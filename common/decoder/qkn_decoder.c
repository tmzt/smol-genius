/*
 * qkn_decoder.c - QK-Norm causal LLM decoder
 *
 * Architecture (per layer):
 *   RMSNorm -> QKV (no bias) -> per-head Q/K RMSNorm -> NeoX RoPE
 *   -> Causal GQA attention -> Output proj -> residual
 *   RMSNorm -> SwiGLU MLP (gate/up/down, no bias) -> residual
 *
 * Features: Q/K per-head RMSNorm, NeoX split-half RoPE, GQA,
 * tied embeddings (tok_embeddings == lm_head).
 */

#include "qkn_decoder.h"
#include "../kernels/smol_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SMOL_TOKEN_IM_END 151645

/* ========================================================================
 * Weight Loading Helpers
 * ======================================================================== */

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "qkn_decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

/* Like load_f32 but returns NULL silently if not found (for optional weights). */
static float *load_f32_optional(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) return NULL;
    return safetensors_get_f32(sf, t);
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

static uint16_t *load_bf16_direct(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "qkn_decoder: weight not found: %s\n", name);
        return NULL;
    }
    /* BF16: return direct mmap pointer */
    if (t->dtype == DTYPE_BF16)
        return safetensors_get_bf16_direct(sf, t);
    /* F32: convert to bf16 (allocate) */
    if (t->dtype == DTYPE_F32) {
        int64_t n = safetensor_numel(t);
        if (n <= 0) return NULL;
        uint16_t *out = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
        if (!out) return NULL;
        const float *src = (const float *)safetensors_data(sf, t);
        for (int64_t i = 0; i < n; i++) out[i] = f32_to_bf16(src[i]);
        return out;
    }
    fprintf(stderr, "qkn_decoder: unsupported dtype %d for %s\n", t->dtype, name);
    return NULL;
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

int qkn_decoder_load(qkn_decoder_t *dec, multi_safetensors_t *ms,
                      const qkn_config_t *cfg, const char *prefix) {
    char name[512];

    /* Token embeddings (large, bf16 mmap direct) */
    snprintf(name, sizeof(name), "%s.embed_tokens.weight", prefix);
    dec->tok_embeddings_bf16 = load_bf16_direct(ms, name);
    if (!dec->tok_embeddings_bf16) {
        fprintf(stderr, "qkn_decoder: embed_tokens failed\n");
        return -1;
    }

    /* Transformer layers */
    for (int i = 0; i < cfg->dec_layers; i++) {
        qkn_dec_layer_t *l = &dec->layers[i];

        /* Attention weights (bf16, no bias) */
        snprintf(name, sizeof(name), "%s.layers.%d.self_attn.q_proj.weight", prefix, i);
        l->wq_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.layers.%d.self_attn.k_proj.weight", prefix, i);
        l->wk_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.layers.%d.self_attn.v_proj.weight", prefix, i);
        l->wv_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.layers.%d.self_attn.o_proj.weight", prefix, i);
        l->wo_weight_bf16 = load_bf16_direct(ms, name);

        /* Per-head Q/K RMSNorm weights (Gemma 2/3 only, optional in Gemma 1) */
        snprintf(name, sizeof(name), "%s.layers.%d.self_attn.q_norm.weight", prefix, i);
        l->q_norm_weight = load_f32_optional(ms, name);
        snprintf(name, sizeof(name), "%s.layers.%d.self_attn.k_norm.weight", prefix, i);
        l->k_norm_weight = load_f32_optional(ms, name);

        /* RMSNorm weights */
        snprintf(name, sizeof(name), "%s.layers.%d.input_layernorm.weight", prefix, i);
        l->input_norm = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.layers.%d.post_attention_layernorm.weight", prefix, i);
        l->post_attn_norm = load_f32(ms, name);

        /* Gemma 3: extra feedforward norms (optional, NULL if not present) */
        snprintf(name, sizeof(name), "%s.layers.%d.pre_feedforward_layernorm.weight", prefix, i);
        l->pre_ffn_norm = load_f32_optional(ms, name);
        snprintf(name, sizeof(name), "%s.layers.%d.post_feedforward_layernorm.weight", prefix, i);
        l->post_ffn_norm = load_f32_optional(ms, name);

        /* MLP weights (bf16, no bias) */
        snprintf(name, sizeof(name), "%s.layers.%d.mlp.gate_proj.weight", prefix, i);
        l->gate_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.layers.%d.mlp.up_proj.weight", prefix, i);
        l->up_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.layers.%d.mlp.down_proj.weight", prefix, i);
        l->down_weight_bf16 = load_bf16_direct(ms, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16 ||
            !l->gate_weight_bf16 || !l->up_weight_bf16 || !l->down_weight_bf16) {
            fprintf(stderr, "qkn_decoder: failed to load layer %d\n", i);
            return -1;
        }

        /* Fuse gate+up weights: interleave rows [gate_row0, up_row0, gate_row1, up_row1, ...] */
        {
            int inter = cfg->dec_intermediate;
            int hidden = cfg->dec_hidden;
            size_t row_bytes = (size_t)hidden * sizeof(uint16_t);
            l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)inter * row_bytes);
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * hidden,
                       l->gate_weight_bf16 + (size_t)r * hidden, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * hidden,
                       l->up_weight_bf16 + (size_t)r * hidden, row_bytes);
            }
        }
    }

    /* Final RMSNorm */
    snprintf(name, sizeof(name), "%s.norm.weight", prefix);
    dec->norm = load_f32(ms, name);
    if (!dec->norm) {
        fprintf(stderr, "qkn_decoder: final norm failed\n");
        return -1;
    }

    return 0;
}

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

static int kv_cache_init(qkn_ctx_t *ctx, int max_seq) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    size_t cache_size = (size_t)ctx->config.dec_layers * max_seq * kv_dim * sizeof(float);
    ctx->kv_cache_k = (float *)calloc(1, cache_size);
    ctx->kv_cache_v = (float *)calloc(1, cache_size);
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;
    return 0;
}

static int kv_cache_grow(qkn_ctx_t *ctx, int required) {
    if (required <= ctx->kv_cache_max) return 0;

    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    int new_max = ctx->kv_cache_max;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * kv_dim;
    size_t old_stride = (size_t)ctx->kv_cache_max * kv_dim;
    size_t total = (size_t)ctx->config.dec_layers * new_stride * sizeof(float);

    float *new_k = (float *)calloc(1, total);
    float *new_v = (float *)calloc(1, total);
    if (!new_k || !new_v) { free(new_k); free(new_v); return -1; }

    size_t copy = (size_t)ctx->kv_cache_len * kv_dim * sizeof(float);
    for (int l = 0; l < ctx->config.dec_layers; l++) {
        memcpy(new_k + l * new_stride, ctx->kv_cache_k + l * old_stride, copy);
        memcpy(new_v + l * new_stride, ctx->kv_cache_v + l * old_stride, copy);
    }

    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    ctx->kv_cache_k = new_k;
    ctx->kv_cache_v = new_v;
    ctx->kv_cache_max = new_max;
    return 0;
}

static float *kv_cache_k_at(qkn_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static float *kv_cache_v_at(qkn_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

/* ========================================================================
 * Buffer Management
 * ======================================================================== */

static int ensure_prefill_buffers(qkn_ctx_t *ctx, int seq_len) {
    const qkn_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;

    if (seq_len <= ctx->pref_seq_cap) return 0;

    int new_cap = ctx->pref_seq_cap > 0 ? ctx->pref_seq_cap : 64;
    while (new_cap < seq_len) new_cap *= 2;

#define REALLOC_PREF(ptr, count) do {                                          \
    void *tmp__ = realloc((ptr), (size_t)(count) * sizeof(float));             \
    if (!tmp__) return -1;                                                      \
    (ptr) = (float *)tmp__;                                                     \
} while (0)

    REALLOC_PREF(ctx->pref_x, new_cap * dim);
    REALLOC_PREF(ctx->pref_x_norm, new_cap * dim);
    REALLOC_PREF(ctx->pref_q, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_k, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_v, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_attn_out, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_proj_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_ffn_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_gate, new_cap * intermediate);
    REALLOC_PREF(ctx->pref_gate_up, new_cap * 2 * intermediate);

#undef REALLOC_PREF

    ctx->pref_seq_cap = new_cap;
    return 0;
}

static int ensure_rope_inv_freq(qkn_ctx_t *ctx, int head_dim, float theta) {
    int half = head_dim / 2;
    if (ctx->rope_inv_freq && ctx->rope_inv_freq_half == half) return 0;

    float *inv = (float *)realloc(ctx->rope_inv_freq, (size_t)half * sizeof(float));
    if (!inv) return -1;
    ctx->rope_inv_freq = inv;

    for (int d = 0; d < half; d++) {
        ctx->rope_inv_freq[d] = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    }
    ctx->rope_inv_freq_half = half;
    return 0;
}

static int fill_rope_cache(float **cos_p, float **sin_p, int *cap_p,
                            const float *inv_freq, int half, int head_dim,
                            int required_pos, int interleaved) {
    if (required_pos <= *cap_p) return 0;

    int new_cap = *cap_p > 0 ? *cap_p : 1024;
    while (new_cap < required_pos) new_cap *= 2;

    size_t n = (size_t)new_cap * head_dim;
    float *new_cos = (float *)realloc(*cos_p, n * sizeof(float));
    if (!new_cos) return -1;
    *cos_p = new_cos;
    float *new_sin = (float *)realloc(*sin_p, n * sizeof(float));
    if (!new_sin) return -1;
    *sin_p = new_sin;

    for (int pos = *cap_p; pos < new_cap; pos++) {
        float p = (float)pos;
        float *cr = new_cos + (size_t)pos * head_dim;
        float *sr = new_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            float angle = p * inv_freq[d];
            float c = cosf(angle);
            float s = sinf(angle);
            if (interleaved) {
                cr[2*d] = c; cr[2*d+1] = c;
                sr[2*d] = s; sr[2*d+1] = s;
            } else {
                cr[d] = c; cr[half+d] = c;
                sr[d] = s; sr[half+d] = s;
            }
        }
    }
    *cap_p = new_cap;
    return 0;
}

static int ensure_rope_caches(qkn_ctx_t *ctx, int required_pos) {
    int head_dim = ctx->config.dec_head_dim;
    int half = head_dim / 2;
    int interleaved = (ctx->config.rope_type == QKN_ROPE_INTERLEAVED);
    float theta = ctx->config.dec_rope_theta;
    float local_theta = ctx->config.dec_rope_local_theta;

    /* Global theta cache (full attention layers) */
    if (ensure_rope_inv_freq(ctx, head_dim, theta) != 0) return -1;
    if (fill_rope_cache(&ctx->rope_cache_cos, &ctx->rope_cache_sin,
                         &ctx->rope_cache_cap, ctx->rope_inv_freq,
                         half, head_dim, required_pos, interleaved) != 0) return -1;

    /* Local theta cache (sliding window layers) — only if different theta */
    if (local_theta > 0.0f && local_theta != theta) {
        if (!ctx->rope_local_inv_freq) {
            ctx->rope_local_inv_freq = (float *)malloc(half * sizeof(float));
            if (!ctx->rope_local_inv_freq) return -1;
            for (int d = 0; d < half; d++)
                ctx->rope_local_inv_freq[d] = 1.0f / powf(local_theta, (float)(2*d) / (float)head_dim);
        }
        if (fill_rope_cache(&ctx->rope_local_cache_cos, &ctx->rope_local_cache_sin,
                             &ctx->rope_local_cache_cap, ctx->rope_local_inv_freq,
                             half, head_dim, required_pos, interleaved) != 0) return -1;
    }

    return 0;
}

static void ensure_dec_buffers(qkn_ctx_t *ctx) {
    if (ctx->dec_x) return;
    const qkn_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    int head_dim = cfg->dec_head_dim;

    ctx->dec_x        = (float *)malloc(dim * sizeof(float));
    ctx->dec_x_norm   = (float *)malloc(dim * sizeof(float));
    ctx->dec_q        = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(dim * sizeof(float));
    ctx->dec_gate     = (float *)malloc(2 * intermediate * sizeof(float));
    ctx->dec_up       = NULL; /* unused: gate buffer holds fused gate+up */
    ctx->dec_ffn_out  = (float *)malloc(dim * sizeof(float));
    ctx->dec_rope_cos = (float *)malloc(head_dim * sizeof(float));
    ctx->dec_rope_sin = (float *)malloc(head_dim * sizeof(float));
}

/* ========================================================================
 * Decoder Prefill (Multiple Tokens)
 * ======================================================================== */

void qkn_decoder_prefill(qkn_ctx_t *ctx, const float *input_embeds, int seq_len) {
    qkn_decoder_t *dec = &ctx->decoder;
    const qkn_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Ensure KV cache */
    if (!ctx->kv_cache_k) {
        if (kv_cache_init(ctx, seq_len + 1024) != 0) return;
    } else if (ctx->kv_cache_len + seq_len > ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, ctx->kv_cache_len + seq_len + 1024) != 0) return;
    }

    if (ensure_prefill_buffers(ctx, seq_len) != 0) return;

    float *x = ctx->pref_x;
    float *x_norm = ctx->pref_x_norm;
    float *q = ctx->pref_q;
    float *k = ctx->pref_k;
    float *v = ctx->pref_v;
    float *attn_out = ctx->pref_attn_out;
    float *proj_out = ctx->pref_proj_out;
    float *ffn_out = ctx->pref_ffn_out;
    float *gate = ctx->pref_gate;
    float *gate_up = ctx->pref_gate_up;

    memcpy(x, input_embeds, (size_t)seq_len * dim * sizeof(float));

    int start_pos = ctx->kv_cache_len;
    if (ensure_rope_caches(ctx, start_pos + seq_len) != 0) return;

    int has_local = (ctx->rope_local_cache_cos != NULL);

    void (*apply_rope)(float *, const float *, const float *, int, int, int) =
        cfg->rope_type == QKN_ROPE_INTERLEAVED
            ? smol_apply_rope_interleaved
            : smol_apply_rope_neox;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qkn_dec_layer_t *l = &dec->layers[layer];

        /* Per-layer RoPE cache selection (sliding layers may use local theta) */
        const float *rope_cos, *rope_sin;
        if (has_local && l->is_sliding) {
            rope_cos = ctx->rope_local_cache_cos + (size_t)start_pos * head_dim;
            rope_sin = ctx->rope_local_cache_sin + (size_t)start_pos * head_dim;
        } else {
            rope_cos = ctx->rope_cache_cos + (size_t)start_pos * head_dim;
            rope_sin = ctx->rope_cache_sin + (size_t)start_pos * head_dim;
        }

        /* Input RMSNorm */
        smol_rms_norm(x_norm, x, l->input_norm, seq_len, dim, eps);

        /* QKV projections (no bias) */
        smol_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, seq_len, dim, q_dim);
        smol_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, seq_len, dim, kv_dim);
        smol_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, seq_len, dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        smol_rms_norm_per_head(q, l->q_norm_weight, seq_len, n_heads, head_dim, eps);
        smol_rms_norm_per_head(k, l->k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        /* Apply RoPE */
        apply_rope(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        apply_rope(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            memcpy(kv_cache_k_at(ctx, layer, start_pos + s),
                   k + s * kv_dim, kv_dim * sizeof(float));
            memcpy(kv_cache_v_at(ctx, layer, start_pos + s),
                   v + s * kv_dim, kv_dim * sizeof(float));
        }

        /* Attention — sliding window or full causal, with optional softcap */
        int total_seq = start_pos + seq_len;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);
        float acap = cfg->attn_logit_softcap;
        if (l->is_sliding && cfg->sliding_window > 0) {
            if (acap > 0.0f)
                smol_sliding_window_attention_softcap(attn_out, q, full_k, full_v,
                    seq_len, total_seq, n_heads, n_kv_heads,
                    head_dim, scale, start_pos, cfg->sliding_window, acap);
            else
                smol_sliding_window_attention(attn_out, q, full_k, full_v,
                    seq_len, total_seq, n_heads, n_kv_heads,
                    head_dim, scale, start_pos, cfg->sliding_window);
        } else {
            if (acap > 0.0f)
                smol_causal_attention_softcap(attn_out, q, full_k, full_v,
                    seq_len, total_seq, n_heads, n_kv_heads,
                    head_dim, scale, start_pos, acap);
            else
                smol_causal_attention(attn_out, q, full_k, full_v,
                    seq_len, total_seq, n_heads, n_kv_heads,
                    head_dim, scale, start_pos);
        }

        /* Output projection */
        smol_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16,
                                 seq_len, q_dim, dim);

        if (l->pre_ffn_norm) {
            /* Gemma 3 style: post-sublayer norm before residual add */
            smol_rms_norm(proj_out, proj_out, l->post_attn_norm, seq_len, dim, eps);
            smol_add_inplace(x, proj_out, seq_len * dim);
            smol_rms_norm(x_norm, x, l->pre_ffn_norm, seq_len, dim, eps);
        } else {
            /* Qwen3 style: residual add, then norm */
            smol_add_inplace(x, proj_out, seq_len * dim);
            smol_rms_norm(x_norm, x, l->post_attn_norm, seq_len, dim, eps);
        }

        /* Gated MLP: GeGLU or SwiGLU */
        smol_linear_nobias_bf16(gate_up, x_norm, l->gate_up_fused_bf16,
                                 seq_len, dim, 2 * intermediate);
        if (cfg->activation == QKN_ACT_GEGLU)
            smol_geglu_multiply(gate, gate_up, seq_len, intermediate);
        else
            smol_swiglu_multiply(gate, gate_up, seq_len, intermediate);
        smol_linear_nobias_bf16(ffn_out, gate, l->down_weight_bf16,
                                 seq_len, intermediate, dim);

        if (l->post_ffn_norm) {
            /* Gemma 3 style: post-sublayer norm before residual add */
            smol_rms_norm(ffn_out, ffn_out, l->post_ffn_norm, seq_len, dim, eps);
        }

        smol_add_inplace(x, ffn_out, seq_len * dim);
    }

    ctx->kv_cache_len = start_pos + seq_len;
}

/* ========================================================================
 * Decoder Forward (Single Token Generation)
 * ======================================================================== */

int qkn_decoder_forward(qkn_ctx_t *ctx, const float *input_embed) {
    qkn_decoder_t *dec = &ctx->decoder;
    const qkn_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    float *x_norm = ctx->dec_x_norm;
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;
    float *attn_out = ctx->dec_attn_out;
    float *proj_out = ctx->dec_proj_out;
    float *gate_buf = ctx->dec_gate;
    float *ffn_out = ctx->dec_ffn_out;
    memcpy(x, input_embed, dim * sizeof(float));

    int pos = ctx->kv_cache_len;

    /* Init or grow KV cache if needed */
    if (!ctx->kv_cache_k) {
        if (kv_cache_init(ctx, pos + 1024) != 0) return SMOL_TOKEN_IM_END;
    } else if (pos >= ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, pos + 1024) != 0) return SMOL_TOKEN_IM_END;
    }

    if (ensure_rope_caches(ctx, pos + 1) != 0) {
        return SMOL_TOKEN_IM_END;
    }
    int has_local = (ctx->rope_local_cache_cos != NULL);

    void (*apply_rope)(float *, const float *, const float *, int, int, int) =
        cfg->rope_type == QKN_ROPE_INTERLEAVED
            ? smol_apply_rope_interleaved
            : smol_apply_rope_neox;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qkn_dec_layer_t *l = &dec->layers[layer];

        const float *rope_cos, *rope_sin;
        if (has_local && l->is_sliding) {
            rope_cos = ctx->rope_local_cache_cos + (size_t)pos * head_dim;
            rope_sin = ctx->rope_local_cache_sin + (size_t)pos * head_dim;
        } else {
            rope_cos = ctx->rope_cache_cos + (size_t)pos * head_dim;
            rope_sin = ctx->rope_cache_sin + (size_t)pos * head_dim;
        }

        smol_rms_norm(x_norm, x, l->input_norm, 1, dim, eps);
        smol_linear_nobias_bf16_qkv(q, k, v, x_norm,
                                    l->wq_weight_bf16,
                                    l->wk_weight_bf16,
                                    l->wv_weight_bf16,
                                    dim, q_dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        smol_rms_norm_per_head(q, l->q_norm_weight, 1, n_heads, head_dim, eps);
        smol_rms_norm_per_head(k, l->k_norm_weight, 1, n_kv_heads, head_dim, eps);

        /* Apply RoPE */
        apply_rope(q, rope_cos, rope_sin, 1, n_heads, head_dim);
        apply_rope(k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        memcpy(kv_cache_k_at(ctx, layer, pos), k, kv_dim * sizeof(float));
        memcpy(kv_cache_v_at(ctx, layer, pos), v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        float acap2 = cfg->attn_logit_softcap;
        if (l->is_sliding && cfg->sliding_window > 0) {
            if (acap2 > 0.0f)
                smol_sliding_window_attention_softcap(attn_out, q, full_k, full_v,
                    1, total_seq, n_heads, n_kv_heads,
                    head_dim, scale, pos, cfg->sliding_window, acap2);
            else
                smol_sliding_window_attention(attn_out, q, full_k, full_v,
                    1, total_seq, n_heads, n_kv_heads,
                    head_dim, scale, pos, cfg->sliding_window);
        } else {
            if (acap2 > 0.0f)
                smol_causal_attention_softcap(attn_out, q, full_k, full_v,
                    1, total_seq, n_heads, n_kv_heads,
                    head_dim, scale, pos, acap2);
            else
                smol_causal_attention(attn_out, q, full_k, full_v,
                    1, total_seq, n_heads, n_kv_heads,
                    head_dim, scale, pos);
        }

        smol_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, 1, q_dim, dim);

        if (l->pre_ffn_norm) {
            /* Gemma 3 style: post-sublayer norm before residual add */
            smol_rms_norm(proj_out, proj_out, l->post_attn_norm, 1, dim, eps);
            smol_add_inplace(x, proj_out, dim);
            smol_rms_norm(x_norm, x, l->pre_ffn_norm, 1, dim, eps);
        } else {
            /* Qwen3 style: residual add, then norm */
            smol_add_inplace(x, proj_out, dim);
            smol_rms_norm(x_norm, x, l->post_attn_norm, 1, dim, eps);
        }

        /* Fused gate+up matvec */
        smol_linear_nobias_bf16(gate_buf, x_norm, l->gate_up_fused_bf16,
                                 1, dim, 2 * intermediate);
        if (cfg->activation == QKN_ACT_GEGLU)
            smol_geglu_multiply(gate_buf, gate_buf, 1, intermediate);
        else
            smol_swiglu_multiply(gate_buf, gate_buf, 1, intermediate);
        smol_linear_nobias_bf16(ffn_out, gate_buf, l->down_weight_bf16, 1, intermediate, dim);

        if (l->post_ffn_norm)
            smol_rms_norm(ffn_out, ffn_out, l->post_ffn_norm, 1, dim, eps);

        smol_add_inplace(x, ffn_out, dim);
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm + streaming argmax (no logits buffer needed) */
    smol_rms_norm(x, x, dec->norm, 1, dim, eps);
    return smol_argmax_matvec_bf16(x, dec->tok_embeddings_bf16, dim, cfg->vocab_size);
}
