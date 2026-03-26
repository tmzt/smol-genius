/*
 * qkn_bf16_decoder.c - BF16 activation Gemma 2 decoder
 *
 * All activations in bf16, matching HF transformers precision.
 * Matmul accumulates in f32, truncates output to bf16.
 * RMSNorm computes in f32, truncates to bf16.
 * Attention softmax in f32, output bf16.
 */

#include "qkn_bf16_decoder.h"
#include "smol_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern int smol_verbose;

/* ========================================================================
 * BF16 ↔ F32 conversion helpers
 * ======================================================================== */

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t u = ((uint32_t)bf) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    /* Truncation (matching PyTorch's bf16 cast) */
    return (uint16_t)(u >> 16);
}

/* Convert buffer f32 → bf16 in-place-ish */
static void f32_to_bf16_buf(uint16_t *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = f32_to_bf16(src[i]);
}

/* Convert buffer bf16 → f32 */
static void bf16_to_f32_buf(float *dst, const uint16_t *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(src[i]);
}

/* ========================================================================
 * BF16 RMSNorm: upcast to f32, compute, truncate to bf16
 * ======================================================================== */

static void bf16_rms_norm(uint16_t *out_bf16, const uint16_t *x_bf16,
                           const float *weight, int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const uint16_t *xr = x_bf16 + s * hidden;
        uint16_t *or_ = out_bf16 + s * hidden;

        /* Compute RMS in f32 */
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float v = bf16_to_f32(xr[i]);
            sum_sq += v * v;
        }
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

        /* Apply norm and truncate to bf16 */
        for (int i = 0; i < hidden; i++) {
            float v = bf16_to_f32(xr[i]) * rms_inv * weight[i];
            or_[i] = f32_to_bf16(v);
        }
    }
}

/* Per-head RMSNorm (for Q/K norms) */
static void bf16_rms_norm_per_head(uint16_t *x_bf16, const float *weight,
                                    int seq_len, int n_heads, int head_dim, float eps) {
    if (!weight) return;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            uint16_t *hv = x_bf16 + s * n_heads * head_dim + h * head_dim;
            float sum_sq = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                float v = bf16_to_f32(hv[i]);
                sum_sq += v * v;
            }
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int i = 0; i < head_dim; i++) {
                float v = bf16_to_f32(hv[i]) * rms_inv * weight[i];
                hv[i] = f32_to_bf16(v);
            }
        }
    }
}

/* ========================================================================
 * BF16 matmul: bf16_weight[out, in] × bf16_input[seq, in] → bf16_output[seq, out]
 * Accumulates in f32, truncates output.
 * ======================================================================== */

static void bf16_linear_nobias(uint16_t *out_bf16, const uint16_t *x_bf16,
                                const uint16_t *w_bf16,
                                int seq_len, int in_dim, int out_dim) {
    for (int s = 0; s < seq_len; s++) {
        const uint16_t *xr = x_bf16 + s * in_dim;
        uint16_t *or_ = out_bf16 + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const uint16_t *wr = w_bf16 + (size_t)o * in_dim;
            float sum = 0.0f;
            for (int i = 0; i < in_dim; i++)
                sum += bf16_to_f32(xr[i]) * bf16_to_f32(wr[i]);
            or_[o] = f32_to_bf16(sum);
        }
    }
}

/* ========================================================================
 * BF16 attention: Q[seq_q, q_dim] × K^T → scores → softmax → × V → out
 * Softmax in f32, everything else bf16.
 * ======================================================================== */

static void bf16_causal_attention_softcap(
        uint16_t *out_bf16, const uint16_t *q_bf16, const uint16_t *k_bf16,
        const uint16_t *v_bf16, int seq_q, int seq_k,
        int n_heads, int n_kv_heads, int head_dim,
        float scale, int q_offset, float cap) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;
    float inv_cap = cap > 0.0f ? 1.0f / cap : 0.0f;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        for (int i = 0; i < seq_q; i++) {
            const uint16_t *q_row = q_bf16 + i * q_hidden + h * head_dim;
            uint16_t *o_row = out_bf16 + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            /* Compute scores in f32 */
            float max_score = -1e30f;
            float *scores = (float *)alloca(k_end * sizeof(float));
            for (int j = 0; j < k_end; j++) {
                const uint16_t *k_row = k_bf16 + j * kv_hidden + kv_h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += bf16_to_f32(q_row[d]) * bf16_to_f32(k_row[d]);
                float score = dot * scale;
                if (cap > 0.0f) score = cap * tanhf(score * inv_cap);
                scores[j] = score;
                if (score > max_score) max_score = score;
            }

            /* Softmax in f32 */
            float sum_exp = 0.0f;
            for (int j = 0; j < k_end; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum_exp += scores[j];
            }
            float inv_sum = 1.0f / sum_exp;

            /* Weighted sum → bf16 */
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int j = 0; j < k_end; j++) {
                    float w = scores[j] * inv_sum;
                    val += w * bf16_to_f32(v_bf16[j * kv_hidden + kv_h * head_dim + d]);
                }
                o_row[d] = f32_to_bf16(val);
            }
        }
    }
}

/* Sliding window variant */
static void bf16_swa_attention_softcap(
        uint16_t *out_bf16, const uint16_t *q_bf16, const uint16_t *k_bf16,
        const uint16_t *v_bf16, int seq_q, int seq_k,
        int n_heads, int n_kv_heads, int head_dim,
        float scale, int q_offset, int window_size, float cap) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;
    float inv_cap = cap > 0.0f ? 1.0f / cap : 0.0f;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        for (int i = 0; i < seq_q; i++) {
            const uint16_t *q_row = q_bf16 + i * q_hidden + h * head_dim;
            uint16_t *o_row = out_bf16 + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;
            int k_start = 0;
            if (global_pos >= window_size)
                k_start = global_pos - window_size + 1;

            int n_keys = k_end - k_start;
            float max_score = -1e30f;
            float *scores = (float *)alloca(n_keys * sizeof(float));

            for (int j = k_start; j < k_end; j++) {
                const uint16_t *k_row = k_bf16 + j * kv_hidden + kv_h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += bf16_to_f32(q_row[d]) * bf16_to_f32(k_row[d]);
                float score = dot * scale;
                if (cap > 0.0f) score = cap * tanhf(score * inv_cap);
                scores[j - k_start] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < n_keys; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum_exp += scores[j];
            }
            float inv_sum = 1.0f / sum_exp;

            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int j = 0; j < n_keys; j++) {
                    float w = scores[j] * inv_sum;
                    val += w * bf16_to_f32(v_bf16[(k_start + j) * kv_hidden + kv_h * head_dim + d]);
                }
                o_row[d] = f32_to_bf16(val);
            }
        }
    }
}

/* ========================================================================
 * BF16 GeLU activation
 * ======================================================================== */

static void bf16_geglu(uint16_t *out_bf16, const uint16_t *gate_up_bf16,
                        int seq_len, int intermediate) {
    for (int s = 0; s < seq_len; s++) {
        const uint16_t *gu = gate_up_bf16 + (size_t)s * 2 * intermediate;
        uint16_t *o = out_bf16 + (size_t)s * intermediate;
        for (int j = 0; j < intermediate; j++) {
            float g = bf16_to_f32(gu[2 * j]);
            float u = bf16_to_f32(gu[2 * j + 1]);
            float x3 = g * g * g;
            float inner = 0.7978845608028654f * (g + 0.044715f * x3);
            g = 0.5f * g * (1.0f + tanhf(inner));
            o[j] = f32_to_bf16(g * u);
        }
    }
}

/* ========================================================================
 * BF16 vector add: a += b (both bf16)
 * ======================================================================== */

static void bf16_add_inplace(uint16_t *a, const uint16_t *b, int n) {
    for (int i = 0; i < n; i++)
        a[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}

/* ========================================================================
 * BF16 RoPE application
 * ======================================================================== */

static void bf16_apply_rope_interleaved(uint16_t *x_bf16, const float *cos_cache,
                                         const float *sin_cache,
                                         int seq_len, int n_heads, int head_dim) {
    int half = head_dim / 2;
    for (int s = 0; s < seq_len; s++) {
        const float *cos_row = cos_cache + s * head_dim;
        const float *sin_row = sin_cache + s * head_dim;
        for (int h = 0; h < n_heads; h++) {
            uint16_t *hv = x_bf16 + s * n_heads * head_dim + h * head_dim;
            for (int i = 0; i < half; i++) {
                float x0 = bf16_to_f32(hv[2 * i]);
                float x1 = bf16_to_f32(hv[2 * i + 1]);
                float c = cos_row[2 * i];
                float sn = sin_row[2 * i];
                hv[2 * i]     = f32_to_bf16(x0 * c - x1 * sn);
                hv[2 * i + 1] = f32_to_bf16(x1 * c + x0 * sn);
            }
        }
    }
}

/* ========================================================================
 * KV Cache Management (bf16)
 * ======================================================================== */

static inline uint16_t *bf16_kv_k_at(qkn_bf16_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_k + (size_t)layer * ctx->kv_cache_max * kv_dim + (size_t)pos * kv_dim;
}
static inline uint16_t *bf16_kv_v_at(qkn_bf16_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_v + (size_t)layer * ctx->kv_cache_max * kv_dim + (size_t)pos * kv_dim;
}

static int bf16_kv_cache_init(qkn_bf16_ctx_t *ctx, int max_seq) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    size_t cache_size = (size_t)ctx->config.dec_layers * max_seq * kv_dim * sizeof(uint16_t);
    ctx->kv_cache_k = (uint16_t *)calloc(1, cache_size);
    ctx->kv_cache_v = (uint16_t *)calloc(1, cache_size);
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;
    return 0;
}

void qkn_bf16_kv_cache_reset(qkn_bf16_ctx_t *ctx) {
    ctx->kv_cache_len = 0;
}

/* ========================================================================
 * Weight Loading (reuse f32 decoder's loader)
 * ======================================================================== */

int qkn_bf16_decoder_load(qkn_decoder_t *dec, multi_safetensors_t *ms,
                           const qkn_config_t *cfg, const char *prefix) {
    return qkn_decoder_load(dec, ms, cfg, prefix);
}

/* ========================================================================
 * RoPE cache (reuse from f32 decoder)
 * ======================================================================== */

/* Reuse the RoPE cache management from qkn_decoder.c.
 * The caches store f32 cos/sin — RoPE is applied by converting
 * bf16 activations to f32, applying rotation, truncating back. */
extern int ensure_rope_caches_bf16(qkn_bf16_ctx_t *ctx, int required_pos);

/* Local implementation — mirrors f32 decoder's ensure_rope_caches */
static int ensure_rope_bf16(qkn_bf16_ctx_t *ctx, int required_pos) {
    const qkn_config_t *cfg = &ctx->config;
    int head_dim = cfg->dec_head_dim;
    int half = head_dim / 2;
    float theta = cfg->dec_rope_theta;

    /* Main RoPE cache */
    if (required_pos > ctx->rope_cache_len) {
        int new_len = required_pos + 256;
        float *new_cos = (float *)realloc(ctx->rope_cache_cos, (size_t)new_len * head_dim * sizeof(float));
        if (!new_cos) return -1;
        float *new_sin = (float *)realloc(ctx->rope_cache_sin, (size_t)new_len * head_dim * sizeof(float));
        if (!new_sin) return -1;

        if (!ctx->rope_inv_freq) {
            ctx->rope_inv_freq = (float *)malloc(half * sizeof(float));
            if (!ctx->rope_inv_freq) return -1;
            for (int i = 0; i < half; i++)
                ctx->rope_inv_freq[i] = 1.0f / powf(theta, (float)(2 * i) / head_dim);
        }

        for (int pos = ctx->rope_cache_len; pos < new_len; pos++) {
            for (int i = 0; i < half; i++) {
                float angle = (float)pos * ctx->rope_inv_freq[i];
                /* Interleaved: cos/sin at [2i] and [2i+1] */
                new_cos[pos * head_dim + 2 * i] = cosf(angle);
                new_cos[pos * head_dim + 2 * i + 1] = cosf(angle);
                new_sin[pos * head_dim + 2 * i] = sinf(angle);
                new_sin[pos * head_dim + 2 * i + 1] = sinf(angle);
            }
        }
        ctx->rope_cache_cos = new_cos;
        ctx->rope_cache_sin = new_sin;
        ctx->rope_cache_len = new_len;
    }

    /* Local theta cache (for sliding window layers) */
    float local_theta = cfg->dec_rope_local_theta;
    if (local_theta > 0.0f && required_pos > ctx->rope_local_cache_len) {
        int new_len = required_pos + 256;
        float *new_cos = (float *)realloc(ctx->rope_local_cache_cos, (size_t)new_len * head_dim * sizeof(float));
        if (!new_cos) return -1;
        float *new_sin = (float *)realloc(ctx->rope_local_cache_sin, (size_t)new_len * head_dim * sizeof(float));
        if (!new_sin) return -1;

        if (!ctx->rope_local_inv_freq) {
            ctx->rope_local_inv_freq = (float *)malloc(half * sizeof(float));
            if (!ctx->rope_local_inv_freq) return -1;
            for (int i = 0; i < half; i++)
                ctx->rope_local_inv_freq[i] = 1.0f / powf(local_theta, (float)(2 * i) / head_dim);
        }

        for (int pos = ctx->rope_local_cache_len; pos < new_len; pos++) {
            for (int i = 0; i < half; i++) {
                float angle = (float)pos * ctx->rope_local_inv_freq[i];
                new_cos[pos * head_dim + 2 * i] = cosf(angle);
                new_cos[pos * head_dim + 2 * i + 1] = cosf(angle);
                new_sin[pos * head_dim + 2 * i] = sinf(angle);
                new_sin[pos * head_dim + 2 * i + 1] = sinf(angle);
            }
        }
        ctx->rope_local_cache_cos = new_cos;
        ctx->rope_local_cache_sin = new_sin;
        ctx->rope_local_cache_len = new_len;
    }

    return 0;
}

/* ========================================================================
 * Ensure prefill buffers
 * ======================================================================== */

static int ensure_bf16_prefill_buffers(qkn_bf16_ctx_t *ctx, int seq_len) {
    if (seq_len <= ctx->pref_cap) return 0;
    int cap = seq_len + 64;
    const qkn_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int inter = cfg->dec_intermediate;

#define REALLOC_BF16(ptr, count) do { \
    uint16_t *tmp = (uint16_t *)realloc(ptr, (size_t)(count) * sizeof(uint16_t)); \
    if (!tmp) return -1; ptr = tmp; } while(0)

    REALLOC_BF16(ctx->pref_x, cap * dim);
    REALLOC_BF16(ctx->pref_x_norm, cap * dim);
    REALLOC_BF16(ctx->pref_q, cap * q_dim);
    REALLOC_BF16(ctx->pref_k, cap * kv_dim);
    REALLOC_BF16(ctx->pref_v, cap * kv_dim);
    REALLOC_BF16(ctx->pref_attn_out, cap * q_dim);
    REALLOC_BF16(ctx->pref_proj_out, cap * dim);
    REALLOC_BF16(ctx->pref_ffn_out, cap * dim);
    REALLOC_BF16(ctx->pref_gate_up, cap * 2 * inter);
    REALLOC_BF16(ctx->pref_gate, cap * inter);
#undef REALLOC_BF16

    ctx->pref_cap = cap;
    return 0;
}

/* ========================================================================
 * Prefill (multiple tokens)
 * ======================================================================== */

void qkn_bf16_decoder_prefill(qkn_bf16_ctx_t *ctx,
                               const uint16_t *input_embeds, int seq_len) {
    qkn_decoder_t *dec = &ctx->decoder;
    const qkn_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    if (!ctx->kv_cache_k) {
        if (bf16_kv_cache_init(ctx, seq_len + 1024) != 0) return;
    }
    if (ensure_bf16_prefill_buffers(ctx, seq_len) != 0) return;

    uint16_t *x = ctx->pref_x;
    uint16_t *x_norm = ctx->pref_x_norm;
    uint16_t *q = ctx->pref_q;
    uint16_t *k = ctx->pref_k;
    uint16_t *v = ctx->pref_v;
    uint16_t *attn_out = ctx->pref_attn_out;
    uint16_t *proj_out = ctx->pref_proj_out;
    uint16_t *ffn_out = ctx->pref_ffn_out;
    uint16_t *gate_up = ctx->pref_gate_up;
    uint16_t *gate = ctx->pref_gate;

    memcpy(x, input_embeds, (size_t)seq_len * dim * sizeof(uint16_t));

    /* Gemma normalizer (in bf16) */
    if (cfg->embed_normalizer > 0.0f) {
        float norm = cfg->embed_normalizer;
        for (int i = 0; i < seq_len * dim; i++)
            x[i] = f32_to_bf16(bf16_to_f32(x[i]) * norm);
    }

    int start_pos = ctx->kv_cache_len;
    if (ensure_rope_bf16(ctx, start_pos + seq_len) != 0) return;

    int has_local = (ctx->rope_local_cache_cos != NULL);
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qkn_dec_layer_t *l = &dec->layers[layer];

        const float *rope_cos, *rope_sin;
        if (has_local && l->is_sliding) {
            rope_cos = ctx->rope_local_cache_cos + (size_t)start_pos * head_dim;
            rope_sin = ctx->rope_local_cache_sin + (size_t)start_pos * head_dim;
        } else {
            rope_cos = ctx->rope_cache_cos + (size_t)start_pos * head_dim;
            rope_sin = ctx->rope_cache_sin + (size_t)start_pos * head_dim;
        }

        /* Input RMSNorm (bf16) */
        bf16_rms_norm(x_norm, x, l->input_norm, seq_len, dim, eps);

        /* QKV projections: bf16 × bf16 → bf16 */
        bf16_linear_nobias(q, x_norm, l->wq_weight_bf16, seq_len, dim, q_dim);
        bf16_linear_nobias(k, x_norm, l->wk_weight_bf16, seq_len, dim, kv_dim);
        bf16_linear_nobias(v, x_norm, l->wv_weight_bf16, seq_len, dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        if (l->q_norm_weight)
            bf16_rms_norm_per_head(q, l->q_norm_weight, seq_len, n_heads, head_dim, eps);
        if (l->k_norm_weight)
            bf16_rms_norm_per_head(k, l->k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        /* RoPE */
        bf16_apply_rope_interleaved(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        bf16_apply_rope_interleaved(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            memcpy(bf16_kv_k_at(ctx, layer, start_pos + s),
                   k + (size_t)s * kv_dim, kv_dim * sizeof(uint16_t));
            memcpy(bf16_kv_v_at(ctx, layer, start_pos + s),
                   v + (size_t)s * kv_dim, kv_dim * sizeof(uint16_t));
        }

        /* Attention */
        int total_seq = start_pos + seq_len;
        uint16_t *full_k = bf16_kv_k_at(ctx, layer, 0);
        uint16_t *full_v = bf16_kv_v_at(ctx, layer, 0);
        float acap = cfg->attn_logit_softcap;

        if (l->is_sliding && cfg->sliding_window > 0)
            bf16_swa_attention_softcap(attn_out, q, full_k, full_v,
                seq_len, total_seq, n_heads, n_kv_heads,
                head_dim, scale, start_pos, cfg->sliding_window, acap);
        else
            bf16_causal_attention_softcap(attn_out, q, full_k, full_v,
                seq_len, total_seq, n_heads, n_kv_heads,
                head_dim, scale, start_pos, acap);

        /* Output projection */
        bf16_linear_nobias(proj_out, attn_out, l->wo_weight_bf16, seq_len, q_dim, dim);

        if (l->pre_ffn_norm) {
            bf16_rms_norm(proj_out, proj_out, l->post_attn_norm, seq_len, dim, eps);
            bf16_add_inplace(x, proj_out, seq_len * dim);
            bf16_rms_norm(x_norm, x, l->pre_ffn_norm, seq_len, dim, eps);
        } else {
            bf16_add_inplace(x, proj_out, seq_len * dim);
            bf16_rms_norm(x_norm, x, l->post_attn_norm, seq_len, dim, eps);
        }

        /* FFN: GeGLU */
        bf16_linear_nobias(gate_up, x_norm, l->gate_up_fused_bf16,
                           seq_len, dim, 2 * intermediate);
        bf16_geglu(gate, gate_up, seq_len, intermediate);
        bf16_linear_nobias(ffn_out, gate, l->down_weight_bf16,
                           seq_len, intermediate, dim);

        if (l->post_ffn_norm)
            bf16_rms_norm(ffn_out, ffn_out, l->post_ffn_norm, seq_len, dim, eps);

        bf16_add_inplace(x, ffn_out, seq_len * dim);

        /* Debug */
        if (smol_verbose >= 2 && layer < 2) {
            uint16_t *last = x + (size_t)(seq_len - 1) * dim;
            fprintf(stderr, "[BF16] L%d last [0:4]: %.4f %.4f %.4f %.4f\n",
                    layer,
                    bf16_to_f32(last[0]), bf16_to_f32(last[1]),
                    bf16_to_f32(last[2]), bf16_to_f32(last[3]));
        }
    }

    ctx->kv_cache_len = start_pos + seq_len;
}

/* ========================================================================
 * Forward (single token generation)
 * ======================================================================== */

int qkn_bf16_decoder_forward(qkn_bf16_ctx_t *ctx, const uint16_t *input_embed) {
    qkn_decoder_t *dec = &ctx->decoder;
    const qkn_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Lazy alloc single-token buffers */
    if (!ctx->dec_x) {
        ctx->dec_x      = (uint16_t *)malloc(dim * sizeof(uint16_t));
        ctx->dec_x_norm  = (uint16_t *)malloc(dim * sizeof(uint16_t));
        ctx->dec_q       = (uint16_t *)malloc(q_dim * sizeof(uint16_t));
        ctx->dec_k       = (uint16_t *)malloc(kv_dim * sizeof(uint16_t));
        ctx->dec_v       = (uint16_t *)malloc(kv_dim * sizeof(uint16_t));
        ctx->dec_attn_out = (uint16_t *)malloc(q_dim * sizeof(uint16_t));
        ctx->dec_proj_out = (uint16_t *)malloc(dim * sizeof(uint16_t));
        ctx->dec_gate_up  = (uint16_t *)malloc(2 * intermediate * sizeof(uint16_t));
        ctx->dec_gate     = (uint16_t *)malloc(intermediate * sizeof(uint16_t));
        ctx->dec_ffn_out  = (uint16_t *)malloc(dim * sizeof(uint16_t));
    }

    uint16_t *x = ctx->dec_x;
    memcpy(x, input_embed, dim * sizeof(uint16_t));

    /* Normalizer */
    if (cfg->embed_normalizer > 0.0f) {
        float norm = cfg->embed_normalizer;
        for (int i = 0; i < dim; i++)
            x[i] = f32_to_bf16(bf16_to_f32(x[i]) * norm);
    }

    int pos = ctx->kv_cache_len;
    if (!ctx->kv_cache_k) {
        if (bf16_kv_cache_init(ctx, pos + 1024) != 0) return 1;
    }
    if (ensure_rope_bf16(ctx, pos + 1) != 0) return 1;

    int has_local = (ctx->rope_local_cache_cos != NULL);
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

        bf16_rms_norm(ctx->dec_x_norm, x, l->input_norm, 1, dim, eps);
        bf16_linear_nobias(ctx->dec_q, ctx->dec_x_norm, l->wq_weight_bf16, 1, dim, q_dim);
        bf16_linear_nobias(ctx->dec_k, ctx->dec_x_norm, l->wk_weight_bf16, 1, dim, kv_dim);
        bf16_linear_nobias(ctx->dec_v, ctx->dec_x_norm, l->wv_weight_bf16, 1, dim, kv_dim);

        if (l->q_norm_weight)
            bf16_rms_norm_per_head(ctx->dec_q, l->q_norm_weight, 1, n_heads, head_dim, eps);
        if (l->k_norm_weight)
            bf16_rms_norm_per_head(ctx->dec_k, l->k_norm_weight, 1, n_kv_heads, head_dim, eps);

        bf16_apply_rope_interleaved(ctx->dec_q, rope_cos, rope_sin, 1, n_heads, head_dim);
        bf16_apply_rope_interleaved(ctx->dec_k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        memcpy(bf16_kv_k_at(ctx, layer, pos), ctx->dec_k, kv_dim * sizeof(uint16_t));
        memcpy(bf16_kv_v_at(ctx, layer, pos), ctx->dec_v, kv_dim * sizeof(uint16_t));

        int total_seq = pos + 1;
        uint16_t *full_k = bf16_kv_k_at(ctx, layer, 0);
        uint16_t *full_v = bf16_kv_v_at(ctx, layer, 0);
        float acap = cfg->attn_logit_softcap;

        if (l->is_sliding && cfg->sliding_window > 0)
            bf16_swa_attention_softcap(ctx->dec_attn_out, ctx->dec_q, full_k, full_v,
                1, total_seq, n_heads, n_kv_heads,
                head_dim, scale, pos, cfg->sliding_window, acap);
        else
            bf16_causal_attention_softcap(ctx->dec_attn_out, ctx->dec_q, full_k, full_v,
                1, total_seq, n_heads, n_kv_heads,
                head_dim, scale, pos, acap);

        bf16_linear_nobias(ctx->dec_proj_out, ctx->dec_attn_out, l->wo_weight_bf16, 1, q_dim, dim);

        if (l->pre_ffn_norm) {
            bf16_rms_norm(ctx->dec_proj_out, ctx->dec_proj_out, l->post_attn_norm, 1, dim, eps);
            bf16_add_inplace(x, ctx->dec_proj_out, dim);
            bf16_rms_norm(ctx->dec_x_norm, x, l->pre_ffn_norm, 1, dim, eps);
        } else {
            bf16_add_inplace(x, ctx->dec_proj_out, dim);
            bf16_rms_norm(ctx->dec_x_norm, x, l->post_attn_norm, 1, dim, eps);
        }

        bf16_linear_nobias(ctx->dec_gate_up, ctx->dec_x_norm, l->gate_up_fused_bf16,
                           1, dim, 2 * intermediate);
        bf16_geglu(ctx->dec_gate, ctx->dec_gate_up, 1, intermediate);
        bf16_linear_nobias(ctx->dec_ffn_out, ctx->dec_gate, l->down_weight_bf16,
                           1, intermediate, dim);

        if (l->post_ffn_norm)
            bf16_rms_norm(ctx->dec_ffn_out, ctx->dec_ffn_out, l->post_ffn_norm, 1, dim, eps);

        bf16_add_inplace(x, ctx->dec_ffn_out, dim);
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm (bf16 → f32 for argmax) */
    float *x_f32 = (float *)alloca(dim * sizeof(float));
    bf16_to_f32_buf(x_f32, x, dim);
    smol_rms_norm(x_f32, x_f32, dec->norm, 1, dim, eps);

    return smol_argmax_matvec_bf16(x_f32, dec->tok_embeddings_bf16, dim, cfg->vocab_size);
}
