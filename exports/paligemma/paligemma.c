/*
 * paligemma.c - PaliGemma model loading and generation orchestration
 *
 * PaliGemma = SigLIP vision encoder + linear projection + Gemma decoder.
 * Simpler than SmolVLM: no pixel shuffle, no chat template.
 *
 * Prompt format: <image_tokens> <text_prompt>\n
 *
 * Weight name prefixes in PaliGemma safetensors:
 *   Vision:    "vision_tower.vision_model."
 *   Connector: "multi_modal_projector.linear."
 *   Decoder:   "language_model.model."
 *
 * HuggingFace models:
 *   google/paligemma-3b-mix-224
 *   google/paligemma2-3b-pt-224
 */

#include "paligemma.h"
#include "../../common/kernels/smol_kernels.h"
#include "../../common/utils/safetensors.h"
#include "../../common/utils/hf_tokenizer.h"
#include "../../common/vision/image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

extern int smol_verbose;

/* Thread pool (from common/kernels/threading.c) */
extern void smol_set_threads(int n);
extern int smol_get_thread_count(void);
extern int smol_get_num_cpus(void);

/* ========================================================================
 * Config Loading (from config.json)
 * ======================================================================== */

/* Minimal JSON value extraction helpers */
static const char *find_key(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static int json_int(const char *json, const char *key, int def) {
    const char *p = find_key(json, key);
    if (!p) return def;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    return neg ? -val : val;
}

static double json_float(const char *json, const char *key, double def) {
    const char *p = find_key(json, key);
    if (!p) return def;
    char buf[64];
    int i = 0;
    while (i < 63 && ((*p >= '0' && *p <= '9') || *p == '.' || *p == '-' || *p == 'e' || *p == 'E' || *p == '+'))
        buf[i++] = *p++;
    buf[i] = '\0';
    return atof(buf);
}

/* Find a sub-object and return pointer to its content (after '{') */
static const char *find_object(const char *json, const char *key) {
    const char *p = find_key(json, key);
    if (!p || *p != '{') return NULL;
    return p + 1;
}

static int load_config(paligemma_config_t *cfg, const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "paligemma: cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc((size_t)size + 1);
    if (!json || fread(json, 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        free(json);
        return -1;
    }
    fclose(f);
    json[size] = '\0';

    /* Vision config */
    const char *vc = find_object(json, "vision_config");
    if (vc) {
        cfg->vis_hidden = json_int(vc, "hidden_size", 1152);
        cfg->vis_heads = json_int(vc, "num_attention_heads", 16);
        cfg->vis_layers = json_int(vc, "num_hidden_layers", 27);
        cfg->vis_ffn_dim = json_int(vc, "intermediate_size", 4304);
        cfg->vis_image_size = json_int(vc, "image_size", 224);
        cfg->vis_patch_size = json_int(vc, "patch_size", 14);
        cfg->vis_layer_norm_eps = (float)json_float(vc, "layer_norm_eps", 1e-6);
    } else {
        /* PaliGemma defaults */
        cfg->vis_hidden = 1152;
        cfg->vis_heads = 16;
        cfg->vis_layers = 27;
        cfg->vis_ffn_dim = 4304;
        cfg->vis_image_size = 224;
        cfg->vis_patch_size = 14;
        cfg->vis_layer_norm_eps = 1e-6f;
    }
    cfg->vis_head_dim = cfg->vis_hidden / cfg->vis_heads;

    /* Text/decoder config */
    const char *tc = find_object(json, "text_config");
    if (tc) {
        cfg->dec_hidden = json_int(tc, "hidden_size", 2048);
        cfg->dec_heads = json_int(tc, "num_attention_heads", 8);
        cfg->dec_kv_heads = json_int(tc, "num_key_value_heads", 1);
        cfg->dec_layers = json_int(tc, "num_hidden_layers", 18);
        cfg->dec_intermediate = json_int(tc, "intermediate_size", 16384);
        cfg->vocab_size = json_int(tc, "vocab_size", 257152);
        cfg->dec_rms_norm_eps = (float)json_float(tc, "rms_norm_eps", 1e-6);
        cfg->dec_rope_theta = (float)json_float(tc, "rope_theta", 10000.0);
        int hd = json_int(tc, "head_dim", 0);
        cfg->dec_head_dim = hd > 0 ? hd : cfg->dec_hidden / cfg->dec_heads;
    } else {
        /* PaliGemma 3B defaults (Gemma 2B decoder) */
        cfg->dec_hidden = 2048;
        cfg->dec_heads = 8;
        cfg->dec_kv_heads = 1;
        cfg->dec_head_dim = 256;
        cfg->dec_layers = 18;
        cfg->dec_intermediate = 16384;
        cfg->vocab_size = 257152;
        cfg->dec_rms_norm_eps = 1e-6f;
        cfg->dec_rope_theta = 10000.0f;
    }

    /* Compute number of image tokens */
    int grid = cfg->vis_image_size / cfg->vis_patch_size;
    cfg->num_image_tokens = grid * grid;

    free(json);
    return 0;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

paligemma_ctx_t *paligemma_load(const char *model_dir) {
    paligemma_ctx_t *ctx = (paligemma_ctx_t *)calloc(1, sizeof(paligemma_ctx_t));
    if (!ctx) return NULL;

    snprintf(ctx->model_dir, sizeof(ctx->model_dir), "%s", model_dir);

    /* Load config */
    if (load_config(&ctx->config, model_dir) != 0) {
        free(ctx);
        return NULL;
    }

    /* Populate siglip_config_t from paligemma_config_t */
    ctx->siglip_cfg.hidden = ctx->config.vis_hidden;
    ctx->siglip_cfg.heads = ctx->config.vis_heads;
    ctx->siglip_cfg.head_dim = ctx->config.vis_head_dim;
    ctx->siglip_cfg.ffn_dim = ctx->config.vis_ffn_dim;
    ctx->siglip_cfg.image_size = ctx->config.vis_image_size;
    ctx->siglip_cfg.patch_size = ctx->config.vis_patch_size;
    ctx->siglip_cfg.layers = ctx->config.vis_layers;
    ctx->siglip_cfg.layer_norm_eps = ctx->config.vis_layer_norm_eps;

    /* Populate qkn_config_t for the Gemma decoder */
    ctx->dec_config.dec_hidden = ctx->config.dec_hidden;
    ctx->dec_config.dec_layers = ctx->config.dec_layers;
    ctx->dec_config.dec_heads = ctx->config.dec_heads;
    ctx->dec_config.dec_kv_heads = ctx->config.dec_kv_heads;
    ctx->dec_config.dec_head_dim = ctx->config.dec_head_dim;
    ctx->dec_config.dec_intermediate = ctx->config.dec_intermediate;
    ctx->dec_config.vocab_size = ctx->config.vocab_size;
    ctx->dec_config.dec_rms_norm_eps = ctx->config.dec_rms_norm_eps;
    ctx->dec_config.dec_rope_theta = ctx->config.dec_rope_theta;
    ctx->dec_config.dec_rope_local_theta = 0.0f;
    ctx->dec_config.sliding_window = 0;
    ctx->dec_config.sliding_window_pattern = 0;
    ctx->dec_config.attn_logit_softcap = 0.0f;
    ctx->dec_config.final_logit_softcap = 0.0f;
    ctx->dec_config.activation = QKN_ACT_GEGLU;
    ctx->dec_config.rope_type = QKN_ROPE_INTERLEAVED;

    if (smol_verbose >= 1) {
        fprintf(stderr, "PaliGemma: vision %d layers (hidden=%d), decoder %d layers (hidden=%d)\n",
                ctx->config.vis_layers, ctx->config.vis_hidden,
                ctx->config.dec_layers, ctx->config.dec_hidden);
        fprintf(stderr, "  image %dx%d patch=%d, num_image_tokens=%d\n",
                ctx->config.vis_image_size, ctx->config.vis_image_size,
                ctx->config.vis_patch_size, ctx->config.num_image_tokens);
        fprintf(stderr, "  decoder: heads=%d kv_heads=%d head_dim=%d rope_theta=%.0f\n",
                ctx->config.dec_heads, ctx->config.dec_kv_heads,
                ctx->config.dec_head_dim, ctx->config.dec_rope_theta);
    }

    /* Open safetensors */
    ctx->safetensors = multi_safetensors_open(model_dir);
    if (!ctx->safetensors) {
        fprintf(stderr, "paligemma: failed to open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }

    /* Load vision encoder + connector weights */
    if (smol_verbose >= 1)
        fprintf(stderr, "  Loading vision encoder + connector...\n");
    if (paligemma_vision_load(&ctx->vision, &ctx->connector,
                              ctx->safetensors, &ctx->config) != 0) {
        fprintf(stderr, "paligemma: failed to load vision weights\n");
        paligemma_free(ctx);
        return NULL;
    }

    /* Load Gemma decoder weights.
     * PaliGemma safetensors use "language_model.model." prefix for decoder. */
    if (smol_verbose >= 1)
        fprintf(stderr, "  Loading decoder...\n");
    if (qkn_decoder_load(&ctx->dec_ctx.decoder, ctx->safetensors,
                          &ctx->dec_config, "language_model.model") != 0) {
        fprintf(stderr, "paligemma: failed to load decoder weights\n");
        paligemma_free(ctx);
        return NULL;
    }
    ctx->dec_ctx.config = ctx->dec_config;

    /* Gemma convention: RMSNorm uses (1 + weight) instead of weight.
     * Add 1.0 to all loaded norm weights so the standard kernel works. */
    {
        int dim = ctx->config.dec_hidden;
        int hdim = ctx->config.dec_head_dim;
        qkn_decoder_t *dec = &ctx->dec_ctx.decoder;

        /* Final norm */
        if (dec->norm)
            for (int i = 0; i < dim; i++) dec->norm[i] += 1.0f;

        for (int l = 0; l < ctx->config.dec_layers; l++) {
            qkn_dec_layer_t *layer = &dec->layers[l];
            if (layer->input_norm)
                for (int i = 0; i < dim; i++) layer->input_norm[i] += 1.0f;
            if (layer->post_attn_norm)
                for (int i = 0; i < dim; i++) layer->post_attn_norm[i] += 1.0f;
            if (layer->pre_ffn_norm)
                for (int i = 0; i < dim; i++) layer->pre_ffn_norm[i] += 1.0f;
            if (layer->post_ffn_norm)
                for (int i = 0; i < dim; i++) layer->post_ffn_norm[i] += 1.0f;
            if (layer->q_norm_weight)
                for (int i = 0; i < hdim; i++) layer->q_norm_weight[i] += 1.0f;
            if (layer->k_norm_weight)
                for (int i = 0; i < hdim; i++) layer->k_norm_weight[i] += 1.0f;
        }
    }

    /* Special tokens (PaliGemma uses Gemma tokenizer) */
    ctx->bos_token = 2;
    ctx->eos_token = 1;    /* <eos> = 1 in Gemma tokenizer */
    ctx->image_token = 257152;  /* <image> token ID — typically last in vocab or configured */

    /* Try to read image_token_index from config if present */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/config.json", model_dir);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *json = (char *)malloc((size_t)sz + 1);
            if (json) {
                if (fread(json, 1, (size_t)sz, f) == (size_t)sz) {
                    json[sz] = '\0';
                    int img_tok = json_int(json, "image_token_index", -1);
                    if (img_tok >= 0)
                        ctx->image_token = img_tok;
                }
                free(json);
            }
            fclose(f);
        }
    }

    /* Init thread pool if not already done */
    if (smol_get_thread_count() < 2)
        smol_set_threads(smol_get_num_cpus());

    if (smol_verbose >= 1)
        fprintf(stderr, "  Model loaded. image_token=%d, eos=%d, threads=%d\n",
                ctx->image_token, ctx->eos_token, smol_get_thread_count());

    return ctx;
}

/* ========================================================================
 * Cleanup
 * ======================================================================== */

void paligemma_free(paligemma_ctx_t *ctx) {
    if (!ctx) return;

    /* Vision encoder layer weights */
    siglip_encoder_t *enc = &ctx->vision;
    free(enc->patch_weight);
    free(enc->patch_bias);
    free(enc->position_embedding);
    for (int i = 0; i < ctx->config.vis_layers; i++) {
        siglip_layer_t *l = &enc->layers[i];
        free(l->ln1_weight); free(l->ln1_bias);
        free(l->wq_weight); free(l->wq_bias);
        free(l->wk_weight); free(l->wk_bias);
        free(l->wv_weight); free(l->wv_bias);
        free(l->wo_weight); free(l->wo_bias);
        free(l->ln2_weight); free(l->ln2_bias);
        free(l->fc1_weight); free(l->fc1_bias);
        free(l->fc2_weight); free(l->fc2_bias);
    }
    free(enc->post_ln_weight);
    free(enc->post_ln_bias);

    /* Connector */
    free(ctx->connector.proj_weight);
    free(ctx->connector.proj_bias);

    /* Decoder (qkn_decoder buffers) */
    free(ctx->dec_ctx.kv_cache_k);
    free(ctx->dec_ctx.kv_cache_v);
    free(ctx->dec_ctx.pref_x);
    free(ctx->dec_ctx.pref_x_norm);
    free(ctx->dec_ctx.pref_q);
    free(ctx->dec_ctx.pref_k);
    free(ctx->dec_ctx.pref_v);
    free(ctx->dec_ctx.pref_attn_out);
    free(ctx->dec_ctx.pref_proj_out);
    free(ctx->dec_ctx.pref_ffn_out);
    free(ctx->dec_ctx.pref_gate);
    free(ctx->dec_ctx.pref_gate_up);
    free(ctx->dec_ctx.dec_x);
    free(ctx->dec_ctx.dec_x_norm);
    free(ctx->dec_ctx.dec_q);
    free(ctx->dec_ctx.dec_k);
    free(ctx->dec_ctx.dec_v);
    free(ctx->dec_ctx.dec_attn_out);
    free(ctx->dec_ctx.dec_proj_out);
    free(ctx->dec_ctx.dec_gate);
    free(ctx->dec_ctx.dec_up);
    free(ctx->dec_ctx.dec_ffn_out);
    free(ctx->dec_ctx.dec_rope_cos);
    free(ctx->dec_ctx.dec_rope_sin);
    free(ctx->dec_ctx.rope_inv_freq);
    free(ctx->dec_ctx.rope_cache_cos);
    free(ctx->dec_ctx.rope_cache_sin);

    /* Decoder norm weights are allocated */
    {
        qkn_decoder_t *dec = &ctx->dec_ctx.decoder;
        free(dec->norm);
        for (int i = 0; i < ctx->config.dec_layers; i++) {
            free(dec->layers[i].input_norm);
            free(dec->layers[i].post_attn_norm);
            free(dec->layers[i].pre_ffn_norm);
            free(dec->layers[i].post_ffn_norm);
            free(dec->layers[i].q_norm_weight);
            free(dec->layers[i].k_norm_weight);
            free(dec->layers[i].gate_up_fused_bf16);
        }
    }

    /* Tokenizer */
    if (ctx->_hf_tok)
        hf_tokenizer_free((hf_tokenizer_t *)ctx->_hf_tok);

    /* Safetensors */
    if (ctx->safetensors)
        multi_safetensors_close(ctx->safetensors);

    free(ctx);
}

void paligemma_set_token_callback(paligemma_ctx_t *ctx, paligemma_token_cb cb, void *userdata) {
    if (!ctx) return;
    ctx->token_cb = cb;
    ctx->token_cb_userdata = userdata;
}

/* ========================================================================
 * Token Embedding Helper
 * ======================================================================== */

static void tok_embed_bf16_to_f32(float *dst, const uint16_t *tok_emb_bf16,
                                  int token_id, int dim) {
    const uint16_t *src = tok_emb_bf16 + (size_t)token_id * dim;
    for (int i = 0; i < dim; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

/* ========================================================================
 * Time helper
 * ======================================================================== */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ========================================================================
 * Generation
 * ======================================================================== */

int paligemma_generate(paligemma_ctx_t *ctx, const char *image_path,
                       const int *prompt_tokens, int n_prompt_tokens,
                       int max_tokens) {
    if (!ctx) return 0;

    const paligemma_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;

    double t_start = time_ms();

    /* ---- Load and preprocess image ---- */
    int img_w, img_h;
    float *image = smol_load_image(image_path, cfg->vis_image_size, &img_w, &img_h);
    if (!image) {
        fprintf(stderr, "paligemma: failed to load image %s\n", image_path);
        return 0;
    }

    /* ---- Vision encoder forward ---- */
    double t_enc_start = time_ms();
    int n_vis_tokens;
    float *vis_embeds = paligemma_vision_forward(ctx, image, 3, img_h, img_w, &n_vis_tokens);
    free(image);
    if (!vis_embeds) {
        fprintf(stderr, "paligemma: vision encoder failed\n");
        return 0;
    }
    double t_enc_end = time_ms();

    if (smol_verbose >= 1) {
        fprintf(stderr, "  Vision: %d tokens, %.0f ms\n",
                n_vis_tokens, t_enc_end - t_enc_start);
    }

    /* ---- Build full input sequence ----
     * PaliGemma prompt format: <image_tokens> <text_prompt_tokens>
     * No BOS — PaliGemma processor handles that.
     * The prompt_tokens may already include the newline at the end.
     */
    int total_seq = n_vis_tokens + n_prompt_tokens;

    if (smol_verbose >= 1) {
        fprintf(stderr, "  Prompt: %d tokens (%d image + %d text)\n",
                total_seq, n_vis_tokens, n_prompt_tokens);
    }

    /* ---- Build embeddings ----
     * Vision tokens use projected vision embeddings.
     * Text tokens use tok_embeddings, scaled by sqrt(hidden_dim) (Gemma convention).
     */
    float embed_scale = sqrtf((float)hidden);
    float *embeddings = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    if (!embeddings) {
        free(vis_embeds);
        return 0;
    }

    /* Vision embeddings (already projected to dec_hidden dim) */
    memcpy(embeddings, vis_embeds, (size_t)n_vis_tokens * hidden * sizeof(float));
    free(vis_embeds);

    /* Text token embeddings (Gemma scaling: multiply by sqrt(dim)) */
    const uint16_t *tok_emb = ctx->dec_ctx.decoder.tok_embeddings_bf16;
    for (int i = 0; i < n_prompt_tokens; i++) {
        float *dst = embeddings + (size_t)(n_vis_tokens + i) * hidden;
        tok_embed_bf16_to_f32(dst, tok_emb, prompt_tokens[i], hidden);
        for (int d = 0; d < hidden; d++)
            dst[d] *= embed_scale;
    }

    /* ---- Reset KV cache and run decoder ---- */
    qkn_kv_cache_reset(&ctx->dec_ctx);

    double t_dec_start = time_ms();

    /* Prefill: all but last token */
    if (total_seq > 1)
        qkn_decoder_prefill(&ctx->dec_ctx, embeddings, total_seq - 1);

    /* First generated token from the last prompt embedding */
    int token = qkn_decoder_forward(&ctx->dec_ctx,
                                     embeddings + (size_t)(total_seq - 1) * hidden);
    free(embeddings);

    if (smol_verbose >= 2)
        fprintf(stderr, "[paligemma] first token: %d (eos=%d)\n", token, ctx->eos_token);

    /* ---- Autoregressive generation ---- */
    float *tmp_embed = (float *)malloc(hidden * sizeof(float));
    if (!tmp_embed) return 0;

    int n_generated = 0;
    for (int i = 0; i < max_tokens; i++) {
        if (token == ctx->eos_token || token < 0 || token >= cfg->vocab_size)
            break;

        n_generated++;

        if (ctx->token_cb)
            ctx->token_cb(token, ctx->token_cb_userdata);

        tok_embed_bf16_to_f32(tmp_embed, tok_emb, token, hidden);
        for (int d = 0; d < hidden; d++) tmp_embed[d] *= embed_scale;
        token = qkn_decoder_forward(&ctx->dec_ctx, tmp_embed);
    }

    free(tmp_embed);

    double t_end = time_ms();

    /* Performance stats */
    ctx->perf_total_ms = t_end - t_start;
    ctx->perf_tokens = n_generated;
    ctx->perf_encode_ms = t_enc_end - t_enc_start;
    ctx->perf_decode_ms = t_end - t_dec_start;

    return n_generated;
}

/* ========================================================================
 * High-level text generation (tokenizes prompt, decodes output to string)
 * ======================================================================== */

/* Accumulator for token callback */
typedef struct {
    int *ids;
    int count;
    int cap;
} token_accum_t;

static void accum_token_cb(int token_id, void *userdata) {
    token_accum_t *acc = (token_accum_t *)userdata;
    if (acc->count >= acc->cap) {
        acc->cap = acc->cap ? acc->cap * 2 : 256;
        acc->ids = (int *)realloc(acc->ids, (size_t)acc->cap * sizeof(int));
    }
    acc->ids[acc->count++] = token_id;
}

char *paligemma_generate_text(paligemma_ctx_t *ctx, const char *image_path,
                              const char *prompt, int max_tokens) {
    if (!ctx) return NULL;

    /* Load tokenizer lazily */
    if (!ctx->vocab_loaded) {
        hf_tokenizer_t *tok = hf_tokenizer_load(ctx->model_dir);
        if (tok) {
            ctx->vocab = tok->id_to_text;
            ctx->vocab_loaded = 1;
            /* Keep the tokenizer around — we need encode too */
            ctx->_hf_tok = tok;
        }
    }

    /* PaliGemma prompt tokens. Hardcoded for known task prefixes since the
     * GPT-2 BPE tokenizer doesn't handle Gemma's SentencePiece vocab.
     * "caption en\n" = [139458, 1584, 108] in Gemma tokenizer */
    int caption_tokens[] = {139458, 1584, 108}; /* "caption en\n" */
    int describe_tokens[] = {8453, 108};         /* "describe\n" */

    int n_prompt_tokens;
    int *prompt_tokens;

    if (!prompt || !prompt[0] || strstr(prompt, "caption")) {
        prompt_tokens = caption_tokens;
        n_prompt_tokens = 3;
    } else if (strstr(prompt, "describe")) {
        prompt_tokens = describe_tokens;
        n_prompt_tokens = 2;
    } else {
        /* Try tokenizer for custom prompts, fall back to caption */
        prompt_tokens = NULL;
        n_prompt_tokens = 0;
        if (ctx->_hf_tok)
            prompt_tokens = hf_tokenizer_encode(ctx->_hf_tok, prompt, &n_prompt_tokens);
        if (!prompt_tokens || n_prompt_tokens == 0) {
            prompt_tokens = caption_tokens;
            n_prompt_tokens = 3;
        }
    }

    /* Set up token accumulator */
    token_accum_t acc = { NULL, 0, 0 };
    paligemma_token_cb old_cb = ctx->token_cb;
    void *old_ud = ctx->token_cb_userdata;
    ctx->token_cb = accum_token_cb;
    ctx->token_cb_userdata = &acc;

    /* Generate */
    paligemma_generate(ctx, image_path, prompt_tokens, n_prompt_tokens, max_tokens);

    /* Restore callback */
    ctx->token_cb = old_cb;
    ctx->token_cb_userdata = old_ud;

    /* Free prompt tokens if we allocated them */
    if (prompt_tokens != caption_tokens && prompt_tokens != describe_tokens)
        free(prompt_tokens);

    /* Decode output tokens to string */
    if (!acc.count) {
        free(acc.ids);
        return strdup("");
    }

    /* Compute total length */
    size_t total_len = 0;
    for (int i = 0; i < acc.count; i++) {
        const char *piece = NULL;
        if (ctx->vocab && acc.ids[i] >= 0 && acc.ids[i] < ctx->config.vocab_size)
            piece = ctx->vocab[acc.ids[i]];
        if (!piece && ctx->_hf_tok)
            piece = hf_tokenizer_decode(ctx->_hf_tok, acc.ids[i]);
        if (piece)
            total_len += strlen(piece);
    }

    char *result = (char *)malloc(total_len + 1);
    if (!result) { free(acc.ids); return NULL; }

    char *p = result;
    for (int i = 0; i < acc.count; i++) {
        const char *piece = NULL;
        if (ctx->vocab && acc.ids[i] >= 0 && acc.ids[i] < ctx->config.vocab_size)
            piece = ctx->vocab[acc.ids[i]];
        if (!piece && ctx->_hf_tok)
            piece = hf_tokenizer_decode(ctx->_hf_tok, acc.ids[i]);
        if (piece) {
            size_t len = strlen(piece);
            memcpy(p, piece, len);
            p += len;
        }
    }
    *p = '\0';

    free(acc.ids);
    return result;
}
