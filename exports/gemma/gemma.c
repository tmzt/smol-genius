/*
 * gemma.c - Gemma text generation export
 *
 * High-level API wrapping qkn_decoder for Gemma 2 family models.
 * Loads BF16 safetensors, auto-detects model size, provides
 * tokenize/generate/reset interface.
 */

#include "gemma.h"
#include "../../common/decoder/qkn_decoder.h"
#include "../../common/utils/safetensors.h"
#include "../../common/utils/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int gemma_verbose = 0;

/* ========================================================================
 * Context
 * ======================================================================== */

struct gemma_ctx_t {
    qkn_ctx_t          dec_ctx;
    qkn_config_t       config;
    multi_safetensors_t *safetensors;
    smol_tokenizer_t   *tokenizer;

    gemma_token_cb      token_cb;
    void               *token_cb_userdata;

    int                 eos_token;
    int                 bos_token;
};

/* ========================================================================
 * Helpers
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
 * Config Detection
 * ======================================================================== */

static int detect_config(gemma_ctx_t *ctx, multi_safetensors_t *ms) {
    qkn_config_t *cfg = &ctx->config;

    /* Probe for layer 18 (0-indexed) to distinguish model sizes */
    const safetensor_t *test = multi_safetensors_find(ms,
        "model.layers.18.self_attn.q_proj.weight", NULL);

    if (!test) {
        /* ≤18 layers — Gemma 2 270M (functiongemma, gemma-2-270m-it) */
        cfg->dec_hidden = 1536;
        cfg->dec_layers = 18;
        cfg->dec_heads = 8;
        cfg->dec_kv_heads = 4;
        cfg->dec_head_dim = 192;
        cfg->dec_intermediate = 6144;
        cfg->vocab_size = 256128;
        if (gemma_verbose >= 1)
            fprintf(stderr, "[gemma] detected: 270M (18 layers, hidden=%d)\n",
                    cfg->dec_hidden);
    } else {
        /* >18 layers — Gemma 2 2B */
        cfg->dec_hidden = 2304;
        cfg->dec_layers = 26;
        cfg->dec_heads = 8;
        cfg->dec_kv_heads = 4;
        cfg->dec_head_dim = 256;
        cfg->dec_intermediate = 9216;
        cfg->vocab_size = 256128;
        if (gemma_verbose >= 1)
            fprintf(stderr, "[gemma] detected: 2B (26 layers, hidden=%d)\n",
                    cfg->dec_hidden);
    }

    cfg->dec_rms_norm_eps = 1e-6f;
    cfg->dec_rope_theta = 10000.0f;

    return 0;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

gemma_ctx_t *gemma_load(const char *model_dir) {
    gemma_ctx_t *ctx = (gemma_ctx_t *)calloc(1, sizeof(gemma_ctx_t));
    if (!ctx) return NULL;

    if (gemma_verbose >= 1)
        fprintf(stderr, "[gemma] loading model from %s\n", model_dir);

    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "[gemma] cannot open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    detect_config(ctx, ms);

    if (gemma_verbose >= 1)
        fprintf(stderr, "[gemma] loading decoder weights...\n");
    if (qkn_decoder_load(&ctx->dec_ctx.decoder, ms, &ctx->config, "model") != 0) {
        fprintf(stderr, "[gemma] failed to load decoder weights\n");
        gemma_free(ctx);
        return NULL;
    }
    ctx->dec_ctx.config = ctx->config;

    /* Load tokenizer (try tokenizer.json then vocab.json) */
    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
    ctx->tokenizer = smol_tokenizer_load(path);
    if (!ctx->tokenizer) {
        snprintf(path, sizeof(path), "%s/vocab.json", model_dir);
        ctx->tokenizer = smol_tokenizer_load(path);
    }
    if (!ctx->tokenizer) {
        fprintf(stderr, "[gemma] failed to load tokenizer from %s\n", model_dir);
        gemma_free(ctx);
        return NULL;
    }

    ctx->eos_token = 1;   /* <eos> */
    ctx->bos_token = 2;   /* <bos> */

    if (gemma_verbose >= 1)
        fprintf(stderr, "[gemma] loaded: %d layers, hidden=%d, vocab=%d\n",
                ctx->config.dec_layers, ctx->config.dec_hidden,
                ctx->config.vocab_size);

    return ctx;
}

void gemma_free(gemma_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->tokenizer) smol_tokenizer_free(ctx->tokenizer);
    if (ctx->safetensors) multi_safetensors_close(ctx->safetensors);
    /* Free KV cache + decode buffers */
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
    free(ctx);
}

/* ========================================================================
 * Token Callback
 * ======================================================================== */

void gemma_set_token_callback(gemma_ctx_t *ctx, gemma_token_cb cb, void *userdata) {
    if (!ctx) return;
    ctx->token_cb = cb;
    ctx->token_cb_userdata = userdata;
}

/* ========================================================================
 * Generation
 * ======================================================================== */

void gemma_reset(gemma_ctx_t *ctx) {
    if (!ctx) return;
    qkn_kv_cache_reset(&ctx->dec_ctx);
}

char *gemma_generate(gemma_ctx_t *ctx, const char *prompt, int max_tokens) {
    if (!ctx || !prompt) return NULL;

    gemma_reset(ctx);

    int dim = ctx->config.dec_hidden;
    const uint16_t *tok_emb = ctx->dec_ctx.decoder.tok_embeddings_bf16;

    /* Tokenize */
    int n_tokens = 0;
    int *tokens = smol_tokenizer_encode(ctx->tokenizer, prompt, &n_tokens);
    if (!tokens || n_tokens == 0) {
        free(tokens);
        return NULL;
    }

    if (gemma_verbose >= 2)
        fprintf(stderr, "[gemma] prompt: %d tokens\n", n_tokens);

    /* Embed all prompt tokens */
    float *embeds = (float *)malloc((size_t)n_tokens * dim * sizeof(float));
    if (!embeds) { free(tokens); return NULL; }
    for (int i = 0; i < n_tokens; i++)
        tok_embed_bf16_to_f32(embeds + (size_t)i * dim, tok_emb, tokens[i], dim);
    free(tokens);

    /* Prefill: process all but the last token to populate KV cache */
    if (n_tokens > 1)
        qkn_decoder_prefill(&ctx->dec_ctx, embeds, n_tokens - 1);

    /* First decode step: feed last prompt token embedding */
    float *last_embed = embeds + (size_t)(n_tokens - 1) * dim;
    int token = qkn_decoder_forward(&ctx->dec_ctx, last_embed);
    free(embeds);

    /* Build result string */
    size_t result_len = 0;
    size_t result_cap = 256;
    char *result = (char *)malloc(result_cap);
    if (!result) return NULL;
    result[0] = '\0';

    float *tmp_embed = (float *)malloc(dim * sizeof(float));
    if (!tmp_embed) { free(result); return NULL; }

    for (int i = 0; i < max_tokens; i++) {
        if (token == ctx->eos_token || token < 0 ||
            token >= ctx->config.vocab_size)
            break;

        const char *piece = smol_tokenizer_decode(ctx->tokenizer, token);
        if (piece) {
            size_t piece_len = strlen(piece);
            if (result_len + piece_len + 1 > result_cap) {
                result_cap = (result_len + piece_len + 1) * 2;
                result = (char *)realloc(result, result_cap);
            }
            memcpy(result + result_len, piece, piece_len);
            result_len += piece_len;
            result[result_len] = '\0';

            if (ctx->token_cb)
                ctx->token_cb(piece, ctx->token_cb_userdata);
        }

        tok_embed_bf16_to_f32(tmp_embed, tok_emb, token, dim);
        token = qkn_decoder_forward(&ctx->dec_ctx, tmp_embed);
    }

    free(tmp_embed);
    return result;
}
