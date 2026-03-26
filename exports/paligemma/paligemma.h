/*
 * paligemma.h - PaliGemma Vision-Language Model (Pure C)
 *
 * PaliGemma = SigLIP vision encoder + linear projection connector + Gemma decoder.
 * Based on PaliGemmaForConditionalGeneration architecture from HuggingFace.
 *
 * The SigLIP encoder is provided by common/vision/siglip.h.
 * The Gemma decoder is provided by common/decoder/qkn_decoder.h.
 * This header defines the PaliGemma-specific connector and orchestration context.
 */

#ifndef PALIGEMMA_H
#define PALIGEMMA_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../../common/vision/siglip.h"
#include "../../common/decoder/qkn_decoder.h"
#include "../../common/utils/safetensors.h"

/* ========================================================================
 * Model Configuration
 * ======================================================================== */

typedef struct {
    /* Vision encoder (used to populate siglip_config_t) */
    int vis_hidden;            /* e.g. 1152 */
    int vis_heads;             /* e.g. 16 */
    int vis_head_dim;          /* e.g. 72 */
    int vis_layers;            /* e.g. 27 */
    int vis_ffn_dim;           /* e.g. 4304 */
    int vis_image_size;        /* e.g. 224 */
    int vis_patch_size;        /* e.g. 14 */
    float vis_layer_norm_eps;  /* e.g. 1e-6 */

    /* Decoder (Gemma) — populated into qkn_config_t */
    int dec_hidden;            /* e.g. 2048 */
    int dec_heads;             /* e.g. 8 */
    int dec_kv_heads;          /* e.g. 1 */
    int dec_head_dim;          /* e.g. 256 */
    int dec_layers;            /* e.g. 18 */
    int dec_intermediate;      /* e.g. 16384 */
    int vocab_size;            /* e.g. 257152 */
    float dec_rms_norm_eps;    /* e.g. 1e-6 */
    float dec_rope_theta;      /* e.g. 10000.0 */

    /* Derived */
    int num_image_tokens;      /* (image_size / patch_size)^2 = 256 */
    int is_gemma2;             /* 1 if text_config.model_type == "gemma2" */
} paligemma_config_t;

/* ========================================================================
 * Connector (simple linear projection)
 * ======================================================================== */

typedef struct {
    /* Linear projection: [dec_hidden, vis_hidden] + bias [dec_hidden] */
    float *proj_weight;
    float *proj_bias;
} paligemma_connector_t;

/* ========================================================================
 * Token Callback
 * ======================================================================== */

typedef void (*paligemma_token_cb)(int token_id, void *userdata);

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    paligemma_config_t config;

    /* Vision: uses common SigLIP encoder */
    siglip_encoder_t vision;
    siglip_config_t  siglip_cfg;

    /* PaliGemma-specific linear projection connector */
    paligemma_connector_t connector;

    /* Decoder (qkn_decoder from common/decoder/) */
    qkn_ctx_t dec_ctx;
    qkn_config_t dec_config;

    /* Model files (kept open for mmap) */
    multi_safetensors_t *safetensors;
    char model_dir[512];

    /* Token streaming callback */
    paligemma_token_cb token_cb;
    void *token_cb_userdata;

    /* Special tokens */
    int bos_token;
    int eos_token;
    int image_token;   /* placeholder token ID for image positions */

    /* Tokenizer (loaded lazily by paligemma_generate_text) */
    char **vocab;       /* [vocab_size] token ID -> string */
    int vocab_loaded;
    void *_hf_tok;      /* hf_tokenizer_t*, kept for encode/decode */

    /* Performance stats */
    double perf_total_ms;
    int perf_tokens;
    double perf_encode_ms;
    double perf_decode_ms;
} paligemma_ctx_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory containing safetensors + config.json */
__attribute__((visibility("default")))
paligemma_ctx_t *paligemma_load(const char *model_dir);

/* Free all resources */
__attribute__((visibility("default")))
void paligemma_free(paligemma_ctx_t *ctx);

/* Set token ID streaming callback (called for each generated token) */
__attribute__((visibility("default")))
void paligemma_set_token_callback(paligemma_ctx_t *ctx, paligemma_token_cb cb, void *userdata);

/* Generate from image + text prompt. Token IDs streamed via callback.
 * Returns number of tokens generated. */
__attribute__((visibility("default")))
int paligemma_generate(paligemma_ctx_t *ctx, const char *image_path,
                       const int *prompt_tokens, int n_prompt_tokens,
                       int max_tokens);

/* High-level: generate text from image + string prompt.
 * Returns malloc'd string (caller must free). Handles tokenization internally.
 * If prompt is NULL, uses "\n" for captioning. */
__attribute__((visibility("default")))
char *paligemma_generate_text(paligemma_ctx_t *ctx, const char *image_path,
                              const char *prompt, int max_tokens);

/* ========================================================================
 * Internal Functions (used across translation units)
 * ======================================================================== */

/* Vision encoder forward: image [3, H, W] -> [num_image_tokens, dec_hidden]
 * Uses common SigLIP encoder, then linear projection connector. */
float *paligemma_vision_forward(paligemma_ctx_t *ctx, const float *image,
                                int channels, int height, int width,
                                int *out_n_tokens);

/* Vision + connector weight loading */
int paligemma_vision_load(siglip_encoder_t *enc, paligemma_connector_t *conn,
                          multi_safetensors_t *ms, const paligemma_config_t *cfg);

#endif /* PALIGEMMA_H */
