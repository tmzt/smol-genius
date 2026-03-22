/*
 * smolvlm.h - SmolVLM-Instruct Pure C Inference Engine
 *
 * SmolVLM = SigLIP vision encoder + pixel shuffle connector + SmolLM2 decoder.
 * Based on Idefics3ForConditionalGeneration architecture.
 *
 * The SigLIP encoder is provided by common/vision/siglip.h. This header
 * defines SmolVLM-specific types: connector, decoder, and orchestration context.
 */

#ifndef SMOLVLM_H
#define SMOLVLM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../../common/vision/siglip.h"
#include "../../common/utils/safetensors.h"

/* ========================================================================
 * Constants
 * ======================================================================== */

/* Vision encoder (passed through to siglip_config_t) */
#define SMOLVLM_VIS_LAYERS      27
#define SMOLVLM_VIS_HIDDEN      1152
#define SMOLVLM_VIS_HEADS       16
#define SMOLVLM_VIS_HEAD_DIM    72     /* 1152 / 16 */
#define SMOLVLM_VIS_FFN         4304
#define SMOLVLM_IMAGE_SIZE      384
#define SMOLVLM_PATCH_SIZE      14
#define SMOLVLM_NUM_PATCHES     729    /* (384/14)^2 = 27*27 */

/* Connector */
#define SMOLVLM_SCALE_FACTOR    3
#define SMOLVLM_IMAGE_SEQ_LEN   81     /* 729 / 9 */
#define SMOLVLM_CONNECTOR_DIM   10368  /* 1152 * 9 */

/* Decoder (SmolLM2-1.7B) */
#define SMOLVLM_DEC_LAYERS      24
#define SMOLVLM_DEC_HIDDEN      2048
#define SMOLVLM_DEC_HEADS       32
#define SMOLVLM_DEC_KV_HEADS    32     /* MHA, not GQA */
#define SMOLVLM_DEC_HEAD_DIM    64
#define SMOLVLM_DEC_FFN         8192
#define SMOLVLM_VOCAB_SIZE      49155

/* Special token IDs */
#define SMOLVLM_TOKEN_IM_START       1       /* <|im_start|> */
#define SMOLVLM_TOKEN_IM_END         2       /* <|im_end|> / pad */
#define SMOLVLM_TOKEN_FAKE_IMAGE     49152   /* <fake_token_around_image> */
#define SMOLVLM_TOKEN_IMAGE          49153   /* <image> placeholder */
#define SMOLVLM_TOKEN_EOS            49154   /* <end_of_utterance> */

/* ========================================================================
 * Model Configuration
 * ======================================================================== */

typedef struct {
    /* Vision encoder (used to populate siglip_config_t) */
    int vis_hidden;            /* 1152 */
    int vis_heads;             /* 16 */
    int vis_head_dim;          /* 72 */
    int vis_layers;            /* 27 */
    int vis_ffn_dim;           /* 4304 */
    int vis_image_size;        /* 384 */
    int vis_patch_size;        /* 14 */
    float vis_layer_norm_eps;  /* 1e-6 */

    /* Connector */
    int scale_factor;          /* 3 */
    int image_seq_len;         /* 81 */

    /* Decoder */
    int dec_hidden;            /* 2048 */
    int dec_heads;             /* 32 */
    int dec_kv_heads;          /* 32 */
    int dec_head_dim;          /* 64 */
    int dec_layers;            /* 24 */
    int dec_intermediate;      /* 8192 */
    int vocab_size;            /* 49155 */
    float dec_rms_norm_eps;    /* 1e-5 */
    float dec_rope_theta;      /* 273768.0 */
} smolvlm_config_t;

/* ========================================================================
 * Connector (pixel shuffle + linear projection)
 * ======================================================================== */

typedef struct {
    /* Linear projection (no bias): [dec_hidden, connector_dim] */
    float *proj_weight;        /* [2048, 10368] */
} smolvlm_connector_t;

/* ========================================================================
 * Decoder Layer (bf16 weights, mmap'd)
 * ======================================================================== */

typedef struct {
    /* Self-attention (no biases) */
    uint16_t *wq_weight_bf16;  /* [dec_hidden, dec_hidden] */
    uint16_t *wk_weight_bf16;  /* [dec_hidden, dec_hidden] */
    uint16_t *wv_weight_bf16;  /* [dec_hidden, dec_hidden] */
    uint16_t *wo_weight_bf16;  /* [dec_hidden, dec_hidden] */

    /* RMSNorm (no bias) */
    float *input_norm;         /* [dec_hidden] */
    float *post_attn_norm;     /* [dec_hidden] */

    /* SwiGLU MLP (no biases) */
    uint16_t *gate_weight_bf16;  /* [intermediate, dec_hidden] */
    uint16_t *up_weight_bf16;    /* [intermediate, dec_hidden] */
    uint16_t *down_weight_bf16;  /* [dec_hidden, intermediate] */

    /* Fused gate+up for single-token matvec [2*intermediate, dec_hidden] */
    uint16_t *gate_up_fused_bf16;
} smolvlm_dec_layer_t;

typedef struct {
    /* Token embeddings (bf16 mmap'd) */
    uint16_t *tok_embeddings_bf16;  /* [vocab_size, dec_hidden] */

    /* Transformer layers */
    smolvlm_dec_layer_t layers[SMOLVLM_DEC_LAYERS];

    /* Final RMSNorm */
    float *norm;               /* [dec_hidden] */

    /* LM head (separate, NOT tied to embeddings) */
    uint16_t *lm_head_bf16;    /* [vocab_size, dec_hidden] */
} smolvlm_decoder_t;

/* ========================================================================
 * Token Callback
 * ======================================================================== */

typedef void (*smolvlm_token_cb)(const char *piece, void *userdata);

/* ========================================================================
 * Tokenizer (opaque, defined in smolvlm_tokenizer.c)
 * ======================================================================== */

typedef struct smolvlm_tokenizer smolvlm_tokenizer_t;

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    smolvlm_config_t config;

    /* Vision: uses common SigLIP encoder */
    siglip_encoder_t vision;
    siglip_config_t  siglip_cfg;

    /* SmolVLM-specific connector */
    smolvlm_connector_t connector;

    /* Decoder */
    smolvlm_decoder_t decoder;

    /* Model files (kept open for mmap) */
    void *safetensors;         /* multi_safetensors_t* */
    char model_dir[512];

    /* Tokenizer */
    smolvlm_tokenizer_t *tokenizer;

    /* KV cache for decoder */
    float *kv_cache_k;         /* [layers, max_seq, kv_heads * head_dim] */
    float *kv_cache_v;
    int kv_cache_len;
    int kv_cache_max;

    /* Persistent decoder buffers (single-token generation) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_gate, *dec_ffn_out;
    float *dec_rope_cos, *dec_rope_sin;

    /* Persistent decoder prefill buffers (multi-token prefill) */
    float *pref_x, *pref_x_norm, *pref_q, *pref_k, *pref_v;
    float *pref_attn_out, *pref_proj_out, *pref_ffn_out;
    float *pref_gate, *pref_gate_up;
    int pref_seq_cap;

    /* Cached RoPE tables */
    float *rope_cache_cos, *rope_cache_sin;
    float *rope_inv_freq;
    int rope_cache_cap;
    int rope_inv_freq_half;

    /* Token streaming callback */
    smolvlm_token_cb token_cb;
    void *token_cb_userdata;

    /* Performance stats */
    double perf_total_ms;
    int perf_tokens;
    double perf_encode_ms;
    double perf_decode_ms;
} smolvlm_ctx_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory */
__attribute__((visibility("default")))
smolvlm_ctx_t *smolvlm_load(const char *model_dir);

/* Free all resources */
__attribute__((visibility("default")))
void smolvlm_free(smolvlm_ctx_t *ctx);

/* Set token streaming callback */
__attribute__((visibility("default")))
void smolvlm_set_token_callback(smolvlm_ctx_t *ctx, smolvlm_token_cb cb, void *userdata);

/* Generate text from image + prompt. Returns allocated string (caller must free).
 * system_prompt may be NULL for no system prompt. */
__attribute__((visibility("default")))
char *smolvlm_generate(smolvlm_ctx_t *ctx, const char *image_path,
                        const char *prompt, const char *system_prompt,
                        int max_tokens);

/* ========================================================================
 * Internal Functions (used across translation units)
 * ======================================================================== */

/* Vision encoder forward: image [3, H, W] -> [num_vis_tokens, dec_hidden]
 * Uses common SigLIP encoder, then pixel shuffle + connector projection. */
float *smolvlm_vision_forward(smolvlm_ctx_t *ctx, const float *image,
                                int channels, int height, int width,
                                int *out_n_tokens);

/* Vision + connector weight loading */
int smolvlm_vision_load(siglip_encoder_t *enc, smolvlm_connector_t *conn,
                          multi_safetensors_t *ms, const smolvlm_config_t *cfg);

/* Decoder weight loading */
int smolvlm_decoder_load(smolvlm_decoder_t *dec, multi_safetensors_t *ms,
                           const smolvlm_config_t *cfg);

/* Decoder prefill (multiple tokens) */
void smolvlm_decoder_prefill(smolvlm_ctx_t *ctx, const float *input_embeds, int seq_len);

/* Decoder forward (single token, uses KV cache, returns greedy token ID) */
int smolvlm_decoder_forward(smolvlm_ctx_t *ctx, const float *input_embed);

/* Tokenizer API */
smolvlm_tokenizer_t *smolvlm_tokenizer_load(const char *model_dir);
void smolvlm_tokenizer_free(smolvlm_tokenizer_t *tok);
int *smolvlm_tokenizer_encode(const smolvlm_tokenizer_t *tok, const char *text, int *out_n);
const char *smolvlm_tokenizer_decode(const smolvlm_tokenizer_t *tok, int token_id);

#endif /* SMOLVLM_H */
