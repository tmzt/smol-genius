/*
 * qkn_decoder.h - QK-Norm causal LLM decoder
 *
 * Architecture (per layer):
 *   RMSNorm -> QKV (no bias) -> per-head Q/K RMSNorm -> NeoX RoPE
 *   -> Causal GQA attention -> Output proj -> residual
 *   RMSNorm -> SwiGLU MLP (gate/up/down, no bias) -> residual
 *
 * Features: Q/K per-head RMSNorm, NeoX split-half RoPE, GQA,
 * tied embeddings (tok_embeddings == lm_head).
 *
 * Covers: Qwen3, Gemma 3, PaliGemma 2, and derivatives.
 */

#ifndef QKN_DECODER_H
#define QKN_DECODER_H

#include <stddef.h>
#include <stdint.h>
#include "../utils/safetensors.h"

#define QKN_MAX_DEC_LAYERS 28

/* ========================================================================
 * Types
 * ======================================================================== */

typedef enum {
    QKN_ACT_SWIGLU = 0,  /* SiLU(gate) * up — Qwen3 */
    QKN_ACT_GEGLU  = 1,  /* GELU(gate) * up — Gemma 3 */
} qkn_activation_t;

typedef enum {
    QKN_ROPE_NEOX        = 0,  /* split-half: (x[d], x[half+d]) — Qwen3 */
    QKN_ROPE_INTERLEAVED = 1,  /* consecutive pairs: (x[2d], x[2d+1]) — Gemma 3 */
} qkn_rope_type_t;

typedef struct {
    /* Attention weights (bf16, no bias) */
    uint16_t *wq_weight_bf16;
    uint16_t *wk_weight_bf16;
    uint16_t *wv_weight_bf16;
    uint16_t *wo_weight_bf16;

    /* Per-head Q/K RMSNorm weights */
    float *q_norm_weight;
    float *k_norm_weight;

    /* Layer norms */
    float *input_norm;
    float *post_attn_norm;

    /* SwiGLU MLP weights (bf16, no bias) */
    uint16_t *gate_weight_bf16;
    uint16_t *up_weight_bf16;
    uint16_t *down_weight_bf16;

    /* Fused gate+up (interleaved rows) */
    uint16_t *gate_up_fused_bf16;

    /* Gemma 3: extra feedforward norms (NULL if not present) */
    float *pre_ffn_norm;
    float *post_ffn_norm;

    /* Sliding window: 1 = use sliding window, 0 = full causal */
    int is_sliding;
} qkn_dec_layer_t;

typedef struct {
    uint16_t *tok_embeddings_bf16;
    float    *norm;
    qkn_dec_layer_t layers[QKN_MAX_DEC_LAYERS];
} qkn_decoder_t;

typedef struct {
    int   dec_hidden;
    int   dec_layers;
    int   dec_heads;
    int   dec_kv_heads;
    int   dec_head_dim;
    int   dec_intermediate;
    int   vocab_size;
    float dec_rms_norm_eps;
    float dec_rope_theta;
    qkn_activation_t activation;
    qkn_rope_type_t  rope_type;
    float dec_rope_local_theta; /* RoPE theta for sliding window layers (0 = same as dec_rope_theta) */
    int   sliding_window;       /* 0 = disabled, >0 = window size */
    int   sliding_window_pattern; /* every Nth layer is full attention (0 = all full) */
    float attn_logit_softcap;   /* 0 = disabled, >0 = softcap value (e.g. 50.0 for Gemma 2) */
    float final_logit_softcap;  /* 0 = disabled, >0 = softcap on final logits (e.g. 30.0) */
    float embed_normalizer;     /* 0 = disabled, >0 = multiply input embeds (Gemma: sqrt(hidden)) */
} qkn_config_t;

typedef struct {
    qkn_config_t   config;
    qkn_decoder_t  decoder;

    /* KV cache */
    float *kv_cache_k;
    float *kv_cache_v;
    int    kv_cache_len;
    int    kv_cache_max;

    /* Prefill buffers */
    float *pref_x;
    float *pref_x_norm;
    float *pref_q;
    float *pref_k;
    float *pref_v;
    float *pref_attn_out;
    float *pref_proj_out;
    float *pref_ffn_out;
    float *pref_gate;
    float *pref_gate_up;
    int    pref_seq_cap;

    /* Single-token decode buffers */
    float *dec_x;
    float *dec_x_norm;
    float *dec_q;
    float *dec_k;
    float *dec_v;
    float *dec_attn_out;
    float *dec_proj_out;
    float *dec_gate;
    float *dec_up;
    float *dec_ffn_out;
    float *dec_rope_cos;
    float *dec_rope_sin;

    /* RoPE cache (global theta — used for full attention layers) */
    float *rope_inv_freq;
    int    rope_inv_freq_half;
    float *rope_cache_cos;
    float *rope_cache_sin;
    int    rope_cache_cap;

    /* RoPE cache (local theta — used for sliding window layers, Gemma 3) */
    float *rope_local_inv_freq;
    float *rope_local_cache_cos;
    float *rope_local_cache_sin;
    int    rope_local_cache_cap;
} qkn_ctx_t;

/* ========================================================================
 * API
 * ======================================================================== */

/* Load decoder weights from safetensors.
 * prefix: weight name prefix, e.g. "thinker.model" gives
 *         "thinker.model.embed_tokens.weight",
 *         "thinker.model.layers.0.self_attn.q_proj.weight", etc.
 *         Pass "" or "model" depending on the model's naming convention. */
int qkn_decoder_load(qkn_decoder_t *dec, multi_safetensors_t *ms,
                      const qkn_config_t *cfg, const char *prefix);

/* Multi-token prefill. Populates KV cache for seq_len input embeddings. */
void qkn_decoder_prefill(qkn_ctx_t *ctx, const float *input_embeds, int seq_len);

/* Single-token decode. Returns greedy-sampled token ID. */
int qkn_decoder_forward(qkn_ctx_t *ctx, const float *input_embed);

/* Reset KV cache position (for new sequence on same loaded weights). */
static inline void qkn_kv_cache_reset(qkn_ctx_t *ctx) {
    ctx->kv_cache_len = 0;
}

#endif /* QKN_DECODER_H */
