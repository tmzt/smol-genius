/*
 * qkn_bf16_decoder.h - BF16 activation decoder for Gemma 2 / PaliGemma 2
 *
 * Same architecture as qkn_decoder but all activations are bf16,
 * matching HF transformers' computation path. Weights are shared
 * with qkn_decoder_t (bf16 mmap'd).
 *
 * Key differences from f32 decoder:
 * - KV cache stored in bf16 (half memory)
 * - Hidden states, Q/K/V, attention output all bf16
 * - Matmul: bf16 × bf16 → f32 accumulate → truncate to bf16
 * - RMSNorm: upcast to f32, compute, truncate to bf16
 * - Attention softmax: f32 internally, output bf16
 */

#ifndef QKN_BF16_DECODER_H
#define QKN_BF16_DECODER_H

#include "qkn_decoder.h"  /* reuse config + weight types */
#include <stdint.h>

typedef struct {
    qkn_config_t   config;
    qkn_decoder_t  decoder;   /* weights (bf16 mmap'd, f32 norms) */

    /* KV cache in bf16 */
    uint16_t *kv_cache_k;     /* [layers, max_seq, kv_dim] bf16 */
    uint16_t *kv_cache_v;
    int kv_cache_len;
    int kv_cache_max;

    /* Prefill buffers (bf16) */
    int pref_cap;
    uint16_t *pref_x;        /* [seq, dim] */
    uint16_t *pref_x_norm;   /* [seq, dim] */
    uint16_t *pref_q;        /* [seq, q_dim] */
    uint16_t *pref_k;        /* [seq, kv_dim] */
    uint16_t *pref_v;        /* [seq, kv_dim] */
    uint16_t *pref_attn_out; /* [seq, q_dim] */
    uint16_t *pref_proj_out; /* [seq, dim] */
    uint16_t *pref_ffn_out;  /* [seq, dim] */
    uint16_t *pref_gate_up;  /* [seq, 2*intermediate] */
    uint16_t *pref_gate;     /* [seq, intermediate] */

    /* Single-token decode buffers (bf16) */
    uint16_t *dec_x;
    uint16_t *dec_x_norm;
    uint16_t *dec_q;
    uint16_t *dec_k;
    uint16_t *dec_v;
    uint16_t *dec_attn_out;
    uint16_t *dec_proj_out;
    uint16_t *dec_gate_up;
    uint16_t *dec_gate;
    uint16_t *dec_ffn_out;

    /* RoPE caches (f32 — sin/cos don't benefit from bf16) */
    float *rope_cache_cos;
    float *rope_cache_sin;
    float *rope_local_cache_cos;
    float *rope_local_cache_sin;
    float *rope_inv_freq;
    float *rope_local_inv_freq;
    int rope_cache_len;
    int rope_local_cache_len;
} qkn_bf16_ctx_t;

/* Load weights into decoder (shared with f32 decoder) */
int qkn_bf16_decoder_load(qkn_decoder_t *dec, multi_safetensors_t *ms,
                           const qkn_config_t *cfg, const char *prefix);

/* Reset KV cache */
void qkn_bf16_kv_cache_reset(qkn_bf16_ctx_t *ctx);

/* Prefill: process multiple tokens, build KV cache.
 * input_embeds is bf16 [seq_len, dim] */
void qkn_bf16_decoder_prefill(qkn_bf16_ctx_t *ctx,
                               const uint16_t *input_embeds, int seq_len);

/* Forward: process single token, return greedy argmax token ID.
 * input_embed is bf16 [dim] */
int qkn_bf16_decoder_forward(qkn_bf16_ctx_t *ctx,
                              const uint16_t *input_embed);

#endif /* QKN_BF16_DECODER_H */
