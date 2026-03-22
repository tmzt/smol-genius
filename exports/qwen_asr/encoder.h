/*
 * encoder.h - Qwen3-ASR audio encoder types and API
 */

#ifndef QWEN_ASR_ENCODER_H
#define QWEN_ASR_ENCODER_H

#include <stddef.h>
#include <stdint.h>
#include "../../common/utils/safetensors.h"
#include "../../common/kernels/smol_kernels.h"

#define QWEN_ASR_MAX_ENC_LAYERS  24
#define QWEN_ASR_CONV_HIDDEN     480

typedef struct {
    /* Audio encoder config (populated from config.json) */
    int enc_d_model;           /* 1024 or 896 */
    int enc_layers;            /* 24 or 18 */
    int enc_heads;             /* 16 or 14 */
    int enc_head_dim;          /* 64 */
    int enc_ffn_dim;           /* 4096 or 3584 */
    int enc_output_dim;        /* 2048 or 1024 */
    int enc_n_window;          /* 50 */
    int enc_n_window_infer;    /* 800 */
    int enc_chunk_size;        /* n_window * 2 = 100 */
    int enc_conv_proj_dim;     /* CONV_HIDDEN * 16 = 7680 */
} qwen_asr_enc_config_t;

typedef struct {
    /* Self-attention (ALL have biases) - pre-converted to f32 */
    float *wq_weight;          /* [d_model, d_model] */
    float *wq_bias;            /* [d_model] */
    float *wk_weight;          /* [d_model, d_model] */
    float *wk_bias;            /* [d_model] */
    float *wv_weight;          /* [d_model, d_model] */
    float *wv_bias;            /* [d_model] */
    float *wo_weight;          /* [d_model, d_model] */
    float *wo_bias;            /* [d_model] */

    /* Pre-attention LayerNorm (with bias) */
    float *attn_norm_weight;   /* [d_model] */
    float *attn_norm_bias;     /* [d_model] */

    /* FFN: GELU(fc1(x)) -> fc2 (ALL have biases) - pre-converted to f32 */
    float *fc1_weight;         /* [ffn_dim, d_model] */
    float *fc1_bias;           /* [ffn_dim] */
    float *fc2_weight;         /* [d_model, ffn_dim] */
    float *fc2_bias;           /* [d_model] */

    /* Pre-FFN LayerNorm (with bias) */
    float *ffn_norm_weight;    /* [d_model] */
    float *ffn_norm_bias;      /* [d_model] */
} qwen_asr_enc_layer_t;

typedef struct {
    /* Conv2D stem (3 layers, each 3x3, stride 2) */
    float *conv1_weight;       /* [480, 1, 3, 3] */
    float *conv1_bias;         /* [480] */
    float *conv2_weight;       /* [480, 480, 3, 3] */
    float *conv2_bias;         /* [480] */
    float *conv3_weight;       /* [480, 480, 3, 3] */
    float *conv3_bias;         /* [480] */

    /* Conv output projection - pre-converted to f32 */
    float *conv_out_weight;    /* [d_model, 7680] */

    /* Transformer layers */
    qwen_asr_enc_layer_t layers[QWEN_ASR_MAX_ENC_LAYERS];

    /* Final LayerNorm */
    float *ln_post_weight;     /* [d_model] */
    float *ln_post_bias;       /* [d_model] */

    /* Projection layers - pre-converted to f32 */
    float *proj1_weight;       /* [d_model, d_model] */
    float *proj1_bias;         /* [d_model] */
    float *proj2_weight;       /* [output_dim, d_model] */
    float *proj2_bias;         /* [output_dim] */
} qwen_asr_encoder_t;

/* Load encoder weights from safetensors */
int qwen_asr_encoder_load(qwen_asr_encoder_t *enc, multi_safetensors_t *ms,
                           const qwen_asr_enc_config_t *cfg);

/* Encoder forward pass.
 * mel: [128, mel_frames] mel spectrogram
 * Returns encoder output embeddings (caller must free), sets *out_seq_len. */
float *qwen_asr_encoder_forward(qwen_asr_encoder_t *enc,
                                 const qwen_asr_enc_config_t *cfg,
                                 const float *mel, int mel_frames,
                                 int *out_seq_len);

#endif /* QWEN_ASR_ENCODER_H */
