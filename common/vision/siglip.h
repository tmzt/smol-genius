/*
 * siglip.h - SigLIP vision encoder (reusable across VLMs)
 *
 * Patch embed + learnable position embeddings + transformer layers + post-layernorm.
 * Parameterized by siglip_config_t. Weights stored in siglip_encoder_t.
 */

#ifndef SMOL_SIGLIP_H
#define SMOL_SIGLIP_H

#include <stddef.h>
#include <stdint.h>

#define SIGLIP_MAX_LAYERS 32

typedef struct {
    int hidden;            /* e.g. 1152 */
    int heads;             /* e.g. 16 */
    int head_dim;          /* hidden / heads */
    int ffn_dim;           /* e.g. 4304 */
    int image_size;        /* e.g. 384 */
    int patch_size;        /* e.g. 14 */
    int layers;            /* e.g. 27 */
    float layer_norm_eps;  /* e.g. 1e-6 */
} siglip_config_t;

typedef struct {
    float *ln1_weight, *ln1_bias;
    float *wq_weight, *wq_bias;
    float *wk_weight, *wk_bias;
    float *wv_weight, *wv_bias;
    float *wo_weight, *wo_bias;
    float *ln2_weight, *ln2_bias;
    float *fc1_weight, *fc1_bias;
    float *fc2_weight, *fc2_bias;
} siglip_layer_t;

typedef struct {
    float *patch_weight;       /* [hidden, 3, patch_size, patch_size] */
    float *patch_bias;         /* [hidden] */
    float *position_embedding; /* [num_positions, hidden] */
    int num_positions;
    siglip_layer_t layers[SIGLIP_MAX_LAYERS];
    float *post_ln_weight;     /* [hidden] */
    float *post_ln_bias;       /* [hidden] */
} siglip_encoder_t;

/* Forward pass: image [3, H, W] -> [num_patches, hidden].
 * Returns malloc'd output. Caller must free. Sets *out_seq_len. */
__attribute__((visibility("default")))
float *siglip_forward(const siglip_encoder_t *enc, const siglip_config_t *cfg,
                      const float *image, int channels, int height, int width,
                      int *out_seq_len);

#endif /* SMOL_SIGLIP_H */
