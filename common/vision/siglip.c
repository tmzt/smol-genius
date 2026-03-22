/*
 * siglip.c - SigLIP vision encoder forward pass
 *
 * Architecture:
 *   Patch embed (Conv2d k=patch_size, s=patch_size) + learnable position embeddings
 *   N SigLIP transformer layers:
 *     LayerNorm -> global self-attention (with bias) -> residual
 *     LayerNorm -> GELU FFN (with bias) -> residual
 *   Post-layernorm
 *
 * Generic encoder only — no pixel shuffle, no connector projection.
 */

#include "siglip.h"
#include "smol_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int smol_verbose;

/* ========================================================================
 * Patch Embedding
 * ======================================================================== */

static float *patch_embed(const siglip_encoder_t *enc, const siglip_config_t *cfg,
                           const float *image, int channels, int height, int width,
                           int *out_num_patches) {
    int patch = cfg->patch_size;
    int hidden = cfg->hidden;
    int pH = height / patch;
    int pW = width / patch;
    int num_patches = pH * pW;

    /* We use smol_conv2d with stride=patch_size, kernel=patch_size, padding=0 */
    int out_h = (height - patch) / patch + 1;
    int out_w = (width - patch) / patch + 1;

    float *conv_out = (float *)malloc((size_t)hidden * out_h * out_w * sizeof(float));
    if (!conv_out) return NULL;

    smol_conv2d(conv_out, image, enc->patch_weight, enc->patch_bias,
                channels, hidden, height, width, patch, patch, patch, 0);

    /* Reshape [hidden, out_h, out_w] -> [num_patches, hidden] */
    float *patches = (float *)malloc((size_t)num_patches * hidden * sizeof(float));
    if (!patches) {
        free(conv_out);
        return NULL;
    }

    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            int patch_idx = h * out_w + w;
            for (int c = 0; c < hidden; c++) {
                patches[patch_idx * hidden + c] = conv_out[c * out_h * out_w + h * out_w + w];
            }
        }
    }
    free(conv_out);

    *out_num_patches = num_patches;
    return patches;
}

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

float *siglip_forward(const siglip_encoder_t *enc, const siglip_config_t *cfg,
                      const float *image, int channels, int height, int width,
                      int *out_seq_len) {
    int hidden = cfg->hidden;
    int n_heads = cfg->heads;
    int head_dim = cfg->head_dim;
    int ffn_dim = cfg->ffn_dim;
    float ln_eps = cfg->layer_norm_eps;

    /* ---- Patch embedding ---- */
    int num_patches;
    float *x = patch_embed(enc, cfg, image, channels, height, width, &num_patches);
    if (!x) return NULL;

    if (smol_verbose >= 1) {
        int grid = (int)sqrtf((float)num_patches);
        fprintf(stderr, "  Vision: %d patches (%dx%d), %d layers\n",
                num_patches, grid, grid, cfg->layers);
    }

    /* ---- Add position embeddings ---- */
    int n_pos = num_patches;
    if (n_pos > enc->num_positions) n_pos = enc->num_positions;
    for (int i = 0; i < n_pos * hidden; i++)
        x[i] += enc->position_embedding[i];

    /* ---- Transformer layers ---- */
    float *x_norm = (float *)malloc((size_t)num_patches * hidden * sizeof(float));
    float *q = (float *)malloc((size_t)num_patches * hidden * sizeof(float));
    float *k = (float *)malloc((size_t)num_patches * hidden * sizeof(float));
    float *v = (float *)malloc((size_t)num_patches * hidden * sizeof(float));
    float *attn_out = (float *)malloc((size_t)num_patches * hidden * sizeof(float));
    float *proj_out = (float *)malloc((size_t)num_patches * hidden * sizeof(float));
    float *ffn_mid = (float *)malloc((size_t)num_patches * ffn_dim * sizeof(float));
    float *ffn_out = (float *)malloc((size_t)num_patches * hidden * sizeof(float));

    /* Single window covering all patches (global attention) */
    int window_starts[2] = {0, num_patches};

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->layers; layer++) {
        const siglip_layer_t *l = &enc->layers[layer];

        /* Pre-LN attention */
        smol_layer_norm(x_norm, x, l->ln1_weight, l->ln1_bias,
                        num_patches, hidden, ln_eps);

        smol_linear(q, x_norm, l->wq_weight, l->wq_bias,
                     num_patches, hidden, hidden);
        smol_linear(k, x_norm, l->wk_weight, l->wk_bias,
                     num_patches, hidden, hidden);
        smol_linear(v, x_norm, l->wv_weight, l->wv_bias,
                     num_patches, hidden, hidden);

        smol_bidirectional_attention(attn_out, q, k, v,
                                      num_patches, n_heads, head_dim, scale,
                                      window_starts, 1);

        smol_linear(proj_out, attn_out, l->wo_weight, l->wo_bias,
                     num_patches, hidden, hidden);
        smol_add_inplace(x, proj_out, num_patches * hidden);

        /* Pre-LN FFN */
        smol_layer_norm(x_norm, x, l->ln2_weight, l->ln2_bias,
                        num_patches, hidden, ln_eps);

        smol_linear(ffn_mid, x_norm, l->fc1_weight, l->fc1_bias,
                     num_patches, hidden, ffn_dim);
        smol_gelu(ffn_mid, num_patches * ffn_dim);
        smol_linear(ffn_out, ffn_mid, l->fc2_weight, l->fc2_bias,
                     num_patches, ffn_dim, hidden);
        smol_add_inplace(x, ffn_out, num_patches * hidden);
    }

    /* Post-layernorm */
    smol_layer_norm(x, x, enc->post_ln_weight, enc->post_ln_bias,
                    num_patches, hidden, ln_eps);

    free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out);
    free(ffn_mid); free(ffn_out);

    *out_seq_len = num_patches;
    return x;
}
