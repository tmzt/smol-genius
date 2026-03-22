/*
 * connector.c - SigLIP weight loading + pixel shuffle connector for SmolVLM
 *
 * Loads vision encoder weights into common siglip_encoder_t,
 * loads connector projection weight, then provides the SmolVLM-specific
 * forward pass: SigLIP forward -> pixel shuffle -> linear projection.
 */

#include "smolvlm.h"
#include "../../common/kernels/smol_kernels.h"
#include "../../common/utils/safetensors.h"
#include "../../common/vision/siglip.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int smol_verbose;

/* ========================================================================
 * Weight Loading Helpers
 * ======================================================================== */

#define VIS_PREFIX "model.vision_model."
#define CON_PREFIX "model.connector."

static float *load_bf16_as_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "vision: weight not found: %s\n", name);
        return NULL;
    }
    uint16_t *bf16 = safetensors_get_bf16_direct(sf, t);
    if (!bf16) return NULL;

    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];

    float *f32 = (float *)malloc(n * sizeof(float));
    if (!f32) return NULL;

    uint32_t *d = (uint32_t *)(void *)f32;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)bf16[i]) << 16;

    return f32;
}

/* Try f32 first, fall back to bf16->f32 conversion */
static float *load_auto_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "vision: weight not found: %s\n", name);
        return NULL;
    }
    if (safetensor_is_bf16(t))
        return load_bf16_as_f32(ms, name);
    return safetensors_get_f32(sf, t);
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

int smolvlm_vision_load(siglip_encoder_t *enc, smolvlm_connector_t *conn,
                          multi_safetensors_t *ms, const smolvlm_config_t *cfg) {
    char name[512];

    /* Patch embedding */
    snprintf(name, sizeof(name), "%sembeddings.patch_embedding.weight", VIS_PREFIX);
    enc->patch_weight = load_auto_f32(ms, name);
    snprintf(name, sizeof(name), "%sembeddings.patch_embedding.bias", VIS_PREFIX);
    enc->patch_bias = load_auto_f32(ms, name);
    if (!enc->patch_weight || !enc->patch_bias) return -1;

    /* Position embedding */
    snprintf(name, sizeof(name), "%sembeddings.position_embedding.weight", VIS_PREFIX);
    enc->position_embedding = load_auto_f32(ms, name);
    if (!enc->position_embedding) return -1;

    /* Determine num_positions from tensor shape */
    {
        safetensors_file_t *sf = NULL;
        const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
        if (t && t->ndim >= 1)
            enc->num_positions = (int)t->shape[0];
        else
            enc->num_positions = SMOLVLM_NUM_PATCHES;
    }

    /* Transformer layers */
    for (int i = 0; i < cfg->vis_layers; i++) {
        siglip_layer_t *l = &enc->layers[i];
        const char *ep = VIS_PREFIX "encoder.layers";

        snprintf(name, sizeof(name), "%s.%d.layer_norm1.weight", ep, i);
        l->ln1_weight = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.layer_norm1.bias", ep, i);
        l->ln1_bias = load_auto_f32(ms, name);

        snprintf(name, sizeof(name), "%s.%d.self_attn.q_proj.weight", ep, i);
        l->wq_weight = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.q_proj.bias", ep, i);
        l->wq_bias = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.k_proj.weight", ep, i);
        l->wk_weight = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.k_proj.bias", ep, i);
        l->wk_bias = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.v_proj.weight", ep, i);
        l->wv_weight = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.v_proj.bias", ep, i);
        l->wv_bias = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.out_proj.weight", ep, i);
        l->wo_weight = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.out_proj.bias", ep, i);
        l->wo_bias = load_auto_f32(ms, name);

        snprintf(name, sizeof(name), "%s.%d.layer_norm2.weight", ep, i);
        l->ln2_weight = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.layer_norm2.bias", ep, i);
        l->ln2_bias = load_auto_f32(ms, name);

        snprintf(name, sizeof(name), "%s.%d.mlp.fc1.weight", ep, i);
        l->fc1_weight = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.mlp.fc1.bias", ep, i);
        l->fc1_bias = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.mlp.fc2.weight", ep, i);
        l->fc2_weight = load_auto_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.mlp.fc2.bias", ep, i);
        l->fc2_bias = load_auto_f32(ms, name);

        if (!l->wq_weight || !l->wk_weight || !l->wv_weight || !l->wo_weight) {
            fprintf(stderr, "vision: failed to load layer %d weights\n", i);
            return -1;
        }
    }

    /* Post-layernorm */
    snprintf(name, sizeof(name), "%spost_layernorm.weight", VIS_PREFIX);
    enc->post_ln_weight = load_auto_f32(ms, name);
    snprintf(name, sizeof(name), "%spost_layernorm.bias", VIS_PREFIX);
    enc->post_ln_bias = load_auto_f32(ms, name);
    if (!enc->post_ln_weight) return -1;

    /* Connector projection */
    snprintf(name, sizeof(name), "%smodality_projection.proj.weight", CON_PREFIX);
    conn->proj_weight = load_auto_f32(ms, name);
    if (!conn->proj_weight) return -1;

    return 0;
}

/* ========================================================================
 * Pixel Shuffle
 * ======================================================================== */

static float *pixel_shuffle(const float *x, int seq_len, int dim, int scale_factor,
                             int *out_seq_len, int *out_dim) {
    int h = (int)sqrtf((float)seq_len);
    int w = h;
    /* Verify square grid */
    if (h * w != seq_len) {
        fprintf(stderr, "pixel_shuffle: non-square grid %d\n", seq_len);
        return NULL;
    }

    int new_h = h / scale_factor;
    int new_w = w / scale_factor;
    int new_dim = dim * scale_factor * scale_factor;
    int new_seq = new_h * new_w;

    float *out = (float *)malloc((size_t)new_seq * new_dim * sizeof(float));
    if (!out) return NULL;

    /*
     * Pixel shuffle groups scale_factor x scale_factor spatial blocks.
     * PyTorch SmolVLM implementation:
     *   x = x.view(h, w, dim)
     *   x = x.view(h, w/sf, dim*sf)
     *   x = x.T(0,1) -> (w/sf, h, dim*sf)
     *   x = x.view(w/sf, h/sf, dim*sf*sf)
     *   x = x.T(0,1) -> (h/sf, w/sf, dim*sf*sf)
     *   x = x.view(new_seq, new_dim)
     */
    for (int oh = 0; oh < new_h; oh++) {
        for (int ow = 0; ow < new_w; ow++) {
            int out_idx = oh * new_w + ow;
            float *out_row = out + (size_t)out_idx * new_dim;

            /* Gather scale_factor x scale_factor block.
             * The PyTorch sequence of reshape+transpose maps (oh, ow) to:
             *   Step 1: view(h, w/sf, dim*sf) groups along w
             *   Step 2: transpose(0,1) -> (w/sf, h, dim*sf)
             *   Step 3: view(w/sf, h/sf, dim*sf*sf) groups along h
             *   Step 4: transpose(0,1) -> (h/sf, w/sf, dim*sf*sf)
             *
             * Working backwards, output[oh][ow] pulls from:
             *   For each sh in [0, sf), sw in [0, sf):
             *     source row = oh*sf + sw, col = ow*sf + sh
             *   (note: sh/sw order is swapped due to the transpose sequence)
             */
            int d = 0;
            for (int sh = 0; sh < scale_factor; sh++) {
                for (int sw = 0; sw < scale_factor; sw++) {
                    int src_row = oh * scale_factor + sw;
                    int src_col = ow * scale_factor + sh;
                    const float *src = x + (size_t)(src_row * w + src_col) * dim;
                    memcpy(out_row + d, src, dim * sizeof(float));
                    d += dim;
                }
            }
        }
    }

    *out_seq_len = new_seq;
    *out_dim = new_dim;
    return out;
}

/* ========================================================================
 * Forward Pass: SigLIP encoder -> pixel shuffle -> linear projection
 * ======================================================================== */

float *smolvlm_vision_forward(smolvlm_ctx_t *ctx, const float *image,
                                int channels, int height, int width,
                                int *out_n_tokens) {
    const smolvlm_config_t *cfg = &ctx->config;
    int hidden = cfg->vis_hidden;

    /* ---- SigLIP encoder forward ---- */
    int num_patches;
    float *x = siglip_forward(&ctx->vision, &ctx->siglip_cfg,
                               image, channels, height, width, &num_patches);
    if (!x) return NULL;

    if (smol_verbose >= 1) {
        int grid = (int)sqrtf((float)num_patches);
        fprintf(stderr, "  Vision: %d patches (%dx%d), %d layers\n",
                num_patches, grid, grid, cfg->vis_layers);
    }

    /* ---- Pixel shuffle ---- */
    int shuffled_seq, shuffled_dim;
    float *shuffled = pixel_shuffle(x, num_patches, hidden, cfg->scale_factor,
                                     &shuffled_seq, &shuffled_dim);
    free(x);
    if (!shuffled) return NULL;

    if (smol_verbose >= 1) {
        fprintf(stderr, "  Connector: %d -> %d tokens (dim %d -> %d)\n",
                num_patches, shuffled_seq, hidden, shuffled_dim);
    }

    /* ---- Linear projection (no bias) ---- */
    int out_dim = cfg->dec_hidden;
    float *output = (float *)malloc((size_t)shuffled_seq * out_dim * sizeof(float));
    if (!output) {
        free(shuffled);
        return NULL;
    }
    smol_linear_nobias(output, shuffled, ctx->connector.proj_weight,
                        shuffled_seq, shuffled_dim, out_dim);
    free(shuffled);

    *out_n_tokens = shuffled_seq;
    return output;
}
