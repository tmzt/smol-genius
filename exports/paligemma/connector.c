/*
 * connector.c - SigLIP weight loading + linear projection connector for PaliGemma
 *
 * Loads vision encoder weights into common siglip_encoder_t,
 * loads connector linear projection weights, then provides the PaliGemma-specific
 * forward pass: SigLIP forward -> linear projection (no pixel shuffle).
 *
 * PaliGemma's connector is simpler than SmolVLM's: just a linear layer that
 * projects from SigLIP's hidden dim to Gemma's hidden dim.
 *
 * Weight name prefixes in PaliGemma safetensors:
 *   Vision:    "vision_tower.vision_model."
 *   Connector: "multi_modal_projector.linear."
 */

#include "paligemma.h"
#include "../../common/kernels/smol_kernels.h"
#include "../../common/utils/safetensors.h"
#include "../../common/vision/siglip.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int smol_verbose;

/* ========================================================================
 * Weight Loading Helpers
 * ======================================================================== */

#define VIS_PREFIX "vision_tower.vision_model."
#define CON_PREFIX "multi_modal_projector.linear."

static float *load_bf16_as_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "paligemma vision: weight not found: %s\n", name);
        return NULL;
    }
    /* Use safetensors_get_f32 which handles unaligned mmap data correctly */
    return safetensors_get_f32(sf, t);
}

/* Try f32 first, fall back to bf16->f32 conversion */
static float *load_auto_f32_impl(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "paligemma vision: weight not found: %s\n", name);
        return NULL;
    }
    if (safetensor_is_bf16(t))
        return load_bf16_as_f32(ms, name);
    return safetensors_get_f32(sf, t);
}

static float *load_auto_f32(multi_safetensors_t *ms, const char *name) {
    return load_auto_f32_impl(ms, name);
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

int paligemma_vision_load(siglip_encoder_t *enc, paligemma_connector_t *conn,
                          multi_safetensors_t *ms, const paligemma_config_t *cfg) {
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
            enc->num_positions = cfg->num_image_tokens;
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
            fprintf(stderr, "paligemma vision: failed to load layer %d weights\n", i);
            return -1;
        }
    }

    /* Post-layernorm */
    snprintf(name, sizeof(name), "%spost_layernorm.weight", VIS_PREFIX);
    enc->post_ln_weight = load_auto_f32(ms, name);
    snprintf(name, sizeof(name), "%spost_layernorm.bias", VIS_PREFIX);
    enc->post_ln_bias = load_auto_f32(ms, name);
    if (!enc->post_ln_weight) return -1;

    /* Connector: linear projection weight + bias */
    snprintf(name, sizeof(name), "%sweight", CON_PREFIX);
    conn->proj_weight = load_auto_f32(ms, name);
    if (!conn->proj_weight) return -1;

    snprintf(name, sizeof(name), "%sbias", CON_PREFIX);
    conn->proj_bias = load_auto_f32(ms, name);
    /* bias may or may not exist — not fatal if missing */

    return 0;
}

/* ========================================================================
 * Forward Pass: SigLIP encoder -> linear projection
 * ======================================================================== */

float *paligemma_vision_forward(paligemma_ctx_t *ctx, const float *image,
                                int channels, int height, int width,
                                int *out_n_tokens) {
    const paligemma_config_t *cfg = &ctx->config;
    int vis_hidden = cfg->vis_hidden;

    /* ---- SigLIP encoder forward ---- */
    int num_patches;
    float *x = siglip_forward(&ctx->vision, &ctx->siglip_cfg,
                               image, channels, height, width, &num_patches);
    if (!x) return NULL;

    if (smol_verbose >= 1) {
        int grid = cfg->vis_image_size / cfg->vis_patch_size;
        fprintf(stderr, "  Vision: %d patches (%dx%d), %d layers\n",
                num_patches, grid, grid, cfg->vis_layers);
    }

    /* ---- Linear projection (with optional bias) ---- */
    int out_dim = cfg->dec_hidden;
    float *output = (float *)malloc((size_t)num_patches * out_dim * sizeof(float));
    if (!output) {
        free(x);
        return NULL;
    }

    if (ctx->connector.proj_bias) {
        smol_linear(output, x, ctx->connector.proj_weight, ctx->connector.proj_bias,
                    num_patches, vis_hidden, out_dim);
    } else {
        smol_linear_nobias(output, x, ctx->connector.proj_weight,
                           num_patches, vis_hidden, out_dim);
    }
    free(x);

    if (smol_verbose >= 1) {
        fprintf(stderr, "  Connector: %d tokens, dim %d -> %d (linear projection)\n",
                num_patches, vis_hidden, out_dim);
    }

    *out_n_tokens = num_patches;
    return output;
}
