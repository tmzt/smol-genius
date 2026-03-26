/*
 * Test bf16 decoder against HF by loading the model and generating tokens.
 * Uses HF-preprocessed pixels (from /tmp/hf_ant_pixels.bin).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "paligemma.h"
#include "qkn_bf16_decoder.h"
#include "smol_kernels.h"

extern int smol_verbose;

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

int main() {
    smol_verbose = 1;
    smol_set_threads(4);

    /* Load model using existing paligemma_load (gets weights + vision) */
    paligemma_ctx_t *pali = paligemma_load(
        "/Users/tmeade/src/common-models/paligemma2-3b-mix-224");
    if (!pali) { fprintf(stderr, "load failed\n"); return 1; }

    /* Read HF pixels */
    FILE *f = fopen("/tmp/hf_ant_pixels.bin", "rb");
    if (!f) { fprintf(stderr, "no HF pixels\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    float *pixels = malloc(sz);
    fread(pixels, 1, sz, f);
    fclose(f);

    /* Run vision forward */
    int n_vis;
    float *vis = paligemma_vision_forward(pali, pixels, 3, 224, 224, &n_vis);
    free(pixels);
    if (!vis) { fprintf(stderr, "vision failed\n"); return 1; }
    fprintf(stderr, "Vision: %d tokens\n", n_vis);

    /* Set up bf16 decoder context */
    qkn_bf16_ctx_t bf16_ctx;
    memset(&bf16_ctx, 0, sizeof(bf16_ctx));
    bf16_ctx.config = pali->dec_config;
    bf16_ctx.decoder = pali->dec_ctx.decoder;  /* share weights */

    int hidden = pali->config.dec_hidden;
    float embed_scale = sqrtf((float)hidden);
    float inv_scale = 1.0f / embed_scale;

    /* Build embeddings in bf16 */
    int prompt[] = {2, 21209, 659, 109};
    int n_prompt = 4;
    int total = n_vis + n_prompt;

    uint16_t *embeds_bf16 = calloc(total * hidden, sizeof(uint16_t));

    /* Vision: divide by sqrt(h) — normalizer will restore */
    for (int i = 0; i < n_vis * hidden; i++)
        embeds_bf16[i] = f32_to_bf16(vis[i] * inv_scale);
    free(vis);

    /* Text: raw bf16 from tok_embeddings */
    const uint16_t *tok_emb = pali->dec_ctx.decoder.tok_embeddings_bf16;
    for (int i = 0; i < n_prompt; i++) {
        memcpy(embeds_bf16 + (n_vis + i) * hidden,
               tok_emb + (size_t)prompt[i] * hidden,
               hidden * sizeof(uint16_t));
    }

    /* Run bf16 decoder */
    qkn_bf16_kv_cache_reset(&bf16_ctx);

    fprintf(stderr, "Prefill %d tokens...\n", total - 1);
    qkn_bf16_decoder_prefill(&bf16_ctx, embeds_bf16, total - 1);

    fprintf(stderr, "First token...\n");
    int token = qkn_bf16_decoder_forward(&bf16_ctx,
        embeds_bf16 + (total - 1) * hidden);
    fprintf(stderr, "Token 1: %d\n", token);

    /* Generate 9 more */
    for (int i = 0; i < 9 && token != 107 && token != 1; i++) {
        uint16_t tmp[2304];
        memcpy(tmp, tok_emb + (size_t)token * hidden, hidden * sizeof(uint16_t));
        token = qkn_bf16_decoder_forward(&bf16_ctx, tmp);
        fprintf(stderr, "Token %d: %d\n", i + 2, token);
    }

    free(embeds_bf16);
    paligemma_free(pali);
    fprintf(stderr, "Done\n");
    return 0;
}
