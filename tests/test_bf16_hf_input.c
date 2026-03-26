/* Feed HF's exact decoder input embeddings to our bf16 decoder */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "paligemma.h"
#include "qkn_bf16_decoder.h"
#include "smol_kernels.h"

extern int smol_verbose;

int main() {
    smol_verbose = 2;
    smol_set_threads(4);

    paligemma_ctx_t *pali = paligemma_load(
        "/Users/tmeade/src/common-models/paligemma2-3b-mix-224");
    if (!pali) return 1;

    /* Read HF's decoder input (260 × 2304 bf16, ALREADY normalized) */
    FILE *f = fopen("/tmp/hf_decoder_input_bf16.bin", "rb");
    if (!f) { fprintf(stderr, "no HF input\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint16_t *hf_embeds = malloc(sz);
    fread(hf_embeds, 1, sz, f);
    fclose(f);

    int total = 260;
    int hidden = 2304;
    fprintf(stderr, "Loaded %ld bytes = %d tokens\n", sz, (int)(sz / sizeof(uint16_t) / hidden));

    /* Set up bf16 decoder — DISABLE normalizer since HF input is already normalized */
    qkn_bf16_ctx_t bf16_ctx;
    memset(&bf16_ctx, 0, sizeof(bf16_ctx));
    bf16_ctx.config = pali->dec_config;
    bf16_ctx.config.embed_normalizer = 0.0f;  /* already applied by HF */
    bf16_ctx.decoder = pali->dec_ctx.decoder;

    qkn_bf16_kv_cache_reset(&bf16_ctx);

    fprintf(stderr, "Prefill 259 tokens...\n");
    qkn_bf16_decoder_prefill(&bf16_ctx, hf_embeds, total - 1);

    fprintf(stderr, "Forward last token...\n");
    int token = qkn_bf16_decoder_forward(&bf16_ctx,
        hf_embeds + (total - 1) * hidden);
    fprintf(stderr, "Token 1: %d\n", token);

    const uint16_t *tok_emb = pali->dec_ctx.decoder.tok_embeddings_bf16;
    for (int i = 0; i < 9 && token != 107 && token != 1; i++) {
        /* For autoregressive, we need raw embed (normalizer disabled) */
        /* But HF normalizes ALL tokens including generated ones */
        /* So we need to normalize the generated token embed manually */
        uint16_t tmp[2304];
        memcpy(tmp, tok_emb + (size_t)token * hidden, hidden * sizeof(uint16_t));
        /* Apply normalizer manually for generated tokens */
        float norm = sqrtf((float)hidden);
        for (int d = 0; d < hidden; d++) {
            uint32_t u = ((uint32_t)tmp[d]) << 16;
            float v; memcpy(&v, &u, 4);
            v *= norm;
            memcpy(&u, &v, 4);
            tmp[d] = (uint16_t)(u >> 16);
        }
        token = qkn_bf16_decoder_forward(&bf16_ctx, tmp);
        fprintf(stderr, "Token %d: %d\n", i + 2, token);
    }

    free(hf_embeds);
    paligemma_free(pali);
    fprintf(stderr, "Done\n");
    return 0;
}
