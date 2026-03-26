#include <math.h>
/* Test SigLIP patch embedding by feeding HF-preprocessed pixels */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "paligemma.h"
#include "smol_kernels.h"

extern int smol_verbose;

int main(int argc, char **argv) {
    const char *model_dir = argc > 1 ? argv[1] :
        "/Users/tmeade/src/common-models/paligemma2-3b-mix-224";
    const char *pixels_path = argc > 2 ? argv[2] : "/tmp/hf_pixels.bin";

    smol_verbose = 0;
    smol_set_threads(4);

    /* Load model */
    paligemma_ctx_t *ctx = paligemma_load(model_dir);
    if (!ctx) { fprintf(stderr, "load failed\n"); return 1; }

    /* Read HF-preprocessed pixels [3, 224, 224] */
    FILE *f = fopen(pixels_path, "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", pixels_path); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *pixels = (float *)malloc(fsize);
    fread(pixels, 1, fsize, f);
    fclose(f);

    int n_pixels = fsize / sizeof(float);
    fprintf(stderr, "Loaded %d floats from %s\n", n_pixels, pixels_path);
    fprintf(stderr, "pixels[0:4]: %.6f %.6f %.6f %.6f\n",
            pixels[0], pixels[1], pixels[2], pixels[3]);

    /* Run SigLIP patch embedding manually using the loaded encoder */
    siglip_encoder_t *enc = &ctx->vision;
    siglip_config_t *cfg = &ctx->siglip_cfg;

    int patch = cfg->patch_size;
    int hidden = cfg->hidden;
    int pH = 224 / patch;
    int pW = 224 / patch;
    int num_patches = pH * pW;
    int out_h = pH, out_w = pW;

    /* Conv2D */
    float *conv_out = (float *)malloc((size_t)hidden * out_h * out_w * sizeof(float));
    smol_conv2d(conv_out, pixels, enc->patch_weight, enc->patch_bias,
                3, hidden, 224, 224, patch, patch, patch, 0);

    /* Reshape [hidden, out_h, out_w] -> [num_patches, hidden] */
    float *patches = (float *)malloc((size_t)num_patches * hidden * sizeof(float));
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            int idx = h * out_w + w;
            for (int c = 0; c < hidden; c++) {
                patches[idx * hidden + c] = conv_out[c * out_h * out_w + h * out_w + w];
            }
        }
    }

    fprintf(stderr, "Patch [0,0:4]: %.6f %.6f %.6f %.6f\n",
            patches[0], patches[1], patches[2], patches[3]);

    /* Compute mean/std */
    double sum = 0, sum2 = 0;
    int n = num_patches * hidden;
    for (int i = 0; i < n; i++) { sum += patches[i]; sum2 += (double)patches[i] * patches[i]; }
    double mean = sum / n;
    double std = sqrt(sum2 / n - mean * mean);
    fprintf(stderr, "Patch mean=%.6f std=%.6f\n", mean, std);

    /* Add position embedding */
    for (int p = 0; p < num_patches; p++) {
        for (int c = 0; c < hidden; c++) {
            patches[p * hidden + c] += enc->position_embedding[p * hidden + c];
        }
    }
    fprintf(stderr, "Embed [0,0:4]: %.6f %.6f %.6f %.6f\n",
            patches[0], patches[1], patches[2], patches[3]);

    free(conv_out);
    free(patches);
    free(pixels);
    paligemma_free(ctx);
    fprintf(stderr, "Done\n");
    return 0;
}
