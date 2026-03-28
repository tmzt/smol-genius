/* Test: feed GPU encoder output to C offline decoder.
 * Build same as test_split_encoder.
 * Run: ./test_gpu_decode <model_dir> <gpu_enc.f32> <n_tokens> <output_dim>
 */
#include "qwen_asr.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <model_dir> <gpu_enc.f32> <n_tokens> <output_dim>\n", argv[0]);
        return 1;
    }
    const char *model_dir = argv[1];
    const char *enc_path = argv[2];
    int n_tokens = atoi(argv[3]);
    int output_dim = atoi(argv[4]);

    /* Load GPU encoder output */
    FILE *f = fopen(enc_path, "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", enc_path); return 1; }
    float *enc_output = (float *)malloc(n_tokens * output_dim * sizeof(float));
    fread(enc_output, sizeof(float), n_tokens * output_dim, f);
    fclose(f);
    fprintf(stderr, "Loaded GPU encoder: %d tokens × %d\n", n_tokens, output_dim);

    /* Load model */
    qwen_ctx_t *ctx = qwen_load(model_dir);
    if (!ctx) { fprintf(stderr, "failed to load model\n"); return 1; }

    /* Decode */
    char *text = qwen_decode_with_encoder_output(ctx, enc_output, n_tokens, output_dim);
    if (text) {
        fprintf(stderr, "GPU→Decoder: \"%s\"\n", text);
        printf("%s\n", text);
        free(text);
    } else {
        fprintf(stderr, "Decode returned NULL\n");
    }

    free(enc_output);
    qwen_free(ctx);
    return 0;
}
