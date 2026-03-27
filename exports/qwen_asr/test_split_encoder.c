/* Test split encoder: compare conv_stem output to full encoder's internal state.
 * Build: clang -O2 -I../../common/kernels -I../../common/utils -I. -o test_split_encoder \
 *        test_split_encoder.c encoder.c qwen_asr.c ../../common/audio/audio.c \
 *        -L../.. -lsmol -lm -lpthread -framework Accelerate
 * Run:   ./test_split_encoder <model_dir> <audio.f32> <n_samples>
 */
#include "qwen_asr.h"
#include "../../common/audio/audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model_dir> <audio.f32> <n_samples>\n", argv[0]);
        return 1;
    }
    const char *model_dir = argv[1];
    const char *audio_path = argv[2];
    int n_samples = atoi(argv[3]);

    /* Load audio */
    FILE *f = fopen(audio_path, "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", audio_path); return 1; }
    float *samples = (float *)malloc(n_samples * sizeof(float));
    fread(samples, sizeof(float), n_samples, f);
    fclose(f);
    fprintf(stderr, "Loaded %d samples from %s\n", n_samples, audio_path);

    /* Load model */
    qwen_ctx_t *ctx = qwen_load(model_dir);
    if (!ctx) { fprintf(stderr, "failed to load model\n"); return 1; }

    /* 1. Mel spectrogram */
    int mel_frames = 0;
    float *mel = qwen_mel_spectrogram(samples, n_samples, &mel_frames);
    fprintf(stderr, "Mel: %d frames\n", mel_frames);

    /* 2. Conv stem */
    int n_tokens = 0, d_model = 0;
    float *tokens = qwen_encoder_conv_stem(ctx, mel, mel_frames, &n_tokens, &d_model);
    fprintf(stderr, "Conv stem: %d tokens, d_model=%d\n", n_tokens, d_model);

    /* Print first few token values for comparison with GPU */
    fprintf(stderr, "Token[0][0:8]: ");
    for (int i = 0; i < 8 && i < d_model; i++)
        fprintf(stderr, "%.6f ", tokens[i]);
    fprintf(stderr, "\n");

    /* 3. Full encoder (reference) */
    int enc_seq_len = 0;
    float *enc_full = qwen_asr_encoder_forward(&ctx->encoder, &ctx->enc_config,
                                                 mel, mel_frames, &enc_seq_len);
    fprintf(stderr, "Full encoder: %d tokens, output_dim=%d\n",
            enc_seq_len, ctx->enc_config.enc_output_dim);

    fprintf(stderr, "EncOut[0][0:8]: ");
    for (int i = 0; i < 8; i++)
        fprintf(stderr, "%.6f ", enc_full[i]);
    fprintf(stderr, "\n");

    /* Write conv stem output for GPU comparison */
    {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/conv_stem_%d_%d.f32", n_tokens, d_model);
        FILE *out = fopen(path, "wb");
        fwrite(tokens, sizeof(float), n_tokens * d_model, out);
        fclose(out);
        fprintf(stderr, "Wrote conv stem to %s\n", path);
    }

    /* Write full encoder output for comparison */
    {
        int out_dim = ctx->enc_config.enc_output_dim;
        char path[256];
        snprintf(path, sizeof(path), "/tmp/enc_ref_%d_%d.f32", enc_seq_len, out_dim);
        FILE *out = fopen(path, "wb");
        fwrite(enc_full, sizeof(float), enc_seq_len * out_dim, out);
        fclose(out);
        fprintf(stderr, "Wrote encoder reference to %s\n", path);
    }

    /* 4. Test decode with C-reference encoder output */
    fprintf(stderr, "\n--- Testing decode_with_encoder_output ---\n");
    char *text = qwen_decode_with_encoder_output(ctx, enc_full, enc_seq_len,
                                                   ctx->enc_config.enc_output_dim);
    if (text) {
        fprintf(stderr, "Transcription: \"%s\"\n", text);
        free(text);
    } else {
        fprintf(stderr, "Decode returned NULL\n");
    }

    free(mel); free(tokens); free(enc_full); free(samples);
    qwen_free(ctx);
    fprintf(stderr, "Done.\n");
    return 0;
}
