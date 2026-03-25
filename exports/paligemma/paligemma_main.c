/*
 * paligemma_main.c - CLI entry point for PaliGemma
 *
 * Usage: paligemma -d <model_dir> -i <image> [options]
 *
 * PaliGemma takes an image and a text prompt, producing a text response.
 * Since tokenization is handled externally (e.g. by the Rust harness),
 * this CLI uses a minimal hardcoded prompt for testing.
 *
 * The token callback receives token IDs (not decoded text), so this CLI
 * just prints the IDs. For actual text output, pair with an external
 * tokenizer (sentencepiece / tokenizers library).
 */

#include "paligemma.h"
#include "../../common/kernels/smol_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int smol_verbose = 1;

/* Token ID streaming callback: print each token ID as it's decoded */
static void stream_token_id(int token_id, void *userdata) {
    (void)userdata;
    printf("%d ", token_id);
    fflush(stdout);
}

static void usage(const char *prog) {
    fprintf(stderr, "paligemma — PaliGemma vision-language inference (pure C)\n\n");
    fprintf(stderr, "Usage: %s -d <model_dir> -i <image> [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>          Model directory (with *.safetensors, config.json)\n");
    fprintf(stderr, "  -i <file>         Input image (PNG, JPG, BMP, PNM, TGA, GIF, PSD)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -t <n>            Number of threads (default: all CPUs)\n");
    fprintf(stderr, "  --max-tokens <n>  Maximum tokens to generate (default: 256)\n");
    fprintf(stderr, "  --debug           Verbose debug output\n");
    fprintf(stderr, "  --silent          No status output (only token IDs on stdout)\n");
    fprintf(stderr, "  -h                Show this help\n");
    fprintf(stderr, "\nNote: This CLI outputs raw token IDs. For decoded text, use an\n");
    fprintf(stderr, "      external tokenizer (sentencepiece) or the Rust harness.\n");
    fprintf(stderr, "\nPaliGemma prompt format: <image_tokens> <text>\\n\n");
    fprintf(stderr, "  The image tokens are generated automatically from the input image.\n");
    fprintf(stderr, "  A default text prompt (newline token) is used if none is provided.\n");
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *image_path = NULL;
    int verbosity = 1;
    int n_threads = 0;
    int max_tokens = 256;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--debug") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--silent") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || !image_path) {
        usage(argv[0]);
        return 1;
    }

    smol_verbose = verbosity;

    /* Initialize thread pool */
    if (n_threads <= 0) n_threads = smol_get_num_cpus();
    smol_set_threads(n_threads);

    /* Load model */
    paligemma_ctx_t *ctx = paligemma_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    /* Set up token streaming */
    int emit_tokens = (verbosity > 0);
    if (emit_tokens)
        paligemma_set_token_callback(ctx, stream_token_id, NULL);

    /* PaliGemma prompt: just a newline token (ID 108 in Gemma tokenizer).
     * In practice, the caller provides tokenized text after the image tokens.
     * For this CLI we use a minimal prompt to trigger captioning. */
    int prompt_tokens[] = { 108 };  /* "\n" in Gemma tokenizer */
    int n_prompt = 1;

    /* Generate */
    int n_generated = paligemma_generate(ctx, image_path, prompt_tokens, n_prompt, max_tokens);

    if (emit_tokens)
        printf("\n");

    /* Performance summary */
    if (verbosity >= 1) {
        double tokens_per_sec = 0.0;
        if (ctx->perf_total_ms > 0) {
            tokens_per_sec = (1000.0 * (double)ctx->perf_tokens) / ctx->perf_total_ms;
        }
        fprintf(stderr,
                "Inference: %.0f ms, %d tokens (%.2f tok/s, encoding: %.0fms, decoding: %.0fms)\n",
                ctx->perf_total_ms, ctx->perf_tokens, tokens_per_sec,
                ctx->perf_encode_ms, ctx->perf_decode_ms);
    }

    if (n_generated == 0) {
        fprintf(stderr, "Generation produced no tokens\n");
        paligemma_free(ctx);
        return 1;
    }

    paligemma_free(ctx);
    return 0;
}
