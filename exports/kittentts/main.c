/*
 * main.c - KittenTTS CLI
 *
 * Usage: kittentts -d <model_dir> [options] "Text to speak"
 */

#include "kittentts.h"
#include "smol_kernels.h"
#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -d <model_dir> [options] \"Text to speak\"\n"
        "\n"
        "Options:\n"
        "  -d <dir>       Model directory (safetensors + phoneme_vocab.json)\n"
        "  -o <file>      Output WAV file (default: stdout raw PCM)\n"
        "  -s <id>        Style/voice ID 0-7 (default: 0)\n"
        "  --speed <f>    Speed multiplier (default: 1.0)\n"
        "  --list-styles  Print available voice names\n"
        "  -t <n>         Number of threads (default: auto)\n"
        "  --debug        Verbose diagnostics\n"
        "  --silent       Suppress status messages\n"
        "\n"
        "Voices: Bella(0) Jasper(1) Luna(2) Bruno(3) Rosie(4) Hugo(5) Kiki(6) Leo(7)\n",
        prog);
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *output_path = NULL;
    const char *text = NULL;
    int style_id = 0;
    float speed = 1.0f;
    int threads = 0;
    int debug = 0;
    int silent = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            style_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--speed") == 0 && i + 1 < argc) {
            speed = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug = 1;
        } else if (strcmp(argv[i], "--silent") == 0) {
            silent = 1;
        } else if (strcmp(argv[i], "--list-styles") == 0) {
            for (int s = 0; s < KTTS_NUM_STYLES; s++) {
                printf("%d: %s\n", s, KTTS_STYLE_NAMES[s]);
            }
            return 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            text = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || !text) {
        usage(argv[0]);
        return 1;
    }

    if (threads > 0) smol_set_threads(threads);
    smol_verbose = debug;

    /* Load model */
    if (!silent) fprintf(stderr, "Loading model from %s...\n", model_dir);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    kittentts_ctx_t *ctx = kittentts_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double load_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    if (!silent) fprintf(stderr, "Model loaded in %.0f ms\n", load_ms);

    /* Synthesize */
    if (!silent) {
        fprintf(stderr, "Synthesizing: \"%s\" (voice: %s, speed: %.1f)\n",
                text, KTTS_STYLE_NAMES[style_id], speed);
    }

    clock_gettime(CLOCK_MONOTONIC, &t0);

    int n_samples;
    float *audio = kittentts_synthesize(ctx, text, style_id, speed, &n_samples);
    if (!audio) {
        fprintf(stderr, "Synthesis failed\n");
        kittentts_free(ctx);
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double synth_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double audio_s = (double)n_samples / KTTS_SAMPLE_RATE;

    if (!silent) {
        fprintf(stderr, "Synthesis: %.0f ms, %d samples (%.2f s audio, %.1fx realtime)\n",
                synth_ms, n_samples, audio_s, audio_s / (synth_ms / 1000.0));
    }

    /* Output */
    if (output_path) {
        smol_write_wav(output_path, audio, n_samples, KTTS_SAMPLE_RATE);
        if (!silent) fprintf(stderr, "Written to %s\n", output_path);
    } else {
        /* Raw PCM to stdout */
        fwrite(audio, sizeof(float), n_samples, stdout);
    }

    free(audio);
    kittentts_free(ctx);
    return 0;
}
