/*
 * compile_wakewords.c - Generate and compile wakeword codebooks
 *
 * Produces:
 *   1. MFCC codebook (wakewords_mfcc.bin) — per-segment MFCC templates
 *   2. Encoder reference (wakewords_enc.bin) — encoder embeddings for full phrases
 *
 * Usage: compile_wakewords [output_dir] [--model-dir path]
 *        Default output_dir: ./wakewords/
 *        --model-dir: path to ASR model for encoder reference generation
 *
 * Without --model-dir, only the MFCC codebook is generated.
 */

/* Globals defined in libsmol.a — just declare extern here */
extern int smol_verbose;
extern int qwen_verbose;
extern int qwen_monitor;

#include "mfcc.h"
#include "audio.h"
#include "qwen_asr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>

/* ========================================================================
 * Segment and Phrase Definitions
 * ======================================================================== */

typedef struct {
    int id;
    const char *tts_text;
    const char *file_prefix;
} segment_def_t;

static const segment_def_t SEGMENTS[] = {
    { 0, "hey",   "seg_hey"   },
    { 1, "ay",    "seg_ay"    },
    { 2, "hi",    "seg_hi"    },
    { 3, "oh",    "seg_oh"    },
    { 4, "okay",  "seg_okay"  },
    { 5, "chit",  "seg_chit"  },
    { 6, "kite",  "seg_kite"  },
    { 7, "tin",   "seg_tin"   },
};
#define N_SEGMENTS (int)(sizeof(SEGMENTS) / sizeof(SEGMENTS[0]))

typedef struct {
    const char *name;
    const char *tts_text;    /* full phrase for TTS */
    const char *file_prefix; /* filename prefix for phrase WAVs */
    int seg_ids[8];
    int n_segs;
    int n_prefix;
} phrase_def_t;

static const phrase_def_t PHRASES[] = {
    { "hey chitin",    "hey chitin",    "phrase_hey_chitin",    { 0, 5, 7 }, 3, 1 },
    { "hi chitin",     "hi chitin",     "phrase_hi_chitin",     { 2, 5, 7 }, 3, 1 },
    { "okay chitin",   "okay chitin",   "phrase_okay_chitin",   { 4, 5, 7 }, 3, 1 },
    { "hey kite-inn",  "hey kite inn",  "phrase_hey_kiteinn",   { 0, 6, 7 }, 3, 1 },
    { "okay kite-inn", "okay kite inn", "phrase_okay_kiteinn",  { 4, 6, 7 }, 3, 1 },
};
#define N_PHRASES (int)(sizeof(PHRASES) / sizeof(PHRASES[0]))

static const char *VOICES[] = {
    "Samantha", "Alex", "Daniel", "Karen", "Moira"
};
#define N_VOICES (int)(sizeof(VOICES) / sizeof(VOICES[0]))

#define N_COEFFS 13

/* ========================================================================
 * TTS Generation
 * ======================================================================== */

static int generate_segment_wavs(const char *outdir) {
    if (access("/usr/bin/say", X_OK) != 0) {
        fprintf(stderr, "Error: /usr/bin/say not found (macOS only)\n");
        return -1;
    }

    for (int s = 0; s < N_SEGMENTS; s++) {
        for (int v = 0; v < N_VOICES; v++) {
            char wav_path[1024];
            snprintf(wav_path, sizeof(wav_path), "%s/%s_%d.wav",
                     outdir, SEGMENTS[s].file_prefix, v + 1);
            if (access(wav_path, F_OK) == 0) continue;

            char cmd[2048];
            snprintf(cmd, sizeof(cmd),
                     "say -v %s -o \"%s\" --data-format=LEI16@16000 \"%s\"",
                     VOICES[v], wav_path, SEGMENTS[s].tts_text);
            fprintf(stderr, "  TTS: %s → %s\n", SEGMENTS[s].tts_text, wav_path);
            if (system(cmd) != 0) {
                fprintf(stderr, "Error: say failed for voice %s\n", VOICES[v]);
                return -1;
            }
        }
    }
    return 0;
}

static int generate_phrase_wavs(const char *outdir) {
    if (access("/usr/bin/say", X_OK) != 0) return -1;

    for (int p = 0; p < N_PHRASES; p++) {
        for (int v = 0; v < N_VOICES; v++) {
            char wav_path[1024];
            snprintf(wav_path, sizeof(wav_path), "%s/%s_%d.wav",
                     outdir, PHRASES[p].file_prefix, v + 1);
            if (access(wav_path, F_OK) == 0) continue;

            char cmd[2048];
            snprintf(cmd, sizeof(cmd),
                     "say -v %s -o \"%s\" --data-format=LEI16@16000 \"%s\"",
                     VOICES[v], wav_path, PHRASES[p].tts_text);
            fprintf(stderr, "  TTS phrase: %s → %s\n", PHRASES[p].tts_text, wav_path);
            if (system(cmd) != 0) {
                fprintf(stderr, "Error: say failed for phrase '%s' voice %s\n",
                        PHRASES[p].name, VOICES[v]);
                return -1;
            }
        }
    }
    return 0;
}

/* ========================================================================
 * MFCC Codebook Compilation
 * ======================================================================== */

static int compile_mfcc_codebook(const char *outdir, const char *bin_path) {
    smol_ww_config_t cfg;
    smol_ww_config_default(&cfg);

    smol_ww_detector_t *det = smol_ww_create(N_COEFFS, &cfg);
    if (!det) { fprintf(stderr, "Error: failed to create detector\n"); return -1; }

    for (int s = 0; s < N_SEGMENTS; s++) {
        int loaded = 0;
        for (int v = 0; v < N_VOICES; v++) {
            char wav_path[1024];
            snprintf(wav_path, sizeof(wav_path), "%s/%s_%d.wav",
                     outdir, SEGMENTS[s].file_prefix, v + 1);
            if (access(wav_path, F_OK) != 0) continue;
            if (smol_ww_add_segment_wav(det, SEGMENTS[s].id, wav_path) >= 0)
                loaded++;
        }
        if (loaded > 0)
            fprintf(stderr, "  Segment %d (%s): %d voices, %d frames\n",
                    SEGMENTS[s].id, SEGMENTS[s].tts_text, loaded,
                    det->segments[SEGMENTS[s].id].n_frames);
    }

    for (int p = 0; p < N_PHRASES; p++) {
        smol_ww_add_phrase(det, PHRASES[p].seg_ids, PHRASES[p].n_segs, PHRASES[p].n_prefix);
    }

    if (smol_ww_save(det, bin_path) != 0) {
        fprintf(stderr, "Error: failed to write %s\n", bin_path);
        smol_ww_free(det);
        return -1;
    }

    fprintf(stderr, "Wrote %s (%d segments, %d phrases)\n",
            bin_path, det->n_segments, det->n_phrases);
    smol_ww_free(det);
    return 0;
}

/* ========================================================================
 * Encoder Reference Compilation
 * ======================================================================== */

static int compile_encoder_refs(const char *outdir, const char *bin_path,
                                 const char *model_dir) {
    fprintf(stderr, "\nLoading ASR model for encoder references...\n");
    qwen_ctx_t *ctx = qwen_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Error: failed to load model from %s\n", model_dir);
        return -1;
    }

    int dim = ctx->enc_config.enc_output_dim;
    fprintf(stderr, "  Encoder output_dim = %d\n", dim);

    /* Accumulate mean-pooled embeddings per phrase, averaged across voices */
    float *phrase_refs = (float *)calloc(N_PHRASES * dim, sizeof(float));

    for (int p = 0; p < N_PHRASES; p++) {
        float *accum = phrase_refs + p * dim;
        int n_voices_ok = 0;

        for (int v = 0; v < N_VOICES; v++) {
            char wav_path[1024];
            snprintf(wav_path, sizeof(wav_path), "%s/%s_%d.wav",
                     outdir, PHRASES[p].file_prefix, v + 1);

            int n_samples = 0;
            float *samples = smol_load_wav(wav_path, &n_samples);
            if (!samples || n_samples < 1600) {
                free(samples);
                continue;
            }

            int mel_frames = 0;
            float *mel = smol_mel_spectrogram(samples, n_samples, &mel_frames);
            free(samples);
            if (!mel) continue;

            int seq_len = 0;
            float *enc = qwen_asr_encoder_forward(&ctx->encoder, &ctx->enc_config,
                                                    mel, mel_frames, &seq_len);
            free(mel);
            if (!enc || seq_len == 0) { free(enc); continue; }

            /* Mean-pool and accumulate */
            for (int t = 0; t < seq_len; t++)
                for (int d = 0; d < dim; d++)
                    accum[d] += enc[t * dim + d] / (float)seq_len;

            free(enc);
            n_voices_ok++;
        }

        if (n_voices_ok > 0) {
            for (int d = 0; d < dim; d++)
                accum[d] /= (float)n_voices_ok;

            /* Normalize to unit length */
            float norm = 0.0f;
            for (int d = 0; d < dim; d++) norm += accum[d] * accum[d];
            norm = sqrtf(norm);
            if (norm > 1e-8f)
                for (int d = 0; d < dim; d++) accum[d] /= norm;

            fprintf(stderr, "  Phrase %d \"%s\": %d voices, norm=%.4f\n",
                    p, PHRASES[p].name, n_voices_ok, norm);
        } else {
            fprintf(stderr, "  Warning: no valid WAVs for phrase %d \"%s\"\n",
                    p, PHRASES[p].name);
        }
    }

    /* Write binary: magic + dim + n_phrases + float data */
    FILE *fp = fopen(bin_path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s for writing\n", bin_path);
        free(phrase_refs);
        qwen_free(ctx);
        return -1;
    }

    uint32_t magic = 0x4B575745; /* 'EWWK' */
    uint32_t udim = (uint32_t)dim;
    uint32_t un = (uint32_t)N_PHRASES;
    fwrite(&magic, 4, 1, fp);
    fwrite(&udim, 4, 1, fp);
    fwrite(&un, 4, 1, fp);
    fwrite(phrase_refs, sizeof(float), N_PHRASES * dim, fp);
    fclose(fp);

    fprintf(stderr, "Wrote %s (%d phrases × %d dims)\n", bin_path, N_PHRASES, dim);

    free(phrase_refs);
    qwen_free(ctx);
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char **argv) {
    const char *outdir = "wakewords";
    const char *model_dir = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (argv[i][0] != '-') {
            outdir = argv[i];
        }
    }

    mkdir(outdir, 0755);

    fprintf(stderr, "Generating segment WAVs via TTS...\n");
    if (generate_segment_wavs(outdir) != 0) return 1;

    char mfcc_path[1024];
    snprintf(mfcc_path, sizeof(mfcc_path), "%s/wakewords_mfcc.bin", outdir);
    fprintf(stderr, "\nCompiling MFCC codebook...\n");
    if (compile_mfcc_codebook(outdir, mfcc_path) != 0) return 1;

    if (model_dir) {
        fprintf(stderr, "\nGenerating phrase WAVs via TTS...\n");
        if (generate_phrase_wavs(outdir) != 0) return 1;

        char enc_path[1024];
        snprintf(enc_path, sizeof(enc_path), "%s/wakewords_enc.bin", outdir);
        fprintf(stderr, "\nCompiling encoder references...\n");
        if (compile_encoder_refs(outdir, enc_path, model_dir) != 0) return 1;
    } else {
        fprintf(stderr, "\nSkipping encoder references (no --model-dir)\n");
    }

    fprintf(stderr, "\nDone.\n");
    return 0;
}
