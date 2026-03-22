/*
 * compile_wakewords.c - Generate and compile MFCC wakeword codebook
 *
 * Uses macOS `say` to generate per-segment WAVs across multiple voices,
 * extracts MFCCs, averages per segment, defines phrases, and writes
 * a binary codebook file.
 *
 * Usage: compile_wakewords [output_dir]
 *        Default output_dir: ./wakewords/
 *
 * Build: cc -O2 -o compile_wakewords tools/compile_wakewords.c \
 *           common/audio/audio.c common/audio/mfcc.c \
 *           -Icommon/audio -Icommon/kernels -Icommon/utils -lm
 *        (or via Makefile target)
 */

/* Provide global expected by audio.c */
int smol_verbose = 0;

#include "mfcc.h"
#include "audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* ========================================================================
 * Segment and Phrase Definitions
 * ======================================================================== */

/* Phoneme segments — each generated as an isolated TTS utterance.
 * ID assignment matches the codebook in mfcc.h. */
typedef struct {
    int id;
    const char *tts_text;   /* text to pass to `say` */
    const char *file_prefix; /* filename prefix for WAVs */
} segment_def_t;

static const segment_def_t SEGMENTS[] = {
    { 0, "hey",   "seg_hey"   },
    { 1, "ay",    "seg_ay"    },  /* /eɪ/ diphthong */
    { 2, "hi",    "seg_hi"    },
    { 3, "oh",    "seg_oh"    },  /* /oʊ/ */
    { 4, "okay",  "seg_okay"  },
    { 5, "chit",  "seg_chit"  },  /* /tʃɪ/ */
    { 6, "kite",  "seg_kite"  },  /* /kaɪ/ */
    { 7, "tin",   "seg_tin"   },  /* /tɪn/ */
};
#define N_SEGMENTS (int)(sizeof(SEGMENTS) / sizeof(SEGMENTS[0]))

/* Wakeword phrases as segment ID sequences.
 * n_prefix: segments checked in stage 1 (fast prefix detection). */
typedef struct {
    const char *name;
    int seg_ids[8];
    int n_segs;
    int n_prefix;
} phrase_def_t;

static const phrase_def_t PHRASES[] = {
    { "hey chitin",   { 0, 5, 7 }, 3, 1 },
    { "hi chitin",    { 2, 5, 7 }, 3, 1 },
    { "okay chitin",  { 4, 5, 7 }, 3, 1 },
    { "hey kite-inn", { 0, 6, 7 }, 3, 1 },
    { "okay kite-inn",{ 4, 6, 7 }, 3, 1 },
};
#define N_PHRASES (int)(sizeof(PHRASES) / sizeof(PHRASES[0]))

/* TTS voices for multi-speaker averaging */
static const char *VOICES[] = {
    "Samantha", "Alex", "Daniel", "Karen", "Moira"
};
#define N_VOICES (int)(sizeof(VOICES) / sizeof(VOICES[0]))

#define N_COEFFS 13

/* ========================================================================
 * TTS Generation
 * ======================================================================== */

static int generate_segment_wavs(const char *outdir) {
    /* Check `say` availability */
    if (access("/usr/bin/say", X_OK) != 0) {
        fprintf(stderr, "Error: /usr/bin/say not found (macOS only)\n");
        return -1;
    }

    for (int s = 0; s < N_SEGMENTS; s++) {
        for (int v = 0; v < N_VOICES; v++) {
            char wav_path[1024];
            snprintf(wav_path, sizeof(wav_path), "%s/%s_%d.wav",
                     outdir, SEGMENTS[s].file_prefix, v + 1);

            /* Skip if already exists */
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

/* ========================================================================
 * Codebook Compilation
 * ======================================================================== */

static int compile_codebook(const char *outdir, const char *bin_path) {
    smol_ww_config_t cfg;
    smol_ww_config_default(&cfg);

    smol_ww_detector_t *det = smol_ww_create(N_COEFFS, &cfg);
    if (!det) {
        fprintf(stderr, "Error: failed to create detector\n");
        return -1;
    }

    /* Load segment WAVs and average across voices */
    for (int s = 0; s < N_SEGMENTS; s++) {
        int loaded = 0;
        for (int v = 0; v < N_VOICES; v++) {
            char wav_path[1024];
            snprintf(wav_path, sizeof(wav_path), "%s/%s_%d.wav",
                     outdir, SEGMENTS[s].file_prefix, v + 1);

            if (access(wav_path, F_OK) != 0) continue;

            if (smol_ww_add_segment_wav(det, SEGMENTS[s].id, wav_path) >= 0) {
                loaded++;
            } else {
                fprintf(stderr, "Warning: failed to load %s\n", wav_path);
            }
        }
        if (loaded > 0) {
            fprintf(stderr, "  Segment %d (%s): %d voices, %d MFCC frames\n",
                    SEGMENTS[s].id, SEGMENTS[s].tts_text,
                    loaded, det->segments[SEGMENTS[s].id].n_frames);
        } else {
            fprintf(stderr, "Warning: no WAVs loaded for segment %d (%s)\n",
                    SEGMENTS[s].id, SEGMENTS[s].tts_text);
        }
    }

    /* Define phrases */
    for (int p = 0; p < N_PHRASES; p++) {
        int idx = smol_ww_add_phrase(det, PHRASES[p].seg_ids,
                                      PHRASES[p].n_segs, PHRASES[p].n_prefix);
        if (idx < 0) {
            fprintf(stderr, "Error: failed to add phrase '%s'\n", PHRASES[p].name);
        } else {
            fprintf(stderr, "  Phrase %d: \"%s\" = [", idx, PHRASES[p].name);
            for (int s = 0; s < PHRASES[p].n_segs; s++) {
                if (s > 0) fprintf(stderr, ", ");
                fprintf(stderr, "%d", PHRASES[p].seg_ids[s]);
            }
            fprintf(stderr, "], prefix=%d\n", PHRASES[p].n_prefix);
        }
    }

    /* Save binary */
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
 * Main
 * ======================================================================== */

int main(int argc, char **argv) {
    const char *outdir = argc > 1 ? argv[1] : "wakewords";

    /* Create output directory */
    mkdir(outdir, 0755);

    fprintf(stderr, "Generating segment WAVs via TTS...\n");
    if (generate_segment_wavs(outdir) != 0) return 1;

    char bin_path[1024];
    snprintf(bin_path, sizeof(bin_path), "%s/wakewords_mfcc.bin", outdir);

    fprintf(stderr, "\nCompiling MFCC codebook...\n");
    if (compile_codebook(outdir, bin_path) != 0) return 1;

    fprintf(stderr, "\nDone. Load at runtime with:\n");
    fprintf(stderr, "  smol_ww_load(det, \"%s\");\n", bin_path);

    return 0;
}
