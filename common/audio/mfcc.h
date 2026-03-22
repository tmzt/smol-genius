/*
 * mfcc.h - Incremental MFCC extraction and segment-based wakeword detection
 *
 * MFCC extractor:
 *   Feed 160 samples (one 10ms hop at 16kHz) per call, get n_coeffs MFCCs.
 *   Uses the same mel filterbank as smol_mel_spectrogram().
 *
 * Wakeword detector:
 *   Phoneme-segment matching on MFCC cosine similarity.
 *   Each wakeword is a sequence of 3-4 segment IDs drawn from a shared
 *   codebook (~8 segments). Segments are averaged MFCC frames from
 *   individual TTS recordings of each phoneme unit:
 *
 *     ID  Segment   Example words
 *     0   /h/       hey, hi
 *     1   /eɪ/      hey, okay (tail)
 *     2   /aɪ/      hi, kite
 *     3   /oʊ/      okay (onset)
 *     4   /keɪ/     okay (full)
 *     5   /tʃɪ/     chitin (onset)
 *     6   /kaɪ/     kite (onset)
 *     7   /tɪn/     chitin, kite-inn (tail)
 *
 *   Two-stage detection:
 *     Stage 1 (prefix): match first 1-2 segments of each phrase
 *     Stage 2 (confirm): match remaining segments within a window
 *
 *   Binary format for compiled segment codebook + phrase definitions.
 */

#ifndef SMOL_MFCC_H
#define SMOL_MFCC_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * MFCC Extractor
 * ======================================================================== */

#define SMOL_MFCC_HOP       160   /* 10ms at 16kHz */
#define SMOL_MFCC_WIN       400   /* 25ms window */
#define SMOL_MFCC_N_FFT     400
#define SMOL_MFCC_N_FREQ    201   /* N_FFT/2 + 1 */
#define SMOL_MFCC_N_MEL     128
#define SMOL_MFCC_MAX_COEFFS 20

typedef struct {
    int n_coeffs;                              /* typically 13 */
    float hann[SMOL_MFCC_WIN];                 /* Hann window */
    float mel_filters[SMOL_MFCC_N_MEL * SMOL_MFCC_N_FREQ]; /* mel filterbank */
    float dct[SMOL_MFCC_MAX_COEFFS * SMOL_MFCC_N_MEL];     /* DCT-II matrix */
    float dft_cos[SMOL_MFCC_N_FREQ * SMOL_MFCC_N_FFT];
    float dft_sin[SMOL_MFCC_N_FREQ * SMOL_MFCC_N_FFT];
} smol_mfcc_t;

/* Create MFCC extractor. n_coeffs is typically 13. */
__attribute__((visibility("default")))
smol_mfcc_t *smol_mfcc_create(int n_coeffs);

/* Free MFCC extractor. */
__attribute__((visibility("default")))
void smol_mfcc_free(smol_mfcc_t *m);

/* Process one 400-sample window, write n_coeffs MFCCs to out[]. */
__attribute__((visibility("default")))
void smol_mfcc_compute(const smol_mfcc_t *m,
                        const float *windowed_400,
                        float *out);

/* ========================================================================
 * Wakeword Detector — Segment Codebook
 * ======================================================================== */

#define SMOL_WW_MAX_SEGMENTS     32  /* max segments in codebook */
#define SMOL_WW_MAX_PHRASES      16  /* max wakeword phrases */
#define SMOL_WW_MAX_PHRASE_SEGS   8  /* max segments per phrase */

/* A segment: averaged MFCC frames representing one phoneme unit.
 * Built by averaging across multiple TTS voices of the same phoneme. */
typedef struct {
    float *mfcc;              /* [n_frames * n_coeffs] averaged MFCCs */
    float *norms;             /* [n_frames] per-frame L2 norms */
    int    n_frames;
} smol_ww_segment_t;

/* A phrase: sequence of segment IDs forming a complete wakeword. */
typedef struct {
    int seg_ids[SMOL_WW_MAX_PHRASE_SEGS];
    int n_segs;               /* total segments in phrase */
    int n_prefix_segs;        /* segments to check in stage 1 */
    int total_frames;         /* sum of segment n_frames (precomputed) */
    int prefix_frames;        /* sum of prefix segment n_frames */
} smol_ww_phrase_t;

/* ========================================================================
 * Wakeword Detector — State Machine
 * ======================================================================== */

typedef enum {
    SMOL_WW_WAITING,          /* idle — running prefix detection */
    SMOL_WW_PREFIX_DETECTED,  /* prefix matched — confirming full phrase */
    SMOL_WW_LISTENING,        /* confirmed — ASR should be active */
} smol_ww_state_t;

typedef struct {
    float threshold;          /* full-phrase cosine sim threshold (default 0.80) */
    float prefix_threshold;   /* prefix cosine sim threshold (default 0.72) */
    int   confirm_frames;     /* frames to wait for full confirmation (default 100 = 1s) */
    int   listen_frames;      /* frames to stay active after detection (default 500 = 5s) */
    int   silence_frames;     /* silence frames to end listening (default 200 = 2s) */
    float energy_threshold;   /* MFCC norm threshold for "has energy" (default 50.0) */
} smol_ww_config_t;

typedef struct {
    smol_ww_config_t    config;
    smol_mfcc_t        *mfcc;
    int                 n_coeffs;

    /* Segment codebook */
    smol_ww_segment_t   segments[SMOL_WW_MAX_SEGMENTS];
    int                 n_segments;

    /* Phrase definitions */
    smol_ww_phrase_t    phrases[SMOL_WW_MAX_PHRASES];
    int                 n_phrases;

    /* MFCC ring buffer for sliding window */
    float              *ring;         /* [ring_cap * n_coeffs] */
    float              *ring_norms;   /* [ring_cap] per-frame L2 norms */
    int                 ring_pos;
    int                 ring_len;
    int                 ring_cap;

    /* State machine */
    smol_ww_state_t     state;
    int                 matched_phrase; /* which phrase triggered prefix */
    int                 confirm_remaining;
    int                 listen_remaining;
    int                 silence_count;

    /* Audio windowing buffer */
    float              *audio_buf;    /* [SMOL_MFCC_WIN] circular */
    int                 audio_pos;
    int                 audio_count;
    int                 hop_count;

    /* Sustained energy counter for prefix gating */
    int                 energy_run;

    /* Adaptive RMS gate — calibrated from ambient noise floor */
    float               ambient_rms;       /* measured ambient RMS (0 = not calibrated) */
    float               rms_gate;          /* gate = ambient_rms * multiplier */
    int                 calibrating;       /* 1 = collecting ambient frames */
    float               calib_rms_sum;
    int                 calib_frames;
} smol_ww_detector_t;

/* ========================================================================
 * API — Lifecycle
 * ======================================================================== */

__attribute__((visibility("default")))
void smol_ww_config_default(smol_ww_config_t *cfg);

__attribute__((visibility("default")))
smol_ww_detector_t *smol_ww_create(int n_coeffs, const smol_ww_config_t *cfg);

__attribute__((visibility("default")))
void smol_ww_free(smol_ww_detector_t *det);

/* ========================================================================
 * API — Segment Codebook Building
 * ======================================================================== */

/* Add a segment to the codebook from a single WAV file of one phoneme.
 * Returns the segment ID (index), or -1 on error.
 * Call multiple times with different voice recordings of the same phoneme
 * using the same segment_id to average them together. */
__attribute__((visibility("default")))
int smol_ww_add_segment_wav(smol_ww_detector_t *det,
                             int segment_id,
                             const char *wav_path);

/* Add a segment from raw audio samples. Same averaging behavior. */
__attribute__((visibility("default")))
int smol_ww_add_segment_audio(smol_ww_detector_t *det,
                               int segment_id,
                               const float *samples, int n_samples);

/* Define a wakeword phrase as a sequence of segment IDs.
 * n_prefix_segs: how many segments to check in stage 1 (typically 1-2).
 * Returns phrase index, or -1 on error. */
__attribute__((visibility("default")))
int smol_ww_add_phrase(smol_ww_detector_t *det,
                        const int *seg_ids, int n_segs,
                        int n_prefix_segs);

/* ========================================================================
 * API — Binary Serialization
 * ======================================================================== */

/* Save compiled codebook + phrases to binary file.
 * Format: header, segments (MFCC frames), phrases (segment ID sequences).
 * Returns 0 on success. */
__attribute__((visibility("default")))
int smol_ww_save(const smol_ww_detector_t *det, const char *path);

/* Load compiled codebook + phrases from binary file.
 * Returns 0 on success. */
__attribute__((visibility("default")))
int smol_ww_load(smol_ww_detector_t *det, const char *path);

/* Load from raw bytes (e.g. embedded via include). */
__attribute__((visibility("default")))
int smol_ww_load_bytes(smol_ww_detector_t *det,
                        const uint8_t *data, size_t size);

/* ========================================================================
 * API — Detection
 * ======================================================================== */

/* Process a chunk of raw audio samples (any length).
 * Returns the current detector state after processing. */
__attribute__((visibility("default")))
smol_ww_state_t smol_ww_process(smol_ww_detector_t *det,
                                 const float *samples, int n_samples);

__attribute__((visibility("default")))
smol_ww_state_t smol_ww_get_state(const smol_ww_detector_t *det);

__attribute__((visibility("default")))
void smol_ww_reset(smol_ww_detector_t *det);

__attribute__((visibility("default")))
void smol_ww_activate(smol_ww_detector_t *det);

/* Start ambient noise calibration. Feed audio via smol_ww_process for ~0.5s,
 * then call smol_ww_calibrate_finish to set the adaptive RMS gate.
 * While calibrating, detection is suppressed. */
__attribute__((visibility("default")))
void smol_ww_calibrate_start(smol_ww_detector_t *det);

/* Finish calibration: compute ambient RMS from collected frames and set
 * the gate to ambient_rms * multiplier. Returns the computed ambient RMS. */
__attribute__((visibility("default")))
float smol_ww_calibrate_finish(smol_ww_detector_t *det, float multiplier);

#endif /* SMOL_MFCC_H */
