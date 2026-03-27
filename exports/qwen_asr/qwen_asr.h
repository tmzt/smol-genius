/*
 * qwen_asr.h - Qwen3-ASR orchestration API
 *
 * High-level API for loading and running the full ASR pipeline:
 * audio -> mel -> encoder -> prompt construction -> decoder -> text.
 *
 * Encoder lives in encoder.{h,c}, decoder in common/decoder/qkn_decoder.{h,c}.
 * This file ties them together with streaming, segmentation, and callbacks.
 */

#ifndef QWEN_ASR_H
#define QWEN_ASR_H

#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include "encoder.h"
#include "../../common/decoder/qkn_decoder.h"
#include "../../common/utils/tokenizer.h"
#include "../../common/audio/audio.h"
#include "../../common/audio/mfcc.h"

/* ========================================================================
 * Constants
 * ======================================================================== */

#define QWEN_SAMPLE_RATE      16000
#define QWEN_MEL_BINS         128
#define QWEN_HOP_LENGTH       160
#define QWEN_WINDOW_SIZE      400
#define QWEN_VOCAB_SIZE       151936

/* Special token IDs */
#define QWEN_TOKEN_IM_START     151644
#define QWEN_TOKEN_IM_END       151645
#define QWEN_TOKEN_ENDOFTEXT    151643
#define QWEN_TOKEN_AUDIO_START  151669
#define QWEN_TOKEN_AUDIO_END    151670
#define QWEN_TOKEN_AUDIO_PAD    151676
#define QWEN_TOKEN_ASR_TEXT     151704

/* Conv2D stem constants */
#define QWEN_CONV_HIDDEN      480

/* ========================================================================
 * Callbacks
 * ======================================================================== */

/* Token callback: each decoded text token during generation. */
typedef void (*qwen_token_cb)(const char *piece, void *userdata);

/* ========================================================================
 * Pipeline Control (atomic, cross-thread)
 * ======================================================================== */

typedef enum {
    QWEN_PIPELINE_IDLE           = 0,  /* wakeword-gated, dropping audio */
    QWEN_PIPELINE_PREFIX_DETECTED = 1, /* prefix matched, confirming */
    QWEN_PIPELINE_LISTENING      = 2,  /* gate open, ASR active */
    QWEN_PIPELINE_PROCESSING     = 3,  /* finalizing ASR result */
} qwen_pipeline_state_t;

typedef enum {
    QWEN_CONTROL_NONE            = 0,
    QWEN_CONTROL_START_LISTENING = 1,  /* bypass wakeword, open gate */
    QWEN_CONTROL_STOP_LISTENING  = 2,  /* close gate, return to idle */
    QWEN_CONTROL_STOP_AND_CLEAR  = 3,  /* close gate + clear text */
} qwen_control_action_t;

/* ========================================================================
 * Live Audio — type alias for smol_live_audio_t
 * ======================================================================== */

typedef smol_live_audio_t qwen_live_audio_t;

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    /* Model components */
    qwen_asr_enc_config_t enc_config;
    qwen_asr_encoder_t    encoder;
    qkn_config_t          dec_config;
    qkn_ctx_t             dec_ctx;

    /* Model files (kept open for mmap) */
    void *safetensors;         /* multi_safetensors_t* */
    char model_dir[512];

    /* Tokenizer (loaded once) */
    smol_tokenizer_t *tokenizer;

    /* Token streaming callback (optional) */
    qwen_token_cb token_cb;
    void *token_cb_userdata;

    /* Segmentation settings */
    float segment_sec;             /* 0 = no splitting, default full-audio decode */
    float search_sec;              /* segment-cutting silence search window ± seconds (default 3) */

    /* Streaming settings */
    float stream_chunk_sec;        /* chunk interval in seconds (default 2.0) */
    int stream_rollback;           /* tokens to roll back per chunk (default 5) */
    int stream_unfixed_chunks;     /* cold-start chunks without prefix (default 2) */
    int stream_max_new_tokens;     /* max generated tokens per streaming step (default 32) */
    int past_text_conditioning;    /* 1=enable past text conditioning */
    int skip_silence;              /* 1=drop long silent spans before transcription */

    /* Optional prompt/language controls */
    char *prompt;                  /* system prompt text (UTF-8) */
    char *force_language;          /* normalized language name, or NULL */
    int *prompt_tokens;            /* cached token ids for prompt text */
    int n_prompt_tokens;
    int *force_prompt_tokens;      /* cached token ids for "language X" + <asr_text> */
    int n_force_prompt_tokens;
    int prompt_tokens_ready;       /* cache valid flag */

    /* Wakeword detector (optional — NULL if disabled) */
    smol_ww_detector_t *wakeword;

    /* Encoder-based wakeword reference embeddings (optional) */
    float *ww_enc_ref;         /* [ww_enc_n_phrases * ww_enc_dim] */
    int    ww_enc_dim;         /* encoder output_dim */
    int    ww_enc_n_phrases;   /* number of reference phrases */
    float  ww_enc_threshold;   /* cosine sim threshold (default 0.85) */

    /* Pipeline state (atomic, cross-thread safe) */
    _Atomic uint32_t pipeline_state;   /* qwen_pipeline_state_t */
    _Atomic uint32_t control_action;   /* qwen_control_action_t */

    /* External shared atomics (optional, set via qwen_set_shared_atomics).
     * When non-NULL, pipeline writes state/diagnostics directly here.
     * Layout: [0]=pipeline_state, [1]=energy_gate, [2]=ww_gate,
     *         [8]=mel_count, [9]=encoder_count, [10]=token_count, [11]=audio_chunks */
    _Atomic uint32_t *shared_atomics;
    int shared_atomics_len;

    /* Per-run performance stats */
    double perf_total_ms;
    int perf_text_tokens;
    double perf_audio_ms;
    double perf_encode_ms;
    double perf_decode_ms;
} qwen_ctx_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory */
__attribute__((visibility("default")))
qwen_ctx_t *qwen_load(const char *model_dir);

/* Free all resources */
__attribute__((visibility("default")))
void qwen_free(qwen_ctx_t *ctx);

/* Set a callback to receive each decoded token as it's generated. */
__attribute__((visibility("default")))
void qwen_set_token_callback(qwen_ctx_t *ctx, qwen_token_cb cb, void *userdata);

/* Set optional system prompt text (UTF-8). Returns 0 on success. */
__attribute__((visibility("default")))
int qwen_set_prompt(qwen_ctx_t *ctx, const char *prompt);

/* Set optional forced language. Returns 0 on success, -1 if unsupported. */
__attribute__((visibility("default")))
int qwen_set_force_language(qwen_ctx_t *ctx, const char *language);

/* Comma-separated supported language names. */
__attribute__((visibility("default")))
const char *qwen_supported_languages_csv(void);

/* Transcribe a WAV file, returns allocated string (caller must free). */
__attribute__((visibility("default")))
char *qwen_transcribe(qwen_ctx_t *ctx, const char *wav_path);

/* Transcribe from raw audio samples (mono float32, 16kHz). */
__attribute__((visibility("default")))
char *qwen_transcribe_audio(qwen_ctx_t *ctx, const float *samples, int n_samples);

/* Transcribe from stdin (auto-detect WAV or raw s16le). */
__attribute__((visibility("default")))
char *qwen_transcribe_stdin(qwen_ctx_t *ctx);

/* Streaming transcription with prefix rollback. */
__attribute__((visibility("default")))
char *qwen_transcribe_stream(qwen_ctx_t *ctx, const float *samples, int n_samples);

/* Live streaming transcription from an incrementally-filled audio source. */
__attribute__((visibility("default")))
char *qwen_transcribe_stream_live(qwen_ctx_t *ctx, qwen_live_audio_t *live);

/* Persistent live streaming: runs back-to-back sessions. Never returns. */
__attribute__((visibility("default")))
void qwen_transcribe_stream_live_persistent(qwen_ctx_t *ctx, qwen_live_audio_t *live);

/* Set the streaming chunk interval in seconds (default 0.5). */
__attribute__((visibility("default")))
void qwen_set_stream_chunk_sec(qwen_ctx_t *ctx, float sec);

/* ========================================================================
 * Wakeword Integration (optional — all NULL/no-op by default)
 * ======================================================================== */

/* Load a wakeword codebook. Enables wakeword gating in persistent mode.
 * Pass NULL to disable. Returns 0 on success. */
__attribute__((visibility("default")))
int qwen_load_wakeword(qwen_ctx_t *ctx, const char *codebook_path);

/* Load wakeword codebook from raw bytes (e.g. embedded binary). */
__attribute__((visibility("default")))
int qwen_load_wakeword_bytes(qwen_ctx_t *ctx, const uint8_t *data, size_t size);

/* Load encoder-based wakeword reference embeddings from raw bytes.
 * Format: u32 magic 'EWWK', u32 dim, u32 n_phrases, then
 * n_phrases * dim floats. Returns 0 on success. */
__attribute__((visibility("default")))
int qwen_load_wakeword_enc_bytes(qwen_ctx_t *ctx, const uint8_t *data, size_t size);

/* ========================================================================
 * Pipeline Control (cross-thread safe, for use with persistent mode)
 * ======================================================================== */

/* Post a control action (caller thread — UI, system, etc.).
 * Latest wins (single atomic, swap-to-consume). */
__attribute__((visibility("default")))
void qwen_post_control(qwen_ctx_t *ctx, qwen_control_action_t action);

/* Read current pipeline state (caller thread). */
__attribute__((visibility("default")))
qwen_pipeline_state_t qwen_get_pipeline_state(const qwen_ctx_t *ctx);

/* Set external shared atomics array. The pipeline writes state and
 * diagnostics directly to this array (no copying). The array must
 * outlive the pipeline. Pass NULL to disable. */
__attribute__((visibility("default")))
void qwen_set_shared_atomics(qwen_ctx_t *ctx, _Atomic uint32_t *atomics, int len);


/* ---- Split encoder API for GPU acceleration ---- */

/* Run mel spectrogram on raw audio samples.
 * Returns mel [128, *out_mel_frames] (caller must free). */
__attribute__((visibility("default")))
float *qwen_mel_spectrogram(const float *samples, int n_samples, int *out_mel_frames);

/* Run conv stem only: mel → token embeddings [*out_tokens, d_model].
 * Returns token embeddings (caller must free).
 * Sets *out_tokens, *out_d_model. */
__attribute__((visibility("default")))
float *qwen_encoder_conv_stem(qwen_ctx_t *ctx,
                               const float *mel, int mel_frames,
                               int *out_tokens, int *out_d_model);

/* Run decoder with externally-provided encoder output.
 * enc_output: [enc_seq_len, output_dim] from GPU encoder.
 * Returns text (caller must free), or NULL on failure. */
__attribute__((visibility("default")))
char *qwen_decode_with_encoder_output(qwen_ctx_t *ctx,
                                       const float *enc_output, int enc_seq_len,
                                       int output_dim);

/* ========================================================================
 * Globals
 * ======================================================================== */

extern int qwen_verbose;
extern int qwen_monitor;

__attribute__((visibility("default")))
void qwen_set_threads(int n);

__attribute__((visibility("default")))
int qwen_get_num_cpus(void);

#endif /* QWEN_ASR_H */
