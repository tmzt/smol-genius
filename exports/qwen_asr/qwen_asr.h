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
#include <pthread.h>
#include "encoder.h"
#include "../../common/decoder/qkn_decoder.h"
#include "../../common/utils/tokenizer.h"
#include "../../common/audio/audio.h"

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
 * Token Callback (streaming output)
 * ======================================================================== */

typedef void (*qwen_token_cb)(const char *piece, void *userdata);

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

/* Global verbose flag */
extern int qwen_verbose;

/* Monitor mode: show inline Unicode symbols on stderr for streaming diagnostics. */
extern int qwen_monitor;

/* Thread count control */
__attribute__((visibility("default")))
void qwen_set_threads(int n);

__attribute__((visibility("default")))
int qwen_get_num_cpus(void);

#endif /* QWEN_ASR_H */
