/*
 * audio.h - Generic audio I/O and mel spectrogram computation
 */

#ifndef SMOL_AUDIO_H
#define SMOL_AUDIO_H

#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

/* ========================================================================
 * Live Audio Buffer (incremental stdin streaming)
 * ======================================================================== */

typedef struct {
    /* Written by reader thread under mutex */
    float *samples;
    int64_t sample_offset;      /* global index of samples[0] */
    int64_t n_samples;          /* number of valid samples in buffer */
    int64_t capacity;           /* allocated capacity (in samples) */
    int eof;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_t thread;
} smol_live_audio_t;

/* ========================================================================
 * WAV File I/O
 * ======================================================================== */

/* Load a WAV file, returns mono float32 samples in [-1,1] at 16kHz.
 * Handles: 16-bit PCM, mono or stereo (mixed to mono).
 * Resamples to 16kHz if needed.
 * Returns NULL on error. Caller must free returned buffer. */
__attribute__((visibility("default")))
float *smol_load_wav(const char *path, int *out_n_samples);

/* Parse a WAV file from a memory buffer. Caller must free returned buffer. */
__attribute__((visibility("default")))
float *smol_parse_wav_buffer(const uint8_t *data, size_t size, int *out_n_samples);

/* Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono).
 * Returns NULL on error. Caller must free returned buffer. */
__attribute__((visibility("default")))
float *smol_read_pcm_stdin(int *out_n_samples);

/* ========================================================================
 * Mel Spectrogram
 * ======================================================================== */

/* Compute log-mel spectrogram from audio samples.
 * Uses dynamic maximum for clamping.
 * samples: mono float32 at 16kHz
 * n_samples: number of samples
 * out_frames: set to number of mel frames produced
 * Returns: [128, n_frames] mel spectrogram (caller must free)
 * Note: Returns in [mel_bins, frames] layout for Conv2D compatibility. */
__attribute__((visibility("default")))
float *smol_mel_spectrogram(const float *samples, int n_samples, int *out_frames);

/* ========================================================================
 * WAV Writing
 * ======================================================================== */

/* Write float32 samples as a 16-bit PCM WAV file.
 * samples: mono float32 in [-1,1]
 * sample_rate: output sample rate (e.g. 24000 for TTS)
 * path: output file path, or "-" for stdout
 * Returns 0 on success, -1 on error. */
__attribute__((visibility("default")))
int smol_write_wav(const char *path, const float *samples, int n_samples, int sample_rate);

/* ========================================================================
 * Inverse STFT
 * ======================================================================== */

/* Reconstruct time-domain audio from magnitude and phase spectrograms.
 * magnitude: [n_freq, n_frames]
 * phase:     [n_freq, n_frames]
 * n_freq = n_fft/2 + 1
 * out_audio: output buffer (must hold at least (n_frames - 1) * hop_size + n_fft samples)
 * Returns number of output samples. */
__attribute__((visibility("default")))
int smol_istft(float *out_audio,
               const float *magnitude, const float *phase,
               int n_freq, int n_frames, int n_fft, int hop_size);

/* ========================================================================
 * Live Audio Streaming
 * ======================================================================== */

/* Start a reader thread that incrementally fills a live audio buffer from stdin.
 * Detects WAV vs raw s16le. For WAV, requires 16kHz sample rate.
 * Returns NULL on error. Caller must call smol_live_audio_free() when done. */
__attribute__((visibility("default")))
smol_live_audio_t *smol_live_audio_start_stdin(void);

/* Join reader thread and free all resources. */
__attribute__((visibility("default")))
void smol_live_audio_free(smol_live_audio_t *la);

#endif /* SMOL_AUDIO_H */
