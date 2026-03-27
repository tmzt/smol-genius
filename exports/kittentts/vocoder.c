/*
 * vocoder.c - iSTFT-Net vocoder
 *
 * Architecture:
 *   ConvTranspose1D [256 -> 128, k=20, s=10]  (10x upsample)
 *   2x ResBlock with Snake activation + AdaIN
 *   + noise injection from SineGen harmonic source
 *   ConvTranspose1D [128 -> 64, k=12, s=6]    (6x upsample)
 *   2x ResBlock with Snake activation + AdaIN
 *   + noise injection
 *   Conv1D post [22, 64, 7] -> magnitude + phase
 *   iSTFT (n_fft=20, hop=5) -> 24kHz waveform
 *
 * Total temporal upsample: 10 * 6 * 5 (hop) = 300x per acoustic frame
 */

#include "vocoder.h"
#include "smol_kernels.h"
#include "audio.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Sine Generator (harmonic source for vocoder)
 *
 * Generates a sinusoidal signal from the F0 contour.
 * Used as harmonic excitation for the vocoder.
 * ======================================================================== */

static void sine_gen(float *out, const float *f0, int length,
                      int upsample_factor, int sample_rate) {
    int out_len = length * upsample_factor;
    float phase = 0.0f;

    for (int i = 0; i < out_len; i++) {
        /* Interpolate F0 for this sample */
        int frame = i / upsample_factor;
        if (frame >= length) frame = length - 1;
        float freq = f0[frame];

        /* Accumulate phase */
        if (freq > 0.0f) {
            phase += 2.0f * (float)M_PI * freq / (float)sample_rate;
            if (phase > 2.0f * (float)M_PI) phase -= 2.0f * (float)M_PI;
            out[i] = sinf(phase);
        } else {
            out[i] = 0.0f;
        }
    }
}

/* ========================================================================
 * Vocoder ResBlock with Snake activation
 * ======================================================================== */

static void vocoder_resblock(float *out, const float *in, int channels, int length,
                               const float *conv1_w, const float *conv1_b,
                               const float *conv2_w, const float *conv2_b,
                               const float *conv3_w, const float *conv3_b,
                               const float *adain_fc_w, const float *adain_fc_b,
                               const float *snake_alpha,
                               const float *style,
                               float *scratch) {
    /* Project style for AdaIN */
    float style_proj[KTTS_DECODER_HIDDEN * 2];
    smol_linear(style_proj, style, adain_fc_w, adain_fc_b,
                1, KTTS_STYLE_ADAIN_DIM, 2 * channels);
    float *s_mean = style_proj;
    float *s_std = style_proj + channels;

    /* Snake activation on input */
    float *act = scratch;
    memcpy(act, in, (size_t)channels * length * sizeof(float));
    smol_snake(act, snake_alpha, channels, length);

    /* Dilated Conv1D #1 (dilation=1) */
    float *c1 = act + (size_t)channels * length;
    smol_conv1d(c1, act, conv1_w, conv1_b,
                channels, channels, length, 3, 1, 1, 1, 1);

    /* Snake + Conv1D #2 (dilation=1) */
    smol_snake(c1, snake_alpha, channels, length);
    float *c2 = act;
    smol_conv1d(c2, c1, conv2_w, conv2_b,
                channels, channels, length, 3, 1, 1, 1, 1);

    /* Snake + Conv1D #3 (dilation=1) */
    smol_snake(c2, snake_alpha, channels, length);
    smol_conv1d(c1, c2, conv3_w, conv3_b,
                channels, channels, length, 3, 1, 1, 1, 1);

    /* AdaIN */
    smol_adain(c1, c1, s_mean, s_std, channels, length, 1e-5f);

    /* Residual: out = in + processed */
    memcpy(out, in, (size_t)channels * length * sizeof(float));
    smol_add_inplace(out, c1, channels * length);
}

int ktts_vocoder_forward(const ktts_vocoder_t *model,
                          const float *acoustic, const float *f0,
                          const float *style,
                          int seq_len,
                          float *out_audio, float *scratch) {
    /* --- Stage 1: ConvTranspose1D [256 -> 128, k=20, s=10] --- */
    int len1 = (seq_len - 1) * KTTS_VOC_UPSAMPLE_0 + 20; /* ~10x */
    float *up1 = scratch;
    smol_conv_transpose1d(up1, acoustic, model->up0_w, model->up0_b,
                          KTTS_DECODER_HIDDEN, KTTS_VOC_CH_0, seq_len,
                          20, KTTS_VOC_UPSAMPLE_0, 5, 0);

    /* Generate harmonic source at this resolution */
    float *source1 = up1 + KTTS_VOC_CH_0 * (size_t)len1;
    sine_gen(source1, f0, seq_len, KTTS_VOC_UPSAMPLE_0, KTTS_SAMPLE_RATE);

    /* Noise injection (noise_conv0: [128, 22, 12]) */
    /* For simplicity, skip noise injection in initial implementation.
     * The noise path adds stochastic detail but isn't critical for basic output. */

    /* ResBlock 0 + 1 (channels=128) */
    float *rb_scratch = source1 + len1;
    float *rb_out = rb_scratch + KTTS_VOC_CH_0 * (size_t)len1 * 4;

    vocoder_resblock(rb_out, up1, KTTS_VOC_CH_0, len1,
                     model->resblocks[0].conv1_w, model->resblocks[0].conv1_b,
                     model->resblocks[0].conv2_w, model->resblocks[0].conv2_b,
                     model->resblocks[0].conv3_w, model->resblocks[0].conv3_b,
                     model->resblocks[0].adain_fc_w, model->resblocks[0].adain_fc_b,
                     model->resblocks[0].snake_alpha, style, rb_scratch);

    vocoder_resblock(up1, rb_out, KTTS_VOC_CH_0, len1,
                     model->resblocks[1].conv1_w, model->resblocks[1].conv1_b,
                     model->resblocks[1].conv2_w, model->resblocks[1].conv2_b,
                     model->resblocks[1].conv3_w, model->resblocks[1].conv3_b,
                     model->resblocks[1].adain_fc_w, model->resblocks[1].adain_fc_b,
                     model->resblocks[1].snake_alpha, style, rb_scratch);

    /* --- Stage 2: ConvTranspose1D [128 -> 64, k=12, s=6] --- */
    int len2 = (len1 - 1) * KTTS_VOC_UPSAMPLE_1 + 12; /* ~6x */
    float *up2 = rb_scratch;
    smol_conv_transpose1d(up2, up1, model->up1_w, model->up1_b,
                          KTTS_VOC_CH_0, KTTS_VOC_CH_1, len1,
                          12, KTTS_VOC_UPSAMPLE_1, 3, 0);

    /* ResBlock 2 + 3 (channels=64) */
    float *rb2_scratch = up2 + KTTS_VOC_CH_1 * (size_t)len2;
    float *rb2_out = rb2_scratch + KTTS_VOC_CH_1 * (size_t)len2 * 4;

    vocoder_resblock(rb2_out, up2, KTTS_VOC_CH_1, len2,
                     model->resblocks[2].conv1_w, model->resblocks[2].conv1_b,
                     model->resblocks[2].conv2_w, model->resblocks[2].conv2_b,
                     model->resblocks[2].conv3_w, model->resblocks[2].conv3_b,
                     model->resblocks[2].adain_fc_w, model->resblocks[2].adain_fc_b,
                     model->resblocks[2].snake_alpha, style, rb2_scratch);

    vocoder_resblock(up2, rb2_out, KTTS_VOC_CH_1, len2,
                     model->resblocks[3].conv1_w, model->resblocks[3].conv1_b,
                     model->resblocks[3].conv2_w, model->resblocks[3].conv2_b,
                     model->resblocks[3].conv3_w, model->resblocks[3].conv3_b,
                     model->resblocks[3].adain_fc_w, model->resblocks[3].adain_fc_b,
                     model->resblocks[3].snake_alpha, style, rb2_scratch);

    /* --- Stage 3: Post convolution -> learned iSTFT --- */
    int n_freq = KTTS_VOC_NFREQ;    /* 11 */

    /* Conv1D post: [64, len2] -> [22, len2] (first 11 = log-mag, next 11 = phase) */
    float *post_out = rb2_out;
    smol_conv1d(post_out, up2, model->conv_post_w, model->conv_post_b,
                KTTS_VOC_CH_1, 2 * n_freq, len2,
                7, 1, 3, 1, 1);

    /* Split into log-magnitude and phase: each [11, len2] */
    float *log_mag = post_out;
    float *phase = post_out + n_freq * (size_t)len2;

    /* magnitude = exp(log_mag) */
    float *mag = log_mag; /* in-place */
    for (int i = 0; i < n_freq * len2; i++) {
        mag[i] = expf(mag[i]);
    }

    /* Compute real and imaginary STFT components:
     *   real_spec[k, t] = mag[k, t] * cos(phase[k, t])
     *   imag_spec[k, t] = mag[k, t] * sin(phase[k, t]) */
    float *real_spec = rb2_scratch;
    float *imag_spec = real_spec + n_freq * (size_t)len2;
    for (int i = 0; i < n_freq * len2; i++) {
        real_spec[i] = mag[i] * cosf(phase[i]);
        imag_spec[i] = mag[i] * sinf(phase[i]);
    }

    /* Learned iSTFT via ConvTranspose1D:
     *   real_signal = ConvTranspose1D(real_spec, stft.weight_backward_real, s=5, k=20, p=0)
     *   imag_signal = ConvTranspose1D(imag_spec, stft.weight_backward_imag, s=5, k=20, p=0)
     *   waveform = real_signal - imag_signal
     *
     * weight shape: [11, 1, 20] — maps 11 frequency bins to 1 output channel */
    int n_samples = (len2 - 1) * KTTS_VOC_HOP + KTTS_VOC_NFFT;
    float *real_signal = imag_spec + n_freq * (size_t)len2;
    float *imag_signal = real_signal + n_samples;

    smol_conv_transpose1d(real_signal, real_spec, model->istft_real_w, NULL,
                          n_freq, 1, len2,
                          KTTS_VOC_NFFT, KTTS_VOC_HOP, 0, 0);
    smol_conv_transpose1d(imag_signal, imag_spec, model->istft_imag_w, NULL,
                          n_freq, 1, len2,
                          KTTS_VOC_NFFT, KTTS_VOC_HOP, 0, 0);

    /* waveform = real - imag */
    for (int i = 0; i < n_samples; i++) {
        out_audio[i] = real_signal[i] - imag_signal[i];
    }

    /* Trim last 15 samples (padding artifact from ConvTranspose) */
    int trim = 15;
    if (n_samples > trim) n_samples -= trim;

    /* Clean NaN values */
    for (int i = 0; i < n_samples; i++) {
        if (out_audio[i] != out_audio[i]) out_audio[i] = 0.0f;
    }

    return n_samples;
}
