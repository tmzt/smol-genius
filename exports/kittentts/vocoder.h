/*
 * vocoder.h - iSTFT-Net vocoder (ConvTranspose upsample + Snake + AdaIN)
 */

#ifndef KTTS_VOCODER_H
#define KTTS_VOCODER_H

#include "kittentts.h"

/* Run vocoder: acoustic features -> waveform.
 * acoustic: [256, seq_len] channel-first acoustic decoder output
 * f0: [seq_len] F0 contour for harmonic source
 * style: [256] style embedding
 * out_audio: output buffer (must be large enough for upsampled audio)
 * Returns number of output audio samples. */
int ktts_vocoder_forward(const ktts_vocoder_t *model,
                          const float *acoustic, const float *f0,
                          const float *style,
                          int seq_len,
                          float *out_audio, float *scratch);

#endif /* KTTS_VOCODER_H */
