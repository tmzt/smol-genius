/*
 * prosody.h - Duration, F0, and noise energy prediction
 */

#ifndef KTTS_PROSODY_H
#define KTTS_PROSODY_H

#include "kittentts.h"

/* Run prosody prediction.
 * text_enc_out: [seq_len, 128] from text encoder
 * style: [256] style embedding
 * speed: speech rate multiplier
 *
 * Outputs:
 *   durations: [seq_len] predicted duration per phoneme (in frames)
 *   f0: [expanded_len] F0 contour
 *   noise: [expanded_len] noise energy
 *   expanded_text: [expanded_len, 128] duration-expanded text features
 *   Returns expanded_len (total frames after duration expansion). */
int ktts_prosody_forward(const ktts_prosody_t *model,
                          const float *text_enc_out, const float *style,
                          int seq_len, float speed,
                          int *durations, float *f0, float *noise,
                          float *expanded_text, float *scratch);

#endif /* KTTS_PROSODY_H */
