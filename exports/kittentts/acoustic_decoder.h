/*
 * acoustic_decoder.h - KittenTTS acoustic decoder
 */

#ifndef KTTS_ACOUSTIC_DECODER_H
#define KTTS_ACOUSTIC_DECODER_H

#include "kittentts.h"

/* Run acoustic decoder.
 * expanded_text: [expanded_len, 128] duration-expanded text features
 * f0: [expanded_len] F0 contour
 * noise: [expanded_len] noise energy
 * style: [256] style embedding
 * out: [256, expanded_len] acoustic features (channel-first)
 * scratch: workspace */
void ktts_acoustic_decoder_forward(const ktts_acoustic_dec_t *model,
                                    const float *expanded_text,
                                    const float *f0, const float *noise,
                                    const float *style,
                                    int expanded_len,
                                    float *out, float *scratch);

#endif /* KTTS_ACOUSTIC_DECODER_H */
