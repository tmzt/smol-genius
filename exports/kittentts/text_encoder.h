/*
 * text_encoder.h - KittenTTS text encoder (Conv1D + BiLSTM)
 */

#ifndef KTTS_TEXT_ENCODER_H
#define KTTS_TEXT_ENCODER_H

#include "kittentts.h"

/* Run text encoder forward pass.
 * token_ids: [seq_len] phoneme token IDs
 * plbert_out: [seq_len, 128] PL-BERT output (added as residual)
 * out: [seq_len, 128] output
 * scratch: workspace */
void ktts_text_encoder_forward(const ktts_text_enc_t *model, const int *token_ids,
                                const float *plbert_out, int seq_len,
                                float *out, float *scratch);

#endif /* KTTS_TEXT_ENCODER_H */
