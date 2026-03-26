/*
 * plbert.h - PL-BERT (ALBERT-style) phoneme language model
 */

#ifndef KTTS_PLBERT_H
#define KTTS_PLBERT_H

#include "kittentts.h"

/* Run PL-BERT forward pass.
 * token_ids: [seq_len] phoneme token IDs
 * out: [seq_len, KTTS_BERT_DIM] output embeddings
 * scratch: workspace buffer (at least KTTS_BERT_HIDDEN * KTTS_BERT_MAX_POS * 2 floats) */
void ktts_plbert_forward(const ktts_plbert_t *model, const int *token_ids,
                          int seq_len, float *out, float *scratch);

#endif /* KTTS_PLBERT_H */
