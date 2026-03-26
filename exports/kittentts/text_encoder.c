/*
 * text_encoder.c - KittenTTS text encoder
 *
 * Architecture:
 *   Phoneme embedding [178, 128]
 *   -> 2x Conv1D [128, 128, kernel=5, pad=2] + LayerNorm
 *   -> Add PL-BERT output (residual)
 *   -> BiLSTM (input=128, hidden=64/dir -> output=128)
 */

#include "text_encoder.h"
#include "smol_kernels.h"
#include <string.h>

void ktts_text_encoder_forward(const ktts_text_enc_t *model, const int *token_ids,
                                const float *plbert_out, int seq_len,
                                float *out, float *scratch) {
    int dim = KTTS_TEXT_ENC_DIM;   /* 128 */
    int lstm_h = KTTS_TEXT_ENC_LSTM_H; /* 64 per direction */

    /* Partition scratch: we need [seq, 128] for conv I/O */
    float *conv_in = scratch;                              /* [128, seq] channel-first */
    float *conv_out = conv_in + (size_t)dim * seq_len;     /* [128, seq] */
    float *lstm_in = conv_out + (size_t)dim * seq_len;     /* [seq, 128] row-major */

    /* --- Step 1: Phoneme embedding lookup --- */
    /* Store in row-major [seq, 128] in lstm_in temporarily */
    for (int t = 0; t < seq_len; t++) {
        int tok_id = token_ids[t];
        if (tok_id < 0 || tok_id >= KTTS_PHONEME_VOCAB) tok_id = 0;
        memcpy(lstm_in + t * dim, model->phoneme_emb + tok_id * dim, dim * sizeof(float));
    }

    /* Transpose to channel-first [128, seq] for Conv1D */
    for (int c = 0; c < dim; c++) {
        for (int t = 0; t < seq_len; t++) {
            conv_in[c * seq_len + t] = lstm_in[t * dim + c];
        }
    }

    /* --- Step 2: 2x Conv1D + LayerNorm --- */
    for (int layer = 0; layer < KTTS_TEXT_ENC_CNN_N; layer++) {
        smol_conv1d(conv_out, conv_in,
                    model->conv_w[layer], model->conv_b[layer],
                    dim, dim, seq_len,
                    KTTS_TEXT_ENC_CNN_K, /*stride=*/1, /*padding=*/2,
                    /*dilation=*/1, /*groups=*/1);

        /* Transpose to row-major for LayerNorm: [seq, 128] */
        for (int c = 0; c < dim; c++) {
            for (int t = 0; t < seq_len; t++) {
                lstm_in[t * dim + c] = conv_out[c * seq_len + t];
            }
        }

        smol_layer_norm(lstm_in, lstm_in,
                        model->conv_ln_w[layer], model->conv_ln_b[layer],
                        seq_len, dim, 1e-5f);

        /* Transpose back to channel-first for next conv */
        if (layer < KTTS_TEXT_ENC_CNN_N - 1) {
            for (int c = 0; c < dim; c++) {
                for (int t = 0; t < seq_len; t++) {
                    conv_in[c * seq_len + t] = lstm_in[t * dim + c];
                }
            }
        }
    }

    /* --- Step 3: Add PL-BERT residual --- */
    /* lstm_in is [seq, 128] row-major from the final LayerNorm */
    smol_add_inplace(lstm_in, plbert_out, seq_len * dim);

    /* --- Step 4: Bidirectional LSTM --- */
    /* Input: [seq, 128], Output: [seq, 128] (64 + 64 per direction) */
    smol_bilstm_forward(out, lstm_in,
                         model->lstm_W_ih_fwd, model->lstm_W_hh_fwd,
                         model->lstm_b_ih_fwd, model->lstm_b_hh_fwd,
                         model->lstm_W_ih_bwd, model->lstm_W_hh_bwd,
                         model->lstm_b_ih_bwd, model->lstm_b_hh_bwd,
                         seq_len, dim, lstm_h);
}
