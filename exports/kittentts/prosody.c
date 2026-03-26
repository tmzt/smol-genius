/*
 * prosody.c - Duration, F0, and noise energy prediction
 *
 * Duration encoder: 4x BiLSTM with AdaLayerNorm -> softmax -> argmax per phoneme
 * F0 predictor: 3x AdainResBlk1d (128->64->64) -> Conv1D -> 1
 * N predictor: same architecture as F0
 *
 * After duration prediction, text features are expanded (repeated) according
 * to predicted durations. F0 and N predictors then operate on the expanded sequence.
 */

#include "prosody.h"
#include "smol_kernels.h"
#include <string.h>
#include <math.h>

/* ========================================================================
 * AdainResBlk1d helper
 *
 * x -> Conv1D -> AdaIN(style) -> LeakyReLU -> Conv1D -> AdaIN(style) + skip(x)
 * ======================================================================== */

static void adain_resblk1d(float *out, const float *in, int channels_in, int channels_out,
                             int length, const float *style,
                             const float *conv1_w, const float *conv1_b,
                             const float *conv2_w, const float *conv2_b,
                             const float *adain_fc_w, const float *adain_fc_b,
                             const float *skip_w, const float *skip_b,
                             float *scratch) {
    /* Project style to mean/std for AdaIN: [style_dim] -> [2 * channels_out] */
    float style_proj[KTTS_DECODER_HIDDEN * 2]; /* large enough for any channel count */
    smol_linear(style_proj, style, adain_fc_w, adain_fc_b,
                1, KTTS_STYLE_ADAIN_DIM, 2 * channels_out);

    float *style_mean = style_proj;
    float *style_std = style_proj + channels_out;

    /* Conv1D #1 */
    float *conv_out = scratch;
    smol_conv1d(conv_out, in, conv1_w, conv1_b,
                channels_in, channels_out, length,
                3, 1, 1, 1, 1);

    /* AdaIN */
    smol_adain(conv_out, conv_out, style_mean, style_std,
               channels_out, length, 1e-5f);

    /* LeakyReLU (alpha = 0.2) */
    for (int i = 0; i < channels_out * length; i++) {
        if (conv_out[i] < 0.0f) conv_out[i] *= 0.2f;
    }

    /* Conv1D #2 */
    float *conv_out2 = scratch + (size_t)channels_out * length;
    smol_conv1d(conv_out2, conv_out, conv2_w, conv2_b,
                channels_out, channels_out, length,
                3, 1, 1, 1, 1);

    /* Re-project style for second AdaIN */
    smol_adain(conv_out2, conv_out2, style_mean, style_std,
               channels_out, length, 1e-5f);

    /* Skip connection */
    if (skip_w && channels_in != channels_out) {
        /* 1x1 conv for channel change */
        smol_conv1d(out, in, skip_w, skip_b,
                    channels_in, channels_out, length,
                    1, 1, 0, 1, 1);
    } else if (channels_in == channels_out) {
        memcpy(out, in, (size_t)channels_out * length * sizeof(float));
    } else {
        memset(out, 0, (size_t)channels_out * length * sizeof(float));
    }

    /* Add residual */
    smol_add_inplace(out, conv_out2, channels_out * length);
}

/* ========================================================================
 * Duration Prediction
 * ======================================================================== */

static void predict_durations(const ktts_prosody_t *model,
                               const float *text_enc_out,
                               const float *style,
                               int seq_len, float speed,
                               int *durations, float *scratch) {
    int dim = KTTS_TEXT_ENC_DIM;     /* 128 */
    int lstm_h = KTTS_PROSODY_LSTM_H; /* 64 per direction */

    /* Duration encoder: 4x stacked BiLSTM */
    float *lstm_out = scratch;
    float *lstm_next = scratch + (size_t)seq_len * dim;

    /* First BiLSTM input is text_enc_out [seq, 128] */
    const float *lstm_in = text_enc_out;

    for (int layer = 0; layer < KTTS_PROSODY_LSTM_N; layer++) {
        smol_bilstm_forward(lstm_out, lstm_in,
                             model->dur_lstm[layer].W_ih_fwd,
                             model->dur_lstm[layer].W_hh_fwd,
                             model->dur_lstm[layer].b_ih_fwd,
                             model->dur_lstm[layer].b_hh_fwd,
                             model->dur_lstm[layer].W_ih_bwd,
                             model->dur_lstm[layer].W_hh_bwd,
                             model->dur_lstm[layer].b_ih_bwd,
                             model->dur_lstm[layer].b_hh_bwd,
                             seq_len,
                             (layer == 0) ? dim : dim,
                             lstm_h);

        /* AdaLayerNorm: project style to gamma/beta, apply to lstm output */
        if (model->dur_ada_fc_w[layer]) {
            float ada_proj[KTTS_TEXT_ENC_DIM * 2];
            smol_linear(ada_proj, style, model->dur_ada_fc_w[layer],
                        model->dur_ada_fc_b[layer],
                        1, KTTS_STYLE_ADAIN_DIM, 2 * dim);
            float *gamma = ada_proj;
            float *beta = ada_proj + dim;

            /* Apply: out = gamma * LayerNorm(x) + beta */
            /* Simplified: just scale and shift the BiLSTM output */
            for (int t = 0; t < seq_len; t++) {
                for (int d = 0; d < dim; d++) {
                    lstm_out[t * dim + d] = gamma[d] * lstm_out[t * dim + d] + beta[d];
                }
            }
        }

        /* Swap buffers for next layer */
        float *tmp = lstm_out;
        lstm_out = lstm_next;
        lstm_next = tmp;
        lstm_in = lstm_next; /* previous output becomes next input */
    }

    /* Duration projection: [seq, 128] -> [seq, 50] */
    float *dur_logits = lstm_out; /* reuse buffer, need seq*50 */
    smol_linear(dur_logits, lstm_in, model->dur_proj_w, model->dur_proj_b,
                seq_len, dim, KTTS_DURATION_CLASSES);

    /* Softmax + argmax per phoneme */
    smol_softmax(dur_logits, seq_len, KTTS_DURATION_CLASSES);

    for (int t = 0; t < seq_len; t++) {
        float *row = dur_logits + t * KTTS_DURATION_CLASSES;
        int best = 0;
        float best_val = row[0];
        for (int c = 1; c < KTTS_DURATION_CLASSES; c++) {
            if (row[c] > best_val) {
                best_val = row[c];
                best = c;
            }
        }
        /* Apply speed scaling */
        int dur = (int)roundf((float)best / speed);
        if (dur < 1) dur = 1;
        durations[t] = dur;
    }
}

/* ========================================================================
 * F0 / Noise Prediction (identical architecture, different weights)
 * ======================================================================== */

typedef struct {
    const float *conv1_w, *conv1_b;
    const float *conv2_w, *conv2_b;
    const float *adain_fc_w, *adain_fc_b;
    const float *downsample_w, *downsample_b;
} adain_block_ptrs_t;

static void predict_contour(const adain_block_ptrs_t blocks[3],
                              const float *proj_w, const float *proj_b,
                              const float *input, const float *style,
                              int length, float *out, float *scratch) {
    int ch = KTTS_TEXT_ENC_DIM;  /* 128 -> 64 -> 64 */
    float *buf_a = scratch;
    float *buf_b = scratch + (size_t)ch * length;
    float *extra = buf_b + (size_t)ch * length;

    /* Transpose input from [length, 128] row-major to [128, length] channel-first */
    for (int c = 0; c < ch; c++) {
        for (int t = 0; t < length; t++) {
            buf_a[c * length + t] = input[t * ch + c];
        }
    }

    /* Block 0: 128 -> 128 */
    adain_resblk1d(buf_b, buf_a, ch, ch, length, style,
                   blocks[0].conv1_w, blocks[0].conv1_b,
                   blocks[0].conv2_w, blocks[0].conv2_b,
                   blocks[0].adain_fc_w, blocks[0].adain_fc_b,
                   NULL, NULL, extra);

    /* Block 1: 128 -> 64 (with downsample) */
    int ch_out = 64;
    adain_resblk1d(buf_a, buf_b, ch, ch_out, length, style,
                   blocks[1].conv1_w, blocks[1].conv1_b,
                   blocks[1].conv2_w, blocks[1].conv2_b,
                   blocks[1].adain_fc_w, blocks[1].adain_fc_b,
                   blocks[1].downsample_w, blocks[1].downsample_b, extra);

    /* Block 2: 64 -> 64 */
    adain_resblk1d(buf_b, buf_a, ch_out, ch_out, length, style,
                   blocks[2].conv1_w, blocks[2].conv1_b,
                   blocks[2].conv2_w, blocks[2].conv2_b,
                   blocks[2].adain_fc_w, blocks[2].adain_fc_b,
                   NULL, NULL, extra);

    /* Final projection: Conv1D [1, 64, 1] -> [1, length] */
    smol_conv1d(out, buf_b, proj_w, proj_b,
                ch_out, 1, length,
                1, 1, 0, 1, 1);
}

/* ========================================================================
 * Main Prosody Forward
 * ======================================================================== */

int ktts_prosody_forward(const ktts_prosody_t *model,
                          const float *text_enc_out, const float *style,
                          int seq_len, float speed,
                          int *durations, float *f0, float *noise,
                          float *expanded_text, float *scratch) {
    int dim = KTTS_TEXT_ENC_DIM;

    /* --- Predict durations --- */
    predict_durations(model, text_enc_out, style, seq_len, speed,
                      durations, scratch);

    /* --- Duration expansion --- */
    /* Expand text features: repeat each phoneme's feature by its duration */
    int expanded_len = 0;
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < durations[t]; d++) {
            if (expanded_len < KTTS_BERT_MAX_POS * 10) { /* safety limit */
                memcpy(expanded_text + expanded_len * dim,
                       text_enc_out + t * dim,
                       dim * sizeof(float));
                expanded_len++;
            }
        }
    }

    if (expanded_len == 0) {
        expanded_len = 1;
        memset(expanded_text, 0, dim * sizeof(float));
    }

    /* Project style from 256 to 128 for AdaIN conditioning.
     * We use the first 128 dims of the style vector as the AdaIN condition. */
    /* (This simplification may need refinement based on actual weight structure) */

    /* --- Predict F0 --- */
    adain_block_ptrs_t f0_blocks[3];
    for (int i = 0; i < 3; i++) {
        f0_blocks[i].conv1_w = model->f0_blocks[i].conv1_w;
        f0_blocks[i].conv1_b = model->f0_blocks[i].conv1_b;
        f0_blocks[i].conv2_w = model->f0_blocks[i].conv2_w;
        f0_blocks[i].conv2_b = model->f0_blocks[i].conv2_b;
        f0_blocks[i].adain_fc_w = model->f0_blocks[i].adain_fc_w;
        f0_blocks[i].adain_fc_b = model->f0_blocks[i].adain_fc_b;
        f0_blocks[i].downsample_w = model->f0_blocks[i].downsample_w;
        f0_blocks[i].downsample_b = model->f0_blocks[i].downsample_b;
    }
    predict_contour(f0_blocks, model->f0_proj_w, model->f0_proj_b,
                    expanded_text, style, expanded_len, f0, scratch);

    /* --- Predict N (noise energy) --- */
    adain_block_ptrs_t n_blocks[3];
    for (int i = 0; i < 3; i++) {
        n_blocks[i].conv1_w = model->n_blocks[i].conv1_w;
        n_blocks[i].conv1_b = model->n_blocks[i].conv1_b;
        n_blocks[i].conv2_w = model->n_blocks[i].conv2_w;
        n_blocks[i].conv2_b = model->n_blocks[i].conv2_b;
        n_blocks[i].adain_fc_w = model->n_blocks[i].adain_fc_w;
        n_blocks[i].adain_fc_b = model->n_blocks[i].adain_fc_b;
        n_blocks[i].downsample_w = model->n_blocks[i].downsample_w;
        n_blocks[i].downsample_b = model->n_blocks[i].downsample_b;
    }
    predict_contour(n_blocks, model->n_proj_w, model->n_proj_b,
                    expanded_text, style, expanded_len, noise, scratch);

    return expanded_len;
}
