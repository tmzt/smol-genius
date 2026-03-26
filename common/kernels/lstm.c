/*
 * lstm.c - LSTM and Bidirectional LSTM kernels
 *
 * Standard LSTM cell with forget/input/cell/output gates.
 * Uses BLAS sgemm for batched gate projections when available.
 */

#include "smol_kernels.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ========================================================================
 * LSTM Forward (single direction)
 *
 * Computes: gates = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh
 *   i = sigmoid(gates[0:H])
 *   f = sigmoid(gates[H:2H])
 *   g = tanh(gates[2H:3H])
 *   o = sigmoid(gates[3H:4H])
 *   c_new = f * c + i * g
 *   h_new = o * tanh(c_new)
 *
 * W_ih: [4*hidden, input_dim]
 * W_hh: [4*hidden, hidden]
 * b_ih: [4*hidden]
 * b_hh: [4*hidden]
 *
 * Input:  [seq_len, input_dim]
 * Output: [seq_len, hidden]
 * h_out:  [hidden] (final hidden state)
 * c_out:  [hidden] (final cell state)
 * ======================================================================== */

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void smol_lstm_forward(float *out, float *h_out, float *c_out,
                       const float *input, const float *h0, const float *c0,
                       const float *W_ih, const float *W_hh,
                       const float *b_ih, const float *b_hh,
                       int seq_len, int input_dim, int hidden_dim) {
    int gate4 = 4 * hidden_dim;

    /* Working buffers for gates and state */
    float *gates = (float *)malloc(gate4 * sizeof(float));
    float *h = (float *)malloc(hidden_dim * sizeof(float));
    float *c = (float *)malloc(hidden_dim * sizeof(float));

    /* Initialize h, c from h0/c0 or zeros */
    if (h0) {
        memcpy(h, h0, hidden_dim * sizeof(float));
    } else {
        memset(h, 0, hidden_dim * sizeof(float));
    }
    if (c0) {
        memcpy(c, c0, hidden_dim * sizeof(float));
    } else {
        memset(c, 0, hidden_dim * sizeof(float));
    }

    for (int t = 0; t < seq_len; t++) {
        const float *x_t = input + t * input_dim;

        /* gates = x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh */
        /* Initialize gates with bias sum */
        for (int j = 0; j < gate4; j++) {
            gates[j] = b_ih[j] + b_hh[j];
        }

        /* gates += x_t @ W_ih^T */
#ifdef USE_BLAS
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    gate4, input_dim, 1.0f,
                    W_ih, input_dim, x_t, 1,
                    1.0f, gates, 1);
#else
        for (int j = 0; j < gate4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < input_dim; k++) {
                sum += W_ih[j * input_dim + k] * x_t[k];
            }
            gates[j] += sum;
        }
#endif

        /* gates += h @ W_hh^T */
#ifdef USE_BLAS
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    gate4, hidden_dim, 1.0f,
                    W_hh, hidden_dim, h, 1,
                    1.0f, gates, 1);
#else
        for (int j = 0; j < gate4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < hidden_dim; k++) {
                sum += W_hh[j * hidden_dim + k] * h[k];
            }
            gates[j] += sum;
        }
#endif

        /* Apply gate activations and compute new state */
        float *out_t = out + t * hidden_dim;
        for (int j = 0; j < hidden_dim; j++) {
            float i_gate = sigmoid_f(gates[j]);
            float f_gate = sigmoid_f(gates[hidden_dim + j]);
            float g_gate = tanhf(gates[2 * hidden_dim + j]);
            float o_gate = sigmoid_f(gates[3 * hidden_dim + j]);

            c[j] = f_gate * c[j] + i_gate * g_gate;
            h[j] = o_gate * tanhf(c[j]);
            out_t[j] = h[j];
        }
    }

    if (h_out) memcpy(h_out, h, hidden_dim * sizeof(float));
    if (c_out) memcpy(c_out, c, hidden_dim * sizeof(float));

    free(gates);
    free(h);
    free(c);
}

/* ========================================================================
 * Bidirectional LSTM Forward
 *
 * Runs forward and backward LSTMs, concatenates outputs.
 *
 * W_ih_fwd, W_hh_fwd, b_ih_fwd, b_hh_fwd: forward LSTM weights
 * W_ih_bwd, W_hh_bwd, b_ih_bwd, b_hh_bwd: backward LSTM weights
 *
 * Input:  [seq_len, input_dim]
 * Output: [seq_len, 2 * hidden_dim]  (fwd || bwd concatenated)
 * ======================================================================== */

void smol_bilstm_forward(float *out,
                          const float *input,
                          const float *W_ih_fwd, const float *W_hh_fwd,
                          const float *b_ih_fwd, const float *b_hh_fwd,
                          const float *W_ih_bwd, const float *W_hh_bwd,
                          const float *b_ih_bwd, const float *b_hh_bwd,
                          int seq_len, int input_dim, int hidden_dim) {
    int out_dim = 2 * hidden_dim;

    /* Forward pass output: [seq_len, hidden_dim] */
    float *fwd_out = (float *)malloc((size_t)seq_len * hidden_dim * sizeof(float));

    smol_lstm_forward(fwd_out, NULL, NULL,
                      input, NULL, NULL,
                      W_ih_fwd, W_hh_fwd, b_ih_fwd, b_hh_fwd,
                      seq_len, input_dim, hidden_dim);

    /* Backward pass: reverse the input sequence */
    float *rev_input = (float *)malloc((size_t)seq_len * input_dim * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        memcpy(rev_input + t * input_dim,
               input + (seq_len - 1 - t) * input_dim,
               input_dim * sizeof(float));
    }

    float *bwd_out = (float *)malloc((size_t)seq_len * hidden_dim * sizeof(float));

    smol_lstm_forward(bwd_out, NULL, NULL,
                      rev_input, NULL, NULL,
                      W_ih_bwd, W_hh_bwd, b_ih_bwd, b_hh_bwd,
                      seq_len, input_dim, hidden_dim);

    /* Concatenate: out[t] = [fwd_out[t], bwd_out[seq_len-1-t]] */
    for (int t = 0; t < seq_len; t++) {
        memcpy(out + t * out_dim,
               fwd_out + t * hidden_dim,
               hidden_dim * sizeof(float));
        memcpy(out + t * out_dim + hidden_dim,
               bwd_out + (seq_len - 1 - t) * hidden_dim,
               hidden_dim * sizeof(float));
    }

    free(fwd_out);
    free(rev_input);
    free(bwd_out);
}
