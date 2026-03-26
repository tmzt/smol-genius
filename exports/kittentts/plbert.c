/*
 * plbert.c - PL-BERT (ALBERT-style) forward pass
 *
 * ALBERT with weight sharing: one transformer layer reused N times.
 * Architecture:
 *   Token embedding [178, 128] + Position embedding [512, 128]
 *   -> LayerNorm -> Linear 128->768
 *   -> N x { LayerNorm -> Self-Attention(768) -> Residual -> LayerNorm -> FFN(768->2048->768) -> Residual }
 *   -> Linear 768->128
 */

#include "plbert.h"
#include "smol_kernels.h"
#include <string.h>

void ktts_plbert_forward(const ktts_plbert_t *model, const int *token_ids,
                          int seq_len, float *out, float *scratch) {
    int bert_dim = KTTS_BERT_DIM;        /* 128 */
    int hidden = KTTS_BERT_HIDDEN;       /* 768 */
    int ffn_dim = KTTS_BERT_FFN;         /* 2048 */
    int n_heads = KTTS_BERT_HEADS;       /* 12 */
    int head_dim = KTTS_BERT_HEAD_DIM;   /* 64 */

    /* Partition scratch buffer */
    float *x = scratch;                                             /* [seq, 768] */
    float *x_norm = x + (size_t)seq_len * hidden;                  /* [seq, 768] */
    float *q = x_norm + (size_t)seq_len * hidden;                  /* [seq, 768] */
    float *k = q + (size_t)seq_len * hidden;                       /* [seq, 768] */
    float *v = k + (size_t)seq_len * hidden;                       /* [seq, 768] */
    float *attn_out = v + (size_t)seq_len * hidden;                /* [seq, 768] */
    float *ffn_scratch = attn_out + (size_t)seq_len * hidden;      /* [seq, 2048] */
    float *emb_buf = ffn_scratch + (size_t)seq_len * ffn_dim;      /* [seq, 128] */

    /* --- Step 1: Token + Position + Type Embedding --- */
    /* emb_buf[t] = tok_emb[token_ids[t]] + pos_emb[t] + tok_type_emb[0] */
    for (int t = 0; t < seq_len; t++) {
        int tok_id = token_ids[t];
        if (tok_id < 0 || tok_id >= KTTS_PHONEME_VOCAB) tok_id = 0;
        int pos = (t < KTTS_BERT_MAX_POS) ? t : KTTS_BERT_MAX_POS - 1;

        float *dst = emb_buf + t * bert_dim;
        const float *tok = model->tok_emb + tok_id * bert_dim;
        const float *pos_e = model->pos_emb + pos * bert_dim;
        const float *type_e = model->tok_type_emb; /* type 0 */
        for (int i = 0; i < bert_dim; i++) {
            dst[i] = tok[i] + pos_e[i] + type_e[i];
        }
    }

    /* Embedding LayerNorm */
    smol_layer_norm(emb_buf, emb_buf, model->emb_ln_w, model->emb_ln_b,
                    seq_len, bert_dim, 1e-12f);

    /* --- Step 2: Linear mapping 128 -> 768 --- */
    smol_linear(x, emb_buf, model->hidden_map_w, model->hidden_map_b,
                seq_len, bert_dim, hidden);

    /* --- Step 3: Shared ALBERT transformer layers --- */
    /* Bidirectional attention: single window covering entire sequence */
    int window_starts[2] = {0, seq_len};
    float scale = 1.0f / 8.0f; /* 1/sqrt(64) */

    for (int iter = 0; iter < model->n_iterations; iter++) {
        /* Pre-attention LayerNorm */
        smol_layer_norm(x_norm, x, model->attn_ln_w, model->attn_ln_b,
                        seq_len, hidden, 1e-12f);

        /* Q/K/V projections (no bias in ALBERT attention) */
        smol_linear_nobias(q, x_norm, model->attn_q_w, seq_len, hidden, hidden);
        smol_linear_nobias(k, x_norm, model->attn_k_w, seq_len, hidden, hidden);
        smol_linear_nobias(v, x_norm, model->attn_v_w, seq_len, hidden, hidden);

        /* Bidirectional self-attention */
        smol_bidirectional_attention(attn_out, q, k, v,
                                     seq_len, n_heads, head_dim, scale,
                                     window_starts, 1);

        /* Output projection + residual */
        smol_linear(x_norm, attn_out, model->attn_o_w, model->attn_o_b,
                    seq_len, hidden, hidden);
        smol_add_inplace(x, x_norm, seq_len * hidden);

        /* Pre-FFN LayerNorm */
        smol_layer_norm(x_norm, x, model->ffn_ln_w, model->ffn_ln_b,
                        seq_len, hidden, 1e-12f);

        /* FFN: up 768->2048 with GELU, then down 2048->768 */
        smol_linear(ffn_scratch, x_norm, model->ffn_up_w, model->ffn_up_b,
                    seq_len, hidden, ffn_dim);
        smol_gelu(ffn_scratch, seq_len * ffn_dim);
        smol_linear(x_norm, ffn_scratch, model->ffn_down_w, model->ffn_down_b,
                    seq_len, ffn_dim, hidden);

        /* Residual */
        smol_add_inplace(x, x_norm, seq_len * hidden);
    }

    /* --- Step 4: Output projection 768 -> 128 --- */
    smol_linear(out, x, model->out_proj_w, model->out_proj_b,
                seq_len, hidden, bert_dim);
}
