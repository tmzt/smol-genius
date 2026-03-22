/* ========================================================================
 * Attention Operations
 * ======================================================================== */

#include "smol_kernels.h"
#include "smol_kernels_impl.h"
#include <math.h>

/* Threading helpers (defined elsewhere) */
extern void smol_parallel_for(void (*fn)(int, int, void *), void *arg);
extern int  smol_get_thread_count(void);

static inline float smol_dot_f32(const float *a, const float *b, int n) {
    return smol_dot_f32_impl(a, b, n);
}

/* dst = dst * scale */
static inline void smol_vec_scale_inplace(float *dst, float scale, int n) {
    smol_vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src */
static inline void smol_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    smol_vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* dst = dst * correction + src */
static inline void smol_vec_scale_add(float *dst, const float *src, float correction, int n) {
    smol_vec_scale_add_impl(dst, src, correction, n);
}

void smol_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq __attribute__((unused)),
                                   int n_heads, int head_dim, float scale,
                                   const int *window_starts, int n_windows) {
    int hidden = n_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        for (int w = 0; w < n_windows; w++) {
            int ws = window_starts[w];
            int we = window_starts[w + 1];

            for (int i = ws; i < we; i++) {
                const float *q_row = Q + i * hidden + h * head_dim;
                float *o_row = out + i * hidden + h * head_dim;

                /* Online softmax */
                float max_score = -1e30f;
                float sum_exp = 0.0f;
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

                for (int j = ws; j < we; j++) {
                    const float *k_row = K + j * hidden + h * head_dim;
                    const float *v_row = V + j * hidden + h * head_dim;

                    float score = smol_dot_f32(q_row, k_row, head_dim) * scale;

                    if (score > max_score) {
                        float correction = expf(max_score - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        smol_vec_scale_add(o_row, v_row, correction, head_dim);
                        max_score = score;
                    } else {
                        float wt = expf(score - max_score);
                        sum_exp += wt;
                        smol_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                    }
                }

                if (sum_exp > 0.0f) {
                    float inv_sum = 1.0f / sum_exp;
                    smol_vec_scale_inplace(o_row, inv_sum, head_dim);
                }
            }
        }
    }
}

static void smol_causal_attention_heads(float *out, const float *Q, const float *K,
                                        const float *V, int seq_q, int seq_k,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int q_offset,
                                        int head_start, int head_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                float score = smol_dot_f32(q_row, k_row, head_dim) * scale;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    smol_vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    float wt = expf(score - max_score);
                    sum_exp += wt;
                    smol_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                smol_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int seq_q, seq_k;
    int n_heads, n_kv_heads;
    int head_dim;
    float scale;
    int q_offset;
} causal_attn_task_t;

static void causal_attn_worker(int tid, int n_threads, void *arg) {
    causal_attn_task_t *t = (causal_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    smol_causal_attention_heads(t->out, t->Q, t->K, t->V,
                                t->seq_q, t->seq_k, t->n_heads, t->n_kv_heads,
                                t->head_dim, t->scale, t->q_offset, h0, h1);
}

void smol_causal_attention(float *out, const float *Q, const float *K, const float *V,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset) {
    if (smol_get_thread_count() > 1 && n_heads >= 2 && (seq_q >= 2 || seq_k >= 128)) {
        causal_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .seq_q = seq_q, .seq_k = seq_k,
            .n_heads = n_heads, .n_kv_heads = n_kv_heads,
            .head_dim = head_dim, .scale = scale, .q_offset = q_offset
        };
        smol_parallel_for(causal_attn_worker, &task);
        return;
    }

    smol_causal_attention_heads(out, Q, K, V,
                                seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset, 0, n_heads);
}
