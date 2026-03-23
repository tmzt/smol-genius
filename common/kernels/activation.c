/*
 * activation.c - Activation function kernels (SiLU, GELU, SwiGLU, Softmax)
 * Extracted from qwen_asr_kernels.c, adapted for smol-genius.
 */

#include "smol_kernels.h"
#include <math.h>
#include <stdlib.h>

#if defined(__APPLE__) && defined(USE_BLAS)
#include <Accelerate/Accelerate.h>
#endif

/* External thread-pool interface (declared in smol_kernels.h or provided by host) */
extern int smol_get_thread_count(void);
extern void smol_parallel_for(void (*fn)(int tid, int n_threads, void *arg), void *arg);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void smol_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

void smol_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

typedef struct {
    float *out;
    const float *gate_up;
    int seq_len;
    int intermediate;
} swiglu_task_t;

static void swiglu_worker(int tid, int n_threads, void *arg) {
    swiglu_task_t *t = (swiglu_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    int alias_inplace = (t->out == t->gate_up);
    for (int s = s0; s < s1; s++) {
        const float *gu = t->gate_up + (size_t)s * 2 * inter;
        float *o = t->out + (size_t)s * inter;
        if (!alias_inplace) {
#if defined(__APPLE__) && defined(USE_BLAS)
            /* Fast path for prefill: vectorized exp(-g) using Accelerate/vForce. */
            for (int j = 0; j < inter; j++) o[j] = -gu[2 * j];
            int n = inter;
            vvexpf(o, o, &n);
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                o[j] = (g / (1.0f + o[j])) * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        } else {
            /* In-place mode (decode seq=1): keep single-pass scalar to avoid alias hazards. */
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
        }
    }
}

void smol_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate) {
    swiglu_task_t task = {
        .out = out,
        .gate_up = gate_up,
        .seq_len = seq_len,
        .intermediate = intermediate
    };

    if (smol_get_thread_count() > 1 && seq_len >= 2 && intermediate >= 256) {
        smol_parallel_for(swiglu_worker, &task);
    } else {
        swiglu_worker(0, 1, &task);
    }
}

static void geglu_worker(int tid, int n_threads, void *arg) {
    swiglu_task_t *t = (swiglu_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    for (int s = s0; s < s1; s++) {
        const float *gu = t->gate_up + (size_t)s * 2 * inter;
        float *o = t->out + (size_t)s * inter;
        for (int j = 0; j < inter; j++) {
            float g = gu[2 * j];
            float u = gu[2 * j + 1];
            /* GELU-tanh: 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3))) */
            float x3 = g * g * g;
            float inner = 0.7978845608028654f * (g + 0.044715f * x3);
            g = 0.5f * g * (1.0f + tanhf(inner));
            o[j] = g * u;
        }
    }
}

void smol_geglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate) {
    swiglu_task_t task = {
        .out = out,
        .gate_up = gate_up,
        .seq_len = seq_len,
        .intermediate = intermediate
    };

    if (smol_get_thread_count() > 1 && seq_len >= 2 && intermediate >= 256) {
        smol_parallel_for(geglu_worker, &task);
    } else {
        geglu_worker(0, 1, &task);
    }
}

void smol_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}
