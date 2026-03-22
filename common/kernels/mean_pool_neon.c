/*
 * mean_pool_neon.c - ARM NEON mean pooling kernel
 *
 * For each sequence in the batch:
 *   1. Sum all token hidden states in the sequence
 *   2. Divide by sequence length
 *   3. Write the result
 *
 * Uses 8-wide NEON accumulation with scalar tail.
 * Scalar fallback provided for non-ARM builds.
 */

#include "smol_kernels.h"
#include <string.h>

#ifdef __ARM_NEON

#include <arm_neon.h>

void smol_mean_pool_neon(float *out, const float *hidden_states,
                          const int *seq_starts, const int *seq_lens,
                          int num_seqs, int hidden) {
    for (int s = 0; s < num_seqs; s++) {
        int start = seq_starts[s];
        int len = seq_lens[s];
        float *o = out + s * hidden;

        /* Zero the output row */
        memset(o, 0, hidden * sizeof(float));

        /* Accumulate all tokens in the sequence */
        for (int t = 0; t < len; t++) {
            const float *tok = hidden_states + (start + t) * hidden;
            int d = 0;
            for (; d + 8 <= hidden; d += 8) {
                float32x4_t a0 = vld1q_f32(o + d);
                float32x4_t a1 = vld1q_f32(o + d + 4);
                float32x4_t b0 = vld1q_f32(tok + d);
                float32x4_t b1 = vld1q_f32(tok + d + 4);
                vst1q_f32(o + d,     vaddq_f32(a0, b0));
                vst1q_f32(o + d + 4, vaddq_f32(a1, b1));
            }
            for (; d < hidden; d++) {
                o[d] += tok[d];
            }
        }

        /* Divide by sequence length */
        if (len > 0) {
            float inv_len = 1.0f / (float)len;
            float32x4_t scale = vdupq_n_f32(inv_len);
            int d = 0;
            for (; d + 8 <= hidden; d += 8) {
                vst1q_f32(o + d,     vmulq_f32(vld1q_f32(o + d),     scale));
                vst1q_f32(o + d + 4, vmulq_f32(vld1q_f32(o + d + 4), scale));
            }
            for (; d < hidden; d++) {
                o[d] *= inv_len;
            }
        }
    }
}

#else /* !__ARM_NEON — scalar fallback for non-ARM builds */

void smol_mean_pool_neon(float *out, const float *hidden_states,
                          const int *seq_starts, const int *seq_lens,
                          int num_seqs, int hidden) {
    for (int s = 0; s < num_seqs; s++) {
        int start = seq_starts[s];
        int len = seq_lens[s];
        float *o = out + s * hidden;

        memset(o, 0, hidden * sizeof(float));
        for (int t = 0; t < len; t++) {
            const float *tok = hidden_states + (start + t) * hidden;
            for (int d = 0; d < hidden; d++) {
                o[d] += tok[d];
            }
        }
        if (len > 0) {
            float inv_len = 1.0f / (float)len;
            for (int d = 0; d < hidden; d++) {
                o[d] *= inv_len;
            }
        }
    }
}

#endif /* __ARM_NEON */
