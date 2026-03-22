/*
 * mean_pool_avx.c - AVX2 mean pooling kernel
 */

#include "smol_kernels_impl.h"

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>
#include <string.h>

void smol_mean_pool_avx(float *out, const float *hidden_states,
                         const int *seq_starts, const int *seq_lens,
                         int num_seqs, int hidden) {
    for (int s = 0; s < num_seqs; s++) {
        int start = seq_starts[s];
        int len = seq_lens[s];
        float *o = out + s * hidden;

        memset(o, 0, hidden * sizeof(float));

        /* Accumulate all tokens */
        for (int t = 0; t < len; t++) {
            const float *tok = hidden_states + (start + t) * hidden;
            int d = 0;
            for (; d + 32 <= hidden; d += 32) {
                _mm256_storeu_ps(o + d,      _mm256_add_ps(_mm256_loadu_ps(o + d),      _mm256_loadu_ps(tok + d)));
                _mm256_storeu_ps(o + d + 8,  _mm256_add_ps(_mm256_loadu_ps(o + d + 8),  _mm256_loadu_ps(tok + d + 8)));
                _mm256_storeu_ps(o + d + 16, _mm256_add_ps(_mm256_loadu_ps(o + d + 16), _mm256_loadu_ps(tok + d + 16)));
                _mm256_storeu_ps(o + d + 24, _mm256_add_ps(_mm256_loadu_ps(o + d + 24), _mm256_loadu_ps(tok + d + 24)));
            }
            for (; d + 8 <= hidden; d += 8) {
                _mm256_storeu_ps(o + d, _mm256_add_ps(_mm256_loadu_ps(o + d), _mm256_loadu_ps(tok + d)));
            }
            for (; d < hidden; d++) {
                o[d] += tok[d];
            }
        }

        /* Divide by length */
        if (len > 0) {
            float inv_len = 1.0f / (float)len;
            __m256 scale = _mm256_set1_ps(inv_len);
            int d = 0;
            for (; d + 32 <= hidden; d += 32) {
                _mm256_storeu_ps(o + d,      _mm256_mul_ps(_mm256_loadu_ps(o + d),      scale));
                _mm256_storeu_ps(o + d + 8,  _mm256_mul_ps(_mm256_loadu_ps(o + d + 8),  scale));
                _mm256_storeu_ps(o + d + 16, _mm256_mul_ps(_mm256_loadu_ps(o + d + 16), scale));
                _mm256_storeu_ps(o + d + 24, _mm256_mul_ps(_mm256_loadu_ps(o + d + 24), scale));
            }
            for (; d + 8 <= hidden; d += 8) {
                _mm256_storeu_ps(o + d, _mm256_mul_ps(_mm256_loadu_ps(o + d), scale));
            }
            for (; d < hidden; d++) {
                o[d] *= inv_len;
            }
        }
    }
}

#endif /* __AVX2__ && __FMA__ */
