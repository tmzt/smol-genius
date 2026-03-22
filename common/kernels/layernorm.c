/*
 * layernorm.c - Layer normalization kernel with inline AVX-512/AVX2/scalar paths
 * Extracted from qwen_asr_kernels.c, adapted for smol-genius.
 */

#include "smol_kernels.h"
#include <math.h>

#if (defined(__AVX512F__) || defined(__AVX2__)) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif

/* ========================================================================
 * Normalization - Layer Norm
 * ======================================================================== */

void smol_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
#if defined(__AVX512F__)
        __m512 sumv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(x_row + i));
        }
        float mean = _mm512_reduce_add_ps(sumv);
        for (; i < hidden; i++) mean += x_row[i];
#elif defined(__AVX2__)
        __m256 sumv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(x_row + i));
        }
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float mean = _mm_cvtss_f32(sum128);
        for (; i < hidden; i++) mean += x_row[i];
#else
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x_row[i];
#endif
        mean /= hidden;

        /* Compute variance */
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv = _mm512_set1_ps(mean);
        __m512 accv = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(x_row + j), meanv);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float var = _mm512_reduce_add_ps(accv);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv = _mm256_set1_ps(mean);
        __m256 accv = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x_row + j), meanv);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float var = _mm_cvtss_f32(acc128);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#else
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float d = x_row[i] - mean;
            var += d * d;
        }
#endif
        var /= hidden;

        float inv_std = 1.0f / sqrtf(var + eps);
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv2 = _mm512_set1_ps(mean);
        __m512 invv = _mm512_set1_ps(inv_std);
        int k = 0;
        for (; k + 16 <= hidden; k += 16) {
            __m512 vx = _mm512_sub_ps(_mm512_loadu_ps(x_row + k), meanv2);
            __m512 vw = _mm512_loadu_ps(weight + k);
            __m512 vb = _mm512_loadu_ps(bias + k);
            __m512 v = _mm512_mul_ps(_mm512_mul_ps(vx, invv), vw);
            v = _mm512_add_ps(v, vb);
            _mm512_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv2 = _mm256_set1_ps(mean);
        __m256 invv = _mm256_set1_ps(inv_std);
        int k = 0;
        for (; k + 8 <= hidden; k += 8) {
            __m256 vx = _mm256_sub_ps(_mm256_loadu_ps(x_row + k), meanv2);
            __m256 vw = _mm256_loadu_ps(weight + k);
            __m256 vb = _mm256_loadu_ps(bias + k);
            __m256 v = _mm256_mul_ps(_mm256_mul_ps(vx, invv), vw);
            v = _mm256_add_ps(v, vb);
            _mm256_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
        }
#endif
    }
}
