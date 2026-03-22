/*
 * vecops_avx.c - x86 SIMD f32 attention helpers (AVX2+FMA, with AVX-512F when available)
 *
 * Operates on L1-resident head vectors.
 */

#include "smol_kernels_impl.h"

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

float smol_dot_f32_avx(const float *a, const float *b, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    int i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float sum = _mm_cvtss_f32(r);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

void smol_vec_scale_inplace_avx(float *dst, float scale, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 s = _mm512_set1_ps(scale);
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(dst + i,      _mm512_mul_ps(_mm512_loadu_ps(dst + i),      s));
        _mm512_storeu_ps(dst + i + 16, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), s));
        _mm512_storeu_ps(dst + i + 32, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), s));
        _mm512_storeu_ps(dst + i + 48, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), s));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(dst + i), s));
    }
    for (; i < n; i++) dst[i] *= scale;
#else
    int i = 0;
    __m256 s = _mm256_set1_ps(scale);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst+i,    _mm256_mul_ps(_mm256_loadu_ps(dst+i),    s));
        _mm256_storeu_ps(dst+i+8,  _mm256_mul_ps(_mm256_loadu_ps(dst+i+8),  s));
        _mm256_storeu_ps(dst+i+16, _mm256_mul_ps(_mm256_loadu_ps(dst+i+16), s));
        _mm256_storeu_ps(dst+i+24, _mm256_mul_ps(_mm256_loadu_ps(dst+i+24), s));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst+i, _mm256_mul_ps(_mm256_loadu_ps(dst+i), s));
    }
    for (; i < n; i++) dst[i] *= scale;
#endif
}

void smol_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 a = _mm512_set1_ps(alpha);
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i), a, _mm512_loadu_ps(dst + i)));
        _mm512_storeu_ps(dst + i + 16,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), a, _mm512_loadu_ps(dst + i + 16)));
        _mm512_storeu_ps(dst + i + 32,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), a, _mm512_loadu_ps(dst + i + 32)));
        _mm512_storeu_ps(dst + i + 48,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), a, _mm512_loadu_ps(dst + i + 48)));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i), a, _mm512_loadu_ps(dst + i)));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
#else
    int i = 0;
    __m256 a = _mm256_set1_ps(alpha);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst+i,    _mm256_fmadd_ps(_mm256_loadu_ps(src+i),    a, _mm256_loadu_ps(dst+i)));
        _mm256_storeu_ps(dst+i+8,  _mm256_fmadd_ps(_mm256_loadu_ps(src+i+8),  a, _mm256_loadu_ps(dst+i+8)));
        _mm256_storeu_ps(dst+i+16, _mm256_fmadd_ps(_mm256_loadu_ps(src+i+16), a, _mm256_loadu_ps(dst+i+16)));
        _mm256_storeu_ps(dst+i+24, _mm256_fmadd_ps(_mm256_loadu_ps(src+i+24), a, _mm256_loadu_ps(dst+i+24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(_mm256_loadu_ps(src+i), a, _mm256_loadu_ps(dst+i)));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
#endif
}

void smol_vec_scale_add_avx(float *dst, const float *src, float correction, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 c = _mm512_set1_ps(correction);
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i), c, _mm512_loadu_ps(src + i)));
        _mm512_storeu_ps(dst + i + 16,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 16), c, _mm512_loadu_ps(src + i + 16)));
        _mm512_storeu_ps(dst + i + 32,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 32), c, _mm512_loadu_ps(src + i + 32)));
        _mm512_storeu_ps(dst + i + 48,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 48), c, _mm512_loadu_ps(src + i + 48)));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i), c, _mm512_loadu_ps(src + i)));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
#else
    int i = 0;
    __m256 c = _mm256_set1_ps(correction);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst+i,    _mm256_fmadd_ps(_mm256_loadu_ps(dst+i),    c, _mm256_loadu_ps(src+i)));
        _mm256_storeu_ps(dst+i+8,  _mm256_fmadd_ps(_mm256_loadu_ps(dst+i+8),  c, _mm256_loadu_ps(src+i+8)));
        _mm256_storeu_ps(dst+i+16, _mm256_fmadd_ps(_mm256_loadu_ps(dst+i+16), c, _mm256_loadu_ps(src+i+16)));
        _mm256_storeu_ps(dst+i+24, _mm256_fmadd_ps(_mm256_loadu_ps(dst+i+24), c, _mm256_loadu_ps(src+i+24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(_mm256_loadu_ps(dst+i), c, _mm256_loadu_ps(src+i)));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
#endif
}

#endif /* __AVX2__ && __FMA__ */
