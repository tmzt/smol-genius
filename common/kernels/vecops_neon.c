/*
 * vecops_neon.c - ARM NEON vector operations
 */

#include "smol_kernels_impl.h"

#ifdef __ARM_NEON

#include <arm_neon.h>

/* Fallback for compilers that don't define vfmaq_f32 (Fused Multiply-Add) */
#ifndef vfmaq_f32
#define vfmaq_f32(acc, a, b) vaddq_f32(acc, vmulq_f32(a, b))
#endif

/* Fallback for horizontal addition (vaddvq_f32 is ARMv8/AArch64 only) */
#if defined(__arm__) && !defined(vaddvq_f32)
static inline float vaddvq_f32(float32x4_t v) {
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}
#endif

float smol_vec_dot_neon(const float *a, const float *b, int n) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void smol_vec_scale_inplace_neon(float *dst, float scale, int n) {
    int i = 0;
    float32x4_t s = vdupq_n_f32(scale);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(vdupq_n_f32(0.0f), d0, s));
        vst1q_f32(dst + i + 4, vfmaq_f32(vdupq_n_f32(0.0f), d1, s));
    }
    for (; i < n; i++) dst[i] *= scale;
}

void smol_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n) {
    int i = 0;
    float32x4_t a = vdupq_n_f32(alpha);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(d0, s0, a));
        vst1q_f32(dst + i + 4, vfmaq_f32(d1, s1, a));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
}

void smol_vec_scale_add_neon(float *dst, const float *src, float correction, int n) {
    int i = 0;
    float32x4_t c = vdupq_n_f32(correction);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(s0, d0, c));
        vst1q_f32(dst + i + 4, vfmaq_f32(s1, d1, c));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
}

#endif /* __ARM_NEON */
