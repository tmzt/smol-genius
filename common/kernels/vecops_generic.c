/*
 * vecops_generic.c - generic vector operations
 */

#include "smol_kernels_impl.h"

float smol_dot_f32_generic(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void smol_vec_scale_inplace_generic(float *dst, float scale, int n) {
    for (int i = 0; i < n; i++) dst[i] *= scale;
}

void smol_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n) {
    for (int i = 0; i < n; i++) dst[i] += alpha * src[i];
}

void smol_vec_scale_add_generic(float *dst, const float *src, float correction, int n) {
    for (int i = 0; i < n; i++) dst[i] = dst[i] * correction + src[i];
}
