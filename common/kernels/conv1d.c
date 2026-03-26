/*
 * conv1d.c - 1D Convolution and Transposed 1D Convolution kernels
 * Follows the same im2col + BLAS sgemm pattern as conv2d.c.
 */

#include "smol_kernels.h"
#include <stdlib.h>
#include <string.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ========================================================================
 * 1D Convolution (im2col + BLAS sgemm)
 *
 * Input:  [c_in, length]
 * Weight: [c_out, c_in, kernel_size]  (or [c_out, c_in/groups, kernel_size])
 * Bias:   [c_out] or NULL
 * Output: [c_out, out_length]
 *
 * out_length = (length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
 * ======================================================================== */

static void im2col_1d(const float *in, float *cols,
                       int c_in, int length,
                       int kernel_size, int stride, int padding, int dilation,
                       int out_length) {
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kernel_size; ki++) {
            int col_row = ic * kernel_size + ki;
            float *col_ptr = cols + (size_t)col_row * out_length;
            for (int oi = 0; oi < out_length; oi++) {
                int ii = oi * stride - padding + ki * dilation;
                if (ii >= 0 && ii < length) {
                    col_ptr[oi] = in[ic * length + ii];
                } else {
                    col_ptr[oi] = 0.0f;
                }
            }
        }
    }
}

void smol_conv1d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int length,
                 int kernel_size, int stride, int padding, int dilation, int groups) {
    int out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    if (groups == 1) {
        /* Standard convolution */
        int patch_size = c_in * kernel_size;

        float *cols = (float *)malloc((size_t)patch_size * out_length * sizeof(float));
        im2col_1d(in, cols, c_in, length, kernel_size, stride, padding, dilation, out_length);

        /* GEMM: weight[c_out, patch_size] @ cols[patch_size, out_length] = out[c_out, out_length] */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    c_out, out_length, patch_size,
                    1.0f, weight, patch_size, cols, out_length,
                    0.0f, out, out_length);
#else
        for (int oc = 0; oc < c_out; oc++) {
            for (int oi = 0; oi < out_length; oi++) {
                float sum = 0.0f;
                for (int p = 0; p < patch_size; p++) {
                    sum += weight[oc * patch_size + p] * cols[p * out_length + oi];
                }
                out[oc * out_length + oi] = sum;
            }
        }
#endif

        free(cols);
    } else {
        /* Grouped / depthwise convolution */
        int c_in_per_group = c_in / groups;
        int c_out_per_group = c_out / groups;
        int patch_size = c_in_per_group * kernel_size;

        float *cols = (float *)malloc((size_t)patch_size * out_length * sizeof(float));

        for (int g = 0; g < groups; g++) {
            const float *in_g = in + g * c_in_per_group * length;
            const float *w_g = weight + g * c_out_per_group * patch_size;
            float *out_g = out + g * c_out_per_group * out_length;

            im2col_1d(in_g, cols, c_in_per_group, length,
                      kernel_size, stride, padding, dilation, out_length);

#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        c_out_per_group, out_length, patch_size,
                        1.0f, w_g, patch_size, cols, out_length,
                        0.0f, out_g, out_length);
#else
            for (int oc = 0; oc < c_out_per_group; oc++) {
                for (int oi = 0; oi < out_length; oi++) {
                    float sum = 0.0f;
                    for (int p = 0; p < patch_size; p++) {
                        sum += w_g[oc * patch_size + p] * cols[p * out_length + oi];
                    }
                    out_g[oc * out_length + oi] = sum;
                }
            }
#endif
        }

        free(cols);
    }

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + oc * out_length;
            for (int oi = 0; oi < out_length; oi++) {
                row[oi] += b;
            }
        }
    }
}

/* ========================================================================
 * Transposed 1D Convolution
 *
 * Input:  [c_in, length]
 * Weight: [c_in, c_out, kernel_size]
 * Bias:   [c_out] or NULL
 * Output: [c_out, out_length]
 *
 * out_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding
 *
 * Implementation: scatter input into col2im buffer, then GEMM.
 * Equivalent to: for each input position, scatter weighted contributions
 * to output positions.
 * ======================================================================== */

void smol_conv_transpose1d(float *out, const float *in, const float *weight, const float *bias,
                            int c_in, int c_out, int length,
                            int kernel_size, int stride, int padding, int output_padding) {
    int out_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding;

    /* Zero output */
    memset(out, 0, (size_t)c_out * out_length * sizeof(float));

    /* For each input position, scatter weighted contributions.
     * weight layout: [c_in, c_out, kernel_size]
     * For input[ic, i], the contribution to output[oc, o] where
     * o = i * stride - padding + ki, is: in[ic, i] * weight[ic, oc, ki]
     */
    for (int ic = 0; ic < c_in; ic++) {
        for (int i = 0; i < length; i++) {
            float x = in[ic * length + i];
            if (x == 0.0f) continue;
            for (int ki = 0; ki < kernel_size; ki++) {
                int o = i * stride - padding + ki;
                if (o < 0 || o >= out_length) continue;
                const float *w_ptr = weight + (ic * c_out * kernel_size) + ki;
                for (int oc = 0; oc < c_out; oc++) {
                    out[oc * out_length + o] += x * w_ptr[oc * kernel_size];
                }
            }
        }
    }

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + oc * out_length;
            for (int oi = 0; oi < out_length; oi++) {
                row[oi] += b;
            }
        }
    }
}
