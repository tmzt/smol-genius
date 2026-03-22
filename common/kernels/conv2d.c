/*
 * conv2d.c - 2D Convolution (im2col + BLAS sgemm) kernel
 * Extracted from qwen_asr_kernels.c, adapted for smol-genius.
 */

#include "smol_kernels.h"
#include <stdlib.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ========================================================================
 * 2D Convolution (im2col + BLAS sgemm)
 * ======================================================================== */

/*
 * im2col: Unroll input patches into a column matrix for GEMM-based conv2d.
 * Input: [C_in, H_in, W_in]
 * Output columns: [C_in * kH * kW, H_out * W_out]
 */
static void im2col(const float *in, float *cols,
                   int c_in, int h_in, int w_in,
                   int kh, int kw, int stride, int padding,
                   int h_out, int w_out) {
    int col_len = h_out * w_out;
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kh; ki++) {
            for (int kj = 0; kj < kw; kj++) {
                int col_row = (ic * kh + ki) * kw + kj;
                float *col_ptr = cols + (size_t)col_row * col_len;
                for (int oh = 0; oh < h_out; oh++) {
                    int ih = oh * stride - padding + ki;
                    for (int ow = 0; ow < w_out; ow++) {
                        int iw = ow * stride - padding + kj;
                        if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                            col_ptr[oh * w_out + ow] = in[ic * h_in * w_in + ih * w_in + iw];
                        } else {
                            col_ptr[oh * w_out + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void smol_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding) {
    int h_out = (h_in + 2 * padding - kh) / stride + 1;
    int w_out = (w_in + 2 * padding - kw) / stride + 1;
    int patch_size = c_in * kh * kw;
    int spatial_out = h_out * w_out;

    /* im2col: input -> column matrix [patch_size, spatial_out] */
    float *cols = (float *)malloc((size_t)patch_size * spatial_out * sizeof(float));
    im2col(in, cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    /* GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, spatial_out, patch_size,
                1.0f, weight, patch_size, cols, spatial_out,
                0.0f, out, spatial_out);
#else
    for (int oc = 0; oc < c_out; oc++) {
        for (int s = 0; s < spatial_out; s++) {
            float sum = 0.0f;
            for (int p = 0; p < patch_size; p++) {
                sum += weight[oc * patch_size + p] * cols[p * spatial_out + s];
            }
            out[oc * spatial_out + s] = sum;
        }
    }
#endif

    free(cols);

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + oc * spatial_out;
            for (int s = 0; s < spatial_out; s++) {
                row[s] += b;
            }
        }
    }
}
