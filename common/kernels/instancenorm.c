/*
 * instancenorm.c - Instance Normalization and Adaptive Instance Normalization (AdaIN)
 *
 * Instance norm: normalize each channel across the sequence/spatial dimension.
 * AdaIN: instance-normalize, then apply style-conditioned scale and shift.
 */

#include "smol_kernels.h"
#include <math.h>

/* ========================================================================
 * Instance Normalization
 *
 * Input/Output: [channels, length]
 * Normalizes each channel independently (mean=0, var=1) across length dim.
 * ======================================================================== */

void smol_instance_norm(float *out, const float *x, int channels, int length, float eps) {
    for (int c = 0; c < channels; c++) {
        const float *x_ch = x + c * length;
        float *out_ch = out + c * length;

        /* Compute mean */
        float mean = 0.0f;
        for (int i = 0; i < length; i++) {
            mean += x_ch[i];
        }
        mean /= length;

        /* Compute variance */
        float var = 0.0f;
        for (int i = 0; i < length; i++) {
            float d = x_ch[i] - mean;
            var += d * d;
        }
        var /= length;

        /* Normalize */
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < length; i++) {
            out_ch[i] = (x_ch[i] - mean) * inv_std;
        }
    }
}

/* ========================================================================
 * Adaptive Instance Normalization (AdaIN)
 *
 * Input:      [channels, length]
 * style_mean: [channels]  (learned linear projection of style vector)
 * style_std:  [channels]  (learned linear projection of style vector)
 * Output:     [channels, length]
 *
 * out[c, i] = style_std[c] * InstanceNorm(x)[c, i] + style_mean[c]
 * ======================================================================== */

void smol_adain(float *out, const float *x,
                const float *style_mean, const float *style_std,
                int channels, int length, float eps) {
    for (int c = 0; c < channels; c++) {
        const float *x_ch = x + c * length;
        float *out_ch = out + c * length;

        /* Instance normalize this channel */
        float mean = 0.0f;
        for (int i = 0; i < length; i++) {
            mean += x_ch[i];
        }
        mean /= length;

        float var = 0.0f;
        for (int i = 0; i < length; i++) {
            float d = x_ch[i] - mean;
            var += d * d;
        }
        var /= length;

        float inv_std = 1.0f / sqrtf(var + eps);
        float s_std = style_std[c];
        float s_mean = style_mean[c];

        for (int i = 0; i < length; i++) {
            out_ch[i] = s_std * (x_ch[i] - mean) * inv_std + s_mean;
        }
    }
}
