/*
 * image.c - Image loading via stb_image, resize, and normalization
 *
 * Supports PNG, JPG, BMP, GIF, PSD, TGA, HDR, PIC, PNM (PPM/PGM) via stb_image.
 * Output: channel-first float [3, H, W] normalized to [-1, 1].
 */

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR          /* skip HDR loader (we don't need float images) */
#define STBI_NO_LINEAR       /* skip linear float conversion */
#include "stb_image.h"

#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int smol_verbose;

/* ========================================================================
 * Bicubic Resize (matches PIL.Image.BICUBIC / HF SiglipImageProcessor)
 * ======================================================================== */

static inline float cubic_weight(float x) {
    /* Keys cubic interpolation (a = -0.5, same as PIL) */
    float ax = x < 0 ? -x : x;
    if (ax <= 1.0f) return (1.5f * ax - 2.5f) * ax * ax + 1.0f;
    if (ax < 2.0f) return ((-0.5f * ax + 2.5f) * ax - 4.0f) * ax + 2.0f;
    return 0.0f;
}

static float *bicubic_resize(const unsigned char *src, int src_w, int src_h,
                              int dst_w, int dst_h) {
    float *dst = (float *)malloc((size_t)dst_h * dst_w * 3 * sizeof(float));
    if (!dst) return NULL;

    for (int y = 0; y < dst_h; y++) {
        /* PIL-compatible coordinate mapping: half-pixel center */
        float src_y = ((float)y + 0.5f) * src_h / dst_h - 0.5f;
        if (src_y < 0.0f) src_y = 0.0f;
        int iy = (int)src_y;
        float fy = src_y - iy;

        for (int x = 0; x < dst_w; x++) {
            float src_x = ((float)x + 0.5f) * src_w / dst_w - 0.5f;
            if (src_x < 0.0f) src_x = 0.0f;
            int ix = (int)src_x;
            float fx = src_x - ix;

            for (int c = 0; c < 3; c++) {
                float val = 0.0f;
                for (int j = -1; j <= 2; j++) {
                    float wy = cubic_weight(fy - j);
                    int sy = iy + j;
                    if (sy < 0) sy = 0;
                    if (sy >= src_h) sy = src_h - 1;
                    for (int i = -1; i <= 2; i++) {
                        float wx = cubic_weight(fx - i);
                        int sx = ix + i;
                        if (sx < 0) sx = 0;
                        if (sx >= src_w) sx = src_w - 1;
                        val += wy * wx * (float)src[(sy * src_w + sx) * 3 + c];
                    }
                }
                /* Clamp to [0, 255] */
                if (val < 0.0f) val = 0.0f;
                if (val > 255.0f) val = 255.0f;
                dst[(y * dst_w + x) * 3 + c] = val;
            }
        }
    }
    return dst;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

float *smol_load_image(const char *path, int target_size, int *out_w, int *out_h) {
    int w, h, channels;
    unsigned char *pixels = stbi_load(path, &w, &h, &channels, 3);  /* force RGB */
    if (!pixels) {
        fprintf(stderr, "smol_image: cannot load %s: %s\n", path, stbi_failure_reason());
        return NULL;
    }

    if (smol_verbose >= 2) {
        fprintf(stderr, "  Image: %dx%d (%d channels) -> %dx%d\n",
                w, h, channels, target_size, target_size);
    }

    /* Bicubic resize to target_size x target_size (matches HF SiglipImageProcessor) */
    float *resized = bicubic_resize(pixels, w, h, target_size, target_size);
    stbi_image_free(pixels);
    if (!resized) return NULL;

    /* Convert to channel-first [3, target_size, target_size] and normalize to [-1, 1] */
    int n = target_size * target_size;
    float *output = (float *)malloc(3 * n * sizeof(float));
    if (!output) {
        free(resized);
        return NULL;
    }

    for (int y = 0; y < target_size; y++) {
        for (int x = 0; x < target_size; x++) {
            int idx = y * target_size + x;
            for (int c = 0; c < 3; c++) {
                float val = resized[idx * 3 + c] / 255.0f * 2.0f - 1.0f;
                output[c * n + idx] = val;
            }
        }
    }

    free(resized);
    *out_w = target_size;
    *out_h = target_size;
    return output;
}
