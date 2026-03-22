/*
 * q4_dequant_generic.c - Generic Q4 dequantization
 */

#include "smol_kernels.h"
#include <string.h>

void smol_dequantize_q4_generic(float *out, const uint8_t *packed,
                                 const uint16_t *scales_f16,
                                 int n, int block_size) {
    int num_blocks = n / block_size;
    int bytes_per_block = block_size / 2;

    for (int b = 0; b < num_blocks; b++) {
        /* f16 to f32 scalar */
        uint16_t h = scales_f16[b];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f32_bits;
        if (exp == 0) {
            f32_bits = sign;
        } else if (exp == 31) {
            f32_bits = sign | 0x7F800000 | (mant << 13);
        } else {
            f32_bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
        float scale;
        memcpy(&scale, &f32_bits, sizeof(float));

        const uint8_t *src = packed + b * bytes_per_block;
        float *dst = out + b * block_size;

        for (int i = 0; i < bytes_per_block; i++) {
            uint8_t byte = src[i];
            dst[i * 2]     = (float)((int)(byte & 0x0F) - 8) * scale;
            dst[i * 2 + 1] = (float)((int)(byte >> 4)   - 8) * scale;
        }
    }
}
