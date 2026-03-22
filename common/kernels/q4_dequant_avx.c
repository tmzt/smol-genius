/*
 * q4_dequant_avx.c - AVX2 Q4 dequantization kernel
 *
 * Processes 32 packed bytes (64 weights) per inner iteration using
 * AVX2 integer shuffle + shift for nibble extraction.
 */

#include "smol_kernels_impl.h"

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>
#include <string.h>

/* f16 to f32 scalar (no AVX F16C dependency for portability) */
static inline float f16_to_f32_scalar(uint16_t h) {
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
    float val;
    memcpy(&val, &f32_bits, sizeof(float));
    return val;
}

void smol_dequantize_q4_avx(float *out, const uint8_t *packed,
                              const uint16_t *scales_f16,
                              int n, int block_size) {
    int num_blocks = n / block_size;
    int bytes_per_block = block_size / 2;
    __m256i mask_lo = _mm256_set1_epi8(0x0F);
    __m256i offset = _mm256_set1_epi32(8);

    for (int b = 0; b < num_blocks; b++) {
        float scale = f16_to_f32_scalar(scales_f16[b]);
        __m256 sv = _mm256_set1_ps(scale);

        const uint8_t *src = packed + b * bytes_per_block;
        float *dst = out + b * block_size;

        int i = 0;
        /* Process 32 packed bytes -> 64 f32 values per iteration */
        for (; i + 32 <= bytes_per_block; i += 32) {
            __m256i raw = _mm256_loadu_si256((const __m256i *)(src + i));

            /* Extract low nibbles (even indices) and high nibbles (odd indices) */
            __m256i lo = _mm256_and_si256(raw, mask_lo);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_lo);

            /* Process low nibbles: 32 x u8 -> 4 groups of 8 x f32 */
            __m128i lo_128_0 = _mm256_castsi256_si128(lo);
            __m128i lo_128_1 = _mm256_extracti128_si256(lo, 1);

            /* First 8 low nibbles */
            __m256i lo32_0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(lo_128_0), offset);
            __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(lo32_0), sv);
            /* Next 8 low nibbles */
            __m256i lo32_1 = _mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(lo_128_0, 8)), offset);
            __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(lo32_1), sv);
            /* Next 8 */
            __m256i lo32_2 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(lo_128_1), offset);
            __m256 f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(lo32_2), sv);
            /* Last 8 */
            __m256i lo32_3 = _mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(lo_128_1, 8)), offset);
            __m256 f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(lo32_3), sv);

            /* Process high nibbles similarly */
            __m128i hi_128_0 = _mm256_castsi256_si128(hi);
            __m128i hi_128_1 = _mm256_extracti128_si256(hi, 1);

            __m256i hi32_0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(hi_128_0), offset);
            __m256 g0 = _mm256_mul_ps(_mm256_cvtepi32_ps(hi32_0), sv);
            __m256i hi32_1 = _mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(hi_128_0, 8)), offset);
            __m256 g1 = _mm256_mul_ps(_mm256_cvtepi32_ps(hi32_1), sv);
            __m256i hi32_2 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(hi_128_1), offset);
            __m256 g2 = _mm256_mul_ps(_mm256_cvtepi32_ps(hi32_2), sv);
            __m256i hi32_3 = _mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(hi_128_1, 8)), offset);
            __m256 g3 = _mm256_mul_ps(_mm256_cvtepi32_ps(hi32_3), sv);

            /* Interleave and store: [lo0,hi0, lo1,hi1, ...] */
            int off = i * 2;
            /* Unpack pairs: lo[0..7] into even slots, hi[0..7] into odd slots */
            __m256 p0 = _mm256_unpacklo_ps(f0, g0);
            __m256 p1 = _mm256_unpackhi_ps(f0, g0);
            _mm256_storeu_ps(dst + off +  0, _mm256_permute2f128_ps(p0, p1, 0x20));
            _mm256_storeu_ps(dst + off +  8, _mm256_permute2f128_ps(p0, p1, 0x31));

            p0 = _mm256_unpacklo_ps(f1, g1);
            p1 = _mm256_unpackhi_ps(f1, g1);
            _mm256_storeu_ps(dst + off + 16, _mm256_permute2f128_ps(p0, p1, 0x20));
            _mm256_storeu_ps(dst + off + 24, _mm256_permute2f128_ps(p0, p1, 0x31));

            p0 = _mm256_unpacklo_ps(f2, g2);
            p1 = _mm256_unpackhi_ps(f2, g2);
            _mm256_storeu_ps(dst + off + 32, _mm256_permute2f128_ps(p0, p1, 0x20));
            _mm256_storeu_ps(dst + off + 40, _mm256_permute2f128_ps(p0, p1, 0x31));

            p0 = _mm256_unpacklo_ps(f3, g3);
            p1 = _mm256_unpackhi_ps(f3, g3);
            _mm256_storeu_ps(dst + off + 48, _mm256_permute2f128_ps(p0, p1, 0x20));
            _mm256_storeu_ps(dst + off + 56, _mm256_permute2f128_ps(p0, p1, 0x31));
        }

        /* Scalar tail */
        for (; i < bytes_per_block; i++) {
            uint8_t byte = src[i];
            dst[i * 2]     = (float)((int)(byte & 0x0F) - 8) * scale;
            dst[i * 2 + 1] = (float)((int)(byte >> 4)   - 8) * scale;
        }
    }
}

#endif /* __AVX2__ && __FMA__ */
