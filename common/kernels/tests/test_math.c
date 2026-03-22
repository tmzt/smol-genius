/*
 * test_math.c - Pure C test harness for smol kernel library
 *
 * Validates correctness of core math operations across all backends.
 */

#include "../smol_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ASSERT_NEAR(a, b, tol, msg) do {                        \
    float _a = (a), _b = (b), _t = (tol);                      \
    if (fabsf(_a - _b) > _t) {                                 \
        fprintf(stderr, "FAIL: %s: %.6f != %.6f (tol=%.6f)\n", \
                msg, _a, _b, _t);                               \
        failures++;                                             \
    }                                                           \
} while (0)

static int failures = 0;

static void test_add_inplace(void) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    smol_add_inplace(a, b, 4);
    ASSERT_NEAR(a[0], 6.0f, 1e-6f, "add_inplace[0]");
    ASSERT_NEAR(a[1], 8.0f, 1e-6f, "add_inplace[1]");
    ASSERT_NEAR(a[2], 10.0f, 1e-6f, "add_inplace[2]");
    ASSERT_NEAR(a[3], 12.0f, 1e-6f, "add_inplace[3]");
}

static void test_scale(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    smol_scale(x, 2.5f, 3);
    ASSERT_NEAR(x[0], 2.5f, 1e-6f, "scale[0]");
    ASSERT_NEAR(x[1], 5.0f, 1e-6f, "scale[1]");
    ASSERT_NEAR(x[2], 7.5f, 1e-6f, "scale[2]");
}

static void test_matmul_t(void) {
    /* A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
     * C = A @ B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
     *             = [[17, 23], [39, 53]] */
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float C[4];
    smol_matmul_t(C, A, B, 2, 2, 2);
    ASSERT_NEAR(C[0], 17.0f, 1e-4f, "matmul_t[0,0]");
    ASSERT_NEAR(C[1], 23.0f, 1e-4f, "matmul_t[0,1]");
    ASSERT_NEAR(C[2], 39.0f, 1e-4f, "matmul_t[1,0]");
    ASSERT_NEAR(C[3], 53.0f, 1e-4f, "matmul_t[1,1]");
}

static void test_rms_norm(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    /* rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386 */
    float rms = sqrtf(30.0f / 4.0f);
    smol_rms_norm(out, x, w, 1, 4, 1e-6f);
    ASSERT_NEAR(out[0], 1.0f / rms, 1e-4f, "rms_norm[0]");
    ASSERT_NEAR(out[1], 2.0f / rms, 1e-4f, "rms_norm[1]");
}

static void test_silu(void) {
    float x[] = {0.0f, 1.0f};
    smol_silu(x, 2);
    ASSERT_NEAR(x[0], 0.0f, 1e-6f, "silu(0)");
    /* silu(1) = 1 / (1 + exp(-1)) ≈ 0.7311 */
    ASSERT_NEAR(x[1], 1.0f / (1.0f + expf(-1.0f)), 1e-4f, "silu(1)");
}

static void test_softmax(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    smol_softmax(x, 1, 3);
    float sum = x[0] + x[1] + x[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f, "softmax_sum");
    /* x[2] should be largest */
    if (x[2] < x[1] || x[1] < x[0]) {
        fprintf(stderr, "FAIL: softmax ordering\n");
        failures++;
    }
}

/* ========================================================================
 * Q4 Dequantization Tests
 * ======================================================================== */

/* Helper: pack two 4-bit nibbles into one byte */
static uint8_t pack_q4(int lo, int hi) { return (uint8_t)((hi << 4) | (lo & 0x0F)); }

/* Helper: f32 to f16 bits (simplified, handles normal range) */
static uint16_t f32_to_f16_bits(float val) {
    uint32_t bits;
    memcpy(&bits, &val, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static void test_dequant_basic(void) {
    /* Single block of 32 elements, scale=0.5
     * byte 0x80: lo=0 (weight=-8), hi=8 (weight=0) */
    const int block_size = 32;
    uint8_t packed[16];
    for (int i = 0; i < 16; i++) packed[i] = pack_q4(0, 8);

    uint16_t scale = f32_to_f16_bits(0.5f);
    float out[32];
    smol_dequantize_q4(out, packed, &scale, 32, block_size);

    /* Even indices: (0-8)*0.5 = -4.0, Odd indices: (8-8)*0.5 = 0.0 */
    for (int i = 0; i < 32; i += 2) {
        ASSERT_NEAR(out[i], -4.0f, 1e-3f, "dequant_basic_even");
        ASSERT_NEAR(out[i + 1], 0.0f, 1e-3f, "dequant_basic_odd");
    }
}

static void test_dequant_zero_scale(void) {
    /* Scale=0 should produce all zeros */
    const int block_size = 32;
    uint8_t packed[16];
    for (int i = 0; i < 16; i++) packed[i] = 0xFF; /* all nibbles=15 */

    uint16_t scale = 0; /* f16 zero */
    float out[32];
    smol_dequantize_q4(out, packed, &scale, 32, block_size);

    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(out[i], 0.0f, 1e-6f, "dequant_zero_scale");
    }
}

static void test_dequant_symmetry(void) {
    /* Nibble=8 is zero point, should produce exactly 0.0 for any scale */
    const int block_size = 32;
    uint8_t packed[16];
    for (int i = 0; i < 16; i++) packed[i] = 0x88; /* lo=8, hi=8 */

    uint16_t scale = f32_to_f16_bits(42.0f);
    float out[32];
    smol_dequantize_q4(out, packed, &scale, 32, block_size);

    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(out[i], 0.0f, 1e-6f, "dequant_symmetry");
    }
}

static void test_dequant_multi_block(void) {
    /* 2 blocks with different scales */
    const int block_size = 32;
    uint8_t packed[32]; /* 2 blocks * 16 bytes */
    for (int i = 0; i < 32; i++) packed[i] = pack_q4(15, 0); /* lo=15(+7), hi=0(-8) */

    uint16_t scales[2] = { f32_to_f16_bits(1.0f), f32_to_f16_bits(2.0f) };
    float out[64];
    smol_dequantize_q4(out, packed, scales, 64, block_size);

    /* Block 0: even=7*1=7, odd=-8*1=-8 */
    ASSERT_NEAR(out[0], 7.0f, 1e-3f, "dequant_multi_b0_even");
    ASSERT_NEAR(out[1], -8.0f, 1e-3f, "dequant_multi_b0_odd");
    /* Block 1: even=7*2=14, odd=-8*2=-16 */
    ASSERT_NEAR(out[32], 14.0f, 1e-2f, "dequant_multi_b1_even");
    ASSERT_NEAR(out[33], -16.0f, 1e-2f, "dequant_multi_b1_odd");
}

/* ========================================================================
 * Mean Pooling Tests
 * ======================================================================== */

static void test_mean_pool_single(void) {
    /* 3 tokens, hidden=4, single sequence */
    float hidden_states[] = {
        1.0f, 2.0f, 3.0f, 4.0f,   /* token 0 */
        5.0f, 6.0f, 7.0f, 8.0f,   /* token 1 */
        3.0f, 3.0f, 3.0f, 3.0f,   /* token 2 */
    };
    int seq_starts[] = {0};
    int seq_lens[] = {3};
    float out[4];

    smol_mean_pool(out, hidden_states, seq_starts, seq_lens, 1, 4);

    /* mean = [(1+5+3)/3, (2+6+3)/3, (3+7+3)/3, (4+8+3)/3] = [3, 11/3, 13/3, 5] */
    ASSERT_NEAR(out[0], 3.0f, 1e-5f, "mean_pool_single[0]");
    ASSERT_NEAR(out[1], 11.0f / 3.0f, 1e-5f, "mean_pool_single[1]");
    ASSERT_NEAR(out[2], 13.0f / 3.0f, 1e-5f, "mean_pool_single[2]");
    ASSERT_NEAR(out[3], 5.0f, 1e-5f, "mean_pool_single[3]");
}

static void test_mean_pool_batch(void) {
    /* 2 sequences: [2 tokens, 1 token], hidden=2 */
    float hidden_states[] = {
        1.0f, 2.0f,    /* seq 0, tok 0 */
        3.0f, 4.0f,    /* seq 0, tok 1 */
        10.0f, 20.0f,  /* seq 1, tok 0 */
    };
    int seq_starts[] = {0, 2};
    int seq_lens[] = {2, 1};
    float out[4];

    smol_mean_pool(out, hidden_states, seq_starts, seq_lens, 2, 2);

    /* seq 0: mean = [(1+3)/2, (2+4)/2] = [2, 3] */
    ASSERT_NEAR(out[0], 2.0f, 1e-5f, "mean_pool_batch_s0[0]");
    ASSERT_NEAR(out[1], 3.0f, 1e-5f, "mean_pool_batch_s0[1]");
    /* seq 1: mean = [10, 20] (single token = identity) */
    ASSERT_NEAR(out[2], 10.0f, 1e-5f, "mean_pool_batch_s1[0]");
    ASSERT_NEAR(out[3], 20.0f, 1e-5f, "mean_pool_batch_s1[1]");
}

static void test_mean_pool_uniform(void) {
    /* All tokens identical -> mean equals any single token */
    float hidden_states[4 * 8]; /* 4 tokens, hidden=8 */
    for (int t = 0; t < 4; t++)
        for (int d = 0; d < 8; d++)
            hidden_states[t * 8 + d] = (float)(d + 1);

    int seq_starts[] = {0};
    int seq_lens[] = {4};
    float out[8];

    smol_mean_pool(out, hidden_states, seq_starts, seq_lens, 1, 8);

    for (int d = 0; d < 8; d++) {
        ASSERT_NEAR(out[d], (float)(d + 1), 1e-5f, "mean_pool_uniform");
    }
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void) {
    printf("Running smol kernel tests...\n");

    test_add_inplace();
    test_scale();
    test_matmul_t();
    test_rms_norm();
    test_silu();
    test_softmax();

    printf("Running Q4 dequantization tests...\n");

    test_dequant_basic();
    test_dequant_zero_scale();
    test_dequant_symmetry();
    test_dequant_multi_block();

    printf("Running mean pooling tests...\n");

    test_mean_pool_single();
    test_mean_pool_batch();
    test_mean_pool_uniform();

    if (failures == 0) {
        printf("All tests passed.\n");
    } else {
        printf("%d test(s) FAILED.\n", failures);
    }
    return failures > 0 ? 1 : 0;
}
