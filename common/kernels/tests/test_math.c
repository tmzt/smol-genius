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

int main(void) {
    printf("Running smol kernel tests...\n");

    test_add_inplace();
    test_scale();
    test_matmul_t();
    test_rms_norm();
    test_silu();
    test_softmax();

    if (failures == 0) {
        printf("All tests passed.\n");
    } else {
        printf("%d test(s) FAILED.\n", failures);
    }
    return failures > 0 ? 1 : 0;
}
