/*
 * q4_dispatch.c - Dispatch wrappers for Q4 and pooling kernels
 */

#include "smol_kernels.h"
#include "smol_kernels_impl.h"

void smol_dequantize_q4(float *out, const uint8_t *packed,
                         const uint16_t *scales_f16, int n, int block_size) {
    smol_dequantize_q4_impl(out, packed, scales_f16, n, block_size);
}

void smol_mean_pool(float *out, const float *hidden_states,
                     const int *seq_starts, const int *seq_lens,
                     int num_seqs, int hidden) {
    smol_mean_pool_impl(out, hidden_states, seq_starts, seq_lens, num_seqs, hidden);
}
