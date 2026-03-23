/*
 * smol_kernels.h - Unified math kernel API for smol-genius inference
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 * Architecture dispatch is handled internally via smol_kernels_impl.h.
 */

#ifndef SMOL_KERNELS_H
#define SMOL_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

__attribute__((visibility("default")))
void smol_add_inplace(float *a, const float *b, int n);
__attribute__((visibility("default")))
void smol_mul_inplace(float *a, const float *b, int n);
__attribute__((visibility("default")))
void smol_scale(float *x, float s, int n);
__attribute__((visibility("default")))
void smol_copy(float *dst, const float *src, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* C = A @ B^T: A[M,K], B[N,K], C[M,N] */
__attribute__((visibility("default")))
void smol_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);

/* y = x @ W^T + b: x[seq,in], W[out,in], b[out], y[seq,out] */
__attribute__((visibility("default")))
void smol_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim);

__attribute__((visibility("default")))
void smol_linear_nobias(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim);

/* bf16 weight variants */
__attribute__((visibility("default")))
void smol_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                      const float *b, int seq_len, int in_dim, int out_dim);

__attribute__((visibility("default")))
void smol_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim);

/* seq=1 decoder fast path: compute Q/K/V matvecs with one threaded dispatch */
__attribute__((visibility("default")))
void smol_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                 const uint16_t *Wq_bf16,
                                 const uint16_t *Wk_bf16,
                                 const uint16_t *Wv_bf16,
                                 int in_dim, int q_dim, int kv_dim);

__attribute__((visibility("default")))
void smol_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                         int M, int K, int N);

/* ========================================================================
 * 2D Convolution
 * ======================================================================== */

__attribute__((visibility("default")))
void smol_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding);

/* ========================================================================
 * Normalization
 * ======================================================================== */

__attribute__((visibility("default")))
void smol_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps);

__attribute__((visibility("default")))
void smol_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps);

__attribute__((visibility("default")))
void smol_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

__attribute__((visibility("default")))
void smol_silu(float *x, int n);
__attribute__((visibility("default")))
void smol_gelu(float *x, int n);
__attribute__((visibility("default")))
void smol_softmax(float *x, int rows, int cols);
__attribute__((visibility("default")))
void smol_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate);
__attribute__((visibility("default")))
void smol_geglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

__attribute__((visibility("default")))
void smol_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq, int n_heads,
                                   int head_dim, float scale,
                                   const int *window_starts, int n_windows);

__attribute__((visibility("default")))
void smol_causal_attention(float *out, const float *Q, const float *K, const float *V,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset);

__attribute__((visibility("default")))
void smol_sliding_window_attention(float *out, const float *Q, const float *K,
                                    const float *V, int seq_q, int seq_k,
                                    int n_heads, int n_kv_heads, int head_dim,
                                    float scale, int q_offset, int window_size);

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

__attribute__((visibility("default")))
void smol_sinusoidal_pe(float *pe, int n_pos, int d_model);

__attribute__((visibility("default")))
void smol_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta);

__attribute__((visibility("default")))
void smol_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim);

/* Streaming argmax: finds argmax(W_bf16 @ x) without materializing full logits. */
__attribute__((visibility("default")))
int smol_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                             int in_dim, int out_dim);

/* ========================================================================
 * Q4 Dequantization
 * ======================================================================== */

/* Dequantize Q4 packed weights to f32.
 * packed: [n/2] bytes, each holding two 4-bit weights
 * scales_f16: [n/block_size] f16 per-block scales
 * out: [n] f32 output
 * Dequant: out[i] = (nibble_i - 8) * scale[i / block_size] */
__attribute__((visibility("default")))
void smol_dequantize_q4(float *out, const uint8_t *packed,
                         const uint16_t *scales_f16, int n, int block_size);

/* ========================================================================
 * Mean Pooling
 * ======================================================================== */

/* Mean pool hidden states across the token dimension.
 * hidden_states: [total_tokens, hidden] row-major
 * seq_starts: [num_seqs] start index of each sequence
 * seq_lens: [num_seqs] length of each sequence
 * out: [num_seqs, hidden] output embeddings */
__attribute__((visibility("default")))
void smol_mean_pool(float *out, const float *hidden_states,
                     const int *seq_starts, const int *seq_lens,
                     int num_seqs, int hidden);

/* ========================================================================
 * Threading
 * ======================================================================== */

__attribute__((visibility("default")))
void smol_set_threads(int n);
__attribute__((visibility("default")))
int smol_get_num_cpus(void);
__attribute__((visibility("default")))
int smol_get_thread_count(void);

/* Internal: parallel dispatch used by kernel implementations */
__attribute__((visibility("default")))
void smol_parallel_for(void (*fn)(int tid, int n_threads, void *arg), void *arg);

/* Global verbose flag */
__attribute__((visibility("default")))
extern int smol_verbose;

#endif /* SMOL_KERNELS_H */
