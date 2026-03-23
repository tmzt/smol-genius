/*
 * gemma.h - Gemma text generation API
 *
 * High-level wrapper over qkn_decoder for Gemma 2 family models.
 * Loads BF16 safetensors, provides tokenize/generate/reset interface.
 */

#ifndef GEMMA_H
#define GEMMA_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Types
 * ======================================================================== */

typedef struct gemma_ctx_t gemma_ctx_t;

typedef void (*gemma_token_cb)(const char *piece, void *userdata);

/* ========================================================================
 * API
 * ======================================================================== */

/* Load a Gemma model from a directory containing safetensors + vocab.json.
 * Auto-detects model size from weight shapes. Returns NULL on error. */
__attribute__((visibility("default")))
gemma_ctx_t *gemma_load(const char *model_dir);

/* Free all resources. */
__attribute__((visibility("default")))
void gemma_free(gemma_ctx_t *ctx);

/* Set a streaming token callback. Called for each generated token.
 * Pass NULL to disable. */
__attribute__((visibility("default")))
void gemma_set_token_callback(gemma_ctx_t *ctx, gemma_token_cb cb, void *userdata);

/* Generate from token IDs. Returns total tokens generated.
 * Stops at EOS or max_tokens. Token callback is invoked per-token with token ID. */
typedef void (*gemma_id_cb)(int token_id, void *userdata);
__attribute__((visibility("default")))
void gemma_set_id_callback(gemma_ctx_t *ctx, gemma_id_cb cb, void *userdata);
__attribute__((visibility("default")))
int gemma_generate(gemma_ctx_t *ctx, const int *prompt_tokens, int n_prompt, int max_tokens);

/* Reset KV cache for a new sequence (called automatically by gemma_generate). */
__attribute__((visibility("default")))
void gemma_reset(gemma_ctx_t *ctx);

/* ========================================================================
 * Globals
 * ======================================================================== */

extern int gemma_verbose;

#endif /* GEMMA_H */
