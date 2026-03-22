/*
 * tokenizer.h - smol BPE tokenizer (GPT-2 byte-level)
 *
 * Supports:
 *  - decode token ID -> UTF-8 text
 *  - encode UTF-8 text -> token IDs (vocab.json + merges.txt)
 */

#ifndef SMOL_TOKENIZER_H
#define SMOL_TOKENIZER_H

typedef struct {
    char **id_to_text;   /* [vocab_size] decoded text strings */
    char **id_to_bpe;    /* [vocab_size] raw BPE token strings from vocab.json */
    int vocab_size;

    /* Internal hash maps (opaque to callers) */
    void *vocab_map;
    int vocab_map_cap;
    void *merge_map;
    int merge_map_cap;
} smol_tokenizer_t;

/* Load tokenizer from vocab.json in model directory */
__attribute__((visibility("default")))
smol_tokenizer_t *smol_tokenizer_load(const char *vocab_json_path);

/* Decode a single token ID to text. Returns pointer to internal string. */
__attribute__((visibility("default")))
const char *smol_tokenizer_decode(const smol_tokenizer_t *tok, int token_id);

/* Encode UTF-8 text into token IDs using BPE.
 * Returns malloc'd array of token IDs and sets *out_n_tokens.
 * Returns NULL on error (and sets *out_n_tokens to 0). */
__attribute__((visibility("default")))
int *smol_tokenizer_encode(const smol_tokenizer_t *tok, const char *text, int *out_n_tokens);

/* Free tokenizer */
__attribute__((visibility("default")))
void smol_tokenizer_free(smol_tokenizer_t *tok);

#endif /* SMOL_TOKENIZER_H */
