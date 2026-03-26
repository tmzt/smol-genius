/*
 * hf_tokenizer.h - HuggingFace tokenizer.json loader (GPT-2 byte-level BPE)
 *
 * Loads vocab + merges from tokenizer.json (HF fast tokenizer format).
 * Provides encode (text -> token IDs) and decode (token ID -> text).
 */

#ifndef HF_TOKENIZER_H
#define HF_TOKENIZER_H

typedef struct hf_tokenizer {
    char **id_to_text;   /* [vocab_size] decoded text strings */
    char **id_to_bpe;    /* [vocab_size] raw BPE token strings */
    int vocab_size;

    /* Internal hash maps */
    void *vocab_map;
    int vocab_map_cap;
    void *merge_map;
    int merge_map_cap;
} hf_tokenizer_t;

/* Load tokenizer from model_dir/tokenizer.json */
__attribute__((visibility("default")))
hf_tokenizer_t *hf_tokenizer_load(const char *model_dir);

/* Decode a single token ID to text. Returns pointer to internal string. */
__attribute__((visibility("default")))
const char *hf_tokenizer_decode(const hf_tokenizer_t *tok, int token_id);

/* Encode UTF-8 text into token IDs using BPE.
 * Returns malloc'd array of token IDs and sets *out_n. */
__attribute__((visibility("default")))
int *hf_tokenizer_encode(const hf_tokenizer_t *tok, const char *text, int *out_n);

/* Free tokenizer */
__attribute__((visibility("default")))
void hf_tokenizer_free(hf_tokenizer_t *tok);

#endif /* HF_TOKENIZER_H */
