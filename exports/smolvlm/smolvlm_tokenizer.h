/*
 * smolvlm_tokenizer.h - Thin wrapper around common hf_tokenizer for SmolVLM
 */

#ifndef SMOLVLM_TOKENIZER_H
#define SMOLVLM_TOKENIZER_H

#include "../../common/utils/hf_tokenizer.h"

typedef hf_tokenizer_t smolvlm_tokenizer_t;

#define smolvlm_tokenizer_load   hf_tokenizer_load
#define smolvlm_tokenizer_decode hf_tokenizer_decode
#define smolvlm_tokenizer_encode hf_tokenizer_encode
#define smolvlm_tokenizer_free   hf_tokenizer_free

/* Back-compat: old code uses `struct smolvlm_tokenizer *` */
#define smolvlm_tokenizer hf_tokenizer

#endif /* SMOLVLM_TOKENIZER_H */
