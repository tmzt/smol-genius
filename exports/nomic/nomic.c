/*
 * nomic.c - Nomic stateless embedding export
 *
 * Skeleton: loads weights via safetensors, computes embeddings
 * via the smol kernel library.
 */

#include "../../common/kernels/smol_kernels.h"
#include "../../common/utils/safetensors.h"
