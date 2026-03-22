/*
 * mean_pool_generic.c - Generic mean pooling
 */

#include "smol_kernels.h"
#include <string.h>

void smol_mean_pool_generic(float *out, const float *hidden_states,
                             const int *seq_starts, const int *seq_lens,
                             int num_seqs, int hidden) {
    for (int s = 0; s < num_seqs; s++) {
        int start = seq_starts[s];
        int len = seq_lens[s];
        float *o = out + s * hidden;
        memset(o, 0, hidden * sizeof(float));
        for (int t = 0; t < len; t++) {
            const float *tok = hidden_states + (start + t) * hidden;
            for (int d = 0; d < hidden; d++) o[d] += tok[d];
        }
        if (len > 0) {
            float inv_len = 1.0f / (float)len;
            for (int d = 0; d < hidden; d++) o[d] *= inv_len;
        }
    }
}
