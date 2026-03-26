/*
 * acoustic_decoder.c - KittenTTS acoustic decoder
 *
 * Architecture:
 *   Adapter: Conv1D [64, 128, 1] (reduce text features)
 *   Concat: adapter(text) [64] + F0 [1] + N [1] = [130, expanded_len]
 *   Encode: AdainResBlk1d [130 -> 256]
 *   4x Decode: concat adapter [64] + prev [256] = [322] -> AdainResBlk1d [322 -> 256]
 */

#include "acoustic_decoder.h"
#include "smol_kernels.h"
#include <string.h>

/* Reuse the same AdainResBlk1d pattern from prosody.c */
static void adain_resblk1d_dec(float *out, const float *in, int ch_in, int ch_out,
                                 int length, const float *style,
                                 const float *conv1_w, const float *conv1_b,
                                 const float *conv2_w, const float *conv2_b,
                                 const float *adain_fc_w, const float *adain_fc_b,
                                 const float *skip_w, const float *skip_b,
                                 float *scratch) {
    /* Project style to mean/std */
    float style_proj[KTTS_DECODER_HIDDEN * 2];
    smol_linear(style_proj, style, adain_fc_w, adain_fc_b,
                1, KTTS_STYLE_ADAIN_DIM, 2 * ch_out);
    float *s_mean = style_proj;
    float *s_std = style_proj + ch_out;

    /* Conv1D #1 */
    float *c1 = scratch;
    smol_conv1d(c1, in, conv1_w, conv1_b,
                ch_in, ch_out, length, 3, 1, 1, 1, 1);
    smol_adain(c1, c1, s_mean, s_std, ch_out, length, 1e-5f);

    /* LeakyReLU */
    for (int i = 0; i < ch_out * length; i++) {
        if (c1[i] < 0.0f) c1[i] *= 0.2f;
    }

    /* Conv1D #2 */
    float *c2 = scratch + (size_t)ch_out * length;
    smol_conv1d(c2, c1, conv2_w, conv2_b,
                ch_out, ch_out, length, 3, 1, 1, 1, 1);
    smol_adain(c2, c2, s_mean, s_std, ch_out, length, 1e-5f);

    /* Skip connection */
    if (skip_w && ch_in != ch_out) {
        smol_conv1d(out, in, skip_w, skip_b,
                    ch_in, ch_out, length, 1, 1, 0, 1, 1);
    } else if (ch_in == ch_out) {
        memcpy(out, in, (size_t)ch_out * length * sizeof(float));
    } else {
        memset(out, 0, (size_t)ch_out * length * sizeof(float));
    }

    smol_add_inplace(out, c2, ch_out * length);
}

void ktts_acoustic_decoder_forward(const ktts_acoustic_dec_t *model,
                                    const float *expanded_text,
                                    const float *f0, const float *noise,
                                    const float *style,
                                    int expanded_len,
                                    float *out, float *scratch) {
    int L = expanded_len;
    int text_dim = KTTS_TEXT_ENC_DIM;  /* 128 */
    int dec_h = KTTS_DECODER_HIDDEN;   /* 256 */

    /* --- Adapter: text [128, L] -> [64, L] via 1x1 conv --- */
    /* First transpose text from [L, 128] row-major to [128, L] channel-first */
    float *text_cf = scratch;
    for (int c = 0; c < text_dim; c++) {
        for (int t = 0; t < L; t++) {
            text_cf[c * L + t] = expanded_text[t * text_dim + c];
        }
    }

    float *adapter_out = text_cf + (size_t)text_dim * L;  /* [64, L] */
    smol_conv1d(adapter_out, text_cf, model->adapter_w, model->adapter_b,
                text_dim, 64, L, 1, 1, 0, 1, 1);

    /* --- Concatenate: adapter [64] + F0 [1] + N [1] = [66, L] then pad to [130, L]
     * Actually the model uses [128 + 1 + 1 = 130] from the full text encoding.
     * Let me use the documented [64 + F0 + N + padding to match] structure. */

    /* Concat into encode input: [adapter(64) + text(64) + F0(1) + N(1) = 130, L] */
    int enc_in_ch = 130;
    float *enc_input = adapter_out + 64 * (size_t)L;  /* [130, L] */

    /* Copy adapter output (64 channels) */
    memcpy(enc_input, adapter_out, 64 * (size_t)L * sizeof(float));
    /* Copy first 64 channels of text (as additional features) */
    memcpy(enc_input + 64 * (size_t)L, text_cf, 64 * (size_t)L * sizeof(float));
    /* Copy F0 (1 channel) */
    memcpy(enc_input + 128 * (size_t)L, f0, L * sizeof(float));
    /* Copy N (1 channel) */
    memcpy(enc_input + 129 * (size_t)L, noise, L * sizeof(float));

    /* --- Encode block: [130, L] -> [256, L] --- */
    float *enc_out = enc_input + enc_in_ch * (size_t)L; /* [256, L] */
    float *extra = enc_out + dec_h * (size_t)L;

    adain_resblk1d_dec(enc_out, enc_input, enc_in_ch, dec_h, L, style,
                       model->enc_conv1_w, model->enc_conv1_b,
                       model->enc_conv2_w, model->enc_conv2_b,
                       model->enc_adain_fc_w, model->enc_adain_fc_b,
                       model->enc_skip_w, model->enc_skip_b,
                       extra);

    /* --- 4 Decode blocks --- */
    /* Each takes concat [adapter(64) + prev(256) + F0(1) + N(1) = 322, L] -> [256, L] */
    int dec_in_ch = 64 + dec_h + 1 + 1; /* 322 */
    float *dec_input = extra;  /* [322, L] */
    float *dec_out = dec_input + dec_in_ch * (size_t)L;
    float *dec_scratch = dec_out + dec_h * (size_t)L;

    for (int blk = 0; blk < KTTS_DECODER_BLOCKS; blk++) {
        /* Build concatenated input */
        memcpy(dec_input, adapter_out, 64 * (size_t)L * sizeof(float));
        memcpy(dec_input + 64 * (size_t)L, enc_out, dec_h * (size_t)L * sizeof(float));
        memcpy(dec_input + (64 + dec_h) * (size_t)L, f0, L * sizeof(float));
        memcpy(dec_input + (64 + dec_h + 1) * (size_t)L, noise, L * sizeof(float));

        adain_resblk1d_dec(dec_out, dec_input, dec_in_ch, dec_h, L, style,
                           model->dec_blocks[blk].conv1_w, model->dec_blocks[blk].conv1_b,
                           model->dec_blocks[blk].conv2_w, model->dec_blocks[blk].conv2_b,
                           model->dec_blocks[blk].adain_fc_w, model->dec_blocks[blk].adain_fc_b,
                           model->dec_blocks[blk].skip_w, model->dec_blocks[blk].skip_b,
                           dec_scratch);

        /* dec_out becomes next enc_out for concatenation */
        memcpy(enc_out, dec_out, dec_h * (size_t)L * sizeof(float));
    }

    /* Copy final output [256, L] */
    memcpy(out, enc_out, dec_h * (size_t)L * sizeof(float));
}
