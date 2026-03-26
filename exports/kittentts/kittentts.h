/*
 * kittentts.h - KittenTTS text-to-speech inference API
 *
 * Pure C reimplementation of the KittenTTS nano model (15M params).
 * Pipeline: text -> espeak-ng phonemization -> PL-BERT -> text encoder ->
 *           prosody prediction -> acoustic decoding -> iSTFT vocoder -> 24kHz audio
 *
 * Usage:
 *   kittentts_ctx_t *ctx = kittentts_load("kitten-tts-nano");
 *   int n_samples;
 *   float *audio = kittentts_synthesize(ctx, "Hello world", 0, 1.0f, &n_samples);
 *   smol_write_wav("out.wav", audio, n_samples, 24000);
 *   free(audio);
 *   kittentts_free(ctx);
 */

#ifndef KITTENTTS_H
#define KITTENTTS_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Model Constants (nano variant)
 * ======================================================================== */

#define KTTS_PHONEME_VOCAB      178
#define KTTS_BERT_DIM           128
#define KTTS_BERT_HIDDEN        768
#define KTTS_BERT_FFN           2048
#define KTTS_BERT_MAX_POS       512
#define KTTS_BERT_HEADS         12      /* 768 / 64 */
#define KTTS_BERT_HEAD_DIM      64

#define KTTS_TEXT_ENC_DIM       128
#define KTTS_TEXT_ENC_CNN_K     5
#define KTTS_TEXT_ENC_CNN_N     2
#define KTTS_TEXT_ENC_LSTM_H    64      /* per direction, 128 total */

#define KTTS_STYLE_DIM          256
#define KTTS_STYLE_ADAIN_DIM    128     /* style projection for AdaIN */

#define KTTS_PROSODY_LSTM_H     64      /* per direction */
#define KTTS_PROSODY_LSTM_N     4
#define KTTS_DURATION_CLASSES   50

#define KTTS_DECODER_HIDDEN     256
#define KTTS_DECODER_BLOCKS     4

#define KTTS_VOC_UPSAMPLE_0     10      /* first upsample stride */
#define KTTS_VOC_UPSAMPLE_1     6       /* second upsample stride */
#define KTTS_VOC_CH_0           128     /* channels after first upsample */
#define KTTS_VOC_CH_1           64      /* channels after second upsample */
#define KTTS_VOC_NFFT           20
#define KTTS_VOC_HOP            5
#define KTTS_VOC_NFREQ          11      /* n_fft/2 + 1 */

#define KTTS_SAMPLE_RATE        24000
#define KTTS_MAX_PHONEMES       512

#define KTTS_NUM_STYLES         8

/* ========================================================================
 * Style / Voice names
 * ======================================================================== */

static const char *KTTS_STYLE_NAMES[KTTS_NUM_STYLES] = {
    "Bella", "Jasper", "Luna", "Bruno",
    "Rosie", "Hugo", "Kiki", "Leo"
};

/* ========================================================================
 * Weight Structures
 * ======================================================================== */

typedef struct {
    /* Token + position embeddings */
    float *tok_emb;             /* [178, 128] */
    float *pos_emb;             /* [512, 128] */
    float *tok_type_emb;        /* [2, 128] */
    float *emb_ln_w;            /* [128] */
    float *emb_ln_b;            /* [128] */

    /* Hidden mapping 128 -> 768 */
    float *hidden_map_w;        /* [768, 128] */
    float *hidden_map_b;        /* [768] */

    /* Shared ALBERT layer */
    float *attn_ln_w;           /* [768] */
    float *attn_ln_b;           /* [768] */
    float *attn_q_w;            /* [768, 768] */
    float *attn_k_w;            /* [768, 768] */
    float *attn_v_w;            /* [768, 768] */
    float *attn_o_w;            /* [768, 768] */
    float *attn_o_b;            /* [768] */
    float *ffn_ln_w;            /* [768] */
    float *ffn_ln_b;            /* [768] */
    float *ffn_up_w;            /* [2048, 768] */
    float *ffn_up_b;            /* [2048] */
    float *ffn_down_w;          /* [768, 2048] */
    float *ffn_down_b;          /* [768] */

    /* Output projection 768 -> 128 */
    float *out_proj_w;          /* [128, 768] */
    float *out_proj_b;          /* [128] */

    int n_iterations;           /* number of times to reuse the shared layer */
} ktts_plbert_t;

typedef struct {
    float *phoneme_emb;         /* [178, 128] */
    /* 2 CNN layers */
    float *conv_w[KTTS_TEXT_ENC_CNN_N];     /* [128, 128, 5] each */
    float *conv_b[KTTS_TEXT_ENC_CNN_N];     /* [128] each */
    float *conv_ln_w[KTTS_TEXT_ENC_CNN_N];  /* [128] each */
    float *conv_ln_b[KTTS_TEXT_ENC_CNN_N];  /* [128] each */
    /* BiLSTM */
    float *lstm_W_ih_fwd;       /* [256, 128] */
    float *lstm_W_hh_fwd;       /* [256, 64] */
    float *lstm_b_ih_fwd;       /* [256] */
    float *lstm_b_hh_fwd;       /* [256] */
    float *lstm_W_ih_bwd;       /* [256, 128] */
    float *lstm_W_hh_bwd;       /* [256, 64] */
    float *lstm_b_ih_bwd;       /* [256] */
    float *lstm_b_hh_bwd;       /* [256] */
} ktts_text_enc_t;

typedef struct {
    /* Duration encoder: 4x BiLSTM + AdaLayerNorm */
    struct {
        float *W_ih_fwd;        /* [256, input] */
        float *W_hh_fwd;        /* [256, 64] */
        float *b_ih_fwd;        /* [256] */
        float *b_hh_fwd;        /* [256] */
        float *W_ih_bwd;
        float *W_hh_bwd;
        float *b_ih_bwd;
        float *b_hh_bwd;
    } dur_lstm[KTTS_PROSODY_LSTM_N];

    /* AdaLayerNorm style projections for duration LSTMs */
    float *dur_ada_fc_w[KTTS_PROSODY_LSTM_N]; /* style -> gamma/beta */
    float *dur_ada_fc_b[KTTS_PROSODY_LSTM_N];

    /* Duration output projection */
    float *dur_proj_w;          /* [50, 128] */
    float *dur_proj_b;          /* [50] */

    /* F0 predictor: 3x AdainResBlk1d */
    struct {
        float *conv1_w;         /* Conv1D weight */
        float *conv1_b;
        float *conv2_w;
        float *conv2_b;
        float *adain_fc_w;      /* style -> mean/std */
        float *adain_fc_b;
        float *downsample_w;    /* optional 1x1 conv for channel change */
        float *downsample_b;
    } f0_blocks[3];
    float *f0_proj_w;           /* [1, 64, 1] */
    float *f0_proj_b;           /* [1] */

    /* N predictor: same structure as F0 */
    struct {
        float *conv1_w;
        float *conv1_b;
        float *conv2_w;
        float *conv2_b;
        float *adain_fc_w;
        float *adain_fc_b;
        float *downsample_w;
        float *downsample_b;
    } n_blocks[3];
    float *n_proj_w;
    float *n_proj_b;
} ktts_prosody_t;

typedef struct {
    /* ASR adapter: Conv1D [64, 128, 1] */
    float *adapter_w;           /* [64, 128, 1] */
    float *adapter_b;           /* [64] */

    /* Encode block */
    float *enc_conv1_w;
    float *enc_conv1_b;
    float *enc_conv2_w;
    float *enc_conv2_b;
    float *enc_adain_fc_w;
    float *enc_adain_fc_b;
    float *enc_skip_w;          /* 1x1 skip conv */
    float *enc_skip_b;

    /* 4 decode blocks */
    struct {
        float *conv1_w;
        float *conv1_b;
        float *conv2_w;
        float *conv2_b;
        float *adain_fc_w;
        float *adain_fc_b;
        float *skip_w;
        float *skip_b;
    } dec_blocks[KTTS_DECODER_BLOCKS];
} ktts_acoustic_dec_t;

typedef struct {
    /* Upsample transposed convolutions */
    float *up0_w;               /* [256, 128, 20] */
    float *up0_b;               /* [128] */
    float *up1_w;               /* [128, 64, 12] */
    float *up1_b;               /* [64] */

    /* Residual blocks (2 per upsample stage) */
    struct {
        float *conv1_w;         /* dilated conv */
        float *conv1_b;
        float *conv2_w;
        float *conv2_b;
        float *conv3_w;
        float *conv3_b;
        float *adain_fc_w;
        float *adain_fc_b;
        float *snake_alpha;     /* per-channel Snake param */
    } resblocks[4];

    /* Noise residual blocks */
    struct {
        float *conv1_w;
        float *conv1_b;
        float *conv2_w;
        float *conv2_b;
    } noise_res[2];

    /* Noise convolutions */
    float *noise_conv0_w;       /* [128, 22, 12] */
    float *noise_conv0_b;
    float *noise_conv1_w;       /* [64, 22, 1] */
    float *noise_conv1_b;

    /* Source module */
    float *source_linear_b;     /* [1] */

    /* Output conv */
    float *conv_post_w;         /* [22, 64, 7] */
    float *conv_post_b;         /* [22] */
} ktts_vocoder_t;

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    /* Sub-model weights */
    ktts_plbert_t       plbert;
    ktts_text_enc_t     text_enc;
    ktts_prosody_t      prosody;
    ktts_acoustic_dec_t decoder;
    ktts_vocoder_t      vocoder;

    /* Style embeddings: [num_styles, style_dim] */
    float *styles;

    /* Phoneme vocabulary map (populated from phoneme_vocab.json) */
    int phoneme_map[256];       /* ASCII/IPA char -> token ID */

    /* Safetensors handle */
    void *safetensors;

    /* Pre-allocated activation buffers */
    float *bert_out;            /* [max_seq, 128] */
    float *text_enc_out;        /* [max_seq, 128] */
    float *style_proj;          /* [style_adain_dim * 2] for mean/std */
    float *expanded_seq;        /* [max_expanded, 128] post-duration */
    float *acoustic_out;        /* [max_expanded, 256] */

    /* Scratch buffers */
    float *scratch_a;
    float *scratch_b;
    size_t scratch_size;

    int max_phonemes;
    int max_expanded_len;
} kittentts_ctx_t;

/* ========================================================================
 * Public API
 * ======================================================================== */

/* Load model from a directory containing safetensors + phoneme_vocab.json.
 * Returns NULL on failure. */
__attribute__((visibility("default")))
kittentts_ctx_t *kittentts_load(const char *model_dir);

/* Free all resources. */
__attribute__((visibility("default")))
void kittentts_free(kittentts_ctx_t *ctx);

/* Synthesize text to audio.
 * style_id: 0-7 for built-in voices (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo)
 * speed: speech rate multiplier (1.0 = normal)
 * Returns allocated float32 buffer at 24kHz. Caller must free.
 * Sets *out_n_samples to number of samples. Returns NULL on error. */
__attribute__((visibility("default")))
float *kittentts_synthesize(kittentts_ctx_t *ctx, const char *text,
                             int style_id, float speed, int *out_n_samples);

/* Convenience: synthesize and write directly to a WAV file.
 * path can be "-" for stdout. Returns 0 on success. */
__attribute__((visibility("default")))
int kittentts_synthesize_wav(kittentts_ctx_t *ctx, const char *text,
                              int style_id, float speed, const char *wav_path);

#endif /* KITTENTTS_H */
