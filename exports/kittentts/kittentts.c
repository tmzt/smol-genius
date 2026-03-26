/*
 * kittentts.c - KittenTTS orchestration layer
 *
 * High-level pipeline: load model, phonemize text, run inference, output audio.
 * All activation buffers are pre-allocated at load time for zero-allocation forward pass.
 */

#include "kittentts.h"
#include "phonemizer.h"
#include "plbert.h"
#include "text_encoder.h"
#include "prosody.h"
#include "acoustic_decoder.h"
#include "vocoder.h"
#include "smol_kernels.h"
#include "safetensors.h"
#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int smol_verbose;

/* ========================================================================
 * Weight Loading Helpers
 * ======================================================================== */

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "kittentts: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static float *try_load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) return NULL;
    return safetensors_get_f32(sf, t);
}

/* ========================================================================
 * PL-BERT Weight Loading
 * ======================================================================== */

static int load_plbert(ktts_plbert_t *m, multi_safetensors_t *ms) {
    m->tok_emb     = load_f32(ms, "plbert.embed.word.weight");
    m->pos_emb     = load_f32(ms, "plbert.embed.pos.weight");
    m->tok_type_emb = load_f32(ms, "plbert.embed.type.weight");
    m->emb_ln_w    = load_f32(ms, "plbert.embed.ln.weight");
    m->emb_ln_b    = load_f32(ms, "plbert.embed.ln.bias");

    m->hidden_map_w = load_f32(ms, "plbert.hidden_map.weight");
    m->hidden_map_b = load_f32(ms, "plbert.hidden_map.bias");

    m->attn_ln_w   = load_f32(ms, "plbert.layer.attn_ln.weight");
    m->attn_ln_b   = load_f32(ms, "plbert.layer.attn_ln.bias");
    m->attn_q_w    = load_f32(ms, "plbert.layer.attn.q.weight");
    m->attn_q_b    = load_f32(ms, "plbert.layer.attn.q.bias");
    m->attn_k_w    = load_f32(ms, "plbert.layer.attn.k.weight");
    m->attn_k_b    = load_f32(ms, "plbert.layer.attn.k.bias");
    m->attn_v_w    = load_f32(ms, "plbert.layer.attn.v.weight");
    m->attn_v_b    = load_f32(ms, "plbert.layer.attn.v.bias");
    m->attn_o_w    = load_f32(ms, "plbert.layer.attn.o.weight");
    m->attn_o_b    = load_f32(ms, "plbert.layer.attn.o.bias");

    m->ffn_ln_w    = load_f32(ms, "plbert.layer.ffn_ln.weight");
    m->ffn_ln_b    = load_f32(ms, "plbert.layer.ffn_ln.bias");
    m->ffn_up_w    = load_f32(ms, "plbert.layer.ffn.up.weight");
    m->ffn_up_b    = load_f32(ms, "plbert.layer.ffn.up.bias");
    m->ffn_down_w  = load_f32(ms, "plbert.layer.ffn.down.weight");
    m->ffn_down_b  = load_f32(ms, "plbert.layer.ffn.down.bias");

    m->out_proj_w  = load_f32(ms, "plbert.out_proj.weight");
    m->out_proj_b  = load_f32(ms, "plbert.out_proj.bias");

    /* Detect number of ALBERT iterations from config or default */
    m->n_iterations = 6; /* ALBERT-base default */

    if (!m->tok_emb || !m->pos_emb || !m->hidden_map_w || !m->attn_q_w) {
        return -1;
    }
    return 0;
}

/* ========================================================================
 * Text Encoder Weight Loading
 * ======================================================================== */

static int load_text_encoder(ktts_text_enc_t *m, multi_safetensors_t *ms) {
    m->phoneme_emb = load_f32(ms, "text_enc.embed.weight");

    char name[256];
    for (int i = 0; i < KTTS_TEXT_ENC_CNN_N; i++) {
        snprintf(name, sizeof(name), "text_enc.conv.%d.weight", i);
        m->conv_w[i] = load_f32(ms, name);
        snprintf(name, sizeof(name), "text_enc.conv.%d.bias", i);
        m->conv_b[i] = load_f32(ms, name);
        snprintf(name, sizeof(name), "text_enc.conv_ln.%d.weight", i);
        m->conv_ln_w[i] = load_f32(ms, name);
        snprintf(name, sizeof(name), "text_enc.conv_ln.%d.bias", i);
        m->conv_ln_b[i] = load_f32(ms, name);
    }

    m->lstm_W_ih_fwd = load_f32(ms, "text_enc.lstm.W_ih_fwd");
    m->lstm_W_hh_fwd = load_f32(ms, "text_enc.lstm.W_hh_fwd");
    m->lstm_b_ih_fwd = load_f32(ms, "text_enc.lstm.b_ih_fwd");
    m->lstm_b_hh_fwd = load_f32(ms, "text_enc.lstm.b_hh_fwd");
    m->lstm_W_ih_bwd = load_f32(ms, "text_enc.lstm.W_ih_bwd");
    m->lstm_W_hh_bwd = load_f32(ms, "text_enc.lstm.W_hh_bwd");
    m->lstm_b_ih_bwd = load_f32(ms, "text_enc.lstm.b_ih_bwd");
    m->lstm_b_hh_bwd = load_f32(ms, "text_enc.lstm.b_hh_bwd");

    if (!m->phoneme_emb || !m->conv_w[0] || !m->lstm_W_ih_fwd) return -1;
    return 0;
}

/* ========================================================================
 * Prosody Weight Loading
 * ======================================================================== */

static int load_prosody(ktts_prosody_t *m, multi_safetensors_t *ms) {
    char name[256];

    /* Duration LSTMs */
    for (int i = 0; i < KTTS_PROSODY_LSTM_N; i++) {
        snprintf(name, sizeof(name), "prosody.dur_lstm.%d.W_ih_fwd", i);
        m->dur_lstm[i].W_ih_fwd = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.dur_lstm.%d.W_hh_fwd", i);
        m->dur_lstm[i].W_hh_fwd = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.dur_lstm.%d.b_ih_fwd", i);
        m->dur_lstm[i].b_ih_fwd = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.dur_lstm.%d.b_hh_fwd", i);
        m->dur_lstm[i].b_hh_fwd = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.dur_lstm.%d.W_ih_bwd", i);
        m->dur_lstm[i].W_ih_bwd = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.dur_lstm.%d.W_hh_bwd", i);
        m->dur_lstm[i].W_hh_bwd = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.dur_lstm.%d.b_ih_bwd", i);
        m->dur_lstm[i].b_ih_bwd = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.dur_lstm.%d.b_hh_bwd", i);
        m->dur_lstm[i].b_hh_bwd = load_f32(ms, name);

        snprintf(name, sizeof(name), "prosody.dur_ada.%d.weight", i);
        m->dur_ada_fc_w[i] = try_load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.dur_ada.%d.bias", i);
        m->dur_ada_fc_b[i] = try_load_f32(ms, name);
    }

    m->dur_proj_w = load_f32(ms, "prosody.dur_proj.weight");
    m->dur_proj_b = load_f32(ms, "prosody.dur_proj.bias");

    /* F0 predictor blocks — AdaIN uses norm{1,2}.fc for style projection */
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "prosody.f0.%d.conv1.weight", i);
        m->f0_blocks[i].conv1_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.f0.%d.conv1.bias", i);
        m->f0_blocks[i].conv1_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.f0.%d.conv2.weight", i);
        m->f0_blocks[i].conv2_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.f0.%d.conv2.bias", i);
        m->f0_blocks[i].conv2_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.f0.%d.norm1.fc.weight", i);
        m->f0_blocks[i].adain_fc_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.f0.%d.norm1.fc.bias", i);
        m->f0_blocks[i].adain_fc_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.f0.%d.conv1x1.weight", i);
        m->f0_blocks[i].downsample_w = try_load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.f0.%d.pool.weight", i);
        m->f0_blocks[i].downsample_b = try_load_f32(ms, name);
    }
    m->f0_proj_w = load_f32(ms, "prosody.f0.proj.weight");
    m->f0_proj_b = try_load_f32(ms, "prosody.f0.proj.bias");

    /* N predictor blocks (same structure) */
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "prosody.n.%d.conv1.weight", i);
        m->n_blocks[i].conv1_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.n.%d.conv1.bias", i);
        m->n_blocks[i].conv1_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.n.%d.conv2.weight", i);
        m->n_blocks[i].conv2_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.n.%d.conv2.bias", i);
        m->n_blocks[i].conv2_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.n.%d.norm1.fc.weight", i);
        m->n_blocks[i].adain_fc_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.n.%d.norm1.fc.bias", i);
        m->n_blocks[i].adain_fc_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.n.%d.conv1x1.weight", i);
        m->n_blocks[i].downsample_w = try_load_f32(ms, name);
        snprintf(name, sizeof(name), "prosody.n.%d.pool.weight", i);
        m->n_blocks[i].downsample_b = try_load_f32(ms, name);
    }
    m->n_proj_w = load_f32(ms, "prosody.n.proj.weight");
    m->n_proj_b = try_load_f32(ms, "prosody.n.proj.bias");

    if (!m->dur_lstm[0].W_ih_fwd || !m->dur_proj_w) return -1;
    return 0;
}

/* ========================================================================
 * Acoustic Decoder Weight Loading
 * ======================================================================== */

static int load_acoustic_decoder(ktts_acoustic_dec_t *m, multi_safetensors_t *ms) {
    m->adapter_w = load_f32(ms, "decoder.adapter.weight");
    m->adapter_b = load_f32(ms, "decoder.adapter.bias");

    m->enc_conv1_w = load_f32(ms, "decoder.encode.conv1.weight");
    m->enc_conv1_b = load_f32(ms, "decoder.encode.conv1.bias");
    m->enc_conv2_w = load_f32(ms, "decoder.encode.conv2.weight");
    m->enc_conv2_b = load_f32(ms, "decoder.encode.conv2.bias");
    m->enc_adain_fc_w = load_f32(ms, "decoder.encode.norm1.fc.weight");
    m->enc_adain_fc_b = load_f32(ms, "decoder.encode.norm1.fc.bias");
    m->enc_skip_w = try_load_f32(ms, "decoder.encode.conv1x1.weight");
    m->enc_skip_b = NULL; /* conv1x1 has no bias */

    char name[256];
    for (int i = 0; i < KTTS_DECODER_BLOCKS; i++) {
        snprintf(name, sizeof(name), "decoder.decode.%d.conv1.weight", i);
        m->dec_blocks[i].conv1_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decode.%d.conv1.bias", i);
        m->dec_blocks[i].conv1_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decode.%d.conv2.weight", i);
        m->dec_blocks[i].conv2_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decode.%d.conv2.bias", i);
        m->dec_blocks[i].conv2_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decode.%d.norm1.fc.weight", i);
        m->dec_blocks[i].adain_fc_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decode.%d.norm1.fc.bias", i);
        m->dec_blocks[i].adain_fc_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decode.%d.conv1x1.weight", i);
        m->dec_blocks[i].skip_w = try_load_f32(ms, name);
        m->dec_blocks[i].skip_b = NULL;
    }

    if (!m->adapter_w || !m->enc_conv1_w) return -1;
    return 0;
}

/* ========================================================================
 * Vocoder Weight Loading
 * ======================================================================== */

static int load_vocoder(ktts_vocoder_t *m, multi_safetensors_t *ms) {
    m->up0_w = load_f32(ms, "vocoder.ups.0.weight");
    m->up0_b = load_f32(ms, "vocoder.ups.0.bias");
    m->up1_w = load_f32(ms, "vocoder.ups.1.weight");
    m->up1_b = load_f32(ms, "vocoder.ups.1.bias");

    char name[256];
    for (int i = 0; i < 4; i++) {
        /* Vocoder resblocks use convs1.0/convs2.0 naming (3 dilated convs each) */
        snprintf(name, sizeof(name), "vocoder.resblocks.%d.convs1.0.weight", i);
        m->resblocks[i].conv1_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "vocoder.resblocks.%d.convs1.0.bias", i);
        m->resblocks[i].conv1_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "vocoder.resblocks.%d.convs2.0.weight", i);
        m->resblocks[i].conv2_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "vocoder.resblocks.%d.convs2.0.bias", i);
        m->resblocks[i].conv2_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "vocoder.resblocks.%d.convs1.1.weight", i);
        m->resblocks[i].conv3_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "vocoder.resblocks.%d.convs1.1.bias", i);
        m->resblocks[i].conv3_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "vocoder.resblocks.%d.adain1.0.fc.weight", i);
        m->resblocks[i].adain_fc_w = load_f32(ms, name);
        snprintf(name, sizeof(name), "vocoder.resblocks.%d.adain1.0.fc.bias", i);
        m->resblocks[i].adain_fc_b = load_f32(ms, name);
        snprintf(name, sizeof(name), "vocoder.resblocks.%d.alpha1.0", i);
        m->resblocks[i].snake_alpha = load_f32(ms, name);
    }

    m->conv_post_w = load_f32(ms, "vocoder.conv_post.weight");
    m->conv_post_b = load_f32(ms, "vocoder.conv_post.bias");

    m->source_linear_b = try_load_f32(ms, "vocoder.source.bias");

    if (!m->up0_w || !m->conv_post_w) return -1;
    return 0;
}

/* ========================================================================
 * Style Embedding Loading
 * ======================================================================== */

static int load_styles(kittentts_ctx_t *ctx, multi_safetensors_t *ms) {
    ctx->styles = (float *)calloc(KTTS_NUM_STYLES * KTTS_STYLE_DIM, sizeof(float));
    if (!ctx->styles) return -1;

    char name[256];
    for (int i = 0; i < KTTS_NUM_STYLES; i++) {
        snprintf(name, sizeof(name), "styles.%d", i);
        float *s = try_load_f32(ms, name);
        if (s) {
            memcpy(ctx->styles + i * KTTS_STYLE_DIM, s, KTTS_STYLE_DIM * sizeof(float));
            free(s);
        }
    }
    return 0;
}

/* ========================================================================
 * kittentts_load
 * ======================================================================== */

kittentts_ctx_t *kittentts_load(const char *model_dir) {
    kittentts_ctx_t *ctx = (kittentts_ctx_t *)calloc(1, sizeof(kittentts_ctx_t));
    if (!ctx) return NULL;

    fprintf(stderr, "kittentts: loading model from %s\n", model_dir);

    /* Initialize phonemizer */
    if (ktts_phonemizer_init() != 0) {
        fprintf(stderr, "kittentts: warning: espeak-ng init failed (phonemization unavailable)\n");
    }

    /* Load phoneme vocabulary */
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/phoneme_vocab.json", model_dir);
    if (ktts_load_phoneme_vocab(vocab_path, ctx->phoneme_map) != 0) {
        fprintf(stderr, "kittentts: warning: phoneme vocab not loaded\n");
    }

    /* Open safetensors */
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "kittentts: failed to open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    /* Load all sub-models */
    if (load_plbert(&ctx->plbert, ms) != 0) {
        fprintf(stderr, "kittentts: failed to load PL-BERT weights\n");
        goto fail;
    }
    fprintf(stderr, "kittentts: PL-BERT loaded (ALBERT, %d iterations)\n", ctx->plbert.n_iterations);

    if (load_text_encoder(&ctx->text_enc, ms) != 0) {
        fprintf(stderr, "kittentts: failed to load text encoder weights\n");
        goto fail;
    }
    fprintf(stderr, "kittentts: text encoder loaded\n");

    if (load_prosody(&ctx->prosody, ms) != 0) {
        fprintf(stderr, "kittentts: failed to load prosody weights\n");
        goto fail;
    }
    fprintf(stderr, "kittentts: prosody predictor loaded\n");

    if (load_acoustic_decoder(&ctx->decoder, ms) != 0) {
        fprintf(stderr, "kittentts: failed to load acoustic decoder weights\n");
        goto fail;
    }
    fprintf(stderr, "kittentts: acoustic decoder loaded\n");

    if (load_vocoder(&ctx->vocoder, ms) != 0) {
        fprintf(stderr, "kittentts: failed to load vocoder weights\n");
        goto fail;
    }
    fprintf(stderr, "kittentts: vocoder loaded\n");

    if (load_styles(ctx, ms) != 0) {
        fprintf(stderr, "kittentts: warning: style embeddings not loaded\n");
    }

    /* Pre-allocate activation buffers */
    ctx->max_phonemes = KTTS_MAX_PHONEMES;
    ctx->max_expanded_len = KTTS_MAX_PHONEMES * 50; /* max 50 frames per phoneme */
    int mp = ctx->max_phonemes;
    int me = ctx->max_expanded_len;

    ctx->bert_out = (float *)malloc((size_t)mp * KTTS_BERT_DIM * sizeof(float));
    ctx->text_enc_out = (float *)malloc((size_t)mp * KTTS_TEXT_ENC_DIM * sizeof(float));
    ctx->style_proj = (float *)malloc(KTTS_STYLE_ADAIN_DIM * 2 * sizeof(float));
    ctx->expanded_seq = (float *)malloc((size_t)me * KTTS_TEXT_ENC_DIM * sizeof(float));
    ctx->acoustic_out = (float *)malloc((size_t)me * KTTS_DECODER_HIDDEN * sizeof(float));

    /* Large scratch buffer for intermediate computations.
     * Sized for the largest consumer (PL-BERT + vocoder). */
    ctx->scratch_size = (size_t)mp * KTTS_BERT_HIDDEN * 8; /* generous */
    if ((size_t)me * KTTS_DECODER_HIDDEN * 8 > ctx->scratch_size) {
        ctx->scratch_size = (size_t)me * KTTS_DECODER_HIDDEN * 8;
    }
    ctx->scratch_a = (float *)malloc(ctx->scratch_size * sizeof(float));
    ctx->scratch_b = (float *)malloc(ctx->scratch_size * sizeof(float));

    if (!ctx->bert_out || !ctx->text_enc_out || !ctx->expanded_seq ||
        !ctx->acoustic_out || !ctx->scratch_a || !ctx->scratch_b) {
        fprintf(stderr, "kittentts: buffer allocation failed\n");
        goto fail;
    }

    fprintf(stderr, "kittentts: model loaded successfully\n");
    return ctx;

fail:
    kittentts_free(ctx);
    return NULL;
}

/* ========================================================================
 * kittentts_free
 * ======================================================================== */

void kittentts_free(kittentts_ctx_t *ctx) {
    if (!ctx) return;

    /* Activation buffers */
    free(ctx->bert_out);
    free(ctx->text_enc_out);
    free(ctx->style_proj);
    free(ctx->expanded_seq);
    free(ctx->acoustic_out);
    free(ctx->scratch_a);
    free(ctx->scratch_b);
    free(ctx->styles);

    /* f32 weight copies freed individually would be extensive.
     * Since all are allocated via load_f32 (which calls safetensors_get_f32
     * and returns malloc'd copies), they need to be freed.
     * For now, they live in the safetensors mmap and are freed by close.
     * TODO: track all f32 allocations for proper cleanup. */

    if (ctx->safetensors)
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);

    ktts_phonemizer_cleanup();
    free(ctx);
}

/* ========================================================================
 * kittentts_synthesize
 * ======================================================================== */

float *kittentts_synthesize(kittentts_ctx_t *ctx, const char *text,
                             int style_id, float speed, int *out_n_samples) {
    *out_n_samples = 0;

    /* --- Step 1: Phonemize --- */
    int token_ids[KTTS_MAX_PHONEMES];
    int n_tokens = ktts_phonemize(text, ctx->phoneme_map,
                                   token_ids, KTTS_MAX_PHONEMES);
    if (n_tokens <= 0) {
        fprintf(stderr, "kittentts: phonemization failed\n");
        return NULL;
    }

    if (smol_verbose) {
        fprintf(stderr, "kittentts: %d phoneme tokens\n", n_tokens);
    }

    /* --- Step 2: PL-BERT --- */
    ktts_plbert_forward(&ctx->plbert, token_ids, n_tokens,
                         ctx->bert_out, ctx->scratch_a);

    /* --- Step 3: Text Encoder --- */
    ktts_text_encoder_forward(&ctx->text_enc, token_ids,
                               ctx->bert_out, n_tokens,
                               ctx->text_enc_out, ctx->scratch_a);

    /* --- Step 4: Get style embedding --- */
    if (style_id < 0 || style_id >= KTTS_NUM_STYLES) style_id = 0;
    const float *style = ctx->styles + style_id * KTTS_STYLE_DIM;

    /* --- Step 5: Prosody prediction (duration + F0 + N) --- */
    int durations[KTTS_MAX_PHONEMES];
    float *f0 = ctx->scratch_b;
    float *noise_energy = f0 + ctx->max_expanded_len;

    int expanded_len = ktts_prosody_forward(&ctx->prosody,
                                             ctx->text_enc_out, style,
                                             n_tokens, speed,
                                             durations, f0, noise_energy,
                                             ctx->expanded_seq, ctx->scratch_a);

    if (smol_verbose) {
        fprintf(stderr, "kittentts: expanded to %d frames\n", expanded_len);
    }

    /* --- Step 6: Acoustic decoder --- */
    ktts_acoustic_decoder_forward(&ctx->decoder,
                                   ctx->expanded_seq, f0, noise_energy,
                                   style, expanded_len,
                                   ctx->acoustic_out, ctx->scratch_a);

    /* --- Step 7: Vocoder --- */
    /* Estimate max output samples: expanded_len * 300 (total upsample factor) */
    int max_samples = expanded_len * 300 + 10000;
    float *audio = (float *)malloc(max_samples * sizeof(float));
    if (!audio) return NULL;

    int n_samples = ktts_vocoder_forward(&ctx->vocoder,
                                          ctx->acoustic_out, f0,
                                          style, expanded_len,
                                          audio, ctx->scratch_a);

    if (n_samples <= 0) {
        free(audio);
        return NULL;
    }

    *out_n_samples = n_samples;
    return audio;
}

/* ========================================================================
 * kittentts_synthesize_wav
 * ======================================================================== */

int kittentts_synthesize_wav(kittentts_ctx_t *ctx, const char *text,
                              int style_id, float speed, const char *wav_path) {
    int n_samples;
    float *audio = kittentts_synthesize(ctx, text, style_id, speed, &n_samples);
    if (!audio) return -1;

    int ret = smol_write_wav(wav_path, audio, n_samples, KTTS_SAMPLE_RATE);
    free(audio);
    return ret;
}
