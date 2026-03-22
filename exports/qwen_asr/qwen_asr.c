/*
 * qwen_asr.c - Qwen3-ASR orchestration layer
 *
 * Pipeline: Load weights -> WAV -> Mel -> Encoder -> Build prompt ->
 *           Prefill decoder -> Autoregressive decode -> Tokenizer -> Text
 *
 * Components:
 *   Encoder:   exports/qwen_asr/encoder.{h,c}
 *   Decoder:   common/decoder/qkn_decoder.{h,c}
 *   Audio:     common/audio/audio.{h,c}
 *   Tokenizer: common/utils/tokenizer.{h,c}
 */

#include "qwen_asr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>

/* ========================================================================
 * Globals
 * ======================================================================== */

int qwen_verbose = 0;
int qwen_monitor = 0;

/* ========================================================================
 * Token Callback
 * ======================================================================== */

void qwen_set_token_callback(qwen_ctx_t *ctx, qwen_token_cb cb, void *userdata) {
    ctx->token_cb = cb;
    ctx->token_cb_userdata = userdata;
}

/* ========================================================================
 * Language Support
 * ======================================================================== */

static const char *QWEN_SUPPORTED_LANGUAGES[] = {
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch",
    "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"
};

static const char *QWEN_SUPPORTED_LANGUAGES_CSV =
    "Chinese,English,Cantonese,Arabic,German,French,Spanish,Portuguese,"
    "Indonesian,Italian,Korean,Russian,Thai,Vietnamese,Japanese,Turkish,"
    "Hindi,Malay,Dutch,Swedish,Danish,Finnish,Polish,Czech,Filipino,"
    "Persian,Greek,Romanian,Hungarian,Macedonian";

const char *qwen_supported_languages_csv(void) {
    return QWEN_SUPPORTED_LANGUAGES_CSV;
}

static void reset_prompt_cache(qwen_ctx_t *ctx) {
    free(ctx->prompt_tokens);
    ctx->prompt_tokens = NULL;
    ctx->n_prompt_tokens = 0;

    free(ctx->force_prompt_tokens);
    ctx->force_prompt_tokens = NULL;
    ctx->n_force_prompt_tokens = 0;

    ctx->prompt_tokens_ready = 0;
}

int qwen_set_prompt(qwen_ctx_t *ctx, const char *prompt) {
    if (!ctx) return -1;

    char *dup = NULL;
    if (prompt && prompt[0] != '\0') {
        dup = strdup(prompt);
        if (!dup) return -1;
    }
    free(ctx->prompt);
    ctx->prompt = dup;
    reset_prompt_cache(ctx);
    return 0;
}

static int normalize_language_name(const char *language, char *out, size_t out_cap) {
    if (!language || !out || out_cap < 2) return -1;

    while (*language && isspace((unsigned char)*language)) language++;
    size_t len = strlen(language);
    while (len > 0 && isspace((unsigned char)language[len - 1])) len--;
    if (len == 0 || len + 1 > out_cap) return -1;

    out[0] = (char)toupper((unsigned char)language[0]);
    for (size_t i = 1; i < len; i++) {
        out[i] = (char)tolower((unsigned char)language[i]);
    }
    out[len] = '\0';
    return 0;
}

static int is_supported_language(const char *language) {
    int n = (int)(sizeof(QWEN_SUPPORTED_LANGUAGES) / sizeof(QWEN_SUPPORTED_LANGUAGES[0]));
    for (int i = 0; i < n; i++) {
        if (strcmp(language, QWEN_SUPPORTED_LANGUAGES[i]) == 0) return 1;
    }
    return 0;
}

int qwen_set_force_language(qwen_ctx_t *ctx, const char *language) {
    if (!ctx) return -1;

    if (!language || language[0] == '\0') {
        free(ctx->force_language);
        ctx->force_language = NULL;
        reset_prompt_cache(ctx);
        return 0;
    }

    char normalized[64];
    if (normalize_language_name(language, normalized, sizeof(normalized)) != 0) return -1;
    if (!is_supported_language(normalized)) return -1;

    char *dup = strdup(normalized);
    if (!dup) return -1;

    free(ctx->force_language);
    ctx->force_language = dup;
    reset_prompt_cache(ctx);
    return 0;
}

/* ========================================================================
 * Threading
 * ======================================================================== */

void qwen_set_threads(int n) {
    smol_set_threads(n);
}

int qwen_get_num_cpus(void) {
    return smol_get_num_cpus();
}

/* ========================================================================
 * Config Detection
 * ======================================================================== */

static int detect_config(qwen_ctx_t *ctx, multi_safetensors_t *ms) {
    qwen_asr_enc_config_t *enc = &ctx->enc_config;
    qkn_config_t *dec = &ctx->dec_config;

    /* Check for layer 18 (0-indexed) in encoder - if it exists, it's 1.7B */
    const safetensor_t *test = multi_safetensors_find(ms,
        "thinker.audio_tower.layers.18.self_attn.q_proj.weight", NULL);

    if (test) {
        /* 1.7B model */
        enc->enc_d_model = 1024;
        enc->enc_layers = 24;
        enc->enc_heads = 16;
        enc->enc_head_dim = 64;
        enc->enc_ffn_dim = 4096;
        enc->enc_output_dim = 2048;
        dec->dec_hidden = 2048;
        dec->dec_layers = 28;
        dec->dec_heads = 16;
        dec->dec_kv_heads = 8;
        dec->dec_head_dim = 128;
        dec->dec_intermediate = 6144;
        if (qwen_verbose >= 1) fprintf(stderr, "Detected: Qwen3-ASR-1.7B\n");
    } else {
        /* 0.6B model */
        enc->enc_d_model = 896;
        enc->enc_layers = 18;
        enc->enc_heads = 14;
        enc->enc_head_dim = 64;
        enc->enc_ffn_dim = 3584;
        enc->enc_output_dim = 1024;
        dec->dec_hidden = 1024;
        dec->dec_layers = 28;
        dec->dec_heads = 16;
        dec->dec_kv_heads = 8;
        dec->dec_head_dim = 128;
        dec->dec_intermediate = 3072;
        if (qwen_verbose >= 1) fprintf(stderr, "Detected: Qwen3-ASR-0.6B\n");
    }

    /* Common encoder parameters */
    enc->enc_n_window = 50;
    enc->enc_n_window_infer = 800;
    enc->enc_chunk_size = enc->enc_n_window * 2; /* 100 */
    enc->enc_conv_proj_dim = QWEN_CONV_HIDDEN * 16; /* 7680 */

    /* Common decoder parameters */
    dec->vocab_size = QWEN_VOCAB_SIZE;
    dec->dec_rms_norm_eps = 1e-6f;
    dec->dec_rope_theta = 1e6f;

    return 0;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

qwen_ctx_t *qwen_load(const char *model_dir) {
    qwen_ctx_t *ctx = (qwen_ctx_t *)calloc(1, sizeof(qwen_ctx_t));
    if (!ctx) return NULL;
    snprintf(ctx->model_dir, sizeof(ctx->model_dir), "%s", model_dir);

    /* Open safetensors (multi-shard) */
    if (qwen_verbose >= 1)
        fprintf(stderr, "Loading model from %s\n", model_dir);

    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "qwen_load: cannot open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    /* Detect model configuration */
    detect_config(ctx, ms);

    /* Load encoder weights */
    if (qwen_verbose >= 1) fprintf(stderr, "Loading encoder weights...\n");
    if (qwen_asr_encoder_load(&ctx->encoder, ms, &ctx->enc_config) != 0) {
        fprintf(stderr, "qwen_load: failed to load encoder\n");
        qwen_free(ctx);
        return NULL;
    }

    /* Load decoder weights */
    if (qwen_verbose >= 1) fprintf(stderr, "Loading decoder weights...\n");
    if (qkn_decoder_load(&ctx->dec_ctx.decoder, ms, &ctx->dec_config,
                          "thinker.model") != 0) {
        fprintf(stderr, "qwen_load: failed to load decoder\n");
        qwen_free(ctx);
        return NULL;
    }
    ctx->dec_ctx.config = ctx->dec_config;

    /* Load tokenizer */
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);
    ctx->tokenizer = smol_tokenizer_load(vocab_path);
    if (!ctx->tokenizer) {
        fprintf(stderr, "qwen_load: failed to load tokenizer from %s\n", vocab_path);
        qwen_free(ctx);
        return NULL;
    }

    /* Default transcription mode: full-audio offline decode (no splitting). */
    ctx->segment_sec = 0.0f;
    ctx->search_sec = 3.0f;

    /* Default streaming parameters */
    ctx->stream_chunk_sec = 0.5f;
    ctx->stream_rollback = 5;
    ctx->stream_unfixed_chunks = 2;
    ctx->stream_max_new_tokens = 32;
    ctx->past_text_conditioning = 0;
    ctx->skip_silence = 0;

    if (qwen_verbose >= 1) fprintf(stderr, "Model loaded.\n");
    return ctx;
}

/* ========================================================================
 * Free
 * ======================================================================== */

void qwen_free(qwen_ctx_t *ctx) {
    if (!ctx) return;

    #define FREE0(p) do { free(p); (p) = NULL; } while (0)

    /* Encoder conv stem */
    FREE0(ctx->encoder.conv1_weight); FREE0(ctx->encoder.conv1_bias);
    FREE0(ctx->encoder.conv2_weight); FREE0(ctx->encoder.conv2_bias);
    FREE0(ctx->encoder.conv3_weight); FREE0(ctx->encoder.conv3_bias);
    FREE0(ctx->encoder.conv_out_weight);

    /* Encoder layers (weights are pre-converted f32, all allocated) */
    for (int i = 0; i < ctx->enc_config.enc_layers; i++) {
        qwen_asr_enc_layer_t *l = &ctx->encoder.layers[i];
        FREE0(l->wq_weight); FREE0(l->wq_bias);
        FREE0(l->wk_weight); FREE0(l->wk_bias);
        FREE0(l->wv_weight); FREE0(l->wv_bias);
        FREE0(l->wo_weight); FREE0(l->wo_bias);
        FREE0(l->attn_norm_weight); FREE0(l->attn_norm_bias);
        FREE0(l->fc1_weight); FREE0(l->fc1_bias);
        FREE0(l->fc2_weight); FREE0(l->fc2_bias);
        FREE0(l->ffn_norm_weight); FREE0(l->ffn_norm_bias);
    }
    FREE0(ctx->encoder.ln_post_weight); FREE0(ctx->encoder.ln_post_bias);
    FREE0(ctx->encoder.proj1_weight); FREE0(ctx->encoder.proj1_bias);
    FREE0(ctx->encoder.proj2_weight); FREE0(ctx->encoder.proj2_bias);

    /* Decoder layers */
    for (int i = 0; i < ctx->dec_config.dec_layers; i++) {
        qkn_dec_layer_t *l = &ctx->dec_ctx.decoder.layers[i];
        FREE0(l->q_norm_weight); FREE0(l->k_norm_weight);
        FREE0(l->input_norm); FREE0(l->post_attn_norm);
        FREE0(l->gate_up_fused_bf16);
    }
    FREE0(ctx->dec_ctx.decoder.norm);

    #undef FREE0

    /* Decoder runtime state */
    free(ctx->dec_ctx.kv_cache_k);
    free(ctx->dec_ctx.kv_cache_v);
    free(ctx->dec_ctx.dec_x); free(ctx->dec_ctx.dec_x_norm);
    free(ctx->dec_ctx.dec_q); free(ctx->dec_ctx.dec_k); free(ctx->dec_ctx.dec_v);
    free(ctx->dec_ctx.dec_attn_out); free(ctx->dec_ctx.dec_proj_out);
    free(ctx->dec_ctx.dec_gate); free(ctx->dec_ctx.dec_up); free(ctx->dec_ctx.dec_ffn_out);
    free(ctx->dec_ctx.dec_rope_cos); free(ctx->dec_ctx.dec_rope_sin);
    free(ctx->dec_ctx.pref_x); free(ctx->dec_ctx.pref_x_norm);
    free(ctx->dec_ctx.pref_q); free(ctx->dec_ctx.pref_k); free(ctx->dec_ctx.pref_v);
    free(ctx->dec_ctx.pref_attn_out); free(ctx->dec_ctx.pref_proj_out);
    free(ctx->dec_ctx.pref_ffn_out);
    free(ctx->dec_ctx.pref_gate); free(ctx->dec_ctx.pref_gate_up);
    free(ctx->dec_ctx.rope_cache_cos); free(ctx->dec_ctx.rope_cache_sin);
    free(ctx->dec_ctx.rope_inv_freq);

    /* Wakeword detector */
    if (ctx->wakeword) smol_ww_free(ctx->wakeword);

    /* Tokenizer */
    if (ctx->tokenizer) smol_tokenizer_free(ctx->tokenizer);

    /* Prompt/language options */
    free(ctx->prompt);
    free(ctx->force_language);
    free(ctx->prompt_tokens);
    free(ctx->force_prompt_tokens);

    /* Close safetensors */
    if (ctx->safetensors) {
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);
    }

    free(ctx);
}

/* ========================================================================
 * Helpers
 * ======================================================================== */

/* Prompt token sequences */
static const int PROMPT_PREFIX_HEAD[] = { 151644, 8948, 198 };
static const int PROMPT_PREFIX_TAIL[] = { 151645, 198, 151644, 872, 198, 151669 };
static const int PROMPT_SUFFIX_BASE[] = { 151670, 151645, 198, 151644, 77091, 198 };
#define PREFIX_HEAD_LEN 3
#define PREFIX_TAIL_LEN 6
#define SUFFIX_BASE_LEN 6

/* Convert a single token embedding from bf16 to f32 */
static void tok_embed_bf16_to_f32(float *dst, const uint16_t *tok_emb_bf16,
                                  int token_id, int dim) {
    const uint16_t *src = tok_emb_bf16 + (size_t)token_id * dim;
    for (int i = 0; i < dim; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Prepare cached prompt-related tokens once per context. */
static int prepare_prompt_tokens(qwen_ctx_t *ctx) {
    if (ctx->prompt_tokens_ready) return 0;

    reset_prompt_cache(ctx);

    if (ctx->prompt && ctx->prompt[0] != '\0') {
        ctx->prompt_tokens = smol_tokenizer_encode(ctx->tokenizer, ctx->prompt,
                                                    &ctx->n_prompt_tokens);
        if (!ctx->prompt_tokens) {
            fprintf(stderr, "qwen: failed to encode --prompt text\n");
            return -1;
        }
    }

    if (ctx->force_language && ctx->force_language[0] != '\0') {
        char force_text[128];
        snprintf(force_text, sizeof(force_text), "language %s", ctx->force_language);

        int n_lang_txt = 0;
        int *lang_txt_tokens = smol_tokenizer_encode(ctx->tokenizer, force_text, &n_lang_txt);
        if (!lang_txt_tokens) {
            fprintf(stderr, "qwen: failed to encode --language text\n");
            return -1;
        }

        ctx->n_force_prompt_tokens = n_lang_txt + 1; /* + <asr_text> marker */
        ctx->force_prompt_tokens = (int *)malloc((size_t)ctx->n_force_prompt_tokens * sizeof(int));
        if (!ctx->force_prompt_tokens) {
            free(lang_txt_tokens);
            return -1;
        }
        if (n_lang_txt > 0) {
            memcpy(ctx->force_prompt_tokens, lang_txt_tokens, (size_t)n_lang_txt * sizeof(int));
        }
        ctx->force_prompt_tokens[n_lang_txt] = QWEN_TOKEN_ASR_TEXT;
        free(lang_txt_tokens);
    }

    ctx->prompt_tokens_ready = 1;
    return 0;
}

/* ========================================================================
 * Silence Compaction
 * ======================================================================== */

static int cmp_float_asc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static float *compact_silence(const float *samples, int n_samples, int *out_samples) {
    if (!samples || n_samples <= 0 || !out_samples) return NULL;

    const int win = 160;               /* 10 ms at 16kHz */
    const float base_thresh = 0.002f;
    const float max_thresh = 0.025f;
    const float smooth_alpha = 0.2f;
    const int min_voice_windows = 5;
    const int pad_voice_windows = 3;
    const int pass_windows = 60;

    int n_win = (n_samples + win - 1) / win;
    float *rms_vals = (float *)malloc((size_t)n_win * sizeof(float));
    float *sorted = (float *)malloc((size_t)n_win * sizeof(float));
    float *smooth_vals = (float *)malloc((size_t)n_win * sizeof(float));
    unsigned char *is_voice = (unsigned char *)malloc((size_t)n_win);
    if (!rms_vals || !sorted || !smooth_vals || !is_voice) {
        free(rms_vals); free(sorted); free(smooth_vals); free(is_voice);
        return NULL;
    }

    for (int w = 0; w < n_win; w++) {
        int start = w * win;
        int end = start + win;
        if (end > n_samples) end = n_samples;
        int len = end - start;
        float energy = 0.0f;
        for (int i = 0; i < len; i++) {
            float v = samples[start + i];
            energy += v * v;
        }
        rms_vals[w] = sqrtf(energy / (float)(len > 0 ? len : 1));
    }

    float smooth = rms_vals[0];
    for (int w = 0; w < n_win; w++) {
        smooth = (1.0f - smooth_alpha) * smooth + smooth_alpha * rms_vals[w];
        smooth_vals[w] = smooth;
    }

    memcpy(sorted, smooth_vals, (size_t)n_win * sizeof(float));
    qsort(sorted, (size_t)n_win, sizeof(float), cmp_float_asc);

    int p25 = (int)((n_win - 1) * 0.25f);
    float noise_floor = sorted[p25];
    float thresh = noise_floor * 1.8f;
    if (thresh < base_thresh) thresh = base_thresh;
    if (thresh > max_thresh) thresh = max_thresh;
    free(sorted);

    for (int w = 0; w < n_win; w++) {
        is_voice[w] = (smooth_vals[w] > thresh) ? 1 : 0;
    }
    free(smooth_vals);

    for (int i = 0; i < n_win; ) {
        if (!is_voice[i]) { i++; continue; }
        int j = i + 1;
        while (j < n_win && is_voice[j]) j++;
        if (j - i < min_voice_windows) {
            memset(is_voice + i, 0, (size_t)(j - i));
        }
        i = j;
    }

    unsigned char *padded = (unsigned char *)calloc((size_t)n_win, 1);
    if (!padded) { free(is_voice); free(rms_vals); return NULL; }
    for (int w = 0; w < n_win; w++) {
        if (!is_voice[w]) continue;
        int a = w - pad_voice_windows; if (a < 0) a = 0;
        int b = w + pad_voice_windows; if (b >= n_win) b = n_win - 1;
        for (int k = a; k <= b; k++) padded[k] = 1;
    }
    free(is_voice);

    float *out = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!out) { free(rms_vals); free(padded); return NULL; }

    int out_n = 0;
    int silence_count = 0;
    for (int w = 0; w < n_win; w++) {
        int start = w * win;
        int end = start + win;
        if (end > n_samples) end = n_samples;
        int len = end - start;

        if (padded[w]) {
            memcpy(out + out_n, samples + start, (size_t)len * sizeof(float));
            out_n += len;
            silence_count = 0;
        } else {
            silence_count++;
            if (silence_count <= pass_windows) {
                memcpy(out + out_n, samples + start, (size_t)len * sizeof(float));
                out_n += len;
            }
        }
    }
    free(padded);
    free(rms_vals);

    if (out_n == 0) {
        int keep = n_samples;
        int min_keep = QWEN_SAMPLE_RATE / 2;
        if (keep > min_keep) keep = min_keep;
        memcpy(out, samples, (size_t)keep * sizeof(float));
        out_n = keep;
    }

    *out_samples = out_n;
    return out;
}

/* ========================================================================
 * Single-Segment Transcription
 * ======================================================================== */

static char *transcribe_segment(qwen_ctx_t *ctx, const float *samples,
                                int n_samples,
                                const int *past_tokens, int n_past_tokens,
                                int *out_text_tokens) {
    int dim = ctx->dec_config.dec_hidden;
    double seg_t0 = get_time_ms();
    int n_text_tokens = 0;

    /* ---- Mel spectrogram ---- */
    double t0 = get_time_ms();
    int mel_frames = 0;
    float *mel = smol_mel_spectrogram(samples, n_samples, &mel_frames);
    if (!mel) return NULL;
    double mel_ms = get_time_ms() - t0;

    if (qwen_verbose >= 2)
        fprintf(stderr, "  Mel: %d frames (%.0f ms)\n", mel_frames, mel_ms);

    /* ---- Encoder ---- */
    t0 = get_time_ms();
    int enc_seq_len = 0;
    float *enc_output = qwen_asr_encoder_forward(&ctx->encoder, &ctx->enc_config,
                                                  mel, mel_frames, &enc_seq_len);
    free(mel);
    if (!enc_output) return NULL;
    double enc_ms = get_time_ms() - t0;

    if (qwen_verbose >= 2)
        fprintf(stderr, "  Encoder: %d tokens (%.0f ms)\n", enc_seq_len, enc_ms);

    if (prepare_prompt_tokens(ctx) != 0) {
        free(enc_output);
        return NULL;
    }

    /* ---- Build input embeddings ---- */
    int prefix_len = PREFIX_HEAD_LEN + ctx->n_prompt_tokens + PREFIX_TAIL_LEN;
    int suffix_len = SUFFIX_BASE_LEN + ctx->n_force_prompt_tokens;
    int n_past_prompt_tokens = (n_past_tokens > 0) ? (n_past_tokens + 1) : 0;
    int total_seq = prefix_len + enc_seq_len + suffix_len + n_past_prompt_tokens;
    float *input_embeds = (float *)malloc((size_t)total_seq * dim * sizeof(float));
    float *tmp_embed = (float *)malloc(dim * sizeof(float));
    if (!input_embeds || !tmp_embed) {
        free(enc_output); free(input_embeds); free(tmp_embed);
        return NULL;
    }

    const uint16_t *tok_emb = ctx->dec_ctx.decoder.tok_embeddings_bf16;

    /* Embed prefix head */
    int off = 0;
    for (int i = 0; i < PREFIX_HEAD_LEN; i++) {
        tok_embed_bf16_to_f32(input_embeds + off * dim, tok_emb,
                              PROMPT_PREFIX_HEAD[i], dim);
        off++;
    }

    /* Embed optional prompt text */
    for (int i = 0; i < ctx->n_prompt_tokens; i++) {
        tok_embed_bf16_to_f32(input_embeds + off * dim, tok_emb,
                              ctx->prompt_tokens[i], dim);
        off++;
    }

    /* Embed prefix tail */
    for (int i = 0; i < PREFIX_TAIL_LEN; i++) {
        tok_embed_bf16_to_f32(input_embeds + off * dim, tok_emb,
                              PROMPT_PREFIX_TAIL[i], dim);
        off++;
    }

    /* Insert encoder output as audio embeddings */
    for (int i = 0; i < enc_seq_len; i++) {
        memcpy(input_embeds + (prefix_len + i) * dim,
               enc_output + i * dim, dim * sizeof(float));
    }
    free(enc_output);

    /* Embed suffix base */
    int suffix_off = prefix_len + enc_seq_len;
    for (int i = 0; i < SUFFIX_BASE_LEN; i++)
        tok_embed_bf16_to_f32(input_embeds + (suffix_off + i) * dim, tok_emb,
                              PROMPT_SUFFIX_BASE[i], dim);

    /* Optional forced-language suffix */
    for (int i = 0; i < ctx->n_force_prompt_tokens; i++)
        tok_embed_bf16_to_f32(input_embeds + (suffix_off + SUFFIX_BASE_LEN + i) * dim,
                              tok_emb, ctx->force_prompt_tokens[i], dim);

    /* Optional past-text conditioning tokens */
    int past_off = suffix_off + suffix_len;
    for (int i = 0; i < n_past_tokens; i++)
        tok_embed_bf16_to_f32(input_embeds + (past_off + i) * dim, tok_emb,
                              past_tokens[i], dim);
    if (n_past_tokens > 0)
        tok_embed_bf16_to_f32(input_embeds + (past_off + n_past_tokens) * dim,
                              tok_emb, QWEN_TOKEN_ASR_TEXT, dim);

    /* ---- Decoder prefill ---- */
    t0 = get_time_ms();
    ctx->dec_ctx.kv_cache_len = 0;
    int prefill_len = total_seq - 1;
    qkn_decoder_prefill(&ctx->dec_ctx, input_embeds, prefill_len);

    /* First token from last prefill position */
    float *last_embed = input_embeds + (size_t)prefill_len * dim;
    int token = qkn_decoder_forward(&ctx->dec_ctx, last_embed);
    free(input_embeds);

    double prefill_ms = get_time_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Prefill: %d tokens (%.0f ms)\n", total_seq, prefill_ms);

    /* ---- Autoregressive decode ---- */
    t0 = get_time_ms();
    int max_tokens = 2048;
    int n_generated = 0;
    int past_asr_text = (ctx->n_force_prompt_tokens > 0 || n_past_tokens > 0) ? 1 : 0;

    size_t text_cap = 4096;
    size_t text_len = 0;
    char *text = (char *)malloc(text_cap);
    text[0] = '\0';

    while (n_generated < max_tokens) {
        n_generated++;

        if (token == QWEN_TOKEN_ENDOFTEXT || token == QWEN_TOKEN_IM_END) break;

        if (token == QWEN_TOKEN_ASR_TEXT) {
            past_asr_text = 1;
        } else if (past_asr_text) {
            const char *piece = smol_tokenizer_decode(ctx->tokenizer, token);
            size_t piece_len = strlen(piece);
            if (text_len + piece_len + 1 > text_cap) {
                while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                text = (char *)realloc(text, text_cap);
            }
            memcpy(text + text_len, piece, piece_len);
            text_len += piece_len;
            text[text_len] = '\0';
            n_text_tokens++;

            if (ctx->token_cb)
                ctx->token_cb(piece, ctx->token_cb_userdata);
        }

        tok_embed_bf16_to_f32(tmp_embed, tok_emb, token, dim);
        token = qkn_decoder_forward(&ctx->dec_ctx, tmp_embed);
    }

    double decode_ms = get_time_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Decode: %d tokens (%.0f ms, %.1f ms/token)\n",
                n_generated, decode_ms,
                n_generated > 0 ? decode_ms / n_generated : 0);

    free(tmp_embed);

    /* Trim whitespace */
    size_t rlen = strlen(text);
    while (rlen > 0 && isspace((unsigned char)text[rlen - 1])) text[--rlen] = '\0';
    char *start = text;
    while (*start && isspace((unsigned char)*start)) start++;
    if (start != text) memmove(text, start, strlen(start) + 1);

    ctx->perf_total_ms += get_time_ms() - seg_t0;
    ctx->perf_text_tokens += n_text_tokens;
    ctx->perf_encode_ms += mel_ms + enc_ms;
    ctx->perf_decode_ms += prefill_ms + decode_ms;
    if (out_text_tokens) *out_text_tokens = n_text_tokens;

    return text;
}

/* ========================================================================
 * Segment Split Helpers
 * ======================================================================== */

#define ENERGY_WINDOW_MS 100

static int find_split_point(const float *samples, int n_samples,
                            int target_sample, float search_sec) {
    int search_half = (int)(search_sec * QWEN_SAMPLE_RATE);
    int lo = target_sample - search_half;
    int hi = target_sample + search_half;
    if (lo < 0) lo = 0;
    if (hi > n_samples) hi = n_samples;

    int win_samples = (ENERGY_WINDOW_MS * QWEN_SAMPLE_RATE) / 1000;
    float best_energy = 1e30f;
    int best_center = target_sample;

    for (int pos = lo; pos + win_samples <= hi; pos += win_samples / 2) {
        float energy = 0;
        int end = pos + win_samples;
        if (end > n_samples) end = n_samples;
        for (int j = pos; j < end; j++) {
            energy += samples[j] * samples[j];
        }
        energy /= (end - pos);
        if (energy < best_energy) {
            best_energy = energy;
            best_center = pos + (end - pos) / 2;
        }
    }
    return best_center;
}

static int should_retry_unconditioned_segment(const char *full_result,
                                              const char *seg_text,
                                              int core_samples,
                                              int n_text_tokens) {
    if (!seg_text || seg_text[0] == '\0') return 1;

    float core_sec = (float)core_samples / (float)QWEN_SAMPLE_RATE;
    if (core_sec >= 8.0f) {
        int min_tokens = (int)(core_sec * 1.75f);
        if (min_tokens < 12) min_tokens = 12;
        if (n_text_tokens < min_tokens) return 1;
    }

    if (full_result && full_result[0] != '\0') {
        size_t seg_len = strlen(seg_text);
        if (seg_len >= 48 && strstr(full_result, seg_text) != NULL) return 1;
    }

    return 0;
}

static int should_insert_boundary_space(int prev_ch, int next_ch) {
    if (prev_ch <= 0 || next_ch <= 0) return 0;
    if (isspace((unsigned char)prev_ch)) return 0;
    if (isspace((unsigned char)next_ch)) return 0;
    if (ispunct((unsigned char)next_ch)) return 0;
    return 1;
}

typedef struct {
    qwen_token_cb downstream_cb;
    void *downstream_userdata;
    int maybe_prepend_space;
    int saw_first_piece;
} segment_emit_state_t;

static void segment_emit_cb(const char *piece, void *userdata) {
    segment_emit_state_t *st = (segment_emit_state_t *)userdata;
    if (!st || !st->downstream_cb || !piece) return;

    if (!st->saw_first_piece) {
        st->saw_first_piece = 1;
        if (st->maybe_prepend_space) {
            unsigned char c0 = (unsigned char)piece[0];
            if (c0 != '\0' && !isspace(c0) && !ispunct(c0)) {
                st->downstream_cb(" ", st->downstream_userdata);
            }
        }
    }
    st->downstream_cb(piece, st->downstream_userdata);
}

/* ========================================================================
 * Main Transcription Entry Point
 * ======================================================================== */

char *qwen_transcribe_audio(qwen_ctx_t *ctx, const float *samples, int n_samples) {
    ctx->perf_total_ms = 0;
    ctx->perf_text_tokens = 0;
    ctx->perf_audio_ms = 1000.0 * (double)n_samples / (double)QWEN_SAMPLE_RATE;
    ctx->perf_encode_ms = 0;
    ctx->perf_decode_ms = 0;

    const float *audio_samples = samples;
    int audio_n_samples = n_samples;
    float *compacted_samples = NULL;
    if (ctx->skip_silence) {
        compacted_samples = compact_silence(samples, n_samples, &audio_n_samples);
        if (compacted_samples) audio_samples = compacted_samples;
        if (qwen_verbose >= 1) {
            float used_pct = 100.0f * (float)audio_n_samples /
                             (float)(n_samples > 0 ? n_samples : 1);
            fprintf(stderr, "Silence skip: used %.1f%%, skipped %.1f%% (%d -> %d samples)\n",
                    used_pct, 100.0f - used_pct, n_samples, audio_n_samples);
        }
    }

    if (qwen_verbose >= 2)
        fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                audio_n_samples, (float)audio_n_samples / QWEN_SAMPLE_RATE);

    if (prepare_prompt_tokens(ctx) != 0) {
        free(compacted_samples);
        return NULL;
    }

    /* Determine segment boundaries */
    float search = ctx->search_sec;
    if (search > ctx->segment_sec / 2.0f) search = ctx->segment_sec / 2.0f;
    int target_samples = (int)(ctx->segment_sec * QWEN_SAMPLE_RATE);
    int margin_samples = (int)(search * QWEN_SAMPLE_RATE);

    /* No splitting if segment_sec is 0 or audio fits in one segment */
    if (ctx->segment_sec <= 0 || audio_n_samples <= target_samples + margin_samples) {
        char *text = transcribe_segment(ctx, audio_samples, audio_n_samples,
                                        NULL, 0, NULL);
        free(compacted_samples);
        return text;
    }

    /* Build split points */
    int splits[128];
    int n_splits = 0;
    splits[n_splits++] = 0;

    int pos = 0;
    while (pos + target_samples + margin_samples < audio_n_samples) {
        int split = find_split_point(audio_samples, audio_n_samples,
                                     pos + target_samples, search);
        splits[n_splits++] = split;
        pos = split;
        if (n_splits >= 127) break;
    }
    splits[n_splits] = audio_n_samples;

    if (qwen_verbose >= 2)
        fprintf(stderr, "Splitting into %d segments\n", n_splits);

    /* Transcribe each segment and concatenate */
    size_t result_cap = 4096;
    size_t result_len = 0;
    char *result = (char *)malloc(result_cap);
    result[0] = '\0';
    int min_samples_seg = QWEN_SAMPLE_RATE / 2;
    int do_boundary_cleanup = (ctx->past_text_conditioning != 0);
    int use_past_conditioning = ctx->past_text_conditioning;
    int conditioning_collapses = 0;
    qwen_token_cb saved_cb = ctx->token_cb;
    void *saved_cb_userdata = ctx->token_cb_userdata;

    for (int s = 0; s < n_splits; s++) {
        int core_start = splits[s];
        int core_end = splits[s + 1];
        int seg_start = core_start;
        int seg_end = core_end;
        int seg_samples = seg_end - seg_start;

        if (qwen_verbose >= 2)
            fprintf(stderr, "Segment %d/%d: %.1f-%.1fs (%d samples)\n",
                    s + 1, n_splits,
                    (float)core_start / QWEN_SAMPLE_RATE,
                    (float)core_end / QWEN_SAMPLE_RATE,
                    seg_samples);

        float *seg_buf = NULL;
        const float *seg_ptr = audio_samples + seg_start;
        if (seg_samples < min_samples_seg) {
            seg_buf = (float *)calloc(min_samples_seg, sizeof(float));
            memcpy(seg_buf, seg_ptr, seg_samples * sizeof(float));
            seg_ptr = seg_buf;
            seg_samples = min_samples_seg;
        }

        int *past_tokens_seg = NULL;
        int n_past_tokens_seg = 0;
        if (use_past_conditioning && result_len > 0) {
            past_tokens_seg = smol_tokenizer_encode(ctx->tokenizer, result,
                                                     &n_past_tokens_seg);
            if (!past_tokens_seg) n_past_tokens_seg = 0;
        }

        segment_emit_state_t emit_state = {0};
        if (do_boundary_cleanup) {
            ctx->token_cb = NULL;
            ctx->token_cb_userdata = NULL;
        } else if (saved_cb) {
            emit_state.downstream_cb = saved_cb;
            emit_state.downstream_userdata = saved_cb_userdata;
            emit_state.maybe_prepend_space =
                (result_len > 0 && !isspace((unsigned char)result[result_len - 1]));
            emit_state.saw_first_piece = 0;
            ctx->token_cb = segment_emit_cb;
            ctx->token_cb_userdata = &emit_state;
        }

        int seg_text_tokens = 0;
        char *seg_text = transcribe_segment(ctx, seg_ptr, seg_samples,
                                            past_tokens_seg, n_past_tokens_seg,
                                            &seg_text_tokens);
        if (do_boundary_cleanup &&
            use_past_conditioning && n_past_tokens_seg > 0 &&
            should_retry_unconditioned_segment(result, seg_text,
                                               core_end - core_start,
                                               seg_text_tokens)) {
            conditioning_collapses++;
            if (qwen_verbose >= 2)
                fprintf(stderr, "Segment mode: retrying segment %d/%d without conditioning\n",
                        s + 1, n_splits);
            free(seg_text);
            seg_text = transcribe_segment(ctx, seg_ptr, seg_samples, NULL, 0,
                                          &seg_text_tokens);
            if (conditioning_collapses >= 2) {
                use_past_conditioning = 0;
                if (qwen_verbose >= 2)
                    fprintf(stderr, "Segment mode: disabling conditioning after %d collapses\n",
                            conditioning_collapses);
            }
        }
        ctx->token_cb = saved_cb;
        ctx->token_cb_userdata = saved_cb_userdata;

        free(past_tokens_seg);
        free(seg_buf);
        if (!seg_text) continue;
        if (seg_text[0] == '\0') { free(seg_text); continue; }

        int cut_pos = 0;
        if (do_boundary_cleanup) {
            while (seg_text[cut_pos] != '\0' && isspace((unsigned char)seg_text[cut_pos])) cut_pos++;
        }
        if (seg_text[cut_pos] == '\0') { free(seg_text); continue; }

        size_t add_len = strlen(seg_text + cut_pos);
        int need_space = should_insert_boundary_space(
            result_len > 0 ? (int)(unsigned char)result[result_len - 1] : 0,
            (int)(unsigned char)seg_text[cut_pos]);
        size_t need = result_len + add_len + (size_t)(need_space ? 2 : 1);
        if (need > result_cap) {
            while (need > result_cap) result_cap *= 2;
            result = (char *)realloc(result, result_cap);
        }
        if (need_space) {
            result[result_len++] = ' ';
            if (do_boundary_cleanup && saved_cb) saved_cb(" ", saved_cb_userdata);
        }
        memcpy(result + result_len, seg_text + cut_pos, add_len);
        result_len += add_len;
        result[result_len] = '\0';
        if (do_boundary_cleanup && saved_cb) saved_cb(seg_text + cut_pos, saved_cb_userdata);
        free(seg_text);
    }

    ctx->token_cb = saved_cb;
    ctx->token_cb_userdata = saved_cb_userdata;
    free(compacted_samples);
    return result;
}

/* ========================================================================
 * Streaming Helpers
 * ======================================================================== */

static int stream_encode_span(qwen_ctx_t *ctx, const float *samples, int n_samples,
                              float **out_enc_output, int *out_seq_len) {
    *out_enc_output = NULL;
    *out_seq_len = 0;
    if (n_samples <= 0) return 0;

    int mel_frames = 0;
    float *mel = smol_mel_spectrogram(samples, n_samples, &mel_frames);
    if (!mel) return -1;

    int seq_len = 0;
    float *enc_output = qwen_asr_encoder_forward(&ctx->encoder, &ctx->enc_config,
                                                  mel, mel_frames, &seq_len);
    free(mel);
    if (!enc_output) return -1;

    *out_enc_output = enc_output;
    *out_seq_len = seq_len;
    return 0;
}

static int stream_tail_repeat_blocks(const int *tokens, int n_tokens, int max_period,
                                     int *out_period) {
    if (out_period) *out_period = 0;
    if (!tokens || n_tokens < 2) return 1;

    int best_reps = 1;
    int best_period = 0;
    int period_cap = n_tokens / 2;
    if (max_period > 0 && period_cap > max_period) period_cap = max_period;

    for (int p = 1; p <= period_cap; p++) {
        int reps = 1;
        while ((reps + 1) * p <= n_tokens) {
            const int *a = tokens + n_tokens - (reps + 1) * p;
            const int *b = tokens + n_tokens - reps * p;
            if (memcmp(a, b, (size_t)p * sizeof(int)) != 0) break;
            reps++;
        }
        if (reps > best_reps) {
            best_reps = reps;
            best_period = p;
        }
    }

    if (out_period) *out_period = best_period;
    return best_reps;
}

typedef struct {
    int64_t start_sample;
    int n_samples;
    int seq_len;
    float *enc_output;
} stream_enc_window_t;

static void stream_clear_enc_cache(stream_enc_window_t *enc_cache,
                                   int *n_enc_cache,
                                   int *enc_cache_start,
                                   int *enc_cached_seq_total,
                                   int64_t *next_window_start,
                                   int64_t new_start_sample) {
    if (!enc_cache || !n_enc_cache || !enc_cache_start ||
        !enc_cached_seq_total || !next_window_start) return;
    for (int i = *enc_cache_start; i < *n_enc_cache; i++) {
        free(enc_cache[i].enc_output);
        enc_cache[i].enc_output = NULL;
    }
    *n_enc_cache = 0;
    *enc_cache_start = 0;
    *enc_cached_seq_total = 0;
    *next_window_start = new_start_sample;
}

static int stream_reanchor_text_state(qwen_ctx_t *ctx,
                                      const int *emitted_text_tokens,
                                      int n_emitted_text_tokens,
                                      int carry_text_tokens,
                                      int **raw_tokens,
                                      int *raw_tokens_cap,
                                      int *n_raw_tokens,
                                      int **stable_text_tokens,
                                      int *stable_text_cap,
                                      int *n_stable_text_tokens) {
    if (!ctx || !raw_tokens || !raw_tokens_cap || !n_raw_tokens ||
        !stable_text_tokens || !stable_text_cap || !n_stable_text_tokens) return -1;

    int carry = n_emitted_text_tokens;
    if (carry_text_tokens > 0 && carry > carry_text_tokens) carry = carry_text_tokens;
    if (carry < 0) carry = 0;

    int raw_lead = (ctx->n_force_prompt_tokens <= 0) ? 1 : 0;
    int raw_need = raw_lead + carry;

    if (raw_need > *raw_tokens_cap) {
        int new_cap = *raw_tokens_cap > 0 ? *raw_tokens_cap : 64;
        while (raw_need > new_cap) new_cap *= 2;
        int *tmp_raw = (int *)realloc(*raw_tokens, (size_t)new_cap * sizeof(int));
        if (!tmp_raw) return -1;
        *raw_tokens = tmp_raw;
        *raw_tokens_cap = new_cap;
    }
    if (carry > *stable_text_cap) {
        int new_cap = *stable_text_cap > 0 ? *stable_text_cap : 64;
        while (carry > new_cap) new_cap *= 2;
        int *tmp_stable = (int *)realloc(*stable_text_tokens, (size_t)new_cap * sizeof(int));
        if (!tmp_stable) return -1;
        *stable_text_tokens = tmp_stable;
        *stable_text_cap = new_cap;
    }

    int tail_off = n_emitted_text_tokens - carry;
    if (tail_off < 0) tail_off = 0;
    if (raw_lead) (*raw_tokens)[0] = QWEN_TOKEN_ASR_TEXT;
    if (carry > 0 && emitted_text_tokens) {
        memcpy((*raw_tokens) + raw_lead,
               emitted_text_tokens + tail_off, (size_t)carry * sizeof(int));
        memcpy(*stable_text_tokens,
               emitted_text_tokens + tail_off, (size_t)carry * sizeof(int));
    }

    *n_raw_tokens = raw_need;
    *n_stable_text_tokens = carry;
    return 0;
}

/* ========================================================================
 * Streaming Implementation
 * ======================================================================== */

#define QWEN_STREAM_MAX_ENC_WINDOWS       4
#define QWEN_STREAM_MAX_PREFIX_TOKENS     150
#define QWEN_STREAM_MAX_REPEAT_TOKEN_RUN  12
#define QWEN_STREAM_OVERLAP_MAX_TOKENS    48
#define QWEN_STREAM_OVERLAP_MIN_TOKENS    4
#define QWEN_STREAM_DEGEN_MAX_PERIOD      6
#define QWEN_STREAM_DEGEN_MIN_REPEATS     4
#define QWEN_STREAM_STALE_CHUNKS          4
#define QWEN_STREAM_RESET_INTERVAL_CHUNKS 45
#define QWEN_STREAM_RESET_CARRY_TOKENS    24

/* Forward declarations — needed by stream_impl for control checking */
static qwen_control_action_t take_control(qwen_ctx_t *ctx);
static void set_state(qwen_ctx_t *ctx, qwen_pipeline_state_t state);

static char *stream_impl(qwen_ctx_t *ctx, const float *samples, int n_samples,
                          qwen_live_audio_t *live) {
    int dim = ctx->dec_config.dec_hidden;
    int chunk_samples = (int)(ctx->stream_chunk_sec * QWEN_SAMPLE_RATE);
    int rollback = ctx->stream_rollback;
    int unfixed_chunks = ctx->stream_unfixed_chunks;
    int max_new_tokens = ctx->stream_max_new_tokens > 0 ? ctx->stream_max_new_tokens : 32;

    const float *audio_samples = samples;
    int64_t audio_n_samples = n_samples;
    float *compacted_samples = NULL;
    if (!live && ctx->skip_silence) {
        int compacted_n = n_samples;
        compacted_samples = compact_silence(samples, n_samples, &compacted_n);
        if (compacted_samples) audio_samples = compacted_samples;
        audio_n_samples = compacted_n;
    }

    /* For live mode, keep a local rolling buffer with global sample base. */
    float *local_samples = NULL;
    int64_t local_n_samples = 0;
    int64_t local_capacity = 0;
    int64_t local_base_sample = 0;
    int live_eof = 0;

    if (live) {
        pthread_mutex_lock(&live->mutex);
        int64_t live_start = live->sample_offset;
        int64_t live_count = live->n_samples;
        live_eof = live->eof;
        local_n_samples = live_count;
        local_base_sample = live_start;
        if (local_n_samples > 0) {
            local_capacity = local_n_samples + chunk_samples * 4;
            if ((uint64_t)local_capacity > (uint64_t)(SIZE_MAX / sizeof(float))) {
                pthread_mutex_unlock(&live->mutex);
                return NULL;
            }
            local_samples = (float *)malloc((size_t)local_capacity * sizeof(float));
            if (!local_samples) { pthread_mutex_unlock(&live->mutex); return NULL; }
            memcpy(local_samples, live->samples, (size_t)local_n_samples * sizeof(float));
        }
        live->sample_offset = live_start + live_count;
        live->n_samples = 0;
        pthread_mutex_unlock(&live->mutex);
        audio_samples = local_samples;
        audio_n_samples = local_base_sample + local_n_samples;
    } else {
        if (audio_n_samples > INT_MAX) { free(compacted_samples); return NULL; }
        local_base_sample = 0;
        local_n_samples = audio_n_samples;
    }

    ctx->perf_total_ms = 0;
    ctx->perf_text_tokens = 0;
    ctx->perf_audio_ms = live ? 0.0 : 1000.0 * (double)n_samples / (double)QWEN_SAMPLE_RATE;
    ctx->perf_encode_ms = 0;
    ctx->perf_decode_ms = 0;
    int enc_window_frames = ctx->enc_config.enc_n_window_infer;
    if (enc_window_frames < 100) enc_window_frames = 100;
    if (enc_window_frames > 800) enc_window_frames = 800;
    int enc_window_samples = enc_window_frames * QWEN_HOP_LENGTH;
    const char *no_cache_env = getenv("QWEN_STREAM_NO_ENC_CACHE");
    int use_enc_cache = 1;
    if (no_cache_env && no_cache_env[0] != '\0' && strcmp(no_cache_env, "0") != 0)
        use_enc_cache = 0;
    if (live && !use_enc_cache) {
        if (qwen_verbose >= 1)
            fprintf(stderr, "Streaming (live): forcing encoder cache on\n");
        use_enc_cache = 1;
    }

    if (prepare_prompt_tokens(ctx) != 0) {
        free(compacted_samples);
        return NULL;
    }

    /* Non-interactive shortcut: skip chunked decoding */
    if (!ctx->token_cb && !live) {
        if (audio_n_samples > INT_MAX) { free(compacted_samples); return NULL; }
        char *text = transcribe_segment(ctx, audio_samples, (int)audio_n_samples,
                                        NULL, 0, NULL);
        free(compacted_samples);
        return text;
    }

    const uint16_t *tok_emb = ctx->dec_ctx.decoder.tok_embeddings_bf16;

    /* Raw decoded history */
    int *raw_tokens = (int *)malloc(8192 * sizeof(int));
    int n_raw_tokens = 0;
    int raw_tokens_cap = 8192;

    /* Stable committed text tokens */
    int *stable_text_tokens = (int *)malloc(8192 * sizeof(int));
    int n_stable_text_tokens = 0;
    int stable_text_cap = 8192;
    int *emitted_text_tokens = (int *)malloc(8192 * sizeof(int));
    int n_emitted_text_tokens = 0;
    int emitted_text_cap = 8192;
    int stagnant_chunks = 0;

    size_t result_cap = 4096;
    size_t result_len = 0;
    char *result = (char *)malloc(result_cap);
    if (!raw_tokens || !stable_text_tokens || !emitted_text_tokens || !result) {
        free(raw_tokens); free(stable_text_tokens); free(emitted_text_tokens);
        free(result); free(compacted_samples);
        return NULL;
    }
    result[0] = '\0';

    float *tmp_embed = (float *)malloc(dim * sizeof(float));
    if (!tmp_embed) {
        free(raw_tokens); free(stable_text_tokens); free(emitted_text_tokens);
        free(result); free(compacted_samples);
        return NULL;
    }

    int chunk_idx = 0;
    int64_t audio_cursor = 0;
    stream_enc_window_t *enc_cache = NULL;
    int n_enc_cache = 0;
    int enc_cache_start = 0;
    int enc_cache_cap = 0;
    int enc_cached_seq_total = 0;
    int64_t next_window_start = 0;
    float *prev_prefill_embeds = NULL;
    int prev_prefill_len = 0;
    int prev_prefill_cap = 0;
    int prefill_total_tokens = 0;
    int prefill_reused_tokens = 0;

    while (audio_cursor < audio_n_samples || (live && !live_eof)) {
        /* Live mode: wait for data */
        if (live) {
            int64_t want = audio_cursor + chunk_samples;
            pthread_mutex_lock(&live->mutex);
            while (live->sample_offset + live->n_samples < want && !live->eof)
                pthread_cond_wait(&live->cond, &live->mutex);

            int64_t live_start = live->sample_offset;
            int64_t live_count = live->n_samples;
            int64_t live_end = live_start + live_count;
            int is_eof_now = live->eof;

            int64_t local_end = local_base_sample + local_n_samples;
            if (local_end < live_start) {
                local_base_sample = live_start;
                local_n_samples = 0;
                local_end = local_base_sample;
            }

            if (live_end > local_end) {
                int64_t delta64 = live_end - local_end;
                int64_t src_off64 = local_end - live_start;
                if (delta64 < 0 || src_off64 < 0 || src_off64 > live_count) {
                    pthread_mutex_unlock(&live->mutex);
                    break;
                }
                if (local_n_samples + delta64 > local_capacity) {
                    int64_t new_cap = local_capacity > 0 ? local_capacity : 32000;
                    while (new_cap < local_n_samples + delta64) new_cap *= 2;
                    if ((uint64_t)new_cap > (uint64_t)(SIZE_MAX / sizeof(float))) {
                        pthread_mutex_unlock(&live->mutex);
                        break;
                    }
                    float *tmp = (float *)realloc(local_samples,
                                                  (size_t)new_cap * sizeof(float));
                    if (!tmp) { pthread_mutex_unlock(&live->mutex); break; }
                    local_samples = tmp;
                    local_capacity = new_cap;
                }
                memcpy(local_samples + (size_t)local_n_samples,
                       live->samples + (size_t)src_off64,
                       (size_t)delta64 * sizeof(float));
                local_n_samples += delta64;
            }

            live->sample_offset = live_end;
            live->n_samples = 0;
            live_eof = is_eof_now;
            pthread_mutex_unlock(&live->mutex);

            audio_samples = local_samples;
            audio_n_samples = local_base_sample + local_n_samples;
            ctx->perf_audio_ms = 1000.0 * (double)audio_n_samples / (double)QWEN_SAMPLE_RATE;

            /* Check for stop command — sets live_eof to trigger natural exit */
            {
                qwen_control_action_t ctl = take_control(ctx);
                if (ctl == QWEN_CONTROL_STOP_LISTENING ||
                    ctl == QWEN_CONTROL_STOP_AND_CLEAR) {
                    live_eof = 1;
                }
            }
        }

        double chunk_t0 = get_time_ms();
        audio_cursor += chunk_samples;
        if (audio_cursor > audio_n_samples) audio_cursor = audio_n_samples;
        int is_final = live ? (live_eof && audio_cursor >= audio_n_samples)
                            : (audio_cursor >= audio_n_samples);

        /* ---- Encoder ---- */
        double t0 = get_time_ms();
        int enc_seq_len = 0;
        float *enc_output = NULL;
        int64_t full_end = (audio_cursor / enc_window_samples) * (int64_t)enc_window_samples;

        if (!use_enc_cache) {
            if (audio_cursor > INT_MAX) { chunk_idx++; continue; }
            if (stream_encode_span(ctx, audio_samples, (int)audio_cursor,
                                   &enc_output, &enc_seq_len) != 0 ||
                !enc_output || enc_seq_len <= 0) {
                free(enc_output);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }
            ctx->perf_encode_ms += get_time_ms() - t0;
        } else {
            int enc_failed = 0;

            while (next_window_start < full_end) {
                int64_t ws = next_window_start;
                int64_t ws_local_off = ws - local_base_sample;
                if (ws_local_off < 0 ||
                    ws_local_off + enc_window_samples > local_n_samples) {
                    enc_failed = 1; break;
                }
                float *win_enc = NULL;
                int win_seq = 0;
                if (stream_encode_span(ctx,
                                       audio_samples + (size_t)ws_local_off,
                                       enc_window_samples,
                                       &win_enc, &win_seq) != 0 ||
                    !win_enc || win_seq <= 0) {
                    free(win_enc); enc_failed = 1; break;
                }

                if (n_enc_cache == enc_cache_cap) {
                    int new_cap = enc_cache_cap > 0 ? enc_cache_cap * 2 : 8;
                    stream_enc_window_t *tmp = (stream_enc_window_t *)realloc(
                        enc_cache, (size_t)new_cap * sizeof(stream_enc_window_t));
                    if (!tmp) { free(win_enc); enc_failed = 1; break; }
                    enc_cache = tmp;
                    enc_cache_cap = new_cap;
                }

                enc_cache[n_enc_cache].start_sample = ws;
                enc_cache[n_enc_cache].n_samples = enc_window_samples;
                enc_cache[n_enc_cache].seq_len = win_seq;
                enc_cache[n_enc_cache].enc_output = win_enc;
                n_enc_cache++;
                enc_cached_seq_total += win_seq;
                next_window_start += enc_window_samples;
            }

            float *partial_enc = NULL;
            int partial_seq = 0;
            if (!enc_failed && full_end < audio_cursor) {
                int64_t partial_samples64 = audio_cursor - full_end;
                int64_t partial_off64 = full_end - local_base_sample;
                if (partial_samples64 > INT_MAX || partial_off64 < 0 ||
                    partial_off64 + partial_samples64 > local_n_samples) {
                    enc_failed = 1;
                } else if (stream_encode_span(ctx,
                                       audio_samples + (size_t)partial_off64,
                                       (int)partial_samples64,
                                       &partial_enc, &partial_seq) != 0) {
                    free(partial_enc); partial_enc = NULL; enc_failed = 1;
                }
            }

            if (enc_failed) {
                free(partial_enc);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }

            /* Evict old windows */
            while (n_enc_cache - enc_cache_start > QWEN_STREAM_MAX_ENC_WINDOWS) {
                enc_cached_seq_total -= enc_cache[enc_cache_start].seq_len;
                free(enc_cache[enc_cache_start].enc_output);
                enc_cache[enc_cache_start].enc_output = NULL;
                enc_cache_start++;
                if (qwen_monitor) { fprintf(stderr, "\xe2\x9f\xb3"); fflush(stderr); }
            }

            enc_seq_len = enc_cached_seq_total + partial_seq;
            if (enc_seq_len <= 0) {
                free(partial_enc);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }

            enc_output = (float *)malloc((size_t)enc_seq_len * dim * sizeof(float));
            if (!enc_output) {
                free(partial_enc);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }

            int enc_off = 0;
            for (int i = enc_cache_start; i < n_enc_cache; i++) {
                memcpy(enc_output + (size_t)enc_off * dim,
                       enc_cache[i].enc_output,
                       (size_t)enc_cache[i].seq_len * dim * sizeof(float));
                enc_off += enc_cache[i].seq_len;
            }
            if (partial_seq > 0 && partial_enc) {
                memcpy(enc_output + (size_t)enc_off * dim,
                       partial_enc, (size_t)partial_seq * dim * sizeof(float));
            }
            free(partial_enc);

            ctx->perf_encode_ms += get_time_ms() - t0;
            if (qwen_monitor) { fprintf(stderr, "\xe2\x96\xb6"); fflush(stderr); }
        }

        /* ---- Build prefix rollback ---- */
        int n_prefix_tokens_full = 0;
        int n_prefix_tokens = 0;
        int prefix_offset = 0;
        if (ctx->past_text_conditioning && chunk_idx >= unfixed_chunks && n_raw_tokens > 0) {
            n_prefix_tokens_full = n_raw_tokens - rollback;
            if (n_prefix_tokens_full < 0) n_prefix_tokens_full = 0;
            n_prefix_tokens = n_prefix_tokens_full;
            if (n_prefix_tokens > QWEN_STREAM_MAX_PREFIX_TOKENS) {
                n_prefix_tokens = QWEN_STREAM_MAX_PREFIX_TOKENS;
                prefix_offset = n_prefix_tokens_full - n_prefix_tokens;
            }
        }

        /* ---- Build input embeddings ---- */
        int prefix_len = PREFIX_HEAD_LEN + ctx->n_prompt_tokens + PREFIX_TAIL_LEN;
        int suffix_len = SUFFIX_BASE_LEN + ctx->n_force_prompt_tokens;
        int total_seq = prefix_len + enc_seq_len + suffix_len + n_prefix_tokens;
        float *input_embeds = (float *)malloc((size_t)total_seq * dim * sizeof(float));
        if (!input_embeds) {
            free(enc_output);
            ctx->perf_total_ms += get_time_ms() - chunk_t0;
            chunk_idx++;
            continue;
        }

        int off = 0;
        for (int i = 0; i < PREFIX_HEAD_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim, tok_emb,
                                  PROMPT_PREFIX_HEAD[i], dim);
            off++;
        }
        for (int i = 0; i < ctx->n_prompt_tokens; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim, tok_emb,
                                  ctx->prompt_tokens[i], dim);
            off++;
        }
        for (int i = 0; i < PREFIX_TAIL_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim, tok_emb,
                                  PROMPT_PREFIX_TAIL[i], dim);
            off++;
        }

        for (int i = 0; i < enc_seq_len; i++)
            memcpy(input_embeds + (prefix_len + i) * dim,
                   enc_output + i * dim, dim * sizeof(float));
        free(enc_output);
        enc_output = NULL;

        int suffix_off = prefix_len + enc_seq_len;
        for (int i = 0; i < SUFFIX_BASE_LEN; i++)
            tok_embed_bf16_to_f32(input_embeds + (suffix_off + i) * dim, tok_emb,
                                  PROMPT_SUFFIX_BASE[i], dim);
        for (int i = 0; i < ctx->n_force_prompt_tokens; i++)
            tok_embed_bf16_to_f32(input_embeds + (suffix_off + SUFFIX_BASE_LEN + i) * dim,
                                  tok_emb, ctx->force_prompt_tokens[i], dim);

        int text_off = suffix_off + suffix_len;
        for (int i = 0; i < n_prefix_tokens; i++)
            tok_embed_bf16_to_f32(input_embeds + (text_off + i) * dim, tok_emb,
                                  raw_tokens[prefix_offset + i], dim);

        /* ---- Decoder prefill + first token ---- */
        t0 = get_time_ms();
        int prefill_len = total_seq - 1;
        int reused_prefill = 0;
        if (prev_prefill_embeds && prev_prefill_len > 0) {
            int cmp_len = prefill_len < prev_prefill_len ? prefill_len : prev_prefill_len;
            size_t row_bytes = (size_t)dim * sizeof(float);
            while (reused_prefill < cmp_len) {
                const float *a = prev_prefill_embeds + (size_t)reused_prefill * dim;
                const float *b = input_embeds + (size_t)reused_prefill * dim;
                if (memcmp(a, b, row_bytes) != 0) break;
                reused_prefill++;
            }
        }
        ctx->dec_ctx.kv_cache_len = reused_prefill;
        int delta_prefill = prefill_len - reused_prefill;
        if (delta_prefill > 0) {
            qkn_decoder_prefill(&ctx->dec_ctx,
                                input_embeds + (size_t)reused_prefill * dim,
                                delta_prefill);
        }
        prefill_total_tokens += prefill_len;
        prefill_reused_tokens += reused_prefill;

        float *last_embed = input_embeds + (size_t)prefill_len * dim;
        int token = qkn_decoder_forward(&ctx->dec_ctx, last_embed);

        if (prefill_len > prev_prefill_cap) {
            int new_cap = prev_prefill_cap > 0 ? prev_prefill_cap : 64;
            while (new_cap < prefill_len) new_cap *= 2;
            float *tmp_prev = (float *)realloc(prev_prefill_embeds,
                                               (size_t)new_cap * dim * sizeof(float));
            if (tmp_prev) { prev_prefill_embeds = tmp_prev; prev_prefill_cap = new_cap; }
            else { prev_prefill_len = 0; }
        }
        if (prev_prefill_embeds && prev_prefill_cap >= prefill_len) {
            memcpy(prev_prefill_embeds, input_embeds,
                   (size_t)prefill_len * dim * sizeof(float));
            prev_prefill_len = prefill_len;
        } else { prev_prefill_len = 0; }
        free(input_embeds);

        double prefill_ms = get_time_ms() - t0;
        ctx->perf_decode_ms += prefill_ms;
        if (qwen_monitor) { fprintf(stderr, "\xc2\xb7"); fflush(stderr); }

        /* ---- Autoregressive decode ---- */
        t0 = get_time_ms();
        int n_generated = 0;

        int *chunk_tokens = (int *)malloc((size_t)max_new_tokens * sizeof(int));
        if (!chunk_tokens) {
            ctx->perf_total_ms += get_time_ms() - chunk_t0;
            chunk_idx++;
            continue;
        }
        int n_chunk_tokens = 0;

        while (n_generated < max_new_tokens) {
            n_generated++;
            if (token == QWEN_TOKEN_ENDOFTEXT || token == QWEN_TOKEN_IM_END) break;
            chunk_tokens[n_chunk_tokens++] = token;
            tok_embed_bf16_to_f32(tmp_embed, tok_emb, token, dim);
            token = qkn_decoder_forward(&ctx->dec_ctx, tmp_embed);
        }

        double decode_ms = get_time_ms() - t0;
        ctx->perf_decode_ms += decode_ms;
        if (qwen_monitor) {
            double ms_per_tok = n_generated > 0 ? decode_ms / n_generated : 0;
            fprintf(stderr, "%s", ms_per_tok > 30 ? "\xe2\x96\xb8" : "\xe2\x96\xaa");
            fflush(stderr);
        }

        /* Update raw token history with repeat suppression */
        int dropped_repeat_tokens = 0;
        if (n_chunk_tokens > 0) {
            int prev_tok = -1;
            int prev_run = 0;
            if (n_prefix_tokens_full > 0) {
                prev_tok = raw_tokens[n_prefix_tokens_full - 1];
                prev_run = 1;
                for (int j = n_prefix_tokens_full - 2; j >= 0; j--) {
                    if (raw_tokens[j] != prev_tok) break;
                    prev_run++;
                    if (prev_run >= QWEN_STREAM_MAX_REPEAT_TOKEN_RUN) break;
                }
            }

            int out_idx = 0;
            for (int i = 0; i < n_chunk_tokens; i++) {
                int tok = chunk_tokens[i];
                int suppress = 0;
                if (tok == prev_tok) {
                    prev_run++;
                    if (prev_run > QWEN_STREAM_MAX_REPEAT_TOKEN_RUN) suppress = 1;
                } else {
                    prev_tok = tok;
                    prev_run = 1;
                }
                if (suppress) { dropped_repeat_tokens++; continue; }
                chunk_tokens[out_idx++] = tok;
            }
            n_chunk_tokens = out_idx;
        }

        int n_raw_new = n_prefix_tokens_full + n_chunk_tokens;
        if (n_raw_new > raw_tokens_cap) {
            while (n_raw_new > raw_tokens_cap) raw_tokens_cap *= 2;
            int *tmp_raw = (int *)realloc(raw_tokens, (size_t)raw_tokens_cap * sizeof(int));
            if (!tmp_raw) {
                free(chunk_tokens);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }
            raw_tokens = tmp_raw;
        }
        if (n_chunk_tokens > 0) {
            memcpy(raw_tokens + n_prefix_tokens_full, chunk_tokens,
                   (size_t)n_chunk_tokens * sizeof(int));
        }
        n_raw_tokens = n_raw_new;
        free(chunk_tokens);

        /* Parse text region */
        int text_start = 0;
        if (ctx->n_force_prompt_tokens <= 0) {
            int asr_text_pos = -1;
            for (int i = 0; i < n_raw_tokens; i++) {
                if (raw_tokens[i] == QWEN_TOKEN_ASR_TEXT) { asr_text_pos = i; break; }
            }
            text_start = (asr_text_pos >= 0) ? asr_text_pos + 1 : 0;
        }
        if (text_start < 0) text_start = 0;
        if (text_start > n_raw_tokens) text_start = n_raw_tokens;
        int n_text_tokens = n_raw_tokens - text_start;

        /* Fixed frontier */
        int candidate_len = 0;
        if (is_final) {
            candidate_len = n_text_tokens;
        } else if (chunk_idx >= unfixed_chunks) {
            candidate_len = n_text_tokens - rollback;
            if (candidate_len <= 0 && n_text_tokens > 0) candidate_len = n_text_tokens - 1;
            if (candidate_len < 0) candidate_len = 0;
        }

        /* Streaming commit */
        int *candidate_tokens = raw_tokens + text_start;
        {
            int tail_period = 0;
            int tail_reps = stream_tail_repeat_blocks(candidate_tokens, candidate_len,
                                                      QWEN_STREAM_DEGEN_MAX_PERIOD,
                                                      &tail_period);
            int candidate_advance = candidate_len - n_stable_text_tokens;
            if (!is_final && n_generated >= max_new_tokens && candidate_advance <= 1)
                stagnant_chunks++;
            else
                stagnant_chunks = 0;

            int recovery_reset = 0;
            if (tail_period > 0 && tail_reps >= QWEN_STREAM_DEGEN_MIN_REPEATS) recovery_reset = 1;
            if (stagnant_chunks >= QWEN_STREAM_STALE_CHUNKS) recovery_reset = 1;
            if (dropped_repeat_tokens >= 8) recovery_reset = 1;

            if (recovery_reset) {
                if (stream_reanchor_text_state(ctx,
                                               emitted_text_tokens, n_emitted_text_tokens,
                                               QWEN_STREAM_RESET_CARRY_TOKENS,
                                               &raw_tokens, &raw_tokens_cap, &n_raw_tokens,
                                               &stable_text_tokens, &stable_text_cap,
                                               &n_stable_text_tokens) != 0) {
                    n_raw_tokens = 0;
                    n_stable_text_tokens = 0;
                }
                prev_prefill_len = 0;
                stream_clear_enc_cache(enc_cache, &n_enc_cache, &enc_cache_start,
                                       &enc_cached_seq_total, &next_window_start, full_end);
                stagnant_chunks = 0;
                if (qwen_monitor) { fprintf(stderr, "!"); fflush(stderr); }
            } else {
                if (candidate_len > stable_text_cap) {
                    while (candidate_len > stable_text_cap) stable_text_cap *= 2;
                    int *tmp_stable = (int *)realloc(stable_text_tokens,
                                                     (size_t)stable_text_cap * sizeof(int));
                    if (!tmp_stable) candidate_len = n_stable_text_tokens;
                    else stable_text_tokens = tmp_stable;
                }

                int lcp = 0;
                while (lcp < n_stable_text_tokens && lcp < candidate_len &&
                       stable_text_tokens[lcp] == candidate_tokens[lcp]) lcp++;
                for (int i = lcp; i < candidate_len; i++)
                    stable_text_tokens[i] = candidate_tokens[i];

                int emit_start = lcp;
                if (emit_start < candidate_len && n_emitted_text_tokens > 0) {
                    int max_overlap = candidate_len - emit_start;
                    if (max_overlap > n_emitted_text_tokens) max_overlap = n_emitted_text_tokens;
                    if (max_overlap > QWEN_STREAM_OVERLAP_MAX_TOKENS)
                        max_overlap = QWEN_STREAM_OVERLAP_MAX_TOKENS;
                    for (int k = max_overlap; k >= QWEN_STREAM_OVERLAP_MIN_TOKENS; k--) {
                        if (memcmp(emitted_text_tokens + (n_emitted_text_tokens - k),
                                   candidate_tokens + emit_start,
                                   (size_t)k * sizeof(int)) == 0) {
                            emit_start += k;
                            break;
                        }
                    }
                }

                for (int i = emit_start; i < candidate_len; i++) {
                    int tok = stable_text_tokens[i];
                    const char *piece = smol_tokenizer_decode(ctx->tokenizer, tok);
                    if (ctx->token_cb) ctx->token_cb(piece, ctx->token_cb_userdata);

                    size_t plen = strlen(piece);
                    if (result_len + plen + 1 > result_cap) {
                        while (result_len + plen + 1 > result_cap) result_cap *= 2;
                        result = (char *)realloc(result, result_cap);
                    }
                    memcpy(result + result_len, piece, plen);
                    result_len += plen;
                    result[result_len] = '\0';
                    ctx->perf_text_tokens++;

                    if (n_emitted_text_tokens == emitted_text_cap) {
                        int new_cap = emitted_text_cap * 2;
                        int *tmp_emit = (int *)realloc(emitted_text_tokens,
                                                       (size_t)new_cap * sizeof(int));
                        if (tmp_emit) { emitted_text_tokens = tmp_emit; emitted_text_cap = new_cap; }
                    }
                    if (n_emitted_text_tokens < emitted_text_cap)
                        emitted_text_tokens[n_emitted_text_tokens++] = tok;
                }

                n_stable_text_tokens = candidate_len;

                int periodic_reset =
                    (!is_final && ctx->past_text_conditioning &&
                     chunk_idx >= unfixed_chunks &&
                     ((chunk_idx + 1) % QWEN_STREAM_RESET_INTERVAL_CHUNKS == 0));
                if (periodic_reset) {
                    if (stream_reanchor_text_state(ctx,
                                                   emitted_text_tokens, n_emitted_text_tokens,
                                                   QWEN_STREAM_RESET_CARRY_TOKENS,
                                                   &raw_tokens, &raw_tokens_cap, &n_raw_tokens,
                                                   &stable_text_tokens, &stable_text_cap,
                                                   &n_stable_text_tokens) != 0) {
                        n_raw_tokens = 0;
                        n_stable_text_tokens = 0;
                    }
                    prev_prefill_len = 0;
                    stream_clear_enc_cache(enc_cache, &n_enc_cache, &enc_cache_start,
                                           &enc_cached_seq_total, &next_window_start, full_end);
                }
            }
        }

        if (live && use_enc_cache) {
            int64_t keep_from = full_end;
            if (keep_from > local_base_sample) {
                int64_t drop64 = keep_from - local_base_sample;
                if (drop64 > local_n_samples) drop64 = local_n_samples;
                if (drop64 > 0) {
                    int64_t remain = local_n_samples - drop64;
                    if (remain > 0) {
                        memmove(local_samples, local_samples + (size_t)drop64,
                                (size_t)remain * sizeof(float));
                    }
                    local_n_samples = remain;
                    local_base_sample += drop64;
                    audio_samples = local_samples;
                    audio_n_samples = local_base_sample + local_n_samples;
                }
            }
        }

        ctx->perf_total_ms += get_time_ms() - chunk_t0;
        chunk_idx++;
    }

    free(tmp_embed);
    for (int i = enc_cache_start; i < n_enc_cache; i++)
        free(enc_cache[i].enc_output);
    free(enc_cache);
    if (qwen_verbose >= 2 && prefill_total_tokens > 0) {
        double reuse_pct = 100.0 * (double)prefill_reused_tokens / (double)prefill_total_tokens;
        fprintf(stderr, "  Prefill reuse: %d/%d tokens (%.1f%%)\n",
                prefill_reused_tokens, prefill_total_tokens, reuse_pct);
    }
    free(prev_prefill_embeds);
    free(raw_tokens);
    free(stable_text_tokens);
    free(emitted_text_tokens);
    free(compacted_samples);
    free(local_samples);

    /* Trim whitespace */
    size_t rlen = strlen(result);
    while (rlen > 0 && isspace((unsigned char)result[rlen - 1])) result[--rlen] = '\0';
    char *start = result;
    while (*start && isspace((unsigned char)*start)) start++;
    if (start != result) memmove(result, start, strlen(start) + 1);

    return result;
}

/* ========================================================================
 * Public Stream API
 * ======================================================================== */

char *qwen_transcribe_stream(qwen_ctx_t *ctx, const float *samples, int n_samples) {
    return stream_impl(ctx, samples, n_samples, NULL);
}

char *qwen_transcribe_stream_live(qwen_ctx_t *ctx, qwen_live_audio_t *live) {
    return stream_impl(ctx, NULL, 0, live);
}

void qwen_set_stream_chunk_sec(qwen_ctx_t *ctx, float sec) {
    ctx->stream_chunk_sec = sec;
}

/* Consume pending control action (called from pipeline thread). */
static qwen_control_action_t take_control(qwen_ctx_t *ctx) {
    return (qwen_control_action_t)atomic_exchange_explicit(
        &ctx->control_action, (uint32_t)QWEN_CONTROL_NONE,
        memory_order_acquire);
}

static void set_state(qwen_ctx_t *ctx, qwen_pipeline_state_t state) {
    atomic_store_explicit(&ctx->pipeline_state, (uint32_t)state,
                          memory_order_release);
}

void qwen_transcribe_stream_live_persistent(qwen_ctx_t *ctx, qwen_live_audio_t *live) {
    const int ww_hop = 160; /* 10ms at 16kHz */
    float ww_buf[160];

    /* Calibrate ambient noise floor before entering wakeword gate.
     * Feed ~500ms of audio through the detector in calibration mode,
     * then set the RMS gate to 3x ambient. */
    if (ctx->wakeword) {
        smol_ww_calibrate_start(ctx->wakeword);
        const int calib_hops = 50; /* ~500ms at 16kHz */
        for (int i = 0; i < calib_hops; i++) {
            pthread_mutex_lock(&live->mutex);
            while (live->n_samples < ww_hop)
                pthread_cond_wait(&live->cond, &live->mutex);
            /* Feed through detector (calibration mode collects RMS) */
            float calib_buf[160];
            memcpy(calib_buf, live->samples, ww_hop * sizeof(float));
            live->n_samples -= ww_hop;
            if (live->n_samples > 0)
                memmove(live->samples, live->samples + ww_hop,
                        (size_t)live->n_samples * sizeof(float));
            live->sample_offset += ww_hop;
            pthread_mutex_unlock(&live->mutex);
            smol_ww_process(ctx->wakeword, calib_buf, ww_hop);
        }
        smol_ww_calibrate_finish(ctx->wakeword, 3.0f);
    }

    for (;;) {
        set_state(ctx, QWEN_PIPELINE_IDLE);

        /* ---- Idle gate: wait for wakeword or START_LISTENING command ---- */
        if (ctx->wakeword) {
            /* Wakeword-gated: process audio hops through detector */
            for (;;) {
                qwen_control_action_t ctl = take_control(ctx);
                if (ctl == QWEN_CONTROL_START_LISTENING) {
                    smol_ww_activate(ctx->wakeword);
                    break;
                }
                if (ctl == QWEN_CONTROL_STOP_AND_CLEAR) {
                    smol_ww_reset(ctx->wakeword);
                }

                /* Wait for audio */
                pthread_mutex_lock(&live->mutex);
                while (live->n_samples < ww_hop)
                    pthread_cond_wait(&live->cond, &live->mutex);

                /* Drain audio in hops through the wakeword detector */
                while (live->n_samples >= ww_hop) {
                    memcpy(ww_buf, live->samples, ww_hop * sizeof(float));
                    live->n_samples -= ww_hop;
                    if (live->n_samples > 0) {
                        memmove(live->samples, live->samples + ww_hop,
                                (size_t)live->n_samples * sizeof(float));
                    }
                    live->sample_offset += ww_hop;
                    pthread_mutex_unlock(&live->mutex);

                    smol_ww_state_t ww = smol_ww_process(ctx->wakeword, ww_buf, ww_hop);
                    if (ww == SMOL_WW_PREFIX_DETECTED) {
                        set_state(ctx, QWEN_PIPELINE_PREFIX_DETECTED);
                    } else if (ww == SMOL_WW_WAITING) {
                        set_state(ctx, QWEN_PIPELINE_IDLE);
                    }

                    if (ww == SMOL_WW_LISTENING)
                        goto gate_open;

                    pthread_mutex_lock(&live->mutex);
                }
                pthread_mutex_unlock(&live->mutex);
            }
        } else {
            /* No wakeword: idle, drop audio, wait for START_LISTENING */
            for (;;) {
                qwen_control_action_t ctl = take_control(ctx);
                if (ctl == QWEN_CONTROL_START_LISTENING)
                    break;

                /* Drain and discard audio so LiveAudio doesn't grow unbounded */
                pthread_mutex_lock(&live->mutex);
                while (live->n_samples == 0 && !live->eof)
                    pthread_cond_wait(&live->cond, &live->mutex);
                live->sample_offset += live->n_samples;
                live->n_samples = 0;
                pthread_mutex_unlock(&live->mutex);
            }
        }
gate_open:

        set_state(ctx, QWEN_PIPELINE_LISTENING);

        /* Reset LiveAudio cursor so stream_impl starts from a clean base.
         * Any buffered audio from wakeword detection is stale — drop it. */
        pthread_mutex_lock(&live->mutex);
        live->n_samples = 0;
        live->sample_offset = 0;
        pthread_mutex_unlock(&live->mutex);

        /* ---- ASR session ---- */
        char *result = stream_impl(ctx, NULL, 0, live);
        if (result) free(result);

        set_state(ctx, QWEN_PIPELINE_PROCESSING);

        qkn_kv_cache_reset(&ctx->dec_ctx);

        /* Session boundary marker */
        if (ctx->token_cb)
            ctx->token_cb(NULL, ctx->token_cb_userdata);

        /* Reset wakeword for next session */
        if (ctx->wakeword)
            smol_ww_reset(ctx->wakeword);
    }
}

/* ========================================================================
 * Convenience Wrappers
 * ======================================================================== */

char *qwen_transcribe(qwen_ctx_t *ctx, const char *wav_path) {
    int n_samples = 0;
    float *samples = smol_load_wav(wav_path, &n_samples);
    if (!samples) {
        fprintf(stderr, "qwen_transcribe: cannot load %s\n", wav_path);
        return NULL;
    }
    char *text = qwen_transcribe_audio(ctx, samples, n_samples);
    free(samples);
    return text;
}

char *qwen_transcribe_stdin(qwen_ctx_t *ctx) {
    int n_samples = 0;
    float *samples = smol_read_pcm_stdin(&n_samples);
    if (!samples) return NULL;
    char *text = qwen_transcribe_audio(ctx, samples, n_samples);
    free(samples);
    return text;
}

/* ========================================================================
 * Wakeword Integration
 * ======================================================================== */

int qwen_load_wakeword(qwen_ctx_t *ctx, const char *codebook_path) {
    if (!ctx || !codebook_path) return -1;
    if (ctx->wakeword) { smol_ww_free(ctx->wakeword); ctx->wakeword = NULL; }

    smol_ww_config_t cfg;
    smol_ww_config_default(&cfg);
    ctx->wakeword = smol_ww_create(13, &cfg);
    if (!ctx->wakeword) return -1;

    if (smol_ww_load(ctx->wakeword, codebook_path) != 0) {
        smol_ww_free(ctx->wakeword);
        ctx->wakeword = NULL;
        return -1;
    }
    return 0;
}

int qwen_load_wakeword_bytes(qwen_ctx_t *ctx, const uint8_t *data, size_t size) {
    if (!ctx || !data || size == 0) return -1;
    if (ctx->wakeword) { smol_ww_free(ctx->wakeword); ctx->wakeword = NULL; }

    smol_ww_config_t cfg;
    smol_ww_config_default(&cfg);
    ctx->wakeword = smol_ww_create(13, &cfg);
    if (!ctx->wakeword) return -1;

    if (smol_ww_load_bytes(ctx->wakeword, data, size) != 0) {
        smol_ww_free(ctx->wakeword);
        ctx->wakeword = NULL;
        return -1;
    }
    return 0;
}

/* ========================================================================
 * Pipeline Control
 * ======================================================================== */

void qwen_post_control(qwen_ctx_t *ctx, qwen_control_action_t action) {
    if (!ctx) return;
    atomic_store_explicit(&ctx->control_action, (uint32_t)action,
                          memory_order_release);
}

qwen_pipeline_state_t qwen_get_pipeline_state(const qwen_ctx_t *ctx) {
    if (!ctx) return QWEN_PIPELINE_IDLE;
    return (qwen_pipeline_state_t)atomic_load_explicit(
        (_Atomic uint32_t *)&ctx->pipeline_state, memory_order_acquire);
}


