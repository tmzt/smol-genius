/*
 * mfcc.c - Incremental MFCC extraction and segment-based wakeword detection
 *
 * MFCC pipeline (per 10ms hop):
 *   400-sample Hann window -> 201-bin power spectrum (DFT)
 *   -> 128-bin mel filterbank -> log -> DCT-II -> 13 MFCCs
 *
 * Wakeword detector (segment-based two-stage):
 *   Codebook of phoneme segments (averaged MFCC frames across voices).
 *   Phrases defined as segment ID sequences.
 *   Stage 1: match prefix segments (1-2 segs) via cosine similarity
 *   Stage 2: match remaining segments within confirmation window
 */

#include "mfcc.h"
#include "audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SAMPLE_RATE 16000

/* ========================================================================
 * Mel filterbank (same as audio.c)
 * ======================================================================== */

static float hertz_to_mel(float freq) {
    const float min_log_hertz = 1000.0f;
    const float min_log_mel = 15.0f;
    const float logstep = logf(6.4f) / 27.0f;
    float mels = 3.0f * freq / 200.0f;
    if (freq >= min_log_hertz) mels = min_log_mel + logf(freq / min_log_hertz) * logstep;
    return mels;
}

static float mel_to_hertz(float mels) {
    const float min_log_hertz = 1000.0f;
    const float min_log_mel = 15.0f;
    const float logstep = logf(6.4f) / 27.0f;
    float freq = 200.0f * mels / 3.0f;
    if (mels >= min_log_mel) freq = min_log_hertz * expf(logstep * (mels - min_log_mel));
    return freq;
}

static void build_mel_filters_into(float *filters) {
    int n_freq = SMOL_MFCC_N_FREQ;
    int n_mel = SMOL_MFCC_N_MEL;

    float fft_freqs[SMOL_MFCC_N_FREQ];
    for (int i = 0; i < n_freq; i++)
        fft_freqs[i] = (float)i * ((float)SAMPLE_RATE / 2.0f) / (float)(n_freq - 1);

    float mel_min = hertz_to_mel(0.0f);
    float mel_max = hertz_to_mel((float)SAMPLE_RATE / 2.0f);

    float filter_freqs[SMOL_MFCC_N_MEL + 2];
    float filter_diff[SMOL_MFCC_N_MEL + 1];
    for (int i = 0; i < n_mel + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * (float)i / (float)(n_mel + 1);
        filter_freqs[i] = mel_to_hertz(mel);
    }
    for (int i = 0; i < n_mel + 1; i++) {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
        if (filter_diff[i] == 0.0f) filter_diff[i] = 1e-6f;
    }

    memset(filters, 0, (size_t)n_mel * n_freq * sizeof(float));
    for (int m = 0; m < n_mel; m++) {
        float enorm = 2.0f / (filter_freqs[m + 2] - filter_freqs[m]);
        for (int f = 0; f < n_freq; f++) {
            float down = (fft_freqs[f] - filter_freqs[m]) / filter_diff[m];
            float up = (filter_freqs[m + 2] - fft_freqs[f]) / filter_diff[m + 1];
            float val = fminf(down, up);
            if (val < 0.0f) val = 0.0f;
            filters[m * n_freq + f] = val * enorm;
        }
    }
}

/* ========================================================================
 * MFCC Extractor
 * ======================================================================== */

smol_mfcc_t *smol_mfcc_create(int n_coeffs) {
    if (n_coeffs < 1 || n_coeffs > SMOL_MFCC_MAX_COEFFS) return NULL;

    smol_mfcc_t *m = (smol_mfcc_t *)calloc(1, sizeof(smol_mfcc_t));
    if (!m) return NULL;
    m->n_coeffs = n_coeffs;

    /* Hann window */
    for (int i = 0; i < SMOL_MFCC_WIN; i++)
        m->hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)SMOL_MFCC_WIN));

    /* Mel filterbank */
    build_mel_filters_into(m->mel_filters);

    /* DCT-II matrix: dct[c * N_MEL + m] = cos(PI * (m + 0.5) * c / N_MEL) */
    for (int c = 0; c < n_coeffs; c++) {
        for (int k = 0; k < SMOL_MFCC_N_MEL; k++) {
            m->dct[c * SMOL_MFCC_N_MEL + k] =
                cosf((float)M_PI * ((float)k + 0.5f) * (float)c / (float)SMOL_MFCC_N_MEL);
        }
    }

    /* DFT tables */
    for (int k = 0; k < SMOL_MFCC_N_FREQ; k++) {
        for (int n = 0; n < SMOL_MFCC_N_FFT; n++) {
            float angle = 2.0f * (float)M_PI * (float)k * (float)n / (float)SMOL_MFCC_N_FFT;
            m->dft_cos[k * SMOL_MFCC_N_FFT + n] = cosf(angle);
            m->dft_sin[k * SMOL_MFCC_N_FFT + n] = sinf(angle);
        }
    }

    return m;
}

void smol_mfcc_free(smol_mfcc_t *m) {
    free(m);
}

void smol_mfcc_compute(const smol_mfcc_t *m,
                        const float *windowed_400,
                        float *out) {
    int n_freq = SMOL_MFCC_N_FREQ;
    int n_mel = SMOL_MFCC_N_MEL;
    int n_coeffs = m->n_coeffs;

    /* Apply Hann window */
    float w[SMOL_MFCC_N_FFT];
    for (int i = 0; i < SMOL_MFCC_N_FFT; i++)
        w[i] = windowed_400[i] * m->hann[i];

    /* Power spectrum via DFT */
    float power[SMOL_MFCC_N_FREQ];
    for (int k = 0; k < n_freq; k++) {
        float re = 0, im = 0;
        const float *cos_row = m->dft_cos + k * SMOL_MFCC_N_FFT;
        const float *sin_row = m->dft_sin + k * SMOL_MFCC_N_FFT;
        for (int n = 0; n < SMOL_MFCC_N_FFT; n++) {
            re += w[n] * cos_row[n];
            im += w[n] * sin_row[n];
        }
        power[k] = re * re + im * im;
    }

    /* Mel filterbank -> log */
    float log_mel[SMOL_MFCC_N_MEL];
    for (int mi = 0; mi < n_mel; mi++) {
        float sum = 0.0f;
        const float *filt = m->mel_filters + mi * n_freq;
        for (int k = 0; k < n_freq; k++) sum += filt[k] * power[k];
        if (sum < 1e-10f) sum = 1e-10f;
        log_mel[mi] = logf(sum);
    }

    /* DCT-II -> MFCCs */
    for (int c = 0; c < n_coeffs; c++) {
        float sum = 0.0f;
        const float *dct_row = m->dct + c * n_mel;
        for (int k = 0; k < n_mel; k++) sum += dct_row[k] * log_mel[k];
        out[c] = sum;
    }
}

/* ========================================================================
 * Helpers
 * ======================================================================== */

static float vec_norm(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrtf(sum);
}

static void precompute_norms(float *norms, const float *frames,
                              int n_frames, int n_coeffs) {
    for (int i = 0; i < n_frames; i++)
        norms[i] = vec_norm(frames + i * n_coeffs, n_coeffs);
}

/* Extract MFCC frame sequence from raw audio.
 * Caller must free *out_frames and *out_norms. */
static int extract_mfcc_frames(const smol_mfcc_t *mfcc,
                                const float *samples, int n_samples,
                                float **out_frames, float **out_norms,
                                int *out_n_frames) {
    int n_coeffs = mfcc->n_coeffs;
    int n_frames = (n_samples - SMOL_MFCC_WIN) / SMOL_MFCC_HOP + 1;
    if (n_frames < 1) return -1;

    float *frames = (float *)malloc((size_t)n_frames * n_coeffs * sizeof(float));
    float *norms = (float *)malloc((size_t)n_frames * sizeof(float));
    if (!frames || !norms) { free(frames); free(norms); return -1; }

    for (int t = 0; t < n_frames; t++)
        smol_mfcc_compute(mfcc, samples + t * SMOL_MFCC_HOP,
                           frames + t * n_coeffs);

    precompute_norms(norms, frames, n_frames, n_coeffs);

    *out_frames = frames;
    *out_norms = norms;
    *out_n_frames = n_frames;
    return 0;
}

/* Cosine similarity between a segment's MFCC sequence and a portion of
 * the ring buffer ending at the current ring_pos.
 * offset: how many frames before ring_pos the segment ends.
 * Returns average per-frame cosine similarity in [0, 1]. */
static float segment_cosine_sim(const float *ring, const float *ring_norms,
                                 int ring_pos, int ring_cap, int ring_len,
                                 const smol_ww_segment_t *seg,
                                 int offset, int n_coeffs) {
    int seg_len = seg->n_frames;
    if (seg_len <= 0 || seg_len + offset > ring_len) return 0.0f;

    float sum = 0.0f;
    int matched = 0;
    for (int i = 0; i < seg_len; i++) {
        int ri = ((ring_pos - offset - seg_len + i) % ring_cap + ring_cap) % ring_cap;
        const float *rv = ring + ri * n_coeffs;
        const float *tv = seg->mfcc + i * n_coeffs;
        float rn = ring_norms[ri];
        float tn = seg->norms[i];

        if (rn < 1e-8f || tn < 1e-8f) continue;

        float dot = 0.0f;
        for (int c = 0; c < n_coeffs; c++) dot += rv[c] * tv[c];
        float sim = dot / (rn * tn);
        if (sim < 0.0f) sim = 0.0f;
        sum += sim;
        matched++;
    }

    return matched > 0 ? sum / (float)matched : 0.0f;
}

/* Match a phrase's segments against the ring buffer.
 * Segments are matched contiguously ending at ring_pos - frame_offset.
 * n_segs_to_check: how many segments of the phrase to check (prefix or full).
 * Returns average cosine similarity across all checked segments. */
static float match_phrase_segments(const smol_ww_detector_t *det,
                                    const smol_ww_phrase_t *phrase,
                                    int n_segs_to_check,
                                    int frame_offset) {
    if (n_segs_to_check <= 0) return 0.0f;

    float total_sim = 0.0f;
    int total_weight = 0;

    /* Walk segments from right to left (most recent first) */
    int offset = frame_offset;
    for (int s = n_segs_to_check - 1; s >= 0; s--) {
        int seg_id = phrase->seg_ids[s];
        if (seg_id < 0 || seg_id >= det->n_segments) return 0.0f;
        const smol_ww_segment_t *seg = &det->segments[seg_id];

        float sim = segment_cosine_sim(det->ring, det->ring_norms,
                                        det->ring_pos, det->ring_cap,
                                        det->ring_len,
                                        seg, offset, det->n_coeffs);
        total_sim += sim * (float)seg->n_frames;
        total_weight += seg->n_frames;
        offset += seg->n_frames;
    }

    return total_weight > 0 ? total_sim / (float)total_weight : 0.0f;
}

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

void smol_ww_config_default(smol_ww_config_t *cfg) {
    cfg->threshold = 0.80f;
    cfg->prefix_threshold = 0.72f;
    cfg->confirm_frames = 100;   /* 1s */
    cfg->listen_frames = 500;    /* 5s */
    cfg->silence_frames = 200;   /* 2s */
    cfg->energy_threshold = 50.0f;
}

smol_ww_detector_t *smol_ww_create(int n_coeffs, const smol_ww_config_t *cfg) {
    if (n_coeffs < 1 || n_coeffs > SMOL_MFCC_MAX_COEFFS) return NULL;

    smol_ww_detector_t *det = (smol_ww_detector_t *)calloc(1, sizeof(smol_ww_detector_t));
    if (!det) return NULL;

    det->mfcc = smol_mfcc_create(n_coeffs);
    if (!det->mfcc) { free(det); return NULL; }
    det->n_coeffs = n_coeffs;
    det->config = *cfg;

    /* Ring buffer: 300 frames = 3 seconds */
    det->ring_cap = 300;
    det->ring = (float *)calloc((size_t)det->ring_cap * n_coeffs, sizeof(float));
    det->ring_norms = (float *)calloc(det->ring_cap, sizeof(float));

    det->state = SMOL_WW_WAITING;

    /* Audio windowing buffer */
    det->audio_buf = (float *)calloc(SMOL_MFCC_WIN, sizeof(float));

    return det;
}

void smol_ww_free(smol_ww_detector_t *det) {
    if (!det) return;
    smol_mfcc_free(det->mfcc);
    for (int i = 0; i < det->n_segments; i++) {
        free(det->segments[i].mfcc);
        free(det->segments[i].norms);
    }
    free(det->ring);
    free(det->ring_norms);
    free(det->audio_buf);
    free(det);
}

/* ========================================================================
 * Segment Codebook Building
 * ======================================================================== */

int smol_ww_add_segment_audio(smol_ww_detector_t *det,
                               int segment_id,
                               const float *samples, int n_samples) {
    if (!det || !samples || segment_id < 0 ||
        segment_id >= SMOL_WW_MAX_SEGMENTS) return -1;

    int n_coeffs = det->n_coeffs;
    float *new_frames = NULL;
    float *new_norms = NULL;
    int new_n_frames = 0;

    if (extract_mfcc_frames(det->mfcc, samples, n_samples,
                             &new_frames, &new_norms, &new_n_frames) != 0)
        return -1;

    /* Extend n_segments if needed */
    while (det->n_segments <= segment_id) {
        memset(&det->segments[det->n_segments], 0, sizeof(smol_ww_segment_t));
        det->n_segments++;
    }

    smol_ww_segment_t *seg = &det->segments[segment_id];

    if (seg->mfcc == NULL) {
        /* First recording for this segment — just store it */
        seg->mfcc = new_frames;
        seg->norms = new_norms;
        seg->n_frames = new_n_frames;
    } else {
        /* Average with existing: use min frame count, running average */
        int min_frames = seg->n_frames < new_n_frames ? seg->n_frames : new_n_frames;

        for (int t = 0; t < min_frames; t++) {
            for (int c = 0; c < n_coeffs; c++) {
                seg->mfcc[t * n_coeffs + c] =
                    (seg->mfcc[t * n_coeffs + c] + new_frames[t * n_coeffs + c]) * 0.5f;
            }
        }
        seg->n_frames = min_frames;

        /* Recompute norms after averaging */
        precompute_norms(seg->norms, seg->mfcc, min_frames, n_coeffs);

        free(new_frames);
        free(new_norms);
    }

    /* Grow ring buffer if needed */
    int max_phrase_frames = 0;
    for (int p = 0; p < det->n_phrases; p++) {
        int frames = 0;
        for (int s = 0; s < det->phrases[p].n_segs; s++) {
            int sid = det->phrases[p].seg_ids[s];
            if (sid < det->n_segments) frames += det->segments[sid].n_frames;
        }
        if (frames > max_phrase_frames) max_phrase_frames = frames;
    }
    /* Also account for individual segment length */
    if (seg->n_frames + 50 > max_phrase_frames) max_phrase_frames = seg->n_frames + 50;

    if (max_phrase_frames + 50 > det->ring_cap) {
        int new_cap = max_phrase_frames + 100;
        float *new_ring = (float *)calloc((size_t)new_cap * n_coeffs, sizeof(float));
        float *nr = (float *)calloc(new_cap, sizeof(float));
        if (new_ring && nr) {
            free(det->ring);
            free(det->ring_norms);
            det->ring = new_ring;
            det->ring_norms = nr;
            det->ring_cap = new_cap;
            det->ring_pos = 0;
            det->ring_len = 0;
        } else {
            free(new_ring);
            free(nr);
        }
    }

    return segment_id;
}

int smol_ww_add_segment_wav(smol_ww_detector_t *det,
                             int segment_id,
                             const char *wav_path) {
    int n_samples = 0;
    float *samples = smol_load_wav(wav_path, &n_samples);
    if (!samples) return -1;
    int ret = smol_ww_add_segment_audio(det, segment_id, samples, n_samples);
    free(samples);
    return ret;
}

int smol_ww_add_phrase(smol_ww_detector_t *det,
                        const int *seg_ids, int n_segs,
                        int n_prefix_segs) {
    if (!det || !seg_ids || n_segs < 1 ||
        n_segs > SMOL_WW_MAX_PHRASE_SEGS ||
        det->n_phrases >= SMOL_WW_MAX_PHRASES) return -1;
    if (n_prefix_segs < 1) n_prefix_segs = 1;
    if (n_prefix_segs > n_segs) n_prefix_segs = n_segs;

    smol_ww_phrase_t *p = &det->phrases[det->n_phrases];
    memcpy(p->seg_ids, seg_ids, (size_t)n_segs * sizeof(int));
    p->n_segs = n_segs;
    p->n_prefix_segs = n_prefix_segs;

    /* Precompute frame counts */
    p->total_frames = 0;
    p->prefix_frames = 0;
    for (int s = 0; s < n_segs; s++) {
        int sid = seg_ids[s];
        int nf = (sid >= 0 && sid < det->n_segments) ? det->segments[sid].n_frames : 0;
        p->total_frames += nf;
        if (s < n_prefix_segs) p->prefix_frames += nf;
    }

    return det->n_phrases++;
}

/* ========================================================================
 * Binary Serialization
 *
 * Format (all little-endian):
 *   u32: magic  (0x57574D46 = "FMWW")
 *   u32: n_coeffs
 *   u32: n_segments
 *   u32: n_phrases
 *   For each segment:
 *     u32: n_frames
 *     float[n_frames * n_coeffs]: mfcc data
 *   For each phrase:
 *     u32: n_segs
 *     u32: n_prefix_segs
 *     u32[n_segs]: seg_ids
 * ======================================================================== */

#define SMOL_WW_MAGIC 0x57574D46u  /* "FMWW" */

int smol_ww_save(const smol_ww_detector_t *det, const char *path) {
    if (!det || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t magic = SMOL_WW_MAGIC;
    uint32_t nc = (uint32_t)det->n_coeffs;
    uint32_t ns = (uint32_t)det->n_segments;
    uint32_t np = (uint32_t)det->n_phrases;
    fwrite(&magic, 4, 1, f);
    fwrite(&nc, 4, 1, f);
    fwrite(&ns, 4, 1, f);
    fwrite(&np, 4, 1, f);

    for (int i = 0; i < det->n_segments; i++) {
        const smol_ww_segment_t *seg = &det->segments[i];
        uint32_t nf = (uint32_t)seg->n_frames;
        fwrite(&nf, 4, 1, f);
        if (nf > 0 && seg->mfcc)
            fwrite(seg->mfcc, sizeof(float), (size_t)nf * det->n_coeffs, f);
    }

    for (int i = 0; i < det->n_phrases; i++) {
        const smol_ww_phrase_t *p = &det->phrases[i];
        uint32_t n = (uint32_t)p->n_segs;
        uint32_t npre = (uint32_t)p->n_prefix_segs;
        fwrite(&n, 4, 1, f);
        fwrite(&npre, 4, 1, f);
        for (int s = 0; s < p->n_segs; s++) {
            uint32_t sid = (uint32_t)p->seg_ids[s];
            fwrite(&sid, 4, 1, f);
        }
    }

    fclose(f);
    return 0;
}

int smol_ww_load_bytes(smol_ww_detector_t *det,
                        const uint8_t *data, size_t size) {
    if (!det || !data || size < 16) return -1;

    const uint8_t *p = data;
    const uint8_t *end = data + size;

#define READ_U32(out) do { \
    if (p + 4 > end) return -1; \
    memcpy(&(out), p, 4); p += 4; \
} while (0)

    uint32_t magic, nc, ns, np;
    READ_U32(magic);
    if (magic != SMOL_WW_MAGIC) return -1;
    READ_U32(nc);
    READ_U32(ns);
    READ_U32(np);

    if ((int)nc != det->n_coeffs) return -1;
    if (ns > SMOL_WW_MAX_SEGMENTS || np > SMOL_WW_MAX_PHRASES) return -1;

    /* Load segments */
    for (uint32_t i = 0; i < ns; i++) {
        uint32_t nf;
        READ_U32(nf);
        size_t data_bytes = (size_t)nf * nc * sizeof(float);
        if (p + data_bytes > end) return -1;

        smol_ww_segment_t *seg = &det->segments[i];
        free(seg->mfcc);
        free(seg->norms);

        seg->n_frames = (int)nf;
        seg->mfcc = (float *)malloc(data_bytes);
        seg->norms = (float *)malloc((size_t)nf * sizeof(float));
        if (!seg->mfcc || !seg->norms) return -1;

        memcpy(seg->mfcc, p, data_bytes);
        p += data_bytes;
        precompute_norms(seg->norms, seg->mfcc, (int)nf, (int)nc);
    }
    det->n_segments = (int)ns;

    /* Load phrases */
    for (uint32_t i = 0; i < np; i++) {
        uint32_t n_segs, n_prefix;
        READ_U32(n_segs);
        READ_U32(n_prefix);
        if (n_segs > SMOL_WW_MAX_PHRASE_SEGS) return -1;

        smol_ww_phrase_t *ph = &det->phrases[i];
        ph->n_segs = (int)n_segs;
        ph->n_prefix_segs = (int)n_prefix;

        for (uint32_t s = 0; s < n_segs; s++) {
            uint32_t sid;
            READ_U32(sid);
            ph->seg_ids[s] = (int)sid;
        }

        /* Recompute frame counts */
        ph->total_frames = 0;
        ph->prefix_frames = 0;
        for (int s = 0; s < ph->n_segs; s++) {
            int sid = ph->seg_ids[s];
            int nf = (sid >= 0 && sid < det->n_segments) ?
                     det->segments[sid].n_frames : 0;
            ph->total_frames += nf;
            if (s < ph->n_prefix_segs) ph->prefix_frames += nf;
        }
    }
    det->n_phrases = (int)np;

#undef READ_U32

    /* Grow ring buffer to fit longest phrase */
    int max_frames = 0;
    for (int i = 0; i < det->n_phrases; i++) {
        if (det->phrases[i].total_frames > max_frames)
            max_frames = det->phrases[i].total_frames;
    }
    if (max_frames + 50 > det->ring_cap) {
        int new_cap = max_frames + 100;
        free(det->ring);
        free(det->ring_norms);
        det->ring = (float *)calloc((size_t)new_cap * det->n_coeffs, sizeof(float));
        det->ring_norms = (float *)calloc(new_cap, sizeof(float));
        det->ring_cap = new_cap;
        det->ring_pos = 0;
        det->ring_len = 0;
    }

    return 0;
}

int smol_ww_load(smol_ww_detector_t *det, const char *path) {
    if (!det || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return -1; }

    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);

    int ret = smol_ww_load_bytes(det, buf, (size_t)sz);
    free(buf);
    return ret;
}

/* ========================================================================
 * Detection State Machine
 * ======================================================================== */

static void process_one_frame(smol_ww_detector_t *det, const float *mfcc_frame) {
    int n_coeffs = det->n_coeffs;

    /* Push into ring buffer */
    memcpy(det->ring + det->ring_pos * n_coeffs, mfcc_frame,
           n_coeffs * sizeof(float));
    det->ring_norms[det->ring_pos] = vec_norm(mfcc_frame, n_coeffs);
    det->ring_pos = (det->ring_pos + 1) % det->ring_cap;
    if (det->ring_len < det->ring_cap) det->ring_len++;

    /* Energy check */
    float energy = det->ring_norms[(det->ring_pos - 1 + det->ring_cap) % det->ring_cap];
    int has_energy = (energy > det->config.energy_threshold);

    switch (det->state) {
    case SMOL_WW_WAITING: {
        if (det->n_phrases == 0 || det->n_segments == 0) break;

        /* Stage 1: check prefix segments of each phrase */
        for (int p = 0; p < det->n_phrases; p++) {
            smol_ww_phrase_t *phrase = &det->phrases[p];
            if (phrase->prefix_frames > det->ring_len) continue;

            float sim = match_phrase_segments(det, phrase,
                                              phrase->n_prefix_segs, 0);
            if (sim >= det->config.prefix_threshold) {
                det->state = SMOL_WW_PREFIX_DETECTED;
                det->matched_phrase = p;
                det->confirm_remaining = det->config.confirm_frames;
                return;
            }
        }
        break;
    }

    case SMOL_WW_PREFIX_DETECTED: {
        det->confirm_remaining--;
        if (det->confirm_remaining <= 0) {
            det->state = SMOL_WW_WAITING;
            return;
        }

        /* Stage 2: check full phrase that triggered prefix */
        smol_ww_phrase_t *phrase = &det->phrases[det->matched_phrase];
        if (phrase->total_frames > det->ring_len) break;

        float sim = match_phrase_segments(det, phrase, phrase->n_segs, 0);
        if (sim >= det->config.threshold) {
            det->state = SMOL_WW_LISTENING;
            det->listen_remaining = det->config.listen_frames;
            det->silence_count = 0;
            return;
        }
        break;
    }

    case SMOL_WW_LISTENING: {
        if (has_energy)
            det->silence_count = 0;
        else
            det->silence_count++;

        det->listen_remaining--;
        if (det->listen_remaining <= 0 ||
            det->silence_count >= det->config.silence_frames) {
            det->state = SMOL_WW_WAITING;
            det->silence_count = 0;
        }
        break;
    }
    }
}

smol_ww_state_t smol_ww_process(smol_ww_detector_t *det,
                                 const float *samples, int n_samples) {
    if (!det || !samples || n_samples <= 0) return det ? det->state : SMOL_WW_WAITING;

    float mfcc_out[SMOL_MFCC_MAX_COEFFS];

    for (int i = 0; i < n_samples; i++) {
        det->audio_buf[det->audio_pos] = samples[i];
        det->audio_pos = (det->audio_pos + 1) % SMOL_MFCC_WIN;
        if (det->audio_count < SMOL_MFCC_WIN) det->audio_count++;
        det->hop_count++;

        if (det->hop_count >= SMOL_MFCC_HOP && det->audio_count >= SMOL_MFCC_WIN) {
            det->hop_count = 0;

            /* Linearize circular buffer */
            float window[SMOL_MFCC_WIN];
            int start = (det->audio_pos - SMOL_MFCC_WIN + SMOL_MFCC_WIN) % SMOL_MFCC_WIN;
            if (start + SMOL_MFCC_WIN <= SMOL_MFCC_WIN) {
                memcpy(window, det->audio_buf + start, SMOL_MFCC_WIN * sizeof(float));
            } else {
                int first = SMOL_MFCC_WIN - start;
                memcpy(window, det->audio_buf + start, first * sizeof(float));
                memcpy(window + first, det->audio_buf, (SMOL_MFCC_WIN - first) * sizeof(float));
            }

            smol_mfcc_compute(det->mfcc, window, mfcc_out);
            process_one_frame(det, mfcc_out);
        }
    }

    return det->state;
}

smol_ww_state_t smol_ww_get_state(const smol_ww_detector_t *det) {
    return det ? det->state : SMOL_WW_WAITING;
}

void smol_ww_reset(smol_ww_detector_t *det) {
    if (!det) return;
    det->state = SMOL_WW_WAITING;
    det->confirm_remaining = 0;
    det->listen_remaining = 0;
    det->silence_count = 0;
}

void smol_ww_activate(smol_ww_detector_t *det) {
    if (!det) return;
    det->state = SMOL_WW_LISTENING;
    det->listen_remaining = det->config.listen_frames;
    det->silence_count = 0;
}
