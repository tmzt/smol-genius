/*
 * phonemizer.c - espeak-ng phonemization + phoneme-to-token ID mapping
 *
 * Uses espeak-ng's C API to convert text to IPA phonemes, then maps
 * each IPA character to a token ID from the KittenTTS vocabulary.
 *
 * The vocabulary has 178 tokens:
 *   0 = pad/$, 1-16 = punctuation, 17-68 = ASCII, 69-177 = IPA symbols
 *
 * Token sequence format: [0, <phoneme_ids...>, 10, 0]
 *   Start marker: 0 (pad)
 *   End marker: 10 (ellipsis '…' in vocab)
 *   Trailing pad: 0
 */

#include "phonemizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ENABLE_ESPEAK
#include <espeak-ng/speak_lib.h>
#endif

#ifdef ENABLE_ESPEAK
static int espeak_initialized = 0;
#endif

int ktts_phonemizer_init(void) {
#ifdef ENABLE_ESPEAK
    if (espeak_initialized) return 0;

    int sr = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, NULL, 0);
    if (sr == -1) {
        fprintf(stderr, "kittentts: espeak_Initialize failed\n");
        return -1;
    }

    /* Set voice to English (US) */
    espeak_SetVoiceByName("en-us");
    espeak_initialized = 1;
    return 0;
#else
    fprintf(stderr, "kittentts: not compiled with ENABLE_KITTENTTS\n");
    return -1;
#endif
}

void ktts_phonemizer_cleanup(void) {
#ifdef ENABLE_ESPEAK
    if (espeak_initialized) {
        espeak_Terminate();
        espeak_initialized = 0;
    }
#endif
}

/* ========================================================================
 * Phoneme Vocabulary Loading
 *
 * Expects a simple JSON file: {"$": 0, ";": 1, ...}
 * We parse it minimally (no full JSON parser needed).
 * ======================================================================== */

int ktts_load_phoneme_vocab(const char *json_path, int *phoneme_map) {
    /* Initialize map to -1 (unknown) */
    for (int i = 0; i < 256; i++) {
        phoneme_map[i] = -1;
    }

    FILE *f = fopen(json_path, "r");
    if (!f) {
        fprintf(stderr, "kittentts: cannot open phoneme vocab: %s\n", json_path);
        return -1;
    }

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);

    /* Simple parser: find "X": N pairs */
    const char *p = buf;
    int count = 0;
    while (*p) {
        /* Find opening quote */
        const char *q1 = strchr(p, '"');
        if (!q1) break;
        q1++;

        /* Find closing quote — handle escape sequences */
        const char *q2 = q1;
        while (*q2 && *q2 != '"') {
            if (*q2 == '\\') q2++; /* skip escaped char */
            q2++;
        }
        if (!*q2) break;

        /* Extract the key character(s) */
        int key_len = (int)(q2 - q1);
        unsigned char key_char = 0;

        if (key_len == 1) {
            key_char = (unsigned char)q1[0];
        } else if (key_len == 2 && q1[0] == '\\') {
            key_char = (unsigned char)q1[1];
        } else if (key_len >= 2) {
            /* Multi-byte UTF-8 IPA characters — use first byte as key.
             * For full Unicode support, this would need a proper map.
             * For now, we handle the common single-byte ASCII + Latin-1 range. */
            key_char = (unsigned char)q1[0];
        }

        /* Find colon and number */
        p = q2 + 1;
        const char *colon = strchr(p, ':');
        if (!colon) break;

        int id = (int)strtol(colon + 1, (char **)&p, 10);
        if (key_char > 0) {
            phoneme_map[key_char] = id;
            count++;
        }
    }

    free(buf);
    fprintf(stderr, "kittentts: loaded %d phoneme vocab entries from %s\n", count, json_path);
    return 0;
}

/* ========================================================================
 * Phonemization
 * ======================================================================== */

int ktts_phonemize(const char *text, const int *phoneme_map,
                   int *out_ids, int max_ids) {
#ifdef ENABLE_ESPEAK
    if (!espeak_initialized) {
        fprintf(stderr, "kittentts: espeak not initialized\n");
        return -1;
    }

    /* Convert text to phonemes using espeak-ng */
    const char *input = text;
    int textmode = espeakCHARS_AUTO;
    int phonememode = espeakPHONEMES_IPA;  /* IPA output with stress marks */

    /* espeak_TextToPhonemes returns a pointer to a static buffer */
    const char *phonemes = espeak_TextToPhonemes(
        (const void **)&input, textmode, phonememode);

    if (!phonemes) {
        fprintf(stderr, "kittentts: espeak_TextToPhonemes failed\n");
        return -1;
    }

    /* Build token sequence: [0, <phoneme_ids>, 10, 0] */
    int n = 0;
    if (n >= max_ids) return -1;
    out_ids[n++] = 0;  /* start pad */

    /* Map each character of the IPA string to a token ID */
    const unsigned char *ph = (const unsigned char *)phonemes;
    while (*ph && n < max_ids - 2) {
        int id = phoneme_map[*ph];
        if (id >= 0) {
            out_ids[n++] = id;
        }
        /* Skip unknown characters silently */
        ph++;
    }

    if (n >= max_ids - 1) {
        /* Truncated — still add end markers */
        n = max_ids - 2;
    }

    out_ids[n++] = 10; /* end marker */
    out_ids[n++] = 0;  /* trailing pad */

    return n;
#else
    (void)text; (void)phoneme_map; (void)out_ids; (void)max_ids;
    return -1;
#endif
}
