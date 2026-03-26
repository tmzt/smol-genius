/*
 * phonemizer.c - Self-contained English grapheme-to-phoneme + token mapping
 *
 * Uses an embedded 18K-word G2P dictionary (from CMU dict) compiled into
 * g2p_table.h. No runtime dependencies — no espeak, no data files.
 *
 * For words not in the dictionary, falls back to simple letter-to-sound rules.
 *
 * Token sequence format: [0, <phoneme_ids...>, 10, 0]
 *   Start marker: 0 (pad)
 *   End marker: 10
 *   Trailing pad: 0
 */

#include "phonemizer.h"
#include "g2p_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int ktts_phonemizer_init(void) {
    /* No initialization needed — dictionary is compiled in */
    return 0;
}

void ktts_phonemizer_cleanup(void) {
    /* Nothing to clean up */
}

/* ========================================================================
 * Phoneme Vocabulary Loading
 *
 * Expects a simple JSON file: {"$": 0, ";": 1, ...}
 * Parses single-byte keys only (sufficient for ASCII + Latin-1 first bytes).
 * Multi-byte IPA characters are handled by the G2P table directly.
 * ======================================================================== */

int ktts_load_phoneme_vocab(const char *json_path, int *phoneme_map) {
    for (int i = 0; i < 256; i++)
        phoneme_map[i] = -1;

    FILE *f = fopen(json_path, "r");
    if (!f) {
        fprintf(stderr, "kittentts: cannot open phoneme vocab: %s\n", json_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);

    const char *p = buf;
    int count = 0;
    while (*p) {
        const char *q1 = strchr(p, '"');
        if (!q1) break;
        q1++;

        const char *q2 = q1;
        while (*q2 && *q2 != '"') {
            if (*q2 == '\\') q2++;
            q2++;
        }
        if (!*q2) break;

        int key_len = (int)(q2 - q1);
        unsigned char key_char = 0;
        if (key_len == 1) {
            key_char = (unsigned char)q1[0];
        } else if (key_len == 2 && q1[0] == '\\') {
            key_char = (unsigned char)q1[1];
        } else if (key_len >= 2) {
            key_char = (unsigned char)q1[0];
        }

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
 * Simple letter-to-sound fallback for unknown words
 *
 * Very basic English pronunciation rules. Not accurate, but ensures
 * every word produces *some* phoneme sequence rather than silence.
 * ======================================================================== */

static int fallback_letter_to_sound(const char *word, int word_len,
                                      const int *phoneme_map,
                                      int *out_ids, int max_ids) {
    int n = 0;
    /* Simple: map each letter to its most common single phoneme.
     * phoneme_map has a-z at positions 97-122 mapped to token IDs 132-157.
     * These correspond to letter names, not sounds, but it's a fallback. */
    for (int i = 0; i < word_len && n < max_ids; i++) {
        unsigned char c = (unsigned char)word[i];
        if (c >= 'a' && c <= 'z') {
            int id = phoneme_map[c];
            if (id >= 0) out_ids[n++] = id;
        }
    }
    return n;
}

/* ========================================================================
 * Phonemization: text -> token IDs
 * ======================================================================== */

int ktts_phonemize(const char *text, const int *phoneme_map,
                   int *out_ids, int max_ids) {
    int n = 0;
    if (max_ids < 4) return -1;

    out_ids[n++] = 0;  /* start pad */

    const char *p = text;
    while (*p && n < max_ids - 2) {
        /* Skip whitespace, emit space token */
        if (isspace((unsigned char)*p)) {
            int space_id = phoneme_map[' '];
            if (space_id >= 0 && n > 1) /* don't lead with space */
                out_ids[n++] = space_id;
            p++;
            continue;
        }

        /* Handle punctuation */
        if (ispunct((unsigned char)*p)) {
            int id = phoneme_map[(unsigned char)*p];
            if (id >= 0 && n < max_ids - 2)
                out_ids[n++] = id;
            p++;
            continue;
        }

        /* Extract word (letters only) */
        const char *word_start = p;
        while (*p && isalpha((unsigned char)*p)) p++;
        int word_len = (int)(p - word_start);
        if (word_len == 0) { p++; continue; }

        /* Lowercase the word into a stack buffer */
        char lower[256];
        int llen = word_len < 255 ? word_len : 255;
        for (int i = 0; i < llen; i++)
            lower[i] = tolower((unsigned char)word_start[i]);
        lower[llen] = '\0';

        /* Look up in G2P dictionary */
        int word_ids[128];
        int nph = g2p_lookup(lower, llen, word_ids, 128);

        if (nph > 0) {
            /* Found in dictionary */
            for (int i = 0; i < nph && n < max_ids - 2; i++)
                out_ids[n++] = word_ids[i];
        } else {
            /* Fallback: letter-to-sound */
            int flen = fallback_letter_to_sound(lower, llen, phoneme_map,
                                                  word_ids, 128);
            for (int i = 0; i < flen && n < max_ids - 2; i++)
                out_ids[n++] = word_ids[i];
        }
    }

    if (n >= max_ids - 1) n = max_ids - 2;
    out_ids[n++] = 10; /* end marker */
    out_ids[n++] = 0;  /* trailing pad */
    return n;
}
