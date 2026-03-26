/*
 * phonemizer.h - Self-contained English grapheme-to-phoneme
 *
 * Uses embedded 18K-word G2P dictionary (CMU dict). No runtime dependencies.
 */

#ifndef KTTS_PHONEMIZER_H
#define KTTS_PHONEMIZER_H

/* Initialize G2P engine. Returns 0 on success. */
int ktts_phonemizer_init(void);

/* Clean up. */
void ktts_phonemizer_cleanup(void);

/* Load phoneme vocabulary from JSON (maps characters to token IDs).
 * Populates phoneme_map[256]. Returns 0 on success. */
int ktts_load_phoneme_vocab(const char *json_path, int *phoneme_map);

/* Phonemize text to token IDs.
 * Uses embedded G2P dictionary for known words, letter-to-sound fallback for unknown.
 * Wraps output as [0, <phoneme_ids...>, 10, 0].
 * Returns number of token IDs written, or -1 on error. */
int ktts_phonemize(const char *text, const int *phoneme_map,
                   int *out_ids, int max_ids);

#endif /* KTTS_PHONEMIZER_H */
