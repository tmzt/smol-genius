/*
 * phonemizer.h - espeak-ng phonemization + phoneme-to-token mapping
 */

#ifndef KTTS_PHONEMIZER_H
#define KTTS_PHONEMIZER_H

/* Initialize espeak-ng. Call once before phonemize(). Returns 0 on success. */
int ktts_phonemizer_init(void);

/* Shut down espeak-ng. */
void ktts_phonemizer_cleanup(void);

/* Load phoneme vocabulary from a JSON file mapping characters to token IDs.
 * Populates phoneme_map[256]. Returns 0 on success. */
int ktts_load_phoneme_vocab(const char *json_path, int *phoneme_map);

/* Phonemize text to token IDs.
 * Wraps output as [0, <phoneme_ids...>, 10, 0] (start + end markers).
 * out_ids: pre-allocated buffer (max_ids capacity)
 * Returns number of token IDs written, or -1 on error. */
int ktts_phonemize(const char *text, const int *phoneme_map,
                   int *out_ids, int max_ids);

#endif /* KTTS_PHONEMIZER_H */
