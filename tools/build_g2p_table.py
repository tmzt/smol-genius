#!/usr/bin/env python3
"""
build_g2p_table.py - Build embedded English G2P lookup table

Takes a word frequency list + CMU pronouncing dictionary, produces a C header
with a perfect hash table mapping English words to KittenTTS phoneme token IDs.

Usage:
    python3 tools/build_g2p_table.py \
        --freq /tmp/freq20k.txt \
        --cmudict /tmp/cmudict.dict \
        --vocab kitten-tts-nano/phoneme_vocab.json \
        --output exports/kittentts/g2p_table.h

The generated header provides:
    int ktts_g2p_lookup(const char *word, int *out_ids, int max_ids);
"""

import argparse
import json
import os
import sys


# ARPAbet to IPA mapping
ARPA_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ',
    'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ',
    'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}

# Stress markers
STRESS_PRIMARY = 'ˈ'
STRESS_SECONDARY = 'ˌ'


def load_cmudict(path):
    """Load CMU dict, return {word: [(arpabet_phones, ...)]}."""
    entries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            parts = line.split()
            word = parts[0].lower()
            # Remove variant markers like "word(2)"
            if '(' in word:
                word = word[:word.index('(')]
            phones = tuple(parts[1:])
            if word not in entries:
                entries[word] = phones  # keep first pronunciation only
    return entries


def arpa_to_ipa_tokens(phones, vocab):
    """Convert ARPAbet phones to list of KittenTTS token IDs."""
    ids = []
    for p in phones:
        base = p.rstrip('012')
        stress = p[-1] if p[-1] in '012' else ''

        if stress == '1':
            # Primary stress mark before the syllable
            ipa_ch = STRESS_PRIMARY
            if ipa_ch in vocab:
                ids.append(vocab[ipa_ch])

        if base not in ARPA_TO_IPA:
            continue

        ipa = ARPA_TO_IPA[base]
        for ch in ipa:
            if ch in vocab:
                ids.append(vocab[ch])
    return ids


def djb2_hash(s):
    """DJB2 hash function matching the C implementation."""
    h = 5381
    for c in s.encode('ascii', errors='ignore'):
        h = ((h * 33) ^ c) & 0xFFFFFFFF
    return h


def build_table(freq_words, cmudict, vocab):
    """Build the G2P lookup table."""
    entries = []  # (word, token_ids)
    missed = 0

    for word in freq_words:
        w = word.lower().strip()
        if not w or not w.isascii():
            continue
        if w in cmudict:
            ids = arpa_to_ipa_tokens(cmudict[w], vocab)
            if ids:
                entries.append((w, ids))
            else:
                missed += 1
        else:
            missed += 1

    print(f"G2P table: {len(entries)} words mapped, {missed} missed")
    return entries


def generate_c_header(entries, output_path):
    """Generate a C header with embedded hash table."""
    # Build hash table with open addressing
    n_entries = len(entries)
    # Use ~1.5x load factor for the hash table
    table_size = 1
    while table_size < n_entries * 3 // 2:
        table_size *= 2

    # Build phoneme data pool and word pool
    phoneme_pool = []  # flat array of all phoneme IDs
    word_pool = []     # flat array of all word chars
    table = [None] * table_size  # (word_offset, word_len, phone_offset, phone_len)

    word_offset = 0
    phone_offset = 0

    collisions = 0
    for word, ids in entries:
        h = djb2_hash(word) % table_size
        probes = 0
        while table[h] is not None:
            h = (h + 1) % table_size
            probes += 1
            if probes > table_size:
                raise RuntimeError("Hash table full")
        if probes > 0:
            collisions += 1

        table[h] = (word_offset, len(word), phone_offset, len(ids))
        for c in word:
            word_pool.append(ord(c))
        phoneme_pool.extend(ids)
        word_offset += len(word)
        phone_offset += len(ids)

    print(f"Hash table: {table_size} slots, {collisions} collisions, "
          f"word pool: {len(word_pool)} bytes, phoneme pool: {len(phoneme_pool)} bytes")

    total_bytes = (table_size * 8  # 4 uint16 per slot
                   + len(word_pool)
                   + len(phoneme_pool))
    print(f"Total data: {total_bytes / 1024:.1f} KB")

    # Generate C code
    with open(output_path, 'w') as f:
        f.write("/*\n")
        f.write(" * g2p_table.h - Embedded English grapheme-to-phoneme lookup table\n")
        f.write(f" * Auto-generated from CMU dict ({n_entries} words)\n")
        f.write(" * Do not edit manually.\n")
        f.write(" */\n\n")
        f.write("#ifndef KTTS_G2P_TABLE_H\n")
        f.write("#define KTTS_G2P_TABLE_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <string.h>\n\n")

        f.write(f"#define G2P_TABLE_SIZE {table_size}\n")
        f.write(f"#define G2P_WORD_POOL_SIZE {len(word_pool)}\n")
        f.write(f"#define G2P_PHONE_POOL_SIZE {len(phoneme_pool)}\n\n")

        # Word pool
        f.write("static const char g2p_word_pool[] = {\n")
        for i in range(0, len(word_pool), 32):
            chunk = word_pool[i:i+32]
            f.write("    " + ",".join(f"'{chr(c)}'" if 32 <= c < 127 and c != ord("'") and c != ord('\\')
                                       else f"0x{c:02x}" for c in chunk) + ",\n")
        f.write("};\n\n")

        # Phoneme ID pool (uint8 since all IDs < 178)
        f.write("static const uint8_t g2p_phone_pool[] = {\n")
        for i in range(0, len(phoneme_pool), 32):
            chunk = phoneme_pool[i:i+32]
            f.write("    " + ",".join(str(v) for v in chunk) + ",\n")
        f.write("};\n\n")

        # Hash table: packed as (word_off:uint16, word_len:uint8, phone_off:uint16, phone_len:uint8)
        # But word_off can exceed 16 bits for 20K words. Use uint32 offsets.
        f.write("/* Hash table slot: word_offset(24bit) | word_len(8bit), phone_offset(24bit) | phone_len(8bit) */\n")
        f.write("static const uint32_t g2p_table[][2] = {\n")
        for slot in table:
            if slot is None:
                f.write("    {0xFFFFFFFF, 0xFFFFFFFF},\n")
            else:
                wo, wl, po, pl = slot
                # Pack: high 24 bits = offset, low 8 bits = length
                w_packed = (wo << 8) | (wl & 0xFF)
                p_packed = (po << 8) | (pl & 0xFF)
                f.write(f"    {{0x{w_packed:08X}, 0x{p_packed:08X}}},\n")
        f.write("};\n\n")

        # DJB2 hash function
        f.write("static inline uint32_t g2p_djb2(const char *s, int len) {\n")
        f.write("    uint32_t h = 5381;\n")
        f.write("    for (int i = 0; i < len; i++)\n")
        f.write("        h = ((h << 5) + h) ^ (uint8_t)s[i];\n")
        f.write("    return h;\n")
        f.write("}\n\n")

        # Lookup function
        f.write("/*\n")
        f.write(" * Look up a word in the G2P table.\n")
        f.write(" * word: lowercase ASCII word (not null-terminated, length given)\n")
        f.write(" * out_ids: output phoneme token IDs\n")
        f.write(" * max_ids: capacity of out_ids\n")
        f.write(" * Returns number of phoneme IDs written, or 0 if not found.\n")
        f.write(" */\n")
        f.write("static int g2p_lookup(const char *word, int word_len, int *out_ids, int max_ids) {\n")
        f.write("    uint32_t h = g2p_djb2(word, word_len) % G2P_TABLE_SIZE;\n")
        f.write("    for (int probe = 0; probe < G2P_TABLE_SIZE; probe++) {\n")
        f.write("        uint32_t w_packed = g2p_table[h][0];\n")
        f.write("        uint32_t p_packed = g2p_table[h][1];\n")
        f.write("        if (w_packed == 0xFFFFFFFF) return 0; /* empty slot */\n")
        f.write("        int wo = (int)(w_packed >> 8);\n")
        f.write("        int wl = (int)(w_packed & 0xFF);\n")
        f.write("        if (wl == word_len && memcmp(g2p_word_pool + wo, word, wl) == 0) {\n")
        f.write("            int po = (int)(p_packed >> 8);\n")
        f.write("            int pl = (int)(p_packed & 0xFF);\n")
        f.write("            if (pl > max_ids) pl = max_ids;\n")
        f.write("            for (int i = 0; i < pl; i++)\n")
        f.write("                out_ids[i] = g2p_phone_pool[po + i];\n")
        f.write("            return pl;\n")
        f.write("        }\n")
        f.write("        h = (h + 1) % G2P_TABLE_SIZE;\n")
        f.write("    }\n")
        f.write("    return 0;\n")
        f.write("}\n\n")

        f.write("#endif /* KTTS_G2P_TABLE_H */\n")

    print(f"Written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Build embedded G2P table')
    parser.add_argument('--freq', required=True, help='Word frequency list (one word per line)')
    parser.add_argument('--cmudict', required=True, help='CMU pronouncing dictionary')
    parser.add_argument('--vocab', required=True, help='KittenTTS phoneme_vocab.json')
    parser.add_argument('--output', required=True, help='Output C header path')
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = json.load(f)

    print(f"Phoneme vocab: {len(vocab)} entries")

    freq_words = open(args.freq).read().split()
    print(f"Frequency list: {len(freq_words)} words")

    cmudict = load_cmudict(args.cmudict)
    print(f"CMU dict: {len(cmudict)} words")

    entries = build_table(freq_words, cmudict, vocab)
    generate_c_header(entries, args.output)


if __name__ == '__main__':
    main()
