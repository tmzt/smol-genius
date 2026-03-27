#!/usr/bin/env python3
"""
kittentts_roundtrip.py - TTS -> ASR roundtrip test harness

Synthesizes words/phrases with kittentts, transcribes with qwen_asr,
and compares input vs output text. Measures Word Error Rate (WER).

Usage:
    # Quick smoke test (10 words)
    python3 tools/kittentts_roundtrip.py --quick

    # Full 20K word sweep
    python3 tools/kittentts_roundtrip.py --freq .kittentts_build/freq20k.txt

    # Custom word list
    python3 tools/kittentts_roundtrip.py --words "hello world" "good morning"

    # Phrases from file
    python3 tools/kittentts_roundtrip.py --phrases phrases.txt

Requirements:
    - ./kittentts binary (make lib MODEL=kittentts APP=1 USE_BLAS=1)
    - ./qwen_asr binary (make lib MODEL=kittentts,qwen_asr APP=1 USE_BLAS=1)
    - kitten-tts-nano/ model directory
    - qwen3-asr-0.6b/ or qwen3-asr-1.7b/ model directory
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time


def synthesize(text, tts_binary, tts_model, wav_path, style=0, speed=1.0):
    """Run kittentts to produce a WAV file. Returns (success, stderr)."""
    cmd = [tts_binary, '-d', tts_model, '-s', str(style),
           '--speed', str(speed), '-o', wav_path, '--silent', text]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.returncode == 0, r.stderr.strip()


def transcribe(wav_path, asr_binary, asr_model):
    """Run qwen_asr on a WAV file. Returns (text, stderr)."""
    cmd = [asr_binary, '-d', asr_model, '-i', wav_path, '--silent']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return r.stdout.strip(), r.stderr.strip()


def normalize(text):
    """Normalize text for comparison: lowercase, strip punctuation."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def word_error_rate(ref, hyp):
    """Compute word error rate between reference and hypothesis."""
    ref_words = ref.split()
    hyp_words = hyp.split()

    # Levenshtein on words
    r = len(ref_words)
    h = len(hyp_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[r][h] / max(r, 1)


def run_roundtrip(texts, tts_binary, tts_model, asr_binary, asr_model,
                  style=0, speed=1.0, verbose=False):
    """Run roundtrip on a list of texts. Returns results list."""
    results = []
    total_tts_ms = 0
    total_asr_ms = 0

    with tempfile.TemporaryDirectory(prefix='kittentts_rt_') as tmpdir:
        for i, text in enumerate(texts):
            wav_path = os.path.join(tmpdir, f'sample_{i:05d}.wav')

            # TTS
            t0 = time.monotonic()
            ok, tts_err = synthesize(text, tts_binary, tts_model, wav_path,
                                     style=style, speed=speed)
            tts_ms = (time.monotonic() - t0) * 1000
            total_tts_ms += tts_ms

            if not ok:
                results.append({
                    'input': text, 'output': '', 'match': False,
                    'wer': 1.0, 'tts_ms': tts_ms, 'asr_ms': 0,
                    'error': f'TTS failed: {tts_err}'
                })
                if verbose:
                    print(f'  [{i+1}/{len(texts)}] TTS FAIL: "{text}" -> {tts_err}')
                continue

            # Check WAV was produced and has content
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 100:
                results.append({
                    'input': text, 'output': '', 'match': False,
                    'wer': 1.0, 'tts_ms': tts_ms, 'asr_ms': 0,
                    'error': 'TTS produced empty/missing WAV'
                })
                continue

            # ASR
            t0 = time.monotonic()
            output, asr_err = transcribe(wav_path, asr_binary, asr_model)
            asr_ms = (time.monotonic() - t0) * 1000
            total_asr_ms += asr_ms

            # Compare
            ref_norm = normalize(text)
            hyp_norm = normalize(output)
            match = ref_norm == hyp_norm
            wer = word_error_rate(ref_norm, hyp_norm)

            results.append({
                'input': text, 'output': output, 'match': match,
                'wer': wer, 'tts_ms': tts_ms, 'asr_ms': asr_ms,
                'ref_norm': ref_norm, 'hyp_norm': hyp_norm,
            })

            if verbose:
                status = 'OK' if match else 'MISMATCH'
                print(f'  [{i+1}/{len(texts)}] {status}: "{text}" -> "{output}" '
                      f'(WER={wer:.0%}, tts={tts_ms:.0f}ms, asr={asr_ms:.0f}ms)')

    return results, total_tts_ms, total_asr_ms


def print_summary(results, total_tts_ms, total_asr_ms):
    """Print summary statistics."""
    n = len(results)
    if n == 0:
        print("No results.")
        return

    exact = sum(1 for r in results if r['match'])
    errors = sum(1 for r in results if 'error' in r)
    avg_wer = sum(r['wer'] for r in results) / n
    avg_tts = total_tts_ms / n
    avg_asr = total_asr_ms / max(n - errors, 1)

    print(f"\n{'=' * 60}")
    print(f"Roundtrip Results: {n} samples")
    print(f"  Exact match:  {exact}/{n} ({exact/n:.1%})")
    print(f"  Avg WER:      {avg_wer:.1%}")
    print(f"  TTS errors:   {errors}")
    print(f"  Avg TTS time: {avg_tts:.0f} ms/sample")
    print(f"  Avg ASR time: {avg_asr:.0f} ms/sample")
    print(f"  Total time:   {(total_tts_ms + total_asr_ms) / 1000:.1f} s")

    # Show worst mismatches
    mismatches = [r for r in results if not r['match'] and 'error' not in r]
    if mismatches:
        mismatches.sort(key=lambda r: -r['wer'])
        print(f"\nWorst mismatches (top {min(10, len(mismatches))}):")
        for r in mismatches[:10]:
            print(f'  "{r["input"]}" -> "{r["output"]}" (WER={r["wer"]:.0%})')


def build_test_phrases(words, group_size=3):
    """Group individual words into short phrases for more realistic TTS."""
    phrases = []
    for i in range(0, len(words), group_size):
        phrase = ' '.join(words[i:i + group_size])
        phrases.append(phrase)
    return phrases


def main():
    parser = argparse.ArgumentParser(description='KittenTTS -> Qwen ASR roundtrip test')
    parser.add_argument('--tts-binary', default='./kittentts')
    parser.add_argument('--tts-model', default='kitten-tts-nano')
    parser.add_argument('--asr-binary', default='./qwen_asr')
    parser.add_argument('--asr-model', default='qwen3-asr-0.6b',
                       help='ASR model dir (qwen3-asr-0.6b or qwen3-asr-1.7b)')
    parser.add_argument('--style', type=int, default=0, help='Voice style 0-7')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--words', nargs='+', help='Specific words to test')
    parser.add_argument('--phrases', help='File with one phrase per line')
    parser.add_argument('--freq', help='Word frequency file (one word per line)')
    parser.add_argument('--limit', type=int, default=100,
                       help='Max words from frequency list (default: 100)')
    parser.add_argument('--group-size', type=int, default=3,
                       help='Words per phrase when using --freq (default: 3)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick smoke test with 10 common phrases')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # Check binaries exist
    for name, path in [('TTS', args.tts_binary), ('ASR', args.asr_binary)]:
        if not os.path.exists(path):
            print(f"Error: {name} binary not found: {path}")
            print(f"Build with: make lib MODEL=kittentts,qwen_asr USE_BLAS=1 APP=1")
            return 1

    # Check model dirs
    for name, path in [('TTS', args.tts_model), ('ASR', args.asr_model)]:
        if not os.path.isdir(path):
            print(f"Error: {name} model dir not found: {path}")
            return 1

    # Build text list
    if args.quick:
        texts = [
            "hello", "world", "hello world",
            "the quick brown fox", "good morning",
            "one two three four five",
            "this is a test", "how are you",
            "computer science", "artificial intelligence",
        ]
    elif args.words:
        texts = args.words
    elif args.phrases:
        with open(args.phrases) as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.freq:
        with open(args.freq) as f:
            words = [w.strip() for w in f if w.strip() and w.strip().isascii()]
        words = words[:args.limit]
        texts = build_test_phrases(words, args.group_size)
    else:
        parser.print_help()
        print("\nSpecify --quick, --words, --phrases, or --freq")
        return 1

    print(f"Roundtrip test: {len(texts)} samples")
    print(f"  TTS: {args.tts_binary} -d {args.tts_model} (voice {args.style})")
    print(f"  ASR: {args.asr_binary} -d {args.asr_model}")
    print()

    results, tts_ms, asr_ms = run_roundtrip(
        texts, args.tts_binary, args.tts_model,
        args.asr_binary, args.asr_model,
        style=args.style, speed=args.speed,
        verbose=args.verbose or len(texts) <= 20)

    print_summary(results, tts_ms, asr_ms)
    return 0


if __name__ == '__main__':
    sys.exit(main())
