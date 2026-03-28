#!/usr/bin/env python3
"""
kittentts_roundtrip.py - TTS -> ASR roundtrip test harness

Synthesizes words/phrases with kittentts (or macOS say), transcribes
with qwen_asr, and compares input vs output text. Measures Word Error Rate.

Usage:
    # Quick smoke test with kittentts
    python3 tools/kittentts_roundtrip.py --quick -v

    # Use macOS say as TTS (ground truth baseline)
    python3 tools/kittentts_roundtrip.py --quick -v --tts say

    # Compare both side by side
    python3 tools/kittentts_roundtrip.py --quick -v --compare

    # Full 20K word sweep
    python3 tools/kittentts_roundtrip.py --freq .kittentts_build/freq20k.txt --compare

Requirements:
    - ./kittentts binary (make lib MODEL=kittentts APP=1 USE_BLAS=1)
    - ./qwen_asr binary (make lib MODEL=kittentts,qwen_asr APP=1 USE_BLAS=1)
    - kitten-tts-nano/ model directory
    - qwen3-asr-0.6b/ or qwen3-asr-1.7b/ model directory
    - macOS (for --tts say / --compare)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time


def synthesize_kittentts(text, wav_path, tts_binary, tts_model, style=0, speed=1.0):
    """Run kittentts to produce a WAV file. Returns (success, stderr)."""
    cmd = [tts_binary, '-d', tts_model, '-s', str(style),
           '--speed', str(speed), '-o', wav_path, '--silent', text]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.returncode == 0, r.stderr.strip()


def synthesize_say(text, wav_path, voice='Samantha'):
    """Use macOS say to produce a 16kHz mono WAV. Returns (success, stderr)."""
    try:
        aiff_path = wav_path + '.aiff'
        cmd_say = ['say', '-v', voice, '-o', aiff_path, text]
        r = subprocess.run(cmd_say, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False, f'say failed: {r.stderr}'
        cmd_conv = ['afconvert', '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1',
                    aiff_path, wav_path]
        r = subprocess.run(cmd_conv, capture_output=True, text=True, timeout=10)
        try:
            os.unlink(aiff_path)
        except OSError:
            pass
        if r.returncode != 0:
            return False, f'afconvert failed: {r.stderr}'
        return True, ''
    except FileNotFoundError:
        return False, 'say or afconvert not found (macOS only)'


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


def run_roundtrip(texts, tts_fn, asr_binary, asr_model, verbose=False):
    """Run roundtrip on a list of texts with a given TTS function.
    tts_fn(text, wav_path) -> (success, error_msg)
    Returns (results, total_tts_ms, total_asr_ms)."""
    results = []
    total_tts_ms = 0
    total_asr_ms = 0

    with tempfile.TemporaryDirectory(prefix='kittentts_rt_') as tmpdir:
        for i, text in enumerate(texts):
            wav_path = os.path.join(tmpdir, f'sample_{i:05d}.wav')

            # TTS
            t0 = time.monotonic()
            ok, tts_err = tts_fn(text, wav_path)
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


def print_summary(label, results, total_tts_ms, total_asr_ms):
    """Print summary statistics."""
    n = len(results)
    if n == 0:
        print(f"[{label}] No results.")
        return

    exact = sum(1 for r in results if r['match'])
    errors = sum(1 for r in results if 'error' in r)
    avg_wer = sum(r['wer'] for r in results) / n
    avg_tts = total_tts_ms / n
    avg_asr = total_asr_ms / max(n - errors, 1)

    print(f"\n{'=' * 60}")
    print(f"[{label}] Roundtrip Results: {n} samples")
    print(f"  Exact match:  {exact}/{n} ({exact/n:.1%})")
    print(f"  Avg WER:      {avg_wer:.1%}")
    print(f"  TTS errors:   {errors}")
    print(f"  Avg TTS time: {avg_tts:.0f} ms/sample")
    print(f"  Avg ASR time: {avg_asr:.0f} ms/sample")
    print(f"  Total time:   {(total_tts_ms + total_asr_ms) / 1000:.1f} s")

    mismatches = [r for r in results if not r['match'] and 'error' not in r]
    if mismatches:
        mismatches.sort(key=lambda r: -r['wer'])
        print(f"\n  Worst mismatches (top {min(10, len(mismatches))}):")
        for r in mismatches[:10]:
            print(f'    "{r["input"]}" -> "{r["output"]}" (WER={r["wer"]:.0%})')


def print_comparison(say_results, ktts_results):
    """Print side-by-side comparison of say vs kittentts."""
    n = len(say_results)

    say_exact = sum(1 for r in say_results if r['match'])
    ktts_exact = sum(1 for r in ktts_results if r['match'])
    say_wer = sum(r['wer'] for r in say_results) / max(n, 1)
    ktts_wer = sum(r['wer'] for r in ktts_results) / max(n, 1)

    print(f"\n{'=' * 60}")
    print(f"Comparison: {n} samples")
    print(f"{'':>25s} {'say':>10s} {'kittentts':>10s}")
    print(f"{'Exact match':>25s} {say_exact:>9d}  {ktts_exact:>9d}")
    print(f"{'Match rate':>25s} {say_exact/max(n,1):>9.1%}  {ktts_exact/max(n,1):>9.1%}")
    print(f"{'Avg WER':>25s} {say_wer:>9.1%}  {ktts_wer:>9.1%}")

    # Per-sample comparison
    both_ok = 0
    say_only = 0
    ktts_only = 0
    neither = 0
    for s, k in zip(say_results, ktts_results):
        sm, km = s['match'], k['match']
        if sm and km:
            both_ok += 1
        elif sm and not km:
            say_only += 1
        elif not sm and km:
            ktts_only += 1
        else:
            neither += 1

    print(f"\n  Both correct:     {both_ok}")
    print(f"  say only:         {say_only}")
    print(f"  kittentts only:   {ktts_only}")
    print(f"  Neither:          {neither}")

    # Show where they differ
    diffs = []
    for s, k in zip(say_results, ktts_results):
        if s['output'] != k['output']:
            diffs.append((s['input'], s['output'], k['output']))
    if diffs:
        print(f"\n  Differing outputs (top {min(10, len(diffs))}):")
        for inp, s_out, k_out in diffs[:10]:
            print(f'    "{inp}"')
            print(f'      say:       "{s_out}"')
            print(f'      kittentts: "{k_out}"')


def main():
    parser = argparse.ArgumentParser(
        description='TTS -> ASR roundtrip test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TTS engines:
  kittentts   Use ./kittentts binary (default)
  say         Use macOS say command (ground truth baseline)

Examples:
  %(prog)s --quick -v                        # kittentts quick test
  %(prog)s --quick -v --tts say              # macOS say quick test
  %(prog)s --quick -v --compare              # side-by-side comparison
  %(prog)s --freq .kittentts_build/freq20k.txt --limit 100 --compare  # top 100 words
""")
    parser.add_argument('--tts', default='kittentts', choices=['kittentts', 'say'],
                       help='TTS engine (default: kittentts)')
    parser.add_argument('--compare', action='store_true',
                       help='Run both kittentts and say, compare results')
    parser.add_argument('--tts-binary', default='./kittentts')
    parser.add_argument('--tts-model', default='kitten-tts-nano')
    parser.add_argument('--say-voice', default='Samantha',
                       help='macOS say voice (default: Samantha)')
    parser.add_argument('--asr-binary', default='./qwen_asr')
    parser.add_argument('--asr-model', default='models/qwen3-asr-0.6b',
                       help='ASR model dir')
    parser.add_argument('--style', type=int, default=0, help='KittenTTS voice style 0-7')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--words', nargs='+', help='Specific words to test')
    parser.add_argument('--phrases', help='File with one phrase per line')
    parser.add_argument('--freq', help='Word frequency file (one word per line)')
    parser.add_argument('--limit', type=int, default=100,
                       help='Max words from frequency list (default: 100)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick smoke test with 20 common words')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # Check ASR binary
    if not os.path.exists(args.asr_binary):
        print(f"Error: ASR binary not found: {args.asr_binary}")
        print(f"Build with: make lib MODEL=kittentts,qwen_asr USE_BLAS=1 APP=1")
        return 1
    if not os.path.isdir(args.asr_model):
        print(f"Error: ASR model dir not found: {args.asr_model}")
        return 1

    # Build text list — one word per sample for clean comparison
    if args.quick:
        texts = [
            "hello", "world", "good", "morning", "test",
            "one", "two", "three", "four", "five",
            "computer", "science", "artificial", "intelligence",
            "the", "quick", "brown", "fox", "jumps", "over",
        ]
    elif args.words:
        texts = args.words
    elif args.phrases:
        with open(args.phrases) as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.freq:
        with open(args.freq) as f:
            words = [w.strip() for w in f if w.strip() and w.strip().isascii()]
        texts = words[:args.limit]
    else:
        parser.print_help()
        print("\nSpecify --quick, --words, --phrases, or --freq")
        return 1

    auto_verbose = args.verbose or len(texts) <= 20

    # Build TTS functions
    def make_kittentts_fn():
        if not os.path.exists(args.tts_binary):
            print(f"Error: kittentts binary not found: {args.tts_binary}")
            sys.exit(1)
        if not os.path.isdir(args.tts_model):
            print(f"Error: kittentts model dir not found: {args.tts_model}")
            sys.exit(1)
        return lambda text, wav: synthesize_kittentts(
            text, wav, args.tts_binary, args.tts_model,
            style=args.style, speed=args.speed)

    def make_say_fn():
        return lambda text, wav: synthesize_say(text, wav, voice=args.say_voice)

    if args.compare:
        # Run both engines
        print(f"Roundtrip comparison: {len(texts)} samples")
        print(f"  ASR: {args.asr_binary} -d {args.asr_model}")
        print()

        print(f"--- macOS say (voice: {args.say_voice}) ---")
        say_fn = make_say_fn()
        say_results, say_tts, say_asr = run_roundtrip(
            texts, say_fn, args.asr_binary, args.asr_model, verbose=auto_verbose)
        print_summary('say', say_results, say_tts, say_asr)

        print(f"\n--- kittentts (voice: {args.style}) ---")
        ktts_fn = make_kittentts_fn()
        ktts_results, ktts_tts, ktts_asr = run_roundtrip(
            texts, ktts_fn, args.asr_binary, args.asr_model, verbose=auto_verbose)
        print_summary('kittentts', ktts_results, ktts_tts, ktts_asr)

        print_comparison(say_results, ktts_results)
    else:
        # Single engine
        if args.tts == 'say':
            label = f'say (voice: {args.say_voice})'
            tts_fn = make_say_fn()
        else:
            label = f'kittentts (voice: {args.style})'
            tts_fn = make_kittentts_fn()

        print(f"Roundtrip test: {len(texts)} samples")
        print(f"  TTS: {label}")
        print(f"  ASR: {args.asr_binary} -d {args.asr_model}")
        print()

        results, tts_ms, asr_ms = run_roundtrip(
            texts, tts_fn, args.asr_binary, args.asr_model, verbose=auto_verbose)
        print_summary(args.tts, results, tts_ms, asr_ms)

    return 0


if __name__ == '__main__':
    sys.exit(main())
