#!/usr/bin/env python3
"""
kittentts_pipeline.py - Complete KittenTTS model conversion pipeline

Orchestrates the full conversion from HuggingFace ONNX model to
self-contained C inference engine artifacts:

  Step 1: Download source files (ONNX model, CMU dict, frequency list)
  Step 2: Convert ONNX weights to safetensors (convert_kittentts.py)
  Step 3: Build embedded G2P table (build_g2p_table.py)
  Step 4: Export phoneme vocabulary JSON

All intermediate and output files are reproducible from the source URLs.

Usage:
    # Full pipeline (downloads + converts + builds G2P):
    python3 tools/kittentts_pipeline.py --build-dir .kittentts_build --output-dir kitten-tts-nano

    # Individual steps:
    python3 tools/kittentts_pipeline.py --step download --build-dir .kittentts_build
    python3 tools/kittentts_pipeline.py --step convert --build-dir .kittentts_build --output-dir kitten-tts-nano
    python3 tools/kittentts_pipeline.py --step g2p --build-dir .kittentts_build
    python3 tools/kittentts_pipeline.py --step inspect --build-dir .kittentts_build

Prerequisites:
    pip install -r tools/kittentts_requirements.txt

Source URLs:
    ONNX model:  https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32/resolve/main/model.onnx
    CMU dict:    https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
    Freq list:   https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt
"""

import argparse
import json
import os
import subprocess
import sys


URLS = {
    'onnx': 'https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32/resolve/main/model.onnx',
    'cmudict': 'https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict',
    'freq20k': 'https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt',
}

FILES = {
    'onnx': 'kitten_tts_nano.onnx',
    'cmudict': 'cmudict.dict',
    'freq20k': 'freq20k.txt',
}


def download_file(url, dest):
    """Download a file if it doesn't exist."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    import urllib.request
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"  -> {dest} ({size / 1024 / 1024:.1f} MB)")


def step_download(build_dir):
    """Step 1: Download all source files."""
    print("=== Step 1: Download source files ===")
    os.makedirs(build_dir, exist_ok=True)
    for key, url in URLS.items():
        dest = os.path.join(build_dir, FILES[key])
        download_file(url, dest)
    print()


def step_convert(build_dir, output_dir):
    """Step 2: Convert ONNX to safetensors."""
    print("=== Step 2: Convert ONNX -> safetensors ===")
    onnx_path = os.path.join(build_dir, FILES['onnx'])
    if not os.path.exists(onnx_path):
        print(f"  ERROR: {onnx_path} not found. Run --step download first.")
        return False

    # Import and run the converter directly
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, tools_dir)
    import convert_kittentts as conv

    tensors = conv.load_onnx_weights(onnx_path)
    remapped = conv.remap_all(tensors)

    os.makedirs(output_dir, exist_ok=True)
    st_path = os.path.join(output_dir, 'model.safetensors')
    conv.save_safetensors(remapped, st_path)
    conv.export_phoneme_vocab(output_dir)
    print()
    return True


def step_g2p(build_dir):
    """Step 3: Build embedded G2P table."""
    print("=== Step 3: Build G2P table ===")
    freq_path = os.path.join(build_dir, FILES['freq20k'])
    cmudict_path = os.path.join(build_dir, FILES['cmudict'])
    # Need the phoneme vocab from the output dir
    # Check common locations
    vocab_path = None
    for p in ['kitten-tts-nano/phoneme_vocab.json', 'phoneme_vocab.json']:
        if os.path.exists(p):
            vocab_path = p
            break
    if not vocab_path:
        print("  ERROR: phoneme_vocab.json not found. Run --step convert first.")
        return False

    if not os.path.exists(freq_path) or not os.path.exists(cmudict_path):
        print("  ERROR: frequency list or CMU dict not found. Run --step download first.")
        return False

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, tools_dir)
    import build_g2p_table as g2p

    with open(vocab_path) as f:
        vocab = json.load(f)
    print(f"  Phoneme vocab: {len(vocab)} entries")

    freq_words = open(freq_path).read().split()
    print(f"  Frequency list: {len(freq_words)} words")

    cmudict = g2p.load_cmudict(cmudict_path)
    print(f"  CMU dict: {len(cmudict)} words")

    entries = g2p.build_table(freq_words, cmudict, vocab)

    output_path = 'exports/kittentts/g2p_table.h'
    g2p.generate_c_header(entries, output_path)
    print()
    return True


def step_inspect(build_dir):
    """Inspect the ONNX model (tensor names, shapes, ops)."""
    print("=== Inspect ONNX model ===")
    onnx_path = os.path.join(build_dir, FILES['onnx'])
    if not os.path.exists(onnx_path):
        print(f"  ERROR: {onnx_path} not found. Run --step download first.")
        return False

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, tools_dir)
    import convert_kittentts as conv
    conv.inspect_onnx(onnx_path)
    print()
    return True


def main():
    parser = argparse.ArgumentParser(
        description='KittenTTS model conversion pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  download   Download ONNX model, CMU dict, and frequency list
  convert    Convert ONNX weights to safetensors + phoneme vocab
  g2p        Build embedded G2P dictionary (C header)
  inspect    Print ONNX model tensor names and shapes
  all        Run all steps (default)

Example - full pipeline from scratch:
  python3 tools/kittentts_pipeline.py --build-dir .kittentts_build --output-dir kitten-tts-nano
""")
    parser.add_argument('--step', default='all',
                       choices=['download', 'convert', 'g2p', 'inspect', 'all'],
                       help='Pipeline step to run (default: all)')
    parser.add_argument('--build-dir', default='.kittentts_build',
                       help='Directory for downloaded/intermediate files')
    parser.add_argument('--output-dir', default='kitten-tts-nano',
                       help='Directory for final model output (safetensors + vocab)')
    args = parser.parse_args()

    print(f"KittenTTS Pipeline")
    print(f"  Build dir:  {args.build_dir}")
    print(f"  Output dir: {args.output_dir}")
    print()

    if args.step in ('all', 'download'):
        step_download(args.build_dir)

    if args.step in ('all', 'convert'):
        if not step_convert(args.build_dir, args.output_dir):
            return 1

    if args.step in ('all', 'g2p'):
        if not step_g2p(args.build_dir):
            return 1

    if args.step == 'inspect':
        step_inspect(args.build_dir)

    if args.step == 'all':
        print("=== Pipeline complete ===")
        print(f"Model:     {args.output_dir}/model.safetensors")
        print(f"Vocab:     {args.output_dir}/phoneme_vocab.json")
        print(f"G2P table: exports/kittentts/g2p_table.h")
        print()
        print("Build and run:")
        print(f"  make clean && make lib MODEL=kittentts USE_BLAS=1")
        print(f"  clang -O3 -ffast-math -std=c11 -Icommon/kernels -Icommon/utils \\")
        print(f"    -Iexports/kittentts -Icommon/audio -DUSE_BLAS -DACCELERATE_NEW_LAPACK \\")
        print(f"    -DENABLE_KITTENTTS -o kittentts exports/kittentts/main.c \\")
        print(f"    -L. -lsmol -framework Accelerate -lm -lpthread")
        print(f"  ./kittentts -d {args.output_dir} -o out.wav \"Hello world\"")

    return 0


if __name__ == '__main__':
    sys.exit(main())
