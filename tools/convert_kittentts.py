#!/usr/bin/env python3
"""
convert_kittentts.py - Convert KittenTTS ONNX model to safetensors format

Extracts all initializer tensors from the ONNX model and saves them as
a single safetensors file with names remapped to match the C inference engine.

Also converts the NPZ voice embeddings and exports the phoneme vocabulary.

Usage:
    python3 tools/convert_kittentts.py \
        --onnx path/to/model.onnx \
        --voices path/to/voices.npz \
        --output-dir kitten-tts-nano

Requirements:
    pip install onnx safetensors numpy
"""

import argparse
import json
import os
import sys

import numpy as np


def load_onnx_initializers(onnx_path):
    """Load all initializer tensors from an ONNX model."""
    import onnx
    from onnx import numpy_helper

    print(f"Loading ONNX model from {onnx_path}...")
    model = onnx.load(onnx_path)
    graph = model.graph

    tensors = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        tensors[init.name] = arr

    print(f"Found {len(tensors)} initializer tensors")
    return tensors, model


def remap_tensor_names(tensors):
    """
    Remap ONNX tensor names to the clean hierarchy expected by the C code.
    This mapping will need to be refined based on the actual ONNX tensor names.
    """
    remapped = {}
    unmapped = []

    # Build a name mapping based on common ONNX naming patterns
    # The exact mapping depends on the specific ONNX export
    for name, arr in tensors.items():
        new_name = name

        # PL-BERT mappings
        if 'bert' in name.lower() or 'plbert' in name.lower():
            new_name = remap_plbert(name)
        elif 'text_encoder' in name.lower():
            new_name = remap_text_encoder(name)
        elif 'predictor' in name.lower():
            new_name = remap_prosody(name)
        elif 'decoder' in name.lower() and 'generator' not in name.lower():
            new_name = remap_decoder(name)
        elif 'generator' in name.lower() or 'ups' in name.lower():
            new_name = remap_vocoder(name)
        else:
            new_name = name  # Keep original for unmapped

        if new_name:
            remapped[new_name] = arr
        else:
            unmapped.append(name)

    if unmapped:
        print(f"\nUnmapped tensors ({len(unmapped)}):")
        for n in sorted(unmapped):
            t = tensors[n]
            print(f"  {n}: {t.shape} {t.dtype}")

    return remapped


def remap_plbert(name):
    """Remap PL-BERT tensor names."""
    # These will be model-specific; return original for now
    replacements = {
        'word_embeddings': 'plbert.embed.word.weight',
        'position_embeddings': 'plbert.embed.pos.weight',
        'token_type_embeddings': 'plbert.embed.type.weight',
        'embedding_hidden_mapping_in': 'plbert.hidden_map',
    }
    for key, val in replacements.items():
        if key in name:
            if 'weight' in name:
                return val + '.weight' if '.weight' not in val else val
            elif 'bias' in name:
                return val + '.bias'
    return name


def remap_text_encoder(name):
    """Remap text encoder tensor names."""
    return name.replace('kmodel.text_encoder.', 'text_enc.')


def remap_prosody(name):
    """Remap prosody predictor tensor names."""
    return name.replace('kmodel.predictor.', 'prosody.')


def remap_decoder(name):
    """Remap acoustic decoder tensor names."""
    return name.replace('kmodel.decoder.', 'decoder.')


def remap_vocoder(name):
    """Remap vocoder tensor names."""
    return name.replace('kmodel.decoder.generator.', 'vocoder.')


def convert_voices(npz_path, tensors):
    """Convert NPZ voice embeddings to safetensors-compatible tensors."""
    if not npz_path or not os.path.exists(npz_path):
        print("No voices.npz found, skipping voice embeddings")
        return

    print(f"Loading voices from {npz_path}...")
    voices = np.load(npz_path)

    voice_names = [
        'expr-voice-2-f', 'expr-voice-2-m',  # Bella, Jasper
        'expr-voice-3-f', 'expr-voice-3-m',  # Luna, Bruno
        'expr-voice-4-f', 'expr-voice-4-m',  # Rosie, Hugo
        'expr-voice-5-f', 'expr-voice-5-m',  # Kiki, Leo
    ]

    for i, vname in enumerate(voice_names):
        if vname in voices:
            arr = voices[vname].astype(np.float32)
            # Use first row (short text style) as the default embedding
            if arr.ndim == 2:
                arr = arr[0:1]  # [1, 256]
            tensors[f'styles.{i}'] = arr.flatten()
            print(f"  Style {i}: {vname} -> styles.{i} {arr.shape}")
        else:
            print(f"  Style {i}: {vname} not found in NPZ")


def export_phoneme_vocab(output_dir):
    """Export KittenTTS phoneme vocabulary as JSON."""
    # KittenTTS vocabulary (178 tokens)
    # This is extracted from the KittenTTS source code
    vocab = {
        '$': 0, ';': 1, ':': 2, ',': 3, '.': 4, '!': 5, '?': 6,
        '\u2014': 7, '\u2026': 10, '"': 11, '(': 12, ')': 13, '\u201c': 14,
        '\u201d': 15, ' ': 16, '\u0250': 17, '\u0251': 18, '\u0252': 19,
        '\u0253': 20, '\u0254': 21, '\u0255': 22, '\u0256': 23, '\u0257': 24,
        '\u0258': 25, '\u0259': 26, '\u025a': 27, '\u025b': 28, '\u025c': 29,
        '\u025d': 30, '\u025e': 31, '\u025f': 32, '\u0260': 33, '\u0261': 34,
        '\u0262': 35, '\u0263': 36, '\u0264': 37, '\u0265': 38, '\u0266': 39,
        '\u0267': 40, '\u0268': 41, '\u0269': 42, '\u026a': 43, '\u026b': 44,
        '\u026c': 45, '\u026d': 46, '\u026e': 47, '\u026f': 48, '\u0270': 49,
        '\u0271': 50, '\u0272': 51, '\u0273': 52, '\u0274': 53, '\u0275': 54,
        '\u0276': 55, '\u0278': 56, '\u0279': 57, '\u027a': 58, '\u027b': 59,
        '\u027d': 60, '\u027e': 61, '\u0280': 62, '\u0281': 63, '\u0282': 64,
        '\u0283': 65, '\u0284': 66, '\u0288': 67, '\u0289': 68, '\u028a': 69,
        '\u028b': 70, '\u028c': 71, '\u028d': 72, '\u028e': 73, '\u028f': 74,
        '\u0290': 75, '\u0291': 76, '\u0292': 77, '\u0294': 78, '\u0295': 79,
        '\u0298': 80, '\u0299': 81, '\u029b': 82, '\u029c': 83, '\u029d': 84,
        '\u029f': 85, '\u02a1': 86, '\u02a2': 87, '\u02b0': 88, '\u02b2': 89,
        '\u02b7': 90, '\u02c8': 91, '\u02cc': 92, '\u02d0': 93, '\u02d1': 94,
        '\u02de': 95, '\u02e0': 96, '\u02e4': 97, '\u0300': 98, '\u0301': 99,
        '\u0302': 100, '\u0303': 101, '\u0304': 102, '\u0305': 103,
        '\u0306': 104, '\u0308': 105, '\u030a': 106, '\u030b': 107,
        '\u030c': 108, '\u030d': 109, '\u030f': 110, '\u0318': 111,
        '\u0319': 112, '\u031a': 113, '\u031c': 114, '\u031d': 115,
        '\u031e': 116, '\u031f': 117, '\u0320': 118, '\u0324': 119,
        '\u0325': 120, '\u032a': 121, '\u032c': 122, '\u032f': 123,
        '\u0330': 124, '\u0334': 125, '\u0339': 126, '\u033a': 127,
        '\u033b': 128, '\u033c': 129, '\u033d': 130, '\u0361': 131,
        'a': 132, 'b': 133, 'c': 134, 'd': 135, 'e': 136, 'f': 137,
        'g': 138, 'h': 139, 'i': 140, 'j': 141, 'k': 142, 'l': 143,
        'm': 144, 'n': 145, 'o': 146, 'p': 147, 'q': 148, 'r': 149,
        's': 150, 't': 151, 'u': 152, 'v': 153, 'w': 154, 'x': 155,
        'y': 156, 'z': 157, 'A': 158, 'B': 159, 'C': 160, 'D': 161,
        'E': 162, 'F': 163, 'G': 164, 'H': 165, 'I': 166, 'J': 167,
        'K': 168, 'L': 169, 'M': 170, 'N': 171, 'O': 172, 'P': 173,
        'Q': 174, 'R': 175, 'S': 176, 'T': 177,
    }

    vocab_path = os.path.join(output_dir, 'phoneme_vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Exported phoneme vocabulary ({len(vocab)} tokens) to {vocab_path}")


def save_safetensors(tensors, output_path):
    """Save tensors as a safetensors file."""
    from safetensors.numpy import save_file

    # Convert all tensors to float32
    f32_tensors = {}
    for name, arr in tensors.items():
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        f32_tensors[name] = arr

    save_file(f32_tensors, output_path)

    total_bytes = sum(arr.nbytes for arr in f32_tensors.values())
    print(f"Saved {len(f32_tensors)} tensors ({total_bytes / 1024 / 1024:.1f} MB) to {output_path}")


def inspect_onnx(onnx_path):
    """Print all tensor names and shapes from an ONNX model (for debugging)."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph

    print(f"\nONNX Model: {onnx_path}")
    print(f"IR Version: {model.ir_version}")
    print(f"Opset: {[op.version for op in model.opset_import]}")

    print(f"\nInputs ({len(graph.input)}):")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")

    print(f"\nOutputs ({len(graph.output)}):")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    print(f"\nInitializers ({len(graph.initializer)}):")
    for init in sorted(graph.initializer, key=lambda x: x.name):
        arr = numpy_helper.to_array(init)
        print(f"  {init.name}: {list(arr.shape)} {arr.dtype}")

    # Count unique op types
    ops = set()
    for node in graph.node:
        ops.add(node.op_type)
    print(f"\nOp types ({len(ops)}): {sorted(ops)}")


def main():
    parser = argparse.ArgumentParser(description='Convert KittenTTS ONNX to safetensors')
    parser.add_argument('--onnx', required=True, help='Path to ONNX model file')
    parser.add_argument('--voices', help='Path to voices.npz file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--inspect', action='store_true', help='Just inspect the ONNX model')
    args = parser.parse_args()

    if args.inspect:
        inspect_onnx(args.onnx)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and remap tensors
    tensors, model = load_onnx_initializers(args.onnx)

    # Print all tensor names for mapping development
    print("\nAll tensor names and shapes:")
    for name in sorted(tensors.keys()):
        arr = tensors[name]
        print(f"  {name}: {list(arr.shape)} {arr.dtype}")

    remapped = remap_tensor_names(tensors)

    # Add voice embeddings
    convert_voices(args.voices, remapped)

    # Save safetensors
    st_path = os.path.join(args.output_dir, 'model.safetensors')
    save_safetensors(remapped, st_path)

    # Export phoneme vocabulary
    export_phoneme_vocab(args.output_dir)

    print(f"\nConversion complete! Model saved to {args.output_dir}/")
    print(f"To use: ./kittentts -d {args.output_dir} \"Hello world\"")


if __name__ == '__main__':
    main()
