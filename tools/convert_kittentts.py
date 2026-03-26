#!/usr/bin/env python3
"""
convert_kittentts.py - Convert KittenTTS ONNX model to safetensors format

Extracts all initializer tensors from the ONNX model and saves them as
a single safetensors file with names remapped to match the C inference engine.

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


def load_onnx_weights(onnx_path):
    """Load all initializer tensors from an ONNX model."""
    import onnx
    from onnx import numpy_helper

    print(f"Loading ONNX model from {onnx_path}...")
    model = onnx.load(onnx_path)
    graph = model.graph

    tensors = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.size > 1:  # skip scalar constants
            tensors[init.name] = arr

    print(f"Found {len(tensors)} tensors (excluding scalars)")
    return tensors


def remap_all(tensors):
    """
    Remap ONNX tensor names to clean hierarchy for the C engine.

    ONNX naming conventions in KittenTTS:
    - Named weights: kmodel.bert.*, kmodel.text_encoder.*, kmodel.predictor.*,
                     kmodel.decoder.*, kmodel.decoder.generator.*
    - Anonymous weights: onnx::MatMul_NNNN, onnx::LSTM_NNNN
    """
    out = {}

    # ========================================================================
    # PL-BERT embeddings
    # ========================================================================
    direct_map = {
        'kmodel.bert.embeddings.word_embeddings.weight': 'plbert.embed.word.weight',
        'kmodel.bert.embeddings.position_embeddings.weight': 'plbert.embed.pos.weight',
        'kmodel.bert.embeddings.token_type_embeddings.weight': 'plbert.embed.type.weight',
        'kmodel.bert.embeddings.LayerNorm.weight': 'plbert.embed.ln.weight',
        'kmodel.bert.embeddings.LayerNorm.bias': 'plbert.embed.ln.bias',
        'kmodel.bert.encoder.embedding_hidden_mapping_in.bias': 'plbert.hidden_map.bias',
        # ALBERT shared layer
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight': 'plbert.layer.attn_ln.weight',
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias': 'plbert.layer.attn_ln.bias',
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias': 'plbert.layer.attn.q.bias',
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias': 'plbert.layer.attn.k.bias',
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias': 'plbert.layer.attn.v.bias',
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias': 'plbert.layer.attn.o.bias',
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias': 'plbert.layer.ffn.up.bias',
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias': 'plbert.layer.ffn.down.bias',
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight': 'plbert.layer.ffn_ln.weight',
        'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias': 'plbert.layer.ffn_ln.bias',
        'kmodel.bert_encoder.bias': 'plbert.out_proj.bias',
    }

    # Anonymous MatMul weights for BERT (identified by shape)
    # From the ONNX inspection:
    #   onnx::MatMul_5661: [128, 768]  -> hidden_map.weight (128->768 is transposed in ONNX)
    #   onnx::MatMul_5662: [768, 768]  -> attn.q.weight
    #   onnx::MatMul_5665: [768, 768]  -> attn.k.weight
    #   onnx::MatMul_5668: [768, 768]  -> attn.v.weight
    #   onnx::MatMul_5672: [768, 768]  -> attn.o.weight
    #   onnx::MatMul_5673: [768, 2048] -> ffn.up.weight
    #   onnx::MatMul_5674: [2048, 768] -> ffn.down.weight
    #   onnx::MatMul_5818: [768, 128]  -> out_proj.weight
    #   onnx::MatMul_5973: [128, 50]   -> duration_proj.weight

    # We need to identify them by their numeric suffix ordering
    matmul_map = {
        '5661': ('plbert.hidden_map.weight', True),    # [128, 768] -> transpose to [768, 128]
        '5662': ('plbert.layer.attn.q.weight', True),  # [768, 768] -> transpose
        '5665': ('plbert.layer.attn.k.weight', True),
        '5668': ('plbert.layer.attn.v.weight', True),
        '5672': ('plbert.layer.attn.o.weight', True),
        '5673': ('plbert.layer.ffn.up.weight', True),  # [768, 2048] -> transpose to [2048, 768]
        '5674': ('plbert.layer.ffn.down.weight', True), # [2048, 768] -> transpose to [768, 2048]
        '5818': ('plbert.out_proj.weight', True),       # [768, 128] -> transpose to [128, 768]
        '5973': ('prosody.dur_proj.weight', True),      # [128, 50] -> transpose to [50, 128]
    }

    # ========================================================================
    # Text encoder
    # ========================================================================
    text_enc_map = {
        'kmodel.text_encoder.embedding.weight': 'text_enc.embed.weight',
        'kmodel.text_encoder.cnn.0.0.weight': 'text_enc.conv.0.weight',
        'kmodel.text_encoder.cnn.0.0.bias': 'text_enc.conv.0.bias',
        'kmodel.text_encoder.cnn.0.1.gamma': 'text_enc.conv_ln.0.weight',
        'kmodel.text_encoder.cnn.0.1.beta': 'text_enc.conv_ln.0.bias',
        'kmodel.text_encoder.cnn.1.0.weight': 'text_enc.conv.1.weight',
        'kmodel.text_encoder.cnn.1.0.bias': 'text_enc.conv.1.bias',
        'kmodel.text_encoder.cnn.1.1.gamma': 'text_enc.conv_ln.1.weight',
        'kmodel.text_encoder.cnn.1.1.beta': 'text_enc.conv_ln.1.bias',
    }
    direct_map.update(text_enc_map)

    # ========================================================================
    # Prosody predictor
    # ========================================================================
    prosody_map = {
        'kmodel.predictor.duration_proj.linear_layer.bias': 'prosody.dur_proj.bias',
        'kmodel.predictor.text_encoder.lstms.1.fc.weight': 'prosody.dur_ada.1.weight',
        'kmodel.predictor.text_encoder.lstms.1.fc.bias': 'prosody.dur_ada.1.bias',
        'kmodel.predictor.text_encoder.lstms.3.fc.weight': 'prosody.dur_ada.3.weight',
        'kmodel.predictor.text_encoder.lstms.3.fc.bias': 'prosody.dur_ada.3.bias',
    }
    direct_map.update(prosody_map)

    # F0 and N predictor blocks
    for prefix, out_prefix in [('F0', 'f0'), ('N', 'n')]:
        for i in range(3):
            src = f'kmodel.predictor.{prefix}.{i}'
            dst = f'prosody.{out_prefix}.{i}'
            for suffix in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
                          'conv1x1.weight',
                          'norm1.fc.weight', 'norm1.fc.bias',
                          'norm1.norm.weight', 'norm1.norm.bias',
                          'norm2.fc.weight', 'norm2.fc.bias',
                          'norm2.norm.weight', 'norm2.norm.bias',
                          'pool.weight', 'pool.bias']:
                key = f'{src}.{suffix}'
                if key in tensors:
                    direct_map[key] = f'{dst}.{suffix}'

        # Projections
        src = f'kmodel.predictor.{prefix}_proj.weight'
        if src in tensors:
            direct_map[src] = f'prosody.{out_prefix}.proj.weight'

    # ========================================================================
    # Acoustic decoder
    # ========================================================================
    dec_map = {
        'kmodel.decoder.asr_res.0.weight': 'decoder.adapter.weight',
        'kmodel.decoder.asr_res.0.bias': 'decoder.adapter.bias',
    }
    direct_map.update(dec_map)

    # Encode block
    for suffix in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
                   'conv1x1.weight',
                   'norm1.fc.weight', 'norm1.fc.bias', 'norm1.norm.weight', 'norm1.norm.bias',
                   'norm2.fc.weight', 'norm2.fc.bias', 'norm2.norm.weight', 'norm2.norm.bias']:
        key = f'kmodel.decoder.encode.{suffix}'
        if key in tensors:
            direct_map[key] = f'decoder.encode.{suffix}'

    # Decode blocks
    for i in range(4):
        for suffix in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
                       'conv1x1.weight',
                       'norm1.fc.weight', 'norm1.fc.bias', 'norm1.norm.weight', 'norm1.norm.bias',
                       'norm2.fc.weight', 'norm2.fc.bias', 'norm2.norm.weight', 'norm2.norm.bias',
                       'pool.weight', 'pool.bias']:
            key = f'kmodel.decoder.decode.{i}.{suffix}'
            if key in tensors:
                direct_map[key] = f'decoder.decode.{i}.{suffix}'

    # ========================================================================
    # Vocoder (generator)
    # ========================================================================
    gen = 'kmodel.decoder.generator'
    direct_map[f'{gen}.ups.0.weight'] = 'vocoder.ups.0.weight'
    direct_map[f'{gen}.ups.0.bias'] = 'vocoder.ups.0.bias'
    direct_map[f'{gen}.ups.1.weight'] = 'vocoder.ups.1.weight'
    direct_map[f'{gen}.ups.1.bias'] = 'vocoder.ups.1.bias'
    direct_map[f'{gen}.conv_post.weight'] = 'vocoder.conv_post.weight'
    direct_map[f'{gen}.conv_post.bias'] = 'vocoder.conv_post.bias'
    direct_map[f'{gen}.noise_convs.0.weight'] = 'vocoder.noise_convs.0.weight'
    direct_map[f'{gen}.noise_convs.0.bias'] = 'vocoder.noise_convs.0.bias'
    direct_map[f'{gen}.noise_convs.1.weight'] = 'vocoder.noise_convs.1.weight'
    direct_map[f'{gen}.noise_convs.1.bias'] = 'vocoder.noise_convs.1.bias'

    # iSTFT weights
    for s in ['weight_backward_imag', 'weight_backward_real', 'weight_forward_imag', 'weight_forward_real']:
        key = f'{gen}.stft.{s}'
        if key in tensors:
            direct_map[key] = f'vocoder.stft.{s}'

    # ResBlocks and noise_res
    for block_type in ['resblocks', 'noise_res']:
        n_blocks = 4 if block_type == 'resblocks' else 2
        for i in range(n_blocks):
            base = f'{gen}.{block_type}.{i}'
            dst = f'vocoder.{block_type}.{i}'
            for sub in ['convs1', 'convs2']:
                for j in range(3):
                    for s in ['weight', 'bias']:
                        key = f'{base}.{sub}.{j}.{s}'
                        if key in tensors:
                            direct_map[key] = f'{dst}.{sub}.{j}.{s}'
            for sub in ['adain1', 'adain2']:
                for j in range(3):
                    for s in ['fc.weight', 'fc.bias', 'norm.weight', 'norm.bias']:
                        key = f'{base}.{sub}.{j}.{s}'
                        if key in tensors:
                            direct_map[key] = f'{dst}.{sub}.{j}.{s}'
            for sub in ['alpha1', 'alpha2']:
                for j in range(3):
                    key = f'{base}.{sub}.{j}'
                    if key in tensors:
                        direct_map[key] = f'{dst}.{sub}.{j}'

    # ========================================================================
    # Apply direct mappings
    # ========================================================================
    mapped = set()
    for src, dst in direct_map.items():
        if src in tensors:
            out[dst] = tensors[src]
            mapped.add(src)

    # ========================================================================
    # Handle anonymous MatMul weights
    # ========================================================================
    for name, arr in tensors.items():
        if name.startswith('onnx::MatMul_'):
            suffix = name.split('_')[-1]
            if suffix in matmul_map:
                dst_name, transpose = matmul_map[suffix]
                if transpose:
                    out[dst_name] = arr.T.copy()
                else:
                    out[dst_name] = arr
                mapped.add(name)

    # ========================================================================
    # Handle LSTM weights (bidirectional format: [2, 4*H, input_dim])
    # ========================================================================
    # ONNX LSTM format: W[2, 4*H, input], R[2, 4*H, H], B[2, 4*2*H]
    # Direction 0 = forward, direction 1 = backward
    # Gate order in ONNX: i, o, f, c (IOFC)
    # We need: i, f, c, o (IFCO) for standard LSTM cell

    lstm_tensors = {}
    for name, arr in tensors.items():
        if name.startswith('onnx::LSTM_'):
            lstm_tensors[name] = arr

    # Sort by numeric suffix to match the 5 LSTMs in order:
    # LSTM 0: text encoder BiLSTM (input=128, hidden=64)
    # LSTM 1-4: prosody duration encoder BiLSTMs
    lstm_keys = sorted(lstm_tensors.keys(), key=lambda x: int(x.split('_')[-1]))

    # Group into triplets (B, W, R per LSTM)
    lstm_groups = []
    i = 0
    while i < len(lstm_keys):
        # Each LSTM has 3 tensors: B (bias), W (input weights), R (recurrent weights)
        group = []
        for j in range(3):
            if i + j < len(lstm_keys):
                group.append(lstm_keys[i + j])
        lstm_groups.append(group)
        i += 3

    def onnx_to_ifco(gates_iofc, hidden_dim):
        """Reorder ONNX LSTM gates from IOFC to IFCO (standard PyTorch order)."""
        # ONNX: i, o, f, c -> need: i, f, g(c), o
        # Actually let's just keep IOFC order and adjust the C code, or
        # reorder to IFGO: i[0:H], f[2H:3H], g[H:2H], o[3H:4H]
        H = hidden_dim
        # ONNX layout: gates[0:H]=i, gates[H:2H]=o, gates[2H:3H]=f, gates[3H:4H]=c
        i_g = gates_iofc[0:H]
        o_g = gates_iofc[H:2*H]
        f_g = gates_iofc[2*H:3*H]
        c_g = gates_iofc[3*H:4*H]
        # Standard layout: i, f, g(cell), o
        return np.concatenate([i_g, f_g, c_g, o_g], axis=0)

    def reorder_lstm_weights(W, hidden_dim):
        """Reorder 2D weight matrix gate order from IOFC to IFGO."""
        H = hidden_dim
        i_g = W[0:H]
        o_g = W[H:2*H]
        f_g = W[2*H:3*H]
        c_g = W[3*H:4*H]
        return np.concatenate([i_g, f_g, c_g, o_g], axis=0)

    lstm_names = [
        'text_enc.lstm',          # LSTM 0: text encoder
        'prosody.dur_lstm.0',     # LSTM 1: duration encoder layer 0
        'prosody.dur_lstm.1',     # LSTM 2: duration encoder layer 1
        'prosody.dur_lstm.2',     # LSTM 3: duration encoder layer 2
        'prosody.dur_lstm.3',     # LSTM 4: duration encoder layer 3
    ]

    for idx, group in enumerate(lstm_groups):
        if idx >= len(lstm_names):
            break
        name_prefix = lstm_names[idx]

        # Identify B (bias), W (input weight), R (recurrent weight) by shape
        B = W = R = None
        for key in group:
            arr = lstm_tensors[key]
            if arr.ndim == 2 and arr.shape[0] == 2:
                # [2, N] -> bias
                B = arr
            elif arr.ndim == 3 and arr.shape[0] == 2:
                # [2, 4*H, input_dim] or [2, 4*H, H]
                if W is None:
                    W = arr  # First 3D tensor = W (input weights)
                else:
                    R = arr  # Second 3D tensor = R (recurrent weights)

        if B is None or W is None or R is None:
            print(f"  Warning: incomplete LSTM group {idx} ({group})")
            continue

        hidden_dim = R.shape[2]  # R is [2, 4*H, H], so R.shape[2] = H

        # Split forward (dir=0) and backward (dir=1)
        for dir_idx, dir_name in [(0, 'fwd'), (1, 'bwd')]:
            w_ih = reorder_lstm_weights(W[dir_idx], hidden_dim)  # [4*H, input_dim]
            w_hh = reorder_lstm_weights(R[dir_idx], hidden_dim)  # [4*H, H]
            # ONNX bias: [4*2*H] = [W_bias(4H) || R_bias(4H)]
            b_all = B[dir_idx]  # [8*H] or [4*2*H]
            half = len(b_all) // 2
            b_ih = onnx_to_ifco(b_all[:half], hidden_dim)
            b_hh = onnx_to_ifco(b_all[half:], hidden_dim)

            out[f'{name_prefix}.W_ih_{dir_name}'] = w_ih.astype(np.float32)
            out[f'{name_prefix}.W_hh_{dir_name}'] = w_hh.astype(np.float32)
            out[f'{name_prefix}.b_ih_{dir_name}'] = b_ih.astype(np.float32)
            out[f'{name_prefix}.b_hh_{dir_name}'] = b_hh.astype(np.float32)

        for key in group:
            mapped.add(key)

    # ========================================================================
    # Report unmapped
    # ========================================================================
    unmapped = []
    for name in sorted(tensors.keys()):
        if name not in mapped:
            arr = tensors[name]
            # Skip ONNX graph constants (small, or with path-like names starting with /)
            if name.startswith('/') or arr.size <= 10:
                continue
            unmapped.append(name)

    if unmapped:
        print(f"\nUnmapped model tensors ({len(unmapped)}):")
        for n in unmapped:
            print(f"  {n}: {list(tensors[n].shape)} {tensors[n].dtype}")

    print(f"\nMapped {len(out)} tensors total")
    return out


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
            if arr.ndim == 2:
                arr = arr[0]  # Use first row (short-text style)
            tensors[f'styles.{i}'] = arr
            print(f"  Style {i}: {vname} ({arr.shape})")
        else:
            # Try alternate naming
            for key in voices.keys():
                print(f"  Available voice: {key}")


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

    f32_tensors = {}
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        elif arr.dtype != np.float32 and arr.dtype != np.int32:
            arr = arr.astype(np.float32)
        f32_tensors[name] = arr

    save_file(f32_tensors, output_path)

    total_bytes = sum(arr.nbytes for arr in f32_tensors.values())
    print(f"Saved {len(f32_tensors)} tensors ({total_bytes / 1024 / 1024:.1f} MB) to {output_path}")


def inspect_onnx(onnx_path):
    """Print all tensor names and shapes from an ONNX model."""
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

    ops = set()
    for node in graph.node:
        ops.add(node.op_type)
    print(f"\nOp types ({len(ops)}): {sorted(ops)}")


def main():
    parser = argparse.ArgumentParser(description='Convert KittenTTS ONNX to safetensors')
    parser.add_argument('--onnx', required=True, help='Path to ONNX model file')
    parser.add_argument('--voices', help='Path to voices.npz file')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--inspect', action='store_true', help='Just inspect the ONNX model')
    args = parser.parse_args()

    if args.inspect:
        inspect_onnx(args.onnx)
        return

    if not args.output_dir:
        parser.error("--output-dir is required for conversion")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load, remap, and save
    tensors = load_onnx_weights(args.onnx)
    remapped = remap_all(tensors)

    # Add voice embeddings
    if args.voices:
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
