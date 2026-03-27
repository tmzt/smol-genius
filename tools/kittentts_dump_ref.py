#!/usr/bin/env python3
"""
kittentts_dump_ref.py - Dump ONNX reference activations for C validation

Runs the KittenTTS ONNX model and saves intermediate tensor shapes and
key architectural details needed to fix the C forward pass.

Usage:
    python3 tools/kittentts_dump_ref.py --onnx .kittentts_build/kitten_tts_nano.onnx
"""

import argparse
import json
import numpy as np
import onnx
from onnx import numpy_helper


def analyze_architecture(model_path):
    """Extract the complete computation flow from ONNX graph."""
    model = onnx.load(model_path)
    graph = model.graph

    print("=" * 70)
    print("KittenTTS ONNX Architecture Analysis")
    print("=" * 70)

    # Categorize all ConvTranspose nodes (these are the upsample/pool layers)
    print("\n=== ConvTranspose (upsample/pool) nodes ===")
    for node in graph.node:
        if node.op_type == 'ConvTranspose':
            attrs = {}
            for a in node.attribute:
                if a.ints:
                    attrs[a.name] = list(a.ints)
                elif a.type == 2:
                    attrs[a.name] = a.i
            stride = attrs.get('strides', [1])[0]
            kernel = attrs.get('kernel_shape', [1])[0]
            pads = attrs.get('pads', [0, 0])
            group = attrs.get('group', 1)
            opad = attrs.get('output_padding', [0])
            if isinstance(opad, list):
                opad = opad[0] if opad else 0
            # Find weight shape
            w_name = node.input[1] if len(node.input) > 1 else '?'
            w_shape = '?'
            for init in graph.initializer:
                if init.name == w_name:
                    w_shape = list(numpy_helper.to_array(init).shape)
                    break
            print(f"  {node.name}")
            print(f"    stride={stride} kernel={kernel} pads={pads} group={group} opad={opad}")
            print(f"    weight: {w_name} {w_shape}")

    # Categorize Conv nodes with their dilations
    print("\n=== Conv nodes with dilation > 1 ===")
    for node in graph.node:
        if node.op_type == 'Conv':
            attrs = {}
            for a in node.attribute:
                if a.ints:
                    attrs[a.name] = list(a.ints)
            dilation = attrs.get('dilations', [1])
            if dilation != [1]:
                print(f"  {node.name}: dilation={dilation}")

    # Count LSTM nodes
    lstm_count = sum(1 for n in graph.node if n.op_type == 'LSTM')
    print(f"\n=== LSTM nodes: {lstm_count} ===")
    for node in graph.node:
        if node.op_type == 'LSTM':
            attrs = {}
            for a in node.attribute:
                if a.name == 'hidden_size':
                    attrs['hidden_size'] = a.i
                elif a.name == 'direction':
                    attrs['direction'] = a.s.decode()
            print(f"  {node.name}: {attrs}")

    # Find the ALBERT layer iteration count
    print("\n=== ALBERT iteration detection ===")
    loop_nodes = [n for n in graph.node if n.op_type == 'Loop']
    for node in loop_nodes:
        # Check if this loop is in the BERT path
        if 'bert' in str(node.input).lower() or 'albert' in node.name.lower():
            # The trip count is an input
            trip_input = node.input[0]
            for init in graph.initializer:
                if init.name == trip_input:
                    val = numpy_helper.to_array(init)
                    print(f"  BERT Loop trip count: {val}")
            print(f"  Loop node: {node.name}, inputs: {list(node.input)[:3]}")

    # Check all Loop nodes for ALBERT
    for node in graph.node:
        if node.op_type == 'Loop':
            trip = node.input[0] if node.input else '?'
            for init in graph.initializer:
                if init.name == trip:
                    val = numpy_helper.to_array(init)
                    print(f"  Loop '{node.name}': trip_count = {val}")


def run_reference(model_path, vocab_path):
    """Run ONNX model and dump reference outputs."""
    import onnxruntime as ort

    with open(vocab_path) as f:
        vocab = json.load(f)

    sess = ort.InferenceSession(model_path)

    # Test input: "hello" in IPA tokens
    input_ids = np.array([[0, 139, 26, 143, 91, 146, 69, 10, 0]], dtype=np.int64)
    style = np.zeros((1, 256), dtype=np.float32)
    speed = np.array([1.0], dtype=np.float32)

    print("\n" + "=" * 70)
    print("Reference Inference")
    print("=" * 70)
    print(f"Input IDs: {input_ids[0].tolist()} (length={input_ids.shape[1]})")

    outputs = sess.run(None, {
        'input_ids': input_ids,
        'style': style,
        'speed': speed,
    })

    waveform = outputs[0]
    duration = outputs[1]

    print(f"\nDuration per phoneme: {duration.tolist()}")
    print(f"Duration sum (acoustic frames): {duration.sum()}")
    print(f"Waveform samples: {len(waveform)}")
    print(f"Waveform duration: {len(waveform)/24000:.2f}s")
    print(f"Upsample ratio: {len(waveform)/duration.sum():.1f}x")
    print(f"Waveform range: [{waveform.min():.4f}, {waveform.max():.4f}]")
    print(f"Waveform RMS: {np.sqrt(np.mean(waveform**2)):.4f}")

    # Compute expected intermediate dimensions
    D = int(duration.sum())
    print(f"\n=== Expected intermediate dimensions ===")
    print(f"PL-BERT output:     [{input_ids.shape[1]}, 128]")
    print(f"Text encoder out:   [{input_ids.shape[1]}, 128]")
    print(f"Duration expanded:  [{D}, 128]")
    print(f"Acoustic decoder:")
    print(f"  adapter:          [{D}, 64]  (from [{D}, 128] via Conv1D 1x1)")
    print(f"  encode input:     [130, {D}] (adapter[64] + text[64] + F0[1] + N[1])")
    print(f"  encode output:    [256, {D}]")
    print(f"  decode.0-2:       [256, {D}] (input: [322, {D}])")
    print(f"  decode.3 + pool:  [256, {D}] -> [322, {D}] -> pool 2x -> [322, {D*2}]")
    print(f"  post-pool:        [256, {D*2}]  (= {D*2} frames)")
    print(f"Vocoder:")
    L1 = (D*2 - 1) * 10 - 2*5 + 20
    L2 = (L1 - 1) * 6 - 2*3 + 12
    print(f"  ups.0 (s=10):     [128, {L1}]")
    print(f"  ups.1 (s=6):      [64, {L2}]")
    print(f"  conv_post:        [22, {L2}]  (11 mag + 11 phase)")
    print(f"  iSTFT (learned):  [{(L2-1)*5+20}] samples (ConvTranspose s=5)")
    print(f"  After trim:       ~{len(waveform)} samples")


def main():
    parser = argparse.ArgumentParser(description='Dump ONNX reference activations')
    parser.add_argument('--onnx', default='.kittentts_build/kitten_tts_nano.onnx')
    parser.add_argument('--vocab', default='kitten-tts-nano/phoneme_vocab.json')
    args = parser.parse_args()

    analyze_architecture(args.onnx)
    run_reference(args.onnx, args.vocab)


if __name__ == '__main__':
    main()
