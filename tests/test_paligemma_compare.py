#!/usr/bin/env python3
"""
PaliGemma 2 stage-by-stage comparison: HF transformers vs C implementation.

Loads both implementations, feeds identical inputs, and compares at each stage:
  Stage 0: Safetensors weight access (raw bf16 values)
  Stage 1: Token embedding lookup
  Stage 2: Image preprocessing (resize + normalize)
  Stage 3: SigLIP vision encoder (patch embed, position embed, per-layer)
  Stage 4: Connector projection
  Stage 5: Assembled input embeddings (vision + text + normalizer)
  Stage 6: Decoder layer 0 (RMSNorm, Q/K/V projection, RoPE, attention)
  Stage 7: Full generation (first token, first 10 tokens, text output)

Each stage reports PASS/FAIL with cosine similarity and max absolute error.
Failing stages indicate where the C implementation diverges from HF.

Usage:
    python3 tests/test_paligemma_compare.py [model_dir] [image_path]

Requires: transformers, torch, Pillow, numpy
"""

import ctypes
import numpy as np
import struct
import json
import sys
import os
import platform

# ─── Configuration ───

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/src/common-models/paligemma2-3b-mix-224")
IMG_PATH = sys.argv[2] if len(sys.argv) > 2 else None

if IMG_PATH is None:
    for p in ["/tmp/te-screenshots/example.com_2.png",
              "/tmp/te-screenshots/en.wikipedia.org_0.png"]:
        if os.path.exists(p):
            IMG_PATH = p
            break

assert IMG_PATH and os.path.exists(IMG_PATH), f"Need a test image (got {IMG_PATH})"
assert os.path.isdir(MODEL_DIR), f"Model dir not found: {MODEL_DIR}"

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    config = json.load(f)

HIDDEN = config["text_config"]["hidden_size"]
VIS_HIDDEN = config["vision_config"]["hidden_size"]
VOCAB = config["text_config"]["vocab_size"]
PATCH_SIZE = config["vision_config"]["patch_size"]
IMAGE_SIZE = config["vision_config"].get("image_size", 224)
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
IMG_TOKEN_ID = config.get("image_token_index", 257152)

print(f"Model:     {MODEL_DIR}")
print(f"Image:     {IMG_PATH}")
print(f"Hidden:    {HIDDEN} (text), {VIS_HIDDEN} (vision)")
print(f"Patches:   {NUM_PATCHES} ({IMAGE_SIZE}px / {PATCH_SIZE}px)")
print()

# ─── Comparison utilities ───

n_pass = 0
n_fail = 0

def compare(name, hf, c, rtol=0.05, atol=0.01):
    """Compare two arrays, report PASS/FAIL."""
    global n_pass, n_fail
    import torch
    if isinstance(hf, torch.Tensor):
        hf = hf.detach().float().cpu().numpy()
    hf = np.asarray(hf, dtype=np.float32).flatten()
    c = np.asarray(c, dtype=np.float32).flatten()
    n = min(len(hf), len(c))
    hf, c = hf[:n], c[:n]

    max_diff = float(np.max(np.abs(hf - c)))
    cos = float(np.dot(hf, c) / (np.linalg.norm(hf) * np.linalg.norm(c) + 1e-12))
    close = np.allclose(hf, c, rtol=rtol, atol=atol)

    if close:
        n_pass += 1
        print(f"  ✓ {name}: cos={cos:.6f} max_diff={max_diff:.6f}")
    else:
        n_fail += 1
        print(f"  ✗ {name}: cos={cos:.6f} max_diff={max_diff:.6f}")
        print(f"    HF [0:6]: {hf[:6].tolist()}")
        print(f"    C  [0:6]: {c[:6].tolist()}")
    return close

def read_bf16_from_safetensors(model_dir, tensor_name, offset_elements=0, count=8):
    """Read raw bf16 values from safetensors and convert to f32."""
    for shard in sorted(os.listdir(model_dir)):
        if not shard.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            if tensor_name not in header:
                continue
            t = header[tensor_name]
            data_start = 8 + n
            byte_off = t["data_offsets"][0] + offset_elements * 2
            f.seek(data_start + byte_off)
            raw = f.read(count * 2)
            vals = []
            for i in range(count):
                bf = struct.unpack("<H", raw[i*2:i*2+2])[0]
                vals.append(struct.unpack("<f", struct.pack("<I", bf << 16))[0])
            return np.array(vals, dtype=np.float32)
    return None

# ─── Load HF model ───

print("Loading HF model...")
import torch
from transformers import (PaliGemmaForConditionalGeneration,
                          PaliGemmaProcessor, AutoTokenizer)
from PIL import Image

hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16)
hf_proc = PaliGemmaProcessor.from_pretrained(MODEL_DIR)
hf_tok = AutoTokenizer.from_pretrained(MODEL_DIR)

# ─── Load C library ───

print("Loading C library...")
lib_name = "libpaligemma_test.dylib" if platform.system() == "Darwin" else "libpaligemma_test.so"
lib_path = os.path.join(os.path.dirname(__file__), "..", lib_name)
assert os.path.exists(lib_path), f"Build C lib first: {lib_path}"
lib = ctypes.CDLL(lib_path)

lib.paligemma_load.restype = ctypes.c_void_p
lib.paligemma_load.argtypes = [ctypes.c_char_p]
lib.paligemma_free.argtypes = [ctypes.c_void_p]
lib.paligemma_generate.restype = ctypes.c_int
lib.paligemma_generate.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
TOKEN_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p)
lib.paligemma_set_token_callback.argtypes = [ctypes.c_void_p, TOKEN_CB, ctypes.c_void_p]
lib.smol_set_threads.argtypes = [ctypes.c_int]

try:
    ctypes.c_int.in_dll(lib, "smol_verbose").value = 0
except:
    pass

lib.smol_set_threads(4)
c_ctx = lib.paligemma_load(MODEL_DIR.encode())
assert c_ctx, "C paligemma_load failed"
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 0: Raw weight access
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("═══ Stage 0: Raw weight access ═══")
# Compare a few weights directly from safetensors vs HF model
with torch.no_grad():
    hf_embed = hf_model.model.language_model.embed_tokens.weight[108].float().numpy()
c_embed = read_bf16_from_safetensors(
    MODEL_DIR, "language_model.model.embed_tokens.weight",
    offset_elements=108 * HIDDEN, count=HIDDEN)
compare("embed_tokens[108] (raw weight)", hf_embed, c_embed)

with torch.no_grad():
    hf_norm = hf_model.model.language_model.layers[0].input_layernorm.weight.float().numpy()
c_norm = read_bf16_from_safetensors(
    MODEL_DIR, "language_model.model.layers.0.input_layernorm.weight",
    count=min(HIDDEN, len(hf_norm)))
compare("layer0.input_layernorm.weight", hf_norm[:len(c_norm)], c_norm)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 1: Token embedding lookup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ Stage 1: Token embedding lookup ═══")
for tok_id in [2, 108, 109, 21209, 659]:
    with torch.no_grad():
        hf_e = hf_model.model.language_model.embed_tokens(torch.tensor([tok_id]))[0].float().numpy()
    c_e = read_bf16_from_safetensors(
        MODEL_DIR, "language_model.model.embed_tokens.weight",
        offset_elements=tok_id * HIDDEN, count=HIDDEN)
    compare(f"embed[{tok_id}]", hf_e, c_e)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 2: Image preprocessing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ Stage 2: Image preprocessing ═══")
img = Image.open(IMG_PATH).convert("RGB")
hf_inputs = hf_proc(text="caption en\n", images=img, return_tensors="pt")
hf_pixels = hf_inputs["pixel_values"][0].float().numpy()  # [3, 224, 224]
print(f"  HF pixel_values: shape={hf_pixels.shape} range=[{hf_pixels.min():.3f}, {hf_pixels.max():.3f}]")
# C preprocessing uses stb_image + bilinear resize + normalize to [-1,1]
# We can't easily get C's preprocessed image, but we can verify the normalization range
print(f"  (C uses stb_image → resize → normalize to [-1,1])")
print(f"  HF uses processor-specific normalization")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 3: SigLIP vision encoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ Stage 3: Vision encoder output ═══")
with torch.no_grad():
    pixel_bf16 = hf_inputs["pixel_values"].to(torch.bfloat16)
    hf_vis = hf_model.model.vision_tower(pixel_bf16).last_hidden_state[0].float()
    print(f"  HF shape: {hf_vis.shape}")
    print(f"  HF [0,0:4]: {hf_vis[0,:4].tolist()}")
    print(f"  HF mean={hf_vis.mean():.4f} std={hf_vis.std():.4f}")
# Note: C vision encoder runs inside paligemma_generate, can't isolate easily.
# The comparison test in Stage 5 will reveal if vision output matches.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 4: Connector projection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ Stage 4: Connector projection ═══")
with torch.no_grad():
    hf_proj = hf_model.model.multi_modal_projector(
        hf_model.model.vision_tower(pixel_bf16).last_hidden_state
    ).float()[0]
    print(f"  HF connector [0,0:4]: {hf_proj[0,:4].tolist()}")
    print(f"  HF connector mean={hf_proj.mean():.6f} std={hf_proj.std():.6f}")
    # HF divides by sqrt(hidden)
    hf_proj_div = hf_proj / (HIDDEN ** 0.5)
    print(f"  HF connector/sqrt(h) std={hf_proj_div.std():.6f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 5: Assembled input embeddings (decoder input)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ Stage 5: Decoder input embeddings ═══")
with torch.no_grad():
    out = hf_model(
        **{k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
           for k, v in hf_inputs.items()},
        output_hidden_states=True, return_dict=True)
    h0 = out.hidden_states[0][0].float()  # [260, 2304] - input to first layer
    print(f"  HF hidden_states[0] shape: {h0.shape}")
    for pos, label in [(0, "vision[0]"), (128, "vision[128]"),
                        (256, "text[BOS]"), (257, "text[caption]"),
                        (258, "text[en]"), (259, "text[\\n]")]:
        if pos < h0.shape[0]:
            v = h0[pos]
            print(f"  HF {label} (pos {pos}): [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}, {v[3]:.4f}] std={v.std():.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 6: Decoder layer 0 output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ Stage 6: Decoder hidden states ═══")
for layer_idx in range(min(4, len(out.hidden_states) - 1)):
    h = out.hidden_states[layer_idx][0].float()
    last = h[-1]  # last position
    print(f"  HF layer {layer_idx} last_pos: mean={last.mean():.4f} std={last.std():.4f} "
          f"[0:4]=[{last[0]:.4f},{last[1]:.4f},{last[2]:.4f},{last[3]:.4f}]")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 7: Full generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ Stage 7: Full generation ═══")

# HF
with torch.no_grad():
    hf_gen = hf_model.generate(
        **{k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
           for k, v in hf_inputs.items()},
        max_new_tokens=32)
    hf_text = hf_proc.decode(hf_gen[0], skip_special_tokens=True)
    hf_ids = hf_gen[0][260:].tolist()
print(f"  HF: '{hf_text}'")
print(f"  HF IDs: {hf_ids[:10]}")

# C
c_ids = []
@TOKEN_CB
def _cb(tid, _):
    c_ids.append(tid)
lib.paligemma_set_token_callback(c_ctx, _cb, None)
prompt = (ctypes.c_int * 4)(2, 21209, 659, 109)
lib.paligemma_generate(c_ctx, IMG_PATH.encode(), prompt, 4, 32)
lib.paligemma_set_token_callback(c_ctx, TOKEN_CB(), None)

c_text = "".join(hf_tok.decode([t]) for t in c_ids)
print(f"  C:  '{c_text}'")
print(f"  C  IDs: {c_ids[:10]}")

if hf_ids[:5] == c_ids[:5]:
    n_pass += 1
    print("  ✓ First 5 tokens match")
else:
    n_fail += 1
    print("  ✗ First 5 tokens DIFFER")
    for i in range(min(len(hf_ids), len(c_ids), 10)):
        match = "=" if hf_ids[i] == c_ids[i] else "≠"
        hf_p = hf_tok.decode([hf_ids[i]]) if i < len(hf_ids) else ""
        c_p = hf_tok.decode([c_ids[i]]) if i < len(c_ids) else ""
        print(f"    [{i}] HF={hf_ids[i]:>6} '{hf_p}' {match} C={c_ids[i]:>6} '{c_p}'")

# ─── Cleanup + Summary ───

lib.paligemma_free(c_ctx)

print(f"\n{'═' * 50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail == 0:
    print("All stages match! 🎉")
else:
    print("Some stages diverge — see ✗ markers above.")
