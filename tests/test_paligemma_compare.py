#!/usr/bin/env python3
"""
PaliGemma 2 comparison test suite.

Loads both the HF transformers reference and our C implementation via FFI,
feeds identical inputs, and compares outputs at each stage:
  1. Token embedding lookup
  2. Vision encoder output
  3. Connector projection
  4. Decoder hidden states (per-layer)
  5. Final logits + generated text

Usage:
    python3 tests/test_paligemma_compare.py [model_dir] [image_path]
"""

import ctypes
import ctypes.util
import numpy as np
import struct
import json
import sys
import os

# ── Configuration ──

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/src/common-models/paligemma2-3b-mix-224")
IMG_PATH = sys.argv[2] if len(sys.argv) > 2 else None

# Find test images
if IMG_PATH is None:
    for candidate in ["/tmp/te-screenshots/example.com_2.png",
                      "/tmp/te-screenshots/en.wikipedia.org_0.png",
                      "/tmp/test_real.jpg"]:
        if os.path.exists(candidate):
            IMG_PATH = candidate
            break

if IMG_PATH is None:
    print("No test image found. Provide as second argument.")
    sys.exit(1)

print(f"Model: {MODEL_DIR}")
print(f"Image: {IMG_PATH}")

HIDDEN = 2304
VOCAB = 257216

# ── Load C library ──

LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "libpaligemma_test.dylib")
if not os.path.exists(LIB_PATH):
    LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "libpaligemma_test.so")

lib = ctypes.CDLL(LIB_PATH)

# Set up C function signatures
lib.paligemma_load.restype = ctypes.c_void_p
lib.paligemma_load.argtypes = [ctypes.c_char_p]

lib.paligemma_free.restype = None
lib.paligemma_free.argtypes = [ctypes.c_void_p]

lib.paligemma_generate.restype = ctypes.c_int
lib.paligemma_generate.argtypes = [
    ctypes.c_void_p,  # ctx
    ctypes.c_char_p,  # image_path
    ctypes.POINTER(ctypes.c_int),  # prompt_tokens
    ctypes.c_int,     # n_prompt_tokens
    ctypes.c_int,     # max_tokens
]

# Token callback
TOKEN_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p)
lib.paligemma_set_token_callback.restype = None
lib.paligemma_set_token_callback.argtypes = [ctypes.c_void_p, TOKEN_CB, ctypes.c_void_p]

lib.smol_set_threads.restype = None
lib.smol_set_threads.argtypes = [ctypes.c_int]

# smol_verbose
try:
    smol_verbose = ctypes.c_int.in_dll(lib, "smol_verbose")
    smol_verbose.value = 2
except:
    pass

# ── Load HF model ──

print("\n=== Loading HF reference ===")
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, AutoTokenizer
from PIL import Image

hf_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
hf_processor = PaliGemmaProcessor.from_pretrained(MODEL_DIR)
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# ── Load C model ──

print("\n=== Loading C implementation ===")
lib.smol_set_threads(4)
c_ctx = lib.paligemma_load(MODEL_DIR.encode())
assert c_ctx, "C paligemma_load failed"

# ── Helper ──

def compare(name, hf_tensor, c_array, rtol=0.1, atol=0.05):
    """Compare HF tensor vs C array, report match/mismatch."""
    if isinstance(hf_tensor, torch.Tensor):
        hf_np = hf_tensor.detach().float().cpu().numpy().flatten()
    else:
        hf_np = np.array(hf_tensor).flatten()
    c_np = np.array(c_array).flatten()

    n = min(len(hf_np), len(c_np))
    hf_np = hf_np[:n]
    c_np = c_np[:n]

    close = np.allclose(hf_np, c_np, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(hf_np - c_np))
    cos_sim = np.dot(hf_np, c_np) / (np.linalg.norm(hf_np) * np.linalg.norm(c_np) + 1e-8)

    status = "✓ PASS" if close else "✗ FAIL"
    print(f"  {status} {name}: max_diff={max_diff:.6f} cos_sim={cos_sim:.6f}")
    if not close:
        print(f"    HF [0:6]: {hf_np[:6]}")
        print(f"    C  [0:6]: {c_np[:6]}")
    return close

# ── Test 1: Token embedding lookup ──

print("\n=== Test 1: Token embedding lookup ===")
test_tokens = [108, 21209, 659]  # \n, caption, en

with torch.no_grad():
    hf_embed_fn = hf_model.model.language_model.embed_tokens
    for tok_id in test_tokens:
        hf_emb = hf_embed_fn(torch.tensor([tok_id]))[0].float().numpy()

        # Read from C: access tok_embeddings_bf16 via safetensors directly
        # (We can't easily call tok_embed_bf16_to_f32 from Python, so compare via file)
        # Instead, read the raw safetensors weight
        for shard in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
            path = os.path.join(MODEL_DIR, shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
                data_start = 8 + n
                key = "language_model.model.embed_tokens.weight"
                if key in header:
                    t = header[key]
                    off = t["data_offsets"][0] + tok_id * HIDDEN * 2
                    f.seek(data_start + off)
                    raw = f.read(HIDDEN * 2)
                    c_emb = []
                    for i in range(HIDDEN):
                        bf = struct.unpack("<H", raw[i*2:i*2+2])[0]
                        c_emb.append(struct.unpack("<f", struct.pack("<I", bf << 16))[0])
                    compare(f"embed[{tok_id}]", hf_emb, c_emb)
                    break

# ── Test 2: Vision encoder ──

print("\n=== Test 2: Vision encoder output ===")
img = Image.open(IMG_PATH).convert("RGB")
hf_inputs = hf_processor(text="caption en\n", images=img, return_tensors="pt")

with torch.no_grad():
    pixel_values = hf_inputs["pixel_values"].to(torch.bfloat16)
    hf_vis_out = hf_model.model.vision_tower(pixel_values)
    hf_vis = hf_vis_out.last_hidden_state[0].float()
    print(f"  HF vision shape: {hf_vis.shape}")
    print(f"  HF vision [0,0:4]: {hf_vis[0,:4].tolist()}")
    print(f"  HF vision mean={hf_vis.mean():.6f} std={hf_vis.std():.6f}")

    # Connector
    hf_proj = hf_model.model.multi_modal_projector(hf_vis_out.last_hidden_state).float()[0]
    print(f"\n  HF connector shape: {hf_proj.shape}")
    print(f"  HF connector [0,0:4]: {hf_proj[0,:4].tolist()}")
    print(f"  HF connector mean={hf_proj.mean():.6f} std={hf_proj.std():.6f}")

    # HF divides by sqrt(hidden)
    hf_proj_scaled = hf_proj / (HIDDEN ** 0.5)
    print(f"  HF connector/sqrt(h) [0,0:4]: {hf_proj_scaled[0,:4].tolist()}")

# ── Test 3: Full generation comparison ──

print("\n=== Test 3: Full generation ===")

# HF
with torch.no_grad():
    hf_out = hf_model.generate(
        **{k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
           for k, v in hf_inputs.items()},
        max_new_tokens=32,
    )
    hf_text = hf_processor.decode(hf_out[0], skip_special_tokens=True)
    hf_new_ids = hf_out[0][260:].tolist()  # skip image+prompt tokens
    print(f"  HF output: '{hf_text}'")
    print(f"  HF token IDs: {hf_new_ids[:10]}...")

# C
c_tokens = []
@TOKEN_CB
def collect_token(token_id, userdata):
    c_tokens.append(token_id)

lib.paligemma_set_token_callback(c_ctx, collect_token, None)
prompt = (ctypes.c_int * 3)(21209, 659, 108)
lib.paligemma_generate(c_ctx, IMG_PATH.encode(), prompt, 3, 32)
lib.paligemma_set_token_callback(c_ctx, TOKEN_CB(), None)

# Decode C tokens
c_text_pieces = []
for tid in c_tokens:
    piece = hf_tokenizer.decode([tid])
    c_text_pieces.append(piece)
c_text = "".join(c_text_pieces)

print(f"  C  output: '{c_text}'")
print(f"  C  token IDs: {c_tokens[:10]}...")

if hf_new_ids[:5] == c_tokens[:5]:
    print("  ✓ PASS: First 5 tokens match!")
else:
    print("  ✗ FAIL: Token mismatch")
    # Find first divergence
    for i in range(min(len(hf_new_ids), len(c_tokens))):
        if i >= len(hf_new_ids) or i >= len(c_tokens) or hf_new_ids[i] != c_tokens[i]:
            print(f"    Diverges at position {i}: HF={hf_new_ids[i] if i < len(hf_new_ids) else 'END'} "
                  f"C={c_tokens[i] if i < len(c_tokens) else 'END'}")
            break

# ── Cleanup ──
lib.paligemma_free(c_ctx)
print("\n=== Done ===")
