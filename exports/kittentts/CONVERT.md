# KittenTTS Model Conversion

Convert the KittenTTS ONNX model into artifacts for the pure C inference engine.

## Prerequisites

```bash
pip install -r tools/kittentts_requirements.txt
```

Requires: `onnx`, `safetensors`, `numpy` (Python 3.9+).

## Quick Start

```bash
python3 tools/kittentts_pipeline.py \
  --build-dir .kittentts_build \
  --output-dir kitten-tts-nano
```

This downloads all source files, converts weights, and builds the G2P table in one step.

## What It Does

| Step | Input | Output | Size |
|------|-------|--------|------|
| Download | HuggingFace, GitHub | ONNX model, CMU dict, freq list | ~58 MB |
| Convert | `kitten_tts_nano.onnx` | `model.safetensors` + `phoneme_vocab.json` | 53.4 MB |
| G2P | CMU dict + freq list | `g2p_table.h` (compiled into binary) | 505 KB |

## Pipeline Steps

### Step 1: Download

```bash
python3 tools/kittentts_pipeline.py --step download --build-dir .kittentts_build
```

Downloads from:
- **ONNX model** (~54 MB): `https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32/resolve/main/model.onnx`
- **CMU dict** (~3.5 MB): `https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict`
- **Frequency list** (~152 KB): `https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt`

Or use the shell script: `./tools/kittentts_download.sh .kittentts_build`

### Step 2: Convert ONNX to Safetensors

```bash
python3 tools/kittentts_pipeline.py --step convert \
  --build-dir .kittentts_build \
  --output-dir kitten-tts-nano
```

Or directly:
```bash
python3 tools/convert_kittentts.py \
  --onnx .kittentts_build/kitten_tts_nano.onnx \
  --output-dir kitten-tts-nano
```

Produces:
- `kitten-tts-nano/model.safetensors` — 475 tensors, all f32
- `kitten-tts-nano/phoneme_vocab.json` — 176-token IPA phoneme vocabulary

The converter remaps ONNX tensor names to a clean hierarchy (`plbert.*`, `text_enc.*`, `prosody.*`, `decoder.*`, `vocoder.*`) and handles LSTM weight gate reordering (ONNX IOFC to standard IFGO).

### Step 3: Build G2P Table

```bash
python3 tools/kittentts_pipeline.py --step g2p --build-dir .kittentts_build
```

Or directly:
```bash
python3 tools/build_g2p_table.py \
  --freq .kittentts_build/freq20k.txt \
  --cmudict .kittentts_build/cmudict.dict \
  --vocab kitten-tts-nano/phoneme_vocab.json \
  --output exports/kittentts/g2p_table.h
```

Produces `exports/kittentts/g2p_table.h` — an 18K-word English pronunciation dictionary compiled into a C hash table. This is checked into the repo so the build has zero Python dependencies.

### Inspect Model

```bash
python3 tools/kittentts_pipeline.py --step inspect --build-dir .kittentts_build
```

Prints all ONNX tensor names, shapes, input/output specs, and op types.

## Build

Library only:
```bash
make lib MODEL=kittentts USE_BLAS=1
```

Library + standalone binary:
```bash
make lib MODEL=kittentts USE_BLAS=1 APP=1
```

On Linux replace `USE_BLAS=1` with `USE_BLAS=1` and ensure OpenBLAS is installed.

## Run

```bash
./kittentts -d kitten-tts-nano -o out.wav "Hello world"
./kittentts -d kitten-tts-nano -s 1 --speed 0.9 -o out.wav "Slower Jasper voice"
./kittentts --list-styles
```

## Runtime Dependencies

None. The binary is fully self-contained:
- Model weights: `model.safetensors` (53 MB, loaded via mmap)
- Phoneme vocab: `phoneme_vocab.json` (loaded at startup)
- G2P dictionary: compiled into the binary (505 KB)

No espeak-ng, no Python, no network access at runtime.

## File Map

```
tools/
  kittentts_pipeline.py       Master pipeline orchestrator
  kittentts_download.sh        Standalone download script
  kittentts_requirements.txt   Python dependencies
  convert_kittentts.py         ONNX -> safetensors converter
  build_g2p_table.py           CMU dict -> C header generator

exports/kittentts/
  g2p_table.h                  Embedded 18K-word G2P dictionary (generated)
  phonemizer.c                 G2P lookup + letter-to-sound fallback
  kittentts.h                  Public API + model constants
  kittentts.c                  Pipeline orchestration + weight loading
  plbert.c                     PL-BERT encoder forward pass
  text_encoder.c               Conv1D + BiLSTM text encoder
  prosody.c                    Duration/F0/noise prediction
  acoustic_decoder.c           AdaIN residual decode chain
  vocoder.c                    iSTFT-Net vocoder
  main.c                       CLI entry point
  build.mk                     Build system integration
```
