#!/bin/bash
# kittentts_download.sh - Download KittenTTS model and data files
#
# Downloads:
#   1. ONNX model (nano fp32, ~54MB)
#   2. CMU pronouncing dictionary (~3.5MB)
#   3. English word frequency list (20K words)
#
# Usage: ./tools/kittentts_download.sh [output_dir]

set -e

OUTDIR="${1:-.kittentts_build}"
mkdir -p "$OUTDIR"

echo "=== KittenTTS Build Data Download ==="
echo "Output directory: $OUTDIR"
echo ""

# 1. ONNX model
ONNX_URL="https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32/resolve/main/model.onnx"
ONNX_FILE="$OUTDIR/kitten_tts_nano.onnx"
if [ ! -f "$ONNX_FILE" ]; then
    echo "Downloading KittenTTS nano ONNX model (~54MB)..."
    curl -L -o "$ONNX_FILE" "$ONNX_URL"
    echo "  -> $ONNX_FILE ($(du -h "$ONNX_FILE" | cut -f1))"
else
    echo "ONNX model already exists: $ONNX_FILE"
fi

# 2. CMU dict
CMUDICT_URL="https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
CMUDICT_FILE="$OUTDIR/cmudict.dict"
if [ ! -f "$CMUDICT_FILE" ]; then
    echo "Downloading CMU pronouncing dictionary..."
    curl -sL -o "$CMUDICT_FILE" "$CMUDICT_URL"
    echo "  -> $CMUDICT_FILE ($(wc -l < "$CMUDICT_FILE") entries)"
else
    echo "CMU dict already exists: $CMUDICT_FILE"
fi

# 3. Word frequency list
FREQ_URL="https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
FREQ_FILE="$OUTDIR/freq20k.txt"
if [ ! -f "$FREQ_FILE" ]; then
    echo "Downloading English word frequency list (20K)..."
    curl -sL -o "$FREQ_FILE" "$FREQ_URL"
    echo "  -> $FREQ_FILE ($(wc -l < "$FREQ_FILE") words)"
else
    echo "Frequency list already exists: $FREQ_FILE"
fi

echo ""
echo "All downloads complete. Next steps:"
echo "  python3 tools/kittentts_pipeline.py --build-dir $OUTDIR --output-dir kitten-tts-nano"
