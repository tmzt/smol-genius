# exports/qwen_asr/build.mk — Qwen3-ASR full pipeline export
# Implies: ENABLE_AUDIO (pulls in common/audio/)
# Implies: ENABLE_DECODER (pulls in common/decoder/)

ENABLE_AUDIO = 1
ENABLE_DECODER = 1

LTO_SRCS += exports/qwen_asr/encoder.c exports/qwen_asr/qwen_asr.c
CFLAGS += -DENABLE_QWEN_ASR -Iexports/qwen_asr
TEST_TARGETS += test-qwen-asr

.PHONY: test-qwen-asr
test-qwen-asr:
	python3 exports/qwen_asr/tests/regression.py
