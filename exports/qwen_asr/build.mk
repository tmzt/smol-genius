# exports/qwen_asr/build.mk — Qwen3-ASR audio encoder export
# Implies: ENABLE_AUDIO (pulls in common/audio/)

ENABLE_AUDIO = 1

LTO_SRCS += exports/qwen_asr/encoder.c
CFLAGS += -DENABLE_QWEN_ASR
TEST_TARGETS += test-qwen-asr

.PHONY: test-qwen-asr
test-qwen-asr:
	python3 exports/qwen_asr/tests/regression.py
