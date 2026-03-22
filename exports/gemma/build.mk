# exports/gemma/build.mk — Gemma causal LLM decoder export
# Implies: ENABLE_DECODER (pulls in common/decoder/)

ENABLE_DECODER = 1

LTO_SRCS += exports/gemma/gemma.c
CFLAGS += -DENABLE_FUNCTION_GEMMA -Iexports/gemma
TEST_TARGETS += test-gemma

.PHONY: test-gemma
test-gemma:
	python3 exports/gemma/tests/regression.py
