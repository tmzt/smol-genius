# exports/gemma/build.mk — Gemma causal LLM decoder export

LTO_SRCS += exports/gemma/gemma.c
CFLAGS += -DENABLE_FUNCTION_GEMMA
TEST_TARGETS += test-gemma

.PHONY: test-gemma
test-gemma:
	python3 exports/gemma/tests/regression.py
