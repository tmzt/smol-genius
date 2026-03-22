# exports/nomic/build.mk — Nomic embedding export

LTO_SRCS += exports/nomic/nomic.c
CFLAGS += -DENABLE_NOMIC_EMBEDDING
TEST_TARGETS += test-nomic

.PHONY: test-nomic
test-nomic:
	python3 exports/nomic/tests/regression.py
