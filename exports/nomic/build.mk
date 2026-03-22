# exports/nomic/build.mk — nomic-embed-text embedding export

LTO_SRCS += exports/nomic/nomic.c
CFLAGS += -DENABLE_NOMIC_EMBEDDING
# test-nomic: requires model weights, not wired up yet
