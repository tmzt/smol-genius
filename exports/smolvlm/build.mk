# exports/smolvlm/build.mk — SmolVLM vision-language model export
# Implies: ENABLE_VISION (pulls in common/vision/)

ENABLE_VISION = 1

LTO_SRCS += exports/smolvlm/connector.c
LTO_SRCS += exports/smolvlm/decoder.c
SRCS += exports/smolvlm/smolvlm.c
CFLAGS += -DENABLE_SMOLVLM
TEST_TARGETS += test-smolvlm

.PHONY: test-smolvlm
test-smolvlm:
	python3 exports/smolvlm/tests/regression.py
