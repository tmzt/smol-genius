# exports/paligemma/build.mk — PaliGemma vision-language model export
# Implies: ENABLE_VISION (pulls in common/vision/), ENABLE_DECODER (pulls in common/decoder/)

ENABLE_VISION = 1
ENABLE_DECODER = 1

LTO_SRCS += exports/paligemma/connector.c
SRCS += exports/paligemma/paligemma.c
CFLAGS += -DENABLE_PALIGEMMA -Iexports/paligemma
