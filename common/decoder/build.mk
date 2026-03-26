# common/decoder/build.mk — QK-Norm causal LLM decoder
# Pulled in by exports that set ENABLE_DECODER = 1

LTO_SRCS += common/decoder/qkn_decoder.c
LTO_SRCS += common/decoder/qkn_bf16_decoder.c
CFLAGS += -Icommon/decoder
