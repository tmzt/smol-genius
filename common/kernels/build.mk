# common/kernels/build.mk — Pure math kernel library
# Conditionally appends arch-specific files to LTO_SRCS.

# Architecture-independent kernels (always included)
LTO_SRCS += common/kernels/threading.c
LTO_SRCS += common/kernels/matmul.c
LTO_SRCS += common/kernels/conv2d.c
LTO_SRCS += common/kernels/layernorm.c
LTO_SRCS += common/kernels/rmsnorm.c
LTO_SRCS += common/kernels/activation.c
LTO_SRCS += common/kernels/attention.c
LTO_SRCS += common/kernels/rope.c

# Generic fallback (always compiled)
LTO_SRCS += common/kernels/bf16_matvec_generic.c
LTO_SRCS += common/kernels/vecops_generic.c

# Architecture-specific hot kernels
ifeq ($(ARCH),neon)
    LTO_SRCS += common/kernels/bf16_matvec_neon.c
    LTO_SRCS += common/kernels/vecops_neon.c
else ifeq ($(ARCH),avx)
    LTO_SRCS += common/kernels/bf16_matvec_avx.c
    LTO_SRCS += common/kernels/vecops_avx.c
else
    # generic or auto-detect: compile all, let dispatch macros select
    LTO_SRCS += common/kernels/bf16_matvec_neon.c
    LTO_SRCS += common/kernels/vecops_neon.c
    LTO_SRCS += common/kernels/bf16_matvec_avx.c
    LTO_SRCS += common/kernels/vecops_avx.c
endif

# Test target
TEST_TARGETS += test-math

.PHONY: test-math
test-math: libsmol.a
	$(CC) $(CFLAGS) -o test_math common/kernels/tests/test_math.c -L. -lsmol $(LDFLAGS)
	./test_math
