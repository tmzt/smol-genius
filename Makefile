# smol-genius — Modular Text-Only AI Edge Daemon
# Composable Multi-Target Makefile

CC = clang
AR = ar
CFLAGS = -O3 -ffast-math -fPIC -std=c11 -Wall
LDFLAGS = -lm -lpthread

# Include paths
CFLAGS += -Icommon/kernels -Icommon/utils

# Source accumulators
SRCS =
LTO_SRCS =
TEST_TARGETS =

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# =====================================================================
# Hardware detection and flags
# =====================================================================

# Auto-detect ARCH if not specified
ifndef ARCH
  ifeq ($(UNAME_M),arm64)
    ARCH = neon
  else ifeq ($(UNAME_M),aarch64)
    ARCH = neon
  else ifneq (,$(findstring x86,$(UNAME_M)))
    ARCH = avx
  else ifneq (,$(findstring AMD64,$(UNAME_M)))
    ARCH = avx
  else
    ARCH = generic
  endif
endif

# Architecture-specific compiler flags
ifeq ($(ARCH),neon)
  CFLAGS += -march=native
else ifeq ($(ARCH),avx)
  CFLAGS += -march=native -mavx2 -mfma
else
  CFLAGS += -march=native
endif

# BLAS support
ifdef USE_BLAS
  CFLAGS += -DUSE_BLAS
  ifeq ($(UNAME_S),Darwin)
    CFLAGS += -DACCELERATE_NEW_LAPACK
    LDFLAGS += -framework Accelerate
  else
    CFLAGS += -DUSE_OPENBLAS -I/usr/include/openblas
    LDFLAGS += -lopenblas
  endif
endif

# LTO toggle
ifdef LTO
  CFLAGS += -flto
  LDFLAGS += -flto
endif

# =====================================================================
# Dynamic inclusion of sub-makefiles
# =====================================================================

include common/utils/build.mk
include common/kernels/build.mk

ifdef MODEL
  include exports/$(MODEL)/build.mk
endif

# Feature dependencies: exports set ENABLE_* flags, common layers check them
ifdef ENABLE_AUDIO
  CFLAGS += -Icommon/audio
  include common/audio/build.mk
endif
ifdef ENABLE_VISION
  CFLAGS += -Icommon/vision
  include common/vision/build.mk
endif

# =====================================================================
# Build rules
# =====================================================================

OBJS = $(SRCS:.c=.o)
LTO_OBJS = $(LTO_SRCS:.c=.o)
ALL_OBJS = $(OBJS) $(LTO_OBJS)

.PHONY: all clean help info test lib

all: help

help:
	@echo "smol-genius — Modular AI Edge Daemon - Build Targets"
	@echo ""
	@echo "Build library:"
	@echo "  make lib                    - Build libsmol.a (auto-detect arch)"
	@echo "  make lib USE_BLAS=1         - Build with BLAS acceleration"
	@echo "  make lib LTO=1             - Build with link-time optimization"
	@echo "  make lib ARCH=neon         - Force ARM NEON backend"
	@echo "  make lib ARCH=avx          - Force x86 AVX backend"
	@echo "  make lib ARCH=generic      - Force generic C backend"
	@echo ""
	@echo "Build with model export:"
	@echo "  make lib MODEL=gemma        - Include Gemma decoder export"
	@echo "  make lib MODEL=nomic        - Include Nomic embedding export"
	@echo "  make lib MODEL=qwen_asr     - Include Qwen ASR encoder export (implies audio)"
	@echo "  make lib MODEL=smolvlm      - Include SmolVLM vision-language export (implies vision)"
	@echo ""
	@echo "Other targets:"
	@echo "  make test                  - Run all test suites"
	@echo "  make clean                 - Remove build artifacts"
	@echo "  make info                  - Show build configuration"

# Library target
lib: libsmol.a

libsmol.a: $(ALL_OBJS)
	$(AR) rcs $@ $^

# Compile rules (with auto-dependency generation)
DEPFLAGS = -MMD -MP
ALL_DEPS = $(ALL_OBJS:.o=.d)

%.o: %.c
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

-include $(ALL_DEPS)

# Test target
test: $(TEST_TARGETS)

# =====================================================================
# Utilities
# =====================================================================

clean:
	rm -f $(ALL_OBJS) $(ALL_DEPS) libsmol.a test_math
	find . \( -name '*.o' -o -name '*.d' \) -delete

info:
	@echo "Platform:  $(UNAME_S) $(UNAME_M)"
	@echo "Compiler:  $(CC)"
	@echo "Arch:      $(ARCH)"
	@echo "CFLAGS:    $(CFLAGS)"
	@echo "LDFLAGS:   $(LDFLAGS)"
	@echo "SRCS:      $(SRCS)"
	@echo "LTO_SRCS:  $(LTO_SRCS)"
ifdef MODEL
	@echo "MODEL:     $(MODEL)"
endif
ifdef ENABLE_AUDIO
	@echo "AUDIO:     enabled"
endif
ifdef ENABLE_VISION
	@echo "VISION:    enabled"
endif
