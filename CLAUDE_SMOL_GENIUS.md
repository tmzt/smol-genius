
When executing this, follow the rules defined in CLAUDE.md as well where they don't conflict with the planned changes.

Master Prompt: The smol-genius Monorepo Refactor (Strict Domain Separation)

System Role: You are the Lead Principal Systems Engineer orchestrating a team of expert C-developer subagents. You write hyper-optimized, zero-dependency C code. Your coding style strictly follows the "antirez" philosophy: explicit memory layouts, highly readable for loops, and zero opaque macro abstractions.

The Mission:
We are transforming the tmzt/smol-genius repository into a modular, text-only AI edge daemon monorepo. We are implementing strict domain separation: pure math operations live in common/kernels/, core utilities (memory/parsing) live in common/utils/, and specific executable topologies live in exports/<model>/. We must retain full support for NEON, AVX, BLAS, and Generic C backends.

Execute this refactor sequentially. Do not proceed to the next phase until the current phase is fully verified.

Phase 1: The Demolition Agent (Purging Audio)
Objective: Strip out all audio processing and legacy ASR topologies.

Delete the following files entirely:

qwen_asr_audio.c and qwen_asr_audio.h

qwen_asr_encoder.c

asr_regression.py

The samples/ directory and any .wav files.

Rename core utilities:

Rename qwen_asr_safetensors.c to safetensors.c

Rename qwen_asr_tokenizer.c to tokenizer.c

Phase 2: The Core Utilities (common/utils/)
Objective: Establish the common utilities tier for memory and string handling.

Create the common/utils/ directory.

Move safetensors.c (and .h) and tokenizer.c (and .h) directly into the common/utils/ directory.

Refactor Namespaces: Globally rename functions/structs inside these files to replace the qwen_ prefix with smol_.

Enforce Cohesion: Create common/utils/build.mk. It must append common/utils/safetensors.c and common/utils/tokenizer.c to the standard SRCS variable (NOT LTO_SRCS, as file I/O and string parsing do not need cross-boundary optimization).

Phase 3: The Math Tier (common/kernels/ Multi-Backend)
Objective: Extract the pure math into a stateless Tier 1 library, preserving all hardware backends.

Create common/kernels/ and common/kernels/tests/.

Explode the Monoliths: Analyze qwen_asr_kernels_neon.c, qwen_asr_kernels_avx.c, and qwen_asr_kernels_generic.c. Extract each distinct neural network operator into its own file inside common/kernels/ using the <operation>_<arch>.c convention.

Examples: matmul_neon.c, matmul_avx.c, rmsnorm_generic.c, etc.

Refactor Namespaces: Replace the qwen_ prefix with smol_.

The Unified API Header: Create common/kernels/smol_kernels.h. This header exposes a unified API (e.g., void smol_matmul(...)) that abstracts the underlying architecture.

Enforce Cohesion: Create common/kernels/build.mk. It must conditionally append the correct _<arch>.c files to the LTO_SRCS variable based on the ARCH variable provided by the root Makefile. Create a pure C test harness in common/kernels/tests/test_math.c.

Phase 4: The Exports Tier (exports/gemma/)
Objective: The legacy decoder must be refactored into a pure text causal LLM export.

Create exports/gemma/ and exports/gemma/tests/.

Move the old qwen_asr_decoder.c into exports/gemma/gemma.c.

Remove Cross-Attention: Edit gemma.c to rip out the cross-attention math. It must strictly be a causal self-attention decoder with its KV cache intact.

Enforce Cohesion: Create exports/gemma/build.mk and a Python regression test exports/gemma/tests/regression.py. In build.mk, append gemma.c to LTO_SRCS, add -D ENABLE_FUNCTION_GEMMA to CFLAGS, and define a test-gemma target.

Phase 5: The Exports Tier (exports/nomic/)
Objective: Establish the directory and build logic for the stateless embedding export.

Create exports/nomic/ and exports/nomic/tests/.

Create exports/nomic/nomic.c (leave it as a skeleton with #include "../../common/kernels/smol_kernels.h" and #include "../../common/utils/safetensors.h").

Enforce Cohesion: Create exports/nomic/build.mk and exports/nomic/tests/regression.py. In build.mk, append nomic.c to LTO_SRCS, add -D ENABLE_NOMIC_EMBEDDING to CFLAGS, and define a test-nomic target.

Phase 6: The Build System Agent (The Multi-Target Kconfig Root)
Objective: Create a composable root Makefile.

Rewrite the root Makefile.

Define global variables: CC=clang, AR=ar, CFLAGS = -O3 -ffast-math -fPIC -std=c11 -Wall.

Initialize accumulators: SRCS = , LTO_SRCS = , TEST_TARGETS = .

Hardware Flags & LTO: Implement the LTO=1 toggle. Support ARCH (neon, avx, generic) and USE_BLAS=1.

Dynamic Inclusion: Include the sub-makefiles exactly like this:

Makefile
include common/utils/build.mk
include common/kernels/build.mk
ifdef MODEL
    include exports/$(MODEL)/build.mk
endif
Define build rules separating OBJS and LTO_OBJS. Define the final libsmol.a target packaging them via ar rcs, and a global test target.
