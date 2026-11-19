#!/usr/bin/env bash
# Install FlashMask (https://github.com/PaddlePaddle/FlashMask) for PyTorch.
#
# Run it on every worker, in the same Python environment as training. The core
# kernels are CuTe DSL and get JIT-compiled at first use, so the install itself
# compiles nothing. They support SM90 (Hopper) and SM100 (Blackwell).
#
#   bash recipe/transfusion/build_env.sh
#   FLASHMASK_SRC=/path/to/FlashMask bash recipe/transfusion/build_env.sh   # existing checkout
#
# Env:
#   PYTHON_BIN      python to install into (default: python)
#   FLASHMASK_SRC   FlashMask checkout; cloned here when missing (default: 3rdparty/FlashMask)
#   FLASHMASK_REF   commit or tag to check out (default: the verified commit below)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
FLASHMASK_SRC="${FLASHMASK_SRC:-${ROOT}/3rdparty/FlashMask}"
FLASHMASK_REF="${FLASHMASK_REF:-5b1d0dcafd2d12e1b7df665223530fcbe140c270}"

if [[ ! -d "${FLASHMASK_SRC}/.git" ]]; then
  git clone https://github.com/PaddlePaddle/FlashMask.git "${FLASHMASK_SRC}"
fi
if ! git -C "${FLASHMASK_SRC}" checkout --quiet "${FLASHMASK_REF}" 2>/dev/null; then
  git -C "${FLASHMASK_SRC}" fetch --quiet origin "${FLASHMASK_REF}"
  git -C "${FLASHMASK_SRC}" checkout --quiet FETCH_HEAD
fi

# FLASHMASK_FRAMEWORK=torch skips the Paddle-only parts. setup.py pip-installs its
# own dependencies (nvidia-cutlass, nvidia-cutlass-dsl[cu13], apache-tvm-ffi)
# before building, which is why build isolation is off.
(cd "${FLASHMASK_SRC}" && FLASHMASK_FRAMEWORK=torch FLASHMASK_BUILD=core \
  "${PYTHON_BIN}" -m pip install . --no-build-isolation)

"${PYTHON_BIN}" -c "from flashmask.frontend.torch import flashmask_attention; print('FlashMask OK')"
