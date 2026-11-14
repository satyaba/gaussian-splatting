#!/usr/bin/env bash
# SGDC: patch CUDA submodules for modern toolchains.
#
# CUDA 12.9 / GCC 14 / Python 3.13 no longer transitively include <cstdint>,
# which breaks diff_gaussian_rasterization (uint32_t/uint64_t/std::uintptr_t
# undefined) and sometimes simple-knn. This inserts the missing include.
#
# Usage (from the repo root):
#   bash scripts/patch_cuda_submodules.sh
# or with explicit paths:
#   bash scripts/patch_cuda_submodules.sh <rasterizer_dir> <simple_knn_dir>
set -euo pipefail

patch_header () {
  local f="$1"
  [ -f "$f" ] || return 0
  if ! grep -q '#include <cstdint>' "$f"; then
    sed -i '1i #include <cstdint>' "$f"
    echo "patched: $f"
  else
    echo "already patched: $f"
  fi
}

RAST="${1:-submodules/diff-gaussian-rasterization}"
KNN="${2:-submodules/simple-knn}"

patch_header "$RAST/cuda_rasterizer/rasterizer_impl.h"
patch_header "$RAST/cuda_rasterizer/auxiliary.h"
patch_header "$RAST/cuda_rasterizer/forward.h"
patch_header "$RAST/cuda_rasterizer/backward.h"
patch_header "$KNN/simple_knn.cu"

echo "done."
