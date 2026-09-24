#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
AIR_ENABLE_NVTX="${AIR_ENABLE_NVTX:-OFF}"

if [[ -z "${AIR_ENABLE_CUDA:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    AIR_ENABLE_CUDA=ON
  else
    AIR_ENABLE_CUDA=OFF
  fi
fi

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DAIR_ENABLE_CUDA="$AIR_ENABLE_CUDA" \
  -DAIR_ENABLE_NVTX="$AIR_ENABLE_NVTX"
cmake --build "$BUILD_DIR" -j"${JOBS:-$(nproc)}"
ctest --test-dir "$BUILD_DIR" --output-on-failure

printf 'AIR build complete: %s (CUDA=%s, NVTX=%s, type=%s)\n' \
  "$BUILD_DIR" "$AIR_ENABLE_CUDA" "$AIR_ENABLE_NVTX" "$BUILD_TYPE"
