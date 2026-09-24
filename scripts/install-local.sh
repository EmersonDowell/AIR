#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-release}"
PREFIX="${PREFIX:-$HOME/.local}"
if [[ -z "${AIR_ENABLE_CUDA:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then AIR_ENABLE_CUDA=ON; else AIR_ENABLE_CUDA=OFF; fi
fi
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DAIR_ENABLE_CUDA="$AIR_ENABLE_CUDA"
cmake --build "$BUILD_DIR" -j"${JOBS:-$(nproc)}"
ctest --test-dir "$BUILD_DIR" --output-on-failure
cmake --install "$BUILD_DIR" --prefix "$PREFIX"
printf 'AIR installed to %s (CUDA=%s)\n' "$PREFIX" "$AIR_ENABLE_CUDA"
