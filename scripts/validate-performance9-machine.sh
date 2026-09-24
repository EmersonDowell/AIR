#!/usr/bin/env bash
# AIR Performance Prompt 9/16 machine evidence collector.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Performance9-$STAMP"
ARCHIVE="$HOME/Downloads/AIR-Performance9-$STAMP.zip"
mkdir -p "$OUT"

{
    echo "harness=AIR-Performance9-R1"
    echo "date=$(date -Is)"
    echo "cmake_sha256=$(sha256sum "$ROOT/CMakeLists.txt" | awk '{print $1}')"
    echo "cuda_backend_sha256=$(sha256sum "$ROOT/src/cuda/cuda_backend.cu" | awk '{print $1}')"
    echo "profile_script_sha256=$(sha256sum "$ROOT/scripts/profile-air-performance.sh" | awk '{print $1}')"
    echo "tactic_contract_sha256=$(sha256sum "$ROOT/docs/TACTIC_CONTRACT.md" | awk '{print $1}')"
    echo "prompt8_result_sha256=$(sha256sum "$ROOT/docs/PROMPT8_SCALING_RESULT.md" | awk '{print $1}')"
} > "$OUT/HARNESS.txt"

set +e
"$ROOT/scripts/profile-air-performance.sh" "$OUT" "$@" 2>&1 | tee "$OUT/harness.txt"
RC=${PIPESTATUS[0]}
set -e

echo "harness_exit=$RC" > "$OUT/exit-codes.txt"

# Preserve source-of-evidence snippets without packaging build products.
mkdir -p "$OUT/source"
cp "$ROOT/docs/PROMPT8_SCALING_RESULT.md" "$OUT/source/"
cp "$ROOT/docs/TACTIC_CONTRACT.md" "$OUT/source/"
cp "$ROOT/docs/PERFORMANCE_ATTRIBUTION.md" "$OUT/source/"
cp "$ROOT/CMakeLists.txt" "$OUT/source/CMakeLists.txt"
cp "$ROOT/src/cuda/cuda_backend.cu" "$OUT/source/cuda_backend.cu"

(
    cd "$(dirname "$OUT")" || exit 1
    zip -qr "$ARCHIVE" "$(basename "$OUT")"
)

echo
echo "========================================"
echo "Prompt 9 harness exit: $RC"
echo "Upload archive:"
echo "$ARCHIVE"
echo "========================================"
exit "$RC"
