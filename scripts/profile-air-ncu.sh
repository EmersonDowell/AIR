#!/usr/bin/env bash
# Optional targeted Nsight Compute follow-up for Prompt 9.
# Run only after Nsight Systems identifies the dominant kernel family.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${AIR_PROFILE_BUILD_DIR:-$ROOT/build-profile}"
MODEL="${1:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
OUT="${2:-$HOME/Downloads/AIR-NCU-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

find_ncu() {
    command -v ncu 2>/dev/null && return 0
    shopt -s nullglob
    local p
    for p in /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu /opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_*/ncu; do
        [ -x "$p" ] && { echo "$p"; shopt -u nullglob; return 0; }
    done
    shopt -u nullglob
    return 1
}
NCU="$(find_ncu || true)"
[ -n "$NCU" ] || { echo "ncu not found"; exit 2; }
[ -x "$BUILD_DIR/air-bench" ] || { echo "profile build missing: $BUILD_DIR/air-bench"; exit 3; }
[ -f "$MODEL" ] || { echo "model missing: $MODEL"; exit 4; }

PROMPT='Adaptive inference runtimes execute transformer layers while tracking memory ownership, scheduling decisions, numerical evidence, and request latency under a controlled workload. Adaptive inference runtimes execute transformer layers while tracking memory ownership, scheduling decisions, numerical evidence, and request latency under a controlled workload. '

# Limit matched launches because metric replay is expensive. This is a diagnostic sample,
# not an end-to-end latency measurement.
"$NCU" \
    --force-overwrite \
    --kernel-name 'regex:specialized_matmul_kernel|specialized_matvec_kernel|attention_batch_paged_kernel|attention_scores_paged_kernel|attention_value_paged_kernel' \
    --launch-count 12 \
    --section SpeedOfLight \
    --section LaunchStats \
    --section Occupancy \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    -o "$OUT/air-dominant-kernels" \
    "$BUILD_DIR/air-bench" -m "$MODEL" --backend cuda --no-manifest \
        --prompt "$PROMPT" --tokens 4 --warmup 0 --runs 1 --concurrency 1 \
        --prefill-quantum 128 --token-budget 256 \
        --output "$OUT/bench.json" \
    > "$OUT/ncu.txt" 2>&1

echo "$OUT"
