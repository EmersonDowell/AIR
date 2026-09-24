#!/usr/bin/env bash
# AIR Performance Prompt 9/16: attribution collection.
# Uses the production InferenceService path through air-bench. No alternate executor.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${AIR_PROFILE_BUILD_DIR:-$ROOT/build-profile}"
OUT="${1:?usage: profile-air-performance.sh OUTPUT_DIR [MODEL ...]}"
shift
mkdir -p "$OUT" "$OUT/correctness" "$OUT/tactic-sweep" "$OUT/nsys" "$OUT/prompts"

find_tool() {
    local name="$1"
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi
    local candidate
    shopt -s nullglob
    for candidate in \
        /usr/local/cuda*/bin/"$name" \
        /opt/nvidia/nsight-systems/*/bin/"$name" \
        /opt/nvidia/nsight-systems/*/target-linux-x64/"$name" \
        /opt/nvidia/nsight-compute/*/"$name" \
        /opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_*/"$name"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            shopt -u nullglob
            return 0
        fi
    done
    shopt -u nullglob
    return 1
}

NSYS="$(find_tool nsys || true)"
NCU="$(find_tool ncu || true)"
NVCC="$(find_tool nvcc || true)"

{
    echo "date=$(date -Is)"
    echo "root=$ROOT"
    echo "build_dir=$BUILD_DIR"
    echo "nsys=${NSYS:-missing}"
    echo "ncu=${NCU:-missing}"
    echo "nvcc=${NVCC:-missing}"
    uname -a
    nvidia-smi || true
    [ -n "$NSYS" ] && "$NSYS" --version || true
    [ -n "$NCU" ] && "$NCU" --version || true
    [ -n "$NVCC" ] && "$NVCC" --version || true
} > "$OUT/preflight.txt" 2>&1

if [ -z "$NVCC" ]; then
    echo "nvcc is required for Prompt 9 CUDA attribution" | tee -a "$OUT/preflight.txt"
    exit 20
fi

# Profile builds are opt-in and separate from the normal release build.
cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DAIR_ENABLE_CUDA=ON \
    -DAIR_ENABLE_NVTX=ON \
    > "$OUT/build.txt" 2>&1 || exit 21
cmake --build "$BUILD_DIR" -j"${JOBS:-$(nproc)}" >> "$OUT/build.txt" 2>&1 || exit 22
ctest --test-dir "$BUILD_DIR" --output-on-failure > "$OUT/ctest.txt" 2>&1 || exit 23

BENCH="$BUILD_DIR/air-bench"
VERIFY="$BUILD_DIR/air-verify"
CLI="$BUILD_DIR/air-cli"

{
    "$BENCH" --version
    "$VERIFY" --version 2>/dev/null || true
    "$CLI" --version 2>/dev/null || true
} > "$OUT/air-version.txt" 2>&1

BASE_SENTENCE='Adaptive inference runtimes execute transformer layers while tracking memory ownership, scheduling decisions, numerical evidence, and request latency under a controlled workload. '
python3 - "$OUT/prompts" "$BASE_SENTENCE" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1]); base=sys.argv[2]
for name,repeats in (("p54",2),("p262",10),("p1042",40)):
    (out/f"{name}.txt").write_text(base*repeats)
PY

# Resolve candidate models. Explicit arguments win. Defaults are the controlled Qwen2.5 ladder.
MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
    MODELS=(
        "$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        "$HOME/Models/AIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        "$HOME/Models/AIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    )
fi

: > "$OUT/models.tsv"
QUALIFIED_MODEL=""
index=0
for model in "${MODELS[@]}"; do
    [ -f "$model" ] || continue
    index=$((index+1))
    tag="model-$(printf '%02d' "$index")"
    sha="$(sha256sum "$model" | awk '{print $1}')"
    bytes="$(stat -c %s "$model")"
    printf '%s\t%s\t%s\t%s\n' "$tag" "$bytes" "$sha" "$model" >> "$OUT/models.tsv"
    "$CLI" inspect "$model" > "$OUT/correctness/$tag-inspect.txt" 2>&1 || true

    set +e
    "$VERIFY" -m "$model" \
        --tokens 1,2,3,4 \
        --generate 4 --top-k 8 --device 0 --atol 0.001 \
        --output "$OUT/correctness/$tag-verification.json" \
        > "$OUT/correctness/$tag-verification.txt" 2>&1
    verify_rc=$?
    set -e
    echo "$verify_rc" > "$OUT/correctness/$tag-verification.exit"

    # One fully traced teacher-forced decode transition. This is diagnostic and
    # intentionally separate from the performance run because trace readbacks synchronize CUDA.
    set +e
    "$VERIFY" -m "$model" \
        --tokens 1,2,3,4 \
        --generate 2 --top-k 8 --device 0 \
        --trace-from 0 --trace-count 1 \
        --output "$OUT/correctness/$tag-trace.json" \
        > "$OUT/correctness/$tag-trace.txt" 2>&1
    trace_rc=$?
    set -e
    echo "$trace_rc" > "$OUT/correctness/$tag-trace.exit"

    if [ "$verify_rc" -eq 0 ] && [ -z "$QUALIFIED_MODEL" ]; then
        QUALIFIED_MODEL="$model"
    fi
done

if [ -z "$QUALIFIED_MODEL" ]; then
    echo "No candidate model passed the frozen atol=0.001 verification gate." | tee -a "$OUT/preflight.txt"
    exit 24
fi
printf '%s\n' "$QUALIFIED_MODEL" > "$OUT/qualified-performance-model.txt"

# Existing-tactic sensitivity. This changes scheduler geometry only; executor code is unchanged.
for quantum in 32 64 128; do
    for prompt in p54 p262 p1042; do
        stem="q${quantum}-${prompt}-c1"
        "$BENCH" -m "$QUALIFIED_MODEL" \
            --backend cuda --no-manifest --prompt-file "$OUT/prompts/$prompt.txt" \
            --tokens 32 --warmup 1 --runs 3 --concurrency 1 \
            --prefill-quantum "$quantum" --token-budget 256 \
            --output "$OUT/tactic-sweep/$stem.json" \
            > "$OUT/tactic-sweep/$stem.txt" 2>&1 || exit 30
    done
done

# One concurrency sensitivity point under the largest existing native prefill quantum.
for prompt in p54 p262; do
    stem="q128-${prompt}-c4"
    "$BENCH" -m "$QUALIFIED_MODEL" \
        --backend cuda --no-manifest --prompt-file "$OUT/prompts/$prompt.txt" \
        --tokens 32 --warmup 1 --runs 3 --concurrency 4 \
        --prefill-quantum 128 --token-budget 512 \
        --output "$OUT/tactic-sweep/$stem.json" \
        > "$OUT/tactic-sweep/$stem.txt" 2>&1 || exit 31
done

# Keep a coarse hardware trace around profiler work. Do not use it as a substitute for Nsight timing.
DMON_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucvmt -d 1 > "$OUT/gpu-dmon.txt" 2>&1 &
    DMON_PID=$!
fi
cleanup() {
    if [ -n "$DMON_PID" ]; then kill "$DMON_PID" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

if [ -z "$NSYS" ]; then
    echo "Nsight Systems (nsys) not found; tactic/correctness evidence collected but Prompt 9 attribution is incomplete." \
        | tee "$OUT/nsys/MISSING.txt"
    exit 25
fi

profile_case() {
    local prompt="$1" quantum="$2" concurrency="$3" tokens="$4"
    local stem="${prompt}-q${quantum}-c${concurrency}"
    local prefix="$OUT/nsys/$stem"
    set +e
    "$NSYS" profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --force-overwrite=true \
        -o "$prefix" \
        "$BENCH" -m "$QUALIFIED_MODEL" \
            --backend cuda --no-manifest --prompt-file "$OUT/prompts/$prompt.txt" \
            --tokens "$tokens" --warmup 0 --runs 1 --concurrency "$concurrency" \
            --prefill-quantum "$quantum" --token-budget 512 \
            --output "$prefix-bench.json" \
        > "$prefix-profile.txt" 2>&1
    local rc=$?
    set -e
    echo "$rc" > "$prefix-profile.exit"
    if [ "$rc" -ne 0 ]; then return; fi
    local rep="$prefix.nsys-rep"
    if [ -f "$rep" ]; then
        "$NSYS" stats \
            --report nvtx_sum \
            --report cuda_gpu_kern_sum \
            --report cuda_gpu_mem_time_sum \
            --report cuda_gpu_mem_size_sum \
            --report cuda_api_sum \
            "$rep" > "$prefix-stats.txt" 2>&1 || true
    fi
}

# Baseline short prompt, long prompt, long prompt at max native quantum, and a concurrency case.
profile_case p54 32 1 16
profile_case p1042 32 1 8
profile_case p1042 128 1 8
profile_case p54 128 4 8

cleanup
trap - EXIT

{
    echo "qualified_model=$QUALIFIED_MODEL"
    echo "nsys=$NSYS"
    echo "ncu=${NCU:-missing}"
    echo "note=Nsight Compute is intentionally not run automatically; use profile-air-ncu.sh after Systems identifies dominant kernels."
} > "$OUT/RESULT.txt"
