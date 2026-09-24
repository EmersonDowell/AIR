#!/usr/bin/env bash
# AIR Performance Prompt 10/16 machine evidence collector.
# Tests quantized-linear prefill tactics through the production InferenceService path.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Performance10-$STAMP"
ARCHIVE="$HOME/Downloads/AIR-Performance10-$STAMP.zip"
BUILD="${AIR_PERF10_BUILD_DIR:-$ROOT/build-performance10}"
PROFILE_BUILD="${AIR_PERF10_PROFILE_BUILD_DIR:-$ROOT/build-performance10-profile}"
MODEL="${AIR_PERF10_MODEL:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
ROUNDS="${AIR_LINEAR_ROUNDS:-5}"
COOLDOWN="${AIR_BENCH_COOLDOWN_SECONDS:-2}"
SEED="${AIR_LINEAR_SEED:-1017}"

mkdir -p "$OUT" "$OUT/correctness" "$OUT/bench" "$OUT/nsys" "$OUT/prompts" "$OUT/source"

find_tool() {
    local name="$1"
    if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return 0; fi
    local candidate
    shopt -s nullglob
    for candidate in \
        /usr/local/cuda*/bin/"$name" \
        /opt/nvidia/nsight-systems/*/bin/"$name" \
        /opt/nvidia/nsight-systems/*/target-linux-x64/"$name"; do
        if [ -x "$candidate" ]; then printf '%s\n' "$candidate"; shopt -u nullglob; return 0; fi
    done
    shopt -u nullglob
    return 1
}

NVCC="$(find_tool nvcc || true)"
NSYS="$(find_tool nsys || true)"

if [ "${AIR_STOP_EXISTING_SERVERS:-0}" = "1" ]; then
    {
        echo "Stopping user-owned AIR/llama benchmark servers before measurement."
        pgrep -a -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' || true
    } > "$OUT/stopped-servers.txt" 2>&1
    pkill -TERM -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >/dev/null 2>&1 || true
    sleep 2
fi

{
    echo "harness=AIR-Performance10-R1"
    echo "date=$(date -Is)"
    echo "air_version_expected=0.9.1"
    echo "model=$MODEL"
    echo "rounds=$ROUNDS"
    echo "seed=$SEED"
    echo "cooldown_seconds=$COOLDOWN"
    echo "nvcc=${NVCC:-missing}"
    echo "nsys=${NSYS:-missing}"
    echo "cmake_sha256=$(sha256sum "$ROOT/CMakeLists.txt" | awk '{print $1}')"
    echo "cuda_backend_sha256=$(sha256sum "$ROOT/src/cuda/cuda_backend.cu" | awk '{print $1}')"
    echo "execution_contract_sha256=$(sha256sum "$ROOT/include/air/execution.hpp" | awk '{print $1}')"
    echo "analyzer_sha256=$(sha256sum "$ROOT/scripts/analyze-linear-tactics.py" | awk '{print $1}')"
    uname -a
    nvidia-smi || true
    echo "--- compute apps before run ---"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
} > "$OUT/preflight.txt" 2>&1

if [ ! -f "$MODEL" ]; then
    echo "missing model: $MODEL" | tee "$OUT/FAILURE.txt"
    exit 20
fi
if [ -z "$NVCC" ]; then
    echo "nvcc is required for Prompt 10 CUDA qualification" | tee "$OUT/FAILURE.txt"
    exit 21
fi

sha256sum "$MODEL" > "$OUT/model-sha256.txt"

# Release build owns correctness and performance qualification.
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=OFF \
    > "$OUT/build-release.txt" 2>&1 || exit 22
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" >> "$OUT/build-release.txt" 2>&1 || exit 23
ctest --test-dir "$BUILD" --output-on-failure > "$OUT/ctest-release.txt" 2>&1 || exit 24

BENCH="$BUILD/air-bench"
VERIFY="$BUILD/air-verify"
CLI="$BUILD/air-cli"

{
    "$BENCH" --version
    "$VERIFY" --version 2>/dev/null || true
    "$CLI" inspect "$MODEL" | sed -n '1,55p'
} > "$OUT/model-and-version.txt" 2>&1

BASE_SENTENCE='Adaptive inference runtimes execute transformer layers while tracking memory ownership, scheduling decisions, numerical evidence, and request latency under a controlled workload. '
python3 - "$OUT/prompts" "$BASE_SENTENCE" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1]); base=sys.argv[2]
for name,repeats in (("p262",10),("p1042",40)):
    (out/f"{name}.txt").write_text(base*repeats)
PY

TOKENS64="$(python3 - <<'PY'
print(','.join(str(i) for i in range(1,65)))
PY
)"

QUALIFIED=()
for tactic in baseline reuse4 reuse8; do
    set +e
    "$VERIFY" -m "$MODEL" --tokens "$TOKENS64" \
        --generate 2 --top-k 8 --device 0 --atol 0.001 \
        --cuda-prefill-block-linear "$tactic" \
        --output "$OUT/correctness/$tactic.json" \
        > "$OUT/correctness/$tactic.txt" 2>&1
    rc=$?
    set -e
    echo "$rc" > "$OUT/correctness/$tactic.exit"
    if [ "$rc" -eq 0 ]; then QUALIFIED+=("$tactic"); fi
done

if ! printf '%s\n' "${QUALIFIED[@]}" | grep -qx baseline; then
    echo "baseline failed differential qualification; Prompt 10 cannot proceed" | tee "$OUT/FAILURE.txt"
    exit 25
fi
printf '%s\n' "${QUALIFIED[@]}" > "$OUT/qualified-tactics.txt"

# Deterministic randomized order per paired round. Every qualified tactic appears
# exactly once in each prompt/round block.
python3 - "$OUT/order.tsv" "$ROUNDS" "$SEED" "${QUALIFIED[@]}" <<'PY'
import random,sys
path=sys.argv[1]; rounds=int(sys.argv[2]); seed=int(sys.argv[3]); tactics=sys.argv[4:]
rng=random.Random(seed)
with open(path,'w') as f:
    for prompt in ('p262','p1042'):
        for r in range(1,rounds+1):
            order=tactics[:]; rng.shuffle(order)
            for tactic in order:
                f.write(f"{r}\t{prompt}\t{tactic}\n")
PY

DMON_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucvmt -d 1 > "$OUT/gpu-dmon.txt" 2>&1 &
    DMON_PID=$!
fi
cleanup() { if [ -n "$DMON_PID" ]; then kill "$DMON_PID" >/dev/null 2>&1 || true; fi; }
trap cleanup EXIT

BENCH_FAILURE=0
while IFS=$'\t' read -r round prompt tactic; do
    stem="round-$(printf '%02d' "$round")-$prompt-$tactic"
    echo "$stem" | tee -a "$OUT/bench/progress.txt"
    set +e
    "$BENCH" -m "$MODEL" --backend cuda --no-manifest \
        --prompt-file "$OUT/prompts/$prompt.txt" \
        --tokens 4 --warmup 0 --runs 1 --concurrency 1 \
        --prefill-quantum 32 --token-budget 256 \
        --cuda-prefill-block-linear "$tactic" \
        --output "$OUT/bench/$stem.json" \
        > "$OUT/bench/$stem.txt" 2>&1
    rc=$?
    set -e
    echo "$rc" > "$OUT/bench/$stem.exit"
    if [ "$rc" -ne 0 ]; then BENCH_FAILURE=1; fi
    sleep "$COOLDOWN"
done < "$OUT/order.tsv"

python3 "$ROOT/scripts/analyze-linear-tactics.py" "$OUT/bench" \
    --json "$OUT/linear-tactic-summary.json" \
    --markdown "$OUT/linear-tactic-summary.md" \
    > "$OUT/analyzer.txt" 2>&1 || ANALYZER_RC=$?
ANALYZER_RC="${ANALYZER_RC:-0}"

# Profiler build is separate so NVTX cannot affect release benchmark results.
if [ -n "$NSYS" ]; then
    cmake -S "$ROOT" -B "$PROFILE_BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=ON \
        > "$OUT/build-profile.txt" 2>&1 || PROFILE_BUILD_RC=$?
    PROFILE_BUILD_RC="${PROFILE_BUILD_RC:-0}"
    if [ "$PROFILE_BUILD_RC" -eq 0 ]; then
        cmake --build "$PROFILE_BUILD" -j"${JOBS:-$(nproc)}" >> "$OUT/build-profile.txt" 2>&1 || PROFILE_BUILD_RC=$?
    fi
    if [ "$PROFILE_BUILD_RC" -eq 0 ]; then
        PBENCH="$PROFILE_BUILD/air-bench"
        for tactic in "${QUALIFIED[@]}"; do
            prefix="$OUT/nsys/p1042-q32-$tactic"
            set +e
            "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
                -o "$prefix" \
                "$PBENCH" -m "$MODEL" --backend cuda --no-manifest \
                    --prompt-file "$OUT/prompts/p1042.txt" \
                    --tokens 2 --warmup 0 --runs 1 --concurrency 1 \
                    --prefill-quantum 32 --token-budget 256 \
                    --cuda-prefill-block-linear "$tactic" \
                    --output "$prefix-bench.json" \
                > "$prefix-profile.txt" 2>&1
            prc=$?
            set -e
            echo "$prc" > "$prefix-profile.exit"
            if [ "$prc" -eq 0 ] && [ -f "$prefix.nsys-rep" ]; then
                "$NSYS" stats \
                    --report nvtx_sum \
                    --report cuda_gpu_kern_sum \
                    --report cuda_gpu_mem_time_sum \
                    --report cuda_gpu_mem_size_sum \
                    --report cuda_api_sum \
                    "$prefix.nsys-rep" > "$prefix-stats.txt" 2>&1 || true
            fi
        done
    fi
else
    echo "Nsight Systems unavailable; randomized release benchmark evidence is still retained." > "$OUT/nsys/MISSING.txt"
fi

cleanup
trap - EXIT

{
    echo "qualified_tactics=$(paste -sd, "$OUT/qualified-tactics.txt")"
    echo "benchmark_failure=$BENCH_FAILURE"
    echo "analyzer_rc=$ANALYZER_RC"
    echo "nsys=${NSYS:-missing}"
    echo "note=No tactic is selected automatically. Candidate acceptance is decided from correctness plus paired evidence."
} > "$OUT/RESULT.txt"

# Preserve exact source/evidence contracts needed to audit the run.
cp "$ROOT/docs/PROMPT9_ATTRIBUTION_RESULT.md" "$OUT/source/"
cp "$ROOT/docs/TACTIC_CONTRACT.md" "$OUT/source/"
cp "$ROOT/docs/QUANTIZED_LINEAR_ENGINE.md" "$OUT/source/"
cp "$ROOT/include/air/execution.hpp" "$OUT/source/execution.hpp"
cp "$ROOT/include/air/cuda.hpp" "$OUT/source/cuda.hpp"
cp "$ROOT/src/cuda/cuda_backend.cu" "$OUT/source/cuda_backend.cu"
cp "$ROOT/src/runtime/execution.cpp" "$OUT/source/execution.cpp"
cp "$ROOT/src/runtime/backend.cpp" "$OUT/source/backend.cpp"
cp "$ROOT/src/runtime/serving.cpp" "$OUT/source/serving.cpp"
cp "$ROOT/scripts/analyze-linear-tactics.py" "$OUT/source/analyze-linear-tactics.py"
cp "$ROOT/CMakeLists.txt" "$OUT/source/CMakeLists.txt"

(
    cd "$(dirname "$OUT")" || exit 1
    zip -qr "$ARCHIVE" "$(basename "$OUT")"
)

echo
echo "========================================"
echo "AIR Prompt 10 evidence archive:"
echo "$ARCHIVE"
echo "Qualified tactics: $(paste -sd, "$OUT/qualified-tactics.txt")"
echo "Benchmark failure flag: $BENCH_FAILURE"
echo "========================================"

# Candidate benchmark failures remain analyzable evidence. Build/baseline correctness
# failures exit earlier. Return nonzero only when the analyzer itself failed.
exit "$ANALYZER_RC"
