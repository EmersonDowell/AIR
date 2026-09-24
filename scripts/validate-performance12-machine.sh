#!/usr/bin/env bash
# AIR Performance Prompt 12/16: true multi-sequence CUDA decode qualification.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Performance12-$STAMP"; ARCHIVE="$HOME/Downloads/AIR-Performance12-$STAMP.zip"
BUILD="${AIR_PERF12_BUILD_DIR:-$ROOT/build-performance12}"; PROFILE_BUILD="${AIR_PERF12_PROFILE_BUILD_DIR:-$ROOT/build-performance12-profile}"
MODEL="${AIR_PERF12_MODEL:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
ROUNDS="${AIR_DECODE_BATCH_ROUNDS:-5}"; COOLDOWN="${AIR_BENCH_COOLDOWN_SECONDS:-2}"; SEED="${AIR_DECODE_BATCH_SEED:-1219}"
TOKENS="${AIR_DECODE_BATCH_TOKENS:-64}"; RUNS="${AIR_DECODE_BATCH_RUNS:-8}"
mkdir -p "$OUT"/{bench,nsys,prompts,source}
find_tool(){ command -v "$1" 2>/dev/null || find /usr/local/cuda* /opt/nvidia/nsight-systems -type f -name "$1" -executable 2>/dev/null | sort -V | tail -1; }
NVCC="$(find_tool nvcc || true)"; NSYS="$(find_tool nsys || true)"
if [ "${AIR_STOP_EXISTING_SERVERS:-0}" = 1 ]; then
  pgrep -a -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >"$OUT/stopped-servers.txt" 2>&1 || true
  pkill -TERM -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >/dev/null 2>&1 || true; sleep 2
fi
{
 echo harness=AIR-Performance12-R1; echo date="$(date -Is)"; echo air_version_expected=0.9.3; echo model="$MODEL"; echo rounds="$ROUNDS"; echo seed="$SEED"; echo generated_tokens="$TOKENS"; echo runs_per_cell="$RUNS"; echo nvcc="${NVCC:-missing}"; echo nsys="${NSYS:-missing}";
 echo cuda_backend_sha256="$(sha256sum "$ROOT/src/cuda/cuda_backend.cu"|awk '{print $1}')"; echo serving_sha256="$(sha256sum "$ROOT/src/runtime/serving.cpp"|awk '{print $1}')"; echo execution_contract_sha256="$(sha256sum "$ROOT/include/air/execution.hpp"|awk '{print $1}')";
 uname -a; nvidia-smi || true;
} >"$OUT/preflight.txt" 2>&1
[ -f "$MODEL" ] || { echo missing model >"$OUT/FAILURE.txt"; exit 20; }
[ -n "$NVCC" ] || { echo nvcc required >"$OUT/FAILURE.txt"; exit 21; }
sha256sum "$MODEL" >"$OUT/model-sha256.txt"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=OFF >"$OUT/build-release.txt" 2>&1 || exit 22
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-release.txt" 2>&1 || exit 23
ctest --test-dir "$BUILD" --output-on-failure >"$OUT/ctest-release.txt" 2>&1 || exit 24
BENCH="$BUILD/air-bench"; CLI="$BUILD/air-cli"
{ "$BENCH" --version; "$CLI" inspect "$MODEL"|sed -n '1,55p'; } >"$OUT/model-and-version.txt" 2>&1
# Keep the prompt short enough that all equal requests finish prefill in the same scheduler cycle.
cat >"$OUT/prompts/decode.txt" <<'EOF'
Explain in one concise paragraph why measured evidence should decide an optimization instead of intuition alone.
EOF
"$CLI" tokenize "$MODEL" "$(cat "$OUT/prompts/decode.txt")" >"$OUT/prompt-tokenization.txt" 2>&1 || true
python3 - "$OUT/order.tsv" "$ROUNDS" "$SEED" <<'PY'
import random,sys
path=sys.argv[1]; rounds=int(sys.argv[2]); rng=random.Random(int(sys.argv[3]))
with open(path,'w') as f:
    for c in (1,2,4,8):
        for r in range(1,rounds+1):
            modes=['serial','native']; rng.shuffle(modes)
            for mode in modes: f.write(f'{r}\t{c}\t{mode}\n')
PY
DMON=""; if command -v nvidia-smi >/dev/null; then nvidia-smi dmon -s pucvmt -d 1 >"$OUT/gpu-dmon.txt" 2>&1 & DMON=$!; fi
cleanup(){ [ -z "$DMON" ] || kill "$DMON" >/dev/null 2>&1 || true; }; trap cleanup EXIT
FAIL=0
while IFS=$'\t' read -r round concurrency mode; do
  stem="round-$(printf '%02d' "$round")-c${concurrency}-${mode}"; echo "$stem"|tee -a "$OUT/bench/progress.txt"
  decode_linear=baseline; [ "$mode" = native ] && decode_linear=reuse8
  set +e
  "$BENCH" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
    --tokens "$TOKENS" --warmup 0 --runs "$RUNS" --concurrency "$concurrency" \
    --token-budget 256 --prefill-quantum 32 --cuda-prefill-block-linear reuse8 \
    --cuda-prefill-attention online-softmax --cuda-decode-block-linear "$decode_linear" \
    --output "$OUT/bench/$stem.json" >"$OUT/bench/$stem.txt" 2>&1
  rc=$?; set -e; echo "$rc">"$OUT/bench/$stem.exit"; [ "$rc" -eq 0 ]||FAIL=1; sleep "$COOLDOWN"
done <"$OUT/order.tsv"
python3 "$ROOT/scripts/analyze-decode-batching.py" "$OUT/bench" --json "$OUT/decode-batching-summary.json" --markdown "$OUT/decode-batching-summary.md" >"$OUT/analyzer.txt" 2>&1; ARC=$?
# Profile the strongest physical-batch case separately with semantic NVTX ranges.
if [ -n "$NSYS" ]; then
  cmake -S "$ROOT" -B "$PROFILE_BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=ON >"$OUT/build-profile.txt" 2>&1 && cmake --build "$PROFILE_BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-profile.txt" 2>&1 || true
  PB="$PROFILE_BUILD/air-bench"
  if [ -x "$PB" ]; then
    for mode in serial native; do
      decode_linear=baseline; [ "$mode" = native ] && decode_linear=reuse8
      prefix="$OUT/nsys/c8-$mode"
      set +e
      "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o "$prefix" \
        "$PB" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
        --tokens 32 --warmup 0 --runs 8 --concurrency 8 --token-budget 256 --prefill-quantum 32 \
        --cuda-prefill-block-linear reuse8 --cuda-prefill-attention online-softmax --cuda-decode-block-linear "$decode_linear" \
        --output "$prefix-bench.json" >"$prefix-profile.txt" 2>&1
      prc=$?; set -e; echo "$prc">"$prefix-profile.exit"
      if [ "$prc" -eq 0 ]&&[ -f "$prefix.nsys-rep" ]; then
        "$NSYS" stats --report nvtx_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum --report cuda_api_sum "$prefix.nsys-rep" >"$prefix-stats.txt" 2>&1 || true
      fi
    done
  fi
fi
cleanup; trap - EXIT
{
 echo benchmark_failure="$FAIL"; echo analyzer_rc="$ARC"; echo prefill_linear_fixed=batch-reuse8; echo prefill_attention_fixed=online-softmax; echo decode_attention_fixed=baseline; echo native_decode_tactic=batch-reuse8;
} >"$OUT/RESULT.txt"
cp "$ROOT/include/air/execution.hpp" "$OUT/source/execution.hpp"; cp "$ROOT/include/air/cuda.hpp" "$OUT/source/cuda.hpp"; cp "$ROOT/include/air/benchmark.hpp" "$OUT/source/benchmark.hpp"
cp "$ROOT/src/cuda/cuda_backend.cu" "$OUT/source/cuda_backend.cu"; cp "$ROOT/src/runtime/backend.cpp" "$OUT/source/backend.cpp"; cp "$ROOT/src/runtime/serving.cpp" "$OUT/source/serving.cpp"; cp "$ROOT/src/benchmark/benchmark.cpp" "$OUT/source/benchmark.cpp"
cp "$ROOT/scripts/analyze-decode-batching.py" "$OUT/source/"; cp "$ROOT/CMakeLists.txt" "$OUT/source/"
(cd "$(dirname "$OUT")" && zip -qr "$ARCHIVE" "$(basename "$OUT")")
echo; echo "AIR Prompt 12 evidence archive:"; echo "$ARCHIVE"
[ "$FAIL" -eq 0 ] || exit 30
exit "$ARC"
