#!/usr/bin/env bash
# AIR Performance Prompt 13/16 R3 machine evidence collector.
# Compares qualified reuse8 against the memory-heavier dense FP32 cuBLAS research tactic.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Performance13-R3-$STAMP"
ARCHIVE="$HOME/Downloads/AIR-Performance13-R3-$STAMP.zip"
BUILD="${AIR_PERF13_R3_BUILD_DIR:-$ROOT/build-performance13-r3}"
PROFILE_BUILD="${AIR_PERF13_R3_PROFILE_BUILD_DIR:-$ROOT/build-performance13-r3-profile}"
MODEL="${AIR_PERF13_MODEL:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
ROUNDS="${AIR_DENSE_ROUNDS:-5}"
COOLDOWN="${AIR_BENCH_COOLDOWN_SECONDS:-2}"
SEED="${AIR_DENSE_SEED:-1327}"
mkdir -p "$OUT"/{correctness,bench,nsys,prompts,source}

find_tool(){ local n="$1"; command -v "$n" 2>/dev/null && return 0; local x; shopt -s nullglob; for x in /usr/local/cuda*/bin/"$n" /opt/nvidia/nsight-systems/*/bin/"$n"; do [ -x "$x" ]&&{ printf '%s\n' "$x"; shopt -u nullglob; return 0; }; done; shopt -u nullglob; return 1; }
NVCC="$(find_tool nvcc||true)"; NSYS="$(find_tool nsys||true)"

if [ "${AIR_STOP_EXISTING_SERVERS:-0}" = 1 ]; then
  { echo 'Stopping user-owned AIR/llama servers'; pgrep -a -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)'||true; } >"$OUT/stopped-servers.txt" 2>&1
  pkill -TERM -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >/dev/null 2>&1||true; sleep 2
fi

{
 echo harness=AIR-Performance13-R3
 echo date="$(date -Is)"
 echo air_version_expected=0.9.6
 echo benchmark_schema_expected=air.benchmark.v8
 echo manifest_schema_expected=8
 echo model="$MODEL"
 echo rounds="$ROUNDS"
 echo seed="$SEED"
 echo nvcc="${NVCC:-missing}"
 echo nsys="${NSYS:-missing}"
 echo cuda_backend_sha256="$(sha256sum "$ROOT/src/cuda/cuda_backend.cu"|awk '{print $1}')"
 echo execution_contract_sha256="$(sha256sum "$ROOT/include/air/execution.hpp"|awk '{print $1}')"
 echo analyzer_sha256="$(sha256sum "$ROOT/scripts/analyze-dense-f32-cublas.py"|awk '{print $1}')"
 uname -a
 nvidia-smi||true
} >"$OUT/preflight.txt" 2>&1

[ -f "$MODEL" ]||{ echo "missing model: $MODEL"|tee "$OUT/FAILURE.txt"; exit 20; }
[ -n "$NVCC" ]||{ echo 'nvcc required'|tee "$OUT/FAILURE.txt"; exit 21; }
sha256sum "$MODEL">"$OUT/model-sha256.txt"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=OFF >"$OUT/build-release.txt" 2>&1||exit 22
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-release.txt" 2>&1||exit 23
ctest --test-dir "$BUILD" --output-on-failure >"$OUT/ctest-release.txt" 2>&1||exit 24
BENCH="$BUILD/air-bench"; VERIFY="$BUILD/air-verify"; CLI="$BUILD/air-cli"
{ "$BENCH" --version; "$CLI" inspect "$MODEL"|sed -n '1,55p'; } >"$OUT/model-and-version.txt" 2>&1

BASE='Adaptive inference runtimes execute transformer layers while tracking memory ownership, scheduling decisions, numerical evidence, and request latency under a controlled workload. '
python3 - "$OUT/prompts" "$BASE" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); b=sys.argv[2]
(p/'p262.txt').write_text(b*10)
(p/'p1042.txt').write_text(b*40)
(p/'decode.txt').write_text('Explain in one compact paragraph why batching can improve GPU inference throughput while preserving independent request state.')
PY
TOKENS1="1"
TOKENS8="$(python3 - <<'PY'
print(','.join(str(i) for i in range(1,9)))
PY
)"
TOKENS64="$(python3 - <<'PY'
print(','.join(str(i) for i in range(1,65)))
PY
)"

# Characterize numerical behavior at three prefill widths before the hard gate.
for width in 1 8 64; do
  token_var="TOKENS${width}"
  token_csv="${!token_var}"
  for t in reuse8 dense-f32-cublas; do
    set +e
    "$VERIFY" -m "$MODEL" --tokens "$token_csv" --generate 2 --top-k 8 --device 0 --atol 0.001 \
      --cuda-prefill-block-linear "$t" --cuda-prefill-attention online-softmax \
      --output "$OUT/correctness/${t}-p${width}.json" >"$OUT/correctness/${t}-p${width}.txt" 2>&1
    rc=$?
    set -e
    echo "$rc">"$OUT/correctness/${t}-p${width}.exit"
  done
done

# The 64-token verification remains the hard admission gate for competitive timing.
cp "$OUT/correctness/reuse8-p64.json" "$OUT/correctness/reuse8.json"
cp "$OUT/correctness/reuse8-p64.txt" "$OUT/correctness/reuse8.txt"
cp "$OUT/correctness/reuse8-p64.exit" "$OUT/correctness/reuse8.exit"
cp "$OUT/correctness/dense-f32-cublas-p64.json" "$OUT/correctness/dense-f32-cublas.json"
cp "$OUT/correctness/dense-f32-cublas-p64.txt" "$OUT/correctness/dense-f32-cublas.txt"
cp "$OUT/correctness/dense-f32-cublas-p64.exit" "$OUT/correctness/dense-f32-cublas.exit"

# The frozen numerical gate is applied before any competitive dense timing.
REUSE_RC="$(cat "$OUT/correctness/reuse8.exit")"
DENSE_RC="$(cat "$OUT/correctness/dense-f32-cublas.exit")"
[ "$REUSE_RC" -eq 0 ]||{ echo 'reuse8 control failed correctness'|tee "$OUT/FAILURE.txt"; exit 25; }
DENSE_QUAL=0
[ "$DENSE_RC" -eq 0 ]&&DENSE_QUAL=1

# If strict verification passes, require exact greedy service output and real physical decode batching.
if [ "$DENSE_QUAL" -eq 1 ]; then
  for t in reuse8 dense-f32-cublas; do
    set +e
    "$BENCH" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
      --tokens 16 --warmup 1 --runs 4 --concurrency 4 --prefill-quantum 32 --token-budget 256 \
      --cuda-prefill-block-linear reuse8 --cuda-prefill-attention online-softmax \
      --cuda-decode-block-linear "$t" --output "$OUT/correctness/decode-$t.json" \
      >"$OUT/correctness/decode-$t.txt" 2>&1
    rc=$?
    set -e
    echo "$rc">"$OUT/correctness/decode-$t.exit"
    [ "$rc" -eq 0 ]||DENSE_QUAL=0
  done
fi
if [ "$DENSE_QUAL" -eq 1 ]; then
  python3 - "$OUT/correctness/decode-reuse8.json" "$OUT/correctness/decode-dense-f32-cublas.json" <<'PY' || DENSE_QUAL=0
import json,sys
A=json.load(open(sys.argv[1])); B=json.load(open(sys.argv[2]))
assert A['output_tokens']==B['output_tokens'], 'decode output mismatch'
assert B['summary']['native_decode_batches']>0, 'dense candidate did not physically batch'
PY
fi
echo "$DENSE_QUAL">"$OUT/correctness/dense-qualified.txt"

# No competitive timing is run for an unqualified dense-control tactic.
TACTICS=(reuse8)
[ "$DENSE_QUAL" -eq 1 ]&&TACTICS+=(dense)
printf '%s\n' "${TACTICS[@]}">"$OUT/qualified-tactics.txt"

python3 - "$OUT/order.tsv" "$ROUNDS" "$SEED" "${TACTICS[@]}" <<'PY'
import random,sys
out=sys.argv[1]; rounds=int(sys.argv[2]); seed=int(sys.argv[3]); ts=sys.argv[4:]; rng=random.Random(seed)
with open(out,'w') as f:
  if len(ts)<2:
    raise SystemExit(0)
  for lane,workloads in [('prefill',['p262','p1042']),('decode',['c8'])]:
    for w in workloads:
      for r in range(1,rounds+1):
        o=ts[:]; rng.shuffle(o)
        for t in o:f.write(f'{lane}\t{r}\t{w}\t{t}\n')
PY

DMON_PID=''
command -v nvidia-smi >/dev/null 2>&1&&{ nvidia-smi dmon -s pucvmt -d 1 >"$OUT/gpu-dmon.txt" 2>&1 & DMON_PID=$!; }
cleanup(){ [ -n "$DMON_PID" ]&&kill "$DMON_PID" >/dev/null 2>&1||true; }
trap cleanup EXIT

FAIL=0
if [ "$DENSE_QUAL" -eq 1 ]; then
  while IFS=$'\t' read -r lane round workload tactic; do
    [ -n "$lane" ]||continue
    stem="$lane-round-$(printf '%02d' "$round")-$workload-$tactic"
    echo "$stem"|tee -a "$OUT/bench/progress.txt"
    emitted="$tactic"; [ "$tactic" = dense ]&&emitted=dense-f32-cublas
    set +e
    if [ "$lane" = prefill ]; then
      "$BENCH" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/$workload.txt" \
        --tokens 4 --warmup 1 --runs 1 --concurrency 1 --prefill-quantum 32 --token-budget 256 \
        --cuda-prefill-block-linear "$emitted" --cuda-prefill-attention online-softmax \
        --cuda-decode-block-linear baseline --output "$OUT/bench/$stem.json" >"$OUT/bench/$stem.txt" 2>&1
    else
      "$BENCH" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
        --tokens 64 --warmup 1 --runs 8 --concurrency 8 --prefill-quantum 32 --token-budget 256 \
        --cuda-prefill-block-linear reuse8 --cuda-prefill-attention online-softmax \
        --cuda-decode-block-linear "$emitted" --output "$OUT/bench/$stem.json" >"$OUT/bench/$stem.txt" 2>&1
    fi
    rc=$?
    set -e
    echo "$rc">"$OUT/bench/$stem.exit"
    [ "$rc" -ne 0 ]&&FAIL=1
    sleep "$COOLDOWN"
  done <"$OUT/order.tsv"
fi

ARC=0
if [ "$DENSE_QUAL" -eq 1 ]; then
  python3 "$ROOT/scripts/analyze-dense-f32-cublas.py" "$OUT/bench" \
    --json "$OUT/dense-f32-summary.json" --markdown "$OUT/dense-f32-summary.md" \
    >"$OUT/analyzer.txt" 2>&1||ARC=$?
else
  cat >"$OUT/dense-f32-summary.md" <<'TXT'
# AIR Prompt 13 R3 Dense FP32 cuBLAS Summary

The candidate did not pass the frozen differential-correctness gate. Competitive performance timing was intentionally not run.
TXT
fi

# Nsight only for a candidate that passed correctness. Both tactics receive identical warmup/run structure.
if [ "$DENSE_QUAL" -eq 1 ] && [ -n "$NSYS" ]; then
  cmake -S "$ROOT" -B "$PROFILE_BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=ON >"$OUT/build-profile.txt" 2>&1||PRC=$?
  PRC="${PRC:-0}"
  [ "$PRC" -eq 0 ]&&cmake --build "$PROFILE_BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-profile.txt" 2>&1||true
  PB="$PROFILE_BUILD/air-bench"
  if [ -x "$PB" ]; then
    for t in reuse8 dense; do
      emitted="$t"; [ "$t" = dense ]&&emitted=dense-f32-cublas
      for lane in prefill decode; do
        prefix="$OUT/nsys/$lane-$t"
        set +e
        if [ "$lane" = prefill ]; then
          "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o "$prefix" \
            "$PB" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/p1042.txt" \
            --tokens 2 --warmup 1 --runs 1 --concurrency 1 --prefill-quantum 32 --token-budget 256 \
            --cuda-prefill-block-linear "$emitted" --cuda-prefill-attention online-softmax \
            --output "$prefix-bench.json" >"$prefix-profile.txt" 2>&1
        else
          "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o "$prefix" \
            "$PB" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
            --tokens 64 --warmup 1 --runs 8 --concurrency 8 --prefill-quantum 32 --token-budget 256 \
            --cuda-prefill-block-linear reuse8 --cuda-prefill-attention online-softmax \
            --cuda-decode-block-linear "$emitted" --output "$prefix-bench.json" >"$prefix-profile.txt" 2>&1
        fi
        x=$?
        set -e
        echo "$x">"$prefix-profile.exit"
        [ "$x" -eq 0 ]&&[ -f "$prefix.nsys-rep" ]&&"$NSYS" stats \
          --report nvtx_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
          --report cuda_gpu_mem_size_sum --report cuda_api_sum "$prefix.nsys-rep" \
          >"$prefix-stats.txt" 2>&1||true
      done
    done
  fi
elif [ -z "$NSYS" ]; then
  echo 'Nsight Systems unavailable' >"$OUT/nsys/MISSING.txt"
else
  echo 'Dense FP32 candidate unqualified; profiler timing intentionally skipped.' >"$OUT/nsys/SKIPPED.txt"
fi

cleanup; trap - EXIT
{
 echo benchmark_failure="$FAIL"
 echo analyzer_rc="$ARC"
 echo dense_correctness_qualified="$DENSE_QUAL"
 echo tactics="$(IFS=,;echo "${TACTICS[*]}")"
 echo note='dense-f32-cublas is a diagnostic/Pareto candidate; no automatic selection is made.'
} >"$OUT/RESULT.txt"

cp "$ROOT/docs/PROMPT13_SHARED_TILE_RESULT.md" "$OUT/source/"
cp "$ROOT/docs/DENSE_F32_CUBLAS_CONTROL.md" "$OUT/source/"
cp "$ROOT/docs/TACTIC_CONTRACT.md" "$OUT/source/"
cp "$ROOT/include/air/execution.hpp" "$OUT/source/execution.hpp"
cp "$ROOT/include/air/cuda.hpp" "$OUT/source/cuda.hpp"
cp "$ROOT/src/cuda/cuda_backend.cu" "$OUT/source/cuda_backend.cu"
cp "$ROOT/src/runtime/backend.cpp" "$OUT/source/backend.cpp"
cp "$ROOT/src/runtime/serving.cpp" "$OUT/source/serving.cpp"
cp "$ROOT/scripts/analyze-dense-f32-cublas.py" "$OUT/source/"
cp "$ROOT/CMakeLists.txt" "$OUT/source/"

(cd "$(dirname "$OUT")"&&zip -qr "$ARCHIVE" "$(basename "$OUT")")
echo
echo '========================================'
echo 'AIR Prompt 13 R3 evidence archive:'
echo "$ARCHIVE"
echo "dense correctness qualified: $DENSE_QUAL"
echo '========================================'
exit "$ARC"
