#!/usr/bin/env bash
# AIR Prompt 13/16 consolidated Sprint-3 WolfCat qualification collector.
# Strict product correctness is always applied before competitive timing.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Performance13-Sprint3-$STAMP"
ARCHIVE="$HOME/Downloads/AIR-Performance13-Sprint3-$STAMP.zip"
BUILD="${AIR_SPRINT3_BUILD_DIR:-$ROOT/build-performance13-sprint3}"
PROFILE_BUILD="${AIR_SPRINT3_PROFILE_BUILD_DIR:-$ROOT/build-performance13-sprint3-profile}"
M05="${AIR_MODEL_05:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
M15="${AIR_MODEL_15:-$HOME/Models/AIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
M7="${AIR_MODEL_7:-$HOME/Models/AIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf}"
ROUNDS="${AIR_SPRINT3_ROUNDS:-5}"
COOLDOWN="${AIR_BENCH_COOLDOWN_SECONDS:-2}"
SEED="${AIR_SPRINT3_SEED:-131013}"
RUN_SCALE_VERIFY="${AIR_RUN_SCALE_VERIFY:-1}"
RUN_NSYS="${AIR_RUN_NSYS:-1}"
RUN_NCU="${AIR_RUN_NCU:-0}"
mkdir -p "$OUT"/{correctness,bench,nsys,ncu,prompts,scale,source,research}

find_tool(){ local n="$1"; command -v "$n" 2>/dev/null && return 0; local x; shopt -s nullglob; for x in /usr/local/cuda*/bin/"$n" /opt/nvidia/nsight-systems/*/bin/"$n" /opt/nvidia/nsight-compute/*/"$n" /opt/nvidia/nsight-compute/*/bin/"$n"; do [ -x "$x" ]&&{ printf '%s\n' "$x"; shopt -u nullglob; return 0; }; done; shopt -u nullglob; return 1; }
NVCC="$(find_tool nvcc||true)"; NSYS="$(find_tool nsys||true)"; NCU="$(find_tool ncu||true)"
DMON_PID=''
cleanup(){ [ -n "$DMON_PID" ]&&kill "$DMON_PID" >/dev/null 2>&1||true; DMON_PID=''; }
copy_source(){
  for f in CMakeLists.txt README.md ROADMAP.md; do [ -f "$ROOT/$f" ]&&cp "$ROOT/$f" "$OUT/source/"; done
  for f in include/air/execution.hpp include/air/cuda.hpp include/air/benchmark.hpp src/cuda/cuda_backend.cu src/runtime/backend.cpp src/runtime/serving.cpp research/packed_quantized.hpp research/packed_quantized_tests.cpp research/packed_model_check.cpp docs/OFFLINE_SPRINT2_NUMERICAL_RESEARCH.md docs/OFFLINE_SPRINT3_PACKED_LINEAR_RESEARCH.md docs/AMPERE_TILED_LINEAR_ENGINE.md scripts/analyze-sprint3-tactics.py scripts/validate-performance13-sprint3-machine.sh; do
    [ -f "$ROOT/$f" ]&&cp "$ROOT/$f" "$OUT/source/$(basename "$f")"
  done
}
finish(){
  cleanup; copy_source
  (cd "$(dirname "$OUT")" && zip -qr "$ARCHIVE" "$(basename "$OUT")") || true
  echo; echo '========================================'; echo 'AIR Prompt 13 Sprint-3 evidence archive:'; echo "$ARCHIVE"; echo '========================================'
}
fail(){ echo "$1" | tee "$OUT/FAILURE.txt"; finish; exit 0; }
trap cleanup EXIT

if [ "${AIR_STOP_EXISTING_SERVERS:-0}" = 1 ]; then
  { echo 'Stopping user-owned AIR/llama servers'; pgrep -a -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)'||true; } >"$OUT/stopped-servers.txt" 2>&1
  pkill -TERM -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >/dev/null 2>&1||true; sleep 2
fi

{
 echo harness=AIR-Performance13-Sprint3
 echo date="$(date -Is)"
 echo air_version_expected=0.9.10
 echo benchmark_schema_expected=air.benchmark.v10
 echo manifest_schema_expected=9
 echo strict_atol=0.001
 echo candidate=q5q8-dp4a-hybrid
 echo model_05="$M05"; echo model_15="$M15"; echo model_7="$M7"
 echo rounds="$ROUNDS"; echo seed="$SEED"; echo run_scale_verify="$RUN_SCALE_VERIFY"; echo run_nsys="$RUN_NSYS"; echo run_ncu="$RUN_NCU"
 echo nvcc="${NVCC:-missing}"; echo nsys="${NSYS:-missing}"; echo ncu="${NCU:-missing}"
 echo cuda_backend_sha256="$(sha256sum "$ROOT/src/cuda/cuda_backend.cu"|awk '{print $1}')"
 echo execution_contract_sha256="$(sha256sum "$ROOT/include/air/execution.hpp"|awk '{print $1}')"
 echo packed_math_sha256="$(sha256sum "$ROOT/research/packed_quantized.hpp"|awk '{print $1}')"
 uname -a; nvidia-smi||true
} >"$OUT/preflight.txt" 2>&1

[ -f "$M05" ]||fail "missing required 0.5B model: $M05"
[ -n "$NVCC" ]||fail 'nvcc required for Sprint-3 qualification'
for m in "$M05" "$M15" "$M7"; do [ -f "$m" ]&&sha256sum "$m" >>"$OUT/model-sha256.txt"; done

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=OFF -DAIR_BUILD_RESEARCH=ON >"$OUT/build-release.txt" 2>&1 || fail 'release CMake configure failed'
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-release.txt" 2>&1 || fail 'release CUDA build failed'
ctest --test-dir "$BUILD" --output-on-failure --timeout 180 >"$OUT/ctest-release.txt" 2>&1 || fail 'release/research ctest failed'
BENCH="$BUILD/air-bench"; VERIFY="$BUILD/air-verify"; CLI="$BUILD/air-cli"; PACKCHECK="$BUILD/air-packed-model-check"
{ "$BENCH" --version; "$CLI" cuda-info; "$CLI" inspect "$M05"|sed -n '1,75p'; } >"$OUT/model-and-version.txt" 2>&1

# CPU/property evidence against actual model files. Row-bounded so larger models stay practical.
for spec in "05:$M05" "15:$M15" "7:$M7"; do
  tag="${spec%%:*}"; model="${spec#*:}"; [ -f "$model" ]||continue
  "$PACKCHECK" "$model" "${AIR_PACKED_CHECK_ROWS:-4}" >"$OUT/research/packed-model-$tag.txt" 2>&1 || echo $? >"$OUT/research/packed-model-$tag.exit"
  [ -f "$OUT/research/packed-model-$tag.exit" ]||echo 0 >"$OUT/research/packed-model-$tag.exit"
done

BASE='Adaptive inference runtimes execute transformer layers while tracking memory ownership, scheduling decisions, numerical evidence, and request latency under a controlled workload. '
python3 - "$OUT/prompts" "$BASE" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); b=sys.argv[2]
(p/'p262.txt').write_text(b*10)
(p/'p1042.txt').write_text(b*40)
(p/'decode.txt').write_text('Explain in one compact paragraph why physical batching can improve GPU inference throughput while preserving independent request state.')
PY
TOKENS1=1
TOKENS8="$(python3 - <<'PY'
print(','.join(str(i) for i in range(1,9)))
PY
)"
TOKENS64="$(python3 - <<'PY'
print(','.join(str(i) for i in range(1,65)))
PY
)"

# Strict 0.5B prefill qualification for all current research/control tactics.
declare -A EMIT=( [reuse8]=batch-reuse8 [dense]=dense-f32-cublas [q5q8]=q5q8-dp4a-hybrid )
for width in 1 8 64; do
  token_var="TOKENS${width}"; token_csv="${!token_var}"
  for t in reuse8 dense q5q8; do
    emitted="${EMIT[$t]}"
    "$VERIFY" -m "$M05" --tokens "$token_csv" --generate 2 --top-k 8 --device 0 --atol 0.001 \
      --cuda-prefill-block-linear "$emitted" --cuda-prefill-attention online-softmax \
      --output "$OUT/correctness/${t}-p${width}.json" >"$OUT/correctness/${t}-p${width}.txt" 2>&1
    echo $? >"$OUT/correctness/${t}-p${width}.exit"
  done
done
REUSE_RC="$(cat "$OUT/correctness/reuse8-p64.exit")"; DENSE_RC="$(cat "$OUT/correctness/dense-p64.exit")"; Q5Q8_RC="$(cat "$OUT/correctness/q5q8-p64.exit")"
[ "$REUSE_RC" -eq 0 ]||fail 'reuse8 strict control failed 0.5B correctness gate'
DENSE_QUAL=0; Q5Q8_QUAL=0; [ "$DENSE_RC" -eq 0 ]&&DENSE_QUAL=1; [ "$Q5Q8_RC" -eq 0 ]&&Q5Q8_QUAL=1

# Native decode exact-output qualification. Decode output remains reuse8 intentionally.
for t in reuse8 dense q5q8; do
  [ "$t" = dense ]&&[ "$DENSE_QUAL" -ne 1 ]&&continue
  [ "$t" = q5q8 ]&&[ "$Q5Q8_QUAL" -ne 1 ]&&continue
  emitted="${EMIT[$t]}"
  "$BENCH" -m "$M05" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
    --tokens 24 --warmup 1 --runs 4 --concurrency 4 --prefill-quantum 32 --token-budget 256 \
    --cuda-prefill-block-linear batch-reuse8 --cuda-prefill-attention online-softmax \
    --cuda-decode-block-linear "$emitted" --cuda-decode-output-linear batch-reuse8 \
    --output "$OUT/correctness/decode-$t.json" >"$OUT/correctness/decode-$t.txt" 2>&1
  rc=$?; echo "$rc" >"$OUT/correctness/decode-$t.exit"
  if [ "$rc" -ne 0 ]; then
    [ "$t" = reuse8 ]&&fail 'reuse8 native decode qualification run failed'
    [ "$t" = dense ]&&DENSE_QUAL=0
    [ "$t" = q5q8 ]&&Q5Q8_QUAL=0
  fi
done
python3 - "$OUT/correctness" "$DENSE_QUAL" "$Q5Q8_QUAL" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1]); dense=int(sys.argv[2]); q5=int(sys.argv[3])
base=json.load(open(p/'decode-reuse8.json'))
def validate(name):
    d=json.load(open(p/f'decode-{name}.json'))
    assert d['output_tokens']==base['output_tokens'], f'{name} decode output mismatch'
    assert d['summary']['native_decode_batches']>0, f'{name} did not physically batch'
    assert d['decode_output_linear_tactic']=='batch-reuse8', f'{name} changed decode output tactic'
    assert d['summary']['prepared_artifact_bytes']>0, f'{name} did not materialize prepared state'
for name,ok in [('dense',dense),('q5q8',q5)]:
    if ok:
        try: validate(name)
        except Exception as e:
            print(f'{name}: {e}')
            (p/f'{name}-decode-qualification-failed.txt').write_text(str(e)+'\n')
PY
[ -f "$OUT/correctness/dense-decode-qualification-failed.txt" ]&&DENSE_QUAL=0
[ -f "$OUT/correctness/q5q8-decode-qualification-failed.txt" ]&&Q5Q8_QUAL=0
printf '%s\n' "$DENSE_QUAL" >"$OUT/correctness/dense-qualified.txt"
printf '%s\n' "$Q5Q8_QUAL" >"$OUT/correctness/q5q8-qualified.txt"

# Optional model-scale diagnostics remain noncompetitive. Baseline verifies runtime scaling;
# packed-model-check measures canonical packing + Q8 activation error without large GPU prepared arenas.
if [ "$RUN_SCALE_VERIFY" = 1 ]; then
  for spec in "15:$M15" "7:$M7"; do
    tag="${spec%%:*}"; model="${spec#*:}"; [ -f "$model" ]||continue
    "$VERIFY" -m "$model" --tokens 1 --generate 2 --top-k 8 --device 0 --atol 0.001 \
      --cuda-prefill-block-linear batch-reuse8 --cuda-prefill-attention online-softmax \
      --output "$OUT/scale/reuse8-$tag.json" >"$OUT/scale/reuse8-$tag.txt" 2>&1
    echo $? >"$OUT/scale/reuse8-$tag.exit"
  done
fi

TACTICS=(reuse8); [ "$DENSE_QUAL" -eq 1 ]&&TACTICS+=(dense); [ "$Q5Q8_QUAL" -eq 1 ]&&TACTICS+=(q5q8)
printf '%s\n' "${TACTICS[@]}" >"$OUT/qualified-tactics.txt"
python3 - "$OUT/order.tsv" "$ROUNDS" "$SEED" "${TACTICS[@]}" <<'PY'
import random,sys
out=sys.argv[1]; rounds=int(sys.argv[2]); seed=int(sys.argv[3]); tactics=sys.argv[4:]; rng=random.Random(seed)
with open(out,'w') as f:
  for lane,workloads in [('prefill',['p262','p1042']),('decode',['c8'])]:
    for w in workloads:
      for r in range(1,rounds+1):
        o=tactics[:]; rng.shuffle(o)
        for t in o:f.write(f'{lane}\t{r}\t{w}\t{t}\n')
PY

command -v nvidia-smi >/dev/null 2>&1&&{ nvidia-smi dmon -s pucvmt -d 1 >"$OUT/gpu-dmon.txt" 2>&1 & DMON_PID=$!; }
BENCH_FAIL=0
while IFS=$'\t' read -r lane round workload tactic; do
  [ -n "$lane" ]||continue; emitted="${EMIT[$tactic]}"; stem="$lane-round-$(printf '%02d' "$round")-$workload-$tactic"
  echo "$stem" | tee -a "$OUT/bench/progress.txt"
  if [ "$lane" = prefill ]; then
    "$BENCH" -m "$M05" --backend cuda --no-manifest --prompt-file "$OUT/prompts/$workload.txt" \
      --tokens 4 --warmup 1 --runs 1 --concurrency 1 --prefill-quantum 32 --token-budget 256 \
      --cuda-prefill-block-linear "$emitted" --cuda-prefill-attention online-softmax \
      --cuda-decode-block-linear baseline --cuda-decode-output-linear batch-reuse8 \
      --output "$OUT/bench/$stem.json" >"$OUT/bench/$stem.txt" 2>&1
  else
    "$BENCH" -m "$M05" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
      --tokens 64 --warmup 1 --runs 8 --concurrency 8 --prefill-quantum 32 --token-budget 256 \
      --cuda-prefill-block-linear batch-reuse8 --cuda-prefill-attention online-softmax \
      --cuda-decode-block-linear "$emitted" --cuda-decode-output-linear batch-reuse8 \
      --output "$OUT/bench/$stem.json" >"$OUT/bench/$stem.txt" 2>&1
  fi
  rc=$?; echo "$rc">"$OUT/bench/$stem.exit"; [ "$rc" -ne 0 ]&&BENCH_FAIL=1; sleep "$COOLDOWN"
done <"$OUT/order.tsv"
cleanup

ANALYZER_RC=0
python3 "$ROOT/scripts/analyze-sprint3-tactics.py" "$OUT/bench" --json "$OUT/sprint3-summary.json" --markdown "$OUT/sprint3-summary.md" >"$OUT/analyzer.txt" 2>&1 || ANALYZER_RC=$?

# Nsight Systems attribution for every qualified tactic, using identical production-path workloads.
if [ "$RUN_NSYS" = 1 ] && [ -n "$NSYS" ]; then
  cmake -S "$ROOT" -B "$PROFILE_BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=ON -DAIR_BUILD_RESEARCH=ON >"$OUT/build-profile.txt" 2>&1
  if [ $? -eq 0 ]; then cmake --build "$PROFILE_BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-profile.txt" 2>&1; fi
  PB="$PROFILE_BUILD/air-bench"
  if [ -x "$PB" ]; then
    for t in "${TACTICS[@]}"; do emitted="${EMIT[$t]}"
      for lane in prefill decode; do
        prefix="$OUT/nsys/$lane-$t"
        if [ "$lane" = prefill ]; then
          "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o "$prefix" \
            "$PB" -m "$M05" --backend cuda --no-manifest --prompt-file "$OUT/prompts/p1042.txt" \
            --tokens 2 --warmup 1 --runs 1 --concurrency 1 --prefill-quantum 32 --token-budget 256 \
            --cuda-prefill-block-linear "$emitted" --cuda-prefill-attention online-softmax --cuda-decode-output-linear batch-reuse8 \
            --output "$prefix-bench.json" >"$prefix-profile.txt" 2>&1
        else
          "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o "$prefix" \
            "$PB" -m "$M05" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
            --tokens 64 --warmup 1 --runs 8 --concurrency 8 --prefill-quantum 32 --token-budget 256 \
            --cuda-prefill-block-linear batch-reuse8 --cuda-prefill-attention online-softmax \
            --cuda-decode-block-linear "$emitted" --cuda-decode-output-linear batch-reuse8 \
            --output "$prefix-bench.json" >"$prefix-profile.txt" 2>&1
        fi
        x=$?; echo "$x">"$prefix-profile.exit"
        if [ "$x" -eq 0 ] && [ -f "$prefix.nsys-rep" ]; then
          "$NSYS" stats --report nvtx_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum --report cuda_api_sum "$prefix.nsys-rep" >"$prefix-stats.txt" 2>&1 || true
        fi
      done
    done
  fi
elif [ "$RUN_NSYS" = 1 ]; then echo 'Nsight Systems unavailable' >"$OUT/nsys/MISSING.txt"; else echo 'Nsight Systems disabled' >"$OUT/nsys/SKIPPED.txt"; fi

# Optional targeted Nsight Compute. Diagnostic only, never a product-qualification gate.
if [ "$RUN_NCU" = 1 ] && [ "$Q5Q8_QUAL" -eq 1 ] && [ -n "$NCU" ]; then
  PB="${PROFILE_BUILD}/air-bench"; [ -x "$PB" ]||PB="$BENCH"
  "$NCU" --set speed-of-light --target-processes all --kernel-name-base demangled \
    --kernel-name regex:q5q8_dp4a_hybrid_matmul_kernel --launch-count 4 --force-overwrite \
    -o "$OUT/ncu/q5q8-dp4a" \
    "$PB" -m "$M05" --backend cuda --no-manifest --prompt-file "$OUT/prompts/p1042.txt" \
    --tokens 1 --warmup 0 --runs 1 --concurrency 1 --prefill-quantum 32 --token-budget 256 \
    --cuda-prefill-block-linear q5q8-dp4a-hybrid --cuda-prefill-attention online-softmax \
    --cuda-decode-output-linear batch-reuse8 --output "$OUT/ncu/q5q8-bench.json" >"$OUT/ncu/q5q8.txt" 2>&1 || true
elif [ "$RUN_NCU" = 1 ]; then echo 'NCU unavailable or q5q8 candidate unqualified' >"$OUT/ncu/SKIPPED.txt"; else echo 'NCU disabled' >"$OUT/ncu/SKIPPED.txt"; fi

{
 echo benchmark_failure="$BENCH_FAIL"
 echo analyzer_rc="$ANALYZER_RC"
 echo dense_correctness_qualified="$DENSE_QUAL"
 echo q5q8_correctness_qualified="$Q5Q8_QUAL"
 echo tactics="$(IFS=,;echo "${TACTICS[*]}")"
 echo strict_gate_atol=0.001
 echo note='Competitive timing occurs only for tactics that pass strict prefill correctness plus exact native-decode output qualification. q5q8-dp4a-hybrid uses DP4A only for Q5_0/Q8_0 block matrices; Q4_K/Q6_K intentionally remain reuse8.'
} >"$OUT/RESULT.txt"
finish
trap - EXIT
exit 0
