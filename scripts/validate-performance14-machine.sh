#!/usr/bin/env bash
# AIR Prompt 14/16 Strategy Lab WolfCat qualification collector.
# Measures correctness, cold preparation, steady-state Pareto evidence, actual
# product-path eviction, then validates the final Strategy Lab manifest.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Performance14-$STAMP"
ARCHIVE="$HOME/Downloads/AIR-Performance14-$STAMP.zip"
BUILD="${AIR_P14_BUILD_DIR:-$ROOT/build-performance14}"
PROFILE_BUILD="${AIR_P14_PROFILE_BUILD_DIR:-$ROOT/build-performance14-profile}"
MODEL="${AIR_MODEL_05:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
ROUNDS="${AIR_P14_ROUNDS:-5}"
TRANSITION_ROUNDS="${AIR_P14_TRANSITION_ROUNDS:-5}"
COOLDOWN="${AIR_BENCH_COOLDOWN_SECONDS:-2}"
SEED="${AIR_P14_SEED:-141011}"
RUN_NSYS="${AIR_RUN_NSYS:-1}"
mkdir -p "$OUT"/{correctness,cold-preparation,bench,transition,adaptive,nsys,prompts,source,telemetry}

find_tool(){ local n="$1"; command -v "$n" 2>/dev/null && return 0; local x; shopt -s nullglob; for x in /usr/local/cuda*/bin/"$n" /opt/nvidia/nsight-systems/*/bin/"$n"; do [ -x "$x" ]&&{ printf '%s\n' "$x"; shopt -u nullglob; return 0; }; done; shopt -u nullglob; return 1; }
NVCC="$(find_tool nvcc||true)"; NSYS="$(find_tool nsys||true)"; DMON_PID=''
cleanup(){ [ -n "$DMON_PID" ]&&kill "$DMON_PID" >/dev/null 2>&1||true; DMON_PID=''; }
copy_source(){
  for f in CMakeLists.txt README.md ROADMAP.md; do [ -f "$ROOT/$f" ]&&cp "$ROOT/$f" "$OUT/source/"; done
  for f in include/air/manifest.hpp include/air/runtime.hpp include/air/serving.hpp include/air/benchmark.hpp src/runtime/manifest.cpp src/runtime/serving.cpp src/benchmark/benchmark.cpp src/server/protocol.cpp src/strategy_probe/main.cpp tests/strategy_lab_tests.cpp scripts/analyze-strategy-lab.py scripts/validate-performance14-machine.sh; do
    [ -f "$ROOT/$f" ]&&cp "$ROOT/$f" "$OUT/source/$(basename "$f")"
  done
}
finish(){
  cleanup; copy_source
  (cd "$(dirname "$OUT")" && zip -qr "$ARCHIVE" "$(basename "$OUT")") || true
  echo; echo '========================================'; echo 'AIR Prompt 14 evidence archive:'; echo "$ARCHIVE"; echo '========================================'
}
fail(){ echo "$1" | tee "$OUT/FAILURE.txt"; finish; exit 0; }
trap cleanup EXIT

if [ "${AIR_STOP_EXISTING_SERVERS:-0}" = 1 ]; then
  { echo 'Stopping user-owned AIR/llama servers'; pgrep -a -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)'||true; } >"$OUT/stopped-servers.txt" 2>&1
  pkill -TERM -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >/dev/null 2>&1||true; sleep 2
fi

{
 echo harness=AIR-Performance14; echo date="$(date -Is)"; echo air_version_expected=0.9.11
 echo benchmark_schema_expected=air.benchmark.v11; echo manifest_schema_expected=10; echo strict_atol=0.001
 echo rounds="$ROUNDS"; echo transition_rounds="$TRANSITION_ROUNDS"; echo seed="$SEED"
 echo model="$MODEL"; echo nvcc="${NVCC:-missing}"; echo nsys="${NSYS:-missing}"
 uname -a; nvidia-smi||true
} >"$OUT/preflight.txt" 2>&1
[ -f "$MODEL" ]||fail "missing required model: $MODEL"
[ -n "$NVCC" ]||fail 'nvcc required for Prompt 14 qualification'
sha256sum "$MODEL" >"$OUT/model-sha256.txt"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=OFF -DAIR_BUILD_RESEARCH=ON >"$OUT/build-release.txt" 2>&1 || fail 'release CMake configure failed'
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-release.txt" 2>&1 || fail 'release CUDA build failed'
ctest --test-dir "$BUILD" --output-on-failure --timeout 180 >"$OUT/ctest-release.txt" 2>&1 || fail 'complete research ctest failed'
BENCH="$BUILD/air-bench"; VERIFY="$BUILD/air-verify"; CLI="$BUILD/air-cli"; PROBE="$BUILD/air-strategy-probe"
for x in "$BENCH" "$VERIFY" "$CLI" "$PROBE"; do [ -x "$x" ]||fail "missing built executable: $x"; done
"$CLI" fingerprint "$MODEL" >"$OUT/fingerprint.json" 2>"$OUT/fingerprint.stderr" || fail 'fingerprint failed'
{ "$BENCH" --version; "$CLI" cuda-info; "$CLI" inspect "$MODEL" | sed -n '1,80p'; } >"$OUT/model-and-version.txt" 2>&1

BASE='Adaptive inference runtimes execute transformer layers while tracking memory ownership, scheduling decisions, numerical evidence, and request latency under a controlled workload. '
python3 - "$OUT/prompts" "$BASE" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); b=sys.argv[2]
(p/'p54.txt').write_text(b*2)
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

# 1. Strict correctness before any competitive timing.
declare -A EMIT=( [reuse8]=batch-reuse8 [dense]=dense-f32-cublas )
for width in 1 8 64; do
  var="TOKENS${width}"; csv="${!var}"
  for t in reuse8 dense; do
    "$VERIFY" -m "$MODEL" --tokens "$csv" --generate 2 --top-k 8 --device 0 --atol 0.001 \
      --cuda-prefill-block-linear "${EMIT[$t]}" --cuda-prefill-attention online-softmax \
      --output "$OUT/correctness/${t}-p${width}.json" >"$OUT/correctness/${t}-p${width}.txt" 2>&1
    echo $? >"$OUT/correctness/${t}-p${width}.exit"
  done
done
for t in reuse8 dense; do
  "$BENCH" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
    --tokens 24 --warmup 1 --runs 4 --concurrency 4 --prefill-quantum 32 --token-budget 256 \
    --cuda-prefill-block-linear batch-reuse8 --cuda-prefill-attention online-softmax \
    --cuda-decode-block-linear "${EMIT[$t]}" --cuda-decode-output-linear batch-reuse8 \
    --output "$OUT/correctness/decode-$t.json" >"$OUT/correctness/decode-$t.txt" 2>&1
  echo $? >"$OUT/correctness/decode-$t.exit"
done
python3 - "$OUT/correctness" <<'PY' || fail 'strict tactic qualification failed'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1])
for t in ('reuse8','dense'):
    for w in (1,8,64):
        assert (p/f'{t}-p{w}.exit').read_text().strip()=='0', f'{t} p{w} strict gate failed'
    assert (p/f'decode-{t}.exit').read_text().strip()=='0', f'{t} decode gate failed'
base=json.load(open(p/'decode-reuse8.json')); dense=json.load(open(p/'decode-dense.json'))
assert dense['output_tokens']==base['output_tokens'], 'dense native decode output mismatch'
assert dense['summary']['native_decode_batches']>0 and base['summary']['native_decode_batches']>0, 'physical native batching missing'
assert dense['decode_output_linear_tactic']=='batch-reuse8'
PY

# GPU telemetry covers transition and steady-state phases.
nvidia-smi dmon -s pucvmt -d 1 >"$OUT/telemetry/nvidia-dmon.txt" 2>&1 & DMON_PID=$!

# 2. Cold dense preparation, one fresh process/request per sample, no warmup.
for r in $(seq -w 1 "$ROUNDS"); do
  "$BENCH" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/p1042.txt" \
    --tokens 2 --warmup 0 --runs 1 --concurrency 1 --prefill-quantum 32 --token-budget 256 \
    --cuda-prefill-block-linear dense-f32-cublas --cuda-decode-block-linear dense-f32-cublas \
    --cuda-decode-output-linear batch-reuse8 --cuda-prefill-attention online-softmax \
    --output "$OUT/cold-preparation/dense-cold-$r.json" >"$OUT/cold-preparation/dense-cold-$r.txt" 2>&1 || fail "cold preparation round $r failed"
  sleep "$COOLDOWN"
done

# 3. Steady-state randomized paired evidence. Small lane intentionally has only reuse8.
python3 - "$OUT/order.tsv" "$ROUNDS" "$SEED" <<'PY'
import random,sys
out,n,seed=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); rng=random.Random(seed)
rows=[]
for r in range(1,n+1):
    rows.append((r,'p54','reuse8'))
    for w in ('p262','p1042','c8'):
        order=['reuse8','dense']; rng.shuffle(order)
        for t in order: rows.append((r,w,t))
open(out,'w').write(''.join(f'{r}\t{w}\t{t}\n' for r,w,t in rows))
PY
while IFS=$'\t' read -r round workload tactic; do
  emitted="${EMIT[$tactic]}"; rr="$(printf '%02d' "$round")"
  if [ "$workload" = c8 ]; then
    stem="decode-round-$rr-c8-$tactic"
    "$BENCH" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/decode.txt" \
      --tokens 64 --warmup 1 --runs 8 --concurrency 8 --prefill-quantum 32 --token-budget 256 \
      --cuda-prefill-block-linear batch-reuse8 --cuda-prefill-attention online-softmax \
      --cuda-decode-block-linear "$emitted" --cuda-decode-output-linear batch-reuse8 \
      --output "$OUT/bench/$stem.json" >"$OUT/bench/$stem.txt" 2>&1 || fail "$stem failed"
  else
    stem="prefill-round-$rr-$workload-$tactic"
    "$BENCH" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/$workload.txt" \
      --tokens 4 --warmup 1 --runs 1 --concurrency 1 --prefill-quantum 32 --token-budget 256 \
      --cuda-prefill-block-linear "$emitted" --cuda-decode-block-linear "$emitted" \
      --cuda-decode-output-linear batch-reuse8 --cuda-prefill-attention online-softmax \
      --output "$OUT/bench/$stem.json" >"$OUT/bench/$stem.txt" 2>&1 || fail "$stem failed"
  fi
  sleep "$COOLDOWN"
done <"$OUT/order.tsv"

# Build bootstrap manifest: prep measured, eviction deliberately unknown for dense.
python3 "$ROOT/scripts/analyze-strategy-lab.py" "$OUT" --seed "$SEED" \
  --manifest-out "$OUT/bootstrap-manifest.json" --summary-json "$OUT/bootstrap-summary.json" --summary-md "$OUT/bootstrap-summary.md" || fail 'bootstrap manifest generation failed'

# 4. First actual dense -> reuse eviction measurement through InferenceService.
for r in $(seq -w 1 "$TRANSITION_ROUNDS"); do
  "$PROBE" -m "$MODEL" --manifest "$OUT/bootstrap-manifest.json" \
    --medium-prompt-file "$OUT/prompts/p1042.txt" --small-prompt-file "$OUT/prompts/p54.txt" \
    --scenario eviction --horizon-tokens 10000 --qualification-allow-unknown-eviction \
    --output "$OUT/transition/eviction-$r.json" >"$OUT/transition/eviction-$r.txt" 2>&1 || fail "qualification eviction round $r failed"
  sleep "$COOLDOWN"
done

# Final manifest now contains measured eviction evidence.
python3 "$ROOT/scripts/analyze-strategy-lab.py" "$OUT" --seed "$SEED" --eviction-dir "$OUT/transition" \
  --manifest-out "$OUT/final-manifest.json" --summary-json "$OUT/strategy-summary.json" --summary-md "$OUT/strategy-summary.md" || fail 'final manifest generation failed'

# 5. Exercise actual Strategy Lab product path with final evidence.
# Minimum VRAM.
"$BENCH" -m "$MODEL" --backend auto --manifest "$OUT/final-manifest.json" --require-manifest --prompt-file "$OUT/prompts/p1042.txt" \
  --tokens 4 --warmup 0 --runs 1 --concurrency 1 --strategy-objective minimum-vram --strategy-horizon-tokens 10000 \
  --output "$OUT/adaptive/minimum-vram.json" >"$OUT/adaptive/minimum-vram.txt" 2>&1 || fail 'minimum-vram adaptive probe failed'
# Explicit budget below dense residency.
"$BENCH" -m "$MODEL" --backend auto --manifest "$OUT/final-manifest.json" --require-manifest --prompt-file "$OUT/prompts/p1042.txt" \
  --tokens 4 --warmup 0 --runs 1 --concurrency 1 --strategy-objective maximum-throughput --strategy-horizon-tokens 10000 \
  --strategy-prepared-memory-budget-bytes 536870912 --output "$OUT/adaptive/limited-vram.json" >"$OUT/adaptive/limited-vram.txt" 2>&1 || fail 'limited-vram adaptive probe failed'
# Cold long-work dense selection.
"$BENCH" -m "$MODEL" --backend auto --manifest "$OUT/final-manifest.json" --require-manifest --prompt-file "$OUT/prompts/p1042.txt" \
  --tokens 4 --warmup 0 --runs 1 --concurrency 1 --strategy-objective maximum-throughput --strategy-horizon-tokens 10000 \
  --output "$OUT/adaptive/cold-long.json" >"$OUT/adaptive/cold-long.txt" 2>&1 || fail 'cold long adaptive probe failed'
# Short exact small region.
"$BENCH" -m "$MODEL" --backend auto --manifest "$OUT/final-manifest.json" --require-manifest --prompt-file "$OUT/prompts/p54.txt" \
  --tokens 4 --warmup 0 --runs 1 --concurrency 1 --strategy-objective maximum-throughput \
  --output "$OUT/adaptive/short-cold.json" >"$OUT/adaptive/short-cold.txt" 2>&1 || fail 'short adaptive probe failed'
# Hot-state behavior and final real eviction with qualification override disabled.
"$PROBE" -m "$MODEL" --manifest "$OUT/final-manifest.json" --medium-prompt-file "$OUT/prompts/p1042.txt" --small-prompt-file "$OUT/prompts/p54.txt" \
  --scenario hot-dense --horizon-tokens 10000 --output "$OUT/adaptive/hot-dense.json" >"$OUT/adaptive/hot-dense.txt" 2>&1 || fail 'hot dense probe failed'
"$PROBE" -m "$MODEL" --manifest "$OUT/final-manifest.json" --medium-prompt-file "$OUT/prompts/p1042.txt" --small-prompt-file "$OUT/prompts/p54.txt" \
  --scenario eviction --horizon-tokens 10000 --output "$OUT/adaptive/final-eviction.json" >"$OUT/adaptive/final-eviction.txt" 2>&1 || fail 'final product eviction probe failed'

python3 - "$OUT" <<'PY' || fail 'adaptive Strategy Lab behavior validation failed'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1])
def j(x): return json.load(open(p/'adaptive'/x))
minv=j('minimum-vram.json'); budget=j('limited-vram.json'); cold=j('cold-long.json'); small=j('short-cold.json')
hot=j('hot-dense.json'); ev=j('final-eviction.json')
assert minv['runs'][0]['strategy_id'].startswith('reuse8-p1042')
assert budget['runs'][0]['strategy_id'].startswith('reuse8-p1042')
assert any(c['strategy_id'].startswith('dense-p1042') and c['disposition']=='rejected:memory-infeasible' for c in budget['runs'][0]['strategy_candidates'])
assert cold['runs'][0]['strategy_id'].startswith('dense-p1042') and cold['runs'][0]['plan_preparation_ms']>0
assert small['runs'][0]['strategy_id'].startswith('reuse8-p54')
assert hot['first']['strategy_id'].startswith('dense-p1042') and hot['second']['strategy_id'].startswith('dense-p1042')
assert hot['second']['strategy_prepared_state_hot'] is True and hot['second']['plan_preparation_ms'] < 1.0
assert ev['first']['strategy_id'].startswith('dense-p1042') and ev['second']['strategy_id'].startswith('reuse8-p54')
assert ev['second']['plan_eviction_ms']>0
assert ev['qualification_allow_unknown_eviction'] is False
# decode_output isolation is preserved by manifest itself
mf=json.load(open(p/'final-manifest.json')); assert all(s['decode_output_quantized_linear']=='batch-reuse8' for s in mf['strategies'])
PY

# Optional Nsight Systems on the actual adaptive cold-long product path.
if [ "$RUN_NSYS" = 1 ] && [ -n "$NSYS" ]; then
  cmake -S "$ROOT" -B "$PROFILE_BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=ON -DAIR_BUILD_RESEARCH=ON >"$OUT/build-profile.txt" 2>&1
  if [ $? -eq 0 ]; then cmake --build "$PROFILE_BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-profile.txt" 2>&1; fi
  PB="$PROFILE_BUILD/air-bench"
  if [ -x "$PB" ]; then
    prefix="$OUT/nsys/adaptive-dense-p1042"
    "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o "$prefix" \
      "$PB" -m "$MODEL" --backend auto --manifest "$OUT/final-manifest.json" --require-manifest --prompt-file "$OUT/prompts/p1042.txt" \
      --tokens 4 --warmup 0 --runs 1 --concurrency 1 --strategy-objective maximum-throughput --strategy-horizon-tokens 10000 \
      --output "$prefix-bench.json" >"$prefix-profile.txt" 2>&1 || true
    [ -f "$prefix.nsys-rep" ] && "$NSYS" stats --report nvtx_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum --report cuda_api_sum "$prefix.nsys-rep" >"$prefix-stats.txt" 2>&1 || true
  fi
else
  echo 'Nsight Systems unavailable or disabled' >"$OUT/nsys/SKIPPED.txt"
fi

cleanup
{
 echo "git_head=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null||echo no-git)"
 find "$ROOT/include/air" "$ROOT/src/runtime" "$ROOT/src/benchmark" "$ROOT/src/strategy_probe" -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum
} >"$OUT/source-sha256.txt" 2>&1
finish
