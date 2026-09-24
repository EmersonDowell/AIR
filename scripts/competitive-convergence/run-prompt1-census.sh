#!/usr/bin/env bash
set +e
set +u
set +E
set +o pipefail 2>/dev/null

ROOT="${AIR_ROOT:-$HOME/Projects/AIR}"
BUILD="${AIR_CC1_BUILD:-$ROOT/build-cc1}"
MODEL="${AIR_CC1_MODEL:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
MANIFEST="${AIR_CC1_MANIFEST:-$ROOT/controls/p14-final-manifest-v10.json}"
ROUNDS="${AIR_CC1_ROUNDS:-5}"
TOKENS="${AIR_CC1_TOKENS:-32}"
PORT="${AIR_CC1_PORT:-18810}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/Downloads/AIR-CC1-Census-$STAMP"

mkdir -p "$OUT"/{source,build,hardware,prompts,bench,nsys,multi,telemetry,analysis}
cd "$ROOT" || {
  echo "AIR root missing: $ROOT"
  return 2 2>/dev/null || true
}

ts(){ date '+%H:%M:%S'; }

gpu_line(){
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm \
      --format=csv,noheader 2>/dev/null | head -n1
  fi
}

run_stage(){
  local label="$1" log="$2"
  shift 2
  mkdir -p "$(dirname "$log")"
  local start pid rc elapsed now newest mtime size file gpu
  start="$(date +%s)"
  echo
  echo "[$(ts)] START $label"
  printf 'command:' >"$log.command"
  printf ' %q' "$@" >>"$log.command"
  printf '\n' >>"$log.command"

  "$@" >"$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
    kill -0 "$pid" 2>/dev/null || break
    now="$(date +%s)"
    elapsed=$((now-start))
    echo "[$(ts)] ACTIVE $label elapsed=${elapsed}s pid=$pid"
    ps -p "$pid" -o pid=,ppid=,etime=,stat=,%cpu=,%mem=,cmd= 2>/dev/null | sed 's/^/  proc: /'
    gpu="$(gpu_line)"
    [ -n "$gpu" ] && echo "  GPU: $gpu"
    newest="$(find "$OUT" -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -n1)"
    if [ -n "$newest" ]; then
      mtime="$(echo "$newest" | awk '{print int($1)}')"
      size="$(echo "$newest" | awk '{print $2}')"
      file="$(echo "$newest" | cut -d' ' -f3-)"
      echo "  newest: $file size=${size}B age=$((now-mtime))s"
    fi
    [ -s "$log" ] && { echo "  log tail:"; tail -n 3 "$log" | sed 's/^/    /'; }
  done
  wait "$pid"; rc=$?
  elapsed=$(( $(date +%s)-start ))
  echo "$rc" >"$log.exit"
  if [ "$rc" -eq 0 ]; then
    echo "[$(ts)] DONE  $label rc=0 elapsed=${elapsed}s"
  else
    echo "[$(ts)] FAIL  $label rc=$rc elapsed=${elapsed}s"
    [ -s "$log" ] && tail -n 20 "$log" | sed 's/^/  /'
  fi
  return "$rc"
}

finish(){
  python3 "$ROOT/scripts/competitive-convergence/prompt1_analyze.py" "$OUT" \
    >"$OUT/analysis/analyzer.txt" 2>&1 || true
  nvidia-smi -q >"$OUT/telemetry/nvidia-smi-q-final.txt" 2>&1 || true
  ps -eo pid,ppid,etime,stat,%cpu,%mem,cmd >"$OUT/telemetry/processes-final.txt" 2>&1 || true
  echo
  echo "[$(ts)] START Packaging Prompt-1 evidence"
  cd "$HOME/Downloads" || return 0 2>/dev/null || true
  local zip="$HOME/Downloads/AIR-CC1-Census-$STAMP.zip"
  zip -qr "$zip" "$(basename "$OUT")"
  echo "[$(ts)] DONE  Packaging Prompt-1 evidence"
  echo "$zip"
  sha256sum "$zip"
}

echo "AIR Competitive Convergence Prompt 1/5"
echo "output=$OUT"
echo "AIR=$ROOT"
echo

# Source fingerprint and context gate.
find CMakeLists.txt include src tests -type f -print0 2>/dev/null \
  | sort -z | xargs -0 sha256sum >"$OUT/source/source-sha256.txt" 2>&1
git status --short >"$OUT/source/git-status.txt" 2>&1 || true
git diff -- . ':!build*' >"$OUT/source/git-diff.patch" 2>&1 || true
bash scripts/check-agent-context.sh >"$OUT/source/agent-context-check.txt" 2>&1 || true

{
  echo "generate_cohort:"
  grep -RIn 'generate_cohort' include/air/serving.hpp src/runtime/serving.cpp || true
  echo "operation-scoped decode:"
  grep -RIn 'decode_output' include/air/cuda.hpp src/cuda/cuda_backend.cu src/runtime/backend.cpp || true
} >"$OUT/source/r7-contract-markers.txt"

# Build and tests.
run_stage "Configure current AIR" "$OUT/build/configure.txt" \
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=OFF -DAIR_BUILD_RESEARCH=ON
CONFIG_RC=$?
if [ "$CONFIG_RC" -ne 0 ]; then finish; exit 0; fi

run_stage "Build current AIR" "$OUT/build/build.txt" \
  cmake --build "$BUILD" -j "${JOBS:-2}"
BUILD_RC=$?
if [ "$BUILD_RC" -ne 0 ]; then finish; exit 0; fi

run_stage "Complete ctest" "$OUT/build/ctest.txt" \
  ctest --test-dir "$BUILD" --output-on-failure --timeout 300
CTEST_RC=$?
if [ "$CTEST_RC" -ne 0 ]; then finish; exit 0; fi

CLI="$BUILD/air-cli"
BENCH="$BUILD/air-bench"
SERVER="$BUILD/air-server"

"$CLI" --version >"$OUT/source/air-version.txt" 2>&1 || true
"$CLI" fingerprint "$MODEL" >"$OUT/source/model-fingerprint.json" 2>"$OUT/source/model-fingerprint.stderr" || true
sha256sum "$MODEL" >"$OUT/source/model-sha256.txt" 2>&1 || true
sha256sum "$MANIFEST" >"$OUT/source/manifest-sha256.txt" 2>&1 || true

# Exact prompt contract from the already-qualified Prompt-15 helper.
if [ -f scripts/p15_prompt_contract.py ]; then
  run_stage "Exact prompt synthesis" "$OUT/prompts/make.txt" \
    python3 scripts/p15_prompt_contract.py make \
      --air-cli "$CLI" --model "$MODEL" --output-dir "$OUT/prompts" \
      --targets 54,160,262,1042
  run_stage "Exact prompt verification" "$OUT/prompts/verify.txt" \
    python3 scripts/p15_prompt_contract.py verify \
      --air-cli "$CLI" --model "$MODEL" --output-dir "$OUT/prompts" \
      --targets 54,160,262,1042
else
  echo "Prompt contract helper missing." >"$OUT/prompts/MISSING.txt"
  finish
  exit 0
fi

# Hardware raw facts.
lscpu -J >"$OUT/hardware/lscpu.json" 2>&1 || lscpu >"$OUT/hardware/lscpu.txt" 2>&1 || true
cat /proc/cpuinfo >"$OUT/hardware/cpuinfo.txt" 2>&1 || true
cat /proc/meminfo >"$OUT/hardware/meminfo.txt" 2>&1 || true
numactl --hardware >"$OUT/hardware/numa.txt" 2>&1 || true
lsblk -J -o NAME,PATH,TYPE,SIZE,ROTA,TRAN,MOUNTPOINTS,FSTYPE,MODEL \
  >"$OUT/hardware/lsblk.json" 2>&1 || true
findmnt -J -T "$ROOT" >"$OUT/hardware/findmnt-air.json" 2>&1 || true
nvidia-smi -q >"$OUT/hardware/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi topo -m >"$OUT/hardware/nvidia-topo.txt" 2>&1 || true

run_stage "Compile host-memory probe" "$OUT/hardware/host-probe-build.txt" \
  g++ -std=c++20 -O3 -pthread -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
    research/prompt1_host_memory_probe.cpp -o "$OUT/hardware/host-probe"
if [ -x "$OUT/hardware/host-probe" ]; then
  run_stage "Host RAM bandwidth/thread scaling" "$OUT/hardware/host-memory.json" \
    "$OUT/hardware/host-probe" "${AIR_CC1_HOST_PROBE_MIB:-256}" "${AIR_CC1_HOST_PROBE_REPEATS:-4}"
fi

if command -v nvcc >/dev/null 2>&1; then
  run_stage "Compile CUDA hardware probe" "$OUT/hardware/cuda-probe-build.txt" \
    nvcc -std=c++20 -O3 research/prompt1_cuda_hardware_probe.cu \
      -o "$OUT/hardware/cuda-probe"
  if [ -x "$OUT/hardware/cuda-probe" ]; then
    run_stage "CUDA transfer/launch/allocation probe" "$OUT/hardware/cuda-hardware.json" \
      "$OUT/hardware/cuda-probe" 0 "${AIR_CC1_CUDA_PROBE_MIB:-64}" "${AIR_CC1_CUDA_PROBE_REPEATS:-20}"
  fi
else
  echo "nvcc unavailable" >"$OUT/hardware/cuda-probe.SKIPPED"
fi

run_stage "Storage read/capability probe" "$OUT/hardware/storage-probe.txt" \
  python3 scripts/competitive-convergence/prompt1_storage_probe.py \
    --path "$MODEL" --output "$OUT/hardware/storage.json" \
    --max-mib "${AIR_CC1_STORAGE_PROBE_MIB:-256}"

python3 scripts/competitive-convergence/prompt1_build_topology.py \
  --host "$OUT/hardware/host-memory.json" \
  --cuda "$OUT/hardware/cuda-hardware.json" \
  --storage "$OUT/hardware/storage.json" \
  --output "$OUT/hardware/hardware-topology.json" \
  >"$OUT/hardware/topology-build.txt" 2>&1 || true

REUSE=(--backend cuda --no-manifest
  --cuda-prefill-block-linear reuse8
  --cuda-decode-block-linear reuse8
  --cuda-decode-output-linear reuse8
  --cuda-prefill-attention online-softmax)

DENSE=(--backend cuda --no-manifest
  --cuda-prefill-block-linear dense-f32-cublas
  --cuda-decode-block-linear dense-f32-cublas
  --cuda-decode-output-linear reuse8
  --cuda-prefill-attention online-softmax)

bench_cell(){
  local name="$1" prompt="$2" concurrency="$3"
  shift 3
  local runs="$ROUNDS"
  if [ "$concurrency" -gt 1 ]; then runs=$((ROUNDS*concurrency)); fi
  run_stage "Benchmark $name" "$OUT/bench/$name.txt" \
    "$BENCH" -m "$MODEL" --prompt-file "$prompt" \
      --tokens "$TOKENS" --warmup 1 --runs "$runs" \
      --concurrency "$concurrency" "$@" --output "$OUT/bench/$name.json"
}

bench_cell "p54-c1-reuse8" "$OUT/prompts/p54.txt" 1 "${REUSE[@]}"
bench_cell "p262-c1-reuse8" "$OUT/prompts/p262.txt" 1 "${REUSE[@]}"
bench_cell "p1042-c1-reuse8" "$OUT/prompts/p1042.txt" 1 "${REUSE[@]}"
bench_cell "p262-c1-dense" "$OUT/prompts/p262.txt" 1 "${DENSE[@]}"
bench_cell "p1042-c1-dense" "$OUT/prompts/p1042.txt" 1 "${DENSE[@]}"

# Exact qualified cohort through adaptive production path.
run_stage "Benchmark exact p160/c8 adaptive cohort" "$OUT/bench/p160-c8-adaptive.txt" \
  "$BENCH" -m "$MODEL" --backend auto --manifest "$MANIFEST" --require-manifest \
    --prompt-file "$OUT/prompts/p160.txt" --tokens "$TOKENS" --warmup 0 \
    --runs $((ROUNDS*8)) --concurrency 8 \
    --strategy-objective maximum-throughput --strategy-horizon-tokens 10000 \
    --output "$OUT/bench/p160-c8-adaptive.json"

# Prefill quantum headroom, static reuse8 production path.
for q in 32 64 128; do
  bench_cell "p262-c1-reuse8-q$q" "$OUT/prompts/p262.txt" 1 \
    "${REUSE[@]}" --prefill-quantum "$q" --token-budget 512
  bench_cell "p1042-c1-reuse8-q$q" "$OUT/prompts/p1042.txt" 1 \
    "${REUSE[@]}" --prefill-quantum "$q" --token-budget 512
done

# Current Nsight Systems traces. Nsight Compute is deliberately deferred until
# Systems identifies the actual important kernels.
if command -v nsys >/dev/null 2>&1; then
  profile_cell(){
    local name="$1" prompt="$2"
    shift 2
    run_stage "Nsight Systems $name" "$OUT/nsys/$name-profile.txt" \
      nsys profile --force-overwrite=true --sample=none --trace=cuda,osrt \
        --output="$OUT/nsys/$name" \
        "$BENCH" -m "$MODEL" --prompt-file "$prompt" \
          --tokens 8 --warmup 0 --runs 1 --concurrency 1 "$@"
    if [ ! -f "$OUT/nsys/$name.nsys-rep" ]; then
      echo "Required Nsight report missing: $OUT/nsys/$name.nsys-rep" >"$OUT/nsys/$name-stats.FAIL"
      return 1
    fi
    nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum \
      "$OUT/nsys/$name.nsys-rep" >"$OUT/nsys/$name-stats.txt" 2>&1
    local stats_rc=$?
    if [ "$stats_rc" -ne 0 ]; then
      echo "nsys stats failed rc=$stats_rc" >"$OUT/nsys/$name-stats.FAIL"
      return "$stats_rc"
    fi
    if ! grep -q 'CUDA GPU Kernel Summary' "$OUT/nsys/$name-stats.txt" || \
       ! grep -q 'CUDA API Summary' "$OUT/nsys/$name-stats.txt"; then
      echo "Required CUDA kernel/API summaries missing after forced export." >"$OUT/nsys/$name-stats.FAIL"
      return 1
    fi
    return 0
  }

  NSYS_STATS_FAILED=0
  profile_cell "p54-reuse8" "$OUT/prompts/p54.txt" "${REUSE[@]}" || NSYS_STATS_FAILED=1
  profile_cell "p1042-reuse8" "$OUT/prompts/p1042.txt" "${REUSE[@]}" || NSYS_STATS_FAILED=1
  profile_cell "p1042-dense" "$OUT/prompts/p1042.txt" "${DENSE[@]}" || NSYS_STATS_FAILED=1
  if [ "$NSYS_STATS_FAILED" -ne 0 ]; then
    echo "Required Nsight statistics are incomplete; failing closed." >"$OUT/nsys/REQUIRED-STATS.FAIL"
    finish
    exit 1
  fi
else
  echo "nsys unavailable; required Prompt-1 profiler evidence cannot be collected." >"$OUT/nsys/REQUIRED-STATS.FAIL"
  finish
  exit 1
fi

if command -v ncu >/dev/null 2>&1; then
  ncu --version >"$OUT/nsys/ncu-version.txt" 2>&1 || true
  echo "Available. Targeted collection intentionally deferred until Systems attribution is reviewed." \
    >"$OUT/nsys/NCU-DEFERRED.txt"
else
  echo "ncu unavailable" >"$OUT/nsys/NCU-DEFERRED.txt"
fi

# Independent multi-client production ingress.
run_server_lane(){
  local lane="$1" port="$2"
  shift 2
  local server_log="$OUT/multi/server-$lane.txt"
  local event_log="$OUT/multi/server-$lane-events.jsonl"

  echo
  echo "[$(ts)] START server lane $lane"
  "$SERVER" -m "$MODEL" --host 127.0.0.1 --port "$port" \
    --workers 16 --max-connections 32 --max-active 8 \
    --token-budget 512 --prefill-quantum 32 \
    --event-log "$event_log" "$@" >"$server_log" 2>&1 &
  local server_pid=$!
  echo "$server_pid" >"$OUT/multi/server-$lane.pid"

  local ready=0
  for _ in $(seq 1 100); do
    if curl -fsS "http://127.0.0.1:$port/health" \
      >"$OUT/multi/health-$lane.json" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 0.1
  done

  if [ "$ready" -eq 1 ]; then
    echo "[$(ts)] DONE  server lane $lane healthy pid=$server_pid"

    run_stage "Independent-arrival c8 [$lane]" "$OUT/multi/independent-c8-$lane.txt" \
      python3 scripts/competitive-convergence/prompt1_multiclient.py \
        --mode independent --url "http://127.0.0.1:$port" \
        --prompt "$OUT/prompts/p160.txt" --concurrency 8 \
        --rounds "$ROUNDS" --tokens "$TOKENS" --stagger-ms 1.0 \
        --output "$OUT/multi/independent-c8-$lane.json"

    run_stage "Mixed long-prefill + short traffic [$lane]" "$OUT/multi/mixed-$lane.txt" \
      python3 scripts/competitive-convergence/prompt1_multiclient.py \
        --mode mixed --url "http://127.0.0.1:$port" \
        --long-prompt "$OUT/prompts/p1042.txt" \
        --short-prompt "$OUT/prompts/p54.txt" --short-count 7 \
        --rounds "$ROUNDS" --tokens "$TOKENS" --long-lead-ms 1.0 \
        --output "$OUT/multi/mixed-$lane.json"

    curl -fsS "http://127.0.0.1:$port/runtime" \
      >"$OUT/multi/runtime-final-$lane.json" 2>&1 || true
    curl -fsS "http://127.0.0.1:$port/events" \
      >"$OUT/multi/events-final-$lane.json" 2>&1 || true
  else
    echo "server lane $lane did not become healthy" \
      >"$OUT/multi/SERVER-FAILED-$lane.txt"
    echo "[$(ts)] FAIL  server lane $lane failed health check"
  fi

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

# Adaptive lane measures planner + scheduler behavior together.
run_server_lane "adaptive" "$PORT" \
  --backend auto --manifest "$MANIFEST" --require-manifest \
  --strategy-objective maximum-throughput --strategy-horizon-tokens 10000

# Fixed reuse8 lane removes Strategy Lab plan variation so scheduler/batching
# behavior can be distinguished from planner compatibility effects.
STATIC_PORT=$((PORT+1))
run_server_lane "reuse8-static" "$STATIC_PORT" \
  --backend cuda --no-manifest \
  --cuda-prefill-block-linear reuse8 \
  --cuda-decode-block-linear reuse8 \
  --cuda-decode-output-linear reuse8 \
  --cuda-prefill-attention online-softmax

finish
