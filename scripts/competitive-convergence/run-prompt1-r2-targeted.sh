#!/usr/bin/env bash
set +e
set +u
set +E
set +o pipefail 2>/dev/null

ROOT="${AIR_ROOT:-$HOME/Projects/AIR}"
BUILD="${AIR_CC1_BUILD:-$ROOT/build-cc1}"
MODEL="${AIR_CC1_MODEL:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/Downloads/AIR-CC1-R2-Targeted-$STAMP"
mkdir -p "$OUT"/{hardware,q,ncu,source}
cd "$ROOT" || exit 2

ts(){ date '+%H:%M:%S'; }
gpu_line(){
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm \
    --format=csv,noheader 2>/dev/null | head -n1
}
run_stage(){
  local label="$1" log="$2"; shift 2
  local start pid rc elapsed gpu
  start="$(date +%s)"
  echo
  echo "[$(ts)] START $label"
  "$@" >"$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
    kill -0 "$pid" 2>/dev/null || break
    elapsed=$(( $(date +%s)-start ))
    echo "[$(ts)] ACTIVE $label elapsed=${elapsed}s pid=$pid"
    ps -p "$pid" -o pid=,ppid=,etime=,stat=,%cpu=,%mem=,cmd= 2>/dev/null | sed 's/^/  proc: /'
    gpu="$(gpu_line)"; [ -n "$gpu" ] && echo "  GPU: $gpu"
    [ -s "$log" ] && { echo "  log tail:"; tail -n 3 "$log" | sed 's/^/    /'; }
  done
  wait "$pid"; rc=$?
  echo "$rc" >"$log.exit"
  elapsed=$(( $(date +%s)-start ))
  if [ "$rc" -eq 0 ]; then
    echo "[$(ts)] DONE  $label rc=0 elapsed=${elapsed}s"
  else
    echo "[$(ts)] FAIL  $label rc=$rc elapsed=${elapsed}s"
    tail -n 20 "$log" 2>/dev/null | sed 's/^/  /'
  fi
  return "$rc"
}

sha256sum research/prompt1_cuda_hardware_probe.cu \
  scripts/competitive-convergence/prompt1_q_randomized.py \
  >"$OUT/source/tool-sha256.txt"

run_stage "Compile corrected CUDA hardware probe" "$OUT/hardware/cuda-probe-build.txt" \
  nvcc -std=c++20 -O3 research/prompt1_cuda_hardware_probe.cu \
    -o "$OUT/hardware/cuda-probe"

if [ -x "$OUT/hardware/cuda-probe" ]; then
  run_stage "Measure CUDA transfer/launch/allocation" "$OUT/hardware/cuda-hardware.json" \
    "$OUT/hardware/cuda-probe" 0 64 20
fi

CENSUS_DIR="$(
  find "$HOME/Downloads" -maxdepth 1 -type d -name 'AIR-CC1-Census-*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
)"
PROMPT_DIR=""
if [ -n "$CENSUS_DIR" ] && [ -f "$CENSUS_DIR/prompts/p1042.txt" ]; then
  PROMPT_DIR="$CENSUS_DIR/prompts"
else
  LATEST_ZIP="$(ls -1t "$HOME"/Downloads/AIR-CC1-Census-*.zip 2>/dev/null | head -n1)"
  if [ -n "$LATEST_ZIP" ]; then
    mkdir -p "$OUT/prompts"
    python3 - "$LATEST_ZIP" "$OUT/prompts" <<'PY'
import sys,zipfile
from pathlib import Path
z=zipfile.ZipFile(sys.argv[1]); out=Path(sys.argv[2])
for name in z.namelist():
    base=Path(name).name
    if base in {"p54.txt","p160.txt","p262.txt","p1042.txt"} and "/prompts/" in name:
        (out/base).write_bytes(z.read(name))
PY
    PROMPT_DIR="$OUT/prompts"
  fi
fi

if [ -n "$PROMPT_DIR" ] && [ -f "$PROMPT_DIR/p1042.txt" ]; then
  run_stage "Randomized q32/q64/q128 confirmation" "$OUT/q/runner.txt" \
    python3 scripts/competitive-convergence/prompt1_q_randomized.py \
      --bench "$BUILD/air-bench" --model "$MODEL" \
      --prompt-dir "$PROMPT_DIR" --output-dir "$OUT/q" \
      --rounds 5 --seed 1501 --cooldown 1
else
  echo "No prior exact Prompt-1 prompt directory or census ZIP found." >"$OUT/q/SKIPPED.txt"
fi

# Targeted Nsight Compute: current long-prompt attention and quantized-linear kernels only.
if command -v ncu >/dev/null 2>&1; then
  NCU_COMMON=(--target-processes all --kernel-name-base demangled
    --section SpeedOfLight --section MemoryWorkloadAnalysis
    --section LaunchStats --section Occupancy --csv --page details)

  APP=("$BUILD/air-bench" -m "$MODEL" --backend cuda --no-manifest
    --cuda-prefill-block-linear reuse8
    --cuda-decode-block-linear reuse8
    --cuda-decode-output-linear reuse8
    --cuda-prefill-attention online-softmax
    --prompt-file "$PROMPT_DIR/p1042.txt" --tokens 1 --warmup 0 --runs 1 --concurrency 1
    --token-budget 512)

  run_stage "NCU late-context attention q32" "$OUT/ncu/ncu-attention-q32.csv" \
    ncu "${NCU_COMMON[@]}" --kernel-name 'regex:attention_batch_paged_online_kernel' \
      --launch-skip 720 --launch-count 3 \
      "${APP[@]}" --prefill-quantum 32

  run_stage "NCU late-context attention q128" "$OUT/ncu/ncu-attention-q128.csv" \
    ncu "${NCU_COMMON[@]}" --kernel-name 'regex:attention_batch_paged_online_kernel' \
      --launch-skip 180 --launch-count 3 \
      "${APP[@]}" --prefill-quantum 128

  run_stage "NCU quantized linear q32" "$OUT/ncu/ncu-linear-q32.csv" \
    ncu "${NCU_COMMON[@]}" --kernel-name 'regex:specialized_matmul_batch_reuse_kernel' \
      --launch-skip 4500 --launch-count 3 \
      "${APP[@]}" --prefill-quantum 32
else
  echo "ncu unavailable" >"$OUT/ncu/SKIPPED.txt"
fi

python3 scripts/competitive-convergence/prompt1_r2_analyze.py "$OUT" \
  >"$OUT/analyze.txt" 2>&1 || true

echo
echo "[$(ts)] START Packaging targeted Prompt-1 R2"
cd "$HOME/Downloads" || exit 0
ZIP="$HOME/Downloads/AIR-CC1-R2-Targeted-$STAMP.zip"
zip -qr "$ZIP" "$(basename "$OUT")"
echo "[$(ts)] DONE  Packaging targeted Prompt-1 R2"
echo "$ZIP"
sha256sum "$ZIP"
