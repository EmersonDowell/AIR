#!/usr/bin/env bash
# AIR Performance Prompt 11/16: paged prefill attention tactic qualification.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Performance11-$STAMP"; ARCHIVE="$HOME/Downloads/AIR-Performance11-$STAMP.zip"
BUILD="${AIR_PERF11_BUILD_DIR:-$ROOT/build-performance11}"; PROFILE_BUILD="${AIR_PERF11_PROFILE_BUILD_DIR:-$ROOT/build-performance11-profile}"
MODEL="${AIR_PERF11_MODEL:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
ROUNDS="${AIR_ATTENTION_ROUNDS:-5}"; COOLDOWN="${AIR_BENCH_COOLDOWN_SECONDS:-2}"; SEED="${AIR_ATTENTION_SEED:-1119}"
mkdir -p "$OUT"/{correctness,bench,nsys,prompts,source}
find_tool(){ command -v "$1" 2>/dev/null || find /usr/local/cuda* /opt/nvidia/nsight-systems -type f -name "$1" -executable 2>/dev/null | sort -V | tail -1; }
NVCC="$(find_tool nvcc || true)"; NSYS="$(find_tool nsys || true)"
if [ "${AIR_STOP_EXISTING_SERVERS:-0}" = 1 ]; then pgrep -a -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >"$OUT/stopped-servers.txt" 2>&1 || true; pkill -TERM -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >/dev/null 2>&1 || true; sleep 2; fi
{
 echo harness=AIR-Performance11-R1; echo date="$(date -Is)"; echo air_version_expected=0.9.2; echo model="$MODEL"; echo rounds="$ROUNDS"; echo seed="$SEED"; echo nvcc="${NVCC:-missing}"; echo nsys="${NSYS:-missing}";
 echo cuda_backend_sha256="$(sha256sum "$ROOT/src/cuda/cuda_backend.cu"|awk '{print $1}')"; echo execution_contract_sha256="$(sha256sum "$ROOT/include/air/execution.hpp"|awk '{print $1}')"; uname -a; nvidia-smi || true;
} >"$OUT/preflight.txt" 2>&1
[ -f "$MODEL" ] || { echo missing model >"$OUT/FAILURE.txt"; exit 20; }; [ -n "$NVCC" ] || { echo nvcc required >"$OUT/FAILURE.txt"; exit 21; }
sha256sum "$MODEL" >"$OUT/model-sha256.txt"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=OFF >"$OUT/build-release.txt" 2>&1 || exit 22
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-release.txt" 2>&1 || exit 23
ctest --test-dir "$BUILD" --output-on-failure >"$OUT/ctest-release.txt" 2>&1 || exit 24
BENCH="$BUILD/air-bench"; VERIFY="$BUILD/air-verify"; CLI="$BUILD/air-cli"
{ "$BENCH" --version; "$VERIFY" --version 2>/dev/null||true; "$CLI" inspect "$MODEL"|sed -n '1,55p'; } >"$OUT/model-and-version.txt" 2>&1
BASE='Adaptive inference runtimes execute transformer layers while tracking memory ownership, scheduling decisions, numerical evidence, and request latency under a controlled workload. '
python3 - "$OUT/prompts" "$BASE" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1]); base=sys.argv[2]
for name,repeats in (("p262",10),("p1042",40)): (out/f"{name}.txt").write_text(base*repeats)
PY
TOKENS64="$(python3 - <<'PY'
print(','.join(str(i) for i in range(1,65)))
PY
)"
QUAL=()
for tactic in baseline online; do
 set +e; "$VERIFY" -m "$MODEL" --tokens "$TOKENS64" --generate 2 --top-k 8 --device 0 --atol 0.001 --cuda-prefill-block-linear reuse8 --cuda-prefill-attention "$tactic" --output "$OUT/correctness/$tactic.json" >"$OUT/correctness/$tactic.txt" 2>&1; rc=$?; set -e
 echo "$rc" >"$OUT/correctness/$tactic.exit"; [ "$rc" -eq 0 ] && QUAL+=("$tactic")
done
printf '%s\n' "${QUAL[@]}" >"$OUT/qualified-tactics.txt"; printf '%s\n' "${QUAL[@]}"|grep -qx baseline || exit 25
python3 - "$OUT/order.tsv" "$ROUNDS" "$SEED" "${QUAL[@]}" <<'PY'
import random,sys
p=sys.argv[1]; n=int(sys.argv[2]); rng=random.Random(int(sys.argv[3])); ts=sys.argv[4:]
with open(p,'w') as f:
 for prompt in ('p262','p1042'):
  for r in range(1,n+1):
   x=ts[:]; rng.shuffle(x)
   for t in x: f.write(f'{r}\t{prompt}\t{t}\n')
PY
DMON=""; if command -v nvidia-smi >/dev/null; then nvidia-smi dmon -s pucvmt -d 1 >"$OUT/gpu-dmon.txt" 2>&1 & DMON=$!; fi
cleanup(){ [ -z "$DMON" ] || kill "$DMON" >/dev/null 2>&1 || true; }; trap cleanup EXIT
FAIL=0
while IFS=$'\t' read -r round prompt tactic; do stem="round-$(printf '%02d' "$round")-$prompt-$tactic"; echo "$stem"|tee -a "$OUT/bench/progress.txt"; set +e; "$BENCH" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/$prompt.txt" --tokens 4 --warmup 0 --runs 1 --concurrency 1 --prefill-quantum 32 --token-budget 256 --cuda-prefill-block-linear reuse8 --cuda-prefill-attention "$tactic" --output "$OUT/bench/$stem.json" >"$OUT/bench/$stem.txt" 2>&1; rc=$?; set -e; echo "$rc">"$OUT/bench/$stem.exit"; [ "$rc" -eq 0 ]||FAIL=1; sleep "$COOLDOWN"; done <"$OUT/order.tsv"
python3 "$ROOT/scripts/analyze-attention-tactics.py" "$OUT/bench" --json "$OUT/attention-tactic-summary.json" --markdown "$OUT/attention-tactic-summary.md" >"$OUT/analyzer.txt" 2>&1 || ARC=$?; ARC="${ARC:-0}"
if [ -n "$NSYS" ]; then
 cmake -S "$ROOT" -B "$PROFILE_BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=ON >"$OUT/build-profile.txt" 2>&1 && cmake --build "$PROFILE_BUILD" -j"${JOBS:-$(nproc)}" >>"$OUT/build-profile.txt" 2>&1 || true
 PB="$PROFILE_BUILD/air-bench"
 if [ -x "$PB" ]; then for tactic in "${QUAL[@]}"; do prefix="$OUT/nsys/p1042-q32-$tactic"; set +e; "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o "$prefix" "$PB" -m "$MODEL" --backend cuda --no-manifest --prompt-file "$OUT/prompts/p1042.txt" --tokens 2 --warmup 0 --runs 1 --concurrency 1 --prefill-quantum 32 --token-budget 256 --cuda-prefill-block-linear reuse8 --cuda-prefill-attention "$tactic" --output "$prefix-bench.json" >"$prefix-profile.txt" 2>&1; prc=$?; set -e; echo "$prc">"$prefix-profile.exit"; if [ "$prc" -eq 0 ]&&[ -f "$prefix.nsys-rep" ]; then "$NSYS" stats --report nvtx_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum --report cuda_api_sum "$prefix.nsys-rep" >"$prefix-stats.txt" 2>&1||true; fi; done; fi
fi
cleanup; trap - EXIT
{ echo qualified_tactics="$(paste -sd, "$OUT/qualified-tactics.txt")"; echo benchmark_failure="$FAIL"; echo analyzer_rc="$ARC"; echo linear_tactic_fixed=batch-reuse8; } >"$OUT/RESULT.txt"
cp "$ROOT/include/air/execution.hpp" "$OUT/source/execution.hpp"; cp "$ROOT/include/air/cuda.hpp" "$OUT/source/cuda.hpp"; cp "$ROOT/src/cuda/cuda_backend.cu" "$OUT/source/cuda_backend.cu"; cp "$ROOT/src/runtime/backend.cpp" "$OUT/source/backend.cpp"; cp "$ROOT/scripts/analyze-attention-tactics.py" "$OUT/source/"; cp "$ROOT/CMakeLists.txt" "$OUT/source/"
(cd "$(dirname "$OUT")" && zip -qr "$ARCHIVE" "$(basename "$OUT")")
echo; echo "AIR Prompt 11 evidence archive:"; echo "$ARCHIVE"; exit "$ARC"
