#!/usr/bin/env bash
set +e
set +u
set +E
set +o pipefail 2>/dev/null
ROOT="${AIR_ROOT:-$HOME/Projects/AIR}"
BUILD="${AIR_P15_BUILD:-$ROOT/build-performance15}"
JOBS="${JOBS:-2}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/Downloads/AIR-P15-R7-Gates5-7-$STAMP"
ZIP="$OUT.zip"
mkdir -p "$OUT"/{build,prompts,gate5,gate7,source}
MODEL05="${AIR_MODEL_05:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
MODEL15="${AIR_MODEL_15:-$HOME/Models/AIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
MODEL7="${AIR_MODEL_7:-$HOME/Models/AIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf}"
MANIFEST="${AIR_P15_MANIFEST:-$ROOT/controls/p14-final-manifest-v10.json}"
if [ ! -f "$MANIFEST" ]; then MANIFEST="$(find "$HOME/Downloads" -maxdepth 2 -type f -path '*/AIR-Performance15-*/prompt14-control-manifest.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"; fi
finish(){ cd "$HOME/Downloads" || return; zip -qr "$ZIP" "$(basename "$OUT")"; echo; echo "R7 targeted evidence archive:"; echo "$ZIP"; sha256sum "$ZIP"; }
trap finish EXIT
cd "$ROOT" || { echo "AIR root missing" >"$OUT/FATAL.txt"; exit 1; }
{ echo "air_root=$ROOT"; echo "build=$BUILD"; echo "model05=$MODEL05"; echo "model15=$MODEL15"; echo "model7=$MODEL7"; echo "manifest=$MANIFEST"; date -u +%FT%TZ; } >"$OUT/CONTEXT.txt"
grep -q 'project(AIR VERSION 0.9.12' CMakeLists.txt || { echo "AIR is not 0.9.12" >"$OUT/FATAL.txt"; exit 2; }
[ -f "$MANIFEST" ] || { echo "Prompt-14 control manifest missing" >"$OUT/FATAL.txt"; exit 3; }
for m in "$MODEL05" "$MODEL15" "$MODEL7"; do [ -f "$m" ] || echo "missing model: $m" >>"$OUT/FATAL.txt"; done
[ ! -f "$OUT/FATAL.txt" ] || exit 4
cmake --build "$BUILD" -j "$JOBS" >"$OUT/build/build.txt" 2>&1; echo $? >"$OUT/build/build.exit"; [ "$(cat "$OUT/build/build.exit")" = 0 ] || exit 10
ctest --test-dir "$BUILD" --output-on-failure >"$OUT/build/ctest.txt" 2>&1; echo $? >"$OUT/build/ctest.exit"; [ "$(cat "$OUT/build/ctest.exit")" = 0 ] || exit 11
CLI="$BUILD/air-cli"; BENCH="$BUILD/air-bench"; VERIFY="$BUILD/air-verify"
"$VERIFY" --help >"$OUT/gate7/air-verify-help.txt" 2>&1
python3 scripts/p15_gate7_profile_guard.py --collector scripts/validate-performance15-machine.sh --verify-help "$OUT/gate7/air-verify-help.txt" --output "$OUT/gate7/PROFILE-GUARD.json" >"$OUT/gate7/profile-guard.txt" 2>&1; echo $? >"$OUT/gate7/PROFILE-GUARD.exit"; [ "$(cat "$OUT/gate7/PROFILE-GUARD.exit")" = 0 ] || exit 12
python3 scripts/p15_prompt_contract.py make --air-cli "$CLI" --model "$MODEL05" --output-dir "$OUT/prompts" --targets 160,262 >"$OUT/prompts/make.txt" 2>&1; echo $? >"$OUT/prompts/make.exit"
python3 scripts/p15_prompt_contract.py verify --air-cli "$CLI" --model "$MODEL05" --output-dir "$OUT/prompts" --targets 160,262 >"$OUT/prompts/verify.txt" 2>&1; echo $? >"$OUT/prompts/verify.exit"
[ "$(cat "$OUT/prompts/make.exit")" = 0 ] && [ "$(cat "$OUT/prompts/verify.exit")" = 0 ] || exit 13
"$BENCH" -m "$MODEL05" --backend auto --manifest "$MANIFEST" --require-manifest --prompt-file "$OUT/prompts/p160.txt" --tokens 32 --warmup 0 --runs 8 --concurrency 8 --strategy-objective maximum-throughput --strategy-horizon-tokens 10000 --output "$OUT/gate5/adaptive-c8.json" >"$OUT/gate5/adaptive-c8.txt" 2>&1; echo $? >"$OUT/gate5/adaptive-c8.exit"
python3 scripts/p15_output_isolation.py "$OUT/gate5/adaptive-c8.json" --expected-prompt-tokens 160 --expected-concurrency 8 --output "$OUT/gate5/output-isolation.json" >"$OUT/gate5/output-isolation.txt" 2>&1; echo $? >"$OUT/gate5/output-isolation.exit"
python3 scripts/p15_gate5_characterize.py "$OUT/gate5/adaptive-c8.json" --expected-prompt-tokens 160 --expected-concurrency 8 --expected-strategy dense-c8 --output "$OUT/gate5/characterization.json" >"$OUT/gate5/characterization.txt" 2>&1; echo $? >"$OUT/gate5/characterization.exit"
LONG_PROMPT="$(cat "$OUT/prompts/p262.txt")"
run_verify(){ local tag="$1" model="$2"; "$VERIFY" -m "$model" --prompt "$LONG_PROMPT" --generate 16 --top-k 8 --atol 0.001 --cuda-prefill-block-linear batch-reuse8 --cuda-decode-block-linear batch-reuse8 --cuda-decode-output-linear batch-reuse8 --cuda-prefill-attention online-softmax --cuda-decode-attention baseline --output "$OUT/gate7/$tag.json" >"$OUT/gate7/$tag.txt" 2>&1; echo $? >"$OUT/gate7/$tag.exit"; }
run_verify model05 "$MODEL05"
run_verify model15 "$MODEL15"
run_verify model7 "$MODEL7"
python3 - "$OUT" <<'PY_SUMMARY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); summary={"schema":"air.prompt15.r7-targeted.v1"}
for name in ("adaptive-c8","output-isolation","characterization"):
 p=r/"gate5"/f"{name}.exit"; summary[f"gate5_{name}_rc"]=int(p.read_text()) if p.exists() else None
for tag in ("model05","model15","model7"):
 p=r/"gate7"/f"{tag}.exit"; summary[f"gate7_{tag}_rc"]=int(p.read_text()) if p.exists() else None
 j=r/"gate7"/f"{tag}.json"
 if j.exists():
  d=json.loads(j.read_text()); s=d.get("summary",{}); dec=d.get("decisions",[])
  summary[tag]={"finite":s.get("finite"),"top1_parity":s.get("top1_parity"),"within_requested_atol":s.get("within_requested_atol"),"decisions":s.get("decisions"),"worst_decision_max_abs_error":max((x.get("logits",{}).get("max_abs_error",0) for x in dec),default=None),"profile":{"prefill_block":d.get("cuda_prefill_block_linear"),"decode_block":d.get("cuda_decode_block_linear"),"decode_output":d.get("cuda_decode_output_linear"),"prefill_attention":d.get("cuda_prefill_attention"),"decode_attention":d.get("cuda_decode_attention")}}
(r/"TARGETED-SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))
PY_SUMMARY
mkdir -p "$OUT/source/files"
for f in include/air/serving.hpp include/air/cuda.hpp src/runtime/serving.cpp src/runtime/backend.cpp src/cuda/cuda_backend.cu src/verification/main.cpp src/benchmark/benchmark.cpp tests/serving_tests.cpp; do mkdir -p "$OUT/source/files/$(dirname "$f")"; cp "$f" "$OUT/source/files/$f"; done
git diff -- include/air/serving.hpp include/air/cuda.hpp src/runtime/serving.cpp src/runtime/backend.cpp src/cuda/cuda_backend.cu src/verification/main.cpp src/benchmark/benchmark.cpp tests/serving_tests.cpp >"$OUT/source/R7.diff" 2>/dev/null || true
sha256sum include/air/serving.hpp include/air/cuda.hpp src/runtime/serving.cpp src/runtime/backend.cpp src/cuda/cuda_backend.cu src/verification/main.cpp src/benchmark/benchmark.cpp tests/serving_tests.cpp >"$OUT/source/R7.sha256" 2>/dev/null || true
exit 0
