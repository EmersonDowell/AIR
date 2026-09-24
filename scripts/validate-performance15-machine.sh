#!/usr/bin/env bash
# AIR Prompt 15/16 Competitive Destruction and Generalization collector.
# Prompt-15 R2 harness hardening: evidence validity is separate from product outcome.
# Prompt-15 R3 harness hardening: product-path, isolation, startup and external-admission contracts.
# Prompt-15 R4 harness hardening: byte-bounded exact-prompt synthesis; AIR_P15_CYCLES canonical alias.
# Prompt-15 R7: atomic production cohorts + executable operation-scoped scalar decode.
# Execution behavior is frozen except for the Prompt-15 manifest integrity hardening.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Performance15-$STAMP"
ARCHIVE="$HOME/Downloads/AIR-Performance15-$STAMP.zip"
BUILD="${AIR_P15_BUILD_DIR:-$ROOT/build-performance15}"
MODEL05="${AIR_MODEL_05:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
MODEL15="${AIR_MODEL_15:-}"
MODEL7="${AIR_MODEL_7:-}"
MANIFEST_CONTROL="$ROOT/controls/p14-final-manifest-v10.json"
CYCLES="${AIR_P15_CYCLES:-${AIR_P15_STRESS_CYCLES:-20}}"
VERIFY_DECISIONS="${AIR_P15_VERIFY_DECISIONS:-16}"
COMPARE_ROUNDS="${AIR_P15_COMPARE_ROUNDS:-5}"
COOLDOWN="${AIR_BENCH_COOLDOWN_SECONDS:-2}"
RUN_EXTERNAL="${AIR_P15_RUN_EXTERNAL:-1}"
mkdir -p "$OUT"/{build,manifest-corruption,out-of-region,memory-budget,stress,concurrency,generalization,control,startup,external,telemetry,prompts,source}

fatal=0
GATE1_OK=1
VERIFY05_OK=0
finish(){
  {
    echo "git_head=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo no-git)"
    find "$ROOT/include/air" "$ROOT/src" "$ROOT/scripts" "$ROOT/tests" "$ROOT/controls" -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum
  } >"$OUT/source-sha256.txt" 2>&1 || true
  python3 "$ROOT/scripts/analyze-performance15.py" "$OUT" >"$OUT/analyzer.txt" 2>&1 || true
  for f in CMakeLists.txt README.md ROADMAP.md; do [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$OUT/source/"; done
  for f in scripts/validate-performance15-machine.sh scripts/p15-corrupt-manifest.py scripts/p15-startup-transition-probe.py scripts/p15_prompt_contract.py scripts/p15_evidence_guard.py scripts/p15_output_isolation.py scripts/p15_gate7_profile_guard.py scripts/p15_external_guard.py scripts/analyze-performance15.py scripts/compare-llama.py src/runtime/manifest.cpp src/strategy_probe/main.cpp tests/core_tests.cpp; do
    [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$OUT/source/$(basename "$f")"
  done
  (cd "$(dirname "$OUT")" && zip -qr "$ARCHIVE" "$(basename "$OUT")") || true
  echo; echo '========================================'; echo 'AIR Prompt 15 evidence archive:'; echo "$ARCHIVE"; echo '========================================'
}
fail_early(){ echo "$1" | tee "$OUT/FATAL.txt"; fatal=1; finish; exit 0; }
trap 'true' EXIT

if [ "${AIR_STOP_EXISTING_SERVERS:-0}" = 1 ]; then
  pgrep -a -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >"$OUT/stopped-servers.txt" 2>&1 || true
  pkill -TERM -u "$(id -u)" -f '(^|/)(air-server|llama-server)( |$)' >/dev/null 2>&1 || true
  sleep 2
fi

find_tool(){ command -v "$1" 2>/dev/null && return 0; local x; shopt -s nullglob; for x in /usr/local/cuda*/bin/"$1" /opt/nvidia/nsight-systems/*/bin/"$1"; do [ -x "$x" ] && { printf '%s\n' "$x"; shopt -u nullglob; return 0; }; done; shopt -u nullglob; return 1; }
NVCC="$(find_tool nvcc || true)"
{
  echo harness=AIR-Performance15
  echo date="$(date -Is)"
  echo expected_air_version=0.9.12
  echo prompt14_control_manifest="$MANIFEST_CONTROL"
  echo stress_cycles="$CYCLES"
  echo verify_decisions="$VERIFY_DECISIONS"
  echo external_rounds="$COMPARE_ROUNDS"
  echo model05="$MODEL05"
  echo nvcc="${NVCC:-missing}"
  uname -a
  nvidia-smi || true
} >"$OUT/preflight.txt" 2>&1
[ -f "$MODEL05" ] || fail_early "missing 0.5B model: $MODEL05"
[ -f "$MANIFEST_CONTROL" ] || fail_early "missing Prompt-14 control manifest: $MANIFEST_CONTROL"
[ -n "$NVCC" ] || fail_early 'nvcc required for Prompt 15'

# Discover optional larger models without inventing names.
if [ -z "$MODEL15" ]; then MODEL15="$(find "$HOME/Models/AIR" -maxdepth 2 -type f \( -iname '*1.5b*.gguf' -o -iname '*1_5b*.gguf' \) 2>/dev/null | head -n1 || true)"; fi
if [ -z "$MODEL7" ]; then MODEL7="$(find "$HOME/Models/AIR" -maxdepth 2 -type f -iname '*7b*.gguf' 2>/dev/null | head -n1 || true)"; fi
printf 'model05=%s\nmodel15=%s\nmodel7=%s\n' "$MODEL05" "${MODEL15:-missing}" "${MODEL7:-missing}" >"$OUT/model-paths.txt"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DAIR_ENABLE_CUDA=ON -DAIR_ENABLE_NVTX=OFF -DAIR_BUILD_RESEARCH=ON >"$OUT/build/configure.txt" 2>&1 || fail_early 'Prompt 15 CUDA configure failed'
cmake --build "$BUILD" -j"${JOBS:-2}" >"$OUT/build/build.txt" 2>&1 || fail_early 'Prompt 15 CUDA build failed'
ctest --test-dir "$BUILD" --output-on-failure --timeout 300 >"$OUT/build/ctest.txt" 2>&1 || fail_early 'Prompt 15 complete ctest failed'
CLI="$BUILD/air-cli"; BENCH="$BUILD/air-bench"; VERIFY="$BUILD/air-verify"; PROBE="$BUILD/air-strategy-probe"; SERVER="$BUILD/air-server"
export PATH="$BUILD:$PATH"
"$CLI" fingerprint "$MODEL05" >"$OUT/fingerprint-05.json" 2>"$OUT/fingerprint-05.stderr" || fail_early '0.5B fingerprint failed'
sha256sum "$MODEL05" >"$OUT/model-sha256-05.txt"
[ -n "$MODEL15" ] && [ -f "$MODEL15" ] && { "$CLI" fingerprint "$MODEL15" >"$OUT/fingerprint-15.json" 2>"$OUT/fingerprint-15.stderr" || true; sha256sum "$MODEL15" >"$OUT/model-sha256-15.txt"; }
[ -n "$MODEL7" ] && [ -f "$MODEL7" ] && { "$CLI" fingerprint "$MODEL7" >"$OUT/fingerprint-7.json" 2>"$OUT/fingerprint-7.stderr" || true; sha256sum "$MODEL7" >"$OUT/model-sha256-7.txt"; }
cp "$MANIFEST_CONTROL" "$OUT/prompt14-control-manifest.json"

# Confirm the carried Prompt-14 evidence identity still matches this exact machine/model.
python3 - "$OUT/fingerprint-05.json" "$OUT/prompt14-control-manifest.json" <<'PY' >"$OUT/control-manifest-identity.txt" 2>&1 || fail_early 'Prompt-14 control manifest identity mismatch after integrity hardening'
import json,sys
f=json.load(open(sys.argv[1]));m=json.load(open(sys.argv[2]))
assert m['air_version']=='0.9.12'
assert f['air_version']=='0.9.12'
assert f['model_digest']==m['model_digest'],(f['model_digest'],m['model_digest'])
assert f['hardware_digest']==m['hardware_digest'],(f['hardware_digest'],m['hardware_digest'])
print('identity=match')
print('manifest_id='+m['manifest_id'])
PY

# Prompt-15 R2 harness hardening: workload labels are evidence and must match AIR's tokenizer.
python3 "$ROOT/scripts/p15_prompt_contract.py" make \
  --air-cli "$CLI" \
  --model "$MODEL05" \
  --output-dir "$OUT/prompts" \
  --targets 54,128,160,262,512,1042,2048 \
  >"$OUT/prompt-contract.txt" 2>&1 || fail_early 'Prompt-15 exact-token prompt synthesis failed'
python3 "$ROOT/scripts/p15_prompt_contract.py" verify \
  --air-cli "$CLI" \
  --model "$MODEL05" \
  --output-dir "$OUT/prompts" \
  --targets 54,128,160,262,512,1042,2048 \
  >"$OUT/prompt-contract-verify.json" 2>&1 || fail_early 'Prompt-15 exact-token prompt contract failed verification'

# Gate 1: manifest-integrity destruction.
# A negative test suite is interpretable only when the same runtime accepts the known-good control.
"$BENCH" -m "$MODEL05" --backend auto --manifest "$OUT/prompt14-control-manifest.json" --require-manifest \
  --prompt-file "$OUT/prompts/p54.txt" --tokens 1 --warmup 0 --runs 1 --concurrency 1 \
  --output "$OUT/manifest-corruption/valid-control.json" >"$OUT/manifest-corruption/valid-control.txt" 2>&1
VALID_CONTROL_RC=$?; echo "$VALID_CONTROL_RC" >"$OUT/manifest-corruption/valid-control.exit"
[ "$VALID_CONTROL_RC" -eq 0 ] || fail_early 'valid Prompt-14 control manifest failed before corruption testing; Gate 1 would be uninterpretable'

python3 "$ROOT/scripts/p15-corrupt-manifest.py" "$OUT/prompt14-control-manifest.json" "$OUT/manifest-corruption/cases" >"$OUT/manifest-corruption/generate.txt" 2>&1 || fail_early 'unable to generate corruption battery'
for bad in "$OUT"/manifest-corruption/cases/*.json; do
  [ "$(basename "$bad")" = EXPECTED.json ] && continue
  name="$(basename "$bad" .json)"
  "$BENCH" -m "$MODEL05" --backend auto --manifest "$bad" --require-manifest --prompt-file "$OUT/prompts/p54.txt" --tokens 1 --warmup 0 --runs 1 --concurrency 1 --output "$OUT/manifest-corruption/$name-output.json" >"$OUT/manifest-corruption/$name.txt" 2>&1
  rc=$?; echo "$rc" >"$OUT/manifest-corruption/$name.exit"
  [ "$rc" -eq 0 ] && GATE1_OK=0
 done
EXPECTED_CORRUPTIONS="$(python3 - "$OUT/manifest-corruption/cases/EXPECTED.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
print(len(j.get('cases',{})))
PY
)"
OBSERVED_CORRUPTIONS="$(find "$OUT/manifest-corruption" -maxdepth 1 -type f -name '*.exit' ! -name 'valid-control.exit' | wc -l)"
[ "$OBSERVED_CORRUPTIONS" -eq "$EXPECTED_CORRUPTIONS" ] || GATE1_OK=0
printf 'valid_control_rc=%s\nexpected_corruptions=%s\nobserved_corruptions=%s\nall_corruptions_rejected=%s\n' \
  "$VALID_CONTROL_RC" "$EXPECTED_CORRUPTIONS" "$OBSERVED_CORRUPTIONS" "$GATE1_OK" >"$OUT/manifest-corruption/RESULT.txt"

# Gate 2: no silent evidence-region extrapolation.
for name in p128 p512 p2048; do
  "$BENCH" -m "$MODEL05" --backend auto --manifest "$OUT/prompt14-control-manifest.json" --require-manifest --prompt-file "$OUT/prompts/$name.txt" --tokens 1 --warmup 0 --runs 1 --concurrency 1 --strategy-objective maximum-throughput --strategy-horizon-tokens 10000 --output "$OUT/out-of-region/$name.json" >"$OUT/out-of-region/$name.txt" 2>&1
  echo $? >"$OUT/out-of-region/$name.exit"
done
for c in 2 4 12; do
  "$BENCH" -m "$MODEL05" --backend auto --manifest "$OUT/prompt14-control-manifest.json" --require-manifest --prompt-file "$OUT/prompts/p160.txt" --tokens 8 --warmup 0 --runs "$c" --concurrency "$c" --strategy-objective maximum-throughput --strategy-horizon-tokens 10000 --output "$OUT/out-of-region/c$c.json" >"$OUT/out-of-region/c$c.txt" 2>&1
  echo $? >"$OUT/out-of-region/c$c.exit"
done

# Gate 3: memory-pressure sweep around the exact dense residency boundary.
DENSE_BYTES=1431306240
for budget in 268435456 536870912 1073741824 1395864371 1431306239 1431306240 1503238554 0; do
  label="$budget"; [ "$budget" = 0 ] && label=unrestricted
  args=(--strategy-objective maximum-throughput --strategy-horizon-tokens 10000)
  [ "$budget" != 0 ] && args+=(--strategy-prepared-memory-budget-bytes "$budget")
  "$BENCH" -m "$MODEL05" --backend auto --manifest "$OUT/prompt14-control-manifest.json" --require-manifest --prompt-file "$OUT/prompts/p1042.txt" --tokens 1 --warmup 0 --runs 1 --concurrency 1 "${args[@]}" --output "$OUT/memory-budget/budget-$label.json" >"$OUT/memory-budget/budget-$label.txt" 2>&1
  echo $? >"$OUT/memory-budget/budget-$label.exit"
done
# minimum-vram must prefer reuse8 even with unrestricted capacity.
"$BENCH" -m "$MODEL05" --backend auto --manifest "$OUT/prompt14-control-manifest.json" --require-manifest --prompt-file "$OUT/prompts/p1042.txt" --tokens 1 --warmup 0 --runs 1 --concurrency 1 --strategy-objective minimum-vram --strategy-horizon-tokens 10000 --output "$OUT/memory-budget/minimum-vram.json" >"$OUT/memory-budget/minimum-vram.txt" 2>&1; echo $? >"$OUT/memory-budget/minimum-vram.exit"

# Gates 4 + 6: repeated real transitions and hot-state stress in one InferenceService.
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits >"$OUT/stress/gpu-before.txt" 2>&1 || true
"$PROBE" -m "$MODEL05" --manifest "$OUT/prompt14-control-manifest.json" --medium-prompt-file "$OUT/prompts/p1042.txt" --small-prompt-file "$OUT/prompts/p54.txt" --scenario oscillation --cycles "$CYCLES" --horizon-tokens 10000 --output "$OUT/stress/oscillation.json" >"$OUT/stress/oscillation.txt" 2>&1; echo $? >"$OUT/stress/oscillation.exit"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits >"$OUT/stress/gpu-after-oscillation.txt" 2>&1 || true
"$PROBE" -m "$MODEL05" --manifest "$OUT/prompt14-control-manifest.json" --medium-prompt-file "$OUT/prompts/p1042.txt" --small-prompt-file "$OUT/prompts/p54.txt" --scenario hot-stress --cycles "$CYCLES" --horizon-tokens 10000 --output "$OUT/stress/hot-stress.json" >"$OUT/stress/hot-stress.txt" 2>&1; echo $? >"$OUT/stress/hot-stress.exit"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits >"$OUT/stress/gpu-after-hot.txt" 2>&1 || true

# Gate 5: concurrency dynamics. Only c8/p160 is a calibrated Strategy-Lab region.
for c in 1 2 4 8 12; do
  "$BENCH" -m "$MODEL05" --backend auto --manifest "$OUT/prompt14-control-manifest.json" --require-manifest --prompt-file "$OUT/prompts/p160.txt" --tokens 32 --warmup 0 --runs "$c" --concurrency "$c" --strategy-objective maximum-throughput --strategy-horizon-tokens 10000 --output "$OUT/concurrency/c$c.json" >"$OUT/concurrency/c$c.txt" 2>&1
  echo $? >"$OUT/concurrency/c$c.exit"
done
python3 "$ROOT/scripts/p15_output_isolation.py" "$OUT/concurrency/c8.json" --expected-prompt-tokens 160 --expected-concurrency 8 --output "$OUT/concurrency/output-isolation.json" >"$OUT/concurrency/output-isolation.txt" 2>&1
echo $? >"$OUT/concurrency/output-isolation.exit"

# Gate 7: longer-history numerical generalization, unchanged atol=0.001.
LONG_PROMPT="$(cat "$OUT/prompts/p262.txt")"
# Pin the complete operation-scoped product path used for this generalization claim.
# AIR has no separate product decode-attention tactic flag; decode attention remains the canonical existing path.
cat >"$OUT/generalization/EXECUTION-PROFILE.json" <<'JSON'
{"schema":"air.prompt15.gate7-execution-profile.v1","prefill_block":"batch-reuse8","decode_block":"batch-reuse8","decode_output":"batch-reuse8","prefill_attention":"online-softmax","decode_attention":"baseline"}
JSON
"$VERIFY" --help >"$OUT/generalization/air-verify-help.txt" 2>&1 || true
python3 "$ROOT/scripts/p15_gate7_profile_guard.py" --collector "$ROOT/scripts/validate-performance15-machine.sh" --verify-help "$OUT/generalization/air-verify-help.txt" --output "$OUT/generalization/PROFILE-GUARD.json" >"$OUT/generalization/profile-guard.txt" 2>&1 || fail_early 'Gate 7 air-verify operation-scoped product-path contract is not supported by this build'
verify_model(){
  local tag="$1" model="$2"
  if [ -z "$model" ] || [ ! -f "$model" ]; then echo missing >"$OUT/generalization/$tag.missing"; return 99; fi
  "$VERIFY" -m "$model" --prompt "$LONG_PROMPT" --generate "$VERIFY_DECISIONS" --top-k 8 --device 0 --atol 0.001 --cuda-prefill-block-linear batch-reuse8 --cuda-decode-block-linear batch-reuse8 --cuda-decode-output-linear batch-reuse8 --cuda-prefill-attention online-softmax --cuda-decode-attention baseline --output "$OUT/generalization/$tag.json" >"$OUT/generalization/$tag.txt" 2>&1
  local rc=$?; echo "$rc" >"$OUT/generalization/$tag.exit"
  if [ "$rc" -ne 0 ] && [ -f "$OUT/generalization/$tag.json" ]; then
    worst="$(python3 - "$OUT/generalization/$tag.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); ds=j.get('decisions',[])
print(max(ds,key=lambda x:x.get('logits',{}).get('max_abs_error',0)).get('index',0) if ds else 0)
PY
)"
    "$VERIFY" -m "$model" --prompt "$LONG_PROMPT" --generate "$VERIFY_DECISIONS" --top-k 8 --device 0 --atol 0.001 --cuda-prefill-block-linear batch-reuse8 --cuda-decode-block-linear batch-reuse8 --cuda-decode-output-linear batch-reuse8 --cuda-prefill-attention online-softmax --cuda-decode-attention baseline --trace-from "$worst" --trace-count 1 --output "$OUT/generalization/$tag-trace.json" >"$OUT/generalization/$tag-trace.txt" 2>&1 || true
  fi
  return "$rc"
}
verify_model model-05 "$MODEL05"; rc05=$?; [ "$rc05" -eq 0 ] && VERIFY05_OK=1 || VERIFY05_OK=0
verify_model model-15 "$MODEL15"; rc15=$?
verify_model model-7 "$MODEL7"; rc7=$?
printf 'model05_rc=%s\nmodel15_rc=%s\nmodel7_rc=%s\n' "$rc05" "$rc15" "$rc7" >"$OUT/generalization/RESULT.txt"

# Gate 8: frozen AIR 0.8 control provenance. Never relabel current reference as 0.8.
cp "$ROOT/controls/AIR-Prompt8-Scaling-Analysis.md" "$OUT/control/AIR-Prompt8-Scaling-Analysis.md"
{
  echo 'historical_control_included=controls/AIR-Prompt8-Scaling-Analysis.md'
  echo 'searching for actual frozen AIR 0.8 artifacts:'
  find "$HOME/Downloads" -maxdepth 1 -type f \( -iname '*AIR*0.8*.zip' -o -iname '*AIR*0.8*.tar*' \) -print -exec sha256sum {} \; 2>/dev/null || true
  for d in "$HOME/Projects/AIR-0.8" "$HOME/Projects/AIR-0.8.0"; do
    if [ -e "$d" ]; then
      echo "source_candidate=$d"
      find "$d" -maxdepth 3 -type f -name air-cli -perm -111 -print 2>/dev/null | while read -r oldcli; do
        echo "binary_candidate=$oldcli"
        "$oldcli" --version 2>&1 || true
        "$oldcli" fingerprint "$MODEL05" 2>&1 || true
        sha256sum "$oldcli" || true
      done
    fi
  done
  if [ -d "$HOME/Projects/AIR/.git" ]; then git -C "$HOME/Projects/AIR" tag --list '*0.8*' || true; fi
} >"$OUT/control/provenance.txt" 2>&1

# Gate 9: process startup versus transition economics.
python3 "$ROOT/scripts/p15-startup-transition-probe.py" --air-server "$SERVER" -m "$MODEL05" --manifest "$OUT/prompt14-control-manifest.json" --small-prompt-file "$OUT/prompts/p54.txt" --medium-prompt-file "$OUT/prompts/p1042.txt" --horizon 10000 --output-dir "$OUT/startup" >"$OUT/startup/probe.txt" 2>&1; echo $? >"$OUT/startup/probe.exit"

# Gate 10: controlled external comparison only after integrity + qualified 0.5B correctness survives.
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/Projects/llama.cpp-tq3/build/bin/llama-server}"
LLAMA_BENCH="${LLAMA_BENCH:-$HOME/Projects/llama.cpp-tq3/build/bin/llama-bench}"
if [ "$RUN_EXTERNAL" = 1 ] && [ "$GATE1_OK" = 1 ] && [ "$VERIFY05_OK" = 1 ] && [ -x "$LLAMA_SERVER" ]; then
  mkdir -p "$OUT/external/reuse8" "$OUT/external/dense"
  python3 "$ROOT/scripts/compare-llama.py" --scaling --model "$MODEL05" --air-server "$SERVER" --air-cli "$CLI" --air-verify "$VERIFY" --llama-server "$LLAMA_SERVER" --llama-bench "$LLAMA_BENCH" --verify-script "$ROOT/scripts/verify-llama-teacher.py" --output-dir "$OUT/external/reuse8" --rounds "$COMPARE_ROUNDS" --cooldown "$COOLDOWN" --prompt-targets 54,160,262,1042 --concurrency-levels 1,4,8 --scaling-generated 32 --skip-microbench --air-server-extra-args='--cuda-prefill-block-linear reuse8 --cuda-decode-block-linear reuse8 --cuda-decode-output-linear reuse8 --cuda-prefill-attention online-softmax' >"$OUT/external/reuse8/harness.txt" 2>&1
  echo $? >"$OUT/external/reuse8/harness.exit"
  python3 "$ROOT/scripts/compare-llama.py" --scaling --model "$MODEL05" --air-server "$SERVER" --air-cli "$CLI" --air-verify "$VERIFY" --llama-server "$LLAMA_SERVER" --llama-bench "$LLAMA_BENCH" --verify-script "$ROOT/scripts/verify-llama-teacher.py" --output-dir "$OUT/external/dense" --rounds "$COMPARE_ROUNDS" --cooldown "$COOLDOWN" --prompt-targets 160,262,1042 --concurrency-levels 1,8 --scaling-generated 32 --skip-microbench --air-server-extra-args='--cuda-prefill-block-linear dense-f32-cublas --cuda-decode-block-linear dense-f32-cublas --cuda-decode-output-linear reuse8 --cuda-prefill-attention online-softmax' >"$OUT/external/dense/harness.txt" 2>&1
  echo $? >"$OUT/external/dense/harness.exit"
  python3 "$ROOT/scripts/p15_external_guard.py" --reuse8-dir "$OUT/external/reuse8" --dense-dir "$OUT/external/dense" --min-rounds "$COMPARE_ROUNDS" --output "$OUT/external/CONTRACT.json" >"$OUT/external/contract.txt" 2>&1
  echo $? >"$OUT/external/CONTRACT.exit"
else
  printf 'external comparison skipped\nrun_external=%s\nmanifest_integrity_ok=%s\nmodel05_correctness_ok=%s\nllama_server=%s\n' "$RUN_EXTERNAL" "$GATE1_OK" "$VERIFY05_OK" "$LLAMA_SERVER" >"$OUT/external/SKIPPED.txt"
fi

# Final telemetry and preliminary non-promotional summary.
nvidia-smi -q >"$OUT/telemetry/nvidia-smi-q.txt" 2>&1 || true
python3 "$ROOT/scripts/analyze-performance15.py" "$OUT" >"$OUT/analyzer.txt" 2>&1 || true
python3 "$ROOT/scripts/p15_evidence_guard.py" "$OUT" --output "$OUT/EVIDENCE-INTEGRITY.json"
EVIDENCE_GUARD_RC=$?
printf 'evidence_guard_rc=%s\n' "$EVIDENCE_GUARD_RC" >"$OUT/EVIDENCE-INTEGRITY.exit"
if [ "$EVIDENCE_GUARD_RC" -ne 0 ]; then
  printf '%s\n' 'Prompt-15 archive is structurally incomplete or internally inconsistent. This is not the same as AIR scientifically failing a gate.' >"$OUT/EVIDENCE-INVALID.txt"
fi
finish
exit 0
