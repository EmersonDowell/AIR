#!/usr/bin/env bash
# AIR Comparison Prompt 8: scaling matrix evidence collection.
# Evidence tooling only. This script does not modify AIR runtime architecture.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_HARNESS_REVISION="AIR-Comparison8-R3"
LLAMA_ROOT="${LLAMA_ROOT:-$HOME/Projects/llama.cpp-tq3}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_ROOT/build/bin/llama-server}"
LLAMA_BENCH="${LLAMA_BENCH:-$LLAMA_ROOT/build/bin/llama-bench}"
BASE_MODEL="${AIR_SCALE_BASE_MODEL:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
ROUNDS="${AIR_SCALE_ROUNDS:-3}"
COOLDOWN="${AIR_SCALE_COOLDOWN:-5}"
PROMPTS="${AIR_SCALE_PROMPTS:-32,256,1024}"
CONCURRENCY="${AIR_SCALE_CONCURRENCY:-1,2,4}"
GENERATED="${AIR_SCALE_GENERATED:-32}"
MAX_MODELS="${AIR_SCALE_MAX_MODELS:-3}"
DISCOVER="${AIR_SCALE_DISCOVER:-1}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Comparison8-$STAMP"
ARCHIVE="$HOME/Downloads/AIR-Comparison8-$STAMP.zip"

mkdir -p "$OUT"

fail=0
for tool in air-server air-cli air-verify air-bench python3 nvidia-smi zip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "missing required tool: $tool" | tee -a "$OUT/preflight.txt"
        fail=1
    fi
done
for path in "$LLAMA_SERVER" "$LLAMA_BENCH" "$ROOT/scripts/compare-llama.py" "$ROOT/scripts/scale-matrix.py" "$ROOT/scripts/verify-llama-teacher.py"; do
    if [ ! -e "$path" ]; then
        echo "missing required path: $path" | tee -a "$OUT/preflight.txt"
        fail=1
    fi
done
if [ "$fail" -ne 0 ]; then
    echo "Preflight failed; evidence directory: $OUT"
    exit 1
fi

ACTUAL_HARNESS_REVISION="$(python3 "$ROOT/scripts/scale-matrix.py" --revision 2>/dev/null || true)"
if [ "$ACTUAL_HARNESS_REVISION" != "$EXPECTED_HARNESS_REVISION" ]; then
    {
        echo "wrong scaling harness installed"
        echo "expected=$EXPECTED_HARNESS_REVISION"
        echo "actual=${ACTUAL_HARNESS_REVISION:-<none>}"
        echo "scale_matrix=$ROOT/scripts/scale-matrix.py"
        echo "Re-copy the current Prompt 8 scripts before running the matrix."
    } | tee -a "$OUT/preflight.txt"
    (
        cd "$(dirname "$OUT")" || exit 1
        zip -qr "$ARCHIVE" "$(basename "$OUT")"
    )
    echo "Upload archive: $ARCHIVE"
    exit 2
fi

{
    echo "date=$(date -Is)"
    echo "base_model=$BASE_MODEL"
    echo "rounds=$ROUNDS"
    echo "cooldown_seconds=$COOLDOWN"
    echo "prompt_targets=$PROMPTS"
    echo "concurrency_levels=$CONCURRENCY"
    echo "generated_tokens_per_request=$GENERATED"
    echo "max_models=$MAX_MODELS"
    echo "discover=$DISCOVER"
    echo "harness_revision=$ACTUAL_HARNESS_REVISION"
    echo "scale_matrix_sha256=$(sha256sum "$ROOT/scripts/scale-matrix.py" | awk '{print $1}')"
    echo "validator_sha256=$(sha256sum "$ROOT/scripts/validate-comparison8-machine.sh" | awk '{print $1}')"
    echo "air_version=$(air-server --version 2>&1 | head -n1)"
    echo "llama_server=$LLAMA_SERVER"
    "$LLAMA_SERVER" --version 2>&1 | head -n3 || true
    uname -a
    nvidia-smi
} > "$OUT/preflight.txt" 2>&1

cmd=(
    python3 "$ROOT/scripts/scale-matrix.py"
    --output-dir "$OUT"
    --base-model "$BASE_MODEL"
    --max-models "$MAX_MODELS"
    --air-cli air-cli
    --air-server air-server
    --air-verify air-verify
    --llama-server "$LLAMA_SERVER"
    --llama-bench "$LLAMA_BENCH"
    --verify-script "$ROOT/scripts/verify-llama-teacher.py"
    --compare-script "$ROOT/scripts/compare-llama.py"
    --rounds "$ROUNDS"
    --cooldown "$COOLDOWN"
    --prompt-targets "$PROMPTS"
    --concurrency-levels "$CONCURRENCY"
    --generated "$GENERATED"
)

if [ "$DISCOVER" = "1" ]; then
    cmd+=(--discover)
fi

# Any explicit arguments are additional candidate model paths. They are retained
# even when automatic discovery is enabled.
for model in "$@"; do
    cmd+=(--model "$model")
done

"${cmd[@]}" 2>&1 | tee "$OUT/harness.txt"
RC=${PIPESTATUS[0]}
echo "harness_exit=$RC" > "$OUT/exit-codes.txt"

(
    cd "$(dirname "$OUT")" || exit 1
    zip -qr "$ARCHIVE" "$(basename "$OUT")"
)

echo
echo "========================================"
echo "Scaling harness exit code: $RC"
echo "Matrix report: $OUT/SCALING_MATRIX.md"
echo "Model inventory: $OUT/MODEL_INVENTORY.json"
echo "Upload archive:"
echo "$ARCHIVE"
echo "========================================"

exit "$RC"
