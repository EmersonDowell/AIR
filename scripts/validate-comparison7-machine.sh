#!/usr/bin/env bash
# AIR Comparison Prompt 7: controlled AIR vs llama.cpp evidence collection.
# This script adds no runtime feature and changes no execution policy.

MODEL="${1:-$HOME/Models/AIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-$HOME/Projects/llama.cpp-tq3}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_ROOT/build/bin/llama-server}"
LLAMA_BENCH="${LLAMA_BENCH:-$LLAMA_ROOT/build/bin/llama-bench}"
ROUNDS="${AIR_COMPARE_ROUNDS:-5}"
COOLDOWN="${AIR_COMPARE_COOLDOWN:-5}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Downloads/AIR-Comparison7-$STAMP"
ARCHIVE="$HOME/Downloads/AIR-Comparison7-$STAMP.zip"

mkdir -p "$OUT"

fail=0
for tool in air-server air-cli air-verify air-bench python3 nvidia-smi; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "missing required tool: $tool" | tee -a "$OUT/preflight.txt"
        fail=1
    fi
done
for path in "$MODEL" "$LLAMA_SERVER" "$LLAMA_BENCH" "$ROOT/scripts/verify-llama-teacher.py"; do
    if [ ! -e "$path" ]; then
        echo "missing required path: $path" | tee -a "$OUT/preflight.txt"
        fail=1
    fi
done
if [ "$fail" -ne 0 ]; then
    echo "Preflight failed; evidence directory: $OUT"
    exit 1
fi

{
    echo "date=$(date -Is)"
    echo "model=$MODEL"
    echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
    echo "model_bytes=$(stat -c %s "$MODEL")"
    echo "air_version=$(air-server --version 2>&1 | head -n1)"
    echo "llama_server=$LLAMA_SERVER"
    echo "llama_bench=$LLAMA_BENCH"
    echo "rounds=$ROUNDS"
    echo "cooldown_seconds=$COOLDOWN"
    uname -a
    nvcc --version 2>/dev/null || true
    nvidia-smi
} > "$OUT/preflight.txt" 2>&1

python3 "$ROOT/scripts/compare-llama.py" \
    --model "$MODEL" \
    --llama-server "$LLAMA_SERVER" \
    --llama-bench "$LLAMA_BENCH" \
    --verify-script "$ROOT/scripts/verify-llama-teacher.py" \
    --output-dir "$OUT" \
    --rounds "$ROUNDS" \
    --order-seed 1337 \
    --sampling-seed 424242 \
    --cooldown "$COOLDOWN" \
    --context 32768 \
    2>&1 | tee "$OUT/harness.txt"
RC=${PIPESTATUS[0]}

echo "harness_exit=$RC" > "$OUT/exit-codes.txt"

python3 - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {"harness_exit": int((root / "exit-codes.txt").read_text().split("=",1)[1])}
comparison = root / "comparison.json"
if comparison.exists():
    obj = json.loads(comparison.read_text())
    summary["schema"] = obj.get("schema")
    summary["model_sha256"] = obj.get("meta", {}).get("model_sha256")
    summary["workloads"] = obj.get("meta", {}).get("workloads")
    summary["ratios"] = obj.get("ratios")
external = root / "external-verification.json"
if external.exists():
    ext = json.loads(external.read_text())
    summary["external_verification"] = ext.get("summary")
(root / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

(
    cd "$(dirname "$OUT")" || exit 1
    zip -qr "$ARCHIVE" "$(basename "$OUT")"
)

echo
echo "========================================"
echo "Comparison harness exit code: $RC"
echo "Report: $OUT/REPORT.md"
echo "Raw comparison: $OUT/comparison.json"
echo "Upload archive:"
echo "$ARCHIVE"
echo "========================================"

exit "$RC"
