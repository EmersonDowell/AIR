#!/usr/bin/env bash
set -euo pipefail

AIR_ROOT="${AIR_ROOT:-$HOME/Projects/AIR}"
MODEL_DIR="${AIR_MODEL_DIR:-$HOME/Models/AIR}"
HOST="127.0.0.1"
PORT="8181"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$AIR_ROOT" ]] || fail "AIR repository not found: $AIR_ROOT"
[[ -x "$AIR_ROOT/run-air.sh" ]] || fail "Canonical AIR launcher missing: $AIR_ROOT/run-air.sh"
[[ -f "$AIR_ROOT/web/index.html" ]] || fail "AIR web application missing: $AIR_ROOT/web/index.html"
grep -q 'name="air-web-version" content="3.1.0"' "$AIR_ROOT/web/index.html" || fail "AIR web tree is not v3.1.0. Re-run the v3 installer."
[[ -d "$MODEL_DIR" ]] || fail "Model directory not found: $MODEL_DIR"

if curl -fsS --max-time 1 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  printf 'AIR is already running at http://%s:%s\n' "$HOST" "$PORT"
  printf 'Stop the existing AIR process before switching models.\n'
  exit 0
fi

mapfile -d '' MODELS < <(find "$MODEL_DIR" -maxdepth 1 -type f -iname '*.gguf' -print0 | sort -z)
(( ${#MODELS[@]} > 0 )) || fail "No GGUF files found directly inside $MODEL_DIR"

printf '\nAIR 0.9.12 · Superior MI Labs\n'
printf 'Interactive local launcher\n\n'
printf 'Repository : %s\n' "$AIR_ROOT"
printf 'Models     : %s\n\n' "$MODEL_DIR"
printf 'Available GGUF models:\n'
for i in "${!MODELS[@]}"; do
  name="$(basename -- "${MODELS[$i]}")"
  note='AIR validates support at launch'
  shopt -s nocasematch
  if [[ "$name" == *qwen2.5* ]]; then note='Qwen2.5 · public supported family';
  elif [[ "$name" == *qwen2* ]]; then note='Qwen2-family · public supported family'; fi
  shopt -u nocasematch
  printf '  %2d) %-54s %s\n' "$((i + 1))" "$name" "$note"
done
printf '\nChoose a model [1-%d] (q to quit): ' "${#MODELS[@]}"
read -r choice
[[ "$choice" != 'q' && "$choice" != 'Q' ]] || exit 0
[[ "$choice" =~ ^[0-9]+$ ]] || fail 'Selection must be a number.'
(( choice >= 1 && choice <= ${#MODELS[@]} )) || fail 'Selection is out of range.'
selected="${MODELS[$((choice - 1))]}"
printf '\nSelected: %s\n' "$selected"
printf 'Launching through canonical AIR run-air.sh…\n'
printf 'Open http://%s:%s after AIR reports healthy.\n\n' "$HOST" "$PORT"
cd "$AIR_ROOT"
exec ./run-air.sh "$selected"
