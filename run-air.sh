#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/model.gguf [air-server options...]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$1"; shift
if [[ -x "$ROOT/build/air-server" ]]; then SERVER="$ROOT/build/air-server"
elif [[ -x "$ROOT/build-release/air-server" ]]; then SERVER="$ROOT/build-release/air-server"
elif command -v air-server >/dev/null 2>&1; then SERVER="$(command -v air-server)"
else
  echo "air-server not found. Build AIR or run scripts/install-local.sh first." >&2
  exit 3
fi
exec "$SERVER" -m "$MODEL" --backend auto "$@"
