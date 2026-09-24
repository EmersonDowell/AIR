#!/usr/bin/env bash
set -euo pipefail
AIR_ROOT="${AIR_ROOT:-$HOME/Projects/AIR}"
OUT="${OUT:-$HOME/Downloads}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/air-web-source.XXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT
mkdir -p "$WORK/repo" "$WORK/runtime"

copy_if() { local p="$1"; local dst="$2"; [[ -e "$p" ]] && cp -a -- "$p" "$dst" || true; }
copy_if "$AIR_ROOT/run-air.sh" "$WORK/repo/"
copy_if "$AIR_ROOT/CMakeLists.txt" "$WORK/repo/"
copy_if "$AIR_ROOT/src/server/main.cpp" "$WORK/repo/"
copy_if "$AIR_ROOT/src/server/protocol.cpp" "$WORK/repo/"
copy_if "$AIR_ROOT/web/index.html" "$WORK/repo/"

{
  printf 'AIR_ROOT=%s\n' "$AIR_ROOT"
  printf 'DATE_UTC=%s\n' "$STAMP"
  printf '\n== git ==\n'; git -C "$AIR_ROOT" status --short --branch 2>&1 || true
  printf '\n== version references ==\n'; grep -R --line-number --include='*.hpp' --include='*.cpp' --include='CMakeLists.txt' '0\.9\.12' "$AIR_ROOT" 2>/dev/null | head -n80 || true
  printf '\n== web roots ==\n'; find "$AIR_ROOT/web" "$HOME/.local/share/air/web" -maxdepth 3 -type f -printf '%p\n' 2>/dev/null | sort || true
  printf '\n== setup.js bounded search ==\n'; find "$AIR_ROOT/web" "$HOME/.local/share/air/web" -type f -name 'setup.js' -print 2>/dev/null || true
  printf '\n== air-server ==\n'; command -v air-server 2>/dev/null || true; ls -l "$HOME/.local/bin/air-server" "$AIR_ROOT/build-release/air-server" "$AIR_ROOT/build/air-server" 2>/dev/null || true
  printf '\n== running ==\n'; pgrep -a -u "$(id -u)" air-server 2>/dev/null || true
} > "$WORK/runtime/state.txt"

for endpoint in health model runtime events metrics v1/models; do
  safe="${endpoint//\//_}"
  curl -sS --max-time 4 "http://127.0.0.1:8181/$endpoint" > "$WORK/runtime/$safe.txt" 2>&1 || true
done

archive="$OUT/AIR-Web-Source-$STAMP.zip"
mkdir -p "$OUT"
( cd "$WORK" && zip -qr "$archive" . )
printf '%s\n' "$archive"
