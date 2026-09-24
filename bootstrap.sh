#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DEPS=0
BUILD=1
INSTALL=0

usage() {
  cat <<'EOF'
AIR public bootstrap helper

Usage:
  ./bootstrap.sh [--install-deps] [--no-build] [--install]

Options:
  --install-deps  On supported Debian/Ubuntu/Mint systems, explicitly install
                  missing base build packages with sudo apt.
  --no-build      Only inspect readiness.
  --install       Run ./scripts/install-local.sh after a successful build.

CUDA is detected but never silently installed.
EOF
}

while (($#)); do
  case "$1" in
    --install-deps) INSTALL_DEPS=1 ;;
    --no-build) BUILD=0 ;;
    --install) INSTALL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$ROOT"

echo "AIR bootstrap"
echo "Repository: $ROOT"
echo

missing=()

check_cmd() {
  local cmd="$1" label="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    printf '  READY %-18s %s\n' "$label" "$("$cmd" --version 2>/dev/null | head -n1 || true)"
  else
    printf '  MISS  %-18s\n' "$label"
    missing+=("$label")
  fi
}

echo "Base tools"
check_cmd c++ "C++ compiler"
check_cmd cmake "CMake"
check_cmd pkg-config "pkg-config"

if pkg-config --exists libpcre2-8 2>/dev/null; then
  echo "  READY PCRE2 development"
else
  echo "  MISS  PCRE2 development"
  missing+=("PCRE2")
fi

if printf '#include <boost/beast.hpp>\n#include <boost/json.hpp>\nint main(){}\n' |
   c++ -std=c++20 -x c++ -fsyntax-only - >/dev/null 2>&1; then
  echo "  READY Boost Beast + JSON headers"
else
  echo "  MISS  Boost Beast + JSON headers"
  missing+=("Boost")
fi

echo
echo "GPU acceleration"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  GPU   /' || true
else
  echo "  INFO  NVIDIA driver/GPU not detected; reference backend remains available."
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n1 | sed 's/^/  CUDA  /'
else
  echo "  INFO  nvcc not detected; CUDA build will not be forced."
fi

if ((${#missing[@]})); then
  echo
  echo "Missing base dependencies: ${missing[*]}"
  if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "Installing allowlisted base build packages with apt..."
      sudo apt-get update
      sudo apt-get install -y build-essential cmake pkg-config libpcre2-dev libboost-all-dev
    else
      echo "--install-deps currently supports apt-based systems only." >&2
      exit 3
    fi
  else
    echo
    echo "On Debian / Ubuntu / Linux Mint:"
    echo "  sudo apt update"
    echo "  sudo apt install -y build-essential cmake pkg-config libpcre2-dev libboost-all-dev"
    echo
    echo "Re-run ./bootstrap.sh after installing dependencies."
    exit 4
  fi
fi

if [[ "$BUILD" -eq 0 ]]; then
  echo
  echo "Readiness check complete."
  exit 0
fi

echo
echo "Building AIR..."
./scripts/build.sh

echo
echo "Running tests..."
ctest --test-dir build --output-on-failure

if [[ "$INSTALL" -eq 1 ]]; then
  echo
  echo "Installing AIR into the configured local prefix..."
  ./scripts/install-local.sh
fi

cat <<'EOF'

AIR build/test completed.

Next:
  ./run-air.sh /path/to/supported-model.gguf

Then open:
  http://127.0.0.1:8181
EOF
