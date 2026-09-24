#!/usr/bin/env bash
set +e
set +u
set +E
set +o pipefail 2>/dev/null

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT="${1:-$DEFAULT_ROOT}"
FAIL=0

pass(){ echo "[PASS] $*"; }
fail(){ echo "[FAIL] $*"; FAIL=1; }

check_lines() {
  local rel="$1" max="$2"
  local path="$ROOT/$rel"
  if [ ! -f "$path" ]; then
    fail "missing $rel"
    return
  fi
  local lines
  lines="$(wc -l <"$path")"
  if [ "$lines" -gt "$max" ]; then
    fail "$rel: $lines lines > budget $max"
  else
    pass "$rel: $lines/$max lines"
  fi
}

require_text(){
  local rel="$1" needle="$2" label="$3"
  if grep -Fq -- "$needle" "$ROOT/$rel" 2>/dev/null; then
    pass "$label"
  else
    fail "$label"
  fi
}

check_lines "AGENTS.md" 120
check_lines "docs/agent/CURRENT.md" 180
check_lines "docs/agent/ARCHITECTURE.md" 220
check_lines "docs/agent/FINDINGS.md" 220
check_lines "docs/agent/EXPERIMENTS.md" 160

for spec in \
  "docs/agent/CURRENT.md|## Status|CURRENT status heading" \
  "docs/agent/CURRENT.md|## NEXT ACTION|CURRENT NEXT ACTION heading" \
  "docs/agent/ARCHITECTURE.md|## State domains|architecture state-domain section" \
  "docs/agent/FINDINGS.md|# AIR Active Findings Index|findings heading" \
  "docs/agent/EXPERIMENTS.md|# AIR Experiment Ledger|experiment-ledger heading"; do
  IFS='|' read -r rel needle label <<<"$spec"
  require_text "$rel" "$needle" "$label"
done

next_count="$(grep -Fc '## NEXT ACTION' "$ROOT/docs/agent/CURRENT.md" 2>/dev/null)"
if [ "$next_count" -eq 1 ]; then
  pass "exactly one CURRENT NEXT ACTION"
else
  fail "CURRENT NEXT ACTION count=$next_count, expected 1"
fi

if grep -RIl --exclude='*.pre-*' 'Prompt 15 is CURRENT' "$ROOT/docs/agent" >/dev/null 2>&1; then
  fail "stale Prompt-15 CURRENT wording in compact context"
else
  pass "no stale Prompt-15 CURRENT wording"
fi

if grep -Eq 'Current substep:.*P3D|P3D.*\*\*ACTIVE\*\*' "$ROOT/docs/agent/CURRENT.md" 2>/dev/null; then
  fail "rejected P3D is still presented as active"
else
  pass "P3D not presented as active"
fi
require_text "docs/agent/CURRENT.md" "P3D batched greedy selection is **REJECTED**" "CURRENT records P3D rejection"
require_text "docs/agent/EXPERIMENTS.md" "| P3D-GREEDY " "experiment ledger records P3D"

cmake_version="$(sed -nE 's/^[[:space:]]*project\(AIR VERSION ([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' "$ROOT/CMakeLists.txt" | head -n1)"
current_version="$(sed -nE 's/^- AIR production baseline: \*\*([0-9]+\.[0-9]+\.[0-9]+)\*\*\.$/\1/p' "$ROOT/docs/agent/CURRENT.md" | head -n1)"
if [ -n "$cmake_version" ] && [ "$cmake_version" = "$current_version" ]; then
  pass "CURRENT version matches CMake project version $cmake_version"
else
  fail "version mismatch CMake=${cmake_version:-missing} CURRENT=${current_version:-missing}"
fi

expected_cuda="$(sed -nE 's/^- Retained CUDA baseline SHA-256: `([0-9a-f]{64})`\.$/\1/p' "$ROOT/docs/agent/CURRENT.md" | head -n1)"
actual_cuda="$(sha256sum "$ROOT/src/cuda/cuda_backend.cu" 2>/dev/null | awk '{print $1}')"
if [ -n "$expected_cuda" ] && [ "$expected_cuda" = "$actual_cuda" ]; then
  pass "retained CUDA fingerprint matches CURRENT"
else
  fail "CUDA fingerprint mismatch expected=${expected_cuda:-missing} actual=${actual_cuda:-missing}"
fi

for rel in \
  scripts/check-agent-context.sh \
  scripts/competitive-convergence/run-prompt1-census.sh \
  CMakeLists.txt; do
  if [ -e "$ROOT/$rel" ]; then pass "required path exists: $rel"; else fail "missing required path: $rel"; fi
done

if grep -Fq 'nsys stats --force-export=true' "$ROOT/scripts/competitive-convergence/run-prompt1-census.sh" 2>/dev/null; then
  pass "Prompt-1 Nsight stats forces export"
else
  fail "Prompt-1 Nsight stats does not force export"
fi
if grep -Fq 'NSYS_STATS_FAILED' "$ROOT/scripts/competitive-convergence/run-prompt1-census.sh" 2>/dev/null; then
  pass "Prompt-1 Nsight stats fail-closed guard present"
else
  fail "Prompt-1 Nsight stats fail-closed guard missing"
fi
if grep -Fq 'nsys unavailable; required Prompt-1 profiler evidence cannot be collected.' "$ROOT/scripts/competitive-convergence/run-prompt1-census.sh" 2>/dev/null && \
   grep -Fq '"$OUT/nsys/REQUIRED-STATS.FAIL"' "$ROOT/scripts/competitive-convergence/run-prompt1-census.sh" 2>/dev/null && \
   ! grep -Fq 'nsys unavailable" >"$OUT/nsys/SKIPPED.txt"' "$ROOT/scripts/competitive-convergence/run-prompt1-census.sh" 2>/dev/null; then
  pass "Prompt-1 Nsight unavailable path fails closed"
else
  fail "Prompt-1 Nsight unavailable path does not fail closed"
fi

if [ "$FAIL" -eq 0 ]; then
  echo "Agent context check PASS."
else
  echo "Agent context check FAIL."
fi

return "$FAIL" 2>/dev/null || exit "$FAIL"
