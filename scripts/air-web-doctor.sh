#!/usr/bin/env bash
set -u

AIR_ROOT="${AIR_ROOT:-$HOME/Projects/AIR}"
PREFIX="${AIR_PREFIX:-$HOME/.local}"
BASE="${AIR_URL:-http://127.0.0.1:8181}"
EXPECTED="3.1.0"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/air-web-doctor-v3.XXXXXX")"
trap 'rm -rf -- "$TMP"' EXIT
local_fail=0
served_fail=0

version_of() {
  local file="$1"
  [[ -f "$file" ]] || { printf 'missing'; return; }
  sed -n 's/.*name="air-web-version" content="\([^"]*\)".*/\1/p' "$file" | head -n1
}

printf 'AIR web doctor v3.1\n'
printf 'AIR root : %s\n' "$AIR_ROOT"
printf 'URL      : %s\n\n' "$BASE"

printf 'A. Local deployment truth\n'
for tree in "$AIR_ROOT/web" "$PREFIX/share/air/web"; do
  printf '  %-42s version=%s\n' "$tree" "$(version_of "$tree/index.html")"
  if [[ "$(version_of "$tree/index.html")" != "$EXPECTED" ]]; then local_fail=1; fi
  if [[ -d "$tree" ]]; then
    stale="$(find "$tree" -type f -name 'setup.js' -print 2>/dev/null | head -n1)"
    if [[ -n "$stale" ]]; then printf '    FAIL stale setup.js: %s\n' "$stale"; local_fail=1; else printf '    PASS no setup.js file\n'; fi
    refs="$(grep -R --fixed-strings --line-number 'setup.js' "$tree" 2>/dev/null | head -n3 || true)"
    if [[ -n "$refs" ]]; then printf '    FAIL setup.js reference:\n%s\n' "$refs"; local_fail=1; else printf '    PASS no setup.js reference\n'; fi
  fi
done

printf '\nB. Local JavaScript parse\n'
if command -v node >/dev/null 2>&1 && [[ -d "$AIR_ROOT/web" ]]; then
  js_count=0
  while IFS= read -r -d '' js; do
    js_count=$((js_count + 1))
    if node --check "$js" >/dev/null 2>"$TMP/node.err"; then
      printf '  PASS %s\n' "${js#$AIR_ROOT/web/}"
    else
      printf '  FAIL %s\n' "${js#$AIR_ROOT/web/}"
      sed 's/^/       /' "$TMP/node.err"
      local_fail=1
    fi
  done < <(find "$AIR_ROOT/web" -type f -name '*.js' -print0 | sort -z)
  printf '  checked %d JavaScript files\n' "$js_count"
else
  printf '  SKIP node or repo web tree unavailable\n'
fi

printf '\nC. Canonical AIR launcher\n'
if [[ -x "$AIR_ROOT/run-air.sh" ]]; then
  printf '  PASS %s/run-air.sh exists and is executable\n' "$AIR_ROOT"
else
  printf '  FAIL canonical run-air.sh missing or not executable\n'; local_fail=1
fi
if [[ -x "$AIR_ROOT/start-air.sh" ]] && grep -q 'exec ./run-air.sh' "$AIR_ROOT/start-air.sh"; then
  printf '  PASS public start-air.sh delegates to run-air.sh\n'
else
  printf '  FAIL public start-air.sh missing or not delegating to run-air.sh\n'; local_fail=1
fi

printf '\nD. Running server\n'
health_code="$(curl -sS --max-time 2 -o "$TMP/health" -w '%{http_code}' "$BASE/health" 2>"$TMP/curl.err" || true)"
if [[ "$health_code" != '200' ]]; then
  printf '  AIR is not listening now (HTTP %s).\n' "${health_code:-000}"
  printf '  This prevents served-byte verification, but does NOT invalidate local syntax checks.\n'
  printf '  Start it with: cd %s && ./start-air.sh\n' "$AIR_ROOT"
else
  printf '  PASS /health HTTP 200\n'
  for endpoint in /model /runtime /events /metrics /v1/models; do
    code="$(curl -sS --max-time 4 -o "$TMP/body" -w '%{http_code}' "$BASE$endpoint" 2>"$TMP/curl.err" || true)"
    printf '  %-12s HTTP %s\n' "$endpoint" "${code:-000}"
    [[ "$code" == '200' ]] || served_fail=1
  done

  printf '\nE. Served web identity\n'
  index_code="$(curl -sS --max-time 4 -o "$TMP/index.html" -w '%{http_code}' "$BASE/" 2>"$TMP/curl.err" || true)"
  served_version="$(version_of "$TMP/index.html")"
  printf '  GET / HTTP %s · web=%s\n' "${index_code:-000}" "$served_version"
  if [[ "$index_code" != '200' || "$served_version" != "$EXPECTED" ]]; then
    printf '  FAIL server is not serving AIR Web %s\n' "$EXPECTED"
    served_fail=1
  fi
  if grep -q '/app/main-v310.js' "$TMP/index.html"; then printf '  PASS index loads main-v310.js\n'; else printf '  FAIL main-v310.js missing from served index\n'; served_fail=1; fi
  if grep -q 'setup.js' "$TMP/index.html"; then printf '  FAIL served index references setup.js\n'; served_fail=1; else printf '  PASS served index does not reference setup.js\n'; fi

  printf '\nF. Served script bytes\n'
  scripts="$(grep -oE '<script[^>]+src="[^"]+"' "$TMP/index.html" | sed -n 's/.*src="\([^"]*\)".*/\1/p')"
  if [[ -z "$scripts" ]]; then printf '  FAIL no external scripts found\n'; served_fail=1; fi
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    out="$TMP/$(basename "$path")"
    code="$(curl -sS --max-time 4 -D "$out.headers" -o "$out" -w '%{http_code}' "$BASE$path" 2>"$TMP/curl.err" || true)"
    ctype="$(tr -d '\r' <"$out.headers" | grep -i '^content-type:' | head -n1 | cut -d: -f2- | sed 's/^[[:space:]]*//' || true)"
    printf '  %-38s HTTP %s · %s' "$path" "${code:-000}" "${ctype:-no content-type}"
    if [[ "$code" == '200' ]] && command -v node >/dev/null 2>&1 && node --check "$out" >/dev/null 2>"$TMP/node.err"; then
      printf ' · syntax PASS\n'
    else
      printf ' · FAIL\n'
      [[ -s "$TMP/node.err" ]] && sed 's/^/       /' "$TMP/node.err"
      served_fail=1
    fi
  done <<< "$scripts"

  legacy_code="$(curl -sS --max-time 3 -o /dev/null -w '%{http_code}' "$BASE/app/views/setup.js" 2>/dev/null || true)"
  printf '\nG. Legacy path probe\n'
  printf '  /app/views/setup.js -> HTTP %s\n' "${legacy_code:-000}"
  if [[ "$legacy_code" == '200' ]]; then
    printf '  FAIL server still exposes legacy setup.js\n'; served_fail=1
  else
    printf '  PASS legacy setup.js is not served\n'
  fi
fi

printf '\nH. Process evidence\n'
pgrep -a -u "$(id -u)" air-server 2>/dev/null || printf '  no air-server process currently visible\n'

printf '\nRESULT\n'
if (( local_fail == 0 )); then printf '  LOCAL DEPLOYMENT: PASS\n'; else printf '  LOCAL DEPLOYMENT: FAIL\n'; fi
if [[ "$health_code" == '200' ]]; then
  if (( served_fail == 0 )); then printf '  SERVED APPLICATION: PASS\n'; else printf '  SERVED APPLICATION: FAIL\n'; fi
else
  printf '  SERVED APPLICATION: NOT TESTED · AIR is stopped\n'
fi

(( local_fail == 0 && served_fail == 0 )) || exit 1
