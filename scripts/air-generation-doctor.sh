#!/usr/bin/env bash
set -u

BASE="${AIR_URL:-http://127.0.0.1:8181}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/air-generation-doctor.XXXXXX")"
trap 'rm -rf -- "$TMP"' EXIT

prompt='Reply with exactly: AIR generation path is working.'
json='{"prompt":"Reply with exactly: AIR generation path is working.","max_tokens":32,"temperature":0,"top_p":1,"top_k":0,"seed":0,"stream":false}'
stream_json='{"prompt":"Reply with exactly: AIR generation path is working.","max_tokens":32,"temperature":0,"top_p":1,"top_k":0,"seed":0,"stream":true}'

printf 'AIR generation doctor\n'
printf 'URL: %s\n\n' "$BASE"

health_code="$(curl -sS --max-time 3 -o "$TMP/health.json" -w '%{http_code}' "$BASE/health" 2>"$TMP/health.err" || true)"
if [[ "$health_code" != "200" ]]; then
  printf 'FAIL /health HTTP %s\n' "${health_code:-000}"
  cat "$TMP/health.err" >&2 || true
  exit 1
fi
printf 'PASS /health HTTP 200\n'

printf '\n1. Native non-streaming /generate\n'
non_code="$(curl -sS --max-time 180 -D "$TMP/non.headers" -o "$TMP/non.body" -w '%{http_code}' \
  "$BASE/generate" -H 'Content-Type: application/json' -H 'Accept: application/json' -d "$json" 2>"$TMP/non.err" || true)"
printf 'HTTP %s\n' "${non_code:-000}"
tr -d '\r' < "$TMP/non.headers" | grep -iE '^(content-type|content-length):' || true
if [[ -s "$TMP/non.err" ]]; then sed 's/^/curl: /' "$TMP/non.err"; fi
printf 'Body:\n'
if command -v jq >/dev/null 2>&1 && jq -e . "$TMP/non.body" >/dev/null 2>&1; then
  jq . "$TMP/non.body"
  non_text="$(jq -r '.text // .choices[0].text // .choices[0].message.content // empty' "$TMP/non.body" 2>/dev/null || true)"
  completion_tokens="$(jq -r '.usage.completion_tokens // .metrics.generated_tokens // empty' "$TMP/non.body" 2>/dev/null || true)"
  printf '\nExtracted text length: %s\n' "${#non_text}"
  printf 'Generated/completion tokens: %s\n' "${completion_tokens:-not reported}"
else
  cat "$TMP/non.body"
  non_text=''
fi

printf '\n2. Native streaming /generate (raw SSE)\n'
stream_code="$(curl -sS -N --max-time 180 -D "$TMP/stream.headers" -o "$TMP/stream.body" -w '%{http_code}' \
  "$BASE/generate" -H 'Content-Type: application/json' -H 'Accept: text/event-stream' -d "$stream_json" 2>"$TMP/stream.err" || true)"
printf 'HTTP %s\n' "${stream_code:-000}"
tr -d '\r' < "$TMP/stream.headers" | grep -iE '^(content-type|transfer-encoding|content-length):' || true
if [[ -s "$TMP/stream.err" ]]; then sed 's/^/curl: /' "$TMP/stream.err"; fi
printf 'Raw bytes: %s\n' "$(wc -c < "$TMP/stream.body" | tr -d ' ')"
printf 'First 40 raw lines (control characters shown):\n'
sed -n '1,40l' "$TMP/stream.body"

printf '\n3. Interpretation\n'
if [[ "$non_code" == "200" && -n "$non_text" ]]; then
  printf 'PASS backend generation produced text through non-streaming /generate.\n'
else
  printf 'FAIL/INCONCLUSIVE non-streaming /generate did not expose response text. This points at AIR/runtime or response-schema behavior, not only the browser stream parser.\n'
fi

if [[ "$stream_code" == "200" ]] && grep -q '^data:' "$TMP/stream.body"; then
  printf 'PASS streaming endpoint returned SSE data frames.\n'
else
  printf 'FAIL/INCONCLUSIVE streaming endpoint did not show normal SSE data frames.\n'
fi

printf '\nSaved temporary evidence was displayed above; no model files were read or copied.\n'
