#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "air.benchmark.v11"

def is_int_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)

def _validate_top_level_output_tokens(doc: dict[str, Any], runs: list[dict[str, Any]]) -> list[list[int]]:
    raw = doc.get("output_tokens")
    if not isinstance(raw, list):
        raise ValueError("top-level output_tokens exists but is not an array")
    if len(raw) != len(runs):
        raise ValueError(f"top-level output_tokens has {len(raw)} rows but runs has {len(runs)}")
    out: list[list[int]] = []
    for i, (ids, run) in enumerate(zip(raw, runs)):
        if not is_int_list(ids):
            raise ValueError(f"output_tokens[{i}] is not an integer token-id array")
        generated = run.get("generated_tokens")
        if not isinstance(generated, int) or generated < 0:
            raise ValueError(f"runs[{i}].generated_tokens is invalid")
        if len(ids) != generated:
            raise ValueError(
                f"output_tokens[{i}] length={len(ids)} does not match "
                f"runs[{i}].generated_tokens={generated}"
            )
        out.append(list(ids))
    return out

def discover_output_id_field(doc: dict[str, Any]) -> tuple[str, list[list[int]]]:
    runs = doc.get("runs")
    if not isinstance(runs, list) or len(runs) < 2 or not all(isinstance(r, dict) for r in runs):
        raise ValueError("benchmark must contain at least two measured run objects")

    # air.benchmark.v11 machine evidence stores measured token sequences here.
    # If the canonical field is present, validate it strictly rather than silently
    # falling back to heuristic discovery.
    if "output_tokens" in doc:
        return "output_tokens", _validate_top_level_output_tokens(doc, runs)

    # Backward/future defensive path: discover a run-local field structurally.
    common = set(runs[0])
    for r in runs[1:]:
        common &= set(r)
    candidates: list[str] = []
    for key in sorted(common):
        vals = [r.get(key) for r in runs]
        if not all(is_int_list(v) for v in vals):
            continue
        if all(isinstance(r.get("generated_tokens"), int) and len(v) == r["generated_tokens"]
               for r, v in zip(runs, vals)):
            candidates.append(key)
    if len(candidates) != 1:
        list_fields = sorted(k for k in common if all(isinstance(r.get(k), list) for r in runs))
        raise ValueError(
            "could not identify exactly one measured-output token-id field; "
            f"candidates={candidates}, run_local_list_fields={list_fields}"
        )
    key = candidates[0]
    return key, [list(r[key]) for r in runs]

def validate(path: Path, expected_prompt_tokens: int | None = None,
             expected_concurrency: int | None = None) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    errors: list[str] = []
    if doc.get("schema") != EXPECTED_SCHEMA:
        errors.append(f"schema={doc.get('schema')!r}, expected {EXPECTED_SCHEMA!r}")
    cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    if expected_concurrency is not None and cfg.get("concurrency") != expected_concurrency:
        errors.append(f"config.concurrency={cfg.get('concurrency')!r}, expected {expected_concurrency}")
    runs = doc.get("runs") if isinstance(doc.get("runs"), list) else []
    if expected_prompt_tokens is not None:
        bad = [r.get("prompt_tokens") for r in runs
               if isinstance(r, dict) and r.get("prompt_tokens") != expected_prompt_tokens]
        if bad:
            errors.append(f"run prompt token counts do not all equal {expected_prompt_tokens}: {bad[:8]}")

    field = None
    outputs: list[list[int]] = []
    try:
        field, outputs = discover_output_id_field(doc)
    except Exception as exc:
        errors.append(str(exc))

    if outputs:
        baseline = outputs[0]
        mismatches = [i for i, ids in enumerate(outputs[1:], 1) if ids != baseline]
        if mismatches:
            errors.append(f"deterministic same-prompt output token IDs differ at run indexes {mismatches}")
        if not baseline:
            errors.append("measured output token IDs are empty")

    return {
        "schema": "air.prompt15.output-isolation.v2",
        "benchmark": str(path),
        "benchmark_schema": doc.get("schema"),
        "token_id_field": field,
        "measured_runs": len(runs),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", type=Path)
    ap.add_argument("--expected-prompt-tokens", type=int)
    ap.add_argument("--expected-concurrency", type=int)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = validate(args.benchmark, args.expected_prompt_tokens, args.expected_concurrency)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__":
    raise SystemExit(main())
