#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, json
from pathlib import Path

def characterize(path: Path, expected_prompt_tokens: int = 160,
                 expected_concurrency: int = 8,
                 expected_strategy: str | None = "dense-c8") -> dict:
    doc = json.loads(path.read_text())
    errors = []
    runs = doc.get("runs") if isinstance(doc.get("runs"), list) else []
    cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    summary = doc.get("summary") if isinstance(doc.get("summary"), dict) else {}

    if doc.get("schema") != "air.benchmark.v11":
        errors.append(f"unexpected benchmark schema {doc.get('schema')!r}")
    if cfg.get("concurrency") != expected_concurrency:
        errors.append(f"concurrency={cfg.get('concurrency')!r}, expected {expected_concurrency}")
    if len(runs) != expected_concurrency:
        errors.append(f"measured runs={len(runs)}, expected {expected_concurrency}")
    bad_prompts = [r.get("prompt_tokens") for r in runs if r.get("prompt_tokens") != expected_prompt_tokens]
    if bad_prompts:
        errors.append(f"not every request used exact p{expected_prompt_tokens}: {bad_prompts}")

    counts = collections.Counter(str(r.get("strategy_id")) for r in runs)
    reasons = collections.Counter(str(r.get("strategy_decision_reason")) for r in runs)

    if expected_strategy:
        if counts != {expected_strategy: expected_concurrency}:
            errors.append(
                f"adaptive cohort did not receive one coherent {expected_strategy} plan: "
                f"strategy_counts={dict(counts)}"
            )
    elif len(counts) != 1 or "static" in counts:
        errors.append(f"adaptive cohort did not receive one coherent qualified plan: {dict(counts)}")

    native_batches = summary.get("native_decode_batches")
    native_sequences = summary.get("native_decode_sequences")
    if not isinstance(native_batches, int) or native_batches <= 0:
        errors.append(f"native_decode_batches={native_batches!r}, expected >0")
    if not isinstance(native_sequences, int) or native_sequences <= 0:
        errors.append(f"native_decode_sequences={native_sequences!r}, expected >0")

    return {
        "schema": "air.prompt15.gate5-characterization.v1",
        "benchmark": str(path),
        "expected_prompt_tokens": expected_prompt_tokens,
        "expected_concurrency": expected_concurrency,
        "expected_strategy": expected_strategy,
        "strategy_counts": dict(counts),
        "decision_reason_counts": dict(reasons),
        "native_decode_batches": native_batches,
        "native_decode_sequences": native_sequences,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", type=Path)
    ap.add_argument("--expected-prompt-tokens", type=int, default=160)
    ap.add_argument("--expected-concurrency", type=int, default=8)
    ap.add_argument("--expected-strategy", default="dense-c8")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = characterize(args.benchmark, args.expected_prompt_tokens,
                          args.expected_concurrency, args.expected_strategy or None)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__":
    raise SystemExit(main())
