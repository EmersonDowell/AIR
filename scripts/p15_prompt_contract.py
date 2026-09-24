#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

DEFAULT_BASE = (
    "Adaptive inference runtimes execute transformer layers while tracking memory ownership, "
    "scheduling decisions, numerical evidence, and request latency under a controlled workload. "
)
DEFAULT_TARGETS = (54, 128, 160, 262, 512, 1042, 2048)


def parse_token_count(output: str) -> int:
    text = output.strip()
    if not text:
        raise ValueError("empty tokenizer output")

    # Prefer machine-readable forms when available.
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
            return len(obj)
        if isinstance(obj, dict):
            for key in ("token_count", "n_tokens", "count", "prompt_tokens"):
                value = obj.get(key)
                if isinstance(value, int) and value >= 0:
                    return value
            for key in ("tokens", "token_ids", "ids"):
                value = obj.get(key)
                if isinstance(value, list) and all(isinstance(x, int) for x in value):
                    return len(value)
    except json.JSONDecodeError:
        pass

    # Common human-readable count formats.
    patterns = (
        r"\btokens?\s*\(\s*(\d+)\s*\)",
        r"\btoken[_ ]count\s*[:=]\s*(\d+)\b",
        r"\bn_tokens\s*[:=]\s*(\d+)\b",
        r"\bcount\s*[:=]\s*(\d+)\s+tokens?\b",
        r"\b(\d+)\s+tokens?\b",
    )
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return int(m.group(1))

    # A labelled token vector such as "tokens: [1, 2, 3]".
    m = re.search(r"\btokens?(?:_ids)?\s*:\s*\[([^\]]*)\]", text, flags=re.I | re.S)
    if m:
        nums = re.findall(r"-?\d+", m.group(1))
        return len(nums)

    raise ValueError(f"unable to parse token count from tokenizer output: {text[:300]!r}")


class AirTokenizerCounter:
    def __init__(self, air_cli: str, model: str):
        self.air_cli = air_cli
        self.model = model
        self.cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        if text in self.cache:
            return self.cache[text]
        proc = subprocess.run(
            [self.air_cli, "tokenize", self.model, text],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"air-cli tokenize failed rc={proc.returncode}: {proc.stdout[-2000:]}")
        count = parse_token_count(proc.stdout)
        self.cache[text] = count
        return count


def _max_repeats_not_over(counter: Callable[[str], int], prefix: str, unit: str, target: int, hi: int) -> tuple[str, int]:
    lo = 0
    best_text = prefix
    best_count = counter(prefix)
    hi = max(0, hi)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = prefix + unit * mid
        count = counter(candidate)
        if count <= target:
            if count >= best_count:
                best_text, best_count = candidate, count
            lo = mid + 1
        else:
            hi = mid - 1
    return best_text, best_count


def synthesize_exact(counter: Callable[[str], int], target: int, base: str = DEFAULT_BASE) -> str:
    if target <= 0:
        raise ValueError("target must be positive")

    empty_count = counter("")
    if empty_count > target:
        raise RuntimeError(f"tokenizer has {empty_count} tokens for empty input, already above target {target}")

    # First use a meaningful repeated technical sentence. Repetition count is found by binary search,
    # so this remains tokenizer-driven rather than relying on the historical Qwen token formula.
    text, count = _max_repeats_not_over(counter, "", base, target, max(1, target))
    if count == target:
        return text

    # Then close the residual with common low-fragmentation suffixes. Each filler is searched in bulk,
    # not appended one subprocess invocation at a time.
    fillers = (
        " token",
        " inference",
        " runtime",
        " data",
        " memory",
        " model",
        " request",
        " evidence",
        " GPU",
        " test",
        ".",
        ",",
        " 0",
        " x",
    )
    progress = True
    rounds = 0
    while count < target and progress and rounds < 8:
        rounds += 1
        progress = False
        start_text, start_count = text, count
        best_text, best_count = text, count
        remaining = target - count
        for filler in fillers:
            candidate_text, candidate_count = _max_repeats_not_over(
                counter, start_text, filler, target, max(1, remaining + 4)
            )
            if candidate_count > best_count:
                best_text, best_count = candidate_text, candidate_count
            if candidate_count == target:
                return candidate_text
        if best_count > count:
            text, count = best_text, best_count
            progress = True
        if count == target:
            return text
        if text == start_text and count == start_count:
            break

    # Last bounded search for a tiny suffix combination. This is deliberately small and deterministic.
    suffixes = (" a", " b", " c", " 1", " 2", " -", "_", ":", ";", "!", "?", "\n")
    for a in ("",) + suffixes:
        c1 = counter(text + a)
        if c1 == target:
            return text + a
        if c1 > target:
            continue
        for b in ("",) + suffixes:
            c2 = counter(text + a + b)
            if c2 == target:
                return text + a + b
            if c2 > target:
                continue
            for c in suffixes:
                if counter(text + a + b + c) == target:
                    return text + a + b + c

    raise RuntimeError(
        f"could not synthesize exactly {target} tokens; closest deterministic candidate had {count}. "
        "Do not relabel an approximate prompt as an exact calibration workload."
    )


def make_contract(air_cli: str, model: str, out: Path, targets: list[int], base: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    counter = AirTokenizerCounter(air_cli, model)
    rows = []
    for target in targets:
        text = synthesize_exact(counter, target, base)
        actual = counter(text)
        if actual != target:
            raise RuntimeError(f"internal prompt contract failure: target={target}, actual={actual}")
        path = out / f"p{target}.txt"
        path.write_text(text)
        rows.append({"name": path.name, "target_tokens": target, "actual_tokens": actual, "bytes": len(text.encode())})
    contract = {
        "schema": "air.prompt15.prompt-contract.v1",
        "model": model,
        "targets": rows,
    }
    (out / "PROMPT-CONTRACT.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def verify_contract(air_cli: str, model: str, directory: Path, targets: list[int]) -> dict:
    counter = AirTokenizerCounter(air_cli, model)
    rows = []
    ok = True
    for target in targets:
        path = directory / f"p{target}.txt"
        if not path.exists():
            rows.append({"name": path.name, "target_tokens": target, "status": "missing"})
            ok = False
            continue
        actual = counter(path.read_text())
        status = "pass" if actual == target else "fail"
        ok &= actual == target
        rows.append({"name": path.name, "target_tokens": target, "actual_tokens": actual, "status": status})
    return {"schema":"air.prompt15.prompt-contract-check.v1","ok":ok,"targets":rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="Create or verify exact-token AIR Prompt-15 workload fixtures using AIR's own tokenizer.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("--air-cli", required=True)
        sp.add_argument("--model", required=True)
        sp.add_argument("--output-dir", type=Path, required=True)
        sp.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)))
        if name == "make":
            sp.add_argument("--base", default=DEFAULT_BASE)
    args = ap.parse_args()
    targets = [int(x) for x in args.targets.split(",") if x.strip()]
    if args.cmd == "make":
        result = make_contract(args.air_cli, args.model, args.output_dir, targets, args.base)
        print(json.dumps(result, indent=2))
        return 0
    result = verify_contract(args.air_cli, args.model, args.output_dir, targets)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
