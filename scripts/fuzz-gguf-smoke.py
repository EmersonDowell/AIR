#!/usr/bin/env python3
"""Small deterministic crash/timeout fuzzer for AIR's GGUF parser.

This is a release-validation smoke fuzzer, not a coverage-guided fuzzer. It
mutates a tiny known-good fixture and treats graceful accept/reject as valid;
signals, hangs, and impossible process exits are failures.
"""
import argparse
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def run_one(cli: str, path: Path, timeout: float) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [cli, "inspect", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if result.returncode < 0:
        return False, f"signal:{-result.returncode}"
    return True, f"exit:{result.returncode}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("--cli", default="air-cli")
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--timeout", type=float, default=2.0)
    args = ap.parse_args()
    original = Path(args.fixture).read_bytes()
    if len(original) < 32:
        print("fixture is too small", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    failures: list[str] = []
    accepted = rejected = 0
    with tempfile.TemporaryDirectory(prefix="air-gguf-fuzz-") as td:
        root = Path(td)
        for case in range(args.cases):
            data = bytearray(original)
            mode = case % 4
            if mode == 0:
                cut = rng.randrange(0, len(data))
                data = data[:cut]
            elif mode == 1:
                for _ in range(rng.randint(1, 8)):
                    i = rng.randrange(0, min(len(data), 512))
                    data[i] ^= 1 << rng.randrange(8)
            elif mode == 2:
                start = rng.randrange(0, min(len(data), 512))
                width = min(rng.randint(1, 32), len(data) - start)
                data[start:start + width] = os.urandom(width)
            else:
                # Damage count/length-heavy header bytes while keeping GGUF magic
                # often intact so deeper parser bounds checks are exercised.
                for _ in range(rng.randint(1, 4)):
                    i = rng.randrange(4, min(len(data), 128))
                    data[i] = rng.randrange(256)
            path = root / f"case-{case:04d}.gguf"
            path.write_bytes(data)
            ok, outcome = run_one(args.cli, path, args.timeout)
            if not ok:
                failures.append(f"case={case} mode={mode} {outcome}")
            elif outcome == "exit:0":
                accepted += 1
            else:
                rejected += 1
    print(f"cases={args.cases} accepted={accepted} rejected={rejected} failures={len(failures)} seed={args.seed}")
    for failure in failures[:20]:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
