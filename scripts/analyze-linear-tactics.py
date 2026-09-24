#!/usr/bin/env python3
"""Summarize randomized AIR Prompt 10 linear-tactic benchmark evidence.

This is evidence tooling only. It consumes air.benchmark.v3 files emitted by the
production InferenceService path and does not implement inference behavior.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Two-sided 95% Student-t critical values by df. Prompt 10 defaults to five
# paired rounds; values through df=30 keep the tool useful without scipy.
T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def ci95(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "low": None, "high": None, "stdev": None}
    mean = statistics.fmean(values)
    if n == 1:
        return {"n": 1, "mean": mean, "low": None, "high": None, "stdev": None}
    sd = statistics.stdev(values)
    t = T975.get(n - 1, 1.96)
    half = t * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "low": mean - half, "high": mean + half, "stdev": sd}


def parse_name(path: Path) -> tuple[int, str, str] | None:
    # round-01-p262-baseline.json
    parts = path.stem.split("-")
    if len(parts) < 4 or parts[0] != "round":
        return None
    try:
        round_id = int(parts[1])
    except ValueError:
        return None
    prompt = parts[2]
    tactic = "-".join(parts[3:])
    return round_id, prompt, tactic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence_dir", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    records: dict[tuple[str, str, int], dict] = {}
    for path in sorted(args.evidence_dir.glob("round-*-*.json")):
        key = parse_name(path)
        if key is None:
            continue
        round_id, prompt, tactic = key
        data = json.loads(path.read_text())
        if data.get("schema") != "air.benchmark.v3":
            raise SystemExit(f"unexpected benchmark schema in {path}: {data.get('schema')}")
        emitted = data.get("prefill_block_linear_tactic")
        expected = {"baseline": "baseline", "reuse4": "batch-reuse4", "reuse8": "batch-reuse8"}.get(tactic, tactic)
        if emitted != expected:
            raise SystemExit(f"tactic identity mismatch in {path}: expected {expected}, got {emitted}")
        records[(prompt, tactic, round_id)] = data

    prompts = sorted({k[0] for k in records})
    tactics = sorted({k[1] for k in records})
    summary: dict = {"schema": "air.prompt10.linear-tactic-summary.v1", "prompts": {}}

    for prompt in prompts:
        out_prompt: dict = {"tactics": {}, "paired_vs_baseline": {}}
        for tactic in tactics:
            rows = [(r, d) for (p, t, r), d in records.items() if p == prompt and t == tactic]
            rows.sort()
            if not rows:
                continue
            prefill = [float(d["summary"]["mean_prefill_tokens_per_second"]) for _, d in rows]
            aggregate = [float(d["summary"]["aggregate_generated_tokens_per_second"]) for _, d in rows]
            ttft = [float(d["summary"]["p50_ttft_ms"]) for _, d in rows]
            total = [float(d["summary"]["p50_total_ms"]) for _, d in rows]
            out_prompt["tactics"][tactic] = {
                "prefill_tokens_per_second": ci95(prefill),
                "aggregate_output_tokens_per_second": ci95(aggregate),
                "p50_ttft_ms": ci95(ttft),
                "p50_total_ms": ci95(total),
            }

        for tactic in tactics:
            if tactic == "baseline":
                continue
            ratios_prefill: list[float] = []
            ratios_total: list[float] = []
            ratios_aggregate: list[float] = []
            rounds = sorted({r for (p, t, r) in records if p == prompt and t == tactic})
            for r in rounds:
                base = records.get((prompt, "baseline", r))
                cand = records.get((prompt, tactic, r))
                if not base or not cand:
                    continue
                bs = base["summary"]; cs = cand["summary"]
                bp = float(bs["mean_prefill_tokens_per_second"])
                cp = float(cs["mean_prefill_tokens_per_second"])
                bt = float(bs["p50_total_ms"]); ct = float(cs["p50_total_ms"])
                ba = float(bs["aggregate_generated_tokens_per_second"])
                ca = float(cs["aggregate_generated_tokens_per_second"])
                if bp > 0: ratios_prefill.append(cp / bp)
                if ct > 0: ratios_total.append(bt / ct)  # >1 means candidate faster
                if ba > 0: ratios_aggregate.append(ca / ba)
            out_prompt["paired_vs_baseline"][tactic] = {
                "prefill_speed_ratio": ci95(ratios_prefill),
                "total_speed_ratio": ci95(ratios_total),
                "aggregate_output_ratio": ci95(ratios_aggregate),
            }
        summary["prompts"][prompt] = out_prompt

    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(rendered)
    else:
        print(rendered)

    if args.markdown:
        lines = ["# AIR Prompt 10 Linear Tactic Summary", "", "Ratios above 1.0 favor the candidate. Confidence intervals are paired two-sided 95% Student-t intervals.", ""]
        for prompt, pdata in summary["prompts"].items():
            lines += [f"## {prompt}", "", "| Tactic | Prefill tok/s mean | 95% CI | Paired prefill ratio vs baseline | 95% CI |", "|---|---:|---:|---:|---:|"]
            for tactic, tdata in pdata["tactics"].items():
                p = tdata["prefill_tokens_per_second"]
                ratio = pdata["paired_vs_baseline"].get(tactic, {}).get("prefill_speed_ratio")
                if ratio:
                    rmean = ratio["mean"]; rlow = ratio["low"]; rhigh = ratio["high"]
                    rtext = f"{rmean:.4f}" if rmean is not None else "n/a"
                    rcitext = f"[{rlow:.4f}, {rhigh:.4f}]" if rlow is not None else "n/a"
                else:
                    rtext = rcitext = "baseline"
                ci = f"[{p['low']:.3f}, {p['high']:.3f}]" if p["low"] is not None else "n/a"
                lines.append(f"| {tactic} | {p['mean']:.3f} | {ci} | {rtext} | {rcitext} |")
            lines.append("")
        args.markdown.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
