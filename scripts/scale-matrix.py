#!/usr/bin/env python3
"""AIR Comparison Prompt 8 multi-model scaling orchestrator.

This is evidence tooling only. It runs the existing compare-llama.py scaling
mode for each compatible model, then aggregates the model x prompt x
concurrency evidence without changing AIR runtime policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

HARNESS_REVISION = "AIR-Comparison8-R3"

def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and cp.returncode != 0:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stdout}")
    return cp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model(air_cli: str, path: Path) -> dict[str, Any] | None:
    """Read discovery metadata only. Execution support is owned by AIR itself.

    Do not duplicate CUDA/reference tensor-format support here. That list belongs
    to the frozen executors and previously caused the scaling harness to reject a
    model that AIR had already executed successfully.
    """
    cp = run([air_cli, "inspect", str(path)], check=False)
    if cp.returncode != 0:
        return None
    text = cp.stdout
    fields: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^(Name|Architecture|Mapped size|Layers|Embedding|Context):\s+(.*)$", line)
        if m:
            fields[m.group(1).lower().replace(" ", "_")] = m.group(2).strip()
    if fields.get("architecture") != "qwen2":
        return None
    types: set[str] = set()
    in_types = False
    for line in text.splitlines():
        if line.strip() == "Tensor types:":
            in_types = True
            continue
        if in_types:
            if not line.startswith("  "):
                if line.strip():
                    break
                continue
            token = line.strip().split()[0]
            types.add(token)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "name": fields.get("name", path.stem),
        "architecture": fields.get("architecture"),
        "layers": fields.get("layers"),
        "embedding": fields.get("embedding"),
        "context": fields.get("context"),
        "tensor_types": sorted(types),
    }


def verify_execution_support(air_verify: str, item: dict[str, Any], probe_dir: Path, atol: float) -> tuple[bool, str]:
    """Ask AIR's canonical verifier whether this model passes the admission correctness gate."""
    probe_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(item["path"]).stem)[:80]
    report = probe_dir / f"{safe_name}.json"
    cp = run([
        air_verify, "-m", item["path"],
        "--tokens", "1", "--generate", "1", "--top-k", "2",
        "--atol", str(atol), "--output", str(report),
    ], check=False)
    log = probe_dir / f"{safe_name}.txt"
    log.write_text(cp.stdout)
    return cp.returncode == 0, str(log)


def candidate_paths() -> list[Path]:
    home = Path.home()
    patterns = [
        home / "Models" / "AIR",
        home / "Models",
    ]
    out: list[Path] = []
    for root in patterns:
        if not root.exists():
            continue
        iterator = root.glob("*.gguf") if root.name == "AIR" else root.glob("**/*.gguf")
        for path in iterator:
            if path.is_file():
                out.append(path.resolve())
    hub = home / ".cache" / "huggingface" / "hub"
    if hub.exists():
        for model_dir in hub.glob("models--*Qwen2*GGUF"):
            for path in model_dir.glob("snapshots/*/*.gguf"):
                if path.is_file():
                    out.append(path.resolve())
    seen: set[Path] = set()
    uniq: list[Path] = []
    for path in out:
        if path not in seen:
            seen.add(path)
            uniq.append(path)
    return uniq


def select_spread(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return sorted(items, key=lambda x: x["bytes"])
    ordered = sorted(items, key=lambda x: x["bytes"])
    if limit == 1:
        return [ordered[0]]
    indexes = [round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)]
    chosen: list[dict[str, Any]] = []
    for idx in indexes:
        if ordered[idx] not in chosen:
            chosen.append(ordered[idx])
    return chosen


def classification(metric: str, evidence: dict[str, Any]) -> str:
    lo, hi = evidence.get("ci95", [None, None])
    if lo is None or hi is None or lo <= 1.0 <= hi:
        return "inconclusive"
    higher = metric in {"aggregate_output_tps", "requests_per_second", "native_prefill_tps", "native_decode_tps"}
    ratio = float(evidence["air_over_llama"])
    if higher:
        return "air-better" if ratio > 1.0 else "llama-better"
    return "air-better" if ratio < 1.0 else "llama-better"


def fmt_ratio(ev: dict[str, Any]) -> str:
    ratio = ev.get("air_over_llama")
    lo, hi = ev.get("ci95", [None, None])
    if ratio is None:
        return "n/a"
    if lo is None or hi is None:
        return f"{ratio:.3f}x"
    return f"{ratio:.3f}x [{lo:.3f},{hi:.3f}]"


def aggregate(output: Path, inventory: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item, run_info in zip(inventory, runs):
        report = json.loads(Path(run_info["scaling_json"]).read_text())
        for key, spec in report["meta"]["cells"].items():
            ratios = report["ratios"][key]
            row = {
                "model": item["name"],
                "model_path": item["path"],
                "model_bytes": item["bytes"],
                "model_sha256": report["meta"]["model_sha256"],
                "target_prompt_tokens": spec["target_prompt_tokens"],
                "actual_prompt_tokens": spec["prompt_tokens"],
                "concurrency": spec["concurrency"],
                "generated_tokens_per_request": spec["generated_tokens_per_request"],
                "ratios": ratios,
            }
            row["classification"] = {m: classification(m, ratios[m]) for m in ratios}
            rows.append(row)

    generalization: list[dict[str, Any]] = []
    dimensions = sorted({(r["target_prompt_tokens"], r["concurrency"]) for r in rows})
    for target, concurrency in dimensions:
        subset = [r for r in rows if r["target_prompt_tokens"] == target and r["concurrency"] == concurrency]
        for metric in ("ttft_latency", "total_latency", "aggregate_output_tps", "requests_per_second"):
            classes = [r["classification"][metric] for r in subset]
            if classes and all(c == classes[0] for c in classes) and classes[0] != "inconclusive":
                result = classes[0]
            else:
                result = "not-generalized"
            generalization.append({
                "target_prompt_tokens": target,
                "concurrency": concurrency,
                "metric": metric,
                "models": len(subset),
                "result": result,
                "classifications": classes,
            })

    obj = {
        "schema": "air.scaling_matrix.v1",
        "harness_revision": HARNESS_REVISION,
        "model_count": len(inventory),
        "model_scaling_complete": len(inventory) >= 2,
        "inventory": inventory,
        "runs": runs,
        "rows": rows,
        "generalization": generalization,
    }
    (output / "SCALING_MATRIX.json").write_text(json.dumps(obj, indent=2) + "\n")

    lines = [
        "# AIR Comparison Prompt 8: Scaling Matrix",
        "",
        f"Compatible models tested: {len(inventory)}",
        f"Additional-model scaling complete: {'yes' if len(inventory) >= 2 else 'no'}",
        "",
        "All cells retain the paired randomized methodology. Ratios are AIR/llama with paired 95% Student-t confidence intervals.",
        "Latency ratios below 1 favor AIR. Throughput ratios above 1 favor AIR.",
        "",
        "| Model | MiB | Prompt target/actual | C | TTFT ratio [95% CI] | Total ratio [95% CI] | Aggregate tok/s ratio [95% CI] |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        ratios = r["ratios"]
        lines.append(
            f"| {r['model']} | {r['model_bytes']/1048576:.1f} | "
            f"{r['target_prompt_tokens']}/{r['actual_prompt_tokens']} | {r['concurrency']} | "
            f"{fmt_ratio(ratios['ttft_latency'])} | {fmt_ratio(ratios['total_latency'])} | "
            f"{fmt_ratio(ratios['aggregate_output_tps'])} |"
        )
    lines += ["", "## Cross-model direction", ""]
    for g in generalization:
        lines.append(
            f"- p{g['target_prompt_tokens']} c{g['concurrency']} {g['metric']}: "
            f"{g['result']} across {g['models']} tested model(s)."
        )
    if len(inventory) < 2:
        lines += [
            "",
            "The prompt/concurrency matrix is valid for the available model, but model-size generalization is not complete.",
            "Add at least one additional compatible Qwen2/Qwen2.5 GGUF and rerun the same harness.",
        ]
    (output / "SCALING_MATRIX.md").write_text("\n".join(lines) + "\n")
    return obj


def main() -> int:
    if "--revision" in sys.argv:
        print(HARNESS_REVISION)
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--base-model", default="")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--max-models", type=int, default=3)
    ap.add_argument("--air-cli", default="air-cli")
    ap.add_argument("--air-server", default="air-server")
    ap.add_argument("--air-verify", default="air-verify")
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--llama-bench", default="")
    ap.add_argument("--verify-script", required=True)
    ap.add_argument("--compare-script", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--prompt-targets", default="32,256,1024")
    ap.add_argument("--concurrency-levels", default="1,2,4")
    ap.add_argument("--generated", type=int, default=32)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--order-seed", type=int, default=7331)
    ap.add_argument("--sampling-seed", type=int, default=424242)
    ap.add_argument("--skip-microbench", action="store_true")
    ap.add_argument("--admission-atol", type=float, default=0.001,
                    help="reference/CUDA max-absolute-error gate used before a model enters the matrix")
    args = ap.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    explicit: list[Path] = []
    if args.base_model:
        explicit.append(Path(args.base_model).expanduser().resolve())
    explicit.extend(Path(x).expanduser().resolve() for x in args.model)

    candidates: list[Path] = []
    seen: set[Path] = set()
    for path in explicit + (candidate_paths() if args.discover else []):
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        candidates.append(path)

    inspected: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in candidates:
        item = inspect_model(args.air_cli, path)
        if item is None:
            rejected.append({"path": str(path), "reason": "not a readable qwen2 GGUF"})
        else:
            inspected.append(item)

    explicit_set = {p for p in explicit}
    explicit_items = [x for x in inspected if Path(x["path"]) in explicit_set]
    discovered_items = [x for x in inspected if Path(x["path"]) not in explicit_set]

    # Preserve explicit models first. For discovery, prefer a size-spread sample,
    # then retain the remaining candidates as deterministic backfill if an AIR
    # execution probe rejects one of the preferred choices.
    remaining = max(0, args.max_models - len(explicit_items))
    preferred = select_spread(discovered_items, remaining) if remaining else []
    preferred_paths = {x["path"] for x in preferred}
    backfill = [x for x in sorted(discovered_items, key=lambda x: x["bytes"]) if x["path"] not in preferred_paths]

    candidate_order: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in explicit_items + preferred + backfill:
        if item["path"] not in seen_paths:
            seen_paths.add(item["path"])
            candidate_order.append(item)

    inventory: list[dict[str, Any]] = []
    probe_dir = output / "admission-probes"
    for item in candidate_order:
        if len(inventory) >= args.max_models:
            break
        ok, log = verify_execution_support(args.air_verify, item, probe_dir, args.admission_atol)
        if ok:
            inventory.append(item)
        else:
            rejected.append({
                "path": item["path"],
                "reason": "AIR reference/CUDA execution probe failed",
                "probe_log": log,
            })

    (output / "MODEL_INVENTORY.json").write_text(json.dumps({
        "harness_revision": HARNESS_REVISION,
        "selected": inventory,
        "metadata_candidates": inspected,
        "rejected": rejected,
        "admission_authority": "air-verify reference/CUDA execution path",
        "admission_atol": args.admission_atol,
    }, indent=2) + "\n")
    if not inventory:
        raise RuntimeError("no compatible Qwen2/Qwen2.5 GGUF models found")

    runs: list[dict[str, Any]] = []
    for index, item in enumerate(inventory):
        model_out = output / f"model-{index+1:02d}"
        model_out.mkdir(exist_ok=True)
        cmd = [
            sys.executable, args.compare_script,
            "--scaling", "--model", item["path"],
            "--air-server", args.air_server, "--air-cli", args.air_cli, "--air-verify", args.air_verify,
            "--llama-server", args.llama_server, "--llama-bench", args.llama_bench,
            "--verify-script", args.verify_script, "--output-dir", str(model_out),
            "--rounds", str(args.rounds), "--order-seed", str(args.order_seed + index * 1009),
            "--sampling-seed", str(args.sampling_seed), "--cooldown", str(args.cooldown),
            "--context", str(args.context), "--prompt-targets", args.prompt_targets,
            "--concurrency-levels", args.concurrency_levels, "--scaling-generated", str(args.generated),
            "--air-port", str(18470 + index * 10), "--llama-port", str(18471 + index * 10),
        ]
        if args.skip_microbench:
            cmd.append("--skip-microbench")
        log = output / f"model-{index+1:02d}-harness.txt"
        with log.open("w") as f:
            proc = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.STDOUT)
        runs.append({
            "model": item["path"], "returncode": proc.returncode,
            "output_dir": str(model_out), "scaling_json": str(model_out / "scaling.json"),
            "log": str(log),
        })
        if proc.returncode != 0:
            raise RuntimeError(f"scaling run failed for {item['path']}; see {log}")

    aggregate(output, inventory, runs)
    print(output / "SCALING_MATRIX.md")
    print(output / "SCALING_MATRIX.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"scaling matrix failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
