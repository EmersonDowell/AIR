#!/usr/bin/env python3
"""Generate Prompt-15 manifest corruption cases from one valid schema-v10 manifest."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("output_dir")
    args = ap.parse_args()
    base = json.loads(Path(args.base).read_text())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cases: dict[str, str] = {}

    (out / "malformed-json.json").write_text('{"schema_version":10,"strategies":[')
    cases["malformed-json"] = "reject"

    def case(name: str, mutator) -> None:
        obj = copy.deepcopy(base); mutator(obj); write(out / f"{name}.json", obj); cases[name] = "reject"

    case("unsupported-schema", lambda d: d.__setitem__("schema_version", 999))
    case("wrong-air-version", lambda d: d.__setitem__("air_version", "0.0.0-destruction"))
    case("wrong-model-digest", lambda d: d.__setitem__("model_digest", "fnv1a64:0000000000000000"))
    case("wrong-hardware-digest", lambda d: d.__setitem__("hardware_digest", "fnv1a64:ffffffffffffffff"))
    case("empty-manifest-id", lambda d: d.__setitem__("manifest_id", ""))
    case("unknown-linear", lambda d: d["strategies"][0].__setitem__("prefill_block_quantized_linear", "magic-linear"))
    case("unknown-attention", lambda d: d["strategies"][0].__setitem__("prefill_attention", "magic-attention"))
    case("inverted-prompt-region", lambda d: (d["strategies"][0].__setitem__("region_min_prompt_tokens", 100), d["strategies"][0].__setitem__("region_max_prompt_tokens", 10)))
    case("inverted-concurrency-region", lambda d: (d["strategies"][0].__setitem__("region_min_active_sequences", 8), d["strategies"][0].__setitem__("region_max_active_sequences", 2)))
    case("zero-prefill-quantum", lambda d: d["strategies"][0].__setitem__("prefill_quantum_tokens", 0))

    def duplicate_strategy(d):
        dup = copy.deepcopy(d["strategies"][0]); dup["evidence_id"] += ":duplicate-row"; d["strategies"].append(dup)
    case("duplicate-strategy-id", duplicate_strategy)

    def duplicate_evidence(d):
        dup = copy.deepcopy(d["strategies"][0]); dup["strategy_id"] += "-duplicate"; d["strategies"].append(dup)
    case("duplicate-evidence-id", duplicate_evidence)

    case("negative-performance", lambda d: d["strategies"][0].__setitem__("mean_prefill_tokens_per_second", -1.0))
    case("negative-preparation", lambda d: d["strategies"][0].__setitem__("preparation_ms_mean", -1.0))

    def missing_performance(d):
        s = d["strategies"][0]
        for k in ("p50_ttft_ms", "p50_total_ms", "mean_prefill_tokens_per_second", "mean_decode_tokens_per_second", "aggregate_generated_tokens_per_second"):
            s[k] = 0.0
    case("strict-missing-performance", missing_performance)

    def missing_preparation(d):
        s = next((x for x in d["strategies"] if int(x.get("prepared_artifact_bytes", 0)) > 0), d["strategies"][0])
        s["prepared_artifact_bytes"] = max(int(s.get("prepared_artifact_bytes", 0)), 1)
        s["preparation_measured"] = False
    case("optional-state-missing-preparation", missing_preparation)

    write(out / "EXPECTED.json", {"schema": "air.prompt15.corruption-cases.v1", "cases": cases})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
