#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

DENSE_BYTES = 1_431_306_240
EXPECTED_CORRUPTIONS = {
    "malformed-json",
    "unsupported-schema",
    "wrong-air-version",
    "wrong-model-digest",
    "wrong-hardware-digest",
    "empty-manifest-id",
    "unknown-linear",
    "unknown-attention",
    "inverted-prompt-region",
    "inverted-concurrency-region",
    "zero-prefill-quantum",
    "duplicate-strategy-id",
    "duplicate-evidence-id",
    "negative-performance",
    "negative-preparation",
    "strict-missing-performance",
    "optional-state-missing-preparation",
}
QUALIFIED_STRATEGIES = {
    "reuse8-p54",
    "reuse8-p262",
    "dense-p262",
    "reuse8-p1042",
    "dense-p1042",
    "reuse8-c8",
    "dense-c8",
}
EXPECTED_OUT_OF_REGION = {"p128", "p512", "p2048", "c2", "c4", "c12"}
EXPECTED_CONCURRENCY = {1, 2, 4, 8, 12}


@dataclass
class Finding:
    severity: str  # error, warning, info
    gate: str
    code: str
    message: str


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def strategy_from_bench(d: dict[str, Any]) -> str | None:
    runs = d.get("runs") or []
    if runs and isinstance(runs[0], dict) and runs[0].get("strategy_id") is not None:
        return str(runs[0].get("strategy_id"))
    if d.get("strategy_id") is not None:
        return str(d.get("strategy_id"))
    return None


def prompt_tokens_from_bench(d: dict[str, Any]) -> int | None:
    runs = d.get("runs") or []
    vals = []
    for row in runs:
        if isinstance(row, dict) and isinstance(row.get("prompt_tokens"), int):
            vals.append(row["prompt_tokens"])
    if vals:
        return vals[0] if all(v == vals[0] for v in vals) else None
    summary = d.get("summary") or {}
    total = summary.get("prompt_tokens")
    cfg = d.get("config") or {}
    measured = cfg.get("measured_runs")
    if isinstance(total, int) and isinstance(measured, int) and measured > 0 and total % measured == 0:
        return total // measured
    return None


def prepared_bytes_from_bench(d: dict[str, Any]) -> int | None:
    summary = d.get("summary") or {}
    value = summary.get("prepared_artifact_bytes")
    return int(value) if isinstance(value, (int, float)) else None


def candidates_from_bench(d: dict[str, Any]) -> list[dict[str, Any]]:
    runs = d.get("runs") or []
    if runs and isinstance(runs[0], dict):
        x = runs[0].get("strategy_candidates")
        if isinstance(x, list):
            return [c for c in x if isinstance(c, dict)]
    return []


def reason_from_bench(d: dict[str, Any]) -> str:
    runs = d.get("runs") or []
    if runs and isinstance(runs[0], dict):
        x = runs[0].get("strategy_decision_reason")
        if x is not None:
            return str(x)
    return ""


def finite_bool(value: Any) -> bool:
    return isinstance(value, bool) and value


def get_error(decision: dict[str, Any]) -> float:
    logits = decision.get("logits") or {}
    try:
        return float(logits.get("max_abs_error", math.inf))
    except Exception:
        return math.inf


def strict_correctness_status(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    decisions = report.get("decisions") or []
    if not isinstance(decisions, list) or not decisions:
        return "INCOMPLETE", {"decisions": 0}
    worst = max(get_error(x) for x in decisions if isinstance(x, dict))
    mismatches = [x.get("index") for x in decisions if isinstance(x, dict) and not bool(x.get("top1_match", False))]
    finite_flags = []
    for x in decisions:
        if not isinstance(x, dict):
            finite_flags.append(False)
            continue
        if "finite" in x:
            finite_flags.append(bool(x.get("finite")))
        else:
            logits = x.get("logits") or {}
            finite_flags.append(bool(logits.get("finite", True)))
    passed = all(finite_flags) and not mismatches and math.isfinite(worst) and worst <= 0.001
    return ("PASS" if passed else "FAIL"), {
        "decisions": len(decisions),
        "worst_max_abs_error": worst,
        "top1_mismatches": mismatches,
        "all_finite": all(finite_flags),
    }


def linear_slope(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    n = len(values)
    xs = list(range(n))
    xbar = statistics.mean(xs)
    ybar = statistics.mean(values)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, values)) / denom


def check_archive(root: Path) -> dict[str, Any]:
    findings: list[Finding] = []
    scientific: dict[str, Any] = {}

    def err(gate: str, code: str, msg: str) -> None:
        findings.append(Finding("error", gate, code, msg))

    def warn(gate: str, code: str, msg: str) -> None:
        findings.append(Finding("warning", gate, code, msg))

    def info(gate: str, code: str, msg: str) -> None:
        findings.append(Finding("info", gate, code, msg))

    # Gate 0: archive/provenance completeness, not scientific success.
    if (root / "FATAL.txt").exists():
        err("0", "collector-fatal", (root / "FATAL.txt").read_text(errors="replace").strip())
    required = [
        root / "build" / "ctest.txt",
        root / "fingerprint-05.json",
        root / "model-sha256-05.txt",
        root / "prompt14-control-manifest.json",
        root / "control-manifest-identity.txt",
        root / "source-sha256.txt",
    ]
    for path in required:
        if not path.exists():
            err("0", "missing-evidence", f"missing required file: {path.relative_to(root)}")
    ctest = root / "build" / "ctest.txt"
    if ctest.exists():
        text = ctest.read_text(errors="replace")
        if "100% tests passed" not in text:
            err("0", "ctest-not-clean", "complete ctest does not report 100% tests passed")
        m = re.search(r"(\d+)/10\s+Test", text)
        if "10/10" not in text and not re.search(r"100% tests passed, 0 tests failed out of 10", text):
            warn("0", "ctest-count-unverified", "could not positively confirm the expected 10-test research suite")

    fp = load_json(root / "fingerprint-05.json") if (root / "fingerprint-05.json").exists() else None
    manifest = load_json(root / "prompt14-control-manifest.json") if (root / "prompt14-control-manifest.json").exists() else None
    if isinstance(fp, dict) and isinstance(manifest, dict):
        for key in ("air_version", "model_digest", "hardware_digest"):
            if fp.get(key) != manifest.get(key):
                err("0", "identity-mismatch", f"fingerprint/control manifest mismatch for {key}: {fp.get(key)!r} != {manifest.get(key)!r}")
        if manifest.get("air_version") != "0.9.12":
            err("0", "wrong-version", f"control manifest air_version={manifest.get('air_version')!r}, expected '0.9.12'")

    # Gate 1: require a valid-manifest positive control AND every expected corruption.
    g1 = root / "manifest-corruption"
    valid_exit = read_int(g1 / "valid-control.exit")
    if valid_exit is None:
        err("1", "missing-positive-control", "valid-manifest positive control exit code is missing")
    elif valid_exit != 0:
        err("1", "positive-control-failed", f"valid manifest could not execute successfully (rc={valid_exit}); corruption rejections are not interpretable")

    exits: dict[str, int] = {}
    for p in g1.glob("*.exit") if g1.exists() else []:
        if p.name == "valid-control.exit":
            continue
        val = read_int(p)
        if val is not None:
            exits[p.stem] = val
    missing = sorted(EXPECTED_CORRUPTIONS - set(exits))
    unexpected = sorted(set(exits) - EXPECTED_CORRUPTIONS)
    if missing:
        err("1", "missing-corruption-cases", f"missing corruption exit evidence: {', '.join(missing)}")
    if unexpected:
        warn("1", "unexpected-corruption-cases", f"unexpected corruption cases present: {', '.join(unexpected)}")
    accepted = sorted(k for k, v in exits.items() if k in EXPECTED_CORRUPTIONS and v == 0)
    if accepted:
        err("1", "corruption-accepted", f"corrupted manifests executed successfully: {', '.join(accepted)}")
    scientific["manifest_integrity"] = "PASS" if valid_exit == 0 and not missing and not accepted else "FAIL"

    # Gate 2: exact evidence must not leak outside its exact regions.
    oor = root / "out-of-region"
    seen_oor: set[str] = set()
    for name in sorted(EXPECTED_OUT_OF_REGION):
        jp = oor / f"{name}.json"
        ep = oor / f"{name}.exit"
        rc = read_int(ep)
        if rc is None:
            err("2", "missing-probe-exit", f"missing {name}.exit")
            continue
        if rc != 0:
            err("2", "probe-failed", f"out-of-region probe {name} did not execute successfully (rc={rc})")
            continue
        d = load_json(jp)
        if not isinstance(d, dict):
            err("2", "missing-probe-json", f"missing or invalid {name}.json")
            continue
        seen_oor.add(name)
        sid = strategy_from_bench(d)
        reason = reason_from_bench(d).lower()
        if sid in QUALIFIED_STRATEGIES:
            err("2", "silent-extrapolation", f"out-of-region probe {name} selected exact-qualified strategy {sid}")
        if sid in (None, "", "static") and not ("fallback" in reason or "no-eligible" in reason):
            warn("2", "fallback-reason-unproven", f"{name} appears to fall back safely but decision reason does not explicitly prove no eligible measured strategy")
    scientific["evidence_region_generalization"] = "PASS" if seen_oor == EXPECTED_OUT_OF_REGION and not any(f.gate == "2" and f.severity == "error" for f in findings) else "FAIL"

    # Gate 3: memory boundary invariants. Scientific selection at/above can be limited by real free memory,
    # but partial materialization below the declared budget is never acceptable.
    mem = root / "memory-budget"
    expected_budgets = [268435456, 536870912, 1073741824, 1395864371, 1431306239, 1431306240, 1503238554]
    for budget in expected_budgets:
        p = mem / f"budget-{budget}.json"
        rc = read_int(mem / f"budget-{budget}.exit")
        if rc is None or not p.exists():
            err("3", "missing-budget-probe", f"missing budget probe {budget}")
            continue
        if rc != 0:
            err("3", "budget-probe-failed", f"budget probe {budget} exited rc={rc}")
            continue
        d = load_json(p)
        if not isinstance(d, dict):
            err("3", "invalid-budget-json", f"invalid budget probe JSON {budget}")
            continue
        sid = strategy_from_bench(d) or ""
        prepared = prepared_bytes_from_bench(d)
        if prepared not in (None, 0, DENSE_BYTES):
            err("3", "partial-prepared-state", f"budget {budget} reports impossible partial dense residency {prepared}")
        if budget < DENSE_BYTES:
            if "dense" in sid:
                err("3", "dense-below-budget", f"budget {budget} selected {sid} below full dense requirement")
            if prepared not in (None, 0):
                err("3", "prepared-below-budget", f"budget {budget} materialized {prepared} bytes below full dense requirement")
            candidates = candidates_from_bench(d)
            dense_rows = [c for c in candidates if "dense" in str(c.get("strategy_id", ""))]
            if dense_rows and not any("memory" in str(c.get("disposition", "")).lower() or "memory" in str(c.get("reason", "")).lower() for c in dense_rows):
                warn("3", "dense-rejection-reason-unproven", f"budget {budget} did not expose an explicit memory rejection for dense")

    for label in ("unrestricted",):
        p = mem / f"budget-{label}.json"
        rc = read_int(mem / f"budget-{label}.exit")
        if rc is None or not p.exists():
            err("3", "missing-budget-probe", f"missing budget probe {label}")
    minp = mem / "minimum-vram.json"
    minrc = read_int(mem / "minimum-vram.exit")
    if minrc is None or not minp.exists():
        err("3", "missing-minimum-vram", "minimum-vram evidence is missing")
    elif minrc != 0:
        err("3", "minimum-vram-failed", f"minimum-vram probe exited rc={minrc}")
    else:
        d = load_json(minp)
        if isinstance(d, dict):
            sid = strategy_from_bench(d) or ""
            prepared = prepared_bytes_from_bench(d)
            if sid != "reuse8-p1042":
                err("3", "minimum-vram-wrong-strategy", f"minimum-vram selected {sid!r}, expected reuse8-p1042")
            if prepared not in (None, 0):
                err("3", "minimum-vram-prepared-state", f"minimum-vram retained {prepared} optional prepared bytes")
    scientific["memory_boundary"] = "PASS" if not any(f.gate == "3" and f.severity == "error" for f in findings) else "FAIL"

    # Gates 4/6: stress structure and diagnostics.
    stress_summary: dict[str, Any] = {}
    for name, expected_steps in (("oscillation", 40), ("hot-stress", 20)):
        rc = read_int(root / "stress" / f"{name}.exit")
        d = load_json(root / "stress" / f"{name}.json")
        if rc is None or not isinstance(d, dict):
            err("4/6", "missing-stress-evidence", f"missing {name} stress evidence")
            continue
        if rc != 0:
            err("4/6", "stress-failed", f"{name} exited rc={rc}")
            continue
        steps = d.get("steps") or []
        if len(steps) != expected_steps:
            err("4/6", "wrong-stress-length", f"{name} has {len(steps)} steps, expected {expected_steps}")
        prepared = []
        dev = []
        rss = []
        prep_ms = []
        evict_ms = []
        labels = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            labels.append(str(step.get("label", "")))
            after = step.get("after") or {}
            metrics = step.get("metrics") or {}
            prepared.append(int(after.get("current_prepared_artifact_bytes", -1)))
            dev.append(float(after.get("current_device_bytes", 0)))
            rss.append(float(step.get("rss_kib", 0)))
            prep_ms.append(float(metrics.get("plan_preparation_ms", 0)))
            evict_ms.append(float(metrics.get("plan_eviction_ms", 0)))
        if name == "oscillation":
            for i, (label, pb) in enumerate(zip(labels, prepared)):
                expected_label = "medium" if i % 2 == 0 else "small"
                if label != expected_label:
                    err("4/6", "oscillation-order", f"step {i} label={label!r}, expected {expected_label!r}")
                if label == "small" and pb != 0:
                    err("4/6", "eviction-not-complete", f"oscillation small step {i} retained {pb} prepared bytes")
                if pb not in (0, DENSE_BYTES):
                    err("4/6", "partial-prepared-state", f"oscillation step {i} has invalid prepared bytes {pb}")
        else:
            for i, pb in enumerate(prepared):
                if pb not in (DENSE_BYTES,):
                    err("4/6", "hot-state-not-resident", f"hot-stress step {i} prepared bytes={pb}, expected {DENSE_BYTES}")
            repeated_prep = [x for x in prep_ms[1:] if x > 0.01]
            if repeated_prep:
                err("4/6", "hot-state-reprepared", f"hot-stress repeated preparation observed after first step: {repeated_prep[:5]}")
        stress_summary[name] = {
            "steps": len(steps),
            "prepared_final": prepared[-1] if prepared else None,
            "device_slope_bytes_per_step": linear_slope(dev),
            "rss_slope_kib_per_step": linear_slope(rss),
            "prep_positive_count": sum(x > 0.01 for x in prep_ms),
            "eviction_positive_count": sum(x > 0.01 for x in evict_ms),
            "device_first_last": [dev[0], dev[-1]] if dev else None,
            "rss_first_last": [rss[0], rss[-1]] if rss else None,
        }
    scientific["stress_diagnostics"] = stress_summary
    scientific["adaptive_runtime_stability"] = "PASS" if not any(f.gate == "4/6" and f.severity == "error" for f in findings) else "FAIL"

    # Gate 5: exact c8 contract and out-of-region concurrency behavior.
    conc = root / "concurrency"
    conc_status: dict[str, Any] = {}
    for c in sorted(EXPECTED_CONCURRENCY):
        p = conc / f"c{c}.json"
        rc = read_int(conc / f"c{c}.exit")
        if rc is None or not p.exists():
            err("5", "missing-concurrency-probe", f"missing c{c} evidence")
            continue
        if rc != 0:
            err("5", "concurrency-probe-failed", f"c{c} exited rc={rc}")
            continue
        d = load_json(p)
        if not isinstance(d, dict):
            err("5", "invalid-concurrency-json", f"invalid c{c}.json")
            continue
        pt = prompt_tokens_from_bench(d)
        sid = strategy_from_bench(d)
        conc_status[str(c)] = {"prompt_tokens": pt, "strategy_id": sid}
        if pt != 160:
            err("5", "calibration-prompt-mismatch", f"c{c} prompt has {pt} tokens; Gate 5 contract requires the 160-token calibration workload")
        if c == 8:
            if sid not in {"reuse8-c8", "dense-c8"}:
                err("5", "c8-not-calibrated-strategy", f"exact c8 workload selected {sid!r}, not a c8 qualified strategy")
        else:
            if sid in QUALIFIED_STRATEGIES:
                err("5", "concurrency-silent-extrapolation", f"out-of-region c{c} selected exact-qualified strategy {sid}")
    scientific["concurrency"] = conc_status
    scientific["concurrency_gate"] = "PASS" if not any(f.gate == "5" and f.severity == "error" for f in findings) else "FAIL"

    # Gate 7: report real scientific result separately from evidence validity.
    gen = root / "generalization"
    gen_status: dict[str, Any] = {}
    for tag in ("model-05", "model-15", "model-7"):
        p = gen / f"{tag}.json"
        missing_marker = gen / f"{tag}.missing"
        exitp = gen / f"{tag}.exit"
        if missing_marker.exists():
            gen_status[tag] = {"status": "MISSING_MODEL"}
            warn("7", "model-missing", f"{tag} was not available for qualification")
            continue
        if not p.exists() or read_int(exitp) is None:
            err("7", "missing-generalization-evidence", f"{tag} verification evidence missing")
            continue
        d = load_json(p)
        if not isinstance(d, dict):
            err("7", "invalid-generalization-json", f"{tag}.json is invalid")
            continue
        status, details = strict_correctness_status(d)
        details["status"] = status
        details["exit_code"] = read_int(exitp)
        gen_status[tag] = details
        # A genuine FAIL is a scientific result, not invalid evidence.
        if status == "PASS" and read_int(exitp) not in (0,):
            err("7", "verification-exit-contradiction", f"{tag} JSON passes strict gate but process exit code is {read_int(exitp)}")
        if status == "FAIL" and read_int(exitp) == 0:
            err("7", "verification-exit-contradiction", f"{tag} JSON fails strict gate but process exit code is 0")
    scientific["correctness_generalization"] = gen_status

    # Gate 8/9 evidence completeness.
    if not (root / "control" / "provenance.txt").exists():
        err("8", "missing-control-provenance", "control/provenance.txt is missing")
    startup = root / "startup" / "startup-transition.json"
    startup_rc = read_int(root / "startup" / "probe.exit")
    if not startup.exists() or startup_rc is None:
        err("9", "missing-startup-evidence", "startup transition evidence is missing")
    elif startup_rc != 0:
        err("9", "startup-probe-failed", f"startup transition probe exited rc={startup_rc}")
    elif not isinstance(load_json(startup), dict):
        err("9", "invalid-startup-json", "startup-transition.json is invalid")

    # Gate 10: check admission consistency only. Competitive interpretation remains separate.
    ext = root / "external"
    skipped = ext / "SKIPPED.txt"
    reuse = ext / "reuse8" / "scaling.json"
    dense = ext / "dense" / "scaling.json"
    g1pass = scientific.get("manifest_integrity") == "PASS"
    g05 = gen_status.get("model-05", {}).get("status") == "PASS"
    if g1pass and g05:
        if skipped.exists():
            warn("10", "external-skipped-despite-admission", skipped.read_text(errors="replace").strip())
        elif not reuse.exists() or not dense.exists():
            err("10", "missing-external-evidence", "external comparison was admitted but reuse8/dense scaling evidence is incomplete")
    else:
        if reuse.exists() or dense.exists():
            err("10", "external-ran-without-admission", "external comparison evidence exists even though integrity or 0.5B strict correctness did not admit it")


    # R3 Gate 5: deterministic output isolation is part of evidence validity.
    isolation_path = root / "concurrency" / "output-isolation.json"
    if not isolation_path.exists():
        err("5", "missing-output-isolation", "Gate 5 did not record deterministic measured-output token isolation evidence")
    else:
        isolation = load_json(isolation_path)
        if not isinstance(isolation, dict) or isolation.get("status") != "PASS":
            err("5", "output-isolation-failed", "Gate 5 same-prompt measured output token IDs are missing, ambiguous, or inconsistent")
        else:
            scientific["output_isolation"] = "PASS"

    # R3 Gate 7: prove every operation-scoped product tactic is pinned by air-verify.
    profile_path = root / "generalization" / "EXECUTION-PROFILE.json"
    profile_guard_path = root / "generalization" / "PROFILE-GUARD.json"
    if not profile_path.exists():
        err("7", "missing-execution-profile", "Gate 7 did not persist its exact operation-scoped execution profile")
    if not profile_guard_path.exists():
        err("7", "missing-profile-guard", "Gate 7 air-verify CLI/profile contract was not checked")
    else:
        pg = load_json(profile_guard_path)
        if not isinstance(pg, dict) or pg.get("status") != "PASS":
            err("7", "profile-guard-failed", "Gate 7 verification path is not proven to pin all required product tactics")

    # R3 Gate 9: require corrected decomposition rather than health-ready + request-local TTFT.
    startup_decomp = root / "startup" / "startup-decomposition.json"
    if not startup_decomp.exists():
        err("9", "missing-startup-decomposition", "Gate 9 corrected startup/first-use/plan-transition decomposition is missing")
    else:
        sd = load_json(startup_decomp)
        if not isinstance(sd, dict) or sd.get("schema") != "air.prompt15.startup-decomposition.v2":
            err("9", "bad-startup-decomposition", "Gate 9 startup decomposition schema is missing or unexpected")
        else:
            for key in ("process_startup_to_health_ms", "first_cuda_reuse8_external_total_ms", "dense_after_cuda_warm_external_total_ms", "dense_plan_preparation_ms", "hot_dense_external_total_ms", "cold_dense_excess_over_hot_ms", "first_dense_non_plan_residual_ms"):
                v = sd.get(key)
                if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    err("9", "nonfinite-startup-component", f"Gate 9 component {key} is missing/non-finite")

    # R3 Gate 10: comparison admission can be INADMISSIBLE without corrupting other gates.
    external = root / "external"
    if external.exists() and not (external / "SKIPPED.txt").exists():
        contract_path = external / "CONTRACT.json"
        if not contract_path.exists():
            warn("10", "external-contract-missing", "external comparison ran but no explicit comparability contract was produced; results are inadmissible")
            scientific["external_comparison"] = "INADMISSIBLE"
        else:
            contract = load_json(contract_path)
            if isinstance(contract, dict) and contract.get("status") == "PASS":
                scientific["external_comparison"] = "ADMITTED"
            else:
                scientific["external_comparison"] = "INADMISSIBLE"
                warn("10", "external-inadmissible", "external comparison lacks sufficient explicit token/pairing/success evidence; do not average it into competitive conclusions")

    # R3 collection depth is fixed before machine results.
    preflight = root / "preflight.txt"
    if preflight.exists():
        t = preflight.read_text(errors="replace")
        expected = {"stress_cycles": "20", "verify_decisions": "16", "external_rounds": "5"}
        for k, v in expected.items():
            m = re.search(rf"^{re.escape(k)}=(.+)$", t, re.M)
            if not m or m.group(1).strip() != v:
                err("0", "collection-depth-mismatch", f"preflight {k} must be {v} for this Prompt-15 qualification")

    structural_errors = [asdict(f) for f in findings if f.severity == "error"]
    warnings = [asdict(f) for f in findings if f.severity == "warning"]
    infos = [asdict(f) for f in findings if f.severity == "info"]
    return {
        "schema": "air.prompt15.evidence-integrity.v2",
        "root": str(root),
        "evidence_valid": not structural_errors,
        "structural_error_count": len(structural_errors),
        "warning_count": len(warnings),
        "errors": structural_errors,
        "warnings": warnings,
        "info": infos,
        "scientific_observations": scientific,
        "note": "Evidence validity is separate from AIR product success. A real correctness/performance failure can be valid evidence.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate AIR Prompt-15 evidence integrity without deciding competitive claims.")
    ap.add_argument("root", type=Path, help="Extracted AIR-Performance15-<timestamp> evidence directory")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = check_archive(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0 if result["evidence_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
