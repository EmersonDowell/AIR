#!/usr/bin/env python3
"""Controlled AIR vs llama.cpp comparison harness.

This tool is evidence-only. It does not modify AIR execution policy or model
state. Primary comparisons use the two HTTP servers with the same raw prompt
text and the same greedy generation limit. The harness asserts that both
servers report the same prompt-token count before a workload is admitted into
the measured set.

Primary comparable metrics are measured by this client: TTFT, total wall time,
and output tokens/s. Engine-native prefill/decode timings are recorded as a
secondary lane because AIR and llama.cpp do not define every internal timing
boundary identically.
"""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import threading
from typing import Any

TCRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    24: 2.064, 29: 2.045, 30: 2.042,
}


def tcrit95(df: int) -> float:
    if df <= 0:
        return math.inf
    if df in TCRIT_975:
        return TCRIT_975[df]
    keys = sorted(TCRIT_975)
    lower = max((k for k in keys if k <= df), default=keys[0])
    if df > keys[-1]:
        return 1.96
    return TCRIT_975[lower]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], *, out: Path | None = None, check: bool = True,
        timeout: float | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=timeout, env=merged)
    if out:
        out.write_text(cp.stdout)
    if check and cp.returncode != 0:
        raise RuntimeError(f"command failed ({cp.returncode}): {shlex.join(cmd)}\n{cp.stdout}")
    return cp


def tokenize_air(air_cli: str, model: Path, text: str) -> list[int]:
    cp = run([air_cli, "tokenize", str(model), text])
    m = re.search(r"tokens\((\d+)\):(.*)", cp.stdout.strip())
    if not m:
        raise RuntimeError(f"unable to parse air-cli tokenize output: {cp.stdout}")
    tokens = [int(x) for x in m.group(2).split()] if m.group(2).strip() else []
    if len(tokens) != int(m.group(1)):
        raise RuntimeError("air-cli token count disagrees with token list")
    return tokens


def build_prompt(air_cli: str, model: Path, target: int) -> tuple[str, list[int]]:
    # Repeated natural-language material avoids pathological all-identical-token
    # prompts while remaining deterministic and easy to reproduce.
    atom = (
        "Adaptive inference runtimes execute transformer layers while tracking "
        "memory ownership, scheduling decisions, numerical evidence, and request "
        "latency under a controlled workload. "
    )
    parts: list[str] = []
    tokens: list[int] = []
    while len(tokens) < target:
        parts.append(atom)
        text = "".join(parts)
        tokens = tokenize_air(air_cli, model, text)
        if len(parts) > 10000:
            raise RuntimeError("unable to construct comparison prompt")
    return "".join(parts), tokens


def http_json(host: str, port: int, method: str, path: str,
              payload: dict[str, Any] | None = None, timeout: float = 120.0) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    body = None if payload is None else json.dumps(payload, separators=(",", ":"))
    headers = {} if body is None else {"Content-Type": "application/json"}
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    status = response.status
    conn.close()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"non-JSON response {status} from {path}: {raw[:500]!r}") from exc
    return status, parsed


def wait_ready(host: str, port: int, path: str, proc: subprocess.Popen[str], timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before ready with code {proc.returncode}")
        try:
            status, payload = http_json(host, port, "GET", path, timeout=2.0)
            if 200 <= status < 300:
                return
            last = f"HTTP {status}: {payload}"
        except Exception as exc:
            last = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"server did not become ready: {last}")


def stop_proc(proc: subprocess.Popen[str] | None, timeout: float = 20.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def gpu_snapshot() -> dict[str, float | None]:
    fields = ["temperature.gpu", "power.draw", "clocks.sm", "memory.used", "utilization.gpu"]
    cp = run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"], check=False)
    if cp.returncode != 0 or not cp.stdout.strip():
        return {"temp_c": None, "power_w": None, "pclk_mhz": None, "memory_mib": None, "util_pct": None}
    values = [x.strip() for x in cp.stdout.splitlines()[0].split(",")]
    def val(i: int) -> float | None:
        try:
            return float(values[i])
        except Exception:
            return None
    return {"temp_c": val(0), "power_w": val(1), "pclk_mhz": val(2), "memory_mib": val(3), "util_pct": val(4)}


def process_vram_mib(pid: int) -> float | None:
    cp = run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], check=False)
    if cp.returncode != 0:
        return None
    total = 0.0
    found = False
    for line in cp.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) == pid:
                total += float(parts[1])
                found = True
        except ValueError:
            continue
    return total if found else None


def sse_request(host: str, port: int, path: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    body = json.dumps(payload, separators=(",", ":"))
    start = time.perf_counter()
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        raw = response.read().decode("utf-8", "replace")
        conn.close()
        raise RuntimeError(f"HTTP {response.status} from {path}: {raw[:1000]}")
    first_token_at: float | None = None
    events: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    while True:
        raw = response.readline()
        if not raw:
            break
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(event)
        is_air_done = event.get("done") is True
        is_llama_done = event.get("stop") is True
        is_error = "error" in event
        if first_token_at is None and not is_air_done and not is_llama_done and not is_error:
            first_token_at = time.perf_counter()
        if is_air_done or is_llama_done or "timings" in event:
            final = event
    end = time.perf_counter()
    conn.close()
    if first_token_at is None:
        raise RuntimeError(f"stream produced no token event: {events[-3:]}")
    return {
        "external_ttft_ms": (first_token_at - start) * 1000.0,
        "external_total_ms": (end - start) * 1000.0,
        "events": len(events),
        "final": final or (events[-1] if events else {}),
    }


def air_payload(prompt: str, generated: int, seed: int, stream: bool) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "max_tokens": generated,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": seed,
        "stream": stream,
    }


def llama_payload(prompt: str, generated: int, seed: int, stream: bool) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": generated,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": seed,
        "stream": stream,
        "cache_prompt": False,
    }


def warm_and_count(engine: str, host: str, port: int, prompt: str, seed: int) -> tuple[int, dict[str, Any]]:
    if engine == "air":
        status, obj = http_json(host, port, "POST", "/generate", air_payload(prompt, 1, seed, False))
        if status != 200:
            raise RuntimeError(f"AIR warmup failed: {status} {obj}")
        return int(obj["usage"]["prompt_tokens"]), obj
    status, obj = http_json(host, port, "POST", "/completion", llama_payload(prompt, 1, seed, False))
    if status != 200:
        raise RuntimeError(f"llama warmup failed: {status} {obj}")
    timings = obj.get("timings", {})
    count = int(timings.get("prompt_n", obj.get("tokens_evaluated", -1)))
    if count < 0:
        raise RuntimeError(f"llama response lacks prompt token count: {obj}")
    return count, obj


def stream_sample(engine: str, host: str, port: int, prompt: str, generated: int,
                  seed: int) -> dict[str, Any]:
    before = gpu_snapshot()
    sample = stream_request_metrics(engine, host, port, prompt, generated, seed)
    after = gpu_snapshot()
    return {
        "engine": engine,
        **sample,
        "gpu_before": before,
        "gpu_after": after,
    }


def numeric(values: list[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out.append(float(v))
    return out


def median(values: list[Any]) -> float | None:
    vals = numeric(values)
    return statistics.median(vals) if vals else None


def mean(values: list[Any]) -> float | None:
    vals = numeric(values)
    return statistics.mean(vals) if vals else None


def paired_ratio_ci(samples: list[dict[str, Any]], field: str, higher_is_better: bool) -> dict[str, Any]:
    by_round: dict[int, dict[str, float]] = {}
    for s in samples:
        v: Any = s
        for part in field.split("."):
            if not isinstance(v, dict):
                v = None
                break
            v = v.get(part)
        if not isinstance(v, (int, float)) or float(v) <= 0 or not math.isfinite(float(v)):
            continue
        by_round.setdefault(int(s["round"]), {})[s["engine"]] = float(v)
    logs: list[float] = []
    for pair in by_round.values():
        if "air" in pair and "llama" in pair:
            ratio = pair["air"] / pair["llama"]
            # Report AIR/llama uniformly. Consumers interpret latency (>1 slower)
            # versus throughput (>1 faster) using higher_is_better.
            logs.append(math.log(ratio))
    if not logs:
        return {"n": 0, "air_over_llama": None, "ci95": [None, None], "higher_is_better": higher_is_better}
    mu = statistics.mean(logs)
    if len(logs) >= 2:
        sd = statistics.stdev(logs)
        half = tcrit95(len(logs) - 1) * sd / math.sqrt(len(logs))
    else:
        half = math.inf
    lo = math.exp(mu - half) if math.isfinite(half) else None
    hi = math.exp(mu + half) if math.isfinite(half) else None
    return {"n": len(logs), "air_over_llama": math.exp(mu), "ci95": [lo, hi], "higher_is_better": higher_is_better}



def percentile(values: list[Any], q: float) -> float | None:
    vals = sorted(numeric(values))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    weight = pos - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def stream_request_metrics(engine: str, host: str, port: int, prompt: str,
                           generated: int, seed: int) -> dict[str, Any]:
    """One request without per-request nvidia-smi calls.

    Scaling groups take one GPU snapshot around the whole concurrent batch so
    telemetry collection does not perturb individual request launch timing.
    """
    if engine == "air":
        raw = sse_request(host, port, "/generate", air_payload(prompt, generated, seed, True))
        final = raw["final"]
        metrics = final.get("metrics", {})
        usage = final.get("usage", {})
        gen = int(usage.get("completion_tokens", metrics.get("generated_tokens", 0)))
        internal = {
            "prompt_tokens": int(usage.get("prompt_tokens", metrics.get("prompt_tokens", 0))),
            "generated_tokens": gen,
            "prefill_ms": metrics.get("prefill_ms"),
            "decode_ms": metrics.get("decode_ms"),
            "internal_ttft_ms": metrics.get("ttft_ms"),
            "internal_total_ms": metrics.get("total_ms"),
            "prefill_tps": metrics.get("prefill_tokens_per_second"),
            "decode_tps": metrics.get("decode_tokens_per_second"),
        }
    else:
        raw = sse_request(host, port, "/completion", llama_payload(prompt, generated, seed, True))
        final = raw["final"]
        timings = final.get("timings", {})
        gen = int(timings.get("predicted_n", final.get("tokens_predicted", 0)))
        prompt_n = int(timings.get("prompt_n", final.get("tokens_evaluated", 0)))
        internal = {
            "prompt_tokens": prompt_n,
            "generated_tokens": gen,
            "prefill_ms": timings.get("prompt_ms"),
            "decode_ms": timings.get("predicted_ms"),
            "internal_ttft_ms": None,
            "internal_total_ms": None,
            "prefill_tps": timings.get("prompt_per_second"),
            "decode_tps": timings.get("predicted_per_second"),
        }
    wall_s = raw["external_total_ms"] / 1000.0
    return {
        "external_ttft_ms": raw["external_ttft_ms"],
        "external_total_ms": raw["external_total_ms"],
        "external_output_tps": (gen / wall_s) if gen > 0 and wall_s > 0 else 0.0,
        "event_count": raw["events"],
        "internal": internal,
    }


def stream_group(engine: str, host: str, port: int, prompt: str,
                 generated: int, seed: int, concurrency: int) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    barrier = threading.Barrier(concurrency + 1)

    def one(index: int) -> dict[str, Any]:
        barrier.wait()
        item = stream_request_metrics(engine, host, port, prompt, generated, seed + index)
        item["request_index"] = index
        return item

    before = gpu_snapshot()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"air-scale-{engine}") as pool:
        futures = [pool.submit(one, i) for i in range(concurrency)]
        group_start = time.perf_counter()
        barrier.wait()
        requests = [f.result() for f in futures]
        group_end = time.perf_counter()
    after = gpu_snapshot()

    generated_total = sum(int(r["internal"].get("generated_tokens") or 0) for r in requests)
    wall_s = group_end - group_start
    return {
        "engine": engine,
        "concurrency": concurrency,
        "group_wall_ms": wall_s * 1000.0,
        "aggregate_output_tps": (generated_total / wall_s) if generated_total > 0 and wall_s > 0 else 0.0,
        "requests_per_second": (concurrency / wall_s) if wall_s > 0 else 0.0,
        "ttft_median_ms": median([r["external_ttft_ms"] for r in requests]),
        "ttft_p95_ms": percentile([r["external_ttft_ms"] for r in requests], 0.95),
        "total_median_ms": median([r["external_total_ms"] for r in requests]),
        "total_p95_ms": percentile([r["external_total_ms"] for r in requests], 0.95),
        "native_prefill_tps_median": median([r["internal"].get("prefill_tps") for r in requests]),
        "native_decode_tps_median": median([r["internal"].get("decode_tps") for r in requests]),
        "generated_tokens": generated_total,
        "gpu_before": before,
        "gpu_after": after,
        "requests": requests,
    }


def parse_int_csv(text: str, *, minimum: int = 1) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value < minimum:
            raise ValueError(f"value must be >= {minimum}: {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("empty integer list")
    return values


def scaling_cell_key(prompt_tokens: int, concurrency: int) -> str:
    return f"p{prompt_tokens}-c{concurrency}"


def render_scaling_report(meta: dict[str, Any], samples: list[dict[str, Any]],
                          ratios: dict[str, Any]) -> str:
    lines = [
        "# AIR vs llama.cpp scaling matrix",
        "",
        f"Model SHA-256: `{meta['model_sha256']}`",
        f"Rounds per cell: {meta['rounds']}  |  order seed: {meta['order_seed']}",
        f"Requested prompt targets: {meta['prompt_targets']}  |  concurrency: {meta['concurrency_levels']}",
        "",
        "Primary scaling metrics are measured by the same client. Concurrent cells launch",
        "the requested number of identical raw requests together. Internal engine timers remain secondary evidence.",
        "",
        "| Cell | Actual prompt tok | C | Engine | TTFT median ms | Total median ms | Aggregate tok/s | Req/s | Start C | End C | Start pclk | End pclk |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cells = meta["cells"]
    for key, spec in cells.items():
        subset = [s for s in samples if s["cell"] == key]
        for engine in ("air", "llama"):
            ss = [s for s in subset if s["engine"] == engine]
            fmt = lambda x, p=2: "" if x is None else f"{x:.{p}f}"
            lines.append(
                f"| {key} | {spec['prompt_tokens']} | {spec['concurrency']} | {engine} | "
                f"{fmt(median([x['ttft_median_ms'] for x in ss]))} | "
                f"{fmt(median([x['total_median_ms'] for x in ss]))} | "
                f"{fmt(median([x['aggregate_output_tps'] for x in ss]))} | "
                f"{fmt(median([x['requests_per_second'] for x in ss]))} | "
                f"{fmt(mean([x['gpu_before'].get('temp_c') for x in ss]), 1)} | "
                f"{fmt(mean([x['gpu_after'].get('temp_c') for x in ss]), 1)} | "
                f"{fmt(mean([x['gpu_before'].get('pclk_mhz') for x in ss]), 0)} | "
                f"{fmt(mean([x['gpu_after'].get('pclk_mhz') for x in ss]), 0)} |"
            )
        r = ratios[key]
        lines += [
            "",
            f"- {key} TTFT AIR/llama: {r['ttft_latency']['air_over_llama']:.3f}x, 95% CI "
            f"[{r['ttft_latency']['ci95'][0]:.3f}, {r['ttft_latency']['ci95'][1]:.3f}]",
            f"- {key} total AIR/llama: {r['total_latency']['air_over_llama']:.3f}x, 95% CI "
            f"[{r['total_latency']['ci95'][0]:.3f}, {r['total_latency']['ci95'][1]:.3f}]",
            f"- {key} aggregate throughput AIR/llama: {r['aggregate_output_tps']['air_over_llama']:.3f}x, 95% CI "
            f"[{r['aggregate_output_tps']['ci95'][0]:.3f}, {r['aggregate_output_tps']['ci95'][1]:.3f}]",
            "",
        ]
    lines += [
        "## Interpretation rules",
        "",
        "- Latency ratios below 1 favor AIR; throughput ratios above 1 favor AIR.",
        "- A 95% interval spanning 1 is inconclusive.",
        "- Model, prompt-length, and concurrency cells are all retained; no workload may be dropped because it is unfavorable.",
        "- Prompt-token agreement is a hard admission gate for every prompt length.",
        "- The model file is identical within every AIR/llama pair and is identified by SHA-256.",
        "- Engine-native timing boundaries are not treated as interchangeable.",
        "- This matrix is evidence only and does not authorize a frozen-runtime architecture change.",
        "",
    ]
    return "\n".join(lines)


def run_scaling(args: argparse.Namespace) -> int:
    model = Path(args.model).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")

    prompt_targets = parse_int_csv(args.prompt_targets)
    concurrency_levels = parse_int_csv(args.concurrency_levels)
    max_concurrency = max(concurrency_levels)
    if args.rounds < 3:
        raise RuntimeError("--rounds must be >= 3")

    system = {
        "schema": "air.scaling.meta.v1",
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": sha256(model),
        "date_unix": time.time(),
        "rounds": args.rounds,
        "order_seed": args.order_seed,
        "sampling_seed": args.sampling_seed,
        "context": args.context,
        "prompt_targets": prompt_targets,
        "concurrency_levels": concurrency_levels,
        "generated_tokens": args.scaling_generated,
        "air_server": args.air_server,
        "air_server_extra_args": args.air_server_extra_args,
        "llama_server": args.llama_server,
    }
    versions = {}
    for name, cmd in (
        ("air_server", [args.air_server, "--version"]),
        ("air_cli", [args.air_cli, "--version"]),
        ("llama_server", [args.llama_server, "--version"]),
        ("nvidia_smi", ["nvidia-smi"]),
    ):
        cp = run(cmd, check=False, timeout=30)
        versions[name] = {"returncode": cp.returncode, "output": cp.stdout.strip()}
    system["versions"] = versions
    system["gpu_initial"] = gpu_snapshot()

    prompts: dict[int, dict[str, Any]] = {}
    for target in prompt_targets:
        text, tokens = build_prompt(args.air_cli, model, target)
        prompts[target] = {"prompt": text, "prompt_tokens": len(tokens)}
    (out / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")

    # Correctness is rechecked for every model in the scaling matrix.
    shortest = min(prompts.values(), key=lambda x: x["prompt_tokens"])
    air_verify_json = out / "air-verification.json"
    run([
        args.air_verify, "-m", str(model), "--prompt", shortest["prompt"],
        "--generate", "16", "--top-k", "8", "--device", "0", "--atol", "0.001",
        "--output", str(air_verify_json),
    ], out=out / "air-verification.txt", check=True, timeout=1200)

    llama_correct_log = (out / "llama-correctness-server.txt").open("w")
    correctness_proc: subprocess.Popen[str] | None = None
    try:
        correctness_proc = subprocess.Popen([
            args.llama_server, "-m", str(model), "-ngl", "99", "-c", str(args.context),
            "-np", "1", "-fa", "on", "--host", "127.0.0.1", "--port", str(args.llama_port),
        ], stdout=llama_correct_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        wait_ready("127.0.0.1", args.llama_port, "/health", correctness_proc)
        run([
            sys.executable, args.verify_script, str(air_verify_json),
            "--url", f"http://127.0.0.1:{args.llama_port}", "--top-k", "8",
            "--output", str(out / "external-verification.json"),
        ], out=out / "external-verification.txt", check=True, timeout=1200)
    finally:
        stop_proc(correctness_proc)
        llama_correct_log.close()
    time.sleep(max(0.0, args.cooldown))

    # Isolated idle residency at the matrix's maximum configured concurrency.
    residency: dict[str, Any] = {}
    for engine in ("air", "llama"):
        log = (out / f"{engine}-residency-server.txt").open("w")
        proc: subprocess.Popen[str] | None = None
        try:
            if engine == "air":
                cmd = [args.air_server, "-m", str(model), "--backend", "cuda", "--no-manifest",
                       "--host", "127.0.0.1", "--port", str(args.air_port),
                       "--max-active", str(max_concurrency), "--prefix-cache", "0",
                       "--workers", str(max(4, max_concurrency)),
                       "--max-connections", str(max(8, max_concurrency * 2)), "--io-timeout", "300"] + shlex.split(args.air_server_extra_args)
                port = args.air_port
            else:
                cmd = [args.llama_server, "-m", str(model), "-ngl", "99", "-c", str(args.context),
                       "-np", str(max_concurrency), "-fa", "on", "--host", "127.0.0.1", "--port", str(args.llama_port)]
                port = args.llama_port
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            wait_ready("127.0.0.1", port, "/health", proc)
            # Warm once before residency measurement so lazy AIR PreparedModel
            # artifacts and the comparator's analogous request-time state are
            # reflected in process VRAM rather than reporting idle model load only.
            warm_prompt = shortest["prompt"]
            warm_and_count(engine, "127.0.0.1", port, warm_prompt, args.sampling_seed)
            time.sleep(1.0)
            record: dict[str, Any] = {"pid": proc.pid, "process_vram_mib": process_vram_mib(proc.pid), "gpu": gpu_snapshot()}
            if engine == "air":
                _, record["runtime"] = http_json("127.0.0.1", port, "GET", "/runtime")
            residency[engine] = record
        finally:
            stop_proc(proc)
            log.close()
        time.sleep(max(0.0, args.cooldown))
    (out / "residency.json").write_text(json.dumps(residency, indent=2) + "\n")

    air_log = (out / "air-server.txt").open("w")
    llama_log = (out / "llama-server.txt").open("w")
    dmon_log = (out / "gpu-dmon.txt").open("w")
    air_proc = llama_proc = dmon_proc = None
    samples: list[dict[str, Any]] = []
    try:
        air_proc = subprocess.Popen([
            args.air_server, "-m", str(model), "--backend", "cuda", "--no-manifest",
            "--host", "127.0.0.1", "--port", str(args.air_port),
            "--max-active", str(max_concurrency), "--prefix-cache", "0",
            "--workers", str(max(4, max_concurrency)),
            "--max-connections", str(max(8, max_concurrency * 2)), "--io-timeout", "600",
        ] + shlex.split(args.air_server_extra_args), stdout=air_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        wait_ready("127.0.0.1", args.air_port, "/health", air_proc)
        llama_proc = subprocess.Popen([
            args.llama_server, "-m", str(model), "-ngl", "99", "-c", str(args.context),
            "-np", str(max_concurrency), "-fa", "on", "--host", "127.0.0.1", "--port", str(args.llama_port),
        ], stdout=llama_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        wait_ready("127.0.0.1", args.llama_port, "/health", llama_proc)
        dmon_proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1"],
                                     stdout=dmon_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)

        checks: dict[str, Any] = {}
        for target, spec in prompts.items():
            a_count, _ = warm_and_count("air", "127.0.0.1", args.air_port, spec["prompt"], args.sampling_seed)
            l_count, _ = warm_and_count("llama", "127.0.0.1", args.llama_port, spec["prompt"], args.sampling_seed)
            checks[str(target)] = {"air": a_count, "llama": l_count, "expected_air_cli": spec["prompt_tokens"]}
            if a_count != l_count or a_count != spec["prompt_tokens"]:
                raise RuntimeError(f"prompt-token mismatch target={target}: {checks[str(target)]}")
        (out / "prompt-token-checks.json").write_text(json.dumps(checks, indent=2) + "\n")

        rng = random.Random(args.order_seed)
        cells: dict[str, Any] = {}
        for target in prompt_targets:
            spec = prompts[target]
            for concurrency in concurrency_levels:
                key = scaling_cell_key(spec["prompt_tokens"], concurrency)
                cells[key] = {
                    "target_prompt_tokens": target,
                    "prompt_tokens": spec["prompt_tokens"],
                    "generated_tokens_per_request": args.scaling_generated,
                    "concurrency": concurrency,
                }
                for round_index in range(1, args.rounds + 1):
                    order = ["air", "llama"]
                    rng.shuffle(order)
                    for position, engine in enumerate(order):
                        if engine == "air":
                            sample = stream_group(engine, "127.0.0.1", args.air_port, spec["prompt"],
                                                  args.scaling_generated, args.sampling_seed, concurrency)
                        else:
                            sample = stream_group(engine, "127.0.0.1", args.llama_port, spec["prompt"],
                                                  args.scaling_generated, args.sampling_seed, concurrency)
                        sample.update({
                            "cell": key, "target_prompt_tokens": target,
                            "prompt_tokens": spec["prompt_tokens"], "round": round_index,
                            "order_position": position,
                        })
                        samples.append(sample)
                        with (out / "scaling-samples.jsonl").open("a") as f:
                            f.write(json.dumps(sample, separators=(",", ":")) + "\n")
                        time.sleep(max(0.0, args.cooldown))
    finally:
        stop_proc(dmon_proc, timeout=3)
        stop_proc(air_proc)
        stop_proc(llama_proc)
        air_log.close(); llama_log.close(); dmon_log.close()

    system["cells"] = cells
    system["residency"] = residency
    ratios: dict[str, Any] = {}
    for key in cells:
        subset = [s for s in samples if s["cell"] == key]
        ratios[key] = {
            "ttft_latency": paired_ratio_ci(subset, "ttft_median_ms", False),
            "total_latency": paired_ratio_ci(subset, "total_median_ms", False),
            "aggregate_output_tps": paired_ratio_ci(subset, "aggregate_output_tps", True),
            "requests_per_second": paired_ratio_ci(subset, "requests_per_second", True),
            "native_prefill_tps": paired_ratio_ci(subset, "native_prefill_tps_median", True),
            "native_decode_tps": paired_ratio_ci(subset, "native_decode_tps_median", True),
        }
    report = {
        "schema": "air.scaling.v1",
        "meta": system,
        "samples": samples,
        "ratios": ratios,
    }
    (out / "scaling.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "SCALING_REPORT.md").write_text(render_scaling_report(system, samples, ratios) + "\n")

    if not args.skip_microbench:
        micro = out / "microbench"
        micro.mkdir(exist_ok=True)
        for target in prompt_targets:
            spec = prompts[target]
            run(["air-bench", "-m", str(model), "--prompt", spec["prompt"], "--backend", "cuda",
                 "--tokens", str(args.scaling_generated), "--warmup", "1", "--runs", str(args.rounds),
                 "--concurrency", "1", "--no-manifest", "--output", str(micro / f"air-p{spec['prompt_tokens']}.json")],
                out=micro / f"air-p{spec['prompt_tokens']}.txt", check=True, timeout=3600)
            if args.llama_bench:
                run([args.llama_bench, "-m", str(model), "-p", str(spec["prompt_tokens"]),
                     "-n", str(args.scaling_generated), "-r", str(args.rounds), "-ngl", "99", "-fa", "on", "-o", "json"],
                    out=micro / f"llama-p{spec['prompt_tokens']}.json", check=True, timeout=3600)

    print(out / "SCALING_REPORT.md")
    print(out / "scaling.json")
    return 0


def render_report(meta: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# AIR vs llama.cpp controlled comparison",
        "",
        f"Model SHA-256: `{meta['model_sha256']}`",
        f"Rounds: {meta['rounds']}  |  order seed: {meta['order_seed']}",
        "",
        "Primary metrics below are measured by the same HTTP/SSE client. Internal prefill/decode",
        "rates are secondary engine-native evidence and should not be interpreted as identical timer boundaries.",
        "",
        "| Workload | Prompt tokens | New tokens | Engine | TTFT median ms | Total median ms | Output median tok/s | Native prefill tok/s | Native decode tok/s | Start temp C | Start pclk MHz |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    workloads = meta["workloads"]
    for name, spec in workloads.items():
        subset = [s for s in samples if s["workload"] == name]
        for engine in ("air", "llama"):
            ss = [s for s in subset if s["engine"] == engine]
            fmt = lambda x, p=2: "" if x is None else f"{x:.{p}f}"
            lines.append(
                f"| {name} | {spec['prompt_tokens']} | {spec['generated_tokens']} | {engine} | "
                f"{fmt(median([x['external_ttft_ms'] for x in ss]))} | "
                f"{fmt(median([x['external_total_ms'] for x in ss]))} | "
                f"{fmt(median([x['external_output_tps'] for x in ss]))} | "
                f"{fmt(median([x['internal'].get('prefill_tps') for x in ss]))} | "
                f"{fmt(median([x['internal'].get('decode_tps') for x in ss]))} | "
                f"{fmt(mean([x['gpu_before'].get('temp_c') for x in ss]), 1)} | "
                f"{fmt(mean([x['gpu_before'].get('pclk_mhz') for x in ss]), 0)} |"
            )
        lines += ["", "Paired AIR/llama ratios (95% Student-t CI over round log-ratios):", ""]
        for label, field, hib in (
            ("TTFT latency", "external_ttft_ms", False),
            ("total latency", "external_total_ms", False),
            ("external output throughput", "external_output_tps", True),
            ("native prefill throughput", "internal.prefill_tps", True),
            ("native decode throughput", "internal.decode_tps", True),
        ):
            r = paired_ratio_ci(subset, field, hib)
            ratio = r["air_over_llama"]
            lo, hi = r["ci95"]
            if ratio is None:
                lines.append(f"- {label}: insufficient paired evidence")
            elif lo is None:
                lines.append(f"- {label}: AIR/llama {ratio:.3f}x (n={r['n']}; CI unavailable)")
            else:
                lines.append(f"- {label}: AIR/llama {ratio:.3f}x, 95% CI [{lo:.3f}, {hi:.3f}], n={r['n']}")
        lines.append("")
    lines += [
        "## Interpretation rules",
        "",
        "- Latency ratios below 1 favor AIR; throughput ratios above 1 favor AIR.",
        "- A ratio whose 95% CI spans 1 is not treated as a demonstrated difference.",
        "- Server-internal prefill/decode metrics are kept separate from client-measured TTFT/total latency.",
        "- Both engines receive the same raw prompt text. Workloads are rejected if reported prompt-token counts differ.",
        "- This prompt compares concurrency=1. Concurrency/context scaling is reserved for Comparison Prompt 8.",
        "- No optimization decision should be made from this report alone.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--air-server", default="air-server")
    ap.add_argument("--air-cli", default="air-cli")
    ap.add_argument("--air-verify", default="air-verify")
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--llama-bench", default="")
    ap.add_argument("--verify-script", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--order-seed", type=int, default=1337)
    ap.add_argument("--sampling-seed", type=int, default=424242)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--air-port", type=int, default=18470)
    ap.add_argument("--llama-port", type=int, default=18471)
    ap.add_argument("--air-server-extra-args", default="",
                    help="fixed extra AIR server arguments for this comparison lane")
    ap.add_argument("--scaling", action="store_true", help="run Prompt 8 prompt/concurrency scaling for one model")
    ap.add_argument("--prompt-targets", default="32,256,1024")
    ap.add_argument("--concurrency-levels", default="1,2,4")
    ap.add_argument("--scaling-generated", type=int, default=32)
    ap.add_argument("--skip-microbench", action="store_true")
    args = ap.parse_args()
    if args.rounds < 3:
        ap.error("--rounds must be >= 3")
    if args.scaling:
        return run_scaling(args)

    model = Path(args.model).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not model.is_file():
        raise SystemExit(f"model does not exist: {model}")

    # System/evidence identity.
    system = {
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": sha256(model),
        "date_unix": time.time(),
        "rounds": args.rounds,
        "order_seed": args.order_seed,
        "sampling_seed": args.sampling_seed,
        "context": args.context,
        "air_server": args.air_server,
        "air_server_extra_args": args.air_server_extra_args,
        "llama_server": args.llama_server,
    }
    versions = {}
    for name, cmd in (
        ("air_server", [args.air_server, "--version"]),
        ("air_cli", [args.air_cli, "--version"]),
        ("air_verify", [args.air_verify, "--version"]),
        ("llama_server", [args.llama_server, "--version"]),
        ("nvidia_smi", ["nvidia-smi"]),
    ):
        cp = run(cmd, check=False, timeout=30)
        versions[name] = {"returncode": cp.returncode, "output": cp.stdout.strip()}
    system["versions"] = versions
    system["gpu_initial"] = gpu_snapshot()
    (out / "system.json").write_text(json.dumps(system, indent=2) + "\n")

    # Build deterministic workload texts and freeze them before any server run.
    decode_text = "The capital of France is"
    decode_tokens = tokenize_air(args.air_cli, model, decode_text)
    balanced_text, balanced_tokens = build_prompt(args.air_cli, model, 128)
    prefill_text, prefill_tokens = build_prompt(args.air_cli, model, 1024)
    workloads: dict[str, dict[str, Any]] = {
        "decode": {"prompt": decode_text, "prompt_tokens": len(decode_tokens), "generated_tokens": 256},
        "balanced": {"prompt": balanced_text, "prompt_tokens": len(balanced_tokens), "generated_tokens": 128},
        "prefill": {"prompt": prefill_text, "prompt_tokens": len(prefill_tokens), "generated_tokens": 32},
    }
    (out / "workloads.json").write_text(json.dumps(workloads, indent=2) + "\n")

    # Correctness lane. AIR report chooses canonical teacher-forced tokens; llama
    # consumes the exact numeric history via /completion.
    air_verify_json = out / "air-verification.json"
    air_verify_txt = out / "air-verification.txt"
    verify_cp = run([
        args.air_verify, "-m", str(model), "--prompt", decode_text,
        "--generate", "32", "--top-k", "8", "--device", "0",
        "--atol", "0.001", "--output", str(air_verify_json),
    ], out=air_verify_txt, check=True, timeout=600)
    _ = verify_cp

    llama_log = (out / "llama-correctness-server.txt").open("w")
    llama_proc: subprocess.Popen[str] | None = None
    try:
        llama_cmd = [
            args.llama_server, "-m", str(model), "-ngl", "99", "-c", str(args.context),
            "-np", "1", "-fa", "on", "--host", "127.0.0.1", "--port", str(args.llama_port),
        ]
        llama_proc = subprocess.Popen(llama_cmd, stdout=llama_log, stderr=subprocess.STDOUT,
                                      text=True, start_new_session=True)
        wait_ready("127.0.0.1", args.llama_port, "/health", llama_proc)
        run([
            sys.executable, args.verify_script, str(air_verify_json),
            "--url", f"http://127.0.0.1:{args.llama_port}", "--top-k", "8",
            "--output", str(out / "external-verification.json"),
        ], out=out / "external-verification.txt", check=True, timeout=600)
    finally:
        stop_proc(llama_proc)
        llama_log.close()
    time.sleep(max(0.0, args.cooldown))

    # Isolated residency. This avoids claiming the combined two-server residency
    # as either engine's own footprint.
    residency: dict[str, Any] = {}
    for engine in ("air", "llama"):
        log = (out / f"{engine}-residency-server.txt").open("w")
        proc = None
        try:
            if engine == "air":
                cmd = [args.air_server, "-m", str(model), "--backend", "cuda", "--no-manifest",
                       "--host", "127.0.0.1", "--port", str(args.air_port), "--max-active", "1",
                       "--prefix-cache", "0", "--workers", "4", "--max-connections", "8",
                       "--io-timeout", "120"] + shlex.split(args.air_server_extra_args)
                port = args.air_port
            else:
                cmd = [args.llama_server, "-m", str(model), "-ngl", "99", "-c", str(args.context),
                       "-np", "1", "-fa", "on", "--host", "127.0.0.1", "--port", str(args.llama_port)]
                port = args.llama_port
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            wait_ready("127.0.0.1", port, "/health", proc)
            # Warm once before residency measurement so lazy execution state is
            # included in the externally observed process footprint.
            warm_and_count(engine, "127.0.0.1", port, decode_text, args.sampling_seed)
            time.sleep(1.0)
            record: dict[str, Any] = {"pid": proc.pid, "process_vram_mib": process_vram_mib(proc.pid), "gpu": gpu_snapshot()}
            if engine == "air":
                _, record["runtime"] = http_json("127.0.0.1", port, "GET", "/runtime")
            residency[engine] = record
        finally:
            stop_proc(proc)
            log.close()
        time.sleep(max(0.0, args.cooldown))
    (out / "residency.json").write_text(json.dumps(residency, indent=2) + "\n")

    # Primary performance lane: both models resident, one request active at a
    # time, randomized engine order within every paired round.
    air_log = (out / "air-server.txt").open("w")
    llama_log = (out / "llama-server.txt").open("w")
    dmon_log = (out / "gpu-dmon.txt").open("w")
    air_proc = llama_proc = dmon_proc = None
    samples: list[dict[str, Any]] = []
    try:
        air_cmd = [args.air_server, "-m", str(model), "--backend", "cuda", "--no-manifest",
                   "--host", "127.0.0.1", "--port", str(args.air_port), "--max-active", "1",
                   "--prefix-cache", "0", "--workers", "4", "--max-connections", "8", "--io-timeout", "300"] + shlex.split(args.air_server_extra_args)
        llama_cmd = [args.llama_server, "-m", str(model), "-ngl", "99", "-c", str(args.context),
                     "-np", "1", "-fa", "on", "--host", "127.0.0.1", "--port", str(args.llama_port)]
        air_proc = subprocess.Popen(air_cmd, stdout=air_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        wait_ready("127.0.0.1", args.air_port, "/health", air_proc)
        llama_proc = subprocess.Popen(llama_cmd, stdout=llama_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        wait_ready("127.0.0.1", args.llama_port, "/health", llama_proc)
        dmon_proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1"],
                                     stdout=dmon_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)

        # Warm each workload and assert exact tokenizer-count agreement.
        prompt_checks = {}
        for name, spec in workloads.items():
            a_count, _ = warm_and_count("air", "127.0.0.1", args.air_port, spec["prompt"], args.sampling_seed)
            l_count, _ = warm_and_count("llama", "127.0.0.1", args.llama_port, spec["prompt"], args.sampling_seed)
            prompt_checks[name] = {"air": a_count, "llama": l_count, "expected_air_cli": spec["prompt_tokens"]}
            if a_count != l_count or a_count != spec["prompt_tokens"]:
                raise RuntimeError(f"prompt-token mismatch for {name}: {prompt_checks[name]}")
        (out / "prompt-token-checks.json").write_text(json.dumps(prompt_checks, indent=2) + "\n")

        rng = random.Random(args.order_seed)
        for name, spec in workloads.items():
            for round_index in range(1, args.rounds + 1):
                order = ["air", "llama"]
                rng.shuffle(order)
                for position, engine in enumerate(order):
                    if engine == "air":
                        sample = stream_sample(engine, "127.0.0.1", args.air_port, spec["prompt"], spec["generated_tokens"], args.sampling_seed)
                    else:
                        sample = stream_sample(engine, "127.0.0.1", args.llama_port, spec["prompt"], spec["generated_tokens"], args.sampling_seed)
                    sample.update({"workload": name, "round": round_index, "order_position": position})
                    samples.append(sample)
                    with (out / "samples.jsonl").open("a") as f:
                        f.write(json.dumps(sample, separators=(",", ":")) + "\n")
                    time.sleep(max(0.0, args.cooldown))
    finally:
        stop_proc(dmon_proc, timeout=3)
        stop_proc(air_proc)
        stop_proc(llama_proc)
        air_log.close(); llama_log.close(); dmon_log.close()

    meta = dict(system)
    meta["workloads"] = {k: {"prompt_tokens": v["prompt_tokens"], "generated_tokens": v["generated_tokens"]} for k, v in workloads.items()}
    report = {
        "schema": "air.comparison.v1",
        "meta": meta,
        "residency": residency,
        "samples": samples,
        "ratios": {
            name: {
                "ttft_latency": paired_ratio_ci([s for s in samples if s["workload"] == name], "external_ttft_ms", False),
                "total_latency": paired_ratio_ci([s for s in samples if s["workload"] == name], "external_total_ms", False),
                "external_output_tps": paired_ratio_ci([s for s in samples if s["workload"] == name], "external_output_tps", True),
                "native_prefill_tps": paired_ratio_ci([s for s in samples if s["workload"] == name], "internal.prefill_tps", True),
                "native_decode_tps": paired_ratio_ci([s for s in samples if s["workload"] == name], "internal.decode_tps", True),
            } for name in workloads
        },
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "REPORT.md").write_text(render_report(meta, samples) + "\n")

    # Secondary length-controlled native benchmark lane. These outputs are raw
    # evidence only and intentionally not folded into the primary paired ratios.
    micro = out / "microbench"
    micro.mkdir(exist_ok=True)
    for name, spec in workloads.items():
        run(["air-bench", "-m", str(model), "--prompt", spec["prompt"], "--backend", "cuda",
             "--tokens", str(spec["generated_tokens"]), "--warmup", "1", "--runs", str(args.rounds),
             "--concurrency", "1", "--no-manifest", "--output", str(micro / f"air-{name}.json")],
            out=micro / f"air-{name}.txt", check=True, timeout=1200)
        if args.llama_bench:
            run([args.llama_bench, "-m", str(model), "-p", str(spec["prompt_tokens"]),
                 "-n", str(spec["generated_tokens"]), "-r", str(args.rounds), "-ngl", "99",
                 "-fa", "on", "-o", "json"], out=micro / f"llama-{name}.json",
                check=True, timeout=1200)

    print(out / "REPORT.md")
    print(out / "comparison.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
