#!/usr/bin/env python3
import argparse, hashlib, json, platform
from pathlib import Path

def load(path):
    if path and path.exists():
        try: return json.loads(path.read_text())
        except Exception: return {}
    return {}

def meminfo():
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            parts = v.strip().split()
            if parts:
                out[k] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1)
    except Exception:
        pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=Path)
    ap.add_argument("--cuda", type=Path)
    ap.add_argument("--storage", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    host = load(args.host)
    cuda = load(args.cuda)
    storage = load(args.storage)
    mem = meminfo()
    nodes = [{
        "id": "cpu0", "kind": "cpu", "name": platform.processor() or platform.machine(),
        "backend": "cpu", "architecture": platform.machine(), "ordinal": 0,
        "numa_node": 0, "total_bytes": 0, "available_bytes": 0,
        "capabilities": [],
    }, {
        "id": "ram0", "kind": "host-memory", "name": "system RAM",
        "backend": "host", "architecture": "dram", "ordinal": 0,
        "numa_node": 0, "total_bytes": mem.get("MemTotal", 0),
        "available_bytes": mem.get("MemAvailable", 0),
        "capabilities": ["pageable", "pinned-capable" if cuda else "pageable-only"],
    }]
    links = []

    memcpy = host.get("memcpy", [])
    if memcpy:
        best = max((x.get("gib_per_second", 0.0) for x in memcpy), default=0.0)
        if best > 0:
            links.append({
                "source_id": "cpu0", "target_id": "ram0", "kind": "memory-access",
                "measured": True,
                "bandwidth_bytes_per_second": best * 1024**3,
                "latency_microseconds": 0.0,
            })

    dev = cuda.get("device")
    if isinstance(dev, dict):
        nodes.append({
            "id": f"gpu{dev.get('ordinal',0)}", "kind": "accelerator",
            "name": dev.get("name","cuda device"), "backend": "cuda",
            "architecture": f"sm{dev.get('compute_major','?')}{dev.get('compute_minor','?')}",
            "ordinal": dev.get("ordinal",0), "numa_node": -1,
            "total_bytes": dev.get("total_global_memory_bytes",0),
            "available_bytes": dev.get("free_memory_bytes",0),
            "capabilities": [
                x for x, enabled in {
                    "graphs": True,
                    "async-copy-engines": dev.get("async_engine_count",0) > 0,
                    "concurrent-kernels": bool(dev.get("concurrent_kernels",0)),
                    "unified-addressing": bool(dev.get("unified_addressing",0)),
                    "managed-memory": bool(dev.get("managed_memory",0)),
                    "memory-pools": bool(dev.get("memory_pools_supported",0)),
                }.items() if enabled
            ],
        })
        gid = f"gpu{dev.get('ordinal',0)}"
        for direction, bw in (
            ("ram-to-gpu", cuda.get("h2d_bytes_per_second")),
            ("gpu-to-ram", cuda.get("d2h_bytes_per_second")),
        ):
            if isinstance(bw, (int,float)) and bw > 0:
                source, target = ("ram0", gid) if direction == "ram-to-gpu" else (gid, "ram0")
                links.append({
                    "source_id": source, "target_id": target, "kind": "host-device",
                    "measured": True, "bandwidth_bytes_per_second": bw,
                    "latency_microseconds": 0.0,
                })

    if storage:
        nodes.append({
            "id": "storage0", "kind": "storage",
            "name": storage.get("path","model storage"), "backend": "filesystem",
            "architecture": "storage", "ordinal": 0, "numa_node": -1,
            "total_bytes": storage.get("filesystem_total_bytes",0),
            "available_bytes": storage.get("filesystem_available_bytes",0),
            "capabilities": [
                x for x, enabled in {
                    "mmap": storage.get("mmap_supported") is True,
                    "direct-io": storage.get("direct_read",{}).get("rc") == 0,
                }.items() if enabled
            ],
        })
        bw = storage.get("direct_read",{}).get("bytes_per_second") or \
             storage.get("sequential_read_bytes_per_second")
        if isinstance(bw, (int,float)) and bw > 0:
            links.append({
                "source_id": "storage0", "target_id": "ram0", "kind": "storage-host",
                "measured": True, "bandwidth_bytes_per_second": bw,
                "latency_microseconds": 0.0,
            })

    body = {"schema_version": 1, "nodes": nodes, "links": links}
    canonical = json.dumps(body, sort_keys=True, separators=(",",":")).encode()
    body["fingerprint"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__":
    main()
