#!/usr/bin/env python3
import argparse, json, mmap, os, shutil, subprocess, time
from pathlib import Path

def run_text(cmd):
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
        return {"rc": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as e:
        return {"rc": -1, "error": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-mib", type=int, default=256)
    args = ap.parse_args()

    path = args.path.resolve()
    size = path.stat().st_size
    bytes_to_read = min(size, args.max_mib * 1024 * 1024)
    vfs = os.statvfs(path)
    mount = run_text(["findmnt", "-J", "-T", str(path)])
    block = run_text(["lsblk", "-J", "-o",
                      "NAME,PATH,TYPE,SIZE,ROTA,TRAN,MOUNTPOINTS,FSTYPE,MODEL"])

    mmap_ok = False
    try:
        with path.open("rb") as f:
            length = min(4096, size)
            if length > 0:
                mm = mmap.mmap(f.fileno(), length, access=mmap.ACCESS_READ)
                _ = mm[0]
                mm.close()
                mmap_ok = True
    except Exception:
        pass

    start = time.perf_counter()
    total = 0
    with path.open("rb", buffering=0) as f:
        try:
            os.posix_fadvise(f.fileno(), 0, bytes_to_read, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
        while total < bytes_to_read:
            data = f.read(min(4 * 1024 * 1024, bytes_to_read - total))
            if not data:
                break
            total += len(data)
    elapsed = time.perf_counter() - start
    seq_bps = total / elapsed if elapsed > 0 else 0.0

    direct = {"rc": None, "bytes": bytes_to_read, "seconds": None,
              "bytes_per_second": None, "stderr": ""}
    if shutil.which("dd") and bytes_to_read >= 4 * 1024 * 1024:
        count = max(1, bytes_to_read // (4 * 1024 * 1024))
        begin = time.perf_counter()
        p = subprocess.run(
            ["dd", f"if={path}", "of=/dev/null", "bs=4M", f"count={count}",
             "iflag=direct", "status=none"],
            text=True, capture_output=True)
        seconds = time.perf_counter() - begin
        direct = {
            "rc": p.returncode,
            "bytes": count * 4 * 1024 * 1024,
            "seconds": seconds,
            "bytes_per_second":
                (count * 4 * 1024 * 1024 / seconds) if p.returncode == 0 and seconds > 0 else None,
            "stderr": p.stderr.strip(),
        }

    gds = {"status": "unproven"}
    candidates = [
        shutil.which("gdscheck"),
        "/usr/local/cuda/gds/tools/gdscheck.py",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            gds = run_text([candidate, "-p"])
            gds["status"] = "tool-ran"
            break

    out = {
        "schema": "air.prompt1.storage.v1",
        "path": str(path),
        "file_size_bytes": size,
        "probe_bytes": total,
        "filesystem_total_bytes": vfs.f_blocks * vfs.f_frsize,
        "filesystem_available_bytes": vfs.f_bavail * vfs.f_frsize,
        "mmap_supported": mmap_ok,
        "sequential_read_bytes_per_second": seq_bps,
        "direct_read": direct,
        "mount": mount,
        "block_devices": block,
        "direct_storage": gds,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__":
    main()
