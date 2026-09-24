#!/usr/bin/env python3
"""Execute one explicitly supplied command and record wall time, CPU and exit status.

Does not select GPUs, inspect/schedule remote hosts, or estimate GPU hours.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", required=True, help="preprocess/train/evaluate/etc.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--metadata", type=Path, help="optional nonsecret JSON: commit/config/data split/GPU identity")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("supply command after --")
    if a.output.exists():
        p.error("refusing to overwrite an existing run record")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(a.metadata.read_text()) if a.metadata else {}
    record = {"schema_version": 1, "run_id": a.run_id, "stage": a.stage,
              "command": command, "cwd": os.getcwd(), "host": socket.gethostname(),
              "start_utc": stamp(), "status": "running", "pid": os.getpid(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "metadata": metadata, "timer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "resource_notes": "CPU/RSS from RUSAGE_CHILDREN; peak RSS is largest child, not aggregate; GPU time/memory not measured"}
    def save():
        temporary = a.output.with_suffix(a.output.suffix+".tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
        temporary.replace(a.output)
    save(); start = time.perf_counter(); before = resource.getrusage(resource.RUSAGE_CHILDREN)
    rc = 127
    try:
        # Inherit stdout/stderr so caller retains standard logging; never invoke a shell.
        rc = subprocess.call(command)
        record.update(status="completed" if rc == 0 else "failed", exit_code=rc)
    except KeyboardInterrupt:
        record.update(status="interrupted", exit_code=130); rc = 130
    except Exception as exc:
        record.update(status="failed_to_launch", exit_code=127, error=f"{type(exc).__name__}: {exc}")
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(end_utc=stamp(), wall_seconds=time.perf_counter()-start,
                      cpu_user_seconds=after.ru_utime-before.ru_utime,
                      cpu_system_seconds=after.ru_stime-before.ru_stime,
                      max_child_rss_kib=after.ru_maxrss)
        save()
    return rc if rc >= 0 else 128-rc


if __name__ == "__main__":
    raise SystemExit(main())
