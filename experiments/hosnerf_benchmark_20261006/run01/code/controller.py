#!/usr/bin/env python3
"""Chain native preflight, full-state recovery checks, and formal training.

Waiting on the running preflight uses a process exit event. Checkpoint audits
run on CPU; recovery checks use the same resource lock as the training queue.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback
import pipeline

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
CODE = RUN / "code"
PYTHON = ROOT / "envs/hosnerf/bin/python"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def wait_exit(pid):
    try:
        fd = pipeline.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        select.select([fd], [], [])
    finally:
        os.close(fd)


def audit(path, expected_step):
    import torch
    torch.set_num_threads(4)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["global_step"] == expected_step, path
    identity = payload["benchmark_identity"]
    assert identity["formal"] is False
    assert identity["scene"] == "Backpack"
    assert identity["max_steps"] == {1: 500000, 2: 400000, 3: 200000}[identity["stage"]]
    assert payload["benchmark_sampling"]["committed"] == expected_step
    assert payload["benchmark_sampling"]["epoch_rng"] is not None
    assert set(payload["benchmark_rng"]) == {"python", "numpy", "torch", "cuda"}
    assert payload["benchmark_rng"]["cuda"]
    assert payload["optimizer_states"]
    checks = {"model_floating_tensors": 0, "adam_floating_tensors": 0}
    for tensor in payload["state_dict"].values():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), path
            checks["model_floating_tensors"] += 1
    optimizer_steps = []
    for optimizer in payload["optimizer_states"]:
        assert optimizer["state"]
        for state in optimizer["state"].values():
            assert "exp_avg" in state and "exp_avg_sq" in state and "step" in state
            optimizer_steps.append(int(state["step"]))
            for tensor in state.values():
                if torch.is_tensor(tensor) and tensor.is_floating_point():
                    assert torch.isfinite(tensor).all(), path
                    checks["adam_floating_tensors"] += 1
    checks.update(global_step=payload["global_step"], stage=identity["stage"],
                  checkpoint=str(path), checkpoint_sha256=pipeline.sha(path),
                  sampler_committed=payload["benchmark_sampling"]["committed"],
                  optimizer_step_min=min(optimizer_steps), optimizer_step_max=max(optimizer_steps))
    return checks


def main(args):
    controller_lock = (RUN / "controller.lock").open("a")
    try:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        write(RUN / f"duplicate_controller_{os.getpid()}.json",
              dict(status="rejected", reason="another controller owns the lock", updated_unix=time.time()))
        return
    state = RUN / "controller.json"
    write(state, dict(status="waiting_for_preflight_exit", preflight_pid=args.preflight_pid,
                      updated_unix=time.time()))
    if args.preflight_pid:
        wait_exit(args.preflight_pid)
    if not (RUN / "preflight.json").exists() or read(RUN / "preflight.json")["status"] != "passed":
        raise RuntimeError("The isolated three-stage preflight did not pass; formal training was not started")
    pipeline.DEADLINE = datetime.fromisoformat(args.cutoff).timestamp()
    # Keep the training queue exclusive during the isolated recovery checks.
    with (RUN / "pipeline.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before, after = [], []
        for stage in (1, 2, 3):
            source = RUN / "runs/smoke/Backpack" / f"stage{stage}" / "last.ckpt"
            before.append(audit(source, 8))
            output = RUN / "runs/smoke_resume/Backpack" / f"stage{stage}"
            output.mkdir(parents=True, exist_ok=True)
            done = output / "receipt.json"
            if done.exists() and read(done).get("global_step") == 10:
                after.append(audit(output / "last.ckpt", 10))
                continue
            if (output / "last.ckpt").exists():
                raise RuntimeError(f"Incomplete recovery check requires inspection: {output}")
            command = [str(PYTHON), "-u", str(CODE / "train_native.py"), "--stage", str(stage),
                       "--scene", "Backpack", "--data-root", str(RUN / "data"),
                       "--output", str(output), "--max-steps", str(pipeline.STEPS[stage]),
                       "--checkpoint-every", "2000", "--stop-after-updates", "2",
                       "--workers", "0", "--no-evaluate", "--smoke", "--resume", str(source),
                       "--deadline", str(pipeline.DEADLINE)]
            write(state, dict(status="recovery_check", stage=stage, updated_unix=time.time()))
            pipeline.task(command, f"Backpack_stage{stage}_resume_check")
            receipt = read(done)
            if receipt["segment_updates"] != 2 or receipt["global_step"] != 10 or receipt["formal"]:
                raise RuntimeError(f"Recovery did not advance exactly two isolated updates: {receipt}")
            after.append(audit(output / "last.ckpt", 10))
        write(RUN / "protocol/recovery_acceptance.json", dict(status="passed", before=before, after=after,
              updated_unix=time.time(), note="Real Lightning restore: isolated 8 to 10 updates, full official schedule denominators; no claim of bitwise equivalence to uninterrupted training."))
    write(state, dict(status="formal_pipeline_running", updated_unix=time.time()))
    subprocess.run([sys.executable, "-u", str(CODE / "pipeline.py"), "--cutoff", args.cutoff], cwd=ROOT, check=True)
    final = read(RUN / "pipeline.json")
    write(state, dict(status=final["status"], pipeline=final, updated_unix=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-pid", type=int, default=0)
    parser.add_argument("--cutoff", default="2026-11-04T23:00:00-08:00")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except pipeline.DeadlineReached:
        write(RUN / "controller.json", dict(status="stopped_at_freeze_deadline", updated_unix=time.time()))
    except BaseException:
        write(RUN / "controller.json", dict(status="failed", traceback=traceback.format_exc(), updated_unix=time.time()))
        raise
