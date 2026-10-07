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
import re
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
    suffix = "_" + args.acceptance_tag if args.acceptance_tag else ""
    preflight_path = RUN / ("preflight" + suffix + ".json")
    if not preflight_path.exists() or read(preflight_path)["status"] != "passed":
        raise RuntimeError("The isolated three-stage preflight did not pass; formal training was not started")
    pipeline.DEADLINE = datetime.fromisoformat(args.cutoff).timestamp()
    pipeline.GPU = args.gpu
    if args.shared_lock:
        pipeline.SHARED_LOCK = args.shared_lock.expanduser().resolve()
    elif args.gpu == '0':
        pipeline.SHARED_LOCK = RUN / 'gpu0_resource.lock'
    # Keep the training queue exclusive during the isolated recovery checks.
    with (RUN / "pipeline.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before, after = [], []
        for stage in (1, 2, 3):
            source = RUN / ("runs/smoke" + suffix + "/Backpack") / f"stage{stage}" / "last.ckpt"
            before.append(audit(source, 8))
            subprocess.run([str(PYTHON), str(CODE / "audit_native_lr.py"), "--checkpoint", str(source),
                            "--output", str(RUN / f"protocol/stage{stage}_fresh_lr{suffix}.json")], check=True)
            output = RUN / ("runs/smoke_resume" + suffix + "/Backpack") / f"stage{stage}"
            output.mkdir(parents=True, exist_ok=True)
            done = output / "receipt.json"
            if done.exists() and read(done).get("global_step") == 10:
                after.append(audit(output / "last.ckpt", 10))
            elif (output / "last.ckpt").exists():
                raise RuntimeError(f"Incomplete recovery check requires inspection: {output}")
            else:
                command = [str(PYTHON), "-u", str(CODE / "train_native.py"), "--stage", str(stage),
                       "--scene", "Backpack", "--data-root", str(RUN / "data"),
                       "--output", str(output), "--max-steps", str(pipeline.STEPS[stage]),
                       "--checkpoint-every", "2000", "--stop-after-updates", "2",
                       "--workers", "0", "--no-evaluate", "--smoke", "--resume", str(source),
                       "--deadline", str(pipeline.DEADLINE)]
                write(state, dict(status="recovery_check", stage=stage, acceptance_tag=args.acceptance_tag, updated_unix=time.time()))
                pipeline.task(command, f"Backpack_stage{stage}_resume_check{suffix}")
                receipt = read(done)
                if receipt["segment_updates"] != 2 or receipt["global_step"] != 10 or receipt["formal"]:
                    raise RuntimeError(f"Recovery did not advance exactly two isolated updates: {receipt}")
                after.append(audit(output / "last.ckpt", 10))
            subprocess.run([str(PYTHON), str(CODE / "audit_native_lr.py"), "--checkpoint", str(output / "last.ckpt"),
                            "--output", str(RUN / f"protocol/stage{stage}_resume_lr{suffix}.json")], check=True)
            if stage == 1:
                subprocess.run([str(PYTHON), str(CODE / "audit_stage1_resume.py"), "--before", str(source),
                                "--after", str(output / "last.ckpt"),
                                "--output", str(RUN / ("protocol/stage1_resume_exact_audit" + suffix + ".json"))], check=True)
        write(RUN / ("protocol/recovery_acceptance" + suffix + ".json"), dict(status="passed", before=before, after=after,
              updated_unix=time.time(), note="Real Lightning restore: isolated 8 to 10 updates, full official schedule denominators; no claim of bitwise equivalence to uninterrupted training."))
    hold_path = RUN / "protocol/HOLD_FORMAL.json"
    if hold_path.exists():
        hold = read(hold_path)
        if hold.get("required_acceptance_tag") != args.acceptance_tag or hold.get("repaired_training_script_sha256") != pipeline.sha(CODE / "train_native.py"):
            write(state, dict(status="formal_dispatch_held", hold=hold, updated_unix=time.time()))
            return
        write(RUN / ("protocol/HOLD_FORMAL_RESOLVED" + suffix + ".json"), dict(original_hold=hold,
              resolved_by_acceptance_tag=args.acceptance_tag, resolved_unix=time.time(),
              recovery_acceptance=str(RUN / ("protocol/recovery_acceptance" + suffix + ".json"))))
        hold_path.unlink()
    write(state, dict(status="formal_pipeline_running", updated_unix=time.time()))
    command = [sys.executable, "-u", str(CODE / "pipeline.py"), "--cutoff", args.cutoff,
               "--acceptance-tag", args.acceptance_tag, "--gpu", args.gpu, "--scenes", args.scenes]
    if args.shared_lock:
        command += ["--shared-lock", str(args.shared_lock)]
    subprocess.run(command, cwd=ROOT, check=True)
    final = read(RUN / "pipeline.json")
    write(state, dict(status=final["status"], pipeline=final, updated_unix=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-pid", type=int, default=0)
    parser.add_argument("--cutoff", default="2026-11-04T23:00:00-08:00")
    parser.add_argument("--acceptance-tag", default="")
    parser.add_argument("--gpu", choices=["0", "1"], default="1")
    parser.add_argument("--shared-lock", type=Path)
    parser.add_argument("--scenes", default=','.join(pipeline.SCENES))
    arguments = parser.parse_args()
    if arguments.acceptance_tag and not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", arguments.acceptance_tag):
        parser.error("acceptance-tag must be a short lowercase identifier")
    if len(set(arguments.scenes.split(','))) != len(arguments.scenes.split(',')) or not set(arguments.scenes.split(',')).issubset(pipeline.SCENES):
        parser.error("scenes must be distinct names from the frozen six-scene benchmark")
    try:
        main(arguments)
    except pipeline.DeadlineReached:
        write(RUN / "controller.json", dict(status="stopped_at_freeze_deadline", updated_unix=time.time()))
    except BaseException:
        write(RUN / "controller.json", dict(status="failed", traceback=traceback.format_exc(), updated_unix=time.time()))
        raise
