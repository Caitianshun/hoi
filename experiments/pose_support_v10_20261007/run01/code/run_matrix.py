"""Persistent V10 exit-event controller; no development evaluation before six terminals.

The controller uses CPU only. Every private CUDA child inherits both shared GPU
lease descriptors, so a supervisor failure cannot silently release its lease.
Run in a persistent user service with Restart=no and KillMode=control-group.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def identity(path):
    path = Path(path).resolve(); digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""): digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(asset):
    actual = identity(asset["path"])
    assert actual["sha256"] == asset["sha256"], ("Changed asset", asset)
    if "bytes" in asset: assert actual["bytes"] == asset["bytes"]
    return Path(actual["path"])


def config(): return read(RUN / "configs/v10.json")


def jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def status(phase, **extra):
    save(RUN / "pipeline.json", dict(status="running", phase=phase,
         pid=os.getpid(), time_unix=time.time(), **extra))


def resource_sample():
    hw = config()["hardware"]
    raw = subprocess.check_output(["nvidia-smi", "-i", str(hw["physical_GPU"]),
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits"], text=True, timeout=30).strip()
    index, uuid, name, total, used, utilization, driver = [x.strip() for x in raw.split(",")]
    assert int(index) == hw["physical_GPU"] and uuid == hw["GPU_uuid"] and name == hw["name"], raw
    processes = subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=30)
    selected = [r for r in processes.splitlines() if r.split(",")[0].strip() == uuid]
    # On this selected GPU even a zero-utilization compute process is protected.
    idle = not selected and float(used) < hw["maximum_idle_memory_MiB"] and float(utilization) < hw["maximum_idle_utilization_percent"]
    return dict(time_unix=time.time(), index=int(index), uuid=uuid, name=name,
        total_MiB=float(total), used_MiB=float(used), utilization_percent=float(utilization),
        driver=driver, compute_processes=selected, load=list(os.getloadavg()), idle=idle)


@contextlib.contextmanager
def gpu_lease(label):
    """Acquire all locks without holding one while waiting for another."""
    hw = config()["hardware"]; start = time.monotonic(); attempts = 0
    paths = sorted(Path(x) for x in hw["shared_locks"])
    while True:
        handles = []
        try:
            for path in paths:
                assert path.parent.exists(), ("Unknown shared lock directory", path)
                handle = path.open("a"); handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            first = resource_sample()
            if not first["idle"]: raise BlockingIOError("Foreign GPU occupancy")
            time.sleep(hw["idle_sample_interval_seconds"])
            second = resource_sample()
            if not second["idle"]: raise BlockingIOError("Foreign GPU occupancy")
            append(RUN / "protocol/resource_checks.jsonl", dict(label=label,
                status="two_idle_readings_under_all_shared_locks", readings=[first, second],
                locks=[str(x) for x in paths], CPU_wait_seconds=time.monotonic()-start))
        except BlockingIOError as error:
            for h in handles: h.close()
            attempts += 1
            if attempts == 1 or attempts % 120 == 0:
                append(RUN / "protocol/resource_checks.jsonl", dict(label=label,
                    status="protected_resource_wait", reason=str(error), locks=[str(x) for x in paths],
                    wait_seconds=time.monotonic()-start, time_unix=time.time()))
            status(label, resource="waiting_for_shared_locks_and_double_idle", resource_wait_seconds=time.monotonic()-start)
            if time.monotonic()-start >= hw["resource_wait_seconds"]:
                raise RuntimeError("Resource wait expired; foreign jobs preserved")
            time.sleep(hw["idle_sample_interval_seconds"])
            continue
        except BaseException:
            for h in handles: h.close()
            raise
        # Never interpret an exception from our yielded CUDA job as a resource
        # acquisition retry; the caller must cost and preserve that failed job.
        try: yield tuple(h.fileno() for h in handles)
        finally:
            for h in handles: h.close()
        return


def budget_remaining():
    used = sum(r["wall_seconds"] for r in jsonl(RUN / "protocol/GPU_jobs.jsonl"))
    # Acceptance and preparation run outside the controller still count.
    used += read(RUN / "protocol/external_GPU_costs.json").get("GPU_seconds", 0) if (RUN / "protocol/external_GPU_costs.json").exists() else 0
    return config()["budgets"]["GPU_seconds"] - used


def job(label, script, args=(), gpu=True):
    status(label)
    directory = RUN / "logs" / (label + "_" + str(time.time_ns()))
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(ROOT / config()["hardware"]["python"]), "-u", str(RUN / "code" / script), *map(str, args)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config()["hardware"]["physical_GPU"]) if gpu else "",
               OMP_NUM_THREADS=str(config()["hardware"]["CPU_threads"]), OPENBLAS_NUM_THREADS=str(config()["hardware"]["CPU_threads"]),
               PYTHONFAULTHANDLER="1")
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    context = gpu_lease(label) if gpu else contextlib.nullcontext(())
    with context as fds:
        if gpu:
            left = budget_remaining()
            assert left > 0, "GPU task budget exhausted"
            # A final ownership sample happens immediately before child launch.
            assert resource_sample()["idle"]
        else: left = None
        start = time.time(); proc = None; timed_out = False
        with (directory / "console.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, pass_fds=fds, start_new_session=True)
            save(directory / "running.json", dict(pid=proc.pid, command=command,
                    start_unix=start, lock_paths=config()["hardware"]["shared_locks"] if gpu else []))
            save(RUN / "active_job.json", dict(status="running", pid=proc.pid,
                    command=command, start_unix=start, log_dir=str(directory), GPU=gpu))
            try: code = proc.wait(timeout=left)
            except subprocess.TimeoutExpired:
                timed_out = True; os.killpg(proc.pid, signal.SIGTERM)
                try: code = proc.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); code = proc.wait()
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                raise
        row = dict(label=label, command=command, pid=proc.pid, returncode=code,
                start_unix=start, end_unix=time.time(), wall_seconds=time.time()-start,
                physical_GPU=config()["hardware"]["physical_GPU"] if gpu else None,
                GPU_uuid=config()["hardware"]["GPU_uuid"] if gpu else None,
                budget_timeout=timed_out, log_dir=str(directory))
        append(RUN / "protocol" / ("GPU_jobs.jsonl" if gpu else "CPU_jobs.jsonl"), row)
        save(directory / "attempt.json", row); save(RUN / "active_job.json", dict(status="exited", **row))
        if timed_out: raise RuntimeError("GPU budget expired; preserved failed process; no automatic restart")
        return row


def completed_run(scene, arm):
    directory = RUN / "scenes" / scene / "runs" / arm
    path = directory / "run.json"
    if not path.exists(): return None
    result = read(path)
    if result.get("status") != "completed": return None
    assert result["completed_updates"] == config()["schedule"]["updates"]
    bound(result["checkpoint"])
    effective_path = directory / "effective_config.json"
    if effective_path.exists():
        effective = read(effective_path)
        if "input" in effective and "config" in effective["input"]:
            assert effective["input"]["config"]["sha256"] == identity(RUN / "configs/v10.json")["sha256"]
    return result


def recovery_state(scene, arm, failed_job):
    directory = RUN / "scenes" / scene / "runs" / arm
    assert not (directory / "controller_recovery_receipt.json").exists(), "This arm already consumed its one recovery"
    assert not (directory / "resume_receipt.json").exists(), "Trainer already consumed this arm's recovery"
    record = read(directory / "latest_checkpoint.json")
    checkpoint = record.get("checkpoint", record)
    bound(checkpoint)
    saved = record.get("completed_updates")
    if saved is None: raise ValueError("latest_checkpoint.json must expose completed_updates")
    progress = read(directory / "attempt_cursor.json")
    frontier = progress["attempted_iteration"]
    replay = frontier - saved
    assert 0 <= replay <= config()["budgets"]["replay_updates_per_job"], ("Recovery replay budget exceeded", replay)
    attempts = [r for r in jsonl(RUN / "protocol/formal_attempts.jsonl") if r["scene"] == scene and r["arm"] == arm]
    confirmed = len(jsonl(directory / "training_metrics.jsonl"))
    unconfirmed = len(attempts) - confirmed
    assert unconfirmed >= 0
    receipt = dict(status="one_technical_recovery_consumed", scene=scene, arm=arm,
        checkpoint=checkpoint, resume_from_updates=saved, attempted_frontier=frontier,
        replay_attempts=replay, failed_job=failed_job, consumed_time_unix=time.time(),
        confirmed_logged_Adam_updates=confirmed, unconfirmed_attempt_outcomes=unconfirmed,
        root_cause="unknown unless separately established; recovery is not a root-cause fix")
    save(directory / "controller_recovery_receipt.json", receipt)
    append(RUN / "protocol/recovery_incidents.jsonl", receipt)
    return checkpoint["path"]


def train():
    acceptance = read(RUN / "protocol/module_acceptance.json")
    assert acceptance["status"] == "passed"
    assert acceptance.get("diagnostic_Adam_updates", acceptance.get("actual_Adam_updates", 0)) <= config()["budgets"]["integrated_Adam_updates"]
    assert acceptance.get("extra_no_update_backwards", 0) <= config()["budgets"]["extra_no_update_backwards"]
    assert read(RUN / "protocol/cost_prediction.json")["status"] == "within_budget"
    assert len(jsonl(RUN / "protocol/formal_attempts.jsonl")) <= config()["budgets"]["formal_attempts"]
    if (RUN / "protocol/terminal_freeze.json").exists(): verify_freeze(); return
    for scene in config()["scenes"]:
        for arm in config()["arms"]:
            if completed_run(scene, arm): continue
            directory = RUN / "scenes" / scene / "runs" / arm
            command = ["--scene", scene, "--arm", arm, "--purpose", "formal", "--target", str(config()["schedule"]["updates"])]
            # An unclosed prior process must be resolved and costed explicitly.
            if directory.exists() and (directory / "latest_checkpoint.json").exists():
                prior = [r for r in jsonl(RUN / "protocol/GPU_jobs.jsonl") if r["label"].startswith(scene + "_" + arm + "_")]
                assert prior and prior[-1]["returncode"] != 0, "Interrupted state lacks a closed failed-job receipt"
                command += ["--resume", recovery_state(scene, arm, prior[-1])]
                result = job(scene + "_" + arm + "_resume1", "train_scene.py", command)
            else:
                result = job(scene + "_" + arm + "_formal", "train_scene.py", command)
                if result["returncode"] != 0:
                    checkpoint = recovery_state(scene, arm, result)
                    result = job(scene + "_" + arm + "_resume1", "train_scene.py", command + ["--resume", checkpoint])
            assert result["returncode"] == 0, ("Second failure: stop this arm and preserve evidence", scene, arm, result)
            assert completed_run(scene, arm)
            assert len(jsonl(RUN / "protocol/formal_attempts.jsonl")) <= config()["budgets"]["formal_attempts"]
    freeze()


def freeze():
    runs = []; assets = [identity(RUN / "configs/v10.json")]
    for scene, cfg in config()["scenes"].items():
        assets.append(identity(ROOT / cfg["parent_checkpoint"]))
        for arm in config()["arms"]:
            result = completed_run(scene, arm); assert result
            runs.append(dict(scene=scene, arm=arm, mode=arm, checkpoint=result["checkpoint"]))
            assets += [result["checkpoint"], identity(RUN / "scenes" / scene / "runs" / arm / "run.json")]
            effective = RUN / "scenes" / scene / "runs" / arm / "effective_config.json"
            if effective.exists(): assets.append(identity(effective))
    save(RUN / "protocol/terminal_freeze.json", dict(status="all_six_terminals_frozen", runs=runs,
        assets=assets, freeze_time_unix=time.time(), development_evaluation_started=False,
        parent_selection="Backpack V9 Q0; Tennis V8 Q; previously developed scene-level selection"))


def verify_freeze():
    value = read(RUN / "protocol/terminal_freeze.json")
    assert value["status"] == "all_six_terminals_frozen" and len(value["runs"]) == 6
    assert {(r["scene"], r.get("arm", r.get("mode"))) for r in value["runs"]} == {(s, a) for s in config()["scenes"] for a in config()["arms"]}
    for asset in value["assets"]: bound(asset)
    return value


def evaluate():
    verify_freeze()
    rendered = RUN / "protocol/all_rendered.json"
    if rendered.exists():
        value = read(rendered); assert value["status"] == "completed"
        assert value["freeze"]["sha256"] == identity(RUN / "protocol/terminal_freeze.json")["sha256"]
        for asset in value.get("assets", []): bound(asset)
    else:
        result = job("all_terminal_render", "render_scene.py", ["--all"])
        assert result["returncode"] == 0
        manifests = [identity(RUN / "evaluation" / scene / arm / "manifest.json") for scene in config()["scenes"] for arm in ["PARENT", *config()["arms"]]]
        save(rendered, dict(status="completed", freeze=identity(RUN / "protocol/terminal_freeze.json"), assets=manifests))
    if not (RUN / "evaluation_summary.json").exists():
        assert job("all_terminal_evaluation", "evaluate.py", gpu=False)["returncode"] == 0
    if not (RUN / "protocol/independent_verification.json").exists():
        assert job("independent_verification", "evaluate.py", ["--verify"], gpu=False)["returncode"] == 0


def report():
    assert read(RUN / "protocol/independent_verification.json")["status"] == "passed"
    assert job("prepare_report_inputs", "build_report.py", ["--prepare"], gpu=False)["returncode"] == 0
    save(RUN / "pipeline.json", dict(status="evaluated_pending_visual_review_and_document_QA",
        phase="report_review", pid=os.getpid(), time_unix=time.time(),
        pending=["Review all fixed figures", "Record scene-level complete-vector decision", "Author DOCX and inspect every rendered page", "Verify private feedback ZIP"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "evaluate", "report", "all", "train-to-report", "freeze"], default="train-to-report")
    args = parser.parse_args()
    with (RUN / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            phases = ["train", "evaluate", "report"] if args.phase in ["all", "train-to-report"] else [args.phase]
            for phase in phases: globals()[phase]()
            if phases[-1] != "report":
                save(RUN / "pipeline.json", dict(status="phase_completed", phase=phases[-1], pid=os.getpid(), time_unix=time.time()))
        except BaseException:
            save(RUN / "pipeline.json", dict(status="failed", requested_phase=args.phase,
                 pid=os.getpid(), time_unix=time.time(), traceback=traceback.format_exc()))
            raise


if __name__ == "__main__": main()
