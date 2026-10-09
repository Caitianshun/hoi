"""Run one V11 GPU job under the shared 4dsr locks after two idle readings; ledger its wall time.

Usage: python v11_gpu.py --label NAME --budget E3_E0b_GPU_seconds -- script.py [args...]
"""
import argparse
import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from v11_common import ROOT, RUN, append_json, config, jsonl, save_json


def resource_sample():
    hw = config()["hardware"]
    raw = subprocess.check_output(["nvidia-smi", "-i", str(hw["physical_GPU"]),
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits"], text=True, timeout=30).strip()
    index, uuid, name, total, used, util, driver = [x.strip() for x in raw.split(",")]
    assert int(index) == hw["physical_GPU"] and uuid == hw["GPU_uuid"] and name == hw["name"], raw
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=30)
    selected = [r for r in apps.splitlines() if r.split(",")[0].strip() == uuid]
    idle = (not selected and float(used) < hw["maximum_idle_memory_MiB"]
            and float(util) < hw["maximum_idle_utilization_percent"])
    return dict(time_unix=time.time(), uuid=uuid, used_MiB=float(used), utilization_percent=float(util),
                driver=driver, compute_processes=selected, idle=idle)


@contextlib.contextmanager
def gpu_lease(label):
    hw = config()["hardware"]
    start = time.monotonic()
    paths = sorted(Path(x) for x in hw["shared_locks"])
    attempts = 0
    while True:
        handles = []
        try:
            for path in paths:
                assert path.parent.exists(), path
                handle = path.open("a")
                handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            first = resource_sample()
            if not first["idle"]:
                raise BlockingIOError("foreign GPU occupancy")
            time.sleep(hw["idle_sample_interval_seconds"])
            second = resource_sample()
            if not second["idle"]:
                raise BlockingIOError("foreign GPU occupancy")
            append_json(RUN / "protocol/resource_checks.jsonl", dict(label=label, status="two_idle_readings_under_all_shared_locks",
                        readings=[first, second], locks=[str(p) for p in paths], wait_seconds=time.monotonic() - start))
        except BlockingIOError as error:
            for h in handles:
                h.close()
            attempts += 1
            if attempts == 1 or attempts % 120 == 0:
                append_json(RUN / "protocol/resource_checks.jsonl", dict(label=label, status="protected_resource_wait",
                            reason=str(error), wait_seconds=time.monotonic() - start, time_unix=time.time()))
            if time.monotonic() - start >= hw["resource_wait_seconds"]:
                raise RuntimeError("resource wait expired; foreign jobs preserved")
            time.sleep(hw["idle_sample_interval_seconds"])
            continue
        except BaseException:
            for h in handles:
                h.close()
            raise
        try:
            yield tuple(h.fileno() for h in handles)
        finally:
            for h in handles:
                h.close()
        return


def used_seconds(budget):
    return sum(r["wall_seconds"] for r in jsonl(RUN / "protocol/GPU_jobs.jsonl") if r["budget"] == budget)


def run_job(label, budget, script_args):
    hw = config()["hardware"]
    limit = config()["budgets"][budget]
    log_dir = RUN / "logs" / f"{label}_{time.time_ns()}"
    log_dir.mkdir(parents=True, exist_ok=False)
    command = [str(ROOT / hw["python"]), "-u", *script_args]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(hw["physical_GPU"]), OMP_NUM_THREADS=str(hw["CPU_threads"]),
               OPENBLAS_NUM_THREADS=str(hw["CPU_threads"]), PYTHONFAULTHANDLER="1")
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    with gpu_lease(label) as fds:
        left = limit - used_seconds(budget)
        assert left > 0, f"{budget} exhausted"
        assert resource_sample()["idle"]
        start = time.time()
        with (log_dir / "console.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=RUN / "code", env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, pass_fds=fds, start_new_session=True)
            save_json(RUN / "active_job.json", dict(status="running", label=label, pid=proc.pid, command=command,
                                                    start_unix=start, log_dir=str(log_dir)))
            timed_out = False
            try:
                code = proc.wait(timeout=left)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait()
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait()
                raise
        row = dict(label=label, budget=budget, command=command, pid=proc.pid, returncode=code, timed_out=timed_out,
                   start_unix=start, end_unix=time.time(), wall_seconds=time.time() - start, log_dir=str(log_dir),
                   physical_GPU=hw["physical_GPU"], GPU_uuid=hw["GPU_uuid"])
        append_json(RUN / "protocol/GPU_jobs.jsonl", row)
        save_json(RUN / "active_job.json", dict(status="finished", **row))
    return row


if __name__ == "__main__":
    argv = sys.argv[1:]
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--budget", required=True, choices=list(config()["budgets"]))
    args = parser.parse_args(argv[:split])
    result = run_job(args.label, args.budget, argv[split + 1:])
    print(result["label"], "returncode", result["returncode"], "wall", round(result["wall_seconds"], 1))
    sys.exit(0 if result["returncode"] == 0 else 1)
