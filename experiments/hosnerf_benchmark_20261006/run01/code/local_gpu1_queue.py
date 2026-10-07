#!/usr/bin/env python3
"""Operational aliases for two disjoint local queues; no scientific source edits.

Default invocation prepares and checks CPU paths only. --run is the explicit
dispatch entry. Canonical data, checkpoints, flows and evaluation stay in run01;
only locks, logs, state and ledgers live under local_gpu0/local_gpu1.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import pipeline

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
CODE = RUN / 'code'
OWNERS = {'0': ['Backpack', 'Suitcase', 'Playground'], '1': ['Dance', 'Lounge'], 'a100-train': ['Tennis']}
TRAIN_SHA = '10c79ca980eaf31dc1a40ff015e23008d719ba4f6aa3263a67a8fd05ac8efdac'
FLOW_SHA = '4279ae34d91ba946260c3bfe0cfb0e8e0364de2b4011666f7a6219514d548ce8'
CONFIG_SHA = '45e25b050a799f50f2ca59bfd2aaf618ed79e1ab709c5b9dce9c33c513775ad7'
SHARED_GPU1 = Path('/home/cai_tianshun/Project/4dsr/output/dynamic_sr_confidence_geometry_20261006/evaluation_gpu.lock')
EXTERNAL = SHARED_GPU1.parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def scientific_identity():
    assert sha(CODE / 'train_native.py') == TRAIN_SHA
    assert sha(CODE / 'prepare_flow.py') == FLOW_SHA
    assert sha(RUN / 'configs/benchmark.json') == CONFIG_SHA
    return dict(schema=1, owners=OWNERS, artifact_root=str(RUN), data_root=str(RUN / 'data'),
                prescribed_steps=pipeline.STEPS, training_script_sha256=TRAIN_SHA,
                flow_script_sha256=FLOW_SHA, benchmark_config_sha256=CONFIG_SHA,
                dataset_manifest_sha256=sha(RUN / 'protocol/dataset_manifest.json'),
                assignment_kind='Operational GPU partition; original full budgets and inputs retained')


def assignment():
    path = RUN / 'protocol/scene_assignment.json'
    expected = json.loads(json.dumps(scientific_identity()))
    assert read(path) == expected, 'Scene ownership or scientific identity changed'
    scenes = [scene for values in OWNERS.values() for scene in values]
    assert len(scenes) == len(set(scenes)) == 6
    return path


def prepare(gpu):
    identity = json.loads(json.dumps(scientific_identity()))
    frozen = RUN / 'protocol/scene_assignment.json'
    with (RUN / 'scene_assignment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if frozen.exists():
            assert read(frozen) == identity, 'Refusing to replace an existing scene assignment'
        else:
            write(frozen, identity)
    ops = RUN / f'local_gpu{gpu}'
    ops.mkdir(exist_ok=True)
    (ops / 'logs').mkdir(exist_ok=True)
    for name in ['data', 'runs', 'evaluation', 'protocol']:
        link = ops / name
        if link.is_symlink():
            assert link.resolve() == RUN / name, f'Wrong operational alias: {link}'
        else:
            assert not link.exists(), f'Refusing to replace a real operational directory: {link}'
            link.symlink_to(RUN / name, target_is_directory=True)
    preflight = RUN / 'preflight_lr_v2.json'
    assert read(preflight)['status'] == 'passed'
    copy = ops / preflight.name
    if copy.exists():
        assert sha(copy) == sha(preflight)
    else:
        shutil.copy2(preflight, copy)
    acceptance = RUN / 'protocol/recovery_acceptance_lr_v2.json'
    assert read(acceptance)['status'] == 'passed'
    snapshot = ops / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in [CODE / name for name in ['local_gpu1_queue.py', 'pipeline.py', 'train_native.py', 'prepare_flow.py', 'evaluate_native.py', 'summarize.py']]:
        target = snapshot / path.name
        if target.exists():
            assert sha(target) == sha(path), 'Operational source snapshot changed'
        else:
            shutil.copy2(path, target)
    evidence = dict(status='cpu_paths_prepared', gpu=gpu, scenes=OWNERS[gpu], operations_root=str(ops),
                    assignment_sha256=sha(frozen), preflight_sha256=sha(preflight),
                    recovery_acceptance_sha256=sha(acceptance), gpu_tasks_launched=0,
                    canonical_aliases={name: str((ops / name).resolve()) for name in ['data', 'runs', 'evaluation', 'protocol']},
                    sources={path.name: sha(path) for path in snapshot.iterdir()},
                    no_repeated_model_warmup='Reuse the completed same-environment native 8->10 and LR/sampler acceptance; first prescribed segment is the new-GPU acceptance.')
    write(ops / 'deployment_identity.json', evidence)
    return ops


def archive_prior_failure(ops):
    archive = ops / 'prior_root_failure'
    archive.mkdir(exist_ok=True)
    records = []
    for name in ['pipeline.json', 'controller.json']:
        source = RUN / name
        if source.exists() and read(source).get('status') == 'failed':
            target = archive / name
            if target.exists():
                assert sha(target) == sha(source), 'Prior failure was replaced'
            else:
                shutil.copy2(source, target)
            records.append(dict(original=str(source), archived=str(target), sha256=sha(target)))
    journal = archive / 'original_controller_journal.log'
    if not journal.exists():
        result = subprocess.run(['journalctl', '--user', '-u', 'hoi-hosnerf-benchmark-lr-v2-20261006.service', '--no-pager'], capture_output=True, text=True)
        journal.write_text(result.stdout + result.stderr)
    write(archive / 'identity.json', dict(status='original_scheduler_failure_preserved', records=records,
          journal_sha256=sha(journal), native_sha256=TRAIN_SHA, pipeline_sha256=sha(CODE / 'pipeline.py'),
          controller_sha256=sha(CODE / 'controller.py'), classification='Predispatch idle guard rejection after a completed native segment'))


class ProcessAPI:
    """Suppress shared summary writes; original scientific commands are unchanged."""
    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) > 1 and Path(command[1]).resolve() == CODE / 'summarize.py':
            return subprocess.CompletedProcess(command, 0)
        return subprocess.run(command, *args, **kwargs)


def configure(gpu, ops):
    assert pipeline.ROOT == ROOT and pipeline.CODE == CODE
    pipeline.RUN = ops
    pipeline.GPU = gpu
    pipeline.SHARED_LOCK = SHARED_GPU1 if gpu == '1' else RUN / 'gpu0_resource.lock'
    pipeline.subprocess = ProcessAPI()


def external_clearance():
    # Known pending 4dsr endpoint evaluation has priority before the first HOI lease.
    for arm in ['G', 'RG']:
        directory = EXTERNAL / f'evaluation/r2_{arm}'
        receipt = read(directory / 'async_complete.json')
        assert receipt['status'] == 'completed_primary_and_extra', arm
        for key in ['primary', 'extra']:
            item = receipt[key]
            path = Path(item['path'])
            if not path.is_absolute():
                path = EXTERNAL.parents[1] / path
            assert path.resolve().is_relative_to(EXTERNAL) and sha(path) == item['sha256']
        primary = read(directory / 'complete.json')
        assert primary['status'] == 'completed_evaluation' and primary['checkpoint'] == receipt['checkpoint']
        extra = read(directory / 'extra/complete.json')
        assert extra['parameter_updates'] == 0 and extra['observations'] == 196
        assert extra['identity']['checkpoint_sha256'] == receipt['checkpoint']['sha256']


def command_test(gpu, ops):
    configure(gpu, ops)
    captured = []
    spec = importlib.util.spec_from_file_location('native_arguments_cpu_test', CODE / 'train_native.py')
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    def capture(command, label):
        with patch.object(sys, 'argv', [command[2], *command[3:]]):
            parsed = native.arguments()
        assert parsed.data_root == RUN / 'data'
        assert parsed.output == RUN / f'runs/formal/{parsed.scene}/stage{parsed.stage}'
        assert parsed.scene in OWNERS[gpu] and parsed.max_steps == pipeline.STEPS[parsed.stage]
        assert parsed.workers == 0 and parsed.chunk == parsed.netchunk == 0 and not parsed.smoke
        assert parsed.stop_after_updates == 2000 and parsed.no_evaluate
        if parsed.stage == 3:
            assert parsed.background_checkpoint == RUN / f'runs/formal/{parsed.scene}/stage1/final.ckpt'
            assert parsed.human_checkpoint == RUN / f'runs/formal/{parsed.scene}/stage2/final.ckpt'
        captured.append(dict(scene=parsed.scene, stage=parsed.stage, canonical_data_root=str(parsed.data_root),
                             canonical_output=str(parsed.output), prescribed_steps=parsed.max_steps,
                             resume=str(parsed.resume) if parsed.resume else None, command=command))
        return dict(exit_code=0, cpu_capture=True)
    with patch.object(pipeline, 'task', capture), patch.object(pipeline, 'checkpoint', lambda directory: directory / 'final.ckpt'), patch.object(Path, 'mkdir', return_value=None):
        for scene in OWNERS[gpu]:
            for stage in [1, 2, 3]:
                pipeline.train(scene, stage)
    result = dict(status='passed_cpu_command_capture', gpu_tasks_launched=0, parameters_updated=0,
                  artifact_directories_created=0, gpu=gpu, scenes=OWNERS[gpu], captured=captured,
                  resource_lock=str(pipeline.SHARED_LOCK), shared_summary_calls_suppressed=True,
                  assignment_sha256=sha(assignment()), wrapper_sha256=sha(Path(__file__)))
    write(ops / 'cpu_command_capture.json', result)
    print(json.dumps(dict(status=result['status'], gpu=gpu, captured=len(captured), gpu_tasks_launched=0)), flush=True)


def run(args, ops):
    assignment()
    archive_prior_failure(ops)
    configure(args.gpu, ops)
    if args.gpu == '1':
        external_clearance()
    original_task = pipeline.task
    def task(command, label):
        assignment()
        # A hot GPU utilization sample can outlast process exit. Retry only this
        # precise admission failure; child failures and busy owners are untouched.
        for attempt in range(6):
            try:
                return original_task(command, label)
            except RuntimeError as error:
                prefix = f'GPU{args.gpu} not confirmed idle with >=20GB available:'
                if not str(error).startswith(prefix) or pipeline.busy_pids() or attempt == 5:
                    raise
                pipeline.check_deadline()
                write(ops / 'idle_admission_retry.json', dict(status='waiting_for_hot_tail_to_settle',
                      attempt=attempt + 1, error=str(error), task=label, parameter_updates=0, observed_unix=time.time()))
                time.sleep(10)
                pipeline.check_deadline()
    pipeline.task = task
    scene_locks = []
    try:
        (RUN / 'scene_locks').mkdir(exist_ok=True)
        for scene in OWNERS[args.gpu]:
            lock = (RUN / f'scene_locks/{scene}.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            scene_locks.append(lock)
        parameters = argparse.Namespace(gpu=args.gpu, scenes=','.join(OWNERS[args.gpu]),
                     shared_lock=pipeline.SHARED_LOCK, acceptance_tag='lr_v2', preflight=False, cutoff=args.cutoff)
        pipeline.main(parameters)
        final = read(ops / 'pipeline.json')
        write(ops / 'terminal_receipt.json', dict(status=final['status'], gpu=args.gpu,
              scenes=OWNERS[args.gpu], assignment_sha256=sha(assignment()),
              pipeline_state_sha256=sha(ops / 'pipeline.json'), ended_unix=time.time(),
              summary='Deferred until both local queues and the verified A100 return finish'))
    except BaseException:
        write(ops / 'wrapper_failure.json', dict(status='failed', traceback=traceback.format_exc(), updated_unix=time.time()))
        raise
    finally:
        for lock in scene_locks:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=['0', '1'], required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--cpu-command-test', action='store_true')
    parser.add_argument('--cutoff', default='2026-11-04T23:00:00-08:00')
    args = parser.parse_args()
    if args.run and args.cpu_command_test:
        parser.error('CPU capture cannot launch the queue')
    ops = prepare(args.gpu)
    assignment()
    if args.run:
        run(args, ops)
    elif args.cpu_command_test:
        command_test(args.gpu, ops)
    else:
        print(json.dumps(dict(status='cpu_ready', gpu=args.gpu, operations_root=str(ops), gpu_tasks_launched=0)), flush=True)


if __name__ == '__main__':
    main()
