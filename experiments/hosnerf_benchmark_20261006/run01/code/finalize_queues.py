#!/usr/bin/env python3
"""CPU exit-event coordinator for the two local queues and A100 return watcher.

No optimizer or renderer is started. Default invocation runs isolated CPU
fixtures; --capture-targets records live process identities; --run waits on
those exact processes and verifies all six endpoints before final summarizing.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
import traceback

RUN = Path(__file__).resolve().parents[1]
OPS = RUN / 'finalize_queues'
SCENES = ['Backpack', 'Tennis', 'Suitcase', 'Playground', 'Dance', 'Lounge']
STEPS = {1: 500000, 2: 400000, 3: 200000}
OWNERS = {'local_gpu0': ['Backpack', 'Suitcase', 'Playground'],
          'local_gpu1': ['Dance', 'Lounge'], 'a100_return': ['Tennis']}
TRAIN_SHA = '10c79ca980eaf31dc1a40ff015e23008d719ba4f6aa3263a67a8fd05ac8efdac'
CONFIG_SHA = '45e25b050a799f50f2ca59bfd2aaf618ed79e1ab709c5b9dce9c33c513775ad7'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def start_ticks(pid):
    # comm may contain spaces or closing parentheses; fields after its final
    # closing parenthesis begin at field 3, and starttime is field 22.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[19])


def pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid, 0)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    descriptor = libc.pidfd_open(pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def capture(args):
    targets = {}
    for name, pid in [('local_gpu0', args.local0_pid), ('local_gpu1', args.local1_pid),
                      ('a100_return', args.return_pid)]:
        if pid <= 0:
            raise ValueError('All three live worker PIDs are required for capture')
        tick = start_ticks(pid)
        descriptor = pidfd(pid)
        try:
            if tick != start_ticks(pid):
                raise RuntimeError('Process identity changed during capture')
        finally:
            os.close(descriptor)
        targets[name] = dict(pid=pid, start_ticks=tick, boot_id=boot())
    if len({item['pid'] for item in targets.values()}) != 3:
        raise ValueError('The three worker PIDs must be distinct')
    value = dict(schema=1, captured_unix=time.time(), owners=OWNERS, targets=targets,
                 coordinator_sha256=sha(__file__), assignment_sha256=sha(RUN / 'protocol/scene_assignment.json'))
    if args.targets.exists():
        raise RuntimeError('Refusing to replace an existing process attachment')
    write(args.targets, value)
    print(json.dumps(dict(status='live_cpu_targets_captured', targets=str(args.targets), gpu_tasks_launched=0)), flush=True)


def wait_targets(targets, event_sink):
    pending = {}
    observations = {}
    with selectors.DefaultSelector() as selector:
        try:
            for name, expected in targets.items():
                if expected['boot_id'] != boot():
                    raise RuntimeError('Host rebooted; original worker exit cannot be inferred from a new boot')
                pid = int(expected['pid'])
                try:
                    actual = start_ticks(pid)
                    if actual != int(expected['start_ticks']):
                        observations[name] = dict(event='original_pid_already_gone_reused', expected=expected)
                        continue
                    descriptor = pidfd(pid)
                    try:
                        if start_ticks(pid) != int(expected['start_ticks']):
                            raise RuntimeError('Process identity changed during attachment')
                    except FileNotFoundError:
                        # A valid pidfd still identifies the process that exited
                        # between opening the descriptor and the final check.
                        pass
                    selector.register(descriptor, selectors.EVENT_READ, name)
                    pending[name] = descriptor
                except ProcessLookupError:
                    observations[name] = dict(event='original_pid_already_exited', expected=expected)
                except FileNotFoundError:
                    observations[name] = dict(event='original_pid_already_exited', expected=expected)
            event_sink(dict(status='waiting_for_exact_worker_exit_events',
                            attached=list(pending), exited=observations, gpu_tasks_launched=0))
            while pending:
                # Completion is the kernel pidfd exit event, not a timer or a
                # metric-file poll. An independent heartbeat handles anomalies.
                for key, _ in selector.select():
                    name = key.data
                    selector.unregister(key.fd)
                    os.close(key.fd)
                    del pending[name]
                    observations[name] = dict(event='pidfd_exit', expected=targets[name], observed_unix=time.time())
                    event_sink(dict(status='waiting_for_exact_worker_exit_events',
                                    attached=list(pending), exited=observations, gpu_tasks_launched=0))
        finally:
            for descriptor in pending.values():
                os.close(descriptor)
    return observations


def child_terminal(run, name):
    directory = run / name
    if name == 'a100_return':
        path = directory / 'watcher.json'
        state = read(path) if path.exists() else {}
        good = state.get('status') == 'completed_returned_and_unified_evaluation_verified'
        if good:
            receipt = read(directory / 'unified_evaluation_receipt.json')
            good = receipt['status'] == 'completed' and receipt['metrics_sha256'] == sha(run / 'evaluation/Tennis/metrics.json')
    else:
        path = directory / 'terminal_receipt.json'
        state = read(path) if path.exists() else {}
        good = state.get('status') == 'complete' and state.get('scenes') == OWNERS[name]
        if good:
            good = state['pipeline_state_sha256'] == sha(directory / 'pipeline.json')
            good = good and state['assignment_sha256'] == sha(run / 'protocol/scene_assignment.json')
    evidence = dict(name=name, accepted_success=good, terminal_state=state, source=str(path))
    for filename in ['wrapper_failure.json', 'watcher_failure.json']:
        path = directory / filename
        if path.exists():
            evidence[filename] = read(path)
    return evidence


def tennis_source_link(run):
    target = run / 'a100_return/remote_snapshot/runs/formal/Tennis'
    source = run / 'runs/formal/Tennis'
    if not target.is_dir():
        return dict(status='no_returned_tennis_stage_directory', target=str(target))
    if source.is_symlink():
        if source.resolve() != target.resolve():
            raise RuntimeError('Existing Tennis source link points to a different experiment')
    elif source.exists():
        raise RuntimeError('Refusing to replace a real Tennis training directory')
    else:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.symlink_to(target, target_is_directory=True)
    return dict(status='read_only_source_index_link', path=str(source), target=str(target),
                mutation_policy='Coordinator reads returned sources; it never modifies or trains this directory',
                original_remote_root='/home/ubuntu/3DGS/HOI/experiments/hosnerf_benchmark_20261006/run01/runs/formal/Tennis',
                returned_provenance=str(run / 'a100_return/return_receipt.json'))


def scene_endpoint(run, scene, manifest):
    result = dict(scene=scene, status='awaiting_endpoint', verified=False)
    try:
        path = run / f'evaluation/{scene}/metrics.json'
        if not path.exists():
            return result
        metrics = read(path)
        identity = metrics['identity']
        native = identity['checkpoint_benchmark_identity']
        expected = manifest['scenes'][scene]
        assert metrics['status'] == 'completed' and metrics['source_kind'] == 'local_retrained'
        assert metrics['scene'] == scene and metrics['frames'] == 16
        assert len(metrics['per_frame']) == 16 and len(metrics['artifacts']) == 48
        assert identity['strict_load'] and identity['global_step'] == 200000
        assert identity['test_ids'] == expected['test_ids']
        assert [row['frame_id'] for row in metrics['per_frame']] == expected['test_ids']
        assert native['formal'] and native['stage'] == 3 and native['scene'] == scene
        assert native['seed'] == 777 and native['max_steps'] == 200000
        assert native['training_script_sha256'] == TRAIN_SHA and native['benchmark_config_sha256'] == CONFIG_SHA
        checkpoints = []
        for stage, budget in STEPS.items():
            directory = run / f'runs/formal/{scene}/stage{stage}'
            receipt = read(directory / 'receipt.json')
            stage_identity = read(directory / 'identity.json')
            assert receipt['status'] == 'completed' and receipt['complete'] and receipt['formal']
            assert receipt['global_step'] == budget and receipt['stage'] == stage
            assert stage_identity['scene'] == scene and stage_identity['stage'] == stage and stage_identity['formal']
            assert stage_identity['max_steps'] == budget and stage_identity['training_script_sha256'] == TRAIN_SHA
            checkpoint = directory / 'final.ckpt'
            checkpoints.append(dict(stage=stage, checkpoint=str(checkpoint), sha256=sha(checkpoint)))
        assert checkpoints[-1]['sha256'] == identity['checkpoint_sha256']
        assert Path(identity['checkpoint']).resolve() == Path(checkpoints[-1]['checkpoint']).resolve()
        data = Path(identity['data_path'])
        allowed_data = run / ('a100_return/local_data/Tennis' if scene == 'Tennis' else f'data/{scene}')
        assert data.resolve() == allowed_data.resolve()
        for resource in identity['test_resources']:
            relative = Path(resource['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            assert sha(data / relative) == resource['sha256']
        expected_artifacts = {f'{folder}/{frame}{suffix}' for frame in expected['test_ids']
                              for folder, suffix in [('floats', '.npz'), ('images', '.png'), ('gt', '.png')]}
        assert {row['path'] for row in metrics['artifacts']} == expected_artifacts
        for artifact in metrics['artifacts']:
            assert sha(path.parent / artifact['path']) == artifact['sha256']
        for key in ['PSNR', 'SSIM', 'LPIPS']:
            values = [row[key] for row in metrics['per_frame']]
            assert all(math.isfinite(value) for value in values)
            assert math.isclose(sum(values) / 16, metrics['mean_metrics'][key], rel_tol=1e-7, abs_tol=1e-9)
        result.update(status='completed_verified', verified=True, metrics=str(path), metrics_sha256=sha(path),
                      checkpoint_index=checkpoints, metric_implementation=identity.get('evaluator_sha256'))
    except Exception:
        result.update(status='endpoint_integrity_failed', error=traceback.format_exc())
    return result


def finalize(run):
    link = tennis_source_link(run)
    manifest = read(run / 'protocol/dataset_manifest.json')
    terminals = [child_terminal(run, name) for name in OWNERS]
    scenes = [scene_endpoint(run, scene, manifest) for scene in SCENES]
    count = sum(item['verified'] for item in scenes)
    good = count == 6 and all(item['accepted_success'] for item in terminals)
    return dict(status='completed_all_six_verified' if good else 'terminal_incomplete_or_failed',
                completed_scenes=count, required_scenes=6, child_terminals=terminals,
                scenes=scenes, tennis_source_index=link, gpu_tasks_launched=0,
                automatic_retraining=False, completed_unix=time.time(),
                evaluation_scope='All six use the same local environment and evaluator; physical local evaluation GPUs 0 and 1 vary. A100 training hardware remains distinct.')


def cpu_check():
    with tempfile.TemporaryDirectory(prefix='hos_finalizer_cpu_') as temporary:
        temp = Path(temporary)
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.1)'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        events = []
        targets = dict(fixture=dict(pid=child.pid, start_ticks=start_ticks(child.pid), boot_id=boot()))
        exits = wait_targets(targets, events.append)
        child.wait()
        assert exits['fixture']['event'] == 'pidfd_exit'
        run = temp / 'run'
        write(run / 'protocol/scene_assignment.json', {'fixture': True})
        for name in ['local_gpu0', 'local_gpu1']:
            write(run / name / 'pipeline.json', dict(status='complete'))
            write(run / name / 'terminal_receipt.json', dict(status='complete', scenes=OWNERS[name],
                  pipeline_state_sha256=sha(run / name / 'pipeline.json'), assignment_sha256=sha(run / 'protocol/scene_assignment.json')))
            assert child_terminal(run, name)['accepted_success']
        write(run / 'local_gpu1/terminal_receipt.json', dict(status='failed', scenes=OWNERS['local_gpu1']))
        assert not child_terminal(run, 'local_gpu1')['accepted_success']
        missing = scene_endpoint(run, 'Backpack', {'scenes': {'Backpack': {}}})
        assert missing['status'] == 'awaiting_endpoint' and not missing['verified']
        target = run / 'a100_return/remote_snapshot/runs/formal/Tennis'
        target.mkdir(parents=True)
        assert tennis_source_link(run)['status'] == 'read_only_source_index_link'
        assert tennis_source_link(run)['status'] == 'read_only_source_index_link'
    value = dict(status='passed_cpu_fixtures', actual_pidfd_exit=True, terminal_success_and_failure_checked=True,
                 missing_endpoint_not_completed=True, strict_tennis_link_checked=True,
                 gpu_tasks_launched=0, parameters_updated=0, coordinator_sha256=sha(__file__), checked_unix=time.time())
    write(OPS / 'cpu_check.json', value)
    print(json.dumps(value), flush=True)


def main(args):
    if args.capture_targets:
        capture(args)
        return
    if not args.run:
        cpu_check()
        return
    OPS.mkdir(exist_ok=True)
    with (OPS / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attachment = read(args.targets)
        assert attachment['owners'] == OWNERS and set(attachment['targets']) == set(OWNERS)
        assert attachment['coordinator_sha256'] == sha(__file__)
        assert attachment['assignment_sha256'] == sha(RUN / 'protocol/scene_assignment.json')
        def sink(value):
            write(OPS / 'state.json', dict(value, targets=str(args.targets), targets_sha256=sha(args.targets)))
        exits = wait_targets(attachment['targets'], sink)
        write(OPS / 'exit_events.json', exits)
        result = finalize(RUN)
        write(OPS / 'terminal_receipt.json', result)
        # The three writers have exited. This is the only final summary write;
        # no simultaneous summary.json.tmp writes can occur from these queues.
        subprocess.run([sys.executable, str(RUN / 'code/summarize.py')], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        summary = read(RUN / 'summary.json')
        result['summary_sha256'] = sha(RUN / 'summary.json')
        result['summary_completed_scenes'] = summary['completed_scenes']
        assert summary['completed_scenes'] == result['completed_scenes']
        write(OPS / 'terminal_receipt.json', result)
        sink(dict(status=result['status'], completed_scenes=result['completed_scenes'], receipt=str(OPS / 'terminal_receipt.json')))
        print('HOS_ALL_QUEUES_TERMINAL', json.dumps(dict(status=result['status'], completed_scenes=result['completed_scenes'])), flush=True)
        if result['status'] != 'completed_all_six_verified':
            raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--capture-targets', action='store_true')
    parser.add_argument('--targets', type=Path, default=OPS / 'targets.json')
    parser.add_argument('--local0-pid', type=int, default=0)
    parser.add_argument('--local1-pid', type=int, default=0)
    parser.add_argument('--return-pid', type=int, default=0)
    args = parser.parse_args()
    try:
        main(args)
    except SystemExit:
        raise
    except BaseException:
        write(OPS / 'coordinator_failure.json', dict(status='failed', error=traceback.format_exc(), observed_unix=time.time(),
              gpu_tasks_launched=0, automatic_retraining=False))
        raise
