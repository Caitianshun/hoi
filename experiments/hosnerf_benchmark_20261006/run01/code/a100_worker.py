#!/usr/bin/env python3
"""Isolated A100 GPU1 Tennis queue; default invocation performs CPU checks only.

The three pending 4dsr r2 module runs have priority. Their training receipts,
checkpoint hashes, user-service exits and processes must pass before this queue
can bind GPU1. Independent HOI locks cannot constrain an unrelated dispatcher;
the coordinator may write a100_worker/HOLD_DISPATCH.json between segments.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time
import traceback

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
CODE = RUN / 'code'
STATE = RUN / 'a100_worker'
PYTHON = ROOT / 'envs/hosnerf/bin/python'
OFFICIAL = ROOT / 'third_party/HOSNeRF'
DATA = RUN / 'data'
REMOTE_ROOT = Path('/home/ubuntu/3DGS/HOI')
EXTERNAL_ROOT = Path('/home/ubuntu/3DGS/4dsr')
EXTERNAL_RUN = EXTERNAL_ROOT / 'output/dynamic_sr_confidence_geometry_20261006'
SCENE = 'Tennis'
GPU = '1'
STEPS = {1: 500000, 2: 400000, 3: 200000}
STAGE_NAMES = {1: '1st_State-Conditional_Scene', 2: '2nd_State_Conditional_Human-Object', 3: '3rd_Complete_HOSNeRF'}
TRAIN_SHA = '10c79ca980eaf31dc1a40ff015e23008d719ba4f6aa3263a67a8fd05ac8efdac'
FLOW_SHA = '4279ae34d91ba946260c3bfe0cfb0e8e0364de2b4011666f7a6219514d548ce8'
CONFIG_SHA = '45e25b050a799f50f2ca59bfd2aaf618ed79e1ab709c5b9dce9c33c513775ad7'
OFFICIAL_REVISION = 'ca169704a9d965e8266c034cceb506ffba72f405'
RAFT_SHA = 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1'


class DeadlineReached(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.pidfd_open
    function.argtypes, function.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.set_inheritable(descriptor, False)
    return descriptor


def frozen_identity():
    assert sha(CODE / 'train_native.py') == TRAIN_SHA, 'Unapproved native harness revision'
    assert sha(CODE / 'prepare_flow.py') == FLOW_SHA, 'Frozen flow source changed'
    assert sha(RUN / 'configs/benchmark.json') == CONFIG_SHA, 'Frozen scientific configuration changed'
    assert sha(ROOT / 'models/RAFT/raft-things.pth') == RAFT_SHA, 'RAFT weight changed'
    revision = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == OFFICIAL_REVISION
    names = subprocess.check_output(['git', '-C', str(OFFICIAL), 'ls-files', '-z']).decode().split('\0')
    sources = ['a100_worker.py', 'train_native.py', 'prepare_flow.py', 'evaluate_native.py',
               'audit_native_lr.py', 'audit_stage1_resume.py', 'prepare_data.py']
    return dict(schema=1, scene=SCENE, physical_gpu=GPU, official_revision=revision,
        sources={name: sha(CODE / name) for name in sources}, benchmark_config_sha256=CONFIG_SHA,
        official_files={name: sha(OFFICIAL / name) for name in names if name},
        dataset_manifest_sha256=sha(RUN / 'protocol/dataset_manifest.json'),
        raft_sha256=RAFT_SHA,
        raft_sources={str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / 'third_party/MoSca/lib_prior/optical_flow/RAFT').rglob('*.py'))},
        prescribed_steps=STEPS, workers=0, chunk=0, netchunk=0, seed=777,
        smoke='Separate 8-step checkpoints and continuation to10, no metric selection',
        formal='From scratch; full official denominators;2000 new updates per resource lease')


def verify_identity(path, full_data=False):
    expected = read(path)
    actual = frozen_identity()
    # JSON serializes integer stage keys to strings.
    assert expected == json.loads(json.dumps(actual)), 'Frozen worker/source identity changed'
    if full_data:
        scene = read(RUN / 'protocol/dataset_manifest.json')['scenes'][SCENE]
        assert len(scene['train_ids']) == 285 and len(scene['test_ids']) == 16
        for row in scene['files']:
            assert sha(DATA / SCENE / row['path']) == row['sha256'], row['path']
    return expected


def checkpoint_audit(args):
    """CPU validation of source/data identity and complete, finite saved state."""
    import torch
    import yaml
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Checkpoint audits are CPU-only'
    torch.set_num_threads(4)
    verify_identity(args.identity)
    payload = torch.load(args.audit_checkpoint, map_location='cpu', weights_only=False)
    identity = payload['benchmark_identity']
    stage, step = identity['stage'], int(payload['global_step'])
    assert stage == args.stage and identity['scene'] == SCENE
    assert identity['formal'] is args.formal and identity['max_steps'] == STEPS[stage]
    if args.expect_step is not None:
        assert step == args.expect_step, (step, args.expect_step)
    assert 0 < step <= STEPS[stage]
    assert identity['training_script_sha256'] == TRAIN_SHA
    assert identity['benchmark_config_sha256'] == CONFIG_SHA
    assert identity['official_revision'] == OFFICIAL_REVISION
    assert identity['official_model_sha256'] == sha(OFFICIAL / STAGE_NAMES[stage] / 'src/model/mipnerf360/model.py')
    assert identity['seed'] == 777 and identity['workers'] == 0
    assert identity['chunk'] == 0 and identity['netchunk'] == 0
    assert identity['data_root'] == str(DATA) and identity['cuda_visible_devices'] == GPU and identity['device'] == 0
    scene = read(RUN / 'protocol/dataset_manifest.json')['scenes'][SCENE]
    assert identity['train_ids'] == scene['train_ids'] and identity['test_ids'] == scene['test_ids']
    for name, digest in identity['metadata_sha256'].items():
        assert sha(DATA / SCENE / name) == digest, name
    if stage > 1:
        assert identity['flow_manifest_sha256'] == sha(RUN / f'protocol/{SCENE}_flow.json')
        config = yaml.safe_load((args.config_dir / 'resolved.yaml').read_text())
        default = yaml.safe_load((OFFICIAL / STAGE_NAMES[stage] / 'configs/default.yaml').read_text())
        adventure = yaml.safe_load((OFFICIAL / STAGE_NAMES[stage] / 'configs/human_nerf/wild/monocular/adventure.yaml').read_text())
        def merge(left, right):
            result = dict(left)
            for key, value in right.items():
                result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
            return result
        expected_train = merge(default, adventure)['train']
        assert config['train'] == expected_train, 'Resolved train settings differ from frozen official YAML'
    assert len(payload['optimizer_states']) == 1
    optimizer = payload['optimizer_states'][0]
    assert optimizer['param_groups'] and optimizer['state'], 'Empty Adam state is not an acceptance'
    for tensor in payload['state_dict'].values():
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), 'Nonfinite model parameter'
    identifiers = [item for group in optimizer['param_groups'] for item in group['params']]
    assert len(identifiers) == len(set(identifiers))
    for parameter, state in optimizer['state'].items():
        assert parameter in identifiers and 0 < int(state['step']) <= step
        for name in ['exp_avg', 'exp_avg_sq']:
            assert torch.isfinite(state[name]).all(), (parameter, name)
        assert (state['exp_avg_sq'] >= 0).all()
    sampling = payload['benchmark_sampling']
    assert 0 <= sampling['committed'] <= sampling['epoch_length'] and sampling['epoch_rng'] is not None
    rng = payload['benchmark_rng']
    assert set(rng) == {'python', 'numpy', 'torch', 'cuda'} and len(rng['cuda']) == 1
    if args.before:
        before = torch.load(args.before, map_location='cpu', weights_only=False)
        previous_identity = before['benchmark_identity']
        # Resume path, parent checkpoint provenance and per-process segment limit
        # change between launches. Every scientific field retains exact identity.
        stable_keys = ['stage', 'scene', 'seed', 'data_root', 'train_ids', 'test_ids', 'metadata_sha256',
                       'official_revision', 'official_model_sha256', 'native_budget', 'max_steps', 'formal',
                       'training_script_sha256', 'benchmark_config_sha256', 'flow_manifest_sha256',
                       'workers', 'chunk', 'netchunk', 'torch_version', 'lightning_version',
                       'cuda_visible_devices', 'device', 'gpu']
        assert all(previous_identity[key] == identity[key] for key in stable_keys), 'Continuation scientific identity changed'
        assert identity['resume']['path'] == str(args.before) and identity['resume']['sha256'] == sha(args.before)
        assert int(before['global_step']) + 2 == step
        old_sampler = before['benchmark_sampling']
        assert sampling['committed'] == old_sampler['committed'] + 2
        assert sampling['epoch'] == old_sampler['epoch'] and sampling['epoch_length'] == old_sampler['epoch_length']
        assert torch.equal(sampling['epoch_rng']['torch'], old_sampler['epoch_rng']['torch'])
        import numpy as np
        assert all(np.array_equal(a, b) for a, b in zip(sampling['epoch_rng']['numpy'], old_sampler['epoch_rng']['numpy']))
        old = before['optimizer_states'][0]
        assert [group['params'] for group in old['param_groups']] == [group['params'] for group in optimizer['param_groups']]
        advances = []
        for parameter, old_state in old['state'].items():
            assert parameter in optimizer['state']
            advance = int(optimizer['state'][parameter]['step']) - int(old_state['step'])
            assert 0 <= advance <= 2
            advances.append(advance)
        for parameter, state in optimizer['state'].items():
            if parameter not in old['state']:
                assert int(state['step']) <= 2
        assert 2 in advances, 'No populated Adam state advanced through both updates'
        assert any(not torch.equal(tensor, before['state_dict'][name]) for name, tensor in payload['state_dict'].items()
                   if tensor.is_floating_point()), 'Short continuation did not change any learned model tensor'
    result = dict(status='passed', stage=stage, scene=SCENE, formal=args.formal, global_step=step,
        checkpoint=str(args.audit_checkpoint), checkpoint_sha256=sha(args.audit_checkpoint),
        training_script_sha256=TRAIN_SHA, groups=len(optimizer['param_groups']),
        populated_adam_states=len(optimizer['state']), finite_state=True, full_rng=True,
        sampler_committed=int(sampling['committed']), short_continuation_checked=bool(args.before))
    write(args.output, result)
    print(json.dumps(result), flush=True)


class Worker:
    def __init__(self, args):
        self.args = args
        self.deadline = dt.datetime.fromisoformat(args.cutoff).timestamp()
        self.queue_lock = None

    def deadline_check(self):
        if time.time() >= self.deadline:
            raise DeadlineReached('Core experiment freeze cutoff reached')

    def status(self, status, **fields):
        write(STATE / 'state.json', dict(status=status, scene=SCENE, host='a100-train',
              worker_pid=os.getpid(), physical_gpu=GPU, updated_unix=time.time(), **fields))

    def wait_processes(self, pids):
        descriptors = []
        try:
            for pid in sorted(set(pids)):
                try:
                    descriptors.append(pidfd_open(pid))
                except ProcessLookupError:
                    pass
            while descriptors:
                self.deadline_check()
                ready, _, _ = select.select(descriptors, [], [], max(0, self.deadline - time.time()))
                for descriptor in ready:
                    os.close(descriptor)
                    descriptors.remove(descriptor)
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    def wait_file_event(self, path, until_absent=False):
        """Wait for creation/replacement using inotify, including missing parents."""
        parent = Path(path).parent
        while not parent.is_dir():
            parent = parent.parent
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_init1.argtypes, libc.inotify_init1.restype = [ctypes.c_int], ctypes.c_int
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        descriptor = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
        if descriptor < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1 failed')
        try:
            # close_write, create, delete, moved_to/from, attrib; also directory removal.
            watch = libc.inotify_add_watch(descriptor, os.fsencode(parent), 0x00000FCE)
            if watch < 0:
                raise OSError(ctypes.get_errno(), f'inotify_add_watch failed: {parent}')
            # Install the watch before rechecking, so a creation/removal between
            # the caller's decision and registration cannot lose the wake-up.
            if (not Path(path).exists()) if until_absent else Path(path).exists():
                return
            self.deadline_check()
            select.select([descriptor], [], [], max(0, self.deadline - time.time()))
            self.deadline_check()
        finally:
            os.close(descriptor)

    def external_ready(self):
        rows, missing, pids = [], [], []
        for arm in ['R', 'G', 'RG']:
            directory = EXTERNAL_RUN / f'runs/r2_{arm}'
            receipt = directory / 'complete.json'
            if (directory / 'failed.json').exists():
                missing.append(directory / 'failed.json')
                continue
            if not receipt.is_file():
                missing.append(receipt)
                continue
            data = read(receipt)
            required = dict(status='completed_training', method=arm, repeat='2', updates=6000,
                            suffix_endpoint=6000, training_rgb_forwards=6000, adam_calls=12000,
                            physical_gpu='1', topology_unchanged=True)
            assert all(data.get(key) == value for key, value in required.items()), (receipt, required)
            assert data['moment_forwards'] == (0 if arm == 'R' else 6000)
            checkpoint = directory / 'checkpoint_12000.pt'
            recorded = Path(data['checkpoint']['path'])
            assert (EXTERNAL_ROOT / recorded).resolve() == checkpoint.resolve() or recorded.resolve() == checkpoint.resolve()
            assert sha(checkpoint) == data['checkpoint']['sha256'], checkpoint
            unit = f'4dsr-cg-r2-{arm.lower()}-20261006.service'
            result = subprocess.run(['systemctl', '--user', 'show', unit,
                      '--property=LoadState,MainPID,ActiveState,SubState,Result,ExecMainStatus'], capture_output=True, text=True)
            fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
            assert fields, (unit, result.stderr)
            pid = int(fields.get('MainPID', '0'))
            if pid:
                pids.append(pid)
            elif fields['LoadState'] != 'not-found':
                assert fields.get('Result') == 'success' and fields.get('ExecMainStatus') == '0', (unit, fields)
                assert fields.get('ActiveState') in ['inactive', 'active'] and fields.get('SubState') in ['exited', 'dead'], (unit, fields)
            rows.append(dict(arm=arm, receipt=str(receipt), receipt_sha256=sha(receipt),
                             checkpoint=str(checkpoint), checkpoint_sha256=data['checkpoint']['sha256'], service=fields))
        return rows, missing, pids

    def resource_reading(self):
        row = subprocess.check_output(['nvidia-smi', '-i', GPU,
              '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
              '--format=csv,noheader,nounits'], text=True).strip().split(',')
        uuid, name = row[0].strip(), row[1].strip()
        assert 'A100' in name, name
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                     '--format=csv,noheader'], text=True)
        pids = [int(line.split(',')[1]) for line in processes.splitlines() if line.split(',')[0].strip() == uuid]
        # Include unrelated 4dsr training processes before their first CUDA context.
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                command = path.read_bytes()
                if b'dynamic_sr_confidence_geometry_20261006/train.py' in command:
                    environment = path.with_name('environ').read_bytes().split(b'\0')
                    visible = next((line.split(b'=', 1)[1] for line in environment if line.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
                    if visible in [None, b'1']:
                        pids.append(int(path.parent.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        return dict(uuid=uuid, name=name, total_mib=float(row[2]), used_mib=float(row[3]),
                    utilization=float(row[4]), observed_unix=time.time(), busy_pids=sorted(set(pids)))

    def cpu(self, command, label):
        verify_identity(self.args.identity)
        log = STATE / 'logs' / f'{label}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   PYTHONDONTWRITEBYTECODE='1', TORCH_HOME='/home/ubuntu/.cache/torch', MPLBACKEND='Agg')
        started = time.time()
        with log.open('a') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        self.ledger(dict(task=label, mode='cpu', command=command, exit_code=result.returncode,
                        started_unix=started, ended_unix=time.time(), log=str(log)))
        if result.returncode:
            raise RuntimeError(f'CPU audit failed: {label}; see {log}')

    def ledger(self, receipt):
        receipt.update(host='a100-train', scene=SCENE, worker_identity_sha256=sha(self.args.identity))
        with (STATE / 'task_ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt) + '\n')

    def gpu(self, command, label):
        while True:
            self.deadline_check()
            hold = STATE / 'HOLD_DISPATCH.json'
            if hold.exists():
                self.status('waiting_for_coordinator_release', task=label, hold=read(hold))
                self.wait_file_event(hold, until_absent=True)
                continue
            external, missing, external_pids = self.external_ready()
            if missing:
                self.status('waiting_for_existing_4dsr_queue', task=label, missing=[str(path) for path in missing])
                self.wait_file_event(missing[0], until_absent=missing[0].name == 'failed.json')
                continue
            if external_pids:
                self.status('waiting_for_existing_process_exit', task=label, pids=external_pids)
                self.wait_processes(external_pids)
                continue
            verify_identity(self.args.identity)
            resource = (STATE / 'gpu1_resource.lock').open('a')
            try:
                try:
                    fcntl.flock(resource, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    owner = read(STATE / 'resource_owner.json')
                    self.status('waiting_for_hoi_resource_lease', task=label, owner=owner)
                    self.wait_processes([int(owner['child_pid'])])
                    continue
                first = self.resource_reading()
                if first['busy_pids']:
                    fcntl.flock(resource, fcntl.LOCK_UN)
                    self.status('waiting_for_existing_process_exit', task=label, pids=first['busy_pids'])
                    self.wait_processes(first['busy_pids'])
                    continue
                for reading in [first]:
                    assert reading['used_mib'] <= 1024 and reading['total_mib'] - reading['used_mib'] >= 30000 and reading['utilization'] <= 5, reading
                time.sleep(self.args.idle_check_seconds)
                self.deadline_check()
                second = self.resource_reading()
                if second['busy_pids']:
                    fcntl.flock(resource, fcntl.LOCK_UN)
                    self.wait_processes(second['busy_pids'])
                    continue
                assert second['uuid'] == first['uuid']
                assert second['used_mib'] <= 1024 and second['total_mib'] - second['used_mib'] >= 30000 and second['utilization'] <= 5, second
                if hold.exists():
                    continue
                self.deadline_check()
                log = STATE / 'logs' / f'{label}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                      TORCH_HOME='/home/ubuntu/.cache/torch', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1',
                      PYTHONUNBUFFERED='1', PL_FAULT_TOLERANT_TRAINING='0', PL_INTER_BATCH_PARALLELISM='0')
                started = time.time()
                self.status('running', task=label, command=command, readings=[first, second], log=str(log))
                with log.open('a') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                            pass_fds=(resource.fileno(), self.queue_lock.fileno()))
                    write(STATE / 'resource_owner.json', dict(worker_pid=os.getpid(), child_pid=child.pid, task=label,
                         physical_gpu=GPU, uuid=second['uuid'], started_unix=started))
                    write(STATE / 'active_process.json', dict(pid=child.pid, task=label, started_unix=started))
                    code = child.wait()
                self.ledger(dict(task=label, mode='gpu', command=command, exit_code=code, physical_gpu=GPU,
                     hardware=second, readings=[first, second], external_clearance=external,
                     started_unix=started, ended_unix=time.time(), seconds=time.time()-started, log=str(log)))
                write(STATE / 'active_process.json', dict(pid=None, task=label, exited_unix=time.time(), exit_code=code))
                if code:
                    raise RuntimeError(f'{label} exited{code}; see{log}')
                return
            finally:
                resource.close()  # Complete child state has been saved; lease ends here.

    def audit(self, checkpoint, stage, formal, step, directory, label, before=None):
        output = STATE / 'audits' / f'{label}_state.json'
        command = [str(PYTHON), str(CODE / 'a100_worker.py'), '--audit-checkpoint', str(checkpoint),
              '--identity', str(self.args.identity), '--stage', str(stage), '--expect-step', str(step),
              '--config-dir', str(directory), '--output', str(output)]
        if formal:
            command.append('--formal')
        if before:
            command += ['--before', str(before)]
        self.cpu(command, label + '_state')
        lr = STATE / 'audits' / f'{label}_lr.json'
        self.cpu([str(PYTHON), str(CODE / 'audit_native_lr.py'), '--checkpoint', str(checkpoint),
                  '--official', str(OFFICIAL), '--config-dir', str(directory), '--expect-step', str(step),
                  '--output', str(lr)], label + '_lr')
        assert read(output)['status'] == read(lr)['status'] == 'passed'
        return dict(state=str(output), state_sha256=sha(output), lr=str(lr), lr_sha256=sha(lr), checkpoint_sha256=sha(checkpoint))

    def train_command(self, stage, output, updates, smoke=False, resume=None):
        command = [str(PYTHON), '-u', str(CODE / 'train_native.py'), '--stage', str(stage), '--scene', SCENE,
              '--data-root', str(DATA), '--output', str(output), '--max-steps', str(STEPS[stage]),
              '--checkpoint-every', '2000', '--stop-after-updates', str(updates), '--no-evaluate',
              '--workers', '0', '--device', '0', '--deadline', str(self.deadline)]
        if smoke:
            command.append('--smoke')
        if resume:
            command += ['--resume', str(resume)]
        elif stage == 3:
            if smoke:
                parent = STATE / 'acceptance'
                background, human = parent / 'stage1/step000000008.ckpt', parent / 'stage2/step000000008.ckpt'
            else:
                parent = RUN / f'runs/formal/{SCENE}'
                background, human = parent / 'stage1/final.ckpt', parent / 'stage2/final.ckpt'
            command += ['--background-checkpoint', str(background), '--human-checkpoint', str(human)]
        return command

    def preflight(self):
        flow = RUN / f'protocol/{SCENE}_flow.json'
        self.gpu([str(PYTHON), '-u', str(CODE / 'prepare_flow.py'), '--scene', SCENE, '--data-root', str(DATA),
             '--checkpoint', str(ROOT / 'models/RAFT/raft-things.pth')], 'Tennis_train_only_flow')
        record = read(flow)
        scene = read(RUN / 'protocol/dataset_manifest.json')['scenes'][SCENE]
        assert record['status'] == 'complete' and record['identity']['heldout_rgb_read'] is False
        assert record['identity']['train_ids'] == scene['train_ids'] and record['identity']['test_ids'] == scene['test_ids']
        assert len(record['pairs']) == len(scene['train_ids']) == 285
        flow_identity = STATE / 'flow_identity.json'
        frozen = dict(manifest_sha256=sha(flow), identity_sha256=record['identity_sha256'])
        if flow_identity.exists():
            assert read(flow_identity) == frozen, 'A completed flow identity changed'
        else:
            write(flow_identity, frozen)
        accepted = {}
        for stage in [1, 2, 3]:
            directory = RUN / f'runs/smoke_a100_r2/{SCENE}/stage{stage}'
            directory.mkdir(parents=True, exist_ok=True)
            saved = STATE / f'acceptance/stage{stage}/step000000008.ckpt'
            last = directory / 'last.ckpt'
            current = read(directory / 'receipt.json')['global_step'] if (directory / 'receipt.json').exists() else 0
            assert current in [0, 8, 10], (stage, current)
            if current == 0:
                assert not last.exists(), 'Unindexed smoke checkpoint exists'
                self.gpu(self.train_command(stage, directory, 8, smoke=True), f'Tennis_stage{stage}_fresh8')
                self.deadline_check()
                current = read(directory / 'receipt.json')['global_step']
                assert current == 8
            if current == 8:
                self.audit(last, stage, False, 8, directory, f'stage{stage}_fresh8')
                saved.parent.mkdir(parents=True, exist_ok=True)
                if saved.exists():
                    assert sha(saved) == sha(last)
                else:
                    shutil.copy2(last, saved)
                self.gpu(self.train_command(stage, directory, 2, smoke=True, resume=saved), f'Tennis_stage{stage}_resume8to10')
                self.deadline_check()
            assert saved.is_file()
            self.audit(saved, stage, False, 8, directory, f'stage{stage}_immutable_fresh8')
            accepted[stage] = self.audit(last, stage, False, 10, directory, f'stage{stage}_resume10', before=saved)
            if stage == 1:
                result = STATE / 'audits/stage1_sampler_adam_resume.json'
                self.cpu([str(PYTHON), str(CODE / 'audit_stage1_resume.py'), '--before', str(saved), '--after', str(last),
                   '--updates', '2', '--official', str(OFFICIAL), '--data-root', str(DATA), '--output', str(result)], 'stage1_sampler_adam_resume')
                assert read(result)['status'] == 'passed'
                accepted[stage]['sampler_adam'] = dict(path=str(result), sha256=sha(result))
        write(STATE / 'preflight.json', dict(status='passed', source_identity_sha256=sha(self.args.identity),
               flow_manifest_sha256=sha(flow), stages=accepted, smoke_steps=30, formal_steps=0, finished_unix=time.time()))

    def formal(self):
        assert read(STATE / 'preflight.json')['status'] == 'passed'
        assert read(STATE / 'preflight.json')['source_identity_sha256'] == sha(self.args.identity)
        for stage in [1, 2, 3]:
            directory = RUN / f'runs/formal/{SCENE}/stage{stage}'
            directory.mkdir(parents=True, exist_ok=True)
            while True:
                self.deadline_check()
                receipt = directory / 'receipt.json'
                current = int(read(receipt)['global_step']) if receipt.exists() else 0
                last = directory / 'last.ckpt'
                if current:
                    self.audit(last, stage, True, current, directory, f'formal_stage{stage}_g{current:09d}')
                else:
                    assert not last.exists() and not (directory / 'final.ckpt').exists(), 'Formal training must start from scratch'
                if current == STEPS[stage]:
                    self.audit(directory / 'final.ckpt', stage, True, current, directory, f'formal_stage{stage}_endpoint')
                    break
                assert 0 <= current < STEPS[stage]
                updates = min(2000, STEPS[stage] - current)
                self.gpu(self.train_command(stage, directory, updates, resume=last if current else None),
                         f'Tennis_formal_stage{stage}_g{current:09d}_plus{updates}')
                self.deadline_check()
                after = read(receipt)
                assert after['global_step'] == current + updates and after['formal'] is True
                self.audit(last, stage, True, current + updates, directory, f'formal_stage{stage}_g{current+updates:09d}')
        output = RUN / f'evaluation/{SCENE}'
        self.gpu([str(PYTHON), '-u', str(CODE / 'evaluate_native.py'), '--scene', SCENE, '--data-root', str(DATA),
             '--checkpoint', str(RUN / f'runs/formal/{SCENE}/stage3/final.ckpt'), '--output', str(output)], 'Tennis_terminal_evaluation')
        metrics = read(output / 'metrics.json')
        assert metrics['status'] == 'completed' and metrics['frames'] == 16 and len(metrics['artifacts']) == 48
        for row in metrics['artifacts']:
            assert sha(output / row['path']) == row['sha256']
        assert metrics['identity']['checkpoint_sha256'] == sha(RUN / f'runs/formal/{SCENE}/stage3/final.ckpt')
        return dict(metrics=str(output / 'metrics.json'), metrics_sha256=sha(output / 'metrics.json'),
                    checkpoint=str(RUN / f'runs/formal/{SCENE}/stage3/final.ckpt'),
                    checkpoint_sha256=metrics['identity']['checkpoint_sha256'])

    def terminal(self, status, evaluation=None, error=None):
        roots = [STATE, RUN / f'runs/formal/{SCENE}', RUN / f'runs/smoke_a100_r2/{SCENE}', RUN / f'evaluation/{SCENE}']
        skip = {'state.json', 'active_process.json', 'resource_owner.json', 'terminal_receipt.json',
                'terminal_files.json', 'worker.log', 'queue.lock', 'gpu1_resource.lock', 'HOLD_DISPATCH.json'}
        rows = []
        for directory in roots:
            if directory.is_dir():
                for path in sorted(directory.rglob('*')):
                    if path.is_file() and path.name not in skip and not path.name.endswith('.tmp'):
                        rows.append(dict(path=str(path.relative_to(RUN)), bytes=path.stat().st_size, sha256=sha(path)))
        manifest = STATE / 'terminal_files.json'
        write(manifest, dict(status=status, files=rows, finished_unix=time.time()))
        receipt = dict(status=status, scene=SCENE, host='a100-train', physical_gpu=GPU,
            prescribed_steps=STEPS, source_identity=str(self.args.identity), source_identity_sha256=sha(self.args.identity),
            file_manifest=str(manifest), file_manifest_sha256=sha(manifest), files=len(rows),
            completion_event='evaluation_integrity_verified' if status == 'completed' else 'worker_terminal_failure_or_cutoff',
            evaluation=evaluation, error=error, finished_unix=time.time(),
            return_to_local_required=True, training_hardware='A100-SXM4-40GB physicalGPU1',
            cross_hardware_optimization_equivalence_claimed=False)
        target = STATE / 'terminal_receipt.json'
        write(target, receipt)
        self.status(status, completion_event=receipt['completion_event'], terminal_receipt=str(target),
                    terminal_receipt_sha256=sha(target), finished_unix=receipt['finished_unix'])
        print('A100_WORKER_TERMINAL', json.dumps(receipt), flush=True)

    def run(self):
        assert ROOT == REMOTE_ROOT, 'GPU run is restricted to the isolated a100-train checkout'
        STATE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = (STATE / 'queue.lock').open('a')
        try:
            fcntl.flock(self.queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            write(STATE / f'duplicate_start_{os.getpid()}.json', dict(status='rejected', reason='A worker or child holds the queue lock'))
            return 2
        try:
            verify_identity(self.args.identity, full_data=True)
            self.preflight()
            evaluation = self.formal()
            self.terminal('completed', evaluation=evaluation)
            return 0
        except DeadlineReached:
            self.terminal('stopped_at_freeze_deadline')
            return 3
        except BaseException:
            self.terminal('failed', error=traceback.format_exc())
            raise
        finally:
            self.queue_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--run', action='store_true', help='Explicitly dispatch the authorized remote queue after all resource gates')
    modes.add_argument('--freeze-identity', action='store_true', help='Save a new private CPU-verified deployment identity')
    modes.add_argument('--audit-checkpoint', type=Path, help='CPU checkpoint audit; does not dispatch any GPU work')
    parser.add_argument('--check-only', action='store_true', help='Default: verify CPU paths, sources and prepared input hashes')
    parser.add_argument('--identity', type=Path, default=STATE / 'deployment_identity.json')
    parser.add_argument('--cutoff', default='2026-11-04T23:00:00-08:00')
    parser.add_argument('--idle-check-seconds', type=float, default=10)
    parser.add_argument('--stage', type=int, choices=STEPS)
    parser.add_argument('--expect-step', type=int)
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--config-dir', type=Path)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.check_only and (args.run or args.audit_checkpoint):
        parser.error('--check-only cannot dispatch or audit a checkpoint')
    if not 2 <= args.idle_check_seconds <= 60:
        parser.error('idle check separation must be2..60 seconds')
    for name in ['identity', 'audit_checkpoint', 'config_dir', 'before', 'output']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.audit_checkpoint:
        if not (args.stage and args.config_dir and args.output):
            parser.error('checkpoint audit requires stage,config-dir andoutput')
        checkpoint_audit(args)
        return
    if args.freeze_identity:
        identity = json.loads(json.dumps(frozen_identity()))
        if args.identity.exists():
            assert read(args.identity) == identity, 'Refusing to overwrite a different frozen worker identity'
        else:
            write(args.identity, identity)
    if not args.run:
        identity = verify_identity(args.identity, full_data=True)
        report = dict(status='cpu_check_passed', gpu_tasks_launched=0, identity=str(args.identity),
          identity_sha256=sha(args.identity), sources=identity['sources'], scene=SCENE,
          run_root=str(RUN), state_root=str(STATE), formal_root=str(RUN / f'runs/formal/{SCENE}'),
          resource_lock=str(STATE / 'gpu1_resource.lock'), physical_gpu=GPU,
          existing_4dsr_priority_receipts=[str(EXTERNAL_RUN / f'runs/r2_{arm}/complete.json') for arm in ['R', 'G', 'RG']],
          pending='GPU flow and30 isolated smoke/resume updates; full native training and rendering remain undispatched')
        write(STATE / 'cpu_check.json', report)
        print(json.dumps(report), flush=True)
        return
    raise SystemExit(Worker(args).run())


if __name__ == '__main__':
    main()
