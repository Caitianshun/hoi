#!/usr/bin/env python3
"""Return the A100 Tennis exit event and recompute endpoint metrics locally.

Default mode is a CPU-only contract/self check. --run watches one identified
worker through remote pidfd, resumes SSH attachment after transport failures,
copies verified terminal artifacts to an isolated snapshot, and evaluates on
physical GPU1 under the shared 4dsr evaluation lock. It never writes the local
training queue, ledger, original data, or remote files.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
RETURN = RUN / 'a100_return'
REMOTE_ROOT = Path('/home/ubuntu/3DGS/HOI')
REMOTE_RUN = REMOTE_ROOT / 'experiments/hosnerf_benchmark_20261006/run01'
REMOTE_STATE = REMOTE_RUN / 'a100_worker'
PYTHON = ROOT / 'envs/hosnerf/bin/python'
SHARED_LOCK = Path('/home/cai_tianshun/Project/4dsr/output/dynamic_sr_confidence_geometry_20261006/evaluation_gpu.lock')
TERMINAL = {'completed', 'failed', 'stopped_at_freeze_deadline'}
SSH = ['ssh', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', 'a100-train']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or '\n' in name or '\r' in name:
        raise ValueError(f'Unsafe manifest path: {name!r}')
    return p


def in_snapshot(snapshot, name):
    relative(name)
    candidate = snapshot / name
    if not candidate.resolve().is_relative_to(snapshot.resolve()):
        raise ValueError(f'Snapshot path escaped through a symlink: {name}')
    return candidate


def choose_files(manifest):
    """Keep a complete index while avoiding redundant checkpoint transfers."""
    result, names = [], set()
    for item in manifest['files']:
        p = relative(item['path'])
        if item['path'] in names:
            raise ValueError('Duplicate terminal file path')
        names.add(item['path'])
        if not re.fullmatch(r'[0-9a-f]{64}', item['sha256']) or not isinstance(item['bytes'], int) or item['bytes'] < 0:
            raise ValueError('Invalid terminal file identity')
        allowed = (p.parts[0] == 'a100_worker' or str(p).startswith(('runs/formal/Tennis/',
                   'runs/smoke_a100_r2/Tennis/', 'evaluation/Tennis/', 'data/Tennis/images_flow/'))
                   or str(p) in {'data/Tennis/cameras_scaleworld.pkl', 'protocol/Tennis_flow.json',
                                   'protocol/dataset_manifest.json', 'configs/benchmark.json'})
        if not allowed:
            raise ValueError(f'Unexpected worker terminal artifact: {p}')
        checkpoint = p.suffix in {'.ckpt', '.pth', '.pt'}
        selected = not checkpoint or str(p) in {f'runs/formal/Tennis/stage{i}/final.ckpt' for i in (1, 2, 3)}
        result.append(dict(item, remote_path=str(REMOTE_RUN / item['path']),
                           transfer_status='selected' if selected else 'remote_retained'))
    return result


# The remote wait does not poll training status. The /proc identity prevents PID
# reuse and a new systemd invocation from silently changing the watched worker.
WAIT_SCRIPT = r'''
import ctypes,json,os,select,subprocess,sys
from pathlib import Path
c=json.loads(sys.argv[1]); state=Path(c['state']); expected=c.get('attachment')
terminal={'completed','failed','stopped_at_freeze_deadline'}
def emit(x): print(json.dumps(x),flush=True)
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if c.get('service'):
    raw=subprocess.check_output(['systemctl','--user','show',c['service'],'--property=MainPID,InvocationID'],text=True)
    service=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    pid=int(service.get('MainPID','0')); invocation=service.get('InvocationID','')
else: pid=c['pid']; invocation=''
if expected: pid=expected['pid']
try:
    stat=Path('/proc/%d/stat'%pid).read_text(); ticks=int(stat[stat.rfind(')')+2:].split()[19])
    cmd=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
except (FileNotFoundError,ProcessLookupError):
    current=json.loads((state/'state.json').read_text())
    if current.get('status') not in terminal: raise RuntimeError('Worker exited without a terminal receipt')
    if expected and current.get('worker_pid')!=expected['pid']: raise RuntimeError('Terminal receipt belongs to another PID')
    if c.get('pid') and current.get('worker_pid')!=c['pid']: raise RuntimeError('Terminal receipt belongs to another requested PID')
    emit({'event':'terminal_after_exit','state':current}); sys.exit(0)
actual={'pid':pid,'start_ticks':ticks,'boot_id':boot,'invocation_id':invocation,'service':c.get('service'),'cmdline':cmd}
if 'a100_worker.py' not in cmd or '--run' not in cmd or '/home/ubuntu/3DGS/HOI/' not in cmd:
    raise RuntimeError('PID is not the authorized isolated HOSNeRF worker')
if expected:
    for key in ['pid','start_ticks','boot_id']:
        if actual[key]!=expected[key]: raise RuntimeError('Worker process identity changed: '+key)
    if expected.get('invocation_id') and invocation and invocation!=expected['invocation_id']:
        raise RuntimeError('Service invocation changed')
if c.get('start_ticks') and ticks!=c['start_ticks']: raise RuntimeError('Requested worker start identity changed')
if hasattr(os,'pidfd_open'): fd=os.pidfd_open(pid)
else:
    libc=ctypes.CDLL(None,use_errno=True); fun=libc.pidfd_open; fun.argtypes=[ctypes.c_int,ctypes.c_uint];fun.restype=ctypes.c_int
    fd=fun(pid,0)
    if fd<0: raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
# Recheck after opening the pidfd to close the PID-reuse window.
again=Path('/proc/%d/stat'%pid).read_text()
if int(again[again.rfind(')')+2:].split()[19])!=ticks: raise RuntimeError('PID changed while attaching')
emit({'event':'attached','attachment':actual})
select.select([fd],[],[]); os.close(fd)
current=json.loads((state/'state.json').read_text())
if current.get('status') not in terminal or current.get('worker_pid')!=pid:
    raise RuntimeError('Worker exit did not produce its terminal receipt')
emit({'event':'terminal_after_exit','attachment':actual,'state':current})
'''


BUNDLE_SCRIPT = r'''
import base64,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); state=root/'a100_worker'
def row(p):
    data=p.read_bytes(); return {'path':str(p.relative_to(root)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'base64':base64.b64encode(data).decode()}
first=row(state/'state.json'); status=json.loads(base64.b64decode(first['base64']))
if status['status'] not in {'completed','failed','stopped_at_freeze_deadline'}: raise RuntimeError('Worker is not terminal')
receipt_path=Path(status['terminal_receipt'])
if receipt_path!=state/'terminal_receipt.json': raise RuntimeError('Unexpected receipt path')
receipt=row(receipt_path)
if receipt['sha256']!=status['terminal_receipt_sha256']: raise RuntimeError('Terminal receipt hash mismatch')
r=json.loads(base64.b64decode(receipt['base64'])); manifest_path=Path(r['file_manifest'])
if manifest_path!=state/'terminal_files.json': raise RuntimeError('Unexpected manifest path')
manifest=row(manifest_path)
if manifest['sha256']!=r['file_manifest_sha256']: raise RuntimeError('Terminal manifest hash mismatch')
identity_path=Path(r['source_identity'])
if not identity_path.is_relative_to(root): raise RuntimeError('Source identity outside isolated run')
identity=row(identity_path)
if identity['sha256']!=r['source_identity_sha256']: raise RuntimeError('Source identity hash mismatch')
if row(state/'state.json')['sha256']!=first['sha256']: raise RuntimeError('Terminal state changed while reading')
print(json.dumps({'state':first,'receipt':receipt,'manifest':manifest,'identity':identity}),flush=True)
'''


def ssh_python(script, *args):
    return SSH + [shlex.join(['python3', '-u', '-c', script, *args])]


def watch(args):
    previous = read(RETURN / 'watcher.json') if (RETURN / 'watcher.json').exists() else {}
    attachment = previous.get('attachment')
    if attachment and (attachment.get('service') != args.remote_service or (args.remote_pid and attachment['pid'] != args.remote_pid)):
        raise RuntimeError('Existing watcher attachment belongs to a different requested worker')
    failures = 0
    while True:
        config = dict(state=str(REMOTE_STATE), service=args.remote_service, pid=args.remote_pid,
                      start_ticks=args.remote_start_ticks, attachment=attachment)
        stderr = RETURN / 'ssh_wait.log'
        with stderr.open('a') as log:
            child = subprocess.Popen(ssh_python(WAIT_SCRIPT, json.dumps(config)), stdout=subprocess.PIPE,
                                     stderr=log, text=True)
            terminal = None
            for line in child.stdout:
                event = json.loads(line)
                if event['event'] == 'attached':
                    attachment = event['attachment']
                    write(RETURN / 'watcher.json', dict(status='attached_remote_pidfd', attachment=attachment,
                          watcher_pid=os.getpid(), updated_unix=time.time(), transport_failures=failures))
                elif event['event'] == 'terminal_after_exit':
                    terminal = event
            code = child.wait()
        if code == 0 and terminal:
            write(RETURN / 'remote_exit_event.json', dict(terminal, transport_failures=failures,
                  received_unix=time.time(), ssh_exit_code=code))
            return terminal
        # SSH 255 is transport failure, never a model completion event. Other
        # failures mean process/contract errors and need operator inspection.
        if code != 255:
            raise RuntimeError(f'Remote pidfd attachment failed ({code}); see {stderr}')
        failures += 1
        write(RETURN / 'watcher.json', dict(status='reconnecting_transport', attachment=attachment,
              transport_failures=failures, updated_unix=time.time(), watcher_pid=os.getpid()))
        time.sleep(min(60, 2 ** min(failures, 6)))


def terminal_bundle():
    result = subprocess.run(ssh_python(BUNDLE_SCRIPT, str(REMOTE_RUN)), check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(result.stdout)


def save_bundle(bundle, snapshot):
    for item in bundle.values():
        data = base64.b64decode(item['base64'], validate=True)
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('Control bundle content hash mismatch')
        path = in_snapshot(snapshot, item['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and sha(path) != item['sha256']:
            raise RuntimeError('Existing terminal snapshot belongs to a different worker receipt')
        path.write_bytes(data)


def transfer():
    before = terminal_bundle()
    snapshot = RETURN / 'remote_snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    save_bundle(before, snapshot)
    receipt = json.loads(base64.b64decode(before['receipt']['base64']))
    manifest = json.loads(base64.b64decode(before['manifest']['base64']))
    if receipt['status'] != manifest['status'] or receipt['files'] != len(manifest['files']):
        raise RuntimeError('Terminal manifest count/status disagrees with receipt')
    if receipt['status'] == 'completed' and receipt['completion_event'] != 'evaluation_integrity_verified':
        raise RuntimeError('Completion lacks endpoint integrity verification')
    index = choose_files(manifest)
    selected = [row for row in index if row['transfer_status'] == 'selected']
    if receipt['status'] == 'completed':
        required = {f'runs/formal/Tennis/stage{i}/final.ckpt' for i in (1, 2, 3)}
        if not required.issubset({row['path'] for row in selected}):
            raise RuntimeError('Completed run lacks three formal final checkpoints')
    # Prevent rsync from following a malicious pre-existing local directory link.
    for row in selected:
        path = in_snapshot(snapshot, row['path'])
        if path.is_symlink():
            raise RuntimeError('Terminal snapshot contains a file symlink')
    listing = RETURN / 'return_files.txt'
    listing.write_text(''.join(row['path'] + '\n' for row in selected))
    log_path = RETURN / 'rsync_return.log'
    command = ['rsync', '-a', '--checksum', '--partial', '--stats', '--files-from=' + str(listing),
               '-e', shlex.join(SSH[:-1]), 'a100-train:' + str(REMOTE_RUN) + '/', str(snapshot) + '/']
    with log_path.open('a') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    for row in selected:
        path = in_snapshot(snapshot, row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError('Returned artifact failed SHA/size verification: ' + row['path'])
        row['transfer_status'] = 'local_verified'
        row['local_path'] = str(path)
    after = terminal_bundle()
    if {k: v['sha256'] for k, v in before.items()} != {k: v['sha256'] for k, v in after.items()}:
        raise RuntimeError('Remote terminal identity changed during transfer')
    evaluation = receipt.get('evaluation')
    if evaluation:
        for key in ['metrics', 'checkpoint']:
            name = str(Path(evaluation[key]).relative_to(REMOTE_RUN))
            if sha(in_snapshot(snapshot, name)) != evaluation[key + '_sha256']:
                raise RuntimeError('Endpoint evaluation receipt identity mismatch')
    write(RETURN / 'returned_file_index.json', dict(status='verified', remote_root=str(REMOTE_RUN),
          terminal_receipt_sha256=before['receipt']['sha256'], returned_files=len(selected),
          remote_retained_files=len(index)-len(selected), files=index,
          policy='Three formal final full-state checkpoints only; original RGB/metadata stay local; remote files retained',
          completed_unix=time.time()))
    write(RETURN / 'return_receipt.json', dict(status=receipt['status'], remote_receipt=receipt,
          remote_receipt_sha256=before['receipt']['sha256'], terminal_files_sha256=before['manifest']['sha256'],
          source_identity_sha256=before['identity']['sha256'], local_snapshot=str(snapshot),
          remote_evaluation=str(snapshot / 'evaluation/Tennis'), all_returned_hashes_verified=True,
          returned_bytes=sum(row['bytes'] for row in selected), completed_unix=time.time()))
    return receipt, snapshot, before


def adapt_data(snapshot):
    manifest_path = snapshot / 'protocol/dataset_manifest.json'
    manifest = read(manifest_path)['scenes']['Tennis']
    source = RUN / 'data/Tennis'
    data = RETURN / 'local_data/Tennis'
    data.mkdir(parents=True, exist_ok=True)
    for row in manifest['files']:
        p = relative(row['path'])
        if str(p) == 'cameras_scaleworld.pkl':
            continue
        candidate = source / str(p)
        if not candidate.is_file() or sha(candidate) != row['sha256']:
            raise RuntimeError('Local original input differs from remote training identity: ' + str(p))
    camera = snapshot / 'data/Tennis/cameras_scaleworld.pkl'
    flow = snapshot / 'data/Tennis/images_flow'
    if not camera.is_file() or not flow.is_dir():
        raise RuntimeError('Returned generated camera/flow inputs are missing')
    for child in source.iterdir():
        if child.name in {'cameras_scaleworld.pkl', 'images_flow'}:
            continue
        target = data / child.name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != child.resolve():
                raise RuntimeError('Local evaluation adapter exists with different input')
        else:
            target.symlink_to(child, target_is_directory=child.is_dir())
    target = data / camera.name
    if target.exists() and sha(target) != sha(camera):
        raise RuntimeError('Local generated camera adapter identity changed')
    shutil.copy2(camera, target)
    target_flow = data / 'images_flow'
    if target_flow.exists() or target_flow.is_symlink():
        if not target_flow.is_symlink() or target_flow.resolve() != flow.resolve():
            raise RuntimeError('Local flow adapter exists with different source')
    else:
        target_flow.symlink_to(flow, target_is_directory=True)
    write(RETURN / 'local_data_identity.json', dict(scene='Tennis', data_root=str(data.parent),
          original_input=str(source), remote_training_input=str(REMOTE_RUN / 'data/Tennis'),
          camera_source=str(camera), camera_sha256=sha(camera), flow_source=str(flow),
          dataset_manifest=str(manifest_path), dataset_manifest_sha256=sha(manifest_path),
          input_files_verified=len(manifest['files'])-1, evaluation_only_adapter=True))
    return data.parent


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    fun = libc.pidfd_open
    fun.argtypes, fun.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    fd = fun(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def wait_pids(pids):
    fds = []
    try:
        for pid in sorted(set(pids)):
            try:
                fds.append(pidfd_open(pid))
            except ProcessLookupError:
                pass
        if fds:
            while fds:
                ready, _, _ = select.select(fds, [], [])
                for fd in ready:
                    os.close(fd)
                    fds.remove(fd)
    finally:
        for fd in fds:
            os.close(fd)


def gpu_reading():
    raw = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    fields = [x.strip() for x in raw.split(',')]
    if len(fields) != 5:
        raise RuntimeError('Unexpected physical GPU1 inventory')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                    '--format=csv,noheader'], text=True)
    pids = [int(line.split(',')[1].strip()) for line in apps.splitlines() if fields[1] in line]
    return dict(raw=raw, uuid=fields[1], pids=pids, available_mib=float(fields[2])-float(fields[3]),
                utilization=float(fields[4]), observed_unix=time.time())


def evaluate(receipt, snapshot, bundle):
    identity = json.loads(base64.b64decode(bundle['identity']['base64']))
    evaluator = RUN / 'code/evaluate_native.py'
    if sha(evaluator) != identity['sources']['evaluate_native.py']:
        raise RuntimeError('Local evaluator differs from frozen remote evaluator')
    for name, digest in identity['official_files'].items():
        if sha(ROOT / 'third_party/HOSNeRF' / name) != digest:
            raise RuntimeError('Local official HOSNeRF source differs from training source')
    data_root = adapt_data(snapshot)
    checkpoint = snapshot / 'runs/formal/Tennis/stage3/final.ckpt'
    output = RUN / 'evaluation/Tennis'
    if output.exists() and any(output.iterdir()):
        previous = read(output / 'metrics.json') if (output / 'metrics.json').exists() else {}
        prior_identity = previous.get('identity', {})
        if (previous.get('status') != 'completed' or prior_identity.get('checkpoint_sha256') != sha(checkpoint)
                or prior_identity.get('data_path') != str(data_root / 'Tennis')
                or prior_identity.get('evaluator_sha256') != sha(evaluator)
                or not prior_identity.get('torch_version')
                or str(output) == str(snapshot / 'evaluation/Tennis')):
            raise RuntimeError('Existing local evaluation cannot be overwritten or reused as unified GPU1 evaluation')
    command = [str(PYTHON), '-u', str(evaluator), '--scene', 'Tennis', '--data-root', str(data_root),
               '--checkpoint', str(checkpoint), '--output', str(output), '--data-manifest',
               str(snapshot / 'protocol/dataset_manifest.json')]
    SHARED_LOCK.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with SHARED_LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            first = gpu_reading()
            if first['pids']:
                pids = first['pids']
            else:
                time.sleep(2)
                second = gpu_reading()
                pids = second['pids']
                if not pids:
                    if any(r['available_mib'] < 20000 or r['utilization'] > 10 for r in [first, second]):
                        raise RuntimeError('GPU1 not confirmed idle with sufficient memory')
                    started = time.time()
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4',
                               OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
                    write(RETURN / 'watcher.json', dict(status='unified_local_evaluation', physical_gpu='1',
                          gpu_readings=[first, second], command=command, started_unix=started))
                    with (RETURN / 'local_evaluation.log').open('a') as log:
                        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 pass_fds=(lock.fileno(),))
                        write(RETURN / 'local_evaluation_process.json', dict(pid=child.pid, command=command, started_unix=started))
                        code = child.wait()
                    if code:
                        raise RuntimeError(f'Unified local evaluation failed ({code})')
                    break
            fcntl.flock(lock, fcntl.LOCK_UN)
        wait_pids(pids)
    metrics = read(output / 'metrics.json')
    if metrics['status'] != 'completed' or metrics['frames'] != 16 or len(metrics['artifacts']) != 48:
        raise RuntimeError('Unified local evaluation is incomplete')
    if metrics['identity']['checkpoint_sha256'] != receipt['evaluation']['checkpoint_sha256']:
        raise RuntimeError('Unified local metrics used a different endpoint')
    for artifact in metrics['artifacts']:
        if sha(output / artifact['path']) != artifact['sha256']:
            raise RuntimeError('Unified local render artifact hash mismatch')
    write(RETURN / 'unified_evaluation_receipt.json', dict(status='completed', physical_gpu='1',
          hardware_readings=[first, second], metrics=str(output / 'metrics.json'),
          metrics_sha256=sha(output / 'metrics.json'), remote_evaluation=str(snapshot / 'evaluation/Tennis/metrics.json'),
          checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), data_root=str(data_root),
          remote_training_checkpoint=receipt['evaluation']['checkpoint'],
          source_identity_sha256=bundle['identity']['sha256'], evaluator_sha256=sha(evaluator),
          elapsed_seconds=time.time()-started, cross_hardware_optimization_equivalence_claimed=False))
    subprocess.run([sys.executable, str(RUN / 'code/summarize.py')], cwd=ROOT, check=True)
    summary = read(RUN / 'summary.json')
    if summary['completed_scenes'] < 6 and summary['status'] == 'completed':
        raise RuntimeError('Summary incorrectly declared a partial benchmark complete')


def check_only():
    """CPU checks exercise parsing, manifest policy and the actual pidfd API."""
    compile(WAIT_SCRIPT, '<remote-pidfd>', 'exec')
    compile(BUNDLE_SCRIPT, '<remote-bundle>', 'exec')
    rejected = []
    for name in ['../data', '/absolute', 'a/../b', 'a\nb', 'a//b']:
        try:
            relative(name)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Unsafe manifest path accepted')
    rows = [dict(path='runs/formal/Tennis/stage3/final.ckpt', bytes=2, sha256='a'*64),
            dict(path='runs/formal/Tennis/stage3/last.ckpt', bytes=2, sha256='b'*64),
            dict(path='runs/smoke_a100_r2/Tennis/stage1/last.ckpt', bytes=2, sha256='c'*64),
            dict(path='data/Tennis/images_flow/0001.npz', bytes=2, sha256='d'*64)]
    chosen = choose_files(dict(files=rows))
    assert [r['transfer_status'] for r in chosen] == ['selected', 'remote_retained', 'remote_retained', 'selected']
    for malformed in [dict(files=rows+rows[:1]), dict(files=[dict(rows[0], path='data/Tennis/images/1.png')])]:
        try:
            choose_files(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe terminal manifest accepted')
    with tempfile.TemporaryDirectory() as name:
        snap = Path(name)
        payload = b'{"status":"completed"}\n'
        control = dict(path='a100_worker/state.json', bytes=len(payload),
                       sha256=hashlib.sha256(payload).hexdigest(), base64=base64.b64encode(payload).decode())
        save_bundle(dict(state=control), snap)
        assert sha(snap / control['path']) == control['sha256']
        (snap / 'escape').symlink_to('/tmp', target_is_directory=True)
        try:
            in_snapshot(snap, 'escape/anything')
        except ValueError:
            pass
        else:
            raise AssertionError('Snapshot symlink escape accepted')
    child = subprocess.Popen([sys.executable, '-c', 'pass'])
    wait_pids([child.pid])
    assert child.wait() == 0
    result = dict(status='cpu_check_passed', watcher_sha256=sha(__file__), gpu_tasks_launched=0,
                  remote_connections=0, unsafe_paths_rejected=len(rejected),
                  terminal_file_selection_checked=True, stable_control_hashes_checked=True,
                  symlink_escape_rejected=True, real_local_pidfd_exit_wait_checked=True,
                  gpu='physical GPU1 only; shared 4dsr evaluation lock; two readings before launch',
                  default_mode='CPU check only; --run required', completed_unix=time.time())
    write(RETURN / 'watcher_cpu_check.json', result)
    print(json.dumps(result), flush=True)


def main(args):
    if not args.run:
        check_only()
        return
    RETURN.mkdir(parents=True, exist_ok=True)
    with (RETURN / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch(args)
        receipt, snapshot, bundle = transfer()
        if receipt['status'] == 'completed':
            evaluate(receipt, snapshot, bundle)
            status = 'completed_returned_and_unified_evaluation_verified'
        else:
            subprocess.run([sys.executable, str(RUN / 'code/summarize.py')], cwd=ROOT, check=True)
            status = 'remote_terminal_failure_or_cutoff_returned'
        write(RETURN / 'watcher.json', dict(status=status, remote_status=receipt['status'],
              finished_unix=time.time(), local_pipeline_modified=False, remote_files_modified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    target = parser.add_mutually_exclusive_group()
    target.add_argument('--remote-pid', type=int, default=0)
    target.add_argument('--remote-service')
    parser.add_argument('--remote-start-ticks', type=int, default=0)
    args = parser.parse_args()
    if args.run and not (args.remote_pid or args.remote_service):
        parser.error('--run requires --remote-pid or --remote-service')
    if args.remote_pid < 0 or args.remote_start_ticks < 0:
        parser.error('PID/start ticks must be nonnegative')
    if args.remote_service and not re.fullmatch(r'[A-Za-z0-9_.@-]+\.service', args.remote_service):
        parser.error('Invalid user service name')
    try:
        main(args)
    except BaseException:
        write(RETURN / 'watcher_failure.json', dict(status='failed', traceback=traceback.format_exc(), updated_unix=time.time()))
        raise
