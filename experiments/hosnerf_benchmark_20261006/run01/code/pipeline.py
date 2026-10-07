#!/usr/bin/env python3
"""Persistent, resource-coordinated native HOSNeRF benchmark queue."""
from __future__ import annotations
import argparse
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import time
import traceback

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
CODE = RUN/'code'
PYTHON = ROOT/'envs/hosnerf/bin/python'
GPU = '1'
SHARED_LOCK = Path('/home/cai_tianshun/Project/4dsr/output/dynamic_sr_confidence_geometry_20261006/evaluation_gpu.lock')
STEPS = {1:500000, 2:400000, 3:200000}
SCENES = ['Backpack','Tennis','Suitcase','Playground','Dance','Lounge']
DEADLINE = None


class DeadlineReached(Exception):
    pass


def check_deadline():
    if DEADLINE is not None and time.time() >= DEADLINE:
        raise DeadlineReached('The frozen core-experiment deadline has been reached')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''):h.update(b)
    return h.hexdigest()


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # The bundled CPython 3.10 omits os.pidfd_open; glibc exposes it here.
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.pidfd_open
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.set_inheritable(fd, False)
    return fd


def wait_processes(pids):
    fds=[]
    for pid in pids:
        try:fds.append(pidfd_open(pid))
        except ProcessLookupError:pass
    while fds:
        ready,_,_=select.select(fds,[],[])
        for fd in ready:os.close(fd);fds.remove(fd)


def busy_pids():
    uuid=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    return [int(line.split(',')[1].strip()) for line in lines.splitlines() if uuid in line and '/opt/todesk/' not in line]


def task(command, label):
    state=RUN/'pipeline.json'
    while True:
        check_deadline()
        write(state, dict(status='waiting_for_shared_gpu', task=label, command=command, updated_unix=time.time()))
        SHARED_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with SHARED_LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            check_deadline()
            pids=busy_pids()
            if pids:
                # Release before waiting for unrelated owners to exit.
                fcntl.flock(lock, fcntl.LOCK_UN)
            else:
                first=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                time.sleep(2)
                pids=busy_pids()
                if not pids:
                    second=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                    for reading in (first,second):
                        fields=reading.split(',')
                        if float(fields[-3])-float(fields[-2])<20000 or float(fields[-1])>10:
                            raise RuntimeError(f'GPU{GPU} not confirmed idle with >=20GB available: {reading}')
                    check_deadline()
                    started=time.time()
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', PL_FAULT_TOLERANT_TRAINING='0', PL_INTER_BATCH_PARALLELISM='0')
                    log=RUN/'logs'/f'{label}.log'
                    write(state, dict(status='running',task=label,command=command,gpu=GPU,readings=[first,second],started_unix=started,log=str(log)))
                    with log.open('a') as f:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                        write(RUN/'active_process.json',dict(pid=child.pid,task=label,started_unix=started))
                        code=child.wait()
                    receipt=dict(task=label, command=command, exit_code=code, seconds=time.time()-started,
                                 hardware=second, physical_gpu=GPU, started_unix=started, ended_unix=time.time(),log=str(log))
                    with (RUN/'task_ledger.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
                    if code:raise RuntimeError(f'{label} exited {code}; see {log}')
                    return receipt
                fcntl.flock(lock, fcntl.LOCK_UN)
        if pids:wait_processes(pids)


def checkpoint(directory):
    p=directory/'last.ckpt'
    if p.exists():return p
    matches=sorted(directory.rglob('last.ckpt'))
    if len(matches)!=1:raise RuntimeError(f'Expected one last.ckpt in {directory}, found {matches}')
    return matches[0]


def train(scene, stage, smoke=False, acceptance_tag=''):
    suffix='_'+acceptance_tag if acceptance_tag else ''
    output=RUN/'runs'/('smoke'+suffix if smoke else 'formal')/scene/f'stage{stage}'
    output.mkdir(parents=True,exist_ok=True)
    command=[str(PYTHON),'-u',str(CODE/'train_native.py'),'--stage',str(stage),'--scene',scene,
             '--data-root',str(RUN/'data'),'--output',str(output),'--max-steps',str(STEPS[stage]),
             '--checkpoint-every','2000','--stop-after-updates',str(8 if smoke else 2000), '--no-evaluate', '--workers','0']
    if smoke:command+=['--smoke']
    if DEADLINE is not None:command+=['--deadline',str(DEADLINE)]
    if stage==3:
        command+=['--background-checkpoint',str(checkpoint(output.parent/'stage1')),
                  '--human-checkpoint',str(checkpoint(output.parent/'stage2'))]
    # A resumed segment always retains the complete target step budget.
    if (output/'last.ckpt').exists():command+=['--resume',str(output/'last.ckpt')]
    result=task(command,f'{scene}_stage{stage}_'+('smoke'+suffix if smoke else 'segment'))
    subprocess.run(['python3',str(CODE/'summarize.py')],cwd=ROOT,check=True)
    return result


def main(a):
    global DEADLINE, GPU, SHARED_LOCK, SCENES
    GPU=a.gpu
    if a.shared_lock:
        SHARED_LOCK=a.shared_lock.expanduser().resolve()
    elif GPU=='0':
        SHARED_LOCK=RUN/'gpu0_resource.lock'
    SCENES=a.scenes.split(',')
    lock=(RUN/'pipeline.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        write(RUN/f'duplicate_launch_{os.getpid()}.json',dict(status='rejected',reason='another pipeline owns the lock',time=time.time()))
        return
    DEADLINE=dt.datetime.fromisoformat(a.cutoff).timestamp()
    suffix='_'+a.acceptance_tag if a.acceptance_tag else ''
    preflight_path=RUN/('preflight'+suffix+'.json')
    if a.preflight:
        task([str(PYTHON),'-u',str(CODE/'prepare_flow.py'),'--scene','Backpack','--data-root',str(RUN/'data'),
              '--checkpoint',str(ROOT/'models/RAFT/raft-things.pth')], 'Backpack_flow')
        for stage in (1,2,3):train('Backpack',stage,smoke=True,acceptance_tag=a.acceptance_tag)
        write(preflight_path,dict(status='passed',finished_unix=time.time(),acceptance_tag=a.acceptance_tag,
                                      note='Three isolated 8-update native stages; no benchmark metrics'))
        write(RUN/'pipeline.json',dict(status='preflight_passed',finished_unix=time.time()))
        return
    if (RUN/'protocol/HOLD_FORMAL.json').exists():
        write(RUN/'pipeline.json',dict(status='formal_dispatch_held',
              hold=read(RUN/'protocol/HOLD_FORMAL.json'), updated_unix=time.time()))
        return
    if not preflight_path.exists() or read(preflight_path)['status']!='passed':
        raise RuntimeError('Three-stage preflight is required before formal training')
    cutoff=DEADLINE
    for scene in SCENES:
        if not (RUN/'data'/scene/'images').exists():
            write(RUN/'pipeline.json',dict(status='missing_dataset',scene=scene,updated_unix=time.time()))
            raise RuntimeError(f'Missing dataset: {scene}')
        task([str(PYTHON),'-u',str(CODE/'prepare_flow.py'),'--scene',scene,'--data-root',str(RUN/'data'),
              '--checkpoint',str(ROOT/'models/RAFT/raft-things.pth')],f'{scene}_flow')
        for stage in (1,2,3):
            output=RUN/'runs/formal'/scene/f'stage{stage}'
            while True:
                if (RUN/'protocol/HOLD_FORMAL.json').exists():
                    write(RUN/'pipeline.json',dict(status='formal_dispatch_held',
                          hold=read(RUN/'protocol/HOLD_FORMAL.json'), updated_unix=time.time()))
                    return
                if time.time()>=cutoff:
                    write(RUN/'pipeline.json',dict(status='stopped_at_freeze_deadline',scene=scene,stage=stage,updated_unix=time.time()))
                    return
                receipt=output/'receipt.json'
                if receipt.exists() and read(receipt).get('global_step',0)>=STEPS[stage]:break
                train(scene,stage)
                check_deadline()
        # Endpoint evaluation is an exit-event continuation, not a polling monitor.
        task([str(PYTHON),'-u',str(CODE/'evaluate_native.py'),'--scene',scene,'--data-root',str(RUN/'data'),
              '--checkpoint',str(checkpoint(RUN/'runs/formal'/scene/'stage3')),'--output',str(RUN/'evaluation'/scene)],f'{scene}_evaluation')
        subprocess.run(['python3',str(CODE/'summarize.py')],cwd=ROOT,check=True)
    write(RUN/'pipeline.json',dict(status='complete',finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    p.add_argument('--cutoff',default='2026-11-04T23:00:00-08:00')
    p.add_argument('--acceptance-tag',default='')
    p.add_argument('--gpu',choices=['0','1'],default='1')
    p.add_argument('--shared-lock',type=Path)
    p.add_argument('--scenes',default=','.join(SCENES))
    args=p.parse_args()
    if args.acceptance_tag and not re.fullmatch(r'[a-z][a-z0-9_]{0,31}',args.acceptance_tag):
        p.error('acceptance-tag must be a short lowercase identifier')
    if len(set(args.scenes.split(','))) != len(args.scenes.split(',')) or not set(args.scenes.split(',')).issubset(SCENES):
        p.error('scenes must be distinct names from the frozen six-scene benchmark')
    try:main(args)
    except DeadlineReached:
        write(RUN/'pipeline.json',dict(status='stopped_at_freeze_deadline',updated_unix=time.time()))
    except BaseException:
        write(RUN/'pipeline.json',dict(status='failed',traceback=traceback.format_exc(),updated_unix=time.time()))
        raise
