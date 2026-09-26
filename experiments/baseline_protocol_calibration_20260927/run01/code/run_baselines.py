"""Persistent fixed-matrix launcher; completion triggers freeze and evaluation.

Every child attempt has a wall-time ledger, including failures. One shared
12-hour budget covers GPU jobs. No repeated training or automatic new seed.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from adapter_4dgs import ROOT,RUN,UPSTREAM,OFFICIAL_COMMIT,sha,save_json,official_config

PYTHON = str(ROOT/'envs/4dgs/bin/python')
S1_PYTHON = str(ROOT/'envs/gvhmr/bin/python')
CODE = RUN/'code'
LEDGER = RUN/'protocol/gpu_cost_ledger.json'


def used_seconds():
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    return sum(r['wall_seconds'] for r in ledger)


def append_cost(row):
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    ledger.append(row);save_json(LEDGER,ledger)


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                                    '--format=csv'],text=True).strip()


def gpu_job(label,command):
    remaining=12*3600-used_seconds()
    if remaining<=0: raise TimeoutError('Cumulative V3 GPU budget exhausted')
    attempts=RUN/'logs'/label
    attempts.mkdir(parents=True,exist_ok=True)
    if (attempts/'attempt.json').exists():
        raise FileExistsError('Existing attempt retained; exact-state recovery must be explicit')
    start=time.time();stop=threading.Event();samples=[]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4',
                                    V3_REMAINING_GPU_SECONDS=str(remaining),PYTHONUNBUFFERED='1')
    with (attempts/'stdout.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save_json(attempts/'attempt.json',dict(status='running',pid=proc.pid,start_unix=start,command=command,
                                             GPU='physical1 RTX3090',remaining_gpu_seconds=remaining))
        def measure():
            while not stop.is_set():
                try:
                    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
                    for line in raw.splitlines():
                        gpu,pid,mem=[x.strip() for x in line.split(',')]
                        if int(pid)==proc.pid:samples.append(dict(time=time.time(),gpu_uuid=gpu,memory_MiB=int(mem)))
                except Exception:pass
                stop.wait(5)
        thread=threading.Thread(target=measure,daemon=True);thread.start()
        try:code=proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            code=-999
        finally:stop.set();thread.join(timeout=10)
    row=dict(label=label,status='completed' if code==0 else 'failed',returncode=code,pid=proc.pid,
             start_unix=start,end_unix=time.time(),wall_seconds=time.time()-start,command=command,
             peak_process_nvidia_MiB=max([s['memory_MiB'] for s in samples],default=None),
             sampled_process_memory=samples,GPU='physical1 RTX3090')
    save_json(attempts/'attempt.json',row);append_cost(row)
    if code:raise RuntimeError(f'{label} exited {code}; see {attempts}')


def freeze_behave():
    assets=[];runs={}
    for dev in ['dev1','dev2']:
        rd=RUN/'runs'/f'behave_{dev}_formal';r=json.loads((rd/'run.json').read_text())
        assert r['status']=='completed' and r['nominal_iterations']==17000
        files=[Path(r['checkpoint']),rd/'effective_config.json',rd/'run.json',
               RUN/'inputs'/f'behave_{dev}'/'manifest.json',RUN/'inputs'/f'behave_{dev}'/'evaluation_manifest.json',
               ROOT/'experiments/structured_hoi_20260923'/f'{dev}_S1_v1/checkpoint_008000.pt',
               ROOT/'experiments/structured_hoi_20260923'/f'{dev}_initialization/initialization.pt']
        for p in files:assets.append(dict(path=str(p),sha256=sha(p)))
        runs[dev]=dict(run_dir=str(rd),checkpoint=r['checkpoint'])
    for name in ['adapter_4dgs.py','train_official.py','render_full_s1.py','render_4dgs.py']:
        p=CODE/name
        assets.append(dict(path=str(p),sha256=sha(p)))
    freeze=RUN/'protocol/finals.json'
    assert not freeze.exists()
    save_json(freeze,dict(status='all_states_frozen',scope='behave_pair',created_unix=time.time(),
                          assets=assets,runs=runs,no_camera1_model_selection=True))
    return freeze


def prepare(include_hos):
    cfg,args=official_config(2)
    datasets=['behave_dev1','behave_dev2']+(['hos_backpack'] if include_hos else [])
    checks=[]
    for ds in datasets:
        p=RUN/'runs'/f'{ds}_check'/'run.json';r=json.loads(p.read_text())
        assert r['status']=='check_completed'
        checks.append(dict(dataset=ds,path=str(p),sha256=sha(p),**{k:r[k] for k in ['seconds','peak_allocated_bytes']}))
    budget_steps=(RUN/'protocol/temporary_steps.jsonl').read_text().splitlines()
    assert len(budget_steps)<=200
    # Small early checks used direct launch; include measured model process times.
    if not LEDGER.exists():
        rows=[dict(label=c['dataset']+'_temporary',wall_seconds=c['seconds'],
                   measurement='model driver wall; Python import overhead not timed',status='completed') for c in checks]
        for p in (RUN/'protocol').glob('behave_*_native_render_check.json'):
            r=json.loads(p.read_text());rows.append(dict(label=p.stem,wall_seconds=r['wall_seconds'],status='completed',measurement='native verification timed block'))
        h0=RUN/'runs/H0_native_first/completion.json'
        if h0.exists():
            r=json.loads(h0.read_text());rows.append(dict(label='H0_native_first',wall_seconds=r['wall_seconds'],status='completed',measurement='native renderer timed block'))
        save_json(LEDGER,rows)
    freeze=RUN/'protocol/training_frozen.json'
    assert not freeze.exists()
    sources=[dict(path=str(CODE/n),sha256=sha(CODE/n)) for n in
             ['adapter_4dgs.py','train_official.py','run_baselines.py','export_protocol.py','export_shared_init.py','export_hos_protocol.py']]
    inputs=[]
    for ds in datasets:
        p=RUN/'inputs'/ds/'manifest.json';m=json.loads(p.read_text())
        inputs.extend([dict(path=str(p),sha256=sha(p)),dict(path=m['point_cloud']['npz_path'],sha256=sha(m['point_cloud']['npz_path']))])
    save_json(freeze,dict(status='training_configuration_frozen',time_unix=time.time(),datasets=datasets,
                         seed=12345,batch_size=2,coarse_iterations=3000,fine_iterations=14000,
                         nominal_steps_per_run=17000,optimizer_steps_per_run=16999,
                         max_formal_runs=3,max_gpu_seconds=43200,max_temporary_steps=200,
                         temporary_steps_used=len(budget_steps),checks=checks,
                         official_commit=OFFICIAL_COMMIT,official_config=vars(cfg),
                         effective_group_config=[vars(a) for a in args],sources=sources,inputs=inputs,
                         timing_estimate={'BEHAVE_each_seconds':[900,3600], 'HOS_seconds':[1200,5400],
                                          'basis':'short measured forward/backward; wider range permits density growth and checkpoints; estimates, not measured full costs'},
                         evaluation='BEHAVE pair frozen together; HOS separate, no test RGB training/reporting/checkpoint selection'))


def run():
    lock=(RUN/'protocol/gpu1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze=json.loads((RUN/'protocol/training_frozen.json').read_text())
    for r in freeze['sources']+freeze['inputs']:assert sha(r['path'])==r['sha256'],r['path']
    save_json(RUN/'pipeline.json',dict(status='running',pid=os.getpid(),started_unix=time.time(),GPU=gpu_snapshot()))
    try:
        for ds in freeze['datasets']:
            rd=RUN/'runs'/f'{ds}_formal'
            gpu_job(ds+'_formal',[PYTHON,str(CODE/'train_official.py'),'--dataset',ds,
                     '--manifest',str(RUN/'inputs'/ds/'manifest.json'),'--output',str(rd)])
            if ds=='behave_dev2':
                final=freeze_behave()
                for dev in ['dev1','dev2']:
                    gpu_job('E0_'+dev,[S1_PYTHON,str(CODE/'render_full_s1.py'),'render','--dev',dev,'--freeze',str(final)])
                    gpu_job('4DGS_render_'+dev,[PYTHON,str(CODE/'render_4dgs.py'),'--dev',dev,
                              '--run-dir',str(RUN/'runs'/f'behave_{dev}_formal'),'--freeze',str(final)])
                # CPU metric computation follows all four frozen output groups.
                with (RUN/'logs/evaluate_frozen.log').open('w') as log:
                    subprocess.run([S1_PYTHON,str(CODE/'evaluate_frozen.py'),'--freeze',str(final)],cwd=ROOT,
                                   env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'2'},stdout=log,stderr=subprocess.STDOUT,check=True)
        save_json(RUN/'pipeline.json',dict(status='formal_and_behave_evaluation_completed',finished_unix=time.time(),
                                         gpu_job_wall_seconds=used_seconds(),HOS_evaluation='separate frozen entry if available'))
    except BaseException:
        save_json(RUN/'pipeline.json',dict(status='failed',finished_unix=time.time(),
                                         gpu_job_wall_seconds=used_seconds(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--include-hos',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.include_hos)
    else:run()
