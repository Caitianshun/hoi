"""One-shot paired training; process exit directly triggers terminal evaluation."""
from pathlib import Path
import os,json,hashlib,time,fcntl,subprocess,traceback
from types import SimpleNamespace
from run_job import run as gpu_job
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
OLD=ROOT/'experiments/baseline_protocol_calibration_20260927/run01'
PYTHON=str(ROOT/'envs/4dgs/bin/python')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def identity(p):
    p=Path(p).absolute();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def save(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def status(stage,**kw):save(RUN/'pipeline.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw))
def free_gpu():
    gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    uuid=gpu.split(',')[0];assert 'RTX 3090' in gpu
    tasks=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    assert uuid not in tasks,'GPU1 is occupied; do not launch a new job'
    return dict(time_unix=time.time(),GPU=gpu,compute_tasks=tasks)
def run():
    lock=(RUN/'protocol/gpu1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'pipeline.json').exists(),'One-shot pipeline; inspect existing state instead of restarting'
    gate=read(RUN/'protocol/formal_gate.json');assert gate['all_conditions_passed']
    for item in gate['frozen_source_files']+gate['frozen_assets']:assert sha(item['path'])==item['sha256'],item['path']
    checks=[free_gpu()];time.sleep(3);checks.append(free_gpu());save(RUN/'protocol/GPU_before_B.json',checks)
    status('B_start',status='running');closed={}
    save(RUN/'continuation.json',dict(status='persistent_pipeline_active',pid=os.getpid(),scope='V5 two fine arms; no automatic restarts',completion_trigger='Synchronous child exit then freeze/render/evaluate; hourly heartbeat is fallback only'))
    try:
        parent=read(RUN/'protocol/fine_transition.json')['parent']['path']
        for branch,policy in [('B_U','uniform_fine'),('B_F','balanced_fine')]:
            used=sum(x['wall_seconds'] for x in map(json.loads,(RUN/'protocol/gpu_jobs.jsonl').read_text().splitlines()))
            if used>=10800:
                closed[branch]=dict(status='not_started_budget_exhausted');continue
            free_gpu();status(branch,status='running',closed=closed)
            attempt=gpu_job(SimpleNamespace(category='B',label=branch+'_formal',command=[str(RUN/'code/train_fine_pair.py'),'--branch-from-coarse',parent,'--output',str(RUN/'runs'/branch),'--policy',policy]))
            rd=RUN/'runs'/branch
            if attempt['returncode']==0 and (rd/'run.json').exists():
                result=read(rd/'run.json');assert result['status']=='completed' and result['nominal_iterations']==14000
                closed[branch]=dict(status='completed',run=identity(rd/'run.json'),attempt=attempt)
            else:
                failure=read(rd/'failure.json') if (rd/'failure.json').exists() else dict(status='external_or_budget_interruption',attempt=attempt)
                closed[branch]=dict(status='failed_no_terminal',failure=failure,attempt=attempt)
            status(branch+'_closed',status='running',closed=closed)
        assets=[];runs={}
        for branch,state in closed.items():
            if state['status']!='completed':continue
            rd=RUN/'runs'/branch;r=read(rd/'run.json');assert sha(r['checkpoint'])==r['checkpoint_sha256']
            assets.extend([identity(r['checkpoint']),identity(rd/'effective_config.json'),identity(rd/'run.json')]);runs[branch]=r
        assets.extend(identity(OLD/'inputs/hos_backpack'/n) for n in ['manifest.json','evaluation_manifest.json'])
        assets.extend(identity(RUN/p) for p in ['protocol/fixed_examples.json','configs/v5.json','protocol/formal_gate.json'])
        save(RUN/'protocol/finals.json',dict(status='all_authorized_attempts_closed',created_unix=time.time(),assets=assets,runs=runs,attempts=closed,no_development_model_selection=True))
        status('terminal_render',status='training_closed_evaluation_running',closed=closed)
        if runs:
            result=gpu_job(SimpleNamespace(category='B',label='render_terminals',command=[str(RUN/'code/render_branches.py')]))
            assert result['returncode']==0,'Rendering failed; do not restart training'
        status('CPU_evaluation',status='training_closed_evaluation_running',closed=closed)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
        with (RUN/'logs/unified_evaluation.log').open('w') as log:
            subprocess.run([PYTHON,str(RUN/'code/evaluate_and_report.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('numerical_evaluation_completed',status='report_and_independent_QA_pending',closed=closed)
        save(RUN/'continuation.json',dict(status='report_and_independent_QA_pending',closed=closed,remaining='Verify hashes/costs/metrics, fixed visual review, DOCX render and every-page QA, minimal ZIP, logs and sync; no further training'))
    except BaseException:
        failure=dict(status='pipeline_failed',time_unix=time.time(),traceback=traceback.format_exc(),closed=closed)
        save(RUN/'pipeline_failure.json',failure);status('failure',**failure)
        save(RUN/'continuation.json',dict(status='exception_requires_interface_review',failure=identity(RUN/'pipeline_failure.json'),no_automatic_training_restart=True));raise
if __name__=='__main__':run()
