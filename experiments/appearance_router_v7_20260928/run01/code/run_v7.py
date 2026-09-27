"""Persistent single entry; child exit directly triggers the next required stage."""
from pathlib import Path
import argparse,subprocess,os,json,time,fcntl,traceback
from types import SimpleNamespace
from run_job import run as gpu_job
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2];PY=str(ROOT/'envs/4dgs/bin/python')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def status(stage,**kw):save(RUN/'pipeline.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw))
def free_gpu():
    s=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    assert 'RTX 3090' in s
    p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    assert s.split(',')[0] not in p
    return dict(time=time.time(),gpu=s,processes=p)
def cpu(name):
    with (RUN/'logs'/(name+'.log')).open('w') as f:
        subprocess.run([PY,str(RUN/'code'/name)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2'})
def gpu(label,cat,args):
    free_gpu();status(label,status='running')
    return gpu_job(SimpleNamespace(label=label,category=cat,command=args))
def run(a):
    assert Path(a.config).resolve()==(RUN/'configs/v7.json').resolve()
    lock=(RUN/'protocol/gpu1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'pipeline.json').exists(),'Existing pipeline must be inspected, never restarted blindly'
    observations=[free_gpu()];time.sleep(3);observations.append(free_gpu());save(RUN/'protocol/GPU_before.json',observations)
    try:
        status('preflight',status='running');cpu('preflight.py')
        assert gpu('integrated_probe','D',[str(RUN/'code/module_acceptance.py')])['returncode']==0
        args=[str(RUN/'code/train_routed_fine.py'),'--purpose','diagnostic','--target','1128','--save-iterations','1104']
        p=ROOT/'experiments/numerical_stability_calibration_20260927/run01/runs/B_U/checkpoint_fine_001000.pt'
        assert gpu('regression_128','D',args+['--start-kind','diagnostic_parent','--parent',str(p),'--output',str(RUN/'diagnostics/regression')])['returncode']==0
        p=RUN/'diagnostics/regression/checkpoint_fine_001104.pt'
        assert gpu('restore_24','D',args+['--start-kind','own_checkpoint','--parent',str(p),'--output',str(RUN/'diagnostics/regression_resume')])['returncode']==0
        status('acceptance_review',status='running');cpu('finish_acceptance.py')
        args=[str(RUN/'code/train_routed_fine.py'),'--purpose','formal','--target','14000']
        p=ROOT/'experiments/numerical_stability_calibration_20260927/run01/protocol/shared_fine_initial.pt';out=RUN/'runs/M1'
        result=gpu('M1_formal','C',args+['--start-kind','shared_fine','--parent',str(p),'--output',str(out)])
        if result['returncode']!=0:
            # Only a genuine external interruption may consume the single replay allowance.
            assert not (out/'failure.json').exists() and not result.get('budget_timeout') and result['returncode'] in [-9,-15,137,143]
            latest=read(out/'latest_checkpoint.json');assert latest['resumable']
            attempts=[json.loads(s) for s in (RUN/'protocol/C_steps.jsonl').read_text().splitlines()]
            assert len(attempts)-latest['iteration']<=256
            save(RUN/'protocol/external_resume_authorization.json',dict(resume_number=1,checkpoint=latest['checkpoint'],external_attempt=result))
            p=Path(latest['checkpoint']['path']);out=RUN/'runs/M1_resume'
            result=gpu('M1_external_resume','C',args+['--start-kind','own_checkpoint','--parent',str(p),'--output',str(out)])
            assert result['returncode']==0
        record=read(out/'run.json');assert record['status']=='completed' and record['nominal_iterations']==14000
        save(RUN/'formal_terminal.json',dict(status='completed',directory=str(out),run=record,process=result))
        status('freeze_terminal',status='running');cpu('freeze_terminal.py')
        assert gpu('terminal_render','E',[str(RUN/'code/render_terminal.py')])['returncode']==0
        status('terminal_evaluation',status='running');cpu('evaluate_terminal.py')
        status('metrics_completed',status='document_and_visual_QA_pending')
        save(RUN/'continuation.json',dict(status='document_and_visual_QA_pending',remaining='Fixed figure visual review, DOCX render and page QA, feedback ZIP, independent CPU verification, final logs and source sync',automatic_new_training=False))
    except BaseException:
        failure=dict(status='technical_failure',traceback=traceback.format_exc(),time_unix=time.time());save(RUN/'pipeline_failure.json',failure);status('failure',**failure);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args())
