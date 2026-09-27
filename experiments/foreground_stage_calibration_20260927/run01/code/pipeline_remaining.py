"""Continue unaffected W_all exactly once and freeze successes plus failures."""
from common import *
from jobs import gpu_job,PYTHON
import os,time,fcntl,traceback,subprocess

def run():
    lock=(RUN/'protocol/gpu1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=read(RUN/'protocol/training_frozen.json')
    for a in frozen['assets']+frozen['training_sources']:assert sha(a['path'])==a['sha256'],a['path']
    assert not list((RUN/'runs/W_fine').glob('*.pt'))
    assert read(RUN/'runs/W_fine/failure.json')['status']=='failed'
    save_json(RUN/'pipeline.json',dict(status='running',pid=os.getpid(),stage='W_all',W_fine='failed_no_legal_resume',updated_unix=time.time()))
    try:
        try:gpu_job('W_all_formal',[PYTHON,str(RUN/'code/train_stage_audited.py'),'--output',str(RUN/'runs/W_all'),'--policy','balanced_all'])
        except RuntimeError:
            assert (RUN/'runs/W_all/failure.json').exists()
        assets=[];runs={};failures={}
        for branch in ['W_fine','W_all']:
            rd=RUN/'runs'/branch
            if (rd/'run.json').exists():
                r=read(rd/'run.json');assert r['status']=='completed';runs[branch]=r
                assets += [identity(r['checkpoint']),identity(rd/'effective_config.json'),identity(rd/'run.json')]
            else:
                failures[branch]=read(rd/'failure.json');assets += [identity(rd/'failure.json'),identity(rd/'effective_config.json')]
        assets += [identity(OLD/'inputs/hos_backpack'/n) for n in ['manifest.json','evaluation_manifest.json']]
        assets += [identity(RUN/'protocol/fixed_examples.json'),identity(RUN/'configs/v4.json')]
        save_json(RUN/'protocol/finals.json',dict(status='all_authorized_attempts_closed',created_unix=time.time(),assets=assets,runs=runs,failures=failures,no_failed_branch_replacement=True))
        save_json(RUN/'pipeline.json',dict(status='attempts_closed_evaluation_running',pid=os.getpid(),updated_unix=time.time()))
        gpu_job('render_available_terminals',[PYTHON,str(RUN/'code/render_branches.py')])
        gpu_job('available_frozen_state_probes',[PYTHON,str(RUN/'code/probe_training_state.py')])
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
        with (RUN/'logs/unified_evaluation.log').open('w') as log:
            subprocess.run([PYTHON,str(RUN/'code/evaluate_and_report.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        save_json(RUN/'pipeline.json',dict(status='available_numerical_evaluation_completed',finished_unix=time.time(),report_status='interpretation_and_DOCX_QA_pending',failed_branches=list(failures)))
    except BaseException:
        save_json(RUN/'continuation_failure.json',dict(status='failed',time_unix=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':run()
