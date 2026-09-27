"""Fixed V4 matrix with completion-triggered freeze, rendering and evaluation."""
from common import *
from jobs import gpu_job,PYTHON
import os,time,fcntl,traceback,subprocess
def run():
    lock=(RUN/'protocol/gpu1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=read(RUN/'protocol/training_frozen.json');assert frozen['status']=='frozen'
    for x in frozen['assets']+frozen['training_sources']:assert sha(x['path'])==x['sha256'],x['path']
    save_json(RUN/'pipeline.json',dict(status='running',pid=os.getpid(),started_unix=time.time(),stage='W_fine'))
    try:
        code=RUN/'code/train_stage.py';parent=OLD/'runs/hos_backpack_formal/checkpoint_coarse_003000.pt'
        for branch,policy in [('W_fine','balanced_fine'),('W_all','balanced_all')]:
            command=[PYTHON,str(code),'--output',str(RUN/'runs'/branch),'--policy',policy]
            if branch=='W_fine':command+=['--branch-from-coarse',str(parent)]
            save_json(RUN/'pipeline.json',dict(status='running',pid=os.getpid(),stage=branch,updated_unix=time.time()))
            gpu_job(branch+'_formal',command)
        assets=[];runs={}
        for branch,nominal in [('W_fine',14000),('W_all',17000)]:
            rd=RUN/'runs'/branch;r=read(rd/'run.json');assert r['status']=='completed' and r['nominal_iterations']==nominal and r['optimizer_updates']==nominal-1
            assets += [identity(r['checkpoint']),identity(rd/'effective_config.json'),identity(rd/'run.json')];runs[branch]=r
        assets += [identity(OLD/'inputs/hos_backpack'/n) for n in ['manifest.json','evaluation_manifest.json']]
        assets += [identity(RUN/'protocol/fixed_examples.json'),identity(RUN/'configs/v4.json')]
        save_json(RUN/'protocol/finals.json',dict(status='both_new_terminal_states_frozen',created_unix=time.time(),assets=assets,runs=runs,no_retained_set_model_selection=True))
        save_json(RUN/'pipeline.json',dict(status='training_completed_evaluation_running',pid=os.getpid(),updated_unix=time.time()))
        gpu_job('render_terminals',[PYTHON,str(RUN/'code/render_branches.py')])
        gpu_job('frozen_state_probes',[PYTHON,str(RUN/'code/probe_training_state.py')])
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
        with (RUN/'logs/unified_evaluation.log').open('w') as log:
            subprocess.run([PYTHON,str(RUN/'code/evaluate_and_report.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        save_json(RUN/'pipeline.json',dict(status='training_rendering_numerical_evaluation_completed',finished_unix=time.time(),report_status='interpretation_and_DOCX_QA_pending'))
    except BaseException:
        save_json(RUN/'pipeline_failure.json',dict(status='failed',time_unix=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':run()
