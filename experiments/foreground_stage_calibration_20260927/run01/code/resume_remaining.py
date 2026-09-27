"""One own-state technical recovery, with synchronous CUDA error evidence."""
from common import *
from jobs import gpu_job,PYTHON
import os,time,fcntl,traceback,shutil,subprocess
def run():
    lock=(RUN/'protocol/gpu1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rd=RUN/'runs/W_all';audit=read(RUN/'protocol/W_all_recovery_audit.json')
    assert sha(rd/'recovery_latest.pt')==audit['checkpoint_sha256']
    for a in read(RUN/'protocol/training_frozen.json')['training_sources']:assert sha(a['path'])==a['sha256']
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    save_json(RUN/'pipeline.json',dict(status='running_same_state_recovery',pid=os.getpid(),stage='W_all',checkpoint=identity(rd/'recovery_latest.pt'),updated_unix=time.time()))
    ok=False
    try:
        gpu_job('W_all_same_state_recovery',[PYTHON,str(RUN/'code/train_stage_audited.py'),'--output',str(rd),'--policy','balanced_all','--resume',str(rd/'recovery_latest.pt')]);ok=True
    except RuntimeError:assert (rd/'failure.json').exists()
    final=read(RUN/'protocol/finals.json');final['created_unix']=time.time();final['recovery_attempt']=identity(RUN/'logs/W_all_same_state_recovery/attempt.json')
    final['assets']=[a for a in final['assets'] if not a['path'].endswith('/W_all/failure.json')]
    if ok:
        result=read(rd/'run.json');final['runs']['W_all']=result;final['failures'].pop('W_all',None)
        final['assets'] += [identity(result['checkpoint']),identity(rd/'run.json')]
    else:
        final['failures']['W_all']=read(rd/'failure.json');final['assets'].append(identity(rd/'failure.json'))
    save_json(RUN/'protocol/finals.json',final)
    if ok:
        archive=RUN/'protocol/pre_recovery_baseline_only_evaluation';archive.mkdir(exist_ok=True)
        for name in ['evaluation','metrics_per_frame.csv','paired_differences.csv','figure_manifest.json']:
            p=RUN/name
            if p.exists():shutil.move(p,archive/p.name)
        (RUN/'evaluation').mkdir()
        gpu_job('render_recovered_terminal',[PYTHON,str(RUN/'code/render_branches.py')])
        gpu_job('probe_recovered_terminal',[PYTHON,str(RUN/'code/probe_training_state.py')])
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
        with (RUN/'logs/recovered_evaluation.log').open('w') as log:subprocess.run([PYTHON,str(RUN/'code/evaluate_and_report.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    save_json(RUN/'pipeline.json',dict(status='available_numerical_evaluation_completed',failed_branches=list(final['failures']),same_state_recovery_succeeded=ok,finished_unix=time.time(),report_status='DOCX_and_evidence_pending'))
if __name__=='__main__':
    try:run()
    except BaseException:save_json(RUN/'recovery_pipeline_failure.json',dict(time_unix=time.time(),traceback=traceback.format_exc()));raise
