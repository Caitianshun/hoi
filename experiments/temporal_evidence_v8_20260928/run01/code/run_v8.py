"""Persistent event-chained V8A pipeline, full matrix before development evaluation."""
from common import *
import argparse,subprocess,time,traceback,types,fcntl
from run_job import run as job

def status(phase,**kw):save_json(RUN/'pipeline.json',dict(status='running',phase=phase,pid=os.getpid(),time_unix=time.time(),**kw))
def gpu(label,cat,script,args=()):
    status(label);r=job(types.SimpleNamespace(label=label,category=cat,command=[str(RUN/'code'/script),*map(str,args)]));assert r['returncode']==0,(label,r)
def train(arm,mode,parent,kind):gpu(arm,'C','train_temporal.py',['--parent',parent,'--output',RUN/'runs'/arm,'--start-kind',kind,'--purpose','formal','--mode',mode,'--target',14000])
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(RUN/'configs/v8.json'));a=p.parse_args();assert Path(a.config).resolve()==(RUN/'configs/v8.json').resolve()
    lock=(RUN/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert read(RUN/'module_acceptance.json')['status']=='pass';assert not any((RUN/'runs'/n).exists() for n in ['B_Q','T_plain','T_mix'])
        cfg=read(RUN/'configs/v8.json');status('startup_GPU_check')
        for i in range(2):
            output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True);assert 'GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec' not in output
            if i==0:time.sleep(15)
        # All interface checks already completed once; no repeated diagnostic budget.
        train('B_Q','off',V5/'protocol/shared_fine_initial.pt','shared_fine')
        gpu('lambda_calibration','D','calibrate_lambda.py')
        parent=RUN/'runs/B_Q/checkpoint_fine_001000.pt'
        train('T_plain','plain',parent,'Q_prefix');train('T_mix','mixture',parent,'Q_prefix')
        assets=[]
        for arm in ['B_Q','T_plain','T_mix']:
            rd=RUN/'runs'/arm;r=read(rd/'run.json');assert r['status']=='completed';assets.extend([identity(r['checkpoint']),identity(rd/'run.json'),identity(rd/'effective_config.json')])
        save_json(RUN/'protocol/finals.json',dict(status='all_three_terminals_frozen',assets=assets,freeze_time_unix=time.time(),development_evaluation_started=False))
        gpu('terminal_render','E','render_terminal.py');gpu('temporal_explanations','E','explain_temporal.py')
        status('CPU_evaluation');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        with (RUN/'logs/CPU_evaluation.log').open('w') as f:subprocess.run([str(ROOT/'envs/4dgs/bin/python'),str(RUN/'code/evaluate_terminal.py')],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        save_json(RUN/'pipeline.json',dict(status='V8A_evaluated_pending_quality_assessment_and_delivery',phase='assessment',pid=os.getpid(),time_unix=time.time(),evaluation=identity(RUN/'evaluation_summary.json')))
    except BaseException:
        save_json(RUN/'pipeline.json',dict(status='failed',pid=os.getpid(),time_unix=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
