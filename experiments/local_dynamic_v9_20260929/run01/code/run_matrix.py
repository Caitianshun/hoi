"""Persistent V9 controller with exit-triggered phases and a global GPU ledger."""
from common import *
import argparse, subprocess, traceback, fcntl, shutil


def status(phase, **extra):
    save_json(RUN/'pipeline.json',dict(status='running',phase=phase,pid=os.getpid(),time_unix=time.time(),**extra))


def check_gpu(twice=False):
    readings=[]
    for i in range(2 if twice else 1):
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        selected=next(row for row in output.splitlines() if row.split(',')[0].strip()==str(config()['hardware']['physical_GPU']))
        fields=[x.strip() for x in selected.split(',')]
        assert fields[2]==config()['hardware']['name']
        assert fields[1] not in procs and int(fields[3].split()[0])<100 and int(fields[4].split()[0])<5,(selected,procs)
        readings.append(dict(time_unix=time.time(),GPUs=output,compute_processes=procs,loadavg=Path('/proc/loadavg').read_text()))
        if twice and i==0:time.sleep(15)
    append_json(RUN/'protocol/resource_checks.jsonl',dict(readings=readings))


def job(label, script, args=(), gpu=True):
    status(label)
    ledger=RUN/'protocol'/('GPU_jobs.jsonl' if gpu else 'CPU_jobs.jsonl')
    prior=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    budget=config()['budgets']['GPU_seconds']-sum(x['wall_seconds'] for x in prior) if gpu else None
    if gpu:
        assert budget>0;check_gpu()
    logs=RUN/'logs'/label;logs.mkdir(parents=True,exist_ok=False)
    command=[str(ROOT/'envs/4dgs/bin/python'),'-u',str(RUN/'code'/script),*args]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1' if gpu else '',OMP_NUM_THREADS='4')
    env.pop('CUDA_LAUNCH_BLOCKING',None)
    start=time.time();timed_out=False
    with (logs/'console.log').open('w') as f:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        save_json(logs/'running.json',dict(pid=p.pid,command=command,start_unix=start))
        try:code=p.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            timed_out=True;p.terminate()
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();code=p.wait()
    row=dict(label=label,command=command,pid=p.pid,returncode=code,start_unix=start,end_unix=time.time(),
             wall_seconds=time.time()-start,physical_GPU=1 if gpu else None,budget_timeout=timed_out)
    append_json(ledger,row);save_json(logs/'attempt.json',row)
    assert code==0,(label,code)


def prepare():
    assert not (RUN/'protocol/prepared.json').exists()
    check_gpu(twice=True)
    assert shutil.disk_usage(ROOT).free>100*(1<<30)
    original=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    assert original==OFFICIAL_COMMIT
    save_json(RUN/'protocol/environment.json',dict(time_unix=time.time(),host=os.uname().nodename,
        python=str(ROOT/'envs/4dgs/bin/python'),official_commit=original,
        project_git_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        free_disk_bytes=shutil.disk_usage(ROOT).free,config=identity(RUN/'configs/v9.json'),
        attachment=identity(RUN/'protocol/user_guidance_source.md')))
    for scene in config()['scenes']:
        job('prepare_'+scene,'prepare_initialization.py',['--scene',scene])
    save_json(RUN/'protocol/prepared.json',dict(status='prepared',scenes=list(config()['scenes']),time_unix=time.time()))


def verify():
    assert read(RUN/'protocol/prepared.json')['status']=='prepared'
    job('integrated_acceptance_finish','verify.py')
    assert read(RUN/'protocol/module_acceptance.json')['status']=='passed'


def freeze():
    assets=[];runs=[]
    for scene in config()['scenes']:
        for mode in config()['modes']:
            rd=scene_dir(scene)/'runs'/mode;r=read(rd/'run.json')
            assert r['status']=='completed' and r['completed_updates']==config()['schedule']['fine_updates']
            assert sha(r['checkpoint']['path'])==r['checkpoint']['sha256']
            assets.extend([r['checkpoint'],identity(rd/'run.json'),identity(rd/'effective_config.json')])
            runs.append(dict(scene=scene,mode=mode,checkpoint=r['checkpoint']))
    save_json(RUN/'protocol/finals.json',dict(status='all_six_terminals_frozen',runs=runs,assets=assets,
        freeze_time_unix=time.time(),development_evaluation_started=False))


def train():
    assert read(RUN/'protocol/module_acceptance.json')['status']=='passed'
    prediction=read(RUN/'protocol/cost_prediction.json')
    assert prediction['status']=='within_budget',prediction
    assert not (RUN/'protocol/finals.json').exists()
    for scene in config()['scenes']:
        job(scene+'_BG','train_scene.py',['--scene',scene,'--mode','BG'])
    for scene in config()['scenes']:
        for mode in config()['modes']:
            job(scene+'_'+mode,'train_scene.py',['--scene',scene,'--mode',mode])
    freeze()


def evaluate():
    assert read(RUN/'protocol/finals.json')['status']=='all_six_terminals_frozen'
    job('all_terminal_render','render_terminal.py')
    job('all_terminal_evaluation','evaluate_terminal.py',gpu=False)
    job('independent_verification','verify_results.py',gpu=False)


def report():
    assert read(RUN/'protocol/independent_verification.json')['status']=='passed'
    job('prepare_report','prepare_report.py',gpu=False)
    save_json(RUN/'pipeline.json',dict(status='evaluated_pending_visual_review_and_delivery_QA',
        phase='report_review',time_unix=time.time(),pid=os.getpid()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','verify','train','evaluate','report','all','train-to-report'],required=True)
    a=p.parse_args();lock=(RUN/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        phases=['prepare','verify','train','evaluate','report'] if a.phase=='all' else ['train','evaluate','report'] if a.phase=='train-to-report' else [a.phase]
        for phase in phases:globals()[phase]()
        if phases[-1]!='report':
            save_json(RUN/'pipeline.json',dict(status='phase_completed',phase=phases[-1],pid=os.getpid(),time_unix=time.time()))
    except BaseException:
        save_json(RUN/'pipeline.json',dict(status='failed',requested_phase=a.phase,pid=os.getpid(),
            time_unix=time.time(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
