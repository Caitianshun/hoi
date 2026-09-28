"""Persistent bounded V8B U/Q pipeline; successful exit chains freeze/render/eval."""
from common import *
import subprocess,time,traceback,fcntl
B=RUN/'tennis'
def status(phase,**kw):
    row=dict(status='V8B_running',V8A='evaluated_verified_and_assessed',phase=phase,pid=os.getpid(),time_unix=time.time(),**kw)
    save_json(B/'pipeline.json',row);save_json(RUN/'pipeline.json',row)
def gpu(label,script,args=()):
    status(label);jobs=B/'protocol/gpu_jobs.jsonl';prior=[json.loads(x) for x in jobs.read_text().splitlines()] if jobs.exists() else []
    remaining=read(RUN/'configs/v8_tennis.json')['GPU_seconds']-sum(x['wall_seconds'] for x in prior);assert remaining>0
    out=B/'logs'/label;out.mkdir(parents=True,exist_ok=False);command=[str(ROOT/'envs/4dgs/bin/python'),str(RUN/'code'/script),*args]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4');env.pop('CUDA_LAUNCH_BLOCKING',None);start=time.time();timed_out=False
    with (out/'console.log').open('w') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env);save_json(out/'running.json',dict(pid=p.pid,command=command,start_unix=start))
        try:code=p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out=True;p.terminate()
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();code=p.wait()
    row=dict(label=label,command=command,pid=p.pid,returncode=code,start_unix=start,end_unix=time.time(),wall_seconds=time.time()-start,physical_GPU=1,budget_timeout=timed_out)
    with jobs.open('a') as f:f.write(json.dumps(row)+'\n')
    save_json(out/'attempt.json',row);assert code==0,(label,row)
def main():
    lock=(B/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert read(RUN/'quality_decision.json')['independent_confirmation']['arms']==['B_U','B_Q']
        assert read(B/'protocol/preprocessing.json')['status']=='completed'
        assert not (B/'runs').exists();status('startup_GPU_check')
        readings=[]
        for i in range(2):
            gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
            assert 'GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec' not in processes
            row=next(s for s in gpu_info.splitlines() if 'GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec' in s)
            parts=[x.strip() for x in row.split(',')];assert parts[0]=='1' and parts[2]=='NVIDIA GeForce RTX 3090';assert int(parts[3].split()[0])<100 and int(parts[4].split()[0])<5
            readings.append(dict(time_unix=time.time(),GPUs=gpu_info,compute_processes=processes,loadavg=Path('/proc/loadavg').read_text()))
            if i==0:time.sleep(15)
        save_json(B/'protocol/GPU_startup.json',dict(readings=readings))
        gpu('coarse_U','train_tennis.py',['--stage','coarse','--objective','U'])
        gpu('B_U','train_tennis.py',['--stage','fine','--objective','U'])
        gpu('B_Q','train_tennis.py',['--stage','fine','--objective','Q'])
        assets=[]
        for arm in ['B_U','B_Q']:
            rd=B/'runs'/arm;r=read(rd/'run.json');assert r['status']=='completed';assets.extend([identity(r['checkpoint']),identity(rd/'run.json'),identity(rd/'effective_config.json')])
        save_json(B/'protocol/finals.json',dict(status='both_terminals_frozen',assets=assets,freeze_time_unix=time.time(),development_evaluation_started=False))
        gpu('terminal_render','render_tennis.py')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='4'
        for script in ['evaluate_tennis.py','verify_tennis.py']:
            status(script)
            with (B/'logs'/(script+'.log')).open('w') as f:subprocess.run([str(ROOT/'envs/4dgs/bin/python'),str(RUN/'code'/script)],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        row=dict(status='V8B_evaluated_verified_pending_visual_assessment_and_delivery',phase='assessment',pid=os.getpid(),time_unix=time.time(),evaluation=identity(B/'evaluation_summary.json'),verification=identity(B/'protocol/independent_verification.json'))
        save_json(B/'pipeline.json',row);save_json(RUN/'pipeline.json',row)
    except BaseException:
        row=dict(status='V8B_failed',V8A='completed_evaluation_preserved',pid=os.getpid(),time_unix=time.time(),traceback=traceback.format_exc())
        save_json(B/'pipeline.json',row);save_json(RUN/'pipeline.json',row);raise
if __name__=='__main__':main()
