"""Persistent serialized GPU jobs with an inclusive wall-time budget ledger."""
from common import *
import os,subprocess,threading,time,fcntl,argparse,traceback
PYTHON=str(ROOT/'envs/4dgs/bin/python')
GPU_UUID='GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec'
def gpu_job(label,command):
    ledger=RUN/'protocol/gpu_cost_ledger.json';rows=read(ledger) if ledger.exists() else []
    remaining=10800-sum(x['wall_seconds'] for x in rows);assert remaining>0
    directory=RUN/'logs'/label;directory.mkdir(parents=True,exist_ok=True);assert not (directory/'attempt.json').exists()
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    assert not any(GPU_UUID in line for line in raw.splitlines()),'GPU1 occupied; no preemption'
    start=time.time();stop=threading.Event();samples=[]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',V4_REMAINING_GPU_SECONDS=str(remaining))
    with (directory/'stdout.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save_json(directory/'attempt.json',dict(status='running',pid=proc.pid,start_unix=start,command=command,remaining_gpu_seconds=remaining,GPU_UUID=GPU_UUID))
        def sample():
            while not stop.is_set():
                try:
                    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
                    for line in output.splitlines():
                        uuid,pid,mem=[x.strip() for x in line.split(',')]
                        if int(pid)==proc.pid:samples.append(dict(time_unix=time.time(),gpu_uuid=uuid,memory_MiB=int(mem)))
                except Exception:pass
                stop.wait(5)
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        try:code=proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            code=-999
        finally:stop.set();thread.join(timeout=10)
    row=dict(label=label,status='completed' if code==0 else 'failed',returncode=code,pid=proc.pid,start_unix=start,end_unix=time.time(),wall_seconds=time.time()-start,
        command=command,GPU_UUID=GPU_UUID,peak_process_nvidia_MiB=max([s['memory_MiB'] for s in samples],default=None),samples=samples)
    save_json(directory/'attempt.json',row);rows.append(row);save_json(ledger,rows)
    if code:raise RuntimeError(f'{label} failed; preserve same-state recovery evidence')
def checks():
    code=RUN/'code/train_stage.py';parent=OLD/'runs/hos_backpack_formal/checkpoint_coarse_003000.pt';out=RUN/'runs/W_fine_check'
    base=[PYTHON,str(code),'--output',str(out),'--policy','balanced_fine','--branch-from-coarse',str(parent),'--check','--fine-steps','10']
    gpu_job('W_fine_check_part1',base+['--stop-after','5'])
    stop=read(out/'planned_stop.json');assert stop['optimizer_updates']==5
    gpu_job('W_fine_check_resume',base+['--resume',stop['checkpoint']])
    gpu_job('W_all_check',[PYTHON,str(code),'--output',str(RUN/'runs/W_all_check'),'--policy','balanced_all','--check','--coarse-steps','5','--fine-steps','5'])
    save_json(RUN/'protocol/checks_completed.json',dict(status='passed',checks=[identity(out/'run.json'),identity(out/'resume_check.json'),identity(out/'branch_validation.json'),identity(RUN/'runs/W_all_check/run.json')],temporary_steps=len((RUN/'protocol/temporary_steps.jsonl').read_text().splitlines()),formal_steps=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['checks']);a=p.parse_args()
    lock=(RUN/'protocol/gpu1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:checks()
    except BaseException:save_json(RUN/'protocol/checks_failure.json',dict(traceback=traceback.format_exc(),time_unix=time.time()));raise
