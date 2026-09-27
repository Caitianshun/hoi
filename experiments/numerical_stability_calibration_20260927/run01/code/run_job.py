"""Persistent bounded GPU job accounting; no automatic restarts."""
from pathlib import Path
import argparse,subprocess,json,os,time
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
def run(a):
    jobs=RUN/'protocol/gpu_jobs.jsonl';prior=[json.loads(s) for s in jobs.read_text().splitlines()] if jobs.exists() else []
    total=sum(x['wall_seconds'] for x in prior);used=sum(x['wall_seconds'] for x in prior if x['category']=='A')
    remaining=min(10800-total,1800-used) if a.category=='A' else 10800-total
    assert remaining>0
    out=RUN/'logs'/a.label;out.mkdir(parents=True,exist_ok=False)
    command=[str(ROOT/'envs/4dgs/bin/python'),*a.command]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4')
    if a.category=='A':env['CUDA_LAUNCH_BLOCKING']='1'
    else:env.pop('CUDA_LAUNCH_BLOCKING',None)
    start=time.time()
    with (out/'console.log').open('w') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        (out/'running.json').write_text(json.dumps(dict(pid=p.pid,start=start,command=command)))
        try:code=p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:p.terminate();code=p.wait(timeout=20)
    row=dict(label=a.label,category=a.category,command=command,pid=p.pid,returncode=code,start_unix=start,end_unix=time.time(),wall_seconds=time.time()-start,physical_GPU=1)
    with jobs.open('a') as f:f.write(json.dumps(row)+'\n')
    (out/'attempt.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--category',choices=['A','B'],default='A');p.add_argument('command',nargs=argparse.REMAINDER);run(p.parse_args())
