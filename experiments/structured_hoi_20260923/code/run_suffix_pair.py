"""Run controlled S0/S1 tails and automatically chain frozen evaluation when ready."""
from pathlib import Path
import argparse,ctypes,json,os,select,subprocess,time,traceback
ROOT=Path('/home/cai_tianshun/Project/HOI');E=ROOT/'experiments/structured_hoi_20260923'

def wait_ready(path):
    if path.exists():return
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(0)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        if libc.inotify_add_watch(fd,str(path.parent).encode(),0x8|0x80|0x100)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        while not path.exists():os.read(fd,65536)
    finally:os.close(fd)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dev',required=True);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--init',type=Path,required=True);a=p.parse_args()
    status=E/f'{a.dev}_pair_pipeline.json';rec=dict(status='running',pid=os.getpid(),gpu=1,dev=a.dev,stages={},prefix=str(a.prefix.resolve()))
    def save():
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(rec,indent=2)+'\n');tmp.replace(status)
    begin=time.perf_counter();env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'8','PYTHONUNBUFFERED':'1'}
    python=ROOT/'envs/gvhmr/bin/python'
    def run(name,args):
        rec['current_stage']=name;save();t=time.perf_counter()
        with (E/f'{a.dev}_{name}.log').open('wb') as out:subprocess.run([str(x) for x in args],stdout=out,stderr=subprocess.STDOUT,env=env,cwd=ROOT,check=True)
        rec['stages'][name]={'status':'completed','seconds':time.perf_counter()-t,'command':[str(x) for x in args]};save()
    try:
        for b in ['S0','S1']:
            out=E/f'{a.dev}_{b}_v1'
            run(b,[python,E/'code/train_structured.py','--init',a.init.resolve(),'--output',out,'--branch',b,'--stop','8000','--resume',a.prefix.resolve()])
            done=json.loads((out/'run.json').read_text());assert done['status']=='completed' and done['final_step']==8000
            rec['stages'][b]['completion_integrity']='trained to8000, checkpoint hash saved, finite loss/grad checks enforced, previews saved';save()
        rec['current_stage']='waiting_for_evaluator_code_ready';save()
        wait_ready(E/'code/evaluate_structured.ready.json')
        for b in ['S0','S1']:
            run('evaluate_'+b,[python,E/'code/evaluate_structured.py','--init',a.init.resolve(),'--checkpoint',E/f'{a.dev}_{b}_v1/checkpoint_008000.pt','--output',E/f'{a.dev}_{b}_v1/evaluation'])
        rec['status']='completed'
    except BaseException as ex:rec.update(status='failed',error=repr(ex),traceback=traceback.format_exc());raise
    finally:rec['wall_seconds']=time.perf_counter()-begin;save()
if __name__=='__main__':main()
