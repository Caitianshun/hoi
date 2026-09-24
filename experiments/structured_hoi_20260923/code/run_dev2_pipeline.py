"""Queue independent dev2 prefix/tails/evaluation after dev1 releases GPU1."""
from pathlib import Path
import ctypes,json,os,subprocess,time,traceback
ROOT=Path('/home/cai_tianshun/Project/HOI');E=ROOT/'experiments/structured_hoi_20260923'

def wait_finished(path):
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(0)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        if libc.inotify_add_watch(fd,str(path.parent).encode(),0x8|0x80|0x100)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        while True:
            if path.exists():
                d=json.loads(path.read_text())
                if d['status']=='completed':return
                if d['status']=='failed':raise RuntimeError('Preceding dev1 pair failed; GPU queue stopped')
            os.read(fd,65536)
    finally:os.close(fd)

def main():
    status=E/'dev2_pipeline.json';rec=dict(status='queued',pid=os.getpid(),gpu=1,host=os.uname().nodename,stages={})
    def save():
        p=status.with_suffix('.tmp');p.write_text(json.dumps(rec,indent=2)+'\n');p.replace(status)
    start=time.perf_counter();save();python=ROOT/'envs/gvhmr/bin/python'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'8','PYTHONUNBUFFERED':'1'}
    try:
        wait_finished(E/'dev1_pair_pipeline.json')
        # The preceding process released its CUDA context before marking completed.
        gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        rec['launch_compute_processes']=gpu;save()
        if gpu.strip():raise RuntimeError('GPU1 acquired by another compute process; refusing concurrent launch: '+gpu)
        init=E/'dev2_initialization/initialization.pt'
        jobs=[('prefix',[python,E/'code/train_structured.py','--init',init,'--output',E/'dev2_prefix01','--branch','prefix','--stop','6000']),
              ('pair',[python,E/'code/run_suffix_pair.py','--dev','dev2','--init',init,'--prefix',E/'dev2_prefix01/checkpoint_006000.pt'])]
        for name,cmd in jobs:
            rec.update(status='running',current_stage=name);save();t=time.perf_counter()
            with (E/f'dev2_{name}_orchestrator.log').open('wb') as out:subprocess.run([str(x) for x in cmd],stdout=out,stderr=subprocess.STDOUT,env=env,cwd=ROOT,check=True)
            rec['stages'][name]={'status':'completed','wall_seconds':time.perf_counter()-t,'command':[str(x) for x in cmd]};save()
        rec['status']='completed'
    except BaseException as ex:
        rec.update(status='failed',error=repr(ex),traceback=traceback.format_exc());raise
    finally:rec['wall_seconds_including_queue']=time.perf_counter()-start;save()

if __name__=='__main__':main()
