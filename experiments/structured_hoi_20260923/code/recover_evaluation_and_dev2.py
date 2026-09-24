"""Preserve first export failure; rerun only evaluation, then queued dev2."""
from pathlib import Path
import json,os,subprocess,time,traceback
ROOT=Path('/home/cai_tianshun/Project/HOI');E=ROOT/'experiments/structured_hoi_20260923'

def main():
    status=E/'recovery_and_dev2_pipeline.json';rec=dict(status='running',pid=os.getpid(),gpu=1,stages={},reason='First dev1 exporter failed on noncontiguous OpenCV annotation array; training checkpoints intact. Original failure preserved.')
    def save():
        p=status.with_suffix('.tmp');p.write_text(json.dumps(rec,indent=2)+'\n');p.replace(status)
    start=time.perf_counter();save();python=ROOT/'envs/gvhmr/bin/python'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'8','PYTHONUNBUFFERED':'1'}
    try:
        gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        rec['launch_compute_processes']=gpu;save()
        if gpu.strip():raise RuntimeError('GPU1 occupied; refusing concurrent launch: '+gpu)
        init=E/'dev2_initialization/initialization.pt';jobs=[]
        for b in ['S0','S1']:
            jobs.append(('dev1_evaluate_'+b,[python,E/'code/evaluate_structured.py','--init',E/'dev1_initialization/initialization.pt','--checkpoint',E/f'dev1_{b}_v1/checkpoint_008000.pt','--output',E/f'dev1_{b}_v1/evaluation_v2']))
        jobs.extend([('dev2_prefix',[python,E/'code/train_structured.py','--init',init,'--output',E/'dev2_prefix01','--branch','prefix','--stop','6000']),
                     ('dev2_pair',[python,E/'code/run_suffix_pair.py','--dev','dev2','--init',init,'--prefix',E/'dev2_prefix01/checkpoint_006000.pt'])])
        for name,cmd in jobs:
            rec['current_stage']=name;save();t=time.perf_counter()
            with (E/f'recovery_{name}.log').open('wb') as out:subprocess.run([str(x) for x in cmd],stdout=out,stderr=subprocess.STDOUT,env=env,cwd=ROOT,check=True)
            rec['stages'][name]={'status':'completed','wall_seconds':time.perf_counter()-t,'command':[str(x) for x in cmd]};save()
        rec['status']='completed'
    except BaseException as ex:
        rec.update(status='failed',error=repr(ex),traceback=traceback.format_exc());raise
    finally:rec['wall_seconds']=time.perf_counter()-start;save()

if __name__=='__main__':main()
