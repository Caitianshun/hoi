"""Completion-driven, persistent S1* pipeline. No resume and no tuning.

Run once per accepted dev after explicit GPU ownership allocation by root.
Errors are atomically recorded; each successful train immediately exports and
scores. Historical code and attributes are reused without mutation.
"""
from pathlib import Path
import argparse, json, hashlib, datetime, subprocess, sys, os, time, traceback
import numpy as np
import torch
ROOT=Path('/home/cai_tianshun/Project/HOI');OLD=ROOT/'experiments/structured_hoi_20260923';E=ROOT/'experiments/object_pose_refinement_20260924/run01'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def save(p,x):
 q=Path(p).with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def eq(a,b):
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b,equal_nan=True)
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev',choices=['dev1','dev2'],required=True);ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--cpu-evaluator',type=Path,required=True);a=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1','GPU1 allocated; never implicit default GPU'
 base=E/'reconstruction';base.mkdir(exist_ok=True);initdir=base/f'{a.dev}_initialization';out=base/f'{a.dev}_S1_star';status=base/f'{a.dev}_pipeline.json';assert not status.exists()
 t=time.perf_counter();rec={'status':'running','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'dev':a.dev,'selection':str(a.selection.resolve()),'selection_sha256':sha(a.selection),'commands':[]}
 save(status,rec)
 def run(cmd):
  rec['commands'].append(cmd);save(status,rec);subprocess.run(cmd,check=True)
 try:
  s=json.loads(a.selection.read_text());assert s['status']=='accepted_for_S1_star' and s['no_independent_reference_used']
  pred=Path(s['selected']['pose_file']);assert sha(pred)==s['selected']['pose_sha256'];pose=np.load(pred)
  oldinit=OLD/f'{a.dev}_initialization/initialization.pt';old=torch.load(oldinit,map_location='cpu',weights_only=False);new=dict(old);new['object_R']=pose['R_world'].copy();new['object_t']=pose['t_world_m'].copy();new['object_init']=str(pred.resolve())
  changed=[k for k in old if not eq(old[k],new[k])];assert set(changed)<=set(['object_R','object_t','object_init']);assert 'object_init' in changed
  initdir.mkdir(exist_ok=False);ip=initdir/'initialization.pt';torch.save(new,ip)
  diff={'old_initialization':str(oldinit),'old_sha256':sha(oldinit),'new_initialization':str(ip),'new_sha256':sha(ip),'changed_fields':changed,'byte_equal_remaining_arrays':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in old.items() if isinstance(v,np.ndarray) and k not in changed},'all_remaining_fields_equal':all(eq(old[k],new[k]) for k in old if k not in changed),'attribution':'Object pose initialization process: R/t initial values, pose-prior targets, correction temporal reference. Prior form, weights, schedule unchanged.','object_colors_resampled':False,'new_tracks_in_reconstruction':False,'resume':None,'step_range':[0,8000]}
  save(initdir/'controlled_difference.json',diff)
  env_identity=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True,check=True).stdout
  rec['launch_gpu_snapshot']=env_identity;rec['phase']='training';save(status,rec)
  run([sys.executable,str(OLD/'code/train_structured.py'),'--init',str(ip),'--output',str(out),'--branch','S1','--stop','8000'])
  tr=json.loads((out/'run.json').read_text());assert tr['status']=='completed' and tr['start_step']==0 and tr['last_checkpoint_step']==8000
  oldtr=json.loads((OLD/f'{a.dev}_S1_v1/run.json').read_text());assert tr['training_code_identity']==oldtr['training_code_identity'];assert tr['frame_schedule_sha256']==oldtr['frame_schedule_sha256']
  for k in ['loss_weights','schedule','densities','residual_bounds_m']:assert tr[k]==oldtr[k],k
  rec['phase']='export';save(status,rec)
  run([sys.executable,str(OLD/'code/evaluate_structured.py'),'--init',str(ip),'--checkpoint',str(out/'checkpoint_008000.pt'),'--output',str(out/'evaluation')])
  rec['phase']='CPU_paired_evaluation';save(status,rec)
  run([sys.executable,str(a.cpu_evaluator),'--dev',a.dev,'--export',str(out/'evaluation'),'--selection',str(a.selection.resolve())])
  rec.update(status='completed',phase='completed',wall_seconds=time.perf_counter()-t,checkpoint=str(out/'checkpoint_008000.pt'),checkpoint_sha256=sha(out/'checkpoint_008000.pt'),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(status,rec)
 except BaseException:
  rec.update(status='failed',wall_seconds=time.perf_counter()-t,error=traceback.format_exc());save(status,rec);raise
if __name__=='__main__':main()
