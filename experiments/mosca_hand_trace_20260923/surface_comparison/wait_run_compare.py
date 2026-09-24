"""Wait for atomic pipeline completion with inotify, then CPU surface audit + comparison."""
from pathlib import Path
import os,json,sys,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
RUN=ROOT/'experiments/mosca_hand_trace_20260923';C=RUN/'c_input_foreground'
os.environ['CUDA_VISIBLE_DEVICES']=''
state=dict(status='waiting_for_C_completion_inotify',pid=os.getpid(),start_unix=time.time(),gpu_used=False,stages=[])
def save():(OUT/'pipeline.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 commands=[('wait_C',[sys.executable,str(ROOT/'experiments/mosca_validation_20260923/code/wait_completion.py'),str(C)]),
  ('audit_C',[sys.executable,str(OUT/'audit_surface_branch.py'),'--branch-dir',str(C),'--output',str(RUN/'surface_audit_c'),'--label','C']),
  ('compare_A_C',[sys.executable,str(OUT/'compare_surfaces.py')])]
 for name,cmd in commands:
  if name!='wait_C' and json.loads((C/'pipeline.json').read_text())['status']!='completed':raise RuntimeError('C did not complete successfully')
  state['status']='waiting_for_C_completion_inotify' if name=='wait_C' else name;save();t=time.perf_counter()
  with (OUT/f'{name}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  state['stages'].append(dict(name=name,status='completed',seconds=time.perf_counter()-t));save()
 state['status']='completed_cpu_pending_visual_QA'
except BaseException as exc:
 state.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
finally:
 state.update(end_unix=time.time(),wall_seconds=time.time()-state['start_unix']);save()
