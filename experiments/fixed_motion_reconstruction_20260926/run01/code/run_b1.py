from pathlib import Path
import os,json,subprocess,time,traceback,sys
E=Path(__file__).resolve().parents[1];start=time.perf_counter();state={'status':'running','pid':os.getpid(),'GPU':'physical1 RTX3090'}
def save(): (E/'pipeline.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for check,dev in [(True,'dev2'),(False,'dev1'),(False,'dev2')]:
  state.update(stage='precheck' if check else 'formal',dev=dev);save()
  subprocess.run([sys.executable,str(E/'code/train_visible_object.py'),'--dev',dev]+(['--check'] if check else []),check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'})
  result=json.loads((E/('preflight' if check else 'runs')/dev/'run.json').read_text());assert result['status']=='completed' and result['steps']==(8 if check else 8000)
 # All final hashes are frozen now; evaluation awaits independent F states.
 import hashlib
 state.update(status='completed_awaiting_F_freeze',checkpoints=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((E/'runs').glob('*/checkpoint_008000.pt'))]);assert len(state['checkpoints'])==2
except BaseException:
 state.update(status='failed',traceback=traceback.format_exc());raise
finally:state['seconds']=time.perf_counter()-start;save()
