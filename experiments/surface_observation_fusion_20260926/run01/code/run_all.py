"""Persistent bounded pipeline. Final freeze triggers evaluation directly."""
from pathlib import Path
import json,os,subprocess,time,traceback,hashlib
E=Path(__file__).resolve().parents[1];R=E.parents[2];B=R/'experiments/fixed_motion_reconstruction_20260926/run01';PY=R/'envs/gvhmr/bin/python';start=time.perf_counter();record={'status':'running','pid':os.getpid(),'GPU':'physical1 RTX3090','completed':[]}
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def call(script,*args):
 record['stage']=[script,*args];save(E/'pipeline.json',record)
 oldcost=sum(json.loads(p.read_text())['seconds'] for p in list((B/'preflight').glob('*/run.json'))+list((B/'runs').glob('*/run.json')))
 remaining=14400-oldcost-(time.perf_counter()-start);assert remaining>0
 subprocess.run([str(PY),str(E/'code'/script),*args],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},timeout=remaining)
 record['completed'].append(record['stage']);save(E/'pipeline.json',record)
try:
 assert json.loads((B/'pipeline.json').read_text())['status']=='completed_awaiting_F_freeze'
 cfg=json.loads((E/'protocol/fusion_frozen.json').read_text())
 for p,h in cfg['codes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 for dev in ['dev1','dev2']:call('bake_fusion_colors.py','--dev',dev,'--variant','F0')
 skips={}
 for dev in ['dev1','dev2']:
  if not cfg['input_check'][dev]['input_candidate_gate']:skips[dev]='input cache lacks measurable source choice';continue
  for variant in ['F1','F2']:call('train_fusion.py','--dev',dev,'--variant',variant,'--check')
  for variant in ['F1','F2']:
   call('train_fusion.py','--dev',dev,'--variant',variant);call('bake_fusion_colors.py','--dev',dev,'--variant',variant)
 assets=[]
 paths=list((B/'runs').glob('*/checkpoint_008000.pt'))+list((E/'runs').glob('*/baked_model.pt'))+list((E/'runs').glob('*/checkpoint_002000.pt'))+list((E/'support').glob('*/*.npz'))+list((E/'code').glob('*.py'))+[E/'protocol/fusion_frozen.json',E/'protocol/frozen.json',B/'protocol/frozen.json']
 for p in paths:assets.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 base=json.loads((E/'protocol/frozen.json').read_text());assets+=base['assets']
 save(E/'protocol/finals.json',{'status':'all_states_frozen','unix_time':time.time(),'assets':assets,'skipped':skips,'evaluation_seen':False})
 call('evaluate_all.py');record['status']='completed'
except BaseException:
 record.update(status='failed',traceback=traceback.format_exc());raise
finally:record['seconds']=time.perf_counter()-start;save(E/'pipeline.json',record)
