"""Persistent fixed four-run plan with immediate final freeze and evaluation."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,socket,subprocess,sys,time,traceback
E=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def main():
 lock=(E/'protocol/gpu1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.perf_counter();status={'protocol_id':'AUX_REF_OBJECT','status':'running','pid':os.getpid(),'host':socket.gethostname(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stages':[]}
 write(E/'pipeline.json',status)
 try:
  assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
  assert json.loads((E/'protocol/native_availability.json').read_text())['training_gate'] is True
  assert json.loads((E/'preflight/result.json').read_text())['status']=='passed'
  codes=['train_aux.py','aux_scene.py','prepare_aux_input.py']
  frozen={'protocol_id':'AUX_REF_OBJECT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code':{n:ident(E/'code'/n) for n in codes},'preflight':ident(E/'preflight/result.json'),'native_availability':ident(E/'protocol/native_availability.json'),'regions_manifest':ident(E/'evaluation/regions/manifest.json'),'frame_schedules':ident(E/'frozen_aux/frame_schedule_manifest.json'),'inputs':{d:ident(E/'inputs'/d/'input_manifest.json') for d in ['dev1','dev2']},'scope':'4 formal runs total; dev1 Pred, dev1 Ref, dev2 Pred, dev2 Ref; each8000; no further training'}
  write(E/'frozen_aux/start.json',frozen)
  for dev in ['dev1','dev2']:
   for arm in ['Pred','Ref']:
    for n,item in frozen['code'].items():assert sha(item['path'])==item['sha256']
    command=[sys.executable,str(E/'code/train_aux.py'),'train','--dev',dev,'--arm',arm]
    status['current_stage']=f'{dev}_{arm}';write(E/'pipeline.json',status)
    with (E/f'{dev}_{arm}.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    run=json.loads((E/'runs'/f'{dev}_{arm}'/'run.json').read_text());assert run['status']=='completed' and run['step']==8000
    status['stages'].append({'dev':dev,'arm':arm,'checkpoint':run['checkpoint'],'seconds':run['wall_seconds']});write(E/'pipeline.json',status)
  runs=[]
  for dev in ['dev1','dev2']:
   pred=json.loads((E/'runs'/f'{dev}_Pred'/'run.json').read_text());ref=json.loads((E/'runs'/f'{dev}_Ref'/'run.json').read_text())
   assert pred['initial_obank_identity']==ref['initial_obank_identity'] and pred['frame_schedule_sha256']==ref['frame_schedule_sha256']
   for arm in ['Pred','Ref']:
    d=E/'runs'/f'{dev}_{arm}';r=json.loads((d/'run.json').read_text());cp=r['checkpoint'];assert sha(cp['path'])==cp['sha256']
    runs.append({'dev':dev,'arm':arm,'step':8000,'checkpoint':cp,'input_manifest':ident(E/'inputs'/dev/'input_manifest.json'),'reference_motion':ident(E/'inputs'/dev/'reference_object_motion.npz'),'run_record':ident(d/'run.json'),'config':ident(d/'config.json')})
  freeze={'protocol_id':'AUX_REF_OBJECT','phase':'four_finals_frozen','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'regions_manifest':frozen['regions_manifest'],'runs':runs,'same_initialization_and_schedule_within_dev':True,'start_freeze':ident(E/'frozen_aux/start.json')}
  write(E/'frozen_aux/finals.json',freeze)
  status['current_stage']='cross_pose_evaluation';write(E/'pipeline.json',status)
  command=[sys.executable,str(E/'code/evaluate_aux.py'),'score','--freeze',str(E/'frozen_aux/finals.json')]
  with (E/'evaluation.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
  status.update(status='completed',current_stage='complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except BaseException as e:
  status.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  status['wall_seconds']=time.perf_counter()-start;write(E/'pipeline.json',status)
if __name__=='__main__':main()
