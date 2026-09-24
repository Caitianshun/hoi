"""Sequential bounded GPU runner; successful exit chains verification and evaluation."""
from pathlib import Path
import subprocess,sys,os,json,hashlib,time,datetime,traceback
import numpy as np
E=Path(__file__).resolve().parents[1]
ROOT=E.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 cfg=json.loads((E/'protocol/frozen_surface_pose.json').read_text())
 for p,h in cfg['code_hashes'].items():assert sha(p)==h,p
 check=json.loads((E/'protocol/implementation_checks.json').read_text())
 assert check['all_numerical_preexecution_checks_passed'] and check['base_forward_passed'],check
 log=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 started=utc();st=time.perf_counter()
 for dev in ['dev1','dev2']:
  for var in ['F00','F01','F10','F11']:
   resource=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True)
   print(resource,flush=True)
   out=E/dev/var;cmd=[sys.executable,str(E/'code/joint_solver.py'),'solve','--dev',dev,'--variant',var,'--steps','300','--output',str(out)]
   row={'dev':dev,'variant':var,'command':cmd,'started_utc':utc(),'resources_before':resource}
   with (E/f'logs/{dev}_{var}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
   row.update(exit_code=r.returncode,finished_utc=utc());log.append(row);save(E/'protocol/execution_status.json',{'status':'running' if not r.returncode else 'failed','runs':log})
   if r.returncode:raise RuntimeError(f'{dev}/{var} failed with {r.returncode}')
   print('completed',dev,var,flush=True)
 # Check the eight output invariants without reference.
 runs={(d,v):json.loads((E/d/v/'metrics.json').read_text()) for d in ['dev1','dev2'] for v in ['F00','F01','F10','F11']}
 invariants={'same_initialization':True,'same_solver_and_budget':True,'common_observation_pool_and_fixed_denominators':True,'q_fixed_for_F00_F01':True,'q_on_fixed_surface':True,'rotation_legal_and_scale_fixed':True}
 for (d,v),m in runs.items():
  z=np.load(E/d/v/'object_init.npz');s=np.load(E/d/v/'surface_points.npz');original=np.load(m['initialization']);obs=np.load(E/f'observations/{d}/observations.npz')
  invariants['same_initialization'] &= m['initialization_sha256']==runs[(d,'F00')]['initialization_sha256']
  invariants['same_solver_and_budget'] &= m['actual_updates']==300 and m['source_sha256']==cfg['code_hashes'][str(E/'code/joint_solver.py')] and m['termination']=='fixed_update_budget'
  invariants['common_observation_pool_and_fixed_denominators'] &= m['observations_sha256']==cfg['data'][d]['observation_sha256']
  if v in ['F00','F01']:invariants['q_fixed_for_F00_F01'] &= m['fixed_q_max_difference_m']<1e-6 and m['total_face_switches']==0
  tri=original['canonical_vertices_m'][original['faces'][s['face']]];q=(tri*s['bary'][:,:,None]).sum(1)
  invariants['q_on_fixed_surface'] &= bool(np.max(abs(q-s['q_final']))<1e-6 and s['surface_distance_bound_m'].max()<=.0500001)
  R=z['R_camera'];invariants['rotation_legal_and_scale_fixed'] &= bool(np.max(abs(R@R.transpose(0,2,1)-np.eye(3)))<1e-5 and np.max(abs(np.linalg.det(R)-1))<1e-5 and np.array_equal(z['canonical_vertices_m'],original['canonical_vertices_m']))
 assert all(invariants.values()),invariants
 checks=dict(check['checks']);checks.update(invariants)
 rules=json.loads((E/'protocol/promotion_rules.json').read_text())
 assert all((checks[k].get('passed',False) if isinstance(checks[k],dict) else checks[k]) for k in rules['implementation_gate']['must_pass'])
 files=[]
 for d in ['dev1','dev2']:
  for v in ['F00','F01','F10','F11']:
   for p in sorted((E/d/v).glob('*')):
    if p.is_file():files.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
 frozen={'all_eight_outputs_frozen':True,'reference_used':False,'utc':utc(),'started_utc':started,'promotion_rules_sha256':sha(E/'protocol/promotion_rules.json'),'frozen_config_sha256':sha(E/'protocol/frozen_surface_pose.json'),'implementation_checks':checks,'implementation_checks_file':{'path':str(E/'protocol/implementation_checks.json'),'sha256':sha(E/'protocol/implementation_checks.json')},'files':files,'run_logs':log,'wall_seconds_before_evaluation':time.perf_counter()-st}
 save(E/'protocol/all_outputs_frozen.json',frozen)
 save(E/'protocol/execution_status.json',{'status':'eight_outputs_frozen_evaluating','runs':log})
 cmd=[sys.executable,str(E/'code/evaluate_prototype.py'),'--all-outputs-freeze',str(E/'protocol/all_outputs_frozen.json')]
 with (E/'logs/evaluation.log').open('w') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 save(E/'protocol/execution_status.json',{'status':'completed' if not r.returncode else 'evaluation_failed','runs':log,'evaluation_exit_code':r.returncode,'completed_utc':utc(),'wall_seconds':time.perf_counter()-st})
 if r.returncode:raise RuntimeError('evaluation failed')
 print('ALL COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(E/'protocol/runner_failure.json',{'status':'failed','utc':utc(),'error':str(e),'traceback':traceback.format_exc()});raise
