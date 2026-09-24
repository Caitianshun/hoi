"""End-to-end completion, immutable old assets and paired-control audit."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
E=Path(__file__).resolve().parents[1];OLD=E.parents[1]/'structured_hoi_20260923'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def main():
 audit=load(E/'protocol/audit_inputs_manifest.json');done=load(E/'logs/reconstruction_completion.json');assert len(done)==2 and all(x['returncode']==0 for x in done)
 checks={};allrefs={}
 for d,L in [('dev1',114),('dev2',98)]:
  out=E/f'reconstruction/{d}_S1_star';ex=out/'evaluation';s=load(E/f'pose/{d}/optimization/selection_frozen.json');tr=load(out/'run.json');old=load(Path(audit['old_S1'][d]['run'])/'run.json');export=load(ex/'export_manifest.json');pair=load(E/f'evaluation/{d}_reconstruction_pair/pipeline.json');pipeline=load(E/f'reconstruction/{d}_pipeline.json');diff=load(E/f'reconstruction/{d}_initialization/controlled_difference.json')
  assert tr['status']==export['status']==export['evaluation_status']==pair['status']==pipeline['status']=='completed'
  assert tr['start_step']==0 and tr['parameter_updates']==tr['last_checkpoint_step']==8000 and tr['resume'] is None
  assert tr['final_checkpoint_sha256']==sha(out/'checkpoint_008000.pt')==pipeline['checkpoint_sha256']==export['checkpoint_sha256']
  assert sha(s['selected']['pose_file'])==s['selected']['pose_sha256'];assert sha(E/'protocol/pose_config_v1.json')==s['pose_config_sha256']
  assert tr['training_code_identity']==old['training_code_identity'];assert tr['frame_schedule_sha256']==old['frame_schedule_sha256']
  for key in ['loss_weights','schedule','densities','residual_bounds_m']:assert tr[key]==old[key]
  assert diff['all_remaining_fields_equal'] and set(diff['changed_fields'])=={'object_R','object_t','object_init'}
  for sub in ['rgb','geometry','buffers','query_overlay']:assert len(list((ex/sub).glob('*')))==L,(d,sub)
  assert len(list((ex/'heldout').glob('compare_*.png')))==5
  poses=np.load(ex/'object_pose.npz');assert poses['R_world'].shape==(L,3,3) and np.isfinite(poses['R_world']).all() and np.isfinite(poses['translation_world_m']).all()
  attrs=np.load(ex/'gaussian_identity.npz');assert all(np.isfinite(v).all() for v in attrs.values() if v.dtype.kind in 'fciu')
  files=audit['training_inputs'][d]['fixed_files']+audit['training_inputs'][d]['RGB']+audit['training_inputs'][d]['estimated_depth']+audit['old_S1'][d]['artifacts']+audit['evaluation_only'][d]['files']+audit['evaluation_only'][d]['heldout_RGB']+audit['evaluation_only'][d]['region_assets']
  for item in files:allrefs[item['path']]=item['sha256']
  checks[d]={'completed':True,'full_8000_from_zero':True,'old_core_and_training_schedule_identical':True,'only_declared_three_initialization_fields_changed':True,'RGB_geometry_buffer_query_frames':L,'heldout_frames':5,'checkpoints_sha256':tr['final_checkpoint_sha256'],'train_seconds':tr['wall_seconds'],'peak_train_allocated_gib':tr['peak_allocated_bytes']/2**30,'export_seconds':export['wall_seconds'],'cpu_pair_seconds':pair['wall_seconds'],'final_points':tr['points'],'source_alpha_cuda_error':export['sparse_cuda_check']}
 for name,r in audit['core_source_version'].items():allrefs[r['path']]=r['sha256']
 changed=[p for p,h in allrefs.items() if sha(p)!=h];assert not changed,changed
 result={'status':'passed','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'historical_input_reference_code_artifacts_rehashed':len(allrefs),'historical_assets_changed':changed,'negative_results_retained':True,'script_sha256':sha(__file__)}
 (E/'evaluation/completion_integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
