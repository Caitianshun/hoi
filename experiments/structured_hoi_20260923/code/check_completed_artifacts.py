"""Numerical acceptance of completed frozen exports, not scientific accuracy."""
from pathlib import Path
import hashlib,json,time
import numpy as np
E=Path('/home/cai_tianshun/Project/HOI/experiments/structured_hoi_20260923')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def main():
 start=time.perf_counter();records=[]
 for dev in ['dev1','dev2']:
  pair=[]
  for branch in ['S0','S1']:
   run=E/f'{dev}_{branch}_v1';r=read(run/'run.json');ev=run/('evaluation_v2' if dev=='dev1' else 'evaluation');m=read(ev/'export_manifest.json')
   assert r['status']=='completed' and m['evaluation_status']=='completed'
   assert sha(run/'checkpoint_008000.pt')==r['final_checkpoint_sha256']==m['checkpoint_sha256']
   for name,digest in r['training_code_identity'].items():assert sha(E/'code'/name)==digest
   pair.append(r);ident=np.load(ev/'gaussian_identity.npz');attr=np.load(ev/'gaussian_static_attributes.npz');ns,nh,no=ident['bank_sizes'];entity=ident['entity'];ids=ident['stable_id'];n=len(entity)
   assert n==sum(r['points']) and len(np.unique(np.c_[entity,ids],axis=0))==n
   assert (attr['scale']>0).all() and np.isfinite(attr['scale']).all()
   assert np.isfinite(attr['opacity']).all() and ((attr['opacity']>=0)&(attr['opacity']<=1)).all()
   pose=np.load(ev/'object_pose.npz');R=pose['R_world'];assert np.max(abs(np.linalg.det(R)-1))<1e-5
   assert np.max(abs(R@R.transpose(0,2,1)-np.eye(3)))<1e-5
   assert np.all(np.diff(pose['timestamp_seconds'])>0)
   rng=np.random.default_rng(76);pairs=rng.integers(0,no,size=(512,2));base=None;worst=0.
   for t in range(m['frame_count']):
    assert (ev/'rgb'/f'{t:05d}.png').exists() and (ev/'buffers'/f'{t:05d}.npz').exists()
    g=np.load(ev/'geometry'/f'{t:05d}.npz');x=g['dynamic_centres_world_m'];f=g['dynamic_affine_frames'];assert x.shape==(nh+no,3) and f.shape==(nh+no,3,3)
    assert np.isfinite(x).all() and np.isfinite(f).all();assert abs(float(g['timestamp_seconds'])-float(pose['timestamp_seconds'][t]))<1e-6
    obj=x[nh:];d=np.linalg.norm(obj[pairs[:,0]]-obj[pairs[:,1]],axis=-1)
    if base is None:base=d
    worst=max(worst,float(abs(d-base).max()))
   assert worst<2e-5,(dev,branch,worst)
   surface=ev/('surface_evaluation' if dev=='dev1' else 'surface_reference')/'metrics.json';support=ev/'support_diagnostic/diagnostic.json'
   assert read(surface)['status']=='completed' and read(support)['status']=='completed'
   records.append({'dev':dev,'branch':branch,'frames':m['frame_count'],'points':r['points'],'checkpoint_sha256':m['checkpoint_sha256'],'actual_object_pairwise_distance_max_change_m':worst,'all_export_geometry_finite':True,'unique_entity_stable_ids':True,'completed_surface_score':True,'completed_support_score':True})
  for key in ['initialization_sha256','resume','data_identity','training_code_identity','frame_schedule_sha256']:assert pair[0][key]==pair[1][key]
 result={'status':'passed','checks':'Completed runs, source/checkpoint identity, same paired inputs/schedule/prefix, all-frame geometry, unique Gaussian identities, SO3 and actual object rigidity. These numerical checks do not establish reconstruction accuracy.','rows':records,'wall_seconds':time.perf_counter()-start}
 (E/'summary/completion_integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
