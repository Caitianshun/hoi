"""Clone frozen structured initialization, changing only object R/t and source.

No RGB colour sampling, body fitting, reference loading, or Gaussian resampling.
Requires prior input-only selection and exact frozen pose identity.
"""
from pathlib import Path
import argparse,copy,hashlib,json,numpy as np,torch

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def same(a,b):
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
 return a==b

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-init',type=Path,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--selection-freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 freeze=json.loads(a.selection_freeze.read_text());assert freeze['input_only_selection_complete'];allowed={(str(Path(r['path']).resolve()),r['sha256']) for r in freeze['frozen_predictions']};assert (str(a.pose.resolve()),sha(a.pose)) in allowed
 old=torch.load(a.old_init,map_location='cpu',weights_only=False);assert old['reference_used'] is False
 pose=np.load(a.pose);R=pose['R_world'];t=pose['t_world_m' if 't_world_m' in pose else 'translation_world_m'];n=len(old['timestamps']);assert R.shape==(n,3,3) and t.shape==(n,3);assert np.isfinite(R).all() and np.isfinite(t).all();assert np.max(np.linalg.norm(np.swapaxes(R,-1,-2)@R-np.eye(3),axis=(1,2)))<1e-4;assert np.max(abs(np.linalg.det(R)-1))<1e-4;assert np.allclose(pose['timestamp_seconds'],old['timestamps'],atol=1e-7,rtol=0)
 oldpose=np.load(old['object_init']);checks={}
 for name in ['canonical_vertices_m','faces','centres_m','sample_face_ids','sample_barycentric_uv','K','c2w']:
  if name in pose:
   checks[name]=bool(np.array_equal(pose[name],oldpose[name]));assert checks[name],f'Unexpected fixed-input difference: {name}'
 new=copy.deepcopy(old);new['object_R']=R.astype(old['object_R'].dtype);new['object_t']=t.astype(old['object_t'].dtype);new['object_init']=str(a.pose.resolve());keep=[k for k in old if k not in ['object_R','object_t','object_init']];assert all(same(old[k],new[k]) for k in keep)
 a.output.mkdir(parents=True,exist_ok=False);out=a.output/'initialization.pt';torch.save(new,out)
 reloaded=torch.load(out,map_location='cpu',weights_only=False);assert all(same(old[k],reloaded[k]) for k in keep)
 meta={'status':'completed','role':'input_only_initialization','reference_used':False,'old_initialization':{'path':str(a.old_init.resolve()),'sha256':sha(a.old_init)},'frozen_pose':{'path':str(a.pose.resolve()),'sha256':sha(a.pose)},'selection_freeze':{'path':str(a.selection_freeze.resolve()),'sha256':sha(a.selection_freeze)},'output_sha256':sha(out),'script_sha256':sha(__file__),'changed_keys':['object_R','object_t','object_init'],'all_other_keys_equal_after_serialization':True,'unchanged_keys':keep,'pose_fixed_geometry_fields_checked':checks,'attribution':'Object-pose initialization process: initial R/t plus soft-prior targets and correction-temporal baseline. No change to prior form, weights, valid-frame rule or schedule.','object_colour_resampled':False,'canonical_anchors_resampled':False,'start_step_required':0,'resume_old_checkpoint_allowed':False}
 (a.output/'initialization_diff.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__':main()
