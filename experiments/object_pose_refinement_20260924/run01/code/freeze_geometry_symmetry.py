"""Freeze a finite approximate symmetry set using texture-free geometry alone."""
from pathlib import Path
import json,hashlib,datetime,time,itertools
import numpy as np
import trimesh
ROOT=Path('/home/cai_tianshun/Project/HOI');E=ROOT/'experiments/object_pose_refinement_20260924/run01';OLD=ROOT/'experiments/structured_hoi_20260923'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sample(v,f,n=4096):
 tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;rng=np.random.default_rng(8293);idx=rng.choice(len(f),n,p=area/area.sum());u=rng.random((n,2));s=np.sqrt(u[:,0]);w=np.c_[1-s,s*(1-u[:,1]),s*u[:,1]];return np.einsum('ni,nij->nj',w,tri[idx])
def distances(mesh,p):return np.concatenate([trimesh.proximity.closest_point(mesh,z)[1] for z in np.array_split(p,max(1,int(np.ceil(len(p)/256))))])
def main():
 start=time.perf_counter();out=E/'protocol/symmetry_geometry_only.json';assert not out.exists()
 config={'candidate_rule':'identity plus 180 degree rotations about each of three area-sampled principal axes and oriented bounding box axes, maximum7; no pose/reference/RGB read','sample_count':4096,'sample_seed':8293,'exact_tolerance_m':1e-4,'approximate_p95_tolerance_diameter_fraction':.01,'approximate_max_tolerance_diameter_fraction':.02,'diameter':'maximum pairwise vertex distance','centre_rule':'oriented bounding box centre, derived only from template geometry; symmetry includes canonical translation c-S@c','accept_rule':'Both forward and inverse transformed fixed samples plus face-referenced vertices pass p95<=1%diameter and max<=2%diameter. Conservative finite hypotheses, not asserted exact physical group.','bop_status':'Custom finite-set symmetry rotation/model distances, not official BOP MSSD/MSPD.'}
 (E/'protocol/symmetry_config_predeclared.json').write_text(json.dumps(config,indent=2)+'\n');result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'configuration':config,'configuration_sha256':sha(E/'protocol/symmetry_config_predeclared.json'),'script_sha256':sha(__file__),'dev':{}}
 for dev,p in [('dev1',OLD/'object_init/run03/object_init.npz'),('dev2',OLD/'object_init/dev2_refined01/object_init.npz')]:
  n=np.load(p);v=n['canonical_vertices_m'].astype(float);f=n['faces'];mesh=trimesh.Trimesh(v,f,process=False);samples=sample(v,f);used=np.unique(f);pts=np.r_[samples,v[used]];mesh.remove_unreferenced_vertices()
  from scipy.spatial.distance import pdist
  diam=float(pdist(v).max());obb=mesh.bounding_box_oriented;centre=obb.primitive.transform[:3,3];axes_obb=obb.primitive.transform[:3,:3];_,axes_area=np.linalg.eigh(np.cov(samples.T));candidates=[('identity',np.eye(3))]
  for tag,axes in [('area_principal',axes_area),('oriented_box',axes_obb)]:
   for i in range(3):
    a=axes[:,i];S=2*np.outer(a,a)-np.eye(3)
    if not any(np.linalg.norm(S-old)<1e-8 for _,old in candidates):candidates.append((f'{tag}_pi_axis{i}',S))
  rec=[]
  for name,S in candidates:
   shift=centre-S@centre
   d=np.r_[distances(mesh,pts@S.T+shift),distances(mesh,(pts-shift)@S)]
   p95=float(np.percentile(d,95));mx=float(d.max());accept=p95<=diam*.01 and mx<=diam*.02
   rec.append({'name':name,'R':S.tolist(),'t_m':shift.tolist(),'surface_distance_p95_m':p95,'surface_distance_max_m':mx,'accepted':bool(accept),'classification':'exact_within_numerical_tolerance' if mx<1e-4 else ('approximate_geometry_only' if accept else 'rejected_non_equivalent')})
  result['dev'][dev]={'canonical_source':str(p),'canonical_file_sha256':sha(p),'vertices_faces_sha256':hashlib.sha256(v.astype(np.float32).tobytes()+f.tobytes()).hexdigest(),'diameter_m':diam,'symmetry_centre_m':centre.tolist(),'all_candidates':rec,'accepted_names':[r['name'] for r in rec if r['accepted']],'reference_or_prediction_read':False,'orphan_vertices_not_in_surface_test':int(len(v)-len(used))}
 result['wall_seconds']=time.perf_counter()-start;out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({d:r['accepted_names'] for d,r in result['dev'].items()}));print('sha256',sha(out))
if __name__=='__main__':main()
