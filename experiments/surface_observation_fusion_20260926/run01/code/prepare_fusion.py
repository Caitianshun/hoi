from pathlib import Path
import sys,json,time
import numpy as np,torch
from scipy.spatial import cKDTree
from build_surface_observations import E,A,save,sha_file,SurfaceIndex,support_query,mesh_maps
from observation_fusion import Memory,Scorer,QD,KD,WIDTH,F2_PARAMS

def main():
 start=time.perf_counter();report={}
 for dev in ['dev1','dev2']:
  p=E/'support'/dev;mem=Memory(dev,'cpu');mem.prepare();obs=mem.obs;ix=SurfaceIndex(obs);x=mem.obj;T=mem.T;meta=json.loads((A/'inputs'/dev/'input_manifest.json').read_text());g=np.load(json.loads((p/'manifest.json').read_text())['geometry']['path']);tri=g['canonical_vertices_m'][g['faces']];lab=np.load(meta['segmentation'])['entity_labels'];schedule=np.load(A/'frozen_aux'/f'{dev}_frame_schedule.npy')[:2000];rows=[]
  for t in range(T):
   m=np.load(p/f'map_{t:03d}.npz');O=lab[t]==2;valid=O&(m['face']>=0);y,z=np.where(valid);face=m['face'][y,z];b=m['bary'][y,z];pts=(tri[face]*b[:,:,None]).sum(1);ids,dd,n,state=support_query(ix,pts,face,T)
   eligible=(mem.features(t)[3].sum(1)>=2).numpy();covered=np.zeros(len(pts),bool);anchor=np.zeros(len(pts),bool)
   for f in np.unique(face):
    qi=np.flatnonzero(face==f);ai=np.flatnonzero(x['attachment_face_id']==f)
    if len(ai):anchor[qi]=cKDTree(x['anchor_canonical_m'][ai]).query(pts[qi],distance_upper_bound=.005)[0]<np.inf
    gi=np.flatnonzero((mem.query['face']==f)&eligible)
    if len(gi):covered[qi]=cKDTree(x['centres_canonical_m'][gi]).query(pts[qi],distance_upper_bound=.005)[0]<np.inf
   feat=mem.features(t);sel=mem.selections[t];rng=np.zeros(len(sel));
   for i in np.flatnonzero(eligible):rng[i]=np.linalg.norm(np.ptp(obs['rgb'][sel[i,sel[i]>=0]],axis=0))
   rows.append(dict(frame=t,time_seconds=meta['timestamp_seconds'][t],O_pixels=int(O.sum()),template_pixels=int(valid.sum()),source_count_pixels={str(c):int((n==c).sum()) for c in [0,1,2]},source_ge3_pixels=int((n>=3).sum()),unmapped_pixels=int((state==0).sum())+int((O&~(m['face']>=0)).sum()),positive_without_anchor_pixels=int(((n>0)&~anchor).sum()),drop_source_candidate_gaussians=int(eligible.sum()),target_pixels_near_eligible=int(covered.sum()),candidate_rgb_range_mean=float(rng[eligible].mean()) if eligible.any() else None))
  eligible_times=[r['frame'] for r in rows if r['drop_source_candidate_gaussians']>0 and r['target_pixels_near_eligible']>0];checks=[int(t) for t in schedule if t in eligible_times][:8]
  # CPU zero-step parity; verify source removal before any Q aggregate.
  for variant in ['F1','F2']:
   torch.manual_seed(12345);scorer=Scorer(variant)
   for key,feat in mem.cache.items():
    assert torch.equal(scorer(feat),Scorer('F0')(feat));sel=mem.selections[key]
    if key>=0:assert not (obs['time'][sel[sel>=0]]==key).any()
  # Barycentric first-surface points project back to exact raster pixel centers.
  ref=np.load(meta['reference_object_motion']);errs=[]
  for t in range(T):
   mp=np.load(p/f'map_{t:03d}.npz');y,z=np.where(mp['face']>=0);pick=np.linspace(0,len(y)-1,min(32,len(y))).astype(int);y=y[pick];z=z[pick];pts=(tri[mp['face'][y,z]]*mp['bary'][y,z,:,None]).sum(1);C=np.array(meta['c2w']);cam=(pts@ref['R_world'][t].T+ref['t_world'][t]-C[:3,3])@C[:3,:3];uv=cam@np.array(meta['K']).T;errs.append(float(np.max(abs(uv[:,:2]/uv[:,2:]-np.c_[z,y]))))
  assert max(errs)<1e-3
  report[dev]={'input_candidate_gate':len(set(checks))>=2,'eligible_times':eligible_times,'precheck_frames':checks,'rows':rows,'projection_max_error_px':max(errs),'gaussian_support_histogram':{str(k):int((mem.query['support_count']==k).sum()) for k in [0,1,2]},'gaussians_ge3':int((mem.query['support_count']>=3).sum()),'zero_step_exact_all_variants':True}
 frozen=dict(qdim=QD,kdim=KD,MLP_hidden=WIDTH,F1_params=sum(p.numel() for p in Scorer('F1').parameters()),F2_params=F2_PARAMS,features='position meters; relative /.005m; normal and view unit; RGB [0,1]; gradients per pixel; distance/5 clipped to1; resolution/.005; count/12; RGB3+patch27+valid9+grad6+gradvalid2+relative3+view3+distance1+purity1+reliability1+resolution1+valid1',mapping='unique closest inherited face; distance <=5mm; nearest/second difference >1e-7m; no adjacent face reassignment',codes={str(p):sha_file(p) for p in (E/'code').glob('*.py')},caches={str(p):sha_file(p) for p in (E/'support').glob('*/*.npz')},input_check=report,seconds=time.perf_counter()-start)
 save(E/'protocol/fusion_frozen.json',frozen)
 print(json.dumps({'dimensions':[QD,KD],'parameters':[frozen['F1_params'],F2_PARAMS],'eligibility':{d:r['precheck_frames'] for d,r in report.items()},'seconds':frozen['seconds']}))
if __name__=='__main__':torch.set_num_threads(2);main()
