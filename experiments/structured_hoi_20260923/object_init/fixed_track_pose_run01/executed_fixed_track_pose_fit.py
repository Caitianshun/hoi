"""One input-only fixed-correspondence rigid pose fit. No reference imports/read paths."""
from pathlib import Path
import json,hashlib,time,os,sys,signal
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT=Path('/home/cai_tianshun/Project/HOI')
E=ROOT/'experiments/structured_hoi_20260923'
BASE=ROOT/'experiments/mosca_baseline_20260922'
HERE=E/'object_init'; OUT=HERE/'fixed_track_pose_run01'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d): (OUT/n).write_text(json.dumps(d,indent=2)+'\n')
def native(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [native(v) for v in x]
 return x
def ray_mesh(uv,R,t,V,F,K):
 rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T; tri=(V@R.T+t)[F];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
 out=[];valid=[]
 for ray in rays:
  h=np.cross(np.broadcast_to(ray,e2.shape),e2);det=np.sum(e1*h,axis=1);ok=np.abs(det)>1e-9;inv=np.divide(1.,det,out=np.zeros_like(det),where=ok);sv=-tri[:,0];u=inv*np.sum(sv*h,axis=1);q=np.cross(sv,e1);v=inv*(q@ray);z=inv*np.sum(e2*q,axis=1);good=ok&(u>=0)&(v>=0)&(u+v<=1)&(z>0);z[~good]=np.inf;j=int(z.argmin());hit=np.isfinite(z[j]);valid.append(hit);out.append((ray*z[j]-t)@R if hit else np.zeros(3))
 return np.asarray(out),np.asarray(valid)
def conditioning(x,uv):
 if len(x)<6:return False,0.,0.
 s=np.linalg.svd(x-x.mean(0),compute_uv=False);u=np.linalg.svd(uv-uv.mean(0),compute_uv=False)
 ratio=float(s[-1]/max(s[0],1e-12));spread=float(u[-1]);return ratio>=.02 and spread>=3,ratio,spread
def project(x,K): p=x@K.T;return p[:,:2]/np.maximum(p[:,2:],1e-6)
def main():
 start=time.perf_counter();OUT.mkdir(exist_ok=False)
 protocol=HERE/'fixed_track_pose_protocol.json'
 allowed=[HERE/'run03/object_init.npz',BASE/'common_input/input_manifest.json',BASE/'common_input/uniform_cotracker_tap.npz',BASE/'segmentation/segmentation.npz',protocol,Path(__file__)]
 hashes={str(p):sha(p) for p in allowed};save('run.json',dict(status='running',input_hashes=hashes,reference_read=False,hardware='CPU only',protocol_sha256=sha(protocol)))
 def alarm(*args):raise TimeoutError('Frozen 600 second total CPU wall budget exceeded')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(590)
 try:
  init=np.load(allowed[0]);m=json.loads(allowed[1].read_text());tr=np.load(allowed[2]);lab=np.load(allowed[3])['entity_labels'];K=np.array(m['K']);C=np.array(m['c2w']);times=np.asarray(m['timestamp_seconds']);T=len(times)
  assert np.array_equal(times,tr['timestamp_seconds']);assert np.array_equal(times,init['timestamp_seconds'])
  tracks=tr['tracks'];vis=tr['visibility'];source=int(np.flatnonzero(init['observed_mask'])[0]);query=tr['query_points'];V=init['canonical_vertices_m'].astype(float);F=init['faces'];R0=init['R_camera'].astype(float);t0=init['t_camera_m'].astype(float)
  masks=np.asarray([cv2.erode((x==2).astype(np.uint8),np.ones((3,3),np.uint8)) for x in lab])
  def inside(t,uv):
   xy=np.rint(np.nan_to_num(uv,nan=-1000)).astype(int);ok=(xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=0)&(xy[:,1]<480);out=np.zeros(len(uv),bool);out[ok]=masks[t,xy[ok,1],xy[ok,0]]>0;return out
  ids=np.flatnonzero((query[:,0]==source)&vis[source]&inside(source,tracks[source]))
  anchors,hits=ray_mesh(tracks[source,ids],R0[source],t0[source],V,F,K);candidate_ids=ids.copy();ids=ids[hits];anchors=anchors[hits]
  ok,ratio,spread=conditioning(anchors,tracks[source,ids])
  source_info=dict(frame=source,mask_candidate_ids=candidate_ids,hit_ids=ids,hit_count=len(ids),canonical_rank_ratio=ratio,image_spread_px=spread,passed=ok)
  save('source_gate.json',native(source_info));np.savez_compressed(OUT/'fixed_anchors.npz',track_ids=ids,canonical_points_m=anchors,source_frame=source,source_uv=tracks[source,ids])
  if not ok:
   save('run.json',dict(status='gated_no_fit',reason='Source correspondence count/planarity/spread failed frozen gate',source=native(source_info),input_hashes=hashes,reference_read=False,wall_seconds=time.perf_counter()-start));return
  gates=[];obs=[];rp=R0.copy();tp=t0.copy();cv2.setRNGSeed(230923)
  for t in range(T):
   keep=vis[t,ids]&inside(t,tracks[t,ids])&np.isfinite(tracks[t,ids]).all(-1);jj=np.flatnonzero(keep);good,ratio,spread=conditioning(anchors[jj],tracks[t,ids[jj]])
   row=dict(frame=t,candidate_count=len(jj),canonical_rank_ratio=ratio,image_spread_px=spread,nondegenerate=good,pnp_success=False,inlier_count=0)
   if good:
    worked,rv,tv,inliers=cv2.solvePnPRansac(anchors[jj].astype(float),tracks[t,ids[jj]].astype(float),K,None,iterationsCount=100,reprojectionError=4,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if worked and inliers is not None and len(inliers)>=6:
     kk=jj[inliers[:,0]];g2,ratio2,spread2=conditioning(anchors[kk],tracks[t,ids[kk]]);r=cv2.Rodrigues(rv)[0];positive=bool(np.all((anchors[kk]@r.T+tv[:,0])[:,2]>.2))
     row.update(inlier_count=len(kk),inlier_canonical_rank_ratio=ratio2,inlier_image_spread_px=spread2,positive_depth=positive)
     if g2 and positive:
      rp[t]=r;tp[t]=tv[:,0];obs.append(kk);row['pnp_success']=True
     else:obs.append(np.empty(0,dtype=int))
    else:obs.append(np.empty(0,dtype=int))
   else:obs.append(np.empty(0,dtype=int))
   gates.append(row)
  save('frame_gates.json',native(gates))
  if not any(len(x) for x in obs):
   save('run.json',dict(status='gated_no_fit',reason='No frame passed nondegenerate PnP support gate',input_hashes=hashes,reference_read=False,wall_seconds=time.perf_counter()-start));return
  sizes=[2*len(x)+7 for x in obs];nres=sum(sizes)+6*(T-2);jac=lil_matrix((nres,T*6),dtype=int);pos=0
  for t,n in enumerate(sizes):jac[pos:pos+n,t*6:(t+1)*6]=1;pos+=n
  for t in range(1,T-1):jac[pos:pos+6,(t-1)*6:(t+2)*6]=1;pos+=6
  dt=np.diff(times)
  def residual(x):
   x=x.reshape(T,6);R=Rotation.from_rotvec(x[:,:3]).as_matrix();trans=x[:,3:];parts=[]
   rprior=Rotation.from_matrix(R@R0.transpose(0,2,1)).as_rotvec()
   for t,ii in enumerate(obs):
    if len(ii):parts.append(((project(anchors[ii]@R[t].T+trans[t],K)-tracks[t,ids[ii]])/(3*np.sqrt(len(ii)))).ravel())
    parts.append(.05*rprior[t]);parts.append(.05*(trans[t]-t0[t])/.1);parts.append(np.array([max(.2-trans[t,2],0)*20]))
   vel=np.diff(trans,axis=0)/dt[:,None];omega=Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).as_rotvec()/dt[:,None];mean_dt=(dt[1:]+dt[:-1])/2
   acc=np.diff(vel,axis=0)/mean_dt[:,None]/10;aa=np.diff(omega,axis=0)/mean_dt[:,None]/80
   for j in range(T-2):parts.append(acc[j]);parts.append(aa[j])
   return np.concatenate(parts)
  p0=np.c_[Rotation.from_matrix(rp).as_rotvec(),tp];res=least_squares(residual,p0.ravel(),jac_sparsity=jac.tocsr(),loss='soft_l1',f_scale=1,max_nfev=50,diff_step=1e-4,ftol=1e-4,xtol=1e-4,gtol=1e-4)
  x=res.x.reshape(T,6);R=Rotation.from_rotvec(x[:,:3]).as_matrix();trans=x[:,3:];Rw=C[:3,:3]@R;tw=trans@C[:3,:3].T+C[:3,3];direct=np.array([len(ii)>=6 for ii in obs]);reprojection=[]
  for t,ii in enumerate(obs):
   er=np.linalg.norm(project(anchors[ii]@R[t].T+trans[t],K)-tracks[t,ids[ii]],axis=-1) if len(ii) else np.array([])
   reprojection.append(dict(frame=t,count=len(ii),median_px=float(np.median(er)) if len(er) else None,max_px=float(er.max()) if len(er) else None))
  out={k:init[k] for k in init.files};out.update(R_world=Rw.astype(np.float32),t_world_m=tw.astype(np.float32),R_camera=R.astype(np.float32),t_camera_m=trans.astype(np.float32),observed_mask=direct,depth_observed_mask=np.zeros(T,bool),reliability=direct.astype(np.float32),rotation_ambiguity=np.ones(T,bool))
  np.savez_compressed(OUT/'object_init.npz',**out)
  np.savez_compressed(OUT/'fixed_anchor_trajectories.npz',world_m=np.einsum('tij,nj->tni',Rw,anchors)+tw[:,None],track_ids=ids,observed_mask=np.array([np.isin(np.arange(len(ids)),ii) for ii in obs]),timestamp_seconds=times)
  save('reprojection.json',reprojection)
  assert all(sha(p)==hashes[str(p)] for p in allowed)
  save('run.json',dict(status='completed',reference_read=False,input_hashes=hashes,prediction_sha256=sha(OUT/'object_init.npz'),source=native(source_info),frames=T,directly_constrained_frames=int(direct.sum()),missing_frames=np.flatnonzero(~direct).tolist(),optimizer_success=bool(res.success),optimizer_message=res.message,nfev=res.nfev,cost=float(res.cost),wall_seconds=time.perf_counter()-start,protocol_sha256=sha(protocol),code_sha256=sha(__file__),note='One ordinary pose diagnostic, not trained Gaussians or S2; no reference-based selection; non-observed frames have priors only.'))
 finally:signal.alarm(0)
if __name__=='__main__':main()
