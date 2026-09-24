"""P1 input-only candidate paths and fixed-shape SE(3) refinement.

No reference paths or reference loaders. Canonical attachments are conditional
on each candidate source pose; overlap/appearance checks are not ground truth.
"""
from pathlib import Path
import argparse, json, time, hashlib, datetime, sys, itertools
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
ROOT=Path('/home/cai_tianshun/Project/HOI')
OLD=ROOT/'experiments/structured_hoi_20260923'
E=ROOT/'experiments/object_pose_refinement_20260924/run01'
sys.path.insert(0,str(OLD/'code'))
from object_init_general import ray_mesh_canonical

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray): return x.tolist()
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)): return [plain(v) for v in x]
 return x
def save(p,x): Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n')
def project(x,K):
 h=x@K.T
 return h[:,:2]/np.maximum(h[:,2:],.01)
def rotvec(R): return Rotation.from_matrix(R).as_rotvec()
def robust(v): return 2*(np.sqrt(1+np.asarray(v)**2)-1)
def pvec(R,t): return np.c_[rotvec(R),t].ravel()

class Problem:
 def __init__(self,dev):
  self.dev=dev; self.D=E/'pose'/dev; self.C=json.loads((E/'protocol/pose_config_v1.json').read_text())
  self.initpath=OLD/('object_init/run03/object_init.npz' if dev=='dev1' else 'object_init/dev2_refined01/object_init.npz')
  self.B=ROOT/'experiments/mosca_baseline_20260922/common_input' if dev=='dev1' else OLD/'data/dev2'
  self.seg=ROOT/'experiments/mosca_baseline_20260922/segmentation/segmentation.npz' if dev=='dev1' else self.B/'segmentation/segmentation.npz'
  self.a=dict(np.load(self.initpath)); self.m=json.loads((self.B/'input_manifest.json').read_text()); assert self.m['role']=='input_only'
  self.ts=self.a['timestamp_seconds']; self.L=len(self.ts); self.K=self.a['K']; self.V=self.a['canonical_vertices_m'].astype(float); self.F=self.a['faces']; self.S=self.a['centres_m'][::2].astype(float)
  self.ST=cKDTree(self.S); self.labels=np.load(self.seg)['entity_labels']; self.R0=self.a['R_camera'].astype(float); self.t0=self.a['t_camera_m'].astype(float)
  self.visible=self.a['observed_mask'].astype(bool); assert self.visible.sum()==self.C['path_selection']['fixed_visible_counts'][dev]
  self.keys=json.loads((self.D/'keyframes.json').read_text());self.kf=[x['frame_index'] for x in self.keys['keyframes']]
  self.obs=[]; self.rgb={}
  for i in range(self.L):
   uv=np.argwhere(self.labels[i]==2)[:,::-1].astype(float); uv=uv[np.linspace(0,len(uv)-1,64,dtype=int)] if len(uv) else np.zeros((64,2))
   # Only known background is a negative. Human/unknown are permitted occlusion.
   field=distance_transform_edt(self.labels[i]==0)
   d=np.load(self.B/f'unidepth_depth/{i:05d}.npz')['dep']; z=d[uv[:,1].astype(int),uv[:,0].astype(int)]
   good=np.isfinite(z)&(z>.1)&self.a['depth_observed_mask'][i]
   if good.any():
    med=np.median(z[good]);mad=np.median(abs(z[good]-med));good &= abs(z-med)<=max(.2,4*1.4826*mad)
   xyz=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(self.K).T*np.nan_to_num(z,nan=0)[:,None]
   self.obs.append((uv,field,xyz,good))
  self.tracklets=[]
  for f in self.kf:
   p=self.D/f'tracklet_keyframe_{f:05d}.npz'
   if not p.exists(): raise FileNotFoundError(p)
   a=dict(np.load(p)); self.tracklets.append(a)
  self.input_hashes={str(p):sha(p) for p in [self.initpath,self.B/'input_manifest.json',self.seg,E/'protocol/pose_config_v1.json',self.D/'keyframes.json',*sorted(self.D.glob('tracklet_keyframe_*.npz'))]}

 def frame_blocks(self,i,R,t):
  uv,field,xyz,good=self.obs[i];cam=self.S@R.T+t;puv=project(cam,self.K)
  if not self.visible[i]: return np.zeros(64),np.zeros(64),np.zeros(64)
  if self.dev=='dev1':
   hull=cv2.convexHull(project(self.V@R.T+t,self.K).astype(np.float32))[:,0]
   fwd=np.array([max(0.,-cv2.pointPolygonTest(hull,tuple(map(float,q)),True)) for q in uv])
   lens=np.linalg.norm(np.roll(hull,-1,axis=0)-hull,axis=1);cum=np.r_[0,np.cumsum(lens)]
   pos=np.linspace(0,cum[-1],64,endpoint=False);idx=np.minimum(np.searchsorted(cum,pos,side='right')-1,len(hull)-1)
   boundary=hull[idx]+((pos-cum[idx])/np.maximum(lens[idx],1e-8))[:,None]*(np.roll(hull,-1,axis=0)[idx]-hull[idx])
   back=map_coordinates(field,[boundary[:,1],boundary[:,0]],order=1,mode='constant',cval=100)
  else:
   fwd=cKDTree(puv).query(uv)[0]
   back=np.quantile(map_coordinates(field,[puv[:,1],puv[:,0]],order=1,mode='constant',cval=100),np.linspace(0,1,64))
  dep=np.zeros(64)
  if good.any(): dep[good]=self.ST.query((xyz[good]-t)@R)[0]/(.1*np.sqrt(good.sum()))*.15
  return fwd/16,back/16,dep

 def ious(self,R,t):
  vals=[]
  for i in range(self.L):
   cam=self.V@R[i].T+t[i];uv=project(cam,self.K); mask=np.zeros((480,640),np.uint8)
   for tri in np.rint(np.clip(uv[self.F],-2000,3000)).astype(np.int32): cv2.fillConvexPoly(mask,tri,1)
   pred=(mask>0)&((self.labels[i]==0)|(self.labels[i]==2)); gt=self.labels[i]==2
   vals.append((pred&gt).sum()/max(1,(pred|gt).sum()))
  return np.array(vals)

 def patch(self,f,uv):
  if f not in self.rgb:self.rgb[f]=cv2.imread(self.m['frame_paths'][f])[...,::-1]
  return cv2.getRectSubPix(self.rgb[f],(9,9),tuple(map(float,uv))).astype(float)/255

 def correspondences(self,R,t,out):
  bindings=[]; audit=[];cross=[]
  for f,tr in zip(self.kf,self.tracklets):
   u=tr['source_uv'][0];ca,hit=ray_mesh_canonical(u,R[f],t[f],self.V,self.F,self.K)
   bindings.append({'frame':f,'canonical':ca,'hit':hit,'confirmed':np.zeros(len(u),bool),'patches':[self.patch(f,x) for x in u]})
  for ia,ib in itertools.combinations(range(len(bindings)),2):
   A,B=self.tracklets[ia],self.tracklets[ib]; ba,bb=bindings[ia],bindings[ib]
   at=A['target_frame'][:,0].astype(int);bt=B['target_frame'][:,0].astype(int); shared=sorted(set(at)&set(bt)); pairs={}
   for f in shared:
    ai=int(np.flatnonzero(at==f)[0]);bi=int(np.flatnonzero(bt==f)[0]);va=np.flatnonzero(A['adopted'][ai]);vb=np.flatnonzero(B['adopted'][bi])
    if not len(va) or not len(vb):continue
    au=A['target_uv'][ai,va];bu=B['target_uv'][bi,vb];dist,near=cKDTree(bu).query(au);back=cKDTree(au).query(bu)[1]
    for j in np.flatnonzero((dist<=2.5)&(back[near]==np.arange(len(va)))):pairs.setdefault((int(va[j]),int(vb[near[j]])),[]).append((f,float(dist[j])))
   for (qa,qb),matches in pairs.items():
    if len(matches)<2:continue
    pa,pb=ba['patches'][qa],bb['patches'][qb];ga=pa.mean(2);gb=pb.mean(2);ga-=ga.mean();gb-=gb.mean();ncc=float((ga*gb).sum()/max(1e-9,np.linalg.norm(ga)*np.linalg.norm(gb)));rgb=float(np.mean(abs(pa.mean((0,1))-pb.mean((0,1)))))
    distance=float(np.linalg.norm(ba['canonical'][qa]-bb['canonical'][qb])) if ba['hit'][qa] and bb['hit'][qb] else None
    good=distance is not None and distance<=.025 and ncc>=.5 and rgb<=35/255
    cross.append(dict(source_a=ba['frame'],query_a=qa,source_b=bb['frame'],query_b=qb,shared_matches=matches,patch_ncc=ncc,rgb_mean_abs=rgb,canonical_separation_m=distance,confirmed_conditionally=good))
    if good:ba['confirmed'][qa]=True;bb['confirmed'][qb]=True
  records=[]
  for j,(b,tr) in enumerate(zip(bindings,self.tracklets)):
   ids=np.flatnonzero(b['confirmed']);ids=sorted(ids,key=lambda q:(-int(tr['adopted'][:,q].sum()),int(q)))[:48];b['confirmed'][:]=False;b['confirmed'][ids]=True
   for q in range(len(b['hit'])):
    audit.append({'source_frame':b['frame'],'query':q,'canonical_candidate':b['canonical'][q].tolist() if b['hit'][q] else None,'state':'reliable_observed_conditional' if b['confirmed'][q] else ('candidate' if b['hit'][q] else 'rejected'),'reason':'cross_source_overlap_appearance_geometry' if b['confirmed'][q] else ('no_sufficient_cross_source_confirmation' if b['hit'][q] else 'no_frontmost_triangle_hit'),'used_observations':int(tr['adopted'][:,q].sum()) if b['confirmed'][q] else 0})
   for q in ids:
    valid=np.flatnonzero(tr['adopted'][:,q]);n=max(1,len(valid));
    for idx in valid:records.append((int(tr['target_frame'][idx,q]),b['canonical'][q],tr['target_uv'][idx,q],.5/(3*np.sqrt(n*max(1,len(ids)))),b['frame'],int(q)))
    # Source self-projection defined the candidate; do not count it as evidence
    # or quietly add a soft tether to the candidate source pose.
  save(out/'canonical_correspondences.json',{'conditional_on_candidate_path':True,'not_ground_truth':True,'bindings':audit,'cross_source_checks':cross,'accepted_queries':sum(int(b['confirmed'].sum()) for b in bindings),'temporal_records':sum(int(r[0]!=r[4]) for r in records),'self_definition_records':sum(int(r[0]==r[4]) for r in records),'raw_track_observation_count':sum(int(tr['adopted'].sum()) for tr in self.tracklets),'no_strong_term_for_unconfirmed':True})
  return records,cross,audit

 def temporal(self,R,t):
  dt=np.diff(self.ts);v=np.diff(t,axis=0)/dt[:,None];w=rotvec(R[1:]@R[:-1].transpose(0,2,1))/dt[:,None];den=(dt[:-1]+dt[1:])/2
  return np.c_[np.diff(v,axis=0)/den[:,None]/10,np.diff(w,axis=0)/den[:,None]/80]*.1

 def blocks(self,R,t,records):
  front=[];back=[];depth=[]
  for i in range(self.L):
   f,b,d=self.frame_blocks(i,R[i],t[i]);front.append(f);back.append(b);depth.append(d)
  tracks=np.array([(project((ca@R[i].T+t[i])[None],self.K)[0]-uv)*weight for i,ca,uv,weight,_,_ in records]).reshape(-1,2)
  return {'silhouette_forward':np.array(front),'silhouette_background':np.array(back),'estimated_depth_weak':np.array(depth),'confirmed_tracks':tracks,'actual_time_acceleration':self.temporal(R,t)}

 def refine(self,R,t,records,out):
  n=self.L;num=192*n+2*len(records)+6*(n-2);sp=lil_matrix((num,6*n),dtype=int)
  # Residual order: per frame 3x64, then tracks, then acceleration.
  for i in range(n):sp[i*192:(i+1)*192,i*6:(i+1)*6]=1
  for k,(i,*_) in enumerate(records):sp[192*n+k*2:192*n+(k+1)*2,i*6:(i+1)*6]=1
  base=192*n+2*len(records)
  for i in range(1,n-1):sp[base+(i-1)*6:base+i*6,(i-1)*6:(i+2)*6]=1
  calls=0;start=time.perf_counter()
  def fun(p):
   nonlocal calls
   calls+=1;x=p.reshape(n,6);rr=Rotation.from_rotvec(x[:,:3]).as_matrix();tt=x[:,3:];vals=[]
   for i in range(n):vals.append(np.concatenate(self.frame_blocks(i,rr[i],tt[i])))
   for i,ca,uv,weight,_,_ in records:vals.append((project((ca@rr[i].T+tt[i])[None],self.K)[0]-uv)*weight)
   vals.append(self.temporal(rr,tt).ravel())
   if calls%150==0:print(self.dev,out.name,'calls',calls,'seconds',round(time.perf_counter()-start,1),flush=True)
   return np.concatenate(vals)
  lo=np.full((n,6),-np.inf);hi=-lo;lo[:,5]=.4;hi[:,5]=8
  p0=pvec(R,t);before=fun(p0)
  result=least_squares(fun,p0,jac_sparsity=sp.tocsr(),bounds=(lo.ravel(),hi.ravel()),loss='soft_l1',f_scale=1,max_nfev=35,diff_step=1e-4,ftol=1e-4,xtol=1e-4,gtol=1e-4)
  xx=result.x.reshape(n,6);rr=Rotation.from_rotvec(xx[:,:3]).as_matrix();tt=xx[:,3:]
  save(out/'optimizer.json',dict(seconds=time.perf_counter()-start,nfev=result.nfev,residual_calls=calls,success=result.success,message=result.message,initial_robust_sum=float(robust(before).sum()),final_robust_sum=float(robust(result.fun).sum()),code_sha256=sha(__file__)))
  return rr,tt

 def coarse_candidates(self,external,out):
  cand=[];rows=[];extra=np.load(external) if external else None
  for frame in self.kf:
   starts=[]
   if extra is not None:
    index=int(np.flatnonzero(extra['frame_indices']==frame)[0])
    for j in range(len(extra['R_camera'][index])):starts.append((extra['R_camera'][index,j],extra['t_camera_m'][index,j],f'megapose_{j}'))
   else:
    val,axis=np.linalg.eigh(self.V.T@self.V);axis=axis[:,np.argmax(val)]
    for deg,fac in itertools.product([0,90,180,270],[.9,1,1.1]):starts.append((self.R0[frame]@Rotation.from_rotvec(axis*np.radians(deg)).as_matrix(),self.t0[frame]*fac,f'fallback_{deg}_{fac}'))
   local=[dict(R=self.R0[frame],t=self.t0[frame],source='old_input',cost=sum(float(robust(b).sum()) for b in self.frame_blocks(frame,self.R0[frame],self.t0[frame])))]
   for R,t,source in starts:
    if not np.isfinite(R).all() or not np.isfinite(t).all() or t[2]<.4 or t[2]>8:rows.append({'frame':frame,'source':source,'rejected':'invalid/unsupported depth'});continue
    def fun(p):return np.concatenate(self.frame_blocks(frame,Rotation.from_rotvec(p[:3]).as_matrix(),p[3:]))
    res=least_squares(fun,np.r_[rotvec(R),t],loss='soft_l1',f_scale=1,max_nfev=24,diff_step=1e-4,ftol=1e-4,xtol=1e-4,gtol=1e-4,bounds=([-np.inf]*5+[.4],[np.inf]*5+[8]))
    rr=Rotation.from_rotvec(res.x[:3]).as_matrix();tt=res.x[3:];cost=float(robust(res.fun).sum());c=dict(R=rr,t=tt,source=source,cost=cost);rows.append({'frame':frame,**c,'nfev':res.nfev})
    if np.min((self.V[np.unique(self.F)]@rr.T+tt)[:,2])>.05:local.append(c)
   # Preserve original input candidate, then best four genuinely different starts.
   selected=[local[0]]
   for c in sorted(local[1:],key=lambda c:c['cost']):
    if all(np.linalg.norm(rotvec(c['R']@d['R'].T))>np.radians(3) or np.linalg.norm(c['t']-d['t'])>.005 for d in selected):selected.append(c)
    if len(selected)==5:break
   cand.append(selected)
  save(out/'candidate_records.json',{'external_candidates':str(external) if external else None,'external_sha256':sha(external) if external else None,'all_local_results':rows,'retained':cand,'counts':[len(c) for c in cand]})
  return cand

 def paths(self,candidates):
  beam=[(0.,[])]
  for k,cs in enumerate(candidates):
   nexts=[]
   for score,ids in beam:
    for j,c in enumerate(cs):
     trans=0.
     if ids:
      prev=candidates[k-1][ids[-1]];dt=self.ts[self.kf[k]]-self.ts[self.kf[k-1]]
      trans=.03*(np.linalg.norm(c['t']-prev['t'])/dt/.5)**2+.02*(np.linalg.norm(rotvec(c['R']@prev['R'].T))/dt/2)**2
     nexts.append((score+c['cost']+trans,ids+[j]))
   beam=sorted(nexts,key=lambda v:v[0])[:30]
  paths=[dict(name='path00_old_start',R=self.R0.copy(),t=self.t0.copy(),ids=[0]*len(self.kf),beam_score=None)]
  for cost,ids in beam:
   if all(i==0 for i in ids):continue
   rc=np.array([candidates[k][j]['R'] for k,j in enumerate(ids)]);tc=np.array([candidates[k][j]['t'] for k,j in enumerate(ids)])
   # Require a decision-relevant alternative to every retained path at one keyframe.
   if any(max(np.max(np.linalg.norm(rotvec(rc@p['R'][self.kf].transpose(0,2,1)),axis=1))/np.radians(15),np.max(np.linalg.norm(tc-p['t'][self.kf],axis=1))/.03)<1 for p in paths):continue
   dr=rc@self.R0[self.kf].transpose(0,2,1);delta=Slerp(self.ts[self.kf],Rotation.from_matrix(dr))(np.clip(self.ts,self.ts[self.kf[0]],self.ts[self.kf[-1]])).as_matrix()
   dt=tc-self.t0[self.kf];tt=self.t0+np.stack([np.interp(self.ts,self.ts[self.kf],dt[:,j]) for j in range(3)],axis=1)
   paths.append(dict(name=f'path{len(paths):02d}_beam',R=delta@self.R0,t=tt,ids=ids,beam_score=cost))
   if len(paths)==3:break
  return paths

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev',choices=['dev1','dev2'],required=True);ap.add_argument('--candidates',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();P=Problem(args.dev)
 save(args.output/'run.json',dict(status='running',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_reference_used=True,input_hashes=P.input_hashes,code_sha256=sha(__file__),GPU_used=False))
 candidates=P.coarse_candidates(args.candidates,args.output);paths=P.paths(candidates);oldiou=P.ious(P.R0,P.t0);results=[]
 for path in paths:
  out=args.output/path['name'];out.mkdir();R,t=path['R'],path['t'];save(out/'path_start.json',{'candidate_ids':path['ids'],'beam_score':path['beam_score']});np.savez_compressed(out/'initial_path.npz',R_camera=R,t_camera_m=t)
  records,cross,bindings=P.correspondences(R,t,out);R,t=P.refine(R,t,records,out);blocks=P.blocks(R,t,records);iou=P.ious(R,t)
  components={k:float(robust(v).sum()/P.L) for k,v in blocks.items()};raw_count=sum(int(a['adopted'].sum()) for a in P.tracklets);used=sum(int(r[0]!=r[4]) for r in records);coverage=used/max(1,raw_count)
  rgbbad=float(np.mean([max(0,1-x['patch_ncc']) for x in cross])) if cross else 1.
  score=sum(components.values())+.1*rgbbad+.1*(1-coverage)
  a=dict(P.a);a['R_camera']=R.astype(np.float32);a['t_camera_m']=t.astype(np.float32);C=a['c2w'];a['R_world']=(C[:3,:3]@R).astype(np.float32);a['t_world_m']=(t@C[:3,:3].T+C[:3,3]).astype(np.float32);a['reliability']=np.where(P.visible,np.minimum(1,(P.labels==2).sum((1,2))/1000)*iou,0).astype(np.float32)
  np.savez_compressed(out/'object_init.npz',**a)
  reasons=[]
  if not np.isfinite(R).all() or not np.isfinite(t).all():reasons.append('nonfinite')
  if not np.allclose(R@R.transpose(0,2,1),np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(R),1,atol=1e-5):reasons.append('illegal_SO3')
  if min(float(np.min((P.V[np.unique(P.F)]@R[i].T+t[i])[:,2])) for i in range(P.L))<=.05:reasons.append('surface_behind_camera')
  if float(iou[P.visible].mean()-oldiou[P.visible].mean())<-.02:reasons.append('mean_visible_IoU_drop_exceeds_0.02')
  diag=dict(path=path['name'],score=score,components=components,rgb_cross_source_inconsistency=rgbbad,confirmed_observation_coverage=coverage,used_temporal_observations=used,raw_reliable_observations=raw_count,old_mean_visible_IoU=float(oldiou[P.visible].mean()),new_mean_visible_IoU=float(iou[P.visible].mean()),fixed_visible_frames=np.flatnonzero(P.visible),iou_gate_pass=not reasons,rejection_reasons=reasons,pose_file=str(out/'object_init.npz'),pose_sha256=sha(out/'object_init.npz'))
  save(out/'input_score.json',diag);results.append(diag)
  np.savez_compressed(out/'input_diagnostics.npz',old_iou=oldiou,new_iou=iou,observed=P.visible,timestamps=P.ts,translation_speed_m_s=np.linalg.norm(np.diff(t,axis=0),axis=1)/np.diff(P.ts),rotation_speed_rad_s=np.linalg.norm(rotvec(R[1:]@R[:-1].transpose(0,2,1)),axis=1)/np.diff(P.ts))
  save(out/'frame_states.json',[dict(frame=i,time=P.ts[i],state='candidate' if P.visible[i] else 'predicted',direct_input_visible=bool(P.visible[i]),observed_object_pixels=int((P.labels[i]==2).sum()),estimated_depth_used=bool(P.a['depth_observed_mask'][i]),iou=iou[i],material_identity='conditional/unresolved; not guaranteed by silhouette') for i in range(P.L)])
 good=sorted((r for r in results if r['iou_gate_pass']),key=lambda r:r['score']);chosen=good[0] if good else None
 freeze={'status':'accepted_for_S1_star' if chosen else 'rejected_no_acceptable_P1','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dev':args.dev,'selected':chosen,'all_candidates':results,'no_independent_reference_used':True,'parameter_tuning_after_reference_allowed':False,'pose_config_sha256':sha(E/'protocol/pose_config_v1.json'),'code_sha256':sha(__file__),'wall_seconds':time.perf_counter()-start}
 save(args.output/'selection_frozen.json',freeze);save(args.output/'run.json',{**freeze,'input_hashes':P.input_hashes,'GPU_used':False});print(json.dumps(plain(freeze),indent=2),flush=True)
if __name__=='__main__':main()
