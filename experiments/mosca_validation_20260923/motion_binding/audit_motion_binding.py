#!/usr/bin/env python3
"""CPU-only frozen MoSca binding audit. No evaluation geometry is read.
Run with CUDA_VISIBLE_DEVICES='' envs/mosca/bin/python <this file>.
All diagnostic identities come from RGB-derived SAM2 masks/visible CoTracker tracks.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,csv
from pathlib import Path
import numpy as np
import torch,cv2,yaml
from scipy.spatial import cKDTree
from pytorch3d.transforms import quaternion_to_matrix
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'experiments/mosca_baseline_20260922'
OUT=Path(__file__).resolve().parent
MOS=ROOT/'third_party/MoSca'
sys.path.insert(0,str(MOS/'lib_mosca/scaffold_utils'))
from dualquat_helper import Rt2dq,dq2unitdq,dq2Rt
# Keep reductions on CPU deterministic and bounded; do not instantiate the model.
torch.set_num_threads(4)
NAMES={-1:'unknown',0:'background',1:'person',2:'object'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o): Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def quant(x):
 x=np.asarray(x); x=x[np.isfinite(x)]
 return None if not len(x) else {'n':int(len(x)),'mean':float(x.mean()),'quantile_0_10_25_50_75_90_99_100':np.quantile(x,[0,.1,.25,.5,.75,.9,.99,1]).tolist()}
def counts(x): return {v:int((x==k).sum()) for k,v in NAMES.items()}
def majority(votes,minimum=3,purity=.8):
 total=votes.sum(-1); labels=votes.argmax(-1).astype(np.int8)
 labels[(total<minimum)|(votes.max(-1)<purity*total)]=-1
 return labels

def project(x,K,C):
 cam=(x-C[:3,3])@C[:3,:3]
 uv=cam[...,:2]/np.maximum(cam[...,2:],1e-8)
 uv=uv*np.array([K[0,0],K[1,1]])+np.array([K[0,2],K[1,2]])
 return uv,cam[...,2]

def sample(a,uv):
 H,W=a.shape[:2]; p=np.trunc(uv).astype(np.int64); valid=(p[...,0]>=0)&(p[...,0]<W)&(p[...,1]>=0)&(p[...,1]<H)&np.isfinite(uv).all(-1)
 return a[p[...,1].clip(0,H-1),p[...,0].clip(0,W-1)],valid

def sparse_raster(xyz,R,scales,opacity,K,C,uv,W=640,H=480):
 """Faithful scalar-pixel native-add3 alpha compositor; checked against GPU export.
 Reproduces native padding/crop, covariance floor, tile bounds and alpha cutoffs.
 Only output known pixel contributors, not full images or radiance.
 """
 xyz=np.asarray(xyz,dtype=np.float64); C=np.asarray(C,dtype=np.float64)
 cam=(xyz-C[:3,3])@C[:3,:3]; z=cam[:,2]; fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
 if abs(H//2-cy)>1 or abs(W//2-cx)>1:
  nw=int(2*max(cx,W-cx));nh=int(2*max(cy,H-cy));crop=np.array([0 if cx>W-cx else nw-W,0 if cy>H-cy else nh-H])
 else: nw,nh=W,H;crop=np.array([0,0])
 mu=cam[:,:2]/(z[:,None]+1e-7)*np.array([fx,fy])+np.array([nw/2-.5,nh/2-.5])
 rc=C[:3,:3].T[None]@R
 cov=(rc*scales[:,None,:]**2)@rc.transpose(0,2,1)
 zz=np.maximum(z,1e-8); clx=np.clip(cam[:,0]/zz,-1.3*nw/(2*fx),1.3*nw/(2*fx));cly=np.clip(cam[:,1]/zz,-1.3*nh/(2*fy),1.3*nh/(2*fy))
 J=np.zeros((len(z),2,3));J[:,0,0]=fx/zz;J[:,1,1]=fy/zz;J[:,0,2]=-fx*clx/zz;J[:,1,2]=-fy*cly/zz
 v=J@cov@J.transpose(0,2,1);v[:,0,0]+=.3;v[:,1,1]+=.3
 aa,bb,cc=v[:,0,0],v[:,0,1],v[:,1,1];det=aa*cc-bb**2
 conic=np.stack([cc,-bb,aa],-1)/np.maximum(det[:,None],1e-20)
 mid=.5*(aa+cc);rad=np.ceil(3*np.sqrt(np.maximum(mid+np.sqrt(np.maximum(.1,mid**2-det)),0)))
 grid=np.array([(nw+15)//16,(nh+15)//16]);rectmin=np.clip(np.trunc((mu-rad[:,None])/16),0,grid).astype(int);rectmax=np.clip(np.trunc((mu+rad[:,None]+15)/16),0,grid).astype(int)
 result=[]
 for q in np.asarray(uv):
  qp=q+crop;tile=np.floor(qp/16).astype(int)
  candidates=np.flatnonzero((z>.02)&(det>0)&((rectmin<=tile)&(rectmax>tile)).all(-1))
  candidates=candidates[np.argsort(z[candidates],kind='stable')];d=mu[candidates]-qp
  c=conic[candidates];power=-.5*(c[:,0]*d[:,0]**2+c[:,2]*d[:,1]**2)-c[:,1]*d[:,0]*d[:,1]
  a=np.minimum(.99,np.asarray(opacity).ravel()[candidates]*np.exp(np.minimum(power,0)))
  ok=(power<=0)&(a>=1/255);candidates=candidates[ok];a=a[ok]
  after=np.cumprod(1-a);before=np.r_[1,after[:-1]];ok=after>=.0001;candidates=candidates[ok];weight=(a*before)[ok]
  result.append((candidates,weight))
 return result

def get_weights(d):
 node=d['scf._node_xyz'];nodeR=quaternion_to_matrix(torch.nn.functional.normalize(d['scf._node_rotation'],dim=-1))
 ti=d['ref_time'];att=d['attach_ind'];idx=d['scf.topo_knn_ind'][att];mask=d['scf.topo_knn_mask'][att]
 src=torch.einsum('nij,nj->ni',nodeR[ti,att],d['_xyz'])+node[ti,att]
 srcR=nodeR[ti,att]@quaternion_to_matrix(torch.nn.functional.normalize(d['_rotation'],dim=-1))
 sig=torch.sigmoid(d['scf._node_sigma_logit'])*d['scf.max_sigma']
 ds=((src[:,None]-node[ti[:,None],idx])**2).sum(-1)
 base=(torch.exp(-ds/(2*sig[idx].squeeze(-1)**2))+1e-6)*mask
 cor=(base+d['_skinning_weight']).abs() if d['w_correction_flag'] else base
 if d['scf.w_corr_maintain_sum_flag']: cor=cor*base.sum(-1,keepdim=True)/cor.sum(-1,keepdim=True).clamp_min(1e-6)
 total=cor.sum(-1);w=cor/total[:,None].clamp_min(1e-6)
 return {'node':node,'nodeR':nodeR,'ti':ti,'idx':idx,'mask':mask,'src':src,'srcR':srcR,'base':base,'total':total,'w':w,'dyn':total.clamp(0,1)}

def warp(d,g,t,subset=None,dyn_off=False):
 sel=slice(None) if subset is None else subset
 idx=g['idx'][sel];ti=g['ti'][sel];R0=g['nodeR'][ti[:,None],idx];x0=g['node'][ti[:,None],idx]
 R1=g['nodeR'][t,idx];x1=g['node'][t,idx];delta=R1@R0.transpose(-1,-2);trans=x1-torch.einsum('nkij,nkj->nki',delta,x0)
 dq=Rt2dq(delta,trans);dq=(dq*g['w'][sel,:,None]).sum(1)
 dyn=torch.full_like(g['dyn'][sel],.999) if dyn_off else g['dyn'][sel]
 unit=torch.zeros_like(dq);unit[:,0]=1;dq=dq*dyn[:,None]+unit*(1-dyn[:,None]);R,tr=dq2Rt(dq2unitdq(dq))
 return torch.einsum('nij,nj->ni',R,g['src'][sel])+tr,R@g['srcR'][sel]

def main():
 start=time.time();OUT.mkdir(exist_ok=True)
 manifest=json.loads((BASE/'common_input/input_manifest.json').read_text());K=np.array(manifest['K']);C=np.array(manifest['c2w']);times=np.array(manifest['timestamp_seconds']);cfg=yaml.safe_load((BASE/'mosca_cotracker/config.yaml').read_text())
 q=json.loads((BASE/'common_input/queries_first_frame.json').read_text());q_uv=np.array(q['points'][:6]);ids=q['point_ids'][:6]
 dpath=BASE/'mosca_cotracker/photometric_d_model_native_add3.pth';spath=BASE/'mosca_cotracker/photometric_s_model_native_add3.pth'
 d=torch.load(dpath,map_location='cpu',weights_only=False);s=torch.load(spath,map_location='cpu',weights_only=False);g=get_weights(d)
 nstatic=len(s['_xyz']); ndyn=len(d['_xyz']);M=d['scf._node_xyz'].shape[1]
 # First priority: exact alpha-weighted source queries, DQB gate and underlying node curves.
 f0=np.load(BASE/'mosca_cotracker/diagnostics/keyframes/00000.npz')
 contrib=sparse_raster(f0['xyz_world'],f0['rotation_frame'],f0['scales'],f0['opacity'],K,C,q_uv)
 old=np.load(BASE/'mosca_cotracker/diagnostics/query_trajectories.npz');check={}
 cpu_a=np.array([w.sum() for _,w in contrib]);cpu_f=np.array([w[i>=nstatic].sum()/w.sum() for i,w in contrib])
 check['alpha_max_abs_difference']=float(np.max(abs(cpu_a-old['source_alpha'])))
 check['dynamic_fraction_max_abs_difference']=float(np.max(abs(cpu_f-old['source_dynamic_fraction'])))
 check['native_pixel_centre_note']='Reproduces native padding crop, not exact manifest principal point; separate root adapter handles calibration change.'
 # DQB coordinate match: validates ALL dynamic Gaussians at two exported times.
 check['dqb_keyframe_xyz_max_m']={}
 for t in [0,28]:
  xyz,_=warp(d,g,t);kf=np.load(BASE/f'mosca_cotracker/diagnostics/keyframes/{t:05d}.npz')
  check['dqb_keyframe_xyz_max_m'][str(t)]=float(np.max(abs(xyz.numpy()-kf['xyz_world'][nstatic:])))
 assert check['alpha_max_abs_difference']<2e-3 and check['dynamic_fraction_max_abs_difference']<2e-3,check
 assert max(check['dqb_keyframe_xyz_max_m'].values())<1e-4,check
 dyn_unique=np.unique(np.concatenate([i[i>=nstatic]-nstatic for i,w in contrib])); unique=torch.tensor(dyn_unique,dtype=torch.long)
 allxyz=[];offxyz=[]
 for t in range(len(times)):
  allxyz.append(warp(d,g,t,unique)[0].numpy());offxyz.append(warp(d,g,t,unique,True)[0].numpy())
 allxyz=np.array(allxyz);offxyz=np.array(offxyz);pred=[];predoff=[];rows=[];qnodes=[]
 for j,(ii,ww) in enumerate(contrib):
  alpha=ww.sum();ii0=ii[ii<nstatic];ww0=ww[ii<nstatic];did=ii[ii>=nstatic]-nstatic;dw=ww[ii>=nstatic];pos=np.searchsorted(dyn_unique,did)
  static=(f0['xyz_world'][ii0]*ww0[:,None]).sum(0)
  p=(static[None]+(allxyz[:,pos]*dw[None,:,None]).sum(1))/alpha
  po=(static[None]+(offxyz[:,pos]*dw[None,:,None]).sum(1))/alpha
  pred.append(p);predoff.append(po)
  nodew=np.zeros(M);np.add.at(nodew,g['idx'][did].numpy().ravel(),(g['w'][did].numpy()*dw[:,None]/alpha).ravel());qnodes.append(nodew)
  nodemotion=np.linalg.norm(g['node'].numpy()-g['node'][0].numpy()[None],axis=-1)
  gate=g['dyn'][did].numpy();sums=g['total'][did].numpy()
  rows.append({'query_id':ids[j],'manual_entity':q['entity'][j],'source_alpha':float(alpha),'static_alpha_fraction':float(ww0.sum()/alpha),'dynamic_alpha_fraction':float(dw.sum()/alpha),'dynamic_conditional_rbf_sum':float((sums*dw).sum()/dw.sum()),'dynamic_conditional_identity_mix':float(((1-gate)*dw).sum()/dw.sum()),'all_alpha_identity_mix_plus_static':float((ww0.sum()+((1-gate)*dw).sum())/alpha),'dynamic_alpha_fraction_rbf_lt_0_1':float(dw[sums<.1].sum()/dw.sum()),'source_dynamic_gs_count':len(did),'source_static_gs_count':len(ii0),'node_weighted_max_displacement_m':float((nodew*nodemotion.max(0)).sum()/nodew.sum()),'node_weighted_trajectory_max_displacement_m':float(np.linalg.norm(np.einsum('tnc,n->tc',g['node'].numpy()-g['node'][0].numpy()[None],nodew/nodew.sum()),axis=-1).max()),'native_query_max_displacement_m':float(np.linalg.norm(p-p[0],axis=-1).max()),'dyn_o_off_query_max_displacement_m':float(np.linalg.norm(po-po[0],axis=-1).max()),'top10_node_weight':sorted([{'node':int(k),'conditional_dynamic_weight':float(nodew[k]/nodew.sum()),'node_max_displacement_m':float(nodemotion[:,k].max())} for k in np.flatnonzero(nodew)],key=lambda z:-z['conditional_dynamic_weight'])[:10]})
 pred=np.stack(pred,1);predoff=np.stack(predoff,1)
 check['export_query_xyz_max_abs_difference_m']=float(np.max(abs(pred-old['predicted'])))
 assert check['export_query_xyz_max_abs_difference_m']<2e-3,check
 dump(OUT/'priority_query_gate_results.json',{'validation':check,'flags':{k:d[k].item() for k in ['dyn_o_flag','w_correction_flag','leaf_local_flag','scf.w_corr_maintain_sum_flag']},'query_results':rows,'counterfactual_note':'Frozen checkpoint only; dyn_o=False uses constant .999. No re-training, no reference coordinates. Normalized DQB is nonlinear: identity mixture is not a linear motion attenuation ratio.'})
 np.savez_compressed(OUT/'frozen_query_motion.npz',actual=pred,dyn_o_off=predoff,frame_times=times,query_id=np.array(ids),source_node_weight=np.array(qnodes))
 print('PRIORITY',json.dumps({'validation':check,'rows':[{k:v for k,v in r.items() if k!='top10_node_weight'} for r in rows]},ensure_ascii=False),flush=True)
 # RGB-only labels: 2px eroded mask interiors; stable track vote >=3, >=80%.
 seg=np.load(BASE/'segmentation/segmentation.npz')['entity_labels'];T,H,W=seg.shape
 interior=np.full_like(seg,255)
 for t in range(T):
  for ent in [0,1,2]:
   m=cv2.erode((seg[t]==ent).astype(np.uint8),np.ones((5,5),np.uint8))>0;interior[t,m]=ent
 tr=np.load(BASE/'common_input/uniform_cotracker_tap.npz');uv=tr['tracks'];vis=tr['visibility'];tv=np.zeros((uv.shape[1],3),np.int32);dep=[];depthmask=[]
 for t in range(T):
  lab,inside=sample(interior[t],uv[t]);ok=vis[t]&inside&(lab<3)
  for e in [0,1,2]:tv[:,e]+=ok&(lab==e)
  z=np.load(BASE/f'common_input/unidepth_depth/{t:05d}.npz')['dep'];dep.append(z)
  err=np.abs(cv2.Laplacian(z.astype(np.float32),cv2.CV_64F,ksize=5));mask=err<=np.median(z)*float(cfg['depth_boundary_th'])
  mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
  depthmask.append(mask&(z>1e-3)&(z<1000))
 dep=np.array(dep);depthmask=np.array(depthmask);tracklabel=majority(tv);validobs=np.zeros_like(vis)
 for t in range(T):
  vals,inside=sample(depthmask[t],uv[t]);validobs[t]=vis[t]&inside&vals
 keep=validobs.sum(0)>=int(cfg['tap_loading_min_valid_cnt']);ident=np.load(BASE/'mosca_cotracker/track_identification.npz')
 filterok=int(keep.sum())==len(ident['dynamic_track_mask'])
 if not filterok: raise RuntimeError(f'Filtered mapping mismatch {keep.sum()} vs {len(ident["dynamic_track_mask"])}')
 # Node labels must have input depth consistency AND proximity to a visible stably labeled track with same mask label.
 def node_labels(xyz,certain):
  votes=np.zeros((xyz.shape[1],3),np.int32);eligible=np.zeros(xyz.shape[:2],bool);obslab=np.full(xyz.shape[:2],-1,np.int8)
  for t in range(T):
   p,z=project(xyz[t],K,C);lab,inside=sample(interior[t],p);dz,_=sample(dep[t],p)
   valid=certain[t]&inside&(lab<3)&(z>0)&(np.abs(z-dz)<=np.maximum(.05,.03*dz))
   near=np.zeros(len(p),bool)
   for ent in [0,1,2]:
    ti=np.flatnonzero(vis[t]&(tracklabel==ent));ii=np.flatnonzero(valid&(lab==ent))
    if len(ti) and len(ii):near[ii]=cKDTree(uv[t,ti]).query(p[ii],k=1)[0]<=3
   ok=valid&near;eligible[t]=ok;obslab[t,ok]=lab[ok]
   for e in [0,1,2]:votes[:,e]+=ok&(lab==e)
  return majority(votes),votes,eligible,obslab
 nl,nv,eligible,nobs=node_labels(g['node'].numpy(),d['scf._node_certain'].numpy())
 geo=torch.load(BASE/'mosca_cotracker/mosca/mosca.pth',map_location='cpu',weights_only=False)
 # geometry checkpoint is scaffold-only with unprefixed keys.
 gl,gv,_,_=node_labels(geo['_node_xyz'].numpy(),geo['_node_certain'].numpy())
 # Independent GS labels from six stored keyframes: input-depth-gated mask projection, not inherited from nodes.
 votes=np.zeros((nstatic+ndyn,3),np.int32);keyframes=[0,16,20,24,28,113];explanation=[]
 for t in keyframes:
  kf=np.load(BASE/f'mosca_cotracker/diagnostics/keyframes/{t:05d}.npz');p,z=project(kf['xyz_world'],K,C);lab,inside=sample(interior[t],p);dz,_=sample(dep[t],p)
  ok=inside&(lab<3)&(z>0)&(np.abs(z-dz)<=np.maximum(.05,.03*dz))&(kf['opacity'].ravel()>.01)
  for e in [0,1,2]:votes[:,e]+=ok&(lab==e)
  # Deterministic evenly spaced object interior pixels. This measures explanation of input mask, not GT visibility.
  yy,xx=np.where(interior[t]==2);pts=np.stack([xx,yy],-1);pts=pts[np.linspace(0,len(pts)-1,min(32,len(pts))).astype(int)] if len(pts) else pts
  rr=sparse_raster(kf['xyz_world'],kf['rotation_frame'],kf['scales'],kf['opacity'],K,C,pts)
  al=[];df=[];im=[]
  for ii,ww in rr:
   a=ww.sum();dynamic=ii>=nstatic;al.append(a);df.append(ww[dynamic].sum()/max(a,1e-8));im.append((ww[~dynamic].sum()+(ww[dynamic]*(1-g['dyn'][ii[dynamic]-nstatic].numpy())).sum())/max(a,1e-8))
  explanation.append({'frame':t,'time_s':float(times[t]),'sam2_object_pixels':int((seg[t]==2).sum()),'sampled_eroded_object_pixels':len(pts),'mean_alpha':float(np.mean(al)) if al else None,'mean_dynamic_fraction':float(np.mean(df)) if df else None,'mean_static_plus_identity_mixture':float(np.mean(im)) if im else None,'interpretation':'Prediction explanation at estimated input object-mask interiors; no GT visibility assertion.'})
 gsl=majority(votes,minimum=2,purity=.8);dl=gsl[nstatic:];skidx=g['idx'].numpy();w=g['w'].numpy();nlab=nl[skidx]
 skstats={}
 for e in [0,1,2,-1]:
  sel=dl==e
  if not sel.any():continue
  skstats[NAMES[e]]={'gaussian_count':int(sel.sum()),'rbf_sum':quant(g['total'].numpy()[sel]),'identity_fraction':quant(1-g['dyn'].numpy()[sel]),'mean_normalized_actual_weight_to':{NAMES[k]:float(np.sum(w[sel]*(nlab[sel]==k))/sel.sum()) for k in [-1,0,1,2]},'mean_masked_neighbor_weight_after_correction':float(np.sum(w[sel]*(~g['mask'][sel].numpy()))/sel.sum())}
 # Directed base-level ARAP edges and the exact per-row normalized topology weight (self excluded only in display).
 edges=d['scf.topo_knn_ind'].numpy();mask=d['scf.topo_knn_mask'].numpy();ew=mask/np.maximum(mask.sum(-1,keepdims=True),1);src=np.repeat(np.arange(M),edges.shape[1]);dst=edges.ravel();valid=mask.ravel()&(src!=dst);ev=ew.ravel();known=(nl[src]>=0)&(nl[dst]>=0);cross=nl[src]!=nl[dst]
 arap={'directed_nonself_active_edges':int(valid.sum()),'known_endpoint_edges':int((valid&known).sum()),'cross_entity_known_edges':int((valid&known&cross).sum()),'cross_entity_known_weight_sum':float(ev[valid&known&cross].sum()),'known_endpoint_weight_sum':float(ev[valid&known].sum()),'pair_counts':{},'multilevel_edges':'NOT SAVED: multilevel_arap_edge_list, weights and cached _D_topo absent from checkpoint. Random topology resampling and stale cached distance prevent exact recovery from final positions. Config enabled two levels; no exact active cross-entity count claimed.'}
 for e1 in [-1,0,1,2]:
  for e2 in [-1,0,1,2]:
   m=valid&(nl[src]==e1)&(nl[dst]==e2)
   if m.any():arap['pair_counts'][f'{NAMES[e1]}->{NAMES[e2]}']={'edges':int(m.sum()),'effective_weight_sum':float(ev[m].sum())}
 for r,nodew in zip(rows,qnodes):
  r['conditional_dynamic_node_weight_by_rgb_diagnostic_label']={NAMES[k]:float(nodew[nl==k].sum()/nodew.sum()) for k in [-1,0,1,2]}
 # Temporal per-stage support, including node certainty; no one-to-one cross-stage identity assertion.
 stage={'raw_visible_track_stable_labels':counts(tracklabel),'depth_filtered_track_stable_labels':counts(tracklabel[keep]),'classified_dynamic_tracks':counts(tracklabel[keep][ident['dynamic_track_mask']]),'classified_static_tracks':counts(tracklabel[keep][ident['static_track_mask']]),'geometry_node_labels':counts(gl),'photometric_node_labels':counts(nl),'dynamic_gs_labels':counts(dl),'static_gs_labels':counts(gsl[:nstatic]),'stages_are_not_same_units':'Tracks, resampled nodes and densified GS are different units. Counts are allocation/support evidence, not a one-to-one survival probability.'}
 eventrows=[]
 for t in range(T):
  eventrows.append({'frame':t,'time_s':float(times[t]),'sam2_object_pixels':int((seg[t]==2).sum()),'visible_object_labeled_tracks':int((vis[t]&(tracklabel==2)).sum()),'depthvalid_object_labeled_tracks':int((validobs[t]&(tracklabel==2)).sum()),'certain_object_labeled_final_nodes':int((d['scf._node_certain'][t].numpy()&(nl==2)).sum()),'certain_object_labeled_geometry_nodes':int((geo['_node_certain'][t].numpy()&(gl==2)).sum())})
 np.savez_compressed(OUT/'diagnostic_labels_and_weights.npz',track_labels=tracklabel,track_votes=tv,raw_track_kept=keep,final_node_labels=nl,final_node_votes=nv,node_observation_label=nobs,geometry_node_labels=gl,gaussian_labels=gsl,gaussian_votes=votes,skinning_indices=skidx,skinning_weights=w,skinning_mask=g['mask'].numpy(),rbf_sum=g['total'].numpy(),dyn_o=g['dyn'].numpy(),gaussian_static_count=nstatic)
 with (OUT/'temporal_object_support.csv').open('w') as f:
  wr=csv.DictWriter(f,fieldnames=list(eventrows[0]));wr.writeheader();wr.writerows(eventrows)
 result={'status':'completed','role':'frozen CPU diagnostic; no training or evaluation geometry accessed','seconds':time.time()-start,'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [dpath,spath,BASE/'common_input/input_manifest.json',BASE/'common_input/uniform_cotracker_tap.npz',BASE/'segmentation/segmentation.npz',BASE/'mosca_cotracker/track_identification.npz',BASE/'mosca_cotracker/mosca/mosca.pth',MOS/'lib_mosca/mosca.py',MOS/'lib_mosca/dynamic_gs.py',MOS/'lib_mosca/scaffold_utils/dualquat_helper.py',Path(__file__)]},'validation':check,'label_protocol':{'track':'SAM2 2px eroded interiors, visible CoTracker only, >=3 votes and >=80% agreement','node':'certain prior; projection in same eroded class; input depth tolerance max(5cm,3% depth); <=3px to visible track of stable same class; >=3 votes and >=80% agreement','gaussian':'Independent mask projection in six stored keyframes [0,16,20,24,28,113], depth tolerance max(5cm,3%); opacity>.01; >=2 votes and >=80% agreement; no node-inherited label','limitations':'All estimated priors; systematic mask/depth/track errors may produce consistent wrong labels. Unknown is not background. Six-frame GS labels have selective coverage. Neither node proximity nor alpha weights establish material-point identity.'},'global_gate':{'dynamic_gs_count':ndyn,'static_gs_count':nstatic,'RBFsum':quant(g['total'].numpy()),'dyn_o':quant(g['dyn'].numpy()),'masked_slots_with_actual_nonzero_weight':int(((~g['mask'])&(g['w']>0)).sum()),'masked_slots_actual_weight_sum':float((g['w']*(~g['mask'])).sum())},'stage_support':stage,'skinning_by_independent_gs_label':skstats,'ARAP_saved_base_edges':arap,'query_results':rows,'object_mask_explanation_keyframes':explanation,'counterfactual':'dyn_o=False is frozen .999 gate, not re-training; source association changes under the off model because ref_time differs per Gaussian. Each trajectory is differenced from its own first frame when reporting motion; this is not reference alignment.'}
 dump(OUT/'motion_binding_audit.json',result)
 print('COMPLETE',json.dumps({'seconds':result['seconds'],'stage':stage,'arap':arap,'query_labels':[{r['query_id']:r['conditional_dynamic_node_weight_by_rgb_diagnostic_label']} for r in rows],'object':explanation},ensure_ascii=False),flush=True)
if __name__=='__main__':
 with torch.no_grad():main()
