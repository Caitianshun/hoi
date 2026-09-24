#!/usr/bin/env python3
"""Independent CPU diagnosis of absolute/relative metric divergence. No training.
Input-derived labels are computed independently of reference; references only score frozen predictions.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,hashlib
from pathlib import Path
import numpy as np,torch,cv2
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sparse_raster_exact import sparse_raster
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
RUN=ROOT/'experiments/mosca_validation_20260923';BASE=ROOT/'experiments/mosca_baseline_20260922';A=RUN/'a_normalized_exact'
sys.path.insert(0,str(RUN/'motion_binding'))
from audit_motion_binding import get_weights,project,sample,majority,counts

def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 start=time.time();torch.set_num_threads(4)
 # Model-only source attribution first; no independent reference coordinate enters this calculation.
 m=json.loads((BASE/'common_input/input_manifest.json').read_text());K=np.array(m['K']);C=np.array(m['c2w']);q=json.loads((BASE/'common_input/queries_first_frame.json').read_text())
 ex=json.loads((A/'diagnostics/export_manifest.json').read_text());scale=ex['world_scale'];ns=ex['static_gaussians'];f0=np.load(A/'diagnostics/keyframes/00000.npz');predfull=np.load(A/'diagnostics/query_trajectories.npz');sem=np.load(A/'diagnostics/posthoc_semantic_labels.npz')['labels']
 contrib=sparse_raster(f0['xyz_world'],f0['rotation_frame'],f0['scales'],f0['opacity'],K,C,np.array(q['points'][:6]));d=torch.load(A/'model/photometric_d_model_native_add3.pth',map_location='cpu',weights_only=False);g=get_weights(d)
 static=[];dynfrac=[];attrib=[];predzero=[]
 for j,(idx,w) in enumerate(contrib):
  isd=idx>=ns;w0=w[~isd];wd=w[isd];did=idx[isd]-ns;alpha=w.sum();frac=wd.sum()/alpha;S=np.sum(f0['xyz_world'][idx[~isd]]*w0[:,None],axis=0)/w0.sum();static.append(S);dynfrac.append(frac);predzero.append(np.sum(f0['xyz_world'][idx]*w[:,None],axis=0)/alpha)
  attr={'query_id':q['point_ids'][j],'static_fraction':float(w0.sum()/alpha),'dynamic_fraction':float(frac),'conditional_dynamic_gate':float(np.sum(g['dyn'][did].numpy()*wd)/wd.sum()),'static_camera_z_m':float(((S-C[:3,3])@C[:3,:3])[2]),'dynamic_camera_z_m':float((((np.sum(f0['xyz_world'][idx[isd]]*wd[:,None],axis=0)/wd.sum())-C[:3,3])@C[:3,:3])[2]),'static_semantic_alpha_fraction_of_total':{str(l):float(w0[sem[idx[~isd]]==l].sum()/alpha) for l in [-1,0,1,2]},'dynamic_semantic_alpha_fraction_of_total':{str(l):float(wd[sem[idx[isd]]==l].sum()/alpha) for l in [-1,0,1,2]}}
  attrib.append(attr)
 static=np.array(static);dynfrac=np.array(dynfrac);D=(predfull['predicted']-(1-dynfrac[None,:,None])*static[None])/dynfrac[None,:,None]
 checks={'source_alpha_max_difference':float(max(abs(w.sum()-predfull['source_alpha'][i]) for i,(_,w) in enumerate(contrib))),'source_dynamic_fraction_max_difference':float(np.max(abs(dynfrac-predfull['source_dynamic_fraction']))),'source_xyz_max_difference_m':float(np.max(abs(np.array(predzero)-predfull['predicted'][0])))}
 assert max(checks.values())<2e-4,checks
 np.savez_compressed(OUT/'source_components_prediction_only.npz',static_centroid=static,dynamic_conditional_centroid=D,dynamic_fraction=dynfrac,frame_times=predfull['frame_times'],query_id=predfull['query_id'],coordinate_frame=np.array('behave_world_k1_color'),units=np.array('m'))
 # Same RGB-prior label protocol as original node audit. Node world positions divide by A input scale.
 tr=np.load(BASE/'common_input/uniform_cotracker_tap.npz');uv=tr['tracks'];vis=tr['visibility'];tl=np.load(RUN/'motion_binding/diagnostic_labels_and_weights.npz')['track_labels'];seg=np.load(BASE/'segmentation/segmentation.npz')['entity_labels'];T,H,W=seg.shape
 nd=g['node'].numpy()/scale;certain=d['scf._node_certain'].numpy();nv=np.zeros((nd.shape[1],3),np.int32);dep_cache=[];nodeobs=np.full(certain.shape,-1,np.int8)
 for t in range(T):
  interior=np.full((H,W),255,np.uint8)
  for ent in [0,1,2]:interior[cv2.erode((seg[t]==ent).astype(np.uint8),np.ones((5,5),np.uint8))>0]=ent
  dep=np.load(BASE/f'common_input/unidepth_depth/{t:05d}.npz')['dep'];p,z=project(nd[t],K,C);lab,inside=sample(interior,p);dz,_=sample(dep,p)
  ok=certain[t]&inside&(lab<3)&(z>0)&(np.abs(z-dz)<=np.maximum(.05,.03*dz));near=np.zeros(len(p),bool)
  for ent in [0,1,2]:
   ii=np.flatnonzero(ok&(lab==ent));tt=np.flatnonzero(vis[t]&(tl==ent))
   if len(ii) and len(tt):near[ii]=cKDTree(uv[t,tt]).query(p[ii],k=1)[0]<=3
  ok&=near;nodeobs[t,ok]=lab[ok]
  for ent in [0,1,2]:nv[:,ent]+=ok&(lab==ent)
 nl=majority(nv);np.savez_compressed(OUT/'a_node_diagnostic_labels.npz',labels=nl,votes=nv,observation_labels=nodeobs)
 for r,(idx,w) in zip(attrib,contrib):
  mask=idx>=ns;did=idx[mask]-ns;dw=w[mask];iw=g['w'][did].numpy()*dw[:,None];labels=nl[g['idx'][did].numpy()]
  r['dynamic_node_weight_by_rgb_label']={str(l):float(iw[labels==l].sum()/dw.sum()) for l in [-1,0,1,2]}
 scale_effect={k:{'stored_value':float(d[k]),'old_effective_m':float(d[k]),'A_effective_m':float(d[k])/scale} for k in ['scf.spatial_unit','scf.max_sigma','scf.init_sigma','max_scale']}
 nodeinfo={'A_final_total':len(nl),'A_diagnostic_labels':counts(nl),'A_native_grouping_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(d['scf._node_grouping'].numpy(),return_counts=True))},'A_node_grouping_is_not_posthoc_labels':True,'source_world_scale':scale,'effective_world_lengths':scale_effect,'node_labels_no_reference_used':True}
 # Only now score already frozen predictions and decompose errors, using exact same reference/time/query arrays.
 paths={'old':BASE/'evaluation/mosca_cotracker_final/evaluation_bundle.npz','A':A/'evaluation/evaluation_bundle.npz'};data={k:dict(np.load(p)) for k,p in paths.items()}
 for k in ['reference','frame_times','query_id','entity','valid_mask','predicted_valid_mask','coordinate_frame','units','visibility']:assert np.array_equal(data['old'][k],data['A'][k]),k
 stats={};queryrows=[];timerows=[];pairrows=[];componentrows=[];ref=data['A']['reference'];times=data['A']['frame_times'];queryids=data['A']['query_id'];obsindices=[int(np.flatnonzero(np.isclose(predfull['frame_times'],t,atol=1e-7,rtol=0))[0]) for t in times]
 for name,z in data.items():
  e=z['predicted']-z['reference'];eh=e[:,4:];eo=e[:,:4];pair=eh[:,:,None]-eo[:,None,:];pn=np.linalg.norm(pair,axis=-1);dot=np.einsum('tid,tjd->tij',eh,eo);cos=dot/(np.linalg.norm(eh,axis=-1)[:,:,None]*np.linalg.norm(eo,axis=-1)[:,None,:]);hn2=np.mean(np.sum(eh**2,-1));on2=np.mean(np.sum(eo**2,-1));pair2=np.mean(pn**2)
  assert abs(pair2-(hn2+on2-2*dot.mean()))<1e-12
  stats[name]={'mean_absolute_epe_m':float(np.linalg.norm(e,axis=-1).mean()),'entity_balanced_absolute_epe_m':float((np.linalg.norm(eh,axis=-1).mean()+np.linalg.norm(eo,axis=-1).mean())/2),'hand_epe_m':float(np.linalg.norm(eh,axis=-1).mean()),'object_epe_m':float(np.linalg.norm(eo,axis=-1).mean()),'mean_pair_error_m':float(pn.mean()),'first_frame_pair_error_m':float(pn[0].mean()),'mean_relative_displacement_error_excluding_first_m':float(np.linalg.norm(pair[1:]-pair[:1],axis=-1).mean()),'pair_MSE_identity':{'mean_hand_squared_error_m2':float(hn2),'mean_object_squared_error_m2':float(on2),'mean_cross_dot_m2':float(dot.mean()),'twice_cross_dot_subtracted_m2':float(2*dot.mean()),'mean_pair_squared_error_m2':float(pair2)},'mean_pair_error_cosine':float(cos.mean()),'first_error_camera_xyz_m':(e[0]@C[:3,:3]).tolist(),'mean_error_camera_xyz_m':(e.mean(0)@C[:3,:3]).tolist()}
  for j,id in enumerate(queryids):
   queryrows.append({'branch':name,'query_id':id,'entity':z['entity'][j],'mean_absolute_epe_m':float(np.linalg.norm(e[:,j],axis=-1).mean()),'mean_displacement_epe_m':float(np.linalg.norm(e[1:,j]-e[:1,j],axis=-1).mean()),'first_epe_m':float(np.linalg.norm(e[0,j])),'initial_camera_x_error_m':float((e[0,j]@C[:3,:3])[0]),'initial_camera_y_error_m':float((e[0,j]@C[:3,:3])[1]),'initial_camera_z_error_m':float((e[0,j]@C[:3,:3])[2])})
  for t,time_s in enumerate(times):
   ehmean=eh[t].mean(0);eomean=eo[t].mean(0)
   timerows.append({'branch':name,'time_s':float(time_s),'mean_hand_epe_m':float(np.linalg.norm(eh[t],axis=-1).mean()),'mean_object_epe_m':float(np.linalg.norm(eo[t],axis=-1).mean()),'mean_pair_epe_m':float(pn[t].mean()),'mean_pair_error_cosine':float(cos[t].mean()),'centroid_relative_error_m':float(np.linalg.norm(ehmean-eomean)),'hand_mean_error_camera_y_m':float((ehmean@C[:3,:3])[1]),'object_mean_error_camera_y_m':float((eomean@C[:3,:3])[1]),'hand_mean_error_camera_z_m':float((ehmean@C[:3,:3])[2]),'object_mean_error_camera_z_m':float((eomean@C[:3,:3])[2])})
   for hi in range(2):
    for oi in range(4):pairrows.append({'branch':name,'time_s':float(time_s),'hand_query':queryids[4+hi],'object_query':queryids[oi],'relative_error_m':float(pn[t,hi,oi]),'relative_displacement_error_m':float(np.linalg.norm(pair[t,hi,oi]-pair[0,hi,oi])) if t>0 else None,'error_vector_dot_m2':float(dot[t,hi,oi]),'error_vector_cosine':float(cos[t,hi,oi])})
 E_dyn=D[obsindices]-ref;E_s=static[None]-ref;E_A=data['A']['predicted']-ref
 for j,id in enumerate(queryids):
  componentrows.append({'query_id':id,'dynamic_fraction':float(dynfrac[j]),'static_fraction':float(1-dynfrac[j]),'observed_composite_epe_m':float(np.linalg.norm(E_A[:,j],axis=-1).mean()),'conditional_dynamic_only_epe_m':float(np.linalg.norm(E_dyn[:,j],axis=-1).mean()),'conditional_static_only_epe_m':float(np.linalg.norm(E_s[:,j],axis=-1).mean()),'conditional_dynamic_displacement_epe_m':float(np.linalg.norm(E_dyn[1:,j]-E_dyn[:1,j],axis=-1).mean()),'conditional_dynamic_initial_epe_m':float(np.linalg.norm(E_dyn[0,j])),'static_initial_epe_m':float(np.linalg.norm(E_s[0,j])),'reference_initial_camera_z_m':float(((ref[0,j]-C[:3,3])@C[:3,:3])[2])})
 writecsv(OUT/'per_query_errors.csv',queryrows);writecsv(OUT/'per_observation_error_vectors.csv',timerows);writecsv(OUT/'all_fixed_pair_errors.csv',pairrows);writecsv(OUT/'conditional_source_component_errors.csv',componentrows)
 result={'status':'completed','wall_seconds':time.time()-start,'same_reference_and_protocol_arrays':True,'coverage':{'absolute':'84/84','relative_pairs':'112/112'},'source_cpu_validation':checks,'source_attribution':attrib,'node_and_scale_diagnostic':nodeinfo,'error_decomposition':stats,'source_component_diagnostic':componentrows,'component_caveat':'Conditional dynamic/static centres are diagnostic algebraic components of the same source weights, not a retrained no-static branch and not material points. Reference does not set these weights or their decomposition.','hashes':{str(p.relative_to(ROOT)):sha(p) for p in [*paths.values(),A/'diagnostics/query_trajectories.npz',A/'diagnostics/export_manifest.json',A/'model/photometric_d_model_native_add3.pth',Path(__file__),OUT/'sparse_raster_exact.py']}}
 dump(OUT/'analysis_a.json',result)
 fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
 for name,col in [('old','tab:blue'),('A','tab:orange')]:
  rr=[r for r in timerows if r['branch']==name]
  for ax,key,title in [(axs[0,0],'mean_object_epe_m','Object absolute error'),(axs[0,1],'mean_hand_epe_m','Hand absolute error'),(axs[1,0],'mean_pair_epe_m','Hand-object relative error'),(axs[1,1],'mean_pair_error_cosine','Alignment of hand/object error vectors')]:
   ax.plot(times,[r[key] for r in rr],'-o',markersize=3,label=name,color=col);ax.set_title(title);ax.set_ylabel('Cosine' if key.endswith('cosine') else 'Error (m)');ax.grid(alpha=.2)
 for ax in axs.flat:ax.legend();ax.axvspan(20.63,21.063,color='gray',alpha=.1)
 axs[1,0].set_xlabel('Matched observation time (s)');axs[1,1].set_xlabel('Matched observation time (s)');fig.suptitle('Same 14 observations: object improvement removes cancellation with hand error');fig.tight_layout();fig.savefig(OUT/'absolute_relative_decomposition.png',dpi=180);plt.close(fig)
 print(json.dumps({'checks':checks,'stats':stats,'components':componentrows,'nodes':nodeinfo},ensure_ascii=False))
if __name__=='__main__':
 with torch.no_grad():main()
