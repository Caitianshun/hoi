#!/usr/bin/env python3
"""Join exact sampler provenance and inspect first-frame raster threshold only. CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np,torch
from pytorch3d.transforms import quaternion_to_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;V=ROOT/'experiments/mosca_validation_20260923';BASE=ROOT/'experiments/mosca_baseline_20260922';A=V/'a_normalized_exact';SEED=OUT.parent/'sampling/run01/actual_dynamic_seeds.npz'
sys.path.insert(0,str(V/'motion_binding'));from audit_motion_binding import get_weights,warp,project

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(path,rows):
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def kernel(xyz,R,scales,opacity,K,C,query):
 # Exact-camera projected covariance and pre-threshold splat alpha. No depth sorting yet.
 cam=(xyz-C[:3,3])@C[:3,:3];z=cam[:,2];fx,fy=K[0,0],K[1,1];zz=np.maximum(z,1e-8);mean=cam[:,:2]/(zz[:,None]+1e-7)*np.array([fx,fy])+K[:2,2]
 rc=C[:3,:3].T[None]@R;cov=(rc*scales[:,None,:]**2)@rc.transpose(0,2,1);cl=cam[:,:2]/zz[:,None];cl=np.clip(cl,-1.3*np.array([640/(2*fx),480/(2*fy)]),1.3*np.array([640/(2*fx),480/(2*fy)]));J=np.zeros((len(z),2,3));J[:,0,0]=fx/zz;J[:,1,1]=fy/zz;J[:,0,2]=-fx*cl[:,0]/zz;J[:,1,2]=-fy*cl[:,1]/zz;v=J@cov@J.transpose(0,2,1);v[:,0,0]+=.3;v[:,1,1]+=.3;delta=mean-query;power=-.5*np.einsum('ni,nij,nj->n',delta,np.linalg.inv(v),delta);alpha=np.minimum(.99,opacity.ravel()*np.exp(np.minimum(power,0)))
 # native tile bounds; this prevents attributing splats never traversed by the pixel.
 aa,bb,cc=v[:,0,0],v[:,0,1],v[:,1,1];det=aa*cc-bb**2;mid=(aa+cc)*.5;rad=np.ceil(3*np.sqrt(np.maximum(mid+np.sqrt(np.maximum(.1,mid**2-det)),0)));grid=np.array([40,30]);rmin=np.clip(np.trunc((mean-rad[:,None])/16),0,grid).astype(int);rmax=np.clip(np.trunc((mean+rad[:,None]+15)/16),0,grid).astype(int);tile=np.floor(query/16).astype(int);eligible=(z>.02)&(det>0)&((rmin<=tile)&(rmax>tile)).all(-1)&(power<=0)
 return {'mean':mean,'z':z,'cov2d':v,'power':power,'alpha_before_threshold':alpha,'tile_eligible':eligible}
def main():
 torch.set_num_threads(4);m=json.loads((BASE/'common_input/input_manifest.json').read_text());K=np.array(m['K']);C=np.array(m['c2w']);queries=json.loads((BASE/'common_input/queries_first_frame.json').read_text());qp=np.array(queries['points'][:6]);seg=np.load(BASE/'segmentation/segmentation.npz')['entity_labels'];dep=np.load(BASE/'common_input/unidepth_depth/00000.npz')['dep'];scale=json.loads((A/'model/run.json').read_text())['world_scale'];seed=np.load(SEED);paths=np.load(OUT/'initial_dynamic_source_paths.npz');rt=np.linalg.norm(seed['raw_world_xyz_m']-paths['reference_world_m'],axis=-1);assert rt.max()<2e-6;assert np.array_equal(seed['raw_source_frame'],paths['ref_time'])
 rows=list(csv.DictReader(open(OUT/'gaussian_contribution_paths.csv')));joined=[]
 for r in rows:
  if r['stage']!='initial':continue
  i=int(r['dynamic_gs_id']);t=int(seed['raw_source_frame'][i]);u,v=seed['raw_pixel_xy'][i];r.update(actual_sample_frame=t,actual_sample_u=float(u),actual_sample_v=float(v),actual_sample_depth_m=float(seed['raw_depth_m'][i]),actual_sample_sam2_label=int(seg[t,int(v),int(u)]),actual_randperm_position=int(seed['original_randperm_position'][i]),local_storage_roundtrip_error_m=float(rt[i]));joined.append(r)
 writecsv(OUT/'initial_actual_sample_paths.csv',joined)
 seedgroups={}
 for query in queries['point_ids'][:6]:
  rr=[r for r in joined if r['query_id']==query];seedgroups[query]={str(l):sum(float(r['conditional_dynamic_weight']) for r in rr if r['actual_sample_sam2_label']==l) for l in [0,1,2]}
 selected=[];near=[];allplot=[]
 for stage,dp,sp in [('initial',A/'model/initial_dynamic.pth',A/'model/initial_static.pth'),('final',A/'model/photometric_d_model_native_add3.pth',A/'model/photometric_s_model_native_add3.pth')]:
  d=torch.load(dp,map_location='cpu',weights_only=False);s=torch.load(sp,map_location='cpu',weights_only=False);g=get_weights(d);dx,dr=warp(d,g,0);xyz=np.r_[s['_xyz'].numpy()/scale,dx.numpy()/scale];R=np.concatenate([quaternion_to_matrix(torch.nn.functional.normalize(s['_rotation'],dim=-1)).numpy(),dr.numpy()]);scales=np.concatenate([(ss['min_scale']+torch.sigmoid(ss['_scaling'])*(ss['max_scale']-ss['min_scale'])).numpy()/scale for ss in [s,d]]);opacity=np.concatenate([torch.sigmoid(ss['_opacity']).numpy() for ss in [s,d]]);ns=len(s['_xyz'])
  for qi in [4,5]:
   q=qp[qi];targetdepth=float(dep[q[1],q[0]]);ker=kernel(xyz,R,scales,opacity,K,C,q);z=ker['z'];rawa=ker['alpha_before_threshold'];eligible=ker['tile_eligible'];nearz=abs(z-targetdepth)<=.1
   ii=np.flatnonzero(nearz&eligible);order=ii[np.argsort(-rawa[ii])];best=[]
   for i in order[:5]:best.append({'component':'dynamic' if i>=ns else 'static','component_id':int(i-ns if i>=ns else i),'depth_m':float(z[i]),'projected_u':float(ker['mean'][i,0]),'projected_v':float(ker['mean'][i,1]),'opacity':float(opacity[i,0]),'prethreshold_alpha':float(rawa[i]),'passes_1_over_255':bool(rawa[i]>=1/255)})
   near.append({'stage':stage,'query_id':queries['point_ids'][qi],'input_depth_m':targetdepth,'near_depth_window_m':.1,'near_tile_candidates':len(ii),'near_candidates_passing_alpha_threshold':int((rawa[ii]>=1/255).sum()),'max_near_prethreshold_alpha':float(rawa[ii].max()) if len(ii) else None,'best_near_candidates':best})
   if stage=='initial':
    gi=86 if qi==4 else 138;i=ns+gi;rawpix=seed['raw_pixel_xy'][gi];offset=rawpix-q;ideal_alpha=float(opacity[i,0]*np.exp(-.5*offset@np.linalg.inv(ker['cov2d'][i])@offset));ownerr=float(np.linalg.norm(dx[gi].numpy()/scale-seed['raw_world_xyz_m'][gi]));selected.append({'query_id':queries['point_ids'][qi],'initial_dynamic_gs_id':gi,'raw_source_frame':int(seed['raw_source_frame'][gi]),'raw_pixel_xy':rawpix.tolist(),'raw_depth_m':float(seed['raw_depth_m'][gi]),'actual_projected_uv':ker['mean'][i].tolist(),'projected_minus_actual_sample_uv':(ker['mean'][i]-rawpix).tolist(),'query_uv':q.tolist(),'projection_minus_query':(ker['mean'][i]-q).tolist(),'opacity':float(opacity[i,0]),'covariance_2d_px2':ker['cov2d'][i].tolist(),'exponent':float(ker['power'][i]),'prethreshold_alpha':float(rawa[i]),'native_alpha_cutoff':1/255,'passes_cutoff':bool(rawa[i]>=1/255),'tile_eligible':bool(eligible[i]),'minimum_opacity_to_pass_current_geometry':float((1/255)/np.exp(ker['power'][i])),'pixel_mean_replaced_by_exact_sample_only_alpha':ideal_alpha,'pixel_mean_counterfactual_note':'Only isolated kernel mean recentered to raw sampling pixel, covariance/opacities fixed; not an actual model change, training or whole render.','warp_to_own_reference_frame_error_m':ownerr})
   # Factual actual compositor weights, sorting and cutoff identical to source raster.
   ix=np.flatnonzero(eligible&(rawa>=1/255));ix=ix[np.argsort(z[ix],kind='stable')];alph=rawa[ix];Tafter=np.cumprod(1-alph);Tbefore=np.r_[1,Tafter[:-1]];keep=Tafter>=.0001;ix=ix[keep];weights=(alph*Tbefore)[keep];allplot.append((stage,qi,z[ix],weights,ix>=ns))
 report={'status':'completed','sampling_sha256':sha(SEED),'all30000_local_storage_roundtrip_max_m':float(rt.max()),'all30000_frame_ids_match':True,'actual_sample_mask_label_conditional_dynamic_weights':seedgroups,'nearby_seed_checks':selected,'near_depth_candidate_checks':near,'interpretation':'Near is a ±10cm input-depth diagnostic window, not independent surface identity. Per-splat alpha threshold and masked tile traversal are counted before occlusion. Alpha from all planes still does not establish material identity. Final IDs have no saved lineage to initial IDs.','script_sha256':sha(__file__)};(OUT/'seed_threshold_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True)
 bins=np.arange(1.6,3.61,.1)
 for stage,qi,z,w,dyn in allplot:
  ax=axs[0 if stage=='initial' else 1,qi-4];total=w.sum();q=qp[qi];d0=float(dep[q[1],q[0]]);edges=np.sort(np.unique(np.r_[bins,d0-.1,d0+.1]));ax.hist([z[~dyn],z[dyn]],bins=edges,weights=[w[~dyn]/total,w[dyn]/total],stacked=True,label=['Static','Dynamic'],color=['#999999','#3287b8']);ax.axvspan(d0-.1,d0+.1,alpha=.15,color='green');ax.axvline(d0,c='green',ls='--');ax.set_title(f'{stage}: {queries["point_ids"][qi]}\nalpha={total:.4f}; transmission={1-total:.4f}');ax.set_ylabel('Fraction of covered alpha');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
 for ax in axs[1]:ax.set_xlabel('Camera0 depth (m)')
 fig.suptitle('First-frame depth contributions; green band is input depth ±0.1m, not ground truth');fig.tight_layout();fig.savefig(OUT/'first_frame_depth_contributions.png',dpi=180);plt.close(fig)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':
 with torch.no_grad():main()
