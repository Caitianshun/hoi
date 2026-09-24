#!/usr/bin/env python3
"""Read-only CPU first-frame binding trace, stage-local GS IDs; no reference geometry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,hashlib,time
from pathlib import Path
import numpy as np,torch
from pytorch3d.transforms import quaternion_to_matrix
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
V=ROOT/'experiments/mosca_validation_20260923';A=V/'a_normalized_exact';BASE=ROOT/'experiments/mosca_baseline_20260922'
sys.path.insert(0,str(V/'analysis_a'));from sparse_raster_exact import sparse_raster
sys.path.insert(0,str(V/'motion_binding'));from audit_motion_binding import get_weights,warp,project,sample

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def v3(row,prefix,x):
 for k,v in zip('xyz',x):row[f'{prefix}_{k}']=float(v)
def summaries(w,x):return float(np.dot(w,x)/w.sum()) if w.sum()>0 else None

def main():
 start=time.time();torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
 m=json.loads((BASE/'common_input/input_manifest.json').read_text());q=json.loads((BASE/'common_input/queries_first_frame.json').read_text());K=np.array(m['K']);C=np.array(m['c2w']);times=np.array(m['timestamp_seconds']);scale=json.loads((A/'model/run.json').read_text())['world_scale'];qs=np.array(q['points'][:6]);seg=np.load(BASE/'segmentation/segmentation.npz')['entity_labels'];dep0=np.load(BASE/'common_input/unidepth_depth/00000.npz')['dep'];depths={0:dep0}
 summary=[];gsrows=[];noderows=[];manifest={};sourcearrays={}
 for stage,dp,sp in [('initial',A/'model/initial_dynamic.pth',A/'model/initial_static.pth'),('final',A/'model/photometric_d_model_native_add3.pth',A/'model/photometric_s_model_native_add3.pth')]:
  d=torch.load(dp,map_location='cpu',weights_only=False);s=torch.load(sp,map_location='cpu',weights_only=False);g=get_weights(d);dx,dr=warp(d,g,0,dyn_off=not bool(d['dyn_o_flag']));pureg={**g,'dyn':torch.ones_like(g['dyn'])};xpure,_=warp(d,pureg,0)
  ti=g['ti'].numpy();att=d['attach_ind'].numpy();node=g['node'].numpy()/scale;nr=g['nodeR'].numpy();src=g['src'].numpy()/scale;dst=dx.numpy()/scale;pure=xpure.numpy()/scale
  rdelta=nr[0,att]@nr[ti,att].transpose(0,2,1);attached=np.einsum('nij,nj->ni',rdelta,src-node[ti,att])+node[0,att]
  angle=np.degrees(np.arccos(np.clip((np.trace(rdelta,axis1=1,axis2=2)-1)/2,-1,1)))
  sx=s['_xyz'].numpy()/scale;sr=quaternion_to_matrix(torch.nn.functional.normalize(s['_rotation'],dim=-1)).numpy();ss=(s['min_scale']+torch.sigmoid(s['_scaling'])*(s['max_scale']-s['min_scale'])).numpy()/scale;ds=(d['min_scale']+torch.sigmoid(d['_scaling'])*(d['max_scale']-d['min_scale'])).numpy()/scale
  xyz=np.r_[sx,dst];rot=np.concatenate([sr,dr.numpy()]);scales=np.r_[ss,ds];op=np.r_[torch.sigmoid(s['_opacity']).numpy(),torch.sigmoid(d['_opacity']).numpy()];ns=len(sx);contrib=sparse_raster(xyz,rot,scales,op,K,C,qs)
  psrc,zsrc=project(src,K,C);pdst,zdst=project(dst,K,C);_,zpure=project(pure,K,C);_,zattach=project(attached,K,C);sigma=(torch.sigmoid(d['scf._node_sigma_logit'])*d['scf.max_sigma']).numpy().ravel()/scale
  saved=np.load(A/'diagnostics/query_trajectories.npz') if stage=='final' else None
  for j,(ix,w) in enumerate(contrib):
   dyn=ix>=ns;di=ix[dyn]-ns;dw=w[dyn];total=w.sum();cond=dw/dw.sum();qdepth=float(dep0[qs[j,1],qs[j,0]]);p=(xyz[ix]*w[:,None]).sum(0)/total
   if saved is not None:assert np.max(abs(p-saved['predicted'][0,j]))<2e-4
   _,zc=project(xyz[ix],K,C);depthbins={'closer_than_input_minus_10cm':zc<qdepth-.1,'within_input_plusminus_10cm':abs(zc-qdepth)<=.1,'10_to_25cm_behind_input':(zc>qdepth+.1)&(zc<=qdepth+.25),'over_25cm_behind_input':zc>qdepth+.25}
   row={'stage':stage,'query_id':q['point_ids'][j],'query_u':int(qs[j,0]),'query_v':int(qs[j,1]),'input_depth_m':qdepth,'alpha':float(total),'uncovered_transmittance':float(1-total),'dynamic_fraction':float(dw.sum()/total),'static_fraction':float(w[~dyn].sum()/total),'contributing_dynamic_gs':len(di),'contributing_static_gs':int((~dyn).sum()),'dynamic_unique_reference_frames':int(len(np.unique(ti[di]))),'dynamic_ref0_conditional_weight':float(cond[ti[di]==0].sum()),'dynamic_source_depth_m':summaries(cond,zsrc[di]),'dynamic_attach_only_depth_at0_m':summaries(cond,zattach[di]),'dynamic_pure_DQB_depth_at0_m':summaries(cond,zpure[di]),'dynamic_actual_depth_at0_m':summaries(cond,zdst[di]),'dynamic_depth_change_due_to_pure_DQB_m':summaries(cond,zpure[di]-zsrc[di]),'dynamic_depth_change_due_to_gate_m':summaries(cond,zdst[di]-zpure[di]),'dynamic_gate':summaries(cond,g['dyn'][di].numpy()),'covered_alpha_depth_bins':{name:float(w[mk].sum()/total) for name,mk in depthbins.items()},'dynamic_conditional_farther25cm_fraction':float(cond[zdst[di]>qdepth+.25].sum())}
   paths=[]
   for rank,k in enumerate(np.argsort(-dw)):
    gi=int(di[k]);absolute=float(dw[k]);tt=int(ti[gi]);aa=int(att[gi]);rowg={'stage':stage,'query_id':q['point_ids'][j],'dynamic_gs_id':gi,'source_alpha_weight':absolute,'fraction_total_covered_alpha':float(absolute/total),'conditional_dynamic_weight':float(cond[k]),'rank_by_dynamic_contribution':rank+1,'ref_frame':tt,'ref_actual_time_s':float(times[tt]),'attach_node_id':aa,'ref_is_not_training_birth_step':True,'src_projected_u':float(psrc[gi,0]),'src_projected_v':float(psrc[gi,1]),'dst_projected_u':float(pdst[gi,0]),'dst_projected_v':float(pdst[gi,1]),'src_camera_z_m':float(zsrc[gi]),'attach_only_camera_z_at0_m':float(zattach[gi]),'pure_dqb_camera_z_at0_m':float(zpure[gi]),'actual_camera_z_at0_m':float(zdst[gi]),'actual_depth_minus_input_query_m':float(zdst[gi]-qdepth),'raw_rbf_sum':float(g['base'][gi].sum()),'corrected_rbf_sum':float(g['total'][gi]),'motion_gate':float(g['dyn'][gi]),'gaussian_opacity':float(op[ns+gi,0]),'attach_sigma_m':float(sigma[aa]),'attach_distance_at_ref_m':float(np.linalg.norm(src[gi]-node[tt,aa])),'attach_node_rotation_ref_to0_deg':float(angle[gi]),'node_certain_at_ref':bool(d['scf._node_certain'][tt,aa]),'node_certain_at0':bool(d['scf._node_certain'][0,aa])}
    for prefix,vec in [('local_raw_normalized',d['_xyz'][gi].numpy()),('local_metric',d['_xyz'][gi].numpy()/scale),('reference_world_m',src[gi]),('first_frame_world_m',dst[gi]),('attach_node_ref_world_m',node[tt,aa]),('attach_node_first_world_m',node[0,aa]),('attach_node_displacement_ref_to0_m',node[0,aa]-node[tt,aa]),('actual_GS_displacement_ref_to0_m',dst[gi]-src[gi])]:v3(rowg,prefix,vec)
    if tt not in depths:depths[tt]=np.load(BASE/f'common_input/unidepth_depth/{tt:05d}.npz')['dep']
    lab,inb=sample(seg[tt],psrc[gi:gi+1]);dd,_=sample(depths[tt],psrc[gi:gi+1]);rowg.update(reference_projection_sam2_label=int(lab[0]) if inb[0] else -1,reference_projection_input_depth_m=float(dd[0]) if inb[0] else None,reference_projection_is_recovered_not_exact_sample_pixel=True)
    actualw=g['w'][gi].numpy();inds=g['idx'][gi].numpy();top=np.argsort(-actualw);rowg['top_weight_node_id']=int(inds[top[0]]);rowg['top_weight']=float(actualw[top[0]])
    gsrows.append(rowg)
    for slot in np.flatnonzero(actualw>1e-8):
     ni=int(inds[slot]);rn={'stage':stage,'query_id':q['point_ids'][j],'dynamic_gs_id':gi,'neighbor_slot':int(slot),'node_id':ni,'original_neighbor_mask':bool(g['mask'][gi,slot]),'actual_normalized_skinning_weight':float(actualw[slot]),'raw_rbf_weight':float(g['base'][gi,slot]),'learned_correction':float(d['_skinning_weight'][gi,slot]),'reference_frame':tt,'node_certain_ref':bool(d['scf._node_certain'][tt,ni]),'node_certain0':bool(d['scf._node_certain'][0,ni]),'node_sigma_m':float(sigma[ni])};v3(rn,'node_world_ref_m',node[tt,ni]);v3(rn,'node_world_0_m',node[0,ni]);v3(rn,'node_displacement_ref_to0_m',node[0,ni]-node[tt,ni]);noderows.append(rn)
    if rank<10:paths.append({key:rowg[key] for key in ['dynamic_gs_id','conditional_dynamic_weight','fraction_total_covered_alpha','ref_frame','ref_actual_time_s','attach_node_id','top_weight_node_id','top_weight','src_camera_z_m','attach_only_camera_z_at0_m','pure_dqb_camera_z_at0_m','actual_camera_z_at0_m','motion_gate','reference_projection_sam2_label','node_certain_at_ref','node_certain_at0']})
   row['top_dynamic_contribution_paths']=paths;summary.append(row)
  sourcearrays[stage]=dict(reference_world_m=src,at0_world_m=dst,attached_only_at0_world_m=attached,pure_dqb_at0_world_m=pure,ref_time=ti,attach_ind=att,skinning_weights=g['w'].numpy(),skinning_indices=g['idx'].numpy(),gate=g['dyn'].numpy())
  np.savez_compressed(OUT/f'{stage}_dynamic_source_paths.npz',**sourcearrays[stage]);manifest[stage]={'dynamic_sha256':sha(dp),'static_sha256':sha(sp),'dynamic_count':len(dx),'static_count':ns,'world_scale':scale,'w_correction':bool(d['w_correction_flag']),'dyn_o':bool(d['dyn_o_flag'])}
 csvwrite(OUT/'gaussian_contribution_paths.csv',gsrows);csvwrite(OUT/'actual_node_support_paths.csv',noderows)
 js(OUT/'binding_trace.json',{'status':'completed','wall_seconds':time.time()-start,'manifests':manifest,'query_summary':summary,'coordinate_convention':'Checkpoint local/native values are normalized, all *_m values divide by input world_scale and are BEHAVE k1 world. Camera0 depth positive away from input camera.','identity_caveat':'GS IDs and node IDs are stage-local; no initial-to-final lineage claimed. ref_time is reference frame and not optimization birth step. Source projection is reconstructed from stored center; exact initial frame/pixel awaits sampler provenance.','gate_decomposition_caveat':'Attached-only, pure DQB and actual DQB are diagnostic transformations with same frozen inputs, not new trained models. Pure DQB here sets gate exactly1; official dyn_o=False is.999. Weighted means are conditioned on tiny alpha at initialization, not physical surfaces.','no_reference_geometry_read':True,'source_code_hash':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),V/'analysis_a/sparse_raster_exact.py',V/'motion_binding/audit_motion_binding.py']}})
 print(json.dumps({'status':'completed','seconds':time.time()-start,'GS_contribution_rows':len(gsrows),'active_node_rows':len(noderows),'hand_summaries':[r for r in summary if r['query_id'].startswith('hand')]},ensure_ascii=False))
if __name__=='__main__':
 with torch.no_grad():main()
