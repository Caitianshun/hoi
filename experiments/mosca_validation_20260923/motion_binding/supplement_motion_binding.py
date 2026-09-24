#!/usr/bin/env python3
"""CPU, input-only supplementary attribution and frozen-gate risk checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,time
from pathlib import Path
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_motion_binding import BASE,OUT,ROOT,get_weights,warp,sparse_raster,dump,quant,NAMES,sha

def weighted_quant(x,w):
 o=np.argsort(x);x=x[o];w=w[o];return {'weighted_mean':float(np.sum(x*w)/np.sum(w)),'weighted_p10_p50_p90':np.interp([.1,.5,.9],np.cumsum(w)/w.sum(),x).tolist()}

def main():
 start=time.time();d=torch.load(BASE/'mosca_cotracker/photometric_d_model_native_add3.pth',map_location='cpu',weights_only=False);g=get_weights(d)
 m=json.loads((BASE/'common_input/input_manifest.json').read_text());q=json.loads((BASE/'common_input/queries_first_frame.json').read_text());a=json.loads((OUT/'motion_binding_audit.json').read_text());labels=np.load(OUT/'diagnostic_labels_and_weights.npz');nl=labels['final_node_labels'];gslabel=labels['gaussian_labels'];ns=int(labels['gaussian_static_count']);dl=gslabel[ns:]
 f=np.load(BASE/'mosca_cotracker/diagnostics/keyframes/00000.npz');c=sparse_raster(f['xyz_world'],f['rotation_frame'],f['scales'],f['opacity'],np.array(m['K']),np.array(m['c2w']),np.array(q['points'][:6]))
 att=d['attach_ind'].numpy();ti=d['ref_time'].numpy();dist=np.linalg.norm(g['src'].numpy()-g['node'].numpy()[ti,att],axis=-1);sigma=(torch.sigmoid(d['scf._node_sigma_logit'])*d['scf.max_sigma']).numpy().ravel()[att]
 bound=[];contribrows=[]
 for qi,(ii,w) in enumerate(c):
  did=ii[ii>=ns]-ns;dw=w[ii>=ns];bound.append({'query_id':q['point_ids'][qi],'attach_distance_m':weighted_quant(dist[did],dw),'attach_sigma_m':weighted_quant(sigma[did],dw),'attach_distance_over_sigma':weighted_quant(dist[did]/sigma[did],dw),'ref_frame_index':weighted_quant(ti[did],dw),'ref_frame_source0_alpha_fraction':float(dw[ti[did]==0].sum()/dw.sum())})
  for j,wj in zip(did,dw):
   nw=g['w'][j].numpy();nd=g['idx'][j].numpy();top=int(nd[nw.argmax()]);contribrows.append({'query_id':q['point_ids'][qi],'dynamic_gaussian_id':int(j),'source_total_alpha_fraction':float(wj/w.sum()),'gs_diagnostic_label':NAMES[int(dl[j])],'attach_node':int(att[j]),'attach_label':NAMES[int(nl[att[j]])],'ref_frame':int(ti[j]),'distance_to_attach_m':float(dist[j]),'attach_sigma_m':float(sigma[j]),'RBF_sum':float(g['total'][j]),'identity_mix':float(1-g['dyn'][j]),'top_actual_weight_node':top,'top_node_label':NAMES[int(nl[top])],'top_actual_weight':float(nw.max())})
 with (OUT/'source_dynamic_gaussian_contributors.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(contribrows[0]));w.writeheader();w.writerows(contribrows)
 # Held frozen source support. These are post-intervention shape changes, not a reconstruction improvement.
 ts=[0,16,20,24,28,113];on=[];off=[];risk=[]
 for t in ts:
  on.append(warp(d,g,t)[0].numpy());off.append(warp(d,g,t,dyn_off=True)[0].numpy())
 on=np.array(on);off=np.array(off);native_disp=np.linalg.norm(on-on[:1],axis=-1).max(0);off_disp=np.linalg.norm(off-off[:1],axis=-1).max(0);shift=np.linalg.norm(off-on,axis=-1).max(0)
 for lab in [-1,0,1,2]:
  sel=dl==lab;risk.append({'label':NAMES[lab],'count':int(sel.sum()),'native_max_displacement_m':quant(native_disp[sel]),'off_max_displacement_m':quant(off_disp[sel]),'max_intervention_shift_m':quant(shift[sel]),'off_displacement_above_1m_count':int((off_disp[sel]>1).sum()),'off_displacement_above_3m_count':int((off_disp[sel]>3).sum())})
 # Inspect candidate neighbors vs actually weighted base ARAP to avoid the KNN-index fallacy.
 edge=d['scf.topo_knn_ind'].numpy();mask=d['scf.topo_knn_mask'].numpy();src=np.broadcast_to(np.arange(len(nl))[:,None],edge.shape);known=(nl[src]>=0)&(nl[edge]>=0);cross=(nl[src]!=nl[edge])&known;nonself=src!=edge
 raw={'known_cross_entity_neighbor_slots_before_mask':int((nonself&cross).sum()),'same_slots_after_mask':int((nonself&cross&mask).sum()),'learned_sk_correction_nonzero_on_original_masked_slots':int(((~g['mask'])&(d['_skinning_weight']!=0)).sum())}
 dump(OUT/'motion_binding_supplement.json',{'seconds':time.time()-start,'source_attach_geometry':bound,'global_off_risk_at_six_keyframes':risk,'raw_vs_effective_neighbor':raw,'finite_off_coordinates':bool(np.isfinite(off).all()),'interpretation':'GS labels are estimated mask/depth projection labels, not GT identities. Displacements use each branch own first-frame positions; intervention also shifts first-frame shapes. Full gate-off may mobilize background dynamic GSs. No reference read.'})
 # Readable static scientific diagnostic, no math-renderer dependency.
 tr=np.load(OUT/'frozen_query_motion.npz');tt=tr['frame_times'];nxyz=g['node'].numpy();fig,ax=plt.subplots(2,3,figsize=(13,6),sharex=True)
 for j,aa in enumerate(ax.flat):
  p=tr['actual'][:,j];o=tr['dyn_o_off'][:,j];nw=tr['source_node_weight'][j];node=(nxyz*nw[None,:,None]).sum(1)/nw.sum()
  aa.plot(tt,np.linalg.norm(p-p[0],axis=-1),label='Native gate on',lw=2)
  aa.plot(tt,np.linalg.norm(o-o[0],axis=-1),label='Frozen gate off (.999)',lw=2)
  aa.plot(tt,np.linalg.norm(node-node[0],axis=-1),label='Weighted node centres',lw=1.4,ls='--')
  aa.axvspan(20.63,21.063,color='gray',alpha=.15);aa.set_title(q['point_ids'][j]);aa.set_ylabel('Displacement (m)');aa.grid(alpha=.2)
  if j>=3:aa.set_xlabel('Actual input time (s)')
 ax[0,0].legend(fontsize=8);fig.suptitle('Frozen source association; displacement from each individual first frame\nShaded: manually selected severe-occlusion RGB interval; not point visibility ground truth',fontsize=11);fig.tight_layout();fig.savefig(OUT/'gate_node_motion.png',dpi=180);plt.close(fig)
 # Full event timeseries support, masks do not define the event boundary.
 rows=list(csv.DictReader(open(OUT/'temporal_object_support.csv')));times=np.array([float(r['time_s']) for r in rows]);fig,aa=plt.subplots(2,1,figsize=(11,5),sharex=True)
 for key,name in [('visible_object_labeled_tracks','Visible object-labeled input tracks'),('depthvalid_object_labeled_tracks','Depth-valid object-labeled input tracks')]:aa[0].plot(times,[float(r[key]) for r in rows],label=name)
 for key,name in [('certain_object_labeled_final_nodes','Photo-stage certain object nodes'),('certain_object_labeled_geometry_nodes','Geometry-stage certain object nodes')]:aa[1].plot(times,[float(r[key]) for r in rows],label=name)
 for p in aa:p.axvspan(20.63,21.063,color='gray',alpha=.15);p.grid(alpha=.2);p.legend(fontsize=8);p.set_ylabel('Count')
 aa[1].set_xlabel('Actual input time (s)');fig.suptitle('RGB-prior object support across representation stages (different units)');fig.tight_layout();fig.savefig(OUT/'object_support_timeline.png',dpi=180);plt.close(fig)
 print(json.dumps({'source':bound,'raw':raw,'risk':risk},ensure_ascii=False))
if __name__=='__main__':
 with torch.no_grad():main()
