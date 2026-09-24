#!/usr/bin/env python3
"""One CPU-only first-frame checkpoint inspection, no temporal snapshots/training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,csv
from pathlib import Path
import numpy as np,torch
from pytorch3d.transforms import quaternion_to_matrix
from sparse_raster_exact import sparse_raster
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;RUN=ROOT/'experiments/mosca_validation_20260923';BASE=ROOT/'experiments/mosca_baseline_20260922'
sys.path.insert(0,str(RUN/'motion_binding'))
from audit_motion_binding import get_weights,warp

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean_depth(points,w,C):
 if not len(w) or w.sum()==0:return None,None
 p=(points*w[:,None]).sum(0)/w.sum();return p,float(((p-C[:3,3])@C[:3,:3])[2])
def main():
 torch.set_num_threads(4);m=json.loads((BASE/'common_input/input_manifest.json').read_text());q=json.loads((BASE/'common_input/queries_first_frame.json').read_text());C=np.array(m['c2w']);K=np.array(m['K']);rows=[];pos=[];hashes={};initial_d=[]
 for name,branch,prefix in [('A_initial','a_normalized_exact','initial'),('B_initial','b_normalized_static_bg','initial'),('A_final','a_normalized_exact','photometric')]:
  model=RUN/branch/'model';scale=json.loads((RUN/'a_normalized_exact/model/run.json').read_text())['world_scale']
  sp=model/(f'{prefix}_static.pth' if prefix=='initial' else 'photometric_s_model_native_add3.pth');dp=model/(f'{prefix}_dynamic.pth' if prefix=='initial' else 'photometric_d_model_native_add3.pth');s=torch.load(sp,map_location='cpu',weights_only=False);d=torch.load(dp,map_location='cpu',weights_only=False)
  if prefix=='initial':initial_d.append(d)
  hashes[str(sp.relative_to(ROOT))]=sha(sp);hashes[str(dp.relative_to(ROOT))]=sha(dp);g=get_weights(d);dx,dr=warp(d,g,0,dyn_off=not bool(d['dyn_o_flag']))
  sx=s['_xyz'];sr=quaternion_to_matrix(torch.nn.functional.normalize(s['_rotation'],dim=-1));scales=[v['min_scale']+torch.sigmoid(v['_scaling'])*(v['max_scale']-v['min_scale']) for v in [s,d]];op=[torch.sigmoid(v['_opacity']) for v in [s,d]]
  x=torch.cat([sx,dx]).numpy()/scale;rot=torch.cat([sr,dr]).numpy();ss=torch.cat(scales).numpy()/scale;oo=torch.cat(op).numpy();ns=len(sx)
  contrib=sparse_raster(x,rot,ss,oo,K,C,np.array(q['points'][:6]));pred=[]
  for j,(idx,w) in enumerate(contrib):
   dyn=idx>=ns;p,z=mean_depth(x[idx],w,C);ps,zs=mean_depth(x[idx[~dyn]],w[~dyn],C);pd,zd=mean_depth(x[idx[dyn]],w[dyn],C)
   pred.append(np.full(3,np.nan) if p is None else p)
   rows.append({'stage':name,'query_id':q['point_ids'][j],'alpha':float(w.sum()),'static_fraction':float(w[~dyn].sum()/w.sum()) if w.sum() else None,'dynamic_fraction':float(w[dyn].sum()/w.sum()) if w.sum() else None,'combined_camera_z_m':z,'static_conditional_camera_z_m':zs,'dynamic_conditional_camera_z_m':zd,'contributing_static_gs':int((~dyn).sum()),'contributing_dynamic_gs':int(dyn.sum()),'conditional_dynamic_gate':float(np.sum(g['dyn'][idx[dyn]-ns].numpy()*w[dyn])/w[dyn].sum()) if w[dyn].sum() else None})
  pred=np.array(pred);pos.append(pred)
  if name=='A_final':assert np.max(abs(pred-np.load(RUN/'a_normalized_exact/diagnostics/query_trajectories.npz')['predicted'][0]))<2e-4
 assert initial_d[0].keys()==initial_d[1].keys();identical=all(torch.equal(initial_d[0][k],initial_d[1][k]) for k in initial_d[0])
 # Persist first-frame prediction-only measurements before loading reference.
 np.savez_compressed(OUT/'initial_source_prediction_only.npz',predicted=np.array(pos),stage=np.array(['A_initial','B_initial','A_final']),query_id=np.array(q['point_ids'][:6]),coordinate_frame=np.array('behave_world_k1_color'),units=np.array('m'))
 ref=np.load(RUN/'a_normalized_exact/evaluation/evaluation_bundle.npz')['reference'][0];dep=np.load(BASE/'common_input/unidepth_depth/00000.npz')['dep']
 for i,row in enumerate(rows):
  stage=i//6;j=i%6;err=pos[stage][j]-ref[j];u,v=q['points'][j]
  row.update(reference_camera_z_m=float(((ref[j]-C[:3,3])@C[:3,:3])[2]),input_depth_m=float(dep[v,u]),source_3d_epe_m=float(np.linalg.norm(err)),source_camera_z_error_m=float((err@C[:3,:3])[2]))
 with (OUT/'initial_final_first_frame.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 record={'status':'completed','cpu_only':True,'initial_dynamic_states_identical_A_B':identical,'source_protocol':'Same exact camera and pixel queries; original saved checkpoint source opacity/scale/warp. Centres and scales divided by common input normalization. No colour or RGB-error inference from low-alpha initialization.','rows':rows,'hashes':hashes,'initial_alpha_note':'Alpha is reported. Conditional centres of weak/transparent initial surfaces are not a final opaque surface and do not establish physical identity. Comparing checkpoints shows where source depth bias appears, not a full causal explanation of training.','script_sha256':sha(Path(__file__))};(OUT/'initial_final_source.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':
 with torch.no_grad():main()
