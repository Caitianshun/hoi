#!/usr/bin/env python3
"""CPU frozen initial-state diagnostic: opacity .012->.02 or remove alpha cutoff. No model write."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,csv
from pathlib import Path
import numpy as np,torch
from pytorch3d.transforms import quaternion_to_matrix
from join_seed_and_threshold import kernel,ROOT,OUT,V,BASE,A
sys.path.insert(0,str(V/'motion_binding'));from audit_motion_binding import get_weights,warp

def main():
 torch.set_num_threads(4);m=json.loads((BASE/'common_input/input_manifest.json').read_text());q=json.loads((BASE/'common_input/queries_first_frame.json').read_text());C=np.array(m['c2w']);K=np.array(m['K']);scale=json.loads((A/'model/run.json').read_text())['world_scale'];dep=np.load(BASE/'common_input/unidepth_depth/00000.npz')['dep'];d=torch.load(A/'model/initial_dynamic.pth',map_location='cpu',weights_only=False);s=torch.load(A/'model/initial_static.pth',map_location='cpu',weights_only=False);g=get_weights(d);dx,dr=warp(d,g,0);ns=len(s['_xyz']);xyz=np.r_[s['_xyz'].numpy()/scale,dx.numpy()/scale];R=np.concatenate([quaternion_to_matrix(torch.nn.functional.normalize(s['_rotation'],dim=-1)).numpy(),dr.numpy()]);scales=np.concatenate([(v['min_scale']+torch.sigmoid(v['_scaling'])*(v['max_scale']-v['min_scale'])).numpy()/scale for v in [s,d]]);opacity=np.concatenate([torch.sigmoid(v['_opacity']).numpy() for v in [s,d]]);rows=[]
 for name,change,cutoff in [('native_initial',False,1/255),('dynamic_opacity_0_02',True,1/255),('diagnostic_no_alpha_cutoff',False,0.)]:
  op=opacity.copy()
  if change:op[ns:]=.02
  for j,uv in enumerate(np.array(q['points'][:6])):
   k=kernel(xyz,R,scales,op,K,C,uv);ia=np.flatnonzero(k['tile_eligible']&(k['alpha_before_threshold']>=cutoff)&(k['alpha_before_threshold']>0));ia=ia[np.argsort(k['z'][ia],kind='stable')];a=k['alpha_before_threshold'][ia];after=np.cumprod(1-a);before=np.r_[1,after[:-1]];valid=after>=.0001;ia=ia[valid];w=(a*before)[valid];alpha=w.sum();z=k['z'][ia];d0=float(dep[uv[1],uv[0]]);near=abs(z-d0)<=.1;dy=ia>=ns
   rows.append({'condition':name,'query_id':q['point_ids'][j],'alpha':float(alpha),'weighted_camera_z_m':float(np.dot(w,z)/alpha),'input_depth_m':d0,'near_input_pm_0_1m_absolute_alpha':float(w[near].sum()),'near_input_pm_0_1m_fraction_covered':float(w[near].sum()/alpha),'farther_input_plus_0_25m_fraction_covered':float(w[z>d0+.25].sum()/alpha),'dynamic_fraction_covered':float(w[dy].sum()/alpha),'dynamic_conditional_depth_m':float(np.dot(w[dy],z[dy])/w[dy].sum()),'dynamic_contributors':int(dy.sum()),'near_dynamic_contributors':int((near&dy).sum()),'GS86_alpha_weight':float(w[ia==ns+86].sum()),'GS138_alpha_weight':float(w[ia==ns+138].sum())})
 with (OUT/'initial_opacity_probe.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 out={'status':'completed','training_performed':False,'checkpoint_modified':False,'reference_geometry_used':False,'change':'Only all initial dynamic opacity set to.02; static opacity, positions, scales, rotations, associations and point count fixed. Separate third diagnostic disables native1/255 per-splat cutoff only, retaining original tile and transmittance rules.','limitations':'Frozen initial rendering diagnostic. Low alpha persists. No evidence of final trained quality or recovery. Depth proximity to estimated input is not GT surface accuracy. No-cutoff is explicitly not the original renderer.','results':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(OUT/'initial_opacity_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':
 with torch.no_grad():main()
