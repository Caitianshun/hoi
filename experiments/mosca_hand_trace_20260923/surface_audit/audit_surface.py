#!/usr/bin/env python3
"""CPU re-query of frozen A at predefined tracker/reference pixels. Evaluation only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,time,hashlib,csv
import numpy as np
import torch
import cv2
from pytorch3d.transforms import quaternion_to_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
RUN=ROOT/'experiments/mosca_validation_20260923';BASE=ROOT/'experiments/mosca_baseline_20260922'
A=RUN/'a_normalized_exact'
sys.path.insert(0,str(RUN/'analysis_a'));from sparse_raster_exact import sparse_raster
sys.path.insert(0,str(RUN/'motion_binding'));from audit_motion_binding import get_weights,warp,project
HAND=['hand_glove_centre','hand_glove_upper']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def dump(p,z):p.write_text(json.dumps(z,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def queries_bilinear(xyz,rot,scales,opacity,K,C,uv,ns):
 """Reuse validated integer compositor; bilinearly combine four integer pixels.
 XYZ buffers are premultiplied then normalized, like the source query exporter.
 Native depth is normalized per pixel first, then bilinearly sampled separately.
 """
 uv=np.asarray(uv,float);corners=[];mixes=[]
 for q in uv:
  if not np.isfinite(q).all() or not(0<=q[0]<=639 and 0<=q[1]<=479):raise ValueError('Out of image query; do not clamp it')
  x,y=np.floor(q).astype(int);dx,dy=q-[x,y]
  corners.extend([[x,y],[min(x+1,639),y],[x,min(y+1,479)],[min(x+1,639),min(y+1,479)]])
  mixes.append([(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy])
 unique,inverse=np.unique(np.array(corners),axis=0,return_inverse=True)
 raster=sparse_raster(xyz,rot,scales,opacity,K,C,unique)
 _,depth=project(xyz,K,C);rows=[]
 for j,q in enumerate(uv):
  ii=[];ww=[];dep=0.;a=0.
  for c,mix in enumerate(mixes[j]):
   ids,w=raster[inverse[j*4+c]]
   if mix<=0:continue
   alpha=w.sum();a+=mix*alpha
   if alpha>0:dep+=mix*np.dot(w,depth[ids])/alpha
   ii.extend(ids.tolist());ww.extend((w*mix).tolist())
  ids,which=np.unique(ii,return_inverse=True);weights=np.zeros(len(ids));np.add.at(weights,which,ww)
  order=np.argsort(depth[ids],kind='stable');ids=ids[order];weights=weights[order]
  if a<=1e-8:raise ValueError('No surface support; report rather than fabricate a position')
  centre=(xyz[ids]*weights[:,None]).sum(0)/a
  z=depth[ids];cdf=np.cumsum(weights)/a;front=np.minimum(weights,np.maximum(.5*a-(np.cumsum(weights)-weights),0))
  top=np.argsort(weights)[::-1][:5]
  ray=np.linalg.solve(K,np.r_[q,1.]);ray/=ray[2]
  world=(ray*dep)@C[:3,:3].T+C[:3,3]
  rows.append(dict(uv=q.tolist(),alpha=float(a),dynamic_fraction=float(weights[ids>=ns].sum()/a),
    static_fraction=float(weights[ids<ns].sum()/a),contributor_count=int(len(ids)),
    rendered_depth_bilinear_m=float(dep),weighted_centroid_world_m=centre.tolist(),
    weighted_centroid_camera_z_m=float(np.dot(weights,z)/a),depth_backprojected_world_m=world.tolist(),
    weighted_depth_quantiles_m={str(v):float(z[min(np.searchsorted(cdf,v),len(z)-1)]) for v in [.1,.5,.9]},
    nearest_half_alpha_mean_depth_m=float(np.dot(front,z)/front.sum()),
    top_contributors=[dict(combined_gaussian_index=int(ids[k]),component='dynamic' if ids[k]>=ns else 'static',
        conditional_alpha_weight=float(weights[k]/a),camera_z_m=float(z[k])) for k in top]))
 return rows

def main():
 begin=time.perf_counter();torch.set_num_threads(4)
 paths={'input_manifest':BASE/'common_input/input_manifest.json','queries':BASE/'common_input/queries_first_frame.json',
        'tracker':BASE/'common_input/uniform_cotracker_tap.npz',
        'reference':BASE/'evaluation/fixed_rgb_queries/same_version_joint_reference_observation_times.npz',
        'fixed_prediction':A/'diagnostics/query_trajectories.npz','export':A/'diagnostics/export_manifest.json',
        'static':A/'model/photometric_s_model_native_add3.pth','dynamic':A/'model/photometric_d_model_native_add3.pth',
        'raster_helper':RUN/'analysis_a/sparse_raster_exact.py','warp_helper':RUN/'motion_binding/audit_motion_binding.py'}
 hashes={str(p):sha(p) for p in paths.values()}
 m=read(paths['input_manifest']);q=read(paths['queries']);tr=load(paths['tracker']);ref=load(paths['reference']);pred=load(paths['fixed_prediction']);ex=read(paths['export'])
 assert sha(paths['input_manifest'])==ex['input_manifest_sha256']
 assert sha(paths['fixed_prediction'])==ex['trajectory_sha256']
 assert sha(paths['static'])==ex['checkpoint_sha256']['static'] and sha(paths['dynamic'])==ex['checkpoint_sha256']['dynamic']
 assert sha(paths['reference'])==read(A/'evaluation/protocol.json')['reference_file_sha256']
 K=np.array(m['K']);C=np.array(m['c2w']);scale=ex['world_scale'];queryids=ref['query_id'].tolist()
 assert pred['query_id'].tolist()==queryids==q['point_ids'][:6]
 assert np.allclose(tr['timestamp_seconds'],m['timestamp_seconds'],rtol=0,atol=1e-9)
 assert np.allclose(pred['frame_times'],m['timestamp_seconds'],rtol=0,atol=1e-9)
 assert np.array_equal(np.array(m['timestamp_seconds'])[ref['prediction_frame_indices']],ref['frame_times'])
 assert pred['coordinate_frame'].item()==ref['coordinate_frame'].item()=='behave_world_k1_color'
 assert pred['units'].item()==ref['units'].item()=='m'
 mapping={}
 for name in HAND:
  qi=queryids.index(name);wanted=np.r_[0.,q['points'][qi]]
  matches=np.flatnonzero(np.all(tr['query_points']==wanted,axis=-1));assert len(matches)==1
  mapping[name]=dict(query_index=qi,training_tracker_index=int(matches[0]),source_uv=q['points'][qi])
 s=torch.load(paths['static'],map_location='cpu',weights_only=False);d=torch.load(paths['dynamic'],map_location='cpu',weights_only=False)
 assert bool(d['leaf_local_flag']);g=get_weights(d);ns=len(s['_xyz'])
 sx=s['_xyz'].numpy()/scale;sr=quaternion_to_matrix(torch.nn.functional.normalize(s['_rotation'],dim=-1)).numpy()
 scales=torch.cat([v['min_scale']+torch.sigmoid(v['_scaling'])*(v['max_scale']-v['min_scale']) for v in [s,d]]).numpy()/scale
 opacity=torch.cat([torch.sigmoid(v['_opacity']) for v in [s,d]]).numpy()
 rows=[];validations={};previous=[]
 for ri,t in enumerate(ref['prediction_frame_indices']):
  t=int(t);dx,dr=warp(d,g,t,dyn_off=not bool(d['dyn_o_flag']))
  xyz=np.concatenate([sx,dx.numpy()/scale]);rot=np.concatenate([sr,dr.numpy()])
  kfpath=A/f'diagnostics/keyframes/{t:05d}.npz'
  if kfpath.exists():
   kf=load(kfpath);err=float(np.max(np.abs(xyz-kf['xyz_world'])));assert err<1e-4
   validations[f'keyframe_{t}_xyz_max_m']=err
  if t==0:
   source=queries_bilinear(xyz,rot,scales,opacity,K,C,np.array(q['points'][:6]),ns)
   validations['source_alpha_max_error']=float(max(abs(r['alpha']-pred['source_alpha'][j]) for j,r in enumerate(source)))
   validations['source_dynamic_fraction_max_error']=float(max(abs(r['dynamic_fraction']-pred['source_dynamic_fraction'][j]) for j,r in enumerate(source)))
   validations['source_xyz_max_error_m']=float(np.max(np.abs(np.array([r['weighted_centroid_world_m'] for r in source])-pred['predicted'][0])))
   assert max(validations.values())<2e-4,validations
  sample_uv=[];metadata=[]
  for name,mp in mapping.items():
   qi=mp['query_index'];ti=mp['training_tracker_index'];rf=ref['reference'][ri,qi];uv,z=project(rf[None],K,C)
   assert np.max(np.abs(uv[0]-ref['projected_uv'][ri,qi]))<1e-7
   fixed=pred['predicted'][t,qi];fuv,fz=project(fixed[None],K,C)
   for location,px in [('training_tracker',tr['tracks'][t,ti]),('reference_projection',uv[0])]:
    sample_uv.append(px);metadata.append(dict(query_id=name,input_frame_index=t,reference_index=ri,time_seconds=float(ref['frame_times'][ri]),
       sample_location=location,tracker_claims_visible=bool(tr['visibility'][t,ti]),
       tracker_uv=tr['tracks'][t,ti].tolist(),reference_uv=uv[0].tolist(),fixed_source_uv=fuv[0].tolist(),
       tracker_reference_pixel_distance=float(np.linalg.norm(tr['tracks'][t,ti]-uv[0])),
       reference_world_m=rf.tolist(),reference_camera_z_m=float(z[0]),
       nominal_reference_time_seconds=float(ref['nominal_frame_times'][ri]),reference_time_offset_seconds=float(ref['source_image_time_offset_seconds'][ri]),
       fixed_source_world_m=fixed.tolist(),fixed_source_camera_z_m=float(fz[0]),
       fixed_source_3d_epe_m=float(np.linalg.norm(fixed-rf))))
  for rr,meta in zip(queries_bilinear(xyz,rot,scales,opacity,K,C,sample_uv,ns),metadata):
   rr.update(meta);rr['depth_signed_error_to_fitted_point_m']=rr['rendered_depth_bilinear_m']-rr['reference_camera_z_m']
   rr['ray_backprojected_epe_to_fitted_point_m']=float(np.linalg.norm(np.array(rr['depth_backprojected_world_m'])-rr['reference_world_m']))
   rows.append(rr)
  print('CPU requery',t,flush=True)
 assert hashes=={str(p):sha(p) for p in paths.values()}
 summary={}
 for name in HAND:
  summary[name]={}
  for loc in ['training_tracker','reference_projection']:
   rr=[r for r in rows if r['query_id']==name and r['sample_location']==loc]
   summary[name][loc]=dict(count=len(rr),tracker_visible_count=sum(r['tracker_claims_visible'] for r in rr),
      alpha_min=min(r['alpha'] for r in rr),alpha_mean=float(np.mean([r['alpha'] for r in rr])),
      dynamic_fraction_mean=float(np.mean([r['dynamic_fraction'] for r in rr])),
      mean_abs_depth_difference_m=float(np.mean([abs(r['depth_signed_error_to_fitted_point_m']) for r in rr])),
      mean_ray_backprojection_difference_m=float(np.mean([r['ray_backprojected_epe_to_fitted_point_m'] for r in rr])),
      mean_fixed_source_3d_epe_m=float(np.mean([r['fixed_source_3d_epe_m'] for r in rr])),
      pixel_distance_mean=float(np.mean([r['tracker_reference_pixel_distance'] for r in rr])),
      abs_depth_within_10cm_count=sum(abs(r['depth_signed_error_to_fitted_point_m'])<=.1 for r in rr),
      abs_depth_within_30cm_count=sum(abs(r['depth_signed_error_to_fitted_point_m'])<=.3 for r in rr))
 result=dict(status='completed_cpu_only_evaluation',mapping=mapping,normalization_scale=scale,coordinate_frame='behave_world_k1_color',units='m',
    independent_reference_used_only_for_evaluation=True,reference_reprojection_pixels_are_evaluation_only=True,
    training_tracker_query_count=int(tr['tracks'].shape[1]),validation=validations,summary=summary,rows=rows,input_sha256=hashes,
    script_sha256=sha(__file__),elapsed_cpu_seconds=time.perf_counter()-begin,
    limitations=['Re-query changes contributing Gaussian identity every time; not material tracking.',
      'Reference-projection ray has reference-supplied 2D location; lower error is not a valid independent method improvement.',
      'Tracker coordinates may be incorrect or invisible; all requested times reported without visibility filtering.',
      'Fitted glove surface has association/timing uncertainty; occluded reference may be behind a legitimately nearer surface.',
      'High alpha means opaque explanation, not accurate hand geometry. Expected depths mix contributing layers.',
      'No training, no checkpoint changes, no fitted alignment, no reference interpolation, no query relocation.',
      'Depth uses bilinear sampling of per-integer-pixel normalized rendered depth; XYZ/weights use bilinear premultiplied buffers then alpha normalization.'])
 dump(OUT/'surface_audit.json',result)
 flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows]
 with (OUT/'surface_samples.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
 create_figures(result,m)
 print(json.dumps({'summary':summary,'validation':validations,'elapsed_cpu_seconds':result['elapsed_cpu_seconds']},indent=2),flush=True)

def create_figures(result,m):
 rows=result['rows'];fig,axs=plt.subplots(2,2,figsize=(14,9),sharex=True)
 fig.subplots_adjust(left=.07,right=.98,bottom=.18,top=.8,wspace=.22,hspace=.35)
 fig.suptitle('Frozen A: fixed identity vs. current rendered surface',fontsize=19,fontweight='bold',y=.97)
 colours={'fixed':'#c45841','training_tracker':'#2b80a5','reference_projection':'#568d37'}
 labels={'fixed':'Fixed source','training_tracker':'At tracker UV','reference_projection':'At fitted UV'}
 for i,name in enumerate(HAND):
  rr=[r for r in rows if r['query_id']==name and r['sample_location']=='reference_projection'];tt=np.array([r['time_seconds'] for r in rr])
  axs[i,0].plot(tt,[r['fixed_source_camera_z_m'] for r in rr],'-o',color=colours['fixed'],markersize=4,label=labels['fixed'])
  axs[i,1].plot(tt,[r['fixed_source_3d_epe_m'] for r in rr],'-o',color=colours['fixed'],markersize=4,label=labels['fixed'])
  for loc in ['training_tracker','reference_projection']:
   aa=[r for r in rows if r['query_id']==name and r['sample_location']==loc]
   axs[i,0].plot(tt,[r['rendered_depth_bilinear_m'] for r in aa],'-^',color=colours[loc],markersize=4,label=labels[loc])
   axs[i,1].plot(tt,[r['ray_backprojected_epe_to_fitted_point_m'] for r in aa],'-^',color=colours[loc],markersize=4,label=labels[loc])
  axs[i,0].scatter(tt,[r['reference_camera_z_m'] for r in rr],color='#222222',marker='D',s=31,label='Fitted ref.',zorder=5)
  title='Glove centre' if i==0 else 'Glove upper'
  axs[i,0].set_title(title+' - camera depth');axs[i,1].set_title(title+' - reference difference')
  axs[i,0].set_ylabel('Camera z (m)');axs[i,1].set_ylabel('3D difference (m)');axs[i,1].set_ylim(bottom=0)
  for ax in axs[i]:
   ax.grid(alpha=.18)
   if i==1:ax.set_xlabel('Actual observation time (s)')
 h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
 fig.text(.07,.09,'Re-query is an evaluation probe, not a material-point trajectory or a replacement tracking score.',fontsize=11)
 fig.text(.07,.058,'Fitted UV supplies the 2D target location. Occlusion and imperfect glove fitting remain unresolved.',fontsize=11)
 fig.text(.07,.026,'Tracker readings include all 14 requested times; the tracker declares only 5 visible times per hand query.',fontsize=11)
 fig.savefig(OUT/'surface_depth_and_reference_difference.png',dpi=170);plt.close(fig)
 # Four source frames fixed before seeing scores: start, occlusion, reappearance, end.
 panels=[]
 for t in [0,25,31,113]:
  rr=[r for r in rows if r['input_frame_index']==t and r['sample_location']=='reference_projection']
  points=np.array([r[k] for r in rr for k in ['reference_uv','tracker_uv','fixed_source_uv']])
  lo=np.maximum(np.floor(points.min(0)-24).astype(int),[0,0]);hi=np.minimum(np.ceil(points.max(0)+24).astype(int),[640,480]);x0,y0=lo;x1,y1=hi
  tiles=[]
  for title,path in [('Input RGB',Path(m['frame_paths'][t])),('A rendered RGB',A/f'diagnostics/rgb/{t:05d}.png')]:
   im=cv2.imread(str(path));crop=im[y0:y1,x0:x1].copy();scale=min(420/crop.shape[1],240/crop.shape[0]);nw,nh=round(crop.shape[1]*scale),round(crop.shape[0]*scale)
   crop=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_NEAREST);tile=np.full((290,455,3),255,np.uint8);ox=(455-nw)//2;oy=44+(240-nh)//2;tile[oy:oy+nh,ox:ox+nw]=crop
   for j,r in enumerate(rr):
    for key,color,marker in [('reference_uv',(40,140,40),cv2.MARKER_CROSS),('tracker_uv',(205,145,20),cv2.MARKER_TILTED_CROSS),('fixed_source_uv',(45,80,210),cv2.MARKER_DIAMOND)]:
     point=np.round((np.array(r[key])-[x0,y0]+.5)*scale-.5+[ox,oy]).astype(int)
     cv2.drawMarker(tile,tuple(point),color,marker,11,1,cv2.LINE_AA)
   cv2.putText(tile,f'{title} | frame {t} | {rr[0]["time_seconds"]:.3f}s',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,20,20),1,cv2.LINE_AA)
   cv2.putText(tile,'+ fitted UV   x tracker UV   diamond fixed source',(8,37),cv2.FONT_HERSHEY_SIMPLEX,.40,(30,30,30),1,cv2.LINE_AA)
   tiles.append(tile)
  panels.append(np.concatenate(tiles,axis=1))
 cv2.imwrite(str(OUT/'fixed_pixels_input_and_A_render.png'),np.concatenate(panels,axis=0))

if __name__=='__main__':
 with torch.no_grad():main()
