"""CPU-only evaluation AFTER frozen GVHMR prediction; never updates inference."""
from pathlib import Path
import hashlib,json,time,datetime
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path('/home/cai_tianshun/Project/HOI');E=ROOT/'experiments/gvhmr_validation_20260923';RUN=E/'run01';OUT=E/'reference_quality';PROTOCOL=E/'reference_evaluation_protocol.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
 a=np.asarray(a,dtype=np.float64); assert np.isfinite(a).all();return dict(mean_cm=float(a.mean()*100),median_cm=float(np.median(a)*100),p90_cm=float(np.quantile(a,.9)*100),max_cm=float(a.max()*100),count=int(a.size))
start=time.perf_counter();protocol=json.loads(PROTOCOL.read_text());run=json.loads((RUN/'run.json').read_text());assert run['status']=='completed' and run['reference_used'] is False and run['parameter_updates']==0
for p,h in protocol['inputs'].items():assert sha(p)==h,(p,'protocol hash mismatch')
for p,h in run['outputs'].items():assert sha(RUN/p)==h,(p,'frozen output hash mismatch')
assert not OUT.exists(),'Refuse overwrite existing evaluation'
OUT.mkdir();(OUT/'protocol_frozen_copy.json').write_bytes(PROTOCOL.read_bytes());(OUT/'inference_frozen_copy.json').write_bytes((RUN/'run.json').read_bytes())
base=ROOT/'third_party/GVHMR/hmr4d/utils/body_model';r=torch.load(base/'smpl_coco17_J_regressor.pt',weights_only=True,map_location='cpu').numpy().astype(np.float64);c=torch.load(base/'smplx2smpl_sparse.pt',weights_only=True,map_location='cpu').to_dense().numpy().astype(np.float64);rc=r@c
refpath=ROOT/'research/2026-09-23/evaluation_pointcloud_reference/person_fitted_vertices_world.npz';ref=np.load(refpath,allow_pickle=False);jref=np.einsum('jv,tvc->tjc',r,ref['xyz_world'].astype(np.float64));idx=ref['input_frame_indices'];tt=ref['frame_times'];assert idx.tolist()==protocol['reference_indices']
body=np.arange(5,17);wrist=[9,10];refs_hip=jref[:,[11,12]].mean(1); reference_offset=tt-ref['nominal_frame_times'];m=json.loads((ROOT/'experiments/mosca_baseline_20260922/common_input/input_manifest.json').read_text());C=np.array(m['c2w']);K=np.array(m['K'])
summary={'status':'completed','evaluation_only':True,'references_never_used_for_inference':True,'protocol_sha256':sha(PROTOCOL),'inference_run_sha256':sha(RUN/'run.json'),'reference_sha256':sha(refpath),'world_frame':'BEHAVE Kinect1 color, metres; no alignment','comparison':'registered fit proxy, no direct sensor GT','input_frame_indices':idx.tolist(),'actual_rgb_seconds':tt.tolist(),'nominal_reference_seconds':ref['nominal_frame_times'].tolist(),'nominal_time_offset_seconds':reference_offset.tolist(),'branches':{},'script_sha256':sha(__file__),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};allj={};arrays={'reference_coco17_world_m':jref,'reference_input_indices':idx,'reference_nominal_seconds':ref['nominal_frame_times'],'matched_rgb_seconds':tt}
for branch in ['raw','official_postproc']:
 pp=RUN/f'{branch}_geometry.npz';g=np.load(pp,allow_pickle=False)
 assert np.array_equal(g['timestamp_seconds'][idx],tt) and np.array_equal(g['input_frame_index'],np.arange(114))
 assert np.array_equal(g['c2w'],C) and np.array_equal(g['K'],K)
 allj[branch]=np.einsum('jv,tvc->tjc',rc,g['vertices_world_m']);j=allj[branch][idx];assert np.isfinite(j).all();predhip=j[:,[11,12]].mean(1)
 err=np.linalg.norm(j-jref,axis=-1);rootrel=np.linalg.norm((j-predhip[:,None])-(jref-refs_hip[:,None]),axis=-1);disp=np.linalg.norm((j-j[:1])-(jref-jref[:1]),axis=-1);hiperr=np.linalg.norm(predhip-refs_hip,axis=-1)
 exported_delta=np.linalg.norm(allj[branch]-g['joints_coco17_world_m'],axis=-1)
 summary['branches'][branch]={'geometry_sha256':sha(pp),'absolute_all17':stats(err),'absolute_body12':stats(err[:,body]),'absolute_left_wrist':stats(err[:,9]),'absolute_right_wrist':stats(err[:,10]),'absolute_wrists_mean':stats(err[:,wrist]),'hip_midpoint_absolute':stats(hiperr),'hip_relative_body12':stats(rootrel[:,body]),'first_frame_displacement_all17':stats(disp),'first_frame_displacement_body12':stats(disp[:,body]),'first_frame_displacement_wrists':stats(disp[:,wrist]),'per_frame':{'absolute_all17_cm':(err.mean(1)*100).tolist(),'absolute_body12_cm':(err[:,body].mean(1)*100).tolist(),'left_wrist_cm':(err[:,9]*100).tolist(),'right_wrist_cm':(err[:,10]*100).tolist(),'hip_midpoint_cm':(hiperr*100).tolist(),'hip_relative_body12_cm':(rootrel[:,body].mean(1)*100).tolist(),'displacement_all17_cm':(disp.mean(1)*100).tolist()},'exported_joint_vs_regressed_mesh_max_m':float(exported_delta.max()),'invalid_joint_count':0}
 arrays[branch+'_coco17_world_m']=allj[branch];arrays[branch+'_error_14x17_m']=err;arrays[branch+'_rootrelative_error_14x17_m']=rootrel;arrays[branch+'_firstframe_displacement_error_14x17_m']=disp
 # Two calculation routes agree up to original regressor row-sum translation convention.
 assert exported_delta.max()<2e-4,('Unexpected full mesh / lite joint mismatch',branch,float(exported_delta.max()))
np.savez_compressed(OUT/'joint_comparison.npz',**arrays)
# Time curves use actual input timestamps; sparse references remain discrete.
colors={'raw':'#2865bd','official_postproc':'#aa4db7'};labels={'raw':'Raw incam','official_postproc':'Official postproc incam'}
fig,axes=plt.subplots(3,3,figsize=(13.5,8.6),dpi=150,sharex=True)
for row,(label,ids) in enumerate([('Left wrist',[9]),('Right wrist',[10]),('Hip midpoint',[11,12])]):
 qref=jref[:,ids].mean(1)
 for axis in range(3):
  ax=axes[row,axis];ax.plot(tt,qref[:,axis],'ko--',lw=1.4,ms=3,label='Registered fit (14 samples)')
  for b in allj:ax.plot(g['timestamp_seconds'],allj[b][:,ids].mean(1)[:,axis],color=colors[b],lw=1.2,label=labels[b])
  ax.set_title(label+' / World '+'XYZ'[axis]);ax.set_ylabel('metres');ax.grid(alpha=.2)
  if row==2:ax.set_xlabel('Actual input time (s)')
handles,leglabels=axes[0,0].get_legend_handles_labels();fig.legend(handles,leglabels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.995));fig.suptitle('Calibrated world trajectories: no alignment | fit timing remains approximate',y=.945);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(OUT/'world_trajectories.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12.5,3.5),dpi=150)
for ax,(key,label) in zip(axes,[('absolute_body12_cm','Absolute body12'),('hip_relative_body12_cm','Hip-relative body12'),('hip_midpoint_cm','Hip midpoint position')]):
 for b in allj:ax.plot(tt,summary['branches'][b]['per_frame'][key],'-o',ms=3,color=colors[b],label=labels[b])
 ax.set_title(label);ax.set_xlabel('Matched input time (s)');ax.set_ylabel('Error vs registered fit (cm)');ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'reference_errors.png');plt.close(fig)
# Predetermined eight quality frames; actual reference nearest mapping already fixed.
selected=[0,16,25,31,55,65,93,113];edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
def project_world(x):
 camera=(x-C[:3,3])@C[:3,:3];h=camera@K.T;assert (camera[...,2]>0).all();return h[...,:2]/h[...,2:]
fig,axes=plt.subplots(2,4,figsize=(14,9),dpi=140)
for ax,i in zip(axes.flat,selected):
 rp=np.where(idx==i)[0];assert len(rp)==1;ri=int(rp[0]);im=Image.open(m['frames'][i]['path']) if 'frames' in m else Image.open(m['frame_paths'][i]);ax.imshow(im)
 for b,pts,color in [('fit',jref[ri],'#0b8b3e'),('raw',allj['raw'][i],colors['raw']),('post',allj['official_postproc'][i],colors['official_postproc'])]:
  uv=project_world(pts)
  for a,z in edges:ax.plot(uv[[a,z],0],uv[[a,z],1],color=color,lw=1.1,alpha=.9)
  ax.scatter(uv[body,0],uv[body,1],s=8,c=color)
 ax.set_xlim(170,470);ax.set_ylim(460,110);ax.set_title(f'Frame {i} | RGB {tt[ri]:.3f}s\nFit nominal {ref["nominal_frame_times"][ri]:.0f}s');ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Input-camera overlay: green=fitted reference, blue=raw, purple=official postproc\nFixed crop for display only; fitted reference did not enter inference',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'reference_overlay.png');plt.close(fig)
summary['wall_seconds']=time.perf_counter()-start;summary['outputs']={p.name:sha(p)for p in OUT.iterdir()if p.suffix in ['npz','png']};(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({b:{k:v['mean_cm']for k,v in vals.items()if isinstance(v,dict)and'mean_cm'in v}for b,vals in summary['branches'].items()},indent=2))
