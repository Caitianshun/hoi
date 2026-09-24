#!/usr/bin/env python3
"""CPU ordinary whole-sequence silhouette/depth/true-time SE(3) refinement."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np,cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt,map_coordinates
from PIL import Image,ImageDraw,ImageFont
ap=argparse.ArgumentParser();ap.add_argument('--initial',type=Path,required=True);ap.add_argument('--input-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ar=ap.parse_args();O=ar.output.resolve();O.mkdir(exist_ok=False,parents=True);B=ar.input_dir.resolve();T=time.perf_counter()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=dict(np.load(ar.initial));L=len(a['timestamp_seconds']);ts=a['timestamp_seconds'];m=json.loads((B/'input_manifest.json').read_text());labels=np.load(B/'segmentation/segmentation.npz')['entity_labels'];K=a['K'];S=a['centres_m'][::2].astype(float);tree=cKDTree(S);obs=[];run={'status':'running','GPU_used':False,'initial_sha256':sha(ar.initial),'code_sha256':sha(__file__),'no_reference_used':True,'loss_weights':{'two_sided_image_average':1,'depth_average':.2,'translation_acceleration_sigma_m_s2':10.,'angular_acceleration_sigma_rad_s2':80.,'initial_pose_soft_tether':.015},'actual_time_differences':True};(O/'run.json').write_text(json.dumps(run,indent=2))
for i in range(L):
 uv=np.argwhere(labels[i]==2)[:,::-1].astype(float);valid=a['observed_mask'][i];ids=np.linspace(0,max(0,len(uv)-1),64,dtype=int);uv=uv[ids] if len(uv) else np.zeros((64,2));field=distance_transform_edt(~((labels[i]==1)|(labels[i]==2)))
 dep=np.load(B/f'unidepth_depth/{i:05d}.npz')['dep'];z=dep[uv[:,1].astype(int),uv[:,0].astype(int)];xyz=np.c_[uv,np.ones(64)]@np.linalg.inv(K).T*z[:,None];obs.append((uv,field,xyz,valid,bool(a['depth_observed_mask'][i])))
Rinit=a['R_camera'].astype(float);tinit=a['t_camera_m'].astype(float);p0=np.c_[Rotation.from_matrix(Rinit).as_rotvec(),tinit].reshape(-1);calls=0
# Per-frame 64 observed-to-surface, 64 surface-to-permitted, 64 depth, 6 soft tether.
N=198; rows=L*N+(L-2)*6;sp=lil_matrix((rows,L*6),dtype=int)
for i in range(L):sp[i*N:(i+1)*N,i*6:(i+1)*6]=1
for i in range(1,L-1):sp[L*N+(i-1)*6:L*N+i*6,(i-1)*6:(i+2)*6]=1
sp=sp.tocsr()
def residual(p):
 global calls
 calls+=1;x=p.reshape(L,6);Rs=Rotation.from_rotvec(x[:,:3]).as_matrix();tt=x[:,3:];vals=[]
 for i,(uv,field,xyz,vis,depthok) in enumerate(obs):
  if vis:
   cam=S@Rs[i].T+tt[i];h=cam@K.T;puv=h[:,:2]/h[:,2:];target=cKDTree(puv).query(uv)[0]/(2*8);back=map_coordinates(field,[puv[:,1],puv[:,0]],order=1,mode='constant',cval=100);back=np.quantile(back,np.linspace(0,1,64))/(2*8);depth=tree.query((xyz-tt[i])@Rs[i])[0]/(.1*8)*.2 if depthok else np.zeros(64)
  else:target=back=depth=np.zeros(64)
  tether=np.r_[Rotation.from_matrix(Rs[i]@Rinit[i].T).as_rotvec(),tt[i]-tinit[i]]*.015;vals.append(np.r_[target,back,depth,tether])
 dt=np.diff(ts);v=np.diff(tt,axis=0)/dt[:,None];w=Rotation.from_matrix(np.einsum('tij,tkj->tik',Rs[1:],Rs[:-1])).as_rotvec()/dt[:,None]
 for i in range(1,L-1):
  den=(dt[i-1]+dt[i])/2;vals.append(np.r_[(v[i]-v[i-1])/den/10.,(w[i]-w[i-1])/den/80.])
 if calls%200==0:print('residual_calls',calls,'elapsed',round(time.perf_counter()-T,2),flush=True)
 return np.concatenate(vals)
initial=residual(p0);lo=np.full_like(p0,-np.inf).reshape(L,6);hi=np.full_like(p0,np.inf).reshape(L,6);lo[:,5]=.4;hi[:,5]=8
res=least_squares(residual,p0,jac_sparsity=sp,bounds=(lo.ravel(),hi.ravel()),loss='soft_l1',f_scale=1,max_nfev=25,diff_step=1e-4,ftol=1e-4,xtol=1e-4,gtol=1e-4,verbose=1)
x=res.x.reshape(L,6);R=Rotation.from_rotvec(x[:,:3]).as_matrix();t=x[:,3:];a['R_camera']=R.astype(np.float32);a['t_camera_m']=t.astype(np.float32);c=a['c2w'];a['R_world']=np.einsum('ij,tjk->tik',c[:3,:3],R).astype(np.float32);a['t_world_m']=(t@c[:3,:3].T+c[:3,3]).astype(np.float32);masks=[];diags=[]
for i in range(L):
 vv=a['canonical_vertices_m']@R[i].T+t[i];uv=vv@K.T;uv=uv[:,:2]/uv[:,2:];mask=np.zeros((480,640),np.uint8)
 for tri in np.rint(uv[a['faces']]).astype(np.int32):cv2.fillConvexPoly(mask,tri,1)
 target=labels[i]==2;pred=mask.astype(bool)&(labels[i]!=1);iou=float((pred&target).sum()/max(1,(pred|target).sum()));rel=min(1,target.sum()/1000)*iou if a['observed_mask'][i] else 0;a['reliability'][i]=rel;masks.append(mask);diags.append({'frame':i,'observed':bool(a['observed_mask'][i]),'object_pixels':int(target.sum()),'visible_silhouette_iou':iou})
np.savez_compressed(O/'object_init.npz',**a);np.savez_compressed(O/'input_diagnostics.npz',visible_silhouette_iou=[d['visible_silhouette_iou'] for d in diags],input_observed=a['observed_mask'],translation_speed_m_s=np.linalg.norm(np.diff(t,axis=0),axis=1)/np.diff(ts),rotation_increment_rad=Rotation.from_matrix(np.einsum('tij,tkj->tik',R[1:],R[:-1])).magnitude());(O/'frame_diagnostics.json').write_text(json.dumps(diags,indent=2));(O/'executed_refinement.py').write_bytes(Path(__file__).read_bytes());run.update(status='completed',wall_seconds=time.perf_counter()-T,optimizer_success=bool(res.success),optimizer_message=res.message,nfev=res.nfev,initial_residual_mean_square=float(np.mean(initial**2)),final_residual_mean_square=float(np.mean(res.fun**2)),output_sha256=sha(O/'object_init.npz'),no_new_observations=True,unobserved_pose_interpretation='weak temporal prediction only; reliability remains zero');(O/'run.json').write_text(json.dumps(run,indent=2))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);tiles=[]
for f in np.unique(np.linspace(0,L-1,12,dtype=int)):
 im=np.array(Image.open(m['frame_paths'][f]).convert('RGB'));ct,_=cv2.findContours((labels[f]==2).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,ct,-1,(0,255,70),1);ct,_=cv2.findContours(masks[f],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,ct,-1,(255,40,220),1);tile=Image.new('RGB',(640,535),'white');tile.paste(Image.fromarray(im),(0,55));d=ImageDraw.Draw(tile);d.text((6,5),f'f{f:03d} {"image fit" if a["observed_mask"][f] else "uncertain temporal pose"} IoU={diags[f]["visible_silhouette_iou"]:.3f}',font=font,fill='black');d.text((6,28),'green=input mask; magenta=full mesh projection',font=font,fill='black');tiles.append(tile)
for k in range(3):
 sheet=Image.new('RGB',(1280,1070),'white')
 for j,im in enumerate(tiles[k*4:k*4+4]):sheet.paste(im,((j%2)*640,(j//2)*535))
 sheet.save(O/f'input_overlay_{k+1}.png')
print(json.dumps(run,indent=2),flush=True)
