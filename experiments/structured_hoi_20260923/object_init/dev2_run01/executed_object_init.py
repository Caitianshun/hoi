#!/usr/bin/env python3
"""Input-only rigid known-mesh initialization; generic source CLI. No published poses/meshes or held-out cameras.
Canonical scan geometry is explicitly admitted by the structured-input protocol.
Output x_world = canonical @ R_world[t].T + t_world_m[t]. CPU-only.
"""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,itertools,datetime
import numpy as np
import cv2,trimesh
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates,distance_transform_edt
ROOT=Path('/home/cai_tianshun/Project/HOI'); BASE=ROOT/'experiments/mosca_baseline_20260922'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def surface_sample(v,f,n,seed):
 rng=np.random.default_rng(seed);tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=-1);ids=rng.choice(len(f),n,p=area/area.sum());uv=rng.random((n,2));swap=uv.sum(1)>1;uv[swap]=1-uv[swap];pts=tri[ids,0]+uv[:,:1]*(tri[ids,1]-tri[ids,0])+uv[:,1:]*(tri[ids,2]-tri[ids,0]);return pts,cross[ids]/area[ids,None],ids,uv

def project(x,K):h=x@K.T;return h[:,:2]/h[:,2:]
def pix_to_xyz(uv,z,K):return np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T*z[:,None]
def kabsch(a,b):
 ac=a-a.mean(0);bc=b-b.mean(0);u,s,v=np.linalg.svd(ac.T@bc);r=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T;return r,b.mean(0)-a.mean(0)@r.T

def ray_mesh_canonical(uv,R,t,V,F,K):
 # Front-most actual triangle hit, no nearest-GT lookup.
 rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T
 cam=V@R.T+t;tri=cam[F];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
 out=[];valid=[]
 for ray in rays:
  h=np.cross(np.broadcast_to(ray,e2.shape),e2);det=np.sum(e1*h,axis=1);ok=np.abs(det)>1e-9;inv=np.divide(1.,det,out=np.zeros_like(det),where=ok);sv=-tri[:,0];u=inv*np.sum(sv*h,axis=1);q=np.cross(sv,e1);vv=inv*(q@ray);tt=inv*np.sum(e2*q,axis=1);good=ok&(u>=0)&(vv>=0)&(u+vv<=1)&(tt>0);tt[~good]=np.inf;i=int(tt.argmin());hit=np.isfinite(tt[i]);valid.append(hit);out.append((ray*tt[i]-t)@R if hit else np.zeros(3))
 return np.asarray(out),np.asarray(valid)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--input-dir',type=Path,required=True);ap.add_argument('--segmentation',type=Path);ap.add_argument('--tracks',type=Path);ap.add_argument('--template',type=Path);ap.add_argument('--template-manifest',type=Path);ap.add_argument('--silhouette-mode',choices=['convex','surface'],default='surface');ap.add_argument('--max-frames',type=int,default=1000000);args=ap.parse_args();B=args.input_dir.resolve();O=args.output.resolve();O.mkdir(parents=True,exist_ok=False);T=time.perf_counter()
 manifest=B/'input_manifest.json';segpath=(args.segmentation or B/'segmentation/segmentation.npz').resolve();trkpath=(args.tracks or B/'uniform_cotracker_tap.npz').resolve();scanpath=(args.template or B/'object_template_geometry.npz').resolve();scanprov=(args.template_manifest or B/'object_template_manifest.json').resolve()
 m=json.loads(manifest.read_text());K=np.asarray(m['K']);c2w=np.asarray(m['c2w']);times=np.asarray(m['timestamp_seconds'])[:args.max_frames];L=len(times)
 config={'source_camera':0,'shape_only_scan_exception':True,'published_pose_used':False,'heldout_used':False,'estimated_depth_is_GT':False,'seed':230923,'sample_count':4096,'fitting_surface_samples':900,'observation_pixels_per_frame':192,'mask_erosion_pixels':1,'minimum_object_area':300,'minimum_valid_depth_pixels':100,'depth_observation_independent_from_2d':True,'depth_sigma_m':0.10,'silhouette_sigma_pixels':2.0,'track_sigma_pixels':4.0,'robust_loss':'soft_l1','max_nfev':45,'temporal_rotation_sigma_rad_per_second':4.0,'temporal_translation_sigma_m_per_second':1.2,'weak_prior_weight':0.08,'version':'v4_general_mesh_cli','silhouette_mode':args.silhouette_mode,'silhouette_policy':'convex only for predeclared closed convex box; surface for nonconvex/open chair mesh based on template topology, not reference performance','minimum_depth_inlier_fraction':0.5,'unobserved_handling':'offline true-time interpolation/Slerp between measured reliable endpoints; endpoints hold; not a new observation'}
 write_json(O/'config.json',config);run={'status':'running','start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_used':False,'pid':os.getpid(),'code_sha256':sha(__file__),'protocol':'structured_known_scan_monocular_input_only','input_hashes':{str(p):sha(p) for p in [manifest,segpath,trkpath,scanpath,scanprov]}}
 write_json(O/'run.json',run)
 scanmeta=json.loads(scanprov.read_text());assert scanmeta.get('colors_or_texture_included',False) is False;assert scanmeta.get('no_per_frame_pose',False) is True;assert scanmeta['sha256']==sha(scanpath)
 if scanpath.suffix=='.npz':
  scan=np.load(scanpath);assert set(scan.files)=={'vertices','faces'};orig=np.asarray(scan['vertices'],dtype=np.float64);F=np.asarray(scan['faces'],dtype=np.int32)
 else:
  mesh=trimesh.load(scanpath,process=False);orig=np.asarray(mesh.vertices,dtype=np.float64);F=np.asarray(mesh.faces,dtype=np.int32)
 origin=orig.mean(0);V=orig-origin;samples,normals,fids,barys=surface_sample(V,F,4096,230923);S=samples[:900];treeS=cKDTree(S)
 # Strip all scan colors/materials; canonical centering recorded, not fitted to a reference.
 trimesh.Trimesh(vertices=V,faces=F,process=False).export(O/'canonical_geometry_only.ply')
 write_json(O/'template_input_manifest.json',{'role':'explicit_known_instance_shape_input_for_new_structured_protocol_only','source':str(scanpath),'source_sha256':sha(scanpath),'source_record':json.loads(scanprov.read_text()),'geometry_only_no_texture':True,'vertex_colors_ignored':True,'canonical_center_in_original_scan_m':origin.tolist(),'canonical_rule':'subtract arithmetic mean of original scan vertices; no pose/reference data','unit':'meters (BEHAVE scan convention)','old_evaluation_directory_unchanged':True})
 labels=np.load(segpath)['entity_labels'][:L];tr=np.load(trkpath);tracks=tr['tracks'][:L];visibility=tr['visibility'][:L];queries=tr['query_points'];qf=queries[:,0].astype(int);quv=np.rint(queries[:,1:]).astype(int);qent=np.load(segpath)['entity_labels'][qf,quv[:,1],quv[:,0]];object_query=qent==2
 deps=[];depth_hash={};rawmed=[];masks=[]
 for i in range(L):
  p=B/f'unidepth_depth/{i:05d}.npz';d=np.load(p)['dep'];deps.append(d);depth_hash[str(p)]=sha(p);mask=cv2.erode((labels[i]==2).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);masks.append(mask);vals=d[mask];rawmed.append(float(np.median(vals)) if len(vals) else np.nan)
 center=np.nanmedian(rawmed);mad=np.nanmedian(np.abs(np.asarray(rawmed)-center));depth_range=(max(.05,center-max(.6,4*1.4826*mad)),center+max(.6,4*1.4826*mad));obs=[]
 for i,(d,mask) in enumerate(zip(deps,masks)):
  good=mask&np.isfinite(d)&(d>depth_range[0])&(d<depth_range[1]);vu=np.argwhere(good);z=d[good]
  if len(z):
   med=np.median(z);spread=np.median(np.abs(z-med));sel=np.abs(z-med)<=max(.20,4*1.4826*spread);vu=vu[sel];z=z[sel]
  valid_count=len(z);rawarea=int((labels[i]==2).sum());observed=rawarea>=300
  depth_frame_reliable=(valid_count>=100 and valid_count/max(1,int(mask.sum()))>=0.5 and np.isfinite(rawmed[i]) and depth_range[0]<rawmed[i]<depth_range[1])
  if not depth_frame_reliable:vu=vu[:0];z=z[:0]
  ids=np.linspace(0,max(0,len(z)-1),min(192,len(z)),dtype=int);uv=vu[ids,::-1].astype(float);xyz=pix_to_xyz(uv,z[ids],K) if len(z) else np.empty((0,3))
  maskuv=np.argwhere(labels[i]==2)[:,::-1].astype(float);mi=np.linspace(0,max(0,len(maskuv)-1),min(192,len(maskuv)),dtype=int);maskuv=maskuv[mi]
  # Model silhouette is permitted behind human pixels, never forced to make occluded object visible.
  permitted=(labels[i]==1)|(labels[i]==2);dist=distance_transform_edt(~permitted)
  tu=np.rint(tracks[i]).astype(int);tv=visibility[i]&object_query&(tu[:,0]>=0)&(tu[:,0]<640)&(tu[:,1]>=0)&(tu[:,1]<480);ids=np.where(tv)[0];tv[ids]&=mask[tu[ids,1],tu[ids,0]]
  ids=np.where(tv)[0];td=np.full(len(tv),np.nan);td[ids]=d[tu[ids,1],tu[ids,0]];tg=tv&np.isfinite(td)&(td>depth_range[0])&(td<depth_range[1])&depth_frame_reliable;tx=np.full((len(tv),3),np.nan);idx=np.where(tg)[0];tx[idx]=pix_to_xyz(tracks[i,idx],td[idx],K)
  obs.append(dict(xyz=xyz,uv=maskuv,dist=dist,area=rawarea,valid_depth_pixels=valid_count,depth_frame_reliable=depth_frame_reliable,observed=observed,track_visible=tv,track_xyz=tx,track_depth_valid=tg))
 run['depth_input_hashes']=depth_hash;run['depth_gate_global_m']=depth_range;run['frame_image_hashes']={str(Path(m['frame_paths'][i])):sha(m['frame_paths'][i]) for i in range(L)};write_json(O/'run.json',run)
 ev,EA=np.linalg.eigh(V.T@V);EA=EA[:,::-1];EA[:,-1]*=np.linalg.det(EA)
 perms=[]
 for perm in itertools.permutations(range(3)):
  for signs in itertools.product([-1,1],repeat=3):
   P=np.eye(3)[:,perm]*signs
   if np.linalg.det(P)>0:perms.append(P)
 Rs=np.zeros((L,3,3));ts=np.zeros((L,3));fitstats=[];last=None
 def residual(p,ob,prior,track_pair):
  R=Rotation.from_rotvec(p[:3]).as_matrix();t=p[3:];cam=S@R.T+t;puv=project(cam,K);parts=[]
  # 3D nearest surface distance is softly weighted because monocular depth can be biased.
  if len(ob['xyz']):
   local=(ob['xyz']-t)@R;dd=treeS.query(local)[0];parts.append(dd/.10*.5)
  # Open/nonconvex geometry retains holes through actual projected surface samples.
  if args.silhouette_mode=='surface':
   alluv=project(samples@R.T+t,K);parts.append(cKDTree(alluv).query(ob['uv'])[0]/2.)
   di=map_coordinates(ob['dist'],[alluv[:,1],alluv[:,0]],order=1,mode='constant',cval=100);parts.append(np.quantile(di,np.linspace(0,1,128))/2.)
  # Convex silhouette is retained solely for the declared box case.
  else:
   try:
    hull=ConvexHull(puv);eq=hull.equations;outside=(ob['uv']@eq[:,:2].T+eq[:,2]).max(1);parts.append(np.maximum(outside,0)/2.)
    poly=puv[hull.vertices];b=[]
    for a0,a1 in zip(poly,np.roll(poly,-1,axis=0)):
     count=max(2,int(np.linalg.norm(a1-a0)/4));b.append(a0+(a1-a0)*np.linspace(0,1,count)[:,None])
    # Variable hull boundary count is summarized by fixed 64 quantiles for stable residual shape.
    buv=np.concatenate(b);di=map_coordinates(ob['dist'],[buv[:,1],buv[:,0]],order=1,mode='constant',cval=100);parts.append(np.quantile(di,np.linspace(0,1,64))/2.)
   except Exception:parts.extend([np.full(len(ob['uv']),100.),np.full(64,100.)])
  if track_pair is not None:
   ca,tu=track_pair;pred=project(ca@R.T+t,K);parts.append(((pred-tu)/4.).reshape(-1))
  if prior is not None:
   pr,pt,dt=prior;parts.append(Rotation.from_matrix(R@pr.T).as_rotvec()/max(.15,4*dt)*.08);parts.append((t-pt)/max(.04,1.2*dt)*.08)
  parts.append(np.array([max(.15-cam[:,2].min(),0)*100]))
  return np.concatenate(parts)
 def fit(R,t,ob,prior=None,track_pair=None,nfev=45):
  p=np.r_[Rotation.from_matrix(R).as_rotvec(),t];res=least_squares(residual,p,args=(ob,prior,track_pair),loss='soft_l1',f_scale=1.,max_nfev=nfev,diff_step=1e-4,ftol=2e-4,xtol=2e-4,gtol=2e-4);return Rotation.from_rotvec(res.x[:3]).as_matrix(),res.x[3:],float(res.cost/len(res.fun))
 for i,ob in enumerate(obs):
  if not ob['observed']:
   if last is not None:Rs[i]=Rs[last];ts[i]=ts[last]
   else:Rs[i]=np.eye(3);ts[i]=[0,0,center]
   fitstats.append({'frame':i,'observed':False,'object_area':ob['area'],'valid_depth_pixels':ob['valid_depth_pixels'],'state':'defer_true_time_interpolation','track_pairs':0});continue
  xyz=ob['xyz'];xyzmean=np.median(xyz,axis=0) if len(xyz) else None;prior=None;pair=None;candidates=[]
  if last is not None:
   pr,pt=Rs[last],ts[last];dt=times[i]-times[last];prior=(pr,pt,dt);candidates.append((pr.copy(),pt.copy()))
   common=obs[last]['track_visible']&ob['track_visible'];ids=np.where(common)[0]
   if len(ids)>=6:
    ids=ids[np.linspace(0,len(ids)-1,min(len(ids),120),dtype=int)];ca,hits=ray_mesh_canonical(tracks[last,ids],pr,pt,V,F,K);ids=ids[hits];ca=ca[hits]
    if len(ids)>=6:
     pair=(ca,tracks[i,ids]);ok,rvec,tvec,inl=cv2.solvePnPRansac(ca.astype(np.float64),tracks[i,ids].astype(np.float64),K,None,iterationsCount=60,reprojectionError=5.,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
     if ok and tvec[2,0]>.2:candidates.insert(0,(cv2.Rodrigues(rvec)[0],tvec[:,0]))
   both=obs[last]['track_depth_valid']&ob['track_depth_valid'];idx=np.where(both)[0]
   if len(idx)>=6:
    aa=obs[last]['track_xyz'][idx];bb=ob['track_xyz'][idx];rr,tt=kabsch(aa,bb)
    for it in range(3):
     er=np.linalg.norm(aa@rr.T+tt-bb,axis=1);keep=er<=max(.025,np.quantile(er,.7));rr,tt=kabsch(aa[keep],bb[keep])
    if np.median(er)<.15:candidates.append((rr@pr,pt@rr.T+tt))
  # Multiple axes/signs keep first/reappearing poses from depending on one arbitrary plane normal.
  if (last is None or i-last>2) and len(xyz)>=100:
   xx=xyz-xyzmean;_,E=np.linalg.eigh(xx.T@xx);E=E[:,::-1];E[:,-1]*=np.linalg.det(E)
   for P in perms:
    r=E@P@EA.T;t=xyzmean.copy();t[2]+=.07;candidates.append((r,t))
  elif last is not None:
   # Missing/unreliable depth never disables valid 2D mask/track observations.
   rr=prior[0]
   if xyzmean is not None:
    visS=S@rr.T;candidates.append((rr,xyzmean-np.median(visS,axis=0)+np.array([0,0,.07])))
   else:
    uvc=ob['uv'].mean(0);guess=pix_to_xyz(uvc[None],np.array([prior[1][2]]),K)[0];candidates.append((rr,guess))
  # Coarse candidate screening uses same allowed observations, never evaluation references.
  costs=[np.mean(np.minimum(residual(np.r_[Rotation.from_matrix(r).as_rotvec(),t],ob,prior,pair)**2,25)) for r,t in candidates]
  order=np.argsort(costs)[:(6 if last is None or i-last>2 else 3)];out=[]
  for j in order:out.append(fit(*candidates[j],ob,prior,pair,nfev=45))
  out.sort(key=lambda x:x[2]);r,t,c=out[0];Rs[i]=r;ts[i]=t;last=i
  gap=float(out[1][2]-c) if len(out)>1 else float('nan')
  fitstats.append({'frame':i,'observed':True,'object_area':ob['area'],'valid_depth_pixels':ob['valid_depth_pixels'],'state':'input_fitted','cost':c,'alternative_cost_gap':gap,'track_pairs':0 if pair is None else len(pair[0]),'candidates_optimized':len(out)})
  if i%10==0 or i==L-1:print(f'frame {i}/{L} cost={c:.4f} track_pairs={fitstats[-1]["track_pairs"]} elapsed={time.perf_counter()-T:.1f}s',flush=True)
  np.savez_compressed(O/'progress.npz',R_camera=Rs,t_camera_m=ts,last_completed_frame=i)
 # Missing / too-small observation intervals are interpolated only as uncertain initialization.
 observed=np.asarray([o['observed'] for o in obs]);valid=np.where(observed)[0];assert len(valid)>=2
 for k in range(3):ts[:,k]=np.interp(times,times[valid],ts[valid,k])
 sl=Slerp(times[valid],Rotation.from_matrix(Rs[valid]));Rs=sl(np.clip(times,times[valid[0]],times[valid[-1]])).as_matrix()
 # Metrics against training-input masks only. Full silhouette ignored where human occludes.
 diag=[];silhouettes=[]
 for i,(r,t,ob) in enumerate(zip(Rs,ts,obs)):
  cam=V@r.T+t;uv=project(cam,K);raster=np.zeros((480,640),np.uint8)
  if args.silhouette_mode=='surface':
   for triuv in np.rint(uv[F]).astype(np.int32):cv2.fillConvexPoly(raster,triuv,1)
  else:cv2.fillConvexPoly(raster,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
  pred=(raster>0)&(labels[i]!=1);target=labels[i]==2;inter=(pred&target).sum();union=(pred|target).sum();iou=float(inter/union) if union else 0.;coverage=float(inter/max(1,target.sum()));leak=float((pred&~target).sum()/max(1,pred.sum()));silhouettes.append(raster)
  if ob['observed']:fitstats[i].update(mask_visible_iou=iou,object_pixel_coverage=coverage,visible_background_fraction=leak)
  diag.append({'frame':i,'timestamp_seconds':float(times[i]),'observed':bool(observed[i]),'object_pixels':int(target.sum()),'visible_silhouette_iou':iou,'observed_object_pixel_coverage':coverage,'predicted_visible_background_fraction':leak})
 Rw=np.einsum('ij,tjk->tik',c2w[:3,:3],Rs);tw=ts@c2w[:3,:3].T+c2w[:3,3]
 # Reliability is transparent heuristic initialization weight, not a calibrated probability.
 rel=np.asarray([min(1,o['area']/1000)*d['visible_silhouette_iou'] if obv else 0. for o,d,obv in zip(obs,diag,observed)])
 np.savez_compressed(O/'object_init.npz',canonical_vertices_m=V.astype(np.float32),faces=F,centres_m=samples.astype(np.float32),canonical_normals=normals.astype(np.float32),sample_face_ids=fids,sample_barycentric_uv=barys,R_world=Rw.astype(np.float32),t_world_m=tw.astype(np.float32),R_camera=Rs.astype(np.float32),t_camera_m=ts.astype(np.float32),timestamp_seconds=times,dt_seconds=np.r_[np.nan,np.diff(times)],reliability=rel.astype(np.float32),observed_mask=observed,depth_observed_mask=np.asarray([o['depth_frame_reliable'] for o in obs]),rotation_ambiguity=np.ones(L,bool),original_scan_center_m=origin,K=K,c2w=c2w,source_video_frame_indices=np.asarray(m['frame_indices'])[:L])
 np.savez_compressed(O/'input_diagnostics.npz',raw_object_depth_median=rawmed,depth_gate_global_m=depth_range,object_mask_pixels=np.asarray([o['area'] for o in obs]),valid_depth_pixels=np.asarray([o['valid_depth_pixels'] for o in obs]),visible_silhouette_iou=np.asarray([d['visible_silhouette_iou'] for d in diag]),input_observed=observed,rotation_increment_rad=Rotation.from_matrix(np.einsum('tij,tkj->tik',Rs[1:],Rs[:-1])).magnitude(),translation_speed_m_s=np.linalg.norm(np.diff(ts,axis=0),axis=1)/np.diff(times))
 write_json(O/'frame_diagnostics.json',diag);write_json(O/'fit_diagnostics.json',fitstats)
 # Verify exact common rigid transformation, determinant and pair distance invariance.
 probe=samples[:50];base_dist=np.linalg.norm(probe[1:]-probe[0],axis=1);err=[]
 for r,t in zip(Rw,tw):ww=probe@r.T+t;err.append(np.max(np.abs(np.linalg.norm(ww[1:]-ww[0],axis=1)-base_dist)))
 (O/'executed_object_init.py').write_bytes(Path(__file__).read_bytes())
 run.update(status='completed',wall_seconds=time.perf_counter()-T,observed_frames=int(observed.sum()),unobserved_or_too_small_frames=np.where(~observed)[0].tolist(),rotation_uniquely_identified=False,rotation_ambiguity_reason='texture-free scan and partial/occluded observations do not establish a unique pose; canonical labels follow selected continuous hypothesis',R_determinant_minmax=[float(np.linalg.det(Rw).min()),float(np.linalg.det(Rw).max())],max_pairwise_rigidity_error_m=float(max(err)),mask_metrics_are_training_input_fit=True,output_sha256=sha(O/'object_init.npz'))
 write_json(O/'run.json',run);print(json.dumps({k:run[k] for k in ['status','wall_seconds','observed_frames','unobserved_or_too_small_frames','max_pairwise_rigidity_error_m']}),flush=True)
 # Fixed input-only illustrative frames, including both missing-mask intervals.
 from PIL import Image,ImageDraw,ImageFont
 ff=np.unique(np.linspace(0,L-1,12,dtype=int)).tolist();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);tiles=[]
 for f in ff:
  im=np.asarray(Image.open(m['frame_paths'][f]).convert('RGB')).copy();gt=(labels[f]==2).astype(np.uint8);pr=silhouettes[f];ct,_=cv2.findContours(gt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,ct,-1,(0,255,70),1);ct,_=cv2.findContours(pr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,ct,-1,(255,40,220),1);crop=Image.fromarray(im).crop((0,0,640,480));tile=Image.new('RGB',(640,535),'white');tile.paste(crop,(0,55));d=ImageDraw.Draw(tile);d.text((6,5),f'f{f:03d} {"fit" if observed[f] else "uncertain interpolation"} IoU={diag[f]["visible_silhouette_iou"]:.3f}',font=font,fill='black');d.text((6,27),'green=input mask; magenta=full projected shape',font=font,fill='black');tiles.append(tile)
 for k in range((len(tiles)+3)//4):
  sheet=Image.new('RGB',(1280,1070),'white')
  for j,im in enumerate(tiles[k*4:k*4+4]):sheet.paste(im,((j%2)*640,(j//2)*535))
  sheet.save(O/f'input_overlay_{k+1}.png')
if __name__=='__main__': main()
