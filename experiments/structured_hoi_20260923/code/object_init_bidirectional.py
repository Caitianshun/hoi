#!/usr/bin/env python3
"""Select ordinary forward/backward rigid initializations using allowed masks and time.
No reference, RGB texture, or held-out view; candidates share exact canonical geometry.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np,cv2
from scipy.spatial.transform import Rotation,Slerp
from PIL import Image,ImageDraw,ImageFont
ap=argparse.ArgumentParser();ap.add_argument('--forward',type=Path,required=True);ap.add_argument('--reverse',type=Path,required=True);ap.add_argument('--input-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();T=time.perf_counter();O=args.output.resolve();O.mkdir(exist_ok=False,parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
runs=[args.forward.resolve(),args.reverse.resolve()];runsj=[json.loads((p/'run.json').read_text()) for p in runs];assert all(r['status']=='completed' for r in runsj);arrays=[dict(np.load(p/'object_init.npz')) for p in runs];diagnostics=[json.loads((p/'frame_diagnostics.json').read_text()) for p in runs]
for k in ['canonical_vertices_m','faces','centres_m','timestamp_seconds','observed_mask']:assert np.array_equal(arrays[0][k],arrays[1][k])
a=arrays[0];L=len(a['timestamp_seconds']);ts=a['timestamp_seconds'];valid=np.where(a['observed_mask'])[0];rots=np.stack([x['R_camera'].astype(float) for x in arrays]);trans=np.stack([x['t_camera_m'].astype(float) for x in arrays]);ious=np.array([[d['visible_silhouette_iou'] for d in ds] for ds in diagnostics]);cost=1-ious[:,valid].T
# Fixed ordinary smooth path: topology-preserving candidate selection, not a reference metric.
# Rate penalties scaled by actual dt. No forced symmetry equivalence / axis relabeling.
w_t=.03;w_r=.02;dp=np.zeros((len(valid),2));par=np.zeros((len(valid),2),int);dp[0]=cost[0]
for k in range(1,len(valid)):
 f0,f1=valid[k-1:k+1];dt=ts[f1]-ts[f0]
 for j in range(2):
  transcost=[]
  for i in range(2):
   speed=np.linalg.norm(trans[j,f1]-trans[i,f0])/dt;ang=Rotation.from_matrix(rots[j,f1]@rots[i,f0].T).magnitude()/dt;transcost.append(dp[k-1,i]+w_t*speed+w_r*ang)
  par[k,j]=np.argmin(transcost);dp[k,j]=cost[k,j]+min(transcost)
sel=np.zeros(len(valid),int);sel[-1]=np.argmin(dp[-1])
for k in range(len(valid)-1,0,-1):sel[k-1]=par[k,sel[k]]
R=rots[sel,valid];t=trans[sel,valid];allR=Slerp(ts[valid],Rotation.from_matrix(R))(np.clip(ts,ts[valid[0]],ts[valid[-1]])).as_matrix();allt=np.stack([np.interp(ts,ts[valid],t[:,k]) for k in range(3)],1);a['R_camera']=allR.astype(np.float32);a['t_camera_m']=allt.astype(np.float32);c=a['c2w'];a['R_world']=np.einsum('ij,tjk->tik',c[:3,:3],allR).astype(np.float32);a['t_world_m']=(allt@c[:3,:3].T+c[:3,3]).astype(np.float32);source=np.full(L,-1);source[valid]=sel;a['selected_direction']=source
B=args.input_dir.resolve();m=json.loads((B/'input_manifest.json').read_text());labels=np.load(B/'segmentation/segmentation.npz')['entity_labels'];V=a['canonical_vertices_m'];F=a['faces'];K=a['K'];diags=[];masks=[]
for f in range(L):
 p=V@allR[f].T+allt[f];uv=p@K.T;uv=uv[:,:2]/uv[:,2:];mask=np.zeros((480,640),np.uint8)
 for tri in np.rint(uv[F]).astype(np.int32):cv2.fillConvexPoly(mask,tri,1)
 pred=mask.astype(bool)&(labels[f]!=1);target=labels[f]==2;inter=(pred&target).sum();union=(pred|target).sum();iou=float(inter/max(1,union));rel=min(1,target.sum()/1000)*iou if a['observed_mask'][f] else 0.;a['reliability'][f]=rel;masks.append(mask);diags.append({'frame':f,'timestamp_seconds':float(ts[f]),'observed':bool(a['observed_mask'][f]),'selected_direction':int(source[f]),'object_pixels':int(target.sum()),'visible_silhouette_iou':iou,'rotation_uniquely_identified':False})
np.savez_compressed(O/'object_init.npz',**a);np.savez_compressed(O/'input_diagnostics.npz',visible_silhouette_iou=[d['visible_silhouette_iou'] for d in diags],input_observed=a['observed_mask'],translation_speed_m_s=np.linalg.norm(np.diff(allt,axis=0),axis=1)/np.diff(ts),rotation_increment_rad=Rotation.from_matrix(np.einsum('tij,tkj->tik',allR[1:],allR[:-1])).magnitude(),selected_direction=source)
(O/'frame_diagnostics.json').write_text(json.dumps(diags,indent=2));(O/'executed_selection.py').write_bytes(Path(__file__).read_bytes());(O/'run.json').write_text(json.dumps({'status':'completed','GPU_used':False,'role':'input-only bidirectional rigid initialization','ordinary_initializer_not_S2_S3':True,'source_runs':[{str(p):sha(p/'object_init.npz')} for p in runs],'code_sha256':sha(__file__),'emission':'1 - input mask visible mesh silhouette IoU','translation_rate_weight':w_t,'rotation_rate_weight':w_r,'times':'real dt','unobserved_frames':'true-time interpolation, zero reliability, no measurement','selected_forward_frames':int((sel==0).sum()),'selected_reverse_frames':int((sel==1).sum()),'switches_at_input_frames':valid[1:][sel[1:]!=sel[:-1]].tolist(),'wall_seconds':time.perf_counter()-T,'source_fitting_seconds':sum(r['wall_seconds'] for r in runsj),'output_sha256':sha(O/'object_init.npz')},indent=2))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);frames=np.unique(np.linspace(0,L-1,12,dtype=int));tiles=[]
for f in frames:
 im=np.array(Image.open(m['frame_paths'][f]).convert('RGB'));ct,_=cv2.findContours((labels[f]==2).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,ct,-1,(0,255,70),1);ct,_=cv2.findContours(masks[f],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,ct,-1,(255,40,220),1);tile=Image.new('RGB',(640,535),'white');tile.paste(Image.fromarray(im),(0,55));d=ImageDraw.Draw(tile);d.text((6,5),f'f{f:03d} source={source[f]} IoU={diags[f]["visible_silhouette_iou"]:.3f} (input fit only)',font=font,fill='black');d.text((6,28),'green=input; magenta=projected mesh; -1=uncertain interpolation',font=font,fill='black');tiles.append(tile)
for k in range(3):
 sheet=Image.new('RGB',(1280,1070),'white')
 for j,im in enumerate(tiles[k*4:k*4+4]):sheet.paste(im,((j%2)*640,(j//2)*535))
 sheet.save(O/f'input_overlay_{k+1}.png')
print((O/'run.json').read_text())
