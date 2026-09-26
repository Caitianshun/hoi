"""Camera0-only mesh first-surface memory. No evaluation imports or paths."""
from pathlib import Path
import sys,json,time
import numpy as np,cv2,torch
from numba import njit
from scipy.spatial import cKDTree
R=Path('/home/cai_tianshun/Project/HOI');A=R/'experiments/aux_ref_object_reconstruction_20260924/run01';E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(A/'code'))
from aux_scene import AuxObjectScene,sha_file,state_identity

def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

@njit(cache=True)
def raster(uv,z,faces,H,W):
 depth=np.full((H,W),np.inf);face=np.full((H,W),-1,np.int32);bary=np.zeros((H,W,3))
 for fi in range(len(faces)):
  a,b,c=faces[fi];x0,y0=uv[a];x1,y1=uv[b];x2,y2=uv[c];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
  if abs(den)<1e-10:continue
  for y in range(max(0,int(np.ceil(min(y0,y1,y2)))),min(H-1,int(np.floor(max(y0,y1,y2))))+1):
   for x in range(max(0,int(np.ceil(min(x0,x1,x2)))),min(W-1,int(np.floor(max(x0,x1,x2))))+1):
    u=((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/den;v=((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/den;w=1-u-v
    if min(u,v,w)<-1e-8:continue
    inv=u/z[a]+v/z[b]+w/z[c];zz=1/inv
    if zz<depth[y,x]:depth[y,x]=zz;face[y,x]=fi;bary[y,x,0]=u/z[a]/inv;bary[y,x,1]=v/z[b]/inv;bary[y,x,2]=w/z[c]/inv
 return depth,face,bary

def mesh_maps(vertices,faces,Rw,tw,K,C,H,W):
 cam=(vertices@Rw.T+tw-C[:3,3])@C[:3,:3]
 assert (cam[:,2]>.02).all()
 uv=cam@K.T;return raster(uv[:,:2]/uv[:,2:],cam[:,2],faces,H,W)

@njit(cache=True)
def closest(p,a,b,c):
 ab=b-a;ac=c-a;ap=p-a;d1=np.dot(ab,ap);d2=np.dot(ac,ap)
 if d1<=0 and d2<=0:return a
 bp=p-b;d3=np.dot(ab,bp);d4=np.dot(ac,bp)
 if d3>=0 and d4<=d3:return b
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:return a+d1/(d1-d3)*ab
 cp=p-c;d5=np.dot(ab,cp);d6=np.dot(ac,cp)
 if d6>=0 and d5<=d6:return c
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:return a+d2/(d2-d6)*ac
 va=d3*d6-d5*d4
 if va<=0 and d4-d3>=0 and d5-d6>=0:return b+(d4-d3)/((d4-d3)+(d5-d6))*(c-b)
 den=1/(va+vb+vc);return a+ab*vb*den+ac*vc*den

@njit(cache=True)
def associate(points,tri,attached):
 ids=np.full(len(points),-1,np.int32);dist=np.full(len(points),np.inf)
 for i in range(len(points)):
  best=1e20;second=1e20;fi=-1
  for j in range(len(tri)):
   q=closest(points[i],tri[j,0],tri[j,1],tri[j,2]);d=np.sum((q-points[i])**2)
   if d<best:second=best;best=d;fi=j
   elif d<second:second=d
  dist[i]=np.sqrt(best)
  # Conservative unique nearest inherited face, no edge/cross-face reassociation.
  if fi==attached[i] and best<=.005**2 and np.sqrt(second)-np.sqrt(best)>1e-7:ids[i]=fi
 return ids,dist

class SurfaceIndex:
 def __init__(self,arrays):
  self.a=arrays;self.trees={}
  for t in np.unique(arrays['time']):
   for f in np.unique(arrays['face'][arrays['time']==t]):
    ii=np.flatnonzero((arrays['time']==t)&(arrays['face']==f));self.trees[(int(t),int(f))]=(cKDTree(arrays['xyz'][ii]),ii)
 def query(self,points,faces,T,positive=True):
  index=np.full((len(points),T),-1,np.int64);distance=np.full((len(points),T),np.inf)
  for f in np.unique(faces[faces>=0]):
   qi=np.flatnonzero(faces==f)
   for t in range(T):
    pair=self.trees.get((t,int(f)))
    if pair is None:continue
    tree,ii=pair
    # Radius pool sorted by distance then pixel ID ensures deterministic ties.
    pools=tree.query_ball_point(points[qi],.005)
    for qid,pool in zip(qi,pools):
     ids=ii[pool]
     if positive:ids=ids[self.a['positive'][ids]]
     if not len(ids):continue
     dd=np.linalg.norm(self.a['xyz'][ids]-points[qid],axis=1);order=np.lexsort((self.a['pixel'][ids],dd));j=order[0];index[qid,t]=ids[j];distance[qid,t]=dd[j]
  return index,distance

def support_query(index,points,faces,T):
 pos,dist=index.query(points,faces,T);geo,_=index.query(points,faces,T,False)
 n=(pos>=0).sum(1);mapped=(faces>=0)&(geo>=0).any(1)
 # 0 unmapped, 1 no positive, 2 single, 3 supported
 state=np.where(mapped,np.where(n>=2,3,np.where(n==1,2,1)),0).astype(np.uint8)
 return pos,dist,n,state

def build(dev):
 start=time.perf_counter();out=E/'support'/dev;out.mkdir(exist_ok=False)
 meta=json.loads((A/'inputs'/dev/'input_manifest.json').read_text());T=len(meta['timestamp_seconds']);H,W=meta['height'],meta['width']
 sc=AuxObjectScene(dev,meta['timestamp_seconds'],device='cpu');ck=A/'runs'/f'{dev}_Ref/checkpoint_008000.pt';state=torch.load(ck,map_location='cpu',weights_only=False)['obank'];sc.obank.resize_for_load(state,'');sc.obank.load_state_dict(state);assert state_identity(sc.obank.state_dict())==state_identity(state)
 x=sc.object_canonical_export();np.savez_compressed(out/'B0_object.npz',**x)
 gp=Path(sc.initialization['object_init']);g=np.load(gp);v=g['canonical_vertices_m'].astype(float);faces=g['faces'];tri=v[faces];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-20)
 ref=np.load(meta['reference_object_motion']);assert sha_file(meta['reference_object_motion'])==meta['reference_motion_sha256'];labels=np.load(meta['segmentation'])['entity_labels'];assert sha_file(meta['segmentation'])==meta['segmentation_sha256']
 K=np.array(meta['K']);C=np.array(meta['c2w']);records=[];frame_rows=[]
 for t,p in enumerate(meta['frame_paths']):
  assert sha_file(p)==meta['frame_sha256'][t];rgb=cv2.imread(p)[...,::-1].astype(np.float32)/255;lab=labels[t];O=lab==2;inside=cv2.erode(O.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);dt=cv2.distanceTransform(O.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
  dep,fi,bc=mesh_maps(v,faces,ref['R_world'][t],ref['t_world'][t],K,C,H,W);valid=fi>=0;y,z=np.where(valid);fid=fi[y,z];b=bc[y,z];xyz=(tri[fid]*b[:,:,None]).sum(1)
  np.savez_compressed(out/f'map_{t:03d}.npz',depth=dep.astype(np.float32),face=fi,bary=bc.astype(np.float32))
  patch=[];pm=[]
  for dy in [-1,0,1]:
   for dx in [-1,0,1]:
    yy=y+dy;xx=z+dx;ok=(yy>=0)&(yy<H)&(xx>=0)&(xx<W);yy=yy.clip(0,H-1);xx=xx.clip(0,W-1);ok&=O[yy,xx];patch.append(rgb[yy,xx]*ok[:,None]);pm.append(ok)
  patch=np.stack(patch,1);pm=np.stack(pm,1);gx=(patch[:,5]-patch[:,3])*.5;gy=(patch[:,7]-patch[:,1])*.5;vx=pm[:,5]&pm[:,3];vy=pm[:,7]&pm[:,1];gx*=vx[:,None];gy*=vy[:,None]
  camera_canonical=(C[:3,3]-ref['t_world'][t])@ref['R_world'][t];direction=camera_canonical-xyz;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-20)
  # Exact pixel-centre colors: interpolation footprint is one valid interior pixel.
  records.append(dict(xyz=xyz.astype(np.float32),face=fid.astype(np.int32),bary=b.astype(np.float32),time=np.full(len(y),t,np.int16),pixel=(y*W+z).astype(np.int32),positive=inside[y,z],rgb=rgb[y,z],normal=norm[fid].astype(np.float32),view=direction.astype(np.float32),distance_px=dt[y,z],purity=pm.mean(1).astype(np.float32),resolution=(dep[y,z]/np.sqrt(K[0,0]*K[1,1])).astype(np.float32),descriptor=np.c_[patch.reshape(len(y),27),pm.astype(float),gx,gy,vx,vy].astype(np.float32)))
  frame_rows.append(dict(time=t,source=meta['timestamp_seconds'][t],O=int(O.sum()),O_interior=int(inside.sum()),surface=int(valid.sum()),positive=int((inside&valid).sum())))
 arrays={k:np.concatenate([row[k] for row in records]) for k in records[0]};np.savez_compressed(out/'observations.npz',**arrays)
 ix=SurfaceIndex(arrays);qfaces,qdist=associate(x['centres_canonical_m'].astype(float),tri,x['attachment_face_id'].astype(np.int64));ids,dd,n,st=support_query(ix,x['centres_canonical_m'],qfaces,T)
 np.savez_compressed(out/'queries.npz',candidate_ids=ids,distance=dd.astype(np.float32),face=qfaces,normal=np.where((qfaces>=0)[:,None],norm[qfaces.clip(0)],0).astype(np.float32),support_count=n,state=st,face_distance=qdist.astype(np.float32))
 # Original anchors separately, no learned center replacement.
 af=g['sample_face_ids'];aid,ad,an,ast=support_query(ix,g['centres_m'],af,T);np.savez_compressed(out/'anchors.npz',candidate_ids=aid,distance=ad.astype(np.float32),support_count=an,state=ast)
 manifest=dict(dev=dev,T=T,frames=frame_rows,source_manifest={'path':str(A/'inputs'/dev/'input_manifest.json'),'sha256':sha_file(A/'inputs'/dev/'input_manifest.json')},B0={'path':str(ck),'sha256':sha_file(ck)},geometry={'path':str(gp),'sha256':sha_file(gp)},code_sha256=sha_file(__file__),observations_sha256=sha_file(out/'observations.npz'),queries_sha256=sha_file(out/'queries.npz'),count=len(arrays['face']),positive=int(arrays['positive'].sum()),query_support={str(k):int((n==k).sum()) for k in range(T+1)},query_states={str(k):int((st==k).sum()) for k in range(4)},seconds=time.perf_counter()-start,GPU_seconds=0,definition='same face 5mm; distinct native times; unmapped includes absent local geometric evidence; proxy not material ground truth',source_reappearance='NA: mask transitions alone do not certify true occlusion')
 save(out/'manifest.json',manifest);print(json.dumps(manifest),flush=True)

if __name__=='__main__':
 torch.set_num_threads(2)
 for dev in ['dev1','dev2']:build(dev)
