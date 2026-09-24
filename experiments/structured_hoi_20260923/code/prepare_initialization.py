"""Build one frozen, legal-input initialization shared by S0/S1."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import torch,cv2
from scipy.spatial import cKDTree

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()

def surface_colors(xyz_frames,rgb,labels,entity,K,C):
    n=xyz_frames.shape[1];total=np.zeros((n,3));cnt=np.zeros(n)
    for t in range(len(rgb)):
        xyz=(xyz_frames[t]-C[:3,3])@C[:3,:3];h=xyz@K.T;uv=np.rint(h[:,:2]/h[:,2:]).astype(int)
        okay=(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)
        ii=np.flatnonzero(okay);pixel=uv[ii,1]*640+uv[ii,0];z=np.full(640*480,np.inf)
        np.minimum.at(z,pixel,xyz[ii,2]);ok=(xyz[ii,2]-z[pixel]<.025)&(labels[t,uv[ii,1],uv[ii,0]]==entity)
        ii=ii[ok];total[ii]+=rgb[t,uv[ii,1],uv[ii,0]]/255.;cnt[ii]+=1
    color=np.full((n,3),.5);valid=cnt>0;color[valid]=total[valid]/cnt[valid,None]
    return color.astype(np.float32),cnt.astype(np.int32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--segmentation',type=Path,required=True)
    p.add_argument('--human-geometry',type=Path,required=True);p.add_argument('--object-init',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    m=json.loads((a.input/'input_manifest.json').read_text());assert m['role']=='input_only'
    for path,digest in zip(m['frame_paths'],m['frame_sha256']):assert sha(path)==digest
    rgb=np.stack([cv2.imread(f)[...,::-1] for f in m['frame_paths']]);labels=np.load(a.segmentation)['entity_labels']
    H,W=rgb.shape[1:3];assert (H,W)==(480,640)
    human=np.load(a.human_geometry);obj=np.load(a.object_init)
    assert np.array_equal(m['timestamp_seconds'],human['timestamp_seconds'])
    assert np.allclose(m['timestamp_seconds'],obj['timestamp_seconds'],atol=1e-7,rtol=0)
    K=np.array(m['K']);C=np.array(m['c2w']);rng=np.random.default_rng(12345)
    anchors=obj['centres_m'];R=obj['R_world'];translation=obj['t_world_m']
    object_world=np.einsum('tij,nj->tni',R,anchors)+translation[:,None]
    hc,hcount=surface_colors(human['vertices_world_m'],rgb,labels,1,K,C)
    oc,ocount=surface_colors(object_world,rgb,labels,2,K,C)
    # Static background receives only pixels seen as background in legal inputs.
    selected=np.unique(np.linspace(0,len(rgb)-1,min(16,len(rgb))).astype(int));depth=[];bgcols=[]
    for t in selected:
        d=np.load(a.input/'unidepth_depth'/f'{t:05d}.npz')['dep'].copy();d[labels[t]!=0]=np.nan;depth.append(d)
        c=rgb[t].astype(float)/255.;c[labels[t]!=0]=np.nan;bgcols.append(c)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);bgdep=np.nanmedian(depth,axis=0);bgcolor=np.nanmedian(bgcols,axis=0)
    okay=np.isfinite(bgdep)&(bgdep>.1)&(bgdep<30)&np.isfinite(bgcolor).all(-1)
    flat=np.flatnonzero(okay);flat=flat[rng.permutation(len(flat))[:60000]]
    uv=np.stack([flat%W,flat//W],-1);dep=bgdep.flat[flat]
    cam=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T*dep[:,None];bg=cam@C[:3,:3].T+C[:3,3]
    hs=cKDTree(human['vertices_camera_m'][0]).query(human['vertices_camera_m'][0],k=2)[0][:,1]*.7
    os=cKDTree(anchors).query(anchors,k=2)[0][:,1]*.7
    bs=dep/K[0,0]*1.3
    data=dict(schema='structured_gaussians_init_v1',input_dir=str(a.input.resolve()),segmentation=str(a.segmentation.resolve()),
        human_geometry=str(a.human_geometry.resolve()),object_init=str(a.object_init.resolve()),
        smplx_model='/home/cai_tianshun/Project/mml/smpl_model/smplx/SMPLX_NEUTRAL.npz',
        c2w=m['c2w'],K=m['K'],timestamps=m['timestamp_seconds'],frame_indices=m['frame_indices'],
        human_colors=hc,human_color_observation_count=hcount,human_scale=np.clip(hs,.0025,.025).astype(np.float32),
        object_anchors=anchors.astype(np.float32),object_R=R.astype(np.float32),object_t=translation.astype(np.float32),
        object_colors=oc,object_color_observation_count=ocount,object_scale=np.clip(os,.0025,.025).astype(np.float32),
        background_anchors=bg.astype(np.float32),background_colors=bgcolor.reshape(-1,3)[flat].astype(np.float32),
        background_scale=np.clip(bs,.003,.1).astype(np.float32),world_scale=1.,seed=12345,reference_used=False)
    torch.save(data,a.output/'initialization.pt')
    record=dict(status='completed',reference_used=False,wall_seconds=time.perf_counter()-start,
        points=dict(human=len(hc),object=len(oc),background=len(bg)),colors_unobserved_default_gray=dict(human=int((hcount==0).sum()),object=int((ocount==0).sum())),
        inputs={str(p.resolve()):sha(p) for p in [a.input/'input_manifest.json',a.segmentation,a.human_geometry,a.object_init]},
        depth_hashes={str(a.input/'unidepth_depth'/f'{t:05d}.npz'):sha(a.input/'unidepth_depth'/f'{t:05d}.npz') for t in selected},
        initialization_sha256=sha(a.output/'initialization.pt'),script_sha256=sha(__file__),
        color_rule='All legal RGB frames; same-entity mask, positive depth and local projected-point z visibility; observed colors averaged, unseen=.5. No scan texture.',
        background_rule='Median legal input background over fixed16times, random60000 eligiblepixels, fixedknownK; no frame-fitted model.')
    (a.output/'initialization.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
