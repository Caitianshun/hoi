"""CPU interface, mesh overlay and input-prior consistency checks; no fitted GT."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import cv2
from numba import njit
from PIL import Image, ImageDraw

ROOT=Path('/home/cai_tianshun/Project/HOI')
EXP=ROOT/'experiments/gvhmr_validation_20260923'
OUT=EXP/'geometry_quality'
FRAMES=[0,16,25,31,55,65,93,113]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p90=float(np.percentile(x,90)),max=float(x.max()))

@njit
def raster(uv,z,faces,H,W):
    """Pinhole z-buffer at integer pixel centres, perspective-correct depth."""
    dep=np.full((H,W),np.inf)
    for f in faces:
        a,b,c=f
        if min(z[a],z[b],z[c])<=0:continue
        x0,y0=uv[a];x1,y1=uv[b];x2,y2=uv[c]
        den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-10:continue
        for y in range(max(0,int(np.ceil(min(y0,y1,y2)))),min(H-1,int(np.floor(max(y0,y1,y2))))+1):
            for x in range(max(0,int(np.ceil(min(x0,x1,x2)))),min(W-1,int(np.floor(max(x0,x1,x2))))+1):
                u=((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/den
                v=((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/den
                w=1-u-v
                if min(u,v,w)>=-1e-8:
                    d=1/(u/z[a]+v/z[b]+w/z[c])
                    if d<dep[y,x]:dep[y,x]=d
    return dep

def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=True)
    run=json.loads((EXP/'run01/run.json').read_text());assert run['status']=='completed'
    for n,h in run['outputs'].items():assert sha(EXP/'run01'/n)==h
    inp=json.loads((ROOT/'experiments/mosca_baseline_20260922/common_input/input_manifest.json').read_text())
    raw=np.load(EXP/'run01/raw_geometry.npz');post=np.load(EXP/'run01/official_postproc_geometry.npz')
    crops=np.load(EXP/'run01/crop_geometry.npz');C=raw['c2w'];R=C[:3,:3];K=raw['K']
    assert np.array_equal(raw['timestamp_seconds'],inp['timestamp_seconds'])
    assert np.array_equal(raw['video_frame_index'],inp['frame_indices'])
    # Recheck actual crop transform from saved centre/scale, plus inverse consistency.
    crop_error=[];roundtrip=[]
    for (x,y,s),A in zip(crops['bbx_xys'],crops['affine_original_to_square']):
        points=np.array([[x-s/2,y-s/2],[x+s/2,y-s/2],[x,y]],np.float32).astype(np.float64)
        dst=points@A[:,:2].T+A[:,2]
        crop_error.append(abs(dst-np.array([[0,0],[255,0],[127.5,127.5]])).max())
        roundtrip.append(abs((dst-A[:,2])@np.linalg.inv(A[:,:2]).T-points).max())
    a=raw['vertices_camera_m'];world=raw['vertices_world_m'];back=(world-C[:3,3])@R
    beta_range=np.ptp(raw['betas'],axis=0);dt=np.diff(raw['timestamp_seconds'])
    check=dict(status='completed',reference_used=False,frames=114,all_arrays_finite=all(np.isfinite(raw[k]).all() and np.isfinite(post[k]).all() for k in raw.files),
        camera_z_range_m=[float(a[...,2].min()),float(a[...,2].max())],nonpositive_vertex_camera_z=int((a[...,2]<=0).sum()),
        c2w_rotation_orthogonality_max=float(abs(R.T@R-np.eye(3)).max()),c2w_rotation_det=float(np.linalg.det(R)),
        camera_world_roundtrip_max_m=float(abs(back-a).max()),crop_control_point_max_error_px=float(max(crop_error)),crop_affine_inverse_max_error_px=float(max(roundtrip)),
        shared_betas_max_temporal_range=float(beta_range.max()),true_dt_min_max_s=[float(dt.min()),float(dt.max())],
        exported_vertices_per_frame=10475,exported_faces=20908,exported_coco_joints=17,exported_body_joints=22,
        timestamp_and_frame_identity_verified=True,checks_do_not_prove_geometric_accuracy=True)
    delta=np.linalg.norm(post['joints_coco17_camera_m']-raw['joints_coco17_camera_m'],axis=-1)
    check['postproc_displacement_m']=dict(all17=stats(delta),wrists=stats(delta[:,9:11]),ankles=stats(delta[:,15:17]))
    check['postproc_transl_change_max_m']=float(abs(post['transl']-raw['transl']).max())
    check['actual_dt_motion_diagnostic']={}
    for name,z in [('raw',raw),('official_postproc',post)]:
        vel=np.linalg.norm(np.diff(z['joints_coco17_camera_m'],axis=0)/dt[:,None,None],axis=-1)
        check['actual_dt_motion_diagnostic'][name]=dict(wrist_speed_m_per_s=stats(vel[:,9:11]),note='Diagnostic only; network did not consume timestamps; no motion accuracy claim')
    masks=np.load(ROOT/'experiments/mosca_baseline_20260922/segmentation/segmentation.npz')['entity_labels']
    rows=[];panels=[];depth_hashes={};dep_exports={}
    for f in FRAMES:
        rgb=cv2.imread(inp['frame_paths'][f])[...,::-1].copy();depthpath=ROOT/f'experiments/mosca_baseline_20260922/common_input/unidepth_depth/{f:05d}.npz'
        ud=np.load(depthpath)['dep'];depth_hashes[str(depthpath)]=sha(depthpath)
        person=masks[f]==1;inside=cv2.erode(person.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        row=[rgb]
        for name,z in [('raw',raw),('official_postproc',post)]:
            dep=raster(z['vertices_uv'][f],z['vertices_camera_m'][f,:,2],z['faces'],480,640);hit=np.isfinite(dep)
            sel=hit&inside&np.isfinite(ud)&(ud>0)
            rows.append(dict(frame=f,variant=name,person_pixels=int(person.sum()),mesh_pixels=int(hit.sum()),
                visible_person_covered_by_mesh=float((person&hit).sum()/person.sum()),mesh_projected_outside_person=float((hit&~person).sum()/hit.sum()),
                interior_common_pixels=int(sel.sum()),input_depth_signed_difference_m=stats(dep[sel]-ud[sel]),input_depth_absolute_difference_m=stats(abs(dep[sel]-ud[sel])),
                note='SAM2 visible silhouette and UniDepth estimates are input-prior consistency proxies; clothing/occlusion/model bias apply; not independent GT'))
            vis=rgb.copy();color=np.zeros_like(rgb);color[:]=[62,177,241] if name=='raw' else [238,160,71]
            vis[hit]=(0.50*rgb[hit]+0.50*color[hit]).astype(np.uint8)
            contour,_=cv2.findContours(hit.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(vis,contour,-1,(10,50,230),1)
            row.append(vis);dep_exports[f'{name}_{f:05d}']=dep.astype(np.float32)
        canvas=Image.new('RGB',(1920,512),'white');d=ImageDraw.Draw(canvas)
        for j,(im,label) in enumerate(zip(row,['Input RGB','Raw mesh projection','Official postproc mesh projection'])):
            canvas.paste(Image.fromarray(im),(640*j,32));d.text((640*j+10,10),f'Frame {f:03d} | {label}',fill='black')
        canvas.save(OUT/f'mesh_{f:03d}.png');panels.append(canvas.resize((1200,320)))
    montage=Image.new('RGB',(1200,320*len(panels)),'white')
    for i,p in enumerate(panels):montage.paste(p,(0,i*320))
    montage.save(OUT/'mesh_montage.png');np.savez_compressed(OUT/'raster_depth_8frames.npz',**dep_exports)
    check.update(per_frame_input_consistency=rows,input_depth_sha256=depth_hashes,run_record_sha256=sha(EXP/'run01/run.json'),script_sha256=sha(__file__),wall_seconds=time.perf_counter()-start)
    (OUT/'geometry_checks.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps({k:v for k,v in check.items() if k not in ['per_frame_input_consistency','input_depth_sha256']},indent=2))

if __name__=='__main__':main()
