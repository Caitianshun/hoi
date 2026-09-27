"""CPU-only full source binding and shared coordinate/time audit."""
from common import *
import csv,pickle,math
import numpy as np
from PIL import Image
def run():
    train=read(OLD/'inputs/hos_backpack/manifest.json');dev=read(OLD/'inputs/hos_backpack/evaluation_manifest.json')
    cp=train['camera_source'];assert sha(cp['path'])==cp['sha256']
    with open(cp['path'],'rb') as f:cameras=pickle.load(f)
    rows=[];lookup={}
    for split,m in [('train',train),('dev',dev)]:
        for f in m['frames']:
            stem=f['frame_id'];c=cameras[stem];K=np.array(f['K']);W=np.array(f['w2c'])
            assert np.array_equal(K,np.asarray(c['intrinsics'])) and np.array_equal(W,np.asarray(c['scaleworld_to_camera']))
            assert f['time']==(int(stem)-1)/282 and Path(f['image_path']).stem==Path(f['mask_path']).stem==stem
            im=identity(f['image_path']);mask=identity(f['mask_path'])
            for key,asset in [('image_sha256',im),('mask_sha256',mask)]:
                if key in f:assert f[key]==asset['sha256']
            assert Image.open(im['path']).size==Image.open(mask['path']).size==(f['width'],f['height'])
            center=np.linalg.inv(W)[:3,3]
            row=dict(frame_id=stem,split=split,source_frame_stem=stem,timestamp=f['time'],timestamp_units='normalized nominal frame index; exposure seconds unknown',source_index=int(stem),K=f['K'],w2c=f['w2c'],width=f['width'],height=f['height'],camera_center=center.tolist(),image=im,mask=mask,camera_source=cp,resize=m['resize'])
            rows.append(row);lookup[stem]=row
    assert len(rows)==284 and len(lookup)==284 and len(train['frames'])==268 and len(dev['frames'])==16
    coverage=[]
    tids=sorted(int(f['frame_id']) for f in train['frames'])
    for f in dev['frames']:
        i=int(f['frame_id']);a=lookup[f['frame_id']];r=dict(frame_id=f['frame_id'],extrapolation=i==0,neighbors={})
        for side,ids in [('left',[x for x in tids if x<i]),('right',[x for x in tids if x>i])]:
            if not ids:r['neighbors'][side]=None;continue
            j=max(ids) if side=='left' else min(ids);b=lookup[f'{j:05d}']
            rel=np.array(a['w2c'])[:3,:3]@np.array(b['w2c'])[:3,:3].T
            r['neighbors'][side]=dict(frame_id=b['frame_id'],time_delta=b['timestamp']-a['timestamp'],camera_distance_over_extent=float(np.linalg.norm(np.array(a['camera_center'])-b['camera_center'])/train['scene_extent']),rotation_degrees=math.degrees(math.acos(float(np.clip((np.trace(rel)-1)/2,-1,1)))),intrinsics_max_abs=float(np.abs(np.array(a['K'])-b['K']).max()))
        coverage.append(r)
    inherited=[identity(V5/'protocol/shared_fine_initial.pt')]
    expected=['ef207932701978cb0f09925ad085378b195e0a514138923567bf717f2d28653d','1e120cfcb332faadd4e17871e4679e0c7a0c7a6dec0ea6a3892d363b5b6170b6','d9ec9dce43a38871fe63abb132a2093b8d9ff64bc50b26e7f5c669e6202d6cff']
    for arm in ['B_U','B_F']:inherited.append(identity(read(V5/'runs'/arm/'run.json')['checkpoint']))
    assert [x['sha256'] for x in inherited]==expected
    inherited.append(identity(V5/'runs/B_U/checkpoint_fine_001000.pt'))
    save_json(RUN/'state_manifest.json',dict(inherited=inherited,history_read_only=True))
    save_json(RUN/'camera_time_audit.json',dict(status='passed',rows=rows,coverage=coverage,source_binding='exact published stem K/w2c',camera_estimation_inputs='unknown; published full-video cameras; not asserted train-only',physical_timestamp='unknown',optimization_attempts=0))
    with (RUN/'camera_time_audit.csv').open('w') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    save_json(RUN/'camera_coverage.json',coverage)
    print('D0 CPU binding passed',len(rows))
if __name__=='__main__':run()
