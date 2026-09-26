#!/usr/bin/env python3
"""Apply a fixed training-side 90k background / 10k foreground point cap."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(2**20),b''):h.update(c)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);a=p.parse_args();base=a.input_dir.resolve()
 raw=base/'initial_points.npz';data=dict(np.load(raw));rng=np.random.default_rng(12345)
 selected=[]
 for label,quota in [(0,90000),(1,10000)]:
  ids=np.flatnonzero(data['label']==label)
  selected.extend((ids if len(ids)<=quota else rng.choice(ids,quota,replace=False)).tolist())
 selected=np.array(sorted(selected),np.int64);xyz=data['xyz'][selected];rgb=data['rgb'][selected]
 output=base/'initial_points_100k.npz'
 np.savez_compressed(output,**{k:v[selected] for k,v in data.items()},raw_selected_indices=selected)
 np.save(base/'selected_raw_indices.npy',selected)
 manifest=json.loads((base/'manifest.json').read_text());rows=manifest['frames']
 support=[];selected_ids=data['source_frame_ids'][selected];selected_xy=data['source_keypoints'][selected];fg=data['label'][selected]==1
 for row in rows:
  f=int(row['frame_id']);matches=np.where((selected_ids==f)&fg[:,None]);xy=selected_xy[matches[0],matches[1]]
  mask=cv2.imread(row['mask_path'],cv2.IMREAD_GRAYSCALE)>=224;coverage=np.zeros(mask.shape,np.uint8)
  for x,y in np.round(xy).astype(int):cv2.circle(coverage,(x,y),8,1,-1)
  support.append(dict(frame_id=row['frame_id'],foreground_observations=len(xy),foreground_mask_pixels=int(mask.sum()),foreground_mask_fraction_within_8px_of_source_keypoint=float((mask&(coverage>0)).sum()/max(mask.sum(),1))))
 center=np.median(xyz,axis=0);extent=float(1.1*np.percentile(np.linalg.norm(xyz-center,axis=1),95))
 # AABB shape matches project adapter {min,max}, no coordinate scaling.
 manifest['scene_center']=center.tolist();manifest['scene_extent']=extent;manifest['aabb']={'min':xyz.min(0).tolist(),'max':xyz.max(0).tolist()}
 manifest['point_cloud']=dict(npz_path=str(output),path=str(output),sha256=sha(output),points=len(xyz),source='frozen train-only RGB/mask two-view triangulation; stratified fixed random cap',raw_path=str(raw),raw_sha256=sha(raw),selected_indices=str(base/'selected_raw_indices.npy'))
 manifest['initialization']=manifest['point_cloud']
 (base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 stats=dict(seed=12345,raw_points=len(data['xyz']),selected_points=len(xyz),raw_background=int((data['label']==0).sum()),raw_foreground=int((data['label']==1).sum()),selected_background=int((data['label'][selected]==0).sum()),selected_foreground=int((data['label'][selected]==1).sum()),quota={'background':90000,'foreground':10000},raw_sha256=sha(raw),sampled_sha256=sha(output),source_support=support,
        foreground_support_scope='published mask combines human and interacted objects; no separate object instance labels, so object-specific support is NA',
        train_only_support_note='8px disks around actual accepted source keypoints; image support proxy, not verified geometry or dense coverage',
        dynamic_note='union of quasi-static two-view foreground triangulations at multiple times; not a canonical body or rigid object initialization',
        scene_center=manifest['scene_center'],scene_extent=extent,aabb=manifest['aabb'])
 (base/'sampling_and_support.json').write_text(json.dumps(stats,indent=2)+'\n')
 print(json.dumps({k:v for k,v in stats.items() if k!='source_support'},indent=2))

if __name__=='__main__':main()
