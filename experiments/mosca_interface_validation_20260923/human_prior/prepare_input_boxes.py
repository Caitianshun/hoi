#!/usr/bin/env python3
"""Export only existing SAM2 person boxes for the frozen 114-RGB HMR2 input."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/cai_tianshun/Project/HOI'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    work=root/'experiments/mosca_interface_validation_20260923/human_prior'
    out=(args.output or work/'prepared_input').resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'input_manifest.json').exists():raise RuntimeError('Use a fresh output directory')
    start=time.perf_counter();(out/'executed_prepare.py').write_bytes(Path(__file__).read_bytes())
    source=root/'experiments/mosca_baseline_20260922/common_input/input_manifest.json'
    inp=json.loads(source.read_text());assert inp['role']=='input_only' and len(inp['frames'])==114
    segdir=root/'experiments/mosca_baseline_20260922/segmentation'
    segmeta=json.loads((segdir/'segmentation_run.json').read_text())
    assert segmeta['status']=='completed' and segmeta['reference_geometry_used'] is False
    seg=np.load(segdir/'segmentation.npz',allow_pickle=False)
    labels=seg['entity_labels'];assert labels.shape==(114,480,640)
    assert segmeta['entity_ids']['person']==1
    boxes=np.full((114,4),np.nan,np.float32);area=np.zeros(114,np.int64)
    for i,frame in enumerate(inp['frames']):
        assert frame['index']==i
        assert segmeta['frames'][i]['source']==frame['path']
        assert segmeta['frames'][i]['sha256']==frame['sha256']==sha(frame['path'])
        assert str(seg['source_frame_names'][i])==Path(frame['path']).name
        y,x=np.nonzero(labels[i]==1);area[i]=len(x)
        if len(x):boxes[i]=[x.min(),y.min(),x.max()+1,y.max()+1]
    present=area>0
    np.savez_compressed(out/'sam2_person_boxes.npz',bbox_xyxy=boxes,bbox_valid=present,
        person_mask_area_pixels=area,input_frame_index=np.arange(114),video_frame_index=inp['frame_indices'],
        timestamp_seconds=inp['timestamp_seconds'],role='estimated_input_person_bbox_not_visibility_GT')
    data=dict(inp)
    data.update(schema_version='human_prior_input_v1',role='frozen_input_only_for_human_prior',status='prepared_inference_not_run',
        source_manifest=str(source),source_manifest_sha256=sha(source),
        bbox_source=str(segdir/'segmentation.npz'),bbox_source_sha256=sha(segdir/'segmentation.npz'),
        bbox_source_run=str(segdir/'segmentation_run.json'),bbox_source_run_sha256=sha(segdir/'segmentation_run.json'),
        bbox_export=str(out/'sam2_person_boxes.npz'),bbox_export_sha256=sha(out/'sam2_person_boxes.npz'),
        bbox_definition='tight union of existing entity_labels==1 pixels; xyxy max-exclusive, no added padding, no smoothing or connected-component selection',
        person_bbox_xyxy=[b.tolist() if ok else None for b,ok in zip(boxes,present)],
        person_bbox_valid=present.tolist(),person_mask_area_pixels=area.tolist(),
        visibility_warning='SAM2 presence is predicted silhouette, not anatomical keypoint visibility or physical correspondence; do not infer confidence from bbox validity',
        inference_run=False,reference_geometry_used=False,code_sha256=sha(Path(__file__)))
    (out/'input_manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    summary={'status':'prepared_inference_not_run','frames':114,'person_boxes_present':int(present.sum()),
        'person_area_min_max':[int(area.min()),int(area.max())],
        'bbox_width_min_max':[float(np.nanmin(boxes[:,2]-boxes[:,0])),float(np.nanmax(boxes[:,2]-boxes[:,0]))],
        'bbox_height_min_max':[float(np.nanmin(boxes[:,3]-boxes[:,1])),float(np.nanmax(boxes[:,3]-boxes[:,1]))],
        'RGB_hashes_verified':114,'source_hashes':{str(source):sha(source),str(segdir/'segmentation.npz'):sha(segdir/'segmentation.npz'),str(segdir/'segmentation_run.json'):sha(segdir/'segmentation_run.json')},
        'GPU_used':False,'wall_seconds':time.perf_counter()-start}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
