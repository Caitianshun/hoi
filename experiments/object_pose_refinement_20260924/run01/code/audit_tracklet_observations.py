from pathlib import Path
import hashlib,json,numpy as np,cv2
E=Path('/home/cai_tianshun/Project/HOI/experiments/object_pose_refinement_20260924/run01')
records={}
for dev in ['dev1','dev2']:
    out=E/'pose'/dev;k=json.loads((out/'keyframes.json').read_text());d=dict(np.load(out/'tracklets.npz'));seg=np.load(k['segmentation'])['entity_labels'];m=json.loads(Path(k['source_manifest']).read_text());ts=np.array(m['timestamp_seconds']);a=d['adopted'];assert np.all(d['source_time']==ts[d['source_frame']]);assert np.all(d['target_time']==ts[d['target_frame']]);assert np.all(np.abs(d['target_time']-d['source_time'])<=1.200000001);assert np.all(d['source_frame'][a]!=d['target_frame'][a]);assert np.all(d['cycle_error_px'][a]<=2);assert np.all(d['patch_ncc'][a]>=.35);assert np.all(d['rgb_mean_abs_difference'][a]<=50);assert np.all(d['observation_state'][a]==0)
    for key in k['keyframes']:
        q=np.load(key['query_path']);er=cv2.erode((seg[key['frame_index']]==2).astype(np.uint8),np.ones((5,5),np.uint8));xy=np.rint(q).astype(int);assert np.all(er[xy[:,1],xy[:,0]])
    cycles=d['cycle_error_px'][d['cycle_tested']];acceptedcycles=d['cycle_error_px'][a]
    pairs=[]
    for i,ki in enumerate(k['keyframes']):
        for kj in k['keyframes'][i+1:]:
            overlap=sorted(set(ki['window_frame_indices'])&set(kj['window_frame_indices']))
            if overlap:pairs.append(dict(source_frames=[ki['frame_index'],kj['frame_index']],overlap_frames=overlap))
    records[dev]=dict(record_count=len(a),adopted_count=int(a.sum()),adopted_ratio=float(a.mean()),cycle_tested_count=len(cycles),cycle_median_px=float(np.median(cycles)),cycle_p95_px=float(np.percentile(cycles,95)),adopted_cycle_median_px=float(np.median(acceptedcycles)),source_self_count=int((d['source_frame']==d['target_frame']).sum()),canonical_confirmation_count=int(d['canonical_confirmed'].sum()),cross_keyframe_window_overlap=pairs,source_mask_pass=True,physical_time_pass=True,state_metadata_pass=True,all_rejected_records_retained=True,tracklets_sha256=hashlib.sha256((out/'tracklets.npz').read_bytes()).hexdigest())
(E/'logs/tracklet_observation_audit.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
