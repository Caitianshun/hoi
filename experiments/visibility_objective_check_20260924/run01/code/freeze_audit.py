"""Freeze input-only frame selection before residual inspection. No reference IO."""
from pathlib import Path
import json, hashlib, datetime, shutil
import numpy as np
from scipy.ndimage import binary_dilation

ROOT=Path('/home/cai_tianshun/Project/HOI')
E=Path(__file__).resolve().parents[1]
PREV=ROOT/'experiments/surface_pose_joint_20260924/run01'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
inputs={};selection={};cross=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
for dev in ['dev1','dev2']:
    run=json.loads((PREV/dev/'F01/run.json').read_text())
    init=Path(run['initialization']);a=np.load(init)
    base=ROOT/('experiments/mosca_baseline_20260922/common_input' if dev=='dev1' else 'experiments/structured_hoi_20260923/data/dev2')
    seg=base.parent/'segmentation/segmentation.npz' if dev=='dev1' else base/'segmentation/segmentation.npz'
    labels=np.load(seg)['entity_labels'];assert set(np.unique(labels))=={0,1,2}
    rows=[]
    for i,lab in enumerate(labels):
        o,h,b=lab==2,lab==1,lab==0
        oh=int((o&binary_dilation(h,structure=cross)).sum());ob=int((o&binary_dilation(b,structure=cross)).sum())
        rows.append(dict(frame=i,O=int(o.sum()),H=int(h.sum()),U=0,B=int(b.sum()),O_adjacent_H=oh,O_adjacent_B=ob,ratio=oh/max(1,oh+ob),eligible=bool(a['observed_mask'][i] and o.sum()>=64 and oh>0 and ob>0)))
    chosen=[]
    for third in np.array_split(np.arange(len(labels)),3):
        candidates=[rows[int(i)] for i in third if rows[int(i)]['eligible']]
        assert candidates
        chosen.append(sorted(candidates,key=lambda x:(-x['ratio'],-x['O_adjacent_H'],x['frame']))[0]['frame'])
    paths=[seg,seg.with_name('segmentation_run.json'),base/'input_manifest.json',init,PREV/dev/'F01/object_init.npz',PREV/dev/'F01/run.json',PREV/f'observations/{dev}/observations.npz',PREV/f'observations/{dev}/features/frozen_descriptors.npz']
    manifest=json.loads((base/'input_manifest.json').read_text())
    paths += [Path(manifest['frame_paths'][i]) for i in chosen]
    inputs[dev]=dict(segmentation=str(seg),base=str(base),initialization=str(init),F01=str(PREV/dev/'F01/object_init.npz'),frame_count=len(labels),visible_count=int(a['observed_mask'].sum()),all_counts={k:int((labels==v).sum()) for k,v in [('B',0),('H',1),('O',2)]},U_pixels=0,hashes={str(p):sha(p) for p in paths})
    selection[dev]=dict(frames=chosen,all_input_candidates=rows)
save(E/'protocol/frame_selection.json',dict(frozen_utc=now,reference_used=False,prediction_used_for_selection=False,rule='Split each existing timeline into three contiguous thirds. In each third select fixed-visible frame with O>=64, nonzero 4-neighbour O-H and O-B boundaries, maximal O-H/(O-H+O-B), tie greater O-H then earlier frame. No dilation of supervision; adjacency only selects frames.',sequences=selection))
save(E/'protocol/frozen_visibility.json',dict(schema='visibility_semantics_audit_v1',frozen_utc=now,status='audit_frozen_before_residual_test',inputs=inputs,labels={'O':'entity_labels==2','H':'entity_labels==1','B':'entity_labels==0','U':'no explicit U in existing input; empty; do not create or expand'},overlap='original maximal positive SAM2 logit, equal logits lower entity id; no raw logits retained for new uncertainty',prediction='complete object alone, no human z-buffer or full SMPL-X mask',fixed_sampling={'positive':64,'box_negative':'64 equal arc-length complete projected convex-hull boundary samples','chair_negative':'64 quantiles of distances at all original centres_m[::2] projections','distance':'scipy distance_transform_edt(B); zero on O/H/U; current projected bilinear sampling','normalization':'rho(residual_px/16) sum divided by full frame count, not by active/background/64 count','visibility':'original observed_mask fixed per frame'},conditional_C0='historical F01, not promoted baseline; no replay absent semantic mismatch',conditional_C1='only if synthetic conflict and real-input path both established',budget={'new_C1':2,'max_total_pose_solves_with_C0_replay':4,'steps':300,'seconds_per_solve':600,'new_gaussian_runs_if_promoted':2,'gaussian_steps':8000,'deadline':'2026-09-27'},promotion={'relative_C0':'same primary centre or displacement improved beyond tolerance both dev; others guarded','relative_old':'same primary centre or displacement improved beyond tolerance both dev; others guarded','linear_tolerance':'max(0.5 cm,5% comparator)','angular_tolerance':'max(2 deg,5% comparator)','depth_P90':'max(0.5 cm,5% comparator)','coverage_drop_max_percentage_points':5,'preserve_all_O':True,'preserve_all_slots_and_chair_64_to_74':True,'no_new_reference_evaluation_before_C1_both_frozen':True},original_config_sha256=sha(PREV/'protocol/frozen_surface_pose.json'),original_promotion_sha256=sha(PREV/'protocol/promotion_rules.json')))
for name in ['frozen_surface_pose.json','promotion_rules.json']:
    shutil.copy2(PREV/'protocol'/name,E/'protocol'/('inherited_'+name))
print({d:selection[d]['frames'] for d in selection})
