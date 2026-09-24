"""Bounded dataset inventory. Reads file metadata/manifests and legal RGB only."""
import datetime, hashlib, json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path('/home/cai_tianshun/Project/HOI')
A=Path(__file__).resolve().parent
OLD=Path('/home/cai_tianshun/Project/4dsr')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(n,d):(A/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    splitpath=A/'sources/official_split_20260923.json';split=load(splitpath)
    existing=ROOT/'experiments/mosca_baseline_20260922/common_input/input_manifest.json'
    m1=load(existing);a2=load(A/'dev2_acceptance.json')
    tests=[
        ('Date03_Sub03_boxlarge','large closed box','boxlarge'),
        ('Date03_Sub03_chairwood_lift','wooden chair lifting','chairwood'),
        ('Date03_Sub03_monitor_move','monitor movement; verify base rigid during chosen event','monitor'),
        ('Date03_Sub04_boxsmall','small closed box','boxsmall'),
        ('Date03_Sub04_stool_move','stool movement','stool'),
        ('Date03_Sub04_plasticcontainer_lift','container lifting; inspect deformation before admission','plasticcontainer'),
        ('Date03_Sub05_tablesmall','small table manipulation','tablesmall'),
        ('Date03_Sub05_trashbin','bin manipulation; inspect lid/handle and nonrigidity','trashbin'),
    ]
    objects=load(ROOT/'data/BEHAVE/source_metadata/objects.zip.index.json')
    rows=[]
    for i,(s,reason,obj) in enumerate(tests):
        assert s in split['test'] and s not in split['train']
        geometry=[x for x in objects['entries'] if x['name'].startswith(f'objects/{obj}/') and x['name'].endswith('.ply')]
        assert len(geometry)==1
        rows.append(dict(id=f'test{i+1:02d}',sequence=s,official_split='test',object=obj,subject=s.split('_')[1],
            reason=reason,selection='Frozen names before structured model results; diversify rigid categories and subjects, never by reconstruction metrics.',
            input_camera=0,heldout_cameras=[1,2,3],window=None,window_status='Requires camera0 RGB event review before optimization; not a verified complete event yet.',
            input_RGB_local=False,source_video_member=f'{s}.0.color.mp4',source_video_archive='https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/video/date03_color.tar',
            timestamps_local=False,timestamp_archive='https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/video/date03_time.tar',
            Date03_extrinsics_local=False,canonical_geometry_local=False,canonical_geometry_archive=objects['url'],canonical_geometry_member=geometry[0],
            evaluation_reference_local=False,download_status='not_started',rigid_event_admission='provisional_category_suitable_RGB_motion_and_articulation_check_pending'))
    test=dict(schema_version=1,status='eight_official_test_sequence_candidates_frozen_not_eight_ready_events',created_at_utc=now,
        official_split_path=str(splitpath),official_split_sha256=sha(splitpath),official_test_count=len(split['test']),
        candidates=rows,scene_coverage='All official test names in the downloaded split are Date03. Do not claim independent multi-location test coverage from these eight.',
        selection_rules=['Use one prescribed camera0 RGB and calibrated K/c2w.','Select first fully observed lift/manipulation/setdown event under camera0 input only; target10–15s and record actual timestamps.',
            'Freeze exact frame list before any method score; do not use fitted masks/poses or heldout images to choose.',
            'Reject articulated/opening or visibly deforming events by documented RGB rule; preserve rejection; replacement requires a versioned protocol before scoring.',
            'No frames excluded based on reconstruction quality. Missing predictions/references retain status and coverage.'],
        acquisition_plan=dict(mode='bounded HTTP Range; no full16.5GB archive',video_size_estimate_MB=[250,450],
            video_size_estimate_basis='Eight cam0 clips, approximate prior38MB; Date03 member index not yet fetched, not a measured budget.',
            elapsed_estimate_minutes=[15,40],elapsed_estimate_basis='Network dependent, planning estimate only; include timestamps/calibration/template members separately.',
            next_action='Index Date03 archive and required members, freeze actual byte budget before transfer; retained cameras/reference are evaluation-only separate budget.'),
        no_test_RGB_or_reference_opened_this_audit=True)
    save('behave_test_candidates_v1.json',test)
    oldpath=OLD/'data/hoi_v1_independent/manifest_frozen_v1.json';old=load(oldpath)
    hrows=[]
    for seq in ['subject05_box','subject06_trashcan','subject09_chair','subject10_book']:
        spec=old['sequences'][seq];obj=seq.split('_',1)[1]
        rgbdir=OLD/'data/hoi_v1_independent'/seq/'cam14/hr'
        inputpaths=[rgbdir/f'{i:06d}.png' for i in range(spec['start'],spec['end']+1)]
        assert all(p.is_file() for p in inputpaths)
        template=OLD/f'data/HODome/independent_v1/scaned_object/{obj}/{obj}_face1000.obj'
        calibration=OLD/f'data/HODome/independent_v1/calibration_ground/{spec["date"]}/calibration.json'
        video=OLD/f'data/HODome/independent_v1/videos/{seq}/data15.mp4'
        samples=[x for x in old['samples'] if x['sequence']==seq and x['camera_id']==14]
        vmeta=next(x for x in samples if x['frame_id']==spec['start'])
        assert template.is_file() and calibration.is_file() and video.is_file()
        # Container PTS is read from the actual decoded video, not calculated from nominal60fps.
        cap=cv2.VideoCapture(str(video));assert cap.isOpened();cap.set(cv2.CAP_PROP_POS_FRAMES,spec['start'])
        pts=[]
        for _ in inputpaths:
            ok,_=cap.read();assert ok;pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)
        cap.release();pts=np.asarray(pts)
        valid=bool(np.isfinite(pts).all() and np.all(np.diff(pts)>0))
        save(f'{seq}_cam14_container_pts.json',dict(source=str(video),frame_ids=list(range(spec['start'],spec['end']+1)),
             timestamps_seconds=pts.tolist(),method='OpenCV CAP_PROP_POS_MSEC after decoding each actual frame; no index/fps construction.',
             valid_monotonic=valid,limitation='Container presentation time only; original sensor acquisition timestamps unavailable locally for this independent package. Not a proof of physical synchronization with published fits.'))
        heldout=[]
        for cam in [22,28,25,31]:
            p=OLD/'data/hoi_v1_independent'/seq/f'cam{cam}'/'hr';heldout.append(dict(camera=cam,path=str(p),png_count=len(list(p.glob('*.png'))),role='evaluation_only'))
        status='rigid_event_RGB_admissible_with_scope_limits' if obj!='book' else 'candidate_closed_book_rigidity_fullframe_review_pending'
        limits={
            'box':'Pickup-to-overhead hold, no putdown/release. Input camera14 clips part of overhead box near top boundary; do not discard those frames.',
            'trashcan':'Pickup-to-carry/turn; no complete object disappearance or putdown/release established.',
            'chair':'Tilt/reorientation and settling. Human partially occludes chair; full disappearance not established.',
            'book':'Closed carry visible in16 sampled camera14 images; tiny page/covers movement between samples not excluded. Do not mark strict rigid acceptance before full-frame review.'}[obj]
        hrows.append(dict(sequence=seq,object=obj,status=status,input_camera=14,input_resolution=[1920,1080],
            new_training_resolution='not_yet_prepared; use native-background RGB, never LR/SR cached images',start_frame=spec['start'],end_frame_inclusive=spec['end'],frames=len(inputpaths),
            input_RGB_dir=str(rgbdir),input_RGB_present=True,first_frame_sha256=sha(inputpaths[0]),last_frame_sha256=sha(inputpaths[-1]),
            source_video=str(video),source_video_bytes=video.stat().st_size,source_video_sha256_from_old_manifest=vmeta['source_video_sha256'],video_full_hash_recomputed=False,
            geometry_template=str(template),geometry_template_sha256=sha(template),template_texture_allowed=False,
            calibration=str(calibration),calibration_sha256=sha(calibration),container_PTS_file=str(A/f'{seq}_cam14_container_pts.json'),container_PTS_monotonic=valid,
            original_sensor_timestamps_available=False,published_pose_allowed_as_input=False,published_fit_masks_allowed_as_input=False,
            heldout=heldout,scope_limits=limits,RGB_review=str(A/f'rgb_review/{seq}_cam14.json'),
            historical_window_provenance_note='Pre-existing package manifest mentions training RGB and released/object-motion or geometry FOV checks during historical window selection. Current audit only rechecks input camera14 RGB; do not rewrite historical origin as RGB-only.',
            old_geometry_and_labels_role='evaluation_only_or_quarantined; never auto-carry old auxiliary geometry/labels into new training',
            pending=['Fresh single-camera RGB SAM2/track/depth/GVHMR priors','No published R/T initialization','Known geometry-only export','New protocol resolution/temporal adapter']))
    hodome=dict(schema_version=1,created_at_utc=now,protocol='four_existing_package_candidates_single_camera_v1',
        original_manifest=str(oldpath),original_manifest_sha256=sha(oldpath),candidates=hrows,
        accepted_rigid_scopes=3,conditional_book_candidates=1,no_complete_four_rigid_claim=True,
        role='Supplemental foreground/heldout evidence only; green studio background does not support natural-scene reconstruction claims.',
        priors_training_status='not_prepared_for_new_protocol',old_SR_labels_geometry_training_prohibited=True,
        evaluation_camera_policy='Only cam14 is the new input; all other cameras are unavailable to training. Existing heldout paths inventoried, never opened for selection.')
    save('hodome_candidates_v1.json',hodome)
    inventory=dict(schema_version=1,created_at_utc=now,scan_scope=['HOI/data/BEHAVE top-level and pre-existing metadata','4dsr/data/HODome and hoi_v1_independent manifests; no weights or logs scan'],
        behave_before_this_audit=dict(local_complete_input_video_sequences=['Date01_Sub01_boxsmall_hand'],local_calibration_dates=['Date01'],other_video_data_local=False),
        dev1=dict(sequence=m1['sequence'],split='train',manifest=str(existing),manifest_sha256=sha(existing),frames=len(m1['frame_paths']),input_camera=0,original_D_protocol='RGB+calibration+predicted priors; no object scan input',
            new_protocol_exception='Canonical object geometry is newly allowed; do not edit or relabel old D manifest/results.'),
        dev2=dict(sequence='Date01_Sub01_chairwood_lift',split='train',acceptance=str(A/'dev2_acceptance.json'),input_manifest=a2['manifest'],input_manifest_sha256=a2['manifest_sha256'],frames=a2['frames'],time_window_s=a2['time_window_s']),
        test_list=str(A/'behave_test_candidates_v1.json'),hodome_list=str(A/'hodome_candidates_v1.json'),
        official_sources=[dict(url='https://github.com/xiexh20/behave-dataset',purpose='sequence-level split; frame_times; object template conventions'),
                          dict(url='https://virtualhumans.mpi-inf.mpg.de/behave/license.html',purpose='official split and archive download paths'),
                          dict(url='https://github.com/Juzezhang/NeuralDome_Toolbox',purpose='HODome modalities/body-model and object-pose distinction')],
        attachment=dict(path='/home/cai_tianshun/下载/CVPR2027_HOI_Codex_执行指导 (1).md',sha256=sha('/home/cai_tianshun/下载/CVPR2027_HOI_Codex_执行指导 (1).md')),
        scripts={str(p):sha(p) for p in [Path(__file__),A/'prepare_dev2.py',A/'review_input.py',ROOT/'scripts/mosca_prepare_rgb.py']},
        acquisition_sources=[load(p) for p in sorted((A/'sources').glob('*.source.json'))],
        immutable_inputs_no_mutations=True,training_or_GPU_operations=False)
    save('inventory.json',inventory)
    print(json.dumps(dict(status='completed',test_candidates=len(rows),hodome_candidates=len(hrows),ready_rigid_scopes=3,conditional=1,dev2_frames=a2['frames'])))

if __name__=='__main__':main()
