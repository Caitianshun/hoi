"""Freeze dev2 from its single legal camera; reuse D's exact RGB map."""
import json,hashlib,subprocess,sys,time
from pathlib import Path
import cv2
import numpy as np
import trimesh

ROOT=Path('/home/cai_tianshun/Project/HOI')
E=ROOT/'experiments/structured_hoi_20260923'
A=E/'data_audit'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    begin=time.perf_counter();out=E/'data/dev2';raw=A/'dev2_native';raw.mkdir(exist_ok=True)
    assert not (out/'input_manifest.json').exists(),'Do not silently replace frozen input.'
    seq='Date01_Sub01_chairwood_lift';video=A/f'sources/{seq}.0.color.mp4';timefile=A/f'sources/{seq}.0.time.json'
    ts=np.asarray(json.loads(timefile.read_text())['color'],dtype=np.float64)/1e6
    target=np.arange(1.26,11.26+1e-7,.1)
    indices=np.unique(np.abs(ts[:,None]-target).argmin(axis=0))
    assert np.all(np.diff(ts[indices])>0)
    cap=cv2.VideoCapture(str(video));assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(ts)
    rows=[]
    for j,i in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,bgr=cap.read();assert ok and bgr.shape[:2]==(1536,2048)
        p=raw/f'{j:05d}.png';assert cv2.imwrite(str(p),bgr,[cv2.IMWRITE_PNG_COMPRESSION,3])
        rows.append(dict(index=j,path=str(p),sha256=sha(p),video_frame_index=int(i),timestamp_seconds=float(ts[i])))
    cap.release()
    calib=ROOT/'data/BEHAVE/calibration/calibs/intrinsics/0/calibration.json'
    extr=ROOT/'data/BEHAVE/calibration/calibs/Date01/config/0/config.json'
    c=json.loads(calib.read_text())['color'];x=json.loads(extr.read_text())
    c2w=np.eye(4);c2w[:3,:3]=np.array(x['rotation']).reshape(3,3);c2w[:3,3]=x['translation']
    event=dict(status='frozen_from_input_RGB_before_new_model_results',requested_time_window_s=[1.26,11.26],
        context='Rigid wooden chair pickup, carrying turn with mutual partial occlusion, reappearance and set-down.',
        visual_brackets_s=dict(ground_before_lift=[1.263,2.263],lift=[2.73,3.263],
          front_carry=[4.763,5.663],body_occludes_much_of_chair=[7.763,8.263],
          visible_again=[8.763,9.263],set_down=[10.230,10.796]),
        limitation='Brackets from camera0 RGB samples only; no exact contact or visibility truth. No full-object disappearance claim. Grip release after set-down is not verified.',
        evidence=[str(A/'rgb_review/chairwood_cam0_1.3_69.6_step2.0.json'),str(A/'rgb_review/chairwood_cam0_1.3_11.3_step0.5.json')],
        inspected_cameras=[0],reference_meshes_or_poses_read=False,method_scores_read=False)
    m=dict(schema_version=1,protocol='structured_hoi_known_shape_v1_20260923',sequence=seq,camera_id=0,role='input_only',
        source_video=str(video),source_video_sha256=sha(video),source_timestamps=str(timefile),source_timestamps_sha256=sha(timefile),
        frame_paths=[r['path'] for r in rows],frame_indices=indices.tolist(),frame_sha256=[r['sha256'] for r in rows],frames=rows,
        timestamp_seconds=ts[indices].tolist(),K=[[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]],
        distortion=c['opencv'][4:],distortion_model='opencv_rational_8',distortion_order=['k1','k2','p1','p2','k3','k4','k5','k6'],
        c2w=c2w.tolist(),coordinate_system='BEHAVE Date01 world = Kinect 1 color camera; metres',original_size=[2048,1536],
        original_size_order='width,height',width=2048,height=1536,images_undistorted=False,images_resized=False,
        sampling=dict(nominal_fps=10,method='nearest actual recording timestamp; duplicate indices removed',start_requested_s=1.26,end_requested_s=11.26,
            num_frames=len(indices),min_gap_s=float(np.diff(ts[indices]).min()),max_gap_s=float(np.diff(ts[indices]).max()),
            warning='Actual capture intervals are irregular. Temporal losses must use timestamp_seconds, not nominal video fps or frame index.'),
        event_selection=event,allowed_template_input='chairwood vertices/faces only; no scan colors or texture and no published per-frame R/T',
        forbidden_training_inputs=['other-camera images','sensor depth','registered human poses/meshes','registered object poses','scan texture/colors','registration-derived masks','contact references'],
        calibration_sources=[str(calib),str(extr)],calibration_sha256=[sha(calib),sha(extr)],selection_script_sha256=sha(__file__))
    rawmanifest=A/'dev2_native_input_manifest.json';rawmanifest.write_text(json.dumps(m,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/mosca_prepare_rgb.py'),'--manifest',str(rawmanifest),'--output',str(out),'--width','640'],check=True)
    prepared=json.loads((out/'input_manifest.json').read_text());d1=json.loads((ROOT/'experiments/mosca_baseline_20260922/common_input/input_manifest.json').read_text())
    np.testing.assert_array_equal(np.asarray(prepared['K']),np.asarray(d1['K']))
    np.testing.assert_array_equal(np.asarray(prepared['c2w']),np.asarray(d1['c2w']))
    template=A/'sources/chairwood_f2500.ply';mesh=trimesh.load(template,process=False)
    dst=out/'object_template_geometry.npz';np.savez_compressed(dst,vertices=np.asarray(mesh.vertices,np.float64),faces=np.asarray(mesh.faces,np.int64))
    meta=dict(role='explicit_known_object_shape_input',object='chairwood',path=str(dst),sha256=sha(dst),
        source_path=str(template),source_sha256=sha(template),source_url='https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/objects.zip',
        source_member='objects/chairwood/chairwood_f2500.ply',vertex_count=len(mesh.vertices),face_count=len(mesh.faces),
        fields=['vertices','faces'],colors_or_texture_included=False,normalization='None; preserve released canonical scan coordinates and units. Initialization may center explicitly with saved transform.',
        bounds=np.asarray(mesh.bounds).tolist(),no_per_frame_pose=True)
    (out/'object_template_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    acceptance=dict(status='RGB_and_template_CPU_ready_priors_not_run',frames=len(indices),time_window_s=[float(ts[indices][0]),float(ts[indices][-1])],
        min_gap_s=m['sampling']['min_gap_s'],max_gap_s=m['sampling']['max_gap_s'],manifest=str(out/'input_manifest.json'),
        manifest_sha256=sha(out/'input_manifest.json'),K_identical_to_dev1=True,c2w_identical_to_dev1=True,
        frame_bytes=sum(Path(q).stat().st_size for q in prepared['frame_paths']),native_frame_bytes=sum(Path(q).stat().st_size for q in m['frame_paths']),
        elapsed_seconds=time.perf_counter()-begin,object_template=meta,RGB_event=event,downstream_pending=['SAM2','UniDepth','CoTracker/flow','GVHMR','RGB-only object pose initialization'])
    (A/'dev2_acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n');print(json.dumps(acceptance,indent=2))

if __name__=='__main__':main()
