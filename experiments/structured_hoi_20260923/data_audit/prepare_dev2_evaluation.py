"""Prepare isolated, predetermined dev2 references; never reads model outputs."""
import concurrent.futures,datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
import cv2
import numpy as np
import trimesh

R=Path('/home/cai_tianshun/Project/HOI');E=R/'experiments/structured_hoi_20260923';O=E/'data_audit/dev2_evaluation'
ARCHIVE=R/'experiments/mosca_baseline_20260922/data/remote_archive.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')

def fetch(kind,index,member,dest,budget=3):
    ix=load(index)
    if dest.exists() and Path(str(dest)+'.source.json').exists():
        src=load(str(dest)+'.source.json');assert src['sha256']==sha(dest);return src|{'path':str(dest)}
    cmd=[sys.executable,str(ARCHIVE),f'fetch_{kind}',ix['url'],str(dest),'--member',member,'--index',str(index),'--budget-mb',str(budget)]
    proc=subprocess.run(cmd,capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(f'{member}: {proc.stderr}')
    return load(str(dest)+'.source.json')|{'path':str(dest)}

def camera(kid):
    ip=R/f'data/BEHAVE/calibration/calibs/intrinsics/{kid}/calibration.json';ep=R/f'data/BEHAVE/calibration/calibs/Date01/config/{kid}/config.json'
    c=load(ip)['color'];x=load(ep);K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]],np.float64);dist=np.array(c['opencv'][4:],np.float64)
    c2w=np.eye(4);c2w[:3,:3]=np.asarray(x['rotation']).reshape(3,3);c2w[:3,3]=x['translation']
    kfull,roi=cv2.getOptimalNewCameraMatrix(K,dist,(2048,1536),0,(2048,1536))
    S=np.array([[640/2048,0,(640/2048-1)/2],[0,480/1536,(480/1536-1)/2],[0,0,1]])
    kout=S@kfull;mx,my=cv2.initUndistortRectifyMap(K,dist,None,kout,(640,480),cv2.CV_32FC1)
    valid=(mx>=0)&(mx<2047)&(my>=0)&(my<1535);assert valid.mean()>.995
    return kout,c2w,mx,my,dict(K_original=K.tolist(),distortion=dist.tolist(),K_rectified_full=kfull.tolist(),K=kout.tolist(),c2w=c2w.tolist(),world_frame='BEHAVE Date01 Kinect1 color; metres',width=640,height=480,source_paths=[str(ip),str(ep)],source_sha256=[sha(ip),sha(ep)],valid_fraction=float(valid.mean()),fixed_map=True,pixel_centres='u_out=scale*(u_rectified+0.5)-0.5',full_roi=list(map(int,roi)))

def main():
    begin=time.perf_counter();plan=load(O/'protocol.json');seq=plan['sequence'];m=load(plan['input_manifest']);original_hash=sha(plan['input_manifest'])
    assert original_hash==plan['input_manifest_sha256']
    timestamp_index=R/'data/BEHAVE/source_metadata/date01_time.tar.index.json';video_index=R/'data/BEHAVE/source_metadata/date01_color.tar.index.json'
    vidmember=f'{seq}.1.color.mp4';timemember=f'{seq}.1.time.json'
    vmeta=next(x for x in load(video_index)['entries'] if x['name']==vidmember);tmeta=next(x for x in load(timestamp_index)['entries'] if x['name']==timemember)
    addendum=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Nominal t2 is absent from sparse ZIP. Keep missing mesh row; use one complete official camera1 video/timestamps to obtain all five predeclared heldout times without replacing any time.',
        heldout_nominal_times_seconds=plan['heldout_nominal_times_seconds'],extra_video=vmeta,extra_timestamp=tmeta,total_expected_transfer_bytes=plan['compressed_bytes']+vmeta['size']+tmeta['size'],
        image_source_protocol='Primary heldout: all five from raw camera1 video with true recording timestamps. Four existing sparse ZIP JPGs kept separately for provenance; never silently mixed.',no_model_outputs_inspected=True)
    save(O/'heldout_source_addendum.json',addendum);assert addendum['total_expected_transfer_bytes']<100*1024**2
    jobs=[('zip',Path(plan['source_index']),x['name'],O/'source'/x['name'],3) for x in plan['required_members']]
    jobs += [('tar',video_index,vidmember,O/'source'/vidmember,50),('tar',timestamp_index,timemember,O/'source'/timemember,1)]
    state=dict(status='downloading',role='evaluation_only',start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(O/'protocol.json'),records=[],failures=[])
    save(O/'status.json',state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(fetch,*job):job[2] for job in jobs}
        for fut in concurrent.futures.as_completed(futures):
            try:state['records'].append(fut.result())
            except Exception as e:state['failures'].append(dict(member=futures[fut],error=str(e)))
            save(O/'status.json',state)
    if state['failures']:
        state['status']='failed_partial_download';state['elapsed_seconds']=time.perf_counter()-begin;save(O/'status.json',state);raise RuntimeError(state['failures'])
    info=load(O/'source'/seq/'info.json');save(O/'sequence_info_checked.json',dict(info=info,path=str(O/'source'/seq/'info.json'),note='Metadata only; chairwood scan corresponds to sparse archive folder chair.'))
    times=np.asarray(plan['nominal_mesh_times_seconds'],np.float64);input_indices=np.asarray(plan['matched_input_indices'],np.int64);inputtimes=np.asarray(plan['matched_input_times_seconds'],np.float64)
    rows=[];hv=[];ov=[];valid=[];hf=None;of=None
    for ti,t in enumerate(times):
        hp=O/'source'/seq/f't{int(t):04d}.000/person/fit02/person_fit.ply';op=O/'source'/seq/f't{int(t):04d}.000/chair/fit01/chair_fit.ply'
        good=hp.exists() and op.exists();valid.append(good)
        row=dict(nominal_reference_time_seconds=float(t),matched_input_frame_index=int(input_indices[ti]),matched_input_time_seconds=float(inputtimes[ti]),matched_time_minus_nominal_seconds=float(inputtimes[ti]-t),available=good)
        if good:
            human=trimesh.load_mesh(hp,process=False);obj=trimesh.load_mesh(op,process=False)
            if hf is None:hf=np.asarray(human.faces,np.int64);of=np.asarray(obj.faces,np.int64)
            np.testing.assert_array_equal(human.faces,hf);np.testing.assert_array_equal(obj.faces,of)
            hv.append(np.asarray(human.vertices,np.float64));ov.append(np.asarray(obj.vertices,np.float64))
            row.update(human_path=str(hp),object_path=str(op),human_sha256=sha(hp),object_sha256=sha(op),mesh_archive_frame=f't{int(t):04d}.000')
        else:row['missing_reason']='Entire nominal time absent in official sparse ZIP; no interpolation or substitute fitted pose.'
        rows.append(row)
    v=np.asarray(valid,bool);human=np.full((len(times),len(hv[0]),3),np.nan);obj=np.full((len(times),len(ov[0]),3),np.nan);human[v]=np.stack(hv);obj[v]=np.stack(ov)
    assert np.isfinite(human[v]).all() and np.isfinite(obj[v]).all()
    ref=O/'reference_meshes_world.npz'
    np.savez_compressed(ref,human_vertices_world_m=human,human_faces=hf,object_vertices_world_m=obj,object_faces=of,reference_available=v,
        reference_nominal_times_seconds=times,matched_input_indices=input_indices,matched_input_times_seconds=inputtimes,matched_time_minus_nominal_seconds=inputtimes-times,
        coordinate_system=np.array('BEHAVE Date01 Kinect1 color world; metres'),role=np.array('evaluation_only'),human_fit_version=np.array('fit02'),object_fit_version=np.array('fit01'))
    save(O/'reference_meshes_manifest.json',dict(role='evaluation_only',npz=str(ref),sha256=sha(ref),reference_rows=rows,requested=10,available=int(v.sum()),
        human_vertex_count=human.shape[1],human_face_count=len(hf),object_vertex_count=obj.shape[1],object_face_count=len(of),common_topology=True,
        coordinates='Published mesh world positions preserved; no world transform, scale fit, ICP, Procrustes or temporal interpolation.',
        limitations=['Fitted surface reference, not exact cloth/glove/material/contact ground truth.','t2 unavailable is retained with NaN vertices and reference_available=false; exclude only by availability and report9/10 coverage.','Reference nominal time vs input match residual is not a sensor synchronization upper bound.']))
    K,c2w,mx,my,cammeta=camera(1);save(O/'heldout_camera1.json',cammeta)
    heldout=O/'heldout_cam1';heldout.mkdir(exist_ok=True);sparse=O/'sparse_cam1';sparse.mkdir(exist_ok=True)
    video=O/'source'/vidmember;tp=O/'source'/timemember;ts=np.asarray(load(tp)['color'],np.float64)/1e6
    cap=cv2.VideoCapture(str(video));assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(ts)
    records=[];sparses=[]
    for t in plan['heldout_nominal_times_seconds']:
        j=int(abs(ts-t).argmin());cap.set(cv2.CAP_PROP_POS_FRAMES,j);ok,bgr=cap.read();assert ok and bgr.shape[:2]==(1536,2048)
        q=heldout/f't{t:04d}.png';assert cv2.imwrite(str(q),cv2.remap(bgr,mx,my,cv2.INTER_LINEAR))
        i=int(abs(np.asarray(m['timestamp_seconds'])-ts[j]).argmin())
        records.append(dict(nominal_time_seconds=t,camera_id=1,raw_video_frame_index=j,actual_camera1_timestamp_seconds=float(ts[j]),
            matched_input_frame_index=i,matched_input_timestamp_seconds=float(m['timestamp_seconds'][i]),input_minus_camera1_time_seconds=float(m['timestamp_seconds'][i]-ts[j]),
            path=str(q),sha256=sha(q),source=str(video),source_sha256=sha(video),source_kind='official_raw_video_frame'))
        sp=O/'source'/seq/f't{t:04d}.000/k1.color.jpg'
        if sp.exists():
            raw=cv2.imread(str(sp));assert raw.shape[:2]==(1536,2048);sq=sparse/f't{t:04d}.png';assert cv2.imwrite(str(sq),cv2.remap(raw,mx,my,cv2.INTER_LINEAR))
            sparses.append(dict(nominal_time_seconds=t,path=str(sq),sha256=sha(sq),source=str(sp),source_sha256=sha(sp),timestamp_note='Nominal archive time only; actual RGB timestamp is not inferred from raw video without image identity matching.'))
    cap.release()
    np.savez_compressed(O/'heldout_camera1.npz',K=K,c2w=c2w,w2c=np.linalg.inv(c2w),nominal_times_seconds=np.array(plan['heldout_nominal_times_seconds']),
        actual_camera1_times_seconds=np.array([r['actual_camera1_timestamp_seconds'] for r in records]),matched_input_indices=np.array([r['matched_input_frame_index'] for r in records]),
        matched_input_times_seconds=np.array([r['matched_input_timestamp_seconds'] for r in records]),frame_paths=np.array([r['path'] for r in records]),role=np.array('evaluation_only'))
    save(O/'heldout_manifest.json',dict(role='evaluation_only',status='five_predeclared_times_ready',records=records,camera=cammeta,
        nominal_request_times=plan['heldout_nominal_times_seconds'],raw_video_timestamp_source=str(tp),raw_video_timestamp_sha256=sha(tp),
        source_protocol=addendum['image_source_protocol'],sparse_archive_records=sparses,model_predictions_read=False,alignment_performed=False,
        limitation='Cross-camera actual recorded timestamp matching is reported, but calibration and sensor-level synchronization have not been independently recalibrated.'))
    assert sha(plan['input_manifest'])==original_hash
    state.update(status='completed',elapsed_seconds=time.perf_counter()-begin,reference_meshes=str(ref),reference_meshes_sha256=sha(ref),available_reference_times=int(v.sum()),missing_reference_times=times[~v].tolist(),
        heldout_images=len(records),sparse_archive_heldout_images=len(sparses),transferred_bytes=sum(x['transferred_bytes'] for x in state['records']),
        input_manifest_unchanged=True,script_sha256=sha(__file__),no_training_writes=True)
    save(O/'status.json',state);print(json.dumps({k:state[k] for k in ['status','elapsed_seconds','available_reference_times','missing_reference_times','heldout_images','transferred_bytes']},indent=2))

if __name__=='__main__':main()
