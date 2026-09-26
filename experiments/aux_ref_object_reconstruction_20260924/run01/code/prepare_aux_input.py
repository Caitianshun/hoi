"""AUX-only native RGB preprocessing and same-rule RGB SAM2 masks. No human fits."""
from pathlib import Path
import sys,json,hashlib,argparse,subprocess,datetime
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
ROOT=Path('/home/cai_tianshun/Project/HOI');E=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/structured_hoi_20260923'
sys.path.insert(0,str(OLD/'data_audit'))
from prepare_dev2_evaluation import camera

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
def prepare(dev):
    a=json.loads((E/'protocol/native_availability.json').read_text());rows=[r for r in a['rows'] if r['dev']==dev and r['candidate_S']]
    assert len(rows)>=3 and all(r['native_S_assets_ready'] for r in rows),'Native asset gate not ready'
    rows.sort(key=lambda x:x['native_nominal_time_seconds']);out=E/f'inputs/{dev}';out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'experiments/mosca_baseline_20260922/common_input' if dev=='dev1' else OLD/'data/dev2'
    oldmeta=json.loads((source/'input_manifest.json').read_text())
    seg=source.parent/'segmentation/segmentation_run.json' if dev=='dev1' else source/'segmentation/segmentation_run.json'
    oldseg=json.loads(seg.read_text());K,C,mx,my,cal=camera(0)
    assert np.allclose(K,oldmeta['K'],atol=1e-8) and np.allclose(C,oldmeta['c2w'],atol=1e-8)
    native=[];RR=[];tt=[];paths=[]
    for j,r in enumerate(rows):
        p=Path(r['native_assets']['camera0']['path']);assert sha(p)==r['native_assets']['camera0']['sha256']
        bgr=cv2.imread(str(p));assert bgr.shape[:2]==(1536,2048)
        q=out/'rgb_native'/f'{j:05d}.png';q.parent.mkdir(exist_ok=True);assert cv2.imwrite(str(q),cv2.remap(bgr,mx,my,cv2.INTER_LINEAR))
        conv=r['official_parameter_conversion_check'];assert conv['official_assertion_sse_lt_1e_8']
        RR.append(Rotation.from_rotvec(conv['angle_rotvec']).as_matrix());tt.append(conv['translation_m']);paths.append(str(q))
        native.append(dict(index=j,sample_id=r['official_capture_sample_id'],query_time=r['native_nominal_time_seconds'],clock='official nominal sample time; actual exposure unknown',raw_image=str(p),raw_sha256=sha(p),image=str(q),image_sha256=sha(q),object_parameters=r['native_assets']['object_parameters'],reference_version='Date01.zip fit01',E_candidate=r['candidate_E']))
    # Reuse original RGB first frame and exact original prompt; add native images in
    # temporal order to original RGB context, never prompt from a reference pose.
    merged=[dict(time=float(t),path=p,source='old_RGB',index=i) for i,(t,p) in enumerate(zip(oldmeta['timestamp_seconds'],oldmeta['frame_paths']))]
    merged += [dict(time=x['query_time'],path=x['image'],source='native_RGB',index=x['index']) for x in native]
    merged.sort(key=lambda x:(x['time'],x['source']=='native_RGB'));assert merged[0]['path']==oldmeta['frame_paths'][0]
    frames=out/'sam_context_rgb';frames.mkdir()
    for i,x in enumerate(merged):
        dest=frames/f'{i:06d}{Path(x["path"]).suffix}';dest.symlink_to(x['path']);x['context_index']=i
        if x['source']=='native_RGB':native[x['index']]['segmentation_context_index']=i
    assert sha(merged[0]['path'])==oldseg['prompts']['first_frame_sha256']
    save(out/'sam_prompt.json',oldseg['prompts']);save(out/'sam_context_manifest.json',{'source_old_input':str(source/'input_manifest.json'),'source_old_segmentation':str(seg),'frames':merged,'reference_used':False,'all_context_RGB_not_additional_training_frames':True})
    np.savez_compressed(out/'reference_object_motion.npz',R_world=np.asarray(RR),t_world=np.asarray(tt),times=np.asarray([r['query_time'] for r in native]),protocol_id=np.array('AUX_REF_OBJECT'),source=np.array('published_fit_actual_native_samples_no_interpolation'))
    meta={'protocol_id':'AUX_REF_OBJECT','dev':dev,'phase':'RGB_prepared_mask_pending','role':'aux_input_only_camera0_plus_object_fit','P0_modified':False,'camera_id':0,'sequence':oldmeta['sequence'],'frame_paths':paths,'frame_sha256':[sha(p) for p in paths],'timestamp_seconds':[r['query_time'] for r in native],'K':K.tolist(),'c2w':C.tolist(),'height':480,'width':640,'calibration':cal,'native_rows':native,'reference_object_motion':str(out/'reference_object_motion.npz'),'reference_motion_sha256':sha(out/'reference_object_motion.npz'),'human_reference_used':False,'camera1_used_for_training':False,'segmentation_source':'same frozen SAM2, unchanged original RGB first-frame prompts; merged RGB context; no fit masks'}
    save(out/'input_manifest.json',meta)
    cmd=[sys.executable,str(ROOT/'scripts/diagnostics/segment_event_rgb.py'),'--frames',str(frames),'--boxes',str(out/'sam_prompt.json'),'--sam2-root',str(ROOT/'third_party/sam2'),'--checkpoint',oldseg['checkpoint'],'--model-config',oldseg['config'],'--output',str(out/'sam_context_output'),'--device','cuda:0','--precision',oldseg['precision']]
    save(out/'sam_command.json',{'command':cmd,'CUDA_VISIBLE_DEVICES':'1','source_checkpoint_sha256':oldseg['checkpoint_sha256'],'source_config_sha256':oldseg['config_sha256'],'source_adapter_sha256':oldseg['adapter_sha256']})
    assert sha(oldseg['checkpoint'])==oldseg['checkpoint_sha256'] and sha(ROOT/'scripts/diagnostics/segment_event_rgb.py')==oldseg['adapter_sha256']
    return cmd

def finish(dev):
    out=E/f'inputs/{dev}';meta=json.loads((out/'input_manifest.json').read_text());r=json.loads((out/'sam_context_output/segmentation_run.json').read_text());assert r['status']=='completed' and r['reference_geometry_used'] is False
    source=np.load(out/'sam_context_output/segmentation.npz')['entity_labels'];ids=[x['segmentation_context_index'] for x in meta['native_rows']];labels=source[ids]
    np.savez_compressed(out/'segmentation.npz',entity_labels=labels,source_frame_ids=np.array([x['sample_id'] for x in meta['native_rows']]),role=np.array('AUX_only_RGB_estimated_native_masks_no_fit_labels'))
    for i,row in enumerate(meta['native_rows']):row['object_pixels']=int((labels[i]==2).sum());row['mask_sha256']=hashlib.sha256(labels[i].tobytes()).hexdigest()
    meta.update(phase='complete',segmentation=str(out/'segmentation.npz'),segmentation_sha256=sha(out/'segmentation.npz'),mask_frame_count=len(labels));save(out/'input_manifest.json',meta)
    print(dev,'prepared',len(labels),'native training frames')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','segment','finish']);ap.add_argument('--dev',required=True,choices=['dev1','dev2']);a=ap.parse_args()
    if a.action=='prepare':print(prepare(a.dev))
    elif a.action=='segment':
        out=E/f'inputs/{a.dev}';cmd=json.loads((out/'sam_command.json').read_text())['command']
        with (out/'segmentation.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        finish(a.dev)
    else:finish(a.dev)
