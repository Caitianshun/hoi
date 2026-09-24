"""Prepare a six-frame input-only resolution ablation; CPU, no fit assets."""
from pathlib import Path
import hashlib,json,time,shutil
import cv2
import numpy as np
ROOT=Path('/home/cai_tianshun/Project/HOI');OUT=Path(__file__).resolve().parent
BASE=ROOT/'experiments/mosca_baseline_20260922'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
 rawpath=BASE/'data/behave_event_input_manifest.json';oldpath=BASE/'common_input/input_manifest.json'
 raw=json.loads(rawpath.read_text());old=json.loads(oldpath.read_text())
 ids=[0,16,25,31,65,113];K0=np.array(raw['K']);D=np.array(raw['distortion']);Kr=np.array(old['processing']['full_rectified_K'])
 W,H=raw['width'],raw['height'];mx,my=cv2.initUndistortRectifyMap(K0,D,None,Kr,(W,H),cv2.CV_32FC1)
 labels=np.load(BASE/'segmentation/segmentation.npz')['entity_labels'];variants={};prepared={}
 for name,width in [('legacy640',640),('area640',640),('area1024',1024),('native_rectified',2048)]:
  (OUT/name/'images').mkdir(parents=True,exist_ok=True);height=round(H*width/W);s=width/W
  S=np.array([[s,0,(s-1)/2],[0,s,(s-1)/2],[0,0,1]]);K=S@Kr
  variants[name]=dict(schema_version=1,role='input_only',variant=name,width=width,height=height,K=K.tolist(),c2w=old['c2w'],
   coordinate_frame='behave_world_k1_color',units='m',images_undistorted=True,distortion=[0.]*8,
   processing='existing frozen direct bilinear remap'if name=='legacy640'else('full native rectification only'if width==2048 else 'native rectification INTER_LINEAR, then INTER_AREA reduction'),
   K_rectified_full=Kr.tolist(),pixel_mapping='u_out=s*(u_full+0.5)-0.5',common_effective_fov='same alpha=0 full-resolution K_rectified',
   frame_paths=[],frame_sha256=[],frames=[],timestamp_seconds=[],source_baseline_frame_indices=ids,source_manifest=str(rawpath),source_manifest_sha256=sha(rawpath))
 for j,i in enumerate(ids):
  src=Path(raw['frame_paths'][i]);assert sha(src)==raw['frame_sha256'][i];image=cv2.imread(str(src));native=cv2.remap(image,mx,my,cv2.INTER_LINEAR)
  for name,v in variants.items():
   dst=OUT/name/'images'/f'{i:05d}.png'
   if name=='legacy640':
    assert sha(old['frame_paths'][i])==old['frame_sha256'][i];shutil.copy2(old['frame_paths'][i],dst)
   else:
    result=native if v['width']==2048 else cv2.resize(native,(v['width'],v['height']),interpolation=cv2.INTER_AREA)
    assert cv2.imwrite(str(dst),result)
   ys,xs=np.where(labels[i]==2);bbox=None
   if len(xs):
    r=v['width']/640;bbox=dict(role='same frozen640 SAM2 ROI mapped for comparison; not a new segmentation or GT',
      width=(int(xs.max())-int(xs.min())+1)*r,height=(int(ys.max())-int(ys.min())+1)*r,mask_area_equivalent=len(xs)*r*r,
      source_bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)])
   row=dict(index=j,source_baseline_index=i,raw_video_index=raw['frame_indices'][i],timestamp_seconds=old['timestamp_seconds'][i],path=str(dst),sha256=sha(dst),source_rgb=str(src),source_sha256=sha(src),mapped_predicted_object_roi=bbox)
   v['frames'].append(row);v['frame_paths'].append(str(dst));v['frame_sha256'].append(row['sha256']);v['timestamp_seconds'].append(row['timestamp_seconds'])
 for name,v in variants.items():
  p=OUT/name/'input_manifest.json';p.write_text(json.dumps(v,indent=2)+'\n');prepared[name]=dict(path=str(p),sha256=sha(p),width=v['width'],height=v['height'])
 assert np.allclose(variants['area640']['K'],old['K'],atol=1e-10)
 configs=[dict(name='legacy640_level3',input_variant='legacy640',resolution_level=3),dict(name='area640_level3',input_variant='area640',resolution_level=3),dict(name='area1024_level3',input_variant='area1024',resolution_level=3),dict(name='area1024_level9',input_variant='area1024',resolution_level=9)]
 for c in configs:
  c['input_manifest']=prepared[c['input_variant']]['path'];c['input_manifest_sha256']=prepared[c['input_variant']]['sha256']
 plan=dict(status='cpu_inputs_prepared_gpu_pending',script_sha256=sha(__file__),frozen_before_prediction=True,seed=12345,
  selected_baseline_indices=ids,inputs=prepared,configs=configs,maximum_frames_per_config=6,maximum_configs=4,
  model='UniDepthV2_vitl14',checkpoint=str(ROOT/'models/UniDepth/unidepth-v2-vitl14.bin'),
  checkpoint_sha256='101d4a941854e7f7ec6839a5a39bfe35bd5cc59ed65d8c5994a55607d8481541',
  inference_precision='FP32 to match baseline; no outer autocast',reference_read_before_prediction=False,
  confidence_policy='Preserve raw error-related confidence; not probability, not higher-is-better certainty, no threshold selection with reference.',
  elapsed_cpu_seconds=time.perf_counter()-start)
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 print(json.dumps({'prepared':prepared,'configurations':configs,'seconds':plan['elapsed_cpu_seconds']},indent=2))
if __name__=='__main__':main()
