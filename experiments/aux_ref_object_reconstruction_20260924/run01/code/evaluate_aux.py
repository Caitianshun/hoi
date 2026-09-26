"""AUX_REF_OBJECT evaluation-only native regions and frozen four-cell rendering.

prepare_regions is CPU-only and runs before any new auxiliary training. It uses
native archive camera1 images and same-sample historical fit01/fit02 solely to
make evaluation regions with the existing raster. No evaluation RGB/fit02 is an
auxiliary training input. score requires all four final checkpoints frozen.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
import cv2
ROOT=Path('/home/cai_tianshun/Project/HOI')
E=Path(__file__).resolve().parents[1]
LEGACY=ROOT/'experiments/structured_hoi_20260923'
VARIANTS=('PP','PR','RP','RR')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x

def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(plain(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def identity(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def fit_root(dev):
 return ROOT/'data/BEHAVE/evaluation_only/sparse_event_reference' if dev=='dev1' else LEGACY/'data_audit/dev2_evaluation/source'

def prepare_regions(native_manifest,output=None):
 """Freeze all available predeclared native E regions before training."""
 import trimesh
 out=Path(output) if output else E/'evaluation/regions'
 if any(E.glob('dev*/**/checkpoint*.pt')):raise RuntimeError('regions must be frozen before new auxiliary checkpoints exist')
 native_manifest=Path(native_manifest);native=read(native_manifest);rows=[r for r in native['rows'] if r.get('candidate_E')]
 assert len(rows)>0 and all(r.get('candidate_S') for r in rows),'E must be a subset of S'
 assert all(sum(r['dev']==d for r in rows)>=2 for d in ['dev1','dev2'])
 original_raster=LEGACY/'code/evaluate_heldout_regions_cpu.py';camera_code=LEGACY/'data_audit/prepare_dev2_evaluation.py'
 raster=load_module('aux_existing_region_raster',original_raster);camera=load_module('aux_existing_camera_mapping',camera_code)
 checks=raster.raster_checks();K,C,mx,my,cammeta=camera.camera(1);H,W=480,640
 for dev in ['dev1','dev2']:
  old=read(LEGACY/f'data_audit/heldout_regions/{dev}/mask_manifest.json')['camera']
  assert np.array_equal(K,np.asarray(old['K'])) and np.array_equal(C,np.asarray(old['c2w'])),'same frozen evaluation projection required'
 out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();results=[];run={'status':'preparing','protocol_id':'AUX_REF_OBJECT','role':'evaluation_only','started_utc':utc(),'native_manifest':identity(native_manifest),'code':identity(__file__),'legacy_raster':identity(original_raster),'legacy_camera_mapping':identity(camera_code),'raster_checks':checks,'GPU_used':False}
 save(out/'run.json',run)
 try:
  for r in sorted(rows,key=lambda r:(r['dev'],r['native_nominal_time_seconds'])):
   dev=r['dev'];sid=r['official_capture_sample_id'];seq,frameid=sid.split('/');assert frameid.startswith('t')
   native_image=r['native_assets']['camera1'];rawpath=Path(native_image['path']);assert sha(rawpath)==native_image['sha256']
   rawsrc=read(str(rawpath)+'.source.json');assert rawsrc['member']==sid+'/k1.color.jpg';assert rawsrc['sha256']==sha(rawpath)
   sample=fit_root(dev)/sid;objname='boxsmall' if dev=='dev1' else 'chair';fits=[sample/'person/fit02/person_fit.ply',sample/f'{objname}/fit01/{objname}_fit.ply'];parts=[];fitrecords=[]
   for entity,p in enumerate(fits,1):
    meta=read(str(p)+'.source.json');assert meta['member'].startswith(sid+'/');assert meta['url']==rawsrc['url'] and meta['etag']==rawsrc['etag'];assert sha(p)==meta['sha256']
    mesh=trimesh.load(str(p),process=False);parts.append((entity,np.asarray(mesh.vertices),np.asarray(mesh.faces)));fitrecords.append({**identity(p),'archive_member':meta['member'],'source':identity(str(p)+'.source.json'),'role':'evaluation_region_only'})
   raw=cv2.imread(str(rawpath));assert raw is not None and raw.shape[:2]==(1536,2048);gt=cv2.remap(raw,mx,my,cv2.INTER_LINEAR)
   labels,depth=raster.camera_mask(parts,K,C,H,W);assert labels.shape==(H,W) and int((labels==2).sum())>0
   d=out/dev/frameid;d.mkdir(parents=True,exist_ok=False);gtpath=d/'native_camera1_rectified.png';maskpath=d/'reference_regions.npz';cv2.imwrite(str(gtpath),gt)
   np.savez_compressed(maskpath,entity_labels=labels,depth_camera_z_m=depth.astype(np.float32),protocol_id=np.array('AUX_REF_OBJECT'),role=np.array('evaluation_only_same_native_capture_sample'))
   overlay=gt.copy();palette=np.array([[0,0,0],[80,210,70],[40,100,240]],np.uint8);overlay[labels>0]=np.rint(.6*overlay[labels>0]+.4*palette[labels[labels>0]]).astype(np.uint8);overlaypath=d/'native_region_overlay.png';cv2.imwrite(str(overlaypath),np.concatenate([gt,overlay],axis=1))
   results.append({'dev':dev,'sample_id':sid,'frame_id':frameid,'nominal_sample_time_seconds':r['native_nominal_time_seconds'],'query_time_seconds':r['native_nominal_time_seconds'],'query_time_basis':'official capture-sample nominal directory time; not asserted exact camera exposure time','camera0_native_actual_time_seconds':None,'camera1_native_actual_time_seconds':None,'official_sync_error_seconds':None,'sync_limitation':'Native same-archive same-sample association established; exact exposure timestamps/sensor sync error not supplied by this sample. Old nearest-frame input-camera1 deltas are not reused.','native_camera1':identity(rawpath),'native_camera1_source':identity(str(rawpath)+'.source.json'),'same_capture_archive':{'url':rawsrc['url'],'etag':rawsrc['etag']},'rgb':identity(gtpath),'regions':identity(maskpath),'overlay':identity(overlaypath),'evaluation_fits':fitrecords,'pixel_counts':{name:int((labels==i).sum()) for i,name in enumerate(['background','human','object'])},'input_S_row_identity':{'new_frame_id':r['new_frame_id'],'candidate_S':True,'pred_evaluation':r.get('pred_evaluation')},'published_object_parameters':native.get('unused',None)})
  manifest={**run,'status':'frozen','frozen_utc':utc(),'prepared_before_auxiliary_training':True,'rows':results,'camera':{'K':K,'c2w':C,'w2c':np.linalg.inv(C),'height':H,'width':W,'metadata':cammeta},'image_count':len(results),'counts_by_dev':{dev:sum(r['dev']==dev for r in results) for dev in ['dev1','dev2']},'metric_definition':'All four cells use same camera1 images and same fixed object region; RGB renderer clip[0,1] versus uint8 native remap/255. PSNR from regional RGB MSE; SSIM is region average of full-image standard 7x7 uniform SSIM map. No dynamic mask shrink, alignment, fitted color correction or output-driven sample selection.','region_limitation':'Approximate labels from same-sample published human fit02/object fit01 with original camera z-buffer; fit/cloth/hair errors and background occlusion remain. Native group identity does not assert zero synchronization error.','LPIPS':'Pending read-only availability check at score; no dependency/weight download allowed.','seconds':time.perf_counter()-started}
  save(out/'manifest.json',manifest);save(out/'run.json',{**run,'status':'complete','manifest':identity(out/'manifest.json'),'seconds':time.perf_counter()-started});print(json.dumps({'status':'frozen','manifest':str(out/'manifest.json'),'image_count':len(results),'counts_by_dev':manifest['counts_by_dev']}),flush=True);return manifest
 except BaseException:
  save(out/'run.json',{**run,'status':'failed','traceback':traceback.format_exc(),'seconds':time.perf_counter()-started});raise


def region_metrics(pred,gt,labels):
 from skimage.metrics import structural_similarity
 assert pred.shape==gt.shape and pred.shape[:2]==labels.shape
 assert np.isfinite(pred).all() and np.isfinite(gt).all()
 pred=np.clip(pred.astype(np.float64),0,1);gt=gt.astype(np.float64)
 _,ss=structural_similarity(gt,pred,data_range=1.,channel_axis=2,full=True);ss=ss.mean(-1);se=((pred-gt)**2).mean(-1)
 masks={'full':np.ones(labels.shape,bool),'object':labels==2,'human':labels==1,'background':labels==0,'foreground':labels>0,'object_interior3px':cv2.erode((labels==2).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)}
 result={}
 for name,mask in masks.items():
  n=int(mask.sum());mse=float(se[mask].mean()) if n else None;result[name]={'pixels':n,'mse':mse,'psnr_db':float(-10*np.log10(max(mse,1e-12))) if n else None,'ssim':float(ss[mask].mean()) if n else None,'squared_error_sum':float(se[mask].sum()) if n else None}
 return result


def summarize_cross_rows(rows):
 """Pure CPU four-cell arithmetic, always paired by exact native sample ID."""
 output={};differences=[];definitions={'swap':('PR','PP'),'learned_ref':('RR','PR'),'total':('RR','PP'),'learned_pred':('RP','PP'),'swap_ref_representation':('RR','RP')}
 for dev in ['dev1','dev2']:
  rr=[r for r in rows if r['dev']==dev];ids=sorted(set(r['sample_id'] for r in rr));matrix={(r['sample_id'],r['cell']):r for r in rr};assert len(matrix)==4*len(ids)
  cells={}
  for cell in VARIANTS:
   cc=[matrix[(sid,cell)] for sid in ids];cells[cell]={}
   for region in ['object','full','human','background','foreground','object_interior3px']:
    cells[cell][region]={}
    for metric in ['psnr_db','ssim']:
     vals=[r['metrics'][region][metric] for r in cc if r['metrics'][region][metric] is not None];cells[cell][region][metric]={'mean':float(np.mean(vals)) if vals else None,'median':float(np.median(vals)) if vals else None,'valid_frames':len(vals),'requested_frames':len(ids)}
  dd={}
  for name,(a,b) in definitions.items():
   dd[name]={}
   for metric in ['psnr_db','ssim']:
    vals=[]
    for sid in ids:
     va=matrix[(sid,a)]['metrics']['object'][metric];vb=matrix[(sid,b)]['metrics']['object'][metric];value=None if va is None or vb is None else va-vb
     differences.append({'protocol_id':'AUX_REF_OBJECT','dev':dev,'sample_id':sid,'region':'object','difference':name,'formula':f'Y_{a}-Y_{b}','metric':metric,'value':value})
     if value is not None:vals.append(value)
    dd[name][metric]={'mean':float(np.mean(vals)) if vals else None,'median':float(np.median(vals)) if vals else None,'valid_frames':len(vals),'requested_frames':len(ids),'formula':f'Y_{a}-Y_{b}'}
  p=dd['learned_ref']['psnr_db'];s=dd['learned_ref']['ssim'];complete=p['valid_frames']==len(ids) and s['valid_frames']==len(ids)
  gate={'complete_pairing':complete,'delta_psnr_mean_at_least_0p5':bool(complete and p['mean']>=.5),'delta_psnr_frame_median_positive':bool(complete and p['median']>0),'delta_ssim_mean_at_least_minus_0p005':bool(complete and s['mean']>=-.005)}
  output[dev]={'samples':ids,'cells':cells,'differences':dd,'numeric_gate':gate,'numeric_gate_passed':all(gate.values())}
 return output,differences


def score(*args,**kwargs):
 raise RuntimeError('GPU score implementation pending final checkpoint schema; prepare_regions is ready. No scoring or GPU allocation was performed.')


def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('prepare_regions');q.add_argument('--native-manifest',type=Path,default=E/'protocol/native_availability.json');q.add_argument('--output',type=Path)
 q=sub.add_parser('score');q.add_argument('--freeze',type=Path,required=True);q.add_argument('--regions',type=Path,default=E/'evaluation/regions/manifest.json');q.add_argument('--output',type=Path,default=E/'evaluation/cross_pose');q.add_argument('--representation-checks',type=Path)
 a=p.parse_args()
 if a.command=='prepare_regions':prepare_regions(a.native_manifest,a.output)
 else:score(a.freeze,a.regions,a.output,a.representation_checks)
if __name__=='__main__':main()
