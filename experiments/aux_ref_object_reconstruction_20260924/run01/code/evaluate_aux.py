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
 (out/'prepare_code_snapshot.py').write_bytes(Path(__file__).read_bytes());run['code']={**identity(out/'prepare_code_snapshot.py'),'original_execution_path':str(Path(__file__).resolve())}
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
   results.append({'dev':dev,'sample_id':sid,'frame_id':frameid,'nominal_sample_time_seconds':r['native_nominal_time_seconds'],'query_time_seconds':r['native_nominal_time_seconds'],'query_time_basis':'official capture-sample nominal directory time; not asserted exact camera exposure time','camera0_native_actual_time_seconds':None,'camera1_native_actual_time_seconds':None,'official_sync_error_seconds':None,'sync_limitation':'Native same-archive same-sample association established; exact exposure timestamps/sensor sync error not supplied by this sample. Old nearest-frame input-camera1 deltas are not reused.','native_camera1':identity(rawpath),'native_camera1_source':identity(str(rawpath)+'.source.json'),'same_capture_archive':{'url':rawsrc['url'],'etag':rawsrc['etag']},'rgb':identity(gtpath),'regions':identity(maskpath),'overlay':identity(overlaypath),'evaluation_fits':fitrecords,'pixel_counts':{name:int((labels==i).sum()) for i,name in enumerate(['background','human','object'])},'input_S_row_identity':{'new_frame_id':r['new_frame_id'],'candidate_S':True,'pred_evaluation':r.get('pred_evaluation')},'published_object_parameters':r['native_assets']['object_parameters'],'official_parameter_conversion_check':r['official_parameter_conversion_check']})
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
    for metric in ['psnr_db','ssim','lpips_spatial_mean']:
     vals=[r['metrics'][region].get(metric) for r in cc if r['metrics'][region].get(metric) is not None];cells[cell][region][metric]={'mean':float(np.mean(vals)) if vals else None,'median':float(np.median(vals)) if vals else None,'valid_frames':len(vals),'requested_frames':len(ids)}
  dd={}
  for name,(a,b) in definitions.items():
   dd[name]={}
   for metric in ['psnr_db','ssim','lpips_spatial_mean']:
    vals=[]
    for sid in ids:
     va=matrix[(sid,a)]['metrics']['object'].get(metric);vb=matrix[(sid,b)]['metrics']['object'].get(metric);value=None if va is None or vb is None else va-vb
     differences.append({'protocol_id':'AUX_REF_OBJECT','dev':dev,'sample_id':sid,'region':'object','difference':name,'formula':f'Y_{a}-Y_{b}','metric':metric,'value':value})
     if value is not None:vals.append(value)
    dd[name][metric]={'mean':float(np.mean(vals)) if vals else None,'median':float(np.median(vals)) if vals else None,'valid_frames':len(vals),'requested_frames':len(ids),'formula':f'Y_{a}-Y_{b}'}
  p=dd['learned_ref']['psnr_db'];s=dd['learned_ref']['ssim'];complete=p['valid_frames']==len(ids) and s['valid_frames']==len(ids)
  gate={'complete_pairing':complete,'delta_psnr_mean_at_least_0p5':bool(complete and p['mean']>=.5),'delta_psnr_frame_median_positive':bool(complete and p['median']>0),'delta_ssim_mean_at_least_minus_0p005':bool(complete and s['mean']>=-.005)}
  output[dev]={'samples':ids,'cells':cells,'differences':dd,'numeric_gate':gate,'numeric_gate_passed':all(gate.values())}
 return output,differences


def checked_identity(record):
 p=Path(record['path']);assert p.is_file(),str(p);assert sha(p)==record['sha256'],f'Frozen file changed: {p}';return p


def validate_freeze(freeze,regions):
 f=read(freeze);m=read(regions)
 assert f['protocol_id']=='AUX_REF_OBJECT' and f['phase']=='four_finals_frozen'
 assert checked_identity(f['regions_manifest']).resolve()==Path(regions).resolve()
 assert m['status']=='frozen' and m['prepared_before_auxiliary_training'] is True
 assert len(f['runs'])==4
 runs={(r['dev'],r['arm']):r for r in f['runs']}
 assert set(runs)=={(d,a) for d in ('dev1','dev2') for a in ('Pred','Ref')}
 for r in f['runs']:
  assert r['step']==8000
  for key in ('checkpoint','input_manifest','reference_motion'):checked_identity(r[key])
 for dev in ('dev1','dev2'):
  a,b=runs[(dev,'Pred')],runs[(dev,'Ref')]
  for key in ('input_manifest','reference_motion'):assert a[key]['sha256']==b[key]['sha256']
  inp=read(a['input_manifest']['path']);assert inp['camera_id']==0 and inp['camera1_used_for_training'] is False and inp['human_reference_used'] is False
  native_ids={r['sample_id'] for r in inp['native_rows']};e=[r for r in m['rows'] if r['dev']==dev]
  assert len(e)>=2 and len(set(r['sample_id'] for r in e))==len(e)
  assert all(r['sample_id'] in native_ids for r in e),'E must use the same actual native samples as S'
 for r in m['rows']:
  for key in ('native_camera1','native_camera1_source','rgb','regions'):checked_identity(r[key])
  for x in r['evaluation_fits']:checked_identity(x)
 checked_identity(m['native_manifest']);checked_identity(m['code']);checked_identity(m['legacy_raster']);checked_identity(m['legacy_camera_mapping'])
 return f,m,runs


def existing_lpips(device='cpu'):
 """Use already installed LPIPS and both local weight files, never download."""
 import torch
 spec=importlib.util.find_spec('lpips')
 if spec is None:return None,{'available':False,'reason':'lpips is not installed; no installation attempted'}
 calibrated=Path(spec.origin).parent/'weights/v0.1/alex.pth'
 trunk=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
 if not calibrated.is_file() or not trunk.is_file():return None,{'available':False,'reason':'Existing AlexNet or LPIPS v0.1 weights missing; downloads forbidden'}
 try:
  import lpips
  # A random constructor cannot invoke torchvision download. Replace EVERY
  # trunk parameter from the existing official cache before any forward pass.
  model=lpips.LPIPS(net='alex',version='0.1',pnet_rand=True,spatial=True,model_path=str(calibrated),verbose=False)
  weights=torch.load(trunk,map_location='cpu',weights_only=True)
  mapped={k:weights['features.'+k.split('.',1)[1]] for k in model.net.state_dict()}
  model.net.load_state_dict(mapped,strict=True);model.eval();model.requires_grad_(False);model=model.to(device)
  return model,{'available':True,'implementation':identity(spec.origin),'calibrated_weights':identity(calibrated),'trunk_weights':identity(trunk),'all_trunk_parameters_loaded_strictly':True,'download_attempted':False,'definition':'LPIPS AlexNet v0.1 spatial map; fixed-region pixel mean. This supplementary regional spatial score is not a crop scalar LPIPS score. Lower is better.'}
 except Exception as exc:return None,{'available':False,'reason':repr(exc),'download_attempted':False,'dependencies_modified':False}


def distribution(values):
 v=np.asarray(values,dtype=np.float64).reshape(-1)
 assert len(v)>0 and np.isfinite(v).all()
 return {'count':len(v),'min':float(v.min()),'q01':float(np.quantile(v,.01)),'q10':float(np.quantile(v,.1)),'median':float(np.median(v)),'mean':float(v.mean()),'q90':float(np.quantile(v,.9)),'q99':float(np.quantile(v,.99)),'max':float(v.max())}


def object_statistics(export):
 opacity=export['opacity'];scales=export['scale_m'];offset=np.linalg.norm(export['local_offset_m'],axis=-1)
 return {'gaussian_count':len(opacity),'opacity':distribution(opacity),'scale_each_axis_m':distribution(scales),'scale_max_axis_m':distribution(scales.max(-1)),'canonical_offset_norm_m':distribution(offset),'opacity_below_0p01_fraction':float((opacity<.01).mean()),'opacity_below_0p1_fraction':float((opacity<.1).mean()),'max_axis_over_0p035m_fraction':float((scales.max(-1)>.035).mean()),'canonical_offset_at_most_5mm':bool((offset<=.005+1e-8).all()),'stable_id_unique':len(np.unique(export['stable_id']))==len(opacity),'local_support_material_matching':'Not measured; no new correspondence or material matcher.'}


def region_masks(labels):
 return {'full':np.ones(labels.shape,bool),'object':labels==2,'human':labels==1,'background':labels==0,'foreground':labels>0,'object_interior3px':cv2.erode((labels==2).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)}


def write_csv(path,rows):
 if not rows:return
 with Path(path).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def score(freeze,regions,output,representation_checks=None):
 """Render the frozen four-cell comparison. No optimizer or backward exists."""
 # Every final file is verified before CUDA initialization or RGB scoring.
 f,m,runs=validate_freeze(freeze,regions)
 import torch
 assert os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,''),'Explicit GPU binding is required'
 assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose exactly one scheduled evaluation GPU'
 expected=os.environ.get('AUX_EXPECTED_GPU','NVIDIA GeForce RTX 3090');actual=torch.cuda.get_device_name(0)
 assert actual==expected,(actual,expected)
 from aux_scene import AuxObjectScene,state_identity
 out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 (out/'evaluate_code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 run={'protocol_id':'AUX_REF_OBJECT','status':'running','started_utc':utc(),'freeze':identity(freeze),'regions_manifest':identity(regions),'code':identity(out/'evaluate_code_snapshot.py'),'aux_scene':identity(Path(__file__).parent/'aux_scene.py'),'hardware':{'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'GPU':actual,'torch':torch.__version__},'motion_update':False,'optimizer_steps':0,'role':'Auxiliary stronger-input conditional object reconstruction; excluded from P0 RGB main results'}
 save(out/'run.json',run);allrows=[];repr_stats={};hsrecords=[]
 try:
  torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();lpips_model,lpips_info=existing_lpips('cpu');save(out/'lpips_availability.json',lpips_info)
  K=torch.tensor(m['camera']['K'],dtype=torch.float32,device='cuda');w2c=torch.tensor(m['camera']['w2c'],dtype=torch.float32,device='cuda');H=m['camera']['height'];W=m['camera']['width']
  with torch.no_grad():
   for dev in ('dev1','dev2'):
    erows=sorted([r for r in m['rows'] if r['dev']==dev],key=lambda r:r['query_time_seconds']);qt=np.asarray([r['query_time_seconds'] for r in erows],np.float64)
    ref=np.load(runs[(dev,'Ref')]['reference_motion']['path']);ids=[]
    for q in qt:
     hit=np.flatnonzero(ref['times']==q);assert len(hit)==1,'Ref must be exactly the same official native sample; interpolation/nearest assignment forbidden';ids.append(int(hit[0]))
    RR=np.asarray(ref['R_world'][ids]);tt=np.asarray(ref['t_world'][ids]);assert RR.shape==(len(qt),3,3) and tt.shape==(len(qt),3)
    assert np.max(np.abs(RR@RR.transpose(0,2,1)-np.eye(3)))<1e-5 and np.max(np.abs(np.linalg.det(RR)-1))<1e-5
    common_frozen=None;repr_stats[dev]={};devout=out/dev;devout.mkdir()
    for arm in ('Pred','Ref'):
     r=runs[(dev,arm)];ck=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=False)
     assert ck['protocol_id']=='AUX_REF_OBJECT' and ck['dev']==dev and ck['arm']==arm and ck['step']==8000
     scene=AuxObjectScene(dev,qt,device='cuda');scene.obank.resize_for_load(ck['obank'],'');scene.obank.load_state_dict(ck['obank'],strict=True);scene.obank.to('cuda');scene.eval();scene.assert_frozen()
     before=state_identity(scene.obank.state_dict());assert before==state_identity(ck['obank'])
     if common_frozen is None:common_frozen=scene.frozen_identity
     else:assert common_frozen==scene.frozen_identity,'Four cells must share the same frozen human/background and motion arrays'
     export=scene.object_canonical_export();exportpath=devout/f'{arm}_actual_object_gaussians.npz';np.savez_compressed(exportpath,**export)
     repr_stats[dev][arm]={'checkpoint':r['checkpoint'],'actual_object_gaussians':identity(exportpath),'statistics':object_statistics(export),'effective_data_gradient_steps':ck.get('effective_data_gradient_steps'),'scene_identity':ck.get('scene_identity')}
     np.savez_compressed(devout/f'{arm}_frozen_motion_conditions.npz',query_times=qt,pred_R_world=scene.pred_R_world.cpu().numpy(),pred_t_world=scene.pred_t_world.cpu().numpy(),ref_R_world=RR,ref_t_world=tt)
     for qi,e in enumerate(erows):
      gt=cv2.cvtColor(cv2.imread(e['rgb']['path']),cv2.COLOR_BGR2RGB).astype(np.float64)/255;labels=np.load(e['regions']['path'])['entity_labels'];fixedO=labels==2;denom=int(fixedO.sum());assert denom==e['pixel_counts']['object'] and denom>0
      sampleout=devout/e['frame_id'];sampleout.mkdir(exist_ok=True)
      if arm=='Pred':
       hs=scene.render_hs_only(qi,K,w2c,H,W);hsrgb=hs['rgb'].detach().cpu().numpy().transpose(1,2,0);hsalpha=hs['alpha'].detach().cpu().numpy().squeeze();hsdepth=hs['dep'].detach().cpu().numpy().squeeze();hspath=sampleout/'frozen_HS_only.npz'
       np.savez_compressed(hspath,rgb=np.clip(hsrgb,0,1),alpha=hsalpha,depth_camera_z_m=hsdepth,transmittance_through_all_HS=1-hsalpha)
       cv2.imwrite(str(sampleout/'frozen_HS_only.png'),cv2.cvtColor(np.rint(np.clip(hsrgb,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
       hsrecords.append({'dev':dev,'sample_id':e['sample_id'],'export':identity(hspath),'fixed_object_pixels':denom,'fixed_object_HS_RGB_metrics':region_metrics(hsrgb,gt,labels)['object'],'HS_alpha_in_fixed_O':distribution(hsalpha[fixedO]),'depth_definition':'Normalized expected HS-only depth. Through-all-HS transmittance is not transmittance before the object depth. The latter and data-gradient coverage are logged separately by training.'})
      for motion in ('Pred','Ref'):
       cell=('P' if arm=='Pred' else 'R')+('P' if motion=='Pred' else 'R');rot=scene.pred_R_world[qi] if motion=='Pred' else RR[qi];trans=scene.pred_t_world[qi] if motion=='Pred' else tt[qi]
       ret,parts=scene.render(qi,K,w2c,H,W,object_R_world=rot,object_t_world=trans);rgb=ret['rgb'].detach().cpu().numpy().transpose(1,2,0);buf=ret['buf'].detach().cpu().numpy();alpha=ret['alpha'].detach().cpu().numpy().squeeze();depth=ret['dep'].detach().cpu().numpy().squeeze()
       assert rgb.shape==(H,W,3) and buf.shape==(3,H,W) and np.isfinite(rgb).all() and np.isfinite(buf).all()
       metrics=region_metrics(rgb,gt,labels)
       if lpips_model is not None:
        px=torch.from_numpy(np.clip(rgb,0,1).transpose(2,0,1).copy()).float()[None];gx=torch.from_numpy(gt.transpose(2,0,1).copy()).float()[None];lmap=lpips_model(px,gx,normalize=True).cpu().numpy().squeeze();assert lmap.shape==(H,W) and np.isfinite(lmap).all()
        for name,mask in region_masks(labels).items():metrics[name]['lpips_spatial_mean']=float(lmap[mask].mean()) if mask.any() else None
       else:
        for metric in metrics.values():metric['lpips_spatial_mean']=None
       predpath=sampleout/f'{cell}.png';cv2.imwrite(str(predpath),cv2.cvtColor(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR));datapath=sampleout/f'{cell}_render_arrays.npz';np.savez_compressed(datapath,rgb_clipped=np.clip(rgb,0,1),entity_contribution_SHO=buf,alpha=alpha,expected_depth_camera_z_m=depth)
       contribution={'fixed_object_pixels':denom,'visible_object_contribution_mean':float(buf[2][fixedO].mean()),'visible_object_contribution_sum':float(buf[2][fixedO].sum()),'coverage_above_0p01':float((buf[2][fixedO]>.01).mean()),'coverage_above_0p1':float((buf[2][fixedO]>.1).mean()),'coverage_above_0p5':float((buf[2][fixedO]>.5).mean()),'definition':'Actual joint depth-sorted alpha contribution of object on the unchanged fixed evaluation O region.'}
       allrows.append({'protocol_id':'AUX_REF_OBJECT','dev':dev,'sample_id':e['sample_id'],'query_time_seconds':float(qt[qi]),'cell':cell,'trained_representation':arm,'render_motion':motion,'checkpoint':r['checkpoint'],'metrics':metrics,'contribution':contribution,'rgb_export':identity(predpath),'float_render_export':identity(datapath)})
       save(out/'per_frame.partial.json',allrows)
      del ret,parts
     assert state_identity(scene.obank.state_dict())==before,'Evaluation changed actual Gaussian state';scene.assert_frozen();del scene,ck;torch.cuda.empty_cache()
   torch.cuda.synchronize()
  summaries,differences=summarize_cross_rows(allrows)
  numeric=all(x['numeric_gate_passed'] for x in summaries.values());external=read(representation_checks) if representation_checks else None
  required=('no_transparency','no_kernel_inflation','no_input_leakage','frozen_scene_comparison_valid')
  checks=(external or {}).get('checks',{});passed=lambda value:value is True or isinstance(value,dict) and value.get('passed') is True
  repr_ok=all(passed(checks.get(k)) for k in required)
  decision={'engineering_numeric_gate_passed_both_events':numeric,'required_representation_checks':{k:checks.get(k) for k in required},'representation_evidence_review_complete':repr_ok,'consistent_representation_difference_worth_further_study':bool(numeric and repr_ok),'status':'threshold_met_with_required_checks' if numeric and repr_ok else ('numeric_threshold_met_pending_representation_review' if numeric else 'numeric_threshold_not_met'),'not_a_method_promotion':True,'not_a_significance_test':True,'no_additional_training_authorized':True}
  save(out/'per_frame.json',allrows);save(out/'five_differences.json',differences);save(out/'summary.json',{'protocol_id':'AUX_REF_OBJECT','input_boundary':run['role'],'per_dev':summaries,'decision_8p1':decision,'LPIPS':lpips_info,'difference_interpretation':'Algebraic comparisons of paired render conditions; not independent causal contributions or percentages. Positive PSNR/SSIM and negative LPIPS are favorable.','fixed_scene_limitation':'Old frozen H/S may retain object traces and occlude Ref geometry. Failure is not evidence against the potential of the complete HOI reconstruction backend.'})
  save(out/'representation_exports.json',{'protocol_id':'AUX_REF_OBJECT','per_dev':repr_stats,'frozen_HS':hsrecords,'external_representation_checks':identity(representation_checks) if representation_checks else None,'unmeasured':'Material correspondence/local support is not measured; no new matcher was built.'})
  flat=[]
  for r in allrows:
   for reg,v in r['metrics'].items():flat.append({'protocol_id':'AUX_REF_OBJECT','dev':r['dev'],'sample_id':r['sample_id'],'cell':r['cell'],'region':reg,**v})
  write_csv(out/'per_frame.csv',flat);write_csv(out/'five_differences.csv',differences)
  validate_freeze(freeze,regions)
  save(out/'run.json',{**run,'status':'complete','completed_utc':utc(),'seconds':time.perf_counter()-started,'rows':len(allrows),'expected_rows':4*len(m['rows']),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'all_frozen_inputs_reverified_after_evaluation':True,'decision_8p1':decision})
  print(json.dumps({'status':'complete','output':str(out),'rows':len(allrows),'decision_8p1':decision}),flush=True)
 except BaseException:
  save(out/'run.json',{**run,'status':'failed','traceback':traceback.format_exc(),'seconds':time.perf_counter()-started,'completed_rows':len(allrows)});raise


def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('prepare_regions');q.add_argument('--native-manifest',type=Path,default=E/'protocol/native_availability.json');q.add_argument('--output',type=Path)
 q=sub.add_parser('score');q.add_argument('--freeze',type=Path,required=True);q.add_argument('--regions',type=Path,default=E/'evaluation/regions/manifest.json');q.add_argument('--output',type=Path,default=E/'evaluation/cross_pose');q.add_argument('--representation-checks',type=Path)
 a=p.parse_args()
 if a.command=='prepare_regions':prepare_regions(a.native_manifest,a.output)
 else:score(a.freeze,a.regions,a.output,a.representation_checks)
if __name__=='__main__':main()
