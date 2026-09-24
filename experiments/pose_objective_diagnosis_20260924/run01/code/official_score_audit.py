"""Input-only official MegaPose scoring audit; no reference loaders or path selection.
Prepare and hash the finite pose pool before scoring. Existing MegaPose assets read-only.
"""
import argparse, csv, datetime, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path('/home/cai_tianshun/Project/HOI')
OLD=ROOT/'experiments/object_pose_refinement_20260924/run01'
NEW=ROOT/'experiments/pose_objective_diagnosis_20260924/run01'
MP=OLD/'megapose'
OUT=NEW/'score_audit/official'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rel(p):return str(Path(p).resolve())
def get_old(dev):return ROOT/'experiments/structured_hoi_20260923'/('object_init/run03/object_init.npz' if dev=='dev1' else 'object_init/dev2_refined01/object_init.npz')
def source(p):return {'path':rel(p),'sha256':sha(p)}
def prepare(tag,extra=None):
 import numpy as np
 from scipy.spatial.transform import Rotation
 dest=OUT/tag;dest.mkdir(parents=True,exist_ok=False)
 rows=[];poses=[];ledger={};assets={}
 def asset(p):assets[rel(p)]=sha(p)
 def add(dev,frame,R,t,category,path_id,stage,pose_source,variant='base',**extra_fields):
  T=np.eye(4,dtype=np.float64);T[:3,:3]=R;T[:3,3]=t
  assert np.isfinite(T).all() and abs(np.linalg.det(R)-1)<1e-4
  idx=len(rows);rows.append(dict(row_id=idx,dev=dev,frame=int(frame),category=category,path_id=path_id,stage=stage,variant=variant,source=rel(pose_source),**extra_fields));poses.append(T)
 def variants(dev,frames,data,pool_id,p,center):
  for f in frames:
   R=data['R_camera'][f].astype(float);t=data['t_camera_m'][f].astype(float);c=R@center+t
   for eps in [-.10,-.05,-.02,0.,.02,.05,.10]:add(dev,f,R,(1+eps)*c-R@center,'perturbation',pool_id,'frozen_pose',p,variant=f'depth_{eps:+.2f}',depth_fraction=eps)
   for ax in range(3):
    for deg in [-15,15]:
     vec=np.zeros(3);vec[ax]=np.radians(deg);rr=R@Rotation.from_rotvec(vec).as_matrix()
     add(dev,f,rr,c-rr@center,'perturbation',pool_id,'frozen_pose',p,variant=f'axis_{ax}_{deg:+d}',axis=ax,angle_deg=deg)
 for dev in ['dev1','dev2']:
  inp_path=MP/'inputs'/dev/'input.json';inp=json.loads(inp_path.read_text());asset(inp_path);asset(inp['mesh_path'])
  oldp=get_old(dev);a=dict(np.load(oldp));asset(oldp);frames=[k['frame_index'] for k in inp['keyframes']];L=len(a['R_camera']);center=a['canonical_vertices_m'].astype(float).mean(0)
  for k in inp['keyframes']:asset(k['source_rgb_path'])
  if extra:
   if dev!=extra['dev']:continue
   p=Path(extra['path']);asset(p);variants(dev,frames,np.load(p),extra['id'],p,center)
   ledger[dev]={'added_path':source(p),'fixed_center':center.tolist(),'frames':frames,'frame_denominator':5};continue
  variants(dev,frames,a,'old_initialization',oldp,center)
  opt=OLD/'pose'/dev/'optimization';selp=opt/'selection_frozen.json';sel=json.loads(selp.read_text());asset(selp)
  p1=Path(sel['selected']['pose_file']);variants(dev,frames,np.load(p1),'P1_selected',p1,center)
  paths=[]
  for pd in sorted(opt.glob('path*')):
   if not pd.is_dir():continue
   canonical=json.loads((pd/'canonical_correspondences.json').read_text());s=json.loads((pd/'input_score.json').read_text());start=json.loads((pd/'path_start.json').read_text())
   for q in [pd/'canonical_correspondences.json',pd/'input_score.json',pd/'path_start.json',pd/'optimizer.json']:asset(q)
   temporal=canonical['temporal_records'];self_n=canonical['self_definition_records'];nrec=temporal+self_n
   path_entry=dict(path_id=pd.name,start_definition=start,selected=pd.name==sel['selected']['path'],before=source(pd/'initial_path.npz'),after=source(pd/'object_init.npz'),official_scores_used_for_full_path_selection=False,original_score=s['score'],original_components=s['components'],input_gate_pass=s['iou_gate_pass'],rejections=s['rejection_reasons'],accepted_queries=canonical['accepted_queries'],temporal_records=temporal,self_definition_records=self_n,raw_track_observations=canonical['raw_track_observation_count'],coverage=temporal/max(1,canonical['raw_track_observation_count']),robust_component_divisor_frames=L,residual_slots={'silhouette_forward':64*L,'silhouette_background':64*L,'estimated_depth_weak':64*L,'confirmed_tracks':2*nrec,'actual_time_acceleration':6*(L-2)},observed_frame_count=len(s['fixed_visible_frames']),zero_correspondence_means_no_track_evidence=True)
   paths.append(path_entry)
   for stage,filename in [('before','initial_path.npz'),('after','object_init.npz')]:
    p=pd/filename;asset(p);b=np.load(p)
    for f in frames:add(dev,f,b['R_camera'][f],b['t_camera_m'][f],'full_path',pd.name,stage,p)
  cp=MP/'predictions'/dev/'candidates.npz';asset(cp);c=np.load(cp)
  for ki,f in enumerate(c['frame_indices']):
   for j in range(c['R_camera'].shape[1]):add(dev,f,c['R_camera'][ki,j],c['t_camera_m'][ki,j],'single_keyframe_candidate',f'megapose_{j}','official_refined',cp,historical_pose_score=float(c['scores'][ki,j]))
  crp=opt/'candidate_records.json';cr=json.loads(crp.read_text());asset(crp)
  ledger[dev]={'old_start':source(oldp),'keyframes':frames,'frozen_center':center.tolist(),'canonical_center_definition':'arithmetic mean of fixed canonical template vertices, camera convention R*c0+t','full_frames':L,'raw_official_candidates':int(c['scores'].size),'official_score_saved':True,'official_scores_for_initial_candidate_top5':True,'official_scores_for_local_refinement_beam_final_selection':False,'local_results_count':len(cr['all_local_results']),'retained_local_counts':cr['counts'],'full_path_count':len(paths),'paths':paths,'final_selection':source(selp),'selected_path':sel['selected']['path'],'official_score_summary':{'minimum':float(c['scores'].min()),'maximum':float(c['scores'].max()),'all_values':c['scores'].tolist()},'original_rgb_term':'0.1 * mean(max(0,1-patch_ncc)) over source overlap checks; frozen source RGB comparisons, constant with respect to optimized R/t within a path; differs across paths'}
 np.savez_compressed(dest/'frozen_poses.npz',T_camera_object=np.stack(poses))
 dump(dest/'frozen_rows.json',rows);dump(dest/'candidate_ledger.json',ledger)
 for p in [MP/'inference_config.json',MP/'weights_manifest.json',MP/'infer_candidates.py',OLD/'code/solve_pose.py',MP/'repo/src/megapose/inference/pose_estimator.py',MP/'repo/src/megapose/models/pose_rigid.py',MP/'repo/src/megapose/panda3d_renderer/panda3d_scene_renderer.py',Path(__file__)]:asset(p)
 for w in json.loads((MP/'weights_manifest.json').read_text())['assets']:asset(w['path'])
 freeze={'created_utc':now(),'tag':tag,'status':'frozen_before_scoring','rows':len(rows),'references_used':False,'assets':assets,'pool':source(dest/'frozen_poses.npz'),'rows_manifest':source(dest/'frozen_rows.json'),'ledger':source(dest/'candidate_ledger.json'),'depth_fractions':[-.1,-.05,-.02,0,.02,.05,.1],'orientation_axes':'canonical template xyz','angles_deg':[-15,15],'center_rule':'mean of canonical template vertices (float64); keep camera center fixed under orientation perturbation','depth_rule':'camera center ray, c=(R*c0+t), t_new=(1+epsilon)*c-R*c0','score':'unmodified official forward_scoring_model -> coarse RGB pose_logit and sigmoid pose_score','crop':'unmodified official hypothesis-dependent crop_inputs; each pose may get different box and resized crop; frozen model surface sampling seed12345','selection':'diagnostic only; no path choice, no reference input','official_frame_denominator':5,'debug':'save official returned observation crop and render for every pose; pass-through crop_inputs wrapper captures original boxes without modifying return values','scope':'existing weights and compatibility patches only; no training, no depth refiner, no new model'}
 dump(NEW/'protocol'/f'official_{tag}_frozen.json',freeze)
 print(json.dumps({'frozen':str(NEW/'protocol'/f'official_{tag}_frozen.json'),'rows':len(rows),'ledger':ledger},ensure_ascii=False),flush=True)

def score(tag):
 import numpy as np
 os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['EGL_VISIBLE_DEVICES']='1';os.environ['HOI_EGL_DEVICE_INDEX']='1'
 os.environ['CONDA_PREFIX']=str(MP/'venv');os.environ['MEGAPOSE_DATA_DIR']=str(MP/'data');sys.path.insert(0,str(MP/'repo/src'))
 import torch, pandas as pd
 from PIL import Image,ImageDraw
 torch.multiprocessing.set_start_method('spawn',force=True);torch.set_num_threads(1);torch.manual_seed(12345);np.random.seed(12345)
 dest=OUT/tag;freeze=json.loads((NEW/'protocol'/f'official_{tag}_frozen.json').read_text())
 for p,h in freeze['assets'].items():assert sha(p)==h,(p,'asset changed after freeze')
 for key in ['pool','rows_manifest','ledger']:assert sha(freeze[key]['path'])==freeze[key]['sha256']
 rows=json.loads((dest/'frozen_rows.json').read_text());poses=np.load(dest/'frozen_poses.npz')['T_camera_object']
 resultdir=dest/'scored';resultdir.mkdir(exist_ok=False)
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total','--format=csv'],text=True)
 proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True)
 line=[x for x in gpu.splitlines() if x.startswith('1,')][0];assert '3090' in line and int(line.split(',')[3].strip().split()[0])==0 and int(line.split(',')[4].strip().split()[0])<1000,(gpu,proc)
 uuid=line.split(',')[2].strip();assert uuid not in proc,(gpu,proc)
 info={'status':'running','started_utc':now(),'gpu_physical_id':1,'hardware':torch.cuda.get_device_name(0),'gpu_before':gpu,'processes_before':proc,'freeze':source(NEW/'protocol'/f'official_{tag}_frozen.json'),'script':source(__file__),'reference_used':False,'frames':[]};assert '3090' in info['hardware'];dump(resultdir/'run.json',info)
 renderer=None;started=time.perf_counter();outputs=[]
 try:
  from megapose.datasets.object_dataset import RigidObject,RigidObjectDataset
  from megapose.utils.load_model import load_named_model
  from megapose.inference.types import ObservationTensor
  from megapose.utils.tensor_collection import PandasTensorCollection
  inputs={d:json.loads((MP/'inputs'/d/'input.json').read_text()) for d in ['dev1','dev2']}
  mesh=RigidObjectDataset([RigidObject(label=d,mesh_path=Path(i['mesh_path']),mesh_units='m') for d,i in inputs.items()]);cfg=json.loads((MP/'inference_config.json').read_text())
  model=load_named_model(cfg['model_name'],mesh,n_workers=cfg['n_workers'],bsz_images=cfg['bsz_images']).cuda().eval();renderer=model.coarse_model.renderer
  info['model_load_seconds']=time.perf_counter()-started
  captured=[];original_crop=model.coarse_model.crop_inputs
  def capture_crop(*args,**kwargs):
   ret=original_crop(*args,**kwargs);captured.append({'K_crop':ret[1].detach().cpu().numpy(),'boxes_rend':ret[2].detach().cpu().numpy(),'boxes_crop':ret[3].detach().cpu().numpy()});return ret
  model.coarse_model.crop_inputs=capture_crop
  for dev,frame in sorted(set((r['dev'],r['frame']) for r in rows)):
   ii=[r['row_id'] for r in rows if r['dev']==dev and r['frame']==frame];selected=[rows[i] for i in ii];T=poses[ii].astype(np.float32);inp=inputs[dev];key=next(k for k in inp['keyframes'] if k['frame_index']==frame)
   rgb=np.array(Image.open(key['source_rgb_path']).convert('RGB'));K=np.array(inp['K']);obs=ObservationTensor.from_numpy(rgb,None,K).cuda()
   df=pd.DataFrame([{'label':dev,'batch_im_id':0,'instance_id':0,'row_id':r['row_id']} for r in selected]);data=PandasTensorCollection(df,poses=torch.tensor(T,device='cuda'));captured.clear()
   t0=time.perf_counter()
   with torch.inference_mode():scored,ex=model.forward_scoring_model(obs,data,return_debug_data=True)
   torch.cuda.synchronize();elapsed=time.perf_counter()-t0
   crops=ex['debug']['images_crop'].detach().cpu().numpy();renders=ex['debug']['renders'].detach().cpu().numpy();boxes=np.concatenate([c['boxes_crop'] for c in captured]);br=np.concatenate([c['boxes_rend'] for c in captured]);kc=np.concatenate([c['K_crop'] for c in captured]);frameout=resultdir/dev/f'{frame:05d}';frameout.mkdir(parents=True,exist_ok=False)
   np.savez_compressed(frameout/'crop_geometry.npz',row_ids=np.array(ii),boxes_crop=boxes,boxes_render=br,K_crop=kc)
   for j,r in enumerate(selected):
    logit=float(scored.infos['pose_logit'].iloc[j]);val=float(scored.infos['pose_score'].iloc[j]);assert np.isfinite(logit) and np.isfinite(val)
    crop=(crops[j,:3].transpose(1,2,0).clip(0,1)*255).astype(np.uint8);render=(renders[j,:3].transpose(1,2,0).clip(0,1)*255).astype(np.uint8)
    h,w=crop.shape[:2];can=Image.new('RGB',(w*3,h+34),'white');can.paste(Image.fromarray(crop),(0,34));can.paste(Image.fromarray(render),(w,34));can.paste(Image.fromarray((.5*crop+.5*render).astype(np.uint8)),(2*w,34));ImageDraw.Draw(can).text((4,4),f'{dev} f{frame} id{r["row_id"]} {r["path_id"]}/{r["stage"]}/{r["variant"]} logit={logit:.5f}',fill='black');pic=frameout/f'{r["row_id"]:05d}_crop_render.png';can.save(pic)
    out={**r,'pose_logit':logit,'pose_score':val,'valid':True,'boxes_crop_xyxy':boxes[j].tolist(),'boxes_render_xyxy':br[j].tolist(),'debug_image':str(pic),'crop_render_sha256':sha(pic),'score_reference':'uncalibrated RGB coarse ranking, not probability of correct pose'};outputs.append(out)
   dump(frameout/'scores.json',[o for o in outputs if o['dev']==dev and o['frame']==frame]);info['frames'].append({'dev':dev,'frame':frame,'rows':len(ii),'seconds':elapsed,'timing':ex['timing_str']});dump(resultdir/'run.json',info);print(dev,frame,len(ii),round(elapsed,3),flush=True)
   del ex,scored,data,obs,crops,renders
  dump(resultdir/'scores.json',outputs)
  full=[]
  for dev,pathid,stage in sorted(set((r['dev'],r['path_id'],r['stage']) for r in outputs if r['category']=='full_path' or (r['category']=='perturbation' and r['variant']=='depth_+0.00'))):
   subset=[r for r in outputs if (r['dev'],r['path_id'],r['stage'])==(dev,pathid,stage) and (r['category']=='full_path' or r['variant']=='depth_+0.00')];valid=len(subset)==5 and all(r['valid'] for r in subset)
   full.append({'dev':dev,'path_id':pathid,'stage':stage,'required_frames':5,'valid_frames':len(subset),'all_valid':valid,'mean_pose_logit':float(np.mean([r['pose_logit'] for r in subset])) if valid else None,'per_frame_logit':{str(r['frame']):r['pose_logit'] for r in subset}})
  dump(resultdir/'full_path_scores.json',full)
  info.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-started,rows=len(outputs),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),all_finite=all(np.isfinite(r['pose_logit']) for r in outputs));dump(resultdir/'run.json',info)
 except BaseException:
  info.update(status='failed',wall_seconds=time.perf_counter()-started,traceback=traceback.format_exc());dump(resultdir/'run.json',info);raise
 finally:
  if renderer is not None:renderer.stop()

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score']);p.add_argument('--tag',default='base');p.add_argument('--extra-dev');p.add_argument('--extra-path',type=Path);p.add_argument('--extra-id');a=p.parse_args()
 extra={'dev':a.extra_dev,'path':str(a.extra_path),'id':a.extra_id} if a.extra_path else None
 if a.action=='prepare':prepare(a.tag,extra)
 else:score(a.tag)
if __name__=='__main__':main()
