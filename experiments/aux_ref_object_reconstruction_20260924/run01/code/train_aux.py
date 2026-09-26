"""Fixed-motion AUX_REF_OBJECT; fresh object-only optimization, no evaluation reader."""
from pathlib import Path
import argparse,datetime,json,os,random,socket,time,traceback
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from aux_scene import AuxObjectScene,sha_file,state_identity
from train_structured import mean_mask,ssim_error

E=Path(__file__).resolve().parents[1]
PROTOCOL='AUX_REF_OBJECT'

def write_json(path,value):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def setup(dev):
 torch.set_num_threads(4);torch.manual_seed(12345);np.random.seed(12345);random.seed(12345)
 meta_path=E/'inputs'/dev/'input_manifest.json';meta=json.loads(meta_path.read_text())
 assert meta['phase']=='complete',meta['phase']
 for p,h in zip(meta['frame_paths'],meta['frame_sha256']):assert sha_file(p)==h
 mask_path=Path(meta['segmentation']);assert sha_file(mask_path)==meta['segmentation_sha256']
 rgb=torch.from_numpy(np.stack([cv2.imread(p)[...,::-1].copy() for p in meta['frame_paths']])).cuda().permute(0,3,1,2).float()/255
 labels=torch.from_numpy(np.load(mask_path)['entity_labels'].astype(np.int64)).cuda()
 assert labels.shape==(len(rgb),rgb.shape[2],rgb.shape[3])
 masks=F.one_hot(labels,3).permute(0,3,1,2).float();interior=-F.max_pool2d(-masks,5,1,2)
 scene=AuxObjectScene(dev,meta['timestamps_seconds'],device='cuda');scene.assert_frozen()
 K=torch.tensor(meta['K'],device='cuda',dtype=torch.float32);w2c=torch.linalg.inv(scene.c2w)
 return scene,meta,meta_path,rgb,labels,masks,interior,K,w2c

def ref_motion(meta,arm):
 if arm=='Pred':return None
 path=Path(meta['reference_object_motion']);assert sha_file(path)==meta['reference_object_motion_sha256']
 data=np.load(path);assert np.array_equal(data['times'],np.asarray(meta['timestamps_seconds']))
 return torch.tensor(data['R_world'],device='cuda',dtype=torch.float32),torch.tensor(data['t_world'],device='cuda',dtype=torch.float32)

def render(scene,t,K,w2c,H,W,motion):
 kwargs={} if motion is None else {'object_R_world':motion[0][t],'object_t_world':motion[1][t]}
 return scene.render(t,K,w2c,H,W,**kwargs)

def step_update(scene,optimizer,t,rgb,labels,masks,interior,K,w2c,motion):
 optimizer.zero_grad(set_to_none=True);H,W=labels.shape[-2:]
 ret,_=render(scene,t,K,w2c,H,W,motion);pred=ret['rgb'];gt=rgb[t]
 lrgb=.8*abs(pred-gt).mean()+.2*ssim_error(pred,gt).mean()+.15*(mean_mask(abs(pred-gt),interior[t,1])+mean_mask(abs(pred-gt),interior[t,2]))
 linstance=sum(mean_mask(abs(ret['buf']-masks[t]).mean(0),interior[t,e]) for e in range(3))/3
 data=lrgb+.2*linstance
 if not bool(torch.isfinite(data)):raise FloatingPointError('Nonfinite data loss')
 data.backward()
 data_grads={k:p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for k,p in scene.obank.named_parameters()}
 active=torch.zeros(scene.obank.n,dtype=torch.bool,device='cuda');norms={}
 for k,g in data_grads.items():
  if not bool(torch.isfinite(g).all()):raise FloatingPointError('Nonfinite data gradient '+k)
  active|=g.reshape(scene.obank.n,-1).abs().amax(1)>0;norms[k]=float(g.norm())
 off=scene.sbank.n+scene.hbank.n;grad=ret['viewspace_points'].grad
 scene.obank.record(grad[off:off+scene.obank.n] if grad is not None else None,ret['radii'][off:off+scene.obank.n])
 reg=scene.object_static_regularization();reg.backward()
 regnorm={}
 for k,p in scene.obank.named_parameters():
  if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('Missing/nonfinite total gradient '+k)
  regnorm[k]=float((p.grad-data_grads[k]).norm())
 own=labels[t]==2;contrib=ret['buf'][2].detach();denom=int(own.sum())
 row={'frame':t,'data_loss':float(data.detach()),'rgb_loss':float(lrgb.detach()),'instance_loss':float(linstance.detach()),'static_reg':float(reg.detach()),'data_grad_norm':norms,'reg_grad_norm':regnorm,'data_grad_gaussians':int(active.sum()),'data_gradient_nonzero':bool(active.any()),'object_points':scene.obank.n,'O_pixels':denom,'O_contribution_mean':float(contrib[own].mean()) if denom else None,'O_contribution_gt_005_fraction':float((contrib[own]>.05).float().mean()) if denom else None}
 optimizer.step();return row

def exported_stats(scene):
 x=scene.object_canonical_export()
 def quantile(a):return {str(q):float(np.quantile(a,q)) for q in [0,.05,.5,.95,1]}
 stats={'count':scene.obank.n,'unique_anchors':int(len(np.unique(x['anchor_id']))),'opacity':quantile(x['opacity']),'scale_m':quantile(x['scale_m']),'offset_norm_m':quantile(np.linalg.norm(x['local_offset_m'],axis=1)),'fraction_alpha_lt002':float((x['opacity']<.02).mean()),'fraction_scale_gt0035':float((x['scale_m']>.035).mean()),'local_material_support':'unmeasured; no trusted local correspondence available'}
 assert stats['offset_norm_m']['1']<=.005000001
 return x,stats

def preflight():
 out=E/'preflight';out.mkdir(exist_ok=False)
 results={};initial=[]
 for arm in ['Pred','Ref']:
  scene,meta,mp,rgb,labels,masks,interior,K,w2c=setup('dev1');motion=ref_motion(meta,arm)
  initial.append(scene.initial_obank_identity);optimizer=torch.optim.Adam(scene.optimizer_groups(),eps=1e-8)
  rows=[]
  for i in range(8):rows.append(step_update(scene,optimizer,i%len(rgb),rgb,labels,masks,interior,K,w2c,motion))
  assert any(r['data_gradient_nonzero'] for r in rows),'No rendered data gradient in temporary check'
  scene.assert_frozen();final=state_identity(scene.obank.state_dict());changed={k:initial[-1]['arrays'][k]!=final['arrays'][k] for k in scene.obank.fields}
  assert all(changed.values()),changed
  results[arm]={'steps':8,'rows':rows,'object_attributes_changed':changed,'H_S_motion_unchanged':True,'initial_object_identity':initial[-1]}
  del scene,optimizer,rgb,labels,masks,interior,motion;torch.cuda.empty_cache()
 assert initial[0]==initial[1]
 report={'protocol_id':PROTOCOL,'status':'passed','temporary_state_discarded':True,'formal_runs_restart_from_original_initialization':True,'same_initial_object_identity':True,'results':results,'source_sha256':sha_file(__file__)}
 write_json(out/'result.json',report);print(json.dumps({'status':'passed','preflight_steps_per_arm':8}),flush=True)

def train(dev,arm):
 out=E/'runs'/f'{dev}_{arm}';out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 record={'protocol_id':PROTOCOL,'status':'running','dev':dev,'arm':arm,'utc_start':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'host':socket.gethostname(),'gpu':torch.cuda.get_device_name(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'budget_steps':8000,'step':0,'effective_data_gradient_steps':0,'published_object_motion_read':arm=='Ref','published_human_depth_texture_camera1_training':False}
 write_json(out/'run.json',record)
 try:
  scene,meta,mp,rgb,labels,masks,interior,K,w2c=setup(dev);motion=ref_motion(meta,arm)
  optimizer=torch.optim.Adam(scene.optimizer_groups(),eps=1e-8);torch.cuda.reset_peak_memory_stats()
  schedule_path=E/'frozen_aux'/f'{dev}_frame_schedule.npy';frames=np.load(schedule_path);assert frames.shape==(8000,)
  config={'protocol_id':PROTOCOL,'dev':dev,'arm':arm,'seed':12345,'steps':8000,'optimizer':'Adam','eps':1e-8,'rates':{g['name']:g['lr'] for g in optimizer.param_groups},'density':{'first':3000,'last':7000,'interval':500,'cap':6000,'threshold':.0002},'all_motion_frozen':True,'ordinary_track_and_motion_prior':False,'input_manifest':{'path':str(mp),'sha256':sha_file(mp)},'frame_schedule':{'path':str(schedule_path),'sha256':sha_file(schedule_path)},'source':scene.source_identity,'training_code':{str(p):sha_file(p) for p in [Path(__file__),Path(__file__).parent/'aux_scene.py',Path(__file__).parent/'prepare_aux_input.py']},'loss_source':{'path':str(Path(__import__('train_structured').__file__)),'sha256':sha_file(__import__('train_structured').__file__)}}
  write_json(out/'config.json',config);write_json(out/'initial_identity.json',scene.identity_snapshot())
  x,stats=exported_stats(scene);np.savez_compressed(out/'object_initial.npz',**x);write_json(out/'object_initial_stats.json',stats)
  record.update(input_manifest=config['input_manifest'],initial_obank_identity=scene.initial_obank_identity,frame_schedule_sha256=sha_file(schedule_path))
  with (out/'steps.jsonl').open('w') as logs,(out/'topology.jsonl').open('w') as events:
   for k,t in enumerate(frames,1):
    row=step_update(scene,optimizer,int(t),rgb,labels,masks,interior,K,w2c,motion);row.update(step=k,seconds=time.perf_counter()-start)
    logs.write(json.dumps(row)+'\n');record['effective_data_gradient_steps']+=int(row['data_gradient_nonzero']);record['step']=k
    if k%500==0 and 3000<=k<=7000:
     event=scene.obank.control(optimizer,'obank',k,6000);events.write(json.dumps(event)+'\n');events.flush()
    if k%500==0:
     logs.flush();write_json(out/'run.json',record);print(json.dumps({'dev':dev,'arm':arm,'step':k,'data_loss':row['data_loss'],'points':scene.obank.n,'seconds':row['seconds']}),flush=True)
    if k in [2000,4000,6000,8000]:
     scene.assert_frozen();path=out/f'checkpoint_{k:06d}.pt'
     torch.save({'protocol_id':PROTOCOL,'dev':dev,'arm':arm,'step':k,'obank':scene.obank.state_dict(),'optimizer':optimizer.state_dict(),'scene_identity':scene.identity_snapshot(),'effective_data_gradient_steps':record['effective_data_gradient_steps'],'config':config,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},path)
     record['checkpoint']={'path':str(path),'sha256':sha_file(path)}
  scene.assert_frozen();x,stats=exported_stats(scene);np.savez_compressed(out/'object_final.npz',**x);write_json(out/'object_final_stats.json',stats);write_json(out/'final_identity.json',scene.identity_snapshot())
  # Training-camera fit is descriptive only and never used for checkpoint choice.
  fit=[];(out/'camera0').mkdir()
  with torch.no_grad():
   for i in range(len(rgb)):
    ret,_=render(scene,i,K,w2c,*labels.shape[-2:],motion);own=labels[i]==2;mse=(ret['rgb']-rgb[i]).square().mean(0)
    fit.append({'frame':i,'O_pixels':int(own.sum()),'object_psnr':float(-10*torch.log10(mse[own].mean().clamp_min(1e-12))) if own.any() else None,'O_contribution_mean':float(ret['buf'][2][own].mean()) if own.any() else None,'O_contribution_gt_005_fraction':float((ret['buf'][2][own]>.05).float().mean()) if own.any() else None})
    cv2.imwrite(str(out/'camera0'/f'{i:05d}.png'),(ret['rgb'].clamp(0,1).permute(1,2,0).cpu().numpy()[...,::-1]*255).astype(np.uint8))
  write_json(out/'camera0_fit.json',{'protocol_id':PROTOCOL,'not_heldout':True,'rows':fit});torch.cuda.synchronize()
  record.update(status='completed',H_S_motion_unchanged=True,object_stats=stats,formal_steps=8000,zero_data_gradient_steps=8000-record['effective_data_gradient_steps'])
 except BaseException as exc:
  record.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  record.update(wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());write_json(out/'run.json',record)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','train']);p.add_argument('--dev',choices=['dev1','dev2']);p.add_argument('--arm',choices=['Pred','Ref']);a=p.parse_args()
 gate=json.loads((E/'protocol/native_availability.json').read_text())
 # Final gate is recorded independently from the training program.
 assert gate.get('training_gate') is True,'Native S/E availability gate not frozen'
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1','This task is bound to local physical GPU1'
 torch.cuda.set_device(0)
 if a.action=='preflight':preflight()
 else:
  assert json.loads((E/'preflight/result.json').read_text())['status']=='passed'
  train(a.dev,a.arm)
