from pathlib import Path
import sys,time,json,os,argparse
import torch,numpy as np
from observation_fusion import Memory,Scorer,E
from train_fusion import setup,frozen,old,save,state_identity,render_parts,sha_file,AUX

def bake(dev,variant):
 start=time.perf_counter();out=E/'runs'/f'{dev}_{variant}';out.mkdir(exist_ok=variant!='F0')
 scene,meta,mp,rgb,labels,masks,interior,K,w2c=setup(dev);motion=old.ref_motion(meta,'Ref');identity=state_identity(scene.obank.state_dict());mem=Memory(dev);torch.manual_seed(12345);net=Scorer(variant).cuda()
 if variant!='F0':net.load_state_dict(torch.load(out/'checkpoint_002000.pt',map_location='cuda',weights_only=False)['model'])
 with torch.no_grad():
  color,weights=net(mem.features(),True);assert torch.isfinite(color).all();audit=mem.audit(net)
  baselinecolor=scene.obank.color_logit.sigmoid();changes=float((color-baselinecolor).abs().max());fused_joint=[];fused_iso=[]
  for t in range(len(rgb)):
   parts=scene.components(t,motion[0][t],motion[1][t]);fused_joint.append(render_parts(scene,parts,K,w2c,*labels.shape[-2:],color)['rgb'].cpu());fused_iso.append(render_parts(scene,parts,K,w2c,*labels.shape[-2:],color,True,True)['rgb'].cpu())
  frozen(scene,identity);state={k:v.detach().cpu().clone() for k,v in scene.obank.state_dict().items()};eps=1e-6;bounded=color.clamp(eps,1-eps);state['color_logit']=torch.logit(bounded).cpu();torch.save({'variant':variant,'dev':dev,'obank':state,'base':str(AUX/'runs'/f'{dev}_Ref/checkpoint_008000.pt'),'cache':sha_file(E/'support'/dev/'observations.npz'),'epsilon':eps},out/'baked_model.pt')
  original_state=scene.obank.state_dict()
  for k in state:
   if k!='color_logit':assert torch.equal(state[k],original_state[k].cpu())
  scene.obank.load_state_dict(state);errs=[];camera0=[];renderdiff=[]
  baseline_scene=setup(dev)[0]
  for t in range(len(rgb)):
   parts=scene.components(t,motion[0][t],motion[1][t]);j=render_parts(scene,parts,K,w2c,*labels.shape[-2:]);i=render_parts(scene,parts,K,w2c,*labels.shape[-2:],only=True,black=True)
   errs.extend([float((j['rgb'].cpu()-fused_joint[t]).abs().max()),float((i['rgb'].cpu()-fused_iso[t]).abs().max())]);O=labels[t]==2;err=(j['rgb']-rgb[t]).square().mean(0);camera0.append(dict(frame=t,O_pixels=int(O.sum()),O_PSNR=float(-10*torch.log10(err[O].mean().clamp_min(1e-12))) if O.any() else None,full_PSNR=float(-10*torch.log10(err.mean().clamp_min(1e-12)))))
   if t==0:
    p0=baseline_scene.components(t,motion[0][t],motion[1][t]);j0=render_parts(baseline_scene,p0,K,w2c,*labels.shape[-2:]);i0=render_parts(baseline_scene,p0,K,w2c,*labels.shape[-2:],only=True,black=True);renderdiff=[float((j['rgb']-j0['rgb']).abs().max()),float((i['rgb']-i0['rgb']).abs().max())]
  assert max(errs)<=1e-5;assert changes>0 and all(x>0 for x in renderdiff)
  np.savez_compressed(out/'baked_colors.npz',colors=color.cpu().numpy(),weights=weights.cpu().numpy(),source_indices=mem.selections[-1]);save(out/'source_audit.json',audit);save(out/'camera0_fit.json',camera0)
  result=dict(status='completed',dev=dev,variant=variant,seconds=time.perf_counter()-start,baked_hash=sha_file(out/'baked_model.pt'),max_render_export_error=max(errs),actual_color_change=changes,first_frame_joint_iso_RGB_change=renderdiff,geometry_topology_opacity_HS_frozen=True,epsilon=eps,color_clamp_error=float((bounded-color).abs().max()),peak_allocated_bytes=torch.cuda.max_memory_allocated())
  save(out/'bake.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dev',required=True);p.add_argument('--variant',required=True);a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';bake(a.dev,a.variant)
