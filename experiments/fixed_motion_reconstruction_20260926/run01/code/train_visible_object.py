"""V2 B1; historical modules remain read-only. Training imports camera0 only."""
from pathlib import Path
import sys,json,time,os,traceback,argparse,random
ROOT=Path('/home/cai_tianshun/Project/HOI')
AUX=ROOT/'experiments/aux_ref_object_reconstruction_20260924/run01'
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(AUX/'code'))
import torch,numpy as np
import train_aux as old
from aux_scene import joint_render,state_identity,sha_file

def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');q.replace(p)

def render_parts(scene,parts,K,w2c,H,W,color=None,only=False,black=False):
 ids=[2] if only else [0,1,2]
 colors=torch.cat([color if i==2 and color is not None else scene.banks[i].color_logit.sigmoid() for i in ids])
 buf=torch.cat([torch.nn.functional.one_hot(torch.full((scene.banks[i].n,),i,device=colors.device,dtype=torch.long),3).float() for i in ids])
 return joint_render([parts[i] for i in ids],H,W,K,w2c,bg_color=[0.,0.,0.] if black else [1.,1.,1.],colors_precomp=colors,add_buffer=buf)

def data_loss(scene,t,rgb,labels,masks,interior,K,w2c,motion,color=None):
 H,W=labels.shape[-2:];parts=scene.components(t,motion[0][t],motion[1][t])
 joint=render_parts(scene,parts,K,w2c,H,W,color);iso=render_parts(scene,parts,K,w2c,H,W,color,True,True)
 pred=joint['rgb'];gt=rgb[t]
 lr=.8*abs(pred-gt).mean()+.2*old.ssim_error(pred,gt).mean()+.15*(old.mean_mask(abs(pred-gt),interior[t,1])+old.mean_mask(abs(pred-gt),interior[t,2]))
 li=sum(old.mean_mask(abs(joint['buf']-masks[t]).mean(0),interior[t,e]) for e in range(3))/3
 ir=.5*old.mean_mask(abs(iso['rgb']-gt),interior[t,2])+.5*old.mean_mask(abs(iso['rgb']),interior[t,0])
 ia=.5*old.mean_mask(abs(iso['alpha']-1),interior[t,2])+.5*old.mean_mask(abs(iso['alpha']),interior[t,0])
 loss=lr+.2*li+.5*ir+.1*ia
 return loss,joint,iso,{'joint_rgb':float(lr.detach()),'joint_instance':float(li.detach()),'iso_rgb':float(ir.detach()),'iso_alpha':float(ia.detach())}

def update(scene,opt,t,rgb,labels,masks,interior,K,w2c,motion):
 opt.zero_grad(set_to_none=True)
 loss,joint,iso,terms=data_loss(scene,t,rgb,labels,masks,interior,K,w2c,motion)
 assert torch.isfinite(loss);loss.backward() # exactly one data backward for both paths
 grads={k:p.grad.detach().clone() for k,p in scene.obank.named_parameters()}
 assert all(torch.isfinite(g).all() for g in grads.values())
 offset=scene.sbank.n+scene.hbank.n;n=scene.obank.n
 gj=joint['viewspace_points'].grad[offset:offset+n];gi=iso['viewspace_points'].grad
 assert gj.shape==gi.shape==(n,3)
 before=scene.obank.seen_count.clone();r=torch.maximum(joint['radii'][offset:offset+n],iso['radii'])
 scene.obank.record(gj+gi,r)
 assert torch.equal(scene.obank.seen_count-before,(r>0).to(before.dtype))
 reg=scene.object_static_regularization();reg.backward();opt.step()
 return dict(frame=t,loss=float(loss.detach()),static_reg=float(reg.detach()),terms=terms,data_grad_norm={k:float(v.norm()) for k,v in grads.items()},nonzero=any(v.abs().max()>0 for v in grads.values()),points=n,screen_joint_norm=float(gj.norm()),screen_iso_norm=float(gi.norm()),screen_sum_norm=float((gj+gi).norm()),seen_union=int((r>0).sum()))

def run(dev,check=False):
 out=HERE/('preflight' if check else 'runs')/dev;out.mkdir(exist_ok=False)
 start=time.perf_counter();rec=dict(status='running',dev=dev,variant='B1',steps=0,precheck=check,pid=os.getpid(),gpu='physical GPU1 RTX3090',protocol='AUX_FIXED_MOTION_RECON_V1')
 save(out/'run.json',rec)
 try:
  scene,meta,mp,rgb,labels,masks,interior,K,w2c=old.setup(dev);motion=old.ref_motion(meta,'Ref')
  schedule=np.load(AUX/'frozen_aux'/f'{dev}_frame_schedule.npy');assert len(schedule)==8000
  opt=torch.optim.Adam(scene.optimizer_groups(),eps=1e-8)
  save(out/'initial_identity.json',scene.identity_snapshot())
  # Same frame schedule, plus check an empty O analytically without optimization.
  assert old.mean_mask(torch.ones_like(interior[0,0]),torch.zeros_like(interior[0,0])).item()==0
  assert not (interior[:,2]*masks[:,1]).any() and not (interior[:,0]*masks[:,1]).any()
  torch.cuda.reset_peak_memory_stats()
  with (out/'steps.jsonl').open('w') as f,(out/'topology.jsonl').open('w') as events:
   for step,t in enumerate(schedule[:8] if check else schedule,1):
    # Persist attempted step before updating; crashes consume precheck budget.
    if check:rec['attempted_steps']=step;save(out/'run.json',rec)
    row=update(scene,opt,int(t),rgb,labels,masks,interior,K,w2c,motion);row['nonzero']=bool(row['nonzero']);row['step']=step;f.write(json.dumps(row)+'\n');rec['steps']=step
    if not check and step%500==0 and 3000<=step<=7000:events.write(json.dumps(scene.obank.control(opt,'obank',step,6000))+'\n');events.flush()
    if step%500==0:save(out/'run.json',rec);f.flush();print(dev,step,row['loss'],flush=True)
    if not check and step%2000==0:
     scene.assert_frozen();torch.save(dict(step=step,obank=scene.obank.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),out/f'checkpoint_{step:06d}.pt')
  scene.assert_frozen();x,stats=old.exported_stats(scene)
  if not check:np.savez_compressed(out/'object_final.npz',**x)
  save(out/'final_identity.json',scene.identity_snapshot());save(out/'object_stats.json',stats)
  if check:
   changes={k:scene.initial_obank_identity['arrays'][k]!=state_identity(scene.obank.state_dict())['arrays'][k] for k in scene.obank.fields};assert all(changes.values());rec['all_five_updated']=changes
   # Same colors, black versus white premultiplied path; no backward/optimizer step.
   with torch.no_grad():
    parts=scene.components(0,motion[0][0],motion[1][0]);b=render_parts(scene,parts,K,w2c,*labels.shape[-2:],only=True,black=True);w=render_parts(scene,parts,K,w2c,*labels.shape[-2:],only=True)
    err=float((b['rgb']-(w['rgb']-(1-w['alpha']))).abs().max());assert err<2e-6;rec['black_white_parity_max']=err
  rec.update(status='completed',frozen_HS_motion=True,temporary_discarded=check)
 except BaseException:
  rec.update(status='failed',traceback=traceback.format_exc());raise
 finally:
  torch.cuda.synchronize();rec.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated());save(out/'run.json',rec)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dev',required=True);ap.add_argument('--check',action='store_true');a=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
 cfg=json.loads((HERE/'protocol/frozen.json').read_text());assert cfg['frozen']
 run(a.dev,a.check)
