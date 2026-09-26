"""Shared score-only training; no evaluation assets. All original state frozen."""
from pathlib import Path
import sys,os,json,time,argparse,traceback
import torch,numpy as np
from observation_fusion import Memory,Scorer,QD,KD,WIDTH,F2_PARAMS,E
R=Path('/home/cai_tianshun/Project/HOI');B=R/'experiments/fixed_motion_reconstruction_20260926/run01';sys.path.insert(0,str(B/'code'))
from train_visible_object import old,AUX,save,state_identity,sha_file,data_loss,render_parts

def setup(dev):
 scene,meta,mp,rgb,labels,masks,interior,K,w2c=old.setup(dev);ck=AUX/'runs'/f'{dev}_Ref/checkpoint_008000.pt';state=torch.load(ck,map_location='cpu',weights_only=False)['obank'];scene.obank.resize_for_load(state,'');scene.obank.load_state_dict(state);scene.to('cuda')
 for p in scene.parameters():p.requires_grad_(False)
 assert state_identity(scene.obank.state_dict())==state_identity(state)
 return scene,meta,mp,rgb,labels,masks,interior,K,w2c

def frozen(scene,identity):
 assert scene._frozen_snapshot()==scene.frozen_identity and state_identity(scene.obank.state_dict())==identity and not any(p.requires_grad for p in scene.parameters())

def run(dev,variant,check):
 out=E/('preflight' if check else 'runs')/f'{dev}_{variant}';out.mkdir(exist_ok=False);start=time.perf_counter();rec=dict(status='running',dev=dev,variant=variant,steps=0,updates=0,precheck=check,pid=os.getpid(),gpu='physical1 RTX3090')
 save(out/'run.json',rec)
 try:
  scene,meta,mp,rgb,labels,masks,interior,K,w2c=setup(dev);motion=old.ref_motion(meta,'Ref');base=state_identity(scene.obank.state_dict());mem=Memory(dev);mem.prepare();torch.manual_seed(12345);net=Scorer(variant).cuda();opt=torch.optim.Adam(net.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
  cfg=dict(qdim=QD,kdim=KD,MLP_hidden=WIDTH,F2_parameters=F2_PARAMS,parameters=sum(p.numel() for p in net.parameters()),B0_identity=base,optimizer={'lr':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':0},source=sha_file(__file__),scorer_source=sha_file(Path(__file__).parent/'observation_fusion.py'))
  save(out/'config.json',cfg)
  schedule=np.load(AUX/'frozen_aux'/f'{dev}_frame_schedule.npy')[:2000];eligible=[int(t) for t in schedule if (mem.features(int(t))[3].sum(1)>=2).any()]
  if check:
   frames=json.loads((E/'protocol/fusion_frozen.json').read_text())['input_check'][dev]['precheck_frames'];assert len(set(frames))>=2
   parity=[]
   for t in [None]+sorted(set(frames)):
    f=mem.features(t);err=float((net(f)-Scorer('F0').cuda()(f)).abs().max());assert err==0;parity.append(err)
   save(out/'zero_step.json',{'max_error':max(parity),'all_cache_and_excluded':True})
  else:
   for v in ['F1','F2']:assert json.loads((E/'preflight'/f'{dev}_{v}'/'run.json').read_text())['status']=='completed'
   frames=schedule
  distinct=set();torch.cuda.reset_peak_memory_stats()
  with (out/'steps.jsonl').open('w') as logfile:
   for step,t in enumerate(frames,1):
    t=int(t);opt.zero_grad(set_to_none=True);feat=mem.features(t);gated=bool((feat[3].sum(1)>=2).any());updated=False;norm=0.;terms={};lossval=None
    if check:rec['attempted_steps']=step;save(out/'run.json',rec)
    if gated:
     color=net(feat);loss,joint,iso,terms=data_loss(scene,t,rgb,labels,masks,interior,K,w2c,motion,color)
     assert torch.isfinite(loss);lossval=float(loss.detach());loss.backward();grads=[p.grad for p in net.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads);norm=sum(float(g.square().sum()) for g in grads)**.5
     if norm>0:opt.step();updated=True;distinct.add(t);rec['updates']+=1
    # No gradient means no optimizer step: Adam momentum must not move weights.
    row=dict(step=step,frame=t,loss=lossval,terms=terms,gradient_norm=norm,updated=updated,eligible_gaussians=int((feat[3].sum(1)>=2).sum()));logfile.write(json.dumps(row)+'\n');rec['steps']=step
    if step%500==0:logfile.flush();save(out/'run.json',rec);print(dev,variant,step,norm,flush=True)
    if not check and step%500==0:torch.save({'step':step,'model':net.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'config':cfg},out/f'checkpoint_{step:06d}.pt')
  frozen(scene,base)
  if check:assert len(distinct)>=2,'fewer than two times with actual rendered learning signal'
  rec.update(status='completed',distinct_gradient_frames=sorted(distinct),frozen_original_state=True,temporary_discarded=check,no_op_steps=rec['steps']-rec['updates'])
 except BaseException:
  rec.update(status='failed',traceback=traceback.format_exc());raise
 finally:
  torch.cuda.synchronize();rec.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated());save(out/'run.json',rec)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dev',required=True);p.add_argument('--variant',choices=['F1','F2'],required=True);p.add_argument('--check',action='store_true');a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';run(a.dev,a.variant,a.check)
