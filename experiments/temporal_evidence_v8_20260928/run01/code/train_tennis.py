"""V8B locked U/Q confirmation, with one train-only coarse parent."""
from common import *
import argparse,time,traceback
import torch
from replay_first_failure import Runtime as BaseRuntime,make_loop
from restore_state import *
from finite_guard import Guard,named_model
from adapter_v4 import ProtocolScene
from loss_policy import regional_rgb
from hoi_modules.projected_motion import make_renderer
from utils.loss_utils import ssim
import train as official

B=RUN/'tennis'

class Runtime(BaseRuntime):
    def __init__(self,a,parent,cfg):
        # Explicit independent ledger: do not invoke an inherited experiment initializer.
        self.a=a;self.output=Path(a.output);self.parent_state=parent;self.resume=parent;self.parent=None
        self.started=time.monotonic();self.elapsed_prior=0;self.prior_steps=0
        self.updates=parent.get('optimizer_updates',0);self.start_updates=self.updates
        self.attempted=0;self.peak_points=0;self.stage=parent['stage'];self.iteration=parent['iteration']
        self.last_checkpoint=None;self.current_density=None;self.phase='setup';self.batch=[];self.detail=False;self.pre_path=Path(a.parent)
        self.logs={n:(self.output/n).open('a',buffering=1) for n in ['training_metrics.jsonl','sampling_order.jsonl','density_events.jsonl']}
        self.events=(self.output/'replay_events.jsonl').open('a',buffering=1)
        lp=B/'protocol/formal_steps.jsonl';self.global_used=len(lp.read_text().splitlines()) if lp.exists() else 0
        self.ledger=lp.open('a',buffering=1);self.optimizer_ledger=(B/'protocol/optimizer_calls.jsonl').open('a',buffering=1)
        self.guard=Guard(self);self.stage_end=a.target;self.current_cfg=cfg
    def before_step(self,stage,iteration,model,stack,temp):
        assert iteration==self.parent_state['iteration']+self.attempted+1
        assert self.global_used+self.attempted<read(B/'protocol/config.json')['attempt_budget']
        self.stage=stage;self.iteration=iteration;self.batch=[];model.optimizer.zero_grad(set_to_none=True)
        self.guard.check('pre_step',named_model(model));self.attempted+=1
        self.ledger.write(json.dumps(dict(run=self.output.name,stage=stage,iteration=iteration,pid=os.getpid(),time_unix=time.time()))+'\n')
    def rgb_loss(self,pred,gt,cameras,stage):
        masks=torch.stack([c.foreground_mask for c in cameras]).to(pred.device)
        self.U,self.rgb_rows=regional_rgb(pred,gt,masks,'uniform',stage)
        self.S=ssim(pred,gt) if self.a.objective=='Q' else None
        self.Q=.8*self.U+.2*(1-self.S) if self.S is not None else self.U
        self.rgb_value=float(self.Q.detach());self.batch=[c.image_name for c in cameras]
        self.logs['sampling_order.jsonl'].write(json.dumps(dict(stage=stage,iteration=self.iteration,frame_ids=self.batch))+'\n')
        return self.Q
    def regularization(self,*args):self.R=super().regularization(*args);return self.R
    def optimizer_step(self,model):
        for event in ['enter','completed']:
            if event=='completed':super().optimizer_step(model)
            self.optimizer_ledger.write(json.dumps(dict(run=self.output.name,stage=self.stage,iteration=self.iteration,event=event,pid=os.getpid()))+'\n')
    def report(self,writer,iteration,l1,loss,loss_fn,ms,tests,scene,render,render_args,stage,dataset_type):
        assert not tests and not scene.getTestCameras()
        row=dict(stage=stage,iteration=iteration,batch=self.batch,U=float(self.U.detach()),SSIM=float(self.S.detach()) if self.S is not None else None,RGB=float(self.Q.detach()),R=float(self.R.detach()),points=len(scene.gaussians.get_xyz),seconds=time.monotonic()-self.started,iter_gpu_ms=float(ms))
        self.logs['training_metrics.jsonl'].write(json.dumps(row)+'\n')
        if iteration%100==0:save_json(self.output/'progress.json',row)
    def after_step(self,stage,iteration,model,stack,temp):
        self.guard.check('completed_model_Adam_buffers',named_model(model));self.peak_points=max(self.peak_points,len(model.get_xyz))
        if iteration%100==0 or iteration==self.stage_end:
            fixed=iteration%1000==0 or iteration==self.stage_end
            path=self.output/(f'checkpoint_{stage}_{iteration:06d}.pt' if fixed else f'rolling_{iteration:06d}.pt')
            state=capture(model,stage,iteration,stack,temp,phase='completed_step',resumable=iteration<14000,optimizer_updates=self.updates,effective_config_sha256=sha(self.output/'effective_config.json'),objective=self.a.objective)
            asset=atomic_checkpoint(path,state);self.last_checkpoint=path;self.pre_path=path
            with (self.output/'checkpoint_index.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=iteration,checkpoint=asset))+'\n')
            save_json(self.output/'latest_checkpoint.json',dict(checkpoint=asset,iteration=iteration,resumable=iteration<14000))
            for p in sorted(self.output.glob('rolling_*.pt'))[:-2]:p.unlink()
        self.U=self.S=self.Q=self.R=None
    def close(self):super().close();self.optimizer_ledger.close()

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    torch.set_num_threads(4);official.setup_seed(12345)
    _,(ds,hidden,opt,pipe)=official_config(2)
    inherited=read(V5/'runs/B_U/effective_config.json')
    for obj,key in [(hidden,'hidden'),(opt,'optimization'),(pipe,'pipeline')]:assert vars(obj)==inherited[key]
    m=read(B/'inputs/hos_tennis/manifest.json');a.scale_bound=m['scene_extent'];a.target=3000 if a.stage=='coarse' else 14000
    a.output=str((B/'runs'/('coarse_U' if a.stage=='coarse' else 'B_'+a.objective)).resolve())
    out=Path(a.output);assert not out.exists();out.mkdir(parents=True)
    ds.source_path=str((B/'inputs/hos_tennis/manifest.json').resolve());ds.model_path=str(out);ds.render_process=False
    model=official.GaussianModel(ds.sh_degree,hidden);scene=ProtocolScene(ds,model)
    if a.stage=='coarse':
        assert a.objective=='U';model.training_setup(opt)
        cams=scene.getTrainCameras();parent=capture(model,'coarse',0,cams,cams,phase='completed_step',resumable=True,optimizer_updates=0)
        a.parent=str((B/'protocol/untrained_coarse_initial.pt').resolve());assert not Path(a.parent).exists();atomic_checkpoint(a.parent,parent)
    else:
        a.parent=str((B/'protocol/shared_fine_initial.pt').resolve());parent=load_completed(a.parent)
        assert parent['iteration']==0 and parent['optimizer_updates']==0
    a.replay_from=a.parent;a.light=True;a.check=False;a.policy='uniform';a.steps=a.target-parent['iteration']
    cfg=dict(model=vars(ds),hidden=vars(hidden),optimization=vars(opt),pipeline=vars(pipe),seed=12345,scale_bound=a.scale_bound,parent=identity(a.parent),objective=a.objective,stage=a.stage,manifest=identity(ds.source_path),temporal_mode='off',M1='off',end=a.target,decision=identity(RUN/'quality_decision.json'))
    save_json(out/'effective_config.json',cfg);rt=Runtime(a,parent,cfg)
    try:
        rt.install_density_audit(model);render=make_renderer(a.scale_bound);loop=make_loop(rt,render)
        (out/'source_snapshots/render.py').write_text(render.expanded_source);official.network_gui.try_connect=lambda:None
        timer=official.Timer();timer.start();loop(ds,opt,hidden,pipe,[],[],[],a.parent,-1,model,scene,a.stage,None,a.target,timer)
        torch.cuda.synchronize()
        assert rt.attempted==a.target and rt.updates==(3000 if a.stage=='coarse' else 13999)
        save_json(out/'run.json',dict(status='completed',stage=a.stage,objective=a.objective,attempted_this_process=rt.attempted,updates_this_process=rt.updates-rt.start_updates,checkpoint=str(rt.last_checkpoint),checkpoint_sha256=sha(rt.last_checkpoint),final_points=len(model.get_xyz),peak_points=rt.peak_points,seconds=time.monotonic()-rt.started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),physical_GPU=1))
        if a.stage=='coarse':
            # Official fine transition: preserve model/SH/radii, reset Adam and statistics,
            # restore coarse RNG after construction, and start with a full camera stack.
            before=clone_cpu(model.capture());model.training_setup(opt)
            for i in [0,1,2,3,4,5,6,7,8,9,13]:exact(before[i],model.capture()[i])
            assert not model.optimizer.state
            for n in ['xyz_gradient_accum','denom','_deformation_accum']:assert torch.count_nonzero(getattr(model,n))==0
            terminal=torch.load(rt.last_checkpoint,map_location='cpu',weights_only=False);rng_restore(terminal['rng'])
            cams=scene.getTrainCameras();state=capture(model,'fine',0,cams,cams,phase='completed_step',resumable=True,optimizer_updates=0,shared_coarse_parent=identity(rt.last_checkpoint))
            path=B/'protocol/shared_fine_initial.pt';assert not path.exists();atomic_checkpoint(path,state)
            save_json(B/'protocol/fine_transition.json',dict(status='passed',parent=identity(rt.last_checkpoint),initial_fine=identity(path),model_SH_radii_exact=True,Adam_statistics_reset=True,full_stack=True,RNG_restored=True,optimization_steps=0))
    except BaseException:
        save_json(out/'failure.json',dict(status='failed',stage=rt.stage,iteration=rt.iteration,attempted_this_process=rt.attempted,updates_this_process=rt.updates-rt.start_updates,phase=rt.phase,last_finite_saved_state=str(rt.pre_path),traceback=traceback.format_exc()));raise
    finally:rt.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['coarse','fine'],required=True);p.add_argument('--objective',choices=['U','Q'],required=True);run(p.parse_args())
