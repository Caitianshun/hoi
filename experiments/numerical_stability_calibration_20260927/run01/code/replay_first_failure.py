"""Budgeted exact-state A replay of the locked loop with boundary observers."""
from common import *
import argparse,inspect,os,time,traceback,copy
import torch
from train_stage import Runtime as OriginalRuntime
from adapter_v4 import ProtocolScene
from restore_state import *
from finite_guard import Guard,named_model
import train as official
import renderer_observer

class WindowComplete(Exception):pass
class Runtime(OriginalRuntime):
    def __init__(self,a,parent,cfg):
        self.a=a;self.output=Path(a.output);self.parent_state=parent;self.resume=parent;self.parent=None
        self.output.mkdir(parents=True,exist_ok=True);self.started=time.monotonic();self.elapsed_prior=0
        self.prior_steps=0;self.updates=0;self.attempted=0;self.peak_points=0;self.stage='fine';self.iteration=parent['iteration']
        self.last_checkpoint=None;self.current_density=None;self.phase='setup';self.batch=[];self.detail=not a.light;self.pre_path=None
        self.logs={n:(self.output/n).open('a',buffering=1) for n in ['training_metrics.jsonl','sampling_order.jsonl','density_events.jsonl']}
        self.events=(self.output/'replay_events.jsonl').open('a',buffering=1)
        self.ledger=(RUN/'protocol/A_steps.jsonl').open('a',buffering=1);self.global_used=sum(1 for _ in (RUN/'protocol/A_steps.jsonl').open())
        self.guard=Guard(self);self.stage_end=parent['iteration']+a.steps;self.current_cfg=cfg
    def event(self,phase,**kw):
        self.events.write(json.dumps(dict(attempt=self.output.name,iteration=self.iteration,phase=phase,batch=self.batch,time_unix=time.time(),**kw),allow_nan=False)+'\n')
    def resume_sampler(self,stage,cameras,stack,temp,model):
        assert stage==self.parent_state['stage'];exact(model.capture(),self.parent_state['model']);exact(model._deformation_accum,self.parent_state['deformation_accum'])
        stack,temp=restore_sampler(self.parent_state,cameras)
        save_json(self.output/'restore_verification.json',dict(status='passed',stage=stage,parent_iteration=self.parent_state['iteration'],expected_first=self.parent_state['iteration']+1,
            model_Adam_groups_steps_buffers_exact=True,RNG_exact=True,stack_exact=[c.uid for c in stack]==self.parent_state['viewpoint_stack'],checkpoint_metadata_dispatch=True,parent=identity(self.a.replay_from)))
        return stack,temp
    def before_step(self,stage,iteration,model,stack,temp):
        assert iteration==self.parent_state['iteration']+self.attempted+1
        if self.global_used+self.attempted>=read(RUN/'configs/v5.json')['budgets']['A_attempts']:raise RuntimeError('A attempt budget exhausted')
        self.stage=stage;self.iteration=iteration;self.batch=[]
        self.guard.check('pre_step',named_model(model),self.detail)
        state=capture(model,stage,iteration-1,stack,temp,phase='completed_step',policy=self.a.policy,effective_config_sha256=sha(self.output/'effective_config.json'),optimizer_updates=self.updates)
        self.pre_path=self.output/f'pre_step_{iteration:06d}.pt'
        torch.save(state,self.pre_path)
        keep=8 if self.detail else 2
        old=sorted(self.output.glob('pre_step_*.pt'))
        for p in old[:-keep]:p.unlink()
        self.attempted+=1;self.ledger.write(json.dumps(dict(attempt=self.output.name,stage=stage,iteration=iteration,time_unix=time.time()))+'\n')
        self.event('schedule_enter',pre_state=str(self.pre_path),stack_ids=state['viewpoint_stack'],RNG_torch=tensor_hash(state['rng']['torch']))
    def selected(self,cameras):
        self.batch=[c.image_name for c in cameras];self.event('batch_selected',frame_ids=self.batch)
    def before_backward(self,loss,model):
        self.guard.check('loss_before_backward',dict(total=loss),self.detail)
        self.event('loss_terms',RGB=self.rgb_value,regions=self.rgb_rows,regularizers=self.reg)
    def gradient_check(self,stage,iteration,model,loss):
        self.guard.check('parameter_gradients',named_model(model,True),self.detail)
    def optimizer_step(self,model):
        self.guard.check('before_Adam',named_model(model),False)
        self.event('Adam_enter');model.optimizer.step();self.updates+=1
        self.guard.check('after_Adam',named_model(model),self.detail)
    def after_step(self,stage,iteration,model,stack,temp):
        self.event('completed_step',optimizer_step_calls=self.updates)
        if iteration==self.stage_end:
            path=self.output/'completed.pt';torch.save(capture(model,stage,iteration,stack,temp,phase='completed_step',policy=self.a.policy,effective_config_sha256=sha(self.output/'effective_config.json'),optimizer_updates=self.updates),path)
            self.last_checkpoint=path
    def install_density_audit(self,model):
        super().install_density_audit(model)
        for name in ['densify','prune','grow','reset_opacity','add_densification_stats']:
            if not hasattr(model,name):continue
            fn=getattr(model,name)
            def wrapper(*args,_name=name,_fn=fn,**kw):
                self.event('density_'+_name+'_enter');result=_fn(*args,**kw)
                self.guard.check('density_'+_name+'_exit',named_model(model),False);return result
            setattr(model,name,wrapper)
    def close(self):
        super().close();self.events.close()

def make_loop(runtime,observed_render):
    original=inspect.getsource(official.scene_reconstruction);source=original
    start=source.index('    if checkpoint:');end=source.index('    bg_color',start)
    source=source[:start]+"    if checkpoint:\n        assert stage == runtime.parent_state['stage']\n        restore_model(gaussians,opt,runtime.parent_state)\n        first_iter = runtime.parent_state['iteration']\n\n"+source[end:]
    changes={
      'count = 0\n    for iteration':'viewpoint_stack,temp_list=runtime.resume_sampler(stage,train_cams,viewpoint_stack,temp_list,gaussians)\n    count = 0\n    for iteration',
      'for iteration in range(first_iter, final_iter+1):':'for iteration in range(first_iter, final_iter+1):\n        runtime.before_step(stage,iteration,gaussians,viewpoint_stack,temp_list)',
      '\n        images = []':'\n        runtime.selected(viewpoint_cams)\n        images = []',
      'Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])':'Ll1=runtime.rgb_loss(image_tensor,gt_image_tensor[:,:3,:,:],viewpoint_cams,stage)',
      'loss = Ll1\n        if stage == "fine" and hyper.time_smoothness_weight != 0:\n            # tv_loss = 0\n            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)\n            loss += tv_loss':'loss=Ll1+runtime.regularization(gaussians,hyper,stage,Ll1)',
      'loss.backward()':'runtime.before_backward(loss,gaussians)\n        loss.backward()',
      'if torch.isnan(loss).any():\n            print("loss is nan,end training, reexecv program now.")\n            os.execv(sys.executable, [sys.executable] + sys.argv)':'runtime.gradient_check(stage,iteration,gaussians,loss)',
      'iter_end.record()':"runtime.guard.check('screen_gradients',[v.grad for v in viewspace_point_tensor_list],runtime.detail)\n        runtime.guard.check('screen_gradient_aggregate',viewspace_point_tensor_grad,runtime.detail)\n        iter_end.record()",
      'gaussians.optimizer.step()':'runtime.optimizer_step(gaussians)'}
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    source+='\n            runtime.after_step(stage,iteration,gaussians,viewpoint_stack,temp_list)\n'
    p=runtime.output/'source_snapshots/train_observed.py';p.parent.mkdir(exist_ok=True);p.write_text(source)
    save_json(runtime.output/'source_adaptation.json',dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),patched=identity(p),changes=changes,restore_change='Explicit metadata stage, no filename dispatch'))
    scope=official.__dict__.copy();scope.update(runtime=runtime,restore_model=restore_model,training_report=runtime.report,render=observed_render)
    exec(compile(source,str(p),'exec'),scope);return scope['scene_reconstruction']

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    out=Path(a.output);assert not out.exists(),'Each A attempt needs a fresh output directory'
    torch.set_num_threads(4);official.setup_seed(12345)
    parent=load_completed(a.replay_from);_,(dataset,hidden,opt,pipe)=official_config(2)
    inherited=read(V4/'runs/W_all/effective_config.json')
    for obj,key in [(hidden,'hidden'),(opt,'optimization'),(pipe,'pipeline')]:assert vars(obj)==inherited[key]
    if Path(a.replay_from).parent==V4/'runs/W_all':assert parent['effective_config_sha256']==sha(V4/'runs/W_all/effective_config.json')
    else:
        pc=read(Path(a.replay_from).parent/'effective_config.json');assert parent['effective_config_sha256']==sha(Path(a.replay_from).parent/'effective_config.json')
        for key,obj in [('hidden',hidden),('optimization',opt),('pipeline',pipe)]:assert pc[key]==vars(obj)
    dataset.source_path=str(OLD/'inputs/hos_backpack/manifest.json');dataset.model_path=str(out.absolute());dataset.render_process=False
    cfg=dict(model=vars(dataset),hidden=vars(hidden),optimization=vars(opt),pipeline=vars(pipe),seed=12345,policy=a.policy,scale_bound=a.scale_bound,parent=identity(a.replay_from),inherited_effective_config=identity(V4/'runs/W_all/effective_config.json'),allowed_cross_directory_changes=['model.model_path','diagnostic observers','output identity','window stop boundary','declared scale_bound if explicitly selected'],scientific_schedule_unchanged=True)
    out.mkdir(parents=True);save_json(out/'effective_config.json',cfg)
    rt=Runtime(a,parent,cfg);model=None
    try:
        model=official.GaussianModel(dataset.sh_degree,hidden);scene=ProtocolScene(dataset,model)
        rt.install_density_audit(model);render=renderer_observer.install(rt);loop=make_loop(rt,render);official.network_gui.try_connect=lambda:None
        timer=official.Timer();timer.start()
        ctx=torch.autograd.detect_anomaly(check_nan=True) if a.anomaly else __import__('contextlib').nullcontext()
        with ctx:loop(dataset,opt,hidden,pipe,[],[],[],a.replay_from,-1,model,scene,'fine',None,rt.stage_end,timer)
        torch.cuda.synchronize();save_json(out/'result.json',dict(status='window_completed',attempted=rt.attempted,optimizer_step_calls=rt.updates,last_iteration=rt.iteration,checkpoint=identity(rt.last_checkpoint)))
    except BaseException:
        save_json(out/'result.json',dict(status='failed',attempted=rt.attempted,optimizer_step_calls=rt.updates,last_iteration=rt.iteration,phase=rt.phase,pre_state=str(rt.pre_path),traceback=traceback.format_exc()))
        if not 'illegal memory access' in traceback.format_exc() and model is not None and model.optimizer is not None:
            torch.save(clone_cpu(dict(model=model.capture(),gradients=named_model(model,True),deformation_accum=model._deformation_accum,phase=rt.phase,iteration=rt.iteration)),out/'failure_state_with_gradients.pt')
        raise
    finally:rt.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay-from',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=8);p.add_argument('--policy',default='balanced_all');p.add_argument('--light',action='store_true');p.add_argument('--anomaly',action='store_true');p.add_argument('--scale-bound',type=float)
    a=p.parse_args();a.check=True;run(a)
