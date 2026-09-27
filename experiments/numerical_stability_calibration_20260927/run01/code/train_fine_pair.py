"""Two conditional fine-only arms with a shared domain bound and strict budgets."""
from common import *
import os,argparse,time,traceback
import torch
from replay_first_failure import Runtime as ReplayRuntime,make_loop
from restore_state import *
from finite_guard import named_model
from adapter_v4 import ProtocolScene
from loss_policy import regional_rgb
import renderer_observer,train as official
class FormalRuntime(ReplayRuntime):
    def __init__(self,a,parent,cfg):
        super().__init__(a,parent,cfg);self.ledger.close();path=RUN/'protocol/B_steps.jsonl'
        self.global_used=sum(1 for _ in path.open()) if path.exists() else 0;self.ledger=path.open('a',buffering=1)
        self.detail=False;self.stage_end=14000;self.updates=parent.get('optimizer_updates',0);self.pre_path=Path(a.resume) if a.resume else RUN/'protocol/shared_fine_initial.pt'
    def before_step(self,stage,iteration,model,stack,temp):
        assert iteration==self.parent_state['iteration']+self.attempted+1
        if self.global_used+self.attempted>=28000:raise RuntimeError('B total attempt budget exhausted; do not restart')
        self.stage=stage;self.iteration=iteration;self.batch=[]
        self.guard.check('pre_step',named_model(model))
        self.attempted+=1;self.ledger.write(json.dumps(dict(run=self.output.name,stage=stage,iteration=iteration,time_unix=time.time()))+'\n')
    def rgb_loss(self,pred,gt,cameras,stage):
        policy='uniform' if self.a.policy=='uniform_fine' else 'balanced_fine'
        masks=torch.stack([c.foreground_mask for c in cameras]).to(pred.device)
        value,rows=regional_rgb(pred,gt,masks,policy,stage)
        self.batch=[c.image_name for c in cameras];self.rgb_rows=rows;self.rgb_value=float(value.detach())
        self.logs['sampling_order.jsonl'].write(json.dumps(dict(stage=stage,iteration=self.iteration,frame_ids=self.batch))+'\n')
        return value
    def after_step(self,stage,iteration,model,stack,temp):
        self.peak_points=max(self.peak_points,len(model.get_xyz));self.event('completed_step',optimizer_step_calls=self.updates)
        if iteration%100==0 or iteration==self.stage_end:
            fixed=iteration%1000==0 or iteration==self.stage_end
            name=f'checkpoint_fine_{iteration:06d}.pt' if fixed else f'rolling_{iteration:06d}.pt'
            path=self.output/name;state=capture(model,stage,iteration,stack,temp,phase='completed_step',resumable=iteration<self.stage_end,policy=self.a.policy,effective_config_sha256=sha(self.output/'effective_config.json'),optimizer_updates=self.updates)
            torch.save(state,path);self.last_checkpoint=path;self.pre_path=path
            for p in sorted(self.output.glob('rolling_*.pt'))[:-2]:p.unlink()
            save_json(self.output/'latest_checkpoint.json',dict(checkpoint=identity(path),stage=stage,iteration=iteration,resumable=iteration<self.stage_end))
def run(a):
    gate=read(RUN/'protocol/formal_gate.json');assert gate['all_conditions_passed']
    for item in gate['frozen_source_files']:assert sha(item['path'])==item['sha256'],item['path']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    torch.set_num_threads(4);official.setup_seed(12345);out=Path(a.output)
    shared=read(RUN/'protocol/fine_transition.json');assert identity(a.branch_from_coarse)==shared['parent']
    parent=load_completed(a.resume if a.resume else shared['initial_fine']['path'])
    if a.resume:assert Path(a.resume).parent.resolve()==out.resolve() and parent['policy']==a.policy
    else:assert not out.exists() and parent['iteration']==0
    _,(dataset,hidden,opt,pipe)=official_config(2);dataset.source_path=str(OLD/'inputs/hos_backpack/manifest.json');dataset.model_path=str(out.resolve());dataset.render_process=False
    inherited=read(OLD/'runs/hos_backpack_formal/effective_config.json')
    for obj,key in [(hidden,'hidden'),(opt,'optimization'),(pipe,'pipeline')]:assert vars(obj)==inherited[key]
    a.scale_bound=shared['scene_extent'];a.light=True;a.check=False;a.steps=14000-parent['iteration'];a.replay_from=a.resume if a.resume else shared['initial_fine']['path']
    cfg=dict(model=vars(dataset),hidden=vars(hidden),optimization=vars(opt),pipeline=vars(pipe),seed=12345,loss_policy=a.policy,stage_ends=dict(coarse=0,fine=14000),parent_coarse=shared['parent'],initial_fine=shared['initial_fine'],scale_bound=a.scale_bound,gate=identity(RUN/'protocol/formal_gate.json'))
    out.mkdir(parents=True,exist_ok=True)
    if a.resume:assert read(out/'effective_config.json')==cfg and parent['effective_config_sha256']==sha(out/'effective_config.json')
    else:save_json(out/'effective_config.json',cfg)
    rt=FormalRuntime(a,parent,cfg);model=None
    try:
        model=official.GaussianModel(dataset.sh_degree,hidden);scene=ProtocolScene(dataset,model);rt.install_density_audit(model)
        render=renderer_observer.install(rt);loop=make_loop(rt,render);official.network_gui.try_connect=lambda:None
        timer=official.Timer();timer.start();loop(dataset,opt,hidden,pipe,[],[],[],a.replay_from,-1,model,scene,'fine',None,14000,timer)
        torch.cuda.synchronize()
        save_json(out/'run.json',dict(status='completed',nominal_iterations=14000,optimizer_updates=rt.updates,attempted_this_process=rt.attempted,checkpoint=str(rt.last_checkpoint),checkpoint_sha256=sha(rt.last_checkpoint),final_points=len(model.get_xyz),peak_points=rt.peak_points,seconds=time.monotonic()-rt.started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),source_config_sha256=sha(out/'effective_config.json'),GPU='physical1 RTX3090'))
    except BaseException:
        error=traceback.format_exc();save_json(out/'failure.json',dict(status='failed',stage=rt.stage,iteration=rt.iteration,attempted_this_process=rt.attempted,optimizer_updates=rt.updates,phase=rt.phase,last_finite_saved_state=str(rt.pre_path),traceback=error))
        if 'illegal memory access' not in error and model is not None and model.optimizer is not None:torch.save(clone_cpu(dict(model=model.capture(),gradients=named_model(model,True),deformation_accum=model._deformation_accum,phase=rt.phase,iteration=rt.iteration)),out/'failure_state_with_gradients.pt')
        raise
    finally:rt.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--branch-from-coarse',required=True);p.add_argument('--output',required=True);p.add_argument('--policy',choices=['uniform_fine','balanced_fine'],required=True);p.add_argument('--resume');run(p.parse_args())
