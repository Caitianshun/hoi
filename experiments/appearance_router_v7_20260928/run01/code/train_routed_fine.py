"""One C-route with exact inherited schedule and bounded diagnostic branches."""
from common import *
import torch,argparse,time,traceback,inspect,csv,math
from replay_first_failure import Runtime as BaseRuntime,make_loop
from restore_state import *
from finite_guard import named_model
from adapter_v4 import ProtocolScene
from loss_policy import regional_rgb
from hoi_modules.attribute_gradient_router import AttributeGradientRouter,partition_current_optimizer_parameters,validate_gradients,aggregate_screen_gradients
from diagnostic_render import make_renderer
from budget import CallBudget
import train as official
class Runtime(BaseRuntime):
    def __init__(self,a,parent,cfg):
        super().__init__(a,parent,cfg);self.ledger.close()
        self.category='D' if a.diagnostic else 'C';self.lp=RUN/'protocol'/(self.category+'_steps.jsonl')
        self.global_used=len(self.lp.read_text().splitlines()) if self.lp.exists() else 0;self.ledger=self.lp.open('a',buffering=1)
        self.pre_path=Path(a.parent);self.updates=parent.get('optimizer_updates',0);self.start_updates=self.updates;self.stage_end=a.target
        self.router=AttributeGradientRouter(a.mode)
        self.optimizer_ledger=(RUN/'protocol/optimizer_calls.jsonl').open('a',buffering=1)
        self.audit=(self.output/'gradient_audit.jsonl').open('a',buffering=1)
        self.bound_file=(self.output/'bound_events_compact.csv').open('w');self.bound_writer=csv.DictWriter(self.bound_file,fieldnames=['arm','iteration','frame_id','total_axes','triggered_axes']);self.bound_writer.writeheader()
    def before_step(self,stage,iteration,model,stack,temp):
        assert iteration==self.parent_state['iteration']+self.attempted+1
        limit=read(RUN/'configs/v7.json')['budgets'][self.category+'_attempts'];assert self.global_used+self.attempted<limit
        self.stage=stage;self.iteration=iteration;self.batch=[];model.optimizer.zero_grad(set_to_none=True)
        self.guard.check('pre_step',named_model(model));partition_current_optimizer_parameters(model)
        self.attempted+=1;self.ledger.write(json.dumps(dict(run=self.output.name,iteration=iteration,kind='training_attempt',purpose=self.a.purpose,pid=os.getpid(),time_unix=time.time()))+'\n')
    def rgb_loss(self,pred,gt,cameras,stage):
        masks=torch.stack([c.foreground_mask for c in cameras]).to(pred.device)
        self.U,rows=regional_rgb(pred,gt,masks,'uniform',stage);self.F,_=regional_rgb(pred,gt,masks,'balanced_fine',stage)
        self.batch=[c.image_name for c in cameras];self.rgb_rows=rows;self.rgb_value=float(self.U.detach())
        self.logs['sampling_order.jsonl'].write(json.dumps(dict(stage=stage,iteration=self.iteration,frame_ids=self.batch))+'\n')
        return self.U
    def regularization(self,*args):self.R=super().regularization(*args);return self.R
    def backward(self,loss,model,qlist):
        self.before_backward(loss,model)
        self.q_uniform,rows,states=self.router.backward(model,self.U,self.F,self.R,qlist)
        if self.attempted==1:save_json(self.output/'gradient_partition.json',dict(parameters=rows,states=states,no_do=True,no_dshs=True))
        if self.iteration%100==0 or self.attempted==1 or self.a.diagnostic:
            norms={}
            G,A,_=partition_current_optimizer_parameters(model)
            for name,grads in [('G',[p.grad for _,p in G]),('A',[p.grad for _,p in A]),('q',self.q_uniform)]:norms[name]=float(torch.sqrt(sum(g.double().square().sum() for g in grads if g is not None)))
            self.audit.write(json.dumps(dict(iteration=self.iteration,batch=self.batch,U=float(self.U.detach()),F=float(self.F.detach()),R=float(self.R.detach()),norms=norms,states=states,screen_grad_source='explicit_returned_uniform',retained_q_grad_used=False))+'\n')
    def screen_grad_source(self,index,q):return self.q_uniform[index]
    def optimizer_step(self,model):
        partition_current_optimizer_parameters(model)
        self.optimizer_ledger.write(json.dumps(dict(purpose=self.a.purpose,run=self.output.name,iteration=self.iteration,event='enter',pid=os.getpid()))+'\n')
        super().optimizer_step(model)
        self.optimizer_ledger.write(json.dumps(dict(purpose=self.a.purpose,run=self.output.name,iteration=self.iteration,event='completed',pid=os.getpid()))+'\n')
    def report(self,writer,iteration,l1,loss,loss_fn,ms,tests,scene,render,render_args,stage,dataset_type):
        assert not tests and not scene.getTestCameras()
        if True:
            row=dict(iteration=iteration,batch=self.batch,U=float(self.U.detach()),F=float(self.F.detach()),R=float(self.R.detach()),points=len(scene.gaussians.get_xyz),seconds=time.monotonic()-self.started,iter_gpu_ms=float(ms),optimization_rule=self.a.policy,scalar_joint_objective=False)
            self.logs['training_metrics.jsonl'].write(json.dumps(row)+'\n');save_json(self.output/'progress.json',row)
    def after_step(self,stage,iteration,model,stack,temp):
        self.guard.check('completed_model_Adam_buffers',named_model(model));partition_current_optimizer_parameters(model)
        self.peak_points=max(self.peak_points,len(model.get_xyz));self.event('completed_step',optimizer_step_calls=self.updates)
        if iteration%100==0 or iteration==self.stage_end or iteration in self.a.save_iterations:
            fixed=iteration%1000==0 or iteration==self.stage_end or iteration in self.a.save_iterations
            path=self.output/(f'checkpoint_fine_{iteration:06d}.pt' if fixed else f'rolling_{iteration:06d}.pt')
            state=capture(model,stage,iteration,stack,temp,phase='completed_step',resumable=iteration<14000,policy=self.a.policy,effective_config_sha256=sha(self.output/'effective_config.json'),optimizer_updates=self.updates,purpose=self.a.purpose,lineage=self.a.lineage,scientific_fingerprint=self.a.fingerprint)
            asset=atomic_checkpoint(path,state);self.last_checkpoint=path;self.pre_path=path
            with (self.output/'checkpoint_index.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=iteration,checkpoint=asset))+'\n')
            save_json(self.output/'latest_checkpoint.json',dict(checkpoint=asset,iteration=iteration,resumable=iteration<14000))
            for p in sorted(self.output.glob('rolling_*.pt'))[:-2]:p.unlink()
        self.q_uniform=[];self.U=self.F=self.R=None
    def close(self):super().close();self.audit.close();self.bound_file.close();self.optimizer_ledger.close()
def routed_loop(rt,render):
    make_loop(rt,render);path=rt.output/'source_snapshots/train_observed.py';source=path.read_text()
    changes={
      'runtime.before_backward(loss,gaussians)\n        loss.backward()':'runtime.backward(loss,gaussians,viewspace_point_tensor_list)',
      'viewspace_point_tensor_list[idx].grad':'runtime.screen_grad_source(idx,viewspace_point_tensor_list[idx])',
      '[v.grad for v in viewspace_point_tensor_list]':'runtime.q_uniform',
    }
    for old,new in changes.items():assert source.count(old)==1;source=source.replace(old,new)
    path=rt.output/'source_snapshots/train_routed.py';path.write_text(source)
    save_json(rt.output/'route_patch.json',dict(changes=changes,patched=identity(path),default_policy='original_uniform preserves V5 one backward',q_source='explicit return; retained .grad ignored in routed modes'))
    ns=official.__dict__.copy();ns.update(runtime=rt,restore_model=restore_model,training_report=rt.report,render=render);exec(compile(source,str(path),'exec'),ns)
    return ns['scene_reconstruction']
def scientific_config(pc,a):
    return dict(hidden=pc['hidden'],optimization=pc['optimization'],pipeline=pc['pipeline'],seed=12345,
        scale_bound=pc['scale_bound'],source_path=pc['model']['source_path'],mode=a.mode,
        purpose=a.purpose,lineage=a.lineage,module_contract='canonical_SH_A_G_uniform_q_uniform',
        config_sha256=sha(RUN/'configs/v7.json'))
def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    torch.set_num_threads(4);official.setup_seed(12345)
    parent=load_completed(a.parent);out=Path(a.output);assert not out.exists()
    a.diagnostic=a.purpose=='diagnostic';a.policy=a.mode
    conf=read(RUN/'configs/v7.json');pc=read(V5/'runs/B_U/effective_config.json')
    _,(dataset,hidden,opt,pipe)=official_config(2)
    for obj,key in [(hidden,'hidden'),(opt,'optimization'),(pipe,'pipeline')]:assert vars(obj)==pc[key]
    a.lineage=conf['parent_hashes']['B_U_fine1000' if a.diagnostic else 'shared_fine0']
    science=scientific_config(pc,a);a.fingerprint=hashlib.sha256(json.dumps(science,sort_keys=True).encode()).hexdigest()
    if a.start_kind=='own_checkpoint':
        assert Path(a.parent).resolve().is_relative_to(RUN.resolve())
        assert parent['purpose']==a.purpose and parent['policy']==a.mode and parent['lineage']==a.lineage
        assert parent['scientific_fingerprint']==a.fingerprint
        source_cfg=Path(a.parent).parent/'effective_config.json';assert parent['effective_config_sha256']==sha(source_cfg)
        assert read(source_cfg)['scientific_config']==science
        # Verify the checkpoint against its previously committed immutable identity.
        indices=list((Path(a.parent).parent/'checkpoint_index.jsonl').open())
        assert any(x['checkpoint']['path']==str(Path(a.parent).absolute()) and x['checkpoint']['sha256']==sha(a.parent) for x in map(json.loads,indices))
        if not a.diagnostic:
            auth=read(RUN/'protocol/external_resume_authorization.json')
            assert auth['checkpoint']==identity(a.parent) and auth['resume_number']==1
    elif a.diagnostic:
        assert a.start_kind=='diagnostic_parent' and parent['iteration']==1000
        assert sha(a.parent)==a.lineage and a.target==1128
    else:
        assert a.start_kind=='shared_fine' and parent['iteration']==0 and a.target==14000
        assert sha(a.parent)==a.lineage
    if not a.diagnostic:
        assert a.mode=='appearance_only' and a.target==14000
        gate=read(RUN/'module_acceptance.json');assert gate['status'] in ['pass','accepted_numerical_variation']
        for asset in gate['frozen_sources']:assert sha(asset['path'])==asset['sha256']
    else:assert a.target==1128 and a.mode=='appearance_only'
    assert parent['iteration']<a.target
    dataset.source_path=pc['model']['source_path'];dataset.model_path=str(out.resolve());dataset.render_process=False
    a.scale_bound=pc['scale_bound'];a.light=True;a.check=a.diagnostic;a.replay_from=a.parent;a.steps=a.target-parent['iteration']
    cfg=dict(model=vars(dataset),hidden=vars(hidden),optimization=vars(opt),pipeline=vars(pipe),seed=12345,
        scale_bound=a.scale_bound,policy=a.mode,parent=identity(a.parent),purpose=a.purpose,lineage=a.lineage,
        scientific_config=science,scientific_fingerprint=a.fingerprint,start_kind=a.start_kind,
        allowed_branch_changes=['gradient policy','output path','diagnostic metadata'],parent_Adam_RNG_stack_preserved=True)
    out.mkdir(parents=True);save_json(out/'effective_config.json',cfg);budget=CallBudget(out.name);budget.mode='optimization'
    rt=Runtime(a,parent,cfg);model=None
    try:
        model=official.GaussianModel(dataset.sh_degree,hidden);scene=ProtocolScene(dataset,model);rt.install_density_audit(model)
        native=make_renderer(a.scale_bound)
        def render(camera,*args,**kwargs):
            p=native(camera,*args,**kwargs)
            rt.guard.check('rendered_attributes_and_outputs',{k:v for k,v in p.items() if k!='canonical_scale_exp'})
            rt.bound_writer.writerow(dict(arm=out.name,iteration=rt.iteration,frame_id=camera.image_name,total_axes=p['deformed_log_scale_raw'].numel(),triggered_axes=int((p['deformed_log_scale_raw']>math.log(a.scale_bound)).sum())))
            return p
        loop=routed_loop(rt,render);official.network_gui.try_connect=lambda:None;timer=official.Timer();timer.start()
        loop(dataset,opt,hidden,pipe,[],[],[],a.parent,-1,model,scene,'fine',None,a.target,timer)
        torch.cuda.synchronize();save_json(out/'run.json',dict(status='completed',nominal_iterations=rt.stage_end,
            attempted_this_process=rt.attempted,optimizer_updates=rt.updates,updates_this_process=rt.updates-rt.start_updates,
            checkpoint=str(rt.last_checkpoint),checkpoint_sha256=sha(rt.last_checkpoint),final_points=len(model.get_xyz),
            peak_points=rt.peak_points,seconds=time.monotonic()-rt.started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),GPU='physical1 RTX3090',purpose=a.purpose,start_kind=a.start_kind,
            source_config_sha256=sha(out/'effective_config.json')))
    except BaseException:
        error=traceback.format_exc();save_json(out/'failure.json',dict(status='failed',iteration=rt.iteration,attempted_this_process=rt.attempted,updates_this_process=rt.updates-rt.start_updates,phase=rt.phase,last_finite_saved_state=str(rt.pre_path),traceback=error))
        if 'illegal memory access' not in error and model is not None and model.optimizer is not None:
            atomic_checkpoint(out/'failure_state.pt',clone_cpu(dict(model=model.capture(),gradients=named_model(model,True),deformation_accum=model._deformation_accum,iteration=rt.iteration)))
        raise
    finally:rt.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start-kind',choices=['shared_fine','own_checkpoint','diagnostic_parent'],required=True)
    p.add_argument('--purpose',choices=['formal','diagnostic'],required=True);p.add_argument('--mode',default='appearance_only',choices=AttributeGradientRouter.MODES)
    p.add_argument('--target',type=int,required=True);p.add_argument('--save-iterations',type=int,nargs='*',default=[]);run(p.parse_args())
