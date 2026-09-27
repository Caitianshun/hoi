"""Locked Wu loop with region loss, explicit coarse fork and audit-only hooks."""
from common import *
import argparse,copy,inspect,os,random,time,traceback,hashlib,functools
import numpy as np
import torch
from adapter_v4 import ProtocolScene
from loss_policy import regional_rgb,sanity_check
from train_official import rng_capture,rng_restore
import train as official

class PlannedStop(Exception):pass

def model_identity(model):
    return dict(active_sh_degree=model.active_sh_degree,spatial_lr_scale=model.spatial_lr_scale,
        tensors={n:tensor_hash(getattr(model,n)) for n in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_deformation_table','max_radii2D']},
        deformation={n:tensor_hash(t) for n,t in model._deformation.state_dict().items()})

class Runtime:
    def __init__(self,a):
        self.a=a;self.output=Path(a.output);self.output.mkdir(parents=True,exist_ok=True)
        self.resume=torch.load(a.resume,weights_only=False) if a.resume else None
        self.parent=torch.load(a.branch_from_coarse,weights_only=False) if a.branch_from_coarse and not a.resume else None
        self.started=time.monotonic();self.elapsed_prior=self.resume['elapsed_seconds'] if self.resume else 0
        self.prior_steps=self.resume['total_nominal_steps'] if self.resume else 0
        self.updates=self.resume['optimizer_updates'] if self.resume else 0
        self.attempted=0;self.peak_points=self.resume['peak_points'] if self.resume else 0
        self.stage=None;self.iteration=0;self.last_checkpoint=None;self.current_density=None
        self.logs={name:(self.output/name).open('a',buffering=1) for name in ['training_metrics.jsonl','sampling_order.jsonl','density_events.jsonl']}
        lp=RUN/'protocol'/('temporary_steps.jsonl' if a.check else 'formal_steps.jsonl')
        self.global_used=len(lp.read_text().splitlines()) if lp.exists() else 0
        self.global_limit=40 if a.check else 31000;self.ledger=lp.open('a',buffering=1)
        self.allowed_seconds=float(os.environ.get('V4_REMAINING_GPU_SECONDS',10800))
    def close(self):
        for f in self.logs.values():f.close()
        self.ledger.close()
    def before_step(self,stage,iteration):
        if self.global_used+self.attempted>=self.global_limit:raise RuntimeError('Nominal-step budget exhausted; no automatic restart')
        if time.monotonic()-self.started>self.allowed_seconds:raise TimeoutError('GPU task budget exhausted')
        self.stage=stage;self.iteration=iteration;self.attempted+=1
        self.ledger.write(json.dumps(dict(run=self.output.name,stage=stage,iteration=iteration,time_unix=time.time()))+'\n')
    def resume_sampler(self,stage,cameras,stack,temp,model):
        if self.resume and self.resume['stage']==stage:
            lookup={c.uid:c for c in cameras};stack=[lookup[i] for i in self.resume['viewpoint_stack']];temp=[lookup[i] for i in self.resume['temp_list']]
            model._deformation_accum=self.resume['deformation_accum'];rng_restore(self.resume['rng'])
            save_json(self.output/'resume_check.json',dict(status='passed',stage=stage,iteration=self.resume['iteration'],own_fine_optimizer_restored=True,sampler_restored=True,branch_reset_not_reapplied=True))
        elif self.parent is not None and stage=='fine':
            assert len(stack)==len(cameras) and [c.uid for c in stack]==[c.uid for c in cameras]
            assert not model.optimizer.state
            assert torch.count_nonzero(model.xyz_gradient_accum)==0 and torch.count_nonzero(model.denom)==0 and torch.count_nonzero(model._deformation_accum)==0
            assert torch.equal(model.max_radii2D,self.parent['model'][9])
            rng_restore(self.parent['rng'])
            # No new random calls follow before the official batch draw.
            save_json(self.output/'fine_transition.json',dict(status='passed',full_stack=True,Adam_reset=True,gradient_denom_deformation_accum_reset=True,max_radii_preserved=True,first_iteration=1,parent=identity(self.a.branch_from_coarse)))
        return stack,temp
    def rgb_loss(self,pred,gt,cameras,stage):
        assert pred.shape[0]==len(cameras)
        masks=torch.stack([c.foreground_mask for c in cameras]).to(pred.device)
        value,rows=regional_rgb(pred,gt,masks,self.a.policy,stage)
        self.batch=[c.image_name for c in cameras];self.rgb_rows=rows;self.rgb_value=float(value.detach())
        self.logs['sampling_order.jsonl'].write(json.dumps(dict(stage=stage,iteration=self.iteration,frame_ids=self.batch))+'\n')
        return value
    def regularization(self,model,hyper,stage,reference):
        vals={n:reference.new_zeros(()) for n in ['plane_tv','time_smoothness','time_plane']}
        if stage=='fine' and hyper.time_smoothness_weight!=0:
            vals=dict(plane_tv=hyper.plane_tv_weight*model._plane_regulation(),time_smoothness=hyper.time_smoothness_weight*model._time_regulation(),time_plane=hyper.l1_time_planes*model._l1_regulation())
        self.reg={n:float(v.detach()) for n,v in vals.items()}
        # Same order as GaussianModel.compute_regulation.
        return vals['plane_tv']+vals['time_smoothness']+vals['time_plane']
    def gradient_check(self,stage,iteration,model,loss):
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite total loss; preserve failure, no execv')
        if iteration==1 or iteration%1000==0:
            assert all(torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'] if p.grad is not None)
    def optimizer_step(self,model):
        model.optimizer.step();self.updates+=1
    def report(self,writer,iteration,l1,loss,loss_fn,ms,tests,scene,render,render_args,stage,dataset_type):
        assert not tests and not scene.getTestCameras()
        self.peak_points=max(self.peak_points,len(scene.gaussians.get_xyz))
        if self.a.check or iteration==1 or iteration%100==0:
            row=dict(stage=stage,iteration=iteration,batch_frame_ids=self.batch,images=self.rgb_rows,
                L_rgb=self.rgb_value,L_full_uniform=float(np.mean([r['L_uniform'] for r in self.rgb_rows])),
                L_fg=float(np.mean([r['L_fg'] for r in self.rgb_rows if r['L_fg'] is not None])) if any(r['L_fg'] is not None for r in self.rgb_rows) else None,
                L_bg=float(np.mean([r['L_bg'] for r in self.rgb_rows if r['L_bg'] is not None])) if any(r['L_bg'] is not None for r in self.rgb_rows) else None,
                regularizers=self.reg,L_total=float(loss),points=len(scene.gaussians.get_xyz),
                iter_gpu_ms=float(ms),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),elapsed_seconds=self.elapsed_prior+time.monotonic()-self.started,
                RGB_policy=self.a.policy,RGB_unclipped=True)
            assert abs(row['L_total']-row['L_rgb']-sum(self.reg.values()))<1e-5
            self.logs['training_metrics.jsonl'].write(json.dumps(row)+'\n');save_json(self.output/'progress.json',row)
    def save_state(self,stage,iteration,model,stack,temp):
        torch.cuda.synchronize();path=self.output/f'checkpoint_{stage}_{iteration:06d}.pt'
        state=dict(model=model.capture(),stage=stage,iteration=iteration,rng=rng_capture(),viewpoint_stack=[c.uid for c in stack],temp_list=[c.uid for c in temp],
            deformation_accum=model._deformation_accum,elapsed_seconds=self.elapsed_prior+time.monotonic()-self.started,
            total_nominal_steps=self.prior_steps+self.attempted,optimizer_updates=self.updates,peak_points=self.peak_points,
            policy=self.a.policy,effective_config_sha256=sha(self.output/'effective_config.json'),parent_coarse=self.a.branch_from_coarse)
        tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path);self.last_checkpoint=path
        save_json(self.output/'latest_checkpoint.json',dict(path=str(path),stage=stage,iteration=iteration))
    def after_step(self,stage,iteration,model,stack,temp):
        self.peak_points=max(self.peak_points,len(model.get_xyz))
        if iteration%1000==0 or iteration==self.stage_end or self.a.stop_after==self.attempted:
            self.save_state(stage,iteration,model,stack,temp)
        if self.a.stop_after==self.attempted:raise PlannedStop('Authorized temporary resume-interface check')
    def install_density_audit(self,model):
        original_add=model.densification_postfix;original_prune=model.prune_points
        def add(*args,**kw):
            n=len(args[0]);out=original_add(*args,**kw)
            if self.current_density is not None:self.current_density['actual_added']+=n
            return out
        def remove(mask):
            n=int(mask.sum());out=original_prune(mask)
            if self.current_density is not None:self.current_density['actual_deleted']+=n
            return out
        model.densification_postfix=add;model.prune_points=remove
        for name in ['densify','prune','grow']:
            if not hasattr(model,name):continue
            fn=getattr(model,name)
            def wrap(*args,_name=name,_fn=fn,**kwargs):
                assert self.current_density is None
                grad=torch.nan_to_num(model.xyz_gradient_accum/model.denom,nan=0.)
                event=dict(stage=self.stage,iteration=self.iteration,operation=_name,points_before=len(model.get_xyz),
                    actual_added=0,actual_deleted=0,screen_gradient=quantiles(grad),
                    thresholds=[v for v in args[:6] if isinstance(v,(int,float)) or v is None])
                self.current_density=event
                try:out=_fn(*args,**kwargs)
                finally:
                    event['points_after']=len(model.get_xyz);assert event['points_after']==event['points_before']+event['actual_added']-event['actual_deleted']
                    self.logs['density_events.jsonl'].write(json.dumps(event)+'\n');self.current_density=None
                return out
            setattr(model,name,wrap)

def patched_loop(runtime):
    original=inspect.getsource(official.scene_reconstruction);source=original
    changes={
        '(model_params, first_iter) = torch.load(checkpoint)':"resume_blob = torch.load(checkpoint, weights_only=False)\n            model_params, first_iter = resume_blob['model'], resume_blob['iteration']",
        'count = 0\n    for iteration':'viewpoint_stack, temp_list = runtime.resume_sampler(stage, train_cams, viewpoint_stack, temp_list, gaussians)\n    count = 0\n    for iteration',
        'for iteration in range(first_iter, final_iter+1):':'for iteration in range(first_iter, final_iter+1):\n        runtime.before_step(stage, iteration)',
        'Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])':'Ll1 = runtime.rgb_loss(image_tensor, gt_image_tensor[:,:3,:,:], viewpoint_cams, stage)',
        'loss = Ll1\n        if stage == "fine" and hyper.time_smoothness_weight != 0:\n            # tv_loss = 0\n            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)\n            loss += tv_loss':
            'loss = Ll1 + runtime.regularization(gaussians, hyper, stage, Ll1)',
        'if torch.isnan(loss).any():\n            print("loss is nan,end training, reexecv program now.")\n            os.execv(sys.executable, [sys.executable] + sys.argv)':'runtime.gradient_check(stage, iteration, gaussians, loss)',
        'gaussians.optimizer.step()':'runtime.optimizer_step(gaussians)',
    }
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    source+='\n            runtime.after_step(stage, iteration, gaussians, viewpoint_stack, temp_list)\n'
    save_json(runtime.output/'upstream_adaptation.json',dict(official_commit=OFFICIAL_COMMIT,original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),replacements=changes,appended_hook='runtime.after_step',density_hook='Only counts actual postfix additions / prune-mask deletions, no RNG or rule changes'))
    namespace=official.__dict__.copy();namespace.update(runtime=runtime,training_report=runtime.report)
    exec(compile(source,str(UPSTREAM/'train.py')+':V4','exec'),namespace);return namespace['scene_reconstruction']

def validate_branch(runtime,model,scene,opt,hidden,pipe,dataset):
    p=runtime.parent;assert p['stage']=='coarse' and p['iteration']==opt.coarse_iterations
    model.restore(p['model'],opt);model._deformation_accum=p['deformation_accum']
    # Compare actual restored tensors and all deformation buffers with parent.
    current=model.capture()
    for i in [0,1,3,4,5,6,7,8,9,10,11,13]:
        assert torch.equal(current[i],p['model'][i]) if torch.is_tensor(current[i]) else current[i]==p['model'][i]
    for key,value in p['model'][2].items():assert torch.equal(current[2][key],value)
    inherited=model_identity(model)
    model.training_setup(opt);assert not model.optimizer.state;model.update_learning_rate(1)
    rng_restore(p['rng']);stack=scene.getTrainCameras().copy();cameras=[]
    for _ in range(opt.batch_size):cameras.append(stack.pop(random.randint(0,len(stack)-1)))
    background=torch.tensor([1,1,1] if dataset.white_background else [0,0,0],dtype=torch.float32,device='cuda')
    with torch.no_grad():
        preds=torch.stack([official.render(c,model,pipe,background,stage='fine')['render'] for c in cameras]);gt=torch.stack([c.original_image for c in cameras]).cuda();masks=torch.stack([c.foreground_mask for c in cameras]).cuda()
        rgb,_=regional_rgb(preds,gt,masks,'uniform','fine');reg=model.compute_regulation(hidden.time_smoothness_weight,hidden.l1_time_planes,hidden.plane_tv_weight)
        assert torch.isfinite(rgb+reg)
        balanced,_=regional_rgb(preds,gt,masks,runtime.a.policy,'fine')
    assert model_identity(model)==inherited,'fine setup must preserve model/SH/radii'
    save_json(runtime.output/'branch_validation.json',dict(status='passed',parent=identity(runtime.a.branch_from_coarse),inherited_model_identity=inherited,
        first_fine_batch_frame_ids=[c.image_name for c in cameras],uniform_RGB=float(rgb),regularization=float(reg),uniform_total=float(rgb+reg),balanced_RGB=float(balanced),
        optimization_steps=0,historical_total_direct_comparison='NA: V3 first-fine log has no batch IDs and l1 aliases total; not used as pure RGB reference',
        RNG_restored_after_model_loader_construction=True,coarse_remaining_stack_not_used=True))
    rng_restore(p['rng'])

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    assert a.policy in ['balanced_fine','balanced_all'];assert bool(a.branch_from_coarse)==(a.policy=='balanced_fine')
    sanity_check();torch.set_num_threads(4);official.setup_seed(12345)
    _,(dataset,hidden,opt,pipe)=official_config(2);parent_cfg=read(OLD/'runs/hos_backpack_formal/effective_config.json')
    for obj,key in [(hidden,'hidden'),(opt,'optimization'),(pipe,'pipeline')]:assert vars(obj)==parent_cfg[key]
    dataset.source_path=str(OLD/'inputs/hos_backpack/manifest.json');dataset.model_path=str(Path(a.output).absolute());dataset.render_process=False
    out=Path(dataset.model_path)
    if (out/'effective_config.json').exists() and not a.resume:raise FileExistsError('Existing attempt: only own full-state resume allowed')
    if a.check:coarse_end,fine_end=a.coarse_steps,a.fine_steps
    else:coarse_end,fine_end=opt.coarse_iterations,opt.iterations
    if a.branch_from_coarse:coarse_end=0
    cfg=dict(model=vars(dataset),hidden=vars(hidden),optimization=vars(opt),pipeline=vars(pipe),seed=12345,official_commit=OFFICIAL_COMMIT,
        check=a.check,loss_policy=a.policy,mask_policy='training mask>=128, foreground_mask only; raw RGB per-image regional mean',
        stage_ends=dict(coarse=coarse_end,fine=fine_end),parent_effective_config=identity(OLD/'runs/hos_backpack_formal/effective_config.json'),
        manifest=identity(dataset.source_path),parent_coarse=identity(a.branch_from_coarse) if a.branch_from_coarse else None,
        sources=[identity(RUN/'code'/n) for n in ['common.py','adapter_v4.py','loss_policy.py','train_stage.py']])
    out.mkdir(parents=True,exist_ok=True)
    if a.resume:
        assert read(out/'effective_config.json')==cfg
        state=torch.load(a.resume,map_location='cpu',weights_only=False);assert state['effective_config_sha256']==sha(out/'effective_config.json')
        assert Path(a.resume).parent.absolute()==out.absolute();del state
    else:save_json(out/'effective_config.json',cfg)
    runtime=Runtime(a)
    try:
        model=official.GaussianModel(dataset.sh_degree,hidden);scene=ProtocolScene(dataset,model)
        if runtime.parent is not None:validate_branch(runtime,model,scene,opt,hidden,pipe,dataset)
        runtime.install_density_audit(model);loop=patched_loop(runtime);official.network_gui.try_connect=lambda:None
        timer=official.Timer();timer.start();resume_stage=runtime.resume['stage'] if runtime.resume else None
        for stage,count in [('coarse',coarse_end),('fine',fine_end)]:
            if not count or (resume_stage=='fine' and stage=='coarse'):continue
            runtime.stage_end=count;checkpoint=a.resume if resume_stage==stage else None
            loop(dataset,opt,hidden,pipe,[],[],[],checkpoint,-1,model,scene,stage,None,count,timer)
            runtime.resume=None
        torch.cuda.synchronize()
        save_json(out/'run.json',dict(status='check_completed' if a.check else 'completed',policy=a.policy,
            nominal_iterations=runtime.prior_steps+runtime.attempted,optimizer_updates=runtime.updates,
            checkpoint=str(runtime.last_checkpoint),checkpoint_sha256=sha(runtime.last_checkpoint),source_config_sha256=sha(out/'effective_config.json'),
            seconds=runtime.elapsed_prior+time.monotonic()-runtime.started,peak_points=runtime.peak_points,final_points=len(model.get_xyz),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),GPU='physical1 RTX3090',
            historical_coarse_nominal_steps=opt.coarse_iterations if a.branch_from_coarse else 0,test_RGB_loaded=False))
    except PlannedStop:
        save_json(out/'planned_stop.json',dict(status='planned_temporary_resume_check_stop',checkpoint=str(runtime.last_checkpoint),steps=runtime.prior_steps+runtime.attempted,optimizer_updates=runtime.updates))
    except BaseException:
        save_json(out/'failure.json',dict(status='failed',stage=runtime.stage,iteration=runtime.iteration,attempted_iterations=runtime.prior_steps+runtime.attempted,optimizer_updates=runtime.updates,seconds=runtime.elapsed_prior+time.monotonic()-runtime.started,traceback=traceback.format_exc()))
        raise
    finally:runtime.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--policy',required=True);p.add_argument('--branch-from-coarse');p.add_argument('--resume');p.add_argument('--check',action='store_true');p.add_argument('--coarse-steps',type=int,default=5);p.add_argument('--fine-steps',type=int,default=10);p.add_argument('--stop-after',type=int);run(p.parse_args())
