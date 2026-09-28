"""Locked official fine schedule: common Q and independent temporal evidence."""
from common import *
import torch,argparse,time,traceback,math
from replay_first_failure import Runtime as BaseRuntime,make_loop
from restore_state import *
from finite_guard import named_model
from adapter_v4 import ProtocolScene
from loss_policy import regional_rgb
from hoi_modules.projected_motion import make_renderer
from hoi_modules.temporal_evidence import TemporalEvidenceModule
from utils.loss_utils import ssim
import train as official
import numpy as np

class Runtime(BaseRuntime):
    def __init__(self,a,parent,cfg):
        super().__init__(a,parent,cfg);self.ledger.close();self.conf=read(RUN/'configs/v8.json')
        self.category='D' if a.diagnostic else 'C';self.lp=RUN/'protocol'/(self.category+'_steps.jsonl');self.global_used=len(self.lp.read_text().splitlines()) if self.lp.exists() else 0;self.ledger=self.lp.open('a',buffering=1)
        self.pre_path=Path(a.parent);self.updates=parent.get('optimizer_updates',0);self.start_updates=self.updates;self.stage_end=a.target
        self.module=TemporalEvidenceModule(a.mode,a.scale_bound)
        self.cache=read(RUN/'track_cache_manifest.json');self.cache_hash=sha(RUN/'track_cache_manifest.json')
        self.pair_data=[dict(np.load(p['cache']['path'])) for p in self.cache['pairs']]
        order=torch.load(RUN/'protocol/pair_order.pt',weights_only=False);self.pair_order=order['order'];self.temporal_generator=order['generator_state'];self.cursor=0
        self.lam=1. if a.diagnostic else (read(RUN/'lambda_calibration.json')['lambda0'] if a.mode!='off' else 0.)
        self.calibration_hash=None if a.diagnostic or a.mode=='off' else sha(RUN/'lambda_calibration.json')
        if a.start_kind=='own_checkpoint':
            t=parent['temporal_state'];assert t['mode']==a.mode and t['cache_sha256']==self.cache_hash and t['lambda0']==self.lam and t['calibration_sha256']==self.calibration_hash
            exact(t['generator_state'],self.temporal_generator);assert t['pair_order_sha256']==sha(RUN/'protocol/pair_order.pt');assert t['global_iteration']==parent['iteration'];self.cursor=t['cursor']
        self.optimizer_ledger=(RUN/'protocol/optimizer_calls.jsonl').open('a',buffering=1)
        self.temporal_log=(self.output/'temporal_stats.jsonl').open('a',buffering=1)
    def resume_sampler(self,stage,cameras,stack,temp,model):
        self.cameras={c.image_name:c for c in cameras};return super().resume_sampler(stage,cameras,stack,temp,model)
    def before_step(self,stage,iteration,model,stack,temp):
        assert iteration==self.parent_state['iteration']+self.attempted+1
        assert self.global_used+self.attempted<self.conf['budgets'][self.category+'_attempts']
        self.stage=stage;self.iteration=iteration;self.batch=[];model.optimizer.zero_grad(set_to_none=True);self.guard.check('pre_step',named_model(model))
        self.attempted+=1;self.ledger.write(json.dumps(dict(run=self.output.name,iteration=iteration,purpose=self.a.purpose,pid=os.getpid(),time_unix=time.time()))+'\n')
    def rgb_loss(self,pred,gt,cameras,stage):
        masks=torch.stack([c.foreground_mask for c in cameras]).to(pred.device)
        self.U,rows=regional_rgb(pred,gt,masks,'uniform',stage)
        self.S=ssim(pred,gt);self.Q=.8*self.U+.2*(1-self.S)
        self.batch=[c.image_name for c in cameras];self.rgb_rows=rows;self.rgb_value=float(self.Q.detach())
        self.logs['sampling_order.jsonl'].write(json.dumps(dict(stage=stage,iteration=self.iteration,frame_ids=self.batch))+'\n');return self.Q
    def regularization(self,*args):self.R=super().regularization(*args);return self.R
    def backward(self,loss,model,qlist):
        self.before_backward(loss,model);loss.backward();self.temporal_stats=None
        # Q-only screen derivatives are already populated. Auxiliary screen does not require grad.
        if self.a.mode!='off' and self.iteration>1000 and self.iteration%self.conf['cadence']==0:
            pid=int(self.pair_order[self.cursor]);p=self.cache['pairs'][pid];self.cursor+=1
            before=[q.grad.clone() for q in qlist] if self.a.diagnostic else None
            l,stats,_=self.module(model,self.cameras[p['source_frame']],self.cameras[p['target_frame']],self.pair_data[pid])
            weight=self.lam*min(1.,(self.iteration-1000)/1000)
            self.guard.check('temporal_loss',l);(weight*l).backward()
            if before is not None:assert all(torch.equal(a,q.grad) for a,q in zip(before,qlist))
            stats.update(iteration=self.iteration,pair_id=pid,pair_cursor=self.cursor,lambda_effective=weight,mode=self.a.mode)
            self.temporal_log.write(json.dumps(stats,allow_nan=False)+'\n');self.temporal_stats=stats
    def optimizer_step(self,model):
        self.optimizer_ledger.write(json.dumps(dict(purpose=self.a.purpose,run=self.output.name,iteration=self.iteration,event='enter',pid=os.getpid()))+'\n');super().optimizer_step(model)
        self.optimizer_ledger.write(json.dumps(dict(purpose=self.a.purpose,run=self.output.name,iteration=self.iteration,event='completed',pid=os.getpid()))+'\n')
    def report(self,writer,iteration,l1,loss,loss_fn,ms,tests,scene,render,render_args,stage,dataset_type):
        assert not tests and not scene.getTestCameras()
        row=dict(iteration=iteration,batch=self.batch,U=float(self.U.detach()),SSIM=float(self.S.detach()),Q=float(self.Q.detach()),R=float(self.R.detach()),points=len(scene.gaussians.get_xyz),seconds=time.monotonic()-self.started,iter_gpu_ms=float(ms),temporal=self.temporal_stats)
        self.logs['training_metrics.jsonl'].write(json.dumps(row)+'\n')
        if iteration%100==0 or self.a.diagnostic:save_json(self.output/'progress.json',row)
    def after_step(self,stage,iteration,model,stack,temp):
        self.guard.check('completed_model_Adam_buffers',named_model(model));self.peak_points=max(self.peak_points,len(model.get_xyz));self.event('completed_step',optimizer_step_calls=self.updates)
        if iteration%100==0 or iteration==self.stage_end or iteration in self.a.save_iterations:
            fixed=iteration%1000==0 or iteration==self.stage_end or iteration in self.a.save_iterations
            path=self.output/(f'checkpoint_fine_{iteration:06d}.pt' if fixed else f'rolling_{iteration:06d}.pt')
            temporal=dict(mode=self.a.mode,cursor=self.cursor,generator_state=self.temporal_generator,pair_order_sha256=sha(RUN/'protocol/pair_order.pt'),cache_sha256=self.cache_hash,lambda0=self.lam,calibration_sha256=self.calibration_hash,cadence=self.conf['cadence'],global_iteration=iteration,warmup_fraction=min(1.,max(0.,(iteration-1000)/1000)))
            state=capture(model,stage,iteration,stack,temp,phase='completed_step',resumable=iteration<14000,policy=self.a.mode,effective_config_sha256=sha(self.output/'effective_config.json'),optimizer_updates=self.updates,purpose=self.a.purpose,lineage=self.a.lineage,scientific_fingerprint=self.a.fingerprint,temporal_state=temporal)
            asset=atomic_checkpoint(path,state);self.last_checkpoint=path;self.pre_path=path
            with (self.output/'checkpoint_index.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=iteration,checkpoint=asset))+'\n')
            save_json(self.output/'latest_checkpoint.json',dict(checkpoint=asset,iteration=iteration,resumable=iteration<14000))
            for p in sorted(self.output.glob('rolling_*.pt'))[:-2]:p.unlink()
        self.U=self.S=self.Q=self.R=None
    def close(self):super().close();self.optimizer_ledger.close();self.temporal_log.close()

def temporal_loop(rt,render):
    make_loop(rt,render);path=rt.output/'source_snapshots/train_observed.py';source=path.read_text();old='runtime.before_backward(loss,gaussians)\n        loss.backward()';assert source.count(old)==1;source=source.replace(old,'runtime.backward(loss,gaussians,viewspace_point_tensor_list)')
    path=rt.output/'source_snapshots/train_temporal.py';path.write_text(source)
    assert rt.current_cfg['optimization']['lambda_dssim']==0 and rt.current_cfg['optimization']['lambda_lpips']==0
    ns=official.__dict__.copy();ns.update(runtime=rt,restore_model=restore_model,training_report=rt.report,render=render);exec(compile(source,str(path),'exec'),ns);return ns['scene_reconstruction']

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    torch.set_num_threads(4);official.setup_seed(12345);parent=load_completed(a.parent);out=Path(a.output);assert not out.exists()
    a.diagnostic=a.purpose=='diagnostic';a.policy=a.mode;conf=read(RUN/'configs/v8.json');pc=read(V5/'runs/B_U/effective_config.json');_,(dataset,hidden,opt,pipe)=official_config(2)
    for obj,key in [(hidden,'hidden'),(opt,'optimization'),(pipe,'pipeline')]:assert vars(obj)==pc[key]
    a.lineage=conf['parent_hashes']['B_U_fine1000'] if a.diagnostic else (conf['parent_hashes']['shared_fine0'] if a.mode=='off' else sha(RUN/'runs/B_Q/checkpoint_fine_001000.pt'))
    science=dict(hidden=pc['hidden'],optimization=pc['optimization'],pipeline=pc['pipeline'],seed=12345,scale_bound=pc['scale_bound'],source_path=pc['model']['source_path'],mode=a.mode,purpose=a.purpose,lineage=a.lineage,config_sha256=sha(RUN/'configs/v8.json'),cache_sha256=sha(RUN/'track_cache_manifest.json'))
    a.fingerprint=hashlib.sha256(json.dumps(science,sort_keys=True).encode()).hexdigest()
    if a.start_kind=='own_checkpoint':
        assert Path(a.parent).resolve().is_relative_to(RUN.resolve());assert parent['scientific_fingerprint']==a.fingerprint
        assert parent['effective_config_sha256']==sha(Path(a.parent).parent/'effective_config.json')
        assert any(x['checkpoint']['sha256']==sha(a.parent) for x in map(json.loads,(Path(a.parent).parent/'checkpoint_index.jsonl').read_text().splitlines()))
    else:
        assert sha(a.parent)==a.lineage
        assert parent['iteration']==(1000 if a.diagnostic or a.mode!='off' else 0)
    if not a.diagnostic:
        assert a.target==14000 and read(RUN/'module_acceptance.json')['status']=='pass'
        if a.mode!='off':assert read(RUN/'lambda_calibration.json')['status']=='pass'
    else:assert a.target==1112
    dataset.source_path=pc['model']['source_path'];dataset.model_path=str(out.resolve());dataset.render_process=False
    a.scale_bound=pc['scale_bound'];a.light=True;a.check=a.diagnostic;a.replay_from=a.parent;a.steps=a.target-parent['iteration']
    cfg=dict(model=vars(dataset),hidden=vars(hidden),optimization=vars(opt),pipeline=vars(pipe),seed=12345,scale_bound=a.scale_bound,policy=a.mode,parent=identity(a.parent),purpose=a.purpose,lineage=a.lineage,scientific_config=science,scientific_fingerprint=a.fingerprint,start_kind=a.start_kind,quality_objective=conf['quality_objective'],parent_Adam_RNG_stack_preserved=True)
    out.mkdir(parents=True);save_json(out/'effective_config.json',cfg);rt=Runtime(a,parent,cfg);model=None
    try:
        model=official.GaussianModel(dataset.sh_degree,hidden);scene=ProtocolScene(dataset,model);rt.install_density_audit(model);render=make_renderer(a.scale_bound)
        (out/'source_snapshots').mkdir(exist_ok=True);(out/'source_snapshots/render.py').write_text(render.expanded_source)
        loop=temporal_loop(rt,render);official.network_gui.try_connect=lambda:None;timer=official.Timer();timer.start();loop(dataset,opt,hidden,pipe,[],[],[],a.parent,-1,model,scene,'fine',None,a.target,timer)
        torch.cuda.synchronize();save_json(out/'run.json',dict(status='completed',nominal_iterations=rt.stage_end,attempted_this_process=rt.attempted,optimizer_updates=rt.updates,updates_this_process=rt.updates-rt.start_updates,checkpoint=str(rt.last_checkpoint),checkpoint_sha256=sha(rt.last_checkpoint),final_points=len(model.get_xyz),peak_points=rt.peak_points,seconds=time.monotonic()-rt.started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),GPU='physical1 RTX3090',purpose=a.purpose,mode=a.mode,start_kind=a.start_kind,temporal_cursor=rt.cursor))
    except BaseException:
        error=traceback.format_exc();save_json(out/'failure.json',dict(status='failed',iteration=rt.iteration,attempted_this_process=rt.attempted,updates_this_process=rt.updates-rt.start_updates,phase=rt.phase,last_finite_saved_state=str(rt.pre_path),traceback=error));raise
    finally:rt.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--output',required=True);p.add_argument('--start-kind',choices=['shared_fine','own_checkpoint','diagnostic_parent','Q_prefix'],required=True);p.add_argument('--purpose',choices=['formal','diagnostic'],required=True);p.add_argument('--mode',choices=TemporalEvidenceModule.MODES,required=True);p.add_argument('--target',type=int,required=True);p.add_argument('--save-iterations',type=int,nargs='*',default=[]);run(p.parse_args())
