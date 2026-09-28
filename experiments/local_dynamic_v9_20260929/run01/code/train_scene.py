"""V9 exact-update trainer: one Q, RGB density gradients, complete recovery."""
from common import *
import argparse, traceback, types
import numpy as np
import torch
from state import *
from render_scene import make_renderer, rgb_background
from hoi_modules.static_dynamic_gaussians import validate_point_state


def finite(tensors, label):
    values=[t.detach() for t in tensors if torch.is_tensor(t) and t.is_floating_point() and t.numel()]
    for device in {t.device for t in values}:
        if not bool(torch.stack([torch.isfinite(t).all() for t in values if t.device==device]).all()):
            raise FloatingPointError(label)


def all_state_tensors(model):
    out=[getattr(model,n) for n in ATTRS]+list(model._deformation.parameters())
    out += [model.max_radii2D, model.xyz_gradient_accum, model.denom, model._deformation_accum]
    out += [v for s in model.optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
    return out


def install_density_audit(model, output, cursor):
    for name in ('densification_postfix','prune_points'):
        original=getattr(model,name)
        def wrapped(*args,_name=name,_original=original,**kw):
            before=len(model._xyz); before_variable=int(model._deformation_table.sum())
            result=_original(*args,**kw)
            validate_point_state(model)
            append_json(output/'density_events.jsonl',dict(iteration=cursor['iteration'],operation=_name,
                points_before=before,points_after=len(model._xyz),variable_before=before_variable,
                variable_after=int(model._deformation_table.sum())))
            return result
        setattr(model,name,wrapped)


def density_step(model, opt, stage, iteration, pkgs, extent, output):
    if iteration >= opt.densify_until_iter:
        return
    radii=torch.stack([p['radii'] for p in pkgs]).max(0).values
    visible=torch.stack([p['visibility_filter'] for p in pkgs]).any(0)
    screens=[p['viewspace_points'].grad for p in pkgs]
    assert all(x is not None for x in screens)
    screen_grad=torch.stack(screens).sum(0)
    model.max_radii2D[visible]=torch.maximum(model.max_radii2D[visible],radii[visible])
    model.add_densification_stats(screen_grad,visible)
    if stage=='coarse':
        opacity=opt.opacity_threshold_coarse; threshold=opt.densify_grad_threshold_coarse
    else:
        opacity=opt.opacity_threshold_fine_init-iteration*(opt.opacity_threshold_fine_init-opt.opacity_threshold_fine_after)/opt.densify_until_iter
        threshold=opt.densify_grad_threshold_fine_init-iteration*(opt.densify_grad_threshold_fine_init-opt.densify_grad_threshold_after)/opt.densify_until_iter
    size=20 if iteration>opt.opacity_reset_interval else None
    if iteration>opt.densify_from_iter and iteration%opt.densification_interval==0 and len(model._xyz)<360000:
        model.densify(threshold,opacity,extent,size,5,5,str(output),iteration,stage)
    if iteration>opt.pruning_from_iter and iteration%opt.pruning_interval==0 and len(model._xyz)>200000:
        model.prune(threshold,opacity,extent,size)
    if iteration%opt.opacity_reset_interval==0:
        model.reset_opacity()


def engine(parent, mode, target, output, kind='formal', checkpoint_at=(), cameras=None, probe=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic(); completed=parent['completed_updates']; initial_completed=completed
    scene=parent['scene']; stage=parent['stage']; cfg=config()
    model,(ds,hidden,opt,pipe)=restore(parent,mode,reset_optimizer=completed==0)
    sched=schedule(scene,stage)
    exact(parent['sampler'],sampler_at(sched,completed))
    cams=cameras if cameras is not None else load_cameras(scene,stage=='coarse')
    render=make_renderer(parent['metadata']['scale_bound'],mode)
    (output/'render_source.txt').write_text(render.expanded_source)
    bg=rgb_background(ds); cursor={'iteration':completed}; install_density_audit(model,output,cursor)
    global_ledger=RUN/'protocol'/('formal_attempts.jsonl' if kind=='formal' else 'diagnostic_attempts.jsonl')
    used=len(global_ledger.read_text().splitlines()) if global_ledger.exists() else 0
    limit=cfg['budgets']['formal_attempts' if kind=='formal' else 'integrated_Adam_updates']
    assert used+target-completed<=limit
    effective=dict(model=vars(ds),hidden=vars(hidden),optimization=vars(opt),pipeline=vars(pipe),
        scene=scene,mode=mode,stage=stage,target=target,scale_bound=parent['metadata']['scale_bound'],
        parent_completed_updates=completed,input=parent['metadata'],RGB_schedule_sha256=sched['schedule_sha256'],
        schema=SCHEMA,background='white',quality='single Q' if stage=='fine' else 'masked background RGB L1')
    save_json(output/'effective_config.json',effective)
    latest=None; attempted=0; peak_points=len(model._xyz); skip_images=0
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    ledger=global_ledger.open('a',buffering=1)
    logs=(output/'training_metrics.jsonl').open('a',buffering=1)
    try:
        for iteration in range(completed+1,target+1):
            cursor['iteration']=iteration; attempted+=1
            ledger.write(json.dumps(dict(scene=scene,mode=mode,stage=stage,iteration=iteration,pid=os.getpid(),
                output=str(output),time_unix=time.time()))+'\n')
            tick=time.monotonic(); model.optimizer.zero_grad(set_to_none=True)
            model.update_learning_rate(iteration)
            if iteration%1000==0:model.oneupSHdegree()
            batch=[cams[i] for i in sched['batches'][iteration-1]]
            pkgs=[render(c,model,pipe,bg,stage=stage) for c in batch]
            pred=torch.stack([p['render'] for p in pkgs]); gt=torch.stack([c.original_image.cuda() for c in batch])
            if stage=='coarse':
                weights=torch.stack([c.background_weight for c in batch]).to(pred.device)
                rgb,skips=background_objective(pred,gt,weights); skip_images+=skips
                l1=rgb; structural=None; regularization=rgb.new_zeros(())
            else:
                rgb,l1,structural=quality_objective(pred,gt)
                regularization=model.compute_regulation(hidden.time_smoothness_weight,hidden.l1_time_planes,hidden.plane_tv_weight)
            loss=rgb+regularization
            finite([loss],f'loss iteration {iteration}')
            if probe is not None:probe(model,rgb,iteration)
            loss.backward()
            gradients=[p.grad for g in model.optimizer.param_groups for p in g['params'] if p.grad is not None]
            finite(gradients,f'gradient iteration {iteration}')
            with torch.no_grad():
                density_step(model,opt,stage,iteration,pkgs,parent['metadata']['scene_extent'],output)
                # Every scheduled iteration, including the last, calls Adam.
                model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
            completed=iteration; peak_points=max(peak_points,len(model._xyz))
            row=dict(scene=scene,mode=mode,stage=stage,completed_updates=completed,
                frame_uids=sched['batches'][iteration-1],frame_ids=[c.image_name for c in batch],
                RGB=float(rgb.detach()),L1=float(l1.detach()),SSIM11=None if structural is None else float(structural.detach()),
                regularization=float(regularization.detach()),points=len(model._xyz),
                variable_points=int(model._deformation_table.sum()),seconds=time.monotonic()-start,
                iteration_seconds=time.monotonic()-tick)
            logs.write(json.dumps(row)+'\n')
            if iteration%100==0 or iteration==target:
                save_json(output/'progress.json',dict(status='running',**row,pid=os.getpid()))
                print(scene,mode,stage,iteration,len(model._xyz),round(row['seconds'],2),flush=True)
            interval=cfg['schedule']['checkpoint_interval']
            if iteration%interval==0 or iteration==target or iteration in checkpoint_at:
                finite(all_state_tensors(model),f'checkpoint state {iteration}')
                state=capture(model,scene,mode,stage,completed,sched,parent['metadata'])
                permanent=iteration==target or iteration in checkpoint_at
                path=output/(f'checkpoint_{stage}_{iteration:06d}.pt' if permanent else f'rolling_{iteration:06d}.pt')
                asset=atomic_checkpoint(path,state); latest=asset
                save_json(output/'latest_checkpoint.json',dict(completed_updates=completed,checkpoint=asset))
                append_json(output/'checkpoint_index.jsonl',dict(completed_updates=completed,checkpoint=asset))
                for old in sorted(output.glob('rolling_*.pt'))[:-2]:old.unlink()
            del pred,gt,pkgs,loss,rgb,l1,structural,regularization,gradients
        torch.cuda.synchronize(); finite(all_state_tensors(model),'terminal model and Adam')
        save_json(output/'run.json',dict(status='completed',scene=scene,mode=mode,stage=stage,
            completed_updates=completed,attempted_this_process=attempted,updates_this_process=completed-initial_completed,
            checkpoint=latest,final_points=len(model._xyz),variable_points=int(model._deformation_table.sum()),
            peak_points=peak_points,seconds=time.monotonic()-start,skipped_BG_images=skip_images,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            physical_GPU=1,pid=os.getpid()))
        return model,torch.load(latest['path'],map_location='cpu',weights_only=False)
    except BaseException:
        save_json(output/'failure.json',dict(status='failed',scene=scene,mode=mode,stage=stage,
            completed_updates=completed,attempted_this_process=attempted,updates_this_process=completed-initial_completed,
            checkpoint=latest,seconds=time.monotonic()-start,traceback=traceback.format_exc()))
        try:
            atomic_checkpoint(output/'failure_state.pt',dict(model=clone_cpu(model.capture()),
                deformation_accum=clone_cpu(model._deformation_accum),completed_updates=completed,
                attempted_iteration=cursor['iteration'],rng=rng_capture(),resumable=False,
                gradients={g['name']:[clone_cpu(p.grad) for p in g['params']] for g in model.optimizer.param_groups}))
        except BaseException:pass
        raise
    finally:
        ledger.close(); logs.close()


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    assert torch.cuda.get_device_name(0)==config()['hardware']['name']
    torch.set_num_threads(config()['hardware']['CPU_threads'])
    sd=scene_dir(a.scene);stage='coarse' if a.mode=='BG' else 'fine'
    out=sd/'runs'/a.mode
    if a.resume:
        assert out.exists() and not (out/'run.json').exists()
        assert not (out/'resume_receipt.json').exists(),'Only one external recovery per job'
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['scene']==a.scene and state['mode']==a.mode
        rows=[json.loads(x) for x in (RUN/'protocol/formal_attempts.jsonl').read_text().splitlines()]
        relevant=[x for x in rows if x['scene']==a.scene and x['mode']==a.mode]
        last=max(x['iteration'] for x in relevant)
        assert 0<=last-state['completed_updates']<=config()['budgets']['replay_updates_per_job']
        save_json(out/'resume_receipt.json',dict(parent=identity(a.resume),replay_updates=last-state['completed_updates'],
            reason=a.external_reason,time_unix=time.time()))
    else:
        assert not out.exists(),str(out)
        path=sd/'protocol'/('coarse_initial_verified.pt' if stage=='coarse' else 'shared_fine0.pt')
        state=torch.load(path,map_location='cpu',weights_only=False)
    for asset in state['metadata']['source_code']:
        assert sha(asset['path'])==asset['sha256'],asset['path']
    target=config()['schedule'][stage+'_updates']
    model,terminal=engine(state,a.mode,target,out,checkpoint_at=([config()['schedule']['middle_snapshot']] if stage=='fine' else []))
    if stage=='coarse':
        from prepare_initialization import compose_fine
        fine,report=compose_fine(model,a.scene,state['metadata'])
        asset=atomic_checkpoint(sd/'protocol/shared_fine0.pt',fine)
        save_json(sd/'protocol/fine_transition.json',dict(status='passed',fine0=asset,**report))
        # Real shared fine0 integrity, no quality scores and no optimizer calls.
        from verify import identity_renders
        save_json(sd/'protocol/fine0_equivalence.json',identity_renders(fine))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True)
    p.add_argument('--mode',required=True,choices=['BG','Q0','S','SL'])
    p.add_argument('--resume');p.add_argument('--external-reason')
    a=p.parse_args();assert not a.resume or a.external_reason
    run(a)
