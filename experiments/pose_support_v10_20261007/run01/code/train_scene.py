"""V10 exact fresh-Adam continuation; base topology fixed, support after-step."""
from common import *
from state import *
from render_scene import make_renderer,rgb_background
import argparse,traceback,cv2
from hoi_modules.pose_prior_adapter import PosePriorAdapter
from hoi_modules.pose_support_gaussians import PoseSupportGaussians
from hoi_modules.region_reconstruction import reconstruction_loss
def initial(scene,arm):
    base,objs,meta=import_parent(scene);seed_all(config()['seed'])
    prior=PosePriorAdapter.from_cache(scene_dir(scene)/'protocol/pose_cache.pt')
    seed_asset=scene_dir(scene)/'protocol/support_seed.pt'
    support=None if arm=='C' else PoseSupportGaussians(torch.load(seed_asset,map_location='cpu',weights_only=False),
        scene_extent=meta['scene_extent'],scale_bound=meta['scale_bound'],max_points=60000).cuda()
    optimizer=new_optimizer(base,support)
    input_identity=scene_dir(scene)/'protocol/input_identity.json'
    meta.update(pose_cache=identity(scene_dir(scene)/'protocol/pose_cache.pt'),support_seed=identity(seed_asset),
        input_identity=identity(input_identity),config=identity(RUN/'configs/v10.json'),
        RGB_schedule=identity(scene_dir(scene)/'protocol/RGB_schedule.json'),
        source_code=read(RUN/'protocol/source_identity.json')['files'] if (RUN/'protocol/source_identity.json').exists() else [])
    return base,support,optimizer,objs,meta,prior
def engine(scene,arm,target,output,purpose='formal',resume=None,probe=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cfg=config();start=time.monotonic()
    if resume:
        s=torch.load(resume,map_location='cpu',weights_only=False);assert s['scene']==scene and s['arm']==arm
        base,support,optimizer,objs,meta=restore(s);completed=s['completed_updates']
        prior=PosePriorAdapter.from_cache(scene_dir(scene)/'protocol/pose_cache.pt')
    else:base,support,optimizer,objs,meta,prior=initial(scene,arm);completed=0
    initial_completed=completed;ds,h,opt,pipe=objs;sched=schedule(scene);cams,frames=load_scene_cameras(scene)
    masks=[torch.from_numpy(cv2.imread(f['mask_path'],0)>=128) for f in frames] if arm=='PQ' else None
    assert len(sched['batches'])==20000 and 0<=completed<target<=20000
    for asset in meta['source_code']:assert sha(asset['path'])==asset['sha256'],asset['path']
    for key in ('pose_cache','support_seed','input_identity','config','RGB_schedule','input_manifest','parent'):
        assert sha(meta[key]['path'])==meta[key]['sha256'],key
    render=make_renderer(meta['scale_bound']);bg=rgb_background(ds);nbase=len(base._xyz);peak_support=0 if support is None else len(support.u)
    ledger=RUN/'protocol'/('formal_attempts.jsonl' if purpose=='formal' else 'diagnostic_attempts.jsonl')
    used=sum(1 for _ in ledger.open()) if ledger.exists() else 0
    limit=cfg['budgets']['formal_attempts' if purpose=='formal' else 'integrated_Adam_updates']
    assert used+target-completed<=limit
    latest=None;attempted=0;torch.cuda.reset_peak_memory_stats();first_lr=None;last_lr=None
    effective=dict(scene=scene,arm=arm,target=target,purpose=purpose,input=meta,
        model=vars(ds),hidden=vars(h),optimization=vars(opt),pipeline=vars(pipe),
        scale_bound=meta['scale_bound'],base_points=nbase,base_topology='fixed',Adam='one_fresh_named_group_optimizer',
        quality='Qr' if arm=='PQ' else 'Q',scene_extent=meta['scene_extent'])
    save_json(output/'effective_config.json',effective)
    if resume is None:
        latest=atomic_checkpoint(output/'checkpoint_000000.pt',capture(base,support,optimizer,scene,arm,0,meta))
        save_json(output/'latest_checkpoint.json',dict(completed_updates=0,checkpoint=latest))
        append_json(output/'checkpoint_index.jsonl',dict(completed_updates=0,checkpoint=latest))
    try:
        for k in range(completed+1,target+1):
            tick=time.monotonic();attempted+=1
            append_json(ledger,dict(scene=scene,arm=arm,purpose=purpose,iteration=k,pid=os.getpid(),output=str(output),time_unix=time.time(),event='attempt_started'))
            save_json(output/'attempt_cursor.json',dict(scene=scene,arm=arm,attempted_iteration=k,attempted_this_process=attempted,
                completed_updates=completed,resume_from_updates=initial_completed,pid=os.getpid()))
            optimizer.zero_grad(set_to_none=True);last_lr=set_learning_rates(optimizer,base,k)
            if first_lr is None:first_lr=last_lr.copy()
            batch_idx=sched['batches'][k-1];batch=[cams[i] for i in batch_idx]
            packages=[]
            for cam in batch:
                bones=None if support is None else prior.bone_transforms(cam.image_name,device='cuda')
                packages.append(render(cam,base,pipe,bg,support=support,bones=bones,scene_extent=meta['scene_extent']))
            pred=torch.stack([p['render'] for p in packages]);gt=torch.stack([c.original_image.cuda() for c in batch])
            mask=None if masks is None else torch.stack([masks[i] for i in batch_idx]).cuda()
            rgb,terms=reconstruction_loss(pred,gt,mask=mask,regional=arm=='PQ',return_terms=True)
            regularization=base.compute_regulation(h.time_smoothness_weight,h.l1_time_planes,h.plane_tv_weight)
            loss=rgb+regularization;finite([loss],f'loss {k}');loss.backward()
            gradients=[p.grad for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
            finite(gradients,f'gradient {k}')
            if probe:probe('after_backward',k,base,support,optimizer,packages,terms)
            with torch.no_grad():
                if support is not None:
                    for pkg in packages:
                        grad=pkg['viewspace_points'].grad;assert grad is not None
                        support.accumulate_density(grad[nbase:],pkg['visibility_filter'][nbase:],pkg['radii'][nbase:],batch_size=len(batch))
                optimizer.step()  # Exactly one Adam call, including final update.
                optimizer.zero_grad(set_to_none=True)
                if support is not None:
                    event=support.after_adam_topology(optimizer,k)
                    if event is not None:append_json(output/'density_events.jsonl',event)
                    support.set_active_sh_after_update(k);peak_support=max(peak_support,len(support.u))
                assert len(base._xyz)==nbase
            completed=k
            if probe:probe('after_update',k,base,support,optimizer,packages,terms)
            row=dict(scene=scene,arm=arm,completed_updates=k,attempted_this_process=attempted,updates_this_process=k-initial_completed,
                frame_ids=[c.image_name for c in batch],frame_uids=batch_idx,RGB=float(rgb.detach()),regularization=float(regularization.detach()),
                points=nbase+(0 if support is None else len(support.u)),base_points=nbase,support_points=0 if support is None else len(support.u),
                seconds=time.monotonic()-start,iteration_seconds=time.monotonic()-tick)
            append_json(output/'training_metrics.jsonl',row)
            if k%100==0 or k==target:
                save_json(output/'progress.json',dict(status='running',**row,pid=os.getpid()))
                print(scene,arm,k,row['support_points'],round(row['seconds'],2),flush=True)
            if k%250==0 or k==target or k==10000:
                finite(all_state_tensors(base,support,optimizer),f'checkpoint {k}')
                payload=capture(base,support,optimizer,scene,arm,k,meta)
                path=output/(f'checkpoint_{k:06d}.pt' if k==target or k==10000 else f'rolling_{k:06d}.pt')
                latest=atomic_checkpoint(path,payload)
                save_json(output/'latest_checkpoint.json',dict(completed_updates=k,checkpoint=latest))
                append_json(output/'checkpoint_index.jsonl',dict(completed_updates=k,checkpoint=latest))
                for old in sorted(output.glob('rolling_*.pt'))[:-2]:old.unlink()
            del pred,gt,loss,rgb,regularization,gradients,packages
        torch.cuda.synchronize();finite(all_state_tensors(base,support,optimizer),'terminal model/Adam')
        receipt=dict(status='completed',scene=scene,arm=arm,completed_updates=completed,attempted_this_process=attempted,
            updates_this_process=completed-initial_completed,checkpoint=latest,final_points=nbase+(0 if support is None else len(support.u)),
            base_points=nbase,support_points=0 if support is None else len(support.u),peak_support_points=peak_support,
            seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            physical_GPU=cfg['hardware']['physical_GPU'],GPU_uuid=cfg['hardware']['GPU_uuid'],pid=os.getpid(),
            actual_first_LR=first_lr,actual_last_LR=last_lr)
        save_json(output/'run.json',receipt)
        return base,support,optimizer,capture(base,support,optimizer,scene,arm,completed,meta)
    except BaseException:
        save_json(output/'failure.json',dict(status='failed',scene=scene,arm=arm,completed_updates=completed,
            attempted_this_process=attempted,updates_this_process=completed-initial_completed,checkpoint=latest,
            seconds=time.monotonic()-start,traceback=traceback.format_exc()))
        raise
def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('1',config()['hardware']['GPU_uuid'])
    assert torch.cuda.get_device_name(0)==config()['hardware']['name'];torch.set_num_threads(4)
    output=Path(args.output) if args.output else scene_dir(args.scene)/'runs'/args.arm
    if args.purpose=='formal' and args.resume:
        assert not (output/'resume_receipt.json').exists(),'one recovery per job'
        s=torch.load(args.resume,map_location='cpu',weights_only=False);cursor=read(output/'attempt_cursor.json')
        replay=cursor['attempted_iteration']-s['completed_updates'];assert 0<=replay<=256
        save_json(output/'resume_receipt.json',dict(parent=identity(args.resume),replay_attempts=replay,
            last_Adam_execution='unknown_if_interrupted',time_unix=time.time()))
    if args.purpose=='formal' and not args.resume:assert not output.exists(),str(output)
    engine(args.scene,args.arm,args.target,output,args.purpose,args.resume)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--arm','--mode',dest='arm',required=True,choices=['C','P','PQ'])
    p.add_argument('--purpose',choices=['formal','acceptance'],default='formal');p.add_argument('--target',type=int,default=20000)
    p.add_argument('--resume');p.add_argument('--output');p.add_argument('--root');p.add_argument('--gpu')
    run(p.parse_args())
