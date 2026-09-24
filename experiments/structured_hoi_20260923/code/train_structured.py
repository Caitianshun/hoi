"""S0/S1 structured HOI, shared initialization and one controlled track schedule.

Training reads only the explicit initialization and input RGB-derived caches.
Run prefixes/checkpoints preserve optimizer and RNG. Evaluation is separate.
"""
from pathlib import Path
import argparse,json,hashlib,time,os,socket,traceback,random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from gaussian_scene import StructuredScene

EXP=Path('/home/cai_tianshun/Project/HOI/experiments/structured_hoi_20260923')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()

def atomic_json(path,x):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)

def mean_mask(x,mask):
    if x.ndim==3 and x.shape[0]==3:x=x.mean(0)
    return (x*mask).sum()/mask.sum().clamp_min(1)

def ssim_error(a,b):
    a=a[None];b=b[None];mu_a=F.avg_pool2d(a,7,1,3);mu_b=F.avg_pool2d(b,7,1,3)
    va=F.avg_pool2d(a*a,7,1,3)-mu_a.square();vb=F.avg_pool2d(b*b,7,1,3)-mu_b.square()
    cov=F.avg_pool2d(a*b,7,1,3)-mu_a*mu_b
    score=((2*mu_a*mu_b+.01**2)*(2*cov+.03**2))/((mu_a.square()+mu_b.square()+.01**2)*(va+vb+.03**2))
    return ((1-score[0])/2).clamp(0,1).mean(0)

def temporal_regularization(scene):
    # Physical-time derivatives of small pose corrections retain estimated real motion.
    # They are not evidence that the initial GVHMR motion itself is correct.
    dt=scene.timestamps.diff();loss=scene.object_translation.sum()*0
    corrections=[scene.object_translation-scene.object_t0,scene.object_rot_delta,
                 scene.human.transl-scene.human.initial_transl,
                 scene.human.body_pose-scene.human.initial_body_pose,
                 scene.human.global_orient-scene.human.initial_global_orient]
    for x in corrections:
        vel=x.diff(dim=0)/dt[:,None];acc=vel.diff(dim=0)/((dt[1:]+dt[:-1])*.5)[:,None]
        loss=loss+.01*vel.square().mean()+.00005*acc.square().mean()
    return loss

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--init',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--branch',choices=['prefix','S0','S1','smoke'],required=True);ap.add_argument('--stop',type=int,default=8000)
    ap.add_argument('--resume',type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(12345);np.random.seed(12345);random.seed(12345)
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();start_time=time.perf_counter()
    record=dict(status='running',branch=a.branch,pid=os.getpid(),host=socket.gethostname(),gpu=torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),initialization=str(a.init.resolve()),initialization_sha256=sha(a.init),
                resume=str(a.resume.resolve()) if a.resume else None,reference_used=False,parameter_updates=0,stages={},
                source_identity={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
                schedule={'warmup_end':1500,'joint_pose_end':2500,'total_steps':8000,'track_start':1500,'track_every':4,'track_end_exclusive':6000 if a.branch=='S0' else 8000},
                geometry_interface='MoSca D native_add3, integer pixels, metres, scale1',
                residual_bounds_m={'human_static':.03,'human_regional':.015,'human_gaussian_attachment':.005,'object_canonical_attachment':.005},
                densities={'interval':500,'start':3000,'last':7000,'screen_gradient_threshold':.0002,'prune_alpha':.02,'prune_requires_seen':3},
                color='learned view-independent RGB, shared protocol in S0/S1')
    atomic_json(a.output/'run.json',record)
    try:
        frozen_code={name:sha(Path(__file__).parent/name) for name in ['train_structured.py','gaussian_scene.py','human_lbs.py']}
        record['training_code_identity']=frozen_code
        record['loss_weights']={'rgb_L1':.8,'rgb_SSIM':.2,'joint_balanced_visible_RGB':.15,'instance':.2,'track_px':.01,'human_pose_rot':.03,'human_transl':.1,'human_beta':.01,'object_rot_delta':.005,'object_transl_delta':.05,'canonical_regional_m2':10.,'canonical_spatial_m2':50.,'regional_velocity':.05,'regional_acceleration':.00005,'correction_velocity':.01,'correction_acceleration':.00005}
        init=torch.load(a.init,map_location='cpu',weights_only=False);ws=Path(init['input_dir']);meta=json.loads((ws/'input_manifest.json').read_text())
        assert meta['role']=='input_only' and init['reference_used'] is False
        frozen_data={str(path):sha(path) for path in [ws/'input_manifest.json',Path(init['segmentation']),ws/'uniform_cotracker_tap.npz',Path(init['human_geometry']),Path(init['smplx_model']),Path(init['object_init'])]}
        for path,digest in zip(meta['frame_paths'],meta['frame_sha256']):assert sha(path)==digest,'RGB identity mismatch'
        scene=StructuredScene(init)
        checkpoint=None
        if a.resume:
            checkpoint=torch.load(a.resume,map_location='cpu',weights_only=False);assert checkpoint['initialization_sha256']==sha(a.init)
            assert checkpoint['training_code_identity']==frozen_code,'Training code changed; use a new controlled prefix'
            assert checkpoint['data_identity']==frozen_data,'Input cache changed; reject checkpoint resume'
            scene.resize_for_load(checkpoint['model']);scene.load_state_dict(checkpoint['model'],strict=True)
        scene=scene.cuda();optimizer=torch.optim.Adam(scene.optimizer_groups(),eps=1e-8)
        begin=0
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer']);begin=checkpoint['step']
            torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state(checkpoint['cuda_rng']);np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
        rgb=torch.from_numpy(np.stack([cv2.imread(p)[...,::-1].copy() for p in meta['frame_paths']])).cuda().permute(0,3,1,2).float()/255
        labels=torch.from_numpy(np.load(init['segmentation'])['entity_labels'].astype(np.int64)).cuda()
        masks=F.one_hot(labels,3).permute(0,3,1,2).float()
        interior=-F.max_pool2d(-masks,5,1,2) # erode away uncertain instance boundaries
        L,_,H,W=rgb.shape;K=torch.tensor(init['K'],dtype=torch.float32,device='cuda');w2c=torch.linalg.inv(scene.c2w)
        tr=np.load(ws/'uniform_cotracker_tap.npz');tracks=torch.tensor(tr['tracks'],device='cuda');vis=torch.tensor(tr['visibility'],device='cuda')
        rng=np.random.default_rng(12345);frames=rng.integers(0,L,size=8000);targets=rng.integers(0,L,size=8000)
        # All branches get the same explicit source and destination choices, including after cutoff.
        object_present=np.flatnonzero((labels==2).sum((1,2)).cpu().numpy()>25)
        for s in range(1500):
            if s%3==2:frames[s]=rng.choice(object_present)
        schedule=np.stack([frames,targets],-1);np.save(a.output/'frame_schedule.npy',schedule)
        record['frame_schedule_sha256']=sha(a.output/'frame_schedule.npy');record['start_step']=begin
        record['data_identity']=frozen_data
        losses=open(a.output/'losses.jsonl','w');events=open(a.output/'topology.jsonl','w')
        def save_checkpoint(step):
            path=a.output/f'checkpoint_{step:06d}.pt'
            torch.save(dict(step=step,initialization_sha256=sha(a.init),model=scene.state_dict(),optimizer=optimizer.state_dict(),
                            training_code_identity=frozen_code,data_identity=frozen_data,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),path)
            record['last_checkpoint']=str(path);record['last_checkpoint_step']=step;atomic_json(a.output/'run.json',record)
        def preview(step):
            with torch.no_grad():
                for t in [0,L//2,L-1]:
                    r,_=scene.render(t,K,w2c,H,W)
                    pic=r['rgb'].clamp(0,1).permute(1,2,0).cpu().numpy()
                    cv2.imwrite(str(a.output/f'preview_{step:06d}_{t:03d}.png'),(pic[...,::-1]*255).astype(np.uint8))
        preview(begin)
        stage_last=None;stage_start=time.perf_counter()
        for step in range(begin,a.stop):
            stage='A' if step<1500 else ('B' if step<2500 else 'C');only=step%3 if stage=='A' else None
            if stage!=stage_last:
                if stage_last:record['stages'][stage_last]={'seconds':time.perf_counter()-stage_start,'end_step':step}
                stage_last=stage;stage_start=time.perf_counter();record['current_stage']=stage;atomic_json(a.output/'run.json',record)
            scene.set_stage(stage,only);optimizer.zero_grad(set_to_none=True)
            t=int(frames[step]);u=int(targets[step]);track_active=(step>=1500 and step%4==0 and step<record['schedule']['track_end_exclusive'] and u!=t)
            result,parts=scene.render(t,K,w2c,H,W,only=only)
            pred=result['rgb'];gt=rgb[t]
            if only is None:
                lrgb=.8*abs(pred-gt).mean()+.2*ssim_error(pred,gt).mean()
                # The same jointly composited RGB receives balanced foreground attention.
                lrgb=lrgb+.15*(mean_mask(abs(pred-gt),interior[t,1])+mean_mask(abs(pred-gt),interior[t,2]))
                linstance=sum(mean_mask(abs(result['buf']-masks[t]).mean(0),interior[t,e]) for e in range(3))/3
            else:
                valid=(labels[t]==only) if only==0 else ((labels[t]==0)|(labels[t]==only))
                own=interior[t,only];lrgb=.8*mean_mask(abs(pred-gt),own)+.2*mean_mask(ssim_error(pred,gt),own)
                linstance=mean_mask(abs(result['buf'][only]-masks[t,only]),valid.float())
            pose=scene.human.pose_prior_losses([t]);res=scene.human.residual_regularization([t])
            lpose=.03*(pose['body_pose']+pose['global_orient'])+pose['transl_m2']*.1+pose['betas']*.01
            lpose=lpose+.005*scene.object_rot_delta[t].square().mean()+.05*(scene.object_translation[t]-scene.object_t0[t]).square().mean()
            lres=10*(res['canonical_l2_m2']+res['regional_l2_m2'])+50*res['canonical_spatial_m2']+.05*res['regional_velocity_m2_s2']+.00005*res['regional_acceleration_m2_s4']
            ltemp=temporal_regularization(scene);lbank=scene.bank_regularization()
            ltrack=pred.sum()*0;track_count=0
            if track_active:
                xy=tracks[t];dst=tracks[u];srcpix=xy.long();dstpix=dst.long()
                inside=(srcpix[:,0]>=0)&(srcpix[:,0]<W)&(srcpix[:,1]>=0)&(srcpix[:,1]<H)&(dstpix[:,0]>=0)&(dstpix[:,0]<W)&(dstpix[:,1]>=0)&(dstpix[:,1]<H)
                ids=(inside&vis[t]&vis[u]).nonzero().flatten()
                sx,sy=srcpix[ids].T;dx,dy=dstpix[ids].T;ent=labels[t,sy,sx]
                ok=(ent>0)&(ent==labels[u,dy,dx])&(interior[t,ent,sy,sx]>.5)&(interior[u,ent,dy,dx]>.5)
                ids=ids[ok][:512] # deterministic shared budget, original order
                if len(ids):
                    corr,_=scene.render(t,K,w2c,H,W,target_frame=u)
                    sx,sy=srcpix[ids].T;xyz=corr['buf'][:,sy,sx].T
                    valid=xyz[:,2]>1e-4
                    proj=xyz@K.T;uv=proj[:,:2]/proj[:,2:].clamp_min(1e-5)
                    distance=(uv-dst[ids]).norm(dim=-1)
                    err=F.smooth_l1_loss(distance,torch.zeros_like(distance),beta=5.,reduction='none')
                    if valid.any():ltrack=err[valid].mean();track_count=int(valid.sum())
            loss=lrgb+.2*linstance+.01*ltrack+lpose+lres+ltemp+lbank
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss step{step}')
            loss.backward()
            # Gradient clipping is applied to pose groups only, not screen density statistics.
            pose_params=[p for group in optimizer.param_groups if group['name'].startswith(('human.','object.')) for p in group['params'] if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(pose_params,5.)
            for name,p in scene.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError(f'nonfinite gradient {name} step{step}')
            g=result['viewspace_points'].grad;rad=result['radii'];off=0
            for i,bank in enumerate(scene.banks):
                if only is None or only==i:
                    bank.record(g[off:off+bank.n] if g is not None else None,rad[off:off+bank.n]);off+=bank.n
            optimizer.step()
            if (step+1)%500==0 and 3000<=step+1<=7000:
                for prefix,bank,cap in [('sbank',scene.sbank,100000),('hbank',scene.hbank,25000),('obank',scene.obank,6000)]:
                    event=bank.control(optimizer,prefix,step+1,cap);events.write(json.dumps(event)+'\n')
                events.flush()
            if step%50==0 or step+1==a.stop:
                row=dict(step=step+1,stage=stage,frame=t,total=float(loss.detach()),rgb=float(lrgb.detach()),instance=float(linstance.detach()),track=float(ltrack.detach()),track_active=track_active,track_count=track_count,pose=float(lpose.detach()),residual=float(lres.detach()),temporal=float(ltemp.detach()),bank=float(lbank.detach()),points=[b.n for b in scene.banks],seconds=time.perf_counter()-start_time)
                losses.write(json.dumps(row)+'\n');losses.flush();print(json.dumps(row),flush=True)
                record['parameter_updates']=step+1-begin;record['current_step']=step+1;atomic_json(a.output/'run.json',record)
            if step+1 in [1500,2500,6000,8000] or step+1==a.stop:
                save_checkpoint(step+1);preview(step+1)
        record['stages'][stage_last]={'seconds':time.perf_counter()-stage_start,'end_step':a.stop}
        losses.close();events.close();torch.cuda.synchronize()
        record.update(status='completed',final_step=a.stop,points=[b.n for b in scene.banks],final_checkpoint_sha256=sha(record['last_checkpoint']))
    except BaseException as e:
        record.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        record.update(wall_seconds=time.perf_counter()-start_time,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());atomic_json(a.output/'run.json',record)
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()
