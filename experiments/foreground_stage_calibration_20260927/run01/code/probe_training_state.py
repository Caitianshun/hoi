"""Independent fixed-snapshot gradient/attribute probes, with no state writes."""
from common import *
from render_utils import diagnostic_renderer,load_model,CalibratedCamera
from train_official import rng_capture
import os,time,torch,numpy as np,cv2,pickle
def rng_digest():return hashlib.sha256(pickle.dumps(rng_capture())).hexdigest()
def gradient_stats(a,b,params):
    def calc(gs):
        valid=[g for g in gs if g is not None]
        if not valid:return None,None,0
        ss=sum(float((g.detach().double()**2).sum()) for g in valid);n=sum(g.numel() for g in valid)
        return ss**.5,(ss/n)**.5,n
    na,ra,da=calc(a);nb,rb,db=calc(b)
    dot=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
    cosine=dot/(na*nb) if na and nb else None
    return dict(FG_L2=na,BG_L2=nb,FG_RMS=ra,BG_RMS=rb,FG_half_weight_L2=None if na is None else na*.5,BG_half_weight_L2=None if nb is None else nb*.5,
        cosine=None if cosine is None else float(np.clip(cosine,-1,1)),angle_degrees=None if cosine is None else float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),
        FG_active_dimensions=da,BG_active_dimensions=db,total_group_dimensions=sum(p.numel() for p in params),FG_missing_gradient_tensors=sum(x is None for x in a),BG_missing_gradient_tensors=sum(x is None for x in b),
        NA_reason='Stage/representation does not connect this group to region RGB, or zero norm/empty region' if na is None or nb is None or not na or not nb else None)
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';torch.set_num_threads(4);started=time.monotonic()
    assert read(RUN/'protocol/finals.json')['status']=='both_new_terminal_states_frozen'
    fixed=read(RUN/'protocol/fixed_examples.json');manifest=read(OLD/'inputs/hos_backpack/manifest.json');lookup={x['frame_id']:x for x in manifest['frames']}
    probe_ids=set(fixed['gradient_probe_frame_ids']);preview_ids=fixed['training_frame_ids'];out=RUN/'diagnostics/state_probes';out.mkdir(exist_ok=True)
    target=RUN/'state_probes.jsonl';assert not target.exists()
    snapshots=[]
    for branch,rd in [('H1',OLD/'runs/hos_backpack_formal'),('W_fine',RUN/'runs/W_fine'),('W_all',RUN/'runs/W_all')]:
        for stage,iteration in [('coarse',3000),('fine',1000),('fine',14000)]:
            if branch=='W_fine' and stage=='coarse':continue
            snapshots.append((branch,rd,stage,iteration))
    summary=[]
    with target.open('w',buffering=1) as log:
        for branch,rd,stage,iteration in snapshots:
            checkpoint=rd/f'checkpoint_{stage}_{iteration:06d}.pt';before=identity(checkpoint);config=rd/'effective_config.json'
            model,(dataset,hidden,opt,pipe),state=load_model(checkpoint,config);assert state['stage']==stage and state['iteration']==iteration;del state
            render,_=diagnostic_renderer();bg=torch.tensor([1,1,1] if dataset.white_background else [0,0,0],dtype=torch.float32,device='cuda')
            groups={'xyz':[model._xyz],'scale_rotation':[model._scaling,model._rotation],'opacity':[model._opacity],'SH_DC':[model._features_dc],'SH_high':[model._features_rest],
                'deformation_network':[],'deformation_grid':[]}
            for name,p in model._deformation.named_parameters():
                if p.requires_grad:groups['deformation_grid' if 'grid' in name else 'deformation_network'].append(p)
            params=[p for ps in groups.values() for p in ps];assert len({id(p) for p in params})==len(params)
            initial_rng=rng_digest();attrs=dict(opacity=quantiles(model.get_opacity),scale_max_axis=quantiles(model.get_scaling.max(1).values),active_sh_degree=model.active_sh_degree,points=len(model.get_xyz));displacements=[];previews=[]
            all_ids=sorted(set(preview_ids)|probe_ids,key=int)
            for fid in all_ids:
                frame=lookup[fid];assert sha(frame['image_path'])==frame['image_sha256'] and sha(frame['mask_path'])==frame['mask_sha256']
                camera=CalibratedCamera(frame,int(fid),True);mask=torch.from_numpy(cv2.imread(frame['mask_path'],0)>=128).cuda();gt=camera.original_image.cuda()
                if fid in probe_ids:
                    pkg=render(camera,model,pipe,bg,stage=stage);err=(pkg['render']-gt).abs().mean(0);lf=err[mask].mean() if mask.any() else None;lb=err[~mask].mean() if (~mask).any() else None
                    gf=torch.autograd.grad(lf,params,retain_graph=True,allow_unused=True) if lf is not None else [None]*len(params)
                    gb=torch.autograd.grad(lb,params,allow_unused=True) if lb is not None else [None]*len(params)
                    pos=0;gradrows={}
                    for name,ps in groups.items():
                        row=gradient_stats(gf[pos:pos+len(ps)],gb[pos:pos+len(ps)],ps);frac=float(mask.float().mean())
                        row.update(FG_uniform_weight_L2=None if row['FG_L2'] is None else row['FG_L2']*frac,BG_uniform_weight_L2=None if row['BG_L2'] is None else row['BG_L2']*(1-frac));gradrows[name]=row;pos+=len(ps)
                    disp=quantiles(torch.linalg.vector_norm(pkg['means3D_final'].detach()-model.get_xyz.detach(),dim=1));displacements.append(dict(frame_id=fid,world_displacement=disp))
                    log.write(json.dumps(dict(branch=branch,stage=stage,iteration=iteration,frame_id=fid,nominal_frame_index=int(fid),normalized_time=frame['time'],snapshot=before,L_FG=None if lf is None else float(lf.detach()),L_BG=None if lb is None else float(lb.detach()),FG_fraction=float(mask.float().mean()),groups=gradrows,attributes=attrs,deformation=disp,optimization_steps=0,scope='Four fixed training frames; group gradients use distinct units and cannot be ranked across groups'))+'\n')
                    rgb=pkg['render'].detach().permute(1,2,0).cpu().numpy();del gf,gb,pkg,err,lf,lb
                else:
                    with torch.no_grad():pkg=render(camera,model,pipe,bg,stage=stage);rgb=pkg['render'].permute(1,2,0).cpu().numpy();del pkg
                if fid in preview_ids:
                    dest=out/branch/f'{stage}_{iteration:06d}'/(fid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,rgb=rgb)
                    jpg=dest.with_suffix('.jpg');cv2.imwrite(str(jpg),cv2.cvtColor(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,92]);previews.append(dict(frame_id=fid,raw=identity(dest),image=identity(jpg)))
                assert all(p.grad is None for p in params),'autograd.grad must not accumulate formal optimizer buffers'
            # Capture RNG tensor bytes explicitly: pickle storage IDs are unstable.
            after=identity(checkpoint);assert before==after
            summary.append(dict(branch=branch,stage=stage,iteration=iteration,checkpoint=before,attributes=attrs,deformation=displacements,previews=previews,checkpoint_unchanged=True,optimizer_steps=0))
            print(branch,stage,iteration,'probed',flush=True);del model,params,groups;torch.cuda.empty_cache()
    save_json(out/'manifest.json',dict(status='completed',snapshots=summary,probe_frame_ids=sorted(probe_ids,key=int),training_preview_frame_ids=preview_ids,rows=len(snapshots)*len(probe_ids),W_fine_coarse='Shared H1 coarse snapshot analyzed once',seconds=time.monotonic()-started,optimization_steps=0,formal_sampler_untouched=True))
if __name__=='__main__':run()
