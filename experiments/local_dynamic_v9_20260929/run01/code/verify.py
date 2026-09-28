"""One bounded engineering acceptance; no development image quality gate."""
from common import *
import numpy as np
import torch
from state import *
from train_scene import engine
from render_scene import make_renderer,rgb_background
from prepare_initialization import compose_fine,source_code_identity
from hoi_modules.static_dynamic_gaussians import deform_attributes,validate_point_state


def identity_renders(state):
    from adapter_4dgs import CalibratedCamera
    frames=read(ROOT/config()['scenes'][state['scene']]['input_dir']/'manifest.json')['frames']
    cams=[CalibratedCamera(frames[i],i,False) for i in [0,len(frames)//2]]
    rendered={}; parameter_counts={}
    with torch.no_grad():
        for mode in config()['modes']:
            model,(ds,h,o,pipe)=restore(state,mode,True)
            attrs=deform_attributes(model,.27,mode)
            for x,y in zip(attrs,(model._xyz,model._scaling,model._rotation,model._opacity,model.get_features)):
                assert torch.equal(x,y),'Zero residual must preserve world attributes'
            parameter_counts[mode]=sum(p.numel() for p in model._deformation.parameters() if p.requires_grad)
            render=make_renderer(state['metadata']['scale_bound'],mode)
            rendered[mode]=[render(c,model,pipe,rgb_background(ds),stage='fine')['render'].cpu() for c in cams]
            del model
        differences={mode:[float((x-y).abs().max()) for x,y in zip(rendered['Q0'],rendered[mode])] for mode in ['S','SL']}
        assert all(x==0 for v in differences.values() for x in v),differences
        assert parameter_counts['SL']-parameter_counts['S']==512
    return dict(status='passed',max_RGB_absolute_difference=differences,train_frames=[c.image_name for c in cams],
                extra_SL_parameters=parameter_counts['SL']-parameter_counts['S'],deformation_parameters=parameter_counts,
                world_attributes_exact=True,optimization_updates=0,development_quality_read=False)


def topology(state):
    model,_=restore(state,'S',True)
    static=torch.nonzero(~model._deformation_table).flatten()[:2]
    dynamic=torch.nonzero(model._deformation_table).flatten()[:2]
    idx=torch.cat((static,dynamic))
    points={n:getattr(model,n).detach()[idx].cpu() for n in ATTRS}
    role=model._deformation_table[idx].cpu()
    set_points(model,points,role); _,_,opt,_=parameters(state['scene'],'fine'); model.training_setup(opt)
    with torch.no_grad():
        # Two small parents, one of each role, must clone with the same role.
        model._scaling.fill_(np.log(model.percent_dense*model.spatial_lr_scale/10))
        grads=torch.tensor([[1.],[0.],[1.],[0.]],device='cuda')
        model.densify_and_clone(grads,.5,model.spatial_lr_scale)
        exact(model._deformation_table,torch.tensor([0,0,1,1,0,1],dtype=torch.bool))
        # Two large parents split; parent pruning and descendants remain aligned.
        model._scaling[[1,3]]=np.log(model.percent_dense*model.spatial_lr_scale*2)
        grads=torch.tensor([[0.],[1.],[0.],[1.],[0.],[0.]],device='cuda')
        model.densify_and_split(grads,.5,model.spatial_lr_scale)
        exact(model._deformation_table,torch.tensor([0,1,0,1,0,1,0,1],dtype=torch.bool))
        mask=torch.zeros(8,dtype=torch.bool,device='cuda');mask[[1,6]]=True;model.prune_points(mask)
        exact(model._deformation_table,torch.tensor([0,0,1,0,1,1],dtype=torch.bool))
        validate_point_state(model)
        try:model.update_deformation_table(0)
        except RuntimeError:pass
        else:raise AssertionError('Role reclassification was not disabled')
        # Empty / singleton variable subsets exercise the real network interface.
        for k in [0,1]:
            model._deformation_table.zero_(); model._deformation_table[:k]=True
            out=deform_attributes(model,.5,'S');assert all(torch.isfinite(x).all() for x in out)
    return dict(status='passed',clone_split_prune_role_inheritance=True,empty_and_singleton=True,
                automatic_role_reclassification_disabled=True,Adam_updates=0)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    assert not (RUN/'protocol/module_acceptance.json').exists()
    torch.set_num_threads(config()['hardware']['CPU_threads'])
    initial=torch.load(scene_dir('Backpack')/'protocol/coarse_initial.pt',map_location='cpu',weights_only=False)
    cams=load_cameras('Backpack',True)
    bg_model,bg_state=engine(initial,'BG',8,RUN/'diagnostics/BG8_fixed',kind='diagnostic',cameras=cams)
    fine,transition=compose_fine(bg_model,'Backpack',initial['metadata']); del bg_model
    atomic_checkpoint(RUN/'diagnostics/shared_short_fine0.pt',fine)
    identity=identity_renders(fine); topo=topology(fine); probes=[]
    def probe(model,rgb,iteration):
        if iteration not in [1,2,64]:return
        ledger=RUN/'protocol/extra_backwards.jsonl'
        n=len(ledger.read_text().splitlines()) if ledger.exists() else 0
        assert n<config()['budgets']['extra_no_update_backwards']
        append_json(ledger,dict(iteration=iteration,purpose='Q-only gradient path, excludes plane/time regularizers'))
        net=model._deformation.deformation_net
        params=[getattr(model,name) for name in ATTRS]+[net.feature_out[0].weight]+list(net.grid.grids.parameters())
        grads=torch.autograd.grad(rgb,params,retain_graph=True,allow_unused=True)
        static=~model._deformation_table
        norms={name:float(g[static].norm()) if g is not None else 0 for name,g in zip(ATTRS,grads[:6])}
        row=dict(iteration=iteration,static_Q_gradient_norms=norms,
                 new_columns_Q_gradient=float(grads[6][:,-4:].norm()),
                 grid_Q_gradient=float(torch.stack([g.norm() for g in grads[7:] if g is not None]).norm()))
        if iteration==1:
            assert row['grid_Q_gradient']==0 and row['new_columns_Q_gradient']==0
        if iteration==64:
            assert row['grid_Q_gradient']>0 and row['new_columns_Q_gradient']>0
            assert all(v>0 for v in norms.values()),norms
        probes.append(row)
    # 64 updates + a 16-update replay, plus BG8 = 88 total Adam calls.
    model,terminal=engine(fine,'SL',64,RUN/'diagnostics/SL64',kind='diagnostic',checkpoint_at=[48],cameras=cams,probe=probe)
    with torch.no_grad():
        a=deform_attributes(model,0.,'SL'); b=deform_attributes(model,1.,'SL');static=~model._deformation_table
        for x,y in zip(a,b):assert torch.equal(x[static],y[static])
    del model
    before=torch.load(RUN/'diagnostics/SL64/checkpoint_fine_000048.pt',map_location='cpu',weights_only=False)
    restored,_=restore(before)
    exact(restored.capture(),before['model']);exact(restored._deformation_accum,before['deformation_accum'])
    exact(rng_capture(),before['rng']); del restored
    model,replayed=engine(before,'SL',64,RUN/'diagnostics/SL_resume16',kind='diagnostic',cameras=cams)
    exact(replayed['sampler'],terminal['sampler']); exact(replayed['rng'],terminal['rng'])
    # CUDA reduction order can vary. Report tensor differences, no science gate.
    differences={}
    for i in [1,4,5,6,7,8]:
        x,y=terminal['model'][i],replayed['model'][i]
        differences[str(i)]=dict(max_abs=float((x-y).abs().max()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-12)))
        assert differences[str(i)]['relative_l2']<=1e-3
    for k,x in terminal['model'][2].items():
        y=replayed['model'][2][k]
        assert torch.allclose(x,y,atol=1e-6,rtol=1e-3),k
    # Direct singleton SL field query, preserving the local/world separation.
    net=model._deformation.deformation_net
    out=net.query_time(model._xyz[:1],None,None,None,model._xyz.new_zeros((1,1)))
    assert out.shape==(1,128) and torch.isfinite(out).all()
    rows=[json.loads(x) for x in (RUN/'diagnostics/SL64/training_metrics.jsonl').read_text().splitlines()]
    measured=float(np.median([x['iteration_seconds'] for x in rows[-32:]]))
    # Prior actual V8 fine throughput plus a measured early-run cost check.
    # Apply 1.6 to early small-point speed for densification/checkpoint overhead.
    fine_estimate=max(.19,measured*1.6)*180000
    predicted=fine_estimate+6000*.15+1800
    save_json(RUN/'protocol/cost_prediction.json',dict(status='within_budget' if predicted<=config()['budgets']['GPU_seconds'] else 'exceeds_budget',
        measured_early_median_seconds=measured,early_to_late_cost_factor=1.6,reference_fine_seconds=.19,
        six_fine_updates=180000,common_BG_updates=6000,render_init_verify_allowance_seconds=1800,
        predicted_GPU_seconds=predicted,limit_GPU_seconds=config()['budgets']['GPU_seconds'],
        estimated_not_guaranteed=True,all_six_arms_included=True))
    actual_updates=len((RUN/'protocol/diagnostic_attempts.jsonl').read_text().splitlines())
    assert actual_updates==96
    save_json(RUN/'protocol/module_acceptance.json',dict(status='passed',diagnostic_Adam_updates=actual_updates,
        failed_technical_check_Adam_updates=8,correction='finite-state checks group CPU Adam step and GPU tensors by device',
        extra_no_update_backwards=len(probes),identity=identity,topology=topo,variable_seed_preservation=transition,
        RGB_only_path_probes=probes,static_time_independent=True,strict_restore_model_Adam_domain_RNG=True,
        replay_tensor_differences=differences,shared_RGB_sequence_and_remaining_stack=True,
        ordinary_RGB_background='white',single_Q=True,TEM=False,M1=False,development_quality_read=False))
    # Preserve the original preparation snapshots, change only implementation
    # provenance after the disclosed checker fix, then freeze formal inputs.
    sources=source_code_identity()
    for scene in config()['scenes']:
        path=scene_dir(scene)/'protocol/coarse_initial.pt'
        s=torch.load(path,map_location='cpu',weights_only=False)
        before=clone_cpu(s['model'])
        s['metadata']['pre_acceptance_source_code']=s['metadata']['source_code']
        s['metadata']['source_code']=sources
        exact(before,s['model'])
        asset=atomic_checkpoint(scene_dir(scene)/'protocol/coarse_initial_verified.pt',s)
        save_json(scene_dir(scene)/'protocol/formal_source_freeze.json',dict(status='frozen',original=identity(path),
            verified=asset,model_Adam_RNG_sampler_unchanged=True,source_code=sources))


if __name__=='__main__':main()
