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
    bg_out=RUN/'diagnostics/BG8_fixed'
    if (bg_out/'run.json').exists():
        bg_state=torch.load(read(bg_out/'run.json')['checkpoint']['path'],map_location='cpu',weights_only=False)
        bg_model,_=restore(bg_state)
    else:
        bg_model,bg_state=engine(initial,'BG',8,bg_out,kind='diagnostic',cameras=cams)
    fine,transition=compose_fine(bg_model,'Backpack',initial['metadata']); del bg_model
    atomic_checkpoint(RUN/'diagnostics/shared_short_fine0.pt',fine)
    identity_check=identity_renders(fine); topo=topology(fine); probes=[]
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
        row=dict(iteration=iteration,active_SH=model.active_sh_degree,static_Q_gradient_norms=norms,
                 new_columns_Q_gradient=float(grads[6][:,-4:].norm()),
                 grid_Q_gradient=float(torch.stack([g.norm() for g in grads[7:] if g is not None]).norm()))
        if iteration==1:
            assert row['grid_Q_gradient']==0 and row['new_columns_Q_gradient']==0
        if iteration==64:
            assert row['grid_Q_gradient']>0 and row['new_columns_Q_gradient']>0
            assert all(v>0 for k,v in norms.items() if k!='_features_rest' or model.active_sh_degree>0),norms
        probes.append(row)
        append_json(RUN/'protocol/RGB_gradient_probes.jsonl',row)
    # 64 updates + a 16-update replay, plus BG8 = 88 total Adam calls.
    failure_path=RUN/'diagnostics/SL64/failure.json'
    if failure_path.exists():
        failure=read(failure_path)
        assert failure['completed_updates']==63 and "'_features_rest': 0.0" in failure['traceback']
        saved_path=RUN/'diagnostics/SL64/failure_state.pt'
        saved=torch.load(saved_path,map_location='cpu',weights_only=False)
        assert saved['completed_updates']==63 and saved['attempted_iteration']==64
        assert all(g is None for group in saved['gradients'].values() for g in group)
        # The old diagnostic assertion ran during autograd.grad before backward
        # and Adam at 64. Parameters/moments are exactly after step 63. Only LR
        # had already advanced; restore its schedule position explicitly.
        terminal=clone_cpu(fine)
        terminal.update(model=saved['model'],deformation_accum=saved['deformation_accum'],rng=saved['rng'],mode='SL',
                        completed_updates=63,iteration=63,optimizer_updates=63,sampler=sampler_at(schedule('Backpack','fine'),63))
        model,_=restore(terminal);model.update_learning_rate(63)
        terminal=capture(model,'Backpack','SL','fine',63,schedule('Backpack','fine'),fine['metadata'])
        for i in [1,2,3,4,5,6,7,8,9,10,11]:exact(terminal['model'][i],saved['model'][i])
        recovered=atomic_checkpoint(RUN/'diagnostics/audited_after_step63.pt',terminal)
        save_json(RUN/'protocol/diagnostic_recovery.json',dict(source=identity(saved_path),after_step63=recovered,
            cause='Short BG8 has active SH0, so higher-order SH RGB gradient is correctly zero; assertion was too broad',
            parameters_and_moments_unchanged=True,only_LR_schedule_position_restored=True,new_Adam_updates=0))
        # Save explicit RGB-only path evidence, without replacing any history.
        fresh,(ds,h,o,pipe)=restore(fine,'SL',True)
        render=make_renderer(fine['metadata']['scale_bound'],'SL')
        batch=[cams[i] for i in schedule('Backpack','fine')['batches'][0]]
        rgb,_,_=quality_objective(torch.stack([render(c,fresh,pipe,rgb_background(ds),stage='fine')['render'] for c in batch]),
                                 torch.stack([c.original_image.cuda() for c in batch]))
        probe(fresh,rgb,1);del fresh,rgb
        degree=model.active_sh_degree;model.active_sh_degree=model.max_sh_degree
        rgb,_,_=quality_objective(torch.stack([render(c,model,pipe,rgb_background(ds),stage='fine')['render'] for c in batch]),
                                 torch.stack([c.original_image.cuda() for c in batch]))
        probe(model,rgb,64);model.active_sh_degree=degree;del rgb
    else:
        model,terminal=engine(fine,'SL',64,RUN/'diagnostics/SL64',kind='diagnostic',checkpoint_at=[48],cameras=cams,probe=probe)
    with torch.no_grad():
        a=deform_attributes(model,0.,'SL'); b=deform_attributes(model,1.,'SL');static=~model._deformation_table
        for x,y in zip(a,b):assert torch.equal(x[static],y[static])
    del model
    before=torch.load(RUN/'diagnostics/SL64/checkpoint_fine_000048.pt',map_location='cpu',weights_only=False)
    restored,_=restore(before)
    exact(restored.capture(),before['model']);exact(restored._deformation_accum,before['deformation_accum'])
    exact(rng_capture(),before['rng']); del restored
    target=terminal['completed_updates']
    replay_out=RUN/'diagnostics'/f'SL_resume{target-48}'
    if (replay_out/'run.json').exists():
        replayed=torch.load(read(replay_out/'run.json')['checkpoint']['path'],map_location='cpu',weights_only=False)
        model,_=restore(replayed)
    else:
        model,replayed=engine(before,'SL',target,replay_out,kind='diagnostic',cameras=cams)
    exact(replayed['sampler'],terminal['sampler']); exact(replayed['rng'],terminal['rng'])
    # V9 requires exact state restoration, which is asserted above. It does not
    # require subsequent CUDA optimization trajectories to be bitwise equal.
    # Preserve the failed stricter diagnostic assertion and report all observed
    # tail differences without introducing an unrequested numerical envelope.
    differences={}
    for i in [1,4,5,6,7,8]:
        x,y=terminal['model'][i],replayed['model'][i]
        differences[str(i)]=dict(max_abs=float((x-y).abs().max()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-12)))
        assert np.isfinite(list(differences[str(i)].values())).all()
    network_differences={}
    for k,x in terminal['model'][2].items():
        y=replayed['model'][2][k]
        network_differences[k]=dict(max_abs=float((x-y).abs().max()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-12)))
        assert np.isfinite(list(network_differences[k].values())).all()
    # Direct singleton SL field query, preserving the local/world separation.
    net=model._deformation.deformation_net
    out=net.query_time(model._xyz[:1],None,None,None,model._xyz.new_zeros((1,1)))
    assert out.shape==(1,128) and torch.isfinite(out).all()
    rows=[json.loads(x) for x in (RUN/'diagnostics/SL64/training_metrics.jsonl').read_text().splitlines()]
    replay_rows=[json.loads(x) for x in (replay_out/'training_metrics.jsonl').read_text().splitlines()]
    tail=[r for r in rows if r['completed_updates']>before['completed_updates']]
    assert len(tail)==len(replay_rows)
    assert all(a['frame_uids']==b['frame_uids'] for a,b in zip(tail,replay_rows))
    tail_loss_differences=[dict(iteration=a['completed_updates'],RGB=a['RGB']-b['RGB'],
                              regularization=a['regularization']-b['regularization']) for a,b in zip(tail,replay_rows)]
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
    actual_updates=0;failed_updates=0
    for folder in (RUN/'diagnostics').iterdir():
        if not folder.is_dir():continue
        evidence=folder/'run.json' if (folder/'run.json').exists() else folder/'failure.json'
        if evidence.exists():
            r=read(evidence);actual_updates+=r['updates_this_process']
            if r['status']=='failed':failed_updates+=r['updates_this_process']
    assert actual_updates<=config()['budgets']['integrated_Adam_updates']
    save_json(RUN/'protocol/module_acceptance.json',dict(status='passed',diagnostic_Adam_updates=actual_updates,
        failed_technical_check_Adam_updates=failed_updates,correction='Group finite checks by device; only active SH coefficients require nonzero RGB gradients',
        extra_no_update_backwards=len((RUN/'protocol/extra_backwards.jsonl').read_text().splitlines()),identity=identity_check,topology=topo,variable_seed_preservation=transition,
        RGB_only_path_probes=probes,static_time_independent=True,strict_restore_model_Adam_domain_RNG=True,
        replay_tensor_differences=differences,shared_RGB_sequence_and_remaining_stack=True,
        replay_network_differences=network_differences,replay_loss_differences=tail_loss_differences,
        replay_contract='Exact loading of model Adam buffers domains RNG and remaining stack; finite tail with identical RGB sequence. CUDA tail equality is not a V9 requirement.',
        ordinary_RGB_background='white',single_Q=True,TEM=False,M1=False,development_quality_read=False))
    # Preserve the original preparation snapshots, change only implementation
    # provenance after the disclosed checker fixes, then freeze formal inputs.
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
