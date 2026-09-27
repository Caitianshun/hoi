"""One predeclared collection; strict assignment and shared-gradient consumers."""
from common import *
import torch,cv2,numpy as np,csv,argparse,types,time
from hoi_modules.attribute_gradient_router import *
from render_utils import load_model,CalibratedCamera
from restore_state import restore_model,clone_cpu,exact
from train_official import rng_restore
from diagnostic_render import make_renderer
from loss_policy import regional_rgb
from budget import CallBudget

def compare(g,ref):
    if ref is None:
        return dict(state='expected_None',passed=g is None)
    assert g is not None and g.shape==ref.shape and g.dtype==ref.dtype
    a=g.detach().cpu().double();b=ref.detach().cpu().double();d=a-b
    rn=float(b.norm());rm=float(b.abs().max());da=float(d.abs().max())
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return dict(state='finite_zero' if rn==0 else 'finite_nonzero',reference_norm=rn,
        reference_max=rm,reference_nonzero=int(torch.count_nonzero(b)),actual_nonzero=int(torch.count_nonzero(a)),
        max_abs=da,rel_l2=float(d.norm())/rn if rn else None,norm_max=da/rm if rm else None,
        finite=finite,passed=finite and (da==0 if rn==0 else float(d.norm())/rn<=1e-3 and da/rm<=1e-3))

def cpu_contract():
    m=types.SimpleNamespace();deform=torch.nn.Module();deform.deformation_net=torch.nn.Module()
    deform.deformation_net.args=types.SimpleNamespace(no_do=True,no_dshs=True)
    deform.w=torch.nn.Parameter(torch.tensor([0.3],dtype=torch.float64))
    deform.aabb=torch.nn.Parameter(torch.ones(1,dtype=torch.float64),requires_grad=False);m._deformation=deform
    for i,(n,a) in enumerate(CANONICAL.items()):setattr(m,a,torch.nn.Parameter(torch.tensor([0.2+i/10],dtype=torch.float64)))
    m.optimizer=torch.optim.Adam([{'params':[getattr(m,a)],'name':n} for n,a in CANONICAL.items()]+[{'params':list(deform.parameters()),'name':'deformation'}])
    G,A,_=partition_current_optimizer_parameters(m);q=torch.tensor([.4,.5],dtype=torch.float64,requires_grad=True)
    U=sum(p.sum() for _,p in G+A)+2*q.sum();F=5*sum(p.sum() for _,p in G)+3*sum(p.sum() for _,p in A)+7*q.sum();R=sum(p.square().sum() for _,p in G)
    expected={r['name']:torch.ones_like(p)+2*p.detach() for r,p in G};expected.update({r['name']:3*torch.ones_like(p) for r,p in A})
    out,_,_=AttributeGradientRouter('appearance_only').backward(m,U,F,R,[q],True)
    assert all(torch.equal(p.grad,expected[r['name']]) for r,p in G+A) and torch.equal(out[0],torch.full_like(q,2))
    wrong_G_detected=all(not torch.equal(p.grad,torch.full_like(p,5)+2*p.detach()) for _,p in G)
    wrong_q_detected=not torch.equal(out[0],torch.full_like(q,7));assert wrong_G_detected and wrong_q_detected
    # Deliberately missing a deformation group must fail, without backward.
    old=m.optimizer.param_groups;m.optimizer.param_groups=old[:-1]
    rejected=False
    try:partition_current_optimizer_parameters(m)
    except AssertionError:rejected=True
    finally:m.optimizer.param_groups=old
    assert rejected
    return dict(status='pass',float64=True,wrong_G_source_detected=wrong_G_detected,wrong_q_source_detected=wrong_q_detected,missing_deformation_rejected=rejected,optimizer_steps=0)

def gradients(model,qg):
    G,A,_=partition_current_optimizer_parameters(model)
    result={r['name']:None if p.grad is None else p.grad.detach().cpu().clone() for r,p in G+A}
    result.update({f'q/{i}':x.detach().cpu().clone() for i,x in enumerate(qg)});return result

def charge_Adam(label):
    p=RUN/'protocol/D_steps.jsonl';used=len(p.read_text().splitlines()) if p.exists() else 0
    assert used<read(RUN/'configs/v7.json')['budgets']['D_attempts']
    with p.open('a') as f:f.write(json.dumps(dict(kind='local_Adam_consumer',run=label,iteration=None,time_unix=time.time()))+'\n')
    with (RUN/'protocol/optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(purpose='diagnostic',run=label,iteration=None,event='enter',pid=os.getpid()))+'\n')

def consume(state,config,grads,q,visible,radii,label,wrapped,topology):
    model,(_,h,opt,pipe),_=load_model(V5/'runs/B_U/checkpoint_fine_001000.pt',config)
    restore_model(model,opt,state);rng_restore(state['rng']);G,A,_=partition_current_optimizer_parameters(model)
    gs=[None if grads[r['name']] is None else grads[r['name']].cuda() for r,p in G+A]
    if wrapped:assign_checked(G+A,gs)
    else:
        for (_,p),g in zip(G+A,gs):p.grad=g
    q=[g.cuda() for g in q];vis=visible.cuda();rad=radii.cuda()
    before={r['name']:p.detach().cpu().clone() for r,p in G+A}
    density=None
    if topology:
        aggregate=aggregate_screen_gradients(q) if wrapped else torch.zeros_like(q[0])+q[0]+q[1]
        old_accum=model.xyz_gradient_accum.clone();old_denom=model.denom.clone()
        model.max_radii2D[vis]=torch.maximum(model.max_radii2D[vis],rad[vis])
        model.add_densification_stats(aggregate,vis)
        assert torch.equal(model.xyz_gradient_accum[vis],old_accum[vis]+torch.norm(aggregate[vis,:2],dim=-1,keepdim=True))
        assert torch.equal(model.denom[vis],old_denom[vis]+1)
        density=dict(signed_aggregate=aggregate.cpu(),accum=model.xyz_gradient_accum.cpu(),denom=model.denom.cpu(),radii=model.max_radii2D.cpu())
        i=1100;opacity=opt.opacity_threshold_fine_init-i*(opt.opacity_threshold_fine_init-opt.opacity_threshold_fine_after)/opt.densify_until_iter
        threshold=opt.densify_grad_threshold_fine_init-i*(opt.densify_grad_threshold_fine_init-opt.densify_grad_threshold_after)/opt.densify_until_iter
        if i>opt.densify_from_iter and i%opt.densification_interval==0 and len(model.get_xyz)<360000:
            model.densify(threshold,opacity,read(config)['scale_bound'],None,5,5,str(RUN/'diagnostics'),i,'fine')
        if i>opt.pruning_from_iter and i%opt.pruning_interval==0 and len(model.get_xyz)>200000:
            model.prune(threshold,opacity,read(config)['scale_bound'],None)
    G,A,rows=partition_current_optimizer_parameters(model)
    n=len(model.get_xyz)
    for row,p in G+A:
        if row['name'] in CANONICAL:assert len(p)==n
        st=model.optimizer.state.get(p,{})
        for k in ('exp_avg','exp_avg_sq'):
            if k in st:assert st[k].shape==p.shape
    skipped=[r['name'] for r,p in G+A if p.grad is None]
    charge_Adam(label);model.optimizer.step();torch.cuda.synchronize()
    with (RUN/'protocol/optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(purpose='diagnostic',run=label,iteration=None,event='completed',pid=os.getpid()))+'\n')
    final=clone_cpu(dict(model=model.capture(),deformation_accum=model._deformation_accum))
    deltas={r['name']:p.detach().cpu()-before[r['name']] for r,p in G+A if p.shape==before[r['name']].shape}
    return final,density,deltas,dict(points=n,post_topology_parameter_optimizer_references='pass',None_before_Adam=skipped)

def run_probe():
    assert not (RUN/'diagnostics/acceptance_probe.json').exists()
    torch.set_num_threads(4);budget=CallBudget('module_acceptance');cpu=cpu_contract()
    parent=V5/'runs/B_U/checkpoint_fine_001000.pt';config=V5/'runs/B_U/effective_config.json'
    model,(_,h,opt,pipe),state=load_model(parent,config);state=clone_cpu(state);restore_model(model,opt,state)
    G,A,partition=partition_current_optimizer_parameters(model);before={r['name']:tensor_hash(p) for r,p in G+A}
    render=make_renderer(read(config)['scale_bound']);by={x['frame_id']:x for x in read(OLD/'inputs/hos_backpack/manifest.json')['frames']}
    frames=[by[n] for n in ['00001','00041']];cams=[CalibratedCamera(f,i) for i,f in enumerate(frames)]
    gt=torch.stack([c.original_image for c in cams]).cuda();masks=torch.stack([torch.tensor(cv2.imread(f['mask_path'],0)>=128) for f in frames]).cuda()
    outputs=[];assets=[];losses=[];semantics=[];out=RUN/'diagnostics/gradients';out.mkdir(exist_ok=True)
    modes=read(RUN/'configs/v7.json')['numerical']['collection']+['appearance_only']
    for k,mode in enumerate(modes):
        model.optimizer.zero_grad(set_to_none=True)
        pkgs=[render(c,model,pipe,torch.zeros(3,device='cuda')) for c in cams]
        pred=torch.stack([p['render'] for p in pkgs]);q=[p['viewspace_points'] for p in pkgs]
        U,_=regional_rgb(pred,gt,masks,'uniform','fine');F,_=regional_rgb(pred,gt,masks,'balanced_fine','fine');R=model.compute_regulation(h.time_smoothness_weight,h.l1_time_planes,h.plane_tv_weight)
        if k==6:
            rg=torch.autograd.grad(R,[p for _,p in A]+q,retain_graph=True,allow_unused=True);assert all(g is None for g in rg)
        router=AttributeGradientRouter(mode);qg,_,states=router.backward(model,U,F,R,q,capture_sources=k==6)
        if k==6:
            assert all((p.grad is None and router.last_sources[r['name']] is None) or torch.equal(p.grad,router.last_sources[r['name']]) for r,p in G+A)
        data=gradients(model,qg);outputs.append(data);assets.append(atomic_checkpoint(out/f'{k}_{mode}.pt',data))
        losses.append(dict(index=k,mode=mode,U=float(U.detach()),F=float(F.detach()),R=float(R.detach())))
        semantics.append(dict(index=k,mode=mode,states=states,source_assignment_exact=True,q_independent_copy=True,R_independent_of_A_q=True if k==6 else None))
        assert before=={r['name']:tensor_hash(p) for r,p in G+A}
        if k==0:
            visible=torch.stack([p['visibility_filter'] for p in pkgs]).any(0).cpu();radii=torch.stack([p['radii'] for p in pkgs]).max(0).values.cpu()
        del pkgs,pred,q,U,F,R,router;torch.cuda.empty_cache()
    rows=[]
    for k in range(1,6):
        for name,g in outputs[k].items():rows.append(dict(index=k,mode=modes[k],name=name,**compare(g,outputs[0][name])))
    numeric=all(x['passed'] for x in rows)
    summary=dict(status='accepted_numerical_variation' if numeric else 'targeted_check_required',cpu_contract=cpu,partition=partition,rows=rows,losses=losses,semantics=semantics,gradient_assets=assets,parameter_hashes_unchanged=before,historical_failed_tolerance=identity(V6/'routing_equivalence.json'))
    save_json(RUN/'diagnostics/acceptance_probe.json',summary)
    assert numeric,'V7 fixed numerical collection needs a targeted implementation review'
    q=[outputs[0][f'q/{i}'] for i in range(2)]
    del model;torch.cuda.empty_cache()
    a=consume(state,config,outputs[0],q,visible,radii,'same_gradient_wrapped',True,True)
    b=consume(state,config,outputs[0],q,visible,radii,'same_gradient_direct',False,True)
    exact(a[0],b[0]);exact(a[1],b[1]);exact(a[2],b[2])
    assets.append(atomic_checkpoint(RUN/'diagnostics/Adam_same_gradient_reference.pt',a[0]))
    shared=dict(status='pass',model_Adam_step_moments_parameters_buffers_exact=True,signed_q_accum_denom_exact=True,wrapped=a[3],direct=b[3],density_before_Adam_preserved=True)
    del a,b;torch.cuda.empty_cache()
    c=consume(state,config,outputs[0],q,visible,radii,'independent_uniform_update',False,False)
    d=consume(state,config,outputs[1],q,visible,radii,'independent_routed_update',True,False)
    update=[]
    for name,v in d[2].items():update.append(dict(name=name,**compare(v,c[2][name])))
    assets.append(atomic_checkpoint(RUN/'diagnostics/Adam_increment_pair.pt',dict(reference=c[2],routed=d[2])))
    save_json(RUN/'diagnostics/consumer_acceptance.json',dict(status='pass',same_saved_gradient=shared,independent_backward_update_differences=update,update_comparison_role='diagnostic, not a quality gate; denominator is actual increment',optimizer_step_calls=4,assets=assets[-2:]))
    fields=['index','mode','name','state','reference_norm','reference_max','reference_nonzero','actual_nonzero','max_abs','rel_l2','norm_max','finite','passed']
    with (RUN/'gradient_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print('Fixed gradient collection and four counted Adam consumers passed')

if __name__=='__main__':run_probe()
