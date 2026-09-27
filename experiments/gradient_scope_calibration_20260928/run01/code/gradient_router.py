"""Explicit parameter-group update rule; density reads returned uniform q only."""
import torch
def partition_current_optimizer_parameters(model):
    hidden=model._deformation.deformation_net.args
    assert hidden.no_do and hidden.no_dshs
    names={id(p):'deformation.'+n for n,p in model._deformation.named_parameters()}
    for name,attr in [('xyz','_xyz'),('f_dc','_features_dc'),('f_rest','_features_rest'),('scaling','_scaling'),('rotation','_rotation'),('opacity','_opacity')]:names[id(getattr(model,attr))]=name
    rows=[];G=[];A=[];seen=set()
    for group in model.optimizer.param_groups:
        for p in group['params']:
            assert id(p) not in seen and id(p) in names;seen.add(id(p));name=names[id(p)]
            scope='A' if name in ['f_dc','f_rest'] else 'G'
            expected_none=any(s in name for s in ['deformation.timenet.','.opacity_deform.','.shs_deform.'])
            row=dict(name=name,optimizer_group=group['name'],scope=scope,requires_grad=p.requires_grad,expected_None=expected_none,shape=list(p.shape),device=str(p.device))
            rows.append(row)
            if p.requires_grad:(A if scope=='A' else G).append((row,p))
    assert {id(p) for p in [model._xyz,model._scaling,model._rotation,model._opacity,model._features_dc,model._features_rest]}<=seen
    assert len(A)==2 and all(r['scope']=='G' for r,p in G)
    return G,A,rows
def validate_gradients(entries,grads):
    result=[]
    for (r,p),g in zip(entries,grads):
        if g is None:
            assert r['expected_None'],f"unexpected_None: {r['name']}";status='expected_None'
        else:
            assert not r['expected_None'],f"inactive parameter became connected: {r['name']}"
            assert g.shape==p.shape and g.device==p.device
            assert torch.isfinite(g).all(),f"nonfinite: {r['name']}"
            status='finite_zero' if not bool(torch.count_nonzero(g)) else 'finite_nonzero'
        result.append(dict(name=r['name'],state=status))
    return result
def compute_routed_gradients(model,U,F,R,q_list,policy='C_route'):
    assert policy in ['C_route','uniform_all']
    G,A,rows=partition_current_optimizer_parameters(model)
    assert all(q.requires_grad for q in q_list)
    for value in [U,F,R]:assert torch.isfinite(value).all()
    gGq=torch.autograd.grad(U+R,[p for _,p in G]+q_list,retain_graph=True,allow_unused=True)
    gA=torch.autograd.grad(F if policy=='C_route' else U,[p for _,p in A],allow_unused=True)
    gG=gGq[:len(G)];gq=gGq[len(G):]
    states=validate_gradients(G,gG)+validate_gradients(A,gA)
    assert all(g is not None and g.shape==q.shape and torch.isfinite(g).all() for g,q in zip(gq,q_list))
    for entries,grads in [(G,gG),(A,gA)]:
        for (_,p),g in zip(entries,grads):p.grad=g
    # Clone to avoid retain_grad hooks or a later VJP mutating density's source.
    q_uniform=[g.detach().clone() for g in gq]
    return q_uniform,rows,states
