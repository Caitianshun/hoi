"""Attribute-scoped RGB gradient routing, independent of datasets and runners.

R must be independent of canonical SH and screen proxies. Routed modes preserve
the legitimate SH viewing-direction derivative to geometry. Density consumes
the returned screen-gradient copies, never retain_grad buffers after two VJPs.
"""
import torch

CANONICAL = {'xyz':'_xyz','f_dc':'_features_dc','f_rest':'_features_rest',
             'scaling':'_scaling','rotation':'_rotation','opacity':'_opacity'}

def partition_current_optimizer_parameters(model):
    hidden=model._deformation.deformation_net.args
    assert hidden.no_do and hidden.no_dshs, 'Dynamic appearance requires a new contract'
    names={id(p):'deformation.'+n for n,p in model._deformation.named_parameters()}
    names.update({id(getattr(model,a)):n for n,a in CANONICAL.items()})
    expected={id(p) for p in model._deformation.parameters()} | {id(getattr(model,a)) for a in CANONICAL.values()}
    G=[];A=[];rows=[];seen=set()
    for group in model.optimizer.param_groups:
        for p in group['params']:
            assert id(p) not in seen and id(p) in names
            seen.add(id(p));name=names[id(p)];scope='A' if name in ('f_dc','f_rest') else 'G'
            inactive=any(s in name for s in ('deformation.timenet.','.opacity_deform.','.shs_deform.'))
            row=dict(name=name,optimizer_group=group['name'],scope=scope,requires_grad=p.requires_grad,
                     expected_None=inactive,shape=list(p.shape),dtype=str(p.dtype),device=str(p.device))
            rows.append(row)
            if p.requires_grad:(A if scope=='A' else G).append((row,p))
    assert seen==expected, 'Optimizer must cover every deformation and canonical parameter, including fixed aabb'
    assert {id(p) for _,p in A}=={id(model._features_dc),id(model._features_rest)}
    assert {id(p) for _,p in G}.isdisjoint({id(p) for _,p in A})
    assert {id(p) for _,p in G+A}=={id(p) for g in model.optimizer.param_groups for p in g['params'] if p.requires_grad}
    return G,A,rows

def validate_gradients(entries,grads):
    assert len(entries)==len(grads), 'Gradient length mismatch'
    result=[]
    for (row,p),g in zip(entries,grads):
        if g is None:
            assert row['expected_None'], 'Unexpected None: '+row['name'];state='expected_None'
        else:
            assert not row['expected_None'], 'Inactive path connected: '+row['name']
            assert g.shape==p.shape and g.dtype==p.dtype and g.device==p.device
            assert torch.isfinite(g).all(), 'Nonfinite gradient: '+row['name']
            state='finite_nonzero' if bool(torch.count_nonzero(g)) else 'finite_zero'
        result.append(dict(name=row['name'],state=state))
    return result

def aggregate_screen_gradients(q_gradients):
    assert q_gradients
    result=torch.zeros_like(q_gradients[0])
    for g in q_gradients:
        assert g.shape==result.shape and g.dtype==result.dtype and g.device==result.device
        result=result+g
    return result

def assign_checked(entries,grads):
    states=validate_gradients(entries,grads)
    for (_,p),g in zip(entries,grads):p.grad=g
    return states

class AttributeGradientRouter:
    MODES=('off','appearance_only','balanced_all','uniform_routed')
    def __init__(self,mode='off'):
        assert mode in self.MODES;self.mode=mode;self.last_sources=None

    def backward(self,model,U,F,R,q_list,capture_sources=False):
        G,A,rows=partition_current_optimizer_parameters(model)
        ids=tuple(id(p) for _,p in G+A)
        assert q_list and len({id(q) for q in q_list})==len(q_list)
        assert all(q.requires_grad for q in q_list)
        assert all(torch.isfinite(x).all() and x.numel()==1 for x in (U,F,R))
        if self.mode in ('off','balanced_all'):
            ((U if self.mode=='off' else F)+R).backward()
            gG=tuple(p.grad for _,p in G);gA=tuple(p.grad for _,p in A);gq=tuple(q.grad for q in q_list)
            states=validate_gradients(G,gG)+validate_gradients(A,gA)
        else:
            gGq=torch.autograd.grad(U+R,[p for _,p in G]+q_list,retain_graph=True,allow_unused=True)
            assert len(gGq)==len(G)+len(q_list)
            gG=gGq[:len(G)];gq=gGq[len(G):]
            gA=torch.autograd.grad(F if self.mode=='appearance_only' else U,[p for _,p in A],allow_unused=True)
            states=assign_checked(G,gG)+assign_checked(A,gA)
        assert len(gq)==len(q_list)
        for g,q in zip(gq,q_list):
            assert g is not None and g.shape==q.shape and g.dtype==q.dtype and g.device==q.device
            assert torch.isfinite(g).all()
        q_copy=[g.detach().clone() for g in gq]
        # Assignment/source equality is exact, independent of CUDA repeat noise.
        for (_,p),g in zip(G+A,tuple(gG)+tuple(gA)):
            assert (p.grad is None and g is None) or p.grad is g
        assert all(torch.equal(c,g) and c.data_ptr()!=g.data_ptr() for c,g in zip(q_copy,gq))
        newG,newA,_=partition_current_optimizer_parameters(model)
        assert ids==tuple(id(p) for _,p in newG+newA)
        self.last_sources=({row['name']:None if g is None else g.detach().clone()
                            for (row,_),g in zip(G+A,tuple(gG)+tuple(gA))} if capture_sources else None)
        return q_copy,rows,states
