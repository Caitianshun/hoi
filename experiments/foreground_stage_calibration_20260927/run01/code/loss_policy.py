"""Per-image, area-normalized RGB losses; never pool masks across a batch."""
import torch
POLICIES=('uniform','balanced_fine','balanced_all')
def regional_rgb(pred,gt,masks,policy,stage):
    assert policy in POLICIES
    assert pred.ndim==4 and pred.shape==gt.shape
    assert masks.shape==(pred.shape[0],pred.shape[2],pred.shape[3]) and masks.dtype==torch.bool
    error=(pred-gt).abs().mean(dim=1)
    balanced=policy=='balanced_all' or (policy=='balanced_fine' and stage=='fine')
    values=[];rows=[]
    for err,fg in zip(error,masks):
        bg=~fg;nf=int(fg.sum());nb=int(bg.sum())
        lf=err[fg].mean() if nf else None;lb=err[bg].mean() if nb else None
        if balanced:
            rgb=(lf+lb)*.5 if nf and nb else lf if nf else lb
        else:rgb=err.mean()
        values.append(rgb)
        rows.append(dict(fg_pixels=nf,bg_pixels=nb,fg_fraction=nf/(nf+nb),
            L_fg=None if lf is None else float(lf.detach()),L_bg=None if lb is None else float(lb.detach()),
            L_uniform=float(err.mean().detach()),L_rgb=float(rgb.detach()),
            empty_region='foreground' if not nf else 'background' if not nb else None))
    return torch.stack(values).mean(),rows
def sanity_check():
    # Unequal region area, unequal masks, empty regions, and raw values > 1.
    p=torch.tensor([[[[2.,2.,0.,0.]]],[[[1.,0.,0.,0.]]]],requires_grad=True).repeat(1,3,1,1)
    g=torch.zeros_like(p);m=torch.tensor([[[True,True,False,False]],[[True,False,False,False]]])
    l,rows=regional_rgb(p,g,m,'balanced_all','coarse');assert abs(float(l)-.75)<1e-7
    grad=torch.autograd.grad(l,p)[0];assert torch.isfinite(grad).all()
    u,_=regional_rgb(p,g,m,'balanced_fine','coarse');assert abs(float(u)-.625)<1e-7
    for m in [torch.zeros_like(m),torch.ones_like(m)]:
        q,_=regional_rgb(p,g,m,'balanced_all','fine');assert abs(float(q)-.625)<1e-7
    return dict(status='passed',scope='per-image region normalization, batch weighting, empty regions, raw RGB, finite derivative',optimization_steps=0)
