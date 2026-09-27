"""Classified finite guards; never repair or replace scientific tensors."""
from common import *
import torch
class FirstNonfinite(FloatingPointError):pass
def walk(v,prefix=''):
    if torch.is_tensor(v) or v is None:yield prefix,v
    elif isinstance(v,dict):
        for k,x in v.items():yield from walk(x,prefix+'/'+str(k))
    elif isinstance(v,(list,tuple)):
        for i,x in enumerate(v):yield from walk(x,prefix+'/'+str(i))
def stats(t):
    if t is None:return dict(state='not_connected')
    x=t.detach();finite=torch.isfinite(x);bad=~finite
    out=dict(shape=list(x.shape),dtype=str(x.dtype),elements=x.numel(),finite=int(finite.sum()),nan=int(torch.isnan(x).sum()),posinf=int(torch.isposinf(x).sum()),neginf=int(torch.isneginf(x).sum()))
    out['state']='finite' if not bool(bad.any()) else 'nonfinite'
    if out['state']=='nonfinite':
        out['indices']=bad.nonzero()[:128].cpu().tolist()
        out['finite_quantiles']=quantiles(x[finite])
    return out
def named_model(model,gradients=False):
    result={}
    for group in model.optimizer.param_groups:
        for i,p in enumerate(group['params']):result[f"{group['name']}/{i}"]=p.grad if gradients else p
    if not gradients:
        result['Adam']=model.optimizer.state_dict()['state']
        result['buffers']={n:getattr(model,n) for n in ['xyz_gradient_accum','denom','max_radii2D','_deformation_accum','_deformation_table']}
    return result
class Guard:
    def __init__(self,runtime):self.r=runtime
    def check(self,phase,values,detailed=False):
        r=self.r;r.phase=phase;bad=[];rows={};missing=[]
        for name,t in walk(values):
            if t is None:missing.append(name);continue
            if not (t.is_floating_point() or t.is_complex()):
                if detailed:rows[name]=dict(state='finite_integer',shape=list(t.shape),dtype=str(t.dtype),elements=t.numel())
                continue
            finite=bool(torch.isfinite(t).all())
            if detailed or not finite:rows[name]=stats(t)
            if not finite:bad.append(name)
        r.event(phase,finite=not bad,bad=bad,not_connected=missing,tensors=rows)
        if bad:
            evidence=dict(attempt=r.output.name,iteration=r.iteration,phase=phase,batch=getattr(r,'batch',[]),tensors=rows,nonfinite_names=bad,pre_state=str(r.pre_path),gaussian_index_basis='Parent row indices valid until next topology event')
            if phase=='forward_internal_geometry':
                evidence['bad_global_gaussian_rows']={n:sorted(set(r.visible_gaussian_ids[i[0]] for i in rows[n]['indices'])) for n in bad}
            save_json(r.output/'first_bad_tensor.json',evidence)
            from restore_state import clone_cpu
            torch.save(clone_cpu(values),r.output/'first_bad_values.pt')
            raise FirstNonfinite(phase+': '+', '.join(bad[:4]))
