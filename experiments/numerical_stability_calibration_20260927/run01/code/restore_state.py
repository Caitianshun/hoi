"""Explicit stage metadata and complete state restoration, without filename dispatch."""
from common import *
import torch,numpy as np
from train_official import rng_capture,rng_restore
def clone_cpu(v):
    if isinstance(v,torch.nn.Parameter):return torch.nn.Parameter(v.detach().cpu().clone(),requires_grad=v.requires_grad)
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:clone_cpu(x) for k,x in v.items()}
    if isinstance(v,tuple):return tuple(clone_cpu(x) for x in v)
    if isinstance(v,list):return [clone_cpu(x) for x in v]
    if isinstance(v,np.ndarray):return v.copy()
    return v
def to_device(v,device):
    if isinstance(v,torch.nn.Parameter):return torch.nn.Parameter(v.detach().to(device).clone(),requires_grad=v.requires_grad)
    if torch.is_tensor(v):return v.to(device).clone()
    if isinstance(v,dict):return {k:(x.cpu().clone() if k=='step' and torch.is_tensor(x) else to_device(x,device)) for k,x in v.items()}
    if isinstance(v,tuple):return tuple(to_device(x,device) for x in v)
    if isinstance(v,list):return [to_device(x,device) for x in v]
    return v
def exact(a,b,path='root'):
    if torch.is_tensor(a):assert torch.equal(a.detach().cpu(),b.detach().cpu()),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:exact(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
    else:assert a==b,path
def load_completed(path):
    s=torch.load(path,map_location='cpu',weights_only=False)
    assert s['stage']=='fine' and s.get('phase','completed_step')=='completed_step'
    assert all(k in s for k in ['model','rng','viewpoint_stack','temp_list','deformation_accum'])
    return s
def restore_model(model,opt,s):
    model.restore(to_device(s['model'],'cuda'),opt)
    model._deformation_accum=to_device(s['deformation_accum'],'cuda')
    exact(model.capture(),s['model']);exact(model._deformation_accum,s['deformation_accum'])
def restore_sampler(s,cameras):
    lookup={c.uid:c for c in cameras}
    stack=[lookup[i] for i in s['viewpoint_stack']];temp=[lookup[i] for i in s['temp_list']]
    rng_restore(s['rng']);exact(rng_capture(),s['rng'])
    return stack,temp
def capture(model,stage,iteration,stack,temp,**meta):
    return clone_cpu(dict(model=model.capture(),stage=stage,iteration=iteration,
        rng=rng_capture(),viewpoint_stack=[c.uid for c in stack],temp_list=[c.uid for c in temp],
        deformation_accum=model._deformation_accum,**meta))
