"""Read-only CPU contract tests for root's actual scene classes; no raster/GPU/reference."""
from pathlib import Path
import ast, hashlib, json, sys, time
import numpy as np
import torch
from torch import nn
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix
E=Path('/home/cai_tianshun/Project/HOI/experiments/structured_hoi_20260923')
sys.path.insert(0,str(E/'code'))
from human_lbs import HumanLBS
start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(439)
source=E/'code/gaussian_scene.py'
# Import the actual class AST verbatim, avoiding the renderer CUDA extension.
a=ast.parse(source.read_text());classes=ast.Module(body=[n for n in a.body if isinstance(n,ast.ClassDef)],type_ignores=[])
ns=dict(torch=torch,np=np,nn=nn,HumanLBS=HumanLBS,axis_angle_to_matrix=axis_angle_to_matrix,quaternion_to_matrix=quaternion_to_matrix)
exec(compile(classes,str(source),'exec'),ns)
Bank,Scene=ns['GaussianBank'],ns['StructuredScene']
report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cpu_only':True,'evaluation_reference_read':False}
# Actual topology routine, initialized Adam state and distinct per-leaf moments.
b=Bank(np.arange(40),np.full((40,3),.4),np.full(40,.01),1,.005)
opt=torch.optim.Adam(b.optimizer_groups('hbank'))
loss=sum((p*torch.arange(1,p.numel()+1,dtype=p.dtype).reshape(p.shape)*.0001).sum() for p in b.parameters())
loss.backward();opt.step();opt.zero_grad(set_to_none=True)
old_params={k:getattr(b,k) for k in b.fields};old_mom={k:opt.state[p]['exp_avg'].clone() for k,p in old_params.items()}
with torch.no_grad():
    b.seen_count[:]=5;b.grad_sum[:]=0;b.grad_sum[[1,7]]=1;b.opacity_logit[3]=-8
old_anchor=b.anchor_id.clone();event=b.control(opt,'hbank',3000,50)
keep=[i for i in range(40) if i not in [1,7,3]]
assert b.n==41 and b.entity==1
assert event['pruned']==[3] and event['split_parents']==[1,7]
assert torch.equal(b.anchor_id,old_anchor[keep+[1,7,1,7]])
assert torch.equal(b.parent_id[-4:],torch.tensor([1,7,1,7]))
assert len(torch.unique(b.stable_id))==b.n
for k in b.fields:
    p=getattr(b,k);assert old_params[k] not in opt.state
    assert next(g for g in opt.param_groups if g['name']=='hbank.'+k)['params'][0] is p
    assert torch.equal(opt.state[p]['exp_avg'][:len(keep)],old_mom[k][keep])
    assert not opt.state[p]['exp_avg'][len(keep):].count_nonzero()
assert b.local_offset().norm(dim=-1).max()<.005
sum(p.square().sum() for p in b.parameters()).backward();opt.step()
# Round-trip points and Adam state through changed tensor lengths.
b2=Bank(np.arange(40),np.full((40,3),.4),np.full(40,.01),1,.005)
b2.resize_for_load(b.state_dict(),'');b2.load_state_dict(b.state_dict())
o2=torch.optim.Adam(b2.optimizer_groups('hbank'));o2.load_state_dict(opt.state_dict())
for k in b.fields:
    assert torch.equal(getattr(b,k),getattr(b2,k))
    assert torch.equal(opt.state[getattr(b,k)]['exp_avg'],o2.state[getattr(b2,k)]['exp_avg'])
report['density_identity_optimizer_checkpoint']={'passed':True,'event':event,'max_attachment_offset_m':float(b.local_offset().norm(dim=-1).max())}
# Actual scene: legal initialization only; no independent evaluation asset access.
p=torch.load(E/'dev1_initialization/initialization.pt',map_location='cpu',weights_only=False)
s=Scene(p);raw=np.load(p['human_geometry']);h=s.components(0)[1][0]
err=(h.detach()-torch.from_numpy(raw['vertices_world_m'][0])).norm(dim=-1).max().item()
assert err<2e-6
stage_report={}
for stage,only in [('A',1),('B',None),('C',None)]:
    s.set_stage(stage,only)
    active=[name for name,v in s.named_parameters() if v.requires_grad]
    if stage=='A':assert all(name.startswith(('human.','hbank.')) for name in active)
    if stage=='B':
        assert set(active)=={'human.body_pose','human.global_orient','human.transl','human.regional_residual_raw','object_rot_delta','object_translation'}
    if stage=='C':assert len(active)==23
    stage_report[stage]=active
s.set_stage('C');s.zero_grad(set_to_none=True)
parts=s.components(1)
# A random linear centre and covariance objective probes both actual transform paths.
loss=0
for xyz,fr,scale,opacity,_ in parts:
    cov=fr@torch.diag_embed(scale.square())@fr.transpose(-1,-2)
    loss=loss+(xyz*torch.randn_like(xyz)).mean()+(cov*torch.randn_like(cov)).sum()
loss.backward()
grad={name:float(v.grad.norm()) for name,v in s.named_parameters() if name in ['human.body_pose','human.global_orient','human.transl','human.betas','human.canonical_offset_raw','human.regional_residual_raw','object_rot_delta','object_translation']}
assert len(grad)==8 and all(v>0 and np.isfinite(v) for v in grad.values())
report['actual_scene']={'passed':True,'world_center_max_error_m':err,'active_parameters_by_stage':stage_report,'nonzero_geometry_gradient_norms':grad}
# Counterexample independent of any data: old track saturation versus linear tail.
r=torch.tensor([10.,40.],requires_grad=True);r.clamp(max=30).mean().backward();clipped=r.grad.tolist()
r=torch.tensor([10.,40.],requires_grad=True);torch.nn.functional.huber_loss(r,torch.zeros_like(r),delta=10).backward();huber=r.grad.tolist()
report['old_track_hard_clip_counterexample']={'residual_px':[10,40],'clamp_mean_gradient':clipped,'huber_delta10_mean_gradient':huber}
report['wall_seconds']=time.perf_counter()-start
(E/'human_model/trainer_review_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
