"""Explicit V8/V9 weight warm start and complete V10 after-Adam recovery."""
from common import *
import random, types
import numpy as np
import torch
from torch import nn
SCHEMA='v10_after_adam_v1'
ATTRS=('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity')
PARENT_PATHS={
 'Backpack': ROOT/'experiments/local_dynamic_v9_20260929/run01/scenes/Backpack/runs/Q0',
 'Tennis':ROOT/'experiments/temporal_evidence_v8_20260928/run01/tennis/runs/B_Q'}
PARENT_NAMES={'Backpack':'checkpoint_fine_030000.pt','Tennis':'checkpoint_fine_014000.pt'}
INPUT_PATHS={
 'Backpack':ROOT/'experiments/baseline_protocol_calibration_20260927/run01/inputs/hos_backpack',
 'Tennis':ROOT/'experiments/temporal_evidence_v8_20260928/run01/tennis/inputs/hos_tennis'}
def rng_capture():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def rng_restore(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def parameters(scene):
    cfg=read(PARENT_PATHS[scene]/'effective_config.json');_,objs=official_config(2)
    for obj,key in zip(objs,('model','hidden','optimization','pipeline')):
        for k,v in cfg[key].items():setattr(obj,k,v)
    assert objs[0].white_background and not objs[3].convert_SHs_python and not objs[3].compute_cov3D_python
    return objs,cfg
def import_parent(scene):
    from scene.gaussian_model import GaussianModel
    objs,cfg=parameters(scene);ds,h,opt,pipe=objs
    path=PARENT_PATHS[scene]/PARENT_NAMES[scene];s=torch.load(path,map_location='cpu',weights_only=False)
    assert len(s['model'])==14
    if scene=='Backpack':assert s['schema']=='v9_after_adam_v1' and s['completed_updates']==30000
    else:assert s['resumable'] is False and s['optimizer_updates']==13999
    a=s['model'];model=GaussianModel(ds.sh_degree,h);model._deformation.cuda()
    for key,index in zip(ATTRS,(1,4,5,6,7,8)):setattr(model,key,nn.Parameter(a[index].detach().cuda().clone()))
    model._deformation.load_state_dict(a[2],strict=True);model.active_sh_degree=a[0];model.spatial_lr_scale=a[13]
    model._deformation_table=a[3].cuda().clone();model.max_radii2D=a[9].cuda().clone()
    model.xyz_gradient_accum=a[10].cuda().clone();model.denom=a[11].cuda().clone()
    model._deformation_accum=s['deformation_accum'].cuda().clone();model.percent_dense=opt.percent_dense
    manifest=read(INPUT_PATHS[scene]/'manifest.json')
    extent=manifest['scene_extent']
    meta=dict(parent=identity(path),parent_effective_config=identity(PARENT_PATHS[scene]/'effective_config.json'),
        parent_optimizer_updates=s.get('optimizer_updates'),parent_schema=s.get('schema','v8_completed_step'),
        parent_resume_used=False,warm_start='render_weights_only_fresh_Adam',scale_bound=cfg['scale_bound'],
        scene_extent=extent,input_manifest=identity(INPUT_PATHS[scene]/'manifest.json'),
        training_scene_selection='previously_developed_scene_level_parent_choice')
    return model,objs,meta
def new_optimizer(base,support=None):
    cfg=config()['learning_rates'];lr=cfg['base']
    groups=[]
    mapping={'xyz':'_xyz','f_dc':'_features_dc','f_rest':'_features_rest','opacity':'_opacity','scaling':'_scaling','rotation':'_rotation'}
    for name,attr in mapping.items():groups.append(dict(params=[getattr(base,attr)],lr=lr[name]*(base.spatial_lr_scale if name=='xyz' else 1),name='base.'+name))
    groups.extend([dict(params=list(base._deformation.get_mlp_parameters()),lr=lr['deformation'],name='base.deformation'),
        dict(params=list(base._deformation.get_grid_parameters()),lr=lr['grid'],name='base.grid')])
    if support is not None:groups.extend(support.optimizer_groups())
    optimizer=torch.optim.Adam(groups,lr=0.,eps=1e-15);base.optimizer=optimizer
    assert not optimizer.state
    return optimizer
def set_learning_rates(optimizer,base,completed_next):
    cfg=config()['learning_rates'];position=(completed_next-1)/19999
    assert 0<=position<=1
    for g in optimizer.param_groups:
        bank,name=g['name'].split('.',1);value=cfg[bank][name]
        if isinstance(value,list):value=float(np.exp((1-position)*np.log(value[0])+position*np.log(value[1])))
        if bank=='base' and name=='xyz':value*=base.spatial_lr_scale
        g['lr']=float(value)
    return {g['name']:g['lr'] for g in optimizer.param_groups}
def schedule(scene):return read(scene_dir(scene)/'protocol/RGB_schedule.json')
def sampler_at(sched,completed):
    used=completed*2;count=sched.get('camera_count',sched.get('training_frames'))
    if count is None:count=len(read(INPUT_PATHS[sched['scene']]/'manifest.json')['frames'])
    flat=[x for batch in sched['batches'] for x in batch];offset=used%count
    epoch_start=used-offset
    epoch_order=flat[epoch_start:epoch_start+count]
    return dict(completed_updates=completed,schedule_sha256=sched.get('schedule_sha256'),
        remaining_stack=epoch_order[offset:] if completed<len(sched['batches']) else [],next_frame_uids=sched['batches'][completed] if completed<len(sched['batches']) else [])
def capture(base,support,optimizer,scene,arm,completed,meta):
    sched=schedule(scene)
    return clone_cpu(dict(schema=SCHEMA,phase='after_Adam_topology_and_zero_grad',resumable=True,
        scene=scene,arm=arm,completed_updates=completed,optimizer_updates=completed,
        base={k:getattr(base,k) for k in ATTRS},deformation=base._deformation.state_dict(),
        active_sh_degree=base.active_sh_degree,spatial_lr_scale=base.spatial_lr_scale,
        deformation_table=base._deformation_table,deformation_accum=base._deformation_accum,
        max_radii2D=base.max_radii2D,xyz_gradient_accum=base.xyz_gradient_accum,denom=base.denom,
        support=None if support is None else support.checkpoint_payload(),optimizer=optimizer.state_dict(),
        rng=rng_capture(),sampler=sampler_at(sched,completed),metadata=meta,
        actual_learning_rates={g['name']:g['lr'] for g in optimizer.param_groups}))
def restore(s):
    from hoi_modules.pose_support_gaussians import PoseSupportGaussians
    assert s['schema']==SCHEMA and s['phase']=='after_Adam_topology_and_zero_grad' and s['resumable']
    base,objs,meta=import_parent(s['scene'])
    for k in ATTRS:setattr(base,k,nn.Parameter(s['base'][k].cuda().clone()))
    base._deformation.load_state_dict(s['deformation'],strict=True);base.active_sh_degree=s['active_sh_degree'];base.spatial_lr_scale=s['spatial_lr_scale']
    for key in ('deformation_table','deformation_accum','max_radii2D','xyz_gradient_accum','denom'):
        setattr(base,'_'+key if key.startswith('deformation') else key,s[key].cuda().clone())
    support=None if s['support'] is None else PoseSupportGaussians.from_checkpoint(s['support'],device='cuda')
    optimizer=new_optimizer(base,support);optimizer.load_state_dict(s['optimizer']);rng_restore(s['rng'])
    assert s['sampler']==sampler_at(schedule(s['scene']),s['completed_updates'])
    return base,support,optimizer,objs,s['metadata']
def load_scene_cameras(scene,split='train',load_rgb=True):
    path=INPUT_PATHS[scene]/('manifest.json' if split=='train' else 'evaluation_manifest.json')
    frames=read(path)['frames'];return [CalibratedCamera(f,i,load_rgb) for i,f in enumerate(frames)],frames
def finite(tensors,label):
    values=[t.detach() for t in tensors if torch.is_tensor(t) and t.is_floating_point() and t.numel()]
    for d in {t.device for t in values}:
        if not bool(torch.stack([torch.isfinite(t).all() for t in values if t.device==d]).all()):raise FloatingPointError(label)
def all_state_tensors(base,support,optimizer):
    values=[getattr(base,k) for k in ATTRS]+list(base._deformation.parameters())
    if support is not None:values+=list(support.parameters())+list(support.buffers())
    values += [v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
    return values
