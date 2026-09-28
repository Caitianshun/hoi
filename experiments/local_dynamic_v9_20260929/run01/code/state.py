"""Explicit after-Adam state; strict schema and loss/sampler implementation."""
from common import *
import types, random
import numpy as np
import torch
from torch import nn
from restore_state import clone_cpu, to_device, exact
from train_official import rng_capture, rng_restore
from scene.gaussian_model import GaussianModel
from hoi_modules.static_dynamic_gaussians import forbid_role_reclassification, validate_point_state
from hoi_modules.local_spacetime_input import install_local_input

ATTRS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
SCHEMA = 'v9_after_adam_v1'


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parameters(scene, stage):
    _, (ds, hidden, opt, pipe) = official_config(config()['batch_size'])
    inherited = read(V5 / 'runs/B_U/effective_config.json')
    for obj, key in ((hidden, 'hidden'), (opt, 'optimization'), (pipe, 'pipeline')):
        assert vars(obj) == inherited[key], key
    inp = ROOT / config()['scenes'][scene]['input_dir']
    ds.source_path = str(inp / 'manifest.json'); ds.render_process = False
    if stage == 'fine':
        opt.iterations = config()['schedule']['fine_updates']
        opt.position_lr_max_steps = config()['schedule']['fine_lr_max_steps']
    assert opt.lambda_dssim == opt.lambda_lpips == 0
    assert not opt.add_point and not opt.dataloader and not opt.zerostamp_init
    assert ds.white_background and not pipe.compute_cov3D_python and not pipe.convert_SHs_python
    return ds, hidden, opt, pipe


def new_model(ds, hidden, mode, domain):
    model = GaussianModel(ds.sh_degree, hidden)
    model.update_deformation_table = types.MethodType(forbid_role_reclassification, model)
    if mode == 'SL':
        assert install_local_input(model, domain) == 512
    model._deformation.cuda()
    return model


def point_attributes(model):
    return {name: getattr(model, name).detach().cpu().clone() for name in ATTRS}


def set_points(model, attrs, role):
    for name in ATTRS:
        setattr(model, name, nn.Parameter(attrs[name].to('cuda').clone()))
    model._deformation_table = role.to('cuda').bool().clone()
    model.max_radii2D = torch.zeros(len(role), device='cuda')


def attrs_identity(attrs):
    return {name: dict(shape=list(t.shape), sha256=tensor_hash(t)) for name, t in attrs.items()}


def schedule(scene, stage):
    return read(scene_dir(scene) / 'protocol' / f'{stage}_RGB_schedule.json')


def sampler_at(sched, completed):
    return dict(completed_updates=completed,
                remaining_stack=sched['remaining_stacks'][completed],
                next_frame_uids=sched['batches'][completed] if completed < len(sched['batches']) else [],
                schedule_sha256=sched['schedule_sha256'])


def capture(model, scene, mode, stage, completed, sched, meta, **extra):
    validate_point_state(model)
    return clone_cpu(dict(schema=SCHEMA, model=model.capture(), scene=scene, mode=mode,
        stage=stage, completed_updates=completed, iteration=completed,
        optimizer_updates=completed, phase='after_Adam_and_zero_grad', resumable=True,
        deformation_accum=model._deformation_accum, rng=rng_capture(),
        sampler=sampler_at(sched, completed), metadata=meta, **extra))


def restore(s, mode=None, reset_optimizer=False):
    assert s['schema'] == SCHEMA and s['phase'] == 'after_Adam_and_zero_grad'
    mode = mode or s['mode']; stage = s['stage']
    ds, hidden, opt, pipe = parameters(s['scene'], stage)
    model = new_model(ds, hidden, mode, s['metadata']['local_domain'])
    if mode == s['mode']:
        model.restore(to_device(s['model'], 'cuda'), opt)
    else:
        assert s['completed_updates'] == 0 and reset_optimizer
        a = s['model']
        attrs = dict(zip(ATTRS, [a[1], a[4], a[5], a[6], a[7], a[8]]))
        set_points(model, attrs, a[3]); model.active_sh_degree = a[0]; model.spatial_lr_scale = a[13]
        state_dict = clone_cpu(a[2])
        if mode == 'SL':
            k = 'deformation_net.feature_out.0.weight'
            old = state_dict[k]
            state_dict[k] = torch.cat((old, torch.zeros((old.shape[0], 4), dtype=old.dtype)), 1)
            state_dict['deformation_net.local_center'] = model._deformation.deformation_net.local_center.cpu()
            state_dict['deformation_net.local_radius'] = model._deformation.deformation_net.local_radius.cpu()
        model._deformation.load_state_dict(state_dict, strict=True)
        model.training_setup(opt)
    model._deformation_accum = s['deformation_accum'].cuda().clone()
    if reset_optimizer:
        model.training_setup(opt)
        model.max_radii2D.zero_()
        assert not model.optimizer.state
    validate_point_state(model)
    rng_restore(s['rng'])
    return model, (ds, hidden, opt, pipe)


def quality_objective(pred, gt):
    from utils.loss_utils import ssim
    weights = config()['quality_objective']
    l1 = (pred - gt).abs().mean()
    structural = ssim(pred, gt, window_size=11)
    return weights['L1'] * l1 + weights['one_minus_SSIM11'] * (1 - structural), l1, structural


def background_objective(pred, gt, weights):
    pixels = weights.sum(dim=(-2, -1)); valid = pixels > 0
    if not bool(valid.any()):
        return pred.sum() * 0, int(len(pixels))
    per_image = ((pred - gt).abs() * weights[:, None]).sum(dim=(1, 2, 3)) / (3 * pixels.clamp_min(1))
    return per_image[valid].mean(), int((~valid).sum())


def load_cameras(scene, background=False):
    from adapter_4dgs import CalibratedCamera
    import cv2
    m = read(ROOT / config()['scenes'][scene]['input_dir'] / 'manifest.json')
    cams = []
    for i, f in enumerate(m['frames']):
        cam = CalibratedCamera(f, i)
        if background:
            mask = cv2.imread(f['mask_path'], 0)
            assert mask is not None and sha(f['mask_path']) == f['mask_sha256']
            size = config()['mask_dilation_square']
            dilated = cv2.dilate((mask >= config()['mask_threshold']).astype(np.uint8),
                                np.ones((size, size), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
            cam.background_weight = torch.from_numpy(dilated == 0)
        cams.append(cam)
    return cams
