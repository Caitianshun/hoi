"""Single RGB background factory and bounded one-raster V9 renderer."""
from common import *
import torch
from hoi_modules.projected_motion import make_renderer as bounded_renderer
from hoi_modules.static_dynamic_gaussians import deform_attributes


def rgb_background(ds, device='cuda'):
    assert ds.white_background
    return torch.ones(3, dtype=torch.float32, device=device)


def make_renderer(bound, mode):
    base = bounded_renderer(bound)
    old = '''pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)'''
    assert base.expanded_source.count(old) == 1
    source = base.expanded_source.replace(old, 'v9_deform_attributes(pc, time, v9_mode)')
    import gaussian_renderer
    ns = gaussian_renderer.__dict__.copy()
    ns.update(bound=bound, v9_deform_attributes=deform_attributes, v9_mode=mode)
    exec(compile(source, '<v9_single_raster>', 'exec'), ns)
    fn = ns['render']; fn.expanded_source = source
    return fn
