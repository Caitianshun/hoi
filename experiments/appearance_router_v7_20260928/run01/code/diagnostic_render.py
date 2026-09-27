"""Bound only final rendered scale; explicit detached precomputed color path."""
from common import *
import inspect,torch
import gaussian_renderer
from scale_domain import bound_log_scale
def make_renderer(bound,legacy_activation=False):
    original=inspect.getsource(gaussian_renderer.render);source=original
    changes={
      'stage="fine", cam_type=None):':'stage="fine", cam_type=None, detach_attributes=False):',
      'scales_final = pc.scaling_activation(scales_final)':
      "canonical_log_scale_raw=pc._scaling\n    deformed_log_scale_raw=scales_final\n    scales_final = pc.scaling_activation(scales_final)" if legacy_activation else
      "canonical_log_scale_raw=pc._scaling\n    deformed_log_scale_raw=scales_final\n    scales_final = torch.exp(bound_log_scale(scales_final,bound))",
      'rendered_image, radii, depth = rasterizer(':
      "if detach_attributes:\n        assert override_color is not None\n        means3D_final=means3D_final.detach();scales_final=scales_final.detach();rotations_final=rotations_final.detach();opacity=opacity.detach()\n    rendered_image, radii, depth = rasterizer(",
      'shs = shs_final,':'shs = shs_final if override_color is None else None,',
      '"depth":depth}':'"depth":depth, "canonical_log_scale_raw":canonical_log_scale_raw, "canonical_scale_exp":torch.exp(canonical_log_scale_raw), "deformed_log_scale_raw":deformed_log_scale_raw, "rendered_scale_bounded":scales_final, "xyz_final":means3D_final, "rotation_final":rotations_final, "opacity_final":opacity}',
    }
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    ns=gaussian_renderer.__dict__.copy();ns.update(bound=bound,bound_log_scale=bound_log_scale)
    path=RUN/'diagnostics/source_snapshots'/('render_legacy.py' if legacy_activation else 'render_final_bound.py');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
    save_json(path.with_suffix('.patch.json'),dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),patched=identity(path),changes=changes))
    exec(compile(source,str(path),'exec'),ns);fn=ns['render']
    def render(*args,**kw):
        pc=args[1];pipe=args[2];assert not pipe.compute_cov3D_python and not pipe.convert_SHs_python
        return fn(*args,**kw)
    return render
def diff(a,b):
    x=a.detach().double();y=b.detach().double();d=x-y
    return dict(max_abs=float(d.abs().max()),relative_L2=float(d.norm()/y.norm().clamp_min(1e-30)),max_relative=float((d.abs()/y.abs().clamp_min(1e-12)).max()),reference_norm=float(y.norm()),finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
