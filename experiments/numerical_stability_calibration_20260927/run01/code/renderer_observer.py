"""Transparent observers around locked render and extension calls."""
from common import *
import inspect,torch
from restore_state import clone_cpu
import gaussian_renderer as renderer
import diff_gaussian_rasterization as backend
def install(runtime):
    original=inspect.getsource(renderer.render);source=original
    edits={
      'scales_final = pc.scaling_activation(scales_final)':"runtime.guard.check('deformation_before_activation',dict(xyz=means3D_final,log_scale=scales_final,quaternion=rotations_final,quaternion_norm=rotations_final.norm(dim=-1),opacity_logit=opacity_final,SH=shs_final),runtime.detail)\n    scales_final = pc.scaling_activation(scales_final)",
      'rendered_image, radii, depth = rasterizer(' : "runtime.guard.check('activation_outputs',dict(exp_scale=scales_final,rotation=rotations_final,opacity=opacity),runtime.detail)\n    rendered_image, radii, depth = rasterizer("}
    for a,b in edits.items():assert source.count(a)==1;source=source.replace(a,b)
    path=runtime.output/'source_snapshots/render_observed.py';path.parent.mkdir(exist_ok=True);path.write_text(source)
    scope=renderer.__dict__.copy();scope['runtime']=runtime;exec(compile(source,str(path),'exec'),scope)
    for name in ['rasterize_gaussians','rasterize_gaussians_backward']:
        original_fn=getattr(backend._C,name)
        def wrapped(*args,_name=name,_fn=original_fn):
            phase='raster_backward' if _name.endswith('backward') else 'raster_forward'
            runtime.guard.check(phase+'_inputs',args,runtime.detail)
            # Capture before risky kernels so a damaged CUDA context is not needed.
            saved=clone_cpu(args) if runtime.detail else None
            if saved is not None:torch.save(saved,runtime.output/(phase+'_last_inputs.pt'))
            runtime.event(phase+'_call',input_shapes=[list(x.shape) if torch.is_tensor(x) else type(x).__name__ for x in args])
            outputs=_fn(*args)
            runtime.guard.check(phase+'_outputs',outputs,runtime.detail)
            return outputs
        setattr(backend._C,name,wrapped)
    return scope['render']
