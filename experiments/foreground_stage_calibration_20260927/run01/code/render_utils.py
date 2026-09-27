"""Evaluation-only renderer overrides on the original deformed geometry path."""
from common import *
import inspect,hashlib,torch
from adapter_4dgs import CalibratedCamera
import gaussian_renderer
def diagnostic_renderer():
    original=inspect.getsource(gaussian_renderer.render);source=original
    replacements={
      'stage="fine", cam_type=None):':'stage="fine", cam_type=None, eval_sh_degree=None, sh_camera_center=None, alpha_only=False):',
      'sh_degree=pc.active_sh_degree,':'sh_degree=pc.active_sh_degree if eval_sh_degree is None else eval_sh_degree,',
      'campos=viewpoint_camera.camera_center.cuda(),':'campos=viewpoint_camera.camera_center.cuda() if sh_camera_center is None else sh_camera_center.cuda(),',
      'colors_precomp = None':'colors_precomp = torch.ones_like(means3D_final) if alpha_only else None',
      'shs = shs_final,':'shs = None if alpha_only else shs_final,',
      '"depth":depth}':'"depth":depth, "means3D_final":means3D_final, "scales_final":scales_final, "opacity_final":opacity}',
    }
    for old,new in replacements.items():assert source.count(old)==1,old;source=source.replace(old,new)
    ns=gaussian_renderer.__dict__.copy();exec(compile(source,str(UPSTREAM/'gaussian_renderer/__init__.py')+':V4_diagnostic','exec'),ns)
    return ns['render'],dict(original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),replacements=replacements,scope='Evaluation only; no formal training renderer edits. SH Python/override_color paths prohibited.')
def load_model(checkpoint,config):
    cfg=read(config);_,(dataset,hidden,opt,pipe)=official_config(2)
    for obj,key in [(dataset,'model'),(hidden,'hidden'),(opt,'optimization'),(pipe,'pipeline')]:
        for name,value in cfg[key].items():setattr(obj,name,value)
    assert not pipe.convert_SHs_python and not pipe.compute_cov3D_python
    from scene.gaussian_model import GaussianModel
    model=GaussianModel(dataset.sh_degree,hidden);model._deformation=model._deformation.cuda()
    state=torch.load(checkpoint,map_location='cuda',weights_only=False);model.restore(state['model'],opt);model._deformation.eval()
    return model,(dataset,hidden,opt,pipe),state
