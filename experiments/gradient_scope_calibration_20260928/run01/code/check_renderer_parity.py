"""Frozen same-camera training/evaluation path comparison with repetition baseline."""
from common import *
import torch,os
from types import SimpleNamespace
from render_utils import load_model,CalibratedCamera
from diagnostic_render import make_renderer,diff
from scale_domain import bound_log_scale
from budget import CallBudget
import gaussian_renderer,renderer_observer
class Guard:
    def check(self,phase,values,detail=False):
        for v in values.values() if isinstance(values,dict) else values:
            if torch.is_tensor(v) and v.is_floating_point():assert torch.isfinite(v).all(),phase
class Runtime:
    def __init__(self,bound):self.a=SimpleNamespace(scale_bound=bound);self.output=RUN/'diagnostics/parity';self.output.mkdir(exist_ok=True);self.guard=Guard();self.detail=False
    def event(self,*a,**k):pass
def run():
    assert not (RUN/'renderer_parity.json').exists(), 'Do not overwrite frozen parity evidence'
    torch.set_num_threads(4);assert os.environ['CUDA_VISIBLE_DEVICES']=='1';budget=CallBudget('D0_parity')
    frames=read(OLD/'inputs/hos_backpack/manifest.json')['frames'];by={x['frame_id']:x for x in frames}
    rows=[];bound=read(V5/'protocol/fine_transition.json')['scene_extent']
    native=make_renderer(bound);legacy=make_renderer(bound,True);observed=renderer_observer.install(Runtime(bound))
    attrs=['render','depth','radii','canonical_log_scale_raw','canonical_scale_exp','deformed_log_scale_raw','rendered_scale_bounded','xyz_final','rotation_final','opacity_final']
    for arm in ['B_U','B_F']:
        r=read(V5/'runs'/arm/'run.json');model,(_,_,_,pipe),state=load_model(r['checkpoint'],V5/'runs'/arm/'effective_config.json');del state
        bg=torch.zeros(3,device='cuda')
        with torch.no_grad():
            for frame_id in ['00001','00041']:
                cam=CalibratedCamera(by[frame_id],0,False)
                a=native(cam,model,pipe,bg);b=native(cam,model,pipe,bg);train=observed(cam,model,pipe,bg)
                model.scaling_activation=lambda x:torch.exp(bound_log_scale(x,bound))
                ev=gaussian_renderer.render(cam,model,pipe,bg);e=legacy(cam,model,pipe,bg)
                model.scaling_activation=torch.exp
                repeat={k:diff(a[k],b[k]) for k in attrs}
                pairs={'train_vs_new':{k:diff(train[k],a[k]) for k in ['render','depth','radii']},'eval_vs_new':{k:diff(ev[k],a[k]) for k in ['render','depth','radii']},'eval_attributes_vs_new':{k:diff(e[k],a[k]) for k in attrs}}
                ok=all(v['finite'] and v['max_abs']<=repeat[k]['max_abs'] for ds in pairs.values() for k,v in ds.items())
                rows.append(dict(arm=arm,frame_id=frame_id,repeat=repeat,comparisons=pairs,passed=ok))
                save_json(RUN/'renderer_parity.json',dict(status='passed' if all(x['passed'] for x in rows) else 'failed',rows=rows,tolerance='observed same-path max absolute repeat error; no inflation',physical_surface_correspondence='NA'))
                assert ok,'Renderer mismatch exceeds measured repeat error'
        del model;torch.cuda.empty_cache()
    import diff_gaussian_rasterization as ext
    save_json(RUN/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),backend=identity(ext._C.__file__),adapter=identity(OLD/'code/adapter_4dgs.py'),upstream=OFFICIAL_COMMIT))
    print('D0 parity passed')
if __name__=='__main__':run()
