"""Synthetic exact-camera and reversible-scale checks before real training."""
from pathlib import Path
import json, os, sys, time, hashlib
import numpy as np
import torch

ROOT = Path('/home/cai_tianshun/Project/HOI')
EXP = ROOT/'experiments/mosca_validation_20260923'
REPO = EXP/'code/MoSca'
INPUT = ROOT/'experiments/mosca_baseline_20260922/common_input'
os.environ['GS_BACKEND'] = 'native_add3'
sys.path.insert(0,str(REPO))
from lib_prior.prior_loading import Saved2D
from lib_render.render_helper import render

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.perf_counter()
m=json.loads((INPUT/'input_manifest.json').read_text())
held=json.loads((ROOT/'experiments/mosca_baseline_20260922/heldout_cpu_preparation_v2/heldout_manifest.json').read_text())
torch.set_num_threads(8)
data=Saved2D(str(INPUT)).load_dep('unidepth_depth',0.3).normalize_depth(1.)
scale=float(data.scale_nw)
del data
if torch.cuda.get_device_capability(0)!=(8,6): raise ValueError('Select assigned RTX3090')
dev='cuda:0'
rows=[]
for name,k0,h,w in [('cam0',m['K'],m['height'],m['width']),('cam1',held['K_rectified'],*held['output_hw'])]:
    k=torch.tensor(k0,dtype=torch.float32,device=dev)
    for target in [[float(k[0,2]),float(k[1,2])],[40.25,40.75],[w-40.75,h-40.25],[.3*w,.7*h],[.8*w,.25*h]]:
        z=3.
        point=torch.tensor([[(target[0]-float(k[0,2]))/float(k[0,0])*z,
                             (target[1]-float(k[1,2]))/float(k[1,1])*z,z]],device=dev,requires_grad=True)
        gs=(point,torch.eye(3,device=dev)[None],torch.full((1,3),.01,device=dev),
            torch.full((1,1),.9,device=dev),torch.zeros((1,3),device=dev))
        out=render(gs,h,w,k,torch.eye(4,device=dev),bg_color=[0.,0.,0.])
        y,x=torch.meshgrid(torch.arange(h,device=dev),torch.arange(w,device=dev),indexing='ij')
        a=out['alpha'][0]
        uv=torch.stack([(a*x).sum(),(a*y).sum()])/a.sum()
        err=float((uv-torch.tensor(target,device=dev)).abs().max())
        uv.sum().backward()
        if not torch.isfinite(point.grad).all() or float(point.grad.abs().max())==0:raise AssertionError('invalid projection gradient')
        rows.append({'camera':name,'expected_uv':target,'measured_uv':uv.detach().cpu().tolist(),'max_error_px':err,
                     'xyz_gradient':point.grad.detach().cpu().tolist()})
        if err>.02:raise AssertionError(rows[-1])
        del out,a,uv

# Coordinate normalization must transform the camera translation as well.
k=torch.tensor(m['K'],dtype=torch.float64)
c2w=torch.tensor(m['c2w'],dtype=torch.float64)
cam=torch.tensor([[.1,.2,2.],[-.3,.4,3.],[.4,-.2,7.]],dtype=torch.float64)
world=cam@c2w[:3,:3].T+c2w[:3,3]
cn=c2w.clone();cn[:3,3]*=scale
wc=torch.linalg.inv(c2w);wn=torch.linalg.inv(cn)
recovered=world@wc[:3,:3].T+wc[:3,3]
normalized=(world*scale)@wn[:3,:3].T+wn[:3,3]
uv0=recovered@k.T;uv0=uv0[:,:2]/uv0[:,2:]
uv1=normalized@k.T;uv1=uv1[:,:2]/uv1[:,2:]
projection_error=float((uv0-uv1).abs().max())
roundtrip_error=float((world*scale/scale-world).abs().max())
assert projection_error<1e-9 and roundtrip_error<1e-12

# Actual rasterization is invariant to the matched geometry/camera/size scaling.
g=(world.float().to(dev),torch.eye(3,device=dev)[None].repeat(3,1,1),torch.full((3,3),.01,device=dev),
   torch.full((3,1),.9,device=dev),torch.zeros((3,3),device=dev))
gn=(g[0]*scale,g[1],g[2]*scale,g[3],g[4])
with torch.no_grad():
    a=render(g,m['height'],m['width'],k.float().to(dev),wc.float().to(dev),bg_color=[0.,0.,0.])
    b=render(gn,m['height'],m['width'],k.float().to(dev),wn.float().to(dev),bg_color=[0.,0.,0.])
    alpha_difference=float((a['alpha']-b['alpha']).abs().max())
    mask=(a['alpha']>.1)&(b['alpha']>.1)
    depth_difference=float((a['dep'][mask]-b['dep'][mask]/scale).abs().max())
assert alpha_difference<.0002 and depth_difference<.0002
record={'status':'completed','world_scale':scale,'scale_source':'1 / torch median over upstream legal depth mask (.3 threshold), whole input video',
        'evaluation_reference_read':False,'calibration_only_cam1':'synthetic projection check',
        'projection_cases':rows,'max_projection_error_px':max(x['max_error_px'] for x in rows),
        'normalization_projection_error_px':projection_error,'normalization_roundtrip_error_m':roundtrip_error,
        'actual_render_scale_alpha_max_difference':alpha_difference,'actual_render_inverse_depth_max_difference_m':depth_difference,
        'gpu':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'wall_seconds':time.perf_counter()-start,'input_manifest_sha256':sha(INPUT/'input_manifest.json'),
        'script_sha256':sha(__file__),'renderer_sha256':sha(REPO/'lib_render/gauspl_renderer_native_add3.py')}
(EXP/'protocol/checks.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
