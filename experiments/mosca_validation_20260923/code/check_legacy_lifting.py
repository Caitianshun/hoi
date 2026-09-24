"""Quantify legacy normalized-grid backprojection against supplied pixel K."""
import sys,json,hashlib
from pathlib import Path
import torch
ROOT=Path('/home/cai_tianshun/Project/HOI');EXP=ROOT/'experiments/mosca_validation_20260923'
sys.path.insert(0,str(ROOT/'third_party/MoSca'))
from lib_moca.camera import MonocularCameras
from lib_prior.prior_loading import get_homo_coordinate_map
old=ROOT/'experiments/mosca_baseline_20260922'
m=json.loads((old/'common_input/input_manifest.json').read_text())
c=MonocularCameras.load_from_ckpt(torch.load(old/'mosca_cotracker/bundle/bundle_cams.pth',map_location='cpu',weights_only=False))
h,w=m['height'],m['width'];grid=torch.tensor(get_homo_coordinate_map(h,w),dtype=torch.float32)
y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
uv=torch.stack([x,y],-1).reshape(-1,2)
with torch.no_grad():
 xyz=c.backproject(grid.reshape(-1,2),torch.full((h*w,),3.))
 projected=xyz@c.K().T;projected=projected[:,:2]/projected[:,2:]
 err=projected-uv
out={'status':'completed','z_for_synthetic_check_m':3.,'max_abs_error_by_axis_px':err.abs().max(0).values.tolist(),
     'mean_abs_error_by_axis_px':err.abs().mean(0).tolist(),
     'interpretation':'Remaining upstream initial lifting pixel convention mismatch, distinct from fixed native raster principal point. Retained identically in current controlled runs, not claimed corrected.',
     'scope':'Synthetic CPU input lifting audit, no reference used. Cannot explain large motion loss by itself; formal geometric benchmarks require a common precise lifting convention.',
     'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'third_party/MoSca/lib_moca/camera.py',ROOT/'third_party/MoSca/lib_prior/prior_loading.py',Path(__file__)]}}
(EXP/'protocol/legacy_lifting_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
