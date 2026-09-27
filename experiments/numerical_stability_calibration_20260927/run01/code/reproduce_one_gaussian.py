"""Zero-optimizer reproduction of the packaged CUDA failure, without HOI data."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import diff_gaussian_rasterization as raster
def run():
    p=argparse.ArgumentParser();p.add_argument('--case-dir',type=Path,required=True);a=p.parse_args()
    case=json.loads((a.case_dir/'case.json').read_text());blob=np.load(a.case_dir/'one_gaussian_case.npz')
    forward=[torch.from_numpy(blob[f'forward_{i}']).cuda() if f'forward_{i}' in blob else case['forward_scalars'][str(i)] for i in range(19)]
    rows=[]
    for mode in ['original','bounded']:
        args=list(forward)
        if mode=='bounded':args[4]=args[4].clamp(max=case['scale_upper_bound'])
        n,rgb,depth,radii,geometry,binning,image=raster._C.rasterize_gaussians(*args)
        back=(args[0],args[1],radii,args[2],args[4],args[5],args[6],args[7],args[8],args[9],args[10],args[11],torch.from_numpy(blob['grad_RGB']).cuda(),torch.from_numpy(blob['grad_depth']).cuda(),args[14],args[15],args[16],geometry,n,binning,image,args[18])
        grads=raster._C.rasterize_gaussians_backward(*back)
        rows.append(dict(mode=mode,RGB_finite=bool(torch.isfinite(rgb).all()),depth_finite=bool(torch.isfinite(depth).all()),radii=radii.cpu().tolist(),gradient_nonfinite={name:int((~torch.isfinite(g)).sum()) for name,g in zip(['screen','color','opacity','xyz','covariance','SH','scale','rotation'],grads)}))
    print(json.dumps(dict(optimization_steps=0,binary=str(raster._C.__file__),rows=rows),indent=2))
if __name__=='__main__':run()
