#!/usr/bin/env python3
"""Measure native_add3 alpha centroid for one camera-axis Gaussian per fixed K.

No checkpoint, dataset RGB, training output or model parameter is read/modified.
Use GPU 1 (RTX 3090, sm_86) after other assigned GPU work completes.
--prepare-only records the two camera configurations without importing torch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
EVENT = ROOT/'experiments/mosca_baseline_20260922'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def source_expected_centre(h, w, k):
    """Read-only reproduction of upstream padding/crop and ndc2Pix origin."""
    cx, cy = k[0][2], k[1][2]
    pad = abs(h//2-cy)>1 or abs(w//2-cx)>1
    nw = int(2*max(cx,w-cx)) if pad else w
    nh = int(2*max(cy,h-cy)) if pad else h
    x0 = 0 if not pad or cx>w-cx else nw-w
    y0 = 0 if not pad or cy>h-cy else nh-h
    return [(nw-1)/2-x0, (nh-1)/2-y0], [nh,nw]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, default=EVENT/'common_input/input_manifest.json')
    p.add_argument('--heldout-manifest', type=Path, default=EVENT/'heldout_cpu_preparation_v2/heldout_manifest.json')
    p.add_argument('--repo', type=Path, default=ROOT/'third_party/MoSca')
    p.add_argument('--output', type=Path, required=True, help='New JSON file; existing results are never overwritten')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    m = json.loads(a.input_manifest.read_text())
    held = json.loads(a.heldout_manifest.read_text())
    if m['camera_id']!=0 or not m['images_undistorted'] or held['camera_id']!=1:
        raise ValueError('Expected fixed rectified camera-0 and camera-1 calibration')
    if held['input_manifest_sha256']!=sha(a.input_manifest): raise ValueError('Heldout/input protocol mismatch')
    views = []
    for name,h,w,k in [('input_cam0',m['height'],m['width'],m['K']),
                       ('heldout_cam1',*held['output_hw'],held['K_rectified'])]:
        centre,padded = source_expected_centre(h,w,k)
        views.append({'camera':name,'height':h,'width':w,'K':k,
                      'K_predicted_uv':[k[0][2],k[1][2]],
                      'source_derived_native_uv':centre,'source_derived_padded_hw':padded})
    repo = a.repo.resolve()
    source_files = [repo/path for path in ['lib_render/render_helper.py','lib_render/gauspl_renderer_native_add3.py',
        'lib_render/diff-gaussian-rasterization-alphadep-add3/cuda_rasterizer/auxiliary.h',
        'lib_render/diff-gaussian-rasterization-alphadep-add3/cuda_rasterizer/forward.cu',
        'lib_render/diff-gaussian-rasterization-alphadep-add3/diff_gaussian_rasterization_add3/__init__.py']]
    context = [a.input_manifest.resolve(),a.heldout_manifest.resolve(),Path(__file__).resolve(),*source_files]
    record = {'status':'prepared_cpu_only','views':views,
              'context_sha256':{str(path):sha(path) for path in context},
              'gaussian':{'camera_xyz_m':[0.,0.,3.],'scale_std_m':[.01,.01,.01],
                          'frame':'identity','opacity':.9,'sh_degree':0,'sh_coefficients':[0.,0.,0.]},
              'renderer':'native_add3 via actual lib_render.render_helper.render',
              'world_to_camera':'identity; Gaussian already placed in camera coordinates',
              'measurement':'Sum alpha*u / sum alpha and sum alpha*v / sum alpha; integer pixel centres start at 0; CPU float64 reduction.',
              'scope':'Synthetic renderer-coordinate diagnostic only, not reconstruction accuracy. No image, checkpoint or training output read.',
              'expected_gpu':'physical GPU 1 RTX 3090; existing extension built for sm_86',
              'source_prediction_note':'Analytical padding/crop prediction is distinct from the measured alpha centroid; finite pixel support can shift the latter slightly.'}
    def save(): a.output.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        if not a.prepare_only:
            if not os.environ.get('CUDA_VISIBLE_DEVICES'):
                raise ValueError('Set CUDA_VISIBLE_DEVICES=1 after the assigned GPU work has finished')
            import numpy as np
            import torch
            os.environ['GS_BACKEND']='native_add3'
            sys.path.insert(0,str(repo))
            from lib_render.render_helper import render
            import diff_gaussian_rasterization_add3 as rasterizer
            torch.cuda.set_device(0)
            if torch.cuda.get_device_capability(0)!=(8,6):
                raise ValueError('This diagnostic uses the existing sm_86 build: select the RTX 3090')
            device=torch.device('cuda:0')
            record.update(status='running',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                          gpu_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,
                          rasterizer_binary=str(Path(rasterizer._C.__file__).resolve()),
                          rasterizer_binary_sha256=sha(rasterizer._C.__file__))
            save()
            torch.cuda.reset_peak_memory_stats(device)
            gs=(torch.tensor([[0.,0.,3.]],device=device),torch.eye(3,device=device)[None],
                torch.full((1,3),.01,device=device),torch.full((1,1),.9,device=device),
                torch.zeros((1,3),device=device))
            with torch.no_grad():
                for view in views:
                    k=torch.tensor(view['K'],dtype=torch.float32,device=device)
                    torch.cuda.synchronize(device); start=time.perf_counter()
                    rendered=render(gs,view['height'],view['width'],k,torch.eye(4,device=device),bg_color=[0.,0.,0.])
                    torch.cuda.synchronize(device)
                    view['render_seconds']=time.perf_counter()-start
                    alpha=rendered['alpha'][0].detach().cpu().numpy().astype(np.float64)
                    if alpha.shape!=(view['height'],view['width']) or not np.isfinite(alpha).all() or alpha.sum()<=0:
                        raise ValueError('Invalid alpha raster')
                    yy,xx=np.indices(alpha.shape,dtype=np.float64)
                    uv=np.array([(alpha*xx).sum(),(alpha*yy).sum()])/alpha.sum()
                    view.update(alpha_centroid_uv=uv.tolist(),alpha_sum=float(alpha.sum()),
                                alpha_max=float(alpha.max()),nonzero_alpha_pixels=int((alpha>0).sum()),
                                measured_minus_K_uv_px=(uv-view['K_predicted_uv']).tolist(),
                                measured_minus_source_prediction_px=(uv-view['source_derived_native_uv']).tolist())
            record.update(status='completed',peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device))
        record['wall_seconds']=time.perf_counter()-begin
        save()
        print(json.dumps({'status':record['status'],'output':str(a.output.resolve()),'views':views},indent=2))
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        save()
        raise


if __name__=='__main__':
    main()
