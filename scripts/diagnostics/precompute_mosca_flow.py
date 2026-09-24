#!/usr/bin/env python3
"""Generate upstream-compatible RAFT optical-flow priors for frozen legal RGB.

Uses MoSca's vendored RAFT, its image/flow resizing and forward-backward mask
functions unchanged. Only the driver, atomic output and audit log are added.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'models/RAFT/raft-things.pth')
    p.add_argument('--config', type=Path, default=ROOT / 'third_party/MoSca/profile/demo/demo_fit.yaml')
    p.add_argument('--repo', type=Path, default=ROOT / 'third_party/MoSca')
    p.add_argument('--device', default='cuda:0', help='Logical device; caller binds physical GPU')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    import cv2
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(a.repo.resolve()))
    from lib_prior.optical_flow.raft_wrapper import (get_raft_model, load_image,
                                                    resize_flow, compute_fwdbwd_mask, InputPadder)
    ws = a.input.resolve()
    manifest_path = ws / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['role'] != 'input_only' or not manifest['images_undistorted']:
        raise ValueError('Use the frozen processed legal RGB manifest')
    cfg = OmegaConf.load(a.config)
    steps = sorted(set(int(x) for x in cfg.track_flow_interval_candidates))
    if not steps or min(steps) < 1 or a.iters < 1: raise ValueError('Invalid intervals/iterations')
    files = [Path(x) for x in manifest['frame_paths']]
    names = [x.name for x in files]
    if len(set(names)) != len(names) or names != sorted(names): raise ValueError('Names must be unique chronological sort order')
    times = np.asarray(manifest['timestamp_seconds'])
    H, W = int(manifest['height']), int(manifest['width'])
    for f, expected in zip(files, manifest['frame_sha256']):
        if sha(f) != expected: raise ValueError(f'Input image hash mismatch: {f}')
    pairs = [(i, i+step) for step in steps for i in range(len(files)-step)]
    expected_directed = {(i,j) for i,j in pairs} | {(j,i) for i,j in pairs}
    flow_dir = ws / 'flow_raft'
    record_path = ws / 'flow_precompute_run.json'
    source_files = [a.repo / 'lib_prior/optical_flow/raft_wrapper.py',
                    a.repo / 'lib_prior/optical_flow/flow_utils.py',
                    a.repo / 'lib_prior/prior_loading.py']
    source_files += sorted((a.repo / 'lib_prior/optical_flow/RAFT').rglob('*.py'))
    identity = {'input_manifest_sha256': sha(manifest_path), 'checkpoint_sha256': sha(a.checkpoint),
                'source_sha256': {str(x.relative_to(a.repo)): sha(x) for x in source_files},
                'intervals': steps, 'iterations': a.iters, 'long_side': 768,
                'small': False, 'mixed_precision': False}
    previous = None
    if flow_dir.exists() and any(flow_dir.glob('*.npz')):
        if not a.resume or not record_path.exists(): raise ValueError('Preserve existing priors; use --resume with matching identity')
        previous = json.loads(record_path.read_text())
        if previous['identity'] != identity: raise ValueError('Existing flow generation identity differs')
    flow_dir.mkdir(exist_ok=True)
    start = time.perf_counter()
    record = {'status':'running', 'identity':identity, 'script_sha256':sha(__file__),
              'source':'MoSca vendored official RAFT architecture, original raft-things.pth; no torchvision model substitution',
              'upstream_commit':subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'], text=True).strip(),
              'input_frames':len(files), 'undirected_pairs':len(pairs), 'expected_directed_files':len(expected_directed),
              'output':str(flow_dir), 'checkpoint':str(a.checkpoint.resolve()),
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'device':a.device,
              'mask_rule':'MoSca compute_fwdbwd_mask unchanged: alpha1=0.5, alpha2=0.5; cubic inverse-flow warping with constant border',
              'information_boundary':'Only the supplied same-camera RGB; no evaluation-only geometry, masks, depth or other cameras',
              'timing_note':'Pair intervals are frame-index offsets from original demo config; actual elapsed seconds are separately recorded',
              'completed_pairs':0, 'files':[], 'pair_times':[]}
    if previous: record['previous_attempt_status'] = previous['status']
    def save():
        tmp=record_path.with_suffix('.json.partial');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(record_path)
    save()
    def validate_file(path):
        with np.load(path, allow_pickle=False) as z:
            flow, mask = z['flow'], z['mask']
            if flow.shape != (H,W,2) or mask.shape != (H,W): raise ValueError(f'Bad output shape: {path}')
            if flow.dtype != np.float16 or mask.dtype != np.float16: raise ValueError(f'Bad output dtype: {path}')
            if not np.isfinite(flow).all() or not np.isin(mask,[0,1]).all(): raise ValueError(f'Invalid output values: {path}')
            return float(mask.mean(dtype=np.float64))
    def append_file(path,i,j):
        valid_fraction=validate_file(path)
        record['files'].append({'path':str(path),'src_index':i,'dst_index':j,'sha256':sha(path),
                                'bytes':path.stat().st_size,'mask_valid_fraction':valid_fraction})
    try:
        device=torch.device(a.device)
        if device.type!='cuda': raise ValueError('Use an explicitly assigned CUDA device')
        torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(12345);np.random.seed(12345)
        record['gpu']=torch.cuda.get_device_name(device)
        model=get_raft_model(str(a.checkpoint),device=device,small=False,mixed_precision=False)
        torch.cuda.synchronize(device)
        record['model_load_seconds']=time.perf_counter()-start
        inference_total=0.0
        for pi,(i,j) in enumerate(pairs):
            fp=flow_dir/f'{names[i]}_to_{names[j]}.npz'
            bp=flow_dir/f'{names[j]}_to_{names[i]}.npz'
            if a.resume and fp.exists() and bp.exists():
                append_file(fp,i,j);append_file(bp,j,i)
                record['pair_times'].append({'i':i,'j':j,'delta_seconds':float(times[j]-times[i]),'resumed':True})
            else:
                if fp.exists() or bp.exists(): raise ValueError('Partial pair exists; inspect before replacing')
                image1,shape1=load_image(str(files[i]));image2,shape2=load_image(str(files[j]))
                if shape1!=(H,W) or shape2!=(H,W): raise ValueError('Image dimensions differ from manifest')
                padder=InputPadder(image1.shape)
                image1,image2=padder.pad(image1.to(device),image2.to(device))
                torch.cuda.synchronize(device);begin=time.perf_counter()
                with torch.no_grad():
                    _,fwd=model(image1,image2,iters=a.iters,test_mode=True)
                    _,bwd=model(image2,image1,iters=a.iters,test_mode=True)
                torch.cuda.synchronize(device);elapsed=time.perf_counter()-begin;inference_total+=elapsed
                fwd=padder.unpad(fwd[0]).cpu().numpy().transpose(1,2,0)
                bwd=padder.unpad(bwd[0]).cpu().numpy().transpose(1,2,0)
                fwd=resize_flow(fwd,H,W);bwd=resize_flow(bwd,H,W)
                mf,mb=compute_fwdbwd_mask(fwd,bwd)
                for path,flow,mask in [(fp,fwd,mf),(bp,bwd,mb)]:
                    tmp=path.with_suffix('.npz.partial')
                    with open(tmp,'wb') as f: np.savez_compressed(f,flow=flow.astype(np.float16),mask=mask.astype(np.float16))
                    tmp.replace(path)
                append_file(fp,i,j);append_file(bp,j,i)
                record['pair_times'].append({'i':i,'j':j,'delta_seconds':float(times[j]-times[i]),'inference_seconds':elapsed})
            record['completed_pairs']=pi+1
            record['inference_seconds']=inference_total
            if pi==0 or (pi+1)%10==0 or pi==len(pairs)-1:
                record['elapsed_seconds']=time.perf_counter()-start;save()
                print(f'RAFT pairs {pi+1}/{len(pairs)}, directed files {len(record["files"])}, wall {record["elapsed_seconds"]:.1f}s',flush=True)
        actual_files=set(flow_dir.glob('*.npz'))
        if len(actual_files)!=len(expected_directed): raise ValueError('Unexpected number of flow files')
        # Exercise the actual upstream Saved2D loader without instantiating or normalizing other priors.
        from lib_prior.prior_loading import Saved2D
        class FlowReceiver:
            def __init__(self):
                self.ws=str(ws);self.frame_names=[f.stem for f in files]
            def register_gradfree_buffer(self,name,value): setattr(self,name,value)
        receiver=FlowReceiver();Saved2D.load_flow(receiver)
        if set(receiver.flow_ij_to_listind_dict)!=expected_directed:
            raise ValueError('Upstream loader pair lookup mismatch')
        if tuple(receiver.flow.shape)!=(len(expected_directed),H,W,2): raise ValueError('Upstream stacked shape differs')
        record['upstream_loader_acceptance']={'status':'passed','lookup_pairs':len(receiver.flow_ij_to_listind_dict),
                                             'flow_shape':list(receiver.flow.shape),'mask_shape':list(receiver.flow_mask.shape)}
        record['output_bytes']=sum(x['bytes'] for x in record['files'])
        record['status']='completed'
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        record['wall_seconds']=time.perf_counter()-start
        if torch.cuda.is_initialized():
            record['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
            record['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device)
        save()
    print(json.dumps({'status':record['status'],'record':str(record_path),'directed_files':len(record['files']),
                      'wall_seconds':record['wall_seconds'],'inference_seconds':record['inference_seconds']}),flush=True)


if __name__=='__main__': main()
