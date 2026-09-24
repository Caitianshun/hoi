#!/usr/bin/env python3
"""Frozen RGB priors for the calibrated MoSca pilot, without evaluation inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['tracker', 'depth'])
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, help='Separate output for a controlled tracker comparison')
    p.add_argument('--segmentation', type=Path)
    p.add_argument('--foreground-queries', type=int, default=2000)
    a = p.parse_args()
    ws = a.input.resolve(); m = json.loads((ws / 'input_manifest.json').read_text())
    out_dir = a.output.resolve() if a.output else ws
    out_dir.mkdir(parents=True, exist_ok=True)
    assert m['role'] == 'input_only' and m['images_undistorted']
    torch.manual_seed(12345); np.random.seed(12345)
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in m['frame_paths']]
    for f, expected in zip(m['frame_paths'], m['frame_sha256']):
        assert sha(f) == expected
    meta = dict(stage=a.stage, status='running', input_manifest_sha256=sha(ws / 'input_manifest.json'),
                input_frames=len(images), device=torch.cuda.get_device_name(0),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                script_sha256=sha(__file__), seed=12345)
    record = out_dir / f'{a.stage}_run.json'; record.write_text(json.dumps(meta, indent=2))
    try:
        if a.stage == 'tracker':
            assert a.segmentation is not None
            sys.path[:0] = [str(ROOT/'third_party/co-tracker'), str(ROOT/'third_party/MoSca/lib_prior/tracking')]
            from cotracker.predictor import CoTrackerOnlinePredictor
            from cotracker_wrapper import online_track_point
            ckpt = ROOT/'models/CoTracker3/scaled_online.pth'
            expected = '205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218'
            assert sha(ckpt) == expected, 'CoTracker checkpoint hash mismatch'
            model = CoTrackerOnlinePredictor(checkpoint=str(ckpt), window_len=16, v2=False).cuda().eval()
            xy = np.load(ws/'queries_first_frame.npy')
            first_queries = np.concatenate([np.zeros((len(xy),1), np.float32), xy], axis=1)
            fg = np.load(a.segmentation)['foreground'].astype(bool)
            assert fg.shape == (len(images), m['height'], m['width'])
            tids, yy, xx = np.where(fg)
            choice = np.random.choice(len(tids), min(a.foreground_queries, len(tids)), replace=False)
            extra = np.stack([tids[choice], xx[choice], yy[choice]], -1).astype(np.float32)
            queries = np.concatenate([first_queries, extra], 0)
            np.save(out_dir/'cotracker_queries.npy', queries)
            video = torch.from_numpy(np.stack(images)).permute(0,3,1,2)[None].float()
            torch.cuda.synchronize(); inference_start = time.perf_counter()
            with torch.inference_mode():
                tracks, visible = online_track_point(video, torch.from_numpy(queries)[None], model, torch.device('cuda:0'))
            torch.cuda.synchronize()
            np.savez_compressed(out_dir/'uniform_cotracker_tap.npz', tracks=tracks.numpy(), visibility=visible.numpy(),
                                query_points=queries, timestamp_seconds=m['timestamp_seconds'])
            meta.update(inference_seconds=time.perf_counter()-inference_start, checkpoint_sha256=expected,
                        shared_first_frame_queries=len(xy), extra_foreground_queries=len(extra),
                        mechanism='Upstream MoSca online_track_point: CoTracker3 online forward/backward fusion; support grid enabled',
                        timing_note='Model reads ordered frames, not physical timestamps; no regular-rate assumption in evaluation')
        else:
            sys.path.insert(0, str(ROOT/'third_party/UniDepth'))
            from unidepth.models import UniDepthV2
            ckpt = ROOT/'models/UniDepth/unidepth-v2-vitl14.bin'
            expected = '101d4a941854e7f7ec6839a5a39bfe35bd5cc59ed65d8c5994a55607d8481541'
            assert sha(ckpt) == expected, 'UniDepth checkpoint incomplete or different'
            config = json.loads((ROOT/'third_party/UniDepth/configs/config_v2_vitl14.json').read_text())
            model = UniDepthV2(config)
            result = model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
            model = model.cuda().eval(); model.resolution_level = 3
            K = torch.tensor(m['K'], dtype=torch.float32, device='cuda')[None]
            out = ws/'unidepth_depth'; out.mkdir(exist_ok=True)
            torch.cuda.synchronize(); inference_start = time.perf_counter()
            ranges = []
            for i, rgb in enumerate(images):
                # UniDepth Camera.crop/resize mutates its supplied tensor in place.
                # Give every frame a fresh calibration copy; retain the shared K.
                pred = model.infer(torch.from_numpy(rgb).permute(2,0,1).cuda(), camera=K.clone())
                depth = pred['depth'][0,0].float().cpu().numpy()
                assert depth.shape == rgb.shape[:2] and np.isfinite(depth).all() and (depth > 0).all(), f'Invalid predicted depth frame {i}: shape={depth.shape}, finite={np.isfinite(depth).mean()}, positive={(depth>0).mean()}'
                np.savez_compressed(out/f'{i:05d}.npz', dep=depth)
                ranges.append([float(np.percentile(depth, x)) for x in (1,50,99)])
                print(f'depth frame {i+1}/{len(images)} p01/50/99={ranges[-1]}', flush=True)
            torch.cuda.synchronize()
            meta.update(inference_seconds=time.perf_counter()-inference_start, checkpoint_sha256=expected,
                        depth_quantiles_p01_p50_p99=ranges, resolution_level=3,
                        mechanism='Frozen UniDepthV2 vitl14 with a fresh copy of legal calibrated pinhole K per frame; no sensor-depth input or reference scale alignment')
        meta['status'] = 'completed'
    except Exception as exc:
        meta.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        meta.update(wall_seconds=time.perf_counter()-begin,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
        record.write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
