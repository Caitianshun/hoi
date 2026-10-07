#!/usr/bin/env python3
"""Frozen RAFT observations between adjacent retained training images only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def consistency(bwd, fwd):
    h, w = bwd.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    u, v = x + bwd[..., 0], y + bwd[..., 1]
    warped = cv2.remap(fwd, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    error = np.linalg.norm(bwd + warped, axis=-1)
    threshold = .5 * (np.linalg.norm(bwd, axis=-1) + np.linalg.norm(warped, axis=-1)) + .5
    return (error < threshold) & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1)


def image_tensor(path, long_edge, device):
    image = np.array(Image.open(path).convert('RGB'))
    h, w = image.shape[:2]
    scale = min(1., long_edge / max(h, w))
    rh, rw = round(h*scale), round(w*scale)
    image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device)
    return tensor, (h, w)


def main(a):
    root = Path(__file__).resolve().parents[4]
    run = Path(__file__).resolve().parents[1]
    scene = (a.data_root / a.scene).resolve()
    paths = sorted((scene/'images').glob('[0-9]*.png'))
    n = len(paths)
    if n < 32:
        raise RuntimeError(('Unexpected frame count', n))
    test = set(range(0, n, n//16)[:16])
    train = [p for i, p in enumerate(paths) if i not in test]
    assert len(test) == 16
    target = scene/'images_flow'
    target.mkdir(exist_ok=True)
    status_path = run/'protocol'/f'{a.scene}_flow.json'
    identity = dict(scene=a.scene, train_ids=[p.stem for p in train], test_ids=[paths[i].stem for i in sorted(test)],
                    checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha(a.checkpoint),
                    inference_long_edge=a.long_edge, iters=a.iters,
                    validity='forward/backward consistency: error < 0.5*(norms)+0.5 px; in bounds',
                    coordinate='original RGB pixel x/y displacement; current retained train -> previous retained train',
                    first_frame='zero displacement and zero validity; no previous observation',
                    heldout_rgb_read=False, mask_source='RAFT consistency only',
                    upstream='MoSca vendored RAFT / Princeton RAFT', torch=torch.__version__)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    old = json.loads(status_path.read_text()) if status_path.exists() else None
    if old and old['identity_sha256'] != identity_sha:
        raise RuntimeError('Existing flow protocol identity differs; use a new directory')
    done = {r['current']: r for r in old.get('pairs', [])} if old else {}
    expected = train[:a.limit] if a.limit else train
    pending = [p for p in expected if p.stem not in done]
    if not pending:
        return
    sys.path.insert(0, str(root/'third_party/MoSca/lib_prior/optical_flow'))
    from RAFT.raft import RAFT
    from RAFT.utils.utils import InputPadder
    args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
    model = RAFT(args)
    weights = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in weights.items()}, strict=True)
    model.to('cuda').eval()
    torch.set_num_threads(4)
    started = time.time()
    for current in pending:
        idx = train.index(current)
        receipt = target/(current.stem+'_bwd.npz')
        if receipt.exists():
            raise RuntimeError(f'Unindexed cache exists: {receipt}')
        t0 = time.time()
        cur, (h, w) = image_tensor(current, a.long_edge, 'cuda')
        prev = train[idx-1] if idx else None
        if prev is None:
            flow = np.zeros((h,w,2), np.float32)
            valid = np.zeros((h,w), bool)
        else:
            previous, previous_shape = image_tensor(prev, a.long_edge, 'cuda')
            assert previous_shape == (h,w)
            padder = InputPadder(cur.shape)
            cur, previous = padder.pad(cur, previous)
            with torch.inference_mode():
                _, bwd = model(cur, previous, iters=a.iters, test_mode=True)
                _, fwd = model(previous, cur, iters=a.iters, test_mode=True)
            bwd = padder.unpad(bwd)[0].cpu().numpy().transpose(1,2,0)
            fwd = padder.unpad(fwd)[0].cpu().numpy().transpose(1,2,0)
            valid_small = consistency(bwd, fwd)
            rh, rw = bwd.shape[:2]
            flow = cv2.resize(bwd, (w,h), interpolation=cv2.INTER_LINEAR)
            flow[...,0] *= w/rw
            flow[...,1] *= h/rh
            valid = cv2.resize(valid_small.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST).astype(bool)
        assert np.isfinite(flow).all()
        tmp = target/(current.stem+'_bwd.tmp.npz')
        np.savez_compressed(tmp, flow=flow.astype(np.float32), mask=valid.astype(np.float32))
        tmp.replace(receipt)
        done[current.stem] = dict(current=current.stem, previous=prev.stem if prev else None,
                                 source_rgb_sha256=sha(current), previous_rgb_sha256=sha(prev) if prev else None,
                                 cache_sha256=sha(receipt), shape=[h,w,2], valid_fraction=float(valid.mean()), seconds=time.time()-t0)
        complete = len(done) == len(train)
        write(status_path, dict(status='complete' if complete else 'partial', identity=identity,
                               identity_sha256=identity_sha, pairs=list(done.values()), expected_pairs=len(train),
                               invocation_seconds=time.time()-started, updated_unix=time.time()))
        print(f'{a.scene} flow {len(done)}/{len(train)} {current.stem} {time.time()-t0:.2f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--long-edge', type=int, default=768)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())
