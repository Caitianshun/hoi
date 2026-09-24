#!/usr/bin/env python3
"""Prepare calibrated monocular RGB only; no evaluation assets are read."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--width', type=int, default=640)
    a = p.parse_args()
    start = time.perf_counter()
    m = json.loads(a.manifest.read_text())
    assert m['role'] == 'input_only'
    assert not m['images_undistorted']
    W, H = m['original_size']
    w, h = a.width, round(H * a.width / W)
    K = np.asarray(m['K'], dtype=np.float64)
    distortion = np.asarray(m['distortion'], dtype=np.float64)
    K_rect, roi = cv2.getOptimalNewCameraMatrix(K, distortion, (W, H), 0, (W, H))
    S = np.array([[w / W, 0, (w / W - 1) / 2],
                  [0, h / H, (h / H - 1) / 2], [0, 0, 1]])
    K_out = S @ K_rect
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, distortion, None, K_out, (w, h), cv2.CV_32FC1)
    valid = (map_x >= 0) & (map_x < W - 1) & (map_y >= 0) & (map_y < H - 1)
    assert valid.mean() > .995, f'Unexpected invalid support: {valid.mean()}'
    out = a.output.resolve()
    (out / 'images').mkdir(parents=True, exist_ok=True)
    records = []
    for i, source in enumerate(m['frame_paths']):
        assert sha(source) == m['frame_sha256'][i], f'Input hash mismatch: {source}'
        rgb = cv2.imread(source, cv2.IMREAD_COLOR)
        assert rgb.shape[:2] == (H, W)
        result = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
        dst = out / 'images' / f'{i:05d}.png'
        assert cv2.imwrite(str(dst), result)
        records.append(dict(index=i, path=str(dst), sha256=sha(dst),
                            video_frame_index=m['frame_indices'][i],
                            timestamp_seconds=m['timestamp_seconds'][i]))
    np.savez_compressed(out / 'rgb_rectification.npz', valid=valid, K_original=K,
                        distortion=distortion, K_rectified_full=K_rect, K=K_out)
    prepared = dict(m)
    prepared.update(frame_paths=[r['path'] for r in records], frames=records,
                    frame_sha256=[r['sha256'] for r in records], K=K_out.tolist(),
                    distortion=[0.] * 8, width=w, height=h,
                    images_undistorted=True, images_resized=True,
                    parent_manifest=str(a.manifest.resolve()),
                    parent_manifest_sha256=sha(a.manifest),
                    processing=dict(method='OpenCV rational distortion rectification, alpha=0; direct bilinear remap to target pinhole grid',
                                    pixel_centres='u_out = scale * (u_rectified + 0.5) - 0.5',
                                    full_rectified_K=K_rect.tolist(), full_roi=list(map(int, roi)),
                                    valid_fraction=float(valid.mean()),
                                    elapsed_seconds=time.perf_counter()-start,
                                    script_sha256=sha(__file__)))
    (out / 'input_manifest.json').write_text(json.dumps(prepared, indent=2))
    print(json.dumps(dict(frames=len(records), resolution=[w, h], output=str(out),
                          K=K_out.tolist(), elapsed_seconds=time.perf_counter()-start)))


if __name__ == '__main__':
    main()
