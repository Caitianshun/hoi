#!/usr/bin/env python3
"""Fit one input-only Sim(3) from Point4D frame-0 dense geometry to UniDepth.

No evaluation geometry or hidden-view paths are accepted. Output metres inherit
the uncertainty of frozen monocular metric depth, not sensor-depth accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


REFERENCE_FRAME_INDICES = np.array([0, 8, 16, 25, 31, 39, 46, 55, 65, 75, 83, 93, 103, 113])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def transform(x, s, r, t):
    return s * np.asarray(x, dtype=np.float64) @ r.T + t


def fit_sim3(x, y):
    """Least-squares y=s R x+t with a proper rotation and positive scale."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    u, singular, vh = np.linalg.svd(yc.T @ xc / len(x))
    d = np.ones(3)
    d[-1] = np.linalg.det(u @ vh)
    r = (u * d) @ vh
    variance = np.square(xc).sum() / len(x)
    if variance < 1e-12 or singular[1] < 1e-12:
        raise ValueError('Degenerate alignment support')
    s = (singular * d).sum() / variance
    if s <= 0 or not np.isfinite(s):
        raise ValueError('Invalid similarity scale')
    return s, r, my - s * r @ mx


def trimmed_sim3(x, y, fraction=0.8, max_iterations=50):
    """Deterministic alternating least-trimmed squares; no reference tuning."""
    keep_count = int(np.floor(len(x) * fraction))
    if keep_count < 100:
        raise ValueError('Insufficient valid dense support')
    keep = np.arange(len(x))
    history = []
    converged = False
    for iteration in range(max_iterations):
        s, r, t = fit_sim3(x[keep], y[keep])
        residual = np.linalg.norm(transform(x, s, r, t) - y, axis=-1)
        new_keep = np.sort(np.argsort(residual, kind='stable')[:keep_count])
        history.append({'iteration': iteration, 'fit_count': len(keep),
                        'next_kept_rms_m': float(np.sqrt(np.mean(residual[new_keep] ** 2)))})
        if np.array_equal(keep, new_keep):
            converged = True
            break
        keep = new_keep
    # Always fit the final retained support, also on an iteration-limit exit.
    s, r, t = fit_sim3(x[keep], y[keep])
    residual = np.linalg.norm(transform(x, s, r, t) - y, axis=-1)
    return s, r, t, keep, residual, history, converged


def bilinear(image, uv):
    x, y = uv.T
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, image.shape[1] - 1), np.minimum(y0 + 1, image.shape[0] - 1)
    dx, dy = x - x0, y - y0
    return ((1-dx)*(1-dy)*image[y0, x0] + dx*(1-dy)*image[y0, x1]
            + (1-dx)*dy*image[y1, x0] + dx*dy*image[y1, x1])


def stats(values):
    return {'count': int(len(values)), 'rms_m': float(np.sqrt(np.mean(values ** 2))),
            'quantiles_p0_p10_p50_p90_p100_m': np.quantile(values, [0, .1, .5, .9, 1]).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--common-input', type=Path, required=True)
    p.add_argument('--point4d-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    start = time.perf_counter()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Alignment output must be empty; preserve previous results')
    common, point4d = args.common_input.resolve(), args.point4d_run.resolve()
    files = {'manifest': common/'input_manifest.json', 'depth_run': common/'depth_run.json',
             'first_depth': common/'unidepth_depth/00000.npz', 'query_metadata': common/'queries_first_frame.json',
             'point4d_run': point4d/'run.json', 'point4d_predictions': point4d/'predictions.npz'}
    m = json.loads(files['manifest'].read_text())
    depth_run = json.loads(files['depth_run'].read_text())
    run = json.loads(files['point4d_run'].read_text())
    qm = json.loads(files['query_metadata'].read_text())
    assert depth_run['status'] == 'completed' and run['status'] == 'complete'
    assert depth_run['input_manifest_sha256'] == sha(files['manifest'])
    assert run['predictions_sha256'] == sha(files['point4d_predictions'])
    depth = np.load(files['first_depth'])['dep'].astype(np.float64)
    z = np.load(files['point4d_predictions'])
    h, w = int(m['height']), int(m['width'])
    mh, mw = map(int, z['model_hw'])
    gs = int(run['configuration']['geometry_stride'])
    assert depth.shape == (h, w) and tuple(z['input_hw']) == (h, w)
    assert np.array_equal(z['frame_ids'].astype(str), np.asarray(m['frame_indices']).astype(str))
    np.testing.assert_allclose(z['timestamps_seconds'], m['timestamp_seconds'], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(z['query_ids'], np.arange(len(qm['point_ids'])))
    np.testing.assert_allclose(z['query_uv_original'], qm['points'], rtol=0, atol=1e-5)
    np.testing.assert_allclose(z['geometry_cam_R'][0], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(z['geometry_cam_t'][0], 0, atol=1e-6)
    yy, xx = np.meshgrid(np.arange(0, mh, gs), np.arange(0, mw, gs), indexing='ij')
    assert z['geometry_points'].shape[1:3] == xx.shape
    uv_model = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float64)
    uv_common = (uv_model + .5) * [w/mw, h/mh] - .5
    source = z['geometry_points'][0].reshape(-1, 3).astype(np.float64)
    projected = source @ z['geometry_intrinsics'][0].astype(np.float64).T
    projected = projected[:, :2] / projected[:, 2:]
    dense_reprojection_error = np.linalg.norm(projected - uv_model, axis=1)
    assert dense_reprojection_error.max() < .001, 'Dense grid mapping disagrees with upstream geometry'
    valid = (np.isfinite(source).all(1) & (source[:, 2] > 0)
             & (uv_common[:, 0] >= 0) & (uv_common[:, 0] <= w-1)
             & (uv_common[:, 1] >= 0) & (uv_common[:, 1] <= h-1))
    flat_indices = np.flatnonzero(valid)
    depths = bilinear(depth, uv_common[valid])
    valid_depth = np.isfinite(depths) & (depths > 0)
    flat_indices, depths = flat_indices[valid_depth], depths[valid_depth]
    source, uv = source[flat_indices], uv_common[flat_indices]
    k, c2w = np.asarray(m['K'], np.float64), np.asarray(m['c2w'], np.float64)
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(k).T
    camera_target = rays * (depths / rays[:, 2])[:, None]
    target = camera_target @ c2w[:3, :3].T + c2w[:3, 3]
    s, r, t, keep, residual, history, converged = trimmed_sim3(source, target)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = s * r, t
    retained = np.zeros(len(source), bool)
    retained[keep] = True
    np.savez_compressed(out/'alignment_support.npz', dense_flat_indices=flat_indices,
                        uv_model=uv_model[flat_indices], uv_common=uv, point4d_source=source,
                        unidepth_target_world=target, unidepth_z_m=depths,
                        retained_mask=retained, residual_m=residual)
    full = transform(z['trajectories'], s, r, t)
    assert full.shape == (114, 822, 3) and np.isfinite(full).all()
    np.savez_compressed(out/'all_tracks_world.npz', trajectories=full.astype(np.float32),
                        confidence=z['confidence'], query_ids=z['query_ids'],
                        query_uv_original=z['query_uv_original'], frame_ids=z['frame_ids'],
                        timestamps_seconds=z['timestamps_seconds'], units=np.array('m'),
                        coordinate_frame=np.array('behave_world_k1_color'))
    selected = full[REFERENCE_FRAME_INDICES, :6]
    np.savez_compressed(out/'prediction_first6_reference_frames.npz', predicted=selected,
                        predicted_valid_mask=np.isfinite(selected).all(-1),
                        frame_times=z['timestamps_seconds'][REFERENCE_FRAME_INDICES],
                        query_id=np.asarray(qm['point_ids'][:6], dtype=str),
                        entity=np.asarray(qm['entity'][:6], dtype=str), units=np.array('m'),
                        coordinate_frame=np.array('behave_world_k1_color'),
                        prediction_frame_indices=REFERENCE_FRAME_INDICES,
                        confidence=z['confidence'][REFERENCE_FRAME_INDICES, :6])
    meta = {'status': 'completed', 'script_sha256': sha(__file__),
            'input_files': {key: {'path': str(path), 'sha256': sha(path)} for key, path in files.items()},
            'information_boundary': 'Only existing Point4D RGB predictions, first-frame frozen UniDepth RGB prediction, legal calibrated K/c2w, and preselected RGB query metadata. No evaluation coordinates, fitted meshes, sensor depth, or hidden views read.',
            'global_transform_fitted_on_evaluation': False, 'alignment_frame_indices': [0],
            'transform_scope': 'One fixed transform shared by all 114 frames and all 822 points/entities.',
            'source_geometry': 'Upstream unproject_depth_map: predicted z depth backprojected with predicted K, then camera-to-frame0; first frame is identity.',
            'target_geometry': 'UniDepthV2 infer depth is points[:, -1:] (z-depth); same-pixel bilinear sample, calibrated K backprojection, fixed calibrated c2w.',
            'metric_scope': 'Estimated metres from frozen monocular metric-depth prior plus known camera. Does not certify sensor or reference accuracy.',
            'support_selection': f'Entire first-frame dense grid at model pixel stride {gs}; finite positive depth and in-bounds common pixel only. No entity labels or query confidence used.',
            'support_reason': 'First-frame decoded query trajectories have a large self-reprojection tail; dense geometry has an explicit pixel-to-depth mapping and avoids using those tracking errors to determine scene alignment.',
            'fit': 'Unweighted proper-rotation Umeyama, initialized on all valid pairs; alternate fitting and retaining exactly lowest 80% residuals; stable sort; at most 50 iterations. Final refit on retained subset.',
            'retained_fraction': .8, 'iterations': history, 'converged': converged,
            'model_hw': [mh, mw], 'common_hw': [h, w], 'geometry_stride': gs,
            'resize_mapping': f'common_uv=(model_uv+0.5)*[{w}/{mw},{h}/{mh}]-0.5',
            'support_common_uv_min': uv.min(0).tolist(), 'support_common_uv_max': uv.max(0).tolist(),
            'dense_self_reprojection_max_px': float(dense_reprojection_error.max()),
            'scale': float(s), 'rotation': r.tolist(), 'translation_m': t.tolist(),
            'matrix_source_to_world': matrix.tolist(), 'matrix_convention': 'Column homogeneous world_point = matrix @ source_point.',
            'residual_all': stats(residual), 'residual_retained': stats(residual[keep]),
            'residual_meaning': 'Agreement between two input-derived geometry estimates; NOT an independent reconstruction or tracking accuracy metric.',
            'confidence_semantics': 'Raw Point4D confidence; not visibility or calibrated probability; finite-only predicted_valid_mask.',
            'prediction_frame_indices': REFERENCE_FRAME_INDICES.tolist(),
            'output_files': {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size} for path in out.glob('*.npz')},
            'cpu_wall_seconds': time.perf_counter() - start}
    (out/'alignment_protocol.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: meta[k] for k in ['status', 'scale', 'converged', 'residual_all', 'residual_retained', 'cpu_wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
