#!/usr/bin/env python3
"""Bounded CPU-only pixel/lift audit. Read frozen A inputs; write this audit only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/mosca_baseline_20260922'
A = ROOT / 'experiments/mosca_validation_20260923/a_normalized_exact'
sys.path.insert(0, str(ROOT / 'third_party/MoSca'))
from lib_moca.camera import MonocularCameras
from lib_prior.prior_loading import get_homo_coordinate_map
from join_seed_and_threshold import kernel, get_weights, warp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_pixel(xyz, K):
    homogeneous = xyz @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def backproject_pixel(uv, camera_z, K):
    """Explicit integer pixel centres; z is camera-axis depth, not ray length."""
    pixel_h = np.c_[uv, np.ones(len(uv), dtype=uv.dtype)]
    rays = pixel_h @ np.linalg.inv(K).T
    return rays * (np.asarray(camera_z) / rays[:, 2])[:, None]


def stats(error):
    error = np.asarray(error, dtype=np.float64)
    return {
        'minimum_signed_error_xy_px': error.min(0).tolist(),
        'maximum_signed_error_xy_px': error.max(0).tolist(),
        'maximum_absolute_error_xy_px': np.abs(error).max(0).tolist(),
        'mean_absolute_error_xy_px': np.abs(error).mean(0).tolist(),
        'rms_error_xy_px': np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        'maximum_euclidean_error_px': float(np.linalg.norm(error, axis=-1).max()),
        'mean_euclidean_error_px': float(np.linalg.norm(error, axis=-1).mean()),
    }


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    inputs = {
        'input_manifest': BASE / 'common_input/input_manifest.json',
        'a_run': A / 'model/run.json',
        'a_initial_camera': A / 'model/bundle/bundle_cams.pth',
        'a_final_camera': A / 'model/photometric_cam.pth',
        'a_initial_dynamic': A / 'model/initial_dynamic.pth',
        'actual_seeds': OUT.parent / 'sampling/run01/actual_dynamic_seeds.npz',
        'camera_source': ROOT / 'third_party/MoSca/lib_moca/camera.py',
        'grid_source': ROOT / 'third_party/MoSca/lib_prior/prior_loading.py',
        'kernel_source': OUT / 'join_seed_and_threshold.py',
    }
    before = {name: sha(path) for name, path in inputs.items()}
    manifest = json.loads(inputs['input_manifest'].read_text())
    K = np.asarray(manifest['K'], dtype=np.float64)
    C = np.asarray(manifest['c2w'], dtype=np.float64)
    scale = float(json.loads(inputs['a_run'].read_text())['world_scale'])
    H, W = manifest['height'], manifest['width']
    assert (H, W) == (480, 640)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    uv = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float64)
    grid = torch.tensor(get_homo_coordinate_map(H, W).reshape(-1, 2), dtype=torch.float32)
    cam = MonocularCameras.load_from_ckpt(torch.load(inputs['a_initial_camera'], map_location='cpu', weights_only=False))
    final_cam = MonocularCameras.load_from_ckpt(torch.load(inputs['a_final_camera'], map_location='cpu', weights_only=False))
    K_a = cam.K().detach().numpy()
    R_a, t_a = [x.detach().numpy() for x in cam.Rt_wc(0)]
    chosen = [(0, 0), (W-1, 0), (0, H-1), (W-1, H-1), (320, 240), (267, 369), (268, 370), (272, 363), (272, 364)]
    # Closed-form expected legacy error for H <= W, exact (unquantized) K.
    expected = np.stack([uv[:, 0] / (W - 1) + (K[0, 2] - W / 2) * (1 - H / W), uv[:, 1] / (H - 1)], axis=-1)
    rows = []
    depths = [0.5, 2.1282360553741455, 5.0]
    for depth_m in depths:
        dep = torch.full((len(uv),), depth_m * scale, dtype=torch.float32)
        legacy_native = cam.backproject(grid, dep)
        legacy_metric = legacy_native.detach().numpy().astype(np.float64) / scale
        legacy_uv = project_pixel(legacy_metric, K)
        legacy_error = legacy_uv - uv
        internal_normalized_error = cam.project(legacy_native).detach().numpy() - grid.numpy()
        exact_metric = backproject_pixel(uv, np.full(len(uv), depth_m), K)
        direct_uv = project_pixel(exact_metric, K)
        # Explicit native-unit conversion, then A's frozen camera-to-world and back.
        exact_native_f32 = torch.tensor(exact_metric * scale, dtype=torch.float32)
        world_native = cam.trans_pts_to_world(0, exact_native_f32)
        camera_native = cam.trans_pts_to_cam(0, world_native)
        roundtrip_uv = project_pixel(camera_native.detach().numpy().astype(np.float64) / scale, K)
        # The intended metric world counterpart uses the public c2w calibration.
        ideal_world_metric = exact_metric @ C[:3, :3].T + C[:3, 3]
        actual_world_metric = world_native.detach().numpy().astype(np.float64) / scale
        row = {
            'camera_z_m': depth_m,
            'pixel_count': int(len(uv)),
            'legacy_lift_to_calibrated_K_projection': stats(legacy_error),
            'legacy_internal_normalized_roundtrip_max_abs': float(np.abs(internal_normalized_error).max()),
            'legacy_measured_minus_closed_form_max_abs_px': float(np.abs(legacy_error - expected).max()),
            'direct_K_inverse_then_K_projection_float64': stats(direct_uv - uv),
            'direct_K_inverse_via_A_normalized_world_float32': stats(roundtrip_uv - uv),
            'A_native_world_divided_by_scale_vs_manifest_world_max_distance_m': float(np.linalg.norm(actual_world_metric - ideal_world_metric, axis=-1).max()),
            'sample_pixels': [{'uv': [u, v], 'legacy_error_xy_px': legacy_error[v * W + u].tolist(), 'closed_form_error_xy_px': expected[v * W + u].tolist()} for u, v in chosen],
        }
        assert np.abs(direct_uv - uv).max() < 1e-9
        assert np.abs(roundtrip_uv - uv).max() < 0.001
        assert np.abs(legacy_error - expected).max() < 0.001
        rows.append(row)

    seeds = np.load(inputs['actual_seeds'])
    d = torch.load(inputs['a_initial_dynamic'], map_location='cpu', weights_only=False)
    g = get_weights(d)
    dx, dr = warp(d, g, 0)
    scales_m = (d['min_scale'] + torch.sigmoid(d['_scaling']) * (d['max_scale'] - d['min_scale'])).numpy() / scale
    opacity = torch.sigmoid(d['_opacity']).numpy()
    checks = []
    for gs_id, query in [(86, [267, 369]), (138, [272, 363])]:
        assert int(seeds['raw_source_frame'][gs_id]) == 0
        pixel = np.asarray(seeds['raw_pixel_xy'][gs_id], dtype=np.float64)
        depth = float(seeds['raw_depth_m'][gs_id])
        corrected_cam = backproject_pixel(pixel[None], np.array([depth]), K)
        corrected_world = corrected_cam @ C[:3, :3].T + C[:3, 3]
        original_world = dx[gs_id:gs_id+1].numpy() / scale
        comparison = {}
        for label, centre in [('original_A', original_world), ('direct_K_inverse_centre_only', corrected_world)]:
            result = kernel(centre, dr[gs_id:gs_id+1].numpy(), scales_m[gs_id:gs_id+1], opacity[gs_id:gs_id+1], K, C, np.array(query))
            comparison[label] = {
                'world_centre_m': centre[0].tolist(),
                'exact_K_projected_uv': project_pixel((centre - C[:3, 3]) @ C[:3, :3], K)[0].tolist(),
                'native_kernel_projected_uv_with_denominator_epsilon': result['mean'][0].tolist(),
                'camera_z_m': float(result['z'][0]),
                'covariance_2d_px2': result['cov2d'][0].tolist(),
                'gaussian_exponent': float(result['power'][0]),
                'pre_compositing_alpha': float(result['alpha_before_threshold'][0]),
                'tile_eligible': bool(result['tile_eligible'][0]),
                'passes_native_alpha_cutoff': bool(result['tile_eligible'][0] and result['alpha_before_threshold'][0] >= 1/255),
            }
        checks.append({
            'initial_dynamic_gs_id': gs_id,
            'source_frame': 0,
            'actual_sample_pixel': pixel.tolist(),
            'actual_sample_camera_z_m': depth,
            'query_pixel': query,
            'opacity_unchanged': float(opacity[gs_id, 0]),
            'gaussian_scales_m_unchanged': scales_m[gs_id].tolist(),
            'sample_radius_native_unchanged': float(np.asarray(seeds['raw_radius_normalized'][gs_id]).reshape(-1)[0]),
            'native_alpha_cutoff': 1/255,
            'centre_displacement_m': float(np.linalg.norm(corrected_world - original_world)),
            'comparison': comparison,
            'scope': 'Single frozen initialization splat: replace its camera-z lift with inverse K; preserve orientation, scales/radius, opacity and query; recompute covariance. Birth frame is 0, so no cross-time node motion is involved. No model/checkpoint is written. This alpha is before compositing, not the full image or a trained result.',
        })
    after = {name: sha(path) for name, path in inputs.items()}
    assert before == after
    result = {
        'status': 'completed', 'device': 'cpu', 'reference_geometry_read': False,
        'size_hw': [H, W], 'K_calibrated': K.tolist(), 'A_K_float32': K_a.tolist(),
        'A_K_minus_calibrated': (K_a - K).tolist(),
        'A_initial_vs_final_camera_max_abs_parameter_difference': {name: float((value-final_cam.state_dict()[name]).abs().max()) if value.dtype != torch.bool else int((value != final_cam.state_dict()[name]).sum()) for name, value in cam.state_dict().items()},
        'world_scale_native_per_meter': scale,
        'A_camera_translation_native_divided_by_scale_vs_metric_max_abs_m': float(np.abs(t_a / scale - C[:3, 3]).max()),
        'A_camera_rotation_vs_manifest_max_abs': float(np.abs(R_a - C[:3, :3]).max()),
        'legacy_closed_form_error_xy_px': ['u/(W-1) + (cx-W/2)*(1-H/W)', 'v/(H-1)'],
        'closed_form_conditions': 'H <= W, integer pixel centres, z>0, ideal exact K; measured float32 A implementation checked separately.',
        'depth_cases': rows, 'seed_kernel_checks': checks,
        'recommended_interface': {
            'input': 'uv pixel centres in [0,W-1] x [0,H-1], camera-axis z in meters, calibrated K, metric c2w',
            'lift': 'r = inverse(K) @ [u,v,1]; x_cam_m = r * (z_m / r[2]); x_world_m = R_wc @ x_cam_m + t_wc_m',
            'native_units': 'x_cam_native = s*x_cam_m; t_wc_native = s*t_wc_m; x_world_native = R_wc @ x_cam_native + t_wc_native; export divides centre and scale by s',
            'projection': 'x_cam_m = inverse(c2w) @ x_world_m; uv = (K @ x_cam_m)[:2] / (K @ x_cam_m)[2]',
            'scope': 'Use one explicit pixel-space wrapper wherever integer pixels/depth enter geometry; inventory dense initialization and sparse track/node lift call sites before any controlled rerun. Keep renderer pixel centres unchanged. A uniform +/-0.5 pixel correction does not match these errors.',
        },
        'input_hashes_before_after_equal': before == after,
        'inputs': {name: {'path': str(path), 'sha256': before[name]} for name,path in inputs.items()},
        'script_sha256': sha(__file__), 'elapsed_cpu_wall_seconds': time.perf_counter() - started,
        'limits': 'Geometric interface and isolated initialization alpha acceptance only. No training, GPU work, whole-frame rendering, or evidence of optimized reconstruction improvement. The old normalized projector is internally inverse to its own lift; that alone does not establish agreement with calibrated K pixel coordinates.',
    }
    (OUT/'lift_projection_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status':result['status'],'wall_seconds':result['elapsed_cpu_wall_seconds'],'grid_first_case':rows[0],'seed_checks':checks},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    with torch.no_grad():
        main()
