#!/usr/bin/env python3
"""Render five frozen, evaluation-only BEHAVE camera-1 views after training.

--prepare-only rectifies the held-out RGB and validates calibration on CPU;
it does not import torch, read model checkpoints, or initialize CUDA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path('/home/cai_tianshun/Project/HOI')
EVENT = ROOT/'experiments/mosca_baseline_20260922'
NOMINAL_TIMES = [20., 21., 22., 25., 26.]
FIXED_INDICES = [16, 25, 31, 55, 65]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_record(out, record):
    (out/'heldout_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


def prepare(args, out):
    import cv2
    import numpy as np
    manifest_path = args.input.resolve()/'input_manifest.json'
    m = json.loads(manifest_path.read_text())
    reference_path = args.reference_manifest.resolve()
    ref = json.loads(reference_path.read_text())
    if m['role'] != 'input_only' or ref['role'] != 'evaluation_only':
        raise ValueError('Input/reference roles do not match the frozen protocol')
    if m['camera_id'] != 0 or not m['images_undistorted']:
        raise ValueError('Expected calibrated, rectified camera-0 training input')
    if ref['sequence'] != m['sequence'] or ref['world_frame'] != 'Kinect 1 color, metres':
        raise ValueError('Reference sequence or world frame mismatch')
    h, w = int(m['height']), int(m['width'])
    if (h, w) != (480, 640):
        raise ValueError('This fixed pilot uses 640 by 480 images')
    times = np.asarray(m['timestamp_seconds'], np.float64)
    indices = [int(np.argmin(abs(times-t))) for t in NOMINAL_TIMES]
    if indices != FIXED_INDICES:
        raise ValueError(f'Time map differs from the preregistered selection: {indices}')
    intrinsic_file = args.calibration_root.resolve()/'calibs/intrinsics/1/calibration.json'
    extrinsic_file = args.calibration_root.resolve()/'calibs/Date01/config/1/config.json'
    color = json.loads(intrinsic_file.read_text())['color']
    extrinsic = json.loads(extrinsic_file.read_text())
    raw_w, raw_h = int(color['width']), int(color['height'])
    k_raw = np.array([[color['fx'], 0, color['cx']], [0, color['fy'], color['cy']], [0, 0, 1]], np.float64)
    distortion = np.array([color[key] for key in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']], np.float64)
    np.testing.assert_allclose(color['opencv'], [color['fx'], color['fy'], color['cx'], color['cy'], *distortion])
    c2w = np.eye(4)
    c2w[:3, :3] = np.asarray(extrinsic['rotation']).reshape(3, 3)
    c2w[:3, 3] = extrinsic['translation']
    # Date01 BEHAVE world is Kinect 1 color, so this target pose must be identity.
    np.testing.assert_allclose(c2w, np.eye(4), rtol=0, atol=1e-10)
    k_full, roi = cv2.getOptimalNewCameraMatrix(k_raw, distortion, (raw_w, raw_h), 0, (raw_w, raw_h))
    scale = np.array([[w/raw_w, 0, (w/raw_w-1)/2], [0, h/raw_h, (h/raw_h-1)/2], [0, 0, 1]])
    k_out = scale @ k_full
    map_x, map_y = cv2.initUndistortRectifyMap(k_raw, distortion, None, k_out, (w, h), cv2.CV_32FC1)
    valid = (map_x >= 0) & (map_x < raw_w-1) & (map_y >= 0) & (map_y < raw_h-1)
    if valid.mean() < .995:
        raise ValueError(f'Unexpectedly large invalid remap area: {1-valid.mean()}')
    # Independent forward projection verifies pixel-centre resize and rational distortion order.
    yy, xx = np.meshgrid(np.arange(0, h, 20), np.arange(0, w, 20), indexing='ij')
    rays = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], -1) @ np.linalg.inv(k_out).T
    projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k_raw, distortion)
    sampled_map = np.stack([map_x[yy, xx], map_y[yy, xx]], -1).reshape(-1, 2)
    mapping_error = float(np.max(abs(projected[:, 0]-sampled_map)))
    if mapping_error > .001:
        raise ValueError(f'Rectification mapping check failed: {mapping_error}')
    for name in ['input_cam0', 'heldout_cam1_gt', 'heldout_cam1_render', 'heldout_cam1_alpha']:
        (out/name).mkdir()
    pairs = []
    for nominal, index in zip(NOMINAL_TIMES, indices):
        member = f"{m['sequence']}/t{nominal:08.3f}/k1.color.jpg"
        entries = [entry for entry in ref['records'] if entry['member'] == member]
        if len(entries) != 1:
            raise ValueError(f'Expected one held-out RGB record: {member}')
        entry = entries[0]
        source = Path(entry['path'])
        if sha(source) != entry['sha256']:
            raise ValueError(f'Held-out RGB hash mismatch: {source}')
        raw = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if raw is None or raw.shape[:2] != (raw_h, raw_w):
            raise ValueError(f'Unexpected held-out RGB resolution: {source}')
        filename = f't{nominal:08.3f}_input{index:05d}.png'
        target = out/'heldout_cam1_gt'/filename
        rectified = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if not cv2.imwrite(str(target), rectified):
            raise IOError(target)
        input_source = Path(m['frame_paths'][index])
        if sha(input_source) != m['frame_sha256'][index]:
            raise ValueError(f'Training RGB hash mismatch: {input_source}')
        input_target = out/'input_cam0'/filename
        shutil.copy2(input_source, input_target)
        pairs.append({'nominal_heldout_time_s': nominal, 'input_index': index,
                      'input_time_s': float(times[index]), 'input_minus_nominal_s': float(times[index]-nominal),
                      'reference_source': str(source), 'reference_source_sha256': entry['sha256'],
                      'input_source': str(input_source), 'input_source_sha256': sha(input_source),
                      'input_copy': str(input_target), 'heldout_gt': str(target), 'heldout_gt_sha256': sha(target),
                      'filename': filename})
    np.savez_compressed(out/'heldout_rectification.npz', K_raw=k_raw, distortion=distortion,
                        K_rectified_full=k_full, K=k_out, c2w=c2w, w2c=np.linalg.inv(c2w), valid=valid)
    record = {'status': 'prepared_cpu_only', 'script_sha256': sha(__file__),
              'information_boundary': 'Evaluation-only images are used only for frozen-checkpoint qualitative comparison; no fitting, tracking alignment, parameter adjustment or checkpoint selection.',
              'input_manifest': str(manifest_path), 'input_manifest_sha256': sha(manifest_path),
              'reference_manifest': str(reference_path), 'reference_manifest_sha256': sha(reference_path),
              'read_reference_assets': 'Only the five listed k1.color.jpg files. No fitted meshes, object poses, sensor depth or joints are read.',
              'calibration': {str(path): sha(path) for path in [intrinsic_file, extrinsic_file]},
              'camera_id': 1, 'world_frame': 'behave_world_k1_color',
              'K_raw': k_raw.tolist(), 'distortion_order': ['k1','k2','p1','p2','k3','k4','k5','k6'],
              'distortion': distortion.tolist(), 'K_rectified': k_out.tolist(), 'c2w': c2w.tolist(),
              'original_hw': [raw_h, raw_w], 'output_hw': [h, w],
              'rectification': 'Independent camera-1 OpenCV rational rectification, alpha=0; direct bilinear remap to pixel-centre-resized 640x480 pinhole grid.',
              'full_roi': list(map(int, roi)), 'valid_fraction': float(valid.mean()),
              'forward_projection_map_max_abs_error_px': mapping_error, 'pairs': pairs,
              'time_limit': 'Held-out times are nominal archive directory labels, not verified exact sensor timestamps. Nearest camera-0 state is rendered with the listed mismatch; no temporal interpolation or multi-camera synchronization claim.',
              'background_limit': 'One camera trained the scene. Camera-1 may reveal surfaces/background never observed in camera-0, so holes and unseen-region errors do not by themselves establish a failure in observed HOI geometry.',
              'interpretation': 'Qualitative held-out camera comparison, not a synchronized standard novel-view benchmark. Full images are retained; no fitted-reference alignment, cropping or per-image photometric adjustment.',
              'render_settings': {'GS_BACKEND': 'native_add3', 'background_color': [1., 1., 1.],
                                  'composition': 'Same full static+dynamic checkpoint jointly alpha-composited',
                                  'scale_factor': 1.0}}
    save_record(out, record)
    return m, k_out, np.linalg.inv(c2w), record


def write_montage(out, record, prepare_only=False):
    from PIL import Image, ImageDraw, ImageFont
    width, height, gap, band = 640, 480, 12, 43
    header, footer = 83, 100
    canvas = Image.new('RGB', (3*width+4*gap, header+5*(height+band+gap)+footer), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 19) if Path(font_path).exists() else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 17) if Path(font_path).exists() else ImageFont.load_default()
    draw.text((gap, 8), 'MoSca: frozen monocular reconstruction, held-out camera 1', fill='black', font=font)
    titles = ['Input camera 0 (nearest training frame)', 'Held-out camera 1: observed RGB (GT)',
              'Render pending (CPU preparation only)' if prepare_only else 'Held-out camera 1: same model render']
    for col, title in enumerate(titles):
        draw.text((gap+col*(width+gap), 47), title, fill='black', font=small)
    for row, pair in enumerate(record['pairs']):
        top = header+row*(height+band+gap)
        texts = [f"index {pair['input_index']}, t={pair['input_time_s']:.6f}s",
                 f"nominal t={pair['nominal_heldout_time_s']:.3f}s (sensor time unavailable)",
                 f"render t={pair['input_time_s']:.6f}s; delta={pair['input_minus_nominal_s']*1000:+.3f}ms"]
        paths = [pair['input_copy'], pair['heldout_gt'], None if prepare_only else pair['render']]
        for col, (text, path) in enumerate(zip(texts, paths)):
            left = gap+col*(width+gap)
            draw.text((left, top+10), text, fill='black', font=small)
            if path is None:
                im = Image.new('RGB', (width,height), '#eeeeee')
                ImageDraw.Draw(im).text((70,210), 'NOT RENDERED - CPU preparation only', fill='black', font=font)
            else:
                im = Image.open(path).convert('RGB')
            if im.size != (width, height): raise ValueError(f'Unexpected montage size: {path}')
            canvas.paste(im, (left, top+band))
    bottom = canvas.height-footer
    notes = ['Single-view input does not observe all camera-1 background/surfaces; render gaps can include unseen regions.',
             'Archive times are nominal; multi-camera synchronization is unverified. Listed deltas are only nominal offsets.',
             'Frozen qualitative diagnostic: no reference-based alignment, tuning, checkpoint selection or image correction.']
    for i, note in enumerate(notes): draw.text((gap, bottom+10+i*26), note, fill='black', font=small)
    canvas.save(out/('preparation_preview_5x3.png' if prepare_only else 'heldout_comparison_5x3.png'))


def render_checkpoint(args, out, manifest, k_target, w2c_target, record):
    import numpy as np
    import cv2
    import torch
    from omegaconf import OmegaConf
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Bind an explicitly allocated GPU with CUDA_VISIBLE_DEVICES')
    os.environ['GS_BACKEND'] = 'native_add3'
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from lib_moca.camera import MonocularCameras
    from lib_mosca.static_gs import StaticGaussian
    from lib_mosca.dynamic_gs import DynSCFGaussian
    from lib_render.render_helper import render
    ckpt = args.checkpoint_dir.resolve()
    cfg = OmegaConf.load(ckpt/'config.yaml')
    training_record = json.loads((ckpt/'run.json').read_text())
    world_scale = float(training_record['world_scale'])
    if cfg.mode != 'behave_calibrated_pilot' or not training_record['exact_principal_point']:
        raise ValueError('Expected controlled exact-camera adapter')
    files = {key: ckpt/name for key, name in {'camera': 'photometric_cam.pth',
              'static': 'photometric_s_model_native_add3.pth', 'dynamic': 'photometric_d_model_native_add3.pth',
              'bundle': 'bundle/bundle.pth', 'config': 'config.yaml'}.items()}
    record.update(status='loading_checkpoint', checkpoint_dir=str(ckpt),
                  checkpoint_sha256={key: sha(path) for key,path in files.items()},
                  cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                  mosca_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
    save_record(out, record)
    device = torch.device(args.device)
    if device.type != 'cuda': raise ValueError('Renderer requires CUDA; use --prepare-only for CPU checks')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    bundle = torch.load(files['bundle'], map_location='cpu', weights_only=False)
    if not torch.allclose(bundle['dep_scale'], torch.ones_like(bundle['dep_scale'])):
        raise ValueError('Unexpected fitted depth scaling')
    del bundle
    cameras = MonocularCameras.load_from_ckpt(torch.load(files['camera'], map_location=device, weights_only=False)).to(device).eval()
    static = StaticGaussian.load_from_ckpt(torch.load(files['static'], map_location=device, weights_only=False), device=device).to(device).eval()
    dynamic = DynSCFGaussian.load_from_ckpt(torch.load(files['dynamic'], map_location=device, weights_only=False), device=device).to(device).eval()
    if int(cameras.T) != len(manifest['frame_paths']) or int(dynamic.T) != int(cameras.T):
        raise ValueError('Model times differ from the frozen input table')
    h, w = record['output_hw']
    if (int(cameras.default_H), int(cameras.default_W)) != (h,w): raise ValueError('Camera size mismatch')
    training_k = torch.tensor(manifest['K'], dtype=torch.float32, device=device)
    training_c2w = torch.tensor(manifest['c2w'], dtype=torch.float32, device=device)
    k_error = float((cameras.K()-training_k).abs().max())
    normalized_c2w = training_c2w.clone(); normalized_c2w[:3,3] *= world_scale
    pose_error = max(float((cameras.T_wc(i)-normalized_c2w).abs().max()) for i in range(int(cameras.T)))
    if k_error > 1e-3 or pose_error > 1e-5:
        raise ValueError(f'Training calibration changed: K {k_error}, pose {pose_error}')
    from validation_utils import expose_metric_gaussians
    expose_metric_gaussians(static, dynamic, world_scale)
    record['exact_principal_point'] = True
    record['coordinate_checks'] = {'scale_nw': world_scale, 'bundle_dep_scale_all_one': True,
                                   'training_K_max_abs_error': k_error, 'training_c2w_max_abs_error': pose_error,
                                   'target_camera1_c2w_identity': True, 'global_reference_alignment': 'none'}
    k = torch.tensor(k_target, dtype=torch.float32, device=device)
    w2c = torch.tensor(w2c_target, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        s_gs = static()
        for pair in record['pairs']:
            result = render([s_gs, dynamic(pair['input_index'])], h, w, k, w2c, bg_color=[1.,1.,1.])
            rgb = result['rgb'].permute(1,2,0).detach().cpu().numpy()
            alpha = result['alpha'][0].detach().cpu().numpy()
            if not np.isfinite(rgb).all() or not np.isfinite(alpha).all(): raise ValueError('Nonfinite render')
            path = out/'heldout_cam1_render'/pair['filename']
            alpha_path = out/'heldout_cam1_alpha'/pair['filename']
            if not cv2.imwrite(str(path), cv2.cvtColor(np.round(np.clip(rgb,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR)):
                raise IOError(path)
            if not cv2.imwrite(str(alpha_path), np.round(np.clip(alpha,0,1)*255).astype(np.uint8)): raise IOError(alpha_path)
            pair.update(render=str(path), render_sha256=sha(path), alpha=str(alpha_path),
                        alpha_sha256=sha(alpha_path), fraction_alpha_above_half=float((alpha>.5).mean()))
    torch.cuda.synchronize(device)
    record.update(render_seconds=time.perf_counter()-start, gpu_name=torch.cuda.get_device_name(device),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  alpha_note='Model opacity only; no visibility truth or unseen-surface mask is claimed.')
    write_montage(out, record)
    record.update(status='completed', comparison_png=str(out/'heldout_comparison_5x3.png'),
                  comparison_png_sha256=sha(out/'heldout_comparison_5x3.png'))
    save_record(out, record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=EVENT/'common_input')
    p.add_argument('--reference-manifest', type=Path, default=EVENT/'data/behave_event_reference_manifest.json')
    p.add_argument('--calibration-root', type=Path, default=ROOT/'data/BEHAVE/calibration')
    p.add_argument('--checkpoint-dir', type=Path)
    p.add_argument('--repo', type=Path, default=Path('/home/cai_tianshun/Project/HOI/experiments/mosca_validation_20260923/code/MoSca'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if not args.prepare_only and args.checkpoint_dir is None: p.error('--checkpoint-dir is required to render')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    record = {'status': 'preparing', 'script_sha256': sha(__file__)}
    try:
        manifest, k, w2c, record = prepare(args, out)
        if not args.prepare_only:
            render_checkpoint(args, out, manifest, k, w2c, record)
        else:
            write_montage(out, record, prepare_only=True)
            record['cpu_preview_png'] = str(out/'preparation_preview_5x3.png')
        record['wall_seconds'] = time.perf_counter()-start
        save_record(out, record)
        print(json.dumps({'status': record['status'], 'output': str(out),
                          'pairs': [{k:p[k] for k in ['nominal_heldout_time_s','input_index','input_time_s','input_minus_nominal_s']} for p in record['pairs']]}, indent=2))
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        save_record(out, record)
        raise


if __name__ == '__main__':
    main()
