#!/usr/bin/env python3
"""Export a calibrated MoSca checkpoint without opening evaluation reference data.

Run in the original MoSca environment after training. The native_add3 renderer
jointly composites static and dynamic Gaussians. Fixed image queries use source
rendering weights to carry destination Gaussian centres, following test_pck's
rendered correspondence mechanism with explicit alpha normalisation for XYZ.
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


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_video(frame_paths, times, target):
    """Encode physical frame durations; no invented regular frame-rate timing."""
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError('ffmpeg or the bundled imageio_ffmpeg binary is required') from exc
    concat = target.with_suffix('.ffconcat')
    gaps = [float(times[i+1] - times[i]) for i in range(len(times)-1)]
    gaps.append(gaps[-1] if gaps else .1)
    with open(concat, 'w') as f:
        f.write('ffconcat version 1.0\n')
        for path, duration in zip(frame_paths, gaps):
            escaped = str(path.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\noption framerate 1000\nduration {duration:.9f}\n")
        escaped = str(frame_paths[-1].resolve()).replace("'", "'\\''")
        f.write(f"file '{escaped}'\noption framerate 1000\n")
    result = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                             '-f', 'concat', '-safe', '0', '-i', str(concat),
                             '-vsync', 'vfr', '-c:v', 'libx264', '-crf', '20',
                             '-pix_fmt', 'yuv420p', '-video_track_timescale', '1000000', str(target)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'ffmpeg failed: {result.stderr[-3000:]}')
    return {'path': str(target), 'sha256': sha(target),
            'timing': 'variable frame duration from actual timestamps, encoded to millisecond precision; last frame held for preceding interval',
            'first_timestamp_s': float(times[0]), 'last_timestamp_s': float(times[-1]),
            'last_frame_hold_s': gaps[-1]}


def write_ply(path, xyz, rgb, component, semantic):
    import numpy as np
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'),
             ('green', 'u1'), ('blue', 'u1'), ('component', 'u1'), ('semantic', '<i4')]
    rows = np.empty(len(xyz), dtype=dtype)
    for i, key in enumerate(['x', 'y', 'z']): rows[key] = xyz[:, i]
    for i, key in enumerate(['red', 'green', 'blue']): rows[key] = rgb[:, i]
    rows['component'] = component
    rows['semantic'] = semantic
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment Gaussian centers; not a reconstructed surface mesh\n'
              'comment component 0 static 1 dynamic; semantic -1 unknown 0 background 1 person 2 object\n'
              f'element vertex {len(xyz)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'property uchar component\nproperty int semantic\nend_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        rows.tofile(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Directory with processed input_manifest.json')
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('/home/cai_tianshun/Project/HOI/experiments/mosca_validation_20260923/code/MoSca'))
    parser.add_argument('--device', default='cuda:0', help='Logical CUDA device; caller binds physical GPU')
    parser.add_argument('--queries', type=Path, help='Default: input/queries_first_frame.json; manual evaluation entries only')
    parser.add_argument('--segmentation', type=Path, help='Optional legal RGB SAM2 segmentation.npz; postprocessing only')
    parser.add_argument('--semantic-vote-stride', type=int, default=10)
    parser.add_argument('--minimum-label-votes', type=int, default=2)
    parser.add_argument('--source-alpha-min', type=float, default=.05)
    parser.add_argument('--visibility-relative-depth-tolerance', type=float, default=.03)
    parser.add_argument('--no-videos', action='store_true', help='Keep all PNG/NPZ exports but skip encoding')
    args = parser.parse_args()
    if not 0 < args.source_alpha_min <= 1:
        parser.error('source-alpha-min must be in (0,1]')
    if args.semantic_vote_stride < 1 or args.minimum_label_votes < 1:
        parser.error('Voting stride/count must be positive')

    # Imports remain after argument parsing so --help and py_compile require no CUDA.
    import numpy as np
    import cv2
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    os.environ['GS_BACKEND'] = 'native_add3'
    sys.path.insert(0, str(args.repo.resolve()))
    from lib_moca.camera import MonocularCameras
    from lib_mosca.static_gs import StaticGaussian
    from lib_mosca.dynamic_gs import DynSCFGaussian
    from lib_render.render_helper import render

    ws = args.input.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    out = args.output.resolve()
    manifest_path = ws / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cfg = OmegaConf.load(checkpoint_dir / 'config.yaml')
    training_record = json.loads((checkpoint_dir/'run.json').read_text())
    world_scale = float(training_record['world_scale'])
    if cfg.mode != 'behave_calibrated_pilot' or world_scale <= 0 or not training_record['exact_principal_point']:
        raise ValueError('Expected this controlled protocol adapter')
    if manifest['role'] != 'input_only' or not manifest['images_undistorted']:
        raise ValueError('Expected legal processed pinhole RGB input')
    H, W = int(manifest['height']), int(manifest['width'])
    times = np.asarray(manifest['timestamp_seconds'], dtype=np.float64)
    if len(times) != len(manifest['frame_paths']) or not np.all(np.diff(times) > 0):
        raise ValueError('Input frame time table is inconsistent')
    for path, expected in zip(manifest['frame_paths'], manifest['frame_sha256']):
        if sha(path) != expected: raise ValueError(f'Input hash mismatch: {path}')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; preserve earlier exports')
    out.mkdir(parents=True, exist_ok=True)
    for sub in ['rgb', 'depth_viz', 'depth', 'query_overlay', 'keyframes']:
        (out / sub).mkdir()
    checkpoints = {name: checkpoint_dir / file for name, file in {
        'camera': 'photometric_cam.pth', 'static': 'photometric_s_model_native_add3.pth',
        'dynamic': 'photometric_d_model_native_add3.pth'}.items()}
    for path in checkpoints.values():
        if not path.is_file(): raise FileNotFoundError(path)
    query_path = (args.queries or ws / 'queries_first_frame.json').resolve()
    q = json.loads(query_path.read_text())
    n = int(q['manual_evaluation_count'])
    uv_np = np.asarray(q['points'][:n], dtype=np.float32)
    entities = np.asarray(q['entity'][:n], dtype=str)
    query_ids = np.asarray(q['point_ids'][:n], dtype=str)
    source_t = int(q['query_frame'])
    if not len(uv_np) or uv_np.shape != (n, 2) or len(query_ids) != n:
        raise ValueError('Malformed frozen RGB query list')
    if not (0 <= source_t < len(times)) or np.any(uv_np < 0) or np.any(uv_np >= [W, H]):
        raise ValueError('Query is outside the declared frame/time')
    if abs(float(q['timestamp_seconds']) - times[source_t]) > 1e-6:
        raise ValueError('Frozen query time does not match input manifest')
    record = {'status': 'running', 'checkpoint_dir': str(checkpoint_dir),
              'input_manifest': str(manifest_path), 'input_manifest_sha256': sha(manifest_path),
              'query_file': str(query_path), 'query_sha256': sha(query_path),
              'checkpoint_sha256': {k: sha(v) for k, v in checkpoints.items()},
              'script_sha256': sha(__file__), 'GS_BACKEND': 'native_add3',
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'device': args.device, 'query_count': n,
              'coordinate_system': manifest['coordinate_system'], 'units': 'estimated metric metres',
              'global_alignment_to_reference': 'none',
              'query_mechanism': 'Source-frame joint static+dynamic alpha-compositing weights transport destination Gaussian centers through add_buffer; sampled numerator divided by sampled source alpha.',
              'source_alpha_min': args.source_alpha_min,
              'query_limitation': 'A fixed weighted Gaussian-center trajectory is not guaranteed to be one physical material surface point. Source mixtures include static background when the reconstruction leaks; no nearest-node or per-frame reference reassignment.',
              'visibility_note': 'Predicted target depth consistency is diagnostic only, never independent occlusion ground truth.',
              'depth_definition': 'native_add3 dep is already alpha-normalized expected camera-space z. Export does not divide it a second time.',
              'semantic_note': 'Optional SAM2 post-hoc projection vote labels, not native MoSca semantics and not training modifications.',
              'renderer_source_sha256': {str(p.relative_to(args.repo)): sha(p) for p in [
                  args.repo / 'mosca_evaluate.py', args.repo / 'lib_render/render_helper.py',
                  args.repo / 'lib_render/gauspl_renderer_native_add3.py']}}
    start = time.perf_counter()
    def save_record(): (out / 'export_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    save_record()
    try:
        device = torch.device(args.device)
        if device.type != 'cuda': raise ValueError('Native renderer requires CUDA; use CPU only for compilation/help')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        # Only self-generated, local adapter checkpoints are loaded, not external dataset pickles.
        bundle = torch.load(checkpoint_dir / 'bundle/bundle.pth', map_location='cpu', weights_only=False)
        if not torch.allclose(bundle['dep_scale'], torch.ones_like(bundle['dep_scale'])):
            raise ValueError('Unexpected per-frame scale; cannot claim original metric coordinate system')
        cams = MonocularCameras.load_from_ckpt(torch.load(checkpoints['camera'], map_location=device, weights_only=False)).to(device).eval()
        static = StaticGaussian.load_from_ckpt(torch.load(checkpoints['static'], map_location=device, weights_only=False), device=device).to(device).eval()
        dynamic = DynSCFGaussian.load_from_ckpt(torch.load(checkpoints['dynamic'], map_location=device, weights_only=False), device=device).to(device).eval()
        if int(cams.T) != len(times) or int(dynamic.T) != len(times):
            raise ValueError('Checkpoint and input frame counts differ; explicit index map required')
        if (int(cams.default_H), int(cams.default_W)) != (H, W): raise ValueError('Checkpoint dimensions differ')
        K = torch.as_tensor(manifest['K'], dtype=torch.float32, device=device)
        c2w = torch.as_tensor(manifest['c2w'], dtype=torch.float32, device=device)
        k_error = float((cams.K() - K).abs().max())
        normalized_c2w = c2w.clone(); normalized_c2w[:3, 3] *= world_scale
        camera_error = max(float((cams.T_wc(t) - normalized_c2w).abs().max()) for t in range(len(times)))
        if k_error > 1e-3 or camera_error > 1e-5:
            raise ValueError(f'Known camera changed: K {k_error}, pose {camera_error}')
        from validation_utils import expose_metric_gaussians
        expose_metric_gaussians(static, dynamic, world_scale)
        record['world_scale'] = world_scale
        record['global_transform_fitted_on_evaluation'] = False
        record['exact_principal_point'] = True
        record['coordinate_checks'] = {'scale_nw': world_scale, 'bundle_dep_scale_all_one': True,
                                       'known_K_max_abs_error': k_error, 'known_c2w_max_abs_error': camera_error,
                                       'source': 'single input-derived world scale; cameras verified in normalized coordinates; Gaussian centers and scales divided by world_scale for metric rendering/export'}
        w2c = torch.linalg.inv(c2w)
        uv = torch.from_numpy(uv_np).to(device)
        grid = torch.stack([2*uv[:, 0]/(W-1)-1, 2*uv[:, 1]/(H-1)-1], -1)[None, None]
        def sample(image):
            return F.grid_sample(image[None], grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0, :, 0].T
        def project(points):
            cam = points @ w2c[:3, :3].T + w2c[:3, 3]
            pix = cam @ K.T
            return pix[:, :2] / pix[:, 2:3].clamp_min(1e-8), cam[:, 2]
        def cpu(tensor): return tensor.detach().cpu().numpy()
        with torch.no_grad():
            s_gs = static()
            d_source = dynamic(source_t)
            source_gs = [s_gs, d_source]
            static_count = len(s_gs[0]); dynamic_count = len(d_source[0])
            total_count = static_count + dynamic_count
            component = np.r_[np.zeros(static_count, np.uint8), np.ones(dynamic_count, np.uint8)]
            source_aux = torch.zeros((total_count, 3), device=device)
            source_aux[:static_count, 0] = 1
            source_aux[static_count:, 1] = 1
            source_render = render(source_gs, H, W, K, w2c, bg_color=[0., 0., 0.], add_buffer=source_aux)
            source_alpha = sample(source_render['alpha'])[:, 0]
            source_fraction = sample(source_render['buf']) / source_alpha[:, None].clamp_min(1e-8)
            record.update(static_gaussians=static_count, dynamic_gaussians=dynamic_count,
                          query_source_alpha=cpu(source_alpha).tolist(),
                          query_source_dynamic_fraction=cpu(source_fraction[:, 1]).tolist())
            labels = np.full(total_count, -1, np.int32)
            votes = np.zeros((total_count, 3), np.uint16)
            if args.segmentation:
                with np.load(args.segmentation, allow_pickle=False) as z:
                    masks = z['entity_labels']
                    role = str(z['role'].item())
                if masks.shape != (len(times), H, W) or role != 'estimated_segmentation_prior_not_reference_visibility':
                    raise ValueError('Expected legal same-input SAM2 masks')
                if not np.isin(masks, [0, 1, 2]).all(): raise ValueError('Unknown segmentation labels')
                for t in range(0, len(times), args.semantic_vote_stride):
                    d_gs = dynamic(t)
                    rd = render([s_gs, d_gs], H, W, K, w2c)
                    xyz = torch.cat([s_gs[0], d_gs[0]])
                    pix, zcoord = project(xyz)
                    xy = torch.round(pix).long()
                    inside = (zcoord > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
                    idx = torch.where(inside)[0]
                    xx, yy = xy[idx, 0], xy[idx, 1]
                    zrender = rd['dep'][0, yy, xx]
                    visible = (rd['alpha'][0, yy, xx] > .3) & ((zcoord[idx] - zrender).abs() <= args.visibility_relative_depth_tolerance * zrender.abs().clamp_min(1e-6))
                    visible &= rd['visibility_filter'][idx]
                    opacity = torch.cat([s_gs[3], d_gs[3]]).reshape(-1)
                    visible &= opacity[idx] > 1e-5
                    idx = idx[visible]
                    xy_valid = cpu(xy[idx])
                    semantic = masks[t, xy_valid[:, 1], xy_valid[:, 0]].astype(int)
                    votes[cpu(idx), semantic] += 1
                winners = votes.argmax(1)
                total_votes = votes.sum(1)
                confident = (total_votes >= args.minimum_label_votes) & (votes.max(1) > total_votes * .5)
                labels[confident] = winners[confident]
                np.savez_compressed(out / 'posthoc_semantic_labels.npz', labels=labels, votes=votes,
                                    component=component, role=np.array('SAM2_posthoc_projection_votes_not_native_semantics'))
                (out / 'semantic').mkdir()
                record['posthoc_semantics'] = {'mask_path': str(args.segmentation.resolve()), 'mask_sha256': sha(args.segmentation),
                    'vote_stride': args.semantic_vote_stride, 'minimum_votes': args.minimum_label_votes,
                    'visibility_relative_depth_tolerance': args.visibility_relative_depth_tolerance,
                    'counts': {str(k): int((labels == k).sum()) for k in [-1, 0, 1, 2]},
                    'limitation': 'Center-projection labels gated by model expected depth can be wrong near overlapping surfaces; unknown retained. Not reference labels.'}
            palette = np.array([[.5, .5, .5], [.12, .8, .25], [1., .4, .05], [.6, .15, .8]], np.float32)
            semantic_colors = torch.from_numpy(palette[np.where(labels < 0, 3, labels)]).to(device)
            event = manifest.get('event_selection', {}).get('first_event', {})
            key_indices = {0, len(times)-1}
            for bounds in event.values():
                if isinstance(bounds, list) and len(bounds) == 2:
                    key_indices.add(int(np.argmin(abs(times - np.mean(bounds)))))
            if len(key_indices) == 2: key_indices.add(len(times)//2)
            predicted = np.full((len(times), n, 3), np.nan, np.float32)
            predicted_uv = np.full((len(times), n, 2), np.nan, np.float32)
            predicted_valid = np.zeros((len(times), n), bool)
            predicted_visible = np.zeros((len(times), n), bool)
            rgb_paths, depth_paths, query_paths, semantic_paths = [], [], [], []
            # Shared visualization range; actual metric depths are saved without clipping.
            source_depth = cpu(source_render['dep'][0]); source_opaque = cpu(source_render['alpha'][0]) > .5
            if not source_opaque.any(): raise ValueError('No sufficiently opaque source pixels for export')
            dmin, dmax = np.percentile(source_depth[source_opaque], [2, 98]).tolist()
            record['depth_visualization_range_m'] = [dmin, dmax]
            del source_render
            for t in range(len(times)):
                d_gs = dynamic(t)
                if len(d_gs[0]) != dynamic_count: raise ValueError('Gaussian identity count changed during frozen checkpoint export')
                target_xyz = torch.cat([s_gs[0], d_gs[0]])
                transported = render(source_gs, H, W, K, w2c, bg_color=[0., 0., 0.], add_buffer=target_xyz)
                query_xyz = sample(transported['buf']) / source_alpha[:, None].clamp_min(1e-8)
                valid = (source_alpha >= args.source_alpha_min) & torch.isfinite(query_xyz).all(1)
                predicted[t] = cpu(query_xyz)
                predicted_valid[t] = cpu(valid)
                target_uv, target_z = project(query_xyz)
                predicted_uv[t] = cpu(target_uv)
                rd = render([s_gs, d_gs], H, W, K, w2c)
                rgb = np.clip(cpu(rd['rgb'].permute(1, 2, 0)), 0, 1)
                depth = cpu(rd['dep'][0]).astype(np.float32)
                alpha = cpu(rd['alpha'][0]).astype(np.float32)
                np.savez_compressed(out / 'depth' / f'{t:05d}.npz', depth_camera_z_m=depth, alpha=alpha,
                                    frame_index=t, timestamp_seconds=times[t], unit=np.array('estimated_metric_metre'))
                bgr = (rgb[:, :, ::-1]*255).round().astype(np.uint8)
                rgb_path = out / 'rgb' / f'{t:05d}.png'; cv2.imwrite(str(rgb_path), bgr); rgb_paths.append(rgb_path)
                depth_u8 = np.clip((depth-dmin)/max(dmax-dmin, 1e-6)*255, 0, 255).astype(np.uint8)
                depth_bgr = cv2.applyColorMap(255-depth_u8, cv2.COLORMAP_TURBO)
                depth_bgr[(alpha < args.source_alpha_min) | ~np.isfinite(depth)] = 0
                depth_path = out / 'depth_viz' / f'{t:05d}.png'; cv2.imwrite(str(depth_path), depth_bgr); depth_paths.append(depth_path)
                target_grid = torch.stack([2*target_uv[:, 0]/(W-1)-1, 2*target_uv[:, 1]/(H-1)-1], -1)[None, None]
                sampled_depth = F.grid_sample(rd['dep'][None], target_grid, align_corners=True)[0, 0, 0]
                sampled_alpha = F.grid_sample(rd['alpha'][None], target_grid, align_corners=True)[0, 0, 0]
                inside = (target_z > 0) & (target_uv[:, 0] >= 0) & (target_uv[:, 0] < W) & (target_uv[:, 1] >= 0) & (target_uv[:, 1] < H)
                vis = valid & inside & (sampled_alpha > .3) & ((target_z-sampled_depth).abs() <= args.visibility_relative_depth_tolerance*sampled_depth.abs().clamp_min(1e-6))
                predicted_visible[t] = cpu(vis)
                overlay = cv2.imread(manifest['frame_paths'][t])
                for qi, point in enumerate(predicted_uv[t]):
                    if not np.isfinite(point).all() or not predicted_valid[t, qi]: continue
                    x, y = np.rint(point).astype(int)
                    if 0 <= x < W and 0 <= y < H:
                        color = (0, 220, 40) if entities[qi] == 'hand' else (0, 120, 255)
                        cv2.circle(overlay, (x, y), 4, color, -1 if predicted_visible[t, qi] else 1)
                        cv2.putText(overlay, str(qi), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
                cv2.putText(overlay, f't={times[t]:.3f}s; hollow=model depth-occluded', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                qp = out / 'query_overlay' / f'{t:05d}.png'; cv2.imwrite(str(qp), overlay); query_paths.append(qp)
                if args.segmentation:
                    sem = render([s_gs, d_gs], H, W, K, w2c, colors_precomp=semantic_colors)
                    sem_bgr = (np.clip(cpu(sem['rgb'].permute(1, 2, 0)), 0, 1)[:, :, ::-1]*255).round().astype(np.uint8)
                    sp = out / 'semantic' / f'{t:05d}.png'; cv2.imwrite(str(sp), sem_bgr); semantic_paths.append(sp)
                if t in key_indices:
                    combined = [torch.cat([sg, dg]) for sg, dg in zip(s_gs, d_gs)]
                    xyz, rotation, scales, opacity, sh = [cpu(x) for x in combined]
                    rgb_dc = np.clip(sh[:, :3]*.28209479177387814 + .5, 0, 1)
                    np.savez_compressed(out / 'keyframes' / f'{t:05d}.npz', xyz_world=xyz, rotation_frame=rotation,
                                        scales=scales, opacity=opacity, spherical_harmonics=sh, component=component,
                                        semantic_posthoc=labels, frame_index=t, timestamp_seconds=times[t],
                                        coordinate_system=np.array(manifest['coordinate_system']))
                    write_ply(out / 'keyframes' / f'{t:05d}.ply', xyz, (rgb_dc*255).round().astype(np.uint8), component, labels)
                print(f'export frame {t+1}/{len(times)}; valid queries {int(valid.sum())}/{n}', flush=True)
            np.savez_compressed(out / 'query_trajectories.npz', predicted=predicted, predicted_valid_mask=predicted_valid,
                                predicted_uv=predicted_uv, predicted_visibility_depth=predicted_visible,
                                frame_times=times, entity=entities, query_id=query_ids, query_uv=uv_np,
                                source_alpha=cpu(source_alpha), source_dynamic_fraction=cpu(source_fraction[:, 1]),
                                coordinate_system=np.array(manifest['coordinate_system']),
                                coordinate_frame=np.array('behave_world_k1_color'), units=np.array('m'),
                                role=np.array('prediction_only_no_reference_alignment'))
            # Node states are an additional diagnostic, never substituted for fixed image queries.
            scf = dynamic.scf
            np.savez_compressed(out / 'scaffold_nodes.npz', node_xyz=cpu(scf._node_xyz) / world_scale,
                                node_rotation=cpu(scf._node_rotation), node_certain_prior=cpu(scf._node_certain),
                                node_grouping=cpu(scf._node_grouping), source_frame_index=cpu(scf._t_list),
                                input_frame_times=times, role=np.array('model_node_diagnostic_not_material_point_reference'))
            record['keyframe_indices'] = sorted(key_indices)
            record['trajectory_sha256'] = sha(out / 'query_trajectories.npz')
            record['videos'] = []
            if not args.no_videos:
                for name, paths in [('combined_rgb', rgb_paths), ('combined_depth_visualization', depth_paths),
                                    ('fixed_query_overlay', query_paths), ('posthoc_semantics', semantic_paths)]:
                    if paths: record['videos'].append(write_video(paths, times, out / f'{name}.mp4'))
            record['status'] = 'completed'
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        record['wall_seconds'] = time.perf_counter()-start
        if torch.cuda.is_initialized():
            record['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
            record['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved()
        save_record()
    print(str(out / 'export_manifest.json'), flush=True)


if __name__ == '__main__':
    main()
