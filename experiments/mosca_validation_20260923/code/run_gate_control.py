#!/usr/bin/env python3
"""MoSca with fixed supplied cameras and RGB-only foreground masks.

The adapter replaces unknown-camera bundle adjustment, not the dynamic solver.
It maps a binary foreground prior to MoSca's legacy `epi` classification field;
these values are explicitly NOT measured epipolar errors. Camera learning rates
are zero at every photometric stage. No evaluation-only asset is opened.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback

ROOT = Path('/home/cai_tianshun/Project/HOI')
REPO = Path('/home/cai_tianshun/Project/HOI/experiments/mosca_validation_20260923/code/MoSca')
os.environ['GS_BACKEND'] = 'native_add3'
# Upstream saves local module configuration alongside tensor state dictionaries.
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from omegaconf import OmegaConf


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--segmentation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--normalize-world', action='store_true')
    p.add_argument('--disable-dynamic-gate', action='store_true')
    p.add_argument('--photo-steps', type=int, default=8000)
    p.add_argument('--geo-steps', type=int, default=4000)
    p.add_argument('--export-script', type=Path)
    p.add_argument('--resume-scaffold-dir', type=Path, help='Reuse a completed geometry stage with identical input and geometric config')
    a = p.parse_args()
    ws, output = a.input.resolve(), a.output.resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('Use an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, output/'runner_snapshot.py')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output/'training.log')], force=True)
    m = json.loads((ws/'input_manifest.json').read_text())
    assert m['role'] == 'input_only' and m['images_undistorted']
    assert len(m['timestamp_seconds']) == len(m['frame_paths'])
    cfg = OmegaConf.load(a.config or REPO/'profile/demo/demo_fit.yaml')
    cfg.mode = 'behave_calibrated_pilot'
    cfg.validation_disable_dynamic_gate = a.disable_dynamic_gate
    cfg.depth_dirname = 'unidepth_depth'; cfg.tap_mode = 'cotracker'
    cfg.dep_median = 1.0 if a.normalize_world else -1.0; cfg.iso_focal = False
    cfg.epi_th = 0.5; cfg.ba_epi_th = 0.5
    cfg.photo_total_steps = a.photo_steps; cfg.geo_mosca_steps = a.geo_steps
    cfg.photo_optim_cam_after_steps = a.photo_steps + 1
    cfg.photo_warm_optim_cam_after_steps = 10**9
    OmegaConf.save(cfg, output/'config.yaml')
    segmentation = np.load(a.segmentation)['foreground'].astype(np.float32)
    assert segmentation.shape == (len(m['frame_paths']), m['height'], m['width'])
    for source, expected in zip(m['frame_paths'], m['frame_sha256']):
        assert sha(source) == expected
    for i in range(len(m['frame_paths'])):
        assert (ws/'unidepth_depth'/f'{i:05d}.npz').exists()
    import mosca_reconstruct as mr
    from lib_prior.prior_loading import Saved2D
    from lib_moca.camera import MonocularCameras
    OriginalOptimCFG = mr.OptimCFG
    world_scale = 1.0
    if a.normalize_world:
        scale_data = Saved2D(str(ws)).load_dep('unidepth_depth', cfg.depth_boundary_th).normalize_depth(1.0)
        world_scale = float(scale_data.scale_nw)
        del scale_data
    # Give dynamic initialization and photo sampling independent RNG streams,
    # so a changed static candidate pool cannot silently change dynamic seeds.
    original_dynamic = mr.DynReconstructionSolver.get_dynamic_model
    original_fit = mr.DynReconstructionSolver.photometric_fit
    def seeded_dynamic(self, *args, **kwargs):
        mr.seed_everything(mr.SEED)
        model = original_dynamic(self, *args, **kwargs)
        if a.disable_dynamic_gate:
            model.dyn_o_flag.fill_(False)
        return model
    def recorded_fit(self, *args, **kwargs):
        from validation_utils import tensor_state_hash
        state = kwargs['d_model'].state_dict()
        if a.resume_scaffold_dir:
            previous_initial = torch.load(a.resume_scaffold_dir/'initial_dynamic.pth', map_location='cpu', weights_only=False)
            tensor_differences = []
            for key in set(state)|set(previous_initial):
                old, new = previous_initial.get(key), state.get(key)
                if torch.is_tensor(old) and torch.is_tensor(new):
                    if not torch.equal(old.cpu(), new.detach().cpu()): tensor_differences.append(key)
                elif repr(old) != repr(new): tensor_differences.append(key)
            (output/'paired_initialization_check.json').write_text(json.dumps({'changed_state_keys': tensor_differences,
                       'expected_only_change': 'dyn_o_flag', 'source': str(a.resume_scaffold_dir)}, indent=2)+'\n')
            assert tensor_differences == ['dyn_o_flag'] or set(tensor_differences)=={'dyn_o_flag'}, tensor_differences
        torch.save(state, output/'initial_dynamic.pth')
        torch.save(kwargs['s_model'].state_dict(), output/'initial_static.pth')
        initial_record = {'dynamic_state_tensor_sha256': tensor_state_hash(state),
                          'dynamic_gaussians': int(kwargs['d_model'].N),
                          'static_gaussians': int(kwargs['s_model'].N),
                          'rng_policy': 'seed 12345 immediately before dynamic initialization and photometric fit'}
        (output/'initialization.json').write_text(json.dumps(initial_record, indent=2)+'\n')
        mr.seed_everything(mr.SEED)
        return original_fit(self, *args, **kwargs)
    mr.DynReconstructionSolver.get_dynamic_model = seeded_dynamic
    mr.DynReconstructionSolver.photometric_fit = recorded_fit

    def load_rgb_foreground(self, epi_dirname='epi'):
        assert Path(self.ws).resolve() == ws
        self.register_gradfree_buffer('epi', torch.from_numpy(segmentation.copy()))
        self.has_epi = True
        logging.info('Adapter: RGB SAM2 foreground in legacy epi field; NOT epipolar residuals')
        return self

    def fixed_camera_optimizer(**kwargs):
        kwargs.update(lr_cam_f=0., lr_cam_q=0., lr_cam_t=0.)
        return OriginalOptimCFG(**kwargs)

    Saved2D.load_epi = load_rgb_foreground
    mr.OptimCFG = fixed_camera_optimizer
    start = time.perf_counter()
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    record = dict(status='running', input_manifest_sha256=sha(ws/'input_manifest.json'),
                  segmentation_sha256=sha(a.segmentation), config_sha256=sha(output/'config.yaml'),
                  script_sha256=sha(__file__), gpu=torch.cuda.get_device_name(0),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  upstream_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip(),
                  adaptations=['Supplied calibrated fixed pinhole cameras; bundle adjustment skipped',
                               'Binary RGB foreground replaces epipolar classifier input',
                               'All camera learning rates are zero; optional single input-only world normalization is reversed for evaluation'],
                  temporal_limit='Upstream regularizers operate on frame indices. Actual intervals are irregular; physical-time metrics must use manifest seconds.',
                  world_scale=world_scale, exact_principal_point=True,
                  rng_policy='isolated dynamic initialization and photo RNG seed 12345',
                  stages={})
    (output/'upstream.patch').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'], text=True))
    def save(): (output/'run.json').write_text(json.dumps(record, indent=2))
    save()
    try:
        poses = torch.tensor(m['c2w'], dtype=torch.float32)[None].repeat(len(m['frame_paths']),1,1)
        poses[:, :3, 3] *= world_scale
        cameras = MonocularCameras(n_time_steps=len(poses), default_H=m['height'], default_W=m['width'],
                                  K=torch.tensor(m['K']), delta_flag=False, init_camera_pose=poses, iso_focal=False)
        (output/'bundle').mkdir()
        torch.save(cameras.state_dict(), output/'bundle/bundle_cams.pth')
        torch.save({'dep_scale':torch.ones(len(poses)), 'adapter':'fixed calibrated cameras, no BA'}, output/'bundle/bundle.pth')
        if a.resume_scaffold_dir:
            source = a.resume_scaffold_dir.resolve()
            previous = json.loads((source/'run.json').read_text())
            assert previous['input_manifest_sha256'] == record['input_manifest_sha256']
            assert previous['segmentation_sha256'] == record['segmentation_sha256']
            assert previous['stages']['scaffold']['status'] == 'completed'
            prior_cfg = OmegaConf.to_container(OmegaConf.load(source/'config.yaml'))
            next_cfg = OmegaConf.to_container(cfg)
            differences = {k: [prior_cfg.get(k), next_cfg.get(k)] for k in set(prior_cfg)|set(next_cfg)
                           if prior_cfg.get(k) != next_cfg.get(k)}
            assert set(differences) <= {'validation_disable_dynamic_gate'}, differences
            assert abs(previous.get('world_scale',1.) - world_scale) < 1e-10
            record['allowed_photo_only_config_differences'] = differences
            (output/'mosca').mkdir()
            files = ['mosca/mosca.pth', 'track_identification.npz', 'bundle/bundle.pth', 'bundle/bundle_cams.pth']
            for name in files: shutil.copy2(source/name, output/name)
            record['reused_scaffold'] = {'source':str(source), 'files':{name:sha(source/name) for name in files},
                                         'original_scaffold_seconds':previous['stages']['scaffold']['seconds']}
            save()
        os.chdir(REPO)
        for name, function in [('photometric_warmup', mr.photometric_warmup),
                               ('scaffold', mr.scaffold_reconstruct),
                               ('photometric', mr.photometric_reconstruct)]:
            if a.resume_scaffold_dir and name in ('photometric_warmup', 'scaffold'):
                record['stages'][name] = {'seconds':0., 'status':'reused_completed_stage'}
                continue
            begin = time.perf_counter(); function(str(ws), str(output), cfg); torch.cuda.synchronize()
            record['stages'][name] = {'seconds':time.perf_counter()-begin, 'status':'completed'}
            save()
        final = torch.load(output/'photometric_cam.pth', map_location='cpu', weights_only=False)
        initial = cameras.state_dict()
        delta = {k:float((final[k]-v).abs().max()) for k,v in initial.items() if v.is_floating_point()}
        record['camera_max_abs_changes'] = delta
        assert max(delta.values()) < 1e-7, f'Known camera unexpectedly changed: {delta}'
        if a.export_script:
            subprocess.run([sys.executable,str(a.export_script.resolve()), '--input', str(ws),
                            '--checkpoint-dir', str(output), '--output', str(output/'diagnostics'),
                            '--segmentation', str(a.segmentation.resolve())], check=True)
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save()


if __name__ == '__main__':
    main()
