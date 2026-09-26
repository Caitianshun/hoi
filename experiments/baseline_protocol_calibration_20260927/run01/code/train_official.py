"""Run the official training loop with narrow I/O, audit and resume hooks.

The upstream function body is loaded verbatim, with asserted replacements for
failure handling, exact RNG/sampler recovery and end-of-step checkpoints. Model,
loss, density control, SH progression and optimizer rules remain upstream.
"""
from __future__ import annotations
import argparse
import copy
import inspect
import json
import os
import random
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from adapter_4dgs import (ROOT, RUN, UPSTREAM, OFFICIAL_COMMIT, ProtocolScene,
                          official_config, save_json, sha, projection_check, load_manifest)
import train as official


def rng_capture():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def rng_restore(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])


class Runtime:
    def __init__(self, output, check, resume=None):
        self.output = Path(output); self.check = check
        self.started = time.monotonic(); self.previous_seconds = 0.
        self.resume = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
        self.last_checkpoint = Path(resume) if resume else None
        self.allowed_seconds = float(os.environ.get('V3_REMAINING_GPU_SECONDS', 12 * 3600))
        self.stage = None; self.step = 0; self.attempted = 0
        self.peak_points = 0; self.gradient_checks = {}; self.loss_rows = []
        self.prior_steps = self.resume.get('total_nominal_steps', 0) if self.resume else 0
        if self.resume:
            self.previous_seconds = self.resume['elapsed_seconds']
            self.peak_points = self.resume['peak_points']
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = (self.output / 'train_metrics.jsonl').open('a', buffering=1)

    def before_step(self, stage, iteration):
        self.stage = stage; self.step = iteration; self.attempted += 1
        if self.check:
            ledger = RUN / 'protocol' / 'temporary_steps.jsonl'
            with ledger.open('a') as f:
                f.write(json.dumps(dict(run=self.output.name, stage=stage,
                                       iteration=iteration, time=time.time())) + '\n')
        if time.monotonic() - self.started > self.allowed_seconds:
            raise TimeoutError('12 GPU-hour ceiling reached; retain incomplete status')

    def gradient_check(self, stage, iteration, model, loss):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite loss at {stage}:{iteration}; no automatic restart')
        if iteration in [1, 2, 10] or iteration % 1000 == 0:
            values = {}
            for group in model.optimizer.param_groups:
                gradients = [p.grad for p in group['params'] if p.grad is not None]
                assert all(bool(torch.isfinite(g).all()) for g in gradients), group['name']
                values[group['name']] = sum(float(g.detach().abs().sum()) for g in gradients)
            self.gradient_checks[f'{stage}_{iteration}'] = values

    def resume_sampler(self, stage, cameras, viewpoint_stack, temp_list, model):
        if self.resume and self.resume['stage'] == stage:
            lookup = {c.uid: c for c in cameras}
            viewpoint_stack = [lookup[i] for i in self.resume['viewpoint_stack']]
            temp_list = [lookup[i] for i in self.resume['temp_list']]
            model._deformation_accum = self.resume['deformation_accum'].cuda()
            rng_restore(self.resume['rng'])
        return viewpoint_stack, temp_list

    def report(self, writer, iteration, l1, loss, loss_fn, ms, tests, scene, render, render_args, stage, dataset_type):
        assert not tests and not scene.getTestCameras()
        points = len(scene.gaussians.get_xyz); self.peak_points = max(self.peak_points, points)
        row = dict(stage=stage, iteration=iteration, l1=float(l1), loss=float(loss),
                   points=points, iter_gpu_ms=float(ms),
                   allocated_bytes=torch.cuda.memory_allocated(),
                   reserved_bytes=torch.cuda.memory_reserved(),
                   seconds=self.previous_seconds + time.monotonic() - self.started)
        if self.check or iteration % 100 == 0 or iteration == 1:
            self.log.write(json.dumps(row) + '\n')
            self.loss_rows.append(row)
        if iteration % 100 == 0 or iteration == 1:
            save_json(self.output / 'progress.json', row)

    def after_step(self, stage, iteration, model, viewpoint_stack, temp_list):
        self.peak_points = max(self.peak_points, len(model.get_xyz))
        if iteration % 1000 == 0 or iteration == self.stage_end:
            self.save_state(stage, iteration, model, viewpoint_stack, temp_list)

    def save_state(self, stage, iteration, model, viewpoint_stack, temp_list):
        path = self.output / f'checkpoint_{stage}_{iteration:06d}.pt'
        torch.cuda.synchronize()
        # Capture sampler/RNG after optimizer and density changes, for exact continuation.
        state = dict(model=model.capture(), stage=stage, iteration=iteration,
                     rng=rng_capture(), viewpoint_stack=[x.uid for x in viewpoint_stack],
                     temp_list=[x.uid for x in temp_list],
                     deformation_accum=model._deformation_accum,
                     elapsed_seconds=self.previous_seconds + time.monotonic() - self.started,
                     total_nominal_steps=self.prior_steps + self.attempted,
                     peak_points=self.peak_points)
        temp = path.with_suffix('.tmp'); torch.save(state, temp); temp.replace(path)
        self.last_checkpoint = path
        save_json(self.output / 'latest_checkpoint.json', dict(path=str(path), stage=stage, iteration=iteration))


def patched_loop(runtime):
    source = inspect.getsource(official.scene_reconstruction)
    original = source
    changes = {
        '(model_params, first_iter) = torch.load(checkpoint)':
            "resume_blob = torch.load(checkpoint, weights_only=False)\n            model_params, first_iter = resume_blob['model'], resume_blob['iteration']",
        'count = 0\n    for iteration':
            'viewpoint_stack, temp_list = runtime.resume_sampler(stage, train_cams, viewpoint_stack, temp_list, gaussians)\n    count = 0\n    for iteration',
        'for iteration in range(first_iter, final_iter+1):':
            'for iteration in range(first_iter, final_iter+1):\n        runtime.before_step(stage, iteration)',
        'if torch.isnan(loss).any():\n            print("loss is nan,end training, reexecv program now.")\n            os.execv(sys.executable, [sys.executable] + sys.argv)':
            'runtime.gradient_check(stage, iteration, gaussians, loss)',
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    source += '\n            runtime.after_step(stage, iteration, gaussians, viewpoint_stack, temp_list)\n'
    save_json(runtime.output / 'upstream_adaptation.json', {
        'official_commit': OFFICIAL_COMMIT, 'original_function_sha256': __import__('hashlib').sha256(original.encode()).hexdigest(),
        'replacements': changes, 'appended_hook': 'runtime.after_step',
        'optimizer_last_fine_step': 'Preserve upstream: iteration 14000 backpropagates but skips optimizer.step; 17000 nominal iterations = 16999 updates.',
        'dataset_replacement': 'ProtocolScene: train manifest only; test/video empty; full K; nonzero extent',
        'disabled': ['GUI listener', 'test RGB reporting', 'render_process previews'],
    })
    namespace = official.__dict__.copy(); namespace.update(runtime=runtime, training_report=runtime.report)
    exec(compile(source, str(UPSTREAM / 'train.py') + ':adapted_function', 'exec'), namespace)
    return namespace['scene_reconstruction']


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    assert torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 3090'
    assert subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'], text=True).strip() == OFFICIAL_COMMIT
    torch.set_num_threads(4); official.setup_seed(12345)
    args, (dataset, hidden, opt, pipe) = official_config(a.batch_size)
    dataset.source_path = str(Path(a.manifest).absolute()); dataset.model_path = str(Path(a.output).absolute())
    dataset.render_process = False
    output = Path(dataset.model_path); output.mkdir(parents=True, exist_ok=True)
    if any((output / n).exists() for n in ['run.json','failure.json','effective_config.json']) and not a.resume:
        raise FileExistsError('Existing attempt cannot restart from zero; exact-state resume required')
    if a.check:
        ledger = RUN / 'protocol/temporary_steps.jsonl'
        rows = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
        limit = 100 if a.dataset == 'hos_backpack' else 50
        used = sum(r['run'].startswith(a.dataset) for r in rows)
        requested = (a.coarse_steps or 0) + (a.fine_steps or 0)
        if a.resume:
            prev = torch.load(a.resume, map_location='cpu', weights_only=False)
            requested -= prev['iteration'] + (a.coarse_steps if prev['stage']=='fine' else 0)
        assert used + requested <= limit, (a.dataset, used, requested, limit)
        assert len(rows) + requested <= 200
    runtime = Runtime(output, a.check, a.resume)
    runtime.started = time.monotonic()
    try:
        manifest = load_manifest(a.manifest)
        save_json(output / 'projection_check.json', projection_check(manifest))
        cfg = dict(model=vars(dataset), hidden=vars(hidden), optimization=vars(opt), pipeline=vars(pipe),
                   seed=12345, official_commit=OFFICIAL_COMMIT, manifest_sha256=sha(a.manifest),
                   GPU='physical1 RTX3090', check=a.check,
                   adapter_sha256=sha(Path(__file__).with_name('adapter_4dgs.py')),
                   training_wrapper_sha256=sha(__file__),
                   official_default_sha256=sha(UPSTREAM/'arguments/hypernerf/default.py'))
        if a.resume:
            assert json.loads((output/'effective_config.json').read_text()) == cfg
        else:
            save_json(output / 'effective_config.json', cfg)
        gaussians = official.GaussianModel(dataset.sh_degree, hidden)
        scene = ProtocolScene(dataset, gaussians)
        loop = patched_loop(runtime)
        official.network_gui.try_connect = lambda: None
        timer = official.Timer(); timer.start()
        coarse_end = a.coarse_steps if a.check else opt.coarse_iterations
        fine_end = a.fine_steps if a.check else opt.iterations
        resume_stage = runtime.resume['stage'] if runtime.resume else None
        for stage, count in [('coarse',coarse_end), ('fine',fine_end)]:
            if not count or (resume_stage == 'fine' and stage == 'coarse'):
                continue
            runtime.stage_end = count
            checkpoint = a.resume if resume_stage == stage else None
            loop(dataset,opt,hidden,pipe,[],[],[],checkpoint,-1,gaussians,scene,stage,None,count,timer)
            runtime.resume = None
        torch.cuda.synchronize()
        if a.check:
            # Forward identity after checkpoint reload; no additional optimization.
            camera = scene.getTrainCameras()[0]
            with torch.no_grad():
                before = official.render(camera,gaussians,pipe,torch.ones(3,device='cuda'),stage=runtime.stage)['render']
                state = torch.load(runtime.last_checkpoint, weights_only=False)
                gaussians.restore(state['model'], opt)
                after = official.render(camera,gaussians,pipe,torch.ones(3,device='cuda'),stage=runtime.stage)['render']
                delta = float((before-after).abs().max())
                assert delta < 1e-6
                np.savez_compressed(output/'train_forward.npz', rgb=after.cpu().numpy())
            save_json(output/'checkpoint_check.json',dict(max_abs_render_difference=delta,passed=True))
        else:
            scene.save(opt.iterations,'fine')
        result = dict(status='check_completed' if a.check else 'completed',
                      nominal_iterations=runtime.prior_steps + runtime.attempted,
                      optimizer_updates=runtime.prior_steps + runtime.attempted - (0 if a.check else 1),
                      checkpoint=str(runtime.last_checkpoint), checkpoint_sha256=sha(runtime.last_checkpoint),
                      seconds=runtime.previous_seconds+time.monotonic()-runtime.started,
                      peak_points=runtime.peak_points, final_points=len(gaussians.get_xyz),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      gpu_process_note='PyTorch memory is not entire-card occupancy',
                      GPU='physical1 NVIDIA GeForce RTX 3090', seed=12345,
                      test_RGB_loaded=False, gradient_checks=runtime.gradient_checks,
                      source_config_sha256=sha(output/'effective_config.json'))
        save_json(output/'run.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='gradient_checks'}),flush=True)
    except BaseException:
        save_json(output/'failure.json',dict(status='failed',stage=runtime.stage,iteration=runtime.step,
                    attempted_iterations=runtime.prior_steps+runtime.attempted,
                    seconds=runtime.previous_seconds+time.monotonic()-runtime.started,traceback=traceback.format_exc()))
        raise
    finally:
        runtime.log.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--check',action='store_true');p.add_argument('--coarse-steps',type=int,default=20)
    p.add_argument('--fine-steps',type=int,default=30);p.add_argument('--resume')
    run(p.parse_args())
