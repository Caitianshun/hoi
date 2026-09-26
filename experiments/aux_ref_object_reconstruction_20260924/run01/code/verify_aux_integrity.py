#!/usr/bin/env python3
"""Independent CPU-only AUX_REF_OBJECT artifact verifier; never loads a renderer.

Run --static-check while training runs. Run without it only after finals.json is
frozen. This script does not wait, poll, train, modify historical files, or load
published human parameters. Only protocol/integrity.json is written on full audit.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[2]
PROTOCOL = 'AUX_REF_OBJECT'
DEVS = ('dev1', 'dev2')
ARMS = ('Pred', 'Ref')
FIELDS = ('offset', 'log_scale', 'quat', 'opacity_logit', 'color_logit')
EXPECTED_TRAINABLE = sorted('scene.obank.' + k for k in FIELDS)
EXPECTED_GROUPS = sorted('obank.' + k for k in FIELDS)
EXPECTED_EVENTS = list(range(3000, 7001, 500))


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def demand(condition, message):
    if not condition:
        raise AssertionError(message)


def arr(value):
    return value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.ascontiguousarray(value)


def tensor_hash(value):
    value = arr(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def state_identity(state):
    arrays = {key: tensor_hash(value) for key, value in sorted(state.items())}
    return {'sha256': hashlib.sha256(json.dumps(arrays, sort_keys=True).encode()).hexdigest(), 'arrays': arrays}


def close(a, b, message, atol=1e-8):
    if not np.allclose(arr(a), arr(b), rtol=0, atol=atol):
        raise AssertionError(message)


class Verifier:
    def __init__(self):
        self.assets = {}
        self.historical = {}

    def checked(self, path, expected=None, role='artifact'):
        path = str(Path(path).resolve())
        if path not in self.assets:
            p = Path(path)
            demand(p.is_file(), 'Missing required file: ' + path)
            self.assets[path] = {'path': path, 'sha256': sha(path), 'bytes': p.stat().st_size, 'roles': []}
        item = self.assets[path]
        if expected is not None:
            demand(item['sha256'] == expected, 'Frozen SHA256 mismatch: ' + path)
        if role not in item['roles']:
            item['roles'].append(role)
        return {'path': path, 'sha256': item['sha256'], 'bytes': item['bytes']}

    def artifact(self, record, role='artifact'):
        return self.checked(record['path'], record['sha256'], role)

    def unchanged_after_audit(self):
        for path, item in self.assets.items():
            demand(sha(path) == item['sha256'], 'File changed during independent audit: ' + path)

    def context(self):
        self.start_path = E / 'frozen_aux/start.json'
        self.checked(self.start_path, role='start_freeze')
        self.start = load(self.start_path)
        demand(self.start['protocol_id'] == PROTOCOL, 'Wrong start protocol')
        for record in self.start['code'].values():
            self.artifact(record, 'frozen_training_source')
        for key in ('preflight', 'native_availability', 'regions_manifest', 'frame_schedules'):
            self.artifact(self.start[key], key)
        self.gate = load(self.start['native_availability']['path'])
        demand(self.gate['training_gate'] is True and self.gate['training_gate_passed'] is True, 'Native gate not passed')
        demand(self.gate['final_gate']['total_S'] == 21 and self.gate['final_gate']['total_E'] == 9, 'Wrong frozen sample counts')
        self.artifact(self.gate['script'], 'frozen_native_audit_source')
        self.artifact(self.gate['final_gate']['frame_manifest'], 'frozen_complete_candidate_CSV')
        for record in self.gate['final_gate']['frozen_dependencies']:
            self.artifact(record, 'native_gate_dependency')
        with Path(self.gate['final_gate']['frame_manifest']['path']).open(newline='') as f:
            csv_rows = list(csv.DictReader(f))
        demand(len(csv_rows) == len(self.gate['rows']) == 26, 'Complete candidate accounting changed')
        selected = [r for r in csv_rows if r['selected_S'] == 'True']
        demand(len(selected) == 21 and sum(r['selected_E'] == 'True' for r in csv_rows) == 9, 'CSV S/E mismatch')
        indexed = {r['official_capture_sample_id']: r for r in self.gate['rows']}
        for row in self.gate['rows']:
            if not row['selected_S']:
                continue
            demand(not row['missing_reasons'] and row['pred_evaluation']['extrapolated'] is False, 'Unresolved native row')
            for record in row['native_assets'].values():
                self.artifact(record, 'native_official_object_or_RGB')
            self.artifact(row['rectified_camera0'], 'camera0_training_RGB')
            self.checked(row['mask']['path'], row['mask']['archive_sha256'], 'RGB_only_training_mask')
            self.artifact(row['reference_motion'], 'isolated_published_object_Rt')
            if row['selected_E']:
                self.artifact(row['evaluation_region'], 'evaluation_region')
                self.artifact(row['rectified_camera1'], 'camera1_evaluation_only_RGB')
        regions = load(self.start['regions_manifest']['path'])
        demand(regions['status'] == 'frozen' and regions['prepared_before_auxiliary_training'] is True, 'Regions not frozen before training')
        demand(len(regions['rows']) == 9, 'Wrong E coverage')
        self.artifact(regions['native_manifest'], 'immutable_native_identity_snapshot')
        self.artifact(regions['code'], 'frozen_region_preparation_source')
        for row in regions['rows']:
            source = indexed[row['sample_id']]
            demand(source['selected_E'] and row['native_camera1']['sha256'] == source['native_assets']['camera1']['sha256'], 'E does not belong to S native group')
            for key in ('native_camera1', 'native_camera1_source', 'rgb', 'regions', 'overlay'):
                self.artifact(row[key], 'evaluation_only_' + key)
        self.inputs, self.schedules = {}, {}
        schedule_manifest = load(self.start['frame_schedules']['path'])
        for dev in DEVS:
            self.artifact(self.start['inputs'][dev], 'frozen_input_manifest')
            meta = load(self.start['inputs'][dev]['path'])
            demand(meta['phase'] == 'complete' and meta['protocol_id'] == PROTOCOL, 'Input not complete')
            demand(meta['camera_id'] == 0 and not meta['human_reference_used'] and not meta['camera1_used_for_training'], 'Input boundary changed')
            for path, digest in zip(meta['frame_paths'], meta['frame_sha256']):
                self.checked(path, digest, 'camera0_RGB')
            self.checked(meta['segmentation'], meta['segmentation_sha256'], 'camera0_RGB_mask')
            self.checked(meta['reference_object_motion'], meta['reference_motion_sha256'], 'Ref_object_Rt_only')
            ref = np.load(meta['reference_object_motion'], allow_pickle=False)
            demand(np.array_equal(ref['times'], meta['timestamp_seconds']), 'Ref/native time mapping changed')
            demand(str(ref['source']) == 'published_fit_actual_native_samples_no_interpolation', 'Ref interpolation not allowed')
            schedule_path = E / f'frozen_aux/{dev}_frame_schedule.npy'
            self.checked(schedule_path, schedule_manifest[dev]['schedule_sha256'], 'shared_frame_schedule')
            frames = np.load(schedule_path, allow_pickle=False)
            demand(frames.shape == (8000,) and np.issubdtype(frames.dtype, np.integer), 'Bad schedule shape/type')
            demand(frames.min() >= 0 and frames.max() < len(meta['timestamp_seconds']), 'Schedule out of range')
            counts = np.bincount(frames, minlength=len(meta['timestamp_seconds'])).tolist()
            demand(counts == schedule_manifest[dev]['frame_counts'], 'Frozen schedule counts changed')
            self.inputs[dev], self.schedules[dev] = meta, frames
        preflight = load(self.start['preflight']['path'])
        demand(preflight['status'] == 'passed' and preflight['temporary_state_discarded'] and preflight['same_initial_object_identity'], 'Preflight incomplete')
        demand(preflight['source_sha256'] == self.start['code']['train_aux.py']['sha256'], 'Preflight/formal code differs')
        # Compare existing historical checksums rather than asserting old files are
        # unchanged merely because their current hashes were collected now.
        historical_path = ROOT / 'experiments/object_pose_refinement_20260924/run01/protocol/audit_inputs_manifest.json'
        self.checked(historical_path, role='historical_identity_ledger')
        historical = load(historical_path)
        for name, record in historical['core_source_version'].items():
            self.historical[name] = self.artifact(record, 'historical_source_unchanged')
        cpu = load(self.gate['final_gate']['CPU_scene_check']['path'])
        for dev in DEVS:
            source = cpu['devs'][dev]['source_identity']
            for kind in ('init', 'checkpoint'):
                self.historical[dev + '_' + kind] = self.checked(source[kind + '_path'], source[kind + '_sha256'], 'historical_model_unchanged')
        self.cpu = cpu
        return {'status': 'passed', 'S': 21, 'E': 9, 'historical_assets_unchanged': len(self.historical)}

    def source_states(self, dev, config):
        source = config['source']
        for kind in ('init', 'checkpoint'):
            self.checked(source[kind + '_path'], source[kind + '_sha256'], 'historical_model_unchanged')
            demand(source[kind + '_sha256'] == self.cpu['devs'][dev]['source_identity'][kind + '_sha256'], 'Wrong old source')
        init = torch.load(source['init_path'], map_location='cpu', weights_only=False)
        old = torch.load(source['checkpoint_path'], map_location='cpu', weights_only=False)
        demand(init['reference_used'] is False and old['step'] == 8000, 'Illegal source scene')
        demand(old['initialization_sha256'] == source['init_sha256'], 'Old init/ckpt pairing mismatch')
        n = len(init['object_anchors'])
        fresh = {'anchor_id': torch.arange(n), 'stable_id': torch.arange(n),
                 'parent_id': torch.full((n,), -1, dtype=torch.long),
                 'grad_sum': torch.zeros(n), 'seen_count': torch.zeros(n), 'next_id': torch.tensor(n),
                 'offset': torch.zeros(n, 3),
                 'log_scale': torch.as_tensor(init['object_scale']).float().reshape(n, 1).repeat(1, 3).log(),
                 'quat': torch.zeros(n, 4), 'opacity_logit': torch.full((n, 1), -1.0986123),
                 'color_logit': torch.logit(torch.as_tensor(init['object_colors']).float().clamp(.01, .99))}
        fresh['quat'][:, 0] = 1
        demand(n == 4096, 'Untrained canonical initialization size changed')
        return init, old['model'], fresh

    def verify_run(self, entry):
        dev, arm = entry['dev'], entry['arm']
        folder = E / f'runs/{dev}_{arm}'
        for key in ('checkpoint', 'input_manifest', 'reference_motion', 'run_record', 'config'):
            self.artifact(entry[key], 'frozen_final_' + key)
        record, config = load(entry['run_record']['path']), load(entry['config']['path'])
        demand(record['status'] == 'completed' and record['step'] == record['formal_steps'] == record['budget_steps'] == 8000, 'Run not exactly complete')
        demand(record['dev'] == dev and record['arm'] == arm and record['protocol_id'] == PROTOCOL, 'Run identity mismatch')
        demand(record['H_S_motion_unchanged'] and record['published_object_motion_read'] == (arm == 'Ref'), 'Incorrect motion mode')
        demand(record['published_human_depth_texture_camera1_training'] is False, 'Forbidden training input')
        demand(record['cuda_visible_devices'] == '1', 'Unexpected GPU assignment')
        demand(config['dev'] == dev and config['arm'] == arm and config['steps'] == 8000 and config['seed'] == 12345, 'Run config differs from plan')
        demand(config['all_motion_frozen'] and config['ordinary_track_and_motion_prior'] is False, 'Motion/track objective unexpectedly enabled')
        demand(config['density'] == {'first': 3000, 'last': 7000, 'interval': 500, 'cap': 6000, 'threshold': .0002}, 'Density rule changed')
        demand(sorted(config['rates']) == EXPECTED_GROUPS, 'Optimizer includes non-object fields')
        demand(config['source']['old_obank_loaded'] is False and config['source']['reference_interpolation'] is False, 'Old object or interpolated Ref used')
        for name, digest in config['training_code'].items():
            self.checked(name, digest, 'run_training_source')
            demand(digest == self.start['code'][Path(name).name]['sha256'], 'Source differs from start freeze')
        self.artifact(config['loss_source'], 'historical_loss_source')
        demand(config['loss_source']['sha256'] == self.historical['train_structured.py']['sha256'], 'Historical loss changed')
        for item in (entry['input_manifest'], config['input_manifest'], record['input_manifest']):
            demand(item['sha256'] == self.start['inputs'][dev]['sha256'], 'Run/input start/final hash mismatch')
        self.artifact(config['frame_schedule'], 'run_frame_schedule')
        demand(config['frame_schedule']['sha256'] == record['frame_schedule_sha256'], 'Run schedule mismatch')
        initial_path, final_path = folder / 'initial_identity.json', folder / 'final_identity.json'
        self.checked(initial_path, role='initial_scene_identity'); self.checked(final_path, role='final_scene_identity')
        initial, final = load(initial_path), load(final_path)
        demand(initial['frozen'] == final['frozen'], 'H/S or frozen motion changed')
        demand(initial['trainable_parameters'] == final['trainable_parameters'] == EXPECTED_TRAINABLE, 'Non-object parameter marked trainable')
        demand(initial['initial_obank_identity'] == initial['object'] == record['initial_obank_identity'], 'Initial object identity mismatch')
        init, old, fresh = self.source_states(dev, config)
        demand(state_identity(fresh) == initial['object'], 'Object did not start from original untrained arrays')
        for prefix in ('human', 'hbank', 'sbank'):
            actual = {k[len(prefix) + 1:]: v for k, v in old.items() if k.startswith(prefix + '.')}
            demand(state_identity(actual) == initial['frozen'][prefix], 'Frozen ' + prefix + ' differs from old final S1')
        for key, old_key in [('object_anchors', 'object_anchors'), ('c2w', 'c2w'),
                             ('original_object_R0', 'object_R0'), ('original_object_translation', 'object_translation'),
                             ('original_object_rot_delta', 'object_rot_delta')]:
            demand(tensor_hash(old[old_key]) == initial['frozen'][key], 'Source frozen tensor changed: ' + key)
        del old, fresh
        native_rows = [r for r in self.gate['rows'] if r['dev'] == dev and r['selected_S']]
        demand(config['source']['query_times_seconds'] == self.inputs[dev]['timestamp_seconds'], 'Prediction query times mismatch')
        demand(len(config['source']['motion_records']) == len(native_rows), 'Motion bracket row count mismatch')
        for motion, native in zip(config['source']['motion_records'], native_rows):
            p = native['pred_evaluation']
            demand(not motion['extrapolated'] and motion['source'] == 'old_S1_RGB_prediction', 'Invalid motion interpolation source')
            for left, right in [('left_frame', 'left_index'), ('right_frame', 'right_index'),
                                ('left_timestamp_seconds', 'left_time'), ('right_timestamp_seconds', 'right_time'),
                                ('interval_seconds', 'interval_seconds'), ('right_weight', 'right_weight')]:
                demand(motion[left] == p[right], 'Prediction bracket differs from frozen native gate')
        steps_path, topology_path = folder / 'steps.jsonl', folder / 'topology.jsonl'
        self.checked(steps_path, role='complete_step_log'); self.checked(topology_path, role='object_topology_log')
        steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line]
        events = [json.loads(line) for line in topology_path.read_text().splitlines() if line]
        demand([r['step'] for r in steps] == list(range(1, 8001)), 'Steps have missing/duplicate/extra budget entries')
        frames = self.schedules[dev]
        demand(np.array_equal([r['frame'] for r in steps], frames), 'Actual frame order differs from shared schedule')
        cumulative = np.cumsum([r['data_gradient_nonzero'] for r in steps])
        for row in steps:
            demand(0 < row['object_points'] <= 6000, 'Point budget exceeded')
            demand(row['data_gradient_nonzero'] == (row['data_grad_gaussians'] > 0), 'Gradient coverage flag mismatch')
            demand(0 <= row['data_grad_gaussians'] <= row['object_points'], 'Invalid gradient coverage count')
            demand(row['O_pixels'] == self.inputs[dev]['native_rows'][row['frame']]['object_pixels'], 'Logged mask identity differs')
            for key in ('data_loss', 'rgb_loss', 'instance_loss', 'static_reg'):
                demand(math.isfinite(row[key]), 'Nonfinite loss ' + key)
            for key in ('data_grad_norm', 'reg_grad_norm'):
                demand(set(row[key]) == set(FIELDS), 'Incomplete gradient attributes')
                demand(all(math.isfinite(x) and x >= 0 for x in row[key].values()), 'Nonfinite gradient norm')
        effective = int(cumulative[-1])
        demand(record['effective_data_gradient_steps'] == effective and record['zero_data_gradient_steps'] == 8000 - effective, 'Effective/zero gradient totals mismatch')
        demand([event['step'] for event in events] == EXPECTED_EVENTS, 'Wrong topology event count/schedule')
        stable = set(range(4096)); ever = set(stable); max_points = 4096; event_by_step = {}
        for event in events:
            demand(event['entity'] == 2 and event['before'] == len(stable), 'Non-object/discontinuous topology event')
            pruned, split, children = map(set, (event['pruned'], event['split_parents'], event['children']))
            demand(len(pruned) == len(event['pruned']) and len(split) == len(event['split_parents']) and len(children) == len(event['children']), 'Duplicate topology IDs')
            demand(pruned <= stable and split <= stable and not pruned & split, 'Invalid pruned/split parents')
            demand(len(children) == 2 * len(split) and not children & ever, 'Children not new paired descendants')
            stable = stable - pruned - split | children; ever |= children
            demand(len(stable) == event['after'] <= 6000, 'Topology count/cap mismatch')
            demand(steps[event['step'] - 1]['object_points'] == event['before'] and steps[event['step']]['object_points'] == event['after'], 'Step/topology counts inconsistent')
            max_points = max(max_points, event['before'], event['after']); event_by_step[event['step']] = event
        n = 4096
        for row in steps:
            demand(row['object_points'] == n, 'Unlogged topology change')
            if row['step'] in event_by_step:
                n = event_by_step[row['step']]['after']
        checkpoints, maximum_offset = [], 0.0
        final_state = None
        for number in (2000, 4000, 6000, 8000):
            path = folder / f'checkpoint_{number:06d}.pt'
            item = self.checked(path, entry['checkpoint']['sha256'] if number == 8000 else None, 'object_checkpoint')
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            demand(ckpt['protocol_id'] == PROTOCOL and ckpt['dev'] == dev and ckpt['arm'] == arm and ckpt['step'] == number, 'Checkpoint identity/step mismatch')
            demand(ckpt['config'] == config and ckpt['effective_data_gradient_steps'] == int(cumulative[number - 1]), 'Checkpoint config/coverage mismatch')
            ident = ckpt['scene_identity']; state = ckpt['obank']
            demand(ident['frozen'] == initial['frozen'] and ident['trainable_parameters'] == EXPECTED_TRAINABLE, 'Frozen scene changed in checkpoint')
            demand(state_identity(state) == ident['object'], 'Checkpoint object hash does not match stored identity')
            count = len(state['anchor_id']); demand(count <= 6000, 'Checkpoint exceeds point cap')
            demand(len(set(arr(state['stable_id']).tolist())) == count, 'Duplicate stable Gaussian IDs')
            demand(bool((state['anchor_id'] >= 0).all() and (state['anchor_id'] < 4096).all()), 'Invalid canonical attachment')
            for value in state.values():
                demand(bool(torch.isfinite(value).all()), 'Nonfinite object checkpoint tensor')
            offset = .005 * state['offset'] / torch.sqrt(1 + state['offset'].square().sum(-1, keepdim=True))
            bound = float(offset.norm(dim=-1).max()); maximum_offset = max(maximum_offset, bound)
            demand(bound <= .005000001, 'Canonical offset exceeds 5 mm')
            optimizer = ckpt['optimizer']; groups = optimizer['param_groups']
            demand(sorted(g['name'] for g in groups) == EXPECTED_GROUPS and all(len(g['params']) == 1 for g in groups), 'Checkpoint optimizer includes non-object parameters')
            ids = [g['params'][0] for g in groups]
            demand(len(set(ids)) == 5 and set(optimizer['state']) == set(ids), 'Unexpected optimizer state')
            for group in groups:
                demand(group['lr'] == config['rates'][group['name']] and group['eps'] == config['eps'], 'Optimizer rate/eps changed')
                demand(int(optimizer['state'][group['params'][0]]['step']) == number, 'Optimizer update budget differs')
            checkpoints.append(item | {'step': number, 'object_points': count, 'max_offset_m': bound})
            if number == 8000:
                demand(ident == final, 'Final identity differs from step 8000 checkpoint')
                demand(set(arr(state['stable_id']).tolist()) == stable, 'Final topology not explained by logged events')
                final_state = state
            del ckpt
        for name in ('object_initial.npz', 'object_final.npz', 'object_initial_stats.json', 'object_final_stats.json', 'camera0_fit.json'):
            self.checked(folder / name, role='representation_or_training_fit_export')
        export = np.load(folder / 'object_final.npz', allow_pickle=False)
        demand(len(export['anchor_id']) == len(stable), 'Export count differs from terminal checkpoint')
        for key in ('anchor_id', 'stable_id', 'parent_id', 'next_id'):
            demand(np.array_equal(export[key], arr(final_state[key])), 'Export differs from checkpoint: ' + key)
        offset = .005 * final_state['offset'] / torch.sqrt(1 + final_state['offset'].square().sum(-1, keepdim=True))
        close(export['local_offset_m'], offset, 'Export offset mismatch')
        close(export['scale_m'], final_state['log_scale'].clamp(-9, 0).exp(), 'Export scale mismatch')
        close(export['opacity'], final_state['opacity_logit'].sigmoid(), 'Export opacity mismatch')
        close(export['colors'], final_state['color_logit'].sigmoid(), 'Export color mismatch')
        anchors = arr(torch.as_tensor(init['object_anchors']).float())[export['anchor_id']]
        close(export['anchor_canonical_m'], anchors, 'Export anchor mismatch')
        close(export['centres_canonical_m'], anchors + arr(offset), 'Export canonical centre mismatch', atol=1e-7)
        demand(np.isfinite(export['centres_canonical_m']).all(), 'Nonfinite canonical geometry')
        statistics = load(folder / 'object_final_stats.json')
        demand(statistics['count'] == record['object_stats']['count'] == len(stable), 'Final count summary mismatch')
        maximum_export_offset = float(np.linalg.norm(export['local_offset_m'], axis=1).max())
        demand(maximum_export_offset <= .005000001, 'Export exceeds 5 mm offset')
        close(statistics['offset_norm_m']['1'], maximum_export_offset, 'Offset statistic mismatch')
        empty_samples = [r['sample_id'] for r in self.inputs[dev]['native_rows'] if r['object_pixels'] == 0]
        return {'protocol_id': PROTOCOL, 'dev': dev, 'arm': arm,
                'object_motion_source': 'rgb_s1' if arm == 'Pred' else 'published_fit',
                'status': 'passed', 'formal_steps': 8000, 'effective_data_gradient_steps': effective,
                'zero_data_gradient_steps': 8000 - effective, 'topology_event_count': len(events),
                'maximum_object_points': max_points, 'final_object_points': len(stable),
                'maximum_checked_canonical_offset_m': max(maximum_offset, maximum_export_offset),
                'shared_initial_object_identity': initial['object'], 'frozen_scene_identity': initial['frozen'],
                'frame_schedule_sha256': config['frame_schedule']['sha256'],
                'empty_RGB_object_mask_samples_retained': empty_samples,
                'empty_RGB_object_mask_scheduled_steps': sum(r['O_pixels'] == 0 for r in steps),
                'checkpoints': checkpoints, 'host': record['host'], 'gpu': record['gpu'],
                'cuda_visible_devices': record['cuda_visible_devices'], 'wall_seconds': record['wall_seconds'],
                'peak_allocated_bytes': record['peak_allocated_bytes'], 'peak_reserved_bytes': record['peak_reserved_bytes']}

    def all_runs(self):
        finals_path = E / 'frozen_aux/finals.json'
        self.checked(finals_path, role='terminal_freeze')
        finals = load(finals_path)
        demand(finals['protocol_id'] == PROTOCOL and finals['phase'] == 'four_finals_frozen', 'Terminal freeze incomplete')
        self.artifact(finals['start_freeze'], 'same_start_freeze')
        demand(finals['start_freeze']['sha256'] == sha(self.start_path), 'Start freeze changed')
        demand(finals['regions_manifest'] == self.start['regions_manifest'], 'E changed across training')
        expected = {(d, a) for d in DEVS for a in ARMS}
        demand(len(finals['runs']) == 4 and {(r['dev'], r['arm']) for r in finals['runs']} == expected, 'Not exactly the four authorized runs')
        actual_dirs = {p.name for p in (E / 'runs').iterdir() if p.is_dir()}
        demand(actual_dirs == {d + '_' + a for d, a in expected}, 'Unexpected extra formal run directories')
        reports = [self.verify_run(row) for row in finals['runs']]
        for dev in DEVS:
            pred, ref = ([r for r in reports if r['dev'] == dev and r['arm'] == a][0] for a in ARMS)
            for key in ('shared_initial_object_identity', 'frozen_scene_identity', 'frame_schedule_sha256', 'empty_RGB_object_mask_samples_retained'):
                demand(pred[key] == ref[key], 'Pred/Ref paired invariant differs: ' + key)
            a = np.load(E / f'runs/{dev}_Pred/object_initial.npz', allow_pickle=False)
            b = np.load(E / f'runs/{dev}_Ref/object_initial.npz', allow_pickle=False)
            demand(set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files), 'Initial canonical exports differ between arms')
        demand(sum(r['formal_steps'] for r in reports) == 32000, 'Wrong four-run total budget')
        return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--static-check', action='store_true', help='Validate frozen startup inputs and historical checksums only; do not read running training outputs or write terminal report')
    args = ap.parse_args()
    torch.set_num_threads(1)
    start = time.perf_counter()
    verifier = Verifier()
    report = {'protocol_id': PROTOCOL, 'status': 'running',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'CPU_only': True, 'GPU_used': False, 'training_performed': False,
              'verifier': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}}
    try:
        report['startup_check'] = verifier.context()
        if args.static_check:
            verifier.unchanged_after_audit()
            print(json.dumps({'status': 'static_check_passed', 'frozen_asset_count': len(verifier.assets),
                              'historical_assets_unchanged': verifier.historical,
                              'full_terminal_audit_run': False}, indent=2))
            return
        report['runs'] = verifier.all_runs()
        verifier.unchanged_after_audit()
        report.update(status='passed', four_runs=4, total_formal_steps=32000,
                      all_frozen_hashes_unchanged=True, historical_assets_unchanged=verifier.historical,
                      only_object_optimizer_parameters=True, H_S_RGB_motion_unchanged=True,
                      paired_initialization_and_schedule_identical=True,
                      native_S_count=21, native_E_count=9,
                      limitations=['Published fits are auxiliary fixed input, not RGB-only motion estimates.',
                                   'Actual native exposures and sensor synchronization error remain unavailable.',
                                   'Runtime H/S immutability is supported by identical checkpoint snapshots, frozen source tensors, trainable/optimizer allowlists, and checkpoint-time assertions; this is not an independent rerender.'])
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if not args.static_check:
            report.update(wall_seconds=time.perf_counter() - start,
                          assets=list(verifier.assets.values()))
            destination = E / 'protocol/integrity.json'
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            temporary.replace(destination)
    print(json.dumps({'status': report['status'], 'runs': [{k: r[k] for k in ('dev', 'arm', 'formal_steps', 'effective_data_gradient_steps', 'final_object_points')} for r in report['runs']],
                      'integrity_sha256': sha(E / 'protocol/integrity.json')}, indent=2))


if __name__ == '__main__':
    main()
