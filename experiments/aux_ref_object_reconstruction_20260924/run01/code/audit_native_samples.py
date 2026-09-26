#!/usr/bin/env python3
"""Audit native object/RGB sample identities; optionally fetch only missing allowed members.

No model fitting/training, human parameters, depth, fitted masks, or texture is read.
The native acquisition group is an identity, not a claim of zero exposure-time error.
"""
import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import io
import json
import math
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'data/BEHAVE/source_metadata/Date01.zip.index.json'
DOWNLOADER = ROOT / 'experiments/mosca_baseline_20260922/data/remote_archive.py'
DEVS = {
    'dev1': dict(sequence='Date01_Sub01_boxsmall_hand', category='boxsmall',
                 input_manifest='experiments/mosca_baseline_20260922/common_input/input_manifest.json',
                 source_root='data/BEHAVE/evaluation_only/sparse_event_reference',
                 template='data/BEHAVE/evaluation_only/templates/boxsmall_f1000.ply',
                 E=[20, 21, 22, 25, 26]),
    'dev2': dict(sequence='Date01_Sub01_chairwood_lift', category='chair',
                 input_manifest='experiments/structured_hoi_20260923/data/dev2/input_manifest.json',
                 source_root='experiments/structured_hoi_20260923/data_audit/dev2_evaluation/source',
                 template='experiments/structured_hoi_20260923/data_audit/sources/chairwood_f2500.ply',
                 E=[4, 6, 8, 10]),
}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


class ObjectParametersOnly(pickle.Unpickler):
    """Allow the NumPy object-pose arrays in the official object-only pickle."""
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
                   ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
                   ('_codecs', 'encode')}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unexpected object parameter pickle class: {module}.{name}')
        return super().find_class(module, name)


def read_object_parameters(path):
    data = ObjectParametersOnly(io.BytesIO(Path(path).read_bytes()), encoding='latin1').load()
    if not isinstance(data, dict) or not {'angle', 'trans'} <= set(data):
        raise ValueError('Not an official angle/trans object parameter dictionary')
    angle = np.asarray(data['angle'], dtype=np.float64).reshape(3)
    trans = np.asarray(data['trans'], dtype=np.float64).reshape(3)
    if not np.isfinite(angle).all() or not np.isfinite(trans).all():
        raise ValueError('Nonfinite published object parameters')
    return angle, trans, sorted(data)


def existing_source(dev, member):
    target = OUT / 'inputs/native' / member
    old = ROOT / dev['source_root'] / member
    return target if target.exists() else old if old.exists() else None


def validate_member(path, member, ix):
    import zlib
    b = path.read_bytes()
    entry = next(e for e in ix['entries'] if e['name'] == member)
    assert len(b) == entry['size'] and zlib.crc32(b) == entry['crc32'], member
    sp = Path(str(path) + '.source.json')
    if sp.exists():
        provenance = load(sp)
        assert provenance['member'] == member and provenance['url'] == ix['url']
        assert provenance['sha256'] == hashlib.sha256(b).hexdigest()
        assert provenance['etag'] == ix['etag']
    return artifact(path)


def materialize(dev, member, ix):
    dest = OUT / 'inputs/native' / member
    prev = existing_source(dev, member)
    if prev:
        validate_member(prev, member, ix)
        if prev != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prev, dest)
            meta = load(str(prev) + '.source.json')
            meta.update(reused_from=str(prev), acquisition='verified_local_copy',
                        local_copy_transferred_bytes=0)
            Path(str(dest) + '.source.json').write_text(json.dumps(meta, indent=2) + '\n')
        return validate_member(dest, member, ix)
    # Member allowlist prevents accidental access to human, depth, or fitted masks.
    allowed = member.endswith('/k0.color.jpg') or member.endswith('/k1.color.jpg')
    allowed |= member.endswith('/fit01/' + dev['category'] + '_fit.pkl')
    assert allowed and '/person/' not in member
    command = [sys.executable, str(DOWNLOADER), 'fetch_zip', ix['url'], str(dest),
               '--member', member, '--index', str(INDEX), '--budget-mb', '2']
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(member + ': ' + result.stderr[-1500:])
    return validate_member(dest, member, ix)


def verified(path, expected=None):
    result = artifact(path)
    if expected is not None and result['sha256'] != expected:
        raise ValueError(f'Frozen artifact hash mismatch: {path}')
    return result


def finalize_native_gate(regions_sha256, verify_only=False):
    """Bind completed RGB-only inputs and frozen evaluation regions, on CPU only."""
    import torch
    torch.set_num_threads(1)
    protocol = OUT / 'protocol'
    native_path = protocol / 'native_availability.json'
    result = load(native_path)
    if verify_only:
        assert result.get('training_gate_passed') is True, 'Native gate is not frozen'
        regions_sha256 = result['final_gate']['regions_manifest']['sha256']
    assert regions_sha256, 'Pass the independently confirmed final regions manifest SHA256'
    region_path = OUT / 'evaluation/regions/manifest.json'
    region_artifact = verified(region_path, regions_sha256)
    regions = load(region_path)
    assert regions['status'] == 'frozen' and regions['protocol_id'] == 'AUX_REF_OBJECT'
    assert regions['prepared_before_auxiliary_training'] is True
    snapshot_artifact = verified(regions['native_manifest']['path'], regions['native_manifest']['sha256'])
    assert Path(snapshot_artifact['path']).resolve() != native_path.resolve(), 'Use an immutable pre-gate identity snapshot to avoid a hash cycle'
    identity_snapshot = load(snapshot_artifact['path'])
    identity_rows = {r['official_capture_sample_id']: r for r in identity_snapshot['rows']}
    evaluation_rows = {r['sample_id']: r for r in regions['rows']}
    assert len(evaluation_rows) == len(regions['rows']) == 9
    expected_E = {r['official_capture_sample_id'] for r in result['rows'] if r['candidate_E']}
    assert set(evaluation_rows) == expected_E, 'Every existing predetermined E sample must be retained'
    for path, digest in zip(regions['camera']['metadata']['source_paths'], regions['camera']['metadata']['source_sha256']):
        verified(path, digest)
    for record in (regions['code'], regions['legacy_raster'], regions['legacy_camera_mapping']):
        verified(record['path'], record['sha256'])
    assert all(regions['raster_checks'].values())
    cpu_path = protocol / 'aux_scene_cpu_check.json'
    cpu = load(cpu_path)
    assert cpu['status'] == 'passed' and cpu['reference_assets_read'] is False
    aux_code = verified(OUT / 'code/aux_scene.py', cpu['code_sha256'])
    frozen_dependencies = [region_artifact, snapshot_artifact, artifact(cpu_path), aux_code]
    for dev in DEVS:
        manifest_path = OUT / f'inputs/{dev}/input_manifest.json'
        meta = load(manifest_path)
        assert meta['phase'] == 'complete', f'{dev}: masks are not complete; leave gate pending'
        assert meta['protocol_id'] == 'AUX_REF_OBJECT' and meta['dev'] == dev
        assert meta['human_reference_used'] is False and meta['camera1_used_for_training'] is False
        assert meta['P0_modified'] is False and meta['camera_id'] == 0
        prepared_artifact = artifact(manifest_path)
        segmentation_artifact = verified(meta['segmentation'], meta['segmentation_sha256'])
        reference_artifact = verified(meta['reference_object_motion'], meta['reference_motion_sha256'])
        frozen_dependencies += [prepared_artifact, segmentation_artifact, reference_artifact]
        rr = [r for r in result['rows'] if r['dev'] == dev and r['candidate_S']]
        prepared = meta['native_rows']
        assert [r['official_capture_sample_id'] for r in rr] == [r['sample_id'] for r in prepared]
        assert len(rr) == meta['mask_frame_count'] == len(meta['timestamp_seconds']) >= 3
        assert len(set(meta['timestamp_seconds'])) == len(rr)
        for path, digest in zip(meta['calibration']['source_paths'], meta['calibration']['source_sha256']):
            verified(path, digest)
        np.testing.assert_array_equal(meta['K'], meta['calibration']['K'])
        np.testing.assert_array_equal(meta['c2w'], meta['calibration']['c2w'])
        masks = np.load(meta['segmentation'], allow_pickle=False)
        labels = masks['entity_labels']
        assert labels.shape == (len(rr), meta['height'], meta['width'])
        assert set(np.unique(labels).tolist()) <= {0, 1, 2}
        assert masks['source_frame_ids'].tolist() == [r['sample_id'] for r in prepared]
        reference = np.load(meta['reference_object_motion'], allow_pickle=False)
        assert str(reference['protocol_id']) == 'AUX_REF_OBJECT'
        assert str(reference['source']) == 'published_fit_actual_native_samples_no_interpolation'
        np.testing.assert_array_equal(reference['times'], meta['timestamp_seconds'])
        seg_run_path = OUT / f'inputs/{dev}/sam_context_output/segmentation_run.json'
        seg_run = load(seg_run_path)
        assert seg_run['status'] == 'completed' and seg_run['reference_geometry_used'] is False
        command_path = OUT / f'inputs/{dev}/sam_command.json'
        command = load(command_path)
        for key in ('checkpoint', 'config', 'adapter'):
            assert seg_run[key + '_sha256'] == command['source_' + key + '_sha256']
        context_path = OUT / f'inputs/{dev}/sam_context_manifest.json'
        context = load(context_path)
        assert context['reference_used'] is False and context['all_context_RGB_not_additional_training_frames'] is True
        prompt_path = OUT / f'inputs/{dev}/sam_prompt.json'
        prompt = load(prompt_path)
        old_seg = load(context['source_old_segmentation'])
        assert prompt == old_seg['prompts'] == seg_run['prompts'], 'Original RGB prompt must remain identical'
        for key in ('checkpoint_sha256', 'config_sha256', 'adapter_sha256'):
            assert seg_run[key] == old_seg[key]
        frozen_dependencies += [artifact(p) for p in [seg_run_path, command_path, context_path, prompt_path]]
        check = cpu['devs'][dev]
        assert check['status'] == 'passed' and check['human_background_no_grad'] is True
        assert check['external_motion_gradient_detached'] is True
        source = check['source_identity']
        assert source['old_checkpoint_step'] == 8000 and source['old_obank_loaded'] is False
        assert source['reference_motion_reader'] is False and source['reference_interpolation'] is False
        assert source['source_code_sha256'] == aux_code['sha256']
        init_artifact = verified(source['init_path'], source['init_sha256'])
        checkpoint_artifact = verified(source['checkpoint_path'], source['checkpoint_sha256'])
        frozen_dependencies += [init_artifact, checkpoint_artifact]
        init = torch.load(source['init_path'], map_location='cpu', weights_only=False)
        assert init.get('reference_used') is False
        times = np.asarray(init['timestamps'], dtype=np.float64)
        assert np.all(np.diff(times) > 0)
        for index, (row, frame) in enumerate(zip(rr, prepared)):
            assert row['native_S_assets_ready'] and row['within_pred_interval']
            sid = row['official_capture_sample_id']
            immutable = identity_rows[sid]
            for key in ('camera0', 'object_parameters'):
                assert row['native_assets'][key]['sha256'] == immutable['native_assets'][key]['sha256']
                verified(row['native_assets'][key]['path'], row['native_assets'][key]['sha256'])
            assert frame['raw_sha256'] == row['native_assets']['camera0']['sha256']
            assert frame['object_parameters']['sha256'] == row['native_assets']['object_parameters']['sha256']
            assert frame['index'] == index and frame['query_time'] == meta['timestamp_seconds'][index]
            assert frame['sample_id'] == sid
            image = verified(frame['image'], frame['image_sha256'])
            assert meta['frame_paths'][index] == image['path'] and meta['frame_sha256'][index] == image['sha256']
            frozen_dependencies.append(image)
            label_hash = hashlib.sha256(np.ascontiguousarray(labels[index]).tobytes()).hexdigest()
            assert label_hash == frame['mask_sha256']
            assert int((labels[index] == 2).sum()) == frame['object_pixels']
            context_row = context['frames'][frame['segmentation_context_index']]
            assert context_row['source'] == 'native_RGB' and context_row['path'] == image['path']
            assert context_row['index'] == index and context_row['time'] == frame['query_time']
            conv = row['official_parameter_conversion_check']
            np.testing.assert_allclose(reference['R_world'][index], Rotation.from_rotvec(conv['angle_rotvec']).as_matrix(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(reference['t_world'][index], conv['translation_m'], rtol=0, atol=1e-12)
            query = float(frame['query_time'])
            assert times[0] <= query <= times[-1]
            right = int(np.searchsorted(times, query, side='left'))
            left = right if times[right] == query else right - 1
            dt = float(times[right] - times[left])
            row['pred_evaluation'] = dict(query_index=index, query_timestamp_seconds=query,
                left_index=left, right_index=right, left_time=float(times[left]), right_time=float(times[right]),
                interval_seconds=dt, right_weight=float((query - times[left]) / dt) if dt > 0 else 0.0,
                method=source['motion_rule'], source='old_final_S1_RGB_prediction',
                actual_native_RGB_exposure_time=False, extrapolated=False,
                checkpoint=checkpoint_artifact, initialization=init_artifact,
                CPU_interface_check=artifact(cpu_path), implementation=aux_code)
            row.update(selected_S=True, new_training_frame_index=index,
                       input_manifest=prepared_artifact, rectified_camera0=image,
                       reference_motion=reference_artifact,
                       mask=dict(path=meta['segmentation'], archive_sha256=segmentation_artifact['sha256'],
                                 array='entity_labels', row_index=index, frame_array_sha256=label_hash,
                                 shape=list(labels[index].shape), dtype=str(labels.dtype),
                                 object_pixels=frame['object_pixels'], source_sample_id=sid),
                       mask_status='verified_native_RGB_SAM2_original_frozen_frontend_and_prompt',
                       input_preparation_phase='complete', missing_reasons=[])
            if row['candidate_E']:
                erow = evaluation_rows[sid]
                assert erow['dev'] == dev and erow['query_time_seconds'] == query
                assert erow['input_S_row_identity']['new_frame_id'] == sid
                assert erow['native_camera1']['sha256'] == row['native_assets']['camera1']['sha256']
                assert erow['published_object_parameters']['sha256'] == row['native_assets']['object_parameters']['sha256']
                assert erow['same_capture_archive']['url'] == result['archive']['url']
                assert erow['same_capture_archive']['etag'] == result['archive']['etag']
                assert erow['official_sync_error_seconds'] is None
                for role in ('native_camera1', 'native_camera1_source', 'rgb', 'regions', 'overlay'):
                    record = erow[role]
                    frozen_dependencies.append(verified(record['path'], record['sha256']))
                for fit in erow['evaluation_fits']:
                    # Hash only provenance JSON here, not published human payloads.
                    assert fit['archive_member'].startswith(sid + '/') and fit['role'] == 'evaluation_region_only'
                    verified(fit['source']['path'], fit['source']['sha256'])
                    fit_source = load(fit['source']['path'])
                    assert fit_source['sha256'] == fit['sha256'] and fit_source['member'] == fit['archive_member']
                    assert fit_source['url'] == result['archive']['url'] and fit_source['etag'] == result['archive']['etag']
                assert erow['pixel_counts']['object'] > 0
                row.update(selected_E=True, evaluation_region=erow['regions'], rectified_camera1=erow['rgb'],
                           E_region_status='frozen_native_same_sample_evaluation_only',
                           E_regions_manifest=region_artifact, E_object_pixels=erow['pixel_counts']['object'])
        summary = result['devs'][dev]
        summary.update(final_S=len(rr), final_E=sum(r['selected_E'] for r in rr),
                       training_gate_passed=True, blockers=[], input_manifest=prepared_artifact,
                       segmentation=segmentation_artifact, reference_motion=reference_artifact)
        assert summary['final_S'] >= 3 and summary['final_E'] >= 2
    unique_dependencies = {r['path']: r for r in frozen_dependencies}
    if verify_only:
        old_dependencies = {r['path']: r for r in result['final_gate']['frozen_dependencies']}
        assert unique_dependencies == old_dependencies, 'Frozen dependency identities changed'
        print(json.dumps(dict(status='frozen_native_gate_verified', S=21, E=9,
                              native_availability_sha256=sha(native_path),
                              frame_manifest_sha256=sha(protocol / 'frame_manifest.csv'))))
        return
    if result.get('training_gate_passed'):
        raise RuntimeError('Gate already frozen; use --verify-frozen for a read-only check')
    # Extend the original complete candidate ledger, preserving all exclusion rows.
    csv_path = protocol / 'frame_manifest.csv'
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        old_columns = reader.fieldnames
        csv_rows = list(reader)
    by_id = {r['new_frame_id']: r for r in result['rows']}
    additions = ['new_training_frame_index', 'input_manifest_path', 'input_manifest_sha256',
                 'rectified_camera0_path', 'rectified_camera0_sha256', 'mask_array', 'mask_row_index',
                 'mask_frame_array_sha256', 'mask_object_pixels', 'reference_motion_path', 'reference_motion_sha256',
                 'rectified_camera1_path', 'rectified_camera1_sha256', 'evaluation_region_path',
                 'evaluation_region_sha256', 'E_regions_manifest_path', 'E_regions_manifest_sha256']
    columns = old_columns + [k for k in additions if k not in old_columns]
    for flat in csv_rows:
        row = by_id[flat['new_frame_id']]
        if not row['selected_S']:
            continue
        for key in ('selected_S', 'selected_E', 'mask_status', 'E_region_status', 'new_training_frame_index'):
            flat[key] = row[key]
        for role in ('input_manifest', 'rectified_camera0', 'reference_motion', 'rectified_camera1', 'evaluation_region', 'E_regions_manifest'):
            if role in row:
                flat[role + '_path'] = row[role]['path']
                flat[role + '_sha256'] = row[role]['sha256']
        flat.update(mask_path=row['mask']['path'], mask_sha256=row['mask']['archive_sha256'],
                    mask_array=row['mask']['array'], mask_row_index=row['mask']['row_index'],
                    mask_frame_array_sha256=row['mask']['frame_array_sha256'],
                    mask_object_pixels=row['mask']['object_pixels'],
                    pred_evaluation_json=json.dumps(row['pred_evaluation'], separators=(',', ':')),
                    missing_reasons='')
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    result.update(status='native_input_and_E_gate_frozen', training_gate=True, training_gate_passed=True,
                  script=artifact(__file__),
                  final_gate=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      total_S=21, total_E=9, all_candidates_accounted=True,
                      scope='native sample availability, legal RGB masks, frozen RGB motion CPU interface, same-sample E identity; GPU training preflight remains a separate required check',
                      regions_manifest=region_artifact, immutable_native_identity=snapshot_artifact,
                      CPU_scene_check=artifact(cpu_path), frame_manifest=artifact(csv_path),
                      frozen_dependencies=list(unique_dependencies.values()),
                      actual_exposure_times_unknown=True, official_sync_errors_unknown=True,
                      reference_interpolated=False, human_reference_training_used=False))
    native_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], S=21, E=9,
                         native_availability_sha256=sha(native_path), frame_manifest_sha256=sha(csv_path))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acquire', action='store_true', help='Copy verified local members and fetch missing allowlisted members only')
    ap.add_argument('--finalize', action='store_true', help='Bind both complete input manifests, verified prediction interface, and frozen native E')
    ap.add_argument('--regions-sha256', help='Independent final E manifest SHA256, required by --finalize')
    ap.add_argument('--verify-frozen', action='store_true', help='Read-only check of a previously finalized gate')
    args = ap.parse_args()
    if args.finalize or args.verify_frozen:
        assert not args.acquire
        return finalize_native_gate(args.regions_sha256, args.verify_frozen)
    existing = OUT / 'protocol/native_availability.json'
    if existing.exists() and load(existing).get('training_gate_passed'):
        raise RuntimeError('Refusing to overwrite a frozen gate; use --verify-frozen')
    ix = load(INDEX)
    entries = {e['name']: e for e in ix['entries']}
    rows, jobs, summaries = [], [], {}
    for name, dev in DEVS.items():
        im = load(ROOT / dev['input_manifest'])
        times = np.asarray(im['timestamp_seconds'], np.float64)
        low, high = float(times.min()), float(times.max())
        for t in range(math.floor(low), math.ceil(high) + 1):
            sample = f"{dev['sequence']}/t{t:04d}.000"
            members = dict(camera0=f'{sample}/k0.color.jpg', camera1=f'{sample}/k1.color.jpg',
                           object_parameters=f"{sample}/{dev['category']}/fit01/{dev['category']}_fit.pkl")
            in_range = low <= t <= high
            group_exists = sample + '/' in entries
            available = {k: m in entries for k, m in members.items()}
            requested_E = t in dev['E']
            can_S = in_range and all(available[k] for k in ('camera0', 'object_parameters'))
            can_E = can_S and requested_E and available['camera1']
            reasons = []
            if not in_range:
                reasons.append('native_nominal_group_time_outside_pred_observation_interval_no_extrapolation')
            if not group_exists:
                reasons.append('official_native_sample_group_absent')
            for k, ok in available.items():
                if not ok:
                    reasons.append('official_member_absent_' + k)
            if can_S:
                jobs.extend((dev, members[k]) for k in ('camera0', 'object_parameters'))
            if can_E:
                jobs.append((dev, members['camera1']))
            right = int(np.searchsorted(times, t, side='left'))
            if in_range:
                left = right if right < len(times) and times[right] == t else right - 1
                bracket = dict(left_index=left, right_index=right,
                               left_time=float(times[left]), right_time=float(times[right]),
                               interval_seconds=float(times[right] - times[left]),
                               requested_group_nominal_time=float(t),
                               method='pending_old_RGB_motion_interface_verification',
                               is_actual_native_RGB_exposure_time=False)
            else:
                bracket = None
            rows.append(dict(dev=name, sequence=dev['sequence'], new_frame_id=sample,
                             official_capture_sample_id=sample, native_frame_times_id=f't{t:04d}.000',
                             native_nominal_time_seconds=float(t), actual_camera0_exposure_seconds=None,
                             actual_camera1_exposure_seconds=None, official_synchronization_error_seconds=None,
                             native_association='same_official_Date01_ZIP_sample_directory',
                             official_group_exists=group_exists, within_pred_interval=in_range,
                             original_frame_id=None, old_input_manifest=artifact(ROOT / dev['input_manifest']),
                             members=members, official_member_available=available,
                             candidate_S=can_S, candidate_E=can_E, selected_S=False, selected_E=False,
                             predetermined_existing_E_requested=requested_E, missing_reasons=reasons,
                             pred_evaluation=bracket,
                             calibration=[artifact(p) for p in im['calibration_sources']],
                             reference_version='Date01.zip/object/fit01',
                             reference_interpolation=False, reference_nearest_assignment=False,
                             mask=None, mask_status='pending_native_RGB_SAM2_or_proven_exact_image_cache',
                             E_region_status='pending_native_image_region_binding' if can_E else 'not_E'))
        summaries[name] = dict(sequence=dev['sequence'], original_RGB_frames=len(times),
                               predicted_observation_interval_seconds=[low, high],
                               input_manifest=artifact(ROOT / dev['input_manifest']))
    jobs = list({m: (d, m) for d, m in jobs}.values())
    missing_jobs = [(d, m) for d, m in jobs if existing_source(d, m) is None]
    predicted_transfer = sum(entries[m]['compressed_size'] + 30 + len(m.encode()) for _, m in missing_jobs)
    assert predicted_transfer < 5 * 1024**2, 'Unexpected acquisition scope'
    failures = []
    if args.acquire:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fs = {pool.submit(materialize, d, m, ix): m for d, m in jobs}
            for future in concurrent.futures.as_completed(fs):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(dict(member=fs[future], error=str(exc)))
    import trimesh
    template_cache = {}
    for name, dev in DEVS.items():
        template_path = ROOT / dev['template']
        template = trimesh.load_mesh(template_path, process=False)
        vertices = np.asarray(template.vertices, np.float64)
        template_cache[name] = (vertices, vertices.mean(0))
        summaries[name]['template'] = artifact(template_path) | dict(
            vertex_count=len(vertices), center_m=vertices.mean(0).tolist(),
            centering='official_parse_obj_pose_numpy_mean_all_vertices_including_isolated', unit='metres')
    for row in rows:
        dev = DEVS[row['dev']]
        assets = {}
        for role, member in row['members'].items():
            p = existing_source(dev, member)
            if p:
                assets[role] = validate_member(p, member, ix)
                assets[role]['source'] = load(str(p) + '.source.json')
        row['native_assets'] = assets
        parameter_ok = False
        if 'object_parameters' in assets:
            try:
                angle, trans, keys = read_object_parameters(assets['object_parameters']['path'])
                vertices, center = template_cache[row['dev']]
                official_world = (vertices - center) @ Rotation.from_rotvec(angle).as_matrix().T + trans
                fit_member = row['members']['object_parameters'].replace('.pkl', '.ply')
                fit_path = ROOT / dev['source_root'] / fit_member
                conversion = dict(angle_rotvec=angle.tolist(), translation_m=trans.tolist(),
                                  parameter_keys=keys, rule='(V - mean(V)) @ Rotation.from_rotvec(angle).as_matrix().T + trans',
                                  coordinate_frame='BEHAVE Date01 Kinect1 color world; metres')
                if fit_path.exists():
                    published = np.asarray(trimesh.load_mesh(fit_path, process=False).vertices, np.float64)
                    assert published.shape == official_world.shape
                    err = published - official_world
                    sse = float(np.sum(err * err))
                    conversion.update(published_object_mesh_check=artifact(fit_path), squared_error_sum_m2=sse,
                                      max_vertex_error_m=float(np.linalg.norm(err, axis=-1).max()),
                                      official_assertion_sse_lt_1e_8=sse < 1e-8)
                    parameter_ok = sse < 1e-8
                else:
                    conversion['missing_check'] = 'published_object_fit_mesh_not_local'
                row['official_parameter_conversion_check'] = conversion
            except Exception as exc:
                row['missing_reasons'].append('object_parameter_conversion_failure:' + str(exc))
        row['native_S_assets_ready'] = row['candidate_S'] and 'camera0' in assets and parameter_ok
        row['native_E_assets_ready'] = row['native_S_assets_ready'] and row['candidate_E'] and 'camera1' in assets
        if row['candidate_S']:
            for role in ('camera0', 'object_parameters'):
                if role not in assets:
                    row['missing_reasons'].append('local_asset_missing_' + role)
            row['missing_reasons'] += ['native_RGB_mask_binding_pending', 'frozen_S1_motion_evaluation_pending']
        if row['candidate_E']:
            row['missing_reasons'].append('native_E_region_manifest_pending')
    for name, summary in summaries.items():
        rr = [r for r in rows if r['dev'] == name]
        summary.update(candidate_rows=len(rr), candidate_S=sum(r['candidate_S'] for r in rr),
                       candidate_E=sum(r['candidate_E'] for r in rr),
                       native_S_assets_ready=sum(r['native_S_assets_ready'] for r in rr),
                       native_E_assets_ready=sum(r['native_E_assets_ready'] for r in rr),
                       final_S=0, final_E=0, training_gate_passed=False,
                       blockers=['native_RGB_masks_and_hashes_not_yet_frozen',
                                 'RGB_only_S1_motion_native_time_evaluation_not_yet_frozen',
                                 'native_E_regions_and_image_identity_not_yet_frozen'])
    copied, fetched = {}, {}
    for row in rows:
        for item in row['native_assets'].values():
            if str(OUT / 'inputs/native') not in item['path']:
                continue
            source = item['source']
            target = copied if source.get('reused_from') else fetched
            target[source['member']] = source
    official_sources = ['tools/parse_obj_pose.py', 'tools/video2images.py', 'data/video_reader.py', 'data/frame_data.py']
    result = dict(protocol_id='AUX_REF_OBJECT', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status='raw_native_identity_audited_training_gate_pending', script=artifact(__file__),
                  archive_index=artifact(INDEX), archive=dict(url=ix['url'], etag=ix['etag'], bytes=ix['archive_bytes']),
                  native_identity_basis='Published sample directory binds camera0, camera1 and fit01 object parameters. Nominal frame label is not a measured exposure timestamp.',
                  synchronization=dict(official_algorithm='video2images picks one common closest-time anchor across Kinect controllers then selects each camera RGB at that anchor. Depth timestamps supply the default anchor.',
                                       release_exact_exposure_times_available=False, per_sample_error_seconds=None,
                                       old_camera0_to_raw_camera1_differences_are_not_sync_errors=True),
                  official_conversion_sources=[artifact(ROOT / 'data/BEHAVE/source_metadata/official_code' / p) for p in official_sources],
                  acquisition=dict(requested=args.acquire, allowed_unique_members=len(jobs), missing_members_before_run=len(missing_jobs),
                                   estimated_missing_range_bytes=predicted_transfer, failures=failures,
                                   isolated_native_local_copy_members=len(copied),
                                   isolated_native_fetched_members=len(fetched),
                                   cumulative_new_HTTP_range_bytes=sum(s['transferred_bytes'] for s in fetched.values()),
                                   no_human_depth_texture_or_fit_mask_downloads=True),
                  forbidden_shortcuts=['old_14_9_nearest_reference_slots_as_labels', 'high_frequency_fit01-smooth_silently_as_fit01',
                                       'reference_interpolation_or_extrapolation', 'nearest_Ref_copy', 'old_raw_video_E_image_identity_relabel'],
                  template_conversion_validation='Official object-only parameters reconstruct same-sample fit01 meshes without alignment; all template vertices included in mean.',
                  minimum_counts=dict(S=3, E=2), devs=summaries, rows=rows,
                  training_started=False, published_human_parameters_read=False,
                  historical_files_modified=False)
    protocol = OUT / 'protocol'
    protocol.mkdir(parents=True, exist_ok=True)
    (protocol / 'native_availability.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    columns = ['dev', 'sequence', 'new_frame_id', 'original_frame_id', 'native_frame_times_id',
               'native_nominal_time_seconds', 'actual_camera0_exposure_seconds', 'actual_camera1_exposure_seconds',
               'official_synchronization_error_seconds', 'native_association', 'within_pred_interval',
               'candidate_S', 'candidate_E', 'native_S_assets_ready', 'native_E_assets_ready', 'selected_S', 'selected_E',
               'camera0_path', 'camera0_sha256', 'camera1_path', 'camera1_sha256', 'object_parameters_path',
               'object_parameters_sha256', 'reference_version', 'mask_status', 'mask_path', 'mask_sha256',
               'E_region_status', 'calibration_json', 'pred_evaluation_json', 'missing_reasons']
    with (protocol / 'frame_manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k) for k in columns}
            for role, asset in row['native_assets'].items():
                flat[role + '_path'] = asset['path']
                flat[role + '_sha256'] = asset['sha256']
            flat['calibration_json'] = json.dumps(row['calibration'], separators=(',', ':'))
            flat['pred_evaluation_json'] = json.dumps(row['pred_evaluation'], separators=(',', ':'))
            flat['missing_reasons'] = ';'.join(row['missing_reasons'])
            writer.writerow(flat)
    print(json.dumps(dict(status=result['status'], devs=summaries, acquisition=result['acquisition']), indent=2))


if __name__ == '__main__':
    main()
