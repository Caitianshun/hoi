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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acquire', action='store_true', help='Copy verified local members and fetch missing allowlisted members only')
    args = ap.parse_args()
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
