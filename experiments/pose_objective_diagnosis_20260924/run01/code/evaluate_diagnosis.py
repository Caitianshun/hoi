"""Evaluation-only entrypoint. Never imported by input solvers or selectors.

freeze-existing reads input assets only and freezes the finite historical pool.
oracle scores complete frozen paths, never constructs a per-frame oracle path.
paired requires an explicit freeze containing every final optimization output.
"""
from pathlib import Path
import argparse
import csv
import datetime
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parents[1] / 'object_pose_refinement_20260924/run01'
SPEC = importlib.util.spec_from_file_location('historical_pose_evaluator', OLD / 'code/evaluate_pose_pairs.py')
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=E.sha(path), bytes=path.stat().st_size)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    E.save(Path(path), value)


def csvwrite(path, rows):
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def freeze_existing():
    dest = ROOT / 'evaluation/historical_pool_freeze.json'
    assert not dest.exists(), 'Existing freeze is immutable; do not overwrite'
    audit = read(OLD / 'protocol/audit_inputs_manifest.json')
    frozen = dict(created_utc=now(), input_only_selection_complete=True,
                  no_independent_reference_read=True, purpose='evaluation-only finite historical pool; no new input selection',
                  symmetry_protocol_sha256=E.sha(OLD / 'protocol/symmetry_geometry_only.json'),
                  script=identity(__file__), historical_evaluator=identity(OLD / 'code/evaluate_pose_pairs.py'),
                  frozen_predictions=[], source_files=[], dev={})
    for dev in ('dev1', 'dev2'):
        old_init = Path(audit['old_S1'][dev]['object_init'])
        ini = np.load(old_init)
        C, ts = ini['c2w'].astype(float), ini['timestamp_seconds']
        base = OLD / f'pose/{dev}/optimization'
        selected = read(base / 'selection_frozen.json')['selected']['path']
        predictions = {'original_initialization': str(old_init)}
        frozen['source_files'] += [identity(old_init), identity(base / 'selection_frozen.json'),
                                   identity(OLD / f'pose/{dev}/tracklets_summary.json')]
        paths = []
        for p in sorted(base.glob('path*')):
            if not p.is_dir():
                continue
            initial = p / 'initial_path.npz'
            final = p / 'object_init.npz'
            a = np.load(initial)
            # Fixed calibration conversion; no pose fitting, alignment or reference read.
            rw = np.einsum('ij,tjk->tik', C[:3, :3], a['R_camera'])
            tw = a['t_camera_m'] @ C[:3, :3].T + C[:3, 3]
            normalized = ROOT / f'evaluation/frozen_input_adapters/{dev}/{p.name}_before.npz'
            normalized.parent.mkdir(parents=True, exist_ok=True)
            assert not normalized.exists()
            np.savez_compressed(normalized, R_world=rw, t_world_m=tw, timestamp_seconds=ts)
            corr = read(p / 'canonical_correspondences.json')
            score = read(p / 'input_score.json')
            row = dict(path=p.name, before=identity(initial), before_world_adapter=identity(normalized),
                       after=identity(final), path_start=identity(p / 'path_start.json'),
                       correspondences=identity(p / 'canonical_correspondences.json'),
                       input_score=score, input_score_file=identity(p / 'input_score.json'),
                       optimizer=read(p / 'optimizer.json'),
                       canonical_queries=corr['accepted_queries'], temporal_records=corr['temporal_records'],
                       raw_2d_records=corr['raw_track_observation_count'],
                       self_definition_records=corr['self_definition_records'],
                       historical_selected=(p.name == selected))
            paths.append(row)
            predictions[p.name + '_before'] = str(normalized)
            predictions[p.name + '_after'] = str(final)
            frozen['source_files'] += [identity(initial), identity(final), identity(p / 'path_start.json'),
                                       identity(p / 'canonical_correspondences.json'), identity(p / 'input_score.json'),
                                       identity(p / 'optimizer.json')]
        candidate_path = OLD / f'megapose/predictions/{dev}/candidates.npz'
        candidate = np.load(candidate_path)
        candidate_rows = [dict(frame=int(fr), candidate_index=k,
                               input_timestamp_seconds=float(candidate['timestamp_seconds'][j]),
                               historical_official_score=float(candidate['scores'][j, k]))
                          for j, fr in enumerate(candidate['frame_indices'])
                          for k in range(candidate['R_camera'].shape[1])]
        frozen['source_files'].append(identity(candidate_path))
        frozen['dev'][dev] = dict(predictions=predictions, complete_paths=paths,
                                  selected_label=selected + '_after',
                                  single_keyframe_candidates=dict(file=identity(candidate_path), rows=candidate_rows),
                                  adapter='R_world=C_R@R_camera; t_world=C_R@t_camera+C_t; original full timestamps')
        frozen['frozen_predictions'] += [dict(label=label, dev=dev, **identity(path)) for label, path in predictions.items()]
    write(dest, frozen)
    print(json.dumps(dict(status='frozen_without_reference_read', path=str(dest), sha256=E.sha(dest))))


def validate_freeze(frozen):
    assert frozen['input_only_selection_complete'] is True
    assert frozen['symmetry_protocol_sha256'] == E.sha(OLD / 'protocol/symmetry_geometry_only.json')
    for row in frozen.get('source_files', []) + frozen['frozen_predictions']:
        assert E.sha(row['path']) == row['sha256'], row['path']


def evaluate(dev, predictions, freeze_path, output):
    frozen = read(freeze_path)
    validate_freeze(frozen)
    # The old evaluator checks all prediction path/hash identities before opening references.
    E.evaluate(dev, {k: Path(v) for k, v in predictions.items()}, output, freeze_path)
    result = read(output / 'paired_results.json')
    track = read(OLD / f'pose/{dev}/tracklets_summary.json')
    vis = set(read(OLD / f'pose/{dev}/optimization/path00_old_start/input_score.json')['fixed_visible_frames'])
    motion = []
    for row in result['rows']:
        f = row['input_frame']
        row['depth_absolute_error_cm'] = abs(row['depth_bias_cm']) if row['depth_bias_cm'] is not None else None
        row['input_2d_reliable_record_count'] = track['adopted_per_frame'][f]
        row['low_2d_observation_lt6'] = track['adopted_per_frame'][f] < 6
        row['input_fixed_visible_set'] = f in vis
        row['retained_without_difficulty_filter'] = True
    for label, summary in result['summaries'].items():
        intervals = summary['reference_interval_motion']
        for r in intervals:
            r['relative_angular_velocity_error_deg_s'] = (r['relative_rotation_error_deg'] / r['dt_seconds']
                                                          if r['relative_rotation_error_deg'] is not None else None)
            r['posthoc_known_chair_64_to_74'] = dev == 'dev2' and r['start_input_frame'] == 64 and r['end_input_frame'] == 74
            motion.append(dict(method=label, **r))
        for metric in ('centre_displacement_error_cm', 'relative_rotation_error_deg',
                       'centre_velocity_error_cm_s', 'relative_angular_velocity_error_deg_s'):
            summary[metric] = E.stats([np.nan if r[metric] is None else r[metric] for r in intervals])
        signed = summary['camera_depth_signed_bias_cm']['mean']
        summary['absolute_mean_signed_depth_bias_cm'] = abs(signed) if signed is not None else None
    result['diagnostic_extension'] = dict(script=identity(__file__), freeze=identity(freeze_path),
        retains_all_reference_slots=True, no_reference_interpolation=True,
        angular_velocity_metric='relative rotation discrepancy divided by actual interval duration; finite-interval proxy',
        low_observation_definition='fewer than 6 accepted 2D nonself records, input-only; never filters evaluation')
    write(output / 'diagnostic_results.json', result)
    csvwrite(output / 'diagnostic_per_slot.csv', result['rows'])
    csvwrite(output / 'diagnostic_per_interval.csv', motion)
    validate_freeze(frozen)
    return result


METRICS = ('centroid_error_cm', 'camera_depth_absolute_error_cm', 'raw_rotation_error_deg',
           'symmetry_rotation_error_deg', 'corresponding_vertex_rmse_cm',
           'centre_displacement_error_cm', 'relative_rotation_error_deg',
           'centre_velocity_error_cm_s', 'relative_angular_velocity_error_deg_s')


def oracle():
    freeze_path = ROOT / 'evaluation/historical_pool_freeze.json'
    frozen = read(freeze_path)
    validate_freeze(frozen)
    output = dict(created_utc=now(), purpose='evaluation-only candidate best reachable diagnostic',
                  freeze=identity(freeze_path), oracle_never_exported_as_prediction=True,
                  per_frame_candidate_selection=False, reference_used_to_select_inputs=False,
                  dev={}, limitations=[
                      'Finite saved candidate pool only; neither deployable result nor reconstruction upper bound.',
                      'All rankings choose one whole path per metric, without per-frame stitching.',
                      'Historical actual selection compared descriptively; same-pool input rescoring is a separate experiment.',
                      'Geometric fitted reference is approximate; material identity accuracy is unavailable.'])
    csv_rows = []
    for dev in ('dev1', 'dev2'):
        spec = frozen['dev'][dev]
        results = evaluate(dev, spec['predictions'], freeze_path, ROOT / f'evaluation/oracle_{dev}')
        summaries = results['summaries']
        chosen = spec['selected_label']
        # Single-keyframe labels require exact identity with the already frozen reference matching.
        representative = [r for r in results['rows'] if r['method'] == chosen]
        frame_to_reference = {r['input_frame']: r for r in representative if r['reference_available']}
        candidate_rows = []
        for r in spec['single_keyframe_candidates']['rows']:
            match = frame_to_reference.get(r['frame'])
            candidate_rows.append(dict(**r, reference_status='same_frozen_reference_slot' if match else 'N/A',
                                       reference_slot=match['slot'] if match else None,
                                       note='No interpolation or nearest-time rematching; excluded from full-path oracle'))
        groups = dict(optimized_paths_only=[k for k in summaries if k.endswith('_after')],
                      all_saved_before_after_and_original=list(summaries))
        comparisons = []
        for group, names in groups.items():
            # Enforce identical full coverage before comparing means; no missing-frame advantage.
            for n in names:
                assert summaries[n]['prediction_slots_valid'] == summaries[chosen]['reference_available']
                assert summaries[n]['full_prediction_frames_valid'] == summaries[n]['full_prediction_frames_requested']
            for metric in METRICS:
                best = min(names, key=lambda x: summaries[x][metric]['mean'])
                value = summaries[best][metric]['mean']
                actual = summaries[chosen][metric]['mean']
                row = dict(dev=dev, pool=group, metric=metric, best_complete_path=best,
                           best_mean=value, historical_selected_path=chosen, historical_selected_mean=actual,
                           gap_selected_minus_best=actual - value, reference_count=summaries[best][metric]['count'],
                           deployable=False)
                comparisons.append(row)
                csv_rows.append(row)
        output['dev'][dev] = dict(comparisons=comparisons, candidate_reference_coverage=candidate_rows,
                                  historical_selection=chosen, complete_path_ledger=spec['complete_paths'],
                                  full_evaluation=str(ROOT / f'evaluation/oracle_{dev}/diagnostic_results.json'),
                                  accepted_symmetries=results['accepted_symmetry_names'])
    validate_freeze(frozen)
    write(ROOT / 'evaluation/oracle_diagnostic.json', output)
    csvwrite(ROOT / 'evaluation/oracle_diagnostic.csv', csv_rows)
    lines = ['# 已存完整路径的仅评价诊断', '',
             '所有源路径在读取本轮参考前冻结。每个指标选整条轨迹；不逐帧拼接、不插值参考、不输出参考选出的初始化。', '',
             '| 事件 | 候选池 | 指标 | 最佳完整路径 | 最佳均值 | 历史选择均值 | 差距 |',
             '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for r in csv_rows:
        if r['metric'] not in ('centroid_error_cm', 'camera_depth_absolute_error_cm', 'raw_rotation_error_deg',
                               'centre_displacement_error_cm', 'relative_rotation_error_deg'):
            continue
        lines.append(f"| {r['dev']} | {r['pool']} | {r['metric']} | {r['best_complete_path']} | {r['best_mean']:.4f} | {r['historical_selected_mean']:.4f} | {r['gap_selected_minus_best']:.4f} |")
    lines += ['', '原参考槽及困难槽全部保留。中心采用全部模板顶点均值，包括历史孤立顶点；对称集合只包含恒等。位移和相对旋转为相邻参考时刻的运动误差；速度使用真实 dt。',
              '', '带符号深度偏差保留在逐槽表和汇总；oracle采用绝对深度误差，避免把更负的偏差误当更好。',
              '', '这些开发事件已多次用于诊断，不属于盲测。完整逐槽、逐间隔、覆盖与低观测标记见 oracle_dev1/ 和 oracle_dev2/。']
    (ROOT / 'evaluation/oracle_diagnostic.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='oracle_completed', path=str(ROOT / 'evaluation/oracle_diagnostic.json'))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze-existing', 'oracle', 'paired'])
    p.add_argument('--dev', choices=['dev1', 'dev2'])
    p.add_argument('--freeze', type=Path)
    p.add_argument('--prediction', action='append', default=[], help='LABEL=/absolute/world_pose.npz')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.mode == 'freeze-existing':
        freeze_existing()
    elif a.mode == 'oracle':
        oracle()
    else:
        assert a.dev and a.freeze and a.prediction and a.output
        frozen = read(a.freeze)
        assert frozen.get('all_required_optimization_outputs_frozen') is True, 'Freeze all 35/140 outputs before evaluation'
        evaluate(a.dev, dict(x.split('=', 1) for x in a.prediction), a.freeze, a.output)


if __name__ == '__main__':
    main()
