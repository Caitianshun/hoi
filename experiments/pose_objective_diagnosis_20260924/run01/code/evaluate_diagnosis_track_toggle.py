"""Adapter from the solver's completed input-only freeze to paired evaluation."""
from pathlib import Path
import json
import numpy as np
import evaluate_diagnosis as D


def main():
    root = D.ROOT
    run = D.read(root / 'track_toggle/run.json')
    assert run['status'] == 'completed_input_outputs_frozen'
    assert run['all_outputs_before_independent_evaluation'] is True and run['reference_used'] is False
    assert run['replay']['passed'] is True
    names = ('on35', 'off35', 'on140', 'off140') if run['continuation']['continue_both'] else ('on35', 'off35')
    audit = D.read(D.OLD / 'protocol/audit_inputs_manifest.json')
    predictions = dict(old_initialization=audit['old_S1']['dev1']['object_init'],
                       historical_P1=str(D.OLD / 'pose/dev1/optimization/path00_old_start/object_init.npz'))
    sources = [D.identity(root / 'track_toggle/run.json'), D.identity(root / 'track_toggle/continuation_decision.json')]
    for name in names:
        row = run['branches'][name]
        assert row['status'] == 'frozen'
        assert D.E.sha(row['pose_path']) == row['pose_sha256']
        predictions[name] = row['pose_path']
        sources.append(D.identity(root / f'track_toggle/{name}/optimizer.json'))
    freeze = dict(created_utc=D.now(), input_only_selection_complete=True,
                  all_required_optimization_outputs_frozen=True, independent_evaluation_not_used_to_stop_solver=True,
                  required_branches=list(names), continuation_decision=run['continuation'],
                  symmetry_protocol_sha256=D.E.sha(D.OLD / 'protocol/symmetry_geometry_only.json'),
                  frozen_predictions=[dict(label=k, **D.identity(v)) for k, v in predictions.items()],
                  source_files=sources)
    freeze_path = root / 'evaluation/track_toggle_evaluation_freeze.json'
    assert not freeze_path.exists()
    D.write(freeze_path, freeze)
    result = D.evaluate('dev1', predictions, freeze_path, root / 'evaluation/track_toggle_dev1')
    summaries = result['summaries']
    rows = []
    for metric in D.METRICS + ('camera_depth_signed_bias_cm',):
        a, b = summaries['on35'][metric]['mean'], summaries['off35'][metric]['mean']
        rows.append(dict(metric=metric, on35=a, off35=b, off_minus_on=b-a,
                         old_initialization=summaries['old_initialization'][metric]['mean']))
    def vals(method, key):
        return np.array([r[key] for r in result['rows'] if r['method'] == method], float)
    paired_direction = {}
    for key in ('centroid_error_cm', 'depth_absolute_error_cm', 'raw_rotation_error_deg'):
        delta = vals('off35', key) - vals('on35', key)
        paired_direction[key] = dict(improved_slots=int((delta < -1e-10).sum()),
                                     degraded_slots=int((delta > 1e-10).sum()),
                                     equal_slots=int((abs(delta) <= 1e-10).sum()), total_slots=len(delta))
    output = dict(status='completed', all_reference_slots_retained=True,
                  evaluation_result=str(root / 'evaluation/track_toggle_dev1/diagnostic_results.json'),
                  metrics=rows, paired_direction=paired_direction,
                  solver_termination={k:run['branches'][k] for k in names},
                  interpretation='Fixed original path and 365 frozen correspondences only; no statement that tracks universally help or harm.')
    D.write(root / 'evaluation/track_toggle_comparison.json', output)
    D.csvwrite(root / 'evaluation/track_toggle_comparison.csv', rows)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
