"""Independent post-freeze evaluation of the two-event surface/pose 2x2.

Never imported by the solver. Requires an explicit all-eight input-only freeze.
No reference is opened by this module import or by --check-freeze.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parents[1] / 'object_pose_refinement_20260924/run01'
SPEC = importlib.util.spec_from_file_location('historical_pose_evaluator', OLD / 'code/evaluate_pose_pairs.py')
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)
VARIANTS = ('F00', 'F01', 'F10', 'F11')
DEVS = ('dev1', 'dev2')
CM = ('centroid_error_cm', 'camera_depth_absolute_error_cm', 'centre_displacement_error_cm')
DEG = ('raw_rotation_error_deg', 'relative_rotation_error_deg')
PRIMARY = ('centroid_error_cm', 'centre_displacement_error_cm')
REPORT_METRICS = CM + DEG + ('camera_depth_signed_bias_cm', 'symmetry_rotation_error_deg',
                            'corresponding_vertex_rmse_cm', 'centre_velocity_error_cm_s',
                            'relative_angular_velocity_error_deg_s')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return E.sha(path)


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvwrite(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def validate_all_outputs(path):
    """Only input manifests and predicted assets; no evaluator/reference calls."""
    frozen = read(path)
    assert frozen.get('all_eight_outputs_frozen') is True, 'Need all eight outputs, not one event at a time'
    assert frozen.get('reference_used') is False, 'Input-only completion must be explicit'
    rules = ROOT / 'protocol/promotion_rules.json'
    assert frozen['promotion_rules_sha256'] == sha(rules)
    files = frozen['files']
    identities = {str(Path(r['path']).resolve()): r['sha256'] for r in files}
    for source, expected in identities.items():
        assert sha(source) == expected, source
    expected_paths = []
    for dev in DEVS:
        for arm in VARIANTS:
            for name in ('object_init.npz', 'metrics.json'):
                p = ROOT / dev / arm / name
                assert str(p.resolve()) in identities, 'Missing from all-eight freeze: ' + str(p)
                expected_paths.append(p)
    checks = frozen.get('implementation_checks', {})
    if 'implementation_checks_file' in frozen:
        src = frozen['implementation_checks_file']
        assert sha(src['path']) == src['sha256']
        preliminary = read(src['path'])
        # Final eight-output invariants override the explicitly pending preflight fields.
        checks = {**preliminary.get('checks', preliminary), **checks}
    return frozen, read(rules), checks, expected_paths


def stats(values):
    return E.stats([np.nan if v is None else v for v in values])


def tolerance(metric, control):
    return max(2. if metric in DEG else .5, .05 * control)


def augment(result):
    intervals = []
    for row in result['rows']:
        row['depth_absolute_error_cm'] = abs(row['depth_bias_cm']) if row['depth_bias_cm'] is not None else None
        row['retained_without_difficulty_filter'] = True
    for label, summary in result['summaries'].items():
        mm = summary['reference_interval_motion']
        for row in mm:
            r = row['relative_rotation_error_deg']
            row['relative_angular_velocity_error_deg_s'] = r / row['dt_seconds'] if r is not None else None
            row['posthoc_known_chair_64_to_74'] = result['dev'] == 'dev2' and row['start_input_frame'] == 64 and row['end_input_frame'] == 74
            intervals.append(dict(dev=result['dev'], method=label, **row))
        for metric in ('centre_displacement_error_cm', 'relative_rotation_error_deg',
                       'centre_velocity_error_cm_s', 'relative_angular_velocity_error_deg_s'):
            summary[metric] = stats([r[metric] for r in mm])
    result['diagnostic_extension'] = dict(script=identity(__file__), alignment='none',
        centre='all original indexed vertices including isolated vertices',
        time='actual input timestamps, no reference interpolation',
        angular_velocity='finite-interval relative-rotation discrepancy divided by dt; diagnostic proxy',
        gaussian_surface_and_heldout_camera1='N/A: no new Gaussian reconstruction at this stage')
    return result, intervals


def flatten_numbers(obj, prefix=''):
    """Keep input residual/coverage/q diagnostics traceable without recoding solver formulas."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from flatten_numbers(value, prefix + ('.' if prefix else '') + str(key))
    elif isinstance(obj, (int, float, bool)) or obj is None:
        yield prefix, obj


def robust_review(metrics):
    """Expected diagnostic shape: robust.initial/final.<family>.component_fraction/edge_fraction.

    Missing schema is explicit, never interpreted as zero rejection.
    """
    data = metrics.get('robust', metrics.get('robust_downweight', metrics.get('robust_downweighting')))
    if data is None and all(stage in metrics for stage in ('initial','final')):
        # Exact schema adapter for joint_solver: active switches are not applied to these diagnostic residuals.
        data = {stage: {'opt': {k:v for k,v in metrics[stage].get('residuals',{}).get('opt',{}).items()
                               if k.startswith('robust_') and k.endswith('_fraction')}} for stage in ('initial','final')}
    if not isinstance(data, dict) or not isinstance(data.get('initial'), dict) or not isinstance(data.get('final'), dict):
        return dict(status='not_evaluable', flags=[], reason='initial/final robust fractions missing from known schema')
    a = dict(flatten_numbers(data['initial']))
    b = dict(flatten_numbers(data['final']))
    rows = []
    for key in sorted(set(a) & set(b)):
        if 'fraction' not in key:
            continue
        x, y = a[key], b[key]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        assert 0 <= x <= 1 and 0 <= y <= 1, (key, x, y)
        rows.append(dict(component=key, initial=x, final=y, change=y-x,
                         flagged=bool(y > x + .05 or y > .5)))
    if not rows:
        return dict(status='not_evaluable', flags=[], reason='no explicitly named fraction fields')
    return dict(status='review_required' if any(x['flagged'] for x in rows) else 'passed',
                flags=[x for x in rows if x['flagged']], all_fractions=rows)


def implementation_gate(checks, rules):
    expected = rules['implementation_gate']['must_pass']
    # Parent should give exact semantic booleans or records with an explicit passed bool.
    source = checks.get('checks', checks)
    values = {}
    for key in expected:
        value = source.get(key)
        if isinstance(value, dict):
            value = value.get('passed')
        values[key] = value if isinstance(value, bool) else None
    return dict(passed=all(v is True for v in values.values()), checks=values,
                failed=[k for k,v in values.items() if v is False],
                missing=[k for k,v in values.items() if v is None])


def compare_geometry(all_results, candidate, control, metrics=CM+DEG):
    rows = []
    pass_all = True
    for dev in DEVS:
        summaries = all_results[dev]['summaries']
        for key in metrics:
            a, b = summaries[control][key], summaries[candidate][key]
            x, y = a['mean'], b['mean']
            valid = x is not None and y is not None and a['count'] == b['count']
            tol = tolerance(key,x) if valid else None
            passed = bool(valid and y-x <= tol + 1e-12)
            pass_all &= passed
            rows.append(dict(dev=dev,candidate=candidate,control=control,metric=key,
                             control_mean=x,candidate_mean=y,candidate_minus_control=y-x if valid else None,
                             tolerance=tol,count_control=a['count'],count_candidate=b['count'],passed=passed))
    return dict(passed=bool(pass_all), rows=rows)


def primary_improvement(all_results,candidate,control):
    rows=[]
    supported=[]
    for key in PRIMARY:
        passes=[]
        for dev in DEVS:
            ss=all_results[dev]['summaries']
            x,y=ss[control][key]['mean'],ss[candidate][key]['mean']
            valid=x is not None and y is not None
            tol=tolerance(key,x) if valid else None
            passed=bool(valid and x-y>tol)
            passes.append(passed)
            rows.append(dict(dev=dev,metric=key,control=control,candidate=candidate,
                             improvement=x-y if valid else None,required_strictly_greater_than=tol,passed=passed))
        if all(passes):supported.append(key)
    return dict(passed=bool(supported),same_primary_metrics_supported_on_both_devs=supported,rows=rows)


def compare_coverage(solver_metrics,candidate,control):
    rows=[]
    for dev in DEVS:
        cc=coverage_stage(solver_metrics[dev][control],'final')
        ca=coverage_stage(solver_metrics[dev][candidate],'final')
        x,y=cc.get('effective_fraction'),ca.get('effective_fraction')
        valid=isinstance(x,(int,float)) and isinstance(y,(int,float))
        if valid:assert 0<=x<=1 and 0<=y<=1
        rows.append(dict(dev=dev,control=control,candidate=candidate,control_effective_fraction=x,
                         candidate_effective_fraction=y,drop=x-y if valid else None,
                         allowed_drop=.05,passed=bool(valid and x-y<=.05+1e-12)))
    return dict(passed=all(r['passed'] for r in rows),rows=rows)


def coverage_stage(metrics,stage,split='opt'):
    if 'coverage' in metrics:
        data=metrics['coverage'].get(stage,{})
        return data.get(split,data)
    return metrics.get(stage,{}).get('coverage',{}).get(split,{})


def tail_review(all_results,candidate,control,metrics=CM+DEG):
    rows=[]
    for dev in DEVS:
        ss=all_results[dev]['summaries']
        for metric in metrics:
            for statistic in ('p90','max'):
                x,y=ss[control][metric][statistic],ss[candidate][metric][statistic]
                valid=x is not None and y is not None
                tol=tolerance(metric,x) if valid else None
                rows.append(dict(dev=dev,candidate=candidate,control=control,metric=metric,statistic=statistic,
                                 control_value=x,candidate_value=y,tolerance=tol,
                                 flagged=bool(not valid or y-x>tol)))
    flags=[r for r in rows if r['flagged']]
    return dict(status='review_required' if flags else 'passed',flags=flags,rows=rows)


def decision(all_results,solver_metrics,checks,rules):
    implementation=implementation_gate(checks,rules)
    arm={}
    for candidate in ('F01','F10','F11'):
        ref_full=all(all_results[d]['summaries'][candidate]['prediction_slots_valid']==all_results[d]['summaries'][candidate]['reference_available']
                     and all_results[d]['summaries'][candidate]['full_prediction_frames_valid']==all_results[d]['summaries'][candidate]['full_prediction_frames_requested'] for d in DEVS)
        geom=compare_geometry(all_results,candidate,'F00')
        old=compare_geometry(all_results,candidate,'old_initialization',CM)
        primary=primary_improvement(all_results,candidate,'F00')
        coverage=compare_coverage(solver_metrics,candidate,'F00')
        tail=tail_review(all_results,candidate,'F00')
        oldtail=tail_review(all_results,candidate,'old_initialization',CM)
        robust={d:robust_review(solver_metrics[d][candidate]) for d in DEVS}
        numeric=all([implementation['passed'],ref_full,geom['passed'],old['passed'],primary['passed'],coverage['passed']])
        review_required=(tail['status']!='passed' or oldtail['status']!='passed' or any(x['status']!='passed' for x in robust.values()))
        arm[candidate]=dict(numerical_base_gate_passed=numeric,eligible_without_additional_review=bool(numeric and not review_required),
                            implementation=implementation,full_reference_and_prediction_coverage=ref_full,
                            F00_geometry=geom,old_initialization_geometry=old,same_primary_improvement=primary,
                            effective_coverage=coverage,long_tail_vs_F00=tail,long_tail_vs_old=oldtail,
                            robust=robust,review_required=review_required)
    simple=next((n for n in ('F01','F10') if arm[n]['eligible_without_additional_review']),None)
    chosen=simple
    extra=None
    if arm['F11']['eligible_without_additional_review']:
        if simple is None:
            chosen='F11'
        else:
            geom=compare_geometry(all_results,'F11',simple)
            primary=primary_improvement(all_results,'F11',simple)
            cover=compare_coverage(solver_metrics,'F11',simple)
            tail=tail_review(all_results,'F11',simple)
            passed=all([geom['passed'],primary['passed'],cover['passed'],tail['status']=='passed'])
            extra=dict(control=simple,geometry=geom,primary_improvement=primary,coverage=cover,long_tail=tail,passed=passed)
            if passed:chosen='F11'
    return dict(created_utc=now(),rules=identity(ROOT/'protocol/promotion_rules.json'),variants=arm,
                preferred_simple_variant=simple,F11_additional_comparison=extra,selected_global_variant=chosen,
                high_level='eligible_for_two_gaussian_runs' if chosen else ('review_required_no_automatic_training' if any(a['numerical_base_gate_passed'] for a in arm.values()) else 'reject_all_no_new_gaussian_training'),
                same_variant_for_both_events=True,new_gaussian_runs_executed_by_evaluator=0,
                no_statistical_significance_claim=True,no_reference_selected_checkpoints_or_per_frame_paths=True)


def factorial(all_results):
    contrasts={'H1_fixed_q':('F01','F00'),'H1_adjustable_q':('F11','F10'),
               'H2_image_off':('F10','F00'),'H2_image_on':('F11','F01')}
    rows=[]
    for dev in DEVS:
        ss=all_results[dev]['summaries']
        for metric in REPORT_METRICS:
            for name,(candidate,control) in contrasts.items():
                y,x=ss[candidate][metric]['mean'],ss[control][metric]['mean']
                rows.append(dict(dev=dev,contrast=name,metric=metric,candidate=candidate,control=control,
                                 candidate_mean=y,control_mean=x,
                                 difference=y-x if x is not None and y is not None else None))
            v=[ss[a][metric]['mean'] for a in ('F00','F01','F10','F11')]
            dd=v[3]-v[2]-v[1]+v[0] if all(x is not None for x in v) else None
            rows.append(dict(dev=dev,contrast='difference_in_differences',metric=metric,candidate='F11-F10',control='F01-F00',
                             candidate_mean=None,control_mean=None,difference=dd))
    return dict(contrasts=contrasts,rows=rows,
                interpretation='Negative differences mean lower error except signed depth bias; negative difference-in-differences is descriptive additive interaction, not statistical synergy.',
                events_are_development_not_blind=True)


def evaluate(freeze_path):
    start=time.perf_counter()
    frozen,rules,checks,expected_paths=validate_all_outputs(freeze_path)
    output=ROOT/'evaluation'
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'paired_metrics.json').exists(),'Preserve previous evaluation; no overwrite'
    # Evaluation adapters are created only after the global eight-output gate above.
    audit=read(OLD/'protocol/audit_inputs_manifest.json')
    adapter=dict(created_utc=now(),input_only_selection_complete=True,
                 all_eight_outputs_frozen=True,all_outputs_source_freeze=identity(freeze_path),
                 symmetry_protocol_sha256=sha(OLD/'protocol/symmetry_geometry_only.json'),frozen_predictions=[])
    predictions={}
    solver_metrics={}
    for dev in DEVS:
        selected=read(OLD/f'pose/{dev}/optimization/selection_frozen.json')['selected']['path']
        predictions[dev]={'old_initialization':Path(audit['old_S1'][dev]['object_init']),
                          'historical_P1':OLD/f'pose/{dev}/optimization/{selected}/object_init.npz'}
        solver_metrics[dev]={}
        for arm in VARIANTS:
            predictions[dev][arm]=ROOT/dev/arm/'object_init.npz'
            solver_metrics[dev][arm]=read(ROOT/dev/arm/'metrics.json')
        adapter['frozen_predictions'] += [dict(dev=dev,label=label,**identity(path)) for label,path in predictions[dev].items()]
    adapter_path=output/'all_predictions_evaluation_freeze.json'
    save(adapter_path,adapter)
    all_results={};summary_rows=[];slot_rows=[];interval_rows=[];input_rows=[]
    for dev in DEVS:
        dest=output/dev
        E.evaluate(dev,predictions[dev],dest,adapter_path)
        result,intervals=augment(read(dest/'paired_results.json'))
        all_results[dev]=result
        slot_rows += [dict(dev=dev,**r) for r in result['rows']]
        interval_rows += intervals
        save(dest/'augmented_results.json',result)
        for label,summary in result['summaries'].items():
            row=dict(dev=dev,method=label,reference_slots_requested=summary['reference_requested'],
                     reference_slots_available=summary['reference_available'],reference_slots_scored=summary['prediction_slots_valid'],
                     reference_intervals_requested=summary['centre_displacement_error_cm']['requested_count'],
                     reference_intervals_scored=summary['centre_displacement_error_cm']['count'],
                     prediction_frames_requested=summary['full_prediction_frames_requested'],prediction_frames_valid=summary['full_prediction_frames_valid'])
            for key in REPORT_METRICS:
                for stat in ('mean','median','p90','max'):
                    row[key+'_'+stat]=summary[key][stat]
            summary_rows.append(row)
        for arm in VARIANTS:
            for key,value in flatten_numbers(solver_metrics[dev][arm]):
                input_rows.append(dict(dev=dev,method=arm,metric=key,value=value,
                                       source=str(ROOT/dev/arm/'metrics.json')))
    fac=factorial(all_results)
    promotion=decision(all_results,solver_metrics,checks,rules)
    fixed_case=[r for r in interval_rows if r['posthoc_known_chair_64_to_74']]
    coverage={d:{a:{stage:{split:coverage_stage(solver_metrics[d][a],stage,split) for split in ('opt','heldout')}
                        for stage in ('initial','final')} for a in VARIANTS} for d in DEVS}
    # Preserve the original solver metrics verbatim: opt/heldout, q motion, boundaries, costs and robust denominators.
    inputdiag=dict(source_metrics=solver_metrics,coverage=coverage,independent_3d_validation=False,
                   heldout_edge_note='Excluded from q/source construction and edge losses; same input RGB video, not independent test data.')
    save(output/'input_edge_diagnostics.json',inputdiag)
    save(output/'factorial_contrasts.json',fac)
    save(output/'promotion_decision.json',promotion)
    save(output/'fixed_failure_interval.json',dict(dev='dev2',start_input_frame=64,end_input_frame=74,
                                                  role='previously exposed posthoc case',rows=fixed_case))
    csvwrite(output/'paired_metrics.csv',summary_rows)
    csvwrite(output/'paired_per_slot.csv',slot_rows)
    csvwrite(output/'paired_per_interval.csv',interval_rows)
    csvwrite(output/'factorial_contrasts.csv',fac['rows'])
    csvwrite(output/'input_edge_diagnostics.csv',input_rows)
    csvwrite(output/'fixed_failure_interval.csv',fixed_case)
    validate_all_outputs(freeze_path)
    result=dict(status='completed',created_utc=now(),script=identity(__file__),rules=identity(ROOT/'protocol/promotion_rules.json'),
                all_outputs_freeze=identity(freeze_path),summaries=summary_rows,
                per_event_outputs={d:str(output/d/'augmented_results.json') for d in DEVS},
                factorial=str(output/'factorial_contrasts.json'),promotion=str(output/'promotion_decision.json'),
                fixed_failure_interval=str(output/'fixed_failure_interval.json'),
                source_hashes_unchanged=True,wall_seconds=time.perf_counter()-start,
                gaussian_metrics=dict(status='not_run',reason='This evaluation covers initialized rigid poses; Gaussian training needs promotion first.',
                                      fixed_queries=None,surface_proxy=None,heldout_camera1=None,LPIPS=None),
                limitations=['Fitted mesh references are approximate; no sensor/material/contact truth claim.',
                             'Both events are development data, not statistically independent repeats or blind tests.',
                             'Reference missing slots retained, no interpolation, ICP, scale fitting or identity switching.'])
    save(output/'paired_metrics.json',result)
    print(json.dumps(dict(status='completed',selected_global_variant=promotion['selected_global_variant'],
                          decision=promotion['high_level'],output=str(output))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all-outputs-freeze',type=Path,required=True)
    p.add_argument('--check-freeze',action='store_true',help='Verify input-only manifests and outputs; does not open references')
    a=p.parse_args()
    if a.check_freeze:
        validate_all_outputs(a.all_outputs_freeze)
        print('All-eight input freeze valid; no reference opened')
    else:
        evaluate(a.all_outputs_freeze)


if __name__=='__main__':main()
