"""Independent evaluation of the predeclared single B ranking rule, same pool."""
import json
import evaluate_diagnosis as D


def main():
    root = D.ROOT
    rulepath = root / 'protocol/official_same_pool_ranking_rule.json'
    rule = D.read(rulepath)
    assert rule['frozen_before_score'] is True
    official_path = root / 'score_audit/official/base/scored/full_path_scores.json'
    official = D.read(official_path)
    freeze = D.read(root / 'evaluation/historical_pool_freeze.json')
    D.validate_freeze(freeze)
    result = dict(created_utc=D.now(), status='completed', branch='B_only',
                  rule=D.identity(rulepath), official_scores=D.identity(official_path),
                  same_pool=True, no_new_pose_optimization=True, new_gaussian_training=False,
                  dev={}, unified_gain_supported=False)
    rows = []
    comparisons = []
    pool = {'old_initialization', 'path00_old_start', 'path01_beam', 'path02_beam'}
    for dev in ('dev1', 'dev2'):
        origpath = root / f'score_audit/same_pool_original/{dev}_scores.json'
        orig = D.read(origpath)
        assert orig['same_pool_complete'] is True
        inputs = {r['name']:r for r in orig['rows']}
        assert set(inputs) == pool
        for r in inputs.values():
            assert D.E.sha(r['path']) == r['sha256']
        score_rows = {r['path_id']:r for r in official if r['dev'] == dev and
                      (r['stage'] == 'after' or r['path_id'] == 'old_initialization')}
        assert set(score_rows) == pool
        valid = [k for k in sorted(pool) if inputs[k]['accepted'] and score_rows[k]['all_valid']]
        assert valid
        for k in valid:
            assert score_rows[k]['required_frames'] == score_rows[k]['valid_frames'] == 5
            assert sorted(map(int,score_rows[k]['per_frame_logit'])) == rule['keyframes'][dev]
        selected_original = min(valid, key=lambda k:(inputs[k]['score'],k))
        assert selected_original == orig['selected']
        selected_official = min(valid, key=lambda k:(-score_rows[k]['mean_pose_logit'],inputs[k]['score'],k))
        evaluated = D.read(root / f'evaluation/oracle_{dev}/diagnostic_results.json')
        summaries = evaluated['summaries']
        metrics = {}
        for role, name in [('old_initialization_baseline','old_initialization'),
                           ('original_same_pool_selection',selected_original),
                           ('official_same_pool_selection',selected_official)]:
            label = 'original_initialization' if name == 'old_initialization' else name + '_after'
            s = summaries[label]
            ident = evaluated['protocol']['prediction_identities'][label]
            assert ident['path'] == inputs[name]['path'] and ident['sha256'] == inputs[name]['sha256']
            assert s['prediction_slots_valid'] == s['reference_available']
            assert s['reference_requested'] == (14 if dev == 'dev1' else 10)
            row = dict(dev=dev, role=role, path=name, input_score=inputs[name]['score'],
                       official_mean_logit=score_rows[name]['mean_pose_logit'],
                       reference_slots_requested=s['reference_requested'], reference_slots_scored=s['prediction_slots_valid'],
                       reference_intervals_requested=s['centre_displacement_error_cm']['requested_count'],
                       reference_intervals_scored=s['centre_displacement_error_cm']['count'],
                       **{k:s[k]['mean'] for k in D.METRICS + ('camera_depth_signed_bias_cm',)})
            metrics[role] = row
            rows.append(row)
        original, rescored = metrics['original_same_pool_selection'], metrics['official_same_pool_selection']
        delta = {k:rescored[k]-original[k] for k in D.METRICS + ('camera_depth_signed_bias_cm',)}
        for key, value in delta.items():
            comparisons.append(dict(dev=dev, metric=key, original_path=selected_original,
                                    official_path=selected_official, original_value=original[key],
                                    official_value=rescored[key], official_minus_original=value))
        result['dev'][dev] = dict(original_scores=D.identity(origpath), identical_candidate_pool=sorted(pool),
                                  jointly_valid_candidates=valid, actual_selection_original=selected_original,
                                  actual_selection_official=selected_official, metrics=metrics,
                                  official_minus_original=delta, all_difficult_reference_slots_retained=True,
                                  complete_per_slot_and_interval_outputs=str(root / f'evaluation/oracle_{dev}'))
    result['decision'] = ('B rejected as a unified correction: dev1 unchanged; dev2 official selection worsens absolute centre, '
                          'absolute depth, raw rotation and centre motion, while relative rotation improves. '
                          'No clear unified gain, so no new Gaussian training or independent-event rollout is justified.')
    result['limitations'] = ['Two repeatedly diagnosed development sequences only; not a blind generalization test.',
                             'Single frozen rule evaluated once; no score fusion tuning or reference-selected candidate export.',
                             'Official scorer may prefer crop/render compatibility rather than physically accurate trajectory.']
    D.write(root / 'evaluation/same_pool_ranking_comparison.json', result)
    D.csvwrite(root / 'evaluation/same_pool_ranking_comparison.csv', rows)
    D.csvwrite(root / 'evaluation/same_pool_ranking_deltas.csv', comparisons)
    print(json.dumps(dict(decision=result['decision'], table=rows), ensure_ascii=False))


if __name__ == '__main__':
    main()
