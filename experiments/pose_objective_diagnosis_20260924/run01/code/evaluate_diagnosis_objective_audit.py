"""Read-only numerical check of finite objective slices against full original blocks."""
import importlib.util
import numpy as np
import evaluate_diagnosis as D


def main():
    source = D.ROOT / 'code/objective_response.py'
    spec = importlib.util.spec_from_file_location('frozen_response_audit_target', source)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    outputs = []
    for dev in ('dev1', 'dev2'):
        path = D.ROOT / f'score_audit/objective/{dev}'
        if dev == 'dev1':
            path /= 'with_track_off'
        response_file = path / 'responses.json'
        response = D.read(response_file)
        problem = obj.Problem(dev)
        cases = {c['name']:c for c in response['freeze']['cases']}
        for name, c in cases.items():
            assert D.E.sha(c['path']) == c['sha256']
            arr = np.load(c['path'])
            records, meta = obj.records_from_cache(problem, D.Path(c['correspondence_path']).parent)
            # Endpoints and the disclosed chair failure interval; exact original 0, +/-10% and axis extremes.
            selected_frames = {problem.kf[0], problem.kf[-1]} | ({64,74} if dev == 'dev2' else set())
            for r in response['rows']:
                if r['case'] != name or r['frame'] not in selected_frames:
                    continue
                chosen = (r['kind'] == 'depth' and r['value'] in (-.1,0.,.1)) or (r['kind'] == 'rotation' and
                            (r['axis'],r['value']) in ((0,15.),(2,-15.)))
                if not chosen:
                    continue
                R, t = arr['R_camera'].astype(float).copy(), arr['t_camera_m'].astype(float).copy()
                i = r['frame']
                R[i], t[i] = obj.perturbed(R[i],t[i],problem.V.mean(0),r['kind'],r['value'],r['axis'])
                _, comp, const, total = obj.score(problem,R,t,records,meta,c['track_scale'])
                diff = {k:float(comp[k]-r['whole_components'][k]) for k in comp}
                outputs.append(dict(dev=dev,case=name,frame=i,kind=r['kind'],value=r['value'],axis=r['axis'],
                                    total_score_difference=float(total-r['whole_score']),component_differences=diff))
    maxdiff = max(max(abs(x['total_score_difference']),max(abs(v) for v in x['component_differences'].values())) for x in outputs)
    assert maxdiff < 1e-10
    report = dict(status='passed',source=D.identity(source),auditor=D.identity(__file__),
                  original_solver=D.identity(D.OLD / 'code/solve_pose.py'),checked_slices=len(outputs),
                  maximum_absolute_score_or_component_difference=maxdiff,rows=outputs,
                  independent_reference_read=False,source_code_or_outputs_modified=False,
                  mathematical_review=[
                      'Depth translation follows the fixed template centre ray; rotation uses template axes and preserves centre.',
                      'Current-frame robust observation delta/P.L plus explicitly separated whole temporal delta equals full original objective.',
                      'Frozen track records keep the original per-source/per-query residual normalization.',
                      'RGB and coverage constants are pose independent; absent pose prior is stated explicitly.',
                      'track_unweighted_robust_sum means original normalized track residual without the on/off multiplier; it is not raw pixel error.'])
    D.write(D.ROOT / 'evaluation/objective_response_audit.json',report)
    print('Objective response audit passed:',len(outputs),'slices; max difference',maxdiff)


if __name__=='__main__':
    main()
