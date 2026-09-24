"""Read-only identity/completion checks; no rendering or model optimization."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT = Path('/home/cai_tianshun/Project/HOI')
EXP = ROOT / 'experiments/mosca_interface_validation_20260923'
D = EXP / 'd_exact_geometry'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def main():
    checks = []
    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
    pipe = read(D / 'pipeline.json')
    run = read(D / 'model/run.json')
    launch = read(D / 'launch.json')
    lineage = read(D / 'model/lineage/summary.json')
    check('pipeline completed', pipe['status'] == 'completed')
    check('five pipeline stages completed', len(pipe['stages']) == 5 and all(x['status']=='completed' and x['exit_code']==0 for x in pipe['stages']))
    check('train completed', run['status'] == 'completed')
    check('camera frozen', all(v == 0 for v in run['camera_max_abs_changes'].values()))
    check('version 2', run['geometry_interface_version'] == 2)
    for p, expected in launch['code_sha256'].items():
        check('launch code '+p, sha(Path(p)) == expected)
    for p, expected in run['source_identity'].items():
        check('training source '+p, sha(EXP/'code/MoSca'/p) == expected)
    historical = read(ROOT/'experiments/mosca_hand_trace_20260923/preservation_checks.json')
    for row in historical['checks']:
        check('historical '+row['path'], sha(Path(row['path'])) == row['expected'])
    for row in read(ROOT/'experiments/mosca_hand_trace_20260923/summary_ac/summary.json')['runs']:
        for p, expected in row['identity_files'].items():
            check('A/C summary identity '+p, sha(Path(p)) == expected)
    check('lineage completed 8000', lineage['status']=='completed' and lineage['completed_steps']==lineage['expected_steps']==8000)
    check('15 prescribed snapshots', [x['completed_steps'] for x in lineage['snapshots']] == launch['lineage_checkpoints'])
    check('lineage frozen code', lineage['script_sha256'] == launch['code_sha256'][str(EXP/'code/lineage_hooks.py')])
    for row in lineage['snapshots']:
        folder = Path(row['directory'])
        for filename, key in [('dynamic_state.pth','state_sha256'),('static_state.pth','static_state_sha256'),('identity.npz','identity_sha256')]:
            check(f'snapshot {folder.name} {filename}', sha(folder/filename) == row[key])
    with np.load(D/'model/lineage/leaf_lineage.npz') as z:
        leaf_keys = z.files
    last = D/'model/lineage/snapshots/08000'
    with np.load(last/'identity.npz') as ids:
        stable_ids = ids['leaf_id']
        check('final leaf IDs unique', len(np.unique(stable_ids)) == len(stable_ids) == lineage['surviving_leaves'])
        check('final attached IDs in node IDs', np.isin(ids['attached_node_id'], ids['node_id']).all())
        check('final node count matches', len(ids['node_id']) == lineage['surviving_nodes'])
    state = torch.load(last/'dynamic_state.pth', map_location='cpu', weights_only=False)
    check('final state matches stable ID rows', state['_xyz'].shape[0] == len(stable_ids))
    result = {'status': 'passed' if all(x['passed'] for x in checks) else 'failed',
              'check_count':len(checks), 'checks':checks, 'leaf_ledger_keys':leaf_keys,
              'reference_used_for_training':pipe['reference_used_for_training'],
              'script_sha256':sha(Path(__file__))}
    (EXP/'delivery_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    if result['status'] != 'passed':
        print(json.dumps([x for x in checks if not x['passed']],indent=2))
        raise RuntimeError('Delivery checks failed')

if __name__ == '__main__':
    main()
