"""Frozen-query distance stability: a proxy, never a rigid-pose accuracy score."""
from pathlib import Path
import argparse, hashlib, itertools, json
import numpy as np
ROOT = Path('/home/cai_tianshun/Project/HOI')
OLD = ROOT/'experiments/mosca_baseline_20260922'
EXP = ROOT/'experiments/mosca_validation_20260923'

def measure(path, key):
    with np.load(path) as z:
        ids = np.flatnonzero(z['entity'] == 'object')
        pairs = list(itertools.combinations(ids.tolist(), 2))
        xyz = z[key].astype(np.float64)
        dist = np.stack([np.linalg.norm(xyz[:, i]-xyz[:, j], axis=-1) for i,j in pairs], axis=-1)
        names = [[str(z['query_id'][i]), str(z['query_id'][j])] for i,j in pairs]
    drift = np.abs(dist[1:]-dist[:1])
    return {'source':str(path), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'frames':len(xyz), 'pairs':names, 'first_distances_m':dist[0].tolist(),
            'mean_abs_distance_change_from_first_m':float(drift.mean()),
            'max_abs_distance_change_from_first_m':float(drift.max()),
            'per_pair_mean_abs_change_m':drift.mean(0).tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--branches', nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    paths={'old_pilot':OLD/'mosca_cotracker/diagnostics/query_trajectories.npz'}
    paths.update({b:EXP/b/'diagnostics/query_trajectories.npz' for b in a.branches})
    result={k:measure(v,'predicted') for k,v in paths.items()}
    ref=OLD/'evaluation/mosca_cotracker_final/evaluation_bundle.npz'
    record={'status':'completed','method':'All six pairwise distances among the four pre-existing fixed source-pixel queries, no alignment, no optimization.',
            'limitation':'These are alpha-weighted Gaussian trajectories, not verified material points. Low distance change alone can result from frozen/wrong motion; pair with independent trajectory errors. Reference has only 14 fitted observations versus 114 predictions.',
            'results':result,'reference_template_check':measure(ref,'reference')}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:{v:x[v] for v in ['mean_abs_distance_change_from_first_m','max_abs_distance_change_from_first_m']} for k,x in result.items()},indent=2))
if __name__=='__main__':main()
