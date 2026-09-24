"""Frozen A/C/D expected-depth diagnostic on six existing fitted-reference frames.

Evaluation only. Reference supports are fixed, with no per-run alignment,
confidence-based selection, or fitted data entering model construction.
"""
from pathlib import Path
import argparse, hashlib, json, time
import cv2
import numpy as np

ROOT = Path('/home/cai_tianshun/Project/HOI')
PREV = ROOT / 'experiments/mosca_validation_20260923'
EXP = ROOT / 'experiments/mosca_interface_validation_20260923'
CASES = {'A': PREV / 'a_normalized_exact', 'C': ROOT/'experiments/mosca_hand_trace_20260923/c_input_foreground', 'D': EXP/'d_exact_geometry'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=['A', 'C', 'D'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    references = sorted((PREV / 'resolution_depth/evaluation_only').glob('reference_*.npz'))
    assert len(references) == 6
    record = {'role': 'evaluation_only_fitted_visible_surface_proxy', 'cases': {},
              'reference_hashes': {str(p): sha(p) for p in references},
              'script_sha256': sha(Path(__file__)),
              'definition': 'Equal-frame mean expected-camera-z absolute error; original fixed fitted masks and 5x5 erosion. No alpha/confidence filtering or alignment. Alpha is reported separately.',
              'limitations': ['Registered human/object fit is not sensor GT; clothes, gloves, boundaries and nominal timing differ.',
                              'Reference z-buffer excludes environment occluders.',
                              'Expected depth mixes contributing layers; not a mesh distance, material trajectory, or contact metric.',
                              'Six pre-existing frames on one development event, not a standard benchmark.']}
    crops = []
    for name in args.cases:
        branch = CASES[name]
        assert json.loads((branch / 'pipeline.json').read_text())['status'] == 'completed'
        rows = []
        for ref_path in references:
            frame = int(ref_path.stem.split('_')[-1])
            pred_path = branch / 'diagnostics/depth' / f'{frame:05d}.npz'
            ref = np.load(ref_path); pred = np.load(pred_path)
            dep, alpha = pred['depth_camera_z_m'], pred['alpha']
            assert dep.shape == ref['dep'].shape == (480, 640)
            row = {'frame': frame, 'prediction_sha256': sha(pred_path), 'entities': {}}
            for entity, eid in [('person', 1), ('object', 2)]:
                mask = ref['entity'] == eid
                interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                zones = {}
                for zone, support in [('all', mask), ('interior2px', interior), ('boundary2px', mask & ~interior)]:
                    x, y, a = dep[support], ref['dep'][support], alpha[support]
                    valid = np.isfinite(x) & (x > 0)
                    # Fail rather than silently omit missing model predictions.
                    assert np.all(valid), (name, frame, entity, zone, np.sum(~valid))
                    zones[zone] = {'count': len(x), 'valid_count': int(valid.sum()),
                                  'mean_abs_m': float(np.abs(x-y).mean()),
                                  'median_abs_m': float(np.median(np.abs(x-y))),
                                  'mean_signed_m': float((x-y).mean()),
                                  'alpha_quantiles': np.quantile(a, [0, .1, .5, .9]).tolist(),
                                  'alpha_under_half_fraction': float(np.mean(a < .5))}
                row['entities'][entity] = zones
            rows.append(row)
        summary = {entity: {zone: {
            'equal_frame_mae_m': float(np.mean([r['entities'][entity][zone]['mean_abs_m'] for r in rows])),
            'equal_frame_median_abs_m': float(np.mean([r['entities'][entity][zone]['median_abs_m'] for r in rows])),
            'equal_frame_signed_m': float(np.mean([r['entities'][entity][zone]['mean_signed_m'] for r in rows])),
            'equal_frame_alpha_under_half_fraction': float(np.mean([r['entities'][entity][zone]['alpha_under_half_fraction'] for r in rows]))}
            for zone in ['all', 'interior2px', 'boundary2px']} for entity in ['person', 'object']}
        record['cases'][name] = {'branch': str(branch), 'pipeline_sha256': sha(branch / 'pipeline.json'), 'rows': rows, 'summary': summary}
    record['elapsed_cpu_seconds'] = time.perf_counter() - start
    (args.output / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({name: data['summary'] for name, data in record['cases'].items()}, indent=2))

if __name__ == '__main__':
    main()
