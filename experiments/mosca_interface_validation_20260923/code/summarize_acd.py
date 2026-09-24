"""Aggregate A/C/D with identical frozen references and matched image crops."""
from pathlib import Path
import hashlib, json
import cv2
import numpy as np

ROOT = Path('/home/cai_tianshun/Project/HOI')
OLD = ROOT / 'experiments/mosca_baseline_20260922'
PREV = ROOT / 'experiments/mosca_validation_20260923'
EXP = ROOT / 'experiments/mosca_interface_validation_20260923'
OUT = EXP / 'summary_acd'
CASES = {'A': PREV / 'a_normalized_exact', 'C': ROOT/'experiments/mosca_hand_trace_20260923/c_input_foreground', 'D': EXP/'d_exact_geometry'}
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    for branch in CASES.values(): assert read(branch / 'pipeline.json')['status'] == 'completed'
    OUT.mkdir()
    rows = []; identity = None
    for name, branch in CASES.items():
        e = branch / 'evaluation'; model = branch / 'model'; diag = branch / 'diagnostics'
        result = read(e / 'three_dimensional/report.json')['results']
        with np.load(e / 'evaluation_bundle.npz') as z:
            current = tuple(z[k].tobytes() for k in ['reference', 'frame_times', 'query_id'])
            if identity is None: identity = current
            assert current == identity
        def stat(key, group):
            return next(x for x in result[key] if x.get('group') == group and x.get('visibility') == 'all')
        train = read(model / 'run.json')
        row = {'name': name, 'branch': str(branch), 'pipeline': read(branch / 'pipeline.json'),
               'absolute': stat('summaries', 'all'), 'object_absolute': stat('summaries', 'entity:object'),
               'hand_absolute': stat('summaries', 'entity:hand'),
               'displacement': stat('displacement_summaries', 'all'),
               'object_displacement': stat('displacement_summaries', 'entity:object'),
               'hand_displacement': stat('displacement_summaries', 'entity:hand'),
               'relative': result['relative_pair_summaries'][0],
               'rgb': read(branch / 'render_summary/summary.json')['summary'],
               'stages': train['stages'], 'peak_allocated_bytes': train['peak_allocated_bytes'],
               'camera_changes': train['camera_max_abs_changes'],
               'identity_files': {str(p): sha(p) for p in [e / 'three_dimensional/report.json', model / 'run.json', diag / 'export_manifest.json']}}
        with np.load(diag / 'query_trajectories.npz') as z:
            row['source_dynamic_fraction'] = z['source_dynamic_fraction'].tolist()
        rows.append(row)
    masks = np.load(OLD / 'segmentation/segmentation.npz')['entity_labels']
    manifest = read(OLD / 'common_input/input_manifest.json')
    panels = []
    for frame in [0, 16, 31, 113]:
        rgb = cv2.imread(manifest['frame_paths'][frame]); yy, xx = np.where(masks[frame] > 0)
        x0, x1 = max(0, int(xx.min())-12), min(rgb.shape[1], int(xx.max())+13)
        y0, y1 = max(0, int(yy.min())-12), min(rgb.shape[0], int(yy.max())+13)
        tiles = []
        images = [('Input', rgb)] + [(name, cv2.imread(str(b / 'diagnostics/rgb' / f'{frame:05d}.png'))) for name, b in CASES.items()]
        for name, im in images:
            crop = im[y0:y1, x0:x1]; scale = min(280/crop.shape[1], 330/crop.shape[0])
            crop = cv2.resize(crop, (round(crop.shape[1]*scale), round(crop.shape[0]*scale)))
            tile = np.full((375, 300, 3), 255, np.uint8)
            x = (300-crop.shape[1])//2; y = 43+(330-crop.shape[0])//2
            tile[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
            cv2.putText(tile, name, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1)
            cv2.putText(tile, f'frame {frame}; {manifest["timestamp_seconds"][frame]:.3f}s', (8, 34), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,0,0), 1)
            tiles.append(tile)
        panels.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(OUT / 'matched_foreground_crops.png'), np.concatenate(panels, axis=0))
    # One fixed evaluation-only crop, identical for every method and time.
    # Includes the human and object; no model-dependent crop or image alignment.
    heldout = []
    names = sorted((CASES['A'] / 'heldout/heldout_cam1_gt').glob('*.png'))
    assert len(names) == 5
    for gt_path in names:
        for branch in CASES.values():
            assert sha(gt_path) == sha(branch / 'heldout/heldout_cam1_gt' / gt_path.name)
        tiles = []
        paths = [('Held-out RGB', gt_path)] + [(name, b / 'heldout/heldout_cam1_render' / gt_path.name) for name, b in CASES.items()]
        for name, path in paths:
            im = cv2.imread(str(path)); assert im.shape[:2] == (480, 640)
            crop = im[120:440, 170:470]
            tile = np.full((360, 300, 3), 255, np.uint8); tile[40:] = crop
            cv2.putText(tile, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .46, (0,0,0), 1)
            cv2.putText(tile, gt_path.stem, (5, 33), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 1)
            tiles.append(tile)
        heldout.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(OUT / 'matched_heldout_crops.png'), np.concatenate(heldout, axis=0))
    record = {'status': 'completed', 'same_reference_identity_verified': True,
              'heldout_fixed_crop_xyxy': [170, 120, 470, 440],
              'heldout_note': 'Nominal archive times only; synchronization unverified. Evaluation-only qualitative crop, no parameter or image alignment.',
              'runs': rows, 'script_sha256': sha(Path(__file__))}
    (OUT / 'summary.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps([{k: r[k] for k in ['name', 'object_absolute', 'hand_absolute', 'relative', 'rgb']} for r in rows], indent=2))

if __name__ == '__main__': main()
