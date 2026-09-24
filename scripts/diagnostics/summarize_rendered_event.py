#!/usr/bin/env python3
"""CPU visual and training-view RGB diagnostics of an already exported event."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--segmentation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    m = json.loads((a.input / 'input_manifest.json').read_text())
    export = json.loads((a.export / 'export_manifest.json').read_text())
    if export['status'] != 'completed':
        raise ValueError('Only summarize a completed real export')
    if export['input_manifest_sha256'] != sha(a.input / 'input_manifest.json'):
        raise ValueError('Input mismatch')
    masks = np.load(a.segmentation, allow_pickle=False)['entity_labels']
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    accum = {k: [0., 0] for k in ['all', 'person', 'object', 'background']}
    key_indices = [0, 19, 22, 25, 31, 93]
    panels = []
    for t, source in enumerate(m['frame_paths']):
        rgb = cv2.imread(source)
        prediction = cv2.imread(str(a.export / 'rgb' / f'{t:05d}.png'))
        if rgb is None or prediction is None or rgb.shape != prediction.shape:
            raise ValueError(f'Bad image pair at frame {t}')
        squared = ((rgb.astype(np.float64) - prediction) / 255.) ** 2
        for region, label in [('all', None), ('person', 1), ('object', 2), ('background', 0)]:
            selected = squared if label is None else squared[masks[t] == label]
            count = selected.size
            mse = float(selected.mean()) if count else None
            psnr = float(-10 * np.log10(max(mse, 1e-15))) if mse is not None else None
            rows.append({'frame_index': t, 'timestamp_s': m['timestamp_seconds'][t],
                         'region': region, 'pixel_count': count // 3, 'mse': mse, 'psnr_db': psnr})
            if count:
                accum[region][0] += float(selected.sum())
                accum[region][1] += count
        if t in key_indices:
            yy, xx = np.where(masks[t] > 0)
            if not len(xx):
                raise ValueError('No estimated foreground for crop')
            x0, x1 = max(0, xx.min()-20), min(rgb.shape[1], xx.max()+21)
            y0, y1 = max(0, yy.min()-20), min(rgb.shape[0], yy.max()+21)
            pieces = []
            for title, im in [('Input RGB', rgb), ('Joint MoSca render', prediction),
                              ('Input foreground crop', rgb[y0:y1, x0:x1]),
                              ('Same crop of render', prediction[y0:y1, x0:x1])]:
                # Fit without geometric stretching; black margins remain visible.
                scale = min(320/im.shape[1], 240/im.shape[0])
                resized = cv2.resize(im, (round(im.shape[1]*scale), round(im.shape[0]*scale)))
                tile = np.full((275, 320, 3), 255, np.uint8)
                tile[35:, :] = 15
                x = (320-resized.shape[1])//2
                y = 35+(240-resized.shape[0])//2
                tile[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
                cv2.putText(tile, title, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, .43, (0,0,0), 1)
                cv2.putText(tile, f'frame {t}; t={m["timestamp_seconds"][t]:.3f}s',
                            (5, 29), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 1)
                pieces.append(tile)
            panels.append(np.concatenate(pieces, axis=1))
    cv2.imwrite(str(a.output/'input_reconstruction_storyboard.png'), np.concatenate(panels, axis=0))
    with open(a.output/'training_view_rgb_metrics.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for region, (error_sum, count) in accum.items():
        mse = error_sum/count if count else None
        summary[region] = {'pixel_count': count//3, 'pooled_mse': mse,
                           'pooled_psnr_db': float(-10*np.log10(max(mse, 1e-15))) if mse is not None else None}
    report = {'status': 'completed', 'input_manifest_sha256': sha(a.input/'input_manifest.json'),
              'export_manifest_sha256': sha(a.export/'export_manifest.json'),
              'script_sha256': sha(__file__), 'summary': summary, 'storyboard_indices': key_indices,
              'interpretation': 'Training-view RGB fit only. Human/object/background regions are estimated SAM2 masks, not reference masks. Empty object masks contribute no pixels. These values do not establish geometry, contact accuracy or novel-view quality.',
              'computation': 'Squared RGB error on 8-bit exported images divided by 255; pooled across pixels then converted to PSNR. No image alignment, crop fitting or reference tuning.'}
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
