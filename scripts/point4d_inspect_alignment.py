#!/usr/bin/env python3
"""Visualize input-prior alignment residuals; never reads evaluation geometry."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from point4d_align_common import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment', type=Path, required=True)
    args = p.parse_args()
    out = args.alignment.resolve()
    protocol = json.loads((out/'alignment_protocol.json').read_text())
    files = protocol['input_files']
    for record in files.values():
        assert sha(record['path']) == record['sha256'], record['path']
    run = json.loads(Path(files['point4d_run']['path']).read_text())
    assert run['source_metadata']['manifest_sha256'] == files['manifest']['sha256']
    assert sha(Path(__file__).with_name('point4d_align_common.py')) == protocol['script_sha256']
    for filename, record in protocol['output_files'].items():
        assert sha(out/filename) == record['sha256'], filename
    support = np.load(out/'alignment_support.npz')
    query = np.load(out/'prediction_first6_reference_frames.npz')
    assert query['predicted'].shape == (14, 6, 3)
    assert query['predicted_valid_mask'].all()
    assert np.isfinite(query['predicted']).all()
    assert str(query['units']) == 'm' and str(query['coordinate_frame']) == 'behave_world_k1_color'
    assert not protocol['global_transform_fitted_on_evaluation']
    gh = len(range(0, protocol['model_hw'][0], protocol['geometry_stride']))
    gw = len(range(0, protocol['model_hw'][1], protocol['geometry_stride']))
    def grid(data):
        a = np.full(gh*gw, np.nan)
        a[support['dense_flat_indices']] = data
        return a.reshape(gh, gw)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    image = axes[0, 0].imshow(grid(support['unidepth_z_m']), cmap='viridis')
    axes[0, 0].set_title('Legal UniDepth first-frame z-depth (m)')
    fig.colorbar(image, ax=axes[0, 0])
    image = axes[0, 1].imshow(grid(support['retained_mask']), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Retained support: white = lowest 80% residuals')
    image = axes[1, 0].imshow(grid(np.log1p(support['residual_m'])), cmap='magma')
    axes[1, 0].set_title('All support log(1 + residual in metres)')
    fig.colorbar(image, ax=axes[1, 0])
    keep = support['retained_mask']
    for mask, label in [(keep, 'retained'), (~keep, 'trimmed')]:
        vals = np.sort(support['residual_m'][mask])
        axes[1, 1].semilogx(vals, np.arange(1, len(vals)+1)/len(vals), label=label)
    axes[1, 1].set(xlabel='Input-prior residual (m), logarithmic axis', ylabel='Cumulative fraction')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=.2)
    for ax in axes.ravel()[:3]:
        ax.set(xlabel='Dense grid column', ylabel='Dense grid row')
    fig.suptitle('One fixed Point4D-to-UniDepth Sim(3); NOT independent reconstruction accuracy', fontsize=12)
    fig.tight_layout()
    fig.savefig(out/'input_prior_alignment_diagnostic.png', dpi=150)
    plt.close(fig)
    validation = {'input_and_output_hashes': 'passed', 'same_RGB_manifest': 'passed',
                  'finite_prediction_schema_14x6x3': 'passed',
                  'alignment_script_hash': 'passed', 'global_transform_fitted_on_evaluation': False,
                  'diagnostic_script_sha256': sha(__file__),
                  'diagnostic_png_sha256': sha(out/'input_prior_alignment_diagnostic.png')}
    (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
