"""Assemble private independent-confirmation delivery indices after actual review.

Does not infer a quality conclusion, execute training or mark visual QA complete.
The reviewer supplies tennis/quality_assessment.json and visual_review.json.
"""
from common import *
import time
B=RUN/'tennis'

def run():
    decision=read(B/'quality_assessment.json');visual=read(B/'visual_review.json')
    assert decision['status']=='completed' and decision['interpretation']
    assert visual['status']=='completed' and visual['all_fixed_images_visually_checked']
    assert read(B/'protocol/independent_verification.json')['status']=='pass'
    expected={str(p.resolve()) for p in (B/'evaluation/figures').glob('*.jpg')}
    assert len(expected)==40 and expected==set(visual['files'])
    assets={}
    def add(p):
        p=Path(p).resolve()
        if str(p) not in assets:assets[str(p)]=identity(p)
    # RGB/masks and cameras remain private local assets, with exact byte identity.
    for name in ['manifest.json','evaluation_manifest.json']:
        manifest=read(B/'inputs/hos_tennis'/name)
        for f in manifest['frames']:
            for key in ['image_path','mask_path']:add(f[key])
        add(manifest['camera_source']['path'])
    for pattern in ['protocol/*.pt','runs/*/checkpoint_*.pt','runs/*/rolling_*.pt','inputs/hos_tennis/*.npz','inputs/hos_tennis/*.npy','inputs/hos_tennis/features/*.npz']:
        for p in B.glob(pattern):add(p)
    # Renders were already hashed by both rendering and independent verification.
    for arm in ['B_U','B_Q']:
        for row in read(B/'evaluation'/arm/'manifest.json')['rows']:
            asset=row['render'];p=Path(asset['path']);assert p.is_file() and p.stat().st_size==asset['bytes'];assets[str(p.resolve())]=asset
    save_json(B/'large_asset_index.json',dict(assets=list(assets.values()),scope='Local data, original-precision renders, initialization and complete states; not copied into small feedback ZIP',time_unix=time.time()))
    selected={}
    for name in ['pipeline.json','pipeline_launcher.json','metrics_per_frame.csv','paired_differences.csv','evaluation_summary.json','figure_manifest.json','costs.json','quality_assessment.json','visual_review.json','large_asset_index.json']:
        p=B/name;assert p.is_file();selected[name]=p
    for pattern in ['protocol/*.json','protocol/*steps.jsonl','protocol/optimizer_calls.jsonl','protocol/gpu_jobs.jsonl','runs/*/*.json','runs/*/sampling_order.jsonl','runs/*/training_metrics.jsonl','runs/*/density_events.jsonl','runs/*/checkpoint_index.jsonl','logs/*/attempt.json','inputs/hos_tennis/*.json','feedback_arrays/*.npz','feedback_arrays/*.json','evaluation/*/manifest.json','evaluation/figures/*.jpg']:
        for p in B.glob(pattern):
            if p.is_file():selected[str(p.relative_to(B))]=p
    for name,p in selected.items():
        assert p.suffix in ['.json','.jsonl','.csv','.jpg','.npz']
        assert not p.is_relative_to(B/'assets')
        if p.suffix=='.npz':assert p.parent==B/'feedback_arrays'
    save_json(RUN/'independent_confirmation.json',dict(status='completed',scene='Tennis',arms=['B_U','B_Q'],scope='Locked independent baseline-objective confirmation; no temporal candidate',evaluation_summary_path=str((B/'evaluation_summary.json').resolve()),interpretation=decision['interpretation'],decision=identity(B/'quality_assessment.json'),visual_review=identity(B/'visual_review.json'),verification=identity(B/'protocol/independent_verification.json'),costs=identity(B/'costs.json'),feedback_files=[dict(name=n,**identity(p)) for n,p in sorted(selected.items())]))
    print('Tennis reviewed results and explicit private feedback inventory ready')

if __name__=='__main__':run()
