"""Bounded CPU-only template/interface audit, never a pose prediction."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPO = ROOT / 'third_party/GVHMR'
ASSET = Path('/home/cai_tianshun/Project/mml/smpl_model/smplx/SMPLX_NEUTRAL.npz')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    sys.path.insert(0, str(REPO))
    import torch
    torch.set_num_threads(2)
    link = REPO / 'inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz'
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(ASSET)
    assert link.resolve() == ASSET.resolve()
    assert not (REPO / 'inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl').exists()
    from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
    decoder = EnDecoder(stats_name='MM_V1_AMASS_LOCAL_BEDLAM_CAM').eval()
    p = dict(body_pose=torch.zeros(1, 1, 63), betas=torch.zeros(1, 1, 10),
             global_orient=torch.zeros(1, 1, 3), transl=torch.zeros(1, 1, 3))
    with torch.no_grad():
        vertices, coco = decoder.smplx_model(**p)
        body = decoder.fk_v2(**p)
    result = dict(status='template_and_constructor_passed', no_gpu=True, real_video_inference=False,
                  learned_weights_loaded=False, test='zero shape/pose only, not an accuracy or motion test',
                  repo=str(REPO), commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
                  asset=str(ASSET), asset_sha256=sha(ASSET), asset_symlink=str(link),
                  vertices_shape=list(vertices.shape), coco17_shape=list(coco.shape), body22_shape=list(body.shape),
                  finite=bool(torch.isfinite(vertices).all() and torch.isfinite(coco).all() and torch.isfinite(body).all()))
    from hmr4d.network.hmr2 import HMR2
    from hmr4d.network.hmr2.configs import get_config
    cfg = get_config(str(REPO/'hmr4d/network/hmr2/configs/model_config.yaml'))
    feature_model = HMR2(cfg)
    result['hmr2_feature_constructor'] = dict(status='passed_on_cpu',
        parameters=sum(p.numel() for p in feature_model.parameters()),
        smpl_layer_constructed=False, smpl_neutral_file_present=False,
        note='ViT and transformer token head only; no trained forward ran')
    try:
        from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import Pipeline
        result['full_pipeline_import'] = 'passed'
    except Exception as e:
        result['full_pipeline_import'] = dict(status='blocked_optional_dependency', error=repr(e))
    sources = ['hmr4d/network/hmr2/__init__.py', 'hmr4d/network/hmr2/hmr2.py',
               'hmr4d/network/hmr2/smpl_head.py', 'hmr4d/model/gvhmr/utils/endecoder.py',
               'hmr4d/utils/body_model/smplx_lite.py', 'hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py',
               'hmr4d/utils/body_model/smplx2smpl_sparse.pt',
               'hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',
               'hmr4d/utils/body_model/smplx_verts437.pt',
               'hmr4d/network/hmr2/configs/smpl_mean_params.npz']
    result['source_hashes'] = {s:sha(REPO/s) for s in sources}
    result['seconds'] = time.perf_counter()-start
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'executed_probe.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
