#!/usr/bin/env python3
"""CPU replay of first-frame MoSca seed-mask queries; no full-image/all-time NN."""
import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '4')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/home/cai_tianshun/Project/HOI'))
    ap.add_argument('--output',type=Path)
    args=ap.parse_args();root=args.root.resolve()
    exp=root/'experiments/mosca_validation_20260923'
    out=(args.output or exp/'analysis_a/seed_mask_probe').resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise RuntimeError('Choose a fresh output directory')
    (out/'executed_probe.py').write_bytes(Path(__file__).read_bytes())
    helper_path=exp/'time_filter/run02_visual_trigger_fixed/executed_diagnostic.py'
    spec=importlib.util.spec_from_file_location('frozen_diagnostic',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from pytorch3d.ops import knn_points
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    start=time.perf_counter();cpu_start=time.process_time()
    repo=exp/'code/MoSca';model=exp/'a_normalized_exact/model'
    ws=root/'experiments/mosca_baseline_20260922/common_input'
    sys.path.insert(0,str(repo))
    from lib_prior.prior_loading import Saved2D
    from lib_moca.camera import MonocularCameras
    from lib_mosca.dynamic_solver_utils import prepare_track_buffers,get_world_points
    cfg=OmegaConf.load(model/'config.yaml')
    s2d=(Saved2D(str(ws)).load_dep('unidepth_depth',float(cfg.depth_boundary_th))
         .normalize_depth(float(cfg.dep_median)).recompute_dep_mask(float(cfg.depth_boundary_th))
         .load_track('cotracker',min_valid_cnt=int(cfg.tap_loading_min_valid_cnt))
         .rescale_perframe_depth_from_bundle(str(model/'bundle/bundle.pth')))
    ident=np.load(model/'track_identification.npz')
    s2d.register_track_indentification(torch.from_numpy(ident['static_track_mask']),torch.from_numpy(ident['dynamic_track_mask']))
    cams=MonocularCameras.load_from_ckpt(torch.load(model/'bundle/bundle_cams.pth',map_location='cpu',weights_only=False))
    scale=float(s2d.scale_nw)
    prior=np.load(exp/'time_filter/run02_visual_trigger_fixed/filter_arrays.npz')
    loaded_ids=prior['loaded_track_ids']
    raw=np.load(ws/'uniform_cotracker_tap.npz')
    assert np.array_equal(s2d.track.numpy(),np.trunc(raw['tracks'][:,loaded_ids]).astype(np.float32))
    assert len(loaded_ids)==2774 and int(s2d.dynamic_track_mask.sum())==2050

    # Compile the exact upstream CPU-capable path without importing CUDA-only render modules.
    source_path=repo/'lib_mosca/dynamic_solver.py';source=source_path.read_text()
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in {'get_dynamic_curves','line_segment_init'}]
    assert len(functions)==2
    (out/'upstream_functions.py').write_text('\n\n'.join(ast.get_source_segment(source,n) for n in functions)+'\n')
    namespace={'torch':torch,'np':np,'logging':logging,'MonocularCameras':MonocularCameras,
               'prepare_track_buffers':prepare_track_buffers,'get_world_points':get_world_points}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source_path),'exec'),namespace)
    curves,mask,_,filter_mask=namespace['get_dynamic_curves'](s2d,cams,return_all_curves=True)
    assert bool(filter_mask.all()) and curves.shape==(114,2774,3)
    # Same expression as photo_recon.identify_fg_mask_by_nearest_curve, including unused curves.
    nd=~s2d.dynamic_track_mask
    static_mean=(curves[:,nd]*mask[:,nd,None]).sum(0,keepdim=True)/mask[:,nd,None].sum(0,keepdim=True).expand(len(curves),-1,-1)
    curves[:,nd]=static_mean
    saved_mean_path=next(model.glob('mosca_photo_viz_*/fg_id_non_dyn_curve_meaned.xyz'))
    saved_mean=np.loadtxt(saved_mean_path)
    mean_difference=float(np.max(np.abs(saved_mean-static_mean.reshape(-1,3).numpy())))
    assert mean_difference<=5.1e-5 # Original artifact was saved with four decimal places.
    meta=json.loads((ws/'queries_first_frame.json').read_text())
    fixed=[tuple(map(int,p)) for p in meta['points'][:6]]
    pixels=list(fixed)
    for x,y in fixed[4:]:
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if (x+dx,y+dy) not in pixels:pixels.append((x+dx,y+dy))
    uv=torch.tensor(pixels,dtype=torch.long);x,y=uv.unbind(1)
    depth=s2d.dep[0,y,x]
    query_cam=cams.backproject(cams.get_homo_coordinate_map()[y,x],depth)
    query_world=cams.trans_pts_to_world(0,query_cam)
    knn=knn_points(query_world[None],curves[0][None],K=2)
    ids=knn.idx[0,:,0]
    squared=((query_world[:,None]-curves[0][None])**2).sum(-1)
    brute=squared.argmin(1)
    assert torch.equal(ids,brute)
    dyn=s2d.dynamic_track_mask
    fg=dyn[ids];valid=s2d.dep_mask[0,y,x]
    records=[]
    for i,(u,v) in enumerate(pixels):
        j=int(ids[i]);raw_id=int(loaded_ids[j])
        dynamic_ids=torch.nonzero(dyn).flatten();non_dynamic_ids=torch.nonzero(~dyn).flatten()
        closest_dyn=int(dynamic_ids[squared[i,dyn].argmin()]);closest_nondyn=int(non_dynamic_ids[squared[i,~dyn].argmin()])
        records.append({'xy':[u,v],'fixed_query_index':i if i<6 else None,
            'query_id':meta['point_ids'][i] if i<6 else None,
            'input_depth_m':float(depth[i]/scale),'depth_valid':bool(valid[i]),
            'dyn_mask':bool(fg[i]),'dynamic_seed_candidate':bool(fg[i]&valid[i]),
            'static_seed_candidate_current_include_fg_true':bool(valid[i]),
            'static_seed_candidate_if_excluding_fg':bool(~fg[i]&valid[i]),
            'nearest_loaded_index':j,'nearest_raw_track_id':raw_id,
            'nearest_track_dynamic':bool(dyn[j]),'nearest_track_explicit_static':bool(s2d.static_track_mask[j]),
            'nearest_track_valid_at_first_frame':bool(mask[0,j]),
            'nearest_position_rule':'first-frame curve position (filled if not valid)' if bool(dyn[j]) else 'valid-observation time mean',
            'nearest_curve_distance_m':float(knn.dists[0,i,0].sqrt()/scale),
            'second_nearest_gap_m':float((knn.dists[0,i,1].sqrt()-knn.dists[0,i,0].sqrt())/scale),
            'nearest_dynamic_raw_track_id':int(loaded_ids[closest_dyn]),'nearest_dynamic_distance_m':float(squared[i,closest_dyn].sqrt()/scale),
            'nearest_non_dynamic_raw_track_id':int(loaded_ids[closest_nondyn]),'nearest_non_dynamic_distance_m':float(squared[i,closest_nondyn].sqrt()/scale),
            'query_world_m':(query_world[i]/scale).tolist(),'nearest_curve_world_m':(curves[0,j]/scale).tolist()})
    neighborhoods=[]
    for i in [4,5]:
        u,v=fixed[i];rr=[r for r in records if abs(r['xy'][0]-u)<=1 and abs(r['xy'][1]-v)<=1]
        assert len(rr)==9
        neighborhoods.append({'query_id':meta['point_ids'][i],'pixels':9,'dynamic_mask_pixels':sum(r['dyn_mask'] for r in rr),
            'depth_valid_pixels':sum(r['depth_valid'] for r in rr),'dynamic_seed_candidate_pixels':sum(r['dynamic_seed_candidate'] for r in rr),
            'static_seed_candidate_pixels_current_include_fg_true':sum(r['static_seed_candidate_current_include_fg_true'] for r in rr)})
    np.savez_compressed(out/'sparse_probe.npz',query_xy=uv.numpy(),query_world_normalized=query_world.numpy(),
        first_frame_curve_world_normalized=curves[0].numpy(),nearest_loaded_index=ids.numpy(),
        loaded_raw_track_ids=loaded_ids,track_dynamic_mask=dyn.numpy(),first_frame_track_valid=mask[0].numpy(),
        dynamic_region_mask=fg.numpy(),depth_valid=valid.numpy(),world_scale=scale)
    files=[Path(__file__).resolve(),helper_path,source_path,repo/'lib_mosca/photo_recon.py',repo/'lib_mosca/dynamic_solver_utils.py',
        repo/'lib_prior/prior_loading.py',repo/'lib_moca/camera.py',repo/'mosca_reconstruct.py',model/'config.yaml',model/'run.json',
        model/'track_identification.npz',model/'bundle/bundle.pth',model/'bundle/bundle_cams.pth',saved_mean_path,
        ws/'queries_first_frame.json',ws/'input_manifest.json',ws/'uniform_cotracker_tap.npz',
        exp/'time_filter/run02_visual_trigger_fixed/filter_arrays.npz']
    assert not torch.cuda.is_initialized()
    result={'status':'completed','scope':'CPU, 22 unique first-frame query pixels; no full-image or all-sequence nearest-neighbor evaluation, no reference/GT',
        'world_scale':scale,'loaded_curves':2774,'dynamic_curve_labels':2050,'explicit_static_labels':686,'unused_labels_treated_non_dynamic':38,
        'method':'Exact upstream get_dynamic_curves(return_all_curves=True); valid time mean for non-dynamic curves; actual pytorch3d.ops.knn_points on CPU, checked against direct squared distances.',
        'saved_non_dynamic_mean_max_difference_normalized':mean_difference,'saved_mean_decimal_precision':4,
        'fixed_queries':records[:6],'hand_3x3_neighborhoods':neighborhoods,'all_sparse_queries':records,
        'limits':['Candidate eligibility is not proof that a Gaussian was sampled at this pixel.',
            'Dynamic scaffold filtering and optimized scaffold geometry are not used by this mask function.',
            'Current static initialization includes foreground, so static and dynamic seed candidates can overlap.',
            'No low-alpha rendered center or independent geometry is used as a surface truth.'],
        'source_sha256':{str(p):helper.sha(p) for p in files},'array_sha256':helper.sha(out/'sparse_probe.npz'),
        'cost':{'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu_start,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'CUDA_initialized':False}}
    helper.json_dump(out/'results.json',result)
    print(json.dumps({k:result[k] for k in ['saved_non_dynamic_mean_max_difference_normalized','fixed_queries','hand_3x3_neighborhoods','cost']},indent=2))


if __name__=='__main__':main()
