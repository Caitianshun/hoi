#!/usr/bin/env python3
"""Replay A-scale shaking on CPU, with fixed inputs and no reference access."""
import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import resource
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("/home/cai_tianshun/Project/HOI"))
    p.add_argument("--output",type=Path)
    args=p.parse_args();root=args.root.resolve()
    base=root/"experiments/mosca_validation_20260923/time_filter"
    output=(args.output or base/"normalized_followup").resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():raise RuntimeError("Use a fresh output directory")
    (output/"executed_analysis.py").write_bytes(Path(__file__).read_bytes())
    spec=importlib.util.spec_from_file_location("frozen_time_diagnostic",base/"run02_visual_trigger_fixed/executed_diagnostic.py")
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(output/"run.log")])
    start=time.perf_counter();cpu_start=time.process_time()
    experiment=root/"experiments/mosca_validation_20260923"
    repo=experiment/"code/MoSca";model=experiment/"a_normalized_exact/model"
    ws=root/"experiments/mosca_baseline_20260922/common_input"
    previous=base/"run02_visual_trigger_fixed/filter_arrays.npz"
    old=np.load(previous)
    record=json.loads((model/"run.json").read_text())
    cfg=OmegaConf.load(model/"config.yaml")
    assert cfg.dep_median==1 and cfg.scf_geo_keyframe_rate==1
    assert not cfg.get_dynamic_curves_filter_factor_in_world
    scale=float(record["world_scale"]);threshold=float(cfg.get_curve_refilter_shaking_th_world)
    assert threshold==.075
    sys.path.insert(0,str(repo))
    from lib_prior.prior_loading import Saved2D
    from lib_moca.camera import MonocularCameras
    from lib_mosca.dynamic_solver_utils import prepare_track_buffers,get_world_points
    pure=helper.extract_upstream_functions(repo/"lib_mosca/dynamic_solver.py",output/"upstream_pure_functions.py")
    s2d=(Saved2D(str(ws)).load_dep("unidepth_depth",float(cfg.depth_boundary_th))
         .normalize_depth(1.).recompute_dep_mask(float(cfg.depth_boundary_th))
         .load_track("cotracker",min_valid_cnt=int(cfg.tap_loading_min_valid_cnt))
         .rescale_perframe_depth_from_bundle(str(model/"bundle/bundle.pth")))
    assert s2d.scale_nw==scale
    identification=np.load(model/"track_identification.npz")
    old_ident=np.load(root/"experiments/mosca_baseline_20260922/mosca_attempt02_missing_flow/track_identification.npz")
    assert all(np.array_equal(identification[k],old_ident[k]) for k in old_ident.files)
    dynamic=identification["dynamic_track_mask"]
    track=s2d.track[:,dynamic];valid=s2d.track_mask[:,dynamic]
    assert np.array_equal(track.numpy(),old["uv_integer"].astype(np.float32))
    assert np.array_equal(valid.numpy(),old["mask_before_o3d"])
    cams=MonocularCameras.load_from_ckpt(torch.load(model/"bundle/bundle_cams.pth",map_location="cpu",weights_only=False))
    assert cams.q_wc.device.type=="cpu"
    time_indices=torch.arange(len(track))
    homo,dep,_=prepare_track_buffers(s2d,track,valid,time_indices)
    observed=get_world_points(homo,dep,cams,time_indices)
    index_xyz=pure["line_segment_init"](valid,observed)
    times=old["timestamps"]
    time_xyz=helper.fill_by_seconds(observed,valid,times)
    o3d=pure["slot_o3d_outlier_identifyication"](index_xyz,valid,nb_neighbors=16,std_ratio=5.)
    common=valid&o3d
    assert np.array_equal(common.numpy(),old["mask_after_o3d"])
    data={
        "original":helper.residuals(index_xyz,times),
        "time_midpoint_only":helper.residuals(index_xyz,times,True,False),
        "endpoint_exempt_only":helper.residuals(index_xyz,times,False,True),
        "time_midpoint_endpoint_exempt":helper.residuals(index_xyz,times,True,True),
        "time_fill_midpoint_endpoint_exempt":helper.residuals(time_xyz,times,True,True),
    }
    assert torch.equal(data["original"]<threshold,pure["curve_shaking_identification"](index_xyz,threshold))
    keep={};bad={}
    for k,v in data.items():keep[k],bad[k]=helper.decisions(v,common,threshold,int(cfg.dyn_id_cnt))
    assert bad["original"].any(0).sum()==515
    assert (~keep["original"]).sum()==516
    entity=old["estimated_curve_entity"];sampled=old["estimated_slot_labels"]
    statistics={}
    for k,kept in keep.items():
        restored=kept&~keep["original"];removed=~kept&keep["original"]
        groups={}
        for code,name in helper.ENTITY_NAMES.items():
            em=entity==code
            groups[name]={"candidates":int(em.sum()),"kept":int((em&kept).sum()),
                "restored_vs_A_original":int((em&restored).sum()),"newly_removed_vs_A_original":int((em&removed).sum())}
        statistics[k]={"kept":int(kept.sum()),"total_removed":int((~kept).sum()),
            "shaking_removed":int(bad[k].any(0).sum()),"additional_min_valid_removed":int((~kept&~bad[k].any(0)).sum()),
            "restored_vs_A_original":int(restored.sum()),"newly_removed_vs_A_original":int(removed.sum()),
            "groups":groups,"estimated_object_label_slots_all_kept_curves":int(((sampled==2)&common.numpy()&kept[None]).sum()),
            "estimated_object_majority_curve_valid_slots":int((common.numpy()&(entity==2)[None]&kept[None]).sum()),
            "cache_scaled_decisions_identical":bool(np.array_equal(kept,(~((old["residual_"+k]*scale>=threshold)&old["mask_after_o3d"]).any(0))&(old["mask_after_o3d"].sum(0)>=4)))}
    low_valid=~keep["original"]&~bad["original"].any(0)
    extra=[{"raw_track_id":int(i),"common_valid_slots":int(v),"estimated_entity":helper.ENTITY_NAMES[int(e)]} for i,v,e in zip(old["raw_track_ids"][low_valid],common.sum(0).numpy()[low_valid],entity[low_valid])]
    filename=output/"normalized_filter_arrays.npz"
    np.savez_compressed(filename,raw_track_ids=old["raw_track_ids"],timestamps=times,world_scale=scale,
        mask_before_o3d=valid.numpy(),mask_after_o3d=common.numpy(),estimated_curve_entity=entity,
        estimated_slot_labels=sampled,index_filled_xyz_normalized=index_xyz.numpy(),time_filled_xyz_normalized=time_xyz.numpy(),
        **{"keep_"+k:v for k,v in keep.items()},**{"residual_normalized_"+k:v.numpy() for k,v in data.items()},
        **{"bad_slot_"+k:v for k,v in bad.items()})
    old_count={k:int(old["keep_"+k].sum()) for k in data}
    old_obj={k:int((old["keep_"+k]&(entity==2)).sum()) for k in data}
    assert not torch.cuda.is_initialized()
    used=[Path(__file__).resolve(),base/"run02_visual_trigger_fixed/executed_diagnostic.py",previous,
        model/"run.json",model/"config.yaml",model/"track_identification.npz",model/"training.log",
        model/"bundle/bundle.pth",model/"bundle/bundle_cams.pth",ws/"input_manifest.json",
        repo/"lib_prior/prior_loading.py",repo/"lib_moca/camera.py",repo/"lib_mosca/dynamic_solver.py",repo/"lib_mosca/dynamic_solver_utils.py"]
    result={"status":"completed","scope":"CPU, frozen inputs and estimated labels only; no reference/GT or training",
        "world_scale":scale,"normalized_threshold":threshold,"equivalent_metric_threshold_m":threshold/scale,
        "threshold_policy":"A actual fixed normalized .075; no search, no tuning based on geometry evaluation",
        "coordinate_comparison":{"same_dynamic_candidate_assignment":True,"same_integer_tracks":True,
            "same_visibility_depth_validity":True,"same_O3D_validity":True,
            "normalized_observed_vs_old_times_scale_max_abs_difference":float((observed-torch.from_numpy(old["observed_xyz"])*scale).abs().max()),
            "index_fill_vs_old_times_scale_max_abs_difference":float((index_xyz-torch.from_numpy(old["index_filled_xyz"])*scale).abs().max())},
        "matched_A_log":{"dynamic_candidates":2050,"valid_before_O3D":int(valid.sum()),"valid_after_O3D":int(common.sum()),
            "shaking_removed":515,"final_less_than_four_removed":516,"kept":1534},
        "extra_min_valid_removal":extra,"branches":statistics,
        "old_metric_result_comparison":{"total_kept":old_count,"object_majority_kept":old_obj},
        "limits":["Counts are retained estimated trajectories, not independently verified correct surface trajectories.",
            "The full A experiment changes physical distance thresholds, node scales, Gaussian scales and optimization behavior; it is not a pure coordinate-unit conversion.",
            "A also repairs rasterization principal-point handling and isolates initialization RNG; full output differences cannot be attributed solely to this filtering replay.",
            "The legacy lifting convention is retained exactly; no normalized branch reference evaluation was read."],
        "source_hashes":{str(f):helper.sha(f) for f in used},"arrays_sha256":helper.sha(filename),
        "depth_and_RGB_hash_provenance":str(base/"run02_visual_trigger_fixed/results.json"),
        "cost":{"wall_seconds":time.perf_counter()-start,"process_cpu_seconds":time.process_time()-cpu_start,
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"CUDA_initialized":False}}
    helper.json_dump(output/"results.json",result)
    print(json.dumps({k:result[k] for k in ['world_scale','equivalent_metric_threshold_m','coordinate_comparison','matched_A_log','extra_min_valid_removal','branches','cost']},indent=2))


if __name__=="__main__":main()
