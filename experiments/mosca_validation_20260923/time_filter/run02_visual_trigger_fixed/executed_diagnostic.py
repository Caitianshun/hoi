#!/usr/bin/env python3
"""CPU-only, input-only replay of MoSca trajectory shaking filtering.

No model source, old input, or old output is modified. Extracted upstream pure
functions retain their implementation; all proposals are local diagnostic code.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import resource
import sys
import time

# Set before importing torch/Open3D; never initialize a CUDA context.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import torch
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

torch.set_num_threads(4)
torch.set_grad_enabled(False)

BRANCHES = (
    "original",
    "time_midpoint_only",
    "endpoint_exempt_only",
    "time_midpoint_endpoint_exempt",
    "time_fill_midpoint_endpoint_exempt",
)
ENTITY_NAMES = {0: "background", 1: "person", 2: "object", 255: "unknown"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def json_dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def extract_upstream_functions(path, output):
    source = path.read_text()
    tree = ast.parse(source)
    names = {"line_segment_init", "slot_o3d_outlier_identifyication", "curve_shaking_identification"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    excerpt = "# Exact AST-extracted upstream functions, CPU globals supplied by caller.\n"
    excerpt += "\n\n".join(ast.get_source_segment(source, n) for n in functions) + "\n"
    output.write_text(excerpt)
    namespace = dict(torch=torch, np=np, logging=logging, o3d=o3d,
                     tqdm=lambda it, **kw: it)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def fill_by_seconds(xyz, visible, timestamps):
    """Only fill invisible slots; keep observed XYZ bit-identical. Endpoint hold."""
    result = xyz.clone()
    t = np.asarray(timestamps, dtype=np.float64)
    x = xyz.numpy()
    for j in range(x.shape[1]):
        ids = np.flatnonzero(visible[:, j].numpy())
        if not len(ids):
            continue
        missing = np.flatnonzero(~visible[:, j].numpy())
        for d in range(3):
            values = np.interp(t[missing], t[ids], x[ids, j, d])
            result[missing, j, d] = torch.from_numpy(values.astype(np.float32))
    assert torch.equal(result[visible], xyz[visible])
    return result


def residuals(xyz, timestamps, use_seconds=False, endpoint_exempt=False):
    ref = (xyz[2:] + xyz[:-2]) / 2.0
    if use_seconds:
        t = torch.as_tensor(timestamps, dtype=torch.float64)
        weight = ((t[1:-1] - t[:-2]) / (t[2:] - t[:-2])).to(xyz.dtype)
        ref = xyz[:-2] * (1.0 - weight[:, None, None]) + xyz[2:] * weight[:, None, None]
    ref = torch.cat([xyz[1:2], ref, xyz[-2:-1]], 0)
    value = (xyz - ref).norm(dim=-1)
    if endpoint_exempt:
        value = value.clone()
        value[0] = 0.0
        value[-1] = 0.0
    return value


def decisions(residual, mask_after_o3d, threshold, min_valid):
    bad_slots = (residual >= threshold) & mask_after_o3d
    any_bad = bad_slots.any(0)
    keep = (~any_bad) & (mask_after_o3d.sum(0) >= min_valid)
    return keep.numpy(), bad_slots.numpy()


def synthetic_checks(times, original_functions, threshold):
    t = torch.tensor(np.asarray(times) - times[0], dtype=torch.float32)
    xyz = torch.zeros(len(t), 1, 3)
    xyz[:, 0, 0] = t  # Known constant velocity: 1 m/s.
    xyz[:, 0, 2] = 3.0
    original = residuals(xyz, times)
    corrected = residuals(xyz, times, True, True)
    assert corrected.max() < 5e-6
    assert torch.equal(original < threshold,
                       original_functions["curve_shaking_identification"](xyz, threshold))
    vis = torch.ones(len(t), 1, dtype=torch.bool)
    vis[18:27] = False
    missing_xyz = xyz.clone()
    missing_xyz[~vis] = 0.0
    index_fill = original_functions["line_segment_init"](vis, missing_xyz)
    time_fill = fill_by_seconds(missing_xyz, vis, times)
    assert (time_fill - xyz).abs().max() < 5e-6
    static = torch.ones_like(xyz)
    assert residuals(static, times, True, True).max() < 1e-6
    corrupted = xyz.clone()
    corrupted[40, 0, 1] += 0.5
    assert residuals(corrupted, times, True, True)[40, 0] >= threshold
    uniform = np.arange(len(t), dtype=np.float64) * .1
    assert torch.allclose(residuals(corrupted, uniform)[1:-1],
                          residuals(corrupted, uniform, True)[1:-1], atol=1e-6)
    return {
        "status": "passed", "is_synthetic_not_real_object_motion": True,
        "known_velocity_m_per_s": [1., 0., 0.],
        "threshold_m": threshold,
        "original_interior_max_residual_m": float(original[1:-1].max()),
        "original_endpoint_residuals_m": original[[0, -1], 0].tolist(),
        "time_corrected_max_residual_m": float(corrected.max()),
        "original_rejects_known_constant_velocity": bool((original >= threshold).any()),
        "time_corrected_rejects_known_constant_velocity": bool((corrected >= threshold).any()),
        "synthetic_gap_indices": list(range(18, 27)),
        "index_gap_fill_max_error_m": float((index_fill - xyz).abs().max()),
        "time_gap_fill_max_error_m": float((time_fill - xyz).abs().max()),
        "uniform_time_interior_equivalence": True,
        "static_passes": True, "single_interior_outlier_0_5m_detected": True,
    }


def entity_accounting(labels, integer_uv, valid_mask):
    T, N = valid_mask.shape
    sampled = np.full((T, N), 255, dtype=np.uint8)
    for i in range(T):
        m = valid_mask[i]
        sampled[i, m] = labels[i, integer_uv[i, m, 1], integer_uv[i, m, 0]]
    counts = np.stack([(sampled == k).sum(0) for k in (0, 1, 2)], 1)
    total = counts.sum(1)
    majority = counts.argmax(1)
    share = counts.max(1) / np.maximum(total, 1)
    entity = np.where((total >= 2) & (share >= .8), majority, 255).astype(np.uint8)
    return sampled, counts, share, entity


def branch_statistics(keep, original_keep, entity, sampled, valid, uv, labels):
    rescued = keep & ~original_keep
    rejected = ~keep & original_keep
    per_entity = {}
    for code, name in ENTITY_NAMES.items():
        m = entity == code
        per_entity[name] = dict(candidates=int(m.sum()), kept=int((keep & m).sum()),
                                rescued=int((rescued & m).sum()),
                                newly_rejected=int((rejected & m).sum()))
    slot_counts = {name: int(((sampled == code) & valid & keep[None]).sum())
                   for code, name in ENTITY_NAMES.items()}
    frames = []
    for t in range(len(valid)):
        row = {"frame_index": t}
        for code, name in ENTITY_NAMES.items():
            m = valid[t] & keep & (sampled[t] == code)
            points = uv[t, m]
            row[name + "_valid_track_slots"] = int(m.sum())
            unique_count = len(np.unique(points, axis=0))
            row[name + "_unique_track_pixels"] = int(unique_count)
            area = int((labels[t] == code).sum()) if code != 255 else 0
            row[name + "_estimated_mask_pixels"] = area
            row[name + "_sampled_pixel_fraction"] = unique_count / area if area else None
        frames.append(row)
    return {"kept": int(keep.sum()), "removed": int((~keep).sum()),
            "rescued_vs_original": int(rescued.sum()),
            "newly_rejected_vs_original": int(rejected.sum()),
            "per_curve_entity": per_entity,
            "valid_slots_by_sampled_rgb_label": slot_counts,
            "per_frame_rgb_label_coverage": frames}


def font(size=18):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_visualizations(output, manifest, input_uv, raw_uv, mask, entity, original_ids,
                        residual_map, keep_map, threshold, times):
    target = "time_fill_midpoint_endpoint_exempt"
    a, b = keep_map["original"], keep_map[target]
    selections = []
    # Deterministic diversity by decision direction then estimated entity; no GT.
    for direction, changed in (("rescued", b & ~a), ("newly_rejected", a & ~b)):
        for code in (2, 1, 0, 255):
            ids = np.flatnonzero(changed & (entity == code))
            if not len(ids):
                continue
            # Prefer long observed trajectories, then stable smallest raw ID.
            order = sorted(ids.tolist(), key=lambda j: (-int(mask[:, j].sum()), int(original_ids[j])))
            j = order[0]
            old_bad = residual_map["original"][:, j] >= threshold
            new_bad = residual_map[target][:, j] >= threshold
            causal = (old_bad & ~new_bad) if direction == "rescued" else (new_bad & ~old_bad)
            causal &= mask[:, j]
            assert causal.any(), "A changed decision needs a causal changed residual slot"
            score = np.where(causal, np.abs(residual_map["original"][:, j] - residual_map[target][:, j]), -1.)
            trigger = int(score.argmax())
            selections.append((j, direction, trigger))
    # Distinguish endpoint-only from temporal weighting effects if available.
    for name, changed in (("endpoint_only_rescue", keep_map["endpoint_exempt_only"] & ~a),
                          ("time_midpoint_only_rescue", keep_map["time_midpoint_only"] & ~a)):
        candidates = np.flatnonzero(changed)
        if len(candidates):
            j = sorted(candidates.tolist(), key=lambda j: (entity[j] != 2, -int(mask[:, j].sum()), int(original_ids[j])))[0]
            if all(j != e[0] for e in selections):
                diff = np.abs(residual_map["original"][:, j] - residual_map[target][:, j])
                changed_branch = "endpoint_exempt_only" if name.startswith("endpoint") else "time_midpoint_only"
                causal = mask[:, j] & (residual_map["original"][:, j] >= threshold) & (residual_map[changed_branch][:, j] < threshold)
                assert causal.any()
                trigger = int(np.where(causal, diff, -1.).argmax())
                selections.append((j, name, trigger))
    selections = selections[:8]
    visdir = output / "visuals"; visdir.mkdir(exist_ok=True)
    metadata = []
    for j, direction, trigger in selections:
        indices = np.arange(max(0, trigger - 2), min(len(times), trigger + 3))
        name = f"track_{int(original_ids[j]):04d}_{direction}"
        canvas = Image.new("RGB", (320 * len(indices), 470), "white")
        draw = ImageDraw.Draw(canvas)
        for col, t in enumerate(indices):
            rgb = Image.open(manifest["frame_paths"][t]).convert("RGB")
            full = rgb.copy(); d = ImageDraw.Draw(full)
            u, v = input_uv[t, j].astype(int)
            d.ellipse((u-6,v-6,u+6,v+6), outline="lime" if mask[t,j] else "red", width=2)
            for k in range(max(0, int(t)-3), int(t)):
                if mask[k,j] and mask[k+1,j]:
                    d.line([tuple(input_uv[k,j]),tuple(input_uv[k+1,j])], fill="yellow", width=2)
            full.thumbnail((320, 240)); canvas.paste(full, (col*320, 34))
            cx, cy = int(np.clip(u, 80, rgb.width-80)), int(np.clip(v, 80, rgb.height-80))
            crop = rgb.crop((cx-80,cy-80,cx+80,cy+80)).resize((180,180))
            cd = ImageDraw.Draw(crop); x=(u-cx+80)*180/160; y=(v-cy+80)*180/160
            cd.ellipse((x-5,y-5,x+5,y+5), outline="lime" if mask[t,j] else "red", width=2)
            canvas.paste(crop,(col*320+70,278))
            draw.text((col*320+4,3),f"f{t}  t={times[t]:.3f}s",font=font(17),fill="black")
        canvas.save(visdir/(name+".png"))
        # A long-range overview reveals entity switches invisible in local crops.
        full_indices = np.unique(np.rint(np.linspace(0,len(times)-1,9)).astype(int))
        overview = Image.new("RGB",(960,810),"white"); od=ImageDraw.Draw(overview)
        for k,t in enumerate(full_indices):
            rgb = Image.open(manifest["frame_paths"][t]).convert("RGB")
            d=ImageDraw.Draw(rgb);u,v=input_uv[t,j].astype(int)
            d.ellipse((u-7,v-7,u+7,v+7),outline="lime" if mask[t,j] else "red",width=3)
            rgb=rgb.resize((320,240));x=(k%3)*320;y=(k//3)*270
            overview.paste(rgb,(x,y+30));od.text((x+4,y+3),f"f{t}  t={times[t]:.3f}s",font=font(17),fill="black")
        overview.save(visdir/(name+"_overview.png"))
        fig,ax=plt.subplots(figsize=(9,3.4))
        for branch in ("original","time_midpoint_only",target):
            values=residual_map[branch][:,j].copy();values[~mask[:,j]]=np.nan
            ax.plot(times,values,label=branch,linewidth=1.2)
        ax.axhline(threshold,color="black",linestyle="--",label="same threshold")
        ax.set(xlabel="Input timestamp (s)",ylabel="Shaking residual (m)",title=f"Raw track {original_ids[j]} / estimated {ENTITY_NAMES[int(entity[j])]} / {direction}")
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(visdir/(name+"_residual.png"),dpi=150);plt.close(fig)
        metadata.append({"raw_track_id":int(original_ids[j]),"candidate_index":int(j),
                         "estimated_entity":ENTITY_NAMES[int(entity[j])],"direction":direction,
                         "trigger_frame":trigger,"storyboard_frames":indices.tolist(),
                         "valid_slots":int(mask[:,j].sum()),
                         "all_branch_keep":{k:bool(v[j]) for k,v in keep_map.items()},
                         "storyboard":str(visdir/(name+".png")),
                         "overview":str(visdir/(name+"_overview.png")),
                         "residual_plot":str(visdir/(name+"_residual.png"))})
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("/home/cai_tianshun/Project/HOI"))
    p.add_argument("--output",type=Path)
    a=p.parse_args();root=a.root.resolve()
    output=(a.output or root/"experiments/mosca_validation_20260923/time_filter").resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():
        raise RuntimeError("Results already exist; use a new --output directory for recomputation")
    # Keep the executed adapter even when its working copy is changed later.
    (output/"executed_diagnostic.py").write_bytes(Path(__file__).read_bytes())
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(output/"run.log")])
    start=time.perf_counter();cpu_start=time.process_time()
    repo=root/"third_party/MoSca";old=root/"experiments/mosca_baseline_20260922"
    ws=old/"common_input";geometry=old/"mosca_attempt02_missing_flow"
    sys.path.insert(0,str(repo))
    from lib_prior.prior_loading import Saved2D
    from lib_moca.camera import MonocularCameras
    from lib_mosca.dynamic_solver_utils import prepare_track_buffers,get_world_points
    funcs=extract_upstream_functions(repo/"lib_mosca/dynamic_solver.py",output/"upstream_pure_functions.py")
    cfg=OmegaConf.load(geometry/"config.yaml")
    manifest=json.loads((ws/"input_manifest.json").read_text())
    times=np.asarray(manifest["timestamp_seconds"],dtype=np.float64)
    assert len(times)==114 and np.all(np.diff(times)>0)
    assert float(cfg.dep_median)==-1 and int(cfg.scf_geo_keyframe_rate)==1
    threshold=float(cfg.get_curve_refilter_shaking_th_world)
    min_valid=int(cfg.dyn_id_cnt)
    assert threshold==.075 and cfg.get_curve_refilter_remove_shaking_curve
    assert not cfg.get_dynamic_curves_filter_factor_in_world
    s2d=(Saved2D(str(ws)).load_dep("unidepth_depth",float(cfg.depth_boundary_th))
         .normalize_depth(float(cfg.dep_median)).recompute_dep_mask(float(cfg.depth_boundary_th))
         .load_track("cotracker",min_valid_cnt=int(cfg.tap_loading_min_valid_cnt))
         .rescale_perframe_depth_from_bundle(str(geometry/"bundle/bundle.pth")))
    # Recover original IDs with the exact loader truncation and validity protocol.
    raw=np.load(ws/"uniform_cotracker_tap.npz")
    raw_tracks=raw["tracks"];raw_vis=raw["visibility"]
    integer=np.trunc(raw_tracks).astype(np.int64)
    inside=((integer[...,0]>=0)&(integer[...,0]<s2d.W)&(integer[...,1]>=0)&(integer[...,1]<s2d.H))
    raw_valid=raw_vis & inside
    depmask=s2d.dep_mask.numpy()
    for t in range(len(times)):
        ids=np.flatnonzero(raw_valid[t]);uv=integer[t,ids]
        raw_valid[t,ids] &= depmask[t,uv[:,1],uv[:,0]]
    loaded_keep=raw_valid.sum(0)>=int(cfg.tap_loading_min_valid_cnt)
    loaded_ids=np.flatnonzero(loaded_keep)
    assert np.array_equal(s2d.track_mask.numpy(),raw_valid[:,loaded_keep])
    assert np.array_equal(s2d.track.numpy(),integer[:,loaded_keep].astype(np.float32))
    ident=np.load(geometry/"track_identification.npz")
    dynamic=ident["dynamic_track_mask"];static=ident["static_track_mask"]
    assert len(dynamic)==len(loaded_ids)
    original_ids=loaded_ids[dynamic]
    track=s2d.track[:,dynamic];valid=s2d.track_mask[:,dynamic]
    assert len(loaded_ids)==2774 and len(original_ids)==2050
    cams=MonocularCameras.load_from_ckpt(torch.load(geometry/"bundle/bundle_cams.pth",map_location="cpu",weights_only=False))
    assert cams.q_wc.device.type=="cpu" and s2d.track.device.type=="cpu"
    homo,dep,_=prepare_track_buffers(s2d,track,valid,torch.arange(len(times)))
    observed=get_world_points(homo,dep,cams,torch.arange(len(times)))
    index_filled=funcs["line_segment_init"](valid,observed)
    time_filled=fill_by_seconds(observed,valid,times)
    assert torch.equal(index_filled[valid],time_filled[valid])
    o3d_mask=funcs["slot_o3d_outlier_identifyication"](index_filled,valid,nb_neighbors=16,std_ratio=5.)
    same_valid=valid & o3d_mask
    residual_map={
        "original":residuals(index_filled,times),
        "time_midpoint_only":residuals(index_filled,times,True,False),
        "endpoint_exempt_only":residuals(index_filled,times,False,True),
        "time_midpoint_endpoint_exempt":residuals(index_filled,times,True,True),
        "time_fill_midpoint_endpoint_exempt":residuals(time_filled,times,True,True),
    }
    assert torch.equal(residual_map["original"] < threshold,
                       funcs["curve_shaking_identification"](index_filled,threshold))
    keep_map={};bad_map={}
    for name,values in residual_map.items():
        keep_map[name],bad_map[name]=decisions(values,same_valid,threshold,min_valid)
    replay={"raw_tracks":int(len(raw_tracks[0])),"loaded_tracks":int(len(loaded_ids)),
            "dynamic_candidates":int(len(original_ids)),"static_tracks_not_filtered":int(static.sum()),
            "unassigned_loaded_tracks_not_filtered":int((~static & ~dynamic).sum()),
            "slots_before_o3d":int(valid.sum()),"slots_after_o3d":int(same_valid.sum()),
            "shaking_invalid_slot_ratio_including_unobserved":float((residual_map["original"]>=threshold).float().mean()),
            "shaking_removed_curves":int(bad_map["original"].any(0).sum()),
            "original_kept_curves":int(keep_map["original"].sum())}
    assert replay["slots_before_o3d"]==126516,replay
    assert replay["slots_after_o3d"]==125080,replay
    assert replay["shaking_removed_curves"]==1289,replay
    assert replay["original_kept_curves"]==761,replay
    logging.info("Original replay matches saved training log: %s",replay)
    synthetic=synthetic_checks(times,funcs,threshold)
    seg=np.load(old/"segmentation/segmentation.npz")
    labels=seg["entity_labels"]
    uv=track.numpy().astype(np.int64)
    sampled,counts,shares,entity=entity_accounting(labels,uv,same_valid.numpy())
    residual_np={k:v.numpy() for k,v in residual_map.items()}
    statistics={k:branch_statistics(v,keep_map["original"],entity,sampled,same_valid.numpy(),uv,labels) for k,v in keep_map.items()}
    # Exact ID-level decisions, comparable across branches with one common validity mask.
    np.savez_compressed(output/"filter_arrays.npz",raw_track_ids=original_ids,loaded_track_ids=loaded_ids,
                        timestamps=times,uv_integer=uv,uv_original_float=raw_tracks[:,original_ids],
                        observed_xyz=observed.numpy(),index_filled_xyz=index_filled.numpy(),
                        time_filled_xyz=time_filled.numpy(),mask_before_o3d=valid.numpy(),
                        mask_after_o3d=same_valid.numpy(),estimated_slot_labels=sampled,
                        estimated_curve_entity=entity,estimated_curve_entity_share=shares,
                        **{"keep_"+k:v for k,v in keep_map.items()},
                        **{"residual_"+k:v for k,v in residual_np.items()},
                        **{"bad_slot_"+k:v for k,v in bad_map.items()})
    with open(output/"curve_decisions.csv","w",newline="") as f:
        w=csv.writer(f);w.writerow(["raw_track_id","estimated_entity","entity_share","valid_slots","background_slots","person_slots","object_slots",*BRANCHES])
        for j,raw_id in enumerate(original_ids):
            w.writerow([raw_id,ENTITY_NAMES[int(entity[j])],shares[j],int(same_valid[:,j].sum()),*counts[j].tolist(),*[int(keep_map[b][j]) for b in BRANCHES]])
    visuals=make_visualizations(output,manifest,uv,raw_tracks[:,original_ids],same_valid.numpy(),entity,
                                original_ids,residual_np,keep_map,threshold,times)
    endpoint_bad=bad_map["original"][[0,-1]].any(0)
    interior_bad=bad_map["original"][1:-1].any(0)
    comparison={"removed_with_only_global_endpoint_failures":int((endpoint_bad & ~interior_bad).sum()),
                "removed_with_only_interior_failures":int((interior_bad & ~endpoint_bad).sum()),
                "removed_with_endpoint_and_interior_failures":int((interior_bad & endpoint_bad).sum()),
                "time_fill_additional_rescues_over_time_midpoint_and_endpoints":int((keep_map[BRANCHES[-1]] & ~keep_map[BRANCHES[-2]]).sum()),
                "time_fill_additional_rejections_over_time_midpoint_and_endpoints":int((~keep_map[BRANCHES[-1]] & keep_map[BRANCHES[-2]]).sum())}
    paths=[ws/"input_manifest.json",ws/"uniform_cotracker_tap.npz",geometry/"config.yaml",geometry/"track_identification.npz",
           geometry/"bundle/bundle_cams.pth",geometry/"bundle/bundle.pth",geometry/"training.log",old/"segmentation/segmentation.npz",
           old/"segmentation/segmentation_run.json",Path(__file__).resolve(),
           repo/"lib_mosca/dynamic_solver.py",repo/"lib_mosca/dynamic_solver_utils.py",repo/"lib_prior/prior_loading.py",repo/"lib_moca/camera.py"]
    files={str(p):sha(p) for p in paths}
    depth_hashes={str(ws/"unidepth_depth"/f"{i:05d}.npz"):sha(ws/"unidepth_depth"/f"{i:05d}.npz") for i in range(len(times))}
    rgb_hashes={p:sha(p) for p in manifest["frame_paths"]}
    assert list(rgb_hashes.values())==manifest["frame_sha256"]
    assert not torch.cuda.is_initialized()
    summary={"status":"completed","scope":"CPU diagnostic only; no training; no evaluation reference geometry used",
             "roles":{"tracking":"frozen estimated CoTracker tracks and visibility","depth":"frozen input-only UniDepth",
                      "camera":"same fixed calibrated camera checkpoint as original geometry stage",
                      "segmentation":"frozen SAM2 estimated labels; used for accounting and examples only, never for filtering"},
             "endpoint_rule":"Only global first and last frame are exempt from the shaking residual test in named branches. Original visibility, depth, O3D and min-valid checks remain unchanged.",
             "entity_rule":"Majority over common valid slots from SAM2: >=80% and >=2 observations -> person/object/background, otherwise unknown. These are estimated labels, not physical point identities.",
             "coverage_definition":"Unique observed integer track pixels / estimated entity mask pixels; no radius dilation, no geometric surface coverage claim.",
             "controlled_variables":{"shaking_threshold_m":threshold,"min_valid_slots":min_valid,"remove_entire_curve_if_any_valid_slot_fails":True,
                                      "o3d_nb_neighbors":16,"o3d_std_ratio":5.,"same_validity_for_all_branches":True,
                                      "same_observed_xyz_for_all_branches":True,"normalized_depth_scale":1.0,
                                      "backprojection":"Exact legacy MoSca homo_map and camera operations replayed; no principal-point correction mixed into this time test."},
             "original_replay":replay,"synthetic_checks":synthetic,"branch_statistics":statistics,
             "endpoint_and_gap_attribution":comparison,"visual_examples":visuals,
             "file_sha256":files,"depth_file_sha256":depth_hashes,"rgb_file_sha256":rgb_hashes,
             "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"open3d":o3d.__version__,
                            "cuda_initialized":False,"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],"torch_cpu_threads":torch.get_num_threads()},
             "cost":{"wall_seconds":time.perf_counter()-start,"process_cpu_seconds":time.process_time()-cpu_start,
                     "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}
    json_dump(output/"results.json",summary)
    compact={"original_replay":replay,"synthetic_checks":synthetic,"branches":{k:{q:v for q,v in s.items() if q!='per_frame_rgb_label_coverage'} for k,s in statistics.items()},"attribution":comparison,"cost":summary["cost"],"visuals":visuals}
    json_dump(output/"summary.json",compact)
    logging.info("Completed: %s",{k:{q:v for q,v in s.items() if q in ['kept','rescued_vs_original','newly_rejected_vs_original']} for k,s in statistics.items()})
    print(json.dumps({"output":str(output),"replay":replay,"attribution":comparison,"cost":summary["cost"]},indent=2))


if __name__=="__main__":
    main()
