"""Bounded CPU-only descriptive audit. No reference geometry or GPU is loaded."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
start = time.perf_counter()
sources = {}

def source(relative):
    path = ROOT / relative
    sources[relative] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return path

def summary(values):
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0}
    qs = np.quantile(a, [0, .1, .25, .5, .75, .9, .95, 1])
    return dict(count=len(a), mean=float(a.mean()), **dict(zip(
        ["min", "p10", "p25", "median", "p75", "p90", "p95", "max"], qs.tolist())))

manifest = json.loads(source("common_input/input_manifest.json").read_text())
queries = json.loads(source("common_input/queries_first_frame.json").read_text())
raw = np.load(source("common_input/uniform_cotracker_tap.npz"))
final = np.load(source("mosca_cotracker/diagnostics/scaffold_nodes.npz"))
query_result = np.load(source("mosca_cotracker/diagnostics/query_trajectories.npz"))
geo = torch.load(source("mosca_cotracker/mosca/mosca.pth"), map_location="cpu", weights_only=False)
bundle = torch.load(source("mosca_cotracker/bundle/bundle.pth"), map_location="cpu", weights_only=False)
identified = np.load(source("mosca_cotracker/track_identification.npz"))
depth = np.load(source("common_input/unidepth_depth/00000.npz"))["dep"]
cfg = yaml.safe_load(source("mosca_cotracker/config.yaml").read_text())
run = json.loads(source("mosca_cotracker/run.json").read_text())
geo_log = source("mosca_attempt02_missing_flow/training.log").read_text().splitlines()
photo_log = source("mosca_cotracker/dynamic_reconstruction_20260922_105919.log").read_text().splitlines()
K = np.asarray(manifest["K"])
c2w = np.asarray(manifest["c2w"])
R, tr = c2w[:3, :3], c2w[:3, 3]
times = np.asarray(manifest["timestamp_seconds"])
T = len(times)
uvq = np.asarray(queries["points"][:6], dtype=float)
zq = depth[uvq[:, 1].astype(int), uvq[:, 0].astype(int)]
cameraq = (np.c_[uvq, np.ones(6)] @ np.linalg.inv(K).T) * zq[:, None]
worldq = cameraq @ R.T + tr

def project(xyz):
    camera = (xyz - tr) @ R
    hom = camera @ K.T
    return hom[..., :2] / hom[..., 2:], camera[..., 2]

stages = {}
for name, xyz, certain, grouping, tids in [
    ("geometry_checkpoint", geo["_node_xyz"].numpy(), geo["_node_certain"].numpy(), geo["_node_grouping"].numpy(), geo["_t_list"].numpy()),
    ("final_photometric_export", final["node_xyz"], final["node_certain_prior"], final["node_grouping"], final["source_frame_index"]),
]:
    xyz = xyz.astype(float)
    certain = certain.astype(bool)
    displacement = np.linalg.norm(xyz - xyz[:1], axis=-1)
    max_motion = displacement.max(0)
    uv, z = project(xyz)
    stage = dict(shape=list(xyz.shape), finite=bool(np.isfinite(xyz).all()),
        source_frame_index=tids.tolist(), time_index_matches_manifest=bool(np.array_equal(tids, np.arange(T))),
        max_from_first_per_node_m=summary(max_motion), final_from_first_per_node_m=summary(displacement[-1]),
        adjacent_step_m=summary(np.linalg.norm(np.diff(xyz, axis=0), axis=-1)),
        nodes_with_max_displacement_under_m={str(th): int((max_motion < th).sum()) for th in [.001, .01, .05, .1]},
        certain_support_frames_per_node=summary(certain.sum(0)), certain_supported_nodes_per_frame=certain.sum(1).tolist(),
        group_counts={str(int(k)): int(v) for k,v in zip(*np.unique(grouping, return_counts=True))},
        per_query_neighborhood=[])
    for qi, point in enumerate(uvq):
        pix_dist = np.linalg.norm(uv[0] - point, axis=-1)
        pix_dist[z[0] <= 0] = np.inf
        world_dist = np.linalg.norm(xyz[0] - worldq[qi], axis=-1)
        output_dist = np.linalg.norm(xyz[0] - query_result["predicted"][0, qi], axis=-1)
        def node_info(ids):
            return [dict(node_index=int(j), pixel_distance_at_first=float(pix_dist[j]),
                input_depth_lift_distance_at_first_m=float(world_dist[j]),
                rendered_query_distance_at_first_m=float(output_dist[j]),
                camera_z_first_m=float(z[0,j]), max_from_first_m=float(max_motion[j]),
                final_from_first_m=float(displacement[-1,j]), prior_certain_frames=int(certain[:,j].sum()),
                certain_frame_indices=np.where(certain[:,j])[0].tolist(),
                prior_support_times_seconds=times[certain[:,j]].tolist(),
                world_positions_at_input_frames={str(t):xyz[t,j].tolist() for t in [0,16,25,31,55,65,113]}) for j in ids]
        local = pix_dist < 25
        stage["per_query_neighborhood"].append(dict(query_id=queries["point_ids"][qi],
            input_uv=point.tolist(), input_depth_m=float(zq[qi]), input_depth_lift_world_m=worldq[qi].tolist(),
            projected_radius25_node_count=int(local.sum()), projected_radius25_max_motion_m=summary(max_motion[local]),
            projected_radius25_certain_node_count_each_frame=certain[:,local].sum(1).tolist(),
            nearest_eight_by_first_projection=node_info(np.argsort(pix_dist)[:8]),
            nearest_eight_by_first_input_depth_lift=node_info(np.argsort(world_dist)[:8]),
            nearest_node_to_first_rendered_query=node_info([np.argmin(output_dist)])[0]))
    stages[name] = stage

raw_tracks = raw["tracks"].astype(float)
raw_visibility = raw["visibility"].astype(bool)
raw_query_diagnostics = []
for qi, point in enumerate(uvq):
    source_query_distance = np.linalg.norm(raw["query_points"][:, 1:] - point, axis=-1)
    source_query_distance[raw["query_points"][:, 0] != 0] = np.inf
    j = int(np.argmin(source_query_distance))
    assert source_query_distance[j] == 0
    visible = raw_visibility[:,j]
    motion = np.linalg.norm(raw_tracks[:,j] - raw_tracks[0,j], axis=-1)
    raw_query_diagnostics.append(dict(query_id=queries["point_ids"][qi], raw_track_index=j,
        visible_frame_indices=np.where(visible)[0].tolist(), visible_count=int(visible.sum()),
        predicted_initial_uv_offset_px=float(np.linalg.norm(raw_tracks[0,j] - point)),
        max_2d_displacement_all_coordinates_px=float(motion.max()),
        max_2d_displacement_visible_coordinates_px=float(motion[visible].max()),
        note="Tracker visibility is model output, not independently annotated visibility; invisible coordinates are descriptive only."))

report = dict(status="completed_cpu_only", coordinate_frame="behave_world_k1_color", units="m",
    protocol={"no_reference_geometry_or_alignment":True, "no_gpu_or_model_mutation":True,
        "max_displacement_definition":"For each node, largest Euclidean displacement from its own first-frame position over all 114 input frames. Descriptive only; not accuracy, velocity or a benchmark score.",
        "neighborhood_definition":"Fixed first-frame projection radius 25 pixels, or 8 nearest first-frame input-UniDepth-lift positions. Neither assigns object identity or rebinds official queries.",
        "support_definition":"Saved node_certain_prior is inherited prior support, not ground-truth visibility or final confidence. Node indices differ across geometry and pruned final stages; no cross-stage index matching is assumed."},
    time={"input_frames":T, "actual_seconds":[float(times[0]),float(times[-1])],
        "delta_seconds":summary(np.diff(times)),
        "final_export_times_equal_manifest":bool(np.array_equal(final["input_frame_times"], times))},
    stages=stages,
    input_tracks={"shape":list(raw_tracks.shape), "visible_frames_per_track":summary(raw_visibility.sum(0)),
        "visible_tracks_per_frame":raw_visibility.sum(1).tolist(),
        "loaded_track_count":len(identified["dynamic_track_mask"]),
        "identified_dynamic":int(identified["dynamic_track_mask"].sum()),
        "identified_static":int(identified["static_track_mask"].sum()),
        "raw_to_loaded_indices_not_assumed":"48 tracks removed by depth+visibility loader; original raw index is not the index in track_identification.",
        "six_fixed_rgb_queries":raw_query_diagnostics},
    final_render_queries={"query_ids":query_result["query_id"].tolist(),
        "max_from_first_m":np.linalg.norm(query_result["predicted"]-query_result["predicted"][:1],axis=-1).max(0).tolist(),
        "source_dynamic_fraction":query_result["source_dynamic_fraction"].tolist(),
        "source_alpha":query_result["source_alpha"].tolist()},
    configuration={"training_status":run["status"], "metric_depth_retained":cfg["dep_median"] < 0,
        "bundle_depth_scale_all_one":bool(torch.all(bundle["dep_scale"]==1)),
        "world_parameter_values":{k:cfg[k] for k in ["dep_median","mosca_unit_world","get_curve_refilter_shaking_th_world","get_dynamic_curves_filter_factor_in_world","get_curve_refilter_remove_shaking_curve","gs_radius_max","photo_lambda_vel_xyz_reg","photo_lambda_acc_xyz_reg"]},
        "geometry_spatial_unit_m":float(geo["spatial_unit"]),
        "geometry_initial_sigma_m":float(geo["init_sigma"]),"geometry_max_sigma_m":float(geo["max_sigma"]),
        "scale_hypothesis":"Original demo normally normalizes depth; this adapter retains estimated metres while reusing fixed distance thresholds and optimizer/regularizer settings. This is an unverified adaptation concern, not a demonstrated cause. No parameter is changed here.",
        "implementation_context":"normalize_depth(negative) sets scale_nw=1; get_dynamic_curves_filter_factor_in_world=false leaves shaking threshold 0.075 in current coordinates; explicit mosca_unit_world overrides its scale_nw-dependent default. Temporal shifts remain frame-index based."},
    log_evidence={"geometry":[l for l in geo_log if any(s in l for s in ["48 tracks are removed", "Identify 686", "shaking outlier ratio", "1289 curves", "M=761", "Resample node from 761", "hard_fix_valid_flag"])],
        "photometric":[l for l in photo_log if "No node to densify" in l or "Node prune:" in l]},
    source_files=sources, elapsed_cpu_seconds=time.perf_counter()-start)
(OUT/"scaffold_motion_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
print(json.dumps({"output":str(OUT/"scaffold_motion_audit.json"),"seconds":report["elapsed_cpu_seconds"],
    "final_node_motion":stages["final_photometric_export"]["max_from_first_per_node_m"]},indent=2))
