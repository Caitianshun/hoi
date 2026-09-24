#!/usr/bin/env python3
"""Input-only CPU export and bounded object-candidate failure accounting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root",type=Path,default=Path("/home/cai_tianshun/Project/HOI"))
    a.add_argument("--run",type=Path)
    a.add_argument("--output",type=Path)
    args=a.parse_args();root=args.root.resolve()
    base=root/"experiments/mosca_validation_20260923/time_filter"
    run=(args.run or base/"run02_visual_trigger_fixed").resolve()
    out=(args.output or base/"prefilter_export").resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/"object_failure_audit.json").exists():raise RuntimeError("Use a new output directory")
    (out/"executed_export.py").write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter();cpustart=time.process_time()
    old=root/"experiments/mosca_baseline_20260922";ws=old/"common_input"
    d=np.load(run/"filter_arrays.npz")
    raw=np.load(ws/"uniform_cotracker_tap.npz")
    manifest=json.loads((ws/"input_manifest.json").read_text())
    labels=np.load(old/"segmentation/segmentation.npz")["entity_labels"]
    ids=d["raw_track_ids"];uv=d["uv_integer"];T,N,_=uv.shape
    vis=raw["visibility"][:,ids].astype(bool)
    inside=(uv[...,0]>=0)&(uv[...,0]<manifest["width"])&(uv[...,1]>=0)&(uv[...,1]<manifest["height"])
    sampled_labels=np.full((T,N),255,np.uint8)
    depth=np.full((T,N),np.nan,np.float32)
    for t in range(T):
        m=inside[t];q=uv[t,m]
        dep=np.load(ws/"unidepth_depth"/f"{t:05d}.npz")["dep"]
        sampled_labels[t,m]=labels[t,q[:,1],q[:,0]]
        depth[t,m]=dep[q[:,1],q[:,0]]
    valid_pre=d["mask_before_o3d"];valid_post=d["mask_after_o3d"]
    xyz=d["observed_xyz"].copy();xyz[~valid_pre]=np.nan
    export=out/"dynamic_candidates_2050.npz"
    np.savez_compressed(export,
        raw_track_id=ids,input_timestamp_seconds=d["timestamps"],
        input_frame_index=np.arange(T,dtype=np.int32),
        video_frame_index=np.asarray(manifest["frame_indices"]),
        track_xy_float=raw["tracks"][:,ids],track_xy_integer=uv,
        cotracker_visibility=vis,in_bounds_after_upstream_integer_truncation=inside,
        valid_before_o3d=valid_pre,valid_after_o3d=valid_post,
        xyz_world_observed_legacy_mosca=xyz,
        sampled_unidepth_at_predicted_pixel=depth,
        estimated_label_at_predicted_pixel=sampled_labels,
        diagnostic_curve_entity=d["estimated_curve_entity"],
        diagnostic_curve_entity_share=d["estimated_curve_entity_share"],
        original_filter_keep=d["keep_original"],
        time_filter_keep=d["keep_time_fill_midpoint_endpoint_exempt"],
        K_input=np.asarray(manifest["K"]),c2w_input=np.asarray(manifest["c2w"]),
        units=np.asarray("m"),coordinate_frame=np.asarray("behave_world_k1_color"),
        role=np.asarray("input_only_frozen_estimates_not_reference_geometry"))
    obj=d["estimated_curve_entity"]==2;assert obj.sum()==280
    original=d["keep_original"];new=d["keep_time_fill_midpoint_endpoint_exempt"]
    bad=d["bad_slot_original"] & obj[None]
    endpoint=bad[[0,-1]].any(0);interior=bad[1:-1].any(0)
    count_bad=bad.sum(0)
    removed=obj & ~original
    # Whether the residual used imputed rather than measured neighboring slots.
    neigh_observed=np.zeros_like(valid_pre)
    neigh_observed[1:-1]=valid_pre[:-2]&valid_pre[2:]
    neigh_observed[0]=valid_pre[1];neigh_observed[-1]=valid_pre[-2]
    bad_with_gap=bad & ~neigh_observed
    bad_all_observed=bad & neigh_observed
    xyz_fill=d["index_filled_xyz"]
    ref=(xyz_fill[2:]+xyz_fill[:-2])/2.
    ref=np.concatenate([xyz_fill[1:2],ref,xyz_fill[-2:-1]],axis=0)
    residual_vector=xyz_fill-ref
    rotation=np.asarray(manifest["c2w"],dtype=np.float32)[:3,:3]
    # Row-vector inverse rotation; translation cancels in the residual.
    residual_cam=residual_vector@rotation
    magnitude=np.linalg.norm(residual_vector,axis=-1)
    depth_dominant=(np.abs(residual_cam[...,2]) >= .8*np.maximum(magnitude,1e-12))
    object_area=(labels==2).sum((1,2));empty=object_area==0
    near_empty=empty.copy()
    for shift in (1,2):
        near_empty[shift:] |= empty[:-shift];near_empty[:-shift] |= empty[shift:]
    frames=[]
    for t in range(T):
        frames.append(dict(frame_index=t,timestamp_seconds=float(d["timestamps"][t]),
            estimated_object_mask_pixels=int(object_area[t]),
            raw_visible_object_candidates=int((vis[t]&inside[t]&obj).sum()),
            valid_pre_o3d_object_candidates=int((valid_pre[t]&obj).sum()),
            valid_post_o3d_object_candidates=int((valid_post[t]&obj).sum()),
            original_kept_valid_object_candidates=int((valid_post[t]&obj&original).sum()),
            time_kept_valid_object_candidates=int((valid_post[t]&obj&new).sum()),
            original_failure_slots=int(bad[t].sum()),
            original_failure_slots_using_imputed_neighbor=int(bad_with_gap[t].sum())))
    label_stats={}
    for label,name in ((0,"background"),(1,"person"),(2,"object"),(255,"unknown")):
        label_stats[name]=dict(
            cotracker_visible_slots=int(((sampled_labels==label)&vis&inside&obj[None]).sum()),
            common_valid_slots=int(((sampled_labels==label)&valid_post&obj[None]).sum()),
            failing_slots=int(((sampled_labels==label)&bad).sum()))
    maxbad=float(d["residual_original"][:,obj].max())
    examples=[]
    chosen=json.loads((run/"results.json").read_text())["visual_examples"]
    for e in chosen:
        j=e["candidate_index"];t=e["trigger_frame"]
        examples.append(dict(raw_track_id=int(ids[j]),estimated_entity=e["estimated_entity"],trigger=t,
            timestamps=[float(x) for x in d["timestamps"][max(0,t-1):min(T,t+2)]],
            unidepth_m=[float(x) for x in depth[max(0,t-1):min(T,t+2),j]],
            source_slot_valid=valid_pre[max(0,t-1):min(T,t+2),j].tolist(),
            residuals_at_trigger={name:float(d["residual_"+name][t,j]) for name in ("original","time_midpoint_only","endpoint_exempt_only","time_fill_midpoint_endpoint_exempt")},
            valid_slots=int(valid_post[:,j].sum())))
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    times=d["timestamps"]
    axes[0].plot(times,(vis&inside&obj[None]).sum(1),label="CoTracker visible / object-majority candidates",alpha=.7)
    axes[0].plot(times,(valid_post&obj[None]).sum(1),label="After depth validity + O3D")
    axes[0].plot(times,(valid_post&obj[None]&original[None]).sum(1),label="Original filter kept",linewidth=2)
    axes[0].plot(times,(valid_post&obj[None]&new[None]).sum(1),label="Time + endpoint + gap fill kept",linewidth=1.5)
    axes[0].set_ylabel("Valid candidate count");axes[0].legend(fontsize=8)
    axes[1].plot(times,bad.sum(1),label="Original threshold failures")
    axes[1].plot(times,bad_with_gap.sum(1),label="Failures with imputed neighbor")
    axes[1].set(xlabel="Input time (s)",ylabel="Failing slots")
    twin=axes[1].twinx();twin.fill_between(times,object_area,alpha=.12,color="gray",label="Estimated SAM2 object mask area")
    twin.set_ylabel("Estimated object mask area (pixels)",color="gray")
    axes[1].legend(fontsize=8,loc="upper left")
    fig.suptitle("280 estimated object-majority dynamic candidates / frozen input-only priors")
    fig.tight_layout();fig.savefig(out/"object_coverage_and_failures.png",dpi=150);plt.close(fig)
    payload={"status":"completed","scope":"CPU, no reference geometry, no training, original 2050 candidates only",
        "object_candidate_definition":"Same >=80% SAM2-label majority over common valid slots as main time-filter experiment; accounting only, not GT",
        "object_candidates":int(obj.sum()),"original_kept":int((obj&original).sum()),"time_kept":int((obj&new).sum()),
        "original_removed":int(removed.sum()),
        "failure_patterns":{"removed_endpoint_only":int((obj&endpoint&~interior).sum()),
            "removed_interior_only":int((obj&interior&~endpoint).sum()),
            "removed_both":int((obj&interior&endpoint).sum()),
            "removed_curves_with_1_or_2_failing_slots":int((removed&(count_bad<=2)).sum()),
            "removed_curves_with_3_to_5_failing_slots":int((removed&(count_bad>=3)&(count_bad<=5)).sum()),
            "removed_curves_with_more_than_5_failing_slots":int((removed&(count_bad>5)).sum()),
            "removed_curves_all_failures_involve_imputed_neighbor":int((removed & ~bad_all_observed.any(0)).sum()),
            "removed_curves_with_at_least_one_all_observed_failure":int((removed & bad_all_observed.any(0)).sum()),
            "total_failing_slots":int(bad.sum()),"failing_slots_with_imputed_neighbor":int(bad_with_gap.sum()),
            "depth_axis_dominant_failing_slots":int((bad&depth_dominant).sum()),
            "depth_axis_dominant_fraction":float((bad&depth_dominant).sum()/bad.sum()),
            "failing_slots_when_estimated_object_mask_empty":int(bad[empty].sum()),
            "failing_slots_within_2_frames_of_empty_object_mask":int(bad[near_empty].sum())},
        "validity_totals":{"all_object_candidate_slots":int(T*obj.sum()),"cotracker_visible_inbounds":int((vis&inside&obj[None]).sum()),
            "before_o3d":int((valid_pre&obj[None]).sum()),"after_o3d":int((valid_post&obj[None]).sum())},
        "rgb_label_evidence":label_stats,"empty_estimated_object_mask_frames":np.flatnonzero(empty).tolist(),
        "per_frame":frames,"selected_example_numbers":examples,
        "interpretation_limits":["A shaking residual is not an independent accuracy metric.",
            "A dominant depth-axis residual suggests inspecting depth estimates; it does not establish a depth error because real 3D motion also changes depth.",
            "Persistent absolute depth bias cannot be diagnosed from shaking or SAM2 masks alone; no independent reference was used.",
            "SAM2 object disappearance or label changes may reflect true occlusion, segmentation error or tracking drift; counts do not disambiguate them.",
            "No normalized/projection-corrected branch outputs were read; this audit preserves the old legacy projection and metric scale."],
        "hashes":{"source_arrays":sha(run/"filter_arrays.npz"),"source_results":sha(run/"results.json"),"export":sha(export),
            "script":sha(__file__),"segmentation":sha(old/"segmentation/segmentation.npz"),"tracks":sha(ws/"uniform_cotracker_tap.npz")},
        "cost":{"wall_seconds":time.perf_counter()-start,"process_cpu_seconds":time.process_time()-cpustart,
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}
    (out/"object_failure_audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    schema={"file":str(export),"sha256":sha(export),"T":T,"N":N,
        "role":"Legal input-only estimated observations, not reference geometry or ground-truth visibility",
        "world_coordinates":"behave_world_k1_color, meters, legacy MoSca backprojection retained to reproduce old filter; do not call exact pinhole K backprojection",
        "projection_recomputation":"track_xy_integer + sampled_unidepth_at_predicted_pixel + K_input + c2w_input allow an explicit separately identified exact-pinhole recomputation; do not silently mix it into these filter results",
        "xyz_validity":"xyz_world_observed_legacy_mosca is NaN wherever valid_before_o3d is false; filter replay's imputed positions are intentionally not called observations",
        "visibility":"cotracker_visibility is a frozen model prediction; valid masks additionally include depth boundaries/O3D, not ground-truth occlusion",
        "depth_at_prediction":"sampled_unidepth_at_predicted_pixel may be another surface at occluded predictions; use only with declared validity",
        "entity_labels":{"0":"estimated background","1":"estimated person","2":"estimated object","255":"unknown/conflicted for diagnostic_curve_entity; out-of-bounds for per-frame labels"},
        "original_indices":"raw_track_id indexes the 2822-track source NPZ; input_frame_index indexes the 114 processed RGB frames; video_frame_index is the original recorded frame ID",
        "source_main_results":str(run/"results.json"),"reference_used":False}
    (out/"dynamic_candidates_2050.schema.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"export":str(export),"failure_patterns":payload["failure_patterns"],"validity":payload["validity_totals"],"labels":label_stats,"examples":examples,"cost":payload["cost"]},indent=2))


if __name__=="__main__":main()
