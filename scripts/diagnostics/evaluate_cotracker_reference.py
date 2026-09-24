#!/usr/bin/env python3
"""CPU 2D/frozen-depth diagnostics for fixed CoTracker queries, no fitted alignment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from evaluate_trajectories import write_csv, main as evaluate_3d
from mesh_query_reference import load_mesh, intersect_mesh


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phases_from_rgb(times, event_selection):
    """Only the pre-existing manually inspected RGB brackets; +/-1ms handles rounded labels."""
    phase=np.full(len(times),"outside_annotated_event",dtype="<U32")
    event=event_selection["first_event"]
    for field,label in [("visible_before_s","before"),("partial_occlusion_s","partial_occlusion"),
                        ("severe_occlusion_visual_s","severe_occlusion"),("reappearance_s","reappearance")]:
        start,end=event[field];phase[(times>=start-.001)&(times<=end+.001)]=label
    start,end=event_selection["putdown_visual_s"];phase[(times>=start)&(times<=end)]="putdown"
    return phase


def summary_2d(error, valid, present):
    available=valid & present & np.isfinite(error);values=error[available];count=int(valid.sum())
    result={"expected_samples":count,"predicted_samples":int(available.sum()),
            "coverage":float(available.sum()/count) if count else None,
            "mean_epe_pixels":float(values.mean()) if len(values) else None,
            "median_epe_pixels":float(np.median(values)) if len(values) else None,
            "p90_epe_pixels":float(np.percentile(values,90)) if len(values) else None}
    for threshold in [5,10,20]:result[f"success_at_{threshold}px_missing_as_failure"]=float((available&(error<=threshold)).sum()/count) if count else None
    return result


def sample_depth(depth, uv):
    h,w=depth.shape; uv=np.asarray(uv,float)
    finite=np.isfinite(uv).all(-1);inside=finite&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
    result=np.full(len(uv),np.nan)
    for i in np.flatnonzero(inside):
        x,y=uv[i];x0=min(int(np.floor(x)),w-2);y0=min(int(np.floor(y)),h-2);dx=x-x0;dy=y-y0
        support=depth[y0:y0+2,x0:x0+2]
        if np.isfinite(support).all() and (support>0).all():
            result[i]=np.sum(support*np.array([[(1-dx)*(1-dy),dx*(1-dy)],[(1-dx)*dy,dx*dy]]))
    return result


def lift_visible_to_world(depth, points, visibility, k, c2w):
    """Use camera-z depth only for tracker-visible points, then one known c2w."""
    z=sample_depth(depth,np.where(np.asarray(visibility)[:,None],points,np.nan))
    camera=np.column_stack([(points[:,0]-k[0,2])*z/k[0,0],(points[:,1]-k[1,2])*z/k[1,1],z])
    return camera@c2w[:3,:3].T+c2w[:3,3]


def mesh_visibility_proxy(ref, origin, mesh_manifests, width, height, tolerance):
    specifications=[json.loads(path.read_text()) for path in mesh_manifests]
    times=ref["nominal_frame_times"]
    sources=[];proxy=np.full(ref["valid_mask"].shape,-1,np.int8);reason=np.full(proxy.shape,"unknown",dtype="<U40")
    for ti,t in enumerate(times):
        meshes=[]
        for spec in specifications:
            matches=[f for f in spec["frames"] if f["time"]==float(t)]
            if len(matches)!=1: raise ValueError("mesh manifest has no unique nominal-time match")
            path=Path(matches[0]["path"]);meshes.append(load_mesh(path));sources.append({"path":str(path),"sha256":sha(path)})
        for qi,xyz in enumerate(ref["reference"][ti]):
            if not ref["valid_mask"][ti,qi]:continue
            u,v=ref["projected_uv"][ti,qi]
            if not np.isfinite([u,v]).all() or not (0<=u<width and 0<=v<height) or ref["camera_depth_m"][ti,qi]<=0:
                proxy[ti,qi]=0;reason[ti,qi]="outside_camera";continue
            direction=xyz-origin;distance=np.linalg.norm(direction);direction/=distance
            hits=[intersect_mesh(origin,direction,vertices,faces) for vertices,faces in meshes]
            hitdist=[hit[2] for hit in hits if hit is not None]
            if not hitdist:reason[ti,qi]="raycast_no_hit";continue
            proxy[ti,qi]=int(min(hitdist)>=distance-tolerance)
            reason[ti,qi]="no_known_mesh_in_front" if proxy[ti,qi] else "known_mesh_occludes"
    return proxy,reason,sources


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks",required=True,type=Path);p.add_argument("--input-manifest",required=True,type=Path)
    p.add_argument("--reference",required=True,type=Path);p.add_argument("--reference-protocol",required=True,type=Path)
    p.add_argument("--frozen-queries",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--mesh-manifests",nargs=2,type=Path,help="same-version hand/object meshes; optional geometric visibility proxy")
    p.add_argument("--ray-tolerance-m",type=float,default=.005)
    p.add_argument("--depth-dir",type=Path);p.add_argument("--depth-run",type=Path)
    a=p.parse_args(argv)
    if a.output.exists() and any(a.output.iterdir()):raise ValueError("output must be new/empty")
    start=time.perf_counter();manifest=json.loads(a.input_manifest.read_text());protocol=json.loads(a.reference_protocol.read_text())
    frozen=json.loads(a.frozen_queries.read_text());n=frozen["manual_evaluation_count"]
    with np.load(a.tracks,allow_pickle=False) as z:tracks=z["tracks"][:,:n].copy();visibility=z["visibility"][:,:n].copy();query=z["query_points"][:n].copy();times=z["timestamp_seconds"].copy()
    with np.load(a.reference,allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
    if not np.array_equal(query[:,1:],np.array(frozen["points"][:n])) or not (query[:,0]==0).all():raise ValueError("tracker query order differs from frozen RGB queries")
    if not np.array_equal(times,np.array(manifest["timestamp_seconds"])):raise ValueError("tracker timestamps differ from input")
    if not np.array_equal(ref["query_id"],np.array(frozen["point_ids"][:n])):raise ValueError("reference query IDs differ")
    indices=ref["prediction_frame_indices"];pred=tracks[indices];predvis=visibility[indices]
    if not np.array_equal(times[indices],ref["frame_times"]):raise ValueError("reference observation times differ")
    phase=phases_from_rgb(times,manifest["event_selection"]);rphase=phase[indices]
    error=np.linalg.norm(pred-ref["projected_uv"],axis=-1)
    valid=ref["valid_mask"] & np.isfinite(ref["projected_uv"]).all(-1)
    present=np.isfinite(pred).all(-1)
    rows=[]
    for mode,available in [("all_tracker_coordinates",present),("tracker_declared_visible_only",present&predvis)]:
        for entity in ["all",*np.unique(ref["entity"]).tolist()]:
            emask=np.ones_like(valid) if entity=="all" else np.broadcast_to(ref["entity"]==entity,valid.shape)
            for label in ["all",*np.unique(rphase).tolist()]:
                pmask=np.ones_like(valid) if label=="all" else np.broadcast_to((rphase==label)[:,None],valid.shape)
                rows.append({"mode":mode,"entity":entity,"manual_rgb_phase":label,**summary_2d(error,valid&emask&pmask,available)})
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/"projection_error_summary.csv",rows)
    perpoint=[]
    for ti,source in enumerate(indices):
        for qi,qid in enumerate(ref["query_id"]):
            perpoint.append({"reference_index":ti,"input_frame_index":int(source),"time_s":float(times[source]),"nominal_reference_time_s":float(ref["nominal_frame_times"][ti]),"query_id":qid,"entity":ref["entity"][qi],"manual_rgb_phase":rphase[ti],"tracker_visible":bool(predvis[ti,qi]),"reference_visibility":-1,"epe_pixels":float(error[ti,qi]) if valid[ti,qi] and present[ti,qi] else None})
    write_csv(a.output/"projection_error_per_query.csv",perpoint)
    phase_rows=[{"frame_index":i,"timestamp_seconds":float(t),"phase":str(phase[i]),"source":"pre-existing manual input-RGB brackets","dense_reference_visibility":"unknown"} for i,t in enumerate(times)]
    write_csv(a.output/"manual_rgb_phases_114.csv",phase_rows)
    visibility_rows=[{"query_id":str(q),"entity":str(e),"tracker_visible_frames":int(visibility[:,i].sum()),"tracker_invisible_frames":int((~visibility[:,i]).sum()),"total_frames":len(times),"reference_visibility":"unknown"} for i,(q,e) in enumerate(zip(ref["query_id"],ref["entity"]))]
    write_csv(a.output/"tracker_visibility.csv",visibility_rows)
    report={"schema_version":1,"method":"CoTracker projected-point diagnostic","input_sha256":{str(x.resolve()):sha(x) for x in (a.tracks,a.input_manifest,a.reference,a.reference_protocol,a.frozen_queries)},"evaluator_sha256":sha(__file__),"image_size":[manifest["width"],manifest["height"]],"reference_role":"registered_reference","reference_visibility":"unknown","reference_used_for_tracker_or_depth":False,"alignment":"none","summary":rows,"tracker_visibility":visibility_rows,"phase_source":manifest["event_selection"],"limitations":["Six fixed queries and14 fitted-reference observations are a diagnostic, not a standard benchmark","Reference is fitted surface projection including hidden surfaces; dense visibility ground truth unavailable","Visibility-conditioned mean must be read with coverage; predicted invisible points remain in all-coordinate 2D diagnostic","Reference sampling is sparse; one severe-occlusion reference time cannot establish a general causal conclusion","Nominal/reference RGB timing and fit/hand-glove association uncertainties remain"]}
    if a.mesh_manifests:
        proxy,reason,sources=mesh_visibility_proxy(ref,np.asarray(manifest["c2w"])[:3,3],a.mesh_manifests,manifest["width"],manifest["height"],a.ray_tolerance_m)
        np.savez_compressed(a.output/"fitted_mesh_visibility_proxy.npz",visibility_proxy=proxy,reason=reason,frame_times=ref["frame_times"],query_id=ref["query_id"])
        eligible=valid&(proxy>=0);report["fitted_mesh_visibility_proxy"]={"role":"same-version registered-geometry proxy, not GT","tolerance_m":a.ray_tolerance_m,"visible_samples":int((proxy==1).sum()),"occluded_or_outside_samples":int((proxy==0).sum()),"agreement_with_tracker":float(((proxy==predvis)&eligible).sum()/eligible.sum()) if eligible.any() else None,"sources":sources,"limitation":"Only human/object meshes; no scene occluders, fitted geometry/material association may be wrong; does not replace unknown GT visibility"}
    # Optional frozen-depth lifting, gated on complete correct preprocessing run.
    if a.depth_dir:
        if not a.depth_run:raise ValueError("--depth-run required with --depth-dir")
        record=json.loads(a.depth_run.read_text())
        if record.get("status")!="completed":raise ValueError("depth run is not completed; partial/failed depth forbidden")
        if record.get("input_manifest_sha256")!=sha(a.input_manifest) or record.get("input_frames")!=len(times):raise ValueError("depth manifest identity/count mismatch")
        k=np.asarray(manifest["K"]);c2w=np.asarray(manifest["c2w"]);world=np.full((len(times),n,3),np.nan);depth_sources=[]
        for ti,points in enumerate(tracks):
            path=a.depth_dir/f"{ti:05d}.npz"
            with np.load(path,allow_pickle=False) as z:depth=z["dep"]
            if depth.shape!=(manifest["height"],manifest["width"]) or not np.isfinite(depth).all() or (depth<=0).any():raise ValueError(f"invalid complete depth frame {ti}")
            world[ti]=lift_visible_to_world(depth,points,visibility[ti],k,c2w)
            depth_sources.append({"path":str(path.resolve()),"sha256":sha(path)})
        np.savez_compressed(a.output/"lifted_tracks_full.npz",predicted=world,predicted_valid_mask=np.isfinite(world).all(-1),frame_times=times,query_id=ref["query_id"],entity=ref["entity"],units=np.array("m"),coordinate_frame=ref["coordinate_frame"],tracker_visibility=visibility)
        np.savez_compressed(a.output/"evaluation_bundle_3d.npz",predicted=world[indices],reference=ref["reference"],valid_mask=ref["valid_mask"],predicted_valid_mask=np.isfinite(world[indices]).all(-1),frame_times=ref["frame_times"],entity=ref["entity"],query_id=ref["query_id"],units=np.array("m"),coordinate_frame=ref["coordinate_frame"],visibility=np.full(valid.shape,-1,np.int8))
        three_protocol={**protocol,"visibility_source":None,"prediction_method":"CoTracker visible-only 2D plus frozen UniDepthV2 camera-z bilinear sample, known global c2w; no scale fitting","reference_used_for_training":False,"depth_run_sha256":sha(a.depth_run),"depth_sources":depth_sources,"global_transform_provenance":"known calibrated BEHAVE camera0 local-to-world supplied as legal input; no reference fitting"}
        protocol_path=a.output/"protocol_3d.json";protocol_path.write_text(json.dumps(three_protocol,indent=2)+"\n")
        events=[]
        before=np.flatnonzero(rphase=="before");during=np.flatnonzero(np.isin(rphase,["partial_occlusion","severe_occlusion"]));after=np.flatnonzero(rphase=="reappearance")
        if len(during):
            events=[{"event_id":"manual_RGB_first_occlusion_sparse_observations","occlusion_start":int(during[0]),"occlusion_end":int(during[-1]+1),"pre_frames":int(during[0]-before[0]) if len(before) else 0,"post_frames":int(after[-1]-during[-1]) if len(after) else 0,"track_indices":list(range(n)),"source":"pre-existing manual RGB event, not tracker/SAM2 labels; sample visibility unknown"}]
        event_path=a.output/"events_3d.json";event_path.write_text(json.dumps(events,indent=2)+"\n")
        evaluate_3d(["--input",str(a.output/"evaluation_bundle_3d.npz"),"--output",str(a.output/"three_dimensional"),"--protocol",str(protocol_path),"--events",str(event_path)])
        report["three_dimensional"]={"status":"completed","protocol":str(protocol_path.resolve()),"note":"Predicted invisible points have no sampled depth and count missing; not an occluded-geometry reconstruction method"}
    else:report["three_dimensional"]={"status":"not_requested","note":"Depth not read; wait for completed preprocessing"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    for qi,qid in enumerate(ref["query_id"]):axes[0].plot(ref["frame_times"],error[:,qi],marker=".",label=qid)
    axes[0].set(ylabel="Projection endpoint error (pixels)",title="CoTracker coordinates vs fitted surface projection (14 observations)");axes[0].legend(fontsize=8,ncol=2)
    axes[1].imshow(visibility.T,aspect="auto",origin="lower",cmap="gray_r",vmin=0,vmax=1)
    axes[1].set(yticks=range(n),yticklabels=ref["query_id"],xlabel="Processed frame index (irregular physical timestamps)",title="Tracker predicted visibility: dark=visible; reference visibility unknown")
    fig.savefig(a.output/"tracker_projection_and_visibility.png",dpi=150);plt.close(fig)
    report["evaluation_wall_seconds"]=time.perf_counter()-start
    (a.output/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"report":str((a.output/"report.json").resolve()),"overall":rows[0]},ensure_ascii=False))
    return 0


if __name__=="__main__":raise SystemExit(main())
