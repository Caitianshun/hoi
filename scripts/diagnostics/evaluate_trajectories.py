#!/usr/bin/env python3
"""CPU-only, fixed-coordinate trajectory evaluation; never estimates alignment.

See README.md beside this script for the NPZ contract and metric limitations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def scalar_string(arr, name):
    a = np.asarray(arr)
    if a.size != 1 or a.dtype.kind not in "US":
        raise ValueError(f"{name} must be one Unicode/string scalar")
    return str(a.item())


def bool_mask(arr, shape, name):
    a = np.asarray(arr)
    if a.shape != shape or not np.isin(a, [0, 1]).all():
        raise ValueError(f"{name} must be a boolean/0-1 array with shape {shape}")
    return a.astype(bool)


@dataclass
class Trajectories:
    predicted: np.ndarray
    reference: np.ndarray
    valid: np.ndarray
    pred_valid: np.ndarray
    visibility: np.ndarray
    times: np.ndarray
    entity: np.ndarray
    frame: str
    units: str
    pred_identity: np.ndarray | None = None
    reference_identity: np.ndarray | None = None
    query_id: np.ndarray | None = None


def load_bundle(path, units_override=None):
    with np.load(path, allow_pickle=False) as z:
        required = {"predicted", "reference", "valid_mask", "frame_times", "entity"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"NPZ missing required keys: {sorted(missing)}")
        pred = np.array(z["predicted"], dtype=np.float64)
        ref = np.array(z["reference"], dtype=np.float64)
        if pred.ndim != 3 or pred.shape[-1] != 3 or pred.shape != ref.shape:
            raise ValueError("predicted and reference must have identical (T,N,3) shapes")
        t, n, _ = pred.shape
        if not t or not n:
            raise ValueError("empty trajectories are not evaluable")
        valid = bool_mask(z["valid_mask"], (t, n), "valid_mask")
        if not np.isfinite(ref[valid]).all():
            raise ValueError("reference contains nonfinite coordinates at valid_mask=True")
        pred_valid = np.isfinite(pred).all(axis=-1)
        if "predicted_valid_mask" in z:
            pred_valid &= bool_mask(z["predicted_valid_mask"], (t, n), "predicted_valid_mask")
        times = np.asarray(z["frame_times"], dtype=np.float64)
        if times.shape != (t,) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("frame_times must be finite, strictly increasing seconds, shape (T,)")
        entity = np.asarray(z["entity"])
        if entity.shape != (n,) or entity.dtype.kind not in "US" or (entity == "").any():
            raise ValueError("entity must be nonempty Unicode labels, shape (N,)")
        if "units" in z:
            units = scalar_string(z["units"], "units")
            if units_override and units != units_override:
                raise ValueError("--units conflicts with NPZ units; do not silently rescale")
        else:
            units = units_override
        if units not in UNIT_SCALE:
            raise ValueError("explicit units are required: NPZ units or --units m/cm/mm")
        if "coordinate_frame" not in z:
            raise ValueError("coordinate_frame is required; camera coordinates must be transformed first")
        frame = scalar_string(z["coordinate_frame"], "coordinate_frame")
        if not frame:
            raise ValueError("coordinate_frame cannot be empty")
        for key in ("predicted_coordinate_frame", "reference_coordinate_frame"):
            if key in z and scalar_string(z[key], key) != frame:
                raise ValueError(f"{key} differs from common coordinate_frame")
        if "visibility" in z:
            vis = np.asarray(z["visibility"])
            if vis.shape != (t, n) or not np.isin(vis, [-1, 0, 1]).all():
                raise ValueError("visibility must have shape (T,N), -1 unknown / 0 occluded / 1 visible")
            vis = vis.astype(np.int8)
        else:
            vis = np.full((t, n), -1, dtype=np.int8)
        pi = ri = None
        if ("predicted_identity" in z) != ("reference_identity" in z):
            raise ValueError("identity evaluation requires both predicted_identity and reference_identity")
        if "predicted_identity" in z:
            pi, ri = np.asarray(z["predicted_identity"]), np.asarray(z["reference_identity"])
            if pi.dtype.kind not in "US" or ri.dtype.kind not in "US":
                raise ValueError("identity arrays must be strings; empty string denotes unknown")
            if pi.shape != (t, n) or ri.shape not in ((t, n), (n,)):
                raise ValueError("predicted_identity: (T,N); reference_identity: (N,) or (T,N)")
            if ri.ndim == 1:
                ri = np.broadcast_to(ri, (t, n)).copy()
        query_id=np.asarray(z["query_id"]).astype(str) if "query_id" in z else np.asarray([f"track{i}" for i in range(n)])
        if query_id.shape!=(n,) or len(set(query_id.tolist()))!=n:
            raise ValueError("query_id must contain one unique string per track")
        scale = UNIT_SCALE[units]
        return Trajectories(pred * scale, ref * scale, valid, pred_valid, vis,
                            times, entity.astype(str), frame, units, pi, ri, query_id)


def validate_protocol(protocol):
    """Detect declared leakage; this cannot certify undeclared data provenance."""
    if protocol.get("reference_used_for_training") is True:
        raise ValueError("reference_used_for_training=true: independent evaluation would be invalid")
    if protocol.get("alignment") not in (None, "none", "predeclared_global_transform"):
        raise ValueError("per-frame/per-entity or fitted-on-evaluation alignment is forbidden")
    if protocol.get("alignment") == "predeclared_global_transform":
        if not protocol.get("global_transform_provenance"):
            raise ValueError("predeclared transform requires global_transform_provenance")
        if protocol.get("global_transform_fitted_on_evaluation") is not False:
            raise ValueError("global transform must explicitly not be fitted on evaluation samples")
    refs, training = set(protocol.get("reference_assets", [])), set(protocol.get("training_assets", []))
    if refs.intersection(training):
        raise ValueError("declared training_assets and reference_assets overlap")
    supported = {"independent_reference", "registered_reference", "tracker_proxy", "synthetic_fixture", "unknown"}
    if protocol.get("reference_role", "unknown") not in supported:
        raise ValueError(f"reference_role must be one of {sorted(supported)}")


def statistics(error, expected, present, thresholds):
    good = expected & present & np.isfinite(error)
    vals = error[good]
    count = int(expected.sum())
    out = {"expected_samples": count, "predicted_samples": int(good.sum()),
           "missing_predictions": int(count - good.sum()),
           "coverage": float(good.sum() / count) if count else None,
           "mean_epe_m": float(vals.mean()) if vals.size else None,
           "median_epe_m": float(np.median(vals)) if vals.size else None,
           "p90_epe_m": float(np.percentile(vals, 90)) if vals.size else None,
           "rmse_m": float(np.sqrt(np.mean(vals**2))) if vals.size else None}
    for threshold in thresholds:
        # Missing predictions count as failures, never disappear from denominator.
        out[f"success_at_{threshold:g}m"] = float((good & (error <= threshold)).sum() / count) if count else None
    return out


def contiguous_true(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    return list(zip(np.where(np.diff(padded) == 1)[0].tolist(),
                    np.where(np.diff(padded) == -1)[0].tolist()))


def automatic_events(data, pre_frames, post_frames):
    events = []
    for n in range(data.predicted.shape[1]):
        for start, end in contiguous_true(data.visibility[:, n] == 0):
            events.append({"event_id": f"track{n}_occlusion_{start}_{end}",
                           "occlusion_start": start, "occlusion_end": end,
                           "track_indices": [n], "pre_frames": pre_frames,
                           "post_frames": post_frames, "source": "provided_visibility"})
    return events


def validate_event(event, data):
    t, n, _ = data.predicted.shape
    for name in ("occlusion_start", "occlusion_end", "pre_frames", "post_frames"):
        if name not in event or isinstance(event[name], bool) or not isinstance(event[name], int):
            raise ValueError(f"event {name} must be an integer")
    a, b = event["occlusion_start"], event["occlusion_end"]
    if not 0 <= a < b <= t or min(event["pre_frames"], event["post_frames"]) < 0:
        raise ValueError("event must use [start,end) inside sequence and nonnegative pre/post lengths")
    if not event.get("event_id"):
        raise ValueError("every event needs event_id")
    if ("track_indices" in event) == ("entity" in event):
        raise ValueError("event needs exactly one of track_indices or entity")
    ids = event.get("track_indices")
    if ids is None:
        ids = np.flatnonzero(data.entity == event["entity"]).tolist()
    if not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= n for i in ids):
        raise ValueError("event track_indices/entity must identify at least one existing track")
    if len(ids) != len(set(ids)):
        raise ValueError("event track_indices cannot contain duplicates")
    return np.asarray(ids, dtype=int)


def evaluate(data, events, thresholds, recovery_threshold, hold_frames):
    error = np.linalg.norm(data.predicted - data.reference, axis=-1)
    # Evaluation-only displacement differences; no transform or GT enters the model.
    displacement_error = np.linalg.norm((data.predicted-data.predicted[:1])-(data.reference-data.reference[:1]),axis=-1)
    displacement_valid = data.valid & data.valid[:1]
    displacement_valid = displacement_valid.copy(); displacement_valid[0] = False
    displacement_present = data.pred_valid & data.pred_valid[:1]
    displacement_summaries, displacement_frames, displacement_events = [], [], []
    frame_rows, summaries = [], []
    all_groups = [("all", np.ones_like(data.valid))]
    all_groups += [(f"entity:{label}", np.broadcast_to(data.entity == label, data.valid.shape))
                   for label in np.unique(data.entity)]
    for group, group_mask in all_groups:
        for state, visibility_mask in [("all", np.ones_like(data.valid)),
                                       ("visible", data.visibility == 1),
                                       ("occluded", data.visibility == 0),
                                       ("unknown", data.visibility == -1)]:
            summaries.append({"group": group, "visibility": state,
                              **statistics(error, data.valid & group_mask & visibility_mask,
                                           data.pred_valid, thresholds)})
            displacement_summaries.append({"group":group,"visibility":state,
                **statistics(displacement_error,displacement_valid & group_mask & visibility_mask,displacement_present,thresholds)})
        for t, timestamp in enumerate(data.times):
            frame_rows.append({"frame_index": t, "time_s": float(timestamp), "group": group,
                               **statistics(error[t], data.valid[t] & group_mask[t],
                                            data.pred_valid[t], thresholds)})
            displacement_frames.append({"frame_index":t,"time_s":float(timestamp),"group":group,
                **statistics(displacement_error[t],displacement_valid[t]&group_mask[t],displacement_present[t],thresholds)})
    event_rows, recovery_rows = [], []
    seen = set()
    for event in events:
        if event["event_id"] in seen:
            raise ValueError("event_id must be unique")
        seen.add(event["event_id"])
        ids = validate_event(event, data)
        a, b = event["occlusion_start"], event["occlusion_end"]
        end = min(len(data.times), b + event["post_frames"])
        for phase, start, stop in [("before", max(0, a - event["pre_frames"]), a),
                                   ("during", a, b), ("after", b, end)]:
            sel = np.zeros_like(data.valid)
            sel[start:stop, ids] = True
            event_rows.append({"event_id": event["event_id"], "phase": phase,
                               "start_frame": start, "end_frame_exclusive": stop,
                               "track_count": len(ids), "source": event.get("source", "manual_annotation"),
                               **statistics(error, data.valid & sel, data.pred_valid, thresholds)})
            displacement_events.append({"event_id":event["event_id"],"phase":phase,"start_frame":start,
                "end_frame_exclusive":stop,"track_count":len(ids),"source":event.get("source","manual_annotation"),
                **statistics(displacement_error,displacement_valid&sel,displacement_present,thresholds)})
        for n in ids:
            result = {"event_id": event["event_id"], "track_index": int(n),
                      "entity": str(data.entity[n]), "threshold_m": recovery_threshold,
                      "hold_frames": hold_frames, "reappearance_frame": None,
                      "recovery_frame": None, "recovery_time_s": None,
                      "status": "no_confirmed_reappearance_in_window"}
            visible = np.flatnonzero(data.visibility[b:end, n] == 1) + b
            if visible.size:
                first = int(visible[0]); result["reappearance_frame"] = first
                good = ((data.visibility[:, n] == 1) & data.valid[:, n] &
                        data.pred_valid[:, n] & (error[:, n] <= recovery_threshold))
                result["status"] = "not_recovered_within_window"
                for k in range(first, end - hold_frames + 1):
                    if good[k:k + hold_frames].all():
                        result.update(status="recovered", recovery_frame=k,
                                      recovery_time_s=float(data.times[k] - data.times[first]))
                        break
            recovery_rows.append(result)
    # Identity labels must be stable dataset identities, not nearest-reference reassignment.
    identity = {"status": "unavailable", "reason": "identity labels not supplied"}
    if data.pred_identity is not None:
        known_ref = data.valid & (data.reference_identity != "")
        known_pred = (data.pred_identity != "") & data.pred_valid
        correct = known_ref & known_pred & (data.pred_identity == data.reference_identity)
        denom = int(known_ref.sum())
        eligible_transition = (known_ref[1:] & known_ref[:-1] & known_pred[1:] & known_pred[:-1] &
                               (data.reference_identity[1:] == data.reference_identity[:-1]))
        switches = eligible_transition & (data.pred_identity[1:] != data.pred_identity[:-1])
        identity = {"status": "available", "expected_samples": denom,
                    "correct_samples": int(correct.sum()),
                    "missing_predicted_identity": int((known_ref & ~known_pred).sum()),
                    "accuracy_missing_as_failure": float(correct.sum()/denom) if denom else None,
                    "switches_across_adjacent_known_frames": int(switches.sum()),
                    "eligible_adjacent_transitions": int(eligible_transition.sum()),
                    "definition": "label change while reference ID stays fixed; gaps are not bridged"}
    velocity = {"status": "unavailable", "reason": "sequence has fewer than two frames"}
    if len(data.times) >= 2:
        dt = np.diff(data.times)[:, None, None]
        verr = np.linalg.norm((np.diff(data.predicted, axis=0) - np.diff(data.reference, axis=0))/dt, axis=-1)
        mask = data.valid[1:] & data.valid[:-1]
        present = data.pred_valid[1:] & data.pred_valid[:-1]
        vals = verr[mask & present]
        velocity = {"status": "available", "expected_adjacent_pairs": int(mask.sum()),
                    "predicted_adjacent_pairs": int((mask & present).sum()),
                    "mean_velocity_error_m_per_s": float(vals.mean()) if vals.size else None,
                    "note": "adjacent finite differences with actual timestamps; no smoothing"}
    relative = {"status": "unavailable", "reason": "requires hand and object entity labels"}
    pair_summaries,pair_frames,pair_events=[],[],[]
    if {"hand", "object"}.issubset(set(data.entity)):
        rel_errors = []
        for t in range(len(data.times)):
            masks = [data.valid[t] & (data.entity == label) for label in ("hand", "object")]
            if not all(m.any() and data.pred_valid[t, m].all() for m in masks):
                continue
            cp = [data.predicted[t, m].mean(axis=0) for m in masks]
            cr = [data.reference[t, m].mean(axis=0) for m in masks]
            rel_errors.append(float(np.linalg.norm((cp[1]-cp[0])-(cr[1]-cr[0]))))
        relative = {"status": "available", "evaluable_frames": len(rel_errors),
                    "mean_centroid_relative_position_error_m": float(np.mean(rel_errors)) if rel_errors else None,
                    "note": "same reference-valid point sets, complete predictions required; not contact or 6DoF"}
        pairs=[(int(h),int(o)) for h in np.flatnonzero(data.entity=="hand") for o in np.flatnonzero(data.entity=="object")]
        labels=data.query_id if data.query_id is not None else np.asarray([f"track{i}" for i in range(data.predicted.shape[1])])
        pvec=np.stack([data.predicted[:,h]-data.predicted[:,o] for h,o in pairs],axis=1)
        rvec=np.stack([data.reference[:,h]-data.reference[:,o] for h,o in pairs],axis=1)
        perr=np.linalg.norm(pvec-rvec,axis=-1)
        pvalid=np.stack([data.valid[:,h]&data.valid[:,o] for h,o in pairs],axis=1)
        ppresent=np.stack([data.pred_valid[:,h]&data.pred_valid[:,o] for h,o in pairs],axis=1)
        pair_summaries.append({"pair_id":"all_fixed_pairs","hand_query_id":"all","object_query_id":"all",**statistics(perr,pvalid,ppresent,thresholds)})
        for pi,(h,o) in enumerate(pairs):
            metadata={"pair_id":f"{labels[h]}__minus__{labels[o]}","hand_query_id":str(labels[h]),"object_query_id":str(labels[o])}
            pair_summaries.append({**metadata,**statistics(perr[:,pi],pvalid[:,pi],ppresent[:,pi],thresholds)})
            for ti,timestamp in enumerate(data.times):
                row={"frame_index":ti,"time_s":float(timestamp),**metadata,"reference_valid":bool(pvalid[ti,pi]),"prediction_valid":bool(ppresent[ti,pi]),
                     "relative_vector_error_m":float(perr[ti,pi]) if pvalid[ti,pi] and ppresent[ti,pi] else None}
                for axis,component in enumerate("xyz"):
                    row[f"predicted_hand_minus_object_{component}_m"]=float(pvec[ti,pi,axis]) if ppresent[ti,pi] else None
                    row[f"reference_hand_minus_object_{component}_m"]=float(rvec[ti,pi,axis]) if pvalid[ti,pi] else None
                pair_frames.append(row)
        for event in events:
            ids=set(validate_event(event,data).tolist());a,b=event["occlusion_start"],event["occlusion_end"]
            pair_selection=np.asarray([h in ids and o in ids for h,o in pairs])
            for phase,lo,hi in [("before",max(0,a-event["pre_frames"]),a),("during",a,b),("after",b,min(len(data.times),b+event["post_frames"]))]:
                sel=np.zeros_like(pvalid);sel[lo:hi]=pair_selection
                pair_events.append({"event_id":event["event_id"],"phase":phase,"pair_id":"all_fixed_pairs",**statistics(perr,pvalid&sel,ppresent,thresholds)})
                for pi,(h,o) in enumerate(pairs):
                    if pair_selection[pi]:pair_events.append({"event_id":event["event_id"],"phase":phase,"pair_id":f"{labels[h]}__minus__{labels[o]}",**statistics(perr[:,pi],pvalid[:,pi]&sel[:,pi],ppresent[:,pi],thresholds)})
        relative["fixed_pairs_note"]="Each fixed hand/object pair uses ||(pred_hand-pred_object)-(ref_hand-ref_object)||. No GT alignment; missing either prediction fails. Not shortest-surface contact distance or penetration. Pairs share points and are correlated, not independent statistical samples."
    return {"summaries": summaries, "per_frame": frame_rows, "events": event_rows,
            "recovery": recovery_rows, "identity": identity, "velocity": velocity,
            "hand_object_relative": relative,"displacement_summaries":displacement_summaries,
            "relative_pair_summaries":pair_summaries,"relative_pair_per_frame":pair_frames,"relative_pair_events":pair_events,
            "displacement_per_frame":displacement_frames,"displacement_events":displacement_events,
            "displacement_definition":"||(prediction_t-prediction_0)-(reference_t-reference_0)||; first frame omitted because difference is trivially zero; missing anchor/current predictions count failure; evaluation difference only, no model alignment"}, error


def write_csv(path, rows):
    if not rows:
        path.write_text("status,reason\nunavailable,no eligible records\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def plot_results(data, results, error, output, reference_role="unknown"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for group in ["all"] + [f"entity:{x}" for x in np.unique(data.entity)]:
        rows = [x for x in results["per_frame"] if x["group"] == group]
        axes[0].plot(data.times, [np.nan if x["mean_epe_m"] is None else x["mean_epe_m"]*1000 for x in rows], label=group)
        axes[1].plot(data.times, [np.nan if x["coverage"] is None else x["coverage"] for x in rows], label=group)
    marker = "SYNTHETIC SELF-CHECK — " if reference_role == "synthetic_fixture" else ""
    axes[0].set(ylabel="3D endpoint error (mm)", title=marker + "Fixed-coordinate evaluation; no alignment")
    axes[0].legend(); axes[1].set(ylabel="Prediction coverage", ylim=(-0.02, 1.02))
    denom = np.maximum(data.valid.sum(axis=1), 1)
    for state, label in [(1, "visible"), (0, "occluded"), (-1, "unknown")]:
        axes[2].plot(data.times, ((data.visibility == state) & data.valid).sum(axis=1)/denom, label=label)
    axes[2].set(ylabel="Reference visibility fraction", xlabel="Time (seconds)", ylim=(-0.02, 1.02)); axes[2].legend()
    for ax in axes: ax.grid(alpha=0.25)
    fig.savefig(output/"trajectory_errors.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, max(2.5, min(8, 0.16*error.shape[1]+1.5))), constrained_layout=True)
    shown = np.where(data.valid & data.pred_valid, error*1000, np.nan).T
    mesh = ax.imshow(shown, aspect="auto", interpolation="nearest", origin="lower", cmap="magma")
    ax.set(xlabel="Frame index (timestamps in per_frame.csv)", ylabel="Persistent track index",
           title=marker + "3D error (mm); missing data blank, counted separately")
    fig.colorbar(mesh, ax=ax, label="mm"); fig.savefig(output/"track_error_heatmap.png", dpi=150); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, help="JSON provenance declaration; unknown provenance is flagged")
    parser.add_argument("--events", type=Path, help="JSON list, explicit half-open frame intervals")
    parser.add_argument("--units", choices=sorted(UNIT_SCALE))
    parser.add_argument("--thresholds-m", nargs="+", type=float, default=[0.01, 0.02, 0.05])
    parser.add_argument("--recovery-threshold-m", type=float, default=0.02)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument("--pre-frames", type=int, default=10)
    parser.add_argument("--post-frames", type=int, default=30)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.hold_frames < 1 or min(args.pre_frames, args.post_frames) < 0:
        parser.error("hold-frames must be positive; pre/post frames nonnegative")
    if any(not np.isfinite(x) or x <= 0 for x in args.thresholds_m + [args.recovery_threshold_m]):
        parser.error("distance thresholds must be positive finite meters")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("output is not empty; select a new directory or explicitly --overwrite")
    start = time.perf_counter()
    protocol = json.loads(args.protocol.read_text()) if args.protocol else {}
    validate_protocol(protocol)
    data = load_bundle(args.input, args.units)
    events = json.loads(args.events.read_text()) if args.events else automatic_events(data, args.pre_frames, args.post_frames)
    if not isinstance(events, list):
        raise ValueError("events JSON must contain a list")
    results, error = evaluate(data, events, sorted(set(args.thresholds_m)), args.recovery_threshold_m, args.hold_frames)
    args.output.mkdir(parents=True, exist_ok=True)
    warnings = []
    if protocol.get("reference_role", "unknown") in ("unknown", "tracker_proxy"):
        warnings.append("Reference provenance is unknown or a tracker proxy: not independent 3D accuracy evidence.")
    if protocol.get("reference_used_for_training") is not False:
        warnings.append("Training/reference separation is not explicitly declared; this report cannot certify no leakage.")
    if not np.any(data.visibility != -1):
        warnings.append("No visibility labels: occlusion buckets and automatic recovery events unavailable.")
    if protocol.get("visibility_source") is None and np.any(data.visibility != -1):
        warnings.append("Visibility source not declared; distinguish reference annotation from model-derived proxies.")
    if protocol.get("reference_role") == "registered_reference":
        warnings.append("Reference is registration-derived and noisy, not direct contact/pressure or perfect geometry truth.")
    for name in ("summaries", "per_frame", "events", "recovery", "displacement_summaries", "displacement_per_frame", "displacement_events", "relative_pair_summaries", "relative_pair_per_frame", "relative_pair_events"):
        write_csv(args.output/f"{name}.csv", results[name])
    if not args.no_plots:
        plot_results(data, results, error, args.output, protocol.get("reference_role", "unknown"))
    report = {"schema_version": 1, "input": str(args.input.resolve()),
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "python": platform.python_version(), "numpy": np.__version__,
              "shape_T_N_3": list(data.predicted.shape), "source_units": data.units,
              "reported_distance_units": "m", "coordinate_frame": data.frame,
              "alignment_performed": "none", "protocol": protocol,
              "events_definition": events, "warnings": warnings,
              "metric_definitions": {
                  "epe": "Euclidean distance in supplied common coordinates, conditional mean plus explicit coverage",
                  "threshold_success": "correct predictions divided by reference-valid samples, missing predictions fail",
                  "displacement_error":results["displacement_definition"],
                  "recovery": "first confirmed visible post-event sample to first run of hold_frames below threshold; finite window",
                  "events": "[occlusion_start,occlusion_end), per-track windows may overlap; not independent samples",
                  "unavailable": ["contact geometry", "penetration", "rigid object rotation/6DoF", "unseen-surface accuracy"]},
              "results": results, "evaluation_wall_seconds": time.perf_counter()-start}
    (args.output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"report": str((args.output/"report.json").resolve()), "warnings": warnings,
                      "overall": results["summaries"][0]}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
