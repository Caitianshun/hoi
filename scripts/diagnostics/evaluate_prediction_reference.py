#!/usr/bin/env python3
"""Join prediction-only trajectories with isolated fitted reference, then score.

No alignment estimation, nearest-point matching, temporal interpolation, or
prediction-visibility filtering. Existing valid masks are retained as supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_trajectories import main as evaluate_3d, write_csv
from evaluate_cotracker_reference import phases_from_rgb, summary_2d
from mesh_query_reference import project_world


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction",type=Path,required=True);p.add_argument("--prediction-protocol",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True);p.add_argument("--reference-protocol",type=Path,required=True)
    p.add_argument("--input-manifest",type=Path,required=True);p.add_argument("--method",required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists() and any(a.output.iterdir()):raise ValueError("output must be new/empty")
    with np.load(a.prediction,allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
    with np.load(a.reference,allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
    for key in ["coordinate_frame","units","entity","query_id"]:
        if not np.array_equal(pred[key],ref[key]):raise ValueError(f"prediction/reference {key} mismatch; no implicit coordinate/query changes")
    if pred["units"].item()!="m":raise ValueError("this adapter requires already converted meter inputs")
    if not np.all(np.diff(pred["frame_times"])>0):raise ValueError("prediction times not strictly increasing")
    indices=[]
    for t in ref["frame_times"]:
        matching=np.flatnonzero(np.isclose(pred["frame_times"],t,atol=1e-7,rtol=0))
        if len(matching)!=1:raise ValueError(f"no unique exact-time prediction at {t}; interpolation forbidden")
        indices.append(int(matching[0]))
    xyz=pred["predicted"][indices];present=np.isfinite(xyz).all(-1)
    if "predicted_valid_mask" in pred:present &= pred["predicted_valid_mask"][indices]
    if xyz.shape!=ref["reference"].shape:raise ValueError("prediction/reference shape mismatch")
    pp=json.loads(a.prediction_protocol.read_text());rp=json.loads(a.reference_protocol.read_text());m=json.loads(a.input_manifest.read_text())
    if pp.get("status")!="completed":raise ValueError("prediction export/protocol must be completed")
    if pp.get("global_transform_fitted_on_evaluation") is True:raise ValueError("evaluation reference used to fit prediction transform")
    protocol={**rp,"prediction_method":a.method,"prediction_protocol":pp,"prediction_protocol_sha256":sha(a.prediction_protocol),
              "prediction_file_sha256":sha(a.prediction),"reference_file_sha256":sha(a.reference),"join_script_sha256":sha(__file__),
              "reference_used_for_training":False,"alignment":"none","evaluation_alignment_performed":"none","visibility_source":None}
    if "matrix_source_to_world" in pp:
        if pp.get("global_transform_fitted_on_evaluation") is not False:raise ValueError("global transform provenance must explicitly exclude reference fitting")
        protocol.update(alignment="predeclared_global_transform",global_transform_fitted_on_evaluation=False,
                        global_transform_provenance="Frozen input-only transform defined before reference evaluation; see complete prediction_protocol. Evaluator fits nothing.")
    times=ref["frame_times"];phase=phases_from_rgb(times,m["event_selection"])
    before=np.flatnonzero(phase=="before");during=np.flatnonzero(np.isin(phase,["partial_occlusion","severe_occlusion"]));after=np.flatnonzero(phase=="reappearance")
    events=[]
    if len(during):events=[{"event_id":"manual_RGB_first_occlusion_sparse_observations","occlusion_start":int(during[0]),"occlusion_end":int(during[-1]+1),"pre_frames":int(during[0]-before[0]) if len(before) else 0,"post_frames":int(after[-1]-during[-1]) if len(after) else 0,"track_indices":list(range(xyz.shape[1])),"source":"pre-existing manual RGB brackets; reference point visibility unknown"}]
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/"evaluation_bundle.npz",predicted=xyz,reference=ref["reference"],valid_mask=ref["valid_mask"],predicted_valid_mask=present,frame_times=times,entity=ref["entity"],query_id=ref["query_id"],visibility=np.full(present.shape,-1,np.int8),coordinate_frame=ref["coordinate_frame"],units=ref["units"])
    (a.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n");(a.output/"events.json").write_text(json.dumps(events,indent=2)+"\n")
    evaluate_3d(["--input",str(a.output/"evaluation_bundle.npz"),"--output",str(a.output/"three_dimensional"),"--protocol",str(a.output/"protocol.json"),"--events",str(a.output/"events.json")])
    k=np.asarray(m["K"]);c=np.asarray(m["c2w"]);uv=[];z=[]
    for points in xyz:
        pixels,depth=project_world(points,k,np.zeros(8),c[:3,:3],c[:3,3]);uv.append(pixels);z.append(depth)
    uv=np.asarray(uv);z=np.asarray(z);error=np.linalg.norm(uv-ref["projected_uv"],axis=-1)
    rows=[]
    for entity in ["all",*np.unique(ref["entity"]).tolist()]:
        emask=np.ones_like(present) if entity=="all" else np.broadcast_to(ref["entity"]==entity,present.shape)
        for label in ["all",*np.unique(phase).tolist()]:
            pmask=np.ones_like(present) if label=="all" else np.broadcast_to((phase==label)[:,None],present.shape)
            rows.append({"method":a.method,"entity":entity,"manual_rgb_phase":label,**summary_2d(error,ref["valid_mask"]&emask&pmask,present&(z>0))})
    write_csv(a.output/"projection_error_summary.csv",rows)
    np.savez_compressed(a.output/"projection_diagnostic.npz",predicted_uv=uv,reference_uv=ref["projected_uv"],error_pixels=error,predicted_camera_z=z,frame_times=times,query_id=ref["query_id"],phase=phase)
    (a.output/"join_report.json").write_text(json.dumps({"method":a.method,"status":"completed","prediction_input_indices":indices,"projection_summary":rows,"prediction_visibility_filtered":False,"coordinate_alignment_performed":False,"reference_visibility":"unknown","comparison_limitations":["Same six fixed queries/14 observations only; fitted-reference and source-image timing uncertainty","Method-specific coordinate calibration, priors, resolutions, and trajectory definitions remain different and must be disclosed","Conditional error means require coverage; reports include first-frame displacement differences without reference alignment"]},indent=2)+"\n")
    print(json.dumps({"method":a.method,"projection_overall":rows[0],"output":str(a.output.resolve())}))


if __name__=="__main__":main()
