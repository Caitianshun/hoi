#!/usr/bin/env python3
"""Evaluation-only fitted-surface tracks from fixed RGB queries and mesh topology.

No learned model or SMPL weights required. First-frame rays select one triangle;
its vertex IDs and barycentric weights remain fixed throughout the sequence.
Output is registration-derived reference, not exact joints/contact ground truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_mesh(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as d:
            vertices, faces = np.asarray(d["vertices"], float), np.asarray(d["faces"], int)
    elif path.suffix.lower() == ".ply":
        from plyfile import PlyData
        d = PlyData.read(path)
        vertices = np.stack([d["vertex"][key] for key in ("x", "y", "z")], axis=-1).astype(float)
        name = "vertex_indices" if "vertex_indices" in d["face"].data.dtype.names else "vertex_index"
        if any(len(x) != 3 for x in d["face"][name]): raise ValueError("only triangular PLY meshes supported")
        faces = np.stack(d["face"][name]).astype(int)
    else: raise ValueError("mesh must be triangular PLY or NPZ(vertices,faces)")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all(): raise ValueError("invalid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("invalid triangle indices")
    return vertices, faces


def intersect_mesh(origin, direction, vertices, faces):
    """Closest positive two-sided Moller-Trumbore hit; direction must be unit norm."""
    tri = vertices[faces]; e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    h = np.cross(np.broadcast_to(direction, e2.shape), e2); a = (e1*h).sum(-1)
    inv = np.divide(1., a, out=np.zeros_like(a), where=np.abs(a) > 1e-12)
    s = origin-tri[:, 0]; u = inv*(s*h).sum(-1); q = np.cross(s, e1)
    v = inv*(direction*q).sum(-1); distance = inv*(e2*q).sum(-1)
    valid = (np.abs(a)>1e-12) & (u>=-1e-8) & (v>=-1e-8) & (u+v<=1+1e-8) & (distance>1e-8)
    if not valid.any(): return None
    index = int(np.argmin(np.where(valid, distance, np.inf)))
    return index, np.array([1-u[index]-v[index], u[index], v[index]]), float(distance[index])


def read_behave_camera(intrinsics_path, extrinsics_path, undistorted=False):
    color = json.loads(Path(intrinsics_path).read_text())["color"]
    ext = json.loads(Path(extrinsics_path).read_text())
    k = np.array([[color["fx"], 0, color["cx"]], [0, color["fy"], color["cy"]], [0, 0, 1.]])
    distortion = np.zeros(8) if undistorted else np.asarray(color["opencv"][4:], float)
    r = np.asarray(ext["rotation"], float).reshape(3, 3); trans = np.asarray(ext["translation"], float)
    if not np.allclose(r.T@r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
        raise ValueError("camera config rotation is not orthonormal")
    return k, distortion, r, trans, (color["width"], color["height"])


def camera_rays(uv, k, distortion, r_world_from_camera):
    import cv2
    uv = np.asarray(uv, float)
    normalized = cv2.undistortPointsIter(uv[:, None, :], k, distortion, None, None,
                                        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))[:, 0]
    camera = np.column_stack([normalized, np.ones(len(uv))])
    reprojected = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k, distortion)[0][:, 0]
    if not np.allclose(reprojected, uv, atol=0.05): raise ValueError("distortion inversion failed query reprojection tolerance")
    world = camera @ r_world_from_camera.T
    return world / np.linalg.norm(world, axis=-1, keepdims=True)


def project_world(points, k, distortion, r_world_from_camera, camera_origin_world):
    import cv2
    camera = (np.asarray(points)-camera_origin_world) @ r_world_from_camera
    projected = np.full((len(camera), 2), np.nan)
    valid = np.isfinite(camera).all(axis=-1) & (camera[:, 2] > 0)
    if valid.any(): projected[valid] = cv2.projectPoints(camera[valid], np.zeros(3), np.zeros(3), k, distortion)[0][:, 0]
    return projected, camera[:, 2]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="JSON frame_times/mesh paths in one evaluation coordinate frame")
    p.add_argument("--queries", type=Path, required=True, help="NPZ query_uv/entity/query_id/image_size/independently_visible_at_query")
    p.add_argument("--query-protocol", type=Path, required=True, help="JSON fixed_queries_before_predictions=true; source=manual_rgb")
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--camera-config", type=Path, required=True, help="BEHAVE config local-to-world, world=k1 color")
    p.add_argument("--processed-input-manifest", type=Path, help="known rectified RGB K/size/c2w from preprocessing; no fitted alignment")
    p.add_argument("--association-frame-index", type=int, default=0)
    p.add_argument("--undistorted", action="store_true", help="only for cv2.undistort RGB using original K, no resize/crop")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()): raise ValueError("output must be new/empty")
    start = time.perf_counter(); manifest = json.loads(args.manifest.read_text())
    protocol = json.loads(args.query_protocol.read_text())
    if protocol.get("fixed_queries_before_predictions") is not True or protocol.get("source") != "manual_rgb":
        raise ValueError("queries must be fixed from RGB before inspecting predictions")
    if protocol.get("reference_used_for_training") is not False: raise ValueError("reference must be isolated from training")
    if manifest.get("units") != "m" or manifest.get("coordinate_frame") != "behave_world_k1_color":
        raise ValueError("this adapter requires BEHAVE world=k1 color, meters; no alignment fitting")
    frames = manifest["frames"]; times = np.asarray([x["time"] for x in frames], float)
    if not len(frames) or not np.isfinite(times).all() or np.any(np.diff(times)<=0): raise ValueError("strict increasing frame times required")
    anchor = args.association_frame_index
    if not 0 <= anchor < len(frames): raise ValueError("association-frame-index out of range")
    with np.load(args.queries, allow_pickle=False) as q:
        uv = np.asarray(q["query_uv"], float); entity = np.asarray(q["entity"], str); ids = np.asarray(q["query_id"], str)
        size = tuple(q["image_size"].tolist()); independent_visible = np.asarray(q["independently_visible_at_query"])
    n = len(uv)
    if uv.shape != (n,2) or not n or not np.isfinite(uv).all(): raise ValueError("query_uv must be nonempty N,2")
    if entity.shape != (n,) or ids.shape != (n,) or len(set(ids.tolist())) != n: raise ValueError("entity and unique query IDs required")
    if independent_visible.shape != (n,) or not np.isin(independent_visible, [0,1]).all(): raise ValueError("visible query declarations required")
    k, distortion, rot, origin, camera_size = read_behave_camera(args.intrinsics, args.camera_config, args.undistorted)
    if args.processed_input_manifest:
        processed=json.loads(args.processed_input_manifest.read_text())
        c2w=np.asarray(processed["c2w"],float)
        if processed.get("images_undistorted") is not True or np.any(np.asarray(processed["distortion"])!=0):
            raise ValueError("processed manifest must describe a rectified pinhole image")
        if not np.allclose(c2w[:3,:3],rot,atol=1e-8) or not np.allclose(c2w[:3,3],origin,atol=1e-8):
            raise ValueError("processed camera frame differs from known BEHAVE config")
        k=np.asarray(processed["K"],float);distortion=np.zeros(8)
        if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0:
            raise ValueError("invalid processed K")
        camera_size=(processed["width"],processed["height"])
    if size != camera_size: raise ValueError("query size must match declared calibration or processed input manifest")
    if (uv < 0).any() or (uv[:,0] >= size[0]).any() or (uv[:,1] >= size[1]).any(): raise ValueError("query outside image")
    def resolve(path):
        path = Path(path); return path if path.is_absolute() else args.manifest.parent / path
    anchor_vertices, faces = load_mesh(resolve(frames[anchor]["path"]))
    occluders = [load_mesh(resolve(path)) for path in frames[anchor].get("occluder_meshes", [])]
    directions = camera_rays(uv, k, distortion, rot)
    triangle = np.full(n, -1, int); weights = np.full((n,3), np.nan); status = []
    for i, ray in enumerate(directions):
        if not independent_visible[i]: status.append("not_independently_visible_at_query"); continue
        hit = intersect_mesh(origin, ray, anchor_vertices, faces)
        if hit is None: status.append("no_fitted_surface_hit"); continue
        obstructed = False
        for ov, of in occluders:
            ohit = intersect_mesh(origin, ray, ov, of)
            if ohit is not None and ohit[2] < hit[2]-0.003: obstructed = True; break
        if obstructed: status.append("known_mesh_occludes_query"); continue
        triangle[i], weights[i], _ = hit; status.append("associated_requires_RGB_geometry_overlay_QA")
    accepted = triangle >= 0; vertex_ids = np.full((n,3), -1, int); vertex_ids[accepted] = faces[triangle[accepted]]
    reference = np.full((len(frames), n, 3), np.nan); projected = np.full((len(frames), n, 2), np.nan)
    depth = np.full((len(frames), n), np.nan); sources = []
    for ti, frame in enumerate(frames):
        path = resolve(frame["path"]); vertices, current_faces = load_mesh(path)
        if vertices.shape != anchor_vertices.shape or not np.array_equal(current_faces, faces):
            raise ValueError(f"mesh topology/order changed at frame {ti}; cannot preserve material-point identity")
        reference[ti, accepted] = np.einsum("nij,ni->nj", vertices[vertex_ids[accepted]], weights[accepted])
        projected[ti], depth[ti] = project_world(reference[ti], k, distortion, rot, origin)
        sources.append({"time": float(times[ti]), "path": str(path.resolve()), "sha256": digest(path)})
    valid = np.broadcast_to(accepted, (len(frames), n)).copy()
    # Visibility remains unknown: mesh self-visibility alone omits external occluders,
    # fitting uncertainty, and RGB evidence. Query declarations are not all-frame labels.
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "reference.npz", reference=reference, valid_mask=valid,
                        frame_times=times, entity=entity, query_id=ids, query_uv=uv,
                        visibility=np.full(valid.shape,-1,np.int8), units=np.array("m"),
                        coordinate_frame=np.array(manifest["coordinate_frame"]),
                        triangle_index=triangle, vertex_ids=vertex_ids, barycentric_weights=weights,
                        projected_uv=projected, camera_depth_m=depth, association_status=np.asarray(status))
    report = {"schema_version":1, "reference_role":"registered_reference", "reference_used_for_training":False,
              "association":"single_anchor_ray_fixed_triangle_barycentric_no_prediction_input_no_rematching",
              "coordinate_frame":manifest["coordinate_frame"], "units":"m", "anchor_index":anchor,
              "anchor_time":float(times[anchor]), "query_status":dict(zip(ids.tolist(), status)),
              "accepted_queries":int(accepted.sum()), "requested_queries":n, "query_protocol":protocol,
              "source_files":sources, "input_sha256":{str(x.resolve()):digest(x) for x in
                  (args.manifest,args.queries,args.query_protocol,args.intrinsics,args.camera_config)},
              "adapter_sha256":digest(__file__), "elapsed_seconds":time.perf_counter()-start,
              "limitations":["Fitted SMPL-H material surface points, not measured joints/contact",
                  "RGB hand labels are manual; ray mesh intersection alone does not prove hand semantics",
                  "Requires anchor RGB/mesh overlay QA; failed associations remain listed",
                  "Visibility unknown until independently annotated; no recovery metric yet",
                  "Reference temporal sampling limits recovery timing; do not invent interpolated ground truth"],
              "undistorted":args.undistorted, "fitted_reference_exposed_to_prediction":False}
    if args.processed_input_manifest:
        report["processed_input_manifest"]={"path":str(args.processed_input_manifest.resolve()),
                                           "sha256":digest(args.processed_input_manifest),"K":k.tolist()}
    (args.output / "reference_protocol.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"reference":str((args.output/"reference.npz").resolve()), "accepted":int(accepted.sum()),"requested":n}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
