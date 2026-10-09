"""Build protocol-B query pose caches: published SMPL for training and retained frames.

The cache is query-only (rendering/evaluation of retained frames under the HOSNeRF
official test condition). It reuses V10's exact body-to-world chain and asserts that
every training-frame entry equals the frozen V10 train-only cache.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from v11_common import ROOT, config, identity, read, save_json, scene_dir, v10_prepare_inputs
from hoi_modules.pose_prior_adapter import (PosePriorAdapter, decompose_similarity, local_pose,
                                            root_transform)


def build(scene):
    prep = v10_prepare_inputs()
    cfg = config()["scenes"][scene]
    inputs = ROOT / cfg["input_dir"]
    train, evaluation = read(inputs / "manifest.json"), read(inputs / "evaluation_manifest.json")
    frames = sorted(train["frames"] + evaluation["frames"], key=lambda f: int(f["frame_id"]))
    ids = [f["frame_id"] for f in frames]
    assert len(ids) == len(set(ids)) == cfg["training_frames"] + cfg["retained_frames"]
    v10_cache_path = ROOT / cfg["v10_scene"] / "protocol/pose_cache.pt"
    v10 = torch.load(v10_cache_path, map_location="cpu", weights_only=False)
    prior_camera = Path(train["camera_source"]["path"])
    assert identity(prior_camera)["sha256"] == train["camera_source"]["sha256"]
    with prior_camera.open("rb") as stream:
        cameras = pickle.load(stream, encoding="latin1")
    mesh_path = prep.metadata_source(scene, prior_camera.parent, "mesh_infos.pkl")
    with mesh_path.open("rb") as stream:
        mesh = pickle.load(stream, encoding="latin1")
    local_q, local_t, parts = [], [], []
    for f in frames:
        record, camera = mesh[f["frame_id"]], cameras[f["frame_id"]]
        for key, matrix in (("K", camera["intrinsics"]), ("w2c", camera["scaleworld_to_camera"])):
            assert np.array_equal(np.asarray(f[key]), np.asarray(matrix)), (scene, f["frame_id"], key)
        rotations, translations = local_pose(record["poses"], record["tpose_joints"])
        body_to_world = np.asarray(camera["smpl_to_scale_world"], dtype=np.float64) @ root_transform(record["Rh"], record["Th"])
        parts.append(decompose_similarity(body_to_world))
        local_q.append(Rotation.from_matrix(rotations).as_quat())
        local_t.append(translations)
    cache = {key: v10[key] for key in ("canonical_joints", "H", "N", "canonical_inverse", "official_world_to_parent_world",
                                       "time_origin", "time_denominator", "native_frame_count", "camera_identity",
                                       "joint_names", "parents", "body_world_chain")}
    cache.update({"schema": "V11.protocolB_query_pose_cache.v1", "scene": scene,
                  "train_frame_ids": ids, "heldout_frame_ids": [],
                  "query_only": True, "training_use": "forbidden",
                  "retained_frame_ids": [f["frame_id"] for f in evaluation["frames"]],
                  "local_quaternions_xyzw": np.stack(local_q), "local_translations": np.stack(local_t),
                  "root_quaternions_xyzw": np.stack([p["quaternion_xyzw"] for p in parts]),
                  "root_translations": np.stack([p["translation"] for p in parts]),
                  "root_scales": np.asarray([p["scale"] for p in parts]),
                  "source": {"mesh_infos": identity(mesh_path), "cameras": identity(prior_camera)},
                  "query_rule": "exact published record at every training and retained frame; no interpolation"})
    # Training rows must equal the frozen V10 train-only cache.
    index = {fid: i for i, fid in enumerate(ids)}
    rows = [index[fid] for fid in v10["train_frame_ids"]]
    checks = {}
    for key in ("local_quaternions_xyzw", "local_translations", "root_quaternions_xyzw", "root_translations", "root_scales"):
        diff = float(np.abs(np.asarray(cache[key])[rows] - np.asarray(v10[key])).max())
        checks[key] = diff
        assert diff == 0.0, (scene, key, diff)
    query, train_only = PosePriorAdapter(cache), PosePriorAdapter(v10)
    bone_diff = max(float(np.abs(query.query(fid)["bone_transforms"] - train_only.query(fid)["bone_transforms"]).max())
                    for fid in v10["train_frame_ids"])
    assert bone_diff == 0.0, bone_diff
    retained = [f["frame_id"] for f in evaluation["frames"]]
    shift = {fid: float(np.abs(query.query(fid)["world_joints"] - train_only.query(fid)["world_joints"]).max()) for fid in retained}
    out = scene_dir(scene) / "protocol/pose_query_cache_B.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    receipt = dict(status="completed", scene=scene, cache=identity(out), v10_train_cache=identity(v10_cache_path),
                   training_rows_equal_v10=checks, training_bone_transform_max_diff=bone_diff,
                   retained_world_joint_max_shift_vs_protocol_A=shift, frames=len(ids))
    save_json(scene_dir(scene) / "protocol/pose_query_cache_B.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default="Backpack,Tennis")
    for s in parser.parse_args().scenes.split(","):
        r = build(s)
        print(s, r["frames"], "train-equal", r["training_bone_transform_max_diff"],
              "max retained shift", round(max(r["retained_world_joint_max_shift_vs_protocol_A"].values()), 4))
