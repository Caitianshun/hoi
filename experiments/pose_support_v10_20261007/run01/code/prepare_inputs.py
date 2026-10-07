#!/usr/bin/env python3
"""Freeze V10 train-only pose, seed/color and RGB sequence inputs on CPU.

Historical camera manifests are authoritative.  Native HOSNeRF stage-1 cameras
are recorded as a separate version and are never substituted into V10.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hoi_modules.pose_prior_adapter import (
    PosePriorAdapter, SMPL_JOINT_IDX, SMPL_PARENTS, canonical_normalization,
    decompose_similarity, forward_kinematics, local_pose, root_transform,
)

SCENES = {
    "Backpack": {
        "inputs": "experiments/baseline_protocol_calibration_20260927/run01/inputs/hos_backpack",
        "parent": "experiments/local_dynamic_v9_20260929/run01/scenes/Backpack/runs/Q0/checkpoint_fine_030000.pt",
        "parent_kind": "V9 Q0 after-Adam 30000", "denominator": 282,
    },
    "Tennis": {
        "inputs": "experiments/temporal_evidence_v8_20260928/run01/tennis/inputs/hos_tennis",
        "parent": "experiments/temporal_evidence_v8_20260928/run01/tennis/runs/B_Q/checkpoint_fine_014000.pt",
        "parent_kind": "V8 Q weight warm-start; nominal 14000, actual 13999 Adam; resumable=False",
        "denominator": 299,
    },
}
OFFICIAL = ROOT / "third_party/HOSNeRF/3rd_Complete_HOSNeRF"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def save_tensor(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def official_helpers():
    body_file = OFFICIAL / "core/utils/body_util.py"
    specification = importlib.util.spec_from_file_location("v10_official_body", body_file)
    body = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(body)
    network_file = OFFICIAL / "core/utils/network_util.py"
    # Compile only the unchanged official class, avoiding unrelated network/config imports.
    tree = ast.parse(network_file.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
               node.name == "MotionBasisComputer")
    namespace = {"torch": torch, "nn": torch.nn, "SMPL_PARENT": body.SMPL_PARENT}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(network_file), "exec"), namespace)
    motion = namespace["MotionBasisComputer"](total_bones=24)
    return body, motion


def metadata_source(scene: str, old_folder: Path, name: str) -> Path:
    legacy = old_folder / name
    if legacy.exists():
        return legacy
    current = ROOT / "experiments/hosnerf_benchmark_20261006/run01/data" / scene / name
    if current.exists():
        return current
    raise FileNotFoundError(f"{scene}: {name} absent from historical and prepared public data")


def training_schedule(frame_ids: list[str], updates: int, seed: int) -> dict:
    rng = random.Random(seed)
    stream = []
    epochs = []
    while len(stream) < updates * 2:
        order = list(range(len(frame_ids)))
        rng.shuffle(order)
        epochs.append({"start_sample": len(stream), "order": order})
        stream.extend(order)
    stream = stream[:updates * 2]
    batches = [stream[i:i + 2] for i in range(0, len(stream), 2)]
    return {"schema": "V10.fixed_RGB_sequence.v1", "seed": seed,
            "updates": updates, "batch_size": 2, "camera_count": len(frame_ids),
            "rule": "independent Python Random; shuffled epochs without replacement, consumed from front",
            "frame_ids": frame_ids, "batches": batches, "batches_indices": batches,
            "batches_ids": [[frame_ids[i] for i in batch] for batch in batches],
            "epochs": epochs,
            "remaining_stack_rule": "after k updates, offset=2*k; find containing epoch by start_sample; remaining is order[offset-start_sample:]",
            "independent_of_initialization_and_density": True}


def projected(joints: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray]:
    camera = joints @ np.asarray(frame["w2c"])[:3, :3].T + np.asarray(frame["w2c"])[:3, 3]
    pixels = camera @ np.asarray(frame["K"]).T
    return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]


def prepare_scene(scene: str, run: Path, updates: int, seed: int) -> dict:
    from hoi_modules.pose_support_gaussians import build_joint_only_seed_cache, initialize_seed_colors
    definition = SCENES[scene]
    inputs = ROOT / definition["inputs"]
    train_path = inputs / "manifest.json"
    evaluation_path = inputs / "evaluation_manifest.json"
    train = json.loads(train_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    frames = train["frames"]
    train_ids = [frame["frame_id"] for frame in frames]
    heldout_ids = [frame["frame_id"] for frame in evaluation["frames"]]
    if set(train_ids) & set(heldout_ids):
        raise ValueError("train/development camera split overlaps")
    if train["time_normalization"]["denominator"] != definition["denominator"]:
        raise ValueError("parent model time mapping changed")
    prior_camera = Path(train["camera_source"]["path"])
    camera_identity = identity(prior_camera)
    if camera_identity["sha256"] != train["camera_source"]["sha256"]:
        raise ValueError("historical camera file hash differs from parent manifest")
    old_folder = prior_camera.parent
    selected_paths = {name: metadata_source(scene, old_folder, name) for name in
                      ("mesh_infos.pkl", "canonical_joints.pkl", "cameras.pkl")}
    # Public files are numpy-only published dictionaries; development records are
    # not indexed or copied.  Container deserialization itself is disclosed below.
    with prior_camera.open("rb") as stream:
        cameras = pickle.load(stream, encoding="latin1")
    with selected_paths["mesh_infos.pkl"].open("rb") as stream:
        mesh = pickle.load(stream, encoding="latin1")
    with selected_paths["canonical_joints.pkl"].open("rb") as stream:
        canonical = np.asarray(pickle.load(stream, encoding="latin1")["joints"], dtype=np.float64)
    with selected_paths["cameras.pkl"].open("rb") as stream:
        world_cameras = pickle.load(stream, encoding="latin1")
    body, motion = official_helpers()
    height, normalizer = canonical_normalization(canonical)
    canonical_global = body.get_canonical_global_tfms(canonical)
    canonical_inverse = np.linalg.inv(canonical_global.astype(np.float64))
    local_quaternions, local_translations, root_parts = [], [], []
    training_records = []
    coordinate_maps = []
    audits = []
    max_local_rotation_difference = 0.
    max_fk_difference = 0.
    max_forward_difference = 0.
    for frame in frames:
        frame_id = frame["frame_id"]
        record = mesh[frame_id]  # Only training records are addressed.
        camera = cameras[frame_id]
        for key, matrix in (("K", camera["intrinsics"]), ("w2c", camera["scaleworld_to_camera"])):
            if not np.array_equal(np.asarray(frame[key]), np.asarray(matrix)):
                raise ValueError(f"{scene}/{frame_id}: historical manifest {key} changed")
        rotations, translations = local_pose(record["poses"], record["tpose_joints"])
        root = root_transform(record["Rh"], record["Th"])
        smpl_to_world = np.asarray(world_cameras[frame_id]["smpl_to_world"], dtype=np.float64)
        smpl_to_old = np.asarray(camera["smpl_to_scale_world"], dtype=np.float64)
        coordinate_maps.append(smpl_to_old @ np.linalg.inv(smpl_to_world))
        # Exactly the official complete-stage loader's root convention.
        body_to_old_world = smpl_to_old @ root
        part = decompose_similarity(body_to_old_world)
        root_parts.append(part)
        local_quaternions.append(Rotation.from_matrix(rotations).as_quat())
        local_translations.append(translations)
        original_r, original_t = body.body_pose_to_body_RTs(record["poses"], record["tpose_joints"])
        fk = forward_kinematics(rotations, translations)
        original_fk = forward_kinematics(original_r.astype(np.float64), original_t.astype(np.float64))
        _, _, forward_r, forward_t = motion(torch.from_numpy(original_r[None]),
                                            torch.from_numpy(original_t[None]),
                                            torch.from_numpy(canonical_global[None]))
        local_forward = fk @ canonical_inverse
        max_local_rotation_difference = max(max_local_rotation_difference, float(abs(rotations - original_r).max()))
        max_fk_difference = max(max_fk_difference, float(abs(fk - original_fk).max()))
        max_forward_difference = max(max_forward_difference,
                                     float(abs(local_forward[:, :3, :3] - forward_r[0].numpy()).max()),
                                     float(abs(local_forward[:, :3, 3] - forward_t[0].numpy()).max()))
        training_records.append({"frame_id": frame_id,
                                 "poses72": np.asarray(record["poses"]).copy(),
                                 "Rh": np.asarray(record["Rh"]).copy(), "Th": np.asarray(record["Th"]).copy(),
                                 "tpose_joints": np.asarray(record["tpose_joints"]).copy()})
        audits.append({"frame_id": frame_id,
                       "similarity_residual": part["similarity_residual"], "scale": part["scale"],
                       "published_camera_chain_note": "official body camera and published scaled-world cameras can have preprocessing differences; the complete-stage body-to-scaleworld chain is used unchanged"})
    coordinate_maps = np.stack(coordinate_maps)
    map_error = float(abs(coordinate_maps - coordinate_maps[0]).max())
    if map_error > 1e-5:
        raise ValueError(f"published old-world mapping is not fixed: {map_error}")
    protocol = run / "scenes" / scene / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    cache = {"schema": "V10.train_pose_cache.v1", "scene": scene,
             "train_frame_ids": train_ids, "heldout_frame_ids": heldout_ids,
             "canonical_joints": canonical, "H": height, "N": normalizer,
             "canonical_inverse": canonical_inverse,
             "local_quaternions_xyzw": np.stack(local_quaternions),
             "local_translations": np.stack(local_translations),
             "root_quaternions_xyzw": np.stack([x["quaternion_xyzw"] for x in root_parts]),
             "root_translations": np.stack([x["translation"] for x in root_parts]),
             "root_scales": np.asarray([x["scale"] for x in root_parts]),
             "training_source_values": training_records,
             "official_world_to_parent_world": coordinate_maps[0],
             "time_origin": 1, "time_denominator": definition["denominator"],
             "native_frame_count": len(train_ids) + len(heldout_ids),
             "camera_identity": camera_identity,
             "query_rule": "local quaternion SLERP and local T linear; complete root similarity R SLERP/t linear/log-positive-scale linear; nearest outside training range",
             "body_world_chain": "old smpl_to_scale_world @ [Rodrigues(Rh),Th]; then FK @ inverse canonical @ N",
             "joint_names": SMPL_JOINT_IDX, "parents": SMPL_PARENTS,
             "canonical_preprocessing": "published scene skeleton prior; all-frame preprocessing provenance not independently established",
             "metadata_container_access": "published pickle dictionaries are deserialized as containers; only train-frame pose/root fields are indexed, used or serialized into this cache; no development pose field is queried",
             "heldout_optimization_fields_used": [], "heldout_color_fields_used": []}
    adapter = PosePriorAdapter(cache)
    configuration = (json.loads((run / "configs/v10.json").read_text())
                     if (run / "configs/v10.json").exists() else {})
    support_random_seed = int(configuration.get("support_seed", 20261007))
    support_seed = build_joint_only_seed_cache(canonical, seed=support_random_seed,
                                              num_points=20000, grid_size=64)
    # Uniform in source frame time, snapped to the closest available training frame.
    numeric_ids = np.asarray([int(fid) for fid in train_ids])
    chosen = sorted(set(int(np.argmin(abs(numeric_ids - time))) for time in
                        np.linspace(numeric_ids[0], numeric_ids[-1], min(16, len(train_ids)))))
    color_frames = []
    for index in chosen:
        frame = frames[index]
        image_path, mask_path = Path(frame["rgb_path"]), Path(frame["mask_path"])
        if sha(image_path) != frame["rgb_sha256"] or sha(mask_path) != frame["mask_sha256"]:
            raise ValueError("selected training initialization image/mask hash changed")
        color_frames.append({"frame_id": frame["frame_id"],
                             "bone_transforms": adapter.query(frame["frame_id"])["bone_transforms"],
                             "K": np.asarray(frame["K"]), "w2c": np.asarray(frame["w2c"]),
                             "rgb": np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.,
                             "mask": np.asarray(Image.open(mask_path).convert("L"))})
    support_seed = initialize_seed_colors(support_seed, color_frames, train_ids)
    color_observations = {int(record["frame_id"]): record for record in
                          support_seed.setdefault("metadata", {}).get("color_sources", [])}
    support_seed["metadata"]["color_sources"] = [
        dict(color_observations.get(int(frames[index]["frame_id"]), {}),
             frame_id=frames[index]["frame_id"], rgb=identity(Path(frames[index]["rgb_path"])),
             mask=identity(Path(frames[index]["mask_path"]))) for index in chosen]
    figure_manifest = json.loads((ROOT / "experiments/local_dynamic_v9_20260929/run01/scenes" /
                                  scene / "figure_manifest.json").read_text())
    fixed_train = list(dict.fromkeys(item["frame_id"] for item in figure_manifest["figures"]
                                    if item.get("split") == "train"))
    if len(fixed_train) != 8 or any(fid not in train_ids for fid in fixed_train):
        raise ValueError("historical fixed eight training examples are not available")
    overlay_records = []
    overlay_folder = protocol / "skeleton_overlays"
    overlay_folder.mkdir(exist_ok=True)
    for frame_id in fixed_train:
        frame = frames[train_ids.index(frame_id)]
        record = mesh[frame_id]
        original_r, original_t = body.body_pose_to_body_RTs(record["poses"], record["tpose_joints"])
        original_fk = forward_kinematics(original_r.astype(np.float64), original_t.astype(np.float64))
        body_world = np.asarray(cameras[frame_id]["smpl_to_scale_world"]) @ root_transform(record["Rh"], record["Th"])
        official_joints = (body_world[None] @ original_fk)[:, :3, 3]
        ours, depth = projected(adapter.query(frame_id)["world_joints"], frame)
        reference, _ = projected(official_joints, frame)
        image = Image.open(frame["rgb_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for pixels, color, width in ((reference, (255, 0, 255), 5), (ours, (0, 255, 255), 2)):
            for child in range(1, 24):
                draw.line([tuple(pixels[SMPL_PARENTS[child]]), tuple(pixels[child])], fill=color, width=width)
            for x, y in pixels:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.rectangle((8, 8, 850, 47), fill=(0, 0, 0))
        draw.text((15, 17), f"{scene} train {frame_id}: official complete-stage chain magenta; adapter cyan", fill=(255,255,255))
        output = overlay_folder / f"{frame_id}.png"
        image.save(output)
        overlay_records.append({"frame_id": frame_id, "overlay": identity(output),
                                "max_projection_difference_pixels": float(abs(ours - reference).max()),
                                "positive_depth_joints": int((depth > 0).sum()),
                                "visual_review": "pending root inspection"})
    # Camera manifests are copied byte-for-byte: the evaluation file contains
    # query cameras and GT paths, never query poses or body-to-world transforms.
    train_manifest_out = protocol / "train_manifest.json"
    eval_manifest_out = protocol / "evaluation_manifest.json"
    train_manifest_out.write_bytes(train_path.read_bytes())
    eval_manifest_out.write_bytes(evaluation_path.read_bytes())
    schedule = training_schedule(train_ids, updates, seed)
    schedule["training_manifest_sha256"] = sha(train_path)
    save_json(protocol / "RGB_schedule.json", schedule)
    save_tensor(protocol / "pose_cache.pt", cache)
    save_tensor(protocol / "support_seed.pt", support_seed)
    save_json(protocol / "coordinate_audit.json", {
        "official_sources": {"body": identity(OFFICIAL / "core/utils/body_util.py"),
                             "network": identity(OFFICIAL / "core/utils/network_util.py"),
                             "loader": identity(OFFICIAL / "core/data/human_nerf/train.py"),
                             "camera": identity(OFFICIAL / "core/utils/camera_util.py")},
        "max_local_R_difference_from_official_epsilon_Rodrigues": max_local_rotation_difference,
        "max_global_FK_difference_from_official": max_fk_difference,
        "max_local_forward_difference_from_official_MotionBasisComputer": max_forward_difference,
        "max_old_world_map_variation_training_frames": map_error,
        "similarity_decomposition": audits,
        "fixed_training_overlays": overlay_records,
        "heldout_query_example": {key: value for key, value in adapter.query(heldout_ids[0]).items()
                                  if key in ("frame_id", "left_train_id", "right_train_id", "fraction",
                                             "outside_training_range", "model_time", "native_time")},
        "optimization_updates": 0, "GPU_used": False,
        "visual_review": "overlay files generated; not claimed reviewed until root inspection"})
    parent = ROOT / definition["parent"]
    result = {"status": "prepared", "scene": scene, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "parent": identity(parent), "parent_kind": definition["parent_kind"],
              "input_manifest": identity(train_path), "evaluation_manifest": identity(evaluation_path),
              "camera_source": camera_identity,
              "metadata": {name: identity(path) for name, path in selected_paths.items()},
              "pose_cache": identity(protocol / "pose_cache.pt"),
              "support_seed": identity(protocol / "support_seed.pt"),
              "RGB_schedule": identity(protocol / "RGB_schedule.json"),
              "train_manifest": identity(train_manifest_out), "query_camera_manifest": identity(eval_manifest_out),
              "coordinate_audit": identity(protocol / "coordinate_audit.json"),
              "train_frame_ids": train_ids, "retained_frame_ids": heldout_ids,
              "fixed_train_frame_ids": fixed_train, "uniform_color_frame_ids": [frames[index]["frame_id"] for index in chosen],
              "seed_points": int(len(support_seed["u"])), "support_seed_rng": support_random_seed,
              "optimization_updates": 0,
              "native_camera_substitution": False,
              "extra_method_condition": "P/PQ additionally use published training poses and scene canonical skeleton; C does not",
              "query_conditions": "frame_id and frozen query K/w2c; body pose from train-cache interpolation only; no query mask used for synthesis",
              "camera_preprocessing_limit": train["camera_source"].get("status"),
              "canonical_preprocessing_limit": cache["canonical_preprocessing"],
              "metadata_container_access": cache["metadata_container_access"],
              "source": {"adapter": identity(ROOT / "hoi_modules/pose_prior_adapter.py"),
                         "prepare": identity(Path(__file__).resolve()),
                         "support": identity(ROOT / "hoi_modules/pose_support_gaussians.py")}}
    if (run / "configs/v10.json").exists():
        result["configuration"] = identity(run / "configs/v10.json")
    save_json(protocol / "input_identity.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--scenes", nargs="+", choices=list(SCENES), default=list(SCENES))
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--rgb-seed", type=int, default=12345)
    args = parser.parse_args()
    if args.updates != 20000:
        raise ValueError("formal V10 preparation freezes exactly 20000 new updates")
    for scene in args.scenes:
        path = args.run / "scenes" / scene / "protocol/input_identity.json"
        if path.exists():
            prior = json.loads(path.read_text())
            for key in ("parent", "input_manifest", "evaluation_manifest", "camera_source", "pose_cache",
                        "support_seed", "RGB_schedule", "coordinate_audit"):
                if identity(Path(prior[key]["path"])) != prior[key]:
                    raise ValueError(f"frozen {scene} {key} changed; refusing overwrite")
            schedule = json.loads(Path(prior["RGB_schedule"]["path"]).read_text())
            configuration_path = args.run / "configs/v10.json"
            configuration = json.loads(configuration_path.read_text()) if configuration_path.exists() else {}
            if schedule["seed"] != args.rgb_seed or schedule["updates"] != args.updates:
                raise ValueError("cached RGB schedule differs from requested seed or updates")
            if prior["support_seed_rng"] != int(configuration.get("support_seed", 20261007)):
                raise ValueError("cached support RNG differs from frozen configuration")
            print(json.dumps({"scene": scene, "status": "reused_frozen_inputs"}), flush=True)
            continue
        result = prepare_scene(scene, args.run.resolve(), args.updates, args.rgb_seed)
        print(json.dumps({"scene": scene, "status": result["status"],
                          "seed_points": result["seed_points"],
                          "pose_cache_sha256": result["pose_cache"]["sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
