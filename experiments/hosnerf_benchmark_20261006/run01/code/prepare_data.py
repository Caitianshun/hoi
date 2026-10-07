#!/usr/bin/env python3
"""Prepare isolated, audited copies of the public HOSNeRF datasets.

Only numeric RGB/mask PNGs and the six published metadata files are selected.
Original archives and historical extracted datasets are never modified. Missing
archives are reported without preventing preparation of the available scenes.
The generated manifests are private experiment records, not source code assets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import shutil
import stat
import tempfile
import zipfile

import numpy as np
from PIL import Image


SCENES = ("Backpack", "Tennis", "Lounge", "Suitcase", "Playground", "Dance")
METADATA = (
    "poses_bounds.npy", "cameras.pkl", "cameras_scaleworld.pkl",
    "mesh_infos.pkl", "canonical_joints.pkl", "transitions_times.json",
)
# Six files are shipped with each public sequence (seven with .DS_Store, omitted).
PNG_NAME = re.compile(r"^[0-9]+\.png$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                     encoding="utf-8") as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


class PublishedNumpyUnpickler(pickle.Unpickler):
    """Read these numpy-only public metadata pickles without arbitrary globals."""

    def find_class(self, module: str, name: str):
        allowed = {
            ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
            ("builtins", "set"), ("builtins", "frozenset"),
            ("_codecs", "encode"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported metadata global {module}.{name}")
        return super().find_class(module, name)


def load_pickle(path: Path):
    return PublishedNumpyUnpickler(io.BytesIO(path.read_bytes()), encoding="latin1").load()


def selected_member(info: zipfile.ZipInfo, scene: str) -> Path | None:
    p = PurePosixPath(info.filename)
    if p.is_absolute() or ".." in p.parts or "\\" in info.filename:
        raise ValueError(f"Unsafe archive member: {info.filename!r}")
    if any(part.startswith(".") or part == "__MACOSX" for part in p.parts):
        return None
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"Archive symlink is not permitted: {info.filename!r}")
    if info.is_dir() or not p.parts or p.parts[0] != scene:
        return None
    relative = p.parts[1:]
    if len(relative) == 1 and relative[0] in METADATA:
        return Path(relative[0])
    if len(relative) == 2 and relative[0] in ("images", "masks") and PNG_NAME.fullmatch(relative[1]):
        return Path(*relative)
    return None


def finite_array(value, shape: tuple[int, ...], label: str) -> None:
    arr = np.asarray(value)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{label}: expected finite shape {shape}, got {arr.shape}")


def audit_scene(folder: Path, scene: str) -> dict:
    rgb = sorted((folder / "images").glob("*.png"))
    mask = sorted((folder / "masks").glob("*.png"))
    files = sorted(rgb + mask + [folder / name for name in METADATA])
    ids = [p.stem for p in rgb]
    mask_ids = [p.stem for p in mask]
    if len(ids) < 16 or len(set(ids)) != len(ids) or ids != mask_ids:
        raise ValueError(f"{scene}: RGB/mask frame IDs do not correspond exactly")
    if [int(i) for i in ids] != list(range(len(ids))):
        raise ValueError(f"{scene}: public numeric frame IDs are not contiguous from zero")
    resolutions = set()
    for image, alpha in zip(rgb, mask):
        with Image.open(image) as im:
            size = im.size
            if im.mode != "RGB":
                raise ValueError(f"{image}: expected RGB PNG, got {im.mode}")
            im.verify()
        with Image.open(alpha) as im:
            if im.size != size or im.mode not in ("L", "RGB", "RGBA", "P"):
                raise ValueError(f"{alpha}: mask size/mode mismatch")
            im.verify()
        resolutions.add(size)
    if len(resolutions) != 1:
        raise ValueError(f"{scene}: varying RGB resolutions")
    for name in METADATA:
        if not (folder / name).is_file():
            raise ValueError(f"{scene}: missing required metadata {name}")
    poses = np.load(folder / "poses_bounds.npy", allow_pickle=False)
    finite_array(poses, (len(ids), 17), f"{scene}/poses_bounds.npy")
    mappings = {name: load_pickle(folder / name) for name in
                ("cameras.pkl", "cameras_scaleworld.pkl", "mesh_infos.pkl")}
    fields = {
        "cameras.pkl": {"intrinsics": (3, 3), "smpl_to_camera": (4, 4),
                        "smpl_to_world": (4, 4), "world_to_camera": (4, 4)},
        "cameras_scaleworld.pkl": {"intrinsics": (3, 3), "smpl_to_camera": (4, 4),
                                   "smpl_to_scale_world": (4, 4), "scaleworld_to_camera": (4, 4)},
        "mesh_infos.pkl": {"Rh": (3,), "Th": (3,), "poses": (72,),
                           "joints": (24, 3), "tpose_joints": (24, 3)},
    }
    for name, mapping in mappings.items():
        if not isinstance(mapping, dict) or set(mapping) != set(ids):
            raise ValueError(f"{scene}/{name}: frame keys disagree with RGB")
        if name.startswith("cameras") and list(mapping) != ids:
            # The public first-stage loader enumerates dictionary order.
            raise ValueError(f"{scene}/{name}: camera dictionary order disagrees with RGB")
        for frame_id in ids:
            for key, shape in fields[name].items():
                if key not in mapping[frame_id]:
                    raise ValueError(f"{scene}/{name}/{frame_id}: missing {key}")
                finite_array(mapping[frame_id][key], shape, f"{scene}/{name}/{frame_id}/{key}")
    canonical = load_pickle(folder / "canonical_joints.pkl")
    finite_array(canonical["joints"], (24, 3), f"{scene}/canonical_joints.pkl")
    transitions = json.loads((folder / "transitions_times.json").read_text())
    if not isinstance(transitions, dict) or not transitions:
        raise ValueError(f"{scene}: transition dictionary is missing/empty")
    for key, value in transitions.items():
        if not isinstance(value, dict) or not np.isfinite(float(value["time"])):
            raise ValueError(f"{scene}: invalid transition {key}")
    interval = len(ids) // 16
    test_indices = list(range(0, len(ids), interval))[:16]
    test_set = set(test_indices)
    test_ids = [ids[i] for i in test_indices]
    train_ids = [frame_id for i, frame_id in enumerate(ids) if i not in test_set]
    width, height = next(iter(resolutions))
    file_records = [{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size,
                     "sha256": digest(p)} for p in files]
    return {
        "status": "prepared", "scene": scene, "data_path": str(folder),
        "frame_count": len(ids), "resolution": {"width": width, "height": height},
        "frame_ids": ids, "train_ids": train_ids, "test_ids": test_ids,
        "test_indices": test_indices, "test_interval_floor": interval,
        "split_rule": "numeric filename order; first 16 indices of range(0, N, N // 16)",
        "stage1_validation_ids": train_ids[:2],
        "time_rule": "published nominal linspace(0, 1, N); physical exposure timestamps not provided",
        "metadata": {r["path"]: r for r in file_records if r["path"] in METADATA},
        "files": file_records,
        "flow_status": "not prepared; current-to-previous retained training frame required",
        "published_prior_condition": "published cameras, pose fits, canonical joints, foreground masks, transition times",
        "audit": {"rgb_mask_ids": "exact", "pose_rows": "exact", "camera_mesh_keys": "exact",
                  "camera_order": "exact", "png_crc": "verified", "metadata_values": "finite"},
    }


def prepare_scene(archive: Path, destination: Path, scene: str) -> dict:
    archive_hash = digest(archive)
    if destination.exists():
        prior_file = destination.parent.parent / "protocol" / "dataset_manifest.json"
        if not prior_file.exists():
            raise FileExistsError(f"Will not overwrite unregistered dataset {destination}")
        prior = json.loads(prior_file.read_text()).get("scenes", {}).get(scene, {})
        if prior.get("archive", {}).get("sha256") != archive_hash:
            raise FileExistsError(f"Will not overwrite dataset with different archive identity: {destination}")
        expected = {r["path"]: r["sha256"] for r in prior.get("files", [])}
        current = audit_scene(destination, scene)
        actual = {r["path"]: r["sha256"] for r in current["files"]}
        if expected != actual:
            raise FileExistsError(f"Existing prepared data changed; refusing to replace: {destination}")
        result = current
        result["reuse"] = "byte-identical original prepared inputs"
    else:
        staging = Path(tempfile.mkdtemp(prefix=f".{scene}.prepare-", dir=destination.parent))
        try:
            selected = set()
            ignored = []
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    rel = selected_member(info, scene)
                    if rel is None:
                        if not info.is_dir():
                            ignored.append(info.filename)
                        continue
                    if rel in selected:
                        raise ValueError(f"Duplicate selected archive member {rel}")
                    selected.add(rel)
                    output = staging / rel
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as source, output.open("xb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            result = audit_scene(staging, scene)
            if destination.exists():
                raise FileExistsError(f"Destination appeared during preparation: {destination}")
            staging.rename(destination)
            result["data_path"] = str(destination)
            result["ignored_archive_members"] = ignored
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    result["archive"] = {"path": str(archive), "bytes": archive.stat().st_size, "sha256": archive_hash}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archives", type=Path)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    args = parser.parse_args()
    run = args.run.resolve()
    archives = args.archives.resolve() if args.archives else run.parents[2] / "other data"
    data = run / "data"
    data.mkdir(parents=True, exist_ok=True)
    previous = run / "protocol" / "dataset_manifest.json"
    prior = json.loads(previous.read_text()) if previous.exists() else {}
    scenes = dict(prior.get("scenes", {}))
    failed = False
    for scene in args.scenes:
        archive = archives / f"{scene}.zip"
        if not archive.is_file():
            scenes[scene] = {"status": "missing", "scene": scene, "expected_archive": str(archive),
                             "reason": "archive is not available; other scenes prepared independently"}
        else:
            try:
                scenes[scene] = prepare_scene(archive, data / scene, scene)
            except Exception as exc:
                failed = True
                scenes[scene] = {"status": "failed", "scene": scene, "archive_path": str(archive),
                                 "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps({"scene": scene, "status": scenes[scene]["status"],
                          "frame_count": scenes[scene].get("frame_count"),
                          "error": scenes[scene].get("error")}), flush=True)
        manifest = {"schema_version": 1, "updated_at_utc": utc_now(),
                    "scope": "isolated published HOSNeRF inputs; no training/evaluation results",
                    "run_path": str(run), "archive_root": str(archives), "scenes": scenes,
                    "source_script": {"path": str(Path(__file__).resolve()),
                                      "sha256": digest(Path(__file__))}}
        write_json(previous, manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
