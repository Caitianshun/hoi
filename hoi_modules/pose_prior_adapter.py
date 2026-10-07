"""Train-pose-only, 24-human-bone HOSNeRF forward motion in a frozen world.

The complete-stage HOSNeRF loader maps body coordinates by
``smpl_to_scale_world @ [Rodrigues(Rh), Th]``.  Local pose FK follows
MotionBasisComputer; it is not a second application of Rh/Th.  This module has
no dependency on SMPL meshes and never opens a query-frame metadata file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import torch


SMPL_JOINT_IDX = {
    "pelvis_root": 0, "left_hip": 1, "right_hip": 2, "belly_button": 3,
    "left_knee": 4, "right_knee": 5, "lower_chest": 6, "left_ankle": 7,
    "right_ankle": 8, "upper_chest": 9, "left_toe": 10, "right_toe": 11,
    "neck": 12, "left_clavicle": 13, "right_clavicle": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18,
    "right_elbow": 19, "left_wrist": 20, "right_wrist": 21,
    "left_thumb": 22, "right_thumb": 23,
}
SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                        9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.zeros((*rotation.shape[:-2], 4, 4), dtype=np.float64)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = translation
    result[..., 3, 3] = 1.0
    return result


def canonical_normalization(joints: np.ndarray) -> tuple[float, np.ndarray]:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (24, 3) or not np.isfinite(joints).all():
        raise ValueError("canonical joints must be finite [24,3]")
    pelvis = joints[SMPL_JOINT_IDX["pelvis_root"]]
    head = joints[SMPL_JOINT_IDX["head"]]
    feet = joints[[SMPL_JOINT_IDX["left_toe"], SMPL_JOINT_IDX["right_toe"]]]
    height = float(np.linalg.norm(head - pelvis) +
                   .5 * np.linalg.norm(feet - pelvis, axis=1).sum())
    if height <= 0:
        raise ValueError("degenerate canonical height")
    normalizer = transform(np.eye(3) * height, pelvis)
    return height, normalizer


def local_pose(poses72: np.ndarray, tpose_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The official local R/T formula, with exact Rodrigues rotations.

    The upstream helper divides its rotation axis by (norm + 1e-5).  Exact
    Rodrigues removes that tiny non-orthogonality before quaternion SLERP;
    the preparation receipt records the difference against upstream itself.
    """
    angles = np.asarray(poses72, dtype=np.float64).reshape(24, 3)
    joints = np.asarray(tpose_joints, dtype=np.float64)
    if joints.shape != (24, 3) or not np.isfinite(angles).all() or not np.isfinite(joints).all():
        raise ValueError("nonfinite or non-human local pose")
    rotations = Rotation.from_rotvec(angles).as_matrix()
    translations = joints.copy()
    translations[1:] -= joints[SMPL_PARENTS[1:]]
    return rotations, translations


def forward_kinematics(rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """24 local transforms -> global body transforms, official parent order."""
    if rotations.shape != (24, 3, 3) or translations.shape != (24, 3):
        raise ValueError("FK requires exactly 24 human bones")
    local = transform(rotations, translations)
    result = local.copy()
    for joint in range(1, 24):
        result[joint] = result[SMPL_PARENTS[joint]] @ local[joint]
    return result


def decompose_similarity(matrix: np.ndarray, tolerance: float = 1e-5) -> dict:
    """Validate similarity before turning its unscaled rotation into a quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("body-to-world must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=tolerance, rtol=0):
        raise ValueError("body-to-world is not an affine homogeneous matrix")
    linear = matrix[:3, :3]
    left, values, right = np.linalg.svd(linear)
    scale = float(values.mean())
    rotation = left @ right
    if scale <= 0 or np.linalg.det(rotation) <= 0:
        raise ValueError("body-to-world has zero scale or reflection")
    error = float(np.max(np.abs(linear - scale * rotation)))
    if error > tolerance * max(1., scale):
        raise ValueError(f"non-similarity body-to-world; residual={error}")
    return {"quaternion_xyzw": Rotation.from_matrix(rotation).as_quat(),
            "translation": matrix[:3, 3].copy(), "scale": scale,
            "similarity_residual": error, "singular_values": values}


def slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-arc normalized quaternion interpolation, xyzw ordering."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0., 1.)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    close = dot > .9995
    denominator = np.maximum(sine, 1e-12)
    spherical = (np.sin((1 - fraction) * angle) / denominator * q0 +
                 np.sin(fraction * angle) / denominator * q1)
    linear = (1 - fraction) * q0 + fraction * q1
    result = np.where(close, linear, spherical)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def root_transform(Rh: np.ndarray, Th: np.ndarray) -> np.ndarray:
    return transform(Rotation.from_rotvec(np.asarray(Rh, dtype=np.float64)).as_matrix(),
                     np.asarray(Th, dtype=np.float64))


class PosePriorAdapter:
    """Query by source frame number, using only a frozen training-pose cache."""

    def __init__(self, cache: dict[str, Any]):
        self.cache = cache
        self.frame_ids = tuple(str(frame) for frame in cache["train_frame_ids"])
        self.frame_numbers = np.asarray([int(frame) for frame in self.frame_ids], dtype=np.int64)
        if len(self.frame_ids) < 1 or np.any(np.diff(self.frame_numbers) <= 0):
            raise ValueError("training frame IDs must be nonempty and ordered")
        self.canonical_joints = np.asarray(cache["canonical_joints"], dtype=np.float64)
        self.H, self.N = canonical_normalization(self.canonical_joints)
        self.canonical_inverse = np.asarray(cache["canonical_inverse"], dtype=np.float64)
        self.local_quaternions = np.asarray(cache["local_quaternions_xyzw"], dtype=np.float64)
        self.local_translations = np.asarray(cache["local_translations"], dtype=np.float64)
        self.root_quaternions = np.asarray(cache["root_quaternions_xyzw"], dtype=np.float64)
        self.root_translations = np.asarray(cache["root_translations"], dtype=np.float64)
        self.root_scales = np.asarray(cache["root_scales"], dtype=np.float64)
        self.time_origin = int(cache["time_origin"])
        self.time_denominator = int(cache["time_denominator"])
        self.native_frame_count = int(cache["native_frame_count"])
        self._numpy_queries: dict[int, dict] = {}
        self._torch_queries: dict[tuple, torch.Tensor] = {}
        if set(self.frame_ids) & set(cache.get("heldout_frame_ids", [])):
            raise ValueError("cache training IDs overlap heldout IDs")

    @classmethod
    def from_cache(cls, path: Path | str) -> "PosePriorAdapter":
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def model_time(self, frame_id: str | int) -> float:
        return (int(frame_id) - self.time_origin) / self.time_denominator

    def native_time(self, frame_id: str | int) -> float:
        return int(frame_id) / (self.native_frame_count - 1)

    def query(self, frame_id: str | int) -> dict:
        number = int(frame_id)
        if number in self._numpy_queries:
            return self._numpy_queries[number]
        right = int(np.searchsorted(self.frame_numbers, number, side="left"))
        if right == 0:
            left = right = 0
        elif right == len(self.frame_numbers):
            left = right = len(self.frame_numbers) - 1
        elif self.frame_numbers[right] == number:
            left = right
        else:
            left = right - 1
        fraction = (0. if left == right else
                    (number - self.frame_numbers[left]) /
                    (self.frame_numbers[right] - self.frame_numbers[left]))
        local_r = Rotation.from_quat(slerp(self.local_quaternions[left],
                                          self.local_quaternions[right], fraction)).as_matrix()
        local_t = ((1 - fraction) * self.local_translations[left] +
                   fraction * self.local_translations[right])
        root_r = Rotation.from_quat(slerp(self.root_quaternions[left],
                                         self.root_quaternions[right], fraction)).as_matrix()
        root_t = ((1 - fraction) * self.root_translations[left] +
                  fraction * self.root_translations[right])
        scale = float(np.exp((1 - fraction) * np.log(self.root_scales[left]) +
                             fraction * np.log(self.root_scales[right])))
        body_world = transform(root_r * scale, root_t)
        global_pose = forward_kinematics(local_r, local_t)
        bones = body_world[None] @ global_pose @ self.canonical_inverse @ self.N
        result = {"frame_id": f"{number:05d}", "left_train_id": self.frame_ids[left],
                  "right_train_id": self.frame_ids[right], "fraction": float(fraction),
                  "outside_training_range": bool(number < self.frame_numbers[0] or
                                                   number > self.frame_numbers[-1]),
                  "local_rotation": local_r, "local_translation": local_t,
                  "global_pose": global_pose, "body_to_world": body_world,
                  "bone_transforms": bones,
                  "world_joints": (body_world[None] @ global_pose)[:, :3, 3],
                  "model_time": self.model_time(number), "native_time": self.native_time(number)}
        self._numpy_queries[number] = result
        return result

    def bone_transforms(self, frame_id: str | int, device=None,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
        key = (int(frame_id), str(device), dtype)
        if key not in self._torch_queries:
            self._torch_queries[key] = torch.as_tensor(
                self.query(frame_id)["bone_transforms"].copy(), device=device, dtype=dtype)
        return self._torch_queries[key]

    def state_dict(self) -> dict:
        return self.cache
