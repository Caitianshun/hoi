"""Differentiable, instance-isolated SMPL-X Gaussian attachments.

All forward outputs are in the GVHMR input-camera coordinate system, in metres.
Canonical static/region offsets are added after pose corrective blend shapes and
before skinning. The returned affine maps local covariance by F @ C @ F.T; it is
not generally orthonormal and must NOT be treated as a quaternion rotation.
No sequence registration, evaluation mesh, or held-out image is read here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
import smplx
from smplx.lbs import batch_rodrigues, batch_rigid_transform


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bounded_vectors(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    """Smooth radial bound: every vector norm is strictly below maximum."""
    return float(maximum) * raw / torch.sqrt(1.0 + raw.square().sum(-1, keepdim=True))


class HumanLBS(nn.Module):
    """SMPL-X vertices carrying Gaussian centres and local covariance frames.

    ``forward(int) -> centres[N,3], affine_frames[N,3,3]``. A 1-D list/tensor
    of frame indices returns batched ``[B,N,3]`` and ``[B,N,3,3]`` instead.
    Shapes/global pose/body pose/root translation are optimizable. Fingers and
    face remain the same default mean poses used by frozen GVHMR geometry.
    No object/background node or cross-instance skinning weight is possible.
    """

    def __init__(
        self,
        raw_geometry_path: str | Path,
        smplx_model_path: str | Path,
        vertex_ids: Iterable[int] | torch.Tensor | None = None,
        max_canonical_offset_m: float = 0.03,
        max_regional_offset_m: float = 0.015,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if max_canonical_offset_m < 0 or max_regional_offset_m < 0:
            raise ValueError("Residual bounds must be nonnegative metres")
        self.max_canonical_offset_m = float(max_canonical_offset_m)
        self.max_regional_offset_m = float(max_regional_offset_m)
        raw_geometry_path, smplx_model_path = Path(raw_geometry_path), Path(smplx_model_path)
        self.source_identity = {
            "raw_geometry_path": str(raw_geometry_path.resolve()),
            "raw_geometry_sha256": _sha(raw_geometry_path),
            "smplx_model_path": str(smplx_model_path.resolve()),
            "smplx_model_sha256": _sha(smplx_model_path),
            "coordinate_system": "GVHMR input-camera axes, metres",
            "reference_used": False,
        }
        with np.load(raw_geometry_path, allow_pickle=False) as raw:
            params = {k: torch.as_tensor(raw[k].copy(), dtype=dtype)
                      for k in ("body_pose", "global_orient", "transl", "betas")}
            timestamps = torch.as_tensor(raw["timestamp_seconds"].copy(), dtype=torch.float64)
            self.register_buffer("c2w", torch.as_tensor(raw["c2w"].copy(), dtype=dtype))
            self.register_buffer("K", torch.as_tensor(raw["K"].copy(), dtype=dtype))
        self.num_frames = int(params["body_pose"].shape[0])
        assert params["body_pose"].shape == (self.num_frames, 63)
        assert params["global_orient"].shape == params["transl"].shape == (self.num_frames, 3)
        assert params["betas"].shape == (self.num_frames, 10)
        if not all(torch.isfinite(v).all() for v in params.values()):
            raise ValueError("Nonfinite GVHMR initialization")
        if (params["betas"] - params["betas"][:1]).abs().max() > 1e-6:
            raise ValueError("Expected sequence-shared frozen GVHMR shape; do not silently average")
        if timestamps.shape != (self.num_frames,) or not (timestamps[1:] > timestamps[:-1]).all():
            raise ValueError("Actual timestamps must be strictly increasing")
        self.register_buffer("timestamps_seconds", timestamps)
        for name in ("body_pose", "global_orient", "transl"):
            self.register_parameter(name, nn.Parameter(params[name].clone()))
            self.register_buffer("initial_" + name, params[name].clone())
        self.betas = nn.Parameter(params["betas"][:1].clone())
        self.register_buffer("initial_betas", params["betas"][:1].clone())

        body = smplx.SMPLX(
            str(smplx_model_path), gender="neutral", num_betas=10,
            num_expression_coeffs=10, use_pca=True, num_pca_comps=12,
            flat_hand_mean=False, batch_size=1, dtype=dtype,
        )
        total_vertices = int(body.v_template.shape[0])
        ids = torch.arange(total_vertices) if vertex_ids is None else torch.as_tensor(vertex_ids, dtype=torch.long).cpu()
        if ids.ndim != 1 or len(ids) == 0 or (ids < 0).any() or (ids >= total_vertices).any():
            raise ValueError("vertex_ids must be a nonempty 1-D valid SMPL-X index array")
        if ids.unique().numel() != ids.numel():
            raise ValueError("Initial attachments must have unique vertex IDs; renderer may explicitly copy later")
        self.num_gaussians = int(ids.numel())
        self.register_buffer("vertex_ids", ids.clone())
        self.register_buffer("entity_ids", torch.ones(len(ids), dtype=torch.long))
        self.register_buffer("template_vertices", body.v_template[ids].detach().clone())
        self.register_buffer("shape_directions", body.shapedirs[ids].detach().clone())
        self.register_buffer("pose_directions", body.posedirs.reshape(486, total_vertices, 3)[:, ids].reshape(486, -1).detach().clone())
        self.register_buffer("joint_template", (body.J_regressor @ body.v_template).detach().clone())
        self.register_buffer("joint_shape_directions", torch.einsum("jv,vck->jck", body.J_regressor, body.shapedirs).detach().clone())
        self.register_buffer("parents", body.parents.detach().clone())
        self.register_buffer("pose_mean", body.pose_mean.detach().clone())
        self.register_buffer("lbs_weights", body.lbs_weights[ids].detach().clone())
        assert self.lbs_weights.shape == (len(ids), 55)
        assert torch.allclose(self.lbs_weights.sum(-1), torch.ones(len(ids), dtype=dtype), atol=1e-5)
        # Collapse each face/finger bone into its first ancestral BODY bone.
        # Face -> head(15), left fingers -> left wrist(20), right -> wrist(21).
        ancestor = []
        for j in range(55):
            parent = j
            while parent >= 22:
                parent = int(body.parents[parent])
            ancestor.append(parent)
        collapse = torch.zeros(55, 22, dtype=dtype)
        collapse[torch.arange(55), torch.tensor(ancestor)] = 1
        self.register_buffer("body_region_weights", self.lbs_weights @ collapse)
        self.register_buffer("bone_to_body_region", torch.tensor(ancestor, dtype=torch.long))
        self.canonical_offset_raw = nn.Parameter(torch.zeros(len(ids), 3, dtype=dtype))
        self.regional_residual_raw = nn.Parameter(torch.zeros(self.num_frames, 22, 3, dtype=dtype))
        # Mesh adjacency among selected attachments, exclusively within human.
        faces = body.faces_tensor.detach().cpu().long()
        edge_ids = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        mapping = torch.full((total_vertices,), -1, dtype=torch.long)
        mapping[ids] = torch.arange(len(ids))
        local = mapping[edge_ids]
        local = local[(local >= 0).all(-1)]
        local = torch.unique(local.sort(-1).values, dim=0)
        self.register_buffer("residual_edges", local)
        self.register_buffer("template_faces_vertex_ids", faces)
        del body

    def canonical_offsets(self) -> torch.Tensor:
        return _bounded_vectors(self.canonical_offset_raw, self.max_canonical_offset_m)

    def regional_offsets(self) -> torch.Tensor:
        return _bounded_vectors(self.regional_residual_raw, self.max_regional_offset_m)

    def canonical_centres(self) -> torch.Tensor:
        """Shaped canonical centres plus static residual; no pose corrective."""
        return self.template_vertices + torch.einsum("k,vck->vc", self.betas[0], self.shape_directions) + self.canonical_offsets()

    def _indices(self, frame_idx):
        indices = torch.as_tensor(frame_idx, dtype=torch.long, device=self.body_pose.device)
        scalar = indices.ndim == 0
        if indices.ndim > 1:
            raise ValueError("frame_idx must be scalar or one-dimensional")
        indices = indices.reshape(-1)
        if not len(indices) or (indices < 0).any() or (indices >= self.num_frames).any():
            raise IndexError("Invalid frame index")
        return indices, scalar

    def forward(self, frame_idx):
        idx, scalar = self._indices(frame_idx)
        batch = len(idx)
        pose = torch.cat([self.global_orient[idx], self.body_pose[idx],
                          torch.zeros(batch, 99, device=self.body_pose.device, dtype=self.body_pose.dtype)], -1)
        pose = pose + self.pose_mean[None]
        rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch, 55, 3, 3)
        joints = self.joint_template[None] + torch.einsum("bk,jck->bjc", self.betas, self.joint_shape_directions)
        joints = joints.expand(batch, -1, -1)
        _, transforms = batch_rigid_transform(rotations, joints, self.parents, dtype=self.body_pose.dtype)
        mixed = torch.einsum("nj,bjxy->bnxy", self.lbs_weights, transforms)
        affine = mixed[..., :3, :3]
        identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        corrective = ((rotations[:, 1:] - identity).reshape(batch, 486) @ self.pose_directions).reshape(batch, self.num_gaussians, 3)
        region_local = torch.einsum("nj,bjc->bnc", self.body_region_weights, self.regional_offsets()[idx])
        canonical_posed = self.canonical_centres()[None] + corrective + region_local
        centres = torch.einsum("bnij,bnj->bni", affine, canonical_posed) + mixed[..., :3, 3] + self.transl[idx, None]
        return (centres[0], affine[0]) if scalar else (centres, affine)

    def pose_parameters(self):
        return [self.body_pose, self.global_orient, self.transl, self.betas]

    def residual_parameters(self):
        return [self.canonical_offset_raw, self.regional_residual_raw]

    def pose_prior_losses(self, frame_idx=None):
        idx = torch.arange(self.num_frames, device=self.body_pose.device) if frame_idx is None else self._indices(frame_idx)[0]
        # Rotation matrix discrepancy avoids axis-angle wrap discontinuities.
        def rotation_error(a, b):
            return (batch_rodrigues(a.reshape(-1, 3)) - batch_rodrigues(b.reshape(-1, 3))).square().mean()
        return {"body_pose": rotation_error(self.body_pose[idx], self.initial_body_pose[idx]),
                "global_orient": rotation_error(self.global_orient[idx], self.initial_global_orient[idx]),
                "transl_m2": (self.transl[idx] - self.initial_transl[idx]).square().mean(),
                "betas": (self.betas - self.initial_betas).square().mean()}

    def residual_regularization(self, frame_idx=None, timestamps=None):
        """Unweighted scalar losses; trainer must explicitly set their weights.

        Temporal losses regularize 22 bounded canonical REGION offsets, not
        arbitrary per-Gaussian trajectories; true seconds set derivative units.
        ``frame_idx`` restricts amplitude only; time regularization uses sequence.
        """
        static = self.canonical_offsets()
        regional = self.regional_offsets()
        idx = torch.arange(self.num_frames, device=regional.device) if frame_idx is None else self._indices(frame_idx)[0]
        edges = self.residual_edges
        spatial = (static[edges[:, 0]] - static[edges[:, 1]]).square().mean() if len(edges) else static.sum() * 0
        ts = self.timestamps_seconds if timestamps is None else torch.as_tensor(timestamps, device=regional.device)
        ts = ts.to(device=regional.device, dtype=torch.float64)
        if ts.shape != (self.num_frames,) or not (ts[1:] > ts[:-1]).all():
            raise ValueError("Temporal regularization requires strictly increasing physical timestamps")
        dt = (ts[1:] - ts[:-1]).to(regional.dtype)
        velocity = (regional[1:] - regional[:-1]) / dt[:, None, None]
        acceleration = (velocity[1:] - velocity[:-1]) / ((dt[1:] + dt[:-1]) * 0.5)[:, None, None]
        zero = regional.sum() * 0
        return {"canonical_l2_m2": static.square().mean(),
                "canonical_spatial_m2": spatial,
                "regional_l2_m2": regional[idx].square().mean(),
                "regional_velocity_m2_s2": velocity.square().mean() if len(velocity) else zero,
                "regional_acceleration_m2_s4": acceleration.square().mean() if len(acceleration) else zero}
