"""V11 persistent rigid object Gaussians with two motion parameterisations.

Every object keeps one canonical Gaussian set for the whole video (rest and carried
states share colour and shape). Its pose T(t) maps canonical points to world:

* ``anchor``: state segments k = 1..K cover the frame axis; segment k has an anchor
  a_k (world identity or an SMPL wrist frame) and
      T(t) = A_{a_k}(t) S_k Delta_k(t),   t in [tau_{k-1}, tau_k).
  S_1 is learnable; S_{k+1} = A_{a_{k+1}}(tau_k)^-1 A_{a_k}(tau_k) S_k Delta_k(tau_k), so the
  world pose is continuous at every switch frame by construction. Delta_k is identity on
  world segments and an se(3) cubic B-spline (starting at identity) on hand segments.
* ``world``: a keyframe trajectory in world coordinates (cubic Hermite translation with
  learnable tangents, quaternion SLERP rotation), the parameterisation used by recent
  HOI Gaussian methods; it does not read any anchor at query time.

Anchors at query frames are supplied by the caller (training poses for training frames,
the protocol's pose source for retained frames). Switch-frame anchors are frozen buffers
computed from training poses only.
"""
import copy
import math

import torch
from torch import nn

SH_C0 = 0.28209479177387814
ANCHORS = ("world", "left", "right")


# ----------------------------------------------------------------------------- SE(3) helpers
def quat_to_matrix(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(R):
    """Rotation matrix (...,3,3) -> unit quaternion (w,x,y,z) with w >= 0."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cands = torch.stack((1 + t, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                         1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]), -1)
    i = cands.argmax(-1)
    out = torch.zeros(*m.shape[:-2], 4, dtype=m.dtype, device=m.device)
    s = torch.sqrt(cands.clamp_min(1e-12)) * 2
    q0 = torch.stack((s[..., 0] / 4, (m[..., 2, 1] - m[..., 1, 2]) / s[..., 0], (m[..., 0, 2] - m[..., 2, 0]) / s[..., 0], (m[..., 1, 0] - m[..., 0, 1]) / s[..., 0]), -1)
    q1 = torch.stack(((m[..., 2, 1] - m[..., 1, 2]) / s[..., 1], s[..., 1] / 4, (m[..., 0, 1] + m[..., 1, 0]) / s[..., 1], (m[..., 0, 2] + m[..., 2, 0]) / s[..., 1]), -1)
    q2 = torch.stack(((m[..., 0, 2] - m[..., 2, 0]) / s[..., 2], (m[..., 0, 1] + m[..., 1, 0]) / s[..., 2], s[..., 2] / 4, (m[..., 1, 2] + m[..., 2, 1]) / s[..., 2]), -1)
    q3 = torch.stack(((m[..., 1, 0] - m[..., 0, 1]) / s[..., 3], (m[..., 0, 2] + m[..., 2, 0]) / s[..., 3], (m[..., 1, 2] + m[..., 2, 1]) / s[..., 3], s[..., 3] / 4), -1)
    for k, q in enumerate((q0, q1, q2, q3)):
        out = torch.where((i == k)[..., None], q, out)
    out = torch.nn.functional.normalize(out, dim=-1)
    return torch.where(out[..., :1] < 0, -out, out)


def hat(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack((z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z), -1).reshape(*v.shape[:-1], 3, 3)


def se3_exp(xi):
    """xi (...,6) = (rho, omega) -> 4x4 rigid transform (exact, small-angle safe)."""
    rho, omega = xi[..., :3], xi[..., 3:]
    theta2 = (omega * omega).sum(-1, keepdim=True)[..., None]
    theta = theta2.clamp_min(1e-12).sqrt()
    W = hat(omega)
    W2 = W @ W
    small = theta2 < 1e-8
    A = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    B = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2.clamp_min(1e-12))
    C = torch.where(small, 1 / 6 - theta2 / 120, (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(1e-12))
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(W.shape)
    R = eye + A * W + B * W2
    V = eye + B * W + C * W2
    T = torch.zeros(*xi.shape[:-1], 4, 4, dtype=xi.dtype, device=xi.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = (V @ rho[..., None])[..., 0]
    T[..., 3, 3] = 1
    return T


def rigid(R, t):
    T = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def rigid_inverse(T):
    R, t = T[..., :3, :3], T[..., :3, 3]
    return rigid(R.transpose(-1, -2), -(R.transpose(-1, -2) @ t[..., None])[..., 0])


def slerp(q0, q1, u):
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin = torch.sin(theta)
    near = sin < 1e-6
    w0 = torch.where(near, 1 - u, torch.sin((1 - u) * theta) / sin.clamp_min(1e-6))
    w1 = torch.where(near, u, torch.sin(u * theta) / sin.clamp_min(1e-6))
    return torch.nn.functional.normalize(w0 * q0 + w1 * q1, dim=-1)


def bspline_basis(u):
    """Uniform cubic B-spline basis weights for local parameter u in [0,1) (4 weights)."""
    u2, u3 = u * u, u * u * u
    return torch.stack(((1 - u) ** 3 / 6, (3 * u3 - 6 * u2 + 4) / 6, (-3 * u3 + 3 * u2 + 3 * u + 1) / 6, u3 / 6), -1)


# ----------------------------------------------------------------------------- object bank
class ObjectAnchorGaussians(nn.Module):
    POINT_GROUPS = {"xyz": "_xyz", "f_dc": "_features_dc", "f_rest": "_features_rest",
                    "opacity": "_opacity", "scaling": "_scaling", "rotation": "_rotation"}

    def __init__(self, seeds, motion, mode, scene_extent, scale_bound, max_points_per_object=30000, device=None):
        """seeds: {'xyz','rgb','scale','object_id'} canonical arrays; motion: per-object spec dict."""
        super().__init__()
        if mode not in ("anchor", "world"):
            raise ValueError(mode)
        device = device or "cpu"
        f = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
        xyz, rgb = f(seeds["xyz"]), f(seeds["rgb"]).clamp(0.02, 0.98)
        n = len(xyz)
        self.mode = mode
        self.objects = [str(o) for o in motion["objects"]]
        self._xyz = nn.Parameter(xyz.clone())
        self._features_dc = nn.Parameter(((rgb - 0.5) / SH_C0)[:, None, :].clone())
        self._features_rest = nn.Parameter(xyz.new_zeros((n, 15, 3)))
        self._scaling = nn.Parameter(f(seeds["scale"]).clamp_min(1e-6).log()[:, None].expand(-1, 3).clone())
        rotation = xyz.new_zeros((n, 4))
        rotation[:, 0] = 1
        self._rotation = nn.Parameter(rotation)
        self._opacity = nn.Parameter(xyz.new_full((n, 1), math.log(0.1 / 0.9)))
        self.register_buffer("object_id", torch.as_tensor(seeds["object_id"], dtype=torch.long, device=device))
        self.register_buffer("point_id", torch.arange(n, dtype=torch.long, device=device))
        self.register_buffer("gradient_accum", xyz.new_zeros(n))
        self.register_buffer("visible_count", xyz.new_zeros(n))
        self.register_buffer("max_radii2D", xyz.new_zeros(n))
        self.scene_extent, self.scale_bound = float(scene_extent), float(scale_bound)
        self.max_points_per_object = int(max_points_per_object)
        self.next_point_id, self.active_sh_degree = n, 0
        self.grad_threshold, self.opacity_threshold = 0.0002, 0.005
        self.object_extent = {o: float(motion["extent"][o]) for o in self.objects}
        self.motion_spec = copy.deepcopy(motion)
        # ---- motion parameters
        self.segments, self.switch_anchor_pairs = {}, {}
        self.S1 = nn.ParameterDict()
        self.xi = nn.ParameterDict()
        self.kf_p, self.kf_v, self.kf_q = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        self.knot_spacing = int(motion.get("residual_knot_spacing", 8))
        self.keyframe_step = int(motion.get("keyframe_step", 4))
        for o in self.objects:
            spec = motion["per_object"][o]
            segments = [dict(start=int(s["start"]), end=int(s["end"]), anchor=s["anchor"]) for s in spec["segments"]]
            if segments[0]["start"] != 0 or any(a["end"] + 1 != b["start"] for a, b in zip(segments, segments[1:])):
                raise ValueError(f"segments of {o} must tile the frame axis")
            if any(s["anchor"] not in ANCHORS for s in segments):
                raise ValueError("unknown anchor")
            self.segments[o] = segments
            if mode == "anchor":
                s1 = torch.as_tensor(spec["S1"], dtype=torch.float32, device=device)
                self.S1[o] = nn.Parameter(torch.cat((s1[:3, 3], matrix_to_quat(s1[:3, :3]))))  # (t, q)
                for k, s in enumerate(segments):
                    if s["anchor"] != "world":
                        knots = (s["end"] - s["start"]) // self.knot_spacing + 4
                        self.xi[f"{o}__{k}"] = nn.Parameter(xyz.new_zeros((knots, 6)))
                # frozen switch anchors (training poses at the switch frame): before / after
                pairs = torch.as_tensor(spec["switch_anchors"], dtype=torch.float32, device=device)  # (K-1, 2, 4, 4)
                if pairs.shape != (len(segments) - 1, 2, 4, 4):
                    raise ValueError("switch anchors must be (K-1,2,4,4)")
                self.register_buffer(f"switch__{o}", pairs)
            else:
                init = torch.as_tensor(spec["keyframe_init"], dtype=torch.float32, device=device)  # (J, 4, 4) world poses
                self.kf_p[o] = nn.Parameter(init[:, :3, 3].clone())
                tangent = torch.zeros_like(init[:, :3, 3])
                tangent[1:-1] = (init[2:, :3, 3] - init[:-2, :3, 3]) / 2
                if len(init) > 1:
                    tangent[0], tangent[-1] = init[1, :3, 3] - init[0, :3, 3], init[-1, :3, 3] - init[-2, :3, 3]
                self.kf_v[o] = nn.Parameter(tangent)
                self.kf_q[o] = nn.Parameter(matrix_to_quat(init[:, :3, :3]))
        self.validate()

    # ------------------------------------------------------------------ bookkeeping
    def __len__(self):
        return len(self._xyz)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features = torch.cat((self._features_dc, self._features_rest), 1)
        mask = features.new_zeros(16)
        mask[:(self.active_sh_degree + 1) ** 2] = 1
        return features * mask[None, :, None]

    @property
    def get_opacity(self):
        return self._opacity.sigmoid()

    def validate(self):
        n = len(self)
        for attr in (*self.POINT_GROUPS.values(), "object_id", "point_id", "gradient_accum", "visible_count", "max_radii2D"):
            if len(getattr(self, attr)) != n:
                raise ValueError(f"misaligned {attr}")
        if n and (self.object_id.min() < 0 or self.object_id.max() >= len(self.objects)):
            raise ValueError("object ids out of range")

    # ------------------------------------------------------------------ poses
    def segment_index(self, o, frame):
        for k, s in enumerate(self.segments[o]):
            if s["start"] <= frame <= s["end"]:
                return k
        raise ValueError(f"frame {frame} outside the segment table of {o}")

    def _delta(self, o, k, frame):
        s = self.segments[o][k]
        if s["anchor"] == "world":
            return torch.eye(4, dtype=self._xyz.dtype, device=self._xyz.device)
        coeff = self.xi[f"{o}__{k}"]

        def spline(f):
            x = (f - s["start"]) / self.knot_spacing
            j = min(int(math.floor(x)), len(coeff) - 4)
            w = bspline_basis(torch.as_tensor(x - j, dtype=coeff.dtype, device=coeff.device))
            return (w[:, None] * coeff[j:j + 4]).sum(0)
        return se3_exp(spline(float(frame)) - spline(float(s["start"])))

    def anchor_pose(self, o, frame, anchors):
        """World pose T_o(frame) for the anchor mode; anchors maps 'left'/'right' to 4x4 rigid frames."""
        k = self.segment_index(o, frame)
        p = self.S1[o]
        S = rigid(quat_to_matrix(p[3:7]), p[:3])
        switches = getattr(self, f"switch__{o}")
        for j in range(k):
            S = rigid_inverse(switches[j, 1]) @ switches[j, 0] @ S @ self._delta(o, j, self.segments[o][j + 1]["start"])
        anchor = self.segments[o][k]["anchor"]
        A = torch.eye(4, dtype=S.dtype, device=S.device) if anchor == "world" else torch.as_tensor(anchors[anchor], dtype=S.dtype, device=S.device)
        return A @ S @ self._delta(o, k, frame)

    def world_pose(self, o, frame):
        p, v, q = self.kf_p[o], self.kf_v[o], self.kf_q[o]
        x = float(frame) / self.keyframe_step
        j = min(int(math.floor(x)), len(p) - 2)
        u = torch.as_tensor(x - j, dtype=p.dtype, device=p.device)
        h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
        t = h00 * p[j] + h10 * v[j] + h01 * p[j + 1] + h11 * v[j + 1]
        R = quat_to_matrix(slerp(torch.nn.functional.normalize(q[j], dim=-1), torch.nn.functional.normalize(q[j + 1], dim=-1), u))
        return rigid(R, t)

    def poses(self, frame, anchors=None):
        return torch.stack([self.anchor_pose(o, frame, anchors) if self.mode == "anchor" else self.world_pose(o, frame)
                            for o in self.objects])

    # ------------------------------------------------------------------ render attributes
    def forward(self, frame, anchors=None):
        T = self.poses(int(frame), anchors)[self.object_id]  # (N,4,4)
        R, t = T[:, :3, :3], T[:, :3, 3]
        means = torch.einsum("nij,nj->ni", R, self._xyz) + t
        scale = torch.exp(self._scaling.clamp(max=math.log(self.scale_bound)))
        Rg = quat_to_matrix(self._rotation)
        L = R @ Rg * scale[:, None, :]
        cov = L @ L.transpose(1, 2)
        eps = (1e-5 * self.scene_extent) ** 2
        cov = cov + torch.eye(3, dtype=cov.dtype, device=cov.device)[None] * eps
        packed = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], -1).contiguous()
        return {"means3D": means, "cov3D_precomp": packed, "shs": self.get_features, "opacities": self.get_opacity}

    # ------------------------------------------------------------------ optimisation
    def optimizer_groups(self, lr):
        groups = [{"name": "object." + k, "params": [getattr(self, a)], "lr": lr[k]} for k, a in self.POINT_GROUPS.items()]
        motion = list(self.S1.parameters()) + list(self.xi.parameters()) + list(self.kf_p.parameters()) + \
            list(self.kf_v.parameters()) + list(self.kf_q.parameters())
        groups.append({"name": "object.motion", "params": motion, "lr": lr["motion"]})
        return groups

    def motion_regulariser(self):
        if self.mode == "anchor":
            values = [c.pow(2).sum() for c in self.xi.values()]
        else:
            values = [((q.norm(dim=-1) - 1) ** 2).sum() for q in self.kf_q.values()]
        return torch.stack(values).sum() if values else self._xyz.new_zeros(())

    def set_active_sh_after_update(self, completed_updates, every=1000):
        self.active_sh_degree = min(3, int(completed_updates) // every)

    @torch.no_grad()
    def accumulate_density(self, screen_grad, visible, radii, batch_size):
        visible = torch.as_tensor(visible, dtype=torch.bool, device=self._xyz.device)
        if screen_grad.shape[0] != len(self) or visible.shape != (len(self),) or len(radii) != len(self):
            raise ValueError("screen statistics must be sliced to object rows")
        score = torch.linalg.vector_norm(screen_grad[:, :2] * int(batch_size), dim=-1)
        self.gradient_accum[visible] += score[visible]
        self.visible_count[visible] += 1
        self.max_radii2D[visible] = torch.maximum(self.max_radii2D[visible], radii.detach()[visible].float())

    def _replace_topology(self, optimizer, keep, child_parent, child_values):
        groups = {g.get("name"): g for g in optimizer.param_groups}
        for semantic, attr in self.POINT_GROUPS.items():
            old = getattr(self, attr)
            group = groups["object." + semantic]
            if len(group["params"]) != 1 or group["params"][0] is not old:
                raise ValueError(f"optimizer is not mapped by object.{semantic}")
            extension = child_values.get(semantic, old.detach()[child_parent])
            new = nn.Parameter(torch.cat((old.detach()[keep], extension), 0).clone())
            state = optimizer.state.pop(old, None)
            if state is not None:
                state = {k: (torch.cat((v[keep], torch.zeros_like(extension)), 0) if torch.is_tensor(v) and v.shape == old.shape else v)
                         for k, v in state.items()}
                optimizer.state[new] = state
            group["params"][0] = new
            setattr(self, attr, new)
        ids = torch.arange(self.next_point_id, self.next_point_id + len(child_parent), dtype=torch.long, device=self._xyz.device)
        self.point_id = torch.cat((self.point_id[keep], ids))
        self.object_id = torch.cat((self.object_id[keep], self.object_id[child_parent]))
        self.next_point_id += len(child_parent)
        for key in ("gradient_accum", "visible_count", "max_radii2D"):
            setattr(self, key, self._xyz.new_zeros(len(self)))
        self.validate()

    @torch.no_grad()
    def after_adam_topology(self, optimizer, completed_updates, start=200, stop=3500, interval=100, generator=None):
        k = int(completed_updates)
        if not (start <= k <= stop and k % interval == 0):
            return None
        before = len(self)
        score = self.gradient_accum / self.visible_count.clamp_min(1)
        prune = (self.get_opacity[:, 0] < self.opacity_threshold) if k > start else torch.zeros(before, dtype=torch.bool, device=self._xyz.device)
        candidates = torch.nonzero((score >= self.grad_threshold) & (self.visible_count > 0) & ~prune, as_tuple=False).flatten()
        candidates = candidates[torch.argsort(self.point_id[candidates], stable=True)]
        candidates = candidates[torch.argsort(score[candidates], descending=True, stable=True)]
        selected = []
        for j, o in enumerate(self.objects):
            mine = candidates[self.object_id[candidates] == j]
            alive = int(((self.object_id == j) & ~prune).sum())
            selected.append(mine[:max(0, self.max_points_per_object - alive)])
        selected = torch.cat(selected) if selected else candidates[:0]
        extent = torch.as_tensor([self.object_extent[o] for o in self.objects], device=self._xyz.device)[self.object_id]
        large = self._scaling.detach().exp().amax(1) > 0.01 * extent
        clone, split = selected[~large[selected]], selected[large[selected]]
        keep_mask = ~prune
        keep_mask[split] = False
        keep = torch.nonzero(keep_mask, as_tuple=False).flatten()
        child_parent = torch.cat((clone, split.repeat_interleave(2)))
        child_values = {}
        if len(child_parent):
            xyz = self._xyz.detach()[child_parent].clone()
            scales = self._scaling.detach()[child_parent].clone()
            if len(split):
                rep = split.repeat_interleave(2)
                noise = torch.randn((len(rep), 3), dtype=xyz.dtype, device=xyz.device, generator=generator) * self._scaling.detach()[rep].exp()
                xyz[len(clone):] += torch.einsum("nij,nj->ni", quat_to_matrix(self._rotation.detach()[rep]), noise)
                scales[len(clone):] -= math.log(1.6)
            child_values = {"xyz": xyz, "scaling": scales}
        if len(child_parent) or prune.any():
            self._replace_topology(optimizer, keep, child_parent, child_values)
        else:
            self.gradient_accum.zero_()
            self.visible_count.zero_()
            self.max_radii2D.zero_()
        return {"before": before, "after": len(self), "opacity_pruned": int(prune.sum()),
                "clone_parents": len(clone), "split_parents": len(split), "new_children": len(child_parent)}

    # ------------------------------------------------------------------ checkpoints
    def get_extra_state(self):
        return {"schema": "v11_object_v1", "mode": self.mode, "objects": self.objects, "segments": self.segments,
                "scene_extent": self.scene_extent, "scale_bound": self.scale_bound, "active_sh_degree": self.active_sh_degree,
                "next_point_id": self.next_point_id, "max_points_per_object": self.max_points_per_object,
                "object_extent": self.object_extent, "motion_spec": self.motion_spec,
                "knot_spacing": self.knot_spacing, "keyframe_step": self.keyframe_step}

    def set_extra_state(self, state):
        if state["schema"] != "v11_object_v1":
            raise ValueError("unknown object checkpoint schema")
        for key, value in state.items():
            if key != "schema":
                setattr(self, key, copy.deepcopy(value))

    def checkpoint_payload(self):
        sizes = {"n": len(self)}
        return {"schema": "v11_object_v1", "sizes": sizes, "state_dict": self.state_dict(),
                "init": {"motion": self.motion_spec, "mode": self.mode, "scene_extent": self.scene_extent,
                         "scale_bound": self.scale_bound, "max_points_per_object": self.max_points_per_object}}

    @classmethod
    def from_checkpoint(cls, payload, device=None):
        if payload["schema"] != "v11_object_v1":
            raise ValueError("unknown object checkpoint schema")
        init, n = payload["init"], payload["sizes"]["n"]
        seeds = {"xyz": torch.zeros(n, 3), "rgb": torch.full((n, 3), 0.5), "scale": torch.ones(n),
                 "object_id": torch.zeros(n, dtype=torch.long)}
        bank = cls(seeds, init["motion"], init["mode"], init["scene_extent"], init["scale_bound"], init["max_points_per_object"], device)
        bank.load_state_dict(payload["state_dict"], strict=True)
        return bank
