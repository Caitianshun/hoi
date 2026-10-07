"""V10 independent pose support Gaussian bank (canonical u coordinates).

The bone-volume initialization below adapts the 24-human-joint portion of
HOSNeRF body_util.approx_gaussian_bone_volumes, ca169704a9d965e8266c034cceb506ffba72f405
(Apache-2.0; Show Lab/NUS, derived from HumanNeRF). It retains its inverse-std
Gaussian fields and torso/head rules; excludes object nodes; uses voxel centers
and a deterministic rotation for antiparallel/zero-length bones. These volumes
are weak support fields, not an occupancy or known surface.
"""
import copy
import math

import numpy as np
import torch
from torch import nn


SMPL_JOINT_IDX = {
    'pelvis_root': 0, 'left_hip': 1, 'right_hip': 2, 'belly_button': 3,
    'left_knee': 4, 'right_knee': 5, 'lower_chest': 6, 'left_ankle': 7,
    'right_ankle': 8, 'upper_chest': 9, 'left_toe': 10, 'right_toe': 11,
    'neck': 12, 'left_clavicle': 13, 'right_clavicle': 14, 'head': 15,
    'left_shoulder': 16, 'right_shoulder': 17, 'left_elbow': 18,
    'right_elbow': 19, 'left_wrist': 20, 'right_wrist': 21,
    'left_thumb': 22, 'right_thumb': 23,
}
SMPL_PARENT = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
               10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13,
               17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
BONE_STDS = np.array([0.03, 0.06, 0.03], dtype=np.float64)
HEAD_STDS = np.array([0.06, 0.06, 0.06], dtype=np.float64)
JOINT_STDS = np.array([0.02, 0.02, 0.02], dtype=np.float64)
TORSO_JOINTS = {SMPL_JOINT_IDX[n] for n in
                ('pelvis_root', 'belly_button', 'lower_chest', 'upper_chest', 'left_clavicle', 'right_clavicle')}
SH_C0 = 0.28209479177387814


def align_bone_rotation(source, target):
    """Deterministic rotation aligning directions; identity for zero bones."""
    a, b = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if min(na, nb) < 1e-12:
        return np.eye(3)
    a, b = a / na, b / nb
    c = np.clip(a @ b, -1, 1)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def canonical_normalization(joints):
    j = np.asarray(joints, dtype=np.float64)
    if j.shape != (24, 3) or not np.isfinite(j).all():
        raise ValueError('canonical_joints must be finite [24,3] in original canonical units')
    p, h, lf, rf = [SMPL_JOINT_IDX[n] for n in ('pelvis_root', 'head', 'left_toe', 'right_toe')]
    height = np.linalg.norm(j[h] - j[p]) + 0.5 * (np.linalg.norm(j[p] - j[lf]) + np.linalg.norm(j[p] - j[rf]))
    if height <= 1e-12:
        raise ValueError('Degenerate canonical height proxy')
    n = np.eye(4)
    n[:3, :3] *= height
    n[:3, 3] = j[p]
    return height, n


def approx_human_bone_volumes(joints, bbox_min, bbox_max, grid_size=64):
    """24 human channels plus background, [25,z,y,x], in original units."""
    j = np.asarray(joints, dtype=np.float64)
    lo, hi = np.asarray(bbox_min), np.asarray(bbox_max)
    axes = [lo[k] + (np.arange(grid_size) + 0.5) * (hi[k] - lo[k]) / grid_size for k in range(3)]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    xyz = np.stack((x, y, z), axis=-1)

    def gaussian(center, std, rotation):
        inv_std = np.diag(1 / std)
        precision = rotation @ inv_std @ inv_std @ rotation.T
        d = xyz - center
        return np.exp(-np.einsum('...i,ij,...j->...', d, precision, d))

    volumes = []
    for parent in range(24):
        children = [c for c, p in SMPL_PARENT.items() if p == parent]
        if children:
            value = np.zeros((grid_size,) * 3, dtype=np.float64)
            for child in children:
                std = BONE_STDS * 2
                if parent in TORSO_JOINTS:
                    std = std * np.array([1.5, 1, 1.5])
                rotation = align_bone_rotation([0, 1, 0], j[child] - j[parent])
                value += gaussian((j[parent] + j[child]) / 2, std, rotation)
        else:
            std = HEAD_STDS if parent == SMPL_JOINT_IDX['head'] else JOINT_STDS
            value = gaussian(j[parent], std * 2, np.eye(3))
        volumes.append(value)
    human = np.stack(volumes)
    background = 1 - np.clip(human.sum(0, keepdims=True), 0, 1)
    all_channels = np.concatenate((human, background))
    return (all_channels / all_channels.sum(0, keepdims=True).clip(min=0.001)).astype(np.float32)


def build_joint_only_seed_cache(canonical_joints, seed=20261007, num_points=20000, grid_size=64):
    """CPU-only, deterministic weak seeds and 3-neighbor mean squared KNN."""
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree
    j = np.asarray(canonical_joints, dtype=np.float64)
    height, normalizer = canonical_normalization(j)
    lo, hi = j.min(0) - 0.15 * height, j.max(0) + 0.15 * height
    volumes = approx_human_bone_volumes(j, lo, hi, grid_size)
    rng = np.random.default_rng(seed)
    quota = np.full(24, num_points // 24, dtype=np.int64)
    quota[:num_points % 24] += 1
    sampled, source = [], []
    for channel, count in enumerate(quota):
        distribution = volumes[channel].astype(np.float64).ravel()
        if distribution.sum() <= 0:
            raise ValueError(f'Bone channel {channel} has no sampling support')
        index = rng.choice(len(distribution), int(count), p=distribution / distribution.sum())
        iz, iy, ix = np.unravel_index(index, (grid_size,) * 3)
        voxel = np.stack((ix, iy, iz), axis=-1)
        sampled.append(lo + (voxel + rng.random((int(count), 3))) * (hi - lo) / grid_size)
        source.append(np.full(int(count), channel, dtype=np.int64))
    xyz = np.concatenate(sampled)
    # Trilinear samples at continuous jittered positions, in field z/y/x order.
    grid_xyz = (xyz - lo) / (hi - lo) * grid_size - 0.5
    human_weights = np.stack([map_coordinates(v, grid_xyz[:, ::-1].T, order=1, mode='nearest')
                              for v in volumes[:24]], axis=-1).astype(np.float64)
    # Stable argsort makes equal weights use the official channel ordering.
    bone_indices = np.argsort(-human_weights, axis=-1, kind='stable')[:, :4]
    bone_weights = np.take_along_axis(human_weights, bone_indices, axis=1).clip(min=0)
    total = bone_weights.sum(1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError('A sampled point has no human skin-weight support')
    bone_weights /= total
    u = ((xyz - j[0]) / height).astype(np.float32)
    if len(u) < 4:
        raise ValueError('At least four points required for standalone 3-neighbor KNN')
    distances, _ = cKDTree(u).query(u, k=4, workers=1)
    knn_dist2 = np.square(distances[:, 1:]).mean(1).clip(min=1e-7).astype(np.float32)
    normalized_box = (np.stack((lo, hi)) - j[0]) / height
    return {'u': u, 'bone_indices': bone_indices, 'bone_weights': bone_weights.astype(np.float32),
            'source': np.concatenate(source), 'knn_dist2': knn_dist2,
            'rgb': np.zeros((num_points, 3), dtype=np.float32),
            'canonical_extent': float(np.linalg.norm(normalized_box[1] - normalized_box[0]) / 2),
            'metadata': {'schema': 'v10_joint_only_seed_v1', 'seed': int(seed),
                         'grid_size': int(grid_size), 'num_points': int(num_points),
                         'quota': quota.tolist(), 'H': float(height), 'N': normalizer.tolist(),
                         'canonical_joints': j.tolist(), 'joint_indices': SMPL_JOINT_IDX.copy(),
                         'human_parents': SMPL_PARENT.copy(), 'bbox_original': np.stack((lo, hi)).tolist(),
                         'bbox_u': normalized_box.tolist(), 'BONE_STDS_original': BONE_STDS.tolist(),
                         'HEAD_STDS_original': HEAD_STDS.tolist(), 'JOINT_STDS_original': JOINT_STDS.tolist(),
                         'helper': 'HOSNeRF-ca169704-human24-voxel_centers-robust_alignment-v1',
                         'knn': 'support-u-only; mean squared 3 nearest other points; scipy.cKDTree',
                         'weak_support_not_surface': True}}


def initialize_seed_colors(seed_cache, frames, train_frame_ids):
    """Use at most 16 uniform training frames, valid FG bilinear RGB medians.

    Each frame dict supplies frame_id, bone_transforms[24,4,4], K[3,3],
    w2c[4,4], raw rgb[H,W,3] and raw mask[H,W]. This helper rejects non-training
    frame IDs before reading their arrays. It returns a copy of the cache.
    """
    from scipy.ndimage import map_coordinates
    frames = sorted(frames, key=lambda f: int(f['frame_id']))
    legal = {int(f) for f in train_frame_ids}
    if not frames or any(int(f['frame_id']) not in legal for f in frames):
        raise ValueError('Color initialization requires only declared training frames')
    choose = np.unique(np.round(np.linspace(0, len(frames) - 1, min(16, len(frames)))).astype(int))
    frames = [frames[i] for i in choose]
    u = np.asarray(seed_cache['u'])
    indices, weights = np.asarray(seed_cache['bone_indices']), np.asarray(seed_cache['bone_weights'])
    observations = np.full((len(frames), len(u), 3), np.nan, dtype=np.float32)
    fg_sum, fg_count, per_frame = np.zeros(3, dtype=np.float64), 0, []
    for slot, frame in enumerate(frames):
        rgb, mask = np.asarray(frame['rgb']), np.asarray(frame['mask'])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or mask.shape != rgb.shape[:2]:
            raise ValueError('Expected RGB HWC and mask HW')
        rgb = rgb.astype(np.float64) / (255 if np.issubdtype(rgb.dtype, np.integer) else 1)
        fg = mask if mask.dtype == np.bool_ else (mask > 0 if np.all((mask == 0) | (mask == 1)) else mask >= 128)
        fg_sum += rgb[fg].sum(0)
        fg_count += int(fg.sum())
        bones = np.asarray(frame['bone_transforms'], dtype=np.float64)
        affine = (bones[indices] * weights[:, :, None, None]).sum(1)
        xyz = np.einsum('nij,nj->ni', affine[:, :3, :3], u) + affine[:, :3, 3]
        w2c, intrinsic = np.asarray(frame['w2c']), np.asarray(frame['K'])
        camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        projected = camera @ intrinsic.T
        xy = projected[:, :2] / projected[:, 2:3].clip(min=1e-12)
        height, width = rgb.shape[:2]
        valid = (camera[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] <= width - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= height - 1)
        selected = np.flatnonzero(valid)
        if len(selected):
            nearest = np.rint(xy[selected]).astype(int)
            selected = selected[fg[nearest[:, 1], nearest[:, 0]]]
            for color in range(3):
                observations[slot, selected, color] = map_coordinates(rgb[:, :, color], xy[selected, ::-1].T,
                                                                      order=1, mode='nearest')
        per_frame.append({'frame_id': int(frame['frame_id']), 'valid_projected_fg_points': int(len(selected)),
                          'fg_pixels': int(fg.sum())})
    if fg_count == 0:
        raise ValueError('All color source training masks are empty')
    have = np.isfinite(observations[:, :, 0]).any(0)
    colors = np.repeat((fg_sum / fg_count)[None], len(u), axis=0).astype(np.float32)
    colors[have] = np.nanmedian(observations[:, have], axis=0)
    result = copy.deepcopy(seed_cache)
    result['rgb'] = colors
    result['metadata']['color_sources'] = per_frame
    result['metadata']['color_fallback_rgb'] = (fg_sum / fg_count).tolist()
    result['metadata']['color_points_without_observation'] = int((~have).sum())
    result['metadata']['color_rule'] = 'training uniform <=16; nearest FG>=128; bilinear RGB; channel median; FG mean fallback; no visibility oracle'
    return result


def quaternion_rotation(q):
    """wxyz rotation with identity fallback for a numerically zero quaternion."""
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    q = torch.where(norm > 1e-12, q / norm.clamp_min(1e-12), identity)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


def pack_covariance(covariance):
    return covariance[..., (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]


class SupportResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(36, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 10))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    @staticmethod
    def encode(u, time):
        t = torch.as_tensor(time, dtype=u.dtype, device=u.device)
        if t.numel() != 1:
            raise ValueError('Support forward expects one scene time per raster')
        inputs = torch.cat((u, (2 * t.reshape(1, 1) - 1).expand(len(u), 1)), -1)
        frequencies = inputs.new_tensor([1, 2, 4, 8]) * math.pi
        angles = inputs[..., None] * frequencies
        return torch.cat((inputs, angles.sin().flatten(1), angles.cos().flatten(1)), -1)

    def forward(self, u, time):
        return self.layers(self.encode(u, time))


class PoseSupportGaussians(nn.Module):
    POINT_GROUPS = {'xyz': '_xyz', 'f_dc': '_features_dc', 'f_rest': '_features_rest',
                    'opacity': '_opacity', 'scaling': '_scaling', 'rotation': '_rotation'}

    def __init__(self, seed_cache, scene_extent, scale_bound, max_points=60000, device=None):
        super().__init__()
        device = device or 'cpu'
        f = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
        u, rgb = f(seed_cache['u']), f(seed_cache['rgb'])
        count = len(u)
        if count > max_points or scene_extent <= 0 or scale_bound <= 0:
            raise ValueError('Invalid support cap/scene extent/scale bound')
        self._xyz = nn.Parameter(u.clone())
        self._features_dc = nn.Parameter(((rgb - 0.5) / SH_C0)[:, None, :].clone())
        self._features_rest = nn.Parameter(u.new_zeros((count, 15, 3)))
        self._scaling = nn.Parameter(f(seed_cache['knn_dist2']).clamp_min(1e-7).sqrt().log()[:, None].expand(-1, 3).clone())
        rotation = u.new_zeros((count, 4))
        rotation[:, 0] = 1
        self._rotation = nn.Parameter(rotation)
        self._opacity = nn.Parameter(u.new_full((count, 1), math.log(0.01 / 0.99)))
        self.residual = SupportResidual().to(device)
        self.register_buffer('bone_indices', torch.as_tensor(seed_cache['bone_indices'], dtype=torch.long, device=device))
        self.register_buffer('bone_weights', f(seed_cache['bone_weights']).clone())
        self.register_buffer('point_id', torch.arange(count, dtype=torch.long, device=device))
        self.register_buffer('parent_id', torch.full((count,), -1, dtype=torch.long, device=device))
        self.register_buffer('source', torch.as_tensor(seed_cache.get('source', np.zeros(count)), dtype=torch.long, device=device))
        self.register_buffer('gradient_accum', u.new_zeros(count))
        self.register_buffer('visible_count', u.new_zeros(count))
        self.register_buffer('max_radii2D', u.new_zeros(count))
        self.register_buffer('transform_norm_min', u.new_tensor(float('inf')))
        self.register_buffer('transform_norm_max', u.new_tensor(0.))
        self.scene_extent, self.scale_bound = float(scene_extent), float(scale_bound)
        self.canonical_extent = float(seed_cache['canonical_extent'])
        self.max_points, self.active_sh_degree, self.next_point_id = int(max_points), 0, count
        self.percent_dense, self.grad_threshold, self.opacity_threshold = 0.01, 0.0002, 0.005
        self.seed_metadata = copy.deepcopy(seed_cache.get('metadata', {}))
        self.validate()

    def __len__(self):
        return len(self._xyz)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def u(self):
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
        count = len(self)
        for attr in (*self.POINT_GROUPS.values(), 'bone_indices', 'bone_weights', 'point_id',
                     'parent_id', 'source', 'gradient_accum', 'visible_count', 'max_radii2D'):
            if len(getattr(self, attr)) != count:
                raise ValueError(f'Misaligned point state: {attr}')
        if self.bone_indices.shape != (count, 4) or self.bone_weights.shape != (count, 4):
            raise ValueError('Exactly four fixed bone bindings required')
        if count and ((self.bone_indices < 0).any() or (self.bone_indices >= 24).any() or
                      (self.bone_weights < 0).any() or not torch.allclose(self.bone_weights.sum(1), self._xyz.new_ones(count), atol=1e-5)):
            raise ValueError('Invalid skin weights')
        if count > self.max_points or len(torch.unique(self.point_id)) != count:
            raise ValueError('Support cap/unique point IDs violated')

    def forward(self, bone_transforms, time):
        b = torch.as_tensor(bone_transforms, dtype=self._xyz.dtype, device=self._xyz.device)
        if b.shape != (24, 4, 4) or not torch.isfinite(b).all():
            raise ValueError('Need finite forward u-to-world bone transforms [24,4,4]')
        # Singular-value bound works for blended affine mappings without
        # replacing the blend with an invented rigid rotation.
        m = torch.linalg.matrix_norm(b[:, :3, :3], ord=2).max().clamp_min(1e-12)
        if self.training:
            with torch.no_grad():
                self.transform_norm_min.copy_(torch.minimum(self.transform_norm_min, m.detach()))
                self.transform_norm_max.copy_(torch.maximum(self.transform_norm_max, m.detach()))
        affine = (b[self.bone_indices] * self.bone_weights[:, :, None, None]).sum(1)
        delta = self.residual(self._xyz, time)
        u = self._xyz + delta[:, :3]
        linear = affine[:, :3, :3]
        means = torch.einsum('nij,nj->ni', linear, u) + affine[:, :3, 3]
        log_scale = self._scaling + delta[:, 3:6]
        log_scale = torch.clamp(log_scale, max=torch.log(log_scale.new_tensor(self.scale_bound) / m))
        rotation = quaternion_rotation(self._rotation + delta[:, 6:10])
        canonical_cov = (rotation * torch.exp(2 * log_scale)[:, None, :]) @ rotation.transpose(1, 2)
        covariance = linear @ canonical_cov @ linear.transpose(1, 2)
        epsilon = (1e-5 * self.scene_extent) ** 2
        covariance = covariance + torch.eye(3, dtype=covariance.dtype, device=covariance.device)[None] * epsilon
        return {'means3D': means, 'cov3D_precomp': pack_covariance(covariance),
                'shs': self.get_features, 'opacities': self.get_opacity,
                'covariance': covariance, 'transform_norm': m}

    def optimizer_groups(self):
        rates = {'xyz': 1.6e-4, 'f_dc': 0.0025, 'f_rest': 0.0025 / 20,
                 'opacity': 0.05, 'scaling': 0.005, 'rotation': 0.001}
        groups = [{'name': 'support.' + key, 'params': [getattr(self, attr)], 'lr': rates[key]}
                  for key, attr in self.POINT_GROUPS.items()]
        groups.append({'name': 'support.residual', 'params': list(self.residual.parameters()), 'lr': 1.6e-4})
        return groups

    @staticmethod
    def update_learning_rates(optimizer, update_number, total_updates=20000):
        position = max(0., min(1., (int(update_number) - 1) / (total_updates - 1)))
        ends = {'support.xyz': (1.6e-4, 1.6e-6), 'support.residual': (1.6e-4, 1.6e-5)}
        for group in optimizer.param_groups:
            if group.get('name') in ends:
                initial, final = ends[group['name']]
                group['lr'] = math.exp(math.log(initial) * (1 - position) + math.log(final) * position)

    def set_active_sh_after_update(self, completed_updates):
        self.active_sh_degree = min(3, int(completed_updates) // 1000)

    @torch.no_grad()
    def accumulate_density(self, screen_grad, visible, radii, batch_size):
        """Call once per view, after backward, with that view's retained grad."""
        if screen_grad is None:
            raise ValueError('Missing per-view screen-space gradient')
        visible = torch.as_tensor(visible, dtype=torch.bool, device=self._xyz.device)
        if screen_grad.shape[0] != len(self) or visible.shape != (len(self),) or len(radii) != len(self):
            raise ValueError('Screen statistics must be sliced to support rows')
        score = torch.linalg.vector_norm(screen_grad[:, :2] * int(batch_size), dim=-1)
        if not torch.isfinite(score[visible]).all():
            raise ValueError('Nonfinite visible screen gradients')
        self.gradient_accum[visible] += score[visible]
        self.visible_count[visible] += 1
        self.max_radii2D[visible] = torch.maximum(self.max_radii2D[visible], radii.detach()[visible])

    def _replace_topology(self, optimizer, keep, child_parent, child_values):
        """Map survivors' Adam rows; children get fresh IDs and zero moments."""
        groups = {g.get('name'): g for g in optimizer.param_groups}
        plans = []
        for semantic, attr in self.POINT_GROUPS.items():
            old = getattr(self, attr)
            group = groups.get('support.' + semantic)
            if group is None or len(group['params']) != 1 or group['params'][0] is not old:
                raise ValueError(f'Optimizer is not mapped by support.{semantic}')
            extension = child_values.get(semantic, old.detach()[child_parent])
            value = torch.cat((old.detach()[keep], extension), 0)
            new = nn.Parameter(value.clone())
            old_state = optimizer.state.get(old)
            new_state = None
            if old_state is not None:
                new_state = {}
                for key, state in old_state.items():
                    if torch.is_tensor(state) and state.shape == old.shape:
                        new_state[key] = torch.cat((state[keep], torch.zeros_like(extension)), 0)
                    else:
                        new_state[key] = state.clone() if torch.is_tensor(state) else copy.deepcopy(state)
            plans.append((attr, old, new, group, new_state))
        # Validate every named group before any mutation.
        for attr, old, new, group, state in plans:
            optimizer.state.pop(old, None)
            group['params'][0] = new
            setattr(self, attr, new)
            if state is not None:
                optimizer.state[new] = state
        children = len(child_parent)
        ids = torch.arange(self.next_point_id, self.next_point_id + children,
                           dtype=torch.long, device=self._xyz.device)
        old_ids = self.point_id
        self.point_id = torch.cat((old_ids[keep], ids))
        self.parent_id = torch.cat((self.parent_id[keep], old_ids[child_parent]))
        for key in ('bone_indices', 'bone_weights', 'source'):
            old = getattr(self, key)
            setattr(self, key, torch.cat((old[keep], old[child_parent]), 0))
        self.next_point_id += children
        for key in ('gradient_accum', 'visible_count', 'max_radii2D'):
            setattr(self, key, self._xyz.new_zeros(len(self)))
        self.validate()

    @torch.no_grad()
    def after_adam_topology(self, optimizer, completed_updates, generator=None):
        """Fixed 500..8000/100 event; must be called after the single Adam step."""
        k = int(completed_updates)
        if not (500 <= k <= 8000 and k % 100 == 0):
            return None
        before = len(self)
        score = self.gradient_accum / self.visible_count.clamp_min(1)
        prune = (self.get_opacity[:, 0] < self.opacity_threshold) if k > 1000 else torch.zeros(before, device=self._xyz.device, dtype=torch.bool)
        candidates = torch.nonzero((score >= self.grad_threshold) & (self.visible_count > 0) & ~prune, as_tuple=False).flatten()
        # Stable ID is the tie-breaker, independent of a previous row ordering.
        candidates = candidates[torch.argsort(self.point_id[candidates], stable=True)]
        candidates = candidates[torch.argsort(score[candidates], descending=True, stable=True)]
        capacity = max(0, self.max_points - (before - int(prune.sum())))
        selected = candidates[:capacity]  # every clone or two-child split adds net one
        large = self._scaling.detach().exp().amax(1) > self.percent_dense * self.canonical_extent
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
                repeated = split.repeat_interleave(2)
                noise = torch.randn((len(repeated), 3), dtype=xyz.dtype, device=xyz.device, generator=generator)
                noise *= self._scaling.detach()[repeated].exp()
                rotation = quaternion_rotation(self._rotation.detach()[repeated])
                xyz[len(clone):] += torch.einsum('nij,nj->ni', rotation, noise)
                scales[len(clone):] -= math.log(1.6)
            child_values = {'xyz': xyz, 'scaling': scales}
        if len(child_parent) or prune.any():
            self._replace_topology(optimizer, keep, child_parent, child_values)
        else:
            self.gradient_accum.zero_()
            self.visible_count.zero_()
            self.max_radii2D.zero_()
        return {'before': before, 'after': len(self),
                'opacity_pruned': int(prune.sum()), 'clone_parents': len(clone), 'split_parents': len(split),
                'new_children': len(child_parent), 'cap': self.max_points}

    def get_extra_state(self):
        return {'schema': 'v10_support_v1', 'scene_extent': self.scene_extent,
                'scale_bound': self.scale_bound, 'canonical_extent': self.canonical_extent,
                'max_points': self.max_points, 'active_sh_degree': self.active_sh_degree,
                'next_point_id': self.next_point_id, 'percent_dense': self.percent_dense,
                'grad_threshold': self.grad_threshold, 'opacity_threshold': self.opacity_threshold,
                'seed_metadata': copy.deepcopy(self.seed_metadata)}

    def set_extra_state(self, state):
        if state['schema'] != 'v10_support_v1':
            raise ValueError('Unknown support checkpoint schema')
        for key, value in state.items():
            if key != 'schema':
                setattr(self, key, copy.deepcopy(value))

    def checkpoint_payload(self):
        """Shared Adam, sample/RNG state are saved by the enclosing trainer."""
        return {'schema': 'v10_support_v1', 'state_dict': self.state_dict()}

    @classmethod
    def from_checkpoint(cls, payload, device=None):
        if payload['schema'] != 'v10_support_v1':
            raise ValueError('Unknown support checkpoint schema')
        state = payload['state_dict']
        extra = state['_extra_state']
        u = state['_xyz']
        cache = {'u': u, 'rgb': u.new_zeros((len(u), 3)),
                 'knn_dist2': u.new_ones(len(u)), 'bone_indices': state['bone_indices'],
                 'bone_weights': state['bone_weights'], 'source': state['source'],
                 'canonical_extent': extra['canonical_extent'], 'metadata': extra['seed_metadata']}
        instance = cls(cache, extra['scene_extent'], extra['scale_bound'], extra['max_points'], device=device or u.device)
        instance.load_state_dict(state)
        instance.validate()
        return instance
