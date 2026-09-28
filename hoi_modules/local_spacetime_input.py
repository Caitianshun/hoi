"""Frozen local field domain, full temporal query and four low-frequency inputs.

Only feature coordinates change; all Gaussian attributes and residual outputs
retain their world units. The normalized query bypasses AABB normalization.
"""
import types
import torch
from torch import nn


def query_normalized(field, u, tau):
    from scene.hexplane import interpolate_ms_features
    if not len(u):
        return u.new_empty((0, field.feat_dim))
    return interpolate_ms_features(
        torch.cat((u, tau), dim=-1), field.grids,
        field.grid_config[0]['grid_dimensions'], field.concat_features, None
    ).reshape(len(u), field.feat_dim)


def local_query_time(self, rays_pts_emb, scales_emb, rotations_emb,
                     time_feature, time_emb):
    z = (rays_pts_emb[:, :3] - self.local_center) / self.local_radius
    tau = 2 * time_emb[:, :1] - 1
    grid_feature = query_normalized(self.grid, -torch.tanh(z), tau)
    low_frequency = torch.cat((z / (1 + z.abs()), tau), dim=-1)
    return self.feature_out(torch.cat((grid_feature, low_frequency), dim=-1))


def install_local_input(model, domain):
    net = model._deformation.deformation_net
    assert not net.no_grid and net.grid_pe == 0
    old = net.feature_out[0]
    assert old.in_features == net.grid.feat_dim == 48
    # Expanding a layer must not consume any training or sampling RNG stream.
    devices = [old.weight.device.index] if old.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        new = nn.Linear(old.in_features + 4, old.out_features,
                        device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_features].copy_(old.weight)
        new.bias.copy_(old.bias)
    net.feature_out[0] = new
    net.register_buffer('local_center', old.weight.new_tensor(domain['center']))
    net.register_buffer('local_radius', old.weight.new_tensor(domain['radius']))
    assert (net.local_radius > 0).all()
    net.query_time = types.MethodType(local_query_time, net)
    return new.weight.numel() - old.weight.numel()
