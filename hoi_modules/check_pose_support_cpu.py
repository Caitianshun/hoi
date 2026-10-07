"""V10 CPU mechanism checks, without any CUDA/Adam budget consumption."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from hoi_modules.pose_support_gaussians import (PoseSupportGaussians, SupportResidual,
    align_bone_rotation, build_joint_only_seed_cache, initialize_seed_colors, quaternion_rotation)
from hoi_modules.region_reconstruction import reconstruction_loss, ssim11_map


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261007)
    upstream_path = Path(__file__).resolve().parents[1] / 'third_party/4DGaussians/utils/loss_utils.py'
    spec = importlib.util.spec_from_file_location('v10_official_loss', upstream_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    x, y = torch.rand(2, 3, 17, 19), torch.rand(2, 3, 17, 19)
    error = (ssim11_map(x, y).mean() - official.ssim(x, y)).abs().item()
    assert error == 0, error
    q = reconstruction_loss(x, y)
    oldq = 0.8 * official.l1_loss(x, y) + 0.2 * (1 - official.ssim(x, y))
    assert torch.allclose(q, oldq, atol=1e-7)
    for fill in (0, 255):
        assert torch.allclose(reconstruction_loss(x, y, torch.full((2, 1, 17, 19), fill), True), q)
    mask = torch.zeros((2, 1, 17, 19), dtype=torch.uint8)
    mask[:, :, 3:8, 4:10] = 255
    qr, terms = reconstruction_loss(x, y, mask, True, True)
    fg = (mask >= 128).expand(-1, 3, -1, -1)
    expect_fg = 0.8 * (x - y).abs()[fg].reshape(2, -1).mean(1) + 0.2 * (1 - ssim11_map(x, y))[fg].reshape(2, -1).mean(1)
    assert torch.allclose(qr, 0.9 * q + 0.1 * expect_fg.mean())
    assert torch.allclose(terms['q_fg'], expect_fg)
    assert np.allclose(align_bone_rotation([0, 1, 0], [0, -1, 0]) @ [0, 1, 0], [0, -1, 0])
    assert np.array_equal(align_bone_rotation([0, 1, 0], [0, 0, 0]), np.eye(3))
    count = 8
    cache = {'u': np.arange(count * 3).reshape(count, 3).astype(np.float32) / 100,
             'rgb': np.full((count, 3), .5, dtype=np.float32),
             'bone_indices': np.tile(np.array([0, 1, 2, 3]), (count, 1)),
             'bone_weights': np.tile(np.array([.4, .3, .2, .1], dtype=np.float32), (count, 1)),
             'knn_dist2': np.full(count, .000001), 'canonical_extent': 1.,
             'source': np.arange(count), 'metadata': {'cpu_probe': True}}
    bank = PoseSupportGaussians(cache, scene_extent=2., scale_bound=.1, max_points=10)
    assert SupportResidual.encode(bank.u, .2).shape == (8, 36)
    bones = torch.eye(4)[None].repeat(24, 1, 1)
    bones[:, :3, :3] *= 2
    bones[:, :3, 3] = torch.tensor([1., 2., 3.])
    output = bank(bones, .2)
    assert torch.allclose(output['means3D'], 2 * bank.u + torch.tensor([1., 2., 3.]))
    expected_cov = torch.diag(torch.tensor([4e-6 + 4e-10] * 3))[None].expand(count, -1, -1)
    assert torch.allclose(output['covariance'], expected_cov, atol=1e-10)
    rotation = quaternion_rotation(torch.tensor([[2 ** -.5, 0., 0., 2 ** -.5]]))[0]
    bones[:, :3, :3] = rotation
    with torch.no_grad(): bank._scaling[:] = torch.tensor([.001, .002, .003]).log()
    output = bank(bones, .2)
    cov = rotation @ torch.diag(torch.tensor([1e-6, 4e-6, 9e-6])) @ rotation.T + torch.eye(3) * 4e-10
    assert torch.allclose(output['covariance'], cov[None].expand(count, -1, -1), atol=1e-10)
    output['shs'].sum().backward()
    assert bank._features_rest.grad.abs().sum() == 0
    # Real Adam state creation makes moment correspondence an executable check.
    optimizer = torch.optim.Adam(bank.optimizer_groups(), eps=1e-15)
    optimizer.zero_grad(set_to_none=True)
    sum(p.sum() for group in optimizer.param_groups for p in group['params']).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before_xyz, before_moment = bank.u.detach().clone(), optimizer.state[bank.u]['exp_avg'].clone()
    # Opposite view gradients must accumulate nonzero norms, not cancel.
    visible = torch.ones(count, dtype=torch.bool)
    radii = torch.ones(count)
    grad = torch.ones((count, 3)) * .001
    bank.accumulate_density(grad, visible, radii, 2)
    bank.accumulate_density(-grad, visible, radii, 2)
    assert torch.allclose(bank.gradient_accum / bank.visible_count, torch.full((count,), 2 * 2 ** .5 * .001))
    # Keep all seeds small: tie score is broken by stable IDs 0,1. Cap=10.
    with torch.no_grad(): bank._scaling[:] = np.log(.001)
    event = bank.after_adam_topology(optimizer, 500)
    assert event['clone_parents'] == 2 and len(bank) == 10
    assert torch.equal(bank.point_id, torch.arange(10))
    assert torch.equal(bank.parent_id[-2:], torch.tensor([0, 1]))
    assert torch.equal(optimizer.state[bank.u]['exp_avg'][:8], before_moment)
    assert optimizer.state[bank.u]['exp_avg'][-2:].abs().sum() == 0
    assert torch.equal(bank.bone_indices[-2:], bank.bone_indices[:2])
    assert torch.equal(bank.u[:8], before_xyz)
    # At cap, prune two low-opacity seeds then two-parent split fills exact cap.
    with torch.no_grad():
        bank._opacity[:2] = -20
        bank._scaling[:] = np.log(.02)
        bank.gradient_accum[:] = .01
        bank.visible_count[:] = 1
    event = bank.after_adam_topology(optimizer, 1100, generator=torch.Generator().manual_seed(7))
    assert event['opacity_pruned'] == 2 and event['split_parents'] == 2 and len(bank) == 10
    assert torch.equal(bank.point_id[-4:], torch.arange(10, 14))
    assert torch.equal(bank.parent_id[-4:], torch.tensor([2, 2, 3, 3]))
    assert optimizer.state[bank.u]['exp_avg'][-4:].abs().sum() == 0
    assert not bank.gradient_accum.any() and not bank.visible_count.any()
    assert bank.after_adam_topology(optimizer, 8100) is None
    bank.set_active_sh_after_update(3000)
    restored = PoseSupportGaussians.from_checkpoint(copy.deepcopy(bank.checkpoint_payload()))
    for key, value in bank.state_dict().items():
        if torch.is_tensor(value): assert torch.equal(restored.state_dict()[key], value), key
    assert restored.next_point_id == 14 and restored.active_sh_degree == 3
    bank.update_learning_rates(optimizer, 1)
    assert abs(optimizer.param_groups[0]['lr'] - 1.6e-4) < 1e-12
    bank.update_learning_rates(optimizer, 20000)
    assert abs(optimizer.param_groups[0]['lr'] - 1.6e-6) < 1e-12
    # Full-size initialization is checked once on a synthetic skeleton, without
    # implying that any real scene's coordinate or projection validation passed.
    skeleton = np.array([[0, 0, 0], [-.1, -.1, 0], [.1, -.1, 0], [0, .1, 0],
        [-.1, -.4, 0], [.1, -.4, 0], [0, .3, 0], [-.1, -.7, 0], [.1, -.7, 0],
        [0, .5, 0], [-.1, -.75, .1], [.1, -.75, .1], [0, .6, 0], [-.1, .5, 0],
        [.1, .5, 0], [0, .75, 0], [-.2, .5, 0], [.2, .5, 0], [-.45, .5, 0],
        [.45, .5, 0], [-.65, .5, 0], [.65, .5, 0], [-.7, .5, 0], [.7, .5, 0]])
    seeds = build_joint_only_seed_cache(skeleton)
    assert seeds['u'].shape == (20000, 3)
    assert seeds['metadata']['quota'] == [834] * 8 + [833] * 16
    assert len(np.unique(seeds['u'], axis=0)) == 20000
    assert np.allclose(seeds['bone_weights'].sum(1), 1)
    assert (seeds['knn_dist2'] > 0).all()
    print(json.dumps({'status': 'passed', 'SSIM11_scalar_error': error,
        'checked': ['Q scalar/map identity', 'Qr empty/full/ordinary masks', '36-input zero-head network',
                    'similarity and rotation covariance', 'SH gradient masking', 'per-view gradient norms',
                    'clone/split/prune IDs/skin/Adam rows', 'exact cap', 'checkpoint tensors', 'LR endpoints',
                    '20k CPU seed quotas/jitter/standalone KNN', 'zero/antiparallel bone rotation'],
        'actual_GPU_Adam_updates': 0}))


if __name__ == '__main__':
    main()
