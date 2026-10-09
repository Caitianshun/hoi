"""CPU unit tests for hoi_modules.object_anchor_gaussians (no rasteriser, no CUDA)."""
import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import v11_common  # noqa: F401  (adds project root to sys.path)
from hoi_modules.object_anchor_gaussians import (ObjectAnchorGaussians, matrix_to_quat, quat_to_matrix, rigid,
                                                 rigid_inverse, se3_exp)

torch.manual_seed(0)
rng = np.random.default_rng(0)


def random_rigid():
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=int(rng.integers(1e6))).as_matrix()
    T[:3, 3] = rng.normal(size=3)
    return T


def test_se3_and_quaternions():
    for _ in range(50):
        omega, rho = rng.normal(size=3) * rng.uniform(0, 3), rng.normal(size=3)
        T = se3_exp(torch.tensor(np.concatenate([rho, omega]), dtype=torch.float64)).numpy()
        assert np.allclose(T[:3, :3], Rotation.from_rotvec(omega).as_matrix(), atol=1e-9)
        assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    small = se3_exp(torch.tensor([0.1, 0.2, 0.3, 1e-7, 0, 0], dtype=torch.float64)).numpy()
    assert np.allclose(small[:3, 3], [0.1, 0.2, 0.3], atol=1e-6)
    for _ in range(50):
        R = torch.tensor(Rotation.random(random_state=int(rng.integers(1e6))).as_matrix())
        assert torch.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-6)
    T = torch.tensor(random_rigid())
    assert torch.allclose(rigid_inverse(T) @ T, torch.eye(4, dtype=T.dtype), atol=1e-9)


def make_bank(mode):
    n_frames = 60
    segments = [dict(start=0, end=19, anchor="world"), dict(start=20, end=44, anchor="right"), dict(start=45, end=59, anchor="world")]
    hand = {f: random_rigid() for f in range(n_frames)}
    pairs = [np.stack([np.eye(4), hand[20]]), np.stack([hand[45], np.eye(4)])]
    kf_frames = list(range(0, n_frames, 4)) + [60]
    kf = [random_rigid() for _ in kf_frames]
    motion = dict(objects=["box"], extent={"box": 1.0}, residual_knot_spacing=8, keyframe_step=4,
                  per_object={"box": dict(segments=segments, S1=random_rigid().tolist(), switch_anchors=np.asarray(pairs).tolist(),
                                          keyframe_init=np.asarray(kf).tolist())})
    seeds = dict(xyz=rng.normal(size=(200, 3)), rgb=rng.uniform(size=(200, 3)), scale=np.full(200, 0.05), object_id=np.zeros(200, int))
    bank = ObjectAnchorGaussians(seeds, motion, mode, scene_extent=5.0, scale_bound=1.0).double()
    return bank, hand, kf_frames, kf


def test_anchor_continuity_and_gradients():
    bank, hand, _, _ = make_bank("anchor")
    with torch.no_grad():
        for c in bank.xi.values():
            c.normal_(0, 0.05)
    anchors = lambda f: {"right": hand[f], "left": np.eye(4)}
    for tau in (20, 45):
        before = bank.anchor_pose("box", tau - 1, anchors(tau - 1))
        # The pose just before the switch, extrapolated by one frame inside the old segment, must equal
        # the new segment's pose at tau: evaluate the old-segment formula at tau with the old anchor.
        k = bank.segment_index("box", tau - 1)
        S = rigid(quat_to_matrix(bank.S1["box"][3:7]), bank.S1["box"][:3])
        sw = bank.switch__box
        for j in range(k):
            S = rigid_inverse(sw[j, 1]) @ sw[j, 0] @ S @ bank._delta("box", j, bank.segments["box"][j + 1]["start"])
        A_old = torch.eye(4, dtype=S.dtype) if bank.segments["box"][k]["anchor"] == "world" else torch.tensor(hand[tau])
        old_at_tau = A_old @ S @ bank._delta("box", k, tau)
        new_at_tau = bank.anchor_pose("box", tau, anchors(tau))
        assert torch.allclose(old_at_tau, new_at_tau, atol=1e-9), (tau, (old_at_tau - new_at_tau).abs().max())
        assert torch.isfinite(before).all()
    out = bank(30, anchors(30))
    loss = out["means3D"].square().mean() + out["cov3D_precomp"].sum() + out["opacities"].sum()
    loss.backward()
    assert bank.S1["box"].grad is not None and bank.S1["box"].grad.abs().sum() > 0
    assert bank._xyz.grad.abs().sum() > 0
    assert any(c.grad is not None and c.grad.abs().sum() > 0 for c in bank.xi.values())
    # rest segment after the put-down is exactly static
    p50, p58 = bank.anchor_pose("box", 50, anchors(50)), bank.anchor_pose("box", 58, anchors(58))
    assert torch.allclose(p50, p58)


def test_world_keyframes():
    bank, _, kf_frames, kf = make_bank("world")
    for f, T in zip(kf_frames[:-1], kf[:-1]):
        P = bank.world_pose("box", f).detach().numpy()
        assert np.allclose(P[:3, 3], T[:3, 3], atol=1e-6) and np.allclose(P[:3, :3], T[:3, :3], atol=1e-5), f
    mid = bank.world_pose("box", 6).detach().numpy()
    assert np.allclose(mid[:3, :3] @ mid[:3, :3].T, np.eye(3), atol=1e-6)


def test_densify_and_optimizer_mapping():
    bank, hand, _, _ = make_bank("anchor")
    bank = bank.float()
    lr = dict(xyz=1e-3, f_dc=1e-3, f_rest=1e-4, opacity=1e-2, scaling=1e-3, rotation=1e-3, motion=1e-3)
    opt = torch.optim.Adam(bank.optimizer_groups(lr), lr=0.0, eps=1e-15)
    out = bank(10, {"right": hand[10], "left": np.eye(4)})
    (out["means3D"].sum() + out["opacities"].sum()).backward()
    opt.step()
    n = len(bank)
    bank.gradient_accum[:20] = 1.0
    bank.visible_count[:20] = 1.0
    with torch.no_grad():
        bank._scaling[:10] = math.log(0.5)    # large -> split
        bank._scaling[10:20] = math.log(0.001)  # small -> clone
    event = bank.after_adam_topology(opt, 200)
    assert event["new_children"] == 30 and len(bank) == n + 20, event
    for g in opt.param_groups:
        if g["name"].startswith("object.") and g["name"] != "object.motion":
            p = g["params"][0]
            assert p is getattr(bank, bank.POINT_GROUPS[g["name"].split(".", 1)[1]])
            assert opt.state[p]["exp_avg"].shape == p.shape
    payload = bank.checkpoint_payload()
    clone = ObjectAnchorGaussians.from_checkpoint(payload)
    for a, b in zip(bank.state_dict().values(), clone.state_dict().values()):
        if torch.is_tensor(a):
            assert torch.equal(a, b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
