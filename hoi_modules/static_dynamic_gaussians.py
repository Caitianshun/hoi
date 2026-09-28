"""Static bypass and differentiable union for the fixed Wu4DGS representation.

The existing boolean deformation table has one meaning: time variation allowed.
It is inherited by the upstream clone/split/prune operations, never estimated
from displacement. Static canonical attributes remain trainable by RGB.
"""
import torch


def canonical_attributes(model):
    return (model._xyz, model._scaling, model._rotation,
            model._opacity, model.get_features)


def deform_attributes(model, time, mode):
    attrs = canonical_attributes(model)
    if mode == 'BG':
        return attrs
    assert mode in ('Q0', 'S', 'SL')
    n = len(attrs[0])
    if not n:
        return attrs
    if not torch.is_tensor(time):
        time = attrs[0].new_full((n, 1), float(time))
    if mode == 'Q0':
        return model._deformation(*attrs, time)
    role = model._deformation_table
    assert role.dtype == torch.bool and role.shape == (n,)
    idx = torch.nonzero(role, as_tuple=False).flatten()
    if not len(idx):
        return attrs
    changed = model._deformation(*(x.index_select(0, idx) for x in attrs),
                                 time.index_select(0, idx))
    # Out-of-place index_copy carries gradients both to untouched canonical
    # points and through the changed subset. Rasterization occurs only once.
    return tuple(a.index_copy(0, idx, b) for a, b in zip(attrs, changed))


def forbid_role_reclassification(self, *args, **kwargs):
    raise RuntimeError('V9 roles are fixed ancestry; automatic reclassification is disabled')


def validate_point_state(model):
    n = len(model._xyz)
    for name in ('_features_dc', '_features_rest', '_scaling', '_rotation',
                 '_opacity', '_deformation_table', 'max_radii2D',
                 'xyz_gradient_accum', 'denom', '_deformation_accum'):
        assert len(getattr(model, name)) == n, name
    assert model._deformation_table.dtype == torch.bool


def zero_residual_heads(model):
    net = model._deformation.deformation_net
    assert not net.args.apply_rotation and net.args.no_do and net.args.no_dshs
    with torch.no_grad():
        for name in ('pos_deform', 'scales_deform', 'rotations_deform'):
            getattr(net, name)[-1].weight.zero_()
            getattr(net, name)[-1].bias.zero_()
