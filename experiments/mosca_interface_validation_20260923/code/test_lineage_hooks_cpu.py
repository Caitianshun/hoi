"""CPU equivalence and row-order tests. Never loads CUDA/rendering extensions.

Actual upstream clone/split/prune/postprocess and optimizer mutation functions
are AST-extracted verbatim. Controlled adapters exercise new-root/node growth.
This is an instrumentation test, not a replacement for real training validation.
"""
import ast
import hashlib
import json
from pathlib import Path
import random
import tempfile
import types

import numpy as np
import torch
from torch import nn
from lineage_hooks import install_lineage_hooks

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'experiments/mosca_validation_20260923/code/MoSca/lib_mosca'


def load_methods(path, names):
    tree = ast.parse(path.read_text())
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
               for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    return ast.Module(body=methods, type_ignores=[])


helper_globals = dict(torch=torch, nn=nn, np=np)
exec(compile((SOURCE / 'gs_utils/gs_optim_helper.py').read_text(), 'upstream_optimizer', 'exec'), helper_globals)
method_globals = dict(helper_globals)
method_globals['quaternion_to_matrix'] = lambda q: torch.eye(3).expand(len(q), 3, 3)
names = ['_densification_postprocess', '_prune_points', '_densify_and_clone', '_densify_and_split']
exec(compile(load_methods(SOURCE / 'dynamic_gs.py', names), 'upstream_dynamic_methods', 'exec'), method_globals)


class Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_node_xyz', torch.zeros(2, 3, 3))

    @property
    def M(self):
        return self._node_xyz.shape[1]

    def append_nodes_traj(self, optimizer, xyz):
        self._node_xyz = torch.cat([self._node_xyz, xyz], 1)

    def remove_nodes(self, optimizer, node_prune_mask):
        self._node_xyz = self._node_xyz[:, ~node_prune_mask]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scf = Node()
        self._xyz = nn.Parameter(torch.arange(18).float().reshape(6, 3) / 100)
        self._rotation = nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(6, 1))
        self._scaling = nn.Parameter(torch.tensor([.1, .9, .2, .8, .1, .9])[:, None].repeat(1, 3))
        self._opacity = nn.Parameter(torch.ones(6, 1) * .1)
        self._features_dc = nn.Parameter(torch.ones(6, 3))
        self._features_rest = nn.Parameter(torch.zeros(6, 3))
        self._skinning_weight = nn.Parameter(torch.zeros(6, 2))
        self._dynamic_logit = nn.Parameter(torch.ones(6, 1))
        for name in ['xyz_gradient_accum', 'xyz_gradient_denom', 'max_radii2D', 'corr_gradient_accum', 'corr_gradient_denom']:
            self.register_buffer(name, torch.zeros(6))
        self.register_buffer('attach_ind', torch.arange(6) % 3)
        self.register_buffer('ref_time', torch.arange(6) % 2)
        self.op_update_exclude = []
        self.max_scale, self.min_scale = 2., 0.
        self.s_inv_act = lambda x: x

    @property
    def N(self):
        return len(self._xyz)

    @property
    def device(self):
        return self._xyz.device

    @property
    def get_s(self):
        return self._scaling

    def copy_args(self, rows):
        attrs = ['_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest', '_skinning_weight', '_dynamic_logit']
        return [getattr(self, n)[rows].detach().clone() for n in attrs]

    def append_new_gs(self, optimizer):
        values = self.copy_args(torch.tensor([0, 1]))
        self._densification_postprocess(optimizer, *values)
        self.attach_ind = torch.cat([self.attach_ind, torch.tensor([0, 1])])
        self.ref_time = torch.cat([self.ref_time, torch.tensor([0, 1])])

    def gradient_based_node_densification(self, optimizer):
        candidate_mask = torch.arange(self.N) == 2
        resample_ind = torch.tensor([0])
        self.scf.append_nodes_traj(optimizer, torch.zeros(2, 1, 3))
        new_gs_ind = torch.tensor([2, 0, 2])
        values = self.copy_args(new_gs_ind)
        self._densification_postprocess(optimizer, *values)
        self.attach_ind = torch.cat([self.attach_ind, torch.full((3,), self.scf.M - 1)])
        self.ref_time = torch.cat([self.ref_time, self.ref_time[new_gs_ind]])


for name in names:
    setattr(Model, name, method_globals[name])


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def photometric_fit(photo, d_model, optimizer, total_steps):
    for step in photo.tqdm(range(total_steps)):
        # An unrelated nested range of the same length must NOT be hooked.
        def nested():
            return list(photo.tqdm(range(total_steps)))
        assert len(nested()) == total_steps
        if step == 0:
            d_model._densify_and_clone(optimizer, torch.ones(d_model.N), .5, .5)
        elif step == 1:
            d_model._densify_and_split(optimizer, torch.ones(d_model.N), .5, .5, N=2)
        elif step == 2:
            d_model._prune_points(optimizer, torch.arange(d_model.N) % 4 == 0)
        elif step == 3:
            d_model.append_new_gs(optimizer)
        elif step == 4:
            d_model.gradient_based_node_densification(optimizer)
        elif step == 5:
            mask = torch.tensor([False, True, False, False])
            d_model.scf.remove_nodes(optimizer, mask)
            # Mirror caller's attachment row conversion after node deletion.
            d_model.attach_ind = torch.where(d_model.attach_ind == 1, 0, d_model.attach_ind)
            d_model.attach_ind = d_model.attach_ind - (d_model.attach_ind > 1).long()
        # Leave actual gradients present at each snapshot.
        for p in d_model.parameters():
            p.grad = torch.ones_like(p) * .031


def run(directory, hooked):
    torch.manual_seed(478); np.random.seed(341); random.seed(903)
    model = Model()
    static_model = nn.Linear(3, 2)
    for p in static_model.parameters():
        p.grad = torch.ones_like(p) * .125
    pairs = [('xyz', '_xyz'), ('rotation', '_rotation'), ('scaling', '_scaling'), ('opacity', '_opacity'),
             ('f_dc', '_features_dc'), ('f_rest', '_features_rest'), ('skinning_w', '_skinning_weight'), ('dyn_logit', '_dynamic_logit')]
    optimizer = torch.optim.Adam([dict(params=[getattr(model, attr)], name=name, lr=.001) for name, attr in pairs])
    # Populate Adam moments to test cat/prune with nonempty optimizer state.
    for p in model.parameters(): p.grad = torch.ones_like(p) * .01
    optimizer.step(); optimizer.zero_grad()
    photo = types.SimpleNamespace(tqdm=lambda iterable, *a, **kw: iterable)
    original_tqdm = photo.tqdm
    handle = install_lineage_hooks(photo, model, directory, static_model=static_model,
                                   checkpoints=range(7)) if hooked else None
    try:
        photometric_fit(photo, model, optimizer, 6)
    finally:
        summary = handle.close() if handle else None
    assert photo.tqdm is original_tqdm
    assert '_densification_postprocess' not in vars(model)
    if hooked:
        for snapshot in (directory/'snapshots').iterdir():
            saved = torch.load(snapshot/'static_state.pth', weights_only=False)
            equal_tree(saved, static_model.state_dict())
    return dict(state=model.state_dict(), optimizer=optimizer.state_dict(),
                static_state=static_model.state_dict(), static_gradients=[p.grad.clone() for p in static_model.parameters()],
                gradients=[p.grad.clone() for p in model.parameters()],
                torch_rng=torch.random.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(), summary=summary)


def main():
    out = Path(__file__).resolve().parents[1] / 'lineage_cpu_test'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lineage_cpu_') as temporary:
        a = run(Path(temporary) / 'plain', False)
        b = run(Path(temporary) / 'observed', True)
        for key in ['state', 'optimizer', 'gradients', 'static_state', 'static_gradients', 'torch_rng', 'python_rng']:
            equal_tree(a[key], b[key])
        assert a['numpy_rng'][0] == b['numpy_rng'][0]
        assert np.array_equal(a['numpy_rng'][1], b['numpy_rng'][1])
        assert a['numpy_rng'][2:] == b['numpy_rng'][2:]
        d = Path(temporary) / 'observed'
        events = list(sorted((d / 'events').glob('*.npz')))
        clone = np.load(events[0]); split = np.load(events[1])
        assert clone['parent_id'].tolist() == [0, 2, 4]
        assert split['parent_id'].tolist() == [1, 3, 5, 1, 3, 5]
        final = np.load(d / 'leaf_lineage.npz')
        assert len(np.unique(final['live_id'])) == len(final['live_id'])
        assert not np.isin([1, 3, 5], final['live_id']).any()
        assert final['live_id'].tolist() == [2, 4, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
        assert final['parent_id'][15:20].tolist() == [-1, -1, 6, 2, 6]
        assert final['root_id'][15:20].tolist() == [15, 16, 0, 2, 0]
        nodes = np.load(d/'node_lineage.npz')
        assert nodes['live_id'].tolist() == [0, 2, 3]
        node_event = np.load(events[5])
        assert node_event['source_leaf_id'].tolist() == [6]
        snap_ids = np.load(d/'snapshots/00006/identity.npz')
        assert np.array_equal(snap_ids['attached_node_id'], nodes['live_id'][snap_ids['attach_ind']])
        assert len(events) == 8  # clone, split append/prune, prune, new roots, node+leaf append, node prune
        assert b['summary']['status'] == 'completed'
        assert len(b['summary']['snapshots']) == 7
        result = dict(status='passed', gpu_used=False, real_upstream_methods=names,
                      append_and_node_scope='Controlled adapters with actual call-site locals; no full MoSca node topology/render execution',
                      parameter_tensors_bitwise_equal=True, optimizer_state_bitwise_equal=True,
                      gradients_bitwise_equal=True, torch_numpy_python_rng_equal=True,
                      clone_parent_order=clone['parent_id'].tolist(), split_parent_order=split['parent_id'].tolist(),
                      events=8, key_snapshots=7, nested_tqdm_not_intercepted=True,
                      static_snapshots_bitwise_equal=True, node_prune_row_identity_verified=True,
                      duplicate_leaf_parent_and_root_ids_verified=True,
                      original_methods_restored=True,
                      instrumentation_seconds=b['summary']['instrumentation_wall_seconds'],
                      source_sha256=hashlib.sha256((SOURCE/'dynamic_gs.py').read_bytes()).hexdigest(),
                      hook_sha256=hashlib.sha256(Path(__file__).with_name('lineage_hooks.py').read_bytes()).hexdigest())
        (out / 'qa.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
