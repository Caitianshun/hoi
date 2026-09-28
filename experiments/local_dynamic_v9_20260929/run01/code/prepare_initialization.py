"""One full-pool KNN initialization, immutable variable seeds, shared fine0."""
from common import *
import argparse, random, subprocess
import numpy as np
import torch
from state import *
from adapter_4dgs import load_manifest
from hoi_modules.static_dynamic_gaussians import zero_residual_heads


def source_code_identity():
    names = ['common.py', 'state.py', 'prepare_initialization.py', 'render_scene.py', 'train_scene.py']
    paths = [RUN / 'code' / n for n in names]
    paths += [ROOT / 'hoi_modules' / n for n in ('static_dynamic_gaussians.py', 'local_spacetime_input.py', 'projected_motion.py')]
    paths += [UPSTREAM / n for n in ('scene/gaussian_model.py', 'scene/deformation.py', 'scene/hexplane.py', 'gaussian_renderer/__init__.py')]
    paths += [OLD / 'code/adapter_4dgs.py', OLD / 'code/train_official.py', V5 / 'code/restore_state.py', RUN / 'configs/v9.json']
    return [identity(p) for p in paths]


def make_schedule(count, updates, seed):
    rng = random.Random(seed); stack = list(range(count)); batches = []
    for _ in range(updates):
        batch = []
        for _ in range(config()['batch_size']):
            batch.append(stack.pop(rng.randrange(len(stack))))
            if not stack:
                stack = list(range(count))
        batches.append(batch)
    content = dict(batches=batches, camera_count=count, batch_size=config()['batch_size'],
                   seed=seed, rule='random pop, immediately refill on empty, including inside an odd-sized final batch')
    content['schedule_sha256'] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return content


def role_partition(pool, extent):
    xyz = pool['xyz'].astype(np.float64); labels = pool['label']
    fg = labels == 1; bg = np.flatnonzero(labels == 0)
    assert fg.sum() == 10000 and len(bg) == 90000
    lo, hi = np.quantile(xyz[fg], config()['local_domain']['quantiles'], axis=0)
    span = np.maximum(hi-lo, config()['local_domain']['minimum_extent_fraction'] * extent)
    outside = np.maximum(np.maximum(lo-xyz[bg], xyz[bg]-hi), 0) / span
    distances = np.linalg.norm(outside, axis=1)
    raw = pool['raw_selected_indices']
    hashes = np.array([hashlib.sha256(f"{config()['role_tie_seed']}:{int(raw[i])}".encode()).hexdigest() for i in bg])
    reserve = bg[np.lexsort((hashes, distances))[:config()['variable_BG_reserve']]]
    role = fg.copy(); role[reserve] = True
    vl, vh = np.quantile(xyz[role], config()['local_domain']['quantiles'], axis=0)
    center = (vl+vh)/2
    radius = np.maximum(config()['local_domain']['radius_factor']*(vh-vl),
                        config()['local_domain']['minimum_extent_fraction']*extent)
    z = (xyz[role]-center)/radius
    domain = dict(center=center.tolist(), radius=radius.tolist(), quantile_lo=vl.tolist(), quantile_hi=vh.tolist(),
                  z_quantiles_per_axis=[quantiles(z[:, i]) for i in range(3)],
                  saturation_definition='abs(tanh(z)) >= 0.99, explanatory only',
                  saturation_fraction_per_axis=(np.abs(np.tanh(z)) >= .99).mean(0).tolist(),
                  center_approx_grid_spacing={str(res):(2*radius/(res-1)).tolist() for res in [64,128,256]})
    return role, reserve, dict(FG_box_lo=lo.tolist(), FG_box_hi=hi.tolist(), distance_axis_denominator=span.tolist(),
        reserve_distances=quantiles(distances[np.isin(bg, reserve)]), local_domain=domain,
        original_labels={'BG': int((labels==0).sum()), 'FG':int(fg.sum())},
        roles={'static_core':int((~role).sum()), 'variable':int(role.sum()), 'BG_reserve':len(reserve)},
        interpretation='Capacity allowance from noisy input provenance; not semantic dynamic truth')


def prepare_scene(scene):
    sd = scene_dir(scene); sd.joinpath('protocol').mkdir(parents=True, exist_ok=True)
    ds, hidden, opt, pipe = parameters(scene, 'coarse')
    m = load_manifest(ds.source_path)
    cfg = config()['scenes'][scene]
    assert len(m['frames']) == cfg['training_frames']
    retained = read(ROOT / cfg['input_dir'] / 'evaluation_manifest.json')
    assert len(retained['frames']) == cfg['retained_frames']
    train_ids = {int(f['frame_id']) for f in m['frames']}
    assert not train_ids.intersection(int(f['frame_id']) for f in retained['frames'])
    pool = np.load(m['point_cloud']['npz_path'])
    assert len(pool['xyz']) == 100000
    assert all(int(i) in train_ids for i in np.unique(pool['source_frame_ids']))
    assert all(abs(f['time']-(int(f['frame_id'])-1)/cfg['time_denominator']) < 1e-12 for f in m['frames'])
    for f in m['frames']:
        assert sha(f['mask_path']) == f['mask_sha256']
    role, reserve, report = role_partition(pool, m['scene_extent'])
    provenance = sd / 'protocol/initial_provenance.npz'
    assert not provenance.exists()
    np.savez_compressed(provenance, **{k:pool[k] for k in pool.files}, role=role, reserve_pool_rows=reserve)
    for stage in ('coarse', 'fine'):
        seq = make_schedule(len(m['frames']), config()['schedule'][stage+'_updates'], config()['rgb_'+stage+'_seed'])
        save_json(sd/'protocol'/f'{stage}_RGB_schedule.json', seq)
    seed_all(config()['seed'])
    model = new_model(ds, hidden, 'BG', report['local_domain'])
    from utils.graphics_utils import BasicPointCloud
    xyz, rgb = pool['xyz'], pool['rgb']
    model._deformation.deformation_net.set_aabb(xyz.max(0), xyz.min(0))
    model.create_from_pcd(BasicPointCloud(xyz, rgb, np.zeros_like(xyz)), m['scene_extent'], 1)
    zero_residual_heads(model)
    full = point_attributes(model)
    variable = {n:x[role].clone() for n,x in full.items()}
    static = {n:x[~role].clone() for n,x in full.items()}
    seed_path = sd/'protocol/variable_seeds.pt'
    seed_asset = atomic_checkpoint(seed_path, dict(attributes=variable, pool_rows=np.flatnonzero(role),
        provenance=identity(provenance), attributes_identity=attrs_identity(variable)))
    meta = dict(input_manifest=identity(ds.source_path), point_pool=identity(m['point_cloud']['npz_path']),
        initial_provenance=identity(provenance), variable_seeds=seed_asset, local_domain=report['local_domain'],
        global_AABB_max_min=np.stack([xyz.max(0), xyz.min(0)]).tolist(), scale_bound=m['scene_extent'],
        scene_extent=m['scene_extent'], source_code=source_code_identity(), seed=config()['seed'],
        input_information_boundary='Original train-only RGB/mask SIFT pool; source SfM scope unknown; FG may omit stationary objects')
    set_points(model, static, torch.zeros(len(static['_xyz']), dtype=torch.bool))
    model.training_setup(opt)
    state = capture(model, scene, 'BG', 'coarse', 0, schedule(scene,'coarse'), meta)
    initial = atomic_checkpoint(sd/'protocol/coarse_initial.pt', state)
    save_json(sd/'protocol/initialization.json',dict(status='prepared', **report, metadata=meta,
        coarse_initial=initial, variable_before=attrs_identity(variable), full_KNN_called_once=True,
        global_grid_spacing={str(res):((xyz.max(0)-xyz.min(0))/(res-1)).tolist() for res in [64,128,256]}))


def compose_fine(model, scene, meta):
    seed_file = Path(meta['variable_seeds']['path'])
    assert sha(seed_file) == meta['variable_seeds']['sha256']
    seeds = torch.load(seed_file, map_location='cpu', weights_only=False)
    assert attrs_identity(seeds['attributes']) == seeds['attributes_identity']
    assert not model._deformation_table.any()
    static = point_attributes(model); active = model.active_sh_degree
    joint = {n:torch.cat((static[n], seeds['attributes'][n]),0) for n in ATTRS}
    role = torch.cat((torch.zeros(len(static['_xyz']),dtype=torch.bool),
                      torch.ones(len(seeds['attributes']['_xyz']),dtype=torch.bool)))
    set_points(model, joint, role)
    model.active_sh_degree = active
    _, _, opt, _ = parameters(scene,'fine'); model.training_setup(opt)
    zero_residual_heads(model)
    seed_all(config()['seed'])
    fine = capture(model,scene,'Q0','fine',0,schedule(scene,'fine'),meta)
    report = dict(variable_seed_identity_before=seeds['attributes_identity'],
        variable_seed_identity_after=attrs_identity({n:joint[n][-len(seeds['pool_rows']):] for n in ATTRS}),
        variable_file_hash_unchanged=True, background_points=len(static['_xyz']), variable_points=len(seeds['pool_rows']),
        fine0_points=len(role), active_SH=active, Adam_empty=not model.optimizer.state,
        density_statistics_zero=all(torch.count_nonzero(getattr(model,n))==0 for n in
            ['max_radii2D','xyz_gradient_accum','denom','_deformation_accum']))
    assert report['variable_seed_identity_before'] == report['variable_seed_identity_after']
    assert report['density_statistics_zero'] and report['Adam_empty']
    return fine, report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    torch.set_num_threads(config()['hardware']['CPU_threads'])
    prepare_scene(a.scene)
