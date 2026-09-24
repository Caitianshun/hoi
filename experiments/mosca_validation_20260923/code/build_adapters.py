"""Build isolated, auditable protocol adapters; never edit historical source."""
from pathlib import Path
import hashlib
import json

ROOT = Path('/home/cai_tianshun/Project/HOI')
EXP = ROOT / 'experiments/mosca_validation_20260923'
CODE = EXP / 'code'
REPO = CODE / 'MoSca'
changes = []

def save_variant(source, destination, edits):
    original = source.read_text()
    value = original
    for old, new in edits:
        if value.count(old) != 1:
            raise ValueError(f'{source}: expected one occurrence: {old[:100]!r}; got {value.count(old)}')
        value = value.replace(old, new, 1)
    destination.write_text(value)
    changes.append({'source': str(source), 'destination': str(destination),
                    'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                    'destination_sha256': hashlib.sha256(value.encode()).hexdigest(),
                    'replacements': [{'old': a, 'new': b} for a,b in edits]})

# CUDA ndc2Pix maps NDC=0 to (size-1)/2. Use an exact asymmetric
# projection matrix; avoid rounding a principal point through integer crops.
original_renderer = ROOT / 'third_party/MoSca/lib_render/gauspl_renderer_native_add3.py'
source = original_renderer.read_text()
begin = source.index('    # * Specially handle the non-centered camera')
end = source.index('    # ! 2023.10.27 Fix this bug', begin)
save_variant(original_renderer, REPO / 'lib_render/gauspl_renderer_native_add3.py', [
    (source[begin:end], '    # Exact calibrated projection, with integer pixel centres.\n'
                       '    center_handling_flag = False\n    new_W, new_H = W, H\n\n'),
    ('    full_proj_transform = (',
     '    projection_matrix[2, 0] = 2.0 * (cx + 0.5) / new_W - 1.0\n'
     '    projection_matrix[2, 1] = 2.0 * (cy + 0.5) / new_H - 1.0\n'
     '    full_proj_transform = (')])

runner_edits = [
 ('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(ROOT)!r})'),
 ("REPO = ROOT/'third_party/MoSca'", f'REPO = Path({str(REPO)!r})'),
 ("    p.add_argument('--photo-steps'", "    p.add_argument('--normalize-world', action='store_true')\n    p.add_argument('--photo-steps'"),
 ('cfg.dep_median = -1.0; cfg.iso_focal = False', 'cfg.dep_median = 1.0 if a.normalize_world else -1.0; cfg.iso_focal = False'),
 ("    OriginalOptimCFG = mr.OptimCFG", """    OriginalOptimCFG = mr.OptimCFG
    world_scale = 1.0
    if a.normalize_world:
        scale_data = Saved2D(str(ws)).load_dep('unidepth_depth', cfg.depth_boundary_th).normalize_depth(1.0)
        world_scale = float(scale_data.scale_nw)
        del scale_data
    # Give dynamic initialization and photo sampling independent RNG streams,
    # so a changed static candidate pool cannot silently change dynamic seeds.
    original_dynamic = mr.DynReconstructionSolver.get_dynamic_model
    original_fit = mr.DynReconstructionSolver.photometric_fit
    def seeded_dynamic(self, *args, **kwargs):
        mr.seed_everything(mr.SEED)
        return original_dynamic(self, *args, **kwargs)
    def recorded_fit(self, *args, **kwargs):
        from validation_utils import tensor_state_hash
        state = kwargs['d_model'].state_dict()
        torch.save(state, output/'initial_dynamic.pth')
        torch.save(kwargs['s_model'].state_dict(), output/'initial_static.pth')
        initial_record = {'dynamic_state_tensor_sha256': tensor_state_hash(state),
                          'dynamic_gaussians': int(kwargs['d_model'].N),
                          'static_gaussians': int(kwargs['s_model'].N),
                          'rng_policy': 'seed 12345 immediately before dynamic initialization and photometric fit'}
        (output/'initialization.json').write_text(json.dumps(initial_record, indent=2)+'\\n')
        mr.seed_everything(mr.SEED)
        return original_fit(self, *args, **kwargs)
    mr.DynReconstructionSolver.get_dynamic_model = seeded_dynamic
    mr.DynReconstructionSolver.photometric_fit = recorded_fit"""),
 ("                  stages={})", "                  world_scale=world_scale, exact_principal_point=True,\n                  rng_policy='isolated dynamic initialization and photo RNG seed 12345',\n                  stages={})"),
 ("'All camera learning rates are zero; estimated metric depth retained without reference scaling'", "'All camera learning rates are zero; optional single input-only world normalization is reversed for evaluation'"),
 ("        poses = torch.tensor(m['c2w'], dtype=torch.float32)[None].repeat(len(m['frame_paths']),1,1)", "        poses = torch.tensor(m['c2w'], dtype=torch.float32)[None].repeat(len(m['frame_paths']),1,1)\n        poses[:, :3, 3] *= world_scale"),
 ("            assert previous['config_sha256'] == record['config_sha256'], 'Refuse reuse with changed config'", """            prior_cfg = OmegaConf.to_container(OmegaConf.load(source/'config.yaml'))
            next_cfg = OmegaConf.to_container(cfg)
            differences = {k: [prior_cfg.get(k), next_cfg.get(k)] for k in set(prior_cfg)|set(next_cfg)
                           if prior_cfg.get(k) != next_cfg.get(k)}
            assert set(differences) <= {'gs_include_fg_in_static'}, differences
            assert abs(previous.get('world_scale',1.) - world_scale) < 1e-10
            record['allowed_photo_only_config_differences'] = differences"""),
]
save_variant(ROOT/'scripts/mosca_run_calibrated_event.py', CODE/'run_control.py', runner_edits)

export_edits = [
 ("ROOT = Path(__file__).resolve().parents[2]", f'ROOT = Path({str(ROOT)!r})'),
 ("default=ROOT / 'third_party/MoSca'", f'default=Path({str(REPO)!r})'),
 ("    if cfg.mode != 'behave_calibrated_pilot' or cfg.dep_median >= 0:\n        raise ValueError('This exporter requires the documented calibrated adapter with scale_nw=1')", """    training_record = json.loads((checkpoint_dir/'run.json').read_text())
    world_scale = float(training_record['world_scale'])
    if cfg.mode != 'behave_calibrated_pilot' or world_scale <= 0 or not training_record['exact_principal_point']:
        raise ValueError('Expected this controlled protocol adapter')"""),
 ("        camera_error = max(float((cams.T_wc(t) - c2w).abs().max()) for t in range(len(times)))", "        normalized_c2w = c2w.clone(); normalized_c2w[:3, 3] *= world_scale\n        camera_error = max(float((cams.T_wc(t) - normalized_c2w).abs().max()) for t in range(len(times)))"),
 ("        record['coordinate_checks'] = {'scale_nw': 1.0", "        from validation_utils import expose_metric_gaussians\n        expose_metric_gaussians(static, dynamic, world_scale)\n        record['world_scale'] = world_scale\n        record['global_transform_fitted_on_evaluation'] = False\n        record['exact_principal_point'] = True\n        record['coordinate_checks'] = {'scale_nw': world_scale"),
 ("'source': 'adapter config dep_median<0, identity bundle scaling, supplied cameras verified'", "'source': 'single input-derived world scale; cameras verified in normalized coordinates; Gaussian centers and scales divided by world_scale for metric rendering/export'"),
 ("node_xyz=cpu(scf._node_xyz),", "node_xyz=cpu(scf._node_xyz) / world_scale,")]
save_variant(ROOT/'scripts/diagnostics/export_mosca_event.py', CODE/'export_control.py', export_edits)

heldout_edits = [
 ("ROOT = Path(__file__).resolve().parents[2]", f'ROOT = Path({str(ROOT)!r})'),
 ("default=ROOT/'third_party/MoSca'", f'default=Path({str(REPO)!r})'),
 ("    if cfg.mode != 'behave_calibrated_pilot' or cfg.dep_median >= 0:\n        raise ValueError('Expected calibrated adapter with scale_nw=1')", "    training_record = json.loads((ckpt/'run.json').read_text())\n    world_scale = float(training_record['world_scale'])\n    if cfg.mode != 'behave_calibrated_pilot' or not training_record['exact_principal_point']:\n        raise ValueError('Expected controlled exact-camera adapter')"),
 ("    pose_error = max(float((cameras.T_wc(i)-training_c2w).abs().max()) for i in range(int(cameras.T)))", "    normalized_c2w = training_c2w.clone(); normalized_c2w[:3,3] *= world_scale\n    pose_error = max(float((cameras.T_wc(i)-normalized_c2w).abs().max()) for i in range(int(cameras.T)))"),
 ("    record['coordinate_checks'] = {'scale_nw': 1.", "    from validation_utils import expose_metric_gaussians\n    expose_metric_gaussians(static, dynamic, world_scale)\n    record['exact_principal_point'] = True\n    record['coordinate_checks'] = {'scale_nw': world_scale")]
save_variant(ROOT/'scripts/diagnostics/render_mosca_heldout.py', CODE/'heldout_control.py', heldout_edits)
(CODE/'adapter_changes.json').write_text(json.dumps(changes, indent=2)+'\n')
print('Built isolated renderer and adapters:', len(changes))
