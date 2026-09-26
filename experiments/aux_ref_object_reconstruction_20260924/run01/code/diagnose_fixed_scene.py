"""CPU-only AUX fixed-H/S occlusion diagnostic at native camera0 object pixels.

No learned object bank, human reference, sensor depth or camera1 is read. Pred
is old S1 RGB motion; Ref is the explicitly allowed native AUX object R/t.
Template depth is a diagnostic surface, not a new point correspondence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import time

import cv2
import numpy as np
from numba import njit

ROOT = Path('/home/cai_tianshun/Project/HOI')
E = Path(__file__).resolve().parents[1]
SPARSE_SOURCE = ROOT / 'experiments/mosca_validation_20260923/analysis_a/sparse_raster_exact.py'
MESH_SOURCE = ROOT / 'experiments/structured_hoi_20260923/code/evaluate_heldout_regions_cpu.py'
NATIVE_SOURCE = ROOT / 'experiments/mosca_interface_validation_20260923/code/MoSca/lib_render/gauspl_renderer_native_add3.py'
FORWARD_SOURCE = NATIVE_SOURCE.parent / 'diff-gaussian-rasterization-alphadep-add3/cuda_rasterizer/forward.cu'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def identity(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


@njit(cache=True)
def mesh_raster(uv, z, faces, depth):
    """Two-sided mesh, integer centres, perspective-correct nearest depth.

    Pure raster from MESH_SOURCE; reject near-plane cases before calling.
    """
    height, width = depth.shape
    for a, b, c in faces:
        x0, y0 = uv[a]
        x1, y1 = uv[b]
        x2, y2 = uv[c]
        denom = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(denom) < 1e-10:
            continue
        xmin = max(0, int(np.ceil(min(x0, x1, x2))))
        xmax = min(width-1, int(np.floor(max(x0, x1, x2))))
        ymin = max(0, int(np.ceil(min(y0, y1, y2))))
        ymax = min(height-1, int(np.floor(max(y0, y1, y2))))
        for yy in range(ymin, ymax+1):
            for xx in range(xmin, xmax+1):
                w0 = ((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/denom
                w1 = ((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/denom
                w2 = 1-w0-w1
                if min(w0, w1, w2) < -1e-8:
                    continue
                zz = 1/(w0/z[a]+w1/z[b]+w2/z[c])
                if zz < depth[yy, xx]:
                    depth[yy, xx] = zz


def template_depth(vertices, faces, R, t, K, C, H, W):
    camera = (vertices @ R.T + t - C[:3, 3]) @ C[:3, :3]
    if not np.isfinite(camera).all() or (camera[:, 2] <= .02).any():
        raise ValueError('Template near-plane/nonfinite case requires explicit clipping; not silently skipped')
    projected = camera @ K.T
    depth = np.full((H, W), np.inf, np.float64)
    mesh_raster(projected[:, :2] / projected[:, 2:], camera[:, 2], faces, depth)
    return depth


@njit(cache=True)
def make_tile_index(rectmin, rectmax, depth_order, gx, gy):
    counts = np.zeros(gx*gy, np.int64)
    for i in depth_order:
        for y in range(rectmin[i, 1], rectmax[i, 1]):
            for x in range(rectmin[i, 0], rectmax[i, 0]):
                counts[y*gx+x] += 1
    ptr = np.zeros(gx*gy+1, np.int64)
    for j in range(len(counts)):
        ptr[j+1] = ptr[j] + counts[j]
    ids = np.empty(ptr[-1], np.int64)
    cursor = ptr[:-1].copy()
    for i in depth_order:
        for y in range(rectmin[i, 1], rectmax[i, 1]):
            for x in range(rectmin[i, 0], rectmax[i, 0]):
                tile = y*gx+x
                ids[cursor[tile]] = i
                cursor[tile] += 1
    return ptr, ids


def project_hs(xyz, frame, scale, opacity, K, C, H, W):
    """Same float64 projection/footprints as frozen sparse_raster_exact.py."""
    # Preserve scale's input precision for scale**2, as the legacy helper does;
    # world/camera products are float64. Promoting scale before squaring changes
    # real float32-bank footprints by small, avoidable amounts.
    xyz = np.asarray(xyz, np.float64)
    frame, scale = np.asarray(frame), np.asarray(scale)
    camera = (xyz-C[:3, 3]) @ C[:3, :3]
    z = camera[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mu = camera[:, :2]/(z[:, None]+1e-7)*[fx, fy]+[cx, cy]
    rc = C[:3, :3].T[None] @ frame
    cov = (rc*scale[:, None, :]**2) @ rc.transpose(0, 2, 1)
    zz = np.maximum(z, 1e-8)
    clx = np.clip(camera[:, 0]/zz, -1.3*W/(2*fx), 1.3*W/(2*fx))
    cly = np.clip(camera[:, 1]/zz, -1.3*H/(2*fy), 1.3*H/(2*fy))
    J = np.zeros((len(z), 2, 3))
    J[:, 0, 0], J[:, 1, 1] = fx/zz, fy/zz
    J[:, 0, 2], J[:, 1, 2] = -fx*clx/zz, -fy*cly/zz
    cov2 = J @ cov @ J.transpose(0, 2, 1)
    cov2[:, 0, 0] += .3
    cov2[:, 1, 1] += .3
    aa, bb, cc = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    det = aa*cc-bb**2
    conic = np.stack([cc, -bb, aa], -1)/np.maximum(det[:, None], 1e-20)
    mid = .5*(aa+cc)
    radius = np.ceil(3*np.sqrt(np.maximum(mid+np.sqrt(np.maximum(.1, mid**2-det)), 0)))
    grid = np.array([(W+15)//16, (H+15)//16])
    rectmin = np.clip(np.trunc((mu-radius[:, None])/16), 0, grid).astype(np.int64)
    rectmax = np.clip(np.trunc((mu+radius[:, None]+15)/16), 0, grid).astype(np.int64)
    valid = (z > .02) & (det > 0)
    depth_order = np.flatnonzero(valid)
    depth_order = depth_order[np.argsort(z[depth_order], kind='stable')]
    ptr, ids = make_tile_index(rectmin, rectmax, depth_order, int(grid[0]), int(grid[1]))
    return dict(mu=mu, z=z, conic=conic, opacity=np.asarray(opacity, np.float64).ravel(), ptr=ptr, ids=ids, gx=int(grid[0]), H=H, W=W)


@njit(cache=True)
def sample_prefix(uv, surface_z, mu, z, conic, opacity, ptr, ids, gx):
    # Both arms use the identical UV sample. Invalid template depths remain NaN.
    n = len(uv)
    mathematical_T = np.full(n, np.nan)
    renderer_T = np.full(n, np.nan)
    reachable = np.zeros(n, np.bool_)
    contributor_count = np.zeros(n, np.int64)
    terminated_at_z = np.full(n, np.nan)
    for q in range(n):
        if not np.isfinite(surface_z[q]):
            continue
        tile = int(uv[q, 1]//16)*gx+int(uv[q, 0]//16)
        product, rt, active = 1., 1., True
        for j in range(ptr[tile], ptr[tile+1]):
            i = ids[j]
            if z[i] >= surface_z[q]:
                break
            dx, dy = mu[i, 0]-uv[q, 0], mu[i, 1]-uv[q, 1]
            power = -.5*(conic[i, 0]*dx*dx+conic[i, 2]*dy*dy)-conic[i, 1]*dx*dy
            if power > 0:
                continue
            a = min(.99, opacity[i]*np.exp(power))
            if a < 1/255:
                continue
            product *= 1-a
            contributor_count[q] += 1
            if active:
                test = rt*(1-a)
                if test < .0001:
                    active = False
                    terminated_at_z[q] = z[i]
                else:
                    rt = test
        mathematical_T[q] = product
        renderer_T[q] = rt
        reachable[q] = active
    return mathematical_T, renderer_T, reachable, contributor_count, terminated_at_z


def prefix(projected, uv, depths):
    return sample_prefix(np.asarray(uv, np.float64), np.asarray(depths, np.float64), *[projected[k] for k in ('mu', 'z', 'conic', 'opacity', 'ptr', 'ids', 'gx')])


@njit(cache=True)
def render_pixels(uv, mu, z, conic, opacity, ptr, ids, gx, colors):
    n = len(uv)
    rgb = np.ones((n, 3), np.float64)
    alpha = np.zeros(n, np.float64)
    expected_depth = np.zeros(n, np.float64)
    for q in range(n):
        tile = int(uv[q, 1]//16)*gx+int(uv[q, 0]//16)
        T, accz = 1., 0.
        out = np.zeros(3, np.float64)
        for j in range(ptr[tile], ptr[tile+1]):
            i = ids[j]
            dx, dy = mu[i, 0]-uv[q, 0], mu[i, 1]-uv[q, 1]
            power = -.5*(conic[i, 0]*dx*dx+conic[i, 2]*dy*dy)-conic[i, 1]*dx*dy
            if power > 0:
                continue
            a = min(.99, opacity[i]*np.exp(power))
            if a < 1/255:
                continue
            test = T*(1-a)
            if test < .0001:
                break
            weight = a*T
            out += weight*colors[i]
            accz += weight*z[i]
            T = test
        rgb[q] = out+T
        alpha[q] = 1-T
        expected_depth[q] = accz/max(1-T, 1e-6)
    return rgb, alpha, expected_depth


def fixed_pixels(mask, maximum=512):
    flat = np.flatnonzero(mask)
    n = min(len(flat), maximum)
    chosen = flat[np.floor((np.arange(n)+.5)*len(flat)/n).astype(np.int64)] if n else flat
    y, x = np.unravel_index(chosen, mask.shape)
    return np.stack([x, y], -1).astype(np.int64)


def stats(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(count=0, min=None, mean=None, median=None, p05=None, p25=None, p75=None, p95=None, max=None)
    return dict(count=len(v), min=float(v.min()), mean=float(v.mean()), median=float(np.median(v)), p05=float(np.quantile(v, .05)), p25=float(np.quantile(v, .25)), p75=float(np.quantile(v, .75)), p95=float(np.quantile(v, .95)), max=float(v.max()))


def fraction(n, d):
    return dict(numerator=int(n), denominator=int(d), fraction=float(n/d) if d else None)


def arm_summary(depth, mask, uv, result):
    physical, rt, reachable, counts, stopdepth = result
    valid = np.isfinite(physical)
    total, ns, nv = int(mask.sum()), len(uv), int(valid.sum())
    effective = np.where(reachable, rt, 0.)[valid]
    return dict(total_fixed_O_pixels=total, sampled_fixed_O_pixels=ns,
                all_O_template_covered=int(np.isfinite(depth[mask]).sum()),
                all_O_template_uncovered=fraction((~np.isfinite(depth[mask])).sum(), total),
                sampled_template_no_depth=fraction((~valid).sum(), ns), valid_template_depth_samples=nv,
                T_HS_before_template=stats(physical[valid]), renderer_retained_T_before_template=stats(rt[valid]),
                renderer_effective_reachability_weight=stats(effective),
                renderer_terminated_before_template=fraction((~reachable[valid]).sum(), nv),
                low_T={str(threshold): fraction((physical[valid] < threshold).sum(), nv) for threshold in (.01, .1, .5)},
                prefix_HS_contributors=stats(counts[valid]),
                hypothetical_alpha_0p5_direct_color_weight=stats(np.where(reachable[valid] & (rt[valid]*.5 >= 1e-4), rt[valid]*.5, 0.)))


def checks():
    """Meaningful CPU checks: legacy parity, depth exclusion and saturation."""
    rng = np.random.default_rng(619)
    H, W = 48, 64
    K = np.array([[48., 0, 30.2], [0, 49., 22.8], [0, 0, 1.]])
    C = np.eye(4)
    xyz = rng.normal(size=(180, 3))*.25
    xyz[:, 2] = rng.uniform(.1, 4, len(xyz))
    frame = np.repeat(np.eye(3)[None], len(xyz), 0)
    frame[:, 0, 1] = .15  # Also check affine, non-SO3 human frames.
    scale = rng.uniform(.005, .1, (len(xyz), 3))
    opacity = rng.uniform(.001, .95, len(xyz))
    colors = rng.random((len(xyz), 3))
    uv = np.stack([rng.integers(W, size=37), rng.integers(H, size=37)], -1)
    p = project_hs(xyz, frame, scale, opacity, K, C, H, W)
    sparse = runpy.run_path(str(SPARSE_SOURCE))['sparse_raster']
    legacy = sparse(xyz, frame, scale, opacity, K, C, uv, W=W, H=H)
    rgb, alpha, _ = render_pixels(uv, *[p[k] for k in ('mu', 'z', 'conic', 'opacity', 'ptr', 'ids', 'gx')], colors)
    alpha_ref = np.array([w.sum() for ids, w in legacy])
    rgb_ref = np.array([(colors[ids]*w[:, None]).sum(0)+1-w.sum() for ids, w in legacy])
    aerr, rerr = float(np.max(abs(alpha-alpha_ref))), float(np.max(abs(rgb-rgb_ref)))
    assert max(aerr, rerr) < 1e-12, (aerr, rerr)
    # Real banks use float32 scales; legacy squares them before promotion.
    f32 = [a.astype(np.float32) for a in (xyz, frame, scale, opacity)]
    p32 = project_hs(*f32, K, C, H, W)
    legacy32 = sparse(*f32, K, C, uv, W=W, H=H)
    _, alpha32, _ = render_pixels(uv, *[p32[k] for k in ('mu', 'z', 'conic', 'opacity', 'ptr', 'ids', 'gx')], colors)
    aerr32 = float(np.max(abs(alpha32-np.array([w.sum() for _, w in legacy32]))))
    assert aerr32 < 1e-12
    # Analytic ray: a=.5 at z1, a=.8 at z3; surface z2 must exclude z3.
    args = dict(mu=np.zeros((2, 2)), z=np.array([1., 3.]), conic=np.tile([1., 0, 1.], (2, 1)), opacity=np.array([.5, .8]), ptr=np.array([0, 2]), ids=np.array([0, 1]), gx=1)
    q = np.zeros((1, 2))
    out = prefix(args, q, [2.])
    assert out[0][0] == .5 and out[2][0]
    assert abs(prefix(args, q, [4.])[0][0]-.1) < 1e-12
    # CUDA rejects the Gaussian that crosses the 1e-4 threshold, then stops.
    args.update(mu=np.zeros((3, 2)), z=np.array([1., 2., 3.]), conic=np.tile([1., 0, 1.], (3, 1)), opacity=np.array([.99, .99, .99]), ptr=np.array([0, 3]), ids=np.arange(3))
    out = prefix(args, q, [4.])
    assert abs(out[0][0]-1e-6)<1e-14 and not out[2][0] and abs(out[1][0]-1e-4)<1e-14
    # Hypothetical color derivative is alpha_obj*T; a central finite difference.
    T, ao, c, eps = .5, .4, .7, 1e-6
    analytic = ao*T
    numerical = ((ao*T*(c+eps)+.12)-(ao*T*(c-eps)+.12))/(2*eps)
    assert abs(analytic-numerical)<1e-9
    depth = np.full((7, 7), np.inf)
    mesh_raster(np.array([[1., 1.], [5., 1.], [1., 5.]]), np.array([1., 2., 4.]), np.array([[0, 1, 2]]), depth)
    assert abs(depth[2, 2]-1/(.5/1+.25/2+.25/4))<1e-12
    empty = prefix(args, np.zeros((0, 2)), np.zeros(0))
    assert not len(empty[0]) and not len(fixed_pixels(np.zeros((7, 7), bool)))
    return dict(status='passed', device='CPU only', synthetic_seed=619, legacy_alpha_max_abs_error=aerr,
                legacy_rgb_max_abs_error=rerr, legacy_float32_bank_alpha_max_abs_error=aerr32,
                sparse_helper=identity(SPARSE_SOURCE),
                behind_surface_excluded=True, native_early_stop_distinguished=True,
                dummy_color_weight_finite_difference_error=abs(analytic-numerical),
                mesh_perspective_depth=True, empty_O_supported=True, GPU_numerical_validation_performed=False)


DEFINITION = {
    'fixed_region': 'Native camera0 RGB-estimated segmentation label 2 (O), unchanged for Pred/Ref. No mask derived from predicted motion.',
    'sampling': 'At most 512 equally spaced midpoint ranks in row-major O pixel order, deterministic without replacement; same UV in both arms. All-O template coverage is also computed exactly.',
    'surface_depth': 'Nearest two-sided known-untextured-template triangle at integer pixel centres, perspective-correct camera z. Pred old S1 world R/t; Ref same-sample native published object R/t. No nearest-time reference filling.',
    'T_definition': 'Product over H/S Gaussian centres with camera z strictly less than template surface z of (1-alpha_i(u)). Alpha includes projected covariance and opacity, capped .99, with native tile culling/near .02 and alpha<1/255 skip. This is depth-truncated, not 1 minus whole H/S alpha.',
    'early_stop': 'Physical product includes every prefix alpha. Renderer-retained T stops before the first alpha that would make T<1e-4. Reachability is false after such a stop; effective reachability weight is then zero, despite nonzero retained T.',
    'gradient_interpretation': 'These are frozen-scene attenuation/visibility diagnostics, not measured trainable-object gradients. A hypothetical object Gaussian at template depth with pixel alpha .5 has direct color coefficient .5*T if reachable and its own threshold survives. Actual RGB/instance data gradients and regularizer gradients must come from training logs.',
    'limitations': ['Template surface depth differs from learned Gaussian centre depths and does not include object self-occlusion.', 'Gaussian renderer sorts by centre depth, not ray-volume intersection. T follows that renderer convention.', 'Fixed RGB masks can be wrong or empty; uncovered/no-depth samples are excluded from T statistics and retain explicit denominators.', 'CPU projection reproduces the frozen float64 sparse helper. CUDA uses float32; exact pixel-cutoff/tile/depth-tie parity is not newly certified.', 'No trustworthy local surface correspondences are inferred; local-support correctness remains unmeasured.', 'Native timestamps are official nominal capture sample IDs; exact exposure synchronization error is unknown.'],
    'use': 'Diagnostic only. Never gates/drops training observations, changes schedules, selects checkpoints or adds budget.',
}


def run(dev, output, maximum=512, hs_images=False, frame_indices=None):
    import torch
    from aux_scene import AuxObjectScene
    torch.set_num_threads(2)
    start = time.perf_counter()
    manifest_path = E/f'inputs/{dev}/input_manifest.json'
    meta = json.loads(manifest_path.read_text())
    assert meta['protocol_id'] == 'AUX_REF_OBJECT' and meta['camera_id'] == 0 and meta['phase'] == 'complete'
    for key, hkey in [('segmentation', 'segmentation_sha256'), ('reference_object_motion', 'reference_motion_sha256')]:
        assert sha(meta[key]) == meta[hkey], f'Input hash mismatch: {key}'
    times = np.asarray(meta['timestamp_seconds'], np.float64)
    ref = np.load(meta['reference_object_motion'])
    assert str(ref['protocol_id']) == 'AUX_REF_OBJECT' and np.array_equal(ref['times'], times)
    labels = np.load(meta['segmentation'])['entity_labels']
    H, W = int(meta['height']), int(meta['width'])
    assert labels.shape == (len(times), H, W)
    K, C = np.array(meta['K'], np.float64), np.array(meta['c2w'], np.float64)
    scene = AuxObjectScene(dev, times, device='cpu', cache_hs=False)
    geometry = np.load(scene.initialization['object_init'])
    vertices, faces = np.asarray(geometry['canonical_vertices_m'], np.float64), np.asarray(geometry['faces'], np.int64)
    selection = list(range(len(times))) if frame_indices is None else frame_indices
    assert len(set(selection)) == len(selection) and all(0 <= i < len(times) for i in selection)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in selection:
        frame_start = time.perf_counter()
        folder = output/f'{index:05d}'
        folder.mkdir()
        s, h = scene.frozen_components(index)
        xyz, affine, scales, opacity = [np.concatenate([p[k].detach().numpy() for p in (s, h)]) for k in range(4)]
        colors = np.concatenate([b.color_logit.sigmoid().detach().numpy() for b in (scene.sbank, scene.hbank)])
        projected = project_hs(xyz, affine, scales, opacity, K, C, H, W)
        mask = labels[index] == 2
        uv = fixed_pixels(mask, maximum)
        raw = dict(uv=uv, total_fixed_O_pixels=np.array(mask.sum()), template_arm_order=np.array(['Pred', 'Ref']))
        row = dict(frame_index=index, time_seconds=float(times[index]), sample_id=meta['native_rows'][index]['sample_id'],
                   fixed_O_pixels=int(mask.sum()), sampled_O_pixels=len(uv), fixed_mask_sha256=hashlib.sha256(labels[index].tobytes()).hexdigest(),
                   old_prediction_motion_record=scene.motion_records[index], arms={})
        for arm, R, t in [('Pred', scene.pred_R_world[index].numpy(), scene.pred_t_world[index].numpy()), ('Ref', ref['R_world'][index], ref['t_world'][index])]:
            depth = template_depth(vertices, faces, R, t, K, C, H, W)
            sampled_depth = depth[uv[:, 1], uv[:, 0]]
            result = prefix(projected, uv, sampled_depth)
            row['arms'][arm] = arm_summary(depth, mask, uv, result)
            raw[f'{arm}_template_depth_camera_z_m'] = sampled_depth
            for name, array in zip(['T_product', 'T_renderer_retained', 'renderer_reachable', 'prefix_contributor_count', 'renderer_stop_camera_z_m'], result):
                raw[f'{arm}_{name}'] = array
        # Check real H/S projection against the independent frozen helper once
        # per frame on the same first 8 fixed samples; never run optimization.
        check_uv = uv[:8]
        legacy = runpy.run_path(str(SPARSE_SOURCE))['sparse_raster'](xyz, affine, scales, opacity, K, C, check_uv, W=W, H=H)
        _, own_alpha, _ = render_pixels(check_uv, *[projected[k] for k in ('mu', 'z', 'conic', 'opacity', 'ptr', 'ids', 'gx')], colors)
        old_alpha = np.array([weight.sum() for _, weight in legacy])
        parity = float(np.max(abs(own_alpha-old_alpha))) if len(uv) else None
        assert parity is None or parity < 1e-11, ('Legacy sparse alpha mismatch', index, parity)
        row['legacy_sparse_alpha_parity'] = dict(sample_count=len(check_uv), max_abs_error=parity)
        np.savez_compressed(folder/'prefix_samples.npz', **raw)
        row['sample_arrays'] = identity(folder/'prefix_samples.npz')
        if hs_images:
            yy, xx = np.indices((H, W))
            all_uv = np.stack([xx.ravel(), yy.ravel()], -1)
            rgb, alpha, dep = render_pixels(all_uv, *[projected[k] for k in ('mu', 'z', 'conic', 'opacity', 'ptr', 'ids', 'gx')], colors)
            rgb, alpha, dep = rgb.reshape(H, W, 3), alpha.reshape(H, W), dep.reshape(H, W)
            cv2.imwrite(str(folder/'HS_only_rgb.png'), np.rint(np.clip(rgb[..., ::-1], 0, 1)*255).astype(np.uint8))
            np.savez_compressed(folder/'HS_only_buffers.npz', rgb=rgb.astype(np.float32), alpha=alpha.astype(np.float32), expected_depth_camera_z_m=dep.astype(np.float32))
            row['HS_only_rgb'] = identity(folder/'HS_only_rgb.png')
            row['HS_only_buffers'] = identity(folder/'HS_only_buffers.npz')
        row['seconds'] = time.perf_counter()-frame_start
        save(folder/'summary.json', row)
        rows.append(row)
        print(json.dumps(dict(dev=dev, frame=index, O=int(mask.sum()), sampled=len(uv), seconds=row['seconds']), ensure_ascii=False), flush=True)
    assert scene.assert_frozen()
    sources = [Path(__file__), E/'code/aux_scene.py', SPARSE_SOURCE, MESH_SOURCE, NATIVE_SOURCE]
    if FORWARD_SOURCE.exists():
        sources.extend([FORWARD_SOURCE, FORWARD_SOURCE.parent/'auxiliary.h'])
    report = dict(protocol_id='AUX_REF_OBJECT', role='diagnostic_native_S_camera0_only', dev=dev, device='CPU',
                  definition=DEFINITION, input_manifest=identity(manifest_path), scene_sources=scene.source_identity,
                  template=identity(scene.initialization['object_init']), source_code=[identity(p) for p in sources],
                  frame_selection=selection, maximum_fixed_O_samples=maximum, HS_only_images_written=hs_images,
                  unchanged_frozen_scene=True, human_reference_read=False, camera1_read=False, learned_object_bank_read=False,
                  formal_training_steps=0, gradient_measurement='No learned data gradient is claimed; see training logs.',
                  rows=rows, seconds=time.perf_counter()-start)
    save(output/'manifest.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--dev', choices=['dev1', 'dev2'])
    p.add_argument('--output', type=Path)
    p.add_argument('--max-pixels', type=int, default=512)
    p.add_argument('--frames', type=int, nargs='+', help='Explicit input indices; default all native S frames, no output-driven selection')
    p.add_argument('--hs-images', action='store_true', help='Also render complete fixed H/S RGB + alpha + expected depth on CPU')
    a = p.parse_args()
    if not 1 <= a.max_pixels <= 512:
        p.error('--max-pixels must be in [1,512]')
    if a.self_test:
        result = checks()
        if a.output:
            save(a.output, result | dict(code=identity(__file__), definition=DEFINITION))
        print(json.dumps(result, indent=2))
    else:
        if not a.dev:
            p.error('--dev is required unless --self-test')
        report = run(a.dev, a.output or E/f'diagnostics/fixed_scene/{a.dev}', a.max_pixels, a.hs_images, a.frames)
        print(json.dumps(dict(dev=a.dev, frames=len(report['rows']), seconds=report['seconds'])))


if __name__ == '__main__':
    main()
