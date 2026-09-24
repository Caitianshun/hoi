"""Evaluation-only fitted mesh masks and held-out RGB region metrics; CPU only.

No fit, alignment, interpolation, checkpoint reads, or train-directory writes.
Masks are fixed from reference meshes/camera before any predictions are opened.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import cv2
import trimesh
from numba import njit
from skimage.metrics import structural_similarity

ROOT = Path('/home/cai_tianshun/Project/HOI')
E = ROOT / 'experiments/structured_hoi_20260923'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + '\n')


@njit(cache=True)
def raster(uv, z, faces, entity, depth, labels):
    """Integer pixel centres, two-sided triangles, perspective-correct depth."""
    height, width = depth.shape
    for face in faces:
        a, b, c = face
        if min(z[a], z[b], z[c]) <= 1e-5:
            continue
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
                    labels[yy, xx] = entity


def camera_mask(parts, K, c2w, height, width):
    depth = np.full((height, width), np.inf, np.float64)
    labels = np.zeros((height, width), np.uint8)
    for entity, vertices, faces in parts:
        cam = (vertices-c2w[:3, 3])@c2w[:3, :3]
        assert np.isfinite(cam).all() and np.all(cam[:, 2] > 1e-5), 'No near-plane clipping case allowed silently'
        projected = cam@K.T
        raster(projected[:, :2]/projected[:, 2:], cam[:, 2], np.asarray(faces, np.int64), entity, depth, labels)
    return labels, depth


def raster_checks():
    uv = np.array([[1., 1.], [5., 1.], [1., 5.]])
    face = np.array([[0, 1, 2]], np.int64)
    depth = np.full((7, 7), np.inf)
    labels = np.zeros((7, 7), np.uint8)
    raster(uv, np.array([4., 4., 4.]), face, 1, depth, labels)
    raster(uv, np.array([2., 2., 2.]), face, 2, depth, labels)
    assert labels[2, 2] == 2 and depth[2, 2] == 2 and labels[6, 6] == 0
    raster(uv, np.array([3., 3., 3.]), face, 1, depth, labels)
    assert labels[2, 2] == 2
    depth[:] = np.inf
    raster(uv, np.array([1., 2., 4.]), face, 1, depth, labels)
    assert abs(depth[2, 2]-1/(.5/1+.25/2+.25/4)) < 1e-10
    return {'near_wins': True, 'far_does_not_overwrite': True, 'perspective_depth': True, 'integer_centres': True}


def prepare(dev, out):
    begin = time.perf_counter()
    checks = raster_checks()
    if dev == 'dev1':
        source = ROOT/'experiments/mosca_interface_validation_20260923/d_exact_geometry/heldout/heldout_manifest.json'
        meta = json.loads(source.read_text())
        K, C = np.array(meta['K_rectified']), np.array(meta['c2w'])
        height, width = meta['output_hw']
        pairs = [dict(nominal_time=p['nominal_heldout_time_s'], input_index=p['input_index'], gt_path=p['heldout_gt'], gt_sha256=p['heldout_gt_sha256'], filename=p['filename'], input_minus_camera1_time_s=None, input_minus_nominal_s=p['input_minus_nominal_s'], camera1_actual_time_s=None) for p in meta['pairs']]
        refroot = ROOT/'data/BEHAVE/evaluation_only/sparse_event_reference/Date01_Sub01_boxsmall_hand'
    else:
        source = E/'data_audit/dev2_evaluation/heldout_manifest.json'
        meta = json.loads(source.read_text())
        K, C = np.array(meta['camera']['K']), np.array(meta['camera']['c2w'])
        height, width = meta['camera']['height'], meta['camera']['width']
        pairs = [dict(nominal_time=p['nominal_time_seconds'], input_index=p['matched_input_frame_index'], gt_path=p['path'], gt_sha256=p['sha256'], filename=Path(p['path']).name, input_minus_camera1_time_s=p['input_minus_camera1_time_seconds'], input_minus_nominal_s=p['matched_input_timestamp_seconds']-p['nominal_time_seconds'], camera1_actual_time_s=p['actual_camera1_timestamp_seconds']) for p in meta['records']]
        refroot = E/'data_audit/dev2_evaluation/source/Date01_Sub01_chairwood_lift'
    out.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []
    for row in pairs:
        gtpath = Path(row['gt_path'])
        assert sha(gtpath) == row['gt_sha256']
        gt = cv2.imread(str(gtpath))
        assert gt.shape[:2] == (height, width)
        timepath = refroot/f"t{row['nominal_time']:08.3f}"
        hp = timepath/'person/fit02/person_fit.ply'
        op = timepath/('boxsmall/fit01/boxsmall_fit.ply' if dev == 'dev1' else 'chair/fit01/chair_fit.ply')
        row['reference_meshes'] = {str(p): sha(p) if p.exists() else None for p in [hp, op]}
        if not hp.exists() or not op.exists():
            row.update(mask_status='missing_reference_keep_full_only', mask_path=None)
            overlay = gt.copy()
            cv2.putText(overlay, 'NO FIT MESH: full image only', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
        else:
            parts = []
            for entity, path in [(1, hp), (2, op)]:
                mesh = trimesh.load(str(path), process=False)
                parts.append((entity, np.asarray(mesh.vertices), np.asarray(mesh.faces)))
            labels, depth = camera_mask(parts, K, C, height, width)
            maskpath = out/f"t{row['nominal_time']:08.3f}_reference_mask.npz"
            np.savez_compressed(maskpath, entity_labels=labels, depth_camera_z_m=depth.astype(np.float32), role=np.array('evaluation_only_approximate_fitted_visibility'))
            palette = np.array([[0, 0, 0], [70, 210, 80], [240, 100, 40]], np.uint8)
            overlay = gt.copy()
            overlay[labels > 0] = np.rint(.6*overlay[labels > 0]+.4*palette[labels[labels > 0]]).astype(np.uint8)
            for entity, color in [(1, (0, 255, 0)), (2, (255, 100, 0))]:
                contours, _ = cv2.findContours((labels == entity).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 1)
            row.update(mask_status='available', mask_path=str(maskpath), mask_sha256=sha(maskpath), pixels={name: int((labels == eid).sum()) for eid, name in enumerate(['background', 'human', 'object'])})
        preview = out/f"t{row['nominal_time']:08.3f}_overlay.png"
        cv2.imwrite(str(preview), np.concatenate([gt, overlay], axis=1))
        row['overlay'] = str(preview)
        panels.append(np.concatenate([gt, overlay], axis=1))
        rows.append(row)
    cv2.imwrite(str(out/'reference_mask_review.jpg'), np.concatenate(panels, axis=0))
    record = dict(status='completed', dev=dev, role='evaluation_only', prepared_before_prediction_reads=True, source_manifest=str(source), source_manifest_sha256=sha(source), script_sha256=sha(__file__), camera=dict(K=K.tolist(), c2w=C.tolist(), height=height, width=width), checks=checks, rows=rows, available_mask_frames=sum(r['mask_status']=='available' for r in rows), seconds=time.perf_counter()-begin,
        definition='Original same-version human fit02 and object fit01 world meshes; integer-centre pinhole projection, perspective-correct triangle z buffer, nearest person/object face label. No alignment, pose fitting, cropping, mask tuning, interpolation or train writes.',
        limitation='Approximate fitted silhouettes; cloth/hair and fit errors remain. Only human/object depth ordering, no scene mesh occlusion. Mask time is nominal reference time; camera/input exposure mismatch recorded. Masks are heldout evaluation annotations and forbidden training inputs.')
    save(out/'mask_manifest.json', record)
    return record


def score(manifest, evaluation, output):
    export = json.loads((evaluation/'export_manifest.json').read_text())
    assert export['status'] == 'completed' and export.get('evaluation_status') == 'completed', 'Only completed frozen exports and held-out evaluation may be scored'
    rows = []
    for row in manifest['rows']:
        path = evaluation/'heldout'/row['filename']
        gt = cv2.imread(row['gt_path'])[..., ::-1].astype(np.float64)/255
        assert sha(row['gt_path']) == row['gt_sha256']
        pred = cv2.imread(str(path))[..., ::-1].astype(np.float64)/255
        assert pred.shape == gt.shape
        _, sm = structural_similarity(gt, pred, data_range=1, channel_axis=2, full=True)
        sm = sm.mean(-1)
        se = ((gt-pred)**2).mean(-1)
        regions = {'full': np.ones(gt.shape[:2], bool)}
        if row['mask_path']:
            assert sha(row['mask_path']) == row['mask_sha256']
            labels = np.load(row['mask_path'])['entity_labels']
            for eid, name in enumerate(['background', 'human', 'object']):
                regions[name] = labels == eid
                regions[name+'_interior3px'] = cv2.erode(regions[name].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            regions['foreground'] = labels > 0
        scores = {}
        for name, mask in regions.items():
            count = int(mask.sum())
            mse = float(se[mask].mean()) if count else None
            scores[name] = dict(pixels=count, mse=mse, squared_error_sum=float(se[mask].sum()), psnr_db=-10*float(np.log10(max(mse, 1e-12))) if count else None, ssim=float(sm[mask].mean()) if count else None)
        rows.append(dict(nominal_time=row['nominal_time'], prediction=str(path), prediction_sha256=sha(path), input_index=row['input_index'], mask_status=row['mask_status'], metrics=scores))
    summary = {}
    for name in dict.fromkeys(k for r in rows for k in r['metrics']):
        rr = [r['metrics'][name] for r in rows if name in r['metrics'] and r['metrics'][name]['pixels'] > 0]
        count = sum(r['pixels'] for r in rr)
        pooled_mse = sum(r['squared_error_sum'] for r in rr)/count if count else None
        summary[name] = dict(frames=len(rr), pixels=count, mean_frame_psnr_db=float(np.mean([r['psnr_db'] for r in rr])) if rr else None, pooled_psnr_db=-10*float(np.log10(max(pooled_mse, 1e-12))) if count else None, mean_frame_ssim=float(np.mean([r['ssim'] for r in rr])) if rr else None)
    result = dict(status='completed', evaluation=str(evaluation), export_manifest_sha256=sha(evaluation/'export_manifest.json'), script_sha256=sha(__file__), mask_manifest_sha256=sha(output.parent/'mask_manifest.json'), rows=rows, summary=summary, metric='Saved uint8 frozen RGB; MSE/PSNR use RGB [0,1]. SSIM is mean full-image 7x7 uniform SSIM map within region. interior3px excludes all region boundary neighbourhoods. Fixed masks shared by S0/S1. No fitted alignment or color adjustment.', limitation=manifest['limitation'])
    save(output, result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev', choices=['dev1', 'dev2'], required=True)
    parser.add_argument('--mode', choices=['prepare', 'score', 'both'], default='prepare')
    parser.add_argument('--evaluation', action='append', default=[], help='name=/absolute/frozen/export')
    args = parser.parse_args()
    out = E/'data_audit/heldout_regions'/args.dev
    manifest = prepare(args.dev, out) if args.mode != 'score' else json.loads((out/'mask_manifest.json').read_text())
    summaries = {}
    if args.mode != 'prepare':
        for specification in args.evaluation:
            name, path = specification.split('=', 1)
            assert name.replace('_', '').isalnum()
            summaries[name] = score(manifest, Path(path), out/f'{name}_metrics.json')
    print(json.dumps(dict(dev=args.dev, masks=manifest['available_mask_frames'], summaries=summaries), indent=2))


if __name__ == '__main__':
    main()
