"""Freeze the original S1 camera0 protocol; evaluation assets stay separate.

This exporter performs no optimization or rendering and does not read held-out
RGB pixels. Hashing evaluation assets records identity, not training information.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
OLD = ROOT / 'experiments/structured_hoi_20260923'
AUX = ROOT / 'experiments/aux_ref_object_reconstruction_20260924/run01'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20): h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(data, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != value:
        raise RuntimeError(f'Refusing to overwrite different frozen output: {path}')
    if not path.exists(): path.write_text(value)


def frame(image_path, time, start, end, K, C, width, height, **extra):
    return dict(image_path=str(Path(image_path).resolve()), image_sha256=sha(image_path),
                width=width, height=height, K=K, c2w=C,
                w2c=np.linalg.inv(np.asarray(C)).tolist(), time_seconds=float(time),
                time=float((time-start)/(end-start)), **extra)


def export(dev):
    init_path = OLD / f'{dev}_initialization/initialization.pt'
    init = torch.load(init_path, map_location='cpu', weights_only=False)
    assert init['reference_used'] is False
    source_path = Path(init['input_dir'])/'input_manifest.json'
    source = json.loads(source_path.read_text())
    n = {'dev1': 114, 'dev2': 98}[dev]
    ts = np.asarray(source['timestamp_seconds'], np.float64)
    assert len(ts) == n and np.all(np.diff(ts) > 0)
    assert np.array_equal(ts, np.asarray(init['timestamps'], np.float64))
    assert source['camera_id'] == 0 and source['role'] == 'input_only'
    assert np.array_equal(source['K'], init['K']) and np.array_equal(source['c2w'], init['c2w'])
    start,end = float(ts[0]),float(ts[-1]); out = RUN/'inputs'/f'behave_{dev}'
    frames=[]
    for i,(path,t,digest) in enumerate(zip(source['frame_paths'],ts,source['frame_sha256'])):
        assert sha(path) == digest
        with Image.open(path) as image: assert image.size == (source['width'],source['height'])
        frames.append(frame(path,t,start,end,source['K'],source['c2w'],source['width'],source['height'],
                            frame_id=f'{i:05d}', camera_id=0, source_frame_index=source['frame_indices'][i]))
    train = dict(schema='baseline_calibration_camera_manifest_v1', dev=dev,
        protocol='BEHAVE_camera0_full_S1_shared_prior_initialization', role='training_only',
        sequence=source['sequence'], frames=frames, frame_count=n,
        time_normalization={'start_seconds':start,'end_seconds':end,'rule':'(time_seconds-start_seconds)/(end_seconds-start_seconds); training endpoints only'},
        timestamps={'source':identity(source['source_timestamps']),'sampling':source['sampling'],
          'basis':'original recording timestamp file indexed by decoded video frame; not nominal fps',
          'actual_capture_timing_claim':'recording timestamps as supplied by BEHAVE; absolute exposure accuracy not independently verified'},
        image_processing={'width':source['width'],'height':source['height'],'images_undistorted':source['images_undistorted'],
          'images_resized':source['images_resized'],'distortion':source['distortion'],'processing':source['processing']},
        source_manifest=identity(source_path), calibration=[identity(p) for p in source['calibration_sources']],
        coordinate_system=source['coordinate_system'], world_transform=np.eye(4).tolist(),
        initialization=identity(init_path), initialization_record=identity(init_path.with_suffix('.json')),
        point_cloud_manifest=str(out/'shared_init.json'),
        information_boundary={'allowed':['original camera0 RGB','published camera calibration','generic SMPL-X','RGB-predicted human/object state','untextured known rigid object geometry','input RGB predicted depth','original untrained H/O/S initialization'],
           'forbidden':['camera1 RGB','sensor depth','published per-frame human/object fits','trained S1/AUX/B/F pointcloud or colors'],
           'legacy_D_boundary_unchanged':True,'system_comparison_not_single_variable_ablation':True})
    init_meta_path=out/'shared_init.json'
    if init_meta_path.exists():
        shared=json.loads(init_meta_path.read_text())
        train.update(point_cloud={'path':shared['point_cloud']['path'],
                                  'npz_path':shared['float_point_cloud']['path'],
                                  'sha256':shared['point_cloud']['sha256'],
                                  'npz_sha256':shared['float_point_cloud']['sha256']},
                     scene_extent=shared['scene_extent'],scene_center=shared['scene_center'],aabb=shared['aabb'])
    save(out/'manifest.json',train)
    # Separate manifest: no evaluation paths are embedded in the training loader input.
    reg_path=AUX/'evaluation/regions/manifest.json';regions=json.loads(reg_path.read_text())
    camera=regions['camera']; aux=json.loads((AUX/'inputs'/dev/'input_manifest.json').read_text())
    native={float(row['query_time']):row for row in aux['native_rows']}
    eframes=[];c0frames=[]
    for r in regions['rows']:
        if r['dev']!=dev: continue
        t=float(r['query_time_seconds']);assert start<=t<=end
        ef=frame(r['rgb']['path'],t,start,end,camera['K'],camera['c2w'],camera['width'],camera['height'],
                 frame_id=r['frame_id'],sample_id=r['sample_id'],camera_id=1,regions=r['regions'],
                 query_time_basis=r['query_time_basis'],sync_limitation=r['sync_limitation'])
        assert ef['image_sha256']==r['rgb']['sha256'];eframes.append(ef)
        nr=native[t]
        c0frames.append(frame(nr['image'],t,start,end,aux['K'],aux['c2w'],aux['width'],aux['height'],
                             frame_id=r['frame_id'],sample_id=r['sample_id'],camera_id=0,
                             query_time_basis=nr['clock'],role='paired_input_view_diagnostic_not_original_training_frame'))
    assert len(eframes)=={'dev1':5,'dev2':4}[dev]
    evaluation=dict(schema='baseline_calibration_evaluation_manifest_v1',dev=dev,role='evaluation_only',
        frames=eframes,paired_camera0_frames=c0frames,source_regions=identity(reg_path),
        metric_definition=regions['metric_definition'],region_limitation=regions['region_limitation'],
        time_normalization=train['time_normalization'],
        timing_limitation='E uses nominal official capture-directory seconds; original S1 samples use recording timestamps. Same capture group is not proof of exact exposure synchronization.')
    save(out/'evaluation_manifest.json',evaluation)
    assets=[{'category':'generic_prior','asset':identity(init['smplx_model'])},
            {'category':'input_RGB_derived_human_state','asset':identity(init['human_geometry'])},
            {'category':'input_RGB_derived_object_pose_and_untextured_geometry','asset':identity(init['object_init'])},
            {'category':'input_RGB_derived_segmentation','asset':identity(init['segmentation'])},
            {'category':'untrained_H_O_S_initialization','asset':identity(init_path)}]
    save(RUN/'protocol'/f'behave_{dev}_assets.json',dict(dev=dev,training_assets=assets,
        training_manifest=identity(out/'manifest.json'),evaluation_manifest=identity(out/'evaluation_manifest.json'),
        original_S1_checkpoint=identity(OLD/f'{dev}_S1_v1/checkpoint_008000.pt'),
        evaluation_assets_excluded_from_initialization=True))
    print(json.dumps({'dev':dev,'training_frames':n,'evaluation_frames':len(eframes),'time_range':[start,end]}))


def attach_shared_init(dev):
    """One permitted staging transition: add immutable untrained point metadata.

    Called before training configuration freeze. It changes neither frames nor
    camera/time identities, and refuses changes to already attached metadata.
    """
    base=RUN/'inputs'/f'behave_{dev}';path=base/'manifest.json'
    train=json.loads(path.read_text());meta=json.loads((base/'shared_init.json').read_text())
    fields=dict(point_cloud={'path':meta['point_cloud']['path'],'npz_path':meta['float_point_cloud']['path'],
                            'sha256':meta['point_cloud']['sha256'],'npz_sha256':meta['float_point_cloud']['sha256']},
                scene_extent=meta['scene_extent'],scene_center=meta['scene_center'],aabb=meta['aabb'])
    if any(k in train for k in fields):
        assert all(train.get(k)==v for k,v in fields.items()),'Initialization metadata already set differently'
    else:
        train.update(fields);path.write_text(json.dumps(train,indent=2,ensure_ascii=False)+'\n')
        assetpath=RUN/'protocol'/f'behave_{dev}_assets.json';assets=json.loads(assetpath.read_text())
        assets['training_manifest']=identity(path)
        assetpath.write_text(json.dumps(assets,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'dev':dev,'shared_init_attached':True,'manifest':identity(path)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dev',choices=['dev1','dev2','both'],default='both')
    parser.add_argument('--attach-shared-init',action='store_true')
    args=parser.parse_args()
    for dev in ['dev1','dev2'] if args.dev=='both' else [args.dev]:
        attach_shared_init(dev) if args.attach_shared_init else export(dev)
