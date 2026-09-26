"""Export all original UNTRAINED S1 H/O/S centres and legal initial RGB.

Uses HumanLBS on CPU at the first original training timestamp, without loading
any checkpoint, reference motion, held-out image, sensor depth, or test fitting.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
from export_protocol import ROOT,RUN,OLD,sha,identity,save
sys.path.insert(0,str(OLD/'code'))
from human_lbs import HumanLBS


def export(dev):
    torch.set_num_threads(2)
    source=OLD/f'{dev}_initialization/initialization.pt'; p=torch.load(source,map_location='cpu',weights_only=False)
    assert p['reference_used'] is False
    record=json.loads(source.with_suffix('.json').read_text());assert sha(source)==record['initialization_sha256']
    for path,digest in record['inputs'].items():assert sha(path)==digest
    for path,digest in record['depth_hashes'].items():assert sha(ROOT/path if not Path(path).is_absolute() else path)==digest
    human=HumanLBS(p['human_geometry'],p['smplx_model'])
    with torch.no_grad():
        h,_=human(0);C=torch.tensor(p['c2w'],dtype=torch.float32);h=h@C[:3,:3].T+C[:3,3]
        o=torch.as_tensor(p['object_anchors']).float()@torch.as_tensor(p['object_R'][0]).float().T+torch.as_tensor(p['object_t'][0]).float()
        xyz=torch.cat([torch.as_tensor(p['background_anchors']).float(),h,o]).numpy()
        rgb=torch.cat([torch.logit(torch.as_tensor(p[k]).float().clamp(.01,.99)).sigmoid() for k in ['background_colors','human_colors','object_colors']]).numpy()
    counts=[len(p['background_anchors']),len(h),len(o)]
    entity=np.repeat(np.arange(3,dtype=np.uint8),counts)
    assert counts==[60000,10475,4096]
    assert np.isfinite(xyz).all() and np.isfinite(rgb).all()
    center=np.median(xyz.astype(np.float64),axis=0)
    distances=np.linalg.norm(xyz-center,axis=1);p95=float(np.percentile(distances,95));extent=1.1*p95
    assert np.isfinite(extent) and extent>0
    out=RUN/'inputs'/f'behave_{dev}';out.mkdir(parents=True,exist_ok=True)
    npz=out/'shared_init.npz';ply=out/'shared_init.ply'
    if npz.exists() or ply.exists():raise RuntimeError('Export output exists; historical/frozen initialization must not be overwritten')
    np.savez_compressed(npz,xyz=xyz,rgb=rgb,entity=entity,source_timestamp_seconds=np.float64(p['timestamps'][0]),source_initialization_sha256=np.array(sha(source)))
    # Standard 3DGS PLY. Float RGB remains losslessly available in NPZ.
    from plyfile import PlyData,PlyElement
    vertex=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,name in enumerate(['x','y','z']):vertex[name]=xyz[:,i]
    for name in ['nx','ny','nz']:vertex[name]=0
    for i,name in enumerate(['red','green','blue']):vertex[name]=np.rint(rgb[:,i]*255).astype(np.uint8)
    PlyData([PlyElement.describe(vertex,'vertex')]).write(ply)
    meta=dict(schema='shared_prior_untrained_HOS_v1',dev=dev,source_initialization=identity(source),
        source_initialization_record=identity(source.with_suffix('.json')),source_human_geometry=identity(p['human_geometry']),source_smplx_model=identity(p['smplx_model']),
        source_object_geometry_pose=identity(p['object_init']),reference_used=False,trained_checkpoint_read=False,test_RGB_read=False,
        point_cloud=identity(ply),float_point_cloud=identity(npz),point_count=len(xyz),counts={'background':counts[0],'human':counts[1],'object':counts[2]},
        first_training_timestamp_seconds=float(p['timestamps'][0]),coordinate_system='BEHAVE Date01 world = Kinect1 color; metres',world_transform=np.eye(4).tolist(),
        color_rule='Exact original untrained GaussianBank color_logit sigmoid, including .01/.99 clamp; all legal original RGB color sources retained. NPZ floats are preferred; PLY rounds to uint8.',
        position_rule='Original HumanLBS with zero trainable residuals at first training timestamp; original rigid object R/t at same timestamp; original predicted-depth background anchors.',
        scene_center=center.tolist(),distance_p95=p95,scene_extent=extent,extent_rule='1.1 * percentile95(norm(xyz - axiswise_median(xyz))); no recentering or rescaling',
        aabb={'min':xyz.min(0).tolist(),'max':xyz.max(0).tolist()},
        source_code={str(Path(__file__).resolve()):sha(__file__),str(OLD/'code/human_lbs.py'):sha(OLD/'code/human_lbs.py')},
        exported_on='CPU',optimization_steps=0)
    save(out/'shared_init.json',meta);print(json.dumps({k:meta[k] for k in ['dev','point_count','scene_extent','scene_center','aabb']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dev',choices=['dev1','dev2','both'],default='both');args=parser.parse_args()
    for dev in ['dev1','dev2'] if args.dev=='both' else [args.dev]:export(dev)
