"""Read-only V8 input identity and independent Tennis asset inventory."""
from common import *
import zipfile,subprocess,platform,pickle,io

def main():
    cfg=read(RUN/'configs/v8.json');assets=[]
    for name,p in [('shared_fine0',V5/'protocol/shared_fine_initial.pt'),('B_U_fine1000',V5/'runs/B_U/checkpoint_fine_001000.pt'),('B_U_terminal',V5/'runs/B_U/checkpoint_fine_014000.pt')]:
        x=identity(p);assert x['sha256']==cfg['parent_hashes'][name];assets.append(x)
    cache=read(RUN/'track_cache_manifest.json');m=read(cache['manifest']['path']);dev=read(OLD/'inputs/hos_backpack/evaluation_manifest.json');assert len(m['frames'])==268 and len(dev['frames'])==16
    ids={f['frame_id'] for f in m['frames']};assert ids.isdisjoint({f['frame_id'] for f in dev['frames']})
    for pair in cache['pairs']:
        assert {pair['source_frame'],pair['target_frame'],*pair['window_frame_ids']}<=ids
        assert pair['cache']==identity(pair['cache']['path'])
    # Metadata only: no Tennis development pixels decoded or quality inspection.
    archive=ROOT/'other data/Tennis.zip';tennis=dict(status='asset_missing',development_images_opened=False)
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            ims=sorted(n for n in z.namelist() if n.startswith('Tennis/images/') and n.endswith('.png'))
            masks=sorted(n for n in z.namelist() if n.startswith('Tennis/masks/') and n.endswith('.png'))
            cams=pickle.loads(z.read('Tennis/cameras_scaleworld.pkl'))
            ids2=[Path(n).stem for n in ims];test=ids2[::(len(ids2)//16)][:16];train=[n for n in ids2 if n not in test]
            assert sorted(cams)==ids2 and {Path(n).stem for n in masks}==set(ids2)
            tennis=dict(status='RGB_masks_cameras_and_standard_split_available',archive=identity(archive),train_ids=train,development_ids=test,development_images_opened=False,initial_pointcloud_status='not yet built; existing fixed-camera train-only triangulation implementation available, no published cloud claimed legal',initialization_plan='If selected after V8A, apply existing frozen train-only SIFT pair triangulation with own time bounds and scale; no heldout RGB or mask reads.',camera_source_limitation='published scaleworld cameras; original SfM image scope unknown')
    save_json(RUN/'protocol/tennis_inventory.json',tennis)
    original=ROOT/'third_party/4DGaussians';assert subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],text=True).strip()==OFFICIAL_COMMIT
    save_json(RUN/'protocol.json',dict(status='input_protocol_frozen',input=identity(cache['manifest']['path']),track_cache=identity(RUN/'track_cache_manifest.json'),parents=assets,Tennis_inventory=identity(RUN/'protocol/tennis_inventory.json'),B_U_reuse='same shared fine0,268/16,seed,batch2,raw uniform L1,optimization/density/evaluator; Q changes only RGB objective, T changes only declared temporal evidence',time=m['time_normalization'],camera_source=m['camera_source'],image_objective=dict(**cfg['quality_objective'],source=identity(original/'utils/loss_utils.py'),SSIM_constants=[.01**2,.03**2],aggregate='all batch/channel/pixels arithmetic mean'),evaluation='unchanged historical clipped7x7 nonGaussian sample covariance SSIM, AlexNet spatial LPIPS, macro and pooled/raw PSNR',pair_order=identity(RUN/'protocol/pair_order.pt'),quality_reference=cfg['quality'],budget=cfg['budgets']))
    save_json(RUN/'environment.json',dict(python=platform.python_version(),upstream_commit=OFFICIAL_COMMIT,git_at_preflight=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),physical_GPU=1,GPU_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv'],text=True),raster_binary=identity(ROOT/'envs/4dgs/lib/python3.10/site-packages/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so')))
    print(json.dumps(dict(status='pass',pairs=len(cache['pairs']),valid_observations=sum(p['valid'] for p in cache['pairs']),tennis=tennis['status'])))
if __name__=='__main__':main()
