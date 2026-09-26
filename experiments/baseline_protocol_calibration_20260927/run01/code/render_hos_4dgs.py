#!/usr/bin/env python3
"""Render the frozen H1 terminal checkpoint, never reading test RGB for rendering."""
import argparse,json,os,time
from pathlib import Path
import cv2
import numpy as np
import torch
from adapter_4dgs import RUN,OFFICIAL_COMMIT,CalibratedCamera,official_config,sha,save_json


def identity(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def run(run_dir,freeze):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    frozen=json.loads(Path(freeze).read_text())
    assert frozen['status']=='all_states_frozen' and frozen['scope']=='hos_backpack'
    assets={str(Path(a['path']).resolve()):a['sha256'] for a in frozen['assets']}
    for path,digest in assets.items():assert sha(path)==digest,f'Frozen asset changed: {path}'
    run_dir=Path(run_dir).resolve();terminal=json.loads((run_dir/'run.json').read_text())
    assert terminal['status']=='completed' and terminal['nominal_iterations']==17000
    checkpoint=Path(terminal['checkpoint']).resolve();assert assets.get(str(checkpoint))==terminal['checkpoint_sha256']
    cfg=json.loads((run_dir/'effective_config.json').read_text())
    assert sha(run_dir/'effective_config.json')==terminal['source_config_sha256']
    assert cfg['official_commit']==OFFICIAL_COMMIT and cfg['seed']==12345 and not cfg['check']
    base=RUN/'inputs/hos_backpack';train=json.loads((base/'manifest.json').read_text());evaluation=json.loads((base/'evaluation_manifest.json').read_text())
    assert cfg['manifest_sha256']==sha(base/'manifest.json') and train['role']=='training_only'
    assert len(train['frames'])==268 and len(evaluation['frames'])==16
    assert assets.get(str((base/'evaluation_manifest.json').resolve()))==sha(base/'evaluation_manifest.json')
    selected=set(np.unique(np.linspace(0,len(train['frames'])-1,16).astype(int)).tolist())
    out=RUN/'evaluation/H1';assert not out.exists(),f'Refusing overwrite {out}';out.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(12345);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    _,(dataset,hidden,opt,pipe)=official_config(cfg['optimization']['batch_size'])
    for obj,key in [(dataset,'model'),(hidden,'hidden'),(opt,'optimization'),(pipe,'pipeline')]:
        for name,value in cfg[key].items():setattr(obj,name,value)
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    model=GaussianModel(dataset.sh_degree,hidden);model._deformation=model._deformation.cuda()
    state=torch.load(checkpoint,map_location='cuda',weights_only=False)
    assert state['stage']=='fine' and state['iteration']==14000 and state['total_nominal_steps']==17000
    model.restore(state['model'],opt);model._deformation.eval()
    bg=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device='cuda')
    records=[]
    with torch.no_grad():
        for group,frames in [('test',evaluation['frames']),('input_fit',train['frames'])]:
            for i,frame in enumerate(frames):
                camera=CalibratedCamera(frame,i,load_rgb=False)
                ret=render(camera,model,pipe,bg,stage='fine',cam_type='explicit_protocol')
                rgb=ret['render'].permute(1,2,0).cpu().numpy();depth=ret['depth'].squeeze().cpu().numpy()
                assert rgb.shape==(frame['height'],frame['width'],3) and np.isfinite(rgb).all()
                dest=out/group/(frame['frame_id']+'.npz');dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest,rgb=rgb,depth=depth)
                visible=group=='test' or i in selected
                if visible:cv2.imwrite(str(dest.with_suffix('.png')),np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1])
                records.append(dict(group=group,frame_id=frame['frame_id'],render=identity(dest),source_frame=frame,visualization_selected=visible,raw_rgb_range=[float(rgb.min()),float(rgb.max())]))
                if (i+1)%32==0:print(group,i+1,'/',len(frames),flush=True)
    torch.cuda.synchronize()
    result=dict(status='completed',method='official_Wu_4DGS_train_RGB_triangulation_initialization',
        source_checkpoint=identity(checkpoint),effective_config=identity(run_dir/'effective_config.json'),all_finals_freeze=identity(freeze),
        official_commit=OFFICIAL_COMMIT,frame_count=len(records),frames=records,wall_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),optimization_steps=0,final_points=len(model.get_xyz),
        RGB_GT_read=False,white_background=bool(dataset.white_background),
        float_render_policy='raw float NPZ; no clipping before metric. Optional PNG clipped for display',
        input_fit_scope='all268 frozen training frames; uniform16 preview choice independent of result',
        H0_comparison='separate table; H0 full native preprocessing and historical split cannot be proven input equivalent')
    save_json(out/'manifest.json',result);print(json.dumps({k:result[k] for k in ['status','frame_count','wall_seconds','peak_allocated_bytes','final_points']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);a=p.parse_args();run(a.run_dir,a.freeze)
