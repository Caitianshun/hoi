"""Render one terminal official 4DGS run only after BEHAVE pair freeze.

Exports raw floating-point RGB without GT reads. PNG is clipped display only;
metrics must consume NPZ. All original train frames are exported for input fit.
"""
from pathlib import Path
import argparse,json,os,time
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from adapter_4dgs import RUN,OFFICIAL_COMMIT,CalibratedCamera,official_config,sha,save_json


def identity(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def render_final(dev,run_dir,freeze):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1','Physical GPU1 is the authorized device'
    frozen=json.loads(Path(freeze).read_text())
    assert frozen.get('status')=='all_states_frozen' and frozen.get('scope')=='behave_pair'
    assets={str(Path(a['path']).resolve()):a['sha256'] for a in frozen['assets']}
    for path,digest in assets.items():assert sha(path)==digest,f'Frozen asset changed: {path}'
    run_dir=Path(run_dir).resolve();terminal=json.loads((run_dir/'run.json').read_text())
    assert terminal['status']=='completed' and terminal['nominal_iterations']==17000
    checkpoint=Path(terminal['checkpoint']).resolve()
    assert assets.get(str(checkpoint))==terminal['checkpoint_sha256'],'Checkpoint must be a frozen terminal asset'
    cfg=json.loads((run_dir/'effective_config.json').read_text())
    assert sha(run_dir/'effective_config.json')==terminal['source_config_sha256']
    assert cfg['official_commit']==OFFICIAL_COMMIT and cfg['seed']==12345 and not cfg['check']
    base=RUN/'inputs'/f'behave_{dev}';train=json.loads((base/'manifest.json').read_text())
    assert cfg['manifest_sha256']==sha(base/'manifest.json')
    evaluation=json.loads((base/'evaluation_manifest.json').read_text())
    assert train['role']=='training_only' and evaluation['role']=='evaluation_only'
    selected=np.unique(np.linspace(0,len(train['frames'])-1,min(16,len(train['frames']))).astype(int)).tolist()
    groups={'camera1_E':evaluation['frames'],'camera0_paired_E':evaluation['paired_camera0_frames'],
            'camera0_full_training_fit':train['frames']}
    out=RUN/'evaluation'/'4DGS'/dev
    if out.exists():raise RuntimeError(f'Refusing to overwrite rendered terminal state: {out}')
    out.mkdir(parents=True)
    torch.set_num_threads(4);start=time.perf_counter()
    # Parse locked official defaults, then use the exact saved effective values.
    _,(dataset,hidden,opt,pipe)=official_config(cfg['optimization']['batch_size'])
    for obj,key in [(dataset,'model'),(hidden,'hidden'),(opt,'optimization'),(pipe,'pipeline')]:
        for name,value in cfg[key].items():setattr(obj,name,value)
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    model=GaussianModel(dataset.sh_degree,hidden)
    model._deformation=model._deformation.cuda()
    state=torch.load(checkpoint,map_location='cuda',weights_only=False)
    assert state['stage']=='fine' and state['iteration']==14000
    assert state['total_nominal_steps']==17000
    model.restore(state['model'],opt)
    # restore creates optimizer state for exact upstream loading. No backward or
    # step is performed; retaining it is harmless and preserves native semantics.
    model._deformation.eval()
    bg=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device='cuda')
    records=[]
    with torch.no_grad():
        for label,frames in groups.items():
            for i,frame in enumerate(frames):
                camera=CalibratedCamera(frame,i,load_rgb=False)
                ret=render(camera,model,pipe,bg,stage='fine',cam_type='explicit_protocol')
                rgb=ret['render'].permute(1,2,0).cpu().numpy()
                depth=ret['depth'].squeeze().cpu().numpy()
                assert rgb.shape==(frame['height'],frame['width'],3) and np.isfinite(rgb).all()
                dest=out/label/f'{frame["frame_id"]}.npz';dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest,rgb=rgb,depth=depth)
                png=dest.with_suffix('.png')
                assert cv2.imwrite(str(png),np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1])
                records.append(dict(group=label,frame_id=frame['frame_id'],time_seconds=frame['time_seconds'],render=identity(dest),
                    png=identity(png),source_frame=frame,visualization_selected=(label!='camera0_full_training_fit' or i in selected),
                    raw_rgb_range=[float(rgb.min()),float(rgb.max())]))
    torch.cuda.synchronize()
    report=dict(status='completed',dev=dev,method='official_Wu_4DGS_shared_prior_initialization',
        source_checkpoint=identity(checkpoint),effective_config=identity(run_dir/'effective_config.json'),
        all_finals_freeze=identity(freeze),official_commit=OFFICIAL_COMMIT,
        frame_count=len(records),frames=records,wall_seconds=time.perf_counter()-start,optimization_steps=0,
        final_points=len(model.get_xyz),RGB_GT_read=False,white_background=bool(dataset.white_background),
        float_render_policy='NPZ RGB is unrounded raw float; PNG is clipped to [0,1] for display only',
        alpha='not returned by official backend; omitted, never fabricated',
        fit_frame_selection='all original 114/98 camera0 frames; preview only uniform at most16; paired E kept separately')
    save_json(out/'manifest.json',report)
    print(json.dumps({k:report[k] for k in ['dev','frame_count','wall_seconds','final_points']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dev',required=True,choices=['dev1','dev2']);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True)
    a=p.parse_args();render_final(a.dev,a.run_dir,a.freeze)
