"""Frozen BEHAVE SH-direction and alpha diagnostics, without optimization."""
from common import *
from render_utils import diagnostic_renderer,load_model,CalibratedCamera
import torch,numpy as np,cv2,time,os
import gaussian_renderer
def camera_check(frame,xyz):
    camera=CalibratedCamera(frame,0,load_rgb=False);c2w=np.asarray(frame['c2w']);w2c=np.asarray(frame['w2c']);center=camera.camera_center.numpy()
    cx=np.c_[xyz,np.ones(len(xyz))]@w2c.T;good=cx[:,2]>.01
    cv=cx[good,:3]@np.asarray(frame['K']).T;uv=cv[:,:2]/cv[:,2:]
    clip=np.c_[xyz[good],np.ones(good.sum())]@camera.full_proj_transform.numpy();pix=((clip[:,:2]/clip[:,3:]+1)*[frame['width'],frame['height']]-1)*.5
    delta=float(np.max(np.abs(pix-uv)));assert delta<.5
    return dict(frame_id=frame['frame_id'],camera_id=frame.get('camera_id'),max_projection_error_px=delta,camera_center_max_error_m=float(np.max(np.abs(center-c2w[:3,3]))),w2c_center_max_error_m=float(np.max(np.abs((w2c@np.r_[center,1])[:3]))),checked_points=int(good.sum()))
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';torch.set_num_threads(4);started=time.monotonic();render,patch=diagnostic_renderer()
    save_json(RUN/'protocol/diagnostic_renderer_patch.json',patch);out=RUN/'diagnostics/appearance';out.mkdir(exist_ok=True);records=[];checks=[]
    for dev in ['dev1','dev2']:
        rd=OLD/'runs'/f'behave_{dev}_formal';terminal=read(rd/'run.json');assert sha(terminal['checkpoint'])==terminal['checkpoint_sha256']
        model,(dataset,hidden,opt,pipe),state=load_model(terminal['checkpoint'],rd/'effective_config.json');assert state['stage']=='fine' and state['iteration']==opt.iterations
        before=identity(terminal['checkpoint']);del state
        ev=read(OLD/'inputs'/f'behave_{dev}'/'evaluation_manifest.json');train=read(OLD/'inputs'/f'behave_{dev}'/'manifest.json');xyz=np.load(train['point_cloud']['npz_path'])['xyz'][::17].astype(np.float64)
        bg=torch.tensor([1,1,1] if dataset.white_background else [0,0,0],dtype=torch.float32,device='cuda');black=torch.zeros(3,device='cuda')
        cam0={f['frame_id']:CalibratedCamera(f,i,False) for i,f in enumerate(ev['paired_camera0_frames'])}
        old_index={(x['group'],x['frame_id']):x for x in read(OLD/'evaluation/4DGS'/dev/'manifest.json')['frames']}
        with torch.no_grad():
            for group,frames in [('camera1_E',ev['frames']),('camera0_paired_E',ev['paired_camera0_frames'])]:
                for i,f in enumerate(frames):
                    checks.append(dict(dev=dev,group=group,**camera_check(f,xyz)));camera=CalibratedCamera(f,i,False)
                    native=gaussian_renderer.render(camera,model,pipe,bg,stage='fine');native_rgb=native['render'].permute(1,2,0).cpu().numpy()
                    old=old_index[group,f['frame_id']]['render'];assert sha(old['path'])==old['sha256'];previous=np.load(old['path'])['rgb'];old_delta=float(np.abs(native_rgb-previous).max());assert old_delta<=1e-5
                    reference_alpha=None
                    for mode in ['native','dc_only','cam0_direction']:
                        kw={} if mode=='native' else dict(eval_sh_degree=0) if mode=='dc_only' else dict(sh_camera_center=cam0[f['frame_id']].camera_center)
                        result=render(camera,model,pipe,bg,stage='fine',**kw);rgb=result['render'].permute(1,2,0).cpu().numpy()
                        delta=float(np.max(np.abs(rgb-native_rgb)))
                        if mode=='native' or (group=='camera0_paired_E' and mode=='cam0_direction'):assert delta<=1e-5
                        assert torch.equal(result['radii'],native['radii']) and torch.equal(result['visibility_filter'],native['visibility_filter'])
                        depth_delta=float((result['depth']-native['depth']).abs().max());assert depth_delta<=1e-5
                        a=render(camera,model,pipe,black,stage='fine',alpha_only=True,**kw)['render'];alpha=a[0].cpu().numpy();assert float((a-a[0:1]).abs().max())<=1e-6
                        if reference_alpha is None:reference_alpha=alpha
                        alpha_delta=float(np.abs(alpha-reference_alpha).max());assert alpha_delta<=1e-6
                        bg_identity=None
                        if i==0:
                            premult=render(camera,model,pipe,black,stage='fine',**kw)['render'].permute(1,2,0).cpu().numpy()
                            bg_identity=float(np.abs(rgb-(premult+(1-alpha[...,None])*bg.cpu().numpy())).max());assert bg_identity<=1e-5
                        dest=out/dev/group/f['frame_id']/f'{mode}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,rgb=rgb,alpha=alpha,depth=result['depth'].squeeze().cpu().numpy(),radii=result['radii'].cpu().numpy(),visibility=result['visibility_filter'].cpu().numpy())
                        png=dest.with_suffix('.png');cv2.imwrite(str(png),np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1])
                        visible=result['visibility_filter'];scale=result['scales_final'][visible].max(1).values
                        records.append(dict(dev=dev,group=group,frame_id=f['frame_id'],mode=mode,render=identity(dest),png=identity(png),source_frame=f,
                            native_old_max_abs=old_delta,mode_native_max_abs=delta,depth_max_abs_delta=depth_delta,alpha_max_abs_delta=alpha_delta,
                            background_rgb=bg.cpu().tolist(),background_compositing_max_abs=bg_identity,visible_scale=quantiles(scale),visible_radius=quantiles(result['radii'][visible]),
                            visible_gaussians=int(visible.sum()),checkpoint=before))
                    print(dev,group,f['frame_id'],'verified',flush=True)
        assert identity(terminal['checkpoint'])==before;del model;torch.cuda.empty_cache()
    save_json(out/'manifest.json',dict(status='completed',records=records,camera_checks=checks,seconds=time.monotonic()-started,optimization_steps=0,all_default_and_geometry_parities_passed=True,
        scope='Frozen-render counterfactuals only, not new method scores or SH0-trained models',source=identity(__file__)))
if __name__=='__main__':run()
