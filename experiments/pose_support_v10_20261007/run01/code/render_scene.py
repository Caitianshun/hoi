"""One joint raster with time-deformed base and articulated support covariance."""
from common import *
import math
import torch
from state import *
def rgb_background(ds,device='cuda'):
    assert ds.white_background;return torch.ones(3,dtype=torch.float32,device=device)
def packed_covariance(cov):return torch.stack([cov[:,0,0],cov[:,0,1],cov[:,0,2],cov[:,1,1],cov[:,1,2],cov[:,2,2]],dim=-1).contiguous()
def base_attributes(base,time,bound):
    xyz=base.get_xyz;t=torch.full((len(xyz),1),float(time),device=xyz.device,dtype=xyz.dtype)
    means,logs,q,opacity,sh=base._deformation(xyz,base._scaling,base._rotation,base._opacity,base.get_features,t)
    scale=torch.exp(logs.clamp(max=math.log(bound)))
    r,x,y,z=torch.nn.functional.normalize(q,dim=-1).unbind(-1)
    rotation=torch.stack((1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
        2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),
        2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)),dim=-1).reshape(-1,3,3)
    L=rotation*scale[:,None,:];cov=L@L.transpose(1,2)
    return dict(means3D=means,cov3D_precomp=packed_covariance(cov),shs=sh,opacities=base.opacity_activation(opacity),
        rendered_scale_bounded=scale,rotation_final=q,opacity_final=opacity)
def make_renderer(bound,mode=None):
    def render(camera,base,pipe,bg,stage='fine',support=None,bones=None,scene_extent=None):
        from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
        assert stage=='fine' and not pipe.compute_cov3D_python and not pipe.convert_SHs_python
        a=base_attributes(base,camera.time,bound);nbase=len(a['means3D'])
        attrs={k:a[k] for k in ('means3D','cov3D_precomp','shs','opacities')}
        if support is not None and len(support.u):
            assert bones is not None
            extra=support(bones,camera.time)
            for key in attrs:attrs[key]=torch.cat([attrs[key],extra[key]],dim=0)
        screen=torch.zeros_like(attrs['means3D'],requires_grad=True)+0
        if screen.requires_grad:screen.retain_grad()
        settings=GaussianRasterizationSettings(image_height=int(camera.image_height),image_width=int(camera.image_width),
            tanfovx=math.tan(camera.FoVx*.5),tanfovy=math.tan(camera.FoVy*.5),bg=bg,
            scale_modifier=1.,viewmatrix=camera.world_view_transform.cuda(),projmatrix=camera.full_proj_transform.cuda(),
            sh_degree=base.active_sh_degree if support is None else max(3,base.active_sh_degree),
            campos=camera.camera_center.cuda(),prefiltered=False,debug=pipe.debug)
        image,radii,depth=GaussianRasterizer(raster_settings=settings)(means3D=attrs['means3D'],means2D=screen,
            shs=attrs['shs'],colors_precomp=None,opacities=attrs['opacities'],scales=None,rotations=None,cov3D_precomp=attrs['cov3D_precomp'])
        return dict(render=image,depth=depth,radii=radii,visibility_filter=radii>0,viewspace_points=screen,
            base_count=nbase,xyz_final=attrs['means3D'],cov3D_precomp=attrs['cov3D_precomp'],
            rendered_scale_bounded=a['rendered_scale_bounded'],rotation_final=a['rotation_final'],opacity_final=a['opacity_final'])
    return render
def render_all():
    from hoi_modules.pose_prior_adapter import PosePriorAdapter
    import cv2,numpy as np
    freeze=read(RUN/'protocol/terminal_freeze.json')
    assert freeze['status']=='all_six_terminals_frozen'
    for scene in config()['scenes']:
        prior=PosePriorAdapter.from_cache(scene_dir(scene)/'protocol/pose_cache.pt')
        fixed_path=ROOT/config()['scenes'][scene]['historical_dir']/'protocol/fixed_examples.json'
        fixed=read(fixed_path)
        selected=set(fixed.get('training_frame_ids',[]))
        for arm in ['PARENT','C','P','PQ']:
            if arm=='PARENT':base,objs,meta=import_parent(scene);support=None;asset=meta['parent']
            else:
                receipt=read(scene_dir(scene)/'runs'/arm/'run.json');asset=receipt['checkpoint'];assert sha(asset['path'])==asset['sha256']
                base,support,optimizer,objs,meta=restore(torch.load(asset['path'],map_location='cpu',weights_only=False))
            out=RUN/'evaluation'/scene/arm
            if (out/'manifest.json').exists():
                manifest=read(out/'manifest.json');assert manifest['checkpoint']==asset
                assert all(sha(x['render']['path'])==x['render']['sha256'] for x in manifest['rows']);continue
            out.mkdir(parents=True,exist_ok=True);rows=[];start=time.monotonic();render=make_renderer(meta['scale_bound']);bg=rgb_background(objs[0])
            base._deformation.eval()
            if support is not None:support.eval()
            with torch.no_grad():
                for split in ('retained','train'):
                    cams,frames=load_scene_cameras(scene,split,load_rgb=False)
                    for cam,f in zip(cams,frames):
                        bones=None if support is None else prior.bone_transforms(f['frame_id'],device='cuda')
                        pkg=render(cam,base,objs[3],bg,support=support,bones=bones,scene_extent=meta['scene_extent'])
                        rgb=pkg['render'].permute(1,2,0).cpu().numpy();assert rgb.dtype==np.float32 and np.isfinite(rgb).all()
                        path=out/split/(f['frame_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(path,rgb=rgb,depth=pkg['depth'].squeeze().cpu().numpy())
                        row=dict(scene=scene,run=arm,split=split,frame_id=f['frame_id'],render=identity(path),source_frame=f)
                        if split=='retained' or f['frame_id'] in selected:
                            png=path.with_suffix('.png');cv2.imwrite(str(png),np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1]);row['display_png']=identity(png)
                        rows.append(row)
            save_json(out/'manifest.json',dict(status='completed',scene=scene,arm=arm,rows=rows,seconds=time.monotonic()-start,
                checkpoint=asset,freeze=identity(RUN/'protocol/terminal_freeze.json'),GT_read=False,query_mask_read=False))
            del base,support;torch.cuda.empty_cache()
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');a=p.parse_args();assert a.all
    torch.set_num_threads(4);render_all()
