"""Bounded official RGB renderer and detached-geometry projected motion moments.

No experiment paths, optimization, density updates, or row-identity state.
The auxiliary render uses the installed 4DGS rasterizer, including signed colors.
"""
import inspect,math
import torch
import torch.nn.functional as F

def make_renderer(bound):
    import gaussian_renderer
    original=inspect.getsource(gaussian_renderer.render);source=original
    edits={
        'scales_final = pc.scaling_activation(scales_final)': 'canonical_log_scale_raw=pc._scaling\n    deformed_log_scale_raw=scales_final\n    scales_final = torch.exp(torch.clamp(scales_final,max=math.log(bound)))',
        'shs = shs_final,':'shs = shs_final if override_color is None else None,',
        '"depth":depth}':'"depth":depth, "canonical_log_scale_raw":canonical_log_scale_raw, "deformed_log_scale_raw":deformed_log_scale_raw, "rendered_scale_bounded":scales_final, "xyz_final":means3D_final, "rotation_final":rotations_final, "opacity_final":opacity}'
    }
    for a,b in edits.items():assert source.count(a)==1;a_count=source.count(a);source=source.replace(a,b)
    ns=gaussian_renderer.__dict__.copy();ns['bound']=bound;exec(compile(source,'<bounded_official_rgb>','exec'),ns)
    fn=ns['render'];fn.expanded_source=source;fn.original_source=original
    return fn

def deform_attributes(model,time):
    xyz=model.get_xyz
    t=torch.full((len(xyz),1),float(time),device=xyz.device,dtype=xyz.dtype)
    # Position path cannot use appearance, opacity or shape gradients.
    return model._deformation(xyz,model._scaling.detach(),model._rotation.detach(),model._opacity.detach(),model.get_features.detach(),t)

def project(xyz,camera):
    h=torch.cat([xyz,torch.ones_like(xyz[:,:1])],dim=1)
    clip=h@camera.full_proj_transform.to(xyz.device)
    view=h@camera.world_view_transform.to(xyz.device)
    valid=torch.isfinite(clip).all(-1)&torch.isfinite(view).all(-1)&(view[:,2]>0)&(clip[:,3]>1e-8)
    safe=torch.where(valid[:,None],clip,torch.zeros_like(clip))
    den=torch.where(valid,clip[:,3],torch.ones_like(clip[:,3]))
    ndc=safe[:,:2]/den[:,None]
    size=xyz.new_tensor([camera.image_width,camera.image_height])
    pixels=((ndc+1)*size-1)/2
    return pixels,valid

def raster_attributes(camera,attributes,colors,bound,sh_degree=0,debug=False):
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    xyz,log_scale,rotation,log_opacity=attributes[:4]
    if not all(torch.isfinite(x.detach()).all() for x in [xyz,log_scale,rotation,log_opacity]):
        raise FloatingPointError('Nonfinite source geometry before auxiliary raster')
    settings=GaussianRasterizationSettings(image_height=int(camera.image_height),image_width=int(camera.image_width),tanfovx=math.tan(camera.FoVx*.5),tanfovy=math.tan(camera.FoVy*.5),bg=torch.zeros(3,device=xyz.device,dtype=xyz.dtype),scale_modifier=1.,viewmatrix=camera.world_view_transform.to(xyz.device),projmatrix=camera.full_proj_transform.to(xyz.device),sh_degree=sh_degree,campos=camera.camera_center.to(xyz.device),prefiltered=False,debug=debug)
    # Unique auxiliary screen tensor with no gradient. No RGB q reuse.
    screen=torch.zeros_like(xyz,requires_grad=False)
    image,radii,depth=GaussianRasterizer(raster_settings=settings)(means3D=xyz.detach(),means2D=screen,shs=None,colors_precomp=colors,opacities=torch.sigmoid(log_opacity.detach()),scales=torch.exp(log_scale.detach().clamp(max=math.log(bound))),rotations=F.normalize(rotation.detach(),dim=-1),cov3D_precomp=None)
    return image

def sample_accumulated(image,uv):
    h,w=image.shape[-2:];size=uv.new_tensor([w,h]);grid=2*(uv+.5)/size-1
    return F.grid_sample(image[None],grid.reshape(1,-1,1,2),mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,:,0].T

def motion_moments(model,source_camera,target_camera,queries,bound,second=True):
    source=deform_attributes(model,source_camera.time);target=deform_attributes(model,target_camera.time)
    ps,vs=project(source[0],source_camera);pt,vt=project(target[0],target_camera)
    valid=vs&vt
    size=ps.new_tensor([source_camera.image_width,source_camera.image_height]);d=torch.where(valid[:,None],(pt-ps)/size,torch.zeros_like(ps))
    # Invalid displacement has zero mass but retains source occlusion/transmittance.
    colors=torch.cat([d,valid[:,None].to(d.dtype)],dim=1)
    first=raster_attributes(source_camera,source,colors,bound,model.active_sh_degree)
    samples=sample_accumulated(first,queries);alpha=samples[:,2].detach();mu=samples[:,:2]/alpha[:,None].clamp_min(1e-6)
    with torch.no_grad():
        if second:
            sq=torch.cat([d.detach().square(),torch.zeros_like(d[:,:1])],dim=1)
            second_img=raster_attributes(source_camera,source,sq,bound,model.active_sh_degree)
            m2=sample_accumulated(second_img,queries)[:,:2]/alpha[:,None].clamp_min(1e-6)
            variance=((m2-mu.detach().square()).clamp_min(0)*size.square()).sum(-1)
        else:variance=torch.zeros_like(alpha)
    return mu,alpha,variance,dict(invalid_gaussians=int((~valid).sum()),gaussians=len(valid))
