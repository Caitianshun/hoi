"""Fixed training-only contribution VJPs and time/camera sensitivity."""
from common import *
import argparse,torch,numpy as np,cv2,copy,math,traceback
from PIL import Image,ImageDraw
from render_utils import load_model,CalibratedCamera
from diagnostic_render import make_renderer,diff
from budget import CallBudget
def run(arm):
    torch.set_num_threads(4);budget=CallBudget('D1_'+arm)
    base=RUN if arm=='C_route' else V5;rd=base/'runs'/arm;r=read(rd/'run.json')
    model,(_,hidden,_,pipe),state=load_model(r['checkpoint'],rd/'effective_config.json');del state
    bound=read(rd/'effective_config.json')['scale_bound'];render=make_renderer(bound);bg=torch.zeros(3,device='cuda')
    frames=read(OLD/'inputs/hos_backpack/manifest.json')['frames'];by={f['frame_id']:f for f in frames}
    ids=read(RUN/'protocol/fixed_examples.json')['training_frame_ids'];assert ids==['00001','00041','00081','00122','00162','00202','00243','00283']
    out=RUN/'diagnostics'/arm;out.mkdir(exist_ok=False)
    contribution=[];probes=[];fgs=[];bgs=[];bound_counts=[]
    row_ids=np.arange(len(model.get_xyz),dtype=np.int64);sample_ids=row_ids[::997]
    for fid in ids:
        f=by[fid];other=min((x for x in frames if x['frame_id']!=fid),key=lambda x:(abs(int(x['frame_id'])-int(fid)),int(x['frame_id'])))
        cami=CalibratedCamera(f,0,False);camj=CalibratedCamera(other,1,False);packages=[]
        for cam,t in [(cami,f['time']),(cami,other['time']),(camj,f['time']),(camj,other['time'])]:
            c=copy.copy(cam);c.time=t
            with torch.no_grad():p=render(c,model,pipe,bg)
            packages.append(p)
        p=packages[0];mask=torch.tensor(cv2.imread(f['mask_path'],0)>=128,device='cuda')
        try:
            colors=torch.ones_like(model.get_xyz,requires_grad=True)
            color=render(cami,model,pipe,bg,override_color=colors,detach_attributes=True)
            alpha=color['render'][0]
            total=torch.autograd.grad(alpha.sum(),colors,retain_graph=True)[0][:,0].detach()
            fg=torch.autograd.grad(alpha[mask].sum(),colors)[0][:,0].detach();back=total-fg
            checks=dict(channels_max_abs=float((color['render']-alpha[None]).abs().max()),radii_equal=torch.equal(p['radii'],color['radii']),geometry_max_abs=diff(p['xyz_final'],color['xyz_final'])['max_abs'],min_total=float(total.min()),min_fg=float(fg.min()),min_bg=float(back.min()),sum_total=float(total.double().sum()),sum_fg=float(fg.double().sum()),alpha_total=float(alpha.detach().double().sum()),alpha_fg=float(alpha.detach()[mask].double().sum()),FG_BG_max_abs=float((fg+back-total).abs().max()))
            checks['sum_total_relative_error']=abs(checks['sum_total']-checks['alpha_total'])/max(1,checks['alpha_total']);checks['sum_fg_relative_error']=abs(checks['sum_fg']-checks['alpha_fg'])/max(1,checks['alpha_fg'])
            assert checks['channels_max_abs']==0 and checks['radii_equal'] and checks['geometry_max_abs']==0
            assert checks['min_total']>=0 and checks['min_fg']>=0 and checks['min_bg']>=-1e-5
            assert checks['sum_total_relative_error']<5e-5 and checks['sum_fg_relative_error']<5e-5
            fa=fg.cpu().numpy();ba=back.cpu().numpy();ta=total.cpu().numpy();active=np.flatnonzero((fa!=0)|(ba!=0)).astype(np.int32)
            path=out/(fid+'_contributions.npz');np.savez_compressed(path,row_ids=active,fg=fa[active],bg=ba[active],total_rows=np.int64(len(ta)))
            hit=(p['deformed_log_scale_raw']>math.log(bound)).any(dim=1)
            contribution.append(dict(frame_id=fid,arrays=identity(path),encoding='lossless sparse float32; absent rows exactly zero',checks=checks,fg_contribution_fraction=checks['sum_fg']/checks['sum_total'],bound_rows=int(hit.sum()),bound_contribution_total=float(total[hit].double().sum()),bound_contribution_fg=float(fg[hit].double().sum()),scale_quantiles=quantiles(p['rendered_scale_bounded']),axis_ratio_quantiles=quantiles(p['rendered_scale_bounded'].max(1).values/p['rendered_scale_bounded'].min(1).values.clamp_min(1e-30)),displacement_quantiles=quantiles((p['xyz_final']-model.get_xyz).norm(dim=1))))
            fgs.append(fa);bgs.append(ba);bound_counts.append(hit.cpu().numpy())
            del color,alpha,total,fg,back,colors
        except Exception:
            save_json(out/'contribution_failure.json',dict(frame_id=fid,traceback=traceback.format_exc(),status='NA_interface_did_not_pass; no training implication'));raise
        # Same canonical rows within this model; never compare row identity across arms.
        arrays=dict(row_ids=sample_ids)
        for n,x in [('ti',packages[0]),('tj',packages[1])]:
            for key in ['xyz_final','deformed_log_scale_raw','rendered_scale_bounded']:arrays[n+'_'+key]=x[key][sample_ids].cpu().numpy()
            xyz=x['xyz_final'][sample_ids];ones=torch.ones((len(xyz),1),device=xyz.device)
            clip=torch.cat([xyz,ones],1)@cami.full_proj_transform.cuda();arrays[n+'_screen_xy']=((clip[:,:2]/clip[:,3:4]+1)*torch.tensor([cami.image_width,cami.image_height],device=xyz.device)-1).cpu().numpy()/2
        ap=out/(fid+'_fixed_rows.npz');np.savez_compressed(ap,**arrays)
        images=[x['render'].permute(1,2,0).cpu().numpy() for x in packages]
        diagonal=[]
        for idx,frame in [(0,f),(3,other)]:
            gt=cv2.imread(frame['image_path'])[...,::-1].astype(np.float32)/255
            mse=float(np.mean((np.clip(images[idx],0,1).astype(np.float64)-gt)**2));diagonal.append(dict(frame_id=frame['frame_id'],full_PSNR=-10*math.log10(mse)))
        sensitivity=dict(change_time_fixed_Ci=float(np.abs(images[1]-images[0]).mean()),change_camera_fixed_ti=float(np.abs(images[2]-images[0]).mean()),change_time_fixed_Cj=float(np.abs(images[3]-images[2]).mean()))
        canvas=Image.new('RGB',(1280,204),'white');draw=ImageDraw.Draw(canvas)
        for k,img in enumerate(images):
            pic=Image.fromarray(np.rint(np.clip(img,0,1)*255).astype('uint8')).resize((320,180))
            canvas.paste(pic,(320*k,24));draw.text((320*k+5,6),['Ci ti (observed)','Ci tj (NO GT)','Cj ti (NO GT)','Cj tj (observed)'][k],fill='black')
        png=out/(fid+'_time_camera.png');canvas.save(png)
        probes.append(dict(frame_id=fid,neighbor=other['frame_id'],time_delta=other['time']-f['time'],selection='nearest other training source index; tie -> smaller index',diagonal_metrics=diagonal,cross_metrics=None,cross_interpretation='model sensitivity only; no GT',sensitivity=sensitivity,fixed_rows=identity(ap),figure=identity(png)))
        del packages,p,images;torch.cuda.empty_cache()
    F=np.stack(fgs);B=np.stack(bgs);T=F+B;den=T.sum(0,dtype=np.float64);supported=den>0
    P=np.zeros_like(T);P[:,supported]=(T[:,supported]/den[supported]).astype(np.float32)
    neff=np.zeros(len(den),np.float32);neff[supported]=1/(P[:,supported]**2).sum(0)
    np.savez_compressed(out/'effective_time.npz',row_ids=row_ids,supported=supported,N_eff=neff,total_contribution=den.astype(np.float32),fg_fraction=np.divide(F.sum(0),den,out=np.zeros_like(den),where=supported).astype(np.float32))
    save_json(out/'contribution_manifest.json',dict(status='passed',arm=arm,checkpoint=identity(r['checkpoint']),rows=contribution,topology_rows=len(row_ids),N_eff_supported=quantiles(neff[supported]),unsupported_in_sample=int((~supported).sum()),N_eff_scope='only eight fixed training times; not true visibility count',physical_surface_correspondence=None,physical_correspondence_NA_reason='No independent material point tracks or geometry reference in this round',summary_arrays=identity(out/'effective_time.npz')))
    save_json(out/'time_camera_probe.json',dict(arm=arm,rows=probes,GT_only_diagonal=True,no_optimization=True))
    print('D1 complete',arm)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['B_U','B_F','C_route']);run(p.parse_args().arm)
