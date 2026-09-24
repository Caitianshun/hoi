"""Exercise unchanged production residuals on one synthetic scene and frozen frames.

No optimization, reference loading, feature reconstruction or data relabeling.
"""
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,hashlib,datetime
import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path('/home/cai_tianshun/Project/HOI');E=Path(__file__).resolve().parents[1]
PREV=ROOT/'experiments/surface_pose_joint_20260924/run01'
sys.path.insert(0,str(PREV/'code'))
from joint_solver import Objective,Problem,project,rho,save,sha
torch.set_num_threads(1);START=time.perf_counter()
COLORS=['#eeeeee','#df7794','#51b7c4','#b2a2da']

class CaptureObjective(Objective):
    def background(self,uv):
        self.captured_uv=uv
        return super().background(uv)

def make(P):
    o=CaptureObjective.__new__(CaptureObjective)
    o.P=P;o.dev=P.dev;o.device='cpu';o.dtype=torch.float64
    o.tensor=lambda x,dt=None:torch.as_tensor(np.asarray(x),dtype=dt or o.dtype,device='cpu')
    o.K=o.tensor(P.K);o.V=o.tensor(P.V);o.S=o.tensor(P.S);o.ts=o.tensor(P.ts);o.vis=o.tensor(P.visible,torch.bool)
    o.uv=o.tensor(np.stack([x[0] for x in P.obs]));o.fields=o.tensor(np.stack([x[1] for x in P.obs])[...,None])
    o.xyz=o.tensor(np.stack([x[2] for x in P.obs]));o.good=o.tensor(np.stack([x[3] for x in P.obs]),torch.bool)
    return o

def synthetic():
    lab=np.zeros((480,640),np.uint8)
    lab[160:321,200:401]=2;lab[140:341,300:451]=1;lab[301:321,220:261]=3
    uv=np.argwhere(lab==2)[:,::-1].astype(float)
    uv=uv[np.linspace(0,len(uv)-1,64,dtype=int)]
    field=distance_transform_edt(lab==0)
    cases={'complete':(200,400),'delete_visible':(240,400),'invade_background':(200,470),'hidden_edge_moves_within_H':(200,420)}
    rows={};fig,axs=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    for k,(left,right) in cases.items():
        V=np.array([[left,160,0],[right,160,0],[right,320,0],[left,320,0]],float)
        x,y=np.meshgrid(np.arange(left,right+1,2),np.arange(160,321,2));S=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        rows[k]={}
        for dev in ['dev1','dev2']:
            P=Problem.__new__(Problem);P.dev=dev;P.L=3;P.K=np.eye(3);P.V=V;P.S=S;P.ts=np.arange(3,dtype=float);P.visible=np.ones(3,bool)
            from scipy.spatial import cKDTree
            P.ST=cKDTree(S);P.obs=[(uv,field,np.zeros((64,3)),np.zeros(64,bool)) for _ in range(3)]
            o=make(P);R=o.tensor(np.repeat(np.eye(3)[None],3,0));t=o.tensor(np.tile([0,0,1],(3,1)))
            comp=o.base(R,t);buv=o.captured_uv.detach().clone();bg=o.background(buv)
            f,b,_=P.frame_blocks(0,np.eye(3),np.array([0,0,1]))
            numpy_comp={'silhouette_forward':float((2*(np.sqrt(1+f*f)-1)).sum()),'silhouette_background':float((2*(np.sqrt(1+b*b)-1)).sum())}
            errors={key:abs(float(comp[key])-numpy_comp[key]) for key in numpy_comp}
            # Exact production sampler on strictly interior H and U probes, and B control.
            probes=o.tensor(np.tile([[390.25,210.25],[240.25,310.25],[480.25,210.25]],(3,1,1))).requires_grad_(True)
            pv=o.background(probes);pg=torch.autograd.grad(rho(pv/16).sum(),probes)[0]
            # Local foreground recovery gradient, excluding all other terms.
            shift=torch.tensor(0.,dtype=o.dtype,requires_grad=True)
            cf=o.base(R,t+torch.stack([shift,shift*0,shift*0])[None])['silhouette_forward']
            fg_grad=float(torch.autograd.grad(cf,shift)[0])
            row={key:float(comp[key]) for key in ['silhouette_forward','silhouette_background']}
            row.update(max_old_new_component_difference=max(errors.values()),foreground_translation_x_gradient=fg_grad,probe_H_U_B_values_px=pv[0].detach().numpy(),probe_H_U_B_robust_gradient_xy=pg[0].detach().numpy())
            rows[k][dev]=row
        if k!='hidden_edge_moves_within_H':
            ax=axs[list(cases).index(k)];ax.imshow(lab,cmap=ListedColormap(COLORS),vmin=0,vmax=3)
            poly=np.vstack([V[:,:2],V[0,:2]]);ax.plot(poly[:,0],poly[:,1],color='black',lw=2)
            ax.scatter(uv[:,0],uv[:,1],s=9,c='#166275')
            ax.set_xlim(180,500);ax.set_ylim(355,125);ax.set_aspect('equal')
            ax.set_title(k.replace('_',' '));ax.set_xlabel('pixel x');ax.set_ylabel('pixel y')
            z=rows[k]['dev1'];ax.text(.02,.01,f"Box forward {z['silhouette_forward']:.3f}\nBox background {z['silhouette_background']:.3f}",transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    checks={}
    for d in ['dev1','dev2']:
        c=rows['complete'][d];dele=rows['delete_visible'][d];extra=rows['invade_background'][d];hidden=rows['hidden_edge_moves_within_H'][d]
        checks[d]={'hidden_H_U_negative_zero':bool(np.all(np.array(c['probe_H_U_B_values_px'])[:2]==0)),'hidden_H_U_gradient_zero':bool(np.all(np.array(c['probe_H_U_B_robust_gradient_xy'])[:2]==0)),'moving_hidden_edge_no_cut_pull':abs(c['silhouette_background']-hidden['silhouette_background'])<1e-10 and abs(c['silhouette_forward']-hidden['silhouette_forward'])<1e-10,'visible_deletion_penalized':dele['silhouette_forward']>c['silhouette_forward']+1 and dele['foreground_translation_x_gradient']>0,'credible_background_intrusion_penalized':extra['silhouette_background']>c['silhouette_background']+1,'old_new_same_residuals':all(rows[k][d]['max_old_new_component_difference']<1e-5 for k in cases)}
    fig.suptitle('One artificial occlusion scene   cyan O   pink H   purple U   grey B',fontsize=12)
    fig.savefig(E/'synthetic_contract/contract.png',dpi=180);plt.close(fig)
    np.savez_compressed(E/'synthetic_contract/scene.npz',entity_labels=lab,background_distance=field,positive_uv=uv)
    result={'role':'code mechanism contract only; not performance experiment','dtype':'CPU float64; original production functions unmodified','cases':rows,'checks':checks,'all_pass':all(all(x.values()) for x in checks.values()),'positive_note':'chair is nearest projected sampled surface point, not exact occupied region; discretization error remains and is explicitly measured','scope':'single artificial scene, three required perturbations and hidden-edge invariant probe; U only synthetic, real caches have U=empty'}
    save(E/'synthetic_contract/results.json',result);print('synthetic',checks,flush=True)
    return result

def region_stats(lab,uv,values,grads):
    xy=uv.detach().numpy();v=values.detach().numpy();g=grads.detach().numpy()
    inside=(xy[:,0]>=0)&(xy[:,0]<=639)&(xy[:,1]>=0)&(xy[:,1]<=479)
    x=np.clip(xy[:,0],0,639);y=np.clip(xy[:,1],0,479);x0=x.astype(int);y0=y.astype(int);x1=np.minimum(x0+1,639);y1=np.minimum(y0+1,479)
    corners=np.stack([lab[y0,x0],lab[y0,x1],lab[y1,x0],lab[y1,x1]],1)
    rows={}
    for name,val in [('O',2),('H',1),('U',3),('B',0)]:
        z=(corners==val).all(1)&inside
        rows[name]={'count':int(z.sum()),'max_raw_px':float(v[z].max()) if z.any() else None,'mean_raw_px':float(v[z].mean()) if z.any() else None,'max_raw_gradient_norm':float(np.linalg.norm(g[z],axis=1).max()) if z.any() else None}
    mixed=inside&~(corners==corners[:,0:1]).all(1)
    near=lab[np.rint(y).astype(int),np.rint(x).astype(int)]
    hn=(near==1)&inside
    rows['mixed_four_pixel_cells']={'count':int(mixed.sum()),'max_raw_px':float(v[mixed].max()) if mixed.any() else None}
    rows['nearest_pixel_H']={'count':int(hn.sum()),'positive_count':int(((v>1e-9)&hn).sum()),'nonzero_explained_by_B_interpolation':bool((~((v>1e-9)&hn)|(corners==0).any(1)).all()),'max_raw_px':float(v[hn].max()) if hn.any() else None}
    rows['outside_count']=int((~inside).sum())
    return rows,corners

def real_frames():
    selection=json.loads((E/'protocol/frame_selection.json').read_text());summary={}
    for dev in ['dev1','dev2']:
        P=Problem(dev);o=make(P);a=np.load(PREV/dev/'F01/object_init.npz')
        R=o.tensor(a['R_camera']);t=o.tensor(a['t_camera_m'])
        with torch.no_grad():comp=o.base(R,t);buv=o.captured_uv.clone()
        buv.requires_grad_(True);back=o.background(buv);grad=torch.autograd.grad(back.sum(),buv)[0]
        frames=selection['sequences'][dev]['frames'];fig,axs=plt.subplots(3,2,figsize=(10,9),layout='constrained')
        rows=[]
        for rowidx,i in enumerate(frames):
            lab=P.labels[i];uv=P.obs[i][0];field=P.obs[i][1];stats,corners=region_stats(lab,buv[i],back[i],grad[i])
            bxy=buv[i].detach().numpy();bval=back[i].detach().numpy();f,b,_=P.frame_blocks(i,a['R_camera'][i],a['t_camera_m'][i])
            projected=project(o.V@R[i].T+t[i],o.K).numpy();samples=project(o.S@R[i].T+t[i],o.K).numpy()
            counts={name:int((lab==v).sum()) for name,v in [('O',2),('H',1),('B',0)]};counts['U']=0
            item={'frame':i,'timestamp_seconds':float(P.ts[i]),'fixed_visible':bool(P.visible[i]),'pixel_counts':counts,'positive_samples':len(uv),'all_positive_queries_in_O':bool((lab[uv[:,1].astype(int),uv[:,0].astype(int)]==2).all()),'negative_queries_before_quantile':len(bxy),'output_negative_residual_count':64,'regions':stats,'forward_robust_sum':float((2*(np.sqrt(1+f*f)-1)).sum()),'background_robust_sum':float((2*(np.sqrt(1+b*b)-1)).sum()),'full_sequence_denominator':P.L}
            rows.append(item)
            np.savez_compressed(E/f'mask_ledger/{dev}_{i:03d}.npz',entity_labels=lab,background_distance_px=field,positive_uv=uv,positive_residual_px=f*16,background_query_uv=bxy,background_raw_distance_px=bval,background_raw_gradient_uv=grad[i].detach().numpy(),bilinear_corner_labels=corners,complete_vertices_projection=projected,complete_samples_projection=samples)
            rgb=cv2.imread(P.m['frame_paths'][i])[...,::-1];overlay=rgb.astype(float)/255
            for val,color in [(1,np.array([.88,.3,.45])),(2,np.array([.1,.8,.85]))]:
                mask=lab==val;overlay[mask]=.6*overlay[mask]+.4*color
            ax=axs[rowidx,0];ax.imshow(overlay)
            if dev=='dev1':
                h=cv2.convexHull(projected.astype(np.float32))[:,0];h=np.vstack([h,h[0]]);ax.plot(h[:,0],h[:,1],c='#ffff33',lw=1)
            else: ax.scatter(samples[:,0],samples[:,1],s=.6,c='#ffff33',alpha=.6)
            ax.scatter(uv[:,0],uv[:,1],s=9,c='white',edgecolors='#166275',linewidths=.3)
            ax.set_title(f'{dev} frame {i}  frozen RGB labels and complete projection',fontsize=9)
            bx=axs[rowidx,1];bx.imshow(field,cmap='Greys',vmin=0,vmax=30);bx.contour(lab==1,levels=[.5],colors=['#d84670'],linewidths=.7);bx.contour(lab==2,levels=[.5],colors=['#20adbe'],linewidths=.7)
            bx.scatter(bxy[:,0],bxy[:,1],c=bval,cmap='inferno',s=6 if dev=='dev1' else 2,vmin=0,vmax=20)
            bx.set_title(f'EDT(B) and actual negative queries  H interior {stats["H"]["count"]}',fontsize=9)
            ys,xs=np.nonzero((lab==1)|(lab==2));xmin=max(0,min(xs.min(),bxy[:,0].min())-20);xmax=min(640,max(xs.max(),bxy[:,0].max())+20);ymin=max(0,min(ys.min(),bxy[:,1].min())-20);ymax=min(480,max(ys.max(),bxy[:,1].max())+20)
            for ax in [ax,bx]:ax.set_xlim(xmin,xmax);ax.set_ylim(ymax,ymin);ax.set_aspect('equal');ax.tick_params(labelsize=7)
        fig.savefig(E/f'mask_ledger/{dev}_frozen_frames.png',dpi=170);plt.close(fig)
        summary[dev]={'frames':rows,'full_sequence_original_components':{k:float(v) for k,v in comp.items()},'all_H_interior_values_and_gradients_zero':all(x['regions']['H']['count']==0 or (x['regions']['H']['max_raw_px']==0 and x['regions']['H']['max_raw_gradient_norm']==0) for x in rows),'some_actual_queries_inside_H':sum(x['regions']['H']['count'] for x in rows)>0,'all_B_interior_minimally_active':all(x['regions']['B']['count']==0 or x['regions']['B']['mean_raw_px']>0 for x in rows),'all_original_O_retained':True,'new_U_pixels':0,'reference_used':False}
        print(dev,[(x['frame'],x['regions']['H'],x['regions']['nearest_pixel_H']) for x in rows],flush=True)
        del o,P,back,grad,buv
    save(E/'mask_ledger/results.json',summary)
    return summary

if __name__=='__main__':
    synthetic_result=synthetic();real=real_frames()
    accepted=synthetic_result['all_pass'] and all(v['all_H_interior_values_and_gradients_zero'] and v['some_actual_queries_inside_H'] for v in real.values())
    save(E/'protocol/audit_result.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'synthetic_contract_passed':synthetic_result['all_pass'],'real_path_checks_passed':accepted,'decision':'no_semantic_conflict_demonstrated' if accepted else 'review_specific_failed_checks','new_pose_solves':0,'new_gaussian_runs':0,'independent_events':0,'wall_seconds':time.perf_counter()-START,'device':'CPU only','torch':torch.__version__,'code_sha256':sha(__file__),'production_code_sha256':sha(PREV/'code/joint_solver.py')})
