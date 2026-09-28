"""Fixed training-only correspondence explanations, no new teachers or optimization."""
from common import *
import torch,numpy as np,cv2
from render_utils import load_model,CalibratedCamera
from hoi_modules.temporal_evidence import TemporalEvidenceModule

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(4);freeze=read(RUN/'protocol/finals.json');assert freeze['status']=='all_three_terminals_frozen'
    cache=read(RUN/'track_cache_manifest.json');order=torch.load(RUN/'protocol/pair_order.pt',weights_only=False)['order'][:4].tolist();frames={r['frame_id']:r for r in read(OLD/'inputs/hos_backpack/manifest.json')['frames']}
    out=RUN/'evaluation/temporal_explanations';out.mkdir(exist_ok=False);results={};stats=[]
    for arm in ['B_Q','T_plain','T_mix']:
        rd=RUN/'runs'/arm;model,_,state=load_model(read(rd/'run.json')['checkpoint'],rd/'effective_config.json');del state
        module=TemporalEvidenceModule('mixture',read(RUN/'configs/v8.json')['scale_bound'])
        for number,pid in enumerate(order):
            p=cache['pairs'][pid];obs=dict(np.load(p['cache']['path']));cams=[CalibratedCamera(frames[p[n]],i,False) for i,n in enumerate(['source_frame','target_frame'])]
            with torch.no_grad():loss,st,d=module(model,*cams,obs)
            assert d is not None
            data={k:v.cpu().numpy() for k,v in d.items() if k!='unit_loss'};data.update(source_uv=obs['source_uv'][obs['valid']],target_uv=obs['target_uv'][obs['valid']],confidence=obs['confidence'][obs['valid']]);path=out/f'{arm}_pair_{number}.npz';np.savez_compressed(path,**data);results[arm,number]=data;stats.append(dict(arm=arm,pair_order_index=number,pair_id=pid,source_frame=p['source_frame'],target_frame=p['target_frame'],stats=st,arrays=identity(path),diagnostic_mode='mixture moments computed for every checkpoint; gate is training-active only for T_mix'))
        del model;torch.cuda.empty_cache()
    figures=[]
    for number,pid in enumerate(order):
        p=cache['pairs'][pid];rgb=cv2.imread(frames[p['source_frame']]['rgb_path'])[...,::-1];fig,axs=plt.subplots(3,5,figsize=(15,6.5));fig.suptitle(f'Train pair {p["source_frame"]} -> {p["target_frame"]} | source-pixel samples; not ground truth')
        for row,arm in enumerate(['B_Q','T_plain','T_mix']):
            d=results[arm,number];uv=d['source_uv'];ix=np.arange(0,len(uv),max(1,len(uv)//64));supported=d['supported']
            for ax in axs[row]:ax.imshow(rgb);ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0);ax.set_xticks([]);ax.set_yticks([])
            for col,dst,label in [(0,d['target_uv'],'teacher flow'),(1,d['predicted'],'predicted flow')]:
                dd=dst[ix]-uv[ix];axs[row,col].quiver(uv[ix,0],uv[ix,1],dd[:,0],dd[:,1],angles='xy',scale_units='xy',scale=1,color='lime',width=.004);axs[row,col].set_title(f'{arm} {label}',fontsize=9)
            for col,key,title,lo,hi in [(2,'alpha','valid alpha',0,1),(3,'variance','log10(1+v px2)',0,5),(4,'gate','mixture gate (diagnostic)',.2,1)]:
                vals=np.log10(1+d[key]) if key=='variance' else d[key];ax=axs[row,col];sc=ax.scatter(uv[:,0],uv[:,1],c=vals,s=8,cmap='viridis',vmin=lo,vmax=hi);ax.set_title(title,fontsize=9);fig.colorbar(sc,ax=ax,fraction=.035,pad=.01)
        fig.tight_layout();path=out/f'training_pair_{number}.jpg';fig.savefig(path,dpi=135);plt.close(fig);figures.append(identity(path))
    save_json(out/'manifest.json',dict(status='completed',fixed_first_four_pair_entries=order,stats=stats,figures=figures,teacher_source='same frozen train-only CoTracker cache',development_track_teacher_evaluation=False,diagnostic_gate_for_B_Q_and_T_plain=True,optimization_steps=0))
if __name__=='__main__':main()
