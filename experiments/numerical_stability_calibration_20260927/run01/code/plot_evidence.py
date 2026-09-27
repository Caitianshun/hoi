"""CPU scientific figures from saved training-only A probes and process logs."""
from common import *
import os,torch,numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def records(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='';torch.set_num_threads(4)
    out=RUN/'diagnostics';out.mkdir(exist_ok=True);manifest=[]
    iso=read(RUN/'runs/A_isolation2/summary.json');frames={f['frame_id']:f for f in read(OLD/'inputs/hos_backpack/manifest.json')['frames']}
    for case in ['normal','failure']:
        original=torch.load(RUN/f'runs/A_isolation2/{case}_original_balanced_rgb_values.pt',map_location='cpu',weights_only=False)['pred'].permute(0,2,3,1).numpy()
        bounded=torch.load(RUN/f'runs/A_isolation2/{case}_bounded_balanced_rgb_values.pt',map_location='cpu',weights_only=False)['pred'].permute(0,2,3,1).numpy()
        ids=next(r['frame_ids'] for r in iso['rows'] if r['case']==case);delta=np.max(np.abs(original-bounded),axis=-1);upper=float(delta.max())
        fig,ax=plt.subplots(len(ids),4,figsize=(16,5.1),layout='constrained')
        for i,fid in enumerate(ids):
            gt=cv2.imread(frames[fid]['image_path'])[...,::-1]/255.
            for j,(im,title) in enumerate([(gt,fid+' GT'),(np.clip(original[i],0,1),'Original'),(np.clip(bounded[i],0,1),'Scale bound'),(delta[i],'Max channel abs change')]):
                art=ax[i,j].imshow(im,cmap='magma',vmin=0,vmax=upper) if j==3 else ax[i,j].imshow(im)
                ax[i,j].set_title(title,fontsize=10);ax[i,j].axis('off')
                if j==3:fig.colorbar(art,ax=ax[i,j],fraction=.032,pad=.015)
        p=out/f'A_{case}_forward.jpg';fig.savefig(p,dpi=150);plt.close(fig);manifest.append(dict(**identity(p),kind=case+'_forward',frame_ids=ids,heatmap_max=upper,display_RGB_clipped=True,difference_computed_from_raw=True))
        del original,bounded
    effects=read(RUN/'patch_effects.json');fig,ax=plt.subplots(1,2,figsize=(11,3.3),layout='constrained')
    row=read(RUN/'A_summary.json')['kernel_case'];pid=str(row['all_bad_attribute_row_intersection'][0]);scale=row['bad_input_geometry'][pid]['scale'];bound=effects['domain']['upper_bound']
    x=np.arange(3);ax[0].bar(x-.18,np.log10(scale),.35,label='Original');ax[0].bar(x+.18,np.log10(np.minimum(scale,bound)),.35,label='Bounded');ax[0].set_xticks(x,['axis 1','axis 2','axis 3']);ax[0].set_ylabel('log10 scale in inherited scene units');ax[0].legend();ax[0].set_title('Actual failing Gaussian rendered scale')
    names=['balanced_rgb','uniform_rgb','regularizers'];labels=['Balanced RGB','Uniform RGB','Regularizers']
    for di,domain in enumerate(['original','bounded']):
        vals=[next(r['nonfinite_gradient_elements'] for r in effects['loss_and_gradient_isolation'] if r['case']=='failure' and r['domain']==domain and r['objective']==name) for name in names]
        bars=ax[1].bar(x+(di-.5)*.35,vals,.35,label=domain)
        for bar,v in zip(bars,vals):ax[1].annotate(str(v),(bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',va='bottom',fontsize=8)
    ax[1].set_xticks(x,labels,fontsize=8);ax[1].set_ylabel('Nonfinite parameter gradient elements');ax[1].set_title('Same fault state isolated backward');ax[1].legend();ax[1].set_ylim(0,ax[1].get_ylim()[1]*1.15)
    p=out/'A_domain_and_gradient.jpg';fig.savefig(p,dpi=170);plt.close(fig);manifest.append(dict(**identity(p),kind='domain_and_gradient',point_id=pid))
    metrics=records(RUN/'runs/A_regression128/training_metrics.jsonl');events=records(RUN/'runs/A_regression128/replay_events.jsonl');density=records(RUN/'runs/A_regression128/density_events.jsonl');fig,ax=plt.subplots(2,1,figsize=(10,4.5),sharex=True,layout='constrained')
    ax[0].plot([r['iteration'] for r in metrics],[r['L_total'] for r in metrics],label='Batch loss');ax[0].set_ylabel('Total training loss');ax[0].legend()
    trigger=[r for r in events if r['phase']=='scale_bound'];ax[1].plot([r['iteration'] for r in trigger],[r['triggered_axes']/r['total_axes'] for r in trigger],'.',ms=2);ax[1].set_ylabel('Bounded axis fraction');ax[1].set_xlabel('Fine iteration')
    for a in ax:
        for i in sorted({r['iteration'] for r in density}):a.axvline(i,color='gray',ls='--',lw=.8)
        a.grid(alpha=.2)
    p=out/'A_regression.jpg';fig.savefig(p,dpi=170);plt.close(fig);manifest.append(dict(**identity(p),kind='regression',density_iterations=sorted({r['iteration'] for r in density})))
    save_json(out/'figure_manifest.json',dict(figures=manifest,source=identity(RUN/'patch_effects.json'),all_training_only=True,optimization_steps=0,GPU_used=False))
    print('A scientific figures',len(manifest))
if __name__=='__main__':run()
