"""Static scientific plots from saved process records, never training feedback."""
from common import *
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def run():
    fig,ax=plt.subplots(1,3,figsize=(14,3.5),layout='constrained')
    for branch,color in [('W_fine','#377eb8'),('W_all','#d95f02')]:
        data=[json.loads(s) for s in (RUN/'runs'/branch/'training_metrics.jsonl').read_text().splitlines()]
        for stage,ls in [('coarse','--'),('fine','-')]:
            a=[r for r in data if r['stage']==stage];x=[r['iteration'] for r in a]
            if not a:continue
            ax[0].plot(x,[r['L_rgb'] for r in a],ls,label=branch+' '+stage,color=color)
            ax[1].plot(x,[r['L_fg'] for r in a],ls,label=branch+' FG '+stage,color=color)
            ax[1].plot(x,[r['L_bg'] for r in a],ls,alpha=.45,color=color)
            ax[2].plot(x,[r['points'] for r in a],ls,label=branch+' '+stage,color=color)
    for a,t in zip(ax,['Pure regional RGB loss','FG loss solid color / BG pale','Total Gaussian count']):a.set(title=t,xlabel='Stage iteration');a.grid(alpha=.2)
    ax[0].legend(fontsize=8);fig.savefig(RUN/'diagnostics/training_summary.jpg',dpi=160);plt.close(fig)
    data=[json.loads(s) for s in (RUN/'state_probes.jsonl').read_text().splitlines()];groups=list(data[0]['groups']);keys=list(dict.fromkeys((r['branch'],r['stage'],r['iteration']) for r in data));mat=np.full((len(keys),len(groups)),np.nan)
    for i,key in enumerate(keys):
        rs=[r for r in data if (r['branch'],r['stage'],r['iteration'])==key]
        for j,g in enumerate(groups):
            vals=[r['groups'][g]['cosine'] for r in rs if r['groups'][g]['cosine'] is not None];mat[i,j]=np.median(vals) if vals else np.nan
    fig,ax=plt.subplots(figsize=(11,max(3,len(keys)*.48)),layout='constrained');im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
    ax.set_xticks(range(len(groups)),[x.replace('_','\n') for x in groups],fontsize=8);ax.set_yticks(range(len(keys)),[' '.join(map(str,k)) for k in keys],fontsize=9)
    for i in range(len(keys)):
        for j in range(len(groups)):ax.text(j,i,'NA' if not np.isfinite(mat[i,j]) else f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Median FG/BG cosine over four fixed train frames');fig.savefig(RUN/'diagnostics/probe_summary.jpg',dpi=160);plt.close(fig)
if __name__=='__main__':run()
