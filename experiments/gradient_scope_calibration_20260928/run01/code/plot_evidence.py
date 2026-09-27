"""Static research figures from saved diagnostics; no model work."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def run():
    dest=RUN/'output/report_figures';dest.mkdir(exist_ok=True)
    route=read(RUN/'routing_equivalence.json');bad=[r for r in route['rows'] if not r['passed']]
    fig,ax=plt.subplots(figsize=(9,3.6));x=np.arange(len(bad));ax.bar(x-.18,[r['difference']['max_abs']/r['atol'] for r in bad],.35,label='Max abs / declared tolerance',color='#346d99');ax.bar(x+.18,[r['difference']['relative_L2']/r['rtol'] for r in bad],.35,label='Relative L2 / declared tolerance',color='#cc8248');ax.axhline(1,c='black',ls='--');ax.set_xticks(x,[r['name'].split('grid.')[-1] for r in bad]);ax.set_ylabel('Ratio (must be <= 1)');ax.legend(loc='lower left',fontsize=8);ax.set_title('First uniform_all prerequisite check');fig.tight_layout();fig.savefig(dest/'routing_gate.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.8));s=read(RUN/'contribution_manifest.json')['summary']
    for arm,sub,col in [('B_U','B_U_verified','#346d99'),('B_F','B_F','#cc8248')]:
        z=np.load(RUN/'diagnostics'/sub/'effective_time.npz');a=np.sort(z['N_eff'][z['supported']]);axs[0].plot(a,np.arange(1,len(a)+1)/len(a),label=arm,color=col)
    axs[0].set(xlabel='Effective sampled time count',ylabel='Fraction of supported rows',xlim=(1,8),ylim=(0,1));axs[0].legend();x=np.arange(2)
    axs[1].bar(x-.18,[v['N_eff_contribution_weighted'] for v in s],.35,label='Total contribution weighted',color='#346d99');axs[1].bar(x+.18,[v['FG_weighted_N_eff'] for v in s],.35,label='FG contribution weighted',color='#cc8248');axs[1].set_xticks(x,[v['arm'] for v in s]);axs[1].set_ylabel('Effective sampled time count');axs[1].legend(fontsize=8);fig.suptitle('Eight frozen training frames; no physical correspondence ground truth');fig.tight_layout();fig.savefig(dest/'contribution_summary.png',dpi=180);plt.close(fig)
if __name__=='__main__':run()
