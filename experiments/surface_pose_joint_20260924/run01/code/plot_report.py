from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).resolve().parents[1];F=E/'output/figures';F.mkdir(parents=True,exist_ok=True)
def load(p):return json.loads(Path(p).read_text())
M={d:{v:load(E/d/v/'metrics.json') for v in ['F00','F01','F10','F11']} for d in ['dev1','dev2']}
colors={'old_initialization':'#222222','historical_P1':'#999999','F00':'#2766a2','F01':'#d16a08','F10':'#40823d','F11':'#9058a7'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(7,4.6),layout='constrained')
for row,d in enumerate(M):
 for col,(key,title,unit) in enumerate([('reprojection_px_mean','Heldout reprojection','px'),('effective_fraction','Heldout effective coverage','%')]):
  a=axs[row,col];vs=list(M[d]);vals=[M[d][v]['final']['residuals' if col==0 else 'coverage']['heldout'][key]*(100 if col else 1) for v in vs]
  a.bar(vs,vals,color=[colors[v] for v in vs]);initial=M[d]['F00']['initial']['residuals' if col==0 else 'coverage']['heldout'][key]*(100 if col else 1);a.axhline(initial,ls='--',color='black',lw=1)
  a.set_title(f'{d}  {title}',fontsize=10);a.set_ylabel(unit);a.set_ylim(0,110 if col else 2.5);a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
  for i,x in enumerate(vals):a.text(i,x+.03 if not col else x+1,f'{x:.2f}',ha='center',fontsize=9)
fig.savefig(F/'input_diagnostics.png',dpi=200);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(7,4.8),layout='constrained')
for r,d in enumerate(M):
 a=load(E/f'evaluation/{d}/augmented_results.json')
 for v in colors:
  rr=[x for x in a['rows'] if x['method']==v] if 'method' in a['rows'][0] else []
  # Result schema uses label; retain all fixed slots, including NaN gaps.
  if not rr:rr=[x for x in a['rows'] if x.get('label')==v]
  if not rr:continue
  times=[x['input_s'] for i,x in enumerate(rr)]
  for c,key in enumerate(['centroid_error_cm','raw_rotation_error_deg']):
   vals=[np.nan if x.get(key) is None else x[key] for x in rr];axs[r,c].plot(times,vals,color=colors[v],label=v.replace('old_initialization','Old').replace('historical_P1','P1'),lw=1.3 if v not in ['old_initialization','historical_P1'] else 1,ls='--' if v=='historical_P1' else '-',marker='o',ms=2)
 for c,t in enumerate(['Centre error (cm)','Rotation error (deg)']):axs[r,c].set_title(d+'  '+t,fontsize=10);axs[r,c].grid(alpha=.2);axs[r,c].set_xlabel('Input time (s)')
axs[0,0].legend(fontsize=7,ncol=3,loc='upper left');fig.savefig(F/'reference_curves.png',dpi=200);plt.close(fig)
