"""Plot frozen diagnostic artifacts without selecting or changing predictions."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).resolve().parents[1]
OUT=E/'output/figures';OUT.mkdir(parents=True,exist_ok=True)
def load(p):return json.loads(Path(p).read_text())
COL={'silhouette_forward':'#2966a3','silhouette_background':'#54a1bd','estimated_depth_weak':'#dc7929','confirmed_tracks':'#a85390'}
def response(dev):
 p=E/f'score_audit/objective/{dev}/responses.json'
 if dev=='dev1':p=E/'score_audit/objective/dev1/with_track_off/responses.json'
 a=load(p);keys=a['freeze']['keyframes'];cases=list(a['cases'])
 fig,axes=plt.subplots(len(cases),5,figsize=(15,3*len(cases)),squeeze=False,constrained_layout=True)
 for j,c in enumerate(cases):
  for k,f in enumerate(keys):
   ax=axes[j,k];rs=[r for r in a['rows'] if r['case']==c and r['frame']==f and r['kind']=='depth'];rs.sort(key=lambda r:r['value']);z=next(r for r in rs if r['value']==0);x=[r['value']*100 for r in rs]
   for name,color in COL.items():
    ax.plot(x,[r['frame_components_robust_sum'][name]-z['frame_components_robust_sum'][name] for r in rs],color=color,label=name.replace('silhouette_','sil. ').replace('estimated_',''),lw=1.25)
   ax.plot(x,[r['whole_observation_delta']*(114 if dev=='dev1' else 98) for r in rs],c='k',lw=1.7,label='observation sum')
   ax.axhline(0,color='#888888',lw=.6);ax.set_title(f'{c}, frame {f}',fontsize=10);ax.grid(alpha=.2)
   if j==len(cases)-1:ax.set_xlabel('Centre-ray depth change (%)')
   if k==0:ax.set_ylabel('Delta robust observation sum')
 axes[0,0].legend(fontsize=6);fig.suptitle(f'{dev}: fixed-frame input response; temporal and constants excluded',fontsize=13)
 fig.savefig(OUT/f'{dev}_depth_response.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,5,figsize=(15,5.8),constrained_layout=True)
 for j,c in enumerate(['old','P1']):
  for k,f in enumerate(keys):
   ax=axes[j,k];rs=[r for r in a['rows'] if r['case']==c and r['frame']==f and r['kind']=='depth'];rs.sort(key=lambda r:r['value']);x=[r['value']*100 for r in rs]
   ax.plot(x,[r['whole_observation_delta'] for r in rs],c='#2966a3',label='observation delta')
   ax.plot(x,[r['whole_temporal_delta'] for r in rs],c='#dc7929',label='temporal delta')
   ax.plot(x,[r['whole_objective_delta'] for r in rs],c='k',label='total delta')
   ax.axhline(0,color='#888',lw=.6);ax.set_title(f'{c}, frame {f}',fontsize=10);ax.grid(alpha=.2)
   if j==1:ax.set_xlabel('Centre-ray depth change (%)')
   if k==0:ax.set_ylabel('Delta whole score')
 axes[0,0].legend(fontsize=7);fig.suptitle(f'{dev}: fixed neighbours add a separate temporal penalty')
 fig.savefig(OUT/f'{dev}_temporal_response.png',dpi=180);plt.close(fig)
 # Orientation slices include both signs of every axis and all original keys.
 fig,axes=plt.subplots(1,len(cases),figsize=(5*len(cases),4.3),squeeze=False,constrained_layout=True)
 for j,c in enumerate(cases):
  matrix=np.array([[next(r['whole_observation_delta'] for r in a['rows'] if r['case']==c and r['frame']==f and r['kind']=='rotation' and r['axis']==ax and r['value']==sgn) for ax in range(3) for sgn in [-15,15]] for f in keys])
  lim=max(float(abs(matrix).max()),1e-8);im=axes[0,j].imshow(matrix,cmap='coolwarm',vmin=-lim,vmax=lim,aspect='auto');axes[0,j].set_xticks(range(6),['x-15','x+15','y-15','y+15','z-15','z+15'],rotation=35);axes[0,j].set_yticks(range(5),[f'f{f}' for f in keys]);axes[0,j].set_title(c);fig.colorbar(im,ax=axes[0,j],label='Whole observation-score delta')
 fig.suptitle(f'{dev}: template-axis orientation slices, centre fixed')
 fig.savefig(OUT/f'{dev}_rotation_response.png',dpi=180);plt.close(fig)
 # Reading-size summary: larger panels comparing only observation deltas.
 fig,axes=plt.subplots(3,2,figsize=(8,8.2),constrained_layout=True)
 for ax,f in zip(axes.ravel(),keys):
  for c in cases:
   rs=sorted([r for r in a['rows'] if r['case']==c and r['frame']==f and r['kind']=='depth'],key=lambda r:r['value'])
   ax.plot([r['value']*100 for r in rs],[r['whole_observation_delta'] for r in rs],'o-',ms=3,label=c)
  ax.set_title(f'Frame {f}');ax.set_xlabel('Depth change (%)');ax.set_ylabel('Observation score delta');ax.grid(alpha=.25);ax.axhline(0,c='gray',lw=.6)
 axes[0,0].legend(fontsize=8);axes.ravel()[-1].axis('off');axes.ravel()[-1].text(.05,.9,'Centre projection held fixed\nScale and rotation fixed\nSame effective samples\n\nTemporal cost is excluded.\nEach curve subtracts its own\nunperturbed observation score.',va='top',fontsize=10)
 fig.savefig(OUT/f'{dev}_depth_reading.png',dpi=180);plt.close(fig)
 # Preserve observation coverage at every frame.
 states=a['all_frame_observation_states'];ts=[r['time_seconds'] for r in states];counts=[r['reliable_2d_record_count'] for r in states]
 fig,ax=plt.subplots(figsize=(12,2.6),constrained_layout=True);ax.plot(ts,counts,lw=1.1);ax.axhline(6,c='orange',ls='--',label='6 reliable 2D records');ax.fill_between(ts,0,max(counts),where=[not r['direct_visible'] for r in states],color='gray',alpha=.2,label='No fixed direct visibility');ax.set_xlabel('Actual camera0 time (s)');ax.set_ylabel('Reliable 2D records');ax.legend(fontsize=8);ax.set_title(dev+' input observation coverage (not material truth)');fig.savefig(OUT/f'{dev}_coverage.png',dpi=170);plt.close(fig)

def toggle():
 fig,axs=plt.subplots(1,2,figsize=(12,3.8),constrained_layout=True)
 for b in ['on35','off35']:
  data=load(E/f'track_toggle/{b}/iterations.json')
  if isinstance(data,dict):data=data.get('iterations',data.get('rows',[]))
  # Inspect keys explicitly rather than assuming an accepted-iteration schema.
  if not data:continue
  keys=data[0].keys()
  costkey='robust_sum' if 'robust_sum' in keys else 'total_robust_sum'
  if costkey in keys:axs[0].plot(range(len(data)),[r[costkey] for r in data],label=b)
 axs[0].set_xlabel('Accepted iteration');axs[0].set_ylabel('Objective (different active track terms)');axs[0].legend()
 s=load(E/'track_toggle/summary.json')
 axs[1].bar(['on35','off35'],[s['branches'][b]['nfev'] for b in ['on35','off35']],color=['#2966a3','#dc7929']);axs[1].axhline(35,c='gray',ls='--',label='Budget 35');axs[1].set_ylabel('nfev (finite differences excluded)');axs[1].legend();fig.savefig(OUT/'toggle_optimization.png',dpi=180);plt.close(fig)

def official():
 rows=load(E/'score_audit/official/base/scored/scores.json')+load(E/'score_audit/official/off35/scored/scores.json')
 for dev in ['dev1','dev2']:
  keys=sorted({r['frame'] for r in rows if r['dev']==dev})
  paths=sorted({r['path_id'] for r in rows if r['dev']==dev and r['category']=='perturbation'})
  fig,axes=plt.subplots(3,2,figsize=(8,8.1),constrained_layout=True)
  for ax,f in zip(axes.ravel(),keys):
   for path in paths:
    rs=sorted([r for r in rows if r['dev']==dev and r['frame']==f and r['path_id']==path and r.get('depth_fraction') is not None],key=lambda r:r['depth_fraction'])
    if rs:ax.plot([r['depth_fraction']*100 for r in rs],[r['pose_logit'] for r in rs],'o-',ms=3,label=path)
   ax.set_title(f'Frame {f}');ax.set_xlabel('Depth change (%)');ax.set_ylabel('Official pose_logit');ax.grid(alpha=.25)
  axes[0,0].legend(fontsize=7);axes.ravel()[-1].axis('off');axes.ravel()[-1].text(.05,.9,'Higher is preferred by scorer.\nNot a calibrated probability.\n\nOfficial hypothesis-dependent\ncrop is preserved; crop/render\nimages saved for every score.\n\nNo reference used in scoring.',va='top',fontsize=10)
  fig.savefig(OUT/f'{dev}_official_response.png',dpi=180);plt.close(fig)

if __name__=='__main__':
 for dev in ['dev1','dev2']:response(dev)
 toggle()
 official()
