"""Predeclared input-evidence figures; run only after all eight outputs AND evaluation.

Fixed frames, deterministic displayed-edge IDs, every arm, no reference-driven
case/point selection. This module does not import a reference evaluator.
"""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARMS=('F00','F01','F10','F11')
FRAMES={'dev1':(2,40,43,83,88),'dev2':(3,24,49,64,71,74,77)}
COLORS={'F00':'#4c78a8','F01':'#f58518','F10':'#54a24b','F11':'#b279a2'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,a):Path(p).write_text(json.dumps(a,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def readiness(freeze,evaluation):
 f=read(freeze);assert f.get('all_eight_outputs_frozen') is True and f.get('reference_used') is False
 identities={str(Path(x['path']).resolve()):x['sha256'] for x in f['files']}
 for dev in FRAMES:
  for arm in ARMS:
   for name in ['object_init.npz','metrics.json']:
    p=(ROOT/dev/arm/name).resolve();assert str(p) in identities and sha(p)==identities[str(p)],str(p)
 e=read(evaluation);assert e['status']=='completed','Figures wait for batch evaluation; do not choose cases while solves remain.'
 return f,e

def savefig(fig,path):
 fig.savefig(path,dpi=165,bbox_inches='tight',facecolor='white')
 import matplotlib.pyplot as plt
 plt.close(fig)

def plot_metrics(metrics,out):
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,4,figsize=(14,6.2));fig.subplots_adjust(wspace=.33,hspace=.42)
 columns=[('Heldout reprojection','px',lambda x:x['final']['residuals']['heldout']['reprojection_px_mean'],lambda x:x['initial']['residuals']['heldout']['reprojection_px_mean']),('Optimization reprojection','px',lambda x:x['final']['residuals']['opt']['reprojection_px_mean'],lambda x:x['initial']['residuals']['opt']['reprojection_px_mean']),('Heldout effective coverage','%',lambda x:100*x['final']['coverage']['heldout']['effective_fraction'],lambda x:100*x['initial']['coverage']['heldout']['effective_fraction']),('Mean q displacement','cm',lambda x:100*x['q_movement_mean_m'],lambda x:0)]
 for row,dev in enumerate(FRAMES):
  for col,(title,unit,fun,initial) in enumerate(columns):
   ax=axs[row,col];vals=[fun(metrics[dev][a]) for a in ARMS];ax.bar(ARMS,vals,color=[COLORS[a] for a in ARMS]);ax.axhline(initial(metrics[dev]['F00']),color='#444444',ls='--',lw=1,label='common initial');ax.set_title(f'{dev}: {title}',fontsize=10);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
   if col==2:ax.set_ylim(0,105)
   for i,v in enumerate(vals):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
 fig.suptitle('All eight input outcomes; heldout edges share the input video (not independent 3D evidence)',fontsize=12)
 savefig(fig,out/'eight_arm_input_comparison.png')
 for dev in FRAMES:
  fig,axs=plt.subplots(1,2,figsize=(11,4));factors=['positive_depth_fraction','in_bounds_fraction','mask_consistent_fraction','template_visible_fraction','effective_fraction'];names=['Positive depth','In bounds','Object mask','Template visible','Combined']
  for ax,split in zip(axs,['opt','heldout']):
   for arm in ARMS:ax.plot(names,[100*metrics[dev][arm]['final']['coverage'][split][k] for k in factors],marker='o',label=arm,color=COLORS[arm])
   ax.plot(names,[100*metrics[dev]['F00']['initial']['coverage'][split][k] for k in factors],color='black',ls='--',label='Initial');ax.set_ylim(0,105);ax.tick_params(axis='x',rotation=20);ax.set_ylabel('Fixed-edge coverage (%)');ax.set_title(f'{dev} / {split}');ax.grid(alpha=.2)
  axs[1].legend(loc='lower left',fontsize=8);fig.tight_layout();savefig(fig,out/f'{dev}_coverage_factors.png')

def plot_curves(dev,metrics,out):
 import matplotlib.pyplot as plt
 obs=np.load(ROOT/f'observations/{dev}/observations.npz');ts=obs['timestamps'];fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
 for arm in ARMS:
  a=np.load(ROOT/dev/arm/'projection_diagnostics.npz');frame=a['edge_frame'];held=a['edge_split']==1;err=np.linalg.norm(a['final_projection']-a['edge_uv'],axis=1);effective=a['final_edge_positive']&a['final_edge_in_bounds']&a['final_edge_mask_consistent']&a['final_edge_template_visible'];curve=np.full(len(ts),np.nan);coverage=curve.copy()
  for f in range(len(ts)):
   ids=(frame==f)&held
   if ids.any():curve[f]=np.mean(err[ids]);coverage[f]=100*np.mean(effective[ids])
  axs[0].plot(ts,curve,label=arm,color=COLORS[arm],lw=1.3);axs[1].plot(ts,coverage,label=arm,color=COLORS[arm],lw=1.3)
 for ax in axs:
  for f in FRAMES[dev]:ax.axvline(ts[f],color='#dddddd',lw=.6)
  ax.grid(alpha=.2)
 axs[0].set_ylabel('Heldout reprojection (px)');axs[0].legend(ncol=4);axs[0].set_title(f'{dev}: complete input timeline; gaps mean no heldout observation')
 axs[1].set_ylabel('Heldout effective coverage (%)');axs[1].set_xlabel('Actual timestamp (s)');axs[1].set_ylim(-2,102);fig.tight_layout();savefig(fig,out/f'{dev}_heldout_timeline.png')


def plot_q(dev,out):
 import matplotlib.pyplot as plt
 from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 obs=np.load(ROOT/f'observations/{dev}/observations.npz');tri=obs['vertices'][obs['faces']];used=obs['vertices'][np.unique(obs['faces'])];lo=used.min(0);hi=used.max(0);extent=(hi-lo).max();center=(lo+hi)/2
 fig=plt.figure(figsize=(15,4));displacements=[]
 for arm in ARMS:
  s=np.load(ROOT/dev/arm/'surface_points.npz');displacements.append(100*np.linalg.norm(s['q_final']-s['q0'],axis=1))
 vmax=max(.1,max(float(d.max()) for d in displacements))
 for i,arm in enumerate(ARMS):
  ax=fig.add_subplot(1,4,i+1,projection='3d');s=np.load(ROOT/dev/arm/'surface_points.npz');q0=s['q0'];q=s['q_final'];ax.add_collection3d(Poly3DCollection(tri,facecolor='#dddddd',edgecolor='#bbbbbb',linewidth=.1,alpha=.08));ax.scatter(*q0.T,c='#888888',s=4,alpha=.4)
  for a,b in zip(q0,q):ax.plot(*np.stack([a,b]).T,color='#555555',lw=.4,alpha=.5)
  sc=ax.scatter(*q.T,c=displacements[i],s=10,cmap='viridis',vmin=0,vmax=vmax);ax.set_xlim(center[0]-extent*.55,center[0]+extent*.55);ax.set_ylim(center[1]-extent*.55,center[1]+extent*.55);ax.set_zlim(center[2]-extent*.55,center[2]+extent*.55);ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-60);ax.set_title(f'{arm}: shared canonical q');ax.set_xlabel('x (m)',fontsize=8);ax.set_ylabel('y (m)',fontsize=8);ax.set_zlabel('z (m)',fontsize=8);ax.tick_params(labelsize=7)
 fig.colorbar(sc,ax=fig.axes,shrink=.65,pad=.015,label='Displacement from q0 (cm)');fig.suptitle(f'{dev}: original fixed template faces; identical view/limits, all candidate points')
 savefig(fig,out/f'{dev}_q_movement.png')
 fig,axs=plt.subplots(1,3,figsize=(12,3.8))
 for ax,(key,label,scale) in zip(axs,[('total_move_m','Accumulated surface travel (cm)',100),('surface_distance_bound_m','Certified distance from original q0 (cm)',100),('face_switch_count','Face switches per q',1)]):
  vals=[np.asarray(np.load(ROOT/dev/arm/'surface_points.npz')[key])*scale for arm in ARMS];ax.boxplot(vals,tick_labels=ARMS,showfliers=True);ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.2)
 axs[1].axhline(5,color='red',ls='--',lw=1);fig.suptitle(dev+' attachment diagnostics; movement is not material-identity proof');fig.tight_layout();savefig(fig,out/f'{dev}_q_statistics.png')


def plot_overlays(dev,out):
 import matplotlib.pyplot as plt
 from PIL import Image
 obs=np.load(ROOT/f'observations/{dev}/observations.npz');s=read(ROOT/f'observations/{dev}/summary.json');manifest=read(s['manifest']);labels=np.load(s['segmentation'])['entity_labels'];edge_frame=obs['edge_frame'];edge_split=obs['edge_split'];coords=obs['edge_uv'];diags={a:np.load(ROOT/dev/a/'projection_diagnostics.npz') for a in ARMS};exports=[]
 for f in FRAMES[dev]:
  rgb=np.asarray(Image.open(manifest['frame_paths'][f]).convert('RGB'));allids=np.flatnonzero(edge_frame==f);ids=allids[np.linspace(0,len(allids)-1,min(48,len(allids)),dtype=int)] if len(allids) else allids
  pixels=np.argwhere(labels[f]==2)
  if len(pixels):y0,x0=np.maximum(0,pixels.min(0)-45);y1,x1=np.minimum([rgb.shape[0]-1,rgb.shape[1]-1],pixels.max(0)+45)
  else:y0,x0=0,0;y1,x1=np.array(rgb.shape[:2])-1
  fig,axs=plt.subplots(1,5,figsize=(16,3.4));names=('Measurements',)+ARMS
  for ax,name in zip(axs,names):
   ax.imshow(rgb)
   for split,marker,color in [(0,'o','#22c55e'),(1,'x','#00ffff')]:
    ii=ids[edge_split[ids]==split];ax.scatter(coords[ii,0],coords[ii,1],s=12,marker=marker,c=color,linewidths=.9,label='train' if split==0 else 'heldout')
   if name!='Measurements':
    pred=diags[name]['final_projection'];ax.scatter(pred[ids,0],pred[ids,1],s=16,marker='+',c='#ff3b30',linewidths=.8)
    for idx in ids:ax.plot([coords[idx,0],pred[idx,0]],[coords[idx,1],pred[idx,1]],c='#ffcc00',lw=.4,alpha=.7)
   ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_axis_off();ax.set_title(name,fontsize=10)
  title=f'{dev} frame {f} / t={obs["timestamps"][f]:.6f}s; green train, cyan heldout, red projected'
  if dev=='dev2' and f in [64,74]:title+=' / previously exposed failure case'
  if not len(ids):title+=' / no candidate edges'
  fig.suptitle(title,fontsize=10);fig.tight_layout();path=out/f'{dev}_projection_{f:05d}.png';savefig(fig,path);exports.append(dict(dev=dev,frame=f,path=str(path),all_edges=len(allids),shown_edge_ids=ids.tolist(),crop_xyxy=[int(x0),int(y0),int(x1),int(y1)],selection='uniform sorted edge IDs, maximum48; crop from fixed input object mask only'))
 return exports

def main():
 p=argparse.ArgumentParser();p.add_argument('--all-outputs-freeze',type=Path,required=True);p.add_argument('--evaluation-summary',type=Path,default=ROOT/'evaluation/paired_metrics.json');p.add_argument('--output',type=Path,default=ROOT/'visualizations');a=p.parse_args();readiness(a.all_outputs_freeze,a.evaluation_summary)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 a.output.mkdir(parents=True,exist_ok=False);metrics={d:{arm:read(ROOT/d/arm/'metrics.json') for arm in ARMS} for d in FRAMES};plot_metrics(metrics,a.output);overlays=[]
 for d in FRAMES:plot_curves(d,metrics,a.output);plot_q(d,a.output);overlays.extend(plot_overlays(d,a.output))
 files=sorted(a.output.glob('*.png'));save(a.output/'manifest.json',{'status':'completed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(__file__),'all_outputs_freeze_sha256':sha(a.all_outputs_freeze),'evaluation_summary_sha256':sha(a.evaluation_summary),'fixed_frames':FRAMES,'point_selection_rule':'uniform sorted graph edge IDs max48/frame, no error ranking','overlays':overlays,'files':[{'path':str(f),'sha256':sha(f)} for f in files],'independent_3d_evidence':False,'note':'Input diagnostics plus observed RGB only; no contact/gaussian/material correctness claim. All arms retained.'});print(json.dumps({'output':str(a.output),'figures':len(files)}))
if __name__=='__main__':main()
