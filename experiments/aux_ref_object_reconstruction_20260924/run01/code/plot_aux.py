"""All frozen E samples shown with one input-defined crop per row."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
E=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def rgb(p):return cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
def main():
 out=E/'figures';out.mkdir(exist_ok=True)
 regions=read(E/'evaluation/regions/manifest.json')['rows'];rows=read(E/'evaluation/cross_pose/per_frame.json');lookup={(r['dev'],r['sample_id'],r['cell']):r for r in rows}
 ledger=[]
 for dev in ['dev1','dev2']:
  rr=sorted([r for r in regions if r['dev']==dev],key=lambda x:x['query_time_seconds'])
  fig,axes=plt.subplots(len(rr),6,figsize=(12,len(rr)*2.05),squeeze=False)
  for i,r in enumerate(rr):
   lab=np.load(r['regions']['path'])['entity_labels'];ys,xs=np.nonzero(lab==2);H,W=lab.shape
   cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2;side=max(200,xs.max()-xs.min()+160,ys.max()-ys.min()+160);side=min(side,H)
   x0=int(np.clip(cx-side/2,0,W-side));y0=int(np.clip(cy-side/2,0,H-side));x1=x0+int(side);y1=y0+int(side)
   sample=E/'evaluation/cross_pose'/dev/r['frame_id'];paths=[r['rgb']['path'],*[lookup[(dev,r['sample_id'],c)]['rgb_export']['path'] for c in ['PP','PR','RP','RR']],sample/'frozen_HS_only.png']
   for j,(name,p) in enumerate(zip(['GT','PP','PR','RP','RR','Frozen H/S'],paths)):
    ax=axes[i,j];ax.imshow(rgb(p)[y0:y1,x0:x1]);ax.contour((lab[y0:y1,x0:x1]==2),levels=[.5],colors=['#00DFFF'],linewidths=.4);ax.set_xticks([]);ax.set_yticks([])
    if i==0:ax.set_title(name,fontsize=11)
    if j==0:ax.set_ylabel(f't={r["query_time_seconds"]:g}s\nO={len(xs)}px',fontsize=9)
    if 1<=j<=4:
     m=lookup[(dev,r['sample_id'],['PP','PR','RP','RR'][j-1])]['metrics']['object'];ax.set_xlabel(f'{m["psnr_db"]:.2f} dB / {m["ssim"]:.3f}',fontsize=8)
   ledger.append({'dev':dev,'sample_id':r['sample_id'],'crop_xyxy':[x0,y0,x1,y1],'rule':'GT fixed O bounding box +80px each side; min200px square; frame-bound clipping; same crop for all six columns','all_E_samples_included':True})
  fig.suptitle(f'AUX_REF_OBJECT  {dev}  held-out camera1',fontsize=13,y=.997);fig.tight_layout(pad=.7,rect=[0,0,1,.98]);fig.savefig(out/f'{dev}_all_E_grid.png',dpi=180);plt.close(fig)
 # Attribute distributions describe actual exported learned Gaussians.
 fig,axes=plt.subplots(2,3,figsize=(10,5.8))
 for i,dev in enumerate(['dev1','dev2']):
  banks=[np.load(E/'runs'/f'{dev}_{arm}'/'object_final.npz') for arm in ['Pred','Ref']]
  for j,(field,title,scale) in enumerate([('opacity','Opacity',1),('scale_m','Largest scale (mm)',1000),('local_offset_m','Canonical offset (mm)',1000)]):
   vals=[]
   for x in banks:
    a=x[field];a=a.max(1) if field=='scale_m' else np.linalg.norm(a,axis=1) if field=='local_offset_m' else a.reshape(-1);vals.append(a*scale)
   axes[i,j].boxplot(vals,tick_labels=['Pred','Ref'],showfliers=False);axes[i,j].set_title(f'{dev}  {title}',fontsize=10);axes[i,j].grid(axis='y',alpha=.2)
   if field=='local_offset_m':axes[i,j].axhline(5,color='r',lw=.7,ls='--')
 fig.suptitle('AUX_REF_OBJECT  actual final object attributes',fontsize=13);fig.tight_layout();fig.savefig(out/'attribute_distributions.png',dpi=180);plt.close(fig)
 (out/'figure_manifest.json').write_text(json.dumps({'protocol_id':'AUX_REF_OBJECT','crop_ledger':ledger,'boxplot':'Median/quartiles and1.5IQR whiskers; extreme points omitted visually only, full min/max reported in JSON.'},indent=2)+'\n')
 print(out)
if __name__=='__main__':main()
