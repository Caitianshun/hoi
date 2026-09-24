"""CPU figures and traceable summary of completed resolution/depth pilot."""
from pathlib import Path
import json,hashlib
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/cai_tianshun/Project/HOI');OUT=Path(__file__).resolve().parent;BASE=ROOT/'experiments/mosca_baseline_20260922'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 run=json.loads((OUT/'prediction_run.json').read_text());assert run['status']=='completed'
 evaluation=json.loads((OUT/'evaluation_summary.json').read_text());assert evaluation['status'].startswith('completed')
 plan=json.loads((OUT/'plan.json').read_text());labels=np.load(BASE/'segmentation/segmentation.npz')['entity_labels']
 variants={k:json.loads(Path(v['path']).read_text())for k,v in plan['inputs'].items()}
 ids=plan['selected_baseline_indices'];configs={r['name']:json.loads(Path(r['manifest']).read_text())for r in run['configs']}
 figures=OUT/'figures';figures.mkdir(exist_ok=True);crops=OUT/'rgb_crops';crops.mkdir(exist_ok=True)
 fig,axs=plt.subplots(6,4,figsize=(12,15),dpi=140);differences=[];rois={}
 for ri,i in enumerate(ids):
  mask=labels[i]==2;ys,xs=np.where(mask);fallback=not len(xs)
  if fallback:ys,xs=np.where(labels[i]==1)
  x0=max(0,(int(xs.min())-10)//5*5);x1=min(640,(int(xs.max())+15)//5*5)
  y0=max(0,(int(ys.min())-10)//5*5);y1=min(480,(int(ys.max())+15)//5*5);rois[i]=[x0,y0,x1,y1]
  for ci,name in enumerate(['native_rectified','legacy640','area640','area1024']):
   rgb=cv2.cvtColor(cv2.imread(variants[name]['frame_paths'][ri]),cv2.COLOR_BGR2RGB);s=variants[name]['width']/640
   xx0,yy0,xx1,yy1=[int(round(a*s))for a in [x0,y0,x1,y1]];crop=rgb[yy0:yy1,xx0:xx1]
   cv2.imwrite(str(crops/f'{i:05d}_{name}.png'),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
   axs[ri,ci].imshow(crop,interpolation='nearest');axs[ri,ci].axis('off')
   title=f'{name}\n{i}: {crop.shape[1]}x{crop.shape[0]} real pixels'
   if fallback:title+=' (person ROI)'
   axs[ri,ci].set_title(title,fontsize=8)
  old=cv2.imread(variants['legacy640']['frame_paths'][ri]).astype(float);new=cv2.imread(variants['area640']['frame_paths'][ri]).astype(float)
  diff=np.abs(new-old);fg=labels[i]>0
  differences.append(dict(input_index=i,old_vs_area640_rgb_mae_0_255=float(diff.mean()),foreground_mae=float(diff[fg].mean()),
     object_mae=float(diff[mask].mean())if mask.any()else None,roi640_xyxy=rois[i],roi_is_person_fallback=fallback))
 fig.suptitle('Real RGB crops, same pinhole ROI | native detail vs processed pixels (no recovery)',fontsize=12)
 fig.tight_layout(rect=(0,0,1,.97));fig.savefig(figures/'rgb_sampling_crops.png');plt.close(fig)

 fig,axs=plt.subplots(6,5,figsize=(13,16),dpi=130);diags=[]
 for ri,i in enumerate(ids):
  x0,y0,x1,y1=rois[i];rgb=cv2.cvtColor(cv2.imread(variants['area640']['frame_paths'][ri]),cv2.COLOR_BGR2RGB)
  axs[ri,0].imshow(rgb[y0:y1,x0:x1]);axs[ri,0].set_title(f'RGB {i}',fontsize=9);axs[ri,0].axis('off')
  olddepth=None
  for ci,(name,cfg)in enumerate(configs.items(),1):
   f=cfg['frames'][ri];p=np.load(f['prediction_path']);d=p['dep'];conf=p['confidence_raw']
   if d.shape!=(480,640):d=cv2.resize(d,(640,480),interpolation=cv2.INTER_LINEAR);conf=cv2.resize(conf,(640,480),interpolation=cv2.INTER_LINEAR)
   if olddepth is None:olddepth=d
   im=axs[ri,ci].imshow(d[y0:y1,x0:x1],vmin=1,vmax=6,cmap='viridis',interpolation='nearest')
   axs[ri,ci].set_title(name,fontsize=8);axs[ri,ci].axis('off')
   parts={}
   for entity,eid in [('person',1),('object',2)]:
    mask=labels[i]==eid;interior=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);edge=mask&~interior
    parts[entity]={}
    for zone,ma in [('interior',interior),('boundary',edge)]:
     parts[entity][zone]=dict(count=int(ma.sum()),depth_quantiles=np.quantile(d[ma],[.1,.5,.9]).tolist(),
        raw_confidence_quantiles=np.quantile(conf[ma],[.1,.5,.9]).tolist(),mean_abs_depth_change_from_old_m=float(abs(d[ma]-olddepth[ma]).mean()))if ma.any()else dict(count=0)
   diags.append(dict(config=name,input_index=i,regions=parts))
 fig.suptitle('Frozen UniDepth outputs | shared 1-6m colour scale (display clipped only)',fontsize=12)
 fig.tight_layout(rect=(0,0,.95,.97));cax=fig.add_axes([.962,.2,.013,.6]);fig.colorbar(im,cax=cax,label='Predicted camera z (m)')
 fig.savefig(figures/'depth_roi_comparison.png');plt.close(fig)
 legacy=[]
 for f in configs['legacy640_level3']['frames']:
  i=f['source_baseline_index'];new=np.load(f['prediction_path'])['dep'];old=np.load(BASE/f'common_input/unidepth_depth/{i:05d}.npz')['dep']
  legacy.append(dict(input_index=i,cross_gpu_old_vs_rerun_depth_mean_abs_m=float(abs(new-old).mean()),max_abs_m=float(abs(new-old).max())))
 record=dict(status='completed',script_sha256=sha(__file__),rgb_differences=differences,input_only_depth_diagnostics=diags,
   legacy_rerun_comparison=legacy,prediction_run_sha256=sha(OUT/'prediction_run.json'),evaluation_sha256=sha(OUT/'evaluation_summary.json'),
   images=[str(figures/'rgb_sampling_crops.png'),str(figures/'depth_roi_comparison.png')],
   interpretation='SAM2 masks are frozen predicted ROIs, not independent geometric GT. Cross-config changes quantify sensitivity, not improved accuracy or temporal stability. Independent post-freeze fitted-surface evaluation is separate.')
 (OUT/'diagnostic_summary.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({'config_costs':run['configs'],'evaluation':[{r['config']:r['equal_frame_summary']}for r in evaluation['results']],
  'legacy_rerun':legacy},indent=2))
if __name__=='__main__':main()
