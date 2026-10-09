"""Read-only audit of frozen V10 results; writes only into this review folder."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/cai_tianshun/Project/HOI')
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'experiments/pose_support_v10_20261007/run01'
KEYS = ('psnr_db', 'ssim', 'lpips_spatial_mean')
SCENES = ('Backpack', 'Tennis')
ARMS = ('C', 'P', 'PQ')
INPUTS = {
 'Backpack': ROOT/'experiments/baseline_protocol_calibration_20260927/run01/inputs/hos_backpack',
 'Tennis': ROOT/'experiments/temporal_evidence_v8_20260928/run01/tennis/inputs/hos_tennis',
}

def read(p): return json.loads(Path(p).read_text())
def asset(p):
 p=Path(p)
 return {'path': str(p), 'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def mean(x): return statistics.mean(x)
def psnr(x): return -10*math.log10(max(x,1e-12))

summary=read(RUN/'evaluation_summary.json')
rows=list(csv.DictReader((RUN/'metrics_per_frame.csv').open()))
idx={(r['scene'],r['run'],r['split'],r['frame_id'],r['region']):r for r in rows}
assert len(idx)==len(rows)
results={'summary_source':asset(RUN/'evaluation_summary.json'),
 'frame_source':asset(RUN/'metrics_per_frame.csv'),'scenes':{},'verification':{},'illustrations':[]}
max_agg=0.; max_psnr=0.; max_mse=0.; psnr_checks=0; agg_checks=0; identity_checks=0

for scene in SCENES:
 manifests={s:read(INPUTS[scene]/f) for s,f in [('train','manifest.json'),('retained','evaluation_manifest.json')]}
 train_ids={f['frame_id'] for f in manifests['train']['frames']}; dev_ids={f['frame_id'] for f in manifests['retained']['frames']}
 assert not(train_ids&dev_ids)
 sr={'input': {'train_frames':len(train_ids),'development_frames':len(dev_ids),
  'development_ids':sorted(dev_ids),'resolution':manifests['train']['resolution'],
  'camera_source':manifests['train']['camera_source'],'disjoint':True}, 'arms':{},'paired':{}}
 for arm in ARMS:
  ar={'metrics':{},'cost':read(RUN/'scenes'/scene/'runs'/arm/'run.json')}
  ar['cost']={k:ar['cost'][k] for k in ['seconds','peak_allocated_bytes','support_points','base_points','final_points','completed_updates']}
  for sp in ('train','retained','retained_without_00000'):
   ar['metrics'][sp]={}
   for region in ('full','foreground','background'):
    rs=[r for r in rows if r['scene']==scene and r['run']==arm and r['split']==('train' if sp=='train' else 'retained') and r['region']==region and (sp!='retained_without_00000' or r['frame_id']!='00000')]
    m={k:mean(float(r[k]) for r in rs) for k in KEYS}
    for k in KEYS:
     max_agg=max(max_agg,abs(m[k]-summary['scenes'][scene]['summaries'][arm][sp][region][k]));agg_checks+=1
    ar['metrics'][sp][region]=m
  for frame in manifests['retained']['frames']:
   fid=frame['frame_id']; rr=idx[scene,arm,'retained',fid,'full']
   raw=Path(rr['raw_render_path'])
   assert asset(raw)['sha256']==rr['raw_render_sha256'];identity_checks+=1
   with np.load(raw,allow_pickle=False) as z: rgb=np.clip(z['rgb'],0,1).astype(np.float64)
   gt=np.array(Image.open(frame['image_path']).convert('RGB'),dtype=np.float64)/255.
   fg=np.array(Image.open(frame['mask_path']).convert('L'))>=128
   assert rgb.shape==gt.shape and np.isfinite(rgb).all()
   err=((rgb-gt)**2).mean(-1)
   for region,mask in [('full',np.ones(fg.shape,bool)),('foreground',fg),('background',~fg)]:
    r=idx[scene,arm,'retained',fid,region];mse=float(err[mask].mean());value=psnr(mse)
    max_psnr=max(max_psnr,abs(value-float(r['psnr_db'])));max_mse=max(max_mse,abs(mse-float(r['mse'])));psnr_checks+=1
  stats={k:[] for k in ['fg_pixel_fraction','fg_error_share','fg_20db_counterfactual','reconstruction_mse_abs_error']}
  for fid in sorted(dev_ids):
   f,b,u=[idx[scene,arm,'retained',fid,r] for r in ('foreground','background','full')]
   q=float(f['pixel_fraction']);fm=float(f['mse']);bm=float(b['mse']);um=float(u['mse'])
   stats['fg_pixel_fraction'].append(q);stats['fg_error_share'].append(q*fm/um)
   stats['fg_20db_counterfactual'].append(psnr(q*min(fm,.01)+(1-q)*bm))
   stats['reconstruction_mse_abs_error'].append(abs(um-q*fm-(1-q)*bm))
  ar['error_budget']={k:mean(v) for k,v in stats.items()}
  ar['error_budget']['counterfactual_is_not_experiment']=True
  sr['arms'][arm]=ar
 for left,right in [('P','C'),('PQ','P')]:
  sr['paired'][left+'-'+right]={}
  for region in ('full','foreground','background'):
   sr['paired'][left+'-'+right][region]={}
   for k in KEYS:
    ds=[float(idx[scene,left,'retained',f,region][k])-float(idx[scene,right,'retained',f,region][k]) for f in sorted(dev_ids)]
    direction=-1 if k=='lpips_spatial_mean' else 1
    sr['paired'][left+'-'+right][region][k]={'mean':mean(ds),'median':statistics.median(ds),
      'wins':sum(d*direction>1e-6 for d in ds),'losses':sum(d*direction < -1e-6 for d in ds),'n':len(ds)}
 results['scenes'][scene]=sr
results['verification']={'aggregate_values_checked':agg_checks,'max_aggregate_absolute_error':max_agg,
 'raw_render_sha_checks':identity_checks,'raw_PSNR_region_checks':psnr_checks,
 'max_PSNR_error_db':max_psnr,'max_MSE_error':max_mse,
 'SSIM_LPIPS':'aggregation verified; no new network inference or second metric implementation',
 'GPU_operations':0,'optimizer_updates':0}
assert max_psnr<1e-9 and max_agg<1e-12

# Same crop and display transform for all methods. ROI is illustrative only.
FONTPATH='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(FONTPATH,22);small=ImageFont.truetype(FONTPATH,18)
def case(scene,split,fid,coords=None):
 m=read(INPUTS[scene]/('manifest.json' if split=='train' else 'evaluation_manifest.json'))
 f=next(x for x in m['frames'] if x['frame_id']==fid)
 gt=np.array(Image.open(f['image_path']).convert('RGB'))
 if coords is None:
  mask=np.array(Image.open(f['mask_path']).convert('L'))>=128
  yy,xx=np.nonzero(mask);pad=45
  coords=[max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(gt.shape[1],int(xx.max())+pad+1),min(gt.shape[0],int(yy.max())+pad+1)]
 x0,y0,x1,y1=coords
 imgs=[gt];assets=[asset(f['image_path']),asset(f['mask_path'])]
 for arm in ARMS:
  p=RUN/'evaluation'/scene/arm/split/(fid+'.npz')
  with np.load(p,allow_pickle=False) as z:imgs.append(np.rint(np.clip(z['rgb'],0,1)*255).astype(np.uint8))
  assets.append(asset(p))
 canvas=Image.new('RGB',(1400,430),'white');draw=ImageDraw.Draw(canvas)
 draw.text((5,5),f'{scene}  {split}  {fid}',font=font,fill='black')
 for i,(label,array) in enumerate(zip(['GT','C','P','PQ'],imgs)):
  p=Image.fromarray(array[y0:y1,x0:x1]);p.thumbnail((338,355),Image.Resampling.LANCZOS)
  canvas.paste(p,(i*350+(350-p.width)//2,67+(355-p.height)//2))
  draw.text((i*350+12,37),label,font=font,fill='black')
 record={'scene':scene,'split':split,'frame_id':fid,'bounds_xyxy':coords,'sources':assets,
  'display':'same ROI all methods; clip float RGB [0,1], round to uint8, Lanczos resize; no retouching'}
 return canvas,record

groups={
 'backpack_dynamic': [('Backpack','retained','00102',None),('Backpack','retained','00119',None)],
 'tennis_dynamic': [('Tennis','retained','00090',None),('Tennis','retained','00180',None)],
 'training_failure': [('Backpack','train','00122',None),('Tennis','train','00130',None)],
 'static_and_remaining': [('Tennis','retained','00054',None),('Backpack','retained','00221',None)]}
for name,cases in groups.items():
 sheet=Image.new('RGB',(1400,870),'white');recs=[]
 for i,args in enumerate(cases):
  pic,record=case(*args);sheet.paste(pic,(0,i*440));recs.append(record)
 p=OUT/'figures'/(name+'.png');p.parent.mkdir(parents=True,exist_ok=True);sheet.save(p)
 results['illustrations'].append({'image':asset(p),'cases':recs})

sources=[RUN/'configs/v10.json',RUN/'input_conditions.json',RUN/'NEXT_DECISION.md',
 ROOT/'hoi_modules/pose_support_gaussians.py',ROOT/'hoi_modules/pose_prior_adapter.py',ROOT/'hoi_modules/region_reconstruction.py',
 RUN/'code/train_scene.py',RUN/'code/render_scene.py',RUN/'code/evaluate.py',
 ROOT/'experiments/baseline_protocol_calibration_20260927/run01/code/evaluate_frozen.py',
 ROOT/'experiments/baseline_protocol_calibration_20260927/run01/code/export_hos_protocol.py',
 ROOT/'experiments/hosnerf_benchmark_20261006/run01/protocol/HOLD_FORMAL.json',
 ROOT/'experiments/local_dynamic_v9_20260929/run01/NEXT_DECISION.md',
 ROOT/'experiments/temporal_evidence_v8_20260928/run01/NEXT_DECISION.md']
results['source_identities']=[asset(p) for p in sources]
save(OUT/'evidence.json',results)
print(json.dumps(results['verification'],indent=2))
for scene,s in results['scenes'].items():
 print(scene,'P full',s['arms']['P']['metrics']['retained']['full'],'budget',s['arms']['P']['error_budget'])
