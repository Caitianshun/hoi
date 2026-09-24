"""CPU-only frozen-input coarse visual check; not dataset GT or formal PCK."""
from pathlib import Path
import json,hashlib,csv,time
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
T=time.perf_counter();O=Path(__file__).resolve().parent;E=O.parent;R=E/'run01';Q=E/'input_quality'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
run=json.loads((R/'run.json').read_text());assert run['status']=='completed'
assert sha(Q/'assistant_visual_proxy.json')=='7f78cbae843a957a754374f3dcfde403e54cdb6a95ad75332f3850a0c5b1e1b0'
ann=json.loads((Q/'assistant_visual_proxy.json').read_text());src=json.loads((Q/'source_manifest.json').read_text());assert ann['fixed_frames']==[0,16,25,31,55,65,93,113]
files=['vitpose.pt','raw_geometry.npz','official_postproc_geometry.npz']
for f in files:assert sha(R/f)==run['outputs'][f]
kp=torch.load(R/'vitpose.pt',map_location='cpu',weights_only=True).numpy(); assert kp.shape==(114,17,3)
npzs={n:np.load(R/(n+'_geometry.npz')) for n in ['raw','official_postproc']}
uv={'ViTPose':kp[...,:2],**{n:z['joints_coco17_uv'] for n,z in npzs.items()}}
for z in npzs.values():assert np.array_equal(z['vitpose'],kp)
lookup={('left','shoulder'):5,('right','shoulder'):6,('left','elbow'):7,('right','elbow'):8,('left','wrist'):9,('right','wrist'):10}
rows=[]
for a in ann['rows']:
 if not a['numeric_comparison_eligible']:continue
 f=a['frame'];j=lookup[(a['anatomical_side'],a['joint'])];target=np.asarray(a['xy_original_640x480'],float);tol=a['uncertainty_plusminus_pixels_per_axis']
 for method,p in uv.items():
  pred=p[f,j];delta=pred-target
  rows.append({'frame':f,'anatomical_side':a['anatomical_side'],'joint':a['joint'],'coco17_index':j,'method':method,'proxy_x':target[0],'proxy_y':target[1],'pred_x':float(pred[0]),'pred_y':float(pred[1]),'dx_pixels':float(delta[0]),'dy_pixels':float(delta[1]),'euclidean_pixels':float(np.linalg.norm(delta)),'tolerance_per_axis_pixels':tol,'within_x_tolerance':bool(abs(delta[0])<=tol),'within_y_tolerance':bool(abs(delta[1])<=tol),'within_both_axis_tolerance':bool((abs(delta)<=tol).all()),'vitpose_confidence':float(kp[f,j,2]) if method=='ViTPose' else None})
summary={n:{'compared_points':8,'within_both_axis_tolerance_count':sum(r['within_both_axis_tolerance'] for r in rows if r['method']==n),'formal_PCK':False} for n in uv}
with (O/'per_point.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result={'status':'completed_numerical_and_images_pending_visual_review','run':str(R),'run_status':'completed','proxy_is_ground_truth':False,'coordinate_system':'original frozen 640x480 image, x right/y down','method':'Direct COCO17 identity comparison; no alignment, side swap, confidence filtering or fine-tuning','uncertainty':'Per-axis assistant conservative localization range; not calibrated statistical CI','scope':'Only 8 input-only pre-frozen eligible elbows/wrists. Count is coarse compatibility check, not formal PCK. Do not use in training or hyperparameter choice. No 3D/contact claim.','summary':summary,'rows':rows,'hashes':{str(p):sha(p) for p in [R/'run.json',Q/'assistant_visual_proxy.json',Q/'source_manifest.json',*[R/f for f in files],Path(__file__)]}}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
CROP=(160,130,380,440);scale=2;tw=440;th=680
colors={'left':(45,255,70),'right':(255,55,225),'central':(245,245,245)}
# Anatomical left/right retain model identity; occluded predictions shown for diagnosis, not treated as observed.
left_edges=[(5,7),(7,9),(5,11),(11,13),(13,15),(1,3)];right_edges=[(6,8),(8,10),(6,12),(12,14),(14,16),(2,4)];central_edges=[(0,1),(0,2),(5,6),(11,12)]
frames=[]
for fr in src['frames']:
 f=fr['input_frame'];p=Path(fr['source']);assert sha(p)==fr['sha256'];original=Image.open(p).convert('RGB').crop(CROP).resize((tw,620),Image.Resampling.NEAREST)
 row=Image.new('RGB',(tw*4,th),'white')
 for ci,name in enumerate(['RGB',*uv]):
  tile=original.copy();d=ImageDraw.Draw(tile)
  if name!='RGB':
   points=(uv[name][f]-np.array(CROP[:2]))*scale
   for edges,col in [(left_edges,colors['left']),(right_edges,colors['right']),(central_edges,colors['central'])]:
    for ia,ib in edges:d.line([tuple(points[ia]),tuple(points[ib])],fill='black',width=5);d.line([tuple(points[ia]),tuple(points[ib])],fill=col,width=3)
   for i,xy in enumerate(points):
    col=colors['left'] if i in [1,3,5,7,9,11,13,15] else colors['right'] if i in [2,4,6,8,10,12,14,16] else colors['central'];x,y=xy;d.ellipse((x-3,y-3,x+3,y+3),fill=col,outline='black')
   for label,j in [('LE',7),('RE',8),('LW',9),('RW',10)]:
    x,y=points[j];d.text((x+4,y+3),label,font=small,fill=colors['left' if label[0]=='L' else 'right'],stroke_width=1,stroke_fill='black')
   for a in ann['rows']:
    if a['frame']!=f or not a['numeric_comparison_eligible']:continue
    x,y=(np.asarray(a['xy_original_640x480'])-np.array(CROP[:2]))*scale;u=a['uncertainty_plusminus_pixels_per_axis']*scale
    d.rectangle((x-u,y-u,x+u,y+u),outline=(255,190,0),width=2)
  row.paste(tile,(ci*tw,60));dr=ImageDraw.Draw(row);label={'RGB':'RGB / no overlay','raw':'GVHMR raw','official_postproc':'GVHMR official postproc'}.get(name,name);dr.text((ci*tw+6,5),f'f{f:03d} {label}',font=font,fill='black');dr.text((ci*tw+6,31),'L: green  R: magenta  Proxy box: orange' if ci else f'Fixed crop x160:380 y130:440',font=small,fill='black')
 row.save(O/f'overlay_frame_{f:03d}.png');frames.append(row)
for k in range(4):
 sheet=Image.new('RGB',(tw*4,th*2),'white');sheet.paste(frames[k*2],(0,0));sheet.paste(frames[k*2+1],(0,th));sheet.save(O/f'overlay_sheet_{k+1}.png')
result['wall_seconds']=time.perf_counter()-T
(O/'scores.json').write_text(json.dumps(result,indent=2))
print(json.dumps(summary,indent=2))
