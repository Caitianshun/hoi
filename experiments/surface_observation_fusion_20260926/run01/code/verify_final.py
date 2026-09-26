"""Independent saved-output audit; no renderer, optimizer, or CUDA calls."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,cv2,torch
from skimage.metrics import structural_similarity
E=Path(__file__).resolve().parents[1];R=E.parents[2];A=R/'experiments/aux_ref_object_reconstruction_20260924/run01';B=R/'experiments/fixed_motion_reconstruction_20260926/run01'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=read(E/'protocol/finals.json');assert all(sha(r['path'])==r['sha256'] for r in freeze['assets'])
meta=read(A/'evaluation/regions/manifest.json');regions={(r['dev'],r['query_time_seconds']):r for r in meta['rows']};rows=read(E/'evaluation/per_frame.json');assert len(rows)==45
maxps=0.;maxss=0.
for row in rows:
 r=regions[row['dev'],row['time']];gt=cv2.imread(r['rgb']['path'])[...,::-1].astype(float)/255;lab=np.load(r['regions']['path'])['entity_labels'];pred=np.load(E/'evaluation'/row['dev']/r['frame_id']/f"{row['variant']}.npz")['rgb'].astype(float);O=lab==2
 ps=-10*np.log10(max(np.mean((pred[O]-gt[O])**2),1e-12));_,ss=structural_similarity(gt,pred,channel_axis=2,data_range=1.,full=True);ss=ss.mean(-1)[O].mean();maxps=max(maxps,abs(ps-row['metrics']['object']['psnr_db']));maxss=max(maxss,abs(ss-row['metrics']['object']['ssim']));assert row['metrics']['object']['pixels']==int(O.sum())
 assert sum(row['metrics'][k]['pixels'] for k in ['supported','single_view','no_positive_evidence','unmapped'])==int(O.sum())
assert maxps<1e-10 and maxss<1e-10
lookup={(r['dev'],r['variant'],r['time']):r for r in rows}
for p in read(E/'evaluation/paired_differences.json'):
 a,b=p['pair'].split('-');x=lookup[p['dev'],a,p['time']]['metrics'][p['region']][p['metric']];y=lookup[p['dev'],b,p['time']]['metrics'][p['region']][p['metric']];assert p['value'] is None if x is None or y is None else abs(p['value']-(x-y))<1e-12
checks={};costs=[]
for d in ['dev1','dev2']:
 b0=torch.load(A/'runs'/f'{d}_Ref/checkpoint_008000.pt',map_location='cpu',weights_only=False)['obank'];schedule=np.load(A/'frozen_aux'/f'{d}_frame_schedule.npy');checks[d]={}
 for v in ['F1','F2']:
  q=E/'runs'/f'{d}_{v}';logs=[json.loads(s) for s in (q/'steps.jsonl').read_text().splitlines()];assert len(logs)==2000 and np.array_equal([r['frame'] for r in logs],schedule[:2000]);assert all(np.isfinite(r['gradient_norm']) for r in logs);assert sum(r['updated'] for r in logs)==read(q/'run.json')['updates'];assert read(E/'preflight'/f'{d}_{v}/zero_step.json')['max_error']==0
  bank=torch.load(q/'baked_model.pt',map_location='cpu',weights_only=False)['obank'];assert all(torch.equal(bank[k],b0[k]) for k in bank if k!='color_logit');checks[d][v]={'nominal_steps':2000,'updates':sum(r['updated'] for r in logs),'geometry_topology_and_original_attributes_equal_except_baked_color':True,'export_error':read(q/'bake.json')['max_render_export_error']}
 b1=torch.load(B/'runs'/d/'checkpoint_008000.pt',map_location='cpu',weights_only=False)['obank'];off=.005*b1['offset']/torch.sqrt(1+(b1['offset']**2).sum(-1,keepdim=True));assert float(off.norm(dim=1).max())<=.005 and len(off)<=6000;checks[d]['B1']={'steps':8000,'points':len(off),'max_offset_m':float(off.norm(dim=1).max())}
for p in list((B/'preflight').glob('*/run.json'))+list((B/'runs').glob('*/run.json'))+list((E/'preflight').glob('*/run.json'))+list((E/'runs').glob('*/run.json'))+list((E/'runs').glob('*/bake.json'))+[E/'evaluation/run.json']:
 j=read(p);assert j['status']=='completed';costs.append({'path':str(p),'seconds':j['seconds'],'peak_allocated_bytes':j.get('peak_allocated_bytes',0)})
short=sum(read(p)['steps'] for p in list((B/'preflight').glob('*/run.json'))+list((E/'preflight').glob('*/run.json')));assert short==48
process_seconds=sum(r['seconds'] for r in costs);assert process_seconds<14400
result={'status':'passed','frozen_assets':len(freeze['assets']),'saved_render_PSNR_SSIM_rows':45,'max_PSNR_difference':maxps,'max_SSIM_difference':maxss,'checks':checks,'precheck_steps':short,'formal_steps':24000,'formal_runs':6,'cost_records':costs,'GPU_process_seconds_sum':process_seconds,'cost_definition':'sum of GPU-using task wall durations including CPU setup; not kernel profiler time','pipeline_wall_upper_bound_seconds':read(B/'pipeline.json')['seconds']+read(B/'preflight/dev1/run.json')['seconds']+read(E/'pipeline.json')['seconds'],'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in costs),'cache_bytes':sum(p.stat().st_size for p in (E/'support').glob('*/*.npz')),'historical_B0_render_max_difference':max(r.get('historical_B0_render_max_difference',0) for r in rows)}
(E/'protocol/final_integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['checks','cost_records']},indent=2))
