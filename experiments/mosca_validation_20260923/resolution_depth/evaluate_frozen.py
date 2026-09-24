"""CPU evaluation AFTER all 24 predictions are frozen; never fits scale."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import cv2
ROOT=Path('/home/cai_tianshun/Project/HOI');OUT=Path(__file__).resolve().parent
REF=ROOT/'research/2026-09-23/evaluation_pointcloud_reference'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(pred,ref,mask):
 a=np.abs(pred[mask]-ref[mask]);r=ref[mask]
 return dict(count=len(a),mean_abs_m=float(a.mean()),median_abs_m=float(np.median(a)),abs_rel=float((a/r).mean()),median_signed_m=float(np.median(pred[mask]-r)))if len(a)else dict(count=0)
def rasterize(entities,t,K,c2w,H,W):
 zbuf=np.full((H,W),np.inf);labels=np.zeros((H,W),np.uint8)
 for eid,d in entities:
  xyz=d['xyz_world'][t].astype(float);cam=(xyz-c2w[:3,3])@c2w[:3,:3];z=cam[:,2];hom=cam@K.T;uv=hom[:,:2]/hom[:,2:]
  for face in d['faces']:
   if np.any(z[face]<=0):continue
   xy=uv[face];lo=np.maximum(np.ceil(xy.min(0)).astype(int),0);hi=np.minimum(np.floor(xy.max(0)).astype(int),[W-1,H-1])
   if np.any(hi<lo):continue
   x,y=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
   a,b,c=xy;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
   if abs(den)<1e-12:continue
   wa=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
   wb=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den;wc=1-wa-wb
   inside=(wa>=-1e-7)&(wb>=-1e-7)&(wc>=-1e-7)
   invz=wa/z[face[0]]+wb/z[face[1]]+wc/z[face[2]]
   zz=np.divide(1.,invz,out=np.full_like(invz,np.inf),where=invz>0)
   sub=zbuf[lo[1]:hi[1]+1,lo[0]:hi[0]+1];target=labels[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
   update=inside&(zz<sub);sub[update]=zz[update];target[update]=eid
 return zbuf,labels
def main():
 start=time.perf_counter();run=json.loads((OUT/'prediction_run.json').read_text())
 assert run['status']=='completed'and run['all_predictions_frozen_before_evaluation']
 for cfg in run['configs']:assert sha(cfg['manifest'])==cfg['sha256']
 # First reference read happens only after above checks.
 refmeta=json.loads((REF/'manifest.json').read_text());entities=[]
 for name,eid in [('person',1),('object',2)]:
  p=refmeta['entities'][name]['path'];assert sha(p)==refmeta['entities'][name]['sha256'];entities.append((eid,np.load(p)))
 inputm=json.loads((OUT/'area640/input_manifest.json').read_text());K=np.array(inputm['K']);c2w=np.array(inputm['c2w']);H,W=480,640
 caches={};time_rows=[]
 evaldir=OUT/'evaluation_only';evaldir.mkdir(exist_ok=True)
 for f in inputm['frames']:
  matches=np.where(entities[0][1]['input_frame_indices']==f['source_baseline_index'])[0];assert len(matches)==1
  ti=int(matches[0]);assert abs(entities[0][1]['frame_times'][ti]-f['timestamp_seconds'])<1e-9
  depth,label=rasterize(entities,ti,K,c2w,H,W);caches[f['source_baseline_index']]=(depth,label)
  np.savez_compressed(evaldir/f'reference_{f["source_baseline_index"]:05d}.npz',dep=depth,entity=label,
     nominal_time=entities[0][1]['nominal_frame_times'][ti],matched_rgb_time=f['timestamp_seconds'],role=np.array('evaluation_only_fitted_visible_surface_proxy'))
  time_rows.append(dict(input_index=f['source_baseline_index'],nominal_s=float(entities[0][1]['nominal_frame_times'][ti]),matched_rgb_s=f['timestamp_seconds']))
  print('CPU fitted reference',f['source_baseline_index'],flush=True)
 results=[]
 for cfg in run['configs']:
  frozen=json.loads(Path(cfg['manifest']).read_text());rows=[]
  for f in frozen['frames']:
   assert sha(f['prediction_path'])==f['prediction_sha256'];pred=np.load(f['prediction_path']);dep=pred['dep'];conf=pred['confidence_raw']
   # Common640 pixel centres; no fitted scale/alignment, no mask derived from predictions.
   if dep.shape!=(H,W):dep=cv2.resize(dep,(W,H),interpolation=cv2.INTER_LINEAR);conf=cv2.resize(conf,(W,H),interpolation=cv2.INTER_LINEAR)
   ref,labels=caches[f['source_baseline_index']];row=dict(input_index=f['source_baseline_index'],entities={})
   for name,eid in [('person',1),('object',2)]:
    mask=labels==eid;interior=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);edge=mask&~interior
    row['entities'][name]={zone:metrics(dep,ref,ma)for zone,ma in [('all_fitted_visible',mask),('interior2px',interior),('boundary2px',edge)]}
    row['entities'][name]['raw_confidence_quantiles_on_interior']=np.quantile(conf[interior],[.1,.5,.9]).tolist()if interior.any()else[]
   rows.append(row)
  average={}
  for entity in ['person','object']:
   average[entity]={}
   for zone in ['all_fitted_visible','interior2px','boundary2px']:
    rr=[r['entities'][entity][zone]for r in rows if r['entities'][entity][zone]['count']]
    average[entity][zone]=dict(frame_count=len(rr),mean_of_frame_mae_m=float(np.mean([r['mean_abs_m']for r in rr])),mean_of_frame_abs_rel=float(np.mean([r['abs_rel']for r in rr])))if rr else dict(frame_count=0)
  results.append(dict(config=cfg['name'],per_frame=rows,equal_frame_summary=average))
 record=dict(status='completed_cpu_evaluation_after_prediction_freeze',script_sha256=sha(__file__),prediction_run_sha256=sha(OUT/'prediction_run.json'),
  reference_manifest=str(REF/'manifest.json'),reference_manifest_sha256=sha(REF/'manifest.json'),time_matches=time_rows,results=results,
  protocol='All predictions frozen before reading fitted geometry; CPU perspective-correct two-sided triangle z-buffer at common640; no scale/alignment fitting;1024 outputs bilinearly sampled to common640 grid; entity interiors eroded by2px for a fixed boundary sensitivity split.',
  limitations=['Registered reference, not sensor GT;14 fitted times have residual temporal/fitting uncertainty.','Visible means closest human/object fitted triangle, not independent visibility; no environment occluders included.','Human fit does not model glove/clothing surfaces precisely; fine contact/hand errors remain uncertain.','Six selected frames are diagnostics, not temporal tracking evaluation or full benchmark.','Raw confidence is an error-related per-image ranking, no calibration/probability or common threshold is assumed.'],elapsed_cpu_seconds=time.perf_counter()-start)
 (OUT/'evaluation_summary.json').write_text(json.dumps(record,indent=2)+'\n');print('evaluation completed',record['elapsed_cpu_seconds'],flush=True)
if __name__=='__main__':main()
