"""Post-freeze CPU projection QA and the previously fixed glove-error case."""
from pathlib import Path
import numpy as np,cv2,json,sys,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/cai_tianshun/Project/HOI');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/diagnostics'))
from evaluate_cotracker_reference import sample_depth
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 run=json.loads((OUT/'prediction_run.json').read_text());assert run['status']=='completed'and run['all_predictions_frozen_before_evaluation']
 m=json.loads((OUT/'area640/input_manifest.json').read_text());sam=np.load(ROOT/'experiments/mosca_baseline_20260922/segmentation/segmentation.npz')['entity_labels']
 fig,axs=plt.subplots(2,3,figsize=(14,8),dpi=120);rows=[]
 for ax,f in zip(axs.flat,m['frames']):
  i=f['source_baseline_index'];ref=np.load(OUT/f'evaluation_only/reference_{i:05d}.npz');lab=ref['entity'];rgb=cv2.imread(f['path']);item={'frame':i,'entities':{}}
  for label,c in [(1,(255,150,30)),(2,(0,220,255))]:
   mask=(lab==label).astype(np.uint8);cnt,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,cnt,-1,c,1)
   interior=cv2.erode(mask,np.ones((5,5),np.uint8)).astype(bool);pred=sam[i]==label
   item['entities'][str(label)]={'fit_visible_pixels':int(mask.sum()),'fit_interior_pixels':int(interior.sum()),'fit_interior_overlap_sam_predicted_region_fraction':float(pred[interior].mean())if interior.any()else None}
  ax.imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB));ax.set_title(f'Frame {i}; blue=person fit, yellow=box fit');ax.axis('off');rows.append(item)
 fig.suptitle('Independent fitted-surface projection audit; outlines are not exact observed silhouettes');fig.tight_layout();fig.savefig(OUT/'figures/reference_projection_qa.png');plt.close(fig)
 (OUT/'evaluation_only/reference_projection_qa.json').write_text(json.dumps({'status':'completed_cpu','script_sha256':sha(__file__),'comparison_role':'fit vs SAM2 prediction; neither assumed exact silhouette GT','frames':rows},indent=2)+'\n')
 source=ROOT/'experiments/mosca_baseline_20260922/evaluation/cotracker_matched_unidepth/largest_error_audit.json';a=json.loads(source.read_text());rows=[]
 for c in run['configs']:
  assert sha(c['manifest'])==c['sha256'];manifest=json.loads(Path(c['manifest']).read_text());f=[r for r in manifest['frames']if r['source_baseline_index']==113][0]
  assert sha(f['prediction_path'])==f['prediction_sha256'];x=np.load(f['prediction_path']);scale=x['dep'].shape[1]/640
  uv=(np.array([a['pred_uv'],a['reference_uv']])+.5)*scale-.5;z=sample_depth(x['dep'],uv);conf=sample_depth(x['confidence_raw'],uv)
  rows.append(dict(config=c['name'],input_scale=scale,sample_uv=uv.tolist(),camera_z_at_fixed_cotracker_pixel_m=float(z[0]),camera_z_at_fixed_reference_projection_m=float(z[1]),confidence_raw_at_these_pixels=conf.tolist(),fitted_reference_surface_z_m=a['fitted_reference_camera_z_m']))
 output=dict(status='completed_postfreeze_evaluation_only',script_sha256=sha(__file__),source=str(source),source_sha256=sha(source),rows=rows,
  protocol='Previously fixed largest-error case; identical two pixels mapped by half-pixel convention; no query relocation or scale fitting. Raw confidence is error-related, not calibrated probability.',
  limitation='One case; fitted glove association uncertain; no 3D surface accuracy improvement claim from sampled depth alone.')
 (OUT/'evaluation_only/fixed_glove_case.json').write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
