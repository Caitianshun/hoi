"""Bounded CPU QA: no model execution or changes to inspected artifacts."""
import hashlib,json,re,time
from pathlib import Path
import numpy as np

ROOT=Path('/home/cai_tianshun/Project/HOI')
BASE=ROOT/'experiments/mosca_baseline_20260922'
EXP=ROOT/'experiments/mosca_validation_20260923'
OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
start=time.perf_counter();checks=[]
def check_file(p,expected,kind):
 actual=sha(p);checks.append(dict(path=str(p),kind=kind,expected=expected,actual=actual,match=actual==expected))
manifest_path=BASE/'common_input/input_manifest.json';m=read(manifest_path)
root_report=EXP/'REPORT.md';report_before=sha(root_report)
parent=read(m['parent_manifest'])
check_file(m['parent_manifest'],m['parent_manifest_sha256'],'raw_frame_manifest')
check_file(m['source_video'],m['source_video_sha256'],'raw_video')
for label,mm in [('processed640_rgb',m),('original_resolution_rgb',parent)]:
 for p,h in zip(mm['frame_paths'],mm['frame_sha256']):check_file(p,h,label)
ta=read(ROOT/'research/2026-09-23/input_sampling_resolution_audit.json')
for k,v in ta.items():
 if isinstance(v,dict) and m['source_timestamps'] in v:
  check_file(m['source_timestamps'],v[m['source_timestamps']],'source_timestamps')
tf=read(EXP/'time_filter/run02_visual_trigger_fixed/results.json')
for p,h in tf['depth_file_sha256'].items():check_file(p,h,'old_depth_prior')
for p,h in tf['file_sha256'].items():
 if '/common_input/' in p or '/segmentation/' in p:
  check_file(p,h,'old_input_side_metadata_or_prior')
arun=read(EXP/'a_normalized_exact/model/run.json')
check_file(manifest_path,arun['input_manifest_sha256'],'A_input_manifest')
summary=read(EXP/'summary_a/summary.json')
check_file(BASE/'mosca_cotracker/diagnostics/query_trajectories.npz',
           read(BASE/'mosca_cotracker/diagnostics/export_manifest.json')['trajectory_sha256'],'old_prediction')
with np.load(BASE/'evaluation/fixed_rgb_queries/same_version_joint_reference_observation_times.npz') as f:
 r={k:f[k].copy() for k in f.files}
rows=[]
preds={'old_pilot':BASE/'mosca_cotracker/diagnostics/query_trajectories.npz',
       'gate_on':EXP/'motion_binding/gate_on_fixed_source_prediction.npz',
       'gate_off':EXP/'motion_binding/gate_off_fixed_source_prediction.npz',
       'a_normalized_exact':EXP/'a_normalized_exact/diagnostics/query_trajectories.npz'}
for name,p in preds.items():
 with np.load(p,allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
 assert z['query_id'].tolist()==r['query_id'].tolist()
 ii=r['prediction_frame_indices'];assert np.allclose(z['frame_times'][ii],r['frame_times'],atol=1e-9,rtol=0)
 assert z['coordinate_frame'].item()==r['coordinate_frame'].item()
 x=z['predicted'][ii];valid=z['predicted_valid_mask'][ii]&r['valid_mask']
 assert valid.all()
 error=np.linalg.norm(x-r['reference'],axis=-1)
 delta=np.linalg.norm((x-x[:1])-(r['reference']-r['reference'][:1]),axis=-1)[1:]
 obj=r['entity']=='object';hand=r['entity']=='hand'
 pair=(x[:,hand,None,:]-x[:,None,obj,:])-(r['reference'][:,hand,None,:]-r['reference'][:,None,obj,:])
 record=dict(name=name,absolute_m=float(error.mean()),object_absolute_m=float(error[:,obj].mean()),
             hand_absolute_m=float(error[:,hand].mean()),displacement_m=float(delta.mean()),
             object_displacement_m=float(delta[:,obj].mean()),hand_displacement_m=float(delta[:,hand].mean()),
             relative_m=float(np.linalg.norm(pair,axis=-1).mean()),absolute_count=int(error.size),
             displacement_count=int(delta.size),relative_count=int(pair.size//3))
 rows.append(record)
 if name in ['old_pilot','a_normalized_exact']:
  s=next(v for v in summary['runs'] if v['name']==name)
  for computed,field in [('absolute_m','absolute'),('object_absolute_m','object_absolute'),('displacement_m','displacement'),('object_displacement_m','object_displacement'),('hand_displacement_m','hand_displacement'),('relative_m','relative')]:
   assert abs(record[computed]-s[field]['mean_epe_m'])<1e-7,(name,computed)
  for entity,v in s['rgb'].items():assert abs(-10*np.log10(v['pooled_mse'])-v['pooled_psnr_db'])<1e-9
  er=EXP/'a_normalized_exact/evaluation/three_dimensional/report.json' if name=='a_normalized_exact' else BASE/'evaluation/mosca_cotracker_final/three_dimensional/report.json'
  # Older evaluator may be stored one level differently; source path is not guessed.
  if er.exists():check_file(er,s['evaluation_sha256'],name+'_evaluation_report')
  rr=EXP/'a_normalized_exact/render_summary/summary.json' if name=='a_normalized_exact' else BASE/'mosca_cotracker/render_summary/summary.json'
  if rr.exists():check_file(rr,s['render_sha256'],name+'_render_summary')
freeze=read(EXP/'motion_binding/prediction_freeze_manifest.json')
for p,h in freeze.items():check_file(EXP/'motion_binding'/p,h,'gate_prediction_freeze')
dep_run=read(EXP/'resolution_depth/prediction_run.json');depth_arrays=[]
for cfg in dep_run['configs']:
 check_file(cfg['manifest'],cfg['sha256'],'depth_prediction_freeze_manifest')
 cm=read(cfg['manifest'])
 for f in cm['frames']:
  check_file(f['prediction_path'],f['prediction_sha256'],'depth_prediction')
  with np.load(f['prediction_path']) as z:
   depth_arrays.append(dict(config=cfg['name'],frame=f['source_baseline_index'],
                            all_depth_finite_positive=bool((np.isfinite(z['dep'])&(z['dep']>0)).all()),
                            all_raw_confidence_finite=bool(np.isfinite(z['confidence_raw']).all())))
ed=read(EXP/'resolution_depth/evaluation_summary.json')
support={}
for cfg in ed['results']:
 counts={e:{zone:sum(f['entities'][e][zone]['count'] for f in cfg['per_frame'])
             for zone in ['all_fitted_visible','interior2px','boundary2px']}
         for e in ['person','object']}
 support[cfg['config']]=counts
assert len({json.dumps(v,sort_keys=True) for v in support.values()})==1
links=[];doc_hashes={}
docs=[p for p in EXP.rglob('*.md') if '/code/' not in str(p) and '/validation_qa/' not in str(p)]
for p in docs:
 doc_hashes[str(p)]=sha(p);text=p.read_text()
 for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)',text):
  dest=match.group(1).strip().strip('<>')
  if re.match(r'^[a-z]+://',dest):links.append(dict(document=str(p),target=dest,kind='external_not_fetched'));continue
  if dest.startswith('#'):continue
  dest=re.sub(r':\d+$','',dest.split('#')[0]);target=Path(dest)
  links.append(dict(document=str(p),target=str(target),kind='absolute_local' if target.is_absolute() else 'relative_local',
                    exists=(target if target.is_absolute() else p.parent/target).exists()))
result=dict(status='completed_cpu_readonly_checks',main_report_sha256_at_start=report_before,
            main_report_sha256_at_end=sha(root_report),main_report_changed_during_check=sha(root_report)!=report_before,
            old_input_hash_checks=checks,hash_check_count=len(checks),hash_mismatches=[v for v in checks if not v['match']],
            recomputed_motion=rows,depth_predictions=depth_arrays,shared_reference_support=support,
            source_A_summary_sha256=sha(EXP/'summary_a/summary.json'),markdown_document_sha256=doc_hashes,
            markdown_links=links,missing_links=[v for v in links if v.get('exists') is False],
            local_relative_links=[v for v in links if v['kind']=='relative_local'],
            script_sha256=sha(__file__),elapsed_cpu_seconds=time.perf_counter()-start)
(OUT/'checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:result[k] for k in ['hash_check_count','hash_mismatches','missing_links','local_relative_links','recomputed_motion','elapsed_cpu_seconds']},indent=2))
