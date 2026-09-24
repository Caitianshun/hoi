"""Summarize only input scores, fixed pool and perturbation responses, never references."""
from pathlib import Path
import json,hashlib,datetime,csv
import numpy as np
ROOT=Path('/home/cai_tianshun/Project/HOI');N=ROOT/'experiments/pose_objective_diagnosis_20260924/run01';O=N/'score_audit/official'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
rows=json.loads((O/'base/scored/scores.json').read_text())
if (O/'off35/scored/scores.json').exists():rows+=json.loads((O/'off35/scored/scores.json').read_text())
summary={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'independent_reference_read':False,'rules_sha256':sha(N/'protocol/official_same_pool_ranking_rule.json'),'scoring_complete':{},'same_pool_rankings':{},'perturbation_response':[],'raw_candidates_reproduction':{},'facts':['Official candidate score was saved for all 50 candidates and used to retain top5 at each keyframe.','Original solve_pose coarse_candidates reads only R/t/frame_indices. Official scores are not read in local refinement, beam construction, full-path scoring or final path selection.','Official scoring crops depend on each pose hypothesis through unmodified crop_inputs, anchored at template origin; our perturbations preserve canonical vertex-mean center by fixed rule.','RGB source-patch term in original objective is constant within each frozen path, not new geometric evidence.','New official scores are coarse RGB logits and sigmoid values, not calibrated correctness probabilities.']}
for tag in ['base','off35']:
 p=O/tag/'scored/run.json'
 if p.exists():
  r=json.loads(p.read_text());summary['scoring_complete'][tag]={k:r.get(k) for k in ['status','rows','wall_seconds','hardware','all_finite','peak_allocated_bytes','peak_reserved_bytes']};summary['scoring_complete'][tag]['gpu_scoring_seconds']=sum(v['seconds'] for v in r['frames'])
for dev in ['dev1','dev2']:
 original_path=N/'score_audit/same_pool_original'/f'{dev}_scores.json';orig=json.loads(original_path.read_text())['rows'];po=[]
 for r in orig:
  name=r['name'];rs=[x for x in rows if x['dev']==dev and ((name=='old_initialization' and x['path_id']==name and x['variant']=='depth_+0.00') or (name!='old_initialization' and x['path_id']==name and x['category']=='full_path' and x['stage']=='after'))]
  assert len(rs)==5
  po.append({'name':name,'pose_file':r['path'],'pose_sha256':r['sha256'],'input_score':r['score'],'accepted':r['accepted'],'rejections':r['rejections'],'valid_official_frames':5,'required_frames':5,'mean_pose_logit':float(np.mean([x['pose_logit'] for x in rs])),'per_frame_logit':{str(x['frame']):x['pose_logit'] for x in rs}})
 good=[r for r in po if r['accepted']]
 original=sorted(good,key=lambda r:(r['input_score'],r['name']));official=sorted(good,key=lambda r:(-r['mean_pose_logit'],r['input_score'],r['name']))
 summary['same_pool_rankings'][dev]={'original_source':str(original_path),'original_source_sha256':sha(original_path),'rows':po,'original_order':[r['name'] for r in original],'official_order':[r['name'] for r in official],'original_first':original[0]['name'],'official_first':official[0]['name'],'diagnostic_only_no_new_training':True}
 raw=[r for r in rows if r['dev']==dev and r['category']=='single_keyframe_candidate'];d=[abs(r['pose_score']-r['historical_pose_score']) for r in raw];summary['raw_candidates_reproduction'][dev]={'n':len(raw),'max_abs_pose_score_difference':max(d),'mean_abs_difference':float(np.mean(d)),'reason_if_nonzero':'Historical scoring occurred after full coarse+refiner pipeline with its RNG consumption; audit directly loads weights with seed12345. Input poses identical, surface sampling and official neural operations remain official.'}
for dev,pathid in sorted(set((r['dev'],r['path_id']) for r in rows if r['category']=='perturbation')):
 subset=[r for r in rows if r['dev']==dev and r['path_id']==pathid and r['category']=='perturbation'];means=[]
 for variant in sorted(set(r['variant'] for r in subset)):
  v=[r for r in subset if r['variant']==variant];assert len(v)==5
  means.append({'variant':variant,'frames':5,'mean_logit':float(np.mean([r['pose_logit'] for r in v])),'min_logit':min(r['pose_logit'] for r in v),'max_logit':max(r['pose_logit'] for r in v),'per_frame':{str(r['frame']):r['pose_logit'] for r in v}})
 baseline=next(r['mean_logit'] for r in means if r['variant']=='depth_+0.00')
 for r in means:r['delta_from_baseline_mean']=r['mean_logit']-baseline
 dep=[r for r in means if r['variant'].startswith('depth_')];ori=[r for r in means if r['variant'].startswith('axis_')]
 summary['perturbation_response'].append({'dev':dev,'path_id':pathid,'means':means,'depth_logit_span':max(r['mean_logit'] for r in dep)-min(r['mean_logit'] for r in dep),'max_depth_variant':max(dep,key=lambda r:r['mean_logit'])['variant'],'max_orientation_variant':max(ori,key=lambda r:r['mean_logit'])['variant'],'depth_endpoint_preference':'descriptive of finite slice only, not a recommended correction'})
dump(O/'summary.json',summary)
with (O/'scores_flat.csv').open('w') as f:
 fields=['row_id','dev','frame','category','path_id','stage','variant','pose_logit','pose_score','valid','debug_image'];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps({'same_pool_rankings':summary['same_pool_rankings'],'raw_reproduction':summary['raw_candidates_reproduction'],'responses':[{k:v for k,v in r.items() if k!='means'} for r in summary['perturbation_response']]},indent=2))
