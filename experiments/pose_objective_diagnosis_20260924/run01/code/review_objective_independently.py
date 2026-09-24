"""Independent input-only numerical review; no solver execution or reference access."""
from pathlib import Path
import collections,datetime,hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/cai_tianshun/Project/HOI');NEW=Path(__file__).resolve().parents[1];OLD=ROOT/'experiments/object_pose_refinement_20260924/run01'
sys.path.insert(0,str(NEW/'code'))
from track_toggle import Problem,load_frozen_records,robust,TERMS,plain

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_pose(P,R,t,records,scale,meta):
 b=P.blocks(R,t,records)
 comp={k:float(robust(v*(scale if k=='confirmed_tracks' else 1.)).sum()/P.L) for k,v in b.items()}
 rgb=.1*(float(np.mean([max(0,1-x['patch_ncc']) for x in meta['cross_source_checks']])) if meta['cross_source_checks'] else 1.)
 coverage=.1*(1-sum(r[0]!=r[4] for r in records)/max(1,sum(int(tr['adopted'].sum()) for tr in P.tracklets)))
 return comp,dict(rgb_constant=rgb,coverage_constant=coverage),sum(comp.values())+rgb+coverage

def main():
 reports=[];assets={};config=json.loads((OLD/'protocol/pose_config_v1.json').read_text());sample_rows=[];pools=[];mechanism=[];validation_errors=[]
 # Review both completed versions so preserved pre-off data are also checked.
 files=sorted((NEW/'score_audit/objective').rglob('responses.json'))
 for dev in ['dev1','dev2']:
  P=Problem(dev);assets[dev]={'L':P.L,'fixed_visible':int(P.visible.sum()),'surface_sample_count':len(P.S),'configured_surface_samples':config['input_terms']['fitting_surface_samples'],'all_vertex_mean_m':P.V.mean(0).tolist()}
  for file in files:
   doc=json.loads(file.read_text())
   if doc['freeze']['dev']!=dev:continue
   maxdiff=0.;maxproj=0.;maxcenter=0.;nsamples=0;countfail=[]
   for case in doc['freeze']['cases']:
    name=case['name'];a=np.load(case['path']);R=a['R_camera'].astype(float);t=a['t_camera_m'].astype(float);cpath=Path(case['correspondence_path']).parent;records=load_frozen_records(P,cpath);meta=json.loads((cpath/'canonical_correspondences.json').read_text());scale=case['track_scale'];base,const,total=check_pose(P,R,t,records,scale,meta)
    for k in base:maxdiff=max(maxdiff,abs(base[k]-doc['cases'][name]['components'][k]))
    for row in [r for r in doc['rows'] if r['case']==name]:
     f=row['frame'];ntrack=sum(r[0]==f for r in records);ndepth=int(P.obs[f][3].sum()) if P.visible[f] else 0
     if row['track_record_count']!=ntrack or row['depth_good_count']!=ndepth or row['counts_changed']:countfail.append(dict(case=name,frame=f,kind=row['kind'],value=row['value']))
     # Deterministic representative subset: all frame/case centers and deepest ray slice;
     # both signs of canonical-axis rotations across all three axes at first keyframe.
     chosen=(row['kind']=='depth' and row['value'] in [-.1,0]) or (row['kind']=='rotation' and f==P.kf[0])
     if not chosen:continue
     rR=R.copy();rt=t.copy();c0=P.V.mean(0);c=R[f]@c0+t[f]
     if row['kind']=='depth':rt[f]=(1+row['value'])*c-R[f]@c0
     else:
      axis=np.eye(3)[row['axis']];rR[f]=R[f]@Rotation.from_rotvec(axis*np.deg2rad(row['value'])).as_matrix();rt[f]=c-rR[f]@c0
     cnew=rR[f]@c0+rt[f]
     if row['kind']=='depth':
      p1=P.K@c;p2=P.K@cnew;maxproj=max(maxproj,float(abs(p1[:2]/p1[2]-p2[:2]/p2[2]).max()))
     else:maxcenter=max(maxcenter,float(np.linalg.norm(cnew-c)))
     comp,constants,full=check_pose(P,rR,rt,records,scale,meta)
     diffs={k:abs(comp[k]-row['whole_components'][k]) for k in comp};diffs['whole_score']=abs(full-row['whole_score']);diffs['whole_objective_delta']=abs((full-total)-row['whole_objective_delta']);diffs['whole_observation_delta']=abs(sum(comp[k]-base[k] for k in comp if k!='actual_time_acceleration')-row['whole_observation_delta']);diffs['whole_temporal_delta']=abs(comp['actual_time_acceleration']-base['actual_time_acceleration']-row['whole_temporal_delta']);diffs['constants']=max(abs(constants[k]-row['constants'][k]) for k in constants)
     mx=max(diffs.values());maxdiff=max(maxdiff,mx);nsamples+=1;sample_rows.append(dict(file=str(file),case=name,frame=f,kind=row['kind'],value=row['value'],axis=row['axis'],maximum_difference=mx))
   reports.append(dict(file=str(file),sha256=sha(file),total_response_rows=len(doc['rows']),full_P_blocks_recomputations=nsamples,maximum_absolute_difference=maxdiff,maximum_depth_center_projection_change_px=maxproj,maximum_rotation_center_change_m=maxcenter,effective_count_mismatches=countfail,passed=maxdiff<1e-10 and not countfail))
  poolfile=NEW/f'score_audit/same_pool_original/{dev}_scores.json';pool=json.loads(poolfile.read_text());poolerrs=[]
  for row in pool['rows']:
   a=np.load(row['path']);cpath=OLD/f'pose/{dev}/optimization'/('path00_old_start' if row['name']=='old_initialization' else row['name']);records=load_frozen_records(P,cpath);meta=json.loads((cpath/'canonical_correspondences.json').read_text());comp,const,total=check_pose(P,a['R_camera'].astype(float),a['t_camera_m'].astype(float),records,1.,meta);mx=max([abs(total-row['score'])]+[abs(comp[k]-row['components'][k]) for k in comp]);poolerrs.append({'path_name':row['name'],'max_score_or_component_difference':mx,'record_count_matches':len(records)==row['record_count'],'accepted':row['accepted']})
  selected=min((r for r in pool['rows'] if r['accepted']),key=lambda x:(x['score'],x['name']))['name'];pools.append({'dev':dev,'file':str(poolfile),'rows':poolerrs,'selected_recomputed':selected,'selected_matches':selected==pool['selected'],'exact_pool_names':sorted(r['name'] for r in pool['rows'])})
 report={'status':'pass_with_metadata_caveats','reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'independent track_toggle subagent','reference_data_read':False,'GPU_used':False,'optimization_rerun':False,'objective_implementation_sha256':sha(NEW/'code/objective_response.py'),'review_script_sha256':sha(__file__),'configuration_sha256':sha(OLD/'protocol/pose_config_v1.json'),'assets':assets,'responses':reports,'all_response_sample_checks_passed':all(x['passed'] for x in reports),'same_pool':pools,'all_same_pool_checks_passed':all(x['selected_matches'] and all(y['max_score_or_component_difference']<1e-10 and y['record_count_matches'] for y in x['rows']) for x in pools),'configuration_vs_code':[{'field':'mask_samples_per_frame / image_sigma_px','configured':[64,2.0],'actual':'64 sampled object pixels per visible frame; forward/back each divided by16=2*sqrt64','mismatch':False},{'field':'fitting_surface_samples','configured':2048,'actual':'Both events historical centres_m[::2] has2048 points','mismatch':False},{'field':'estimated_depth','configured':'sigma0.1 m, multiplier0.15, sqrt(frozen good count)','actual':'Exact match; zeros in unusable slots retained','mismatch':False},{'field':'tracks','configured':'sigma3 px, multiplier0.5','actual':'Exact values; sqrt(per-query reliable rows * confirmed queries in same source), not global per-frame track normalization','mismatch':False,'clarification':'General config block_normalization prose says per-frame blocks, but pre_solver_clarification explicitly documents track per-source/query exception.'},{'field':'temporal','configured':'translation accel sigma10, rotation accel sigma80, multiplier0.1','actual':'Exact; actual time dt used; final score divided byL','mismatch':False},{'field':'old_pose_tether','configured':0,'actual':'No absolute pose prior residual','mismatch':False},{'field':'config operationality','actual':'Problem.C numerical fields except fixed_visible_counts mostly describe hardcoded constants; altering JSON alone would not alter these residual weights. Present numerical values match.','mismatch':False,'future_risk':True}], 'metadata_caveats':[{'field':'track_unweighted_robust_sum','issue':'Original field is calculated from bb confirmed_tracks, which already includes original sigma/multiplier/source-query normalization. It is independent of the on/off multiplier only, not raw unweighted pixel robust error.','correct_interpretation':'original_normalized_track_robust_sum_before_toggle','numeric_scores_affected':False},{'field':'tolerance_termination','issue':'on satisfied ftol+xtol; off satisfied xtol; optimality nonzero. Do not equate stopping criteria to strict stationary/global optimum.','numeric_scores_affected':False},{'field':'same_pool aggregation','issue':'Original score retains all-frame objective and its actual-time term; official proposed mean uses the same fixed five observed keyframes for every complete path. Same candidate pool does not mean identical evidence aggregation, and RGB score need not be calibrated geometry probability.','numeric_scores_affected':False}], 'physics_and_information_checks':{'depth_along_camera_center_ray':True,'rotation_about_fixed_historical_center_in_template_axes':True,'neighbor_poses_frozen_for_slices':True,'temporal_delta_separate':True,'RGB_and_coverage_constants_explicit':True,'fixed_visibility_depth_good_and_correspondence_identities':True,'raw_track_inputs_not_rebound':True,'world_camera_conversion_not_needed':'All source poses already R_camera/t_camera_m. Fixed K and metric template.'},'sample_rows':sample_rows}
 assert report['all_response_sample_checks_passed'] and report['all_same_pool_checks_passed']
 dest=NEW/'protocol/objective_independent_review.json';dest.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2)+'\n');print(json.dumps({'output':str(dest),'recomputations':len(sample_rows),'max_error':max(x['maximum_difference'] for x in sample_rows),'status':report['status']}))
if __name__=='__main__':main()
