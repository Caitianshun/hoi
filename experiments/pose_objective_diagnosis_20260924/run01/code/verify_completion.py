"""Read-only numerical/artifact checks, followed by a frozen final index."""
from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np
E=Path(__file__).resolve().parents[1]
O=E.parents[1]/'object_pose_refinement_20260924/run01'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def save(p,a):Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
historical={}
def walk(v):
 if isinstance(v,dict):
  if isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str) and len(v['sha256'])==64:historical.setdefault(v['path'],set()).add(v['sha256'])
  for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
for name in ['audit_inputs_manifest.json','final_artifact_manifest.json']:walk(load(O/'protocol'/name))
walk(load(E/'protocol/frozen_diagnostic.json'))
checks=[];mismatch=[]
for p,expected in historical.items():
 path=Path(p)
 if not path.is_absolute():continue
 actual=sha(path) if path.is_file() else None;ok=actual in expected
 row={'path':p,'expected':sorted(expected),'actual':actual,'passed':ok};checks.append(row)
 if not ok:mismatch.append(row)
assert not mismatch,str(mismatch)
t=load(E/'track_toggle/summary.json');assert t['status']=='completed_input_outputs_frozen' and t['replay']['passed']
assert not t['continuation']['continue_both']
for name,n in [('on35',19),('off35',17)]:
 b=t['branches'][name];assert b['nfev']==n and b['success'] and sha(b['pose_path'])==b['pose_sha256']
 a=np.load(b['pose_path']);assert a['R_world'].shape==(114,3,3) and a['t_world_m'].shape==(114,3)
q=load(E/'evaluation/track_toggle_dev1/diagnostic_results.json');assert q['summaries']['on35']['prediction_slots_valid']==14 and q['summaries']['off35']['prediction_slots_valid']==14
for dev in ['dev1','dev2']:
 original=load(E/f'score_audit/same_pool_original/{dev}_scores.json');assert len(original['rows'])==4 and all(r['accepted'] for r in original['rows'])
 o=load(E/f'evaluation/oracle_{dev}/diagnostic_results.json');assert o['source_hashes_unchanged']
assert load(E/'evaluation/same_pool_ranking_comparison.json')
scored=0
for tag,num in [('base',370),('off35',65)]:
 run=load(E/f'score_audit/official/{tag}/scored/run.json');assert run['status']=='complete' and run['rows']==num and run['all_finite']
 rows=load(E/f'score_audit/official/{tag}/scored/scores.json');assert len(rows)==num
 for r in rows:assert r['valid'] and sha(r['debug_image'])==r['crop_render_sha256']
 scored+=len(rows)
pdf=E/'output/pdf/Pose_objective_diagnosis.pdf';qa=load(E/'protocol/pdf_qa.json');assert qa['status']=='passed' and qa['pdf_sha256']==sha(pdf) and qa['pages']==9
integrity={'status':'completed','utc':utc(),'historical_identity_checks':checks,'historical_file_count':len(checks),'historical_mismatches':[],'track_replay_exact':True,'full_path_frames':114,'toggle_output_frozen':True,'budget_extension_triggered':False,'paired_reference_slots':{'dev1':[14,14],'dev2':[9,10]},'paired_reference_intervals':{'dev1':[13,13],'dev2':[8,9]},'same_pool_per_event':4,'official_valid_scores':scored,'official_valid_crop_render_images':scored,'new_complete_gaussian_trainings':0,'independent_events_consumed':0,'new_model_export_and_heldout':'not triggered; no new model exists','pdf_qa':'passed,9pages,embedded_fonts/images,standalone_copy_identical_render','stage8_executed':'B same-pool scoring only; rejected by paired evidence','active_experiment_processes_required':False}
save(E/'protocol/completion_integrity.json',integrity)
save(E/'STATUS.json',{'status':'completed','completed_utc':utc(),'decision':'retain_old_S1_reject_track_off_and_official_ranking','training_gate':'not_met','new_8000_step_trainings':0,'stage8_branch':'B completed and rejected','report':str(pdf),'next_decision':str(E/'NEXT_DECISION.md'),'reproduce':str(E/'REPRODUCE.md'),'deadline':'stage decision completed2026-09-24; writing2026-11-05through15unchanged'})
artifacts=[]
for p in sorted(E.rglob('*')):
 if not p.is_file() or 'tmp' in p.relative_to(E).parts or '__pycache__' in p.parts or p.name=='final_artifact_manifest.json':continue
 artifacts.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
for p in [O/'code/build_report.py',O/'code/solve_pose.py',O/'code/evaluate_pose_pairs.py']:
 artifacts.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'role':'read_only_reused_source'})
save(E/'protocol/final_artifact_manifest.json',{'status':'frozen','utc':utc(),'root_git_repository':False,'source_control':'file SHA256 snapshots; no invented git commit','artifacts':artifacts})
print(json.dumps({'status':'completed','historical_checks':len(checks),'manifest_items':len(artifacts),'official_scores':scored,'new_trainings':0,'pdf_pages':9}))
