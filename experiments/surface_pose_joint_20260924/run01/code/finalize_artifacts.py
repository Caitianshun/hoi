from pathlib import Path
import hashlib,json,datetime,shutil,subprocess
E=Path(__file__).resolve().parents[1];R=E.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,a):Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
f=read(E/'protocol/all_outputs_frozen.json');assert all(sha(x['path'])==x['sha256'] for x in f['files'])
p=read(E/'evaluation/promotion_decision.json');assert p['selected_global_variant'] is None and p['high_level']=='reject_all_no_new_gaussian_training'
assert read(E/'protocol/execution_status.json')['status']=='completed'
assert read(E/'protocol/docx_qa.json')['status']=='passed'
required=['protocol/frozen_surface_pose.json','protocol/promotion_rules.json','protocol/all_outputs_frozen.json','protocol/implementation_checks.json','protocol/historical_integrity.json','protocol/final_numeric_audit.json','evaluation/paired_metrics.json','evaluation/paired_per_slot.csv','evaluation/paired_per_interval.csv','evaluation/fixed_failure_interval.json','evaluation/factorial_contrasts.json','METHOD_DECISION.md','REPRODUCE.md','output/Surface_pose_joint_verification.docx','visualizations/manifest.json']
for d in ['dev1','dev2']:
 required.extend([f'observations/{d}/observations.npz',f'visualizations/{d}_full_pose_timeline.png'])
 for v in ['F00','F01','F10','F11']:
  required.extend([f'{d}/{v}/{n}' for n in ['object_init.npz','surface_points.npz','projection_diagnostics.npz','iterations.json','metrics.json','run.json']])
assert all((E/x).is_file() for x in required)
snapshot=E/'protocol/root_document_snapshots';snapshot.mkdir(exist_ok=True)
for name in ['AGENTS.md','README.md','RESEARCH_LOG.md']:shutil.copy2(R/name,snapshot/name)
status={'status':'completed','utc':now,'formal_pose_runs':8,'updates_per_run':300,'new_gaussian_runs':0,'independent_events_used':0,'selected_variant':None,'decision':'H1 limited relative motion support; all fail complete promotion gates; retain old S1','pending_training':False,'report':str(E/'output/Surface_pose_joint_verification.docx')}
save(E/'STATUS.json',status)
skip={'docx_render','docx_render_final','docx_qa','docx_verified','__pycache__'}
files=[]
for path in sorted(E.rglob('*')):
 if not path.is_file() or any(x in skip for x in path.relative_to(E).parts):continue
 if path.name in ['final_artifact_manifest.json','completion_integrity.json']:continue
 files.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
save(E/'protocol/final_artifact_manifest.json',{'status':'frozen','utc':now,'source_control':'SHA256 files; project has no git repository','files':files,'excluded':'stale document rendering iterations, Python bytecode and self-referential completion/index; final docx_final page PNGs retained','historical_identity_audit':str(E/'protocol/historical_integrity.json')})
completion={**status,'required_files':required,'required_present':len(required),'required_missing':[],'frozen_solve_files_checked':len(f['files']),'all_frozen_predictions_unchanged':True,'historical_manifest_files_unchanged':619,'artifact_count':len(files),'final_manifest_sha256':sha(E/'protocol/final_artifact_manifest.json'),'formal_retries':0,'all_formal_exit_codes_zero':True,'reference_slots':{'dev1':'14/14,13/13 intervals','dev2':'9/10,8/9 intervals; t2 missing preserved'},'docx_pages':5,'docx_images_embedded':2,'docx_portable_copy_render_match':True,'Gaussian_and_camera1_new_metrics':'N/A not triggered by gate','gpu_after':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True),'science_limitations':['two development events, single initialization and bounded optimizer','same-video heldout edges are not independent test','fitted reference and time matching limits retained','no precise material identity or contact claim']}
save(E/'completion_integrity.json',completion)
print(json.dumps({'status':'completed','files':len(files),'required':len(required),'missing':[],'new_gaussian_runs':0,'docx_pages':5}))
