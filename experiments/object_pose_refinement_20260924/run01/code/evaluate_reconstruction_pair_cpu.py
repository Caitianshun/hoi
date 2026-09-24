"""Chain CPU S1/S1* pose, surface, support and heldout evaluation after export.

Old assets remain read-only. Reuses historical surface/support metric files only
when their scorer SHA matches today's unchanged old scorer. Heldout old/new RGB
is rescored in one new directory with the exact original region masks.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
BASE=Path(__file__).resolve().parents[1]
OLD=Path('/home/cai_tianshun/Project/HOI/experiments/structured_hoi_20260923')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def load(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dev',required=True,choices=['dev1','dev2']);ap.add_argument('--export',type=Path);ap.add_argument('--new-init',type=Path);ap.add_argument('--selection-freeze','--selection',dest='selection_freeze',type=Path,required=True);ap.add_argument('--output',type=Path);a=ap.parse_args()
 new=a.export or BASE/f'reconstruction/{a.dev}_S1_star/evaluation';init=a.new_init or BASE/f'reconstruction/{a.dev}_initialization/initialization.pt';out=a.output or BASE/f'evaluation/{a.dev}_reconstruction_pair';out.mkdir(parents=True,exist_ok=False)
 begin=time.perf_counter();rec={'status':'running','role':'evaluation_only_no_optimization','dev':a.dev,'pid':os.getpid(),'new_export':str(new),'new_init':str(init),'stages':{},'source_script_sha256':sha(__file__)};atomic(out/'pipeline.json',rec)
 def run(name,command):
  stamp=time.perf_counter();rec['current_stage']=name;rec['stages'][name]={'status':'running','command':command};atomic(out/'pipeline.json',rec)
  with (out/f'{name}.log').open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd='/home/cai_tianshun/Project/HOI')
  rec['stages'][name].update(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode,wall_seconds=time.perf_counter()-stamp);atomic(out/'pipeline.json',rec)
  if r.returncode:raise RuntimeError(f'{name} failed, see {out/name}.log')
 try:
  audit=load(BASE/'protocol/audit_inputs_manifest.json');oldmeta=audit['old_S1'][a.dev];old=Path(oldmeta['evaluation']);ex=load(new/'export_manifest.json');assert ex['status']==ex.get('evaluation_status')=='completed';assert ex['prediction_geometry_frozen_before_reference'];assert ex['global_transform_fitted_on_evaluation'] is False;assert ex['initialization_sha256']==sha(init)
  parent=load(a.selection_freeze)
  selected_pose=None
  if 'selected' in parent:
   assert parent['status']=='accepted_for_S1_star' and parent['no_independent_reference_used'] is True
   selected_pose=Path(parent['selected']['pose_file']);assert sha(selected_pose)==parent['selected']['pose_sha256']
  else:
   assert parent['input_only_selection_complete'] is True
   candidates=[Path(r['path']) for r in parent['frozen_predictions'] if 'object_pose' not in Path(r['path']).name]
   if len(candidates)==1:selected_pose=candidates[0]
  sym=BASE/'protocol/symmetry_geometry_only.json'
  if 'symmetry_protocol_sha256' in parent:assert parent['symmetry_protocol_sha256']==sha(sym)
  selection_records={}
  for devname in ['dev1','dev2']:
   selection_path=BASE/f'pose/{devname}/optimization/selection_frozen.json'
   if 'selected' in parent:
    assert selection_path.exists(),f'Both dev input selections must freeze before any independent evaluation: {selection_path}'
    decision=load(selection_path);assert decision.get('no_independent_reference_used') is True
    assert decision.get('status') not in [None,'running','pending','candidate']
    selection_records[devname]={'path':str(selection_path),'sha256':sha(selection_path),'status':decision['status']}

  # This is a post-training export freeze; it does not select or rescore a P1 candidate.
  freeze={'input_only_selection_complete':True,'parent_selection_freeze':str(a.selection_freeze),'parent_selection_freeze_sha256':sha(a.selection_freeze),'symmetry_protocol_sha256':sha(sym),'frozen_predictions':[{'path':str((new/'object_pose.npz').resolve()),'sha256':sha(new/'object_pose.npz')}],'completed_export_manifest_sha256':sha(new/'export_manifest.json'),'prediction_geometry_frozen_before_reference':True,'both_dev_input_freezes':selection_records,'selection_note':'Only one completed S1* model for preselected input initialization. All predictions hashed before CPU paired reference scores; no candidate reselection.'};
  if selected_pose is not None:freeze['frozen_predictions'].append({'path':str(selected_pose.resolve()),'sha256':sha(selected_pose)})
  atomic(out/'posttraining_prediction_freeze.json',freeze)
  surface_script=OLD/'code/score_structured_surface_cpu.py';support_script=OLD/'code/diagnose_structured_support_cpu.py';heldout_script=BASE/'code/evaluate_heldout_regions_paired.py'
  for script in [surface_script,support_script]:assert sha(script)==audit['core_source_version'][script.name]['sha256']
  oldsurface=old/('surface_evaluation' if a.dev=='dev1' else 'surface_reference');oldsupport=old/'support_diagnostic';surface_code_matches=load(oldsurface/'protocol.json')['script_sha256']==sha(surface_script);support_code_matches=load(oldsupport/'protocol.json')['script_sha256']==sha(support_script)
  reference=Path(next(x['path'] for x in audit['evaluation_only'][a.dev]['files'] if x['path'].endswith('reference_meshes_world.npz')))
  import torch
  ip=torch.load(init,map_location='cpu',weights_only=False)
  if not surface_code_matches:
   run('surface_old_rescore',[sys.executable,str(surface_script),'--export',str(old),'--reference',str(reference),'--output',str(out/'surface_old_S1_rescore')]);oldsurface=out/'surface_old_S1_rescore'
  if not support_code_matches:
   old_init=torch.load(oldmeta['initialization'],map_location='cpu',weights_only=False)
   run('support_old_rescore',[sys.executable,str(support_script),'--export',str(old),'--segmentation',old_init['segmentation'],'--output',str(out/'support_old_S1_rescore')]);oldsupport=out/'support_old_S1_rescore'
  if selected_pose is not None:
   run('initialization_pose',[sys.executable,str(BASE/'code/evaluate_pose_pairs.py'),'--dev',a.dev,'--prediction',f'old_init={oldmeta["object_init"]}','--prediction',f'P1={selected_pose}','--selection-freeze',str(out/'posttraining_prediction_freeze.json'),'--output',str(out/'initialization_pose')])
  run('pose',[sys.executable,str(BASE/'code/evaluate_pose_pairs.py'),'--dev',a.dev,'--prediction',f'old_S1={old}/object_pose.npz','--prediction',f'S1_star={new}/object_pose.npz','--selection-freeze',str(out/'posttraining_prediction_freeze.json'),'--output',str(out/'pose')])
  run('surface',[sys.executable,str(surface_script),'--export',str(new),'--reference',str(reference),'--output',str(out/'surface_S1_star')])
  run('support',[sys.executable,str(support_script),'--export',str(new),'--segmentation',ip['segmentation'],'--output',str(out/'support_S1_star')])
  run('heldout',[sys.executable,str(heldout_script),'--dev',a.dev,'--evaluation',f'old_S1={old}','--evaluation',f'S1_star={new}','--output',str(out/'heldout')])
  # Historical scores are included with hashes, never overwritten.
  results={'status':'completed','dev':a.dev,'old_export':str(old),'new_export':str(new),'pose':load(out/'pose/paired_results.json'),'surface':{'old_S1':load(oldsurface/'metrics.json'),'S1_star':load(out/'surface_S1_star/metrics.json')},'support':{'old_S1':load(oldsupport/'diagnostic.json'),'S1_star':load(out/'support_S1_star/diagnostic.json')},'heldout':{key:load(out/f'heldout/{key}_metrics.json') for key in ['old_S1','S1_star']},'old_result_reuse':{'surface_protocol':str(oldsurface/'protocol.json'),'surface_protocol_sha256':sha(oldsurface/'protocol.json'),'surface_metrics_sha256':sha(oldsurface/'metrics.json'),'support_protocol':str(oldsupport/'protocol.json'),'support_protocol_sha256':sha(oldsupport/'protocol.json'),'support_metrics_sha256':sha(oldsupport/'diagnostic.json'),'surface_reused':surface_code_matches,'support_reused':support_code_matches,'reason':'Historical metric reused only when scorer SHA matches; otherwise old export rescored into new directory with the same current scorer as S1_star.'},'fixed_query_policy':'dev1 original 6 alpha-bound queries/14 slots remain in exporter reference_evaluation. dev2 matching 6-query material truth is N/A.','cost':{'cpu_pair_seconds':time.perf_counter()-begin,'new_export_record':ex,'old_equivalent_full_train_seconds':oldmeta['equivalent_full_train_seconds']},'limitations':['Input support is a proxy reliability gate, not independent material identity.','Whole-entity contribution and frozen-local-family contribution are distinct.','No LPIPS addition; preserve historical N/A.','One seed and two previously diagnosed development events cannot support statistical significance or blind-test claims.']}
  if selected_pose is not None:results['initialization_pose']=load(out/'initialization_pose/paired_results.json')
  for key,directory in [('old_S1',old),('S1_star',new)]:
   q=directory/'reference_evaluation/three_dimensional/report.json'
   if q.exists():results.setdefault('fixed_query_results',{})[key]={'path':str(q),'sha256':sha(q),'metrics':load(q)}
   elif (directory/'reference_evaluation').exists():results.setdefault('fixed_query_results',{})[key]={'path':str(directory/'reference_evaluation'),'files':[str(p) for p in (directory/'reference_evaluation').glob('*.json')]}
   else:results.setdefault('fixed_query_results',{})[key]={'status':'not_available','reason':'No matching reference queries for this dev.'}
  atomic(out/'paired_reconstruction_results.json',results);rec['status']='completed';rec['result']=str(out/'paired_reconstruction_results.json')
 except BaseException as e:
  rec.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
 finally:rec['wall_seconds']=time.perf_counter()-begin;atomic(out/'pipeline.json',rec)
if __name__=='__main__':main()
