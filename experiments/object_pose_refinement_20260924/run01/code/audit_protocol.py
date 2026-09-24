"""P0 provenance / prior audit. Reads historical assets; never fits or trains."""
from pathlib import Path
import json, hashlib, datetime, numpy as np, torch
ROOT=Path('/home/cai_tianshun/Project/HOI'); OLD=ROOT/'experiments/structured_hoi_20260923'; OUT=ROOT/'experiments/object_pose_refinement_20260924/run01'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def item(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def main():
 rows=json.loads((OLD/'summary/results.json').read_text())['rows'];result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed','role':'provenance_audit_not_prediction','path_source':item(OLD/'summary/results.json'),'training_inputs':{},'evaluation_only':{},'old_S1':{},'core_source_version':{},'seed':12345,'new_predictions_scored':False}
 for d in ['dev1','dev2']:
  r=next(x for x in rows if x['dev']==d and x['branch']=='S1'); run=Path(r['run']);ev=Path(r['evaluation']);runmeta=json.loads((run/'run.json').read_text()); ip=Path(runmeta['initialization']);data=torch.load(ip,map_location='cpu',weights_only=False);ws=Path(data['input_dir']);m=json.loads((ws/'input_manifest.json').read_text());op=Path(data['object_init']);o=np.load(op)
  rgb=[item(p) for p in m['frame_paths']];assert [x['sha256'] for x in rgb]==m['frame_sha256']
  caches=[ws/'input_manifest.json',Path(data['segmentation']),Path(data['human_geometry']),Path(data['smplx_model']),ws/'uniform_cotracker_tap.npz',op,ip]
  depth=[item(p) for p in sorted((ws/'unidepth_depth').glob('*.npz'))]
  result['training_inputs'][d]={'fixed_files':[item(p) for p in caches],'RGB':rgb,'estimated_depth':depth,'frame_indices':m['frame_indices'],'timestamp_seconds':m['timestamp_seconds'],'K':m['K'],'c2w':m['c2w'],'input_frame_count':len(rgb),'fixed_visible_frame_indices':np.flatnonzero(o['observed_mask']).tolist(),'unobserved_frame_indices':np.flatnonzero(~o['observed_mask']).tolist(),'object_template':{'units':'metres','vertex_count':len(o['canonical_vertices_m']),'face_count':len(o['faces']),'original_scan_center_m':o['original_scan_center_m'].tolist(),'canonical_vertex_mean_m':o['canonical_vertices_m'].astype(float).mean(0).tolist(),'transform':'X_world = R_world @ X_canonical + t_world_m; column-vector convention','canonical_geometry_sha256':hashlib.sha256(o['canonical_vertices_m'].tobytes()+o['faces'].tobytes()).hexdigest(),'scan_texture_used':False},'initialization_array_identities':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in data.items() if isinstance(v,np.ndarray)}}
  result['old_S1'][d]={'run':str(run),'evaluation':str(ev),'initialization':str(ip),'object_init':str(op),'artifacts':[item(p) for p in [run/'run.json',run/'checkpoint_008000.pt',run/'frame_schedule.npy',ev/'export_manifest.json',ev/'object_pose.npz',ev/'gaussian_identity.npz']], 'loss_weights':runmeta['loss_weights'],'schedule':runmeta['schedule'],'densities':runmeta['densities'],'training_code_identity':runmeta['training_code_identity'],'equivalent_full_train_seconds':r['equivalent_full_train_seconds'],'actual_prefix_shared':str(runmeta['resume']),'point_counts':r['final_points']}
  refs=[OLD/('data_audit/dev1_surface_reference/reference_meshes_world.npz' if d=='dev1' else 'data_audit/dev2_evaluation/reference_meshes_world.npz')]
  if d=='dev1':
   refs += list((ROOT/'research/2026-09-23/evaluation_pointcloud_reference').glob('*.json'))
   refs += [ROOT/'research/2026-09-23/evaluation_pointcloud_reference/object_fitted_vertices_world.npz']
   hp=Path(json.loads((ev/'heldout_metrics.json').read_text())['source_manifest'])
  else:hp=OLD/'data_audit/dev2_evaluation/heldout_manifest.json';refs+=[OLD/'data_audit/dev2_evaluation/reference_meshes_manifest.json']
  refs.append(hp)
  hm=json.loads(hp.read_text());refimages=[Path(x['heldout_gt']) for x in hm.get('pairs',[])] if 'pairs' in hm else [Path(x['path']) for x in hm['records']]
  regions=OLD/'data_audit/heldout_regions'/d
  result['evaluation_only'][d]={'files':[item(p) for p in refs],'heldout_RGB':[item(p) for p in refimages], 'region_assets':[item(p) for p in regions.rglob('*') if p.is_file()] if regions.exists() else [],'reference_policy':'Only independent scorer after a new prediction hash/selection is frozen. No reference-based parameter selection, alignment, initialization or colour sampling.','old_export_query':item(ev/'query_trajectories.npz')}
 for n in ['train_structured.py','gaussian_scene.py','human_lbs.py','prepare_initialization.py','evaluate_structured.py','score_structured_surface_cpu.py','diagnose_structured_support_cpu.py','evaluate_heldout_regions_cpu.py']:
  result['core_source_version'][n]=item(OLD/'code'/n)
 result['runtime']=json.loads((OLD/'protocol/frozen_training_v1/runtime.json').read_text())
 result['pose_prior_audit']={'source_fields':['object_R','object_t'],'rotation':'R(t)=Exp(object_rot_delta[t]) @ object_R0[t]; initial delta=0','translation':'object_translation initialized to object_t0','per_sample_soft_prior':{'rotation_axis_angle_delta_squared_mean':.005,'translation_delta_m_squared_mean':.05},'temporal_prior':{'quantities':['object_translation-object_t0','object_rot_delta'],'velocity_weight':.01,'acceleration_weight':.00005,'time':'actual timestamp finite differences'},'valid_frames':'No observed_mask/reliability gating. Soft prior at every sampled frame; correction temporal prior over whole sequence.','train_schedule':'Identical to historical S1, including warmup/stages and tracking end 8000.','implication':'Replacing object_R and object_t changes both initial learnable pose, soft prior targets, and correction temporal reference. Attribution is object pose initialization process, not initial R/t only.'}
 result['declared_S1_star_difference_plan']={'replace':['object_R','object_t','object_init provenance pointer'],'keep_byte_identical':[k for k in data if k not in ['object_R','object_t','object_init']],'do_not_rerun_prepare_initialization':True,'reason':'Recomputing colour from new projection would alter object RGB attributes and could change anchors/scales. Clone old torch init dict, replace exactly the three fields, and verify all remaining entries recursively.','training_from_step':0,'training_to_step':8000,'resume':None,'ordinary_tracker_cache':'same historical cache and frequency; new P1 tracklets do not enter S1* reconstruction loss','no_reference_read_by_initialization_entry':True}
 (OUT/'protocol/audit_inputs_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':'completed','path':str(OUT/'protocol/audit_inputs_manifest.json'),'sha256':sha(OUT/'protocol/audit_inputs_manifest.json')}))
if __name__=='__main__':main()
