"""Read-only P1 replay and single-variable frozen-track toggle; no reference loader.

The original correspondence JSON supplies canonical attachments. We never call
correspondences()/ray casting. All 365 residual rows remain in the sparse system
when their weight is set to zero. Public helpers support separate score slices.
"""
from pathlib import Path
import argparse, collections, datetime, hashlib, importlib, inspect, io, json, os, platform, sys, time
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
ROOT=Path('/home/cai_tianshun/Project/HOI')
OLD=ROOT/'experiments/object_pose_refinement_20260924/run01'
NEW=ROOT/'experiments/pose_objective_diagnosis_20260924/run01'
sys.path.insert(0,str(OLD/'code'))
from solve_pose import Problem, plain, pvec, robust, project, sha, rotvec

def save(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(plain(a),indent=2,ensure_ascii=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load_frozen_records(P,path):
 """Reconstruct original residual rows from saved canonical JSON + immutable tracks."""
 meta=json.loads((Path(path)/'canonical_correspondences.json').read_text())
 lookup={(int(b['source_frame']),int(b['query'])):b for b in meta['bindings']}
 records=[]
 for f,tr in zip(P.kf,P.tracklets):
  ids=[q for q in range(tr['source_uv'].shape[1]) if lookup[(f,q)]['state']=='reliable_observed_conditional']
  ids=sorted(ids,key=lambda q:(-int(tr['adopted'][:,q].sum()),int(q)))
  for q in ids:
   valid=np.flatnonzero(tr['adopted'][:,q]);ca=np.array(lookup[(f,q)]['canonical_candidate'],dtype=float)
   for idx in valid:
    records.append((int(tr['target_frame'][idx,q]),ca,tr['target_uv'][idx,q],.5/(3*np.sqrt(max(1,len(valid))*max(1,len(ids)))),f,q))
 assert len({(r[4],r[5]) for r in records})==meta['accepted_queries']
 assert sum(r[0]!=r[4] for r in records)==meta['temporal_records']
 return records

def residual_blocks(P,R,t,records,track_scale=1.):
 """Exactly original pre-soft-L1 normalized residual blocks; track_scale only toggle."""
 out=P.blocks(R,t,records)
 out['confirmed_tracks_diagnostic']=out['confirmed_tracks'].copy()
 out['confirmed_tracks']=out['confirmed_tracks']*track_scale
 return out

TERMS=['silhouette_forward','silhouette_background','estimated_depth_weak','confirmed_tracks','actual_time_acceleration']
def residual_vector(blocks):
 return np.concatenate([np.stack([blocks[k] for k in TERMS[:3]],axis=1).ravel(),blocks['confirmed_tracks'].ravel(),blocks['actual_time_acceleration'].ravel()])

def summarize_blocks(P,blocks,records):
 weights=np.array([r[3] for r in records]);counts=np.array([int(o[3].sum()) if P.visible[i] else 0 for i,o in enumerate(P.obs)])
 raw={TERMS[0]:blocks[TERMS[0]]*16,TERMS[1]:blocks[TERMS[1]]*16,TERMS[2]:blocks[TERMS[2]]*(.1*np.sqrt(np.maximum(1,counts))/.15)[:,None],TERMS[3]:blocks['confirmed_tracks_diagnostic']/weights[:,None] if len(weights) else np.zeros((0,2)),TERMS[4]:blocks[TERMS[4]]*np.array([100]*3+[800]*3)}
 effective={TERMS[0]:int(P.visible.sum()*64),TERMS[1]:int(P.visible.sum()*64),TERMS[2]:int(counts.sum()),TERMS[3]:2*len(records),TERMS[4]:6*(P.L-2)}
 return {k:{'raw_squared_sum':float(np.square(raw[k]).sum()),'raw_abs_sum':float(abs(raw[k]).sum()),'weighted_squared_sum':float(np.square(blocks[k]).sum()),'weighted_robust_sum':float(robust(blocks[k]).sum()),'weighted_robust_mean_per_frame':float(robust(blocks[k]).sum()/P.L),'scalar_rows':int(blocks[k].size),'fixed_effective_scalar_count':effective[k]} for k in TERMS}

def pose_step(x,y):
 if y is None:return {'translation_max_m':0.,'translation_mean_m':0.,'rotation_max_deg':0.,'parameter_l2':0.}
 a=x.reshape(-1,6);b=y.reshape(-1,6);d=np.linalg.norm(a[:,3:]-b[:,3:],axis=1)
 angle=np.degrees(Rotation.from_matrix(Rotation.from_rotvec(a[:,:3]).as_matrix()@Rotation.from_rotvec(b[:,:3]).as_matrix().transpose(0,2,1)).magnitude())
 return dict(translation_max_m=float(d.max()),translation_mean_m=float(d.mean()),rotation_max_deg=float(angle.max()),parameter_l2=float(np.linalg.norm(x-y)))

def freeze(P,oldpath,out):
 records=load_frozen_records(P,oldpath);a=np.load(oldpath/'initial_path.npz');R=a['R_camera'];t=a['t_camera_m']
 assert P.L==114 and P.visible.sum()==103 and len(records)==365 and len({(r[4],r[5]) for r in records})==26
 assert np.array_equal(R,P.R0) and np.array_equal(t,P.t0)
 np.savez_compressed(out/'frozen_start.npz',R_camera=R,t_camera_m=t,timestamp_seconds=P.ts,observed_mask=P.visible,K=P.K,c2w=P.a['c2w'],canonical_vertices_m=P.V,faces=P.F)
 np.savez_compressed(out/'frozen_records.npz',frame=np.array([r[0] for r in records]),canonical=np.array([r[1] for r in records]),uv=np.array([r[2] for r in records]),weight=np.array([r[3] for r in records]),source_frame=np.array([r[4] for r in records]),query=np.array([r[5] for r in records]))
 filepaths=[oldpath/'initial_path.npz',oldpath/'canonical_correspondences.json',oldpath/'object_init.npz',oldpath/'optimizer.json',OLD/'code/solve_pose.py',OLD/'protocol/pose_config_v1.json',OLD/'protocol/pre_solver_clarification.json',OLD/'pose/dev1/optimization/selection_frozen.json',OLD/'reconstruction/dev1_initialization/controlled_difference.json',*sorted(P.B.glob('unidepth_depth/*.npz'))]
 manifest={**P.input_hashes,**{str(p):sha(p) for p in filepaths},str(Path(__file__).resolve()):sha(__file__)}
 source_counts=collections.Counter(r[4] for r in records);query_counts=collections.Counter((r[4],r[5]) for r in records);frame_counts=collections.Counter(r[0] for r in records)
 frames=[dict(frame=i,timestamp_seconds=float(P.ts[i]),fixed_visible=bool(P.visible[i]),track_records=frame_counts[i],low_observation_frame=frame_counts[i]<6,reliable_2d_records=sum(int(tr['adopted'][tr['target_frame'][:,0].astype(int)==i].sum()) for tr in P.tracklets),depth_samples=int(P.obs[i][3].sum()) if P.visible[i] else 0) for i in range(P.L)]
 save(out/'frame_observation_counts.json',frames)
 protocol={'frozen_utc':utc(),'old_path':str(oldpath),'dev':'dev1','input_only':True,'reference_accessed':False,'GPU_used':False,'original_code_unmodified':True,'original_gate_behavior':'Canonical acceptance and depth good frozen before fitting; no dynamic correspondence re-gating or re-binding. Projection-dependent convex hull and boundary positions are residual evaluation, not dropping observation identities.','record_source':'Saved canonical coordinates + original adopted tracklet rows, original stable order and weights; ray casting never rerun.','accepted_queries':26,'temporal_records':365,'self_records':sum(r[0]==r[4] for r in records),'budget':{'definition':'scipy least_squares max_nfev: objective calls excluding numerical Jacobian calls, not outer iterations; initial manual diagnostic call also excluded','initial':35,'cumulative_maximum':140,'continuation_trigger_predeclared':'Either branch status==0 AND (relative accepted objective reduction over last 5 recorded iterations >1e-3 OR final accepted max translation step >0.001 m OR rotation step >0.1 deg). Evaluated input-only; both branches continued once with 140-initial_nfev remaining.','continuation_optimizer_state':'Same algorithm/bounds/diff_step/tolerances; restart at own 35-budget endpoint (SciPy has no resume-state API), trust radius resets. Explicit new segment; cumulative nfev reported.','no_reference_trigger':True},'replay_tolerance':{'max_translation_m':1e-5,'max_rotation_deg':.001,'relative_robust_sum':1e-5},'unique_changed_variable':'confirmed_tracks residual multiplier 1 versus 0; keep same 365 rows and same Jacobian sparsity, all other weights and counts fixed','normalization':{'silhouette':'px/(2*sqrt(64)) = px/16; 64 each forward/background per frame','depth':'nearest surface distance_m/(0.1*sqrt(frozen good sample count))*0.15','tracks':'pixel component * 0.5/(3*sqrt(query reliable count * source confirmed query count)); SOFT_L1 applied after scaling','temporal':'actual-dt translation acceleration/10*0.1, rotation acceleration/80*0.1','pose_prior':'none; config old_pose_tether=0','RGB':'fixed 0.1*mean(max(0,1-NCC)) path score only; absent optimizer gradient','coverage':'fixed 0.1*(1-365/1794) path score only'},'source_record_counts':dict(source_counts),'query_record_counts':{f'{s}:{q}':v for (s,q),v in query_counts.items()},'hashes':manifest,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}
 save(NEW/'protocol/track_toggle_frozen.json',protocol)
 return records,R,t,protocol

class Tee:
 def __init__(self,*targets):self.targets=targets
 def write(self,s):
  for t in self.targets:t.write(s);t.flush()
 def flush(self):
  for t in self.targets:t.flush()

def optimize(P,R,t,records,out,track_scale,budget):
 out.mkdir(exist_ok=False);n=P.L;num=192*n+2*len(records)+6*(n-2);sp=lil_matrix((num,6*n),dtype=int)
 for i in range(n):sp[i*192:(i+1)*192,i*6:(i+1)*6]=1
 for k,(i,*_) in enumerate(records):sp[192*n+k*2:192*n+(k+1)*2,i*6:(i+1)*6]=1
 base=192*n+2*len(records)
 for i in range(1,n-1):sp[base+(i-1)*6:base+i*6,(i-1)*6:(i+2)*6]=1
 calls=[];vectors=[];iterations=[];iter_x=[];start=time.perf_counter()
 fhandle=(out/'objective_evaluations.jsonl').open('w')
 def fun(p):
  x=p.reshape(n,6);rr=Rotation.from_rotvec(x[:,:3]).as_matrix();tt=x[:,3:];b=residual_blocks(P,rr,tt,records,track_scale);v=residual_vector(b)
  frames=[];fr=inspect.currentframe().f_back
  while fr is not None:
   frames.append(fr.f_code.co_name);fr=fr.f_back
  role='finite_difference_jacobian' if '_sparse_difference' in frames else ('solver_trial' if 'trf_bounds' in frames else ('solver_initial' if 'least_squares' in frames else 'manual_initial_diagnostic'))
  entry={'call':len(calls)+1,'role':role,'elapsed_seconds':time.perf_counter()-start,'total_robust_sum':float(robust(v).sum()),'scipy_cost':float(robust(v).sum()/2),'terms':summarize_blocks(P,b,records),'disabled_track_diagnostic_robust_sum':float(robust(b['confirmed_tracks_diagnostic']).sum()),'step_from_previous_call':pose_step(p,vectors[-1] if vectors else None)}
  calls.append(entry);vectors.append(p.copy());fhandle.write(json.dumps(plain(entry))+'\n');fhandle.flush()
  return v
 lo=np.full((n,6),-np.inf);hi=-lo;lo[:,5]=.4;hi[:,5]=8;p0=pvec(R,t);before=fun(p0)
 trf=importlib.import_module('scipy.optimize._lsq.trf');original=trf.print_iteration_nonlinear
 def record_iteration(iteration,nfev,cost,actual_reduction,step_norm,g_norm):
  frame=inspect.currentframe().f_back;x=frame.f_locals['x'].copy();entry=dict(iteration=int(iteration),nfev=int(nfev),scipy_cost=float(cost),robust_sum=float(2*cost),actual_reduction=actual_reduction,step_norm=step_norm,optimality=float(g_norm),accepted_pose_step=pose_step(x,iter_x[-1] if iter_x else None))
  iterations.append(entry);iter_x.append(x);save(out/'iterations.json',iterations);original(iteration,nfev,cost,actual_reduction,step_norm,g_norm)
 trf.print_iteration_nonlinear=record_iteration
 oldstdout=sys.stdout
 try:
  with (out/'scipy_iteration_stdout.log').open('w') as log:
   sys.stdout=Tee(oldstdout,log)
   result=least_squares(fun,p0,jac_sparsity=sp.tocsr(),bounds=(lo.ravel(),hi.ravel()),loss='soft_l1',f_scale=1,max_nfev=budget,diff_step=1e-4,ftol=1e-4,xtol=1e-4,gtol=1e-4,verbose=2)
 finally:
  trf.print_iteration_nonlinear=original;sys.stdout=oldstdout;fhandle.close()
 xx=result.x.reshape(n,6);rr=Rotation.from_rotvec(xx[:,:3]).as_matrix();tt=xx[:,3:];np.savez_compressed(out/'all_evaluated_parameters.npz',parameters=np.array(vectors),accepted_iteration_parameters=np.array(iter_x))
 blocks=residual_blocks(P,rr,tt,records,track_scale);save(out/'final_components.json',summarize_blocks(P,blocks,records));np.savez_compressed(out/'final_residual_blocks.npz',**blocks)
 iou=P.ious(rr,tt);a=dict(P.a);a['R_camera']=rr.astype(np.float32);a['t_camera_m']=tt.astype(np.float32);C=a['c2w'];a['R_world']=(C[:3,:3]@rr).astype(np.float32);a['t_world_m']=(tt@C[:3,:3].T+C[:3,3]).astype(np.float32);a['reliability']=np.where(P.visible,np.minimum(1,(P.labels==2).sum((1,2))/1000)*iou,0).astype(np.float32)
 np.savez_compressed(out/'object_init.npz',**a);np.savez_compressed(out/'pose_float64.npz',R_camera=rr,t_camera_m=tt,timestamp_seconds=P.ts)
 outcome=dict(status='frozen',frozen_utc=utc(),seconds=time.perf_counter()-start,nfev=int(result.nfev),njev=int(result.njev),nit=int(iterations[-1]['iteration']),residual_calls_including_manual_initial=len(calls),call_counts=dict(collections.Counter(c['role'] for c in calls)),success=bool(result.success),termination_status=int(result.status),message=result.message,initial_robust_sum=float(robust(before).sum()),final_robust_sum=float(robust(result.fun).sum()),track_scale=track_scale,max_nfev=budget,optimality=float(result.optimality),mean_fixed_visible_iou=float(iou[P.visible].mean()),pose_path=str(out/'object_init.npz'),pose_sha256=sha(out/'object_init.npz'),float64_pose_sha256=sha(out/'pose_float64.npz'),rotation_orthonormal_max_error=float(abs(rr@rr.transpose(0,2,1)-np.eye(3)).max()),rotation_det_max_error=float(abs(np.linalg.det(rr)-1).max()),minimum_used_vertex_camera_depth_m=min(float((P.V[np.unique(P.F)]@rr[i].T+tt[i])[:,2].min()) for i in range(n)),all_nonpose_arrays_identical=all(np.array_equal(a[k],P.a[k],equal_nan=True) for k in P.a if k not in ['R_camera','t_camera_m','R_world','t_world_m','reliability']))
 save(out/'optimizer.json',outcome)
 return rr,tt,outcome,iterations

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=NEW/'track_toggle');args=ap.parse_args();out=args.output;out.mkdir(exist_ok=True,parents=True)
 if (out/'summary.json').exists() or (out/'on35').exists():raise RuntimeError('Refusing overwrite')
 P=Problem('dev1');oldpath=OLD/'pose/dev1/optimization/path00_old_start';records,R,t,protocol=freeze(P,oldpath,out)
 save(out/'run.json',dict(status='running',started_utc=utc(),pid=os.getpid(),no_reference_used=True))
 onR,ont,on,oni=optimize(P,R,t,records,out/'on35',1.,35)
 historical=np.load(oldpath/'object_init.npz');oldopt=json.loads((oldpath/'optimizer.json').read_text());trans=float(abs(ont-historical['t_camera_m']).max());angle=float(np.degrees(Rotation.from_matrix(onR@historical['R_camera'].transpose(0,2,1)).magnitude()).max());reldiff=abs(on['final_robust_sum']-oldopt['final_robust_sum'])/max(1,oldopt['final_robust_sum']);passed=trans<=1e-5 and angle<=.001 and reldiff<=1e-5
 replay=dict(passed=passed,maximum_translation_component_difference_m=trans,maximum_rotation_difference_deg=angle,relative_final_robust_sum_difference=reldiff,historical=oldopt,new=on,comparison_storage='Historical float32 pose versus new float64; serialization error included',frozen_records_only=True)
 save(out/'replay_comparison.json',replay)
 if not passed:
  save(out/'run.json',dict(status='blocked_replay_mismatch',replay=replay));raise RuntimeError('Replay mismatch; off branch not run')
 offR,offt,off,offi=optimize(P,R,t,records,out/'off35',0.,35)
 def trigger(result,it):
  last=it[-1];prior=it[max(0,len(it)-5)];reduction=(prior['robust_sum']-last['robust_sum'])/max(abs(prior['robust_sum']),1e-12);step=last['accepted_pose_step'];moving=reduction>1e-3 or step['translation_max_m']>.001 or step['rotation_max_deg']>.1
  return dict(trigger=bool(result['termination_status']==0 and moving),budget_terminated=result['termination_status']==0,last5_relative_reduction=reduction,last_step=step)
 decision={'on':trigger(on,oni),'off':trigger(off,offi)};decision['continue_both']=decision['on']['trigger'] or decision['off']['trigger'];save(out/'continuation_decision.json',decision)
 results={'on35':on,'off35':off}
 if decision['continue_both']:
  for name,rr,tt,result,scale in [('on',onR,ont,on,1.),('off',offR,offt,off,0.)]:
   _,_,long,li=optimize(P,rr,tt,records,out/(name+'140'),scale,140-result['nfev']);long['cumulative_nfev']=result['nfev']+long['nfev'];long['continuation_restart_disclosed']=True;long['optimization_unresolved_at_cap']=bool(long['cumulative_nfev']==140 and trigger(long,li)['trigger']);save(out/(name+'140')/'optimizer.json',long);results[name+'140']=long
 summary=dict(status='completed_input_outputs_frozen',frozen_utc=utc(),reference_used=False,replay=replay,continuation=decision,branches=results,all_outputs_before_independent_evaluation=True,code_sha256=sha(__file__),protocol_sha256=sha(NEW/'protocol/track_toggle_frozen.json'))
 save(out/'summary.json',summary);save(out/'run.json',summary);print(json.dumps(plain(summary),indent=2))
if __name__=='__main__':main()
