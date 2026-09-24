"""Frozen paired object-pose evaluation, no prediction fitting/alignment.

Reference rotations are read analytically from three corresponding vertex IDs,
only after whole-mesh correspondence is verified. Rotations are never inferred
by aligning a prediction to a reference. New predictions require a prior
input-only selection/freeze file with exact path + SHA256 identities.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,time
import numpy as np
from scipy.spatial.distance import pdist
BASE=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def angle(a,b):
 rel=a@np.swapaxes(b,-1,-2);s=np.stack([rel[...,2,1]-rel[...,1,2],rel[...,0,2]-rel[...,2,0],rel[...,1,0]-rel[...,0,1]],-1)
 return np.degrees(np.arctan2(np.linalg.norm(s,axis=-1)/2,(np.trace(rel,axis1=-2,axis2=-1)-1)/2))
def basis(v,ids):
 a,b,c=ids;u=v[b]-v[a];u=u/np.linalg.norm(u);z=v[c]-v[a];z-=u*(u@z);z/=np.linalg.norm(z);return np.stack([u,z,np.cross(u,z)],-1)
def gauge(v,faces,ref):
 valid=ref['reference_available'].astype(bool);y=ref['object_vertices_world_m'].astype(np.float64);ids=[0,int(np.argmax(np.linalg.norm(v-v[0],axis=1))),None];ids[2]=int(np.argmax(np.linalg.norm(np.cross(v-v[0],v[ids[1]]-v[0]),axis=1)));B=basis(v,ids)
 rr=np.full((len(y),3,3),np.nan);tt=np.full((len(y),3),np.nan);errors=[];rigids=[]
 for i in np.flatnonzero(valid):
  rr[i]=basis(y[i],ids)@B.T;tt[i]=y[i].mean(0)-rr[i]@v.mean(0);errors.append(float(np.linalg.norm(v@rr[i].T+tt[i]-y[i],axis=1).max()));rigids.append(float(np.abs(pdist(v)-pdist(y[i])).max()))
 equal=np.array_equal(faces,ref['object_faces']);tol=1e-4;okay=equal and max(errors)<tol and max(rigids)<tol
 return rr,tt,{'verified':bool(okay),'faces_identical':bool(equal),'selected_vertex_ids_from_canonical_only':ids,'full_mesh_reconstruction_max_error_m':max(errors),'all_pair_distance_max_difference_m':max(rigids),'fixed_tolerance_m':tol,'method':'Two canonical corresponding edges form orthonormal bases; R_ref=B_ref@B_can.T; t_ref=mean(ref)-R_ref@mean(can). No SVD/ICP/optimization or prediction-dependent frame.'}
def stats(a):
 a=np.asarray(a,dtype=float);z=a[np.isfinite(a)]
 return {'mean':float(z.mean()),'median':float(np.median(z)),'p90':float(np.percentile(z,90)),'max':float(z.max()),'count':len(z),'requested_count':len(a)} if len(z) else {'mean':None,'median':None,'p90':None,'max':None,'count':0,'requested_count':len(a)}
def pose_arrays(p,ts):
 a=np.load(p);R=a['R_world'].astype(float);t=a['t_world_m' if 't_world_m' in a else 'translation_world_m'].astype(float)
 assert R.shape==(len(ts),3,3) and t.shape==(len(ts),3)
 assert np.allclose(a['timestamp_seconds'],ts,atol=1e-7,rtol=0),'Timestamp identity mismatch'
 finite=np.isfinite(R).all((1,2))&np.isfinite(t).all(1);valid=finite.copy();valid[finite]&=(np.linalg.norm(np.swapaxes(R[finite],1,2)@R[finite]-np.eye(3),axis=(1,2))<1e-4)&(np.abs(np.linalg.det(R[finite])-1)<1e-4)
 return R,t,valid,a

def evaluate(dev,predictions,out,selection):
 start=time.perf_counter();auditfile=BASE/'protocol/audit_inputs_manifest.json';symfile=BASE/'protocol/symmetry_geometry_only.json';audit=json.loads(auditfile.read_text());sy=json.loads(symfile.read_text());old=audit['old_S1'][dev]
 # Hashes and prior selection checked before the independent reference is opened.
 allowed_old={old['object_init'],str(Path(old['evaluation'])/'object_pose.npz')};freeze=json.loads(selection.read_text()) if selection else None
 if freeze:
  assert freeze['input_only_selection_complete'] is True
  assert freeze['symmetry_protocol_sha256']==sha(symfile),'Symmetry protocol must be frozen with prediction selection'
  allowed={(str(Path(r['path']).resolve()),r['sha256']) for r in freeze['frozen_predictions']}
 else:allowed=set()
 identities={label:{'path':str(p.resolve()),'sha256':sha(p)} for label,p in predictions.items()}
 for label,p in predictions.items():
  if str(p.resolve()) not in allowed_old:
   assert freeze is not None and (str(p.resolve()),sha(p)) in allowed,'New prediction not frozen by input-only selection'
 out.mkdir(parents=True,exist_ok=False)
 proto={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dev':dev,'script_sha256':sha(__file__),'audit_sha256':sha(auditfile),'symmetry_protocol_sha256':sha(symfile),'selection_freeze':str(selection) if selection else None,'selection_freeze_sha256':sha(selection) if selection else None,'prediction_identities':identities,'global_alignment':'none','time_interpolation':False,'centroid':'arithmetic mean of same indexed full template vertices including historical unreferenced vertices; same old definition','symmetry':'predeclared finite geometry-only set; raw metrics always retained; per-frame minimum is shape tolerance, not material correctness','material_accuracy':'not available; template fitted identities are not directly measured real material identity','bop_metrics':'not implemented; custom finite-set rotation/vertex model distances are not official BOP MSSD/MSPD'};save(out/'protocol.json',proto)
 ini=np.load(old['object_init']);v=ini['canonical_vertices_m'].astype(float);C=ini['c2w'];ts=ini['timestamp_seconds'];faces=ini['faces'];npred=len(ts);refentry=next(x for x in audit['evaluation_only'][dev]['files'] if x['path'].endswith('reference_meshes_world.npz'));assert sha(refentry['path'])==refentry['sha256'];ref=np.load(refentry['path']);Rref,tref,gv=gauge(v,faces,ref);save(out/'reference_gauge_verification.json',gv)
 frames=ref['matched_input_indices'];nominal=ref['reference_nominal_times_seconds'];available=ref['reference_available'];assert np.allclose(ref['matched_input_times_seconds'],ts[frames],atol=1e-7,rtol=0)
 yy=ref['object_vertices_world_m'].astype(float);cy=yy.mean(1);sym=[s for s in sy['dev'][dev]['all_candidates'] if s['accepted']];assert sym and sym[0]['name']=='identity';S=np.array([s['R'] for s in sym]);d=np.array([s['t_m'] for s in sym]);rows=[];summaries={};arrays={};curves=[]
 for label,path in predictions.items():
  R,t,valid,raw=pose_arrays(path,ts);cx=np.einsum('tij,j->ti',R,v.mean(0))+t;pred_v=np.einsum('tij,nj->tni',R[frames],v)+t[frames,None];delt=cx[frames]-cy;dc=delt@C[:3,:3];norm=np.linalg.norm(delt,axis=1);rawrot=np.full(len(frames),np.nan);allrot=np.full((len(frames),len(sym)),np.nan);allmodel=allrot.copy();rawrmse=np.sqrt(np.mean(np.sum((pred_v-yy)**2,axis=2),axis=1));validslots=available&valid[frames]
  if gv['verified']:
   for k in np.flatnonzero(validslots):
    rawrot[k]=angle(R[frames[k]],Rref[k]);allrot[k]=angle(R[frames[k]],Rref[k][None]@S)
    for j in range(len(sym)):
     vr=v@S[j].T+d[j];yr=vr@Rref[k].T+tref[k];allmodel[k,j]=np.sqrt(np.mean(np.sum((pred_v[k]-yr)**2,axis=1)))
  # One constant symmetry for the complete event is reported separately. Never
  # replace trajectory by a different best symmetry at each frame.
  bestglobal=int(np.nanargmin(np.nanmean(allrot,axis=0))) if gv['verified'] and validslots.any() else 0
  symrot=np.full(len(frames),np.nan);symmodel=symrot.copy();bestframe=np.full(len(frames),-1,int)
  for k in np.flatnonzero(validslots):
   if gv['verified']:bestframe[k]=int(np.argmin(allrot[k]));symrot[k]=allrot[k,bestframe[k]];symmodel[k]=float(allmodel[k].min())
  for z in [norm,dc,rawrmse]:z[~validslots]=np.nan
  stepdeg=angle(R[1:],R[:-1]);speed=np.linalg.norm(np.diff(cx,axis=0),axis=1)/np.diff(ts);dt=np.diff(ts)
  motion_rows=[]
  for k in range(1,len(frames)):
   j=k-1;ok=bool(validslots[k] and validslots[j]);f0,f1=int(frames[j]),int(frames[k]);dur=float(ts[f1]-ts[f0]);dis=float(np.linalg.norm(delt[k]-delt[j])*100) if ok else None;rd=float(angle(R[f1]@R[f0].T,Rref[k]@Rref[j].T)) if ok and gv['verified'] else None
   motion_rows.append({'start_input_frame':f0,'end_input_frame':f1,'dt_seconds':dur,'reference_available_at_both':bool(available[k] and available[j]),'prediction_valid_at_both':bool(valid[f0] and valid[f1]),'centre_displacement_error_cm':dis,'relative_rotation_error_deg':rd,'centre_velocity_error_cm_s':dis/dur if dis is not None else None})
  eligible=np.flatnonzero(validslots);branches=bestframe[eligible];switches=int(np.sum(branches[1:]!=branches[:-1])) if len(branches)>1 else 0
  for k,f in enumerate(frames):
   row={'method':label,'slot':k,'input_frame':int(f),'nominal_s':float(nominal[k]),'input_s':float(ts[f]),'input_minus_nominal_s':float(ts[f]-nominal[k]),'reference_available':bool(available[k]),'prediction_valid':bool(valid[f]),'status':'scored' if validslots[k] else ('missing_reference' if not available[k] else 'invalid_prediction'),'centroid_error_cm':float(norm[k]*100) if validslots[k] else None,'depth_bias_cm':float(dc[k,2]*100) if validslots[k] else None,'raw_rotation_error_deg':float(rawrot[k]) if np.isfinite(rawrot[k]) else None,'symmetry_rotation_error_deg':float(symrot[k]) if np.isfinite(symrot[k]) else None,'constant_event_symmetry_rotation_error_deg':float(allrot[k,bestglobal]) if np.isfinite(allrot[k,bestglobal]) else None,'best_frame_symmetry':sym[bestframe[k]]['name'] if bestframe[k]>=0 else None,'corresponding_vertex_rmse_cm':float(rawrmse[k]*100) if validslots[k] else None,'symmetry_model_rmse_cm':float(symmodel[k]*100) if np.isfinite(symmodel[k]) else None,'old_init_direct_observed':bool(ini['observed_mask'][f])};rows.append(row)
  summaries[label]={'reference_available':int(available.sum()),'reference_requested':len(available),'prediction_slots_valid':int(validslots.sum()),'full_prediction_frames_valid':int(valid.sum()),'full_prediction_frames_requested':npred,'centroid_error_cm':stats(norm*100),'camera_depth_signed_bias_cm':stats(dc[:,2]*100),'camera_depth_absolute_error_cm':stats(abs(dc[:,2])*100),'raw_rotation_error_deg':stats(rawrot),'symmetry_rotation_error_deg':stats(symrot),'constant_event_symmetry_rotation_error_deg':stats(allrot[:,bestglobal]),'constant_event_symmetry_name':sym[bestglobal]['name'],'per_reference_frame_best_branch_switch_count':switches,'corresponding_vertex_rmse_cm':stats(rawrmse*100),'symmetry_model_rmse_cm':stats(symmodel*100),'whole_sequence_motion':{'adjacent_angle_deg':stats(stepdeg),'angular_speed_deg_s':stats(stepdeg/dt),'centre_speed_m_s':stats(speed),'adjacent_over_90deg_indices':(np.flatnonzero(stepdeg>90)+1).tolist(),'note':'Raw unmodified trajectory; large rotation may be real motion. A threshold flag is not automatic evidence of identity error.'},'reference_interval_motion':motion_rows}
  arrays.update({label+'_centroid_world_m':cx,label+'_slot_error_world_m':delt,label+'_slot_rotation_error_deg':rawrot,label+'_slot_symmetry_rotation_deg':symrot,label+'_all_symmetry_rotation_deg':allrot,label+'_valid_frames':valid});curves.append((label,norm*100,dc[:,2]*100,rawrot,symrot))
 result={'status':'completed','dev':dev,'protocol':proto,'reference_source':refentry,'reference_gauge':gv,'accepted_symmetry_names':[s['name'] for s in sym],'summaries':summaries,'rows':rows,'limitations':['Fitted meshes are approximate evaluation references, not sensor or material ground truth.','Input-minus-nominal timestamp offsets are reported and are not actual inter-camera synchronization guarantees.','No per-frame alignment, scale fitting, reference interpolation or reference-dependent candidate selection.','Approximate symmetry and indexed-vertex residual are geometric diagnostics, not a material accuracy score.','This CPU scorer covers object pose; Gaussian queries, opacity-filtered surface proxies and heldout images must use existing dedicated frozen-export evaluators.'],'maximum_input_nominal_time_offset_s':float(abs(ts[frames]-nominal).max()),'source_hashes_unchanged':all(sha(r['path'])==r['sha256'] for r in identities.values()),'wall_seconds':time.perf_counter()-start};assert result['source_hashes_unchanged'];save(out/'paired_results.json',result)
 with (out/'paired_per_slot.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(out/'arrays.npz',**arrays,frame_indices=frames,reference_available=available,reference_nominal_times_seconds=nominal,actual_input_times_seconds=ts[frames])
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 for lab,ce,de,re,se in curves:
  for ax,values,title in zip(axes.ravel(),[ce,de,re,se],['Absolute template-centroid error (cm)','Signed camera-depth bias (cm)','Raw rotation error (degrees)','Predeclared symmetry rotation (degrees)']):ax.plot(ts[frames],values,'o-',label=lab);ax.set_title(title);ax.set_xlabel('Actual camera0 time (s)');ax.grid(alpha=.25)
 axes[0,0].legend();fig.savefig(out/'paired_pose_curves.png',dpi=160);plt.close(fig)
 print(json.dumps({'status':'completed','dev':dev,'output':str(out),'summary':{k:{'centre_cm':v['centroid_error_cm']['mean'],'raw_rotation_deg':v['raw_rotation_error_deg']['mean'],'symmetry_rotation_deg':v['symmetry_rotation_error_deg']['mean']} for k,v in summaries.items()}}))
def selftest():
 from scipy.spatial.transform import Rotation
 v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]]);R=Rotation.from_rotvec([.2,-.7,.3]).as_matrix();t=np.array([.1,.2,-.3]);f=np.array([[0,1,2],[0,1,3]]);y=v@R.T+t
 ref={'reference_available':np.array([True]),'object_vertices_world_m':y[None],'object_faces':f};rr,tt,g=gauge(v,f,ref);assert g['verified'] and np.allclose(rr[0],R) and np.allclose(tt[0],t);assert abs(float(angle(R,R)))<1e-10;assert abs(float(angle(np.eye(3),Rotation.from_rotvec([0,0,np.pi]).as_matrix()))-180)<1e-10
 y2=y.copy();y2[2,0]+=.1;ref['object_vertices_world_m']=y2[None];assert not gauge(v,f,ref)[2]['verified'];print(json.dumps({'selftest':'passed','analytic_known_transform':'exact','deformed_reference_gauge':'rejected','identity_and_180deg_angles':'passed'}))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dev',choices=['dev1','dev2']);p.add_argument('--prediction',action='append',default=[],help='LABEL=/absolute/path.npz');p.add_argument('--selection-freeze',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:selftest();return
 assert a.dev and a.prediction and a.output;pred={k:Path(v) for k,v in (x.split('=',1) for x in a.prediction)};evaluate(a.dev,pred,a.output,a.selection_freeze)
if __name__=='__main__':main()
