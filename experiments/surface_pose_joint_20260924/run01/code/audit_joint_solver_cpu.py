"""Independent CPU read-only checks of the prototype's inherited objective."""
from pathlib import Path
import datetime,hashlib,inspect,json,sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).parent))
import joint_solver as J
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stub(P,dtype):
 x=J.Objective.__new__(J.Objective);x.P=P;x.dev=P.dev;x.device='cpu';x.dtype=dtype;x.tensor=lambda a,dt=None:torch.as_tensor(np.asarray(a),dtype=dt or dtype,device='cpu');x.K=x.tensor(P.K);x.V=x.tensor(P.V);x.S=x.tensor(P.S);x.ts=x.tensor(P.ts);x.vis=x.tensor(P.visible,torch.bool);x.uv=x.tensor(np.stack([a[0] for a in P.obs]));x.fields=x.tensor(np.stack([a[1] for a in P.obs])[...,None]);x.xyz=x.tensor(np.stack([a[2] for a in P.obs]));x.good=x.tensor(np.stack([a[3] for a in P.obs]),torch.bool);return x

def main():
 torch.set_num_threads(2);rows=[]
 for dev in ['dev1','dev2']:
  P=J.Problem(dev)
  for dtype in [torch.float64,torch.float32]:
   obj=stub(P,dtype)
   for variant in ['old','fixed_input_only_perturbation']:
    R=P.R0.copy();t=P.t0.copy()
    if variant!='old':
     R[::5]=R[::5]@Rotation.from_rotvec([.01,-.005,.002]).as_matrix();t[::7]+=[.001,-.002,.003]
    old=P.blocks(R,t,[]);reference={k:float((2*(np.sqrt(1+a*a)-1)).sum()/P.L) for k,a in old.items() if k!='confirmed_tracks'}
    with torch.no_grad():new={k:float(v) for k,v in obj.base(obj.tensor(R),obj.tensor(t)).items()}
    diff={k:abs(new[k]-reference[k]) for k in reference};rows.append(dict(dev=dev,dtype=str(dtype),variant=variant,original=reference,new=new,absolute_difference=diff,passed=max(diff.values())<2e-4))
 source=(ROOT/'code/joint_solver.py').read_text();surface=(ROOT/'code/surface_constraints.py').read_text();findings=[{'priority':'resolved_blocker','issue':'surface_points NPZ initially repeated initial_face/initial_bary in explicit kwargs and state_np; fixed by parent during audit','current_fixed':"q0=o['q0'],q_final=qn,**state" in source},{'priority':'resolved_blocker','issue':'SurfaceConstraintSet has metadata(), not summary(); parent corrected the call','current_fixed':'surface.metadata()' in source and 'surface.summary()' not in source},{'priority':'note','issue':'The new2D and image robust residuals average vector channels before edge weighting; freeze explicitly: mean over2 coordinates /5 feature channels. This is a valid declared new-term choice, not inherited raw old-track normalization.'},{'priority':'note','issue':'Historical baseline depth is nearest sampled surface. The world/camera Euclidean distance implementation equals historical canonical distance for rotations in SO(3); stored float32 matrices and float32 timestamps introduce small numerical differences recorded here.'},{'priority':'note','issue':'Validity uses all optimization edges and fixed weights; heldout does not enter optimization, RGB fixed-source descriptor or sigma_F estimation. Out-of-bounds feature sampling clamps but separate unclamped uv and positive-depth penalties remain.'},{'priority':'note','issue':'Final surface_points has q_final from live float32 parameter and state q/bary float64 projected state; these can differ by float32 serialization. Use q_final for actual solved-point diagnostics and face/bary for surface certificates; check small residual explicitly.'},{'priority':'note','issue':'Model ray visibility is diagnostics only. Effective coverage combines template visibility and input mask, so report all factors rather than equating alpha/geometric visibility with material correctness.'}]
 checks={'old_base_cpu_port_parity':all(r['passed'] for r in rows),'all_sources_in_common_2d':'self.train[self.is_source].all()' in source,'image_only_nonsource_training':'self.opt_image=self.train&~self.is_source' in source,'fixed_W_and_WI':'self.W=self.weights[self.train].sum();self.WI=self.weights[self.opt_image].sum()' in source,'no_absolute_pose_prior':"old_pose_tether" not in inspect.getsource(J.Objective.loss),'fixed_and_movable_flags_correct':"fixed=variant[1]=='0';image_on=variant[2]=='1'" in source,'same_camera_coordinates_no_extra_extrinsic':"R[self.edge_frame],q[self.edge_track]" in source,'final_endpoint_not_reference_selected':"end point (no best-checkpoint search)" in source,'all_original_q_neighborhood_centres_fixed':'never reset' in surface,'surface_state_dump_collision_fixed':findings[0]['current_fixed'],'surface_metadata_method_correct':findings[1]['current_fixed']}
 result={'status':'passed_with_disclosed_numerical_and_metadata_notes','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reference_accessed':False,'GPU_used':False,'solver_modified_by_reviewer':False,'source_sha256':sha(ROOT/'code/joint_solver.py'),'surface_sha256':sha(ROOT/'code/surface_constraints.py'),'review_script_sha256':sha(__file__),'checks':checks,'base_numerical_checks':rows,'findings':findings,'limits':'CPU baseline-value parity and source audit; production CUDA gradients, eight-arm output integrity and post-freeze evaluation are separate required checks.'}
 assert all(checks.values()),checks
 (ROOT/'protocol/joint_solver_independent_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'checks':checks,'max_absolute_base_difference':max(max(r['absolute_difference'].values()) for r in rows)},indent=2))
if __name__=='__main__':main()
