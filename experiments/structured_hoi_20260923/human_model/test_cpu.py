"""Human branch acceptance using allowed frozen RGB prediction and template only."""
from pathlib import Path
import sys,time,json,hashlib
import numpy as np,torch
from smplx.lbs import batch_rodrigues
E=Path('/home/cai_tianshun/Project/HOI/experiments/structured_hoi_20260923');sys.path.insert(0,str(E/'code'));from human_lbs import HumanLBS
ROOT=E.parent.parent;RAW=ROOT/'experiments/gvhmr_validation_20260923/run01/raw_geometry.npz';ASSET=Path('/home/cai_tianshun/Project/mml/smpl_model/smplx/SMPLX_NEUTRAL.npz');torch.set_num_threads(2);start=time.perf_counter()
m=HumanLBS(RAW,ASSET);raw=np.load(RAW,allow_pickle=False);maxerr=0.;maxworlderr=0.
with torch.no_grad():
 for ids in torch.arange(m.num_frames).split(8):
  c,f=m(ids);assert c.shape==(len(ids),10475,3) and f.shape==(len(ids),10475,3,3)
  maxerr=max(maxerr,float(np.max(abs(c.numpy()-raw['vertices_camera_m'][ids.numpy()]))))
  world=c.numpy().astype(np.float64)@raw['c2w'][:3,:3].T+raw['c2w'][:3,3]
  maxworlderr=max(maxworlderr,float(np.max(abs(world-raw['vertices_world_m'][ids.numpy()]))))
assert maxerr<3e-6,(maxerr,'initial position mismatch')
# Centre and covariance path gradients must reach every parameter group.
c,f=m(17);wc=torch.linspace(.4,1.3,10475)[:,None]*torch.tensor([.2,-.3,.5]);wf=torch.arange(9).reshape(3,3).float()/11
loss=(c*wc).mean()+.04*(f*wf).mean();loss.backward();grad={n:float(p.grad.norm())for n,p in m.named_parameters()};assert all(np.isfinite(v)and v>1e-10 for v in grad.values()),grad
outsideframe_gradmax={n:float(p.grad[torch.arange(114)!=17].abs().max())for n,p in m.named_parameters()if n in ['body_pose','global_orient','transl','regional_residual_raw']};assert max(outsideframe_gradmax.values())==0
m.zero_grad(set_to_none=True)
# Pure covariance objective must update rotation, not silently detach frames.
c,f=m(17);cov0=torch.diag(torch.tensor([1e-4,3e-4,8e-4]));cov=f@cov0@f.transpose(-1,-2);covloss=(cov*torch.tensor([[.1,.3,.2],[.3,.6,-.2],[.2,-.2,.9]])).mean();covloss.backward();cov_grads={n:float(p.grad.norm())if p.grad is not None else None for n,p in m.named_parameters()};assert cov_grads['body_pose']>1e-10 and cov_grads['global_orient']>1e-10
with torch.no_grad():
 eig=torch.linalg.eigvalsh(cov);mineig=float(eig.min());assert mineig>0
 c0,f0=m(17);old=m.global_orient[17].clone();new=old+torch.tensor([.14,-.09,.07]);ro=batch_rodrigues(old[None])[0];rn=batch_rodrigues(new[None])[0];dr=rn@ro.T
 pivot=m.joint_template[0]+torch.einsum('k,ck->c',m.betas[0],m.joint_shape_directions[0])+m.transl[17]
 m.global_orient[17].copy_(new);c1,f1=m(17);rooterr=float((c1-((c0-pivot)@dr.T+pivot)).abs().max());frameerr=float((f1-torch.einsum('ij,njk->nik',dr,f0)).abs().max());assert rooterr<2e-6 and frameerr<2e-6;m.global_orient[17].copy_(old)
 # Bound stressed static and regional offsets independently and together.
 m.canonical_offset_raw.copy_(torch.arange(10475*3).reshape(-1,3)%7+100);m.regional_residual_raw.copy_(torch.arange(114*22*3).reshape(114,22,3)%11+150)
 staticmax=float(m.canonical_offsets().norm(dim=-1).max());regionalmax=float(m.regional_offsets().norm(dim=-1).max());assert staticmax<=.03000001 and regionalmax<=.01500001
 boundedc,_=m(17);combinedmax=float((boundedc-c0).norm(dim=-1).max());assert combinedmax<.045001
 m.canonical_offset_raw.zero_();m.regional_residual_raw.zero_()
 # Different frames remain independently correctable; time prior uses actual dt.
 m.regional_residual_raw[17,20]=torch.tensor([.2,-.1,.1]);offsets=m.regional_offsets();dtime=torch.diff(m.timestamps_seconds).float();vel=(offsets[1:]-offsets[:-1])/dtime[:,None,None];expected=vel.square().mean();reg=m.residual_regularization();assert torch.allclose(reg['regional_velocity_m2_s2'],expected)
 regularization_values={k:float(v)for k,v in reg.items()};m.regional_residual_raw.zero_()
# Explicit subset matches full attachment and identity; no cross-instance state.
ids=torch.tensor([0,10,100,522,1234,4321,9876,10474]);sub=HumanLBS(RAW,ASSET,vertex_ids=ids)
with torch.no_grad():
 sc,sf=sub(17);fc,ff=m(17);subseterr=float((sc-fc[ids]).abs().max());assert subseterr<2e-6;assert torch.allclose(sf,ff[ids],atol=1e-6)
 assert sub.body_pose.data_ptr()!=m.body_pose.data_ptr();before,_=sub(17);m.transl[17,0]+=.01;after,_=sub(17);assert torch.equal(before,after);m.transl[17,0]-=.01
assert set(n for n,_ in m.named_parameters())=={'body_pose','global_orient','transl','betas','canonical_offset_raw','regional_residual_raw'}
assert m.lbs_weights.shape[-1]==55 and m.body_region_weights.shape[-1]==22 and m.entity_ids.unique().tolist()==[1]
assert m.bone_to_body_region[22:25].tolist()==[15]*3 and m.bone_to_body_region[25:40].tolist()==[20]*15 and m.bone_to_body_region[40:].tolist()==[21]*15
record={'status':'passed_cpu','wall_seconds':time.perf_counter()-start,'frames_checked':114,'vertices_per_frame':10475,'reference_used':False,'init_camera_max_abs_error_m':maxerr,'init_world_max_abs_error_m':maxworlderr,'gradient_norms':grad,'other_frame_grad_max':outsideframe_gradmax,'covariance_only_gradient_norms':cov_grads,'covariance_min_eigenvalue_m2':mineig,'root_rotation_pelvis_position_max_error_m':rooterr,'root_rotation_frame_max_error':frameerr,'subset_position_max_error_m':subseterr,'residual_stress_max_canonical_m':staticmax,'residual_stress_max_regional_m':regionalmax,'residual_stress_max_combined_posed_m':combinedmax,'temporal_actual_dt_losses':regularization_values,'only_human_bones_and_body_regions':True,'instances_storage_independent':True,'source_identity':m.source_identity,'source_sha256':hashlib.sha256((E/'code/human_lbs.py').read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(E/'human_model/acceptance_cpu.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
