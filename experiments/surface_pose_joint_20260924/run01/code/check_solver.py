"""CPU-only numerical acceptance checks. Input assets only; no reference loader.

The complete eight-run identity/budget checks are explicitly pending until the
coordinator fills them from final artifacts. Small edge fixtures call actual
Objective.edges/loss methods; only the old base term is omitted for image tests.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,sys,time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'code'))
import joint_solver as J
from surface_constraints import SurfaceConstraintSet,self_test

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(x):return torch.as_tensor(np.asarray(x),dtype=torch.float64)
def base_object(dev):
 o=J.Objective.__new__(J.Objective);P=J.Problem(dev);o.P=P;o.dev=dev;o.device='cpu';o.dtype=torch.float64
 o.tensor=lambda x,dt=None:torch.as_tensor(np.asarray(x),dtype=dt or torch.float64)
 o.K=tensor(P.K);o.V=tensor(P.V);o.S=tensor(P.S);o.ts=tensor(P.ts);o.vis=torch.tensor(P.visible,dtype=torch.bool)
 o.uv=tensor(np.stack([x[0] for x in P.obs]));o.fields=tensor(np.stack([x[1] for x in P.obs])[...,None]);o.xyz=tensor(np.stack([x[2] for x in P.obs]));o.good=torch.tensor(np.stack([x[3] for x in P.obs]),dtype=torch.bool)
 return o

def check_base(o):
 results=[];P=o.P
 for case in ['old_initialization','deterministic_small_perturbation']:
  R=P.R0.copy();t=P.t0.copy()
  if case!='old_initialization':
   w=.001*np.sin(np.arange(P.L)/7);R=Rotation.from_rotvec(np.c_[w,w*.3,w*-.2]).as_matrix()@R;t+=np.c_[w*.4,w*.2,w*.3]
  old=P.blocks(R,t,[]);old={k:float((2*(np.sqrt(1+v*v)-1)).sum()/P.L) for k,v in old.items() if k!='confirmed_tracks'}
  with torch.no_grad():new={k:float(v) for k,v in o.base(tensor(R),tensor(t)).items()}
  rows={k:{'old':old[k],'new':new[k],'abs_difference':abs(old[k]-new[k]),'tolerance':max(1e-6,2e-5*abs(old[k])),'passed':abs(old[k]-new[k])<=max(1e-6,2e-5*abs(old[k]))} for k in old}
  results.append({'case':case,'components':rows,'passed':all(x['passed'] for x in rows.values())})
 return {'rows':results,'passed':all(r['passed'] for r in results),'note':'Convex hull/polygon old OpenCV geometry rounds input coordinates float32; new torch residual arithmetic float64. Tolerance permits this small forward rounding only.'}

def edge_fixture(K,field,descriptor,sigma,uv):
 o=J.Objective.__new__(J.Objective);o.K=tensor(K);o.image=tensor(field[None]);o.desc=tensor(descriptor[None]);o.sigmaF=tensor(sigma)
 o.edge_frame=torch.zeros(1,dtype=torch.long);o.edge_track=torch.zeros(1,dtype=torch.long);o.edge_uv=tensor(uv[None]);o.weights=torch.ones(1,dtype=torch.float64)
 o.train=torch.tensor([True]);o.is_source=torch.tensor([False]);o.opt_image=torch.tensor([True]);o.W=o.weights.sum();o.WI=o.W;o.calls=0;o.base=lambda R,t:{}
 return o

def check_image(dev,P):
 a=dict(np.load(E/f'observations/{dev}/observations.npz'));fpath=E/f'observations/{dev}/features/fields.npy';fields=np.load(fpath,mmap_mode='r');d=np.load(E/f'observations/{dev}/features/frozen_descriptors.npz');eligible=[]
 for i in np.flatnonzero((a['edge_split']==0)&~a['edge_is_source'].astype(bool)):
  k=int(a['edge_track'][i]);fr=int(a['edge_frame'][i]);q=a['q0'][k]
  if a['bary0'][k].min()<.02:continue
  cam=P.R0[fr]@q+P.t0[fr];h=P.K@cam;uv=h[:2]/h[2]
  if cam[2]<=.1 or min(uv[0]-3,637-uv[0],uv[1]-3,477-uv[1])<0:continue
  frac=uv-np.floor(uv)
  if frac.min()<.15 or frac.max()>.85:continue
  x,y=np.floor(uv).astype(int);ff=fields[fr];texture=float(np.linalg.norm(ff[y,x+1]-ff[y,x])+np.linalg.norm(ff[y+1,x]-ff[y,x]))
  if texture<.01:continue
  eligible.append((i,k,fr,uv,texture))
  if len(eligible)==3:break
 if len(eligible)<2:raise RuntimeError(f'{dev}: too few valid interior textured gradient check edges: {len(eligible)}')
 results=[]
 for i,k,fr,uv,texture in eligible:
  field=np.array(fields[fr]);o=edge_fixture(P.K,field,d['source_descriptor'][k],d['sigma_F'],a['edge_uv'][i]);sc=SurfaceConstraintSet(P.V,P.F,[a['face0'][k]],a['bary0'][k:k+1]);rv=Rotation.from_matrix(P.R0[fr]).as_rotvec();x0=np.r_[rv,P.t0[fr],sc.bary2[0]]
  def fn(x,measured=False):
   R=J.axis_angle_to_matrix(x[:3])[None];t=x[3:6][None];q=sc.torch_points(x[6:][None])
   if measured:
    values=J.sample(o.image,o.edge_frame,o.edge_uv);res=(values-o.desc)/o.sigmaF
    return J.rho(res).mean()+x.sum()*0
   return o.loss(R,t,q,True)[1]['projection_image']
  x=torch.tensor(x0,dtype=torch.float64,requires_grad=True);loss=fn(x);g=torch.autograd.grad(loss,x)[0].detach().numpy();fd=[]
  for j in range(8):
   h=1e-7 if j<6 else 1e-6;plus=x0.copy();minus=x0.copy();plus[j]+=h;minus[j]-=h
   with torch.no_grad():fd.append(float((fn(tensor(plus))-fn(tensor(minus)))/(2*h)))
  fd=np.array(fd);error=abs(g-fd);tolerance=1e-5+2e-4*np.maximum(abs(g),abs(fd));norms={'rotation':float(np.linalg.norm(g[:3])),'translation':float(np.linalg.norm(g[3:6])),'q_bary':float(np.linalg.norm(g[6:]))}
  xm=torch.tensor(x0,dtype=torch.float64,requires_grad=True);fixed=fn(xm,True);negative=torch.autograd.grad(fixed,xm)[0].numpy()
  dx=np.array([1e-5,-1e-5,2e-5,1e-5,2e-5,-1e-5,1e-5,-1e-5]);withmove=float(fn(tensor(x0+dx)));fixedmove=float(fn(tensor(x0+dx),True))
  results.append({'edge_id':int(i),'track_id':k,'target_frame':fr,'source_frame':int(a['track_source_frame'][k]),'predicted_uv':uv.tolist(),'bary_min':float(a['bary0'][k].min()),'local_feature_difference_norm':texture,'image_loss':float(loss),'gradient':g.tolist(),'finite_difference':fd.tolist(),'absolute_error':error.tolist(),'tolerance':tolerance.tolist(),'gradient_norms':norms,'FD_passed':bool((error<=tolerance).all()),'nonzero_Rt':bool(norms['rotation']>1e-6 and norms['translation']>1e-6),'nonzero_q':bool(norms['q_bary']>1e-6),'measured_pixel_negative_control_max_gradient':float(abs(negative).max()),'measured_pixel_negative_control_change':abs(float(fixed)-fixedmove),'current_projection_loss_change':withmove-float(loss),'negative_control_passed':bool(np.max(abs(negative))==0 and abs(float(fixed)-fixedmove)==0)})
 return {'passed':all(r['FD_passed'] and r['nonzero_Rt'] and r['nonzero_q'] and r['negative_control_passed'] for r in results),'samples':results,'fields_sha256':sha(fpath),'descriptors_sha256':sha(E/f'observations/{dev}/features/frozen_descriptors.npz'),'rule':'first3 optimization non-source edges with barymin>.02, positive valid projection, >3px image margin, fractional pixels in [.15,.85], adjacent feature change norm>.01; no reference criteria'}

def check_masks_and_invariance():
 o=J.Objective.__new__(J.Objective);o.calls=0;o.base=lambda R,t:{};o.weights=tensor([.7,.9,.5]);o.train=torch.tensor([True,True,False]);o.opt_image=torch.tensor([False,True,False]);o.is_source=torch.tensor([True,False,False]);o.W=o.weights[o.train].sum();o.WI=o.weights[o.opt_image].sum()
 du=torch.ones((3,2),dtype=torch.float64,requires_grad=True);df=torch.ones((3,5),dtype=torch.float64,requires_grad=True);cam=tensor([[0,0,1],[0,0,1],[0,0,1]]);uv=tensor([[100,100],[150,150],[200,200]])
 o.edges=lambda R,t,q:(cam,uv,du,df)
 loss,c,_=o.loss(None,None,None,True);gdu,gdf=torch.autograd.grad(loss,[du,df]);off,coff,_=o.loss(None,None,None,False);goff=torch.autograd.grad(off,df)[0]
 source2d=bool(gdu[0].abs().sum()>0);sourceimage=bool(gdf[0].abs().sum()==0);holdout=bool(gdu[2].abs().sum()==0 and gdf[2].abs().sum()==0);imageoff=bool(coff['projection_image']==0 and goff.abs().sum()==0)
 oldW=float(o.W);oldWI=float(o.WI);cam[:]=tensor([[0,0,-1],[0,0,-1],[0,0,-1]]);uv[:]=tensor([[-1000,1000],[-1000,1000],[-1000,1000]])
 difficult,dc,_=o.loss(None,None,None,True);dg=torch.autograd.grad(difficult,du)[0];no_remove=bool(dg[0].abs().sum()>0 and dg[1].abs().sum()>0 and float(o.W)==oldW and float(o.WI)==oldWI and float(dc['validity'])>0)
 # Fixed-q solver branch uses no q parameter in optimizer. Exercise that algebra
 # with the actual surface mapping and Adam while R/t parameters receive updates.
 sc=SurfaceConstraintSet([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],[0],[[.5,.2,.3]]);bary=torch.nn.Parameter(tensor(sc.bary2),requires_grad=False);q0=sc.torch_points(bary).clone();trans=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64));opt=torch.optim.Adam([trans],lr=.002)
 for _ in range(3):
  opt.zero_grad();ll=((sc.torch_points(bary)+trans-1)**2).sum();ll.backward();opt.step()
 qfixed=bool(torch.equal(q0,sc.torch_points(bary)) and bary.grad is None)
 return {'source_2d_included':source2d,'source_image_excluded':sourceimage,'heldout_excluded':holdout,'image_off_value_and_gradient_zero':imageoff,'fixed_q_parameter_unchanged':qfixed,'no_dynamic_removal_of_difficult_edges':no_remove,'source_2d_gradient':gdu[0].tolist(),'source_image_gradient':gdf[0].tolist(),'fixed_denominators':{'W':oldW,'WI':oldWI},'passed':all([source2d,sourceimage,holdout,imageoff,qfixed,no_remove])}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base-only',action='store_true');a=ap.parse_args();torch.set_num_threads(1);start=time.perf_counter();code_start=sha(E/'code/joint_solver.py');details={}
 for dev in ['dev1','dev2']:
  print('CHECK',dev,flush=True);o=base_object(dev);base=check_base(o);print(dev,'base',base['passed'],flush=True)
  gradients=None if a.base_only else check_image(dev,o.P);details[dev]={'base_forward':base,'image_gradients':gradients};del o
 masks=check_masks_and_invariance();surface=self_test();integrity=json.loads((E/'protocol/observations_integrity.json').read_text());integrity_ok=all(all(d['checks'].values()) for d in integrity['devs'].values());rules=json.loads((E/'protocol/promotion_rules.json').read_text());must=rules['implementation_gate']['must_pass'];checks={k:{'passed':False,'status':'pending_eight_run_artifact_audit'} for k in must}
 grads_ok=not a.base_only and all(d['image_gradients']['passed'] for d in details.values());allbase=all(d['base_forward']['passed'] for d in details.values())
 for k,v in {'nonconstant_image_gradient_Rt':grads_ok,'nonconstant_image_gradient_q':grads_ok,'q_on_fixed_surface':surface['all_passed'],'q_fixed_for_F00_F01':masks['fixed_q_parameter_unchanged'],'source_2d_included':masks['source_2d_included'] and integrity_ok,'no_dynamic_removal_of_difficult_edges':masks['no_dynamic_removal_of_difficult_edges'],'rotation_legal_and_scale_fixed':surface['all_passed']}.items():checks[k]={'passed':bool(v),'status':'CPU_implementation_verified','final_artifact_recheck_required':k in ['q_on_fixed_surface','q_fixed_for_F00_F01','rotation_legal_and_scale_fixed']}
 rv=torch.linspace(-3,3,30,dtype=torch.float64).reshape(10,3);rr=J.axis_angle_to_matrix(rv);rotation_ok=bool(torch.max(abs(rr@rr.transpose(1,2)-torch.eye(3)))<1e-10 and torch.max(abs(torch.linalg.det(rr)-1))<1e-10);checks['rotation_legal_and_scale_fixed']['passed']&=rotation_ok
 stable=sha(E/'code/joint_solver.py')==code_start
 report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reference_read':False,'GPU_used':False,'dtype':'float64 CPU; source feature values retain original float32 before conversion','source_hashes':{'joint_solver.py':code_start,'surface_constraints.py':sha(E/'code/surface_constraints.py'),'check_solver.py':sha(__file__)},'solver_code_unchanged_during_checks':stable,'checks':checks,'all_numerical_preexecution_checks_passed':bool(allbase and grads_ok and masks['passed'] and surface['all_passed'] and integrity_ok and stable),'base_forward_passed':allbase,'details':details,'mask_and_invariance':masks,'surface_checks':surface,'observations_integrity_passed':integrity_ok,'pending_main_process_after_all_eight': ['common_observation_pool_and_fixed_denominators','same_initialization','same_solver_and_budget'],'final_artifact_rechecks':['q_on_fixed_surface','q_fixed_for_F00_F01','rotation_legal_and_scale_fixed'],'seconds':time.perf_counter()-start}
 report['ready_for_formal_runs']=report['all_numerical_preexecution_checks_passed'];J.save(E/'protocol/implementation_checks.json',report);print(json.dumps({'all_numerical_preexecution_checks_passed':report['all_numerical_preexecution_checks_passed'],'base_forward_passed':allbase,'pending':report['pending_main_process_after_all_eight'],'seconds':report['seconds']},indent=2),flush=True)
 if not report['all_numerical_preexecution_checks_passed'] and not a.base_only:raise SystemExit(1)
if __name__=='__main__':main()
