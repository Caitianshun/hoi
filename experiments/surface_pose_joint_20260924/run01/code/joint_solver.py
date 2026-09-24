"""Input-only shared surface attachment / projection-image 2x2 solver.

Historical silhouette/depth/time terms are ported piecewise differentiably;
new observations replace the historical conditional track term. No evaluation
reference imports or paths are used by this module.
"""
from pathlib import Path
import argparse,json,hashlib,datetime,sys,time,os,math,traceback
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle
ROOT=Path('/home/cai_tianshun/Project/HOI')
E=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/object_pose_refinement_20260924/run01'
sys.path.insert(0,str(OLD/'code'))
from solve_pose import Problem,plain

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(plain(a),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rho(x):return 2*(torch.sqrt(1+x*x)-1)
def project(x,K):
 h=x@K.T
 return h[...,:2]/h[...,2:].clamp_min(.01)

def sample(fields,frame,uv):
 """Bilinear NHWC frozen field sampling, gradient only through coordinates."""
 H,W=fields.shape[1:3];x=uv[...,0].clamp(0,W-1);y=uv[...,1].clamp(0,H-1)
 x0=x.floor().long();y0=y.floor().long();x1=(x0+1).clamp(max=W-1);y1=(y0+1).clamp(max=H-1)
 wx=(x-x0).unsqueeze(-1);wy=(y-y0).unsqueeze(-1)
 return (fields[frame,y0,x0]*(1-wx)+fields[frame,y0,x1]*wx)*(1-wy)+(fields[frame,y1,x0]*(1-wx)+fields[frame,y1,x1]*wx)*wy

def feature_field(rgb):
 """Five frozen channels; no learned features or per-pose normalization."""
 x=rgb.astype(np.float32)/255.;x=cv2.GaussianBlur(x,(0,0),.8)
 mu=cv2.GaussianBlur(x,(0,0),3);var=np.maximum(cv2.GaussianBlur(x*x,(0,0),3)-mu*mu,0)
 color=(x-mu)/np.sqrt(np.maximum(var,.05**2))
 gray=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
 gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8.;gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8.
 return np.concatenate([color,gx[...,None],gy[...,None]],axis=-1).astype(np.float32)

def build_features(dev):
 P=Problem(dev);O=E/f'observations/{dev}';obs=np.load(O/'observations.npz')
 out=O/'features';out.mkdir(exist_ok=False)
 field=np.lib.format.open_memmap(out/'fields.npy',mode='w+',dtype=np.float32,shape=(P.L,480,640,5))
 hashes={};st=time.perf_counter()
 for i,path in enumerate(P.m['frame_paths']):
  rgb=cv2.imread(path)[...,::-1];field[i]=feature_field(rgb);hashes[path]=sha(path)
 field.flush();del field
 a=np.load(out/'fields.npy',mmap_mode='r');fields=torch.from_numpy(np.asarray(a).copy())
 sf=torch.tensor(obs['track_source_frame'],dtype=torch.long);su=torch.tensor(obs['track_source_uv'],dtype=torch.float32)
 desc=sample(fields,sf,su).numpy();et=obs['edge_track'];ef=obs['edge_frame'];uv=obs['edge_uv'];train=(obs['edge_split']==0)&(~obs['edge_is_source'].astype(bool))
 measured=sample(fields,torch.tensor(ef[train]),torch.tensor(uv[train],dtype=torch.float32)).numpy()
 diff=measured-desc[et[train]]
 sigma=np.maximum(.15,1.4826*np.median(abs(diff-np.median(diff,axis=0)),axis=0)).astype(np.float32)
 np.savez_compressed(out/'frozen_descriptors.npz',source_descriptor=desc,sigma_F=sigma)
 save(out/'manifest.json',{'status':'frozen','utc':utc(),'no_reference_used':True,'source':'camera0 RGB only; no dense cached learned feature maps in original tracklets, which store only discrete NCC/RGB observations','channels':['locally_normalized_R','locally_normalized_G','locally_normalized_B','gray_Sobel_x_div8','gray_Sobel_y_div8'],'preblur_sigma_px':.8,'normalization_gaussian_sigma_px':3.,'color_std_floor':.05,'sigma_F_rule':'max(0.15,1.4826*MAD(measured target minus other fixed source descriptor)) per channel from optimization non-source edges only; same rule both devs','sigma_F':sigma,'descriptor':'one fixed source at measured pixel; no heldout edge used','input_image_hashes':hashes,'fields_sha256':sha(out/'fields.npy'),'descriptors_sha256':sha(out/'frozen_descriptors.npz'),'observations_sha256':sha(O/'observations.npz'),'code_sha256':sha(__file__),'wall_seconds':time.perf_counter()-st})
 print(dev,'features frozen',sigma,flush=True)

class Objective:
 def __init__(self,dev,device='cuda',dtype=torch.float32):
  self.P=P=Problem(dev);self.dev=dev;self.device=device;self.dtype=dtype
  self.obs=dict(np.load(E/f'observations/{dev}/observations.npz'));self.a=self.obs
  self.tensor=lambda x,dt=None:torch.as_tensor(np.asarray(x),dtype=dt or dtype,device=device)
  self.K=self.tensor(P.K);self.V=self.tensor(P.V);self.S=self.tensor(P.S);self.ts=self.tensor(P.ts);self.vis=self.tensor(P.visible,torch.bool)
  self.uv=self.tensor(np.stack([x[0] for x in P.obs]));self.fields=self.tensor(np.stack([x[1] for x in P.obs])[...,None]);self.xyz=self.tensor(np.stack([x[2] for x in P.obs]));self.good=self.tensor(np.stack([x[3] for x in P.obs]),torch.bool)
  self.image=self.tensor(np.load(E/f'observations/{dev}/features/fields.npy',mmap_mode='c'))
  d=np.load(E/f'observations/{dev}/features/frozen_descriptors.npz');self.desc=self.tensor(d['source_descriptor']);self.sigmaF=self.tensor(d['sigma_F'])
  for k in ['edge_track','edge_frame','track_source_frame']:setattr(self,k,self.tensor(self.a[k],torch.long))
  self.edge_uv=self.tensor(self.a['edge_uv']);self.weights=self.tensor(self.a['edge_weight']);self.is_source=self.tensor(self.a['edge_is_source'],torch.bool);self.train=self.tensor(self.a['edge_split']==0,torch.bool);self.heldout=~self.train
  self.opt_image=self.train&~self.is_source;self.hold_image=self.heldout&~self.is_source
  assert bool(self.train[self.is_source].all()) and self.opt_image.any()
  self.W=self.weights[self.train].sum();self.WI=self.weights[self.opt_image].sum();self.calls=0

 def silhouette_box(self,Vuv):
  B=len(Vuv);coords=Vuv.detach().cpu().numpy().astype(np.float32);hulls=[cv2.convexHull(v,returnPoints=False)[:,0] for v in coords];M=max(map(len,hulls))
  ids=np.array([np.pad(h,(0,M-len(h)),constant_values=h[-1]) for h in hulls]);endids=np.array([np.pad(np.roll(h,-1),(0,M-len(h)),constant_values=h[-1]) for h in hulls])
  batch=torch.arange(B,device=self.device)[:,None];a=Vuv[batch,self.tensor(ids,torch.long)];b=Vuv[batch,self.tensor(endids,torch.long)];e=b-a;length=torch.linalg.vector_norm(e,dim=-1);good=length>1e-7
  delta=self.uv[:,:,None,:]-a[:,None,:,:];alpha=(delta*e[:,None]).sum(-1)/(length[:,None]**2).clamp_min(1e-12);near=a[:,None]+alpha.clamp(0,1)[...,None]*e[:,None]
  dist=torch.linalg.vector_norm(self.uv[:,:,None]-near,dim=-1);dist=dist.masked_fill(~good[:,None],float('inf'));mindist=dist.min(-1).values
  cross=e[:,None,:,0]*delta[...,1]-e[:,None,:,1]*delta[...,0];inside=((cross>=-1e-7)|~good[:,None]).all(-1)|((cross<=1e-7)|~good[:,None]).all(-1)
  fwd=torch.where(inside,torch.zeros_like(mindist),mindist)
  cum=torch.cat([torch.zeros((B,1),device=self.device,dtype=self.dtype),length.cumsum(-1)],-1);pos=cum[:,-1:]*torch.arange(64,device=self.device,dtype=self.dtype)[None]/64
  idx=(torch.searchsorted(cum.contiguous(),pos.contiguous(),right=True)-1).clamp(0,M-1)
  start=cum.gather(1,idx);seglen=length.gather(1,idx);aa=a.gather(1,idx[:,:,None].expand(-1,-1,2));ee=e.gather(1,idx[:,:,None].expand(-1,-1,2));boundary=aa+((pos-start)/seglen.clamp_min(1e-8))[...,None]*ee
  return fwd,self.background(boundary)

 def background(self,uv):
  ids=torch.arange(len(uv),device=self.device)[:,None].expand(uv.shape[:-1]);v=sample(self.fields,ids,uv)[...,0]
  outside=(uv[...,0]<0)|(uv[...,0]>639)|(uv[...,1]<0)|(uv[...,1]>479)
  return torch.where(outside,torch.full_like(v,100),v)

 def base(self,R,t):
  P=self.P;cam=torch.einsum('tij,nj->tni',R,self.S)+t[:,None];puv=project(cam,self.K)
  if self.dev=='dev1':
   vcam=torch.einsum('tij,nj->tni',R,self.V)+t[:,None];f,b=self.silhouette_box(project(vcam,self.K))
  else:
   with torch.no_grad():indices=torch.cdist(self.uv,puv,compute_mode='donot_use_mm_for_euclid_dist').argmin(-1)
   nearest=puv.gather(1,indices[...,None].expand(-1,-1,2));f=torch.linalg.vector_norm(nearest-self.uv,dim=-1)
   back=self.background(puv);b=torch.quantile(back,torch.linspace(0,1,64,device=self.device,dtype=self.dtype),dim=1).T
  with torch.no_grad():indices=torch.cdist(self.xyz,cam,compute_mode='donot_use_mm_for_euclid_dist').argmin(-1)
  nearest=cam.gather(1,indices[...,None].expand(-1,-1,3));dep=torch.linalg.vector_norm(self.xyz-nearest,dim=-1)/(.1*torch.sqrt(self.good.sum(-1).clamp_min(1))[:,None])*.15
  dep=dep*self.good;visible=self.vis[:,None]
  dt=torch.diff(self.ts);v=torch.diff(t,dim=0)/dt[:,None];w=matrix_to_axis_angle(R[1:]@R[:-1].transpose(1,2))/dt[:,None];den=(dt[:-1]+dt[1:])/2
  temp=torch.cat([torch.diff(v,dim=0)/den[:,None]/10,torch.diff(w,dim=0)/den[:,None]/80],-1)*.1
  return {'silhouette_forward':rho(f/16*visible).sum()/P.L,'silhouette_background':rho(b/16*visible).sum()/P.L,'estimated_depth_weak':rho(dep*visible).sum()/P.L,'actual_time_acceleration':rho(temp).sum()/P.L}

 def edges(self,R,t,q):
  cam=torch.einsum('eij,ej->ei',R[self.edge_frame],q[self.edge_track])+t[self.edge_frame];uv=project(cam,self.K)
  diff=(uv-self.edge_uv)/3.;feature=sample(self.image,self.edge_frame,uv);fdiff=(feature-self.desc[self.edge_track])/self.sigmaF
  return cam,uv,diff,fdiff

 def loss(self,R,t,q,image_on=True):
  self.calls+=1;comp=self.base(R,t);cam,uv,diff,fdiff=self.edges(R,t,q);W=self.weights
  comp['candidate_2d']=(rho(diff).mean(-1)[self.train]*W[self.train]).sum()/self.W
  val=torch.stack([(-uv[:,0]).clamp_min(0)/3,(uv[:,0]-639).clamp_min(0)/3,(-uv[:,1]).clamp_min(0)/3,(uv[:,1]-479).clamp_min(0)/3,(.05-cam[:,2]).clamp_min(0)/.01],-1)
  comp['validity']=(rho(val).sum(-1)[self.train]*W[self.train]).sum()/self.W
  img=(rho(fdiff).mean(-1)[self.opt_image]*W[self.opt_image]).sum()/self.WI
  comp['projection_image']=img if image_on else img*0
  return sum(comp.values()),comp,{'raw_image_diagnostic':img,'edge_cam':cam,'edge_uv':uv,'diff2d':diff,'diffimage':fdiff}

 def input_metrics(self,R,t,q):
  """Fixed edge denominators, no input-visibility removal by prediction."""
  with torch.no_grad():cam,uv,du,df=self.edges(R,t,q)
  ca=cam.cpu().numpy();xy=uv.cpu().numpy();u=self.a['edge_uv'];frames=self.a['edge_frame'];inside=(xy[:,0]>=0)&(xy[:,0]<=639)&(xy[:,1]>=0)&(xy[:,1]<=479);positive=ca[:,2]>.05;ix=np.rint(np.clip(xy[:,0],0,639)).astype(int);iy=np.rint(np.clip(xy[:,1],0,479)).astype(int);mask=(self.P.labels[frames,iy,ix]==2)&inside
  # Front-most geometric ray intersection is DIAGNOSTIC ONLY, never a loss gate.
  from object_init_general import ray_mesh_canonical
  rn=R.detach().cpu().numpy();tn=t.detach().cpu().numpy();qn=q.detach().cpu().numpy();visible=np.zeros(len(xy),bool)
  for f in np.unique(frames):
   ids=np.flatnonzero((frames==f)&inside&positive)
   if not len(ids):continue
   front,hit=ray_mesh_canonical(xy[ids],rn[f],tn[f],self.P.V,self.P.F,self.P.K);visible[ids]=hit&(np.linalg.norm(front-qn[self.a['edge_track'][ids]],axis=1)<.002)
  coverage={};res={}
  for name,subset in [('opt',self.a['edge_split']==0),('heldout',self.a['edge_split']==1)]:
   z=subset;ei=z&~self.a['edge_is_source'].astype(bool);err=np.linalg.norm(xy[z]-u[z],axis=1);fd=df.cpu().numpy();dd=du.cpu().numpy();imageerr=np.mean(abs(fd[ei]),axis=1)
   coverage[name]={'fixed_edges':int(z.sum()),'positive_depth_fraction':float(positive[z].mean()),'in_bounds_fraction':float(inside[z].mean()),'mask_consistent_fraction':float(mask[z].mean()),'template_visible_fraction':float(visible[z].mean()),'effective_fraction':float((positive&inside&mask&visible)[z].mean())}
   res[name]={'reprojection_px_mean':float(err.mean()),'reprojection_px_median':float(np.median(err)),'reprojection_px_p90':float(np.percentile(err,90)),'reprojection_px_max':float(err.max()),'image_normalized_abs_mean':float(imageerr.mean()),'image_target_edges':int(ei.sum()),'robust_2d_edge_downweight_lt_0p1_fraction':float((1/np.sqrt(1+dd[z]**2)<.1).any(axis=1).mean()),'robust_image_edge_downweight_lt_0p1_fraction':float((1/np.sqrt(1+fd[ei]**2)<.1).any(axis=1).mean()),'robust_2d_component_downweight_lt_0p1_fraction':float((1/np.sqrt(1+dd[z]**2)<.1).mean()),'robust_image_component_downweight_lt_0p1_fraction':float((1/np.sqrt(1+fd[ei]**2)<.1).mean())}
  per_frame=[]
  for f in range(self.P.L):
   row={'frame':f,'timestamp_seconds':float(self.P.ts[f])}
   for split,code in [('opt',0),('heldout',1)]:
    z=(frames==f)&(self.a['edge_split']==code);count=int(z.sum());row[split]={'fixed_edges':count,**{key:(float(values[z].mean()) if count else None) for key,values in [('positive_depth_fraction',positive),('in_bounds_fraction',inside),('mask_consistent_fraction',mask),('template_visible_fraction',visible),('effective_fraction',positive&inside&mask&visible)]}}
   per_frame.append(row)
  return {'coverage':coverage,'residuals':res,'per_frame_coverage':per_frame,'projection':xy,'edge_positive':positive,'edge_in_bounds':inside,'edge_mask_consistent':mask,'edge_template_visible':visible}

def gpu_check():
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 assert '3090' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0)
 return {'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'visible_device':torch.cuda.get_device_name(0)}

def solve(dev,variant,output,steps=300,preflight=False):
 from surface_constraints import SurfaceConstraintSet
 total_st=time.perf_counter();gpu=gpu_check();torch.manual_seed(12345);np.random.seed(12345);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 output=Path(output);output.mkdir(parents=True,exist_ok=False);Pobj=Objective(dev);P=Pobj.P;o=Pobj.obs
 surface=SurfaceConstraintSet(P.V,P.F,o['face0'],o['bary0'],radius_m=.05)
 fixed=variant[1]=='0';image_on=variant[2]=='1';R0=Rotation.from_matrix(P.R0).as_rotvec()
 rv=torch.nn.Parameter(Pobj.tensor(R0));trans=torch.nn.Parameter(Pobj.tensor(P.t0));bary=torch.nn.Parameter(Pobj.tensor(o['bary0'][:,1:]),requires_grad=not fixed)
 groups=[{'params':[rv],'lr':.002},{'params':[trans],'lr':.002}]
 if not fixed:groups.append({'params':[bary],'lr':.005})
 opt=torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8);st=time.perf_counter();history=[];switches=0;boundary=0
 cfg={'status':'running','utc':utc(),'dev':dev,'variant':variant,'preflight':preflight,'no_reference_used':True,'steps_budget':steps,'wall_clock_cap_seconds':600,'optimizer':'Adam full batch with fixed cosine LR schedule, end point (no best-checkpoint search)','learning_rates':{'rotation_vector':.002,'translation_m':.002,'barycentric_uv':.005},'fixed_q':fixed,'image_on':image_on,'gpu':gpu,'initialization':str(P.initpath),'initialization_sha256':sha(P.initpath),'observations_sha256':sha(E/f'observations/{dev}/observations.npz'),'source_sha256':sha(__file__),'surface_module_sha256':sha(E/'code/surface_constraints.py')}
 save(output/'run.json',cfg)
 q0=surface.torch_points(bary).detach();r0=axis_angle_to_matrix(rv).detach();initmet=Pobj.input_metrics(r0,trans.detach(),q0);st=time.perf_counter()
 for iteration in range(steps):
  if time.perf_counter()-st>600:break
  opt.zero_grad(set_to_none=True);R=axis_angle_to_matrix(rv);q=surface.torch_points(bary);loss,comp,extra=Pobj.loss(R,trans,q,image_on)
  if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
  loss.backward()
  if not all(torch.isfinite(p.grad).all() for group in groups for p in group['params']):raise RuntimeError('nonfinite gradients')
  factor=.1+.9*.5*(1+math.cos(math.pi*iteration/max(1,steps-1)))
  for g,lr in zip(opt.param_groups,[.002,.002]+([] if fixed else [.005])):g['lr']=lr*factor
  oldrv=rv.detach().clone();oldt=trans.detach().clone();oldq=q.detach().clone();opt.step()
  with torch.no_grad():trans[:,2].clamp_(.4,8)
  stepstate=None
  if not fixed:
   stepstate=surface.project_step(bary.detach().cpu().numpy())
   with torch.no_grad():bary.copy_(Pobj.tensor(stepstate['projected_bary2']))
   changed=np.asarray(stepstate['face_changed'],bool);switches+=int(changed.sum());boundary+=int(np.asarray(stepstate['boundary_hit']).sum())
   if changed.any():
    for k in ['exp_avg','exp_avg_sq']:opt.state[bary][k][Pobj.tensor(changed,torch.bool)]=0
  row={'iteration':iteration+1,'objective_before_update':float(loss.detach()),'components_before_update':{k:float(v.detach()) for k,v in comp.items()},'raw_image_diagnostic':float(extra['raw_image_diagnostic'].detach()),'lr_factor':factor,'translation_step_max_m':float(torch.linalg.vector_norm(trans.detach()-oldt,dim=-1).max()),'rotation_vector_step_max_rad':float(torch.linalg.vector_norm(rv.detach()-oldrv,dim=-1).max()),'q_step_max_m':float(torch.linalg.vector_norm(surface.torch_points(bary).detach()-oldq,dim=-1).max()),'elapsed_seconds':time.perf_counter()-st}
  history.append(row)
  if (iteration+1)%25==0:print(dev,variant,iteration+1,round(row['objective_before_update'],5),round(row['elapsed_seconds'],2),flush=True)
 R=axis_angle_to_matrix(rv).detach();t=trans.detach();q=surface.torch_points(bary).detach();loss,comp,extra=Pobj.loss(R,t,q,image_on);torch.cuda.synchronize();solve_seconds=time.perf_counter()-st
 finalmet=Pobj.input_metrics(R,t,q);rn=R.cpu().numpy();tn=t.cpu().numpy();qn=q.cpu().numpy();a=dict(P.a);a['R_camera']=rn;a['t_camera_m']=tn;a['R_world']=(P.a['c2w'][:3,:3]@rn).astype(np.float32);a['t_world_m']=(tn@P.a['c2w'][:3,:3].T+P.a['c2w'][:3,3]).astype(np.float32)
 iou=P.ious(rn,tn);a['reliability']=np.where(P.visible,np.minimum(1,(P.labels==2).sum((1,2))/1000)*iou,0).astype(np.float32)
 np.savez_compressed(output/'object_init.npz',**a)
 state=surface.state_np();np.savez_compressed(output/'surface_points.npz',q0=o['q0'],q_final=qn,**state)
 np.savez_compressed(output/'projection_diagnostics.npz',edge_track=o['edge_track'],edge_frame=o['edge_frame'],edge_uv=o['edge_uv'],edge_split=o['edge_split'],initial_projection=initmet.pop('projection'),final_projection=finalmet.pop('projection'),**{f'initial_{k}':initmet.pop(k) for k in list(initmet) if k.startswith('edge_')},**{f'final_{k}':finalmet.pop(k) for k in list(finalmet) if k.startswith('edge_')})
 save(output/'iterations.json',history)
 result={**cfg,'status':'completed','completed_utc':utc(),'actual_updates':len(history),'objective_calls':Pobj.calls,'jacobian_mode':'reverse autograd, one backward per update; no numerical Jacobian','termination':'fixed_update_budget' if len(history)==steps else 'wall_clock_budget','solve_wall_seconds':solve_seconds,'total_wall_seconds':time.perf_counter()-total_st,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'final_objective':float(loss),'final_components':{k:float(v) for k,v in comp.items()},'initial':initmet,'final':finalmet,'mean_visible_iou':float(iou[P.visible].mean()),'fixed_visible_frames':int(P.visible.sum()),'q_movement_mean_m':float(np.linalg.norm(qn-o['q0'],axis=-1).mean()),'q_movement_max_m':float(np.linalg.norm(qn-o['q0'],axis=-1).max()),'total_face_switches':switches,'boundary_update_point_count':boundary,'fixed_q_max_difference_m':float(np.max(abs(qn-o['q0']))),'pose_sha256':sha(output/'object_init.npz'),'surface_sha256':sha(output/'surface_points.npz'),'surface_constraint_summary':surface.metadata()}
 save(output/'metrics.json',result);save(output/'run.json',result);print('DONE',dev,variant,round(solve_seconds,2),flush=True);return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['features','solve']);ap.add_argument('--dev',choices=['dev1','dev2'],required=True);ap.add_argument('--variant',choices=['F00','F01','F10','F11']);ap.add_argument('--output',type=Path);ap.add_argument('--steps',type=int,default=300);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
 if args.mode=='features':build_features(args.dev)
 else:
  try:solve(args.dev,args.variant,args.output,args.steps,args.preflight)
  except Exception as exc:
   if args.output:save(args.output/'failure.json',{'status':'failed','utc':utc(),'error':str(exc),'traceback':traceback.format_exc()})
   raise
