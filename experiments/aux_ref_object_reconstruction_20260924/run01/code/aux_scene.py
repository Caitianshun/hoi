"""Isolated AUX_REF_OBJECT scene adapter; no published/reference-data reader.

Only the caller may supply explicit object R_world/t_world. Old S1 supplies
frozen H/S and RGB-predicted motion. The object bank is NEVER loaded from S1.
Interpolated motion here is RGB prediction evaluation, never reference filling.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path('/home/cai_tianshun/Project/HOI')
OLD=ROOT/'experiments/structured_hoi_20260923'
sys.path.insert(0,str(OLD/'code'))
from gaussian_scene import StructuredScene,GaussianBank
from human_lbs import batch_rodrigues,batch_rigid_transform
from pytorch3d.transforms import quaternion_to_matrix
from lib_render.render_helper import render as joint_render


def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while block:=stream.read(1<<20):h.update(block)
 return h.hexdigest()


def tensor_identity(value):
 a=value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.ascontiguousarray(value)
 h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()


def state_identity(state):
 identities={key:tensor_identity(value) for key,value in sorted(state.items())}
 aggregate=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()
 return {'sha256':aggregate,'arrays':identities}


def _linear(values,left,right,alpha):
 values=np.asarray(values);shape=(len(alpha),)+(1,)*(values.ndim-1)
 return values[left]*(1-alpha.reshape(shape))+values[right]*alpha.reshape(shape)


def _slerp(times,rotations,queries):
 # Rotations are matrices. SciPy rejects queries outside its closed source span.
 return Slerp(times,Rotation.from_matrix(rotations))(queries).as_matrix()


class AuxObjectScene(nn.Module):
 """Frozen S1 scene plus a fresh object bank, sampled at fixed query times.

 Construction is valid on CPU. Rendering requires the existing CUDA renderer.
 Instantiate directly on the intended device; caches are built on that device.
 motion_records/interpolation_records document each prediction interpolation.
 """
 def __init__(self,dev,query_times,device='cpu',*,init_path=None,checkpoint_path=None,cache_hs=True):
  super().__init__()
  if dev not in ('dev1','dev2'):raise ValueError('Only the two frozen development sequences are supported')
  self.dev=dev;self.device_name=str(device);self.init_path=Path(init_path or OLD/f'{dev}_initialization/initialization.pt').resolve();self.checkpoint_path=Path(checkpoint_path or OLD/f'{dev}_S1_v1/checkpoint_008000.pt').resolve()
  init=torch.load(self.init_path,map_location='cpu',weights_only=False)
  if init.get('reference_used') is not False:raise ValueError('Old initialization must have reference_used=False')
  checkpoint=torch.load(self.checkpoint_path,map_location='cpu',weights_only=False)
  if checkpoint['step']!=8000:raise ValueError('AUX requires the original final8000 S1 checkpoint')
  if checkpoint['initialization_sha256']!=sha_file(self.init_path):raise ValueError('Checkpoint/init identity mismatch')
  state=checkpoint['model'];self.scene=StructuredScene(init)
  if not torch.equal(state['object_anchors'].cpu(),self.scene.object_anchors):raise ValueError('Old S1 object anchors differ from the shared untrained initialization')
  # Resize only frozen banks. obank stays the brand-new constructor bank;
  # no trained obank attribute or topology ever enters load_state_dict.
  self.scene.sbank.resize_for_load(state,'sbank.');self.scene.hbank.resize_for_load(state,'hbank.')
  filtered={k:v for k,v in state.items() if not k.startswith('obank.') and k!='object_anchors'}
  incompatible=self.scene.load_state_dict(filtered,strict=False)
  allowed={k for k in self.scene.state_dict() if k.startswith('obank.')}|{'object_anchors'}
  if set(incompatible.missing_keys)!=allowed or incompatible.unexpected_keys:raise ValueError(('Unexpected partial-load fields',incompatible))
  for p in self.scene.parameters():p.requires_grad_(False)
  for p in self.scene.obank.parameters():p.requires_grad_(True)
  self.initialization=init
  self.source_times=np.asarray(init['timestamps'],dtype=np.float64)
  query=np.asarray(query_times,dtype=np.float64)
  if query.ndim!=1 or not len(query) or not np.isfinite(query).all():raise ValueError('query_times must be a finite nonempty1D array')
  if not (np.diff(self.source_times)>0).all():raise ValueError('Original source times must increase strictly')
  if query.min()<self.source_times[0] or query.max()>self.source_times[-1]:raise ValueError('Prediction extrapolation is forbidden')
  right=np.searchsorted(self.source_times,query,side='left');right=np.minimum(right,len(self.source_times)-1);exact=self.source_times[right]==query;left=np.where(exact,right,right-1)
  if (left<0).any():raise ValueError('Prediction extrapolation is forbidden')
  interval=self.source_times[right]-self.source_times[left];alpha=np.divide(query-self.source_times[left],interval,out=np.zeros_like(query),where=interval>0)
  self.query_times=query.copy();self.motion_records=[{'query_index':i,'query_timestamp_seconds':float(q),'left_frame':int(l),'right_frame':int(r),'left_timestamp_seconds':float(self.source_times[l]),'right_timestamp_seconds':float(self.source_times[r]),'interval_seconds':float(dt),'right_weight':float(a),'interpolated_prediction':bool(l!=r),'extrapolated':False,'source':'old_S1_RGB_prediction'} for i,(q,l,r,dt,a) in enumerate(zip(query,left,right,interval,alpha))]
  self.interpolation_records=self.motion_records
  with torch.no_grad():
   predR=self.scene.object_R(torch.arange(len(self.source_times))).cpu().numpy().astype(np.float64)
   predt=self.scene.object_translation.cpu().numpy().astype(np.float64)
   human=self.scene.human;mean=human.pose_mean.cpu().numpy().astype(np.float64)
   # Slerp actual joint rotations (including the fixed pose mean), then subtract
   # that mean only to keep the original HumanLBS pose-construction convention.
   angles=np.concatenate([human.global_orient.cpu().numpy(),human.body_pose.cpu().numpy()],1).reshape(len(self.source_times),22,3).astype(np.float64)
   angles=angles+mean[:66].reshape(1,22,3)
   qangles=np.empty((len(query),22,3),np.float64)
   for j in range(22):qangles[:,j]=Rotation.from_matrix(_slerp(self.source_times,Rotation.from_rotvec(angles[:,j]).as_matrix(),query)).as_rotvec()
   qangles-=mean[:66].reshape(1,22,3)
   htranslation=_linear(human.transl.cpu().numpy(),left,right,alpha)
   # Interpolate the actual bounded residual vectors, retaining the1.5cm bound
   # by convex interpolation. Static canonical residual/shape remain untouched.
   regional=_linear(human.regional_offsets().cpu().numpy(),left,right,alpha)
  self.scene=self.scene.to(device)
  self.register_buffer('pred_R_world',torch.as_tensor(_slerp(self.source_times,predR,query),dtype=torch.float32,device=device))
  self.register_buffer('pred_t_world',torch.as_tensor(_linear(predt,left,right,alpha),dtype=torch.float32,device=device))
  self.register_buffer('_query_human_global',torch.as_tensor(qangles[:,0],dtype=torch.float32,device=device))
  self.register_buffer('_query_human_body',torch.as_tensor(qangles[:,1:].reshape(len(query),63),dtype=torch.float32,device=device))
  self.register_buffer('_query_human_translation',torch.as_tensor(htranslation,dtype=torch.float32,device=device))
  self.register_buffer('_query_human_region',torch.as_tensor(regional,dtype=torch.float32,device=device))
  self._hs_cache={};self._cache_hs=bool(cache_hs)
  with torch.no_grad():
   identity=torch.eye(3,device=device).expand(len(self.scene.background_anchors),3,3)
   self._static_part=tuple(x.detach() for x in self.sbank(self.scene.background_anchors,identity))
   if cache_hs:
    for i in range(len(query)):self._hs_cache[i]=self._human_component(i)
  self.initial_obank_identity=state_identity(self.obank.state_dict())
  self.frozen_identity=self._frozen_snapshot()
  self.source_identity={'protocol_id':'AUX_REF_OBJECT','init_path':str(self.init_path),'init_sha256':sha_file(self.init_path),'checkpoint_path':str(self.checkpoint_path),'checkpoint_sha256':sha_file(self.checkpoint_path),'old_checkpoint_step':8000,'old_obank_loaded':False,'object_initialization':'original untrained canonical arrays; colors inherit old RGB-pose sampling bias, common to Pred/Ref','motion_rule':'Object world rotation SLERP + translation linear. Human actual22body/global SO3 joint rotations SLERP, translation linear, actual bounded canonical regional offsets linear. No extrapolation. Static shape/canonical offsets unchanged.','reference_motion_reader':False,'reference_interpolation':False,'query_times_seconds':query.tolist(),'motion_records':self.motion_records,'source_code_sha256':sha_file(__file__)}
  geom=np.load(init['object_init'])
  self._sample_faces=np.asarray(geom['sample_face_ids']).copy() if 'sample_face_ids' in geom else None
  self._sample_uv=np.asarray(geom['sample_barycentric_uv']).copy() if 'sample_barycentric_uv' in geom else None
  del checkpoint,state,filtered

 @property
 def obank(self):return self.scene.obank
 @property
 def hbank(self):return self.scene.hbank
 @property
 def sbank(self):return self.scene.sbank
 @property
 def object_anchors(self):return self.scene.object_anchors
 @property
 def c2w(self):return self.scene.c2w
 @property
 def banks(self):return [self.sbank,self.hbank,self.obank]
 @property
 def human(self):return self.scene.human

 def optimizer_groups(self):return self.obank.optimizer_groups('obank')

 def _index(self,index):
  if isinstance(index,float) or (torch.is_tensor(index) and index.is_floating_point()):raise TypeError('Use integer query index, never a float timestamp')
  index=int(index)
  if not 0<=index<len(self.query_times):raise IndexError(index)
  return index

 @torch.no_grad()
 def _human_component(self,index):
  h=self.human;pose=torch.cat([self._query_human_global[index],self._query_human_body[index],torch.zeros(99,dtype=h.body_pose.dtype,device=h.body_pose.device)])+h.pose_mean
  rotations=batch_rodrigues(pose.reshape(-1,3)).reshape(1,55,3,3)
  joints=h.joint_template[None]+torch.einsum('bk,jck->bjc',h.betas,h.joint_shape_directions)
  _,transforms=batch_rigid_transform(rotations,joints,h.parents,dtype=h.body_pose.dtype)
  mixed=torch.einsum('nj,bjxy->bnxy',h.lbs_weights,transforms);affine=mixed[0,:,:3,:3]
  corrective=((rotations[:,1:]-torch.eye(3,dtype=h.body_pose.dtype,device=h.body_pose.device)).reshape(1,486)@h.pose_directions).reshape(h.num_gaussians,3)
  local_region=torch.einsum('nj,jc->nc',h.body_region_weights,self._query_human_region[index])
  canonical=h.canonical_centres()+corrective+local_region
  centres=torch.einsum('nij,nj->ni',affine,canonical)+mixed[0,:,:3,3]+self._query_human_translation[index]
  worldR=self.c2w[:3,:3];worldt=self.c2w[:3,3];centres=centres@worldR.T+worldt;affine=worldR[None]@affine
  return tuple(x.detach() for x in self.hbank(centres,affine))

 def frozen_components(self,index):
  index=self._index(index)
  human=self._hs_cache.get(index)
  if human is None:
   human=self._human_component(index)
   if self._cache_hs:self._hs_cache[index]=human
  return self._static_part,human

 def object_motion(self,index,object_R_world=None,object_t_world=None):
  index=self._index(index)
  if (object_R_world is None)!=(object_t_world is None):raise ValueError('Provide both explicit object world rotation and translation, or neither')
  if object_R_world is None:return self.pred_R_world[index].detach(),self.pred_t_world[index].detach()
  R=torch.as_tensor(object_R_world,dtype=self.object_anchors.dtype,device=self.object_anchors.device).detach();t=torch.as_tensor(object_t_world,dtype=self.object_anchors.dtype,device=self.object_anchors.device).detach()
  if R.shape!=(3,3) or t.shape!=(3,):raise ValueError('Explicit motion must be one3x3 R_world and one3-vector t_world')
  if not bool(torch.isfinite(R).all() and torch.isfinite(t).all()):raise ValueError('Nonfinite explicit motion')
  return R,t

 def components(self,index,object_R_world=None,object_t_world=None):
  index=self._index(index);s,h=self.frozen_components(index);R,t=self.object_motion(index,object_R_world,object_t_world)
  anchors=self.object_anchors@R.T+t
  # This call remains in autograd; only R/t and H/S are frozen.
  o=self.obank(anchors,R.expand(len(anchors),3,3))
  return [s,h,o]

 def _render_parts(self,parts,indices,K,w2c,H,W):
  color=torch.cat([self.banks[i].color_logit.sigmoid() for i in indices]);buffer=torch.cat([torch.nn.functional.one_hot(torch.full((self.banks[i].n,),i,device=color.device,dtype=torch.long),3).float() for i in indices])
  return joint_render([parts[i] for i in indices],H,W,K,w2c,bg_color=[1.,1.,1.],colors_precomp=color,add_buffer=buffer)

 def render(self,index,K,w2c,H,W,only=None,object_R_world=None,object_t_world=None):
  parts=self.components(index,object_R_world,object_t_world);indices=[0,1,2] if only is None else [int(only)]
  if any(i not in [0,1,2] for i in indices):raise ValueError('only must be0,1,2 orNone')
  return self._render_parts(parts,indices,K,w2c,H,W),parts

 def render_hs_only(self,index,K,w2c,H,W):
  s,h=self.frozen_components(index)
  return self._render_parts([s,h], [0,1],K,w2c,H,W)

 def object_static_regularization(self):
  bank=self.obank
  return 100*torch.relu(bank.log_scale.exp()-.035).square().mean()+.01*(bank.local_offset()/.005).square().mean()

 def object_regularization_components(self):
  bank=self.obank
  return {'scale':100*torch.relu(bank.log_scale.exp()-.035).square().mean(),'canonical_offset':.01*(bank.local_offset()/.005).square().mean()}

 def _frozen_snapshot(self):
  snapshot={'human':state_identity(self.human.state_dict()),'hbank':state_identity(self.hbank.state_dict()),'sbank':state_identity(self.sbank.state_dict()),'object_anchors':tensor_identity(self.object_anchors),'c2w':tensor_identity(self.c2w),'original_object_R0':tensor_identity(self.scene.object_R0),'original_object_translation':tensor_identity(self.scene.object_translation),'original_object_rot_delta':tensor_identity(self.scene.object_rot_delta),'query_pred_R':tensor_identity(self.pred_R_world),'query_pred_t':tensor_identity(self.pred_t_world),'query_human_global':tensor_identity(self._query_human_global),'query_human_body':tensor_identity(self._query_human_body),'query_human_translation':tensor_identity(self._query_human_translation),'query_human_region':tensor_identity(self._query_human_region)}
  return snapshot

 def identity_snapshot(self):
  return {'protocol_id':'AUX_REF_OBJECT','frozen':self._frozen_snapshot(),'object':state_identity(self.obank.state_dict()),'initial_obank_identity':self.initial_obank_identity,'query_times_seconds':self.query_times.tolist(),'trainable_parameters':sorted(name for name,p in self.named_parameters() if p.requires_grad)}

 def assert_frozen(self):
  if self._frozen_snapshot()!=self.frozen_identity:raise AssertionError('Frozen H/S, anchors, or prediction motion changed')
  actual={name for name,p in self.named_parameters() if p.requires_grad};expected={'scene.obank.'+k for k in GaussianBank.fields}
  if actual!=expected:raise AssertionError(('Unexpected trainable parameters',actual,expected))
  return True

 @torch.no_grad()
 def object_canonical_export(self):
  bank=self.obank;cpu=lambda x:x.detach().cpu().numpy().copy();ids=cpu(bank.anchor_id);local=bank.local_offset();out={'centres_canonical_m':cpu(self.object_anchors[bank.anchor_id]+local),'anchor_canonical_m':cpu(self.object_anchors[bank.anchor_id]),'local_offset_m':cpu(local),'canonical_frame':cpu(quaternion_to_matrix(bank.quat)),'scale_m':cpu(bank.log_scale.clamp(-9,0).exp()),'opacity':cpu(bank.opacity_logit.sigmoid()),'colors':cpu(bank.color_logit.sigmoid()),'anchor_id':ids,'stable_id':cpu(bank.stable_id),'parent_id':cpu(bank.parent_id),'next_id':cpu(bank.next_id)}
  if self._sample_faces is not None:out['attachment_face_id']=self._sample_faces[ids].copy()
  if self._sample_uv is not None:
   uv=self._sample_uv[ids];out['attachment_barycentric']=np.c_[1-uv.sum(-1),uv]
  return out


def cpu_check():
 torch.set_num_threads(2);reports={}
 for dev in ['dev1','dev2']:
  init=torch.load(OLD/f'{dev}_initialization/initialization.pt',map_location='cpu',weights_only=False);times=np.asarray(init['timestamps'],float);mid=len(times)//2;query=np.array([times[0],times[mid],(times[mid]+times[mid+1])/2,times[-1]])
  obj=AuxObjectScene(dev,query,device='cpu');assert obj.assert_frozen()
  parity=[]
  with torch.no_grad():
   for qi,oldi in [(0,0),(1,mid),(3,len(times)-1)]:
    old=obj.scene.components(oldi);new=obj.components(qi)
    parity.append({'query_index':qi,'original_frame':oldi,'max_human_xyz_difference_m':float((old[1][0]-new[1][0]).abs().max()),'max_human_affine_difference':float((old[1][1]-new[1][1]).abs().max()),'max_object_xyz_difference_m':float((old[2][0]-new[2][0]).abs().max())})
  assert max(r['max_human_xyz_difference_m'] for r in parity)<3e-6
  assert max(r['max_human_affine_difference'] for r in parity)<3e-6
  assert obj.obank.n==4096
  # CPU autograd geometry check only; CUDA-render data gradients are a separate
  # necessary preflight. Explicit external motion is always detached.
  R=torch.eye(3,requires_grad=True);t=torch.tensor([0.,0.,2.],requires_grad=True);parts=obj.components(2,R,t);loss=parts[2][0].sum()+parts[2][1].sum()+parts[2][2].sum()+parts[2][3].sum()+obj.obank.color_logit.sigmoid().sum();loss.backward()
  assert R.grad is None and t.grad is None
  grad={name:bool(p.grad is not None and torch.isfinite(p.grad).all()) for name,p in obj.obank.named_parameters()}
  assert all(grad.values()) and not any(p.grad is not None for p in obj.human.parameters())
  assert obj.assert_frozen();obj.zero_grad(set_to_none=True)
  export=obj.object_canonical_export();assert (np.linalg.norm(export['local_offset_m'],axis=-1)<.005).all();assert not any(x.requires_grad for x in obj.frozen_components(0)[0]+obj.frozen_components(0)[1])
  # Boundary validation does not construct a second scene/renderer.
  reports[dev]={'status':'passed','source_identity':obj.source_identity,'initial_obank_identity':obj.initial_obank_identity,'frozen_identity':obj.frozen_identity,'snapshot':obj.identity_snapshot(),'source_grid_parity':parity,'all5_object_attribute_geometry_gradients_finite':grad,'external_motion_gradient_detached':True,'human_background_no_grad':True,'object_n':obj.obank.n,'interpolated_query':obj.motion_records[2],'scale_fixed_world':float(init['world_scale']),'render_gradient_check':'not_run_on_CPU; required GPU preflight remains'}
  del obj
 return {'status':'passed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_used':False,'reference_assets_read':False,'code_sha256':sha_file(__file__),'devs':reports}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--cpu-check',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
 if not args.cpu_check:raise SystemExit('Import AuxObjectScene or request --cpu-check')
 result=cpu_check()
 if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'devs':{k:{'object_n':v['object_n'],'parity':v['source_grid_parity']} for k,v in result['devs'].items()}},indent=2))
