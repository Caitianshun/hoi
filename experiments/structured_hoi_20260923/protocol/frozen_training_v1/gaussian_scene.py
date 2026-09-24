"""Three permanently separate Gaussian banks using the calibrated MoSca D renderer.

Learned scene parameters contain no evaluation assets. Positions are in metres.
Human affine frames deform full covariances; object uses one shared rigid pose.
"""
from pathlib import Path
import sys,os
import numpy as np
import torch
from torch import nn
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, quaternion_to_matrix
from human_lbs import HumanLBS

ROOT=Path('/home/cai_tianshun/Project/HOI')
DREPO=ROOT/'experiments/mosca_interface_validation_20260923/code/MoSca'
os.environ['GS_BACKEND']='native_add3'
sys.path.insert(0,str(DREPO))
from lib_render.render_helper import render


class GaussianBank(nn.Module):
    fields=('offset','log_scale','quat','opacity_logit','color_logit')
    buffers_point=('anchor_id','stable_id','parent_id','grad_sum','seen_count')
    def __init__(self,anchor_id,colors,scale,entity,offset_bound):
        super().__init__();n=len(anchor_id);self.entity=int(entity);self.offset_bound=float(offset_bound)
        self.register_buffer('anchor_id',torch.as_tensor(anchor_id,dtype=torch.long))
        self.register_buffer('stable_id',torch.arange(n,dtype=torch.long));self.register_buffer('parent_id',torch.full((n,),-1,dtype=torch.long))
        self.register_buffer('grad_sum',torch.zeros(n));self.register_buffer('seen_count',torch.zeros(n))
        self.register_buffer('next_id',torch.tensor(n,dtype=torch.long))
        self.offset=nn.Parameter(torch.zeros(n,3));self.log_scale=nn.Parameter(torch.as_tensor(scale).float().reshape(n,1).repeat(1,3).log())
        q=torch.zeros(n,4);q[:,0]=1;self.quat=nn.Parameter(q)
        self.opacity_logit=nn.Parameter(torch.full((n,1),-1.0986123))
        self.color_logit=nn.Parameter(torch.logit(torch.as_tensor(colors).float().clamp(.01,.99)))

    @property
    def n(self):return len(self.anchor_id)

    def local_offset(self):
        return self.offset_bound*self.offset/torch.sqrt(1+self.offset.square().sum(-1,keepdim=True))

    def forward(self,anchors,affine):
        fr=affine[self.anchor_id]
        xyz=anchors[self.anchor_id]+torch.einsum('nij,nj->ni',fr,self.local_offset())
        frames=fr@quaternion_to_matrix(self.quat)
        # No opacity floor. Scales have only finite numerical bounds here.
        scale=self.log_scale.clamp(-9,0).exp()
        return xyz,frames,scale,self.opacity_logit.sigmoid(),torch.zeros_like(xyz)

    def optimizer_groups(self,prefix):
        rates={'offset':.003,'log_scale':.003,'quat':.001,'opacity_logit':.03,'color_logit':.008}
        return [{'params':[getattr(self,k)],'lr':v,'name':prefix+'.'+k} for k,v in rates.items()]

    def resize_for_load(self,state,prefix):
        for k in self.fields:setattr(self,k,nn.Parameter(torch.empty_like(state[prefix+k])))
        for k in self.buffers_point:setattr(self,k,torch.empty_like(state[prefix+k]))

    @torch.no_grad()
    def record(self,grad,radii):
        if grad is None:return
        valid=radii>0
        self.grad_sum[valid]+=grad[valid,:2].norm(dim=-1)
        self.seen_count[valid]+=1

    @torch.no_grad()
    def control(self,optimizer,prefix,step,max_points,threshold=.0002):
        """Split selected rendered leaves; all children inherit entity and attachment.

        Adam moments are retained for surviving leaves and zeroed for new children.
        Empty visibility is not itself a pruning criterion; no permanent alpha floor.
        """
        n=self.n;old_ids=self.stable_id.clone()
        prune=(self.opacity_logit.sigmoid().flatten()<.02)&(self.seen_count>=3)
        # Keep a diagnostic failure legible instead of silently deleting a full bank.
        if int((~prune).sum())<32:prune.zero_()
        candidates=((self.grad_sum/self.seen_count.clamp_min(1)>threshold)&(~prune)&(self.seen_count>=3)).nonzero().flatten()
        room=max(0,max_points-int((~prune).sum()))
        if len(candidates)>min(1024,room):
            score=self.grad_sum[candidates]/self.seen_count[candidates]
            candidates=candidates[score.topk(min(1024,room)).indices]
        split=torch.zeros(n,dtype=torch.bool,device=old_ids.device);split[candidates]=True
        keep=(~prune&~split).nonzero().flatten()
        select=torch.cat([keep,candidates,candidates]);born=2*len(candidates)
        if not bool(prune.any()) and not born:
            self.grad_sum.zero_();self.seen_count.zero_()
            return dict(step=step,entity=self.entity,before=n,after=n,pruned=[],split_parents=[],children=[])
        copied={k:getattr(self,k).detach()[select].clone() for k in self.fields}
        if born:
            nk=len(keep);m=len(candidates)
            local=self.local_offset()[candidates]
            jitter=torch.randn_like(local)*self.log_scale[candidates].exp()*.35
            # Bounded canonical displacement; never moves a child to another entity.
            def inverse_bound(x):
                v=x/self.offset_bound;v=v/torch.clamp(v.norm(dim=-1,keepdim=True)/.98,min=1)
                return v/torch.sqrt(1-v.square().sum(-1,keepdim=True))
            plus=inverse_bound(local+jitter);minus=inverse_bound(local-jitter)
            copied['offset'][nk:nk+m]=plus;copied['offset'][nk+m:]=minus
            copied['log_scale'][nk:]-=np.log(1.4)
            alpha=self.opacity_logit[candidates].sigmoid()
            child_alpha=1-torch.sqrt(1-alpha)
            copied['opacity_logit'][nk:]=torch.logit(child_alpha).repeat(2,1)
        for k in self.fields:
            old=getattr(self,k);new=nn.Parameter(copied[k]);setattr(self,k,new)
            for group in optimizer.param_groups:
                if group['name']==prefix+'.'+k:
                    group['params']=[new];break
            state=optimizer.state.pop(old,{})
            for key,val in list(state.items()):
                if torch.is_tensor(val) and val.shape==old.shape:
                    state[key]=val[select].clone()
                    if born:state[key][len(keep):]=0
            optimizer.state[new]=state
        self.anchor_id=self.anchor_id[select]
        self.parent_id=self.parent_id[select]
        self.stable_id=old_ids[select]
        newids=torch.arange(int(self.next_id),int(self.next_id)+born,device=old_ids.device)
        if born:
            self.parent_id[len(keep):]=old_ids[candidates].repeat(2)
            self.stable_id[len(keep):]=newids;self.next_id+=born
        self.grad_sum=torch.zeros(len(select),device=old_ids.device);self.seen_count=torch.zeros_like(self.grad_sum)
        return dict(step=step,entity=self.entity,before=n,after=self.n,pruned=old_ids[prune].tolist(),split_parents=old_ids[candidates].tolist(),children=newids.tolist())


class StructuredScene(nn.Module):
    def __init__(self,initialization):
        super().__init__();p=initialization
        self.human=HumanLBS(p['human_geometry'],p['smplx_model'])
        self.register_buffer('c2w',torch.tensor(p['c2w'],dtype=torch.float32))
        self.register_buffer('timestamps',torch.tensor(p['timestamps'],dtype=torch.float32))
        self.register_buffer('object_anchors',torch.as_tensor(p['object_anchors']).float())
        self.register_buffer('background_anchors',torch.as_tensor(p['background_anchors']).float())
        self.register_buffer('object_R0',torch.as_tensor(p['object_R']).float())
        self.register_buffer('object_t0',torch.as_tensor(p['object_t']).float())
        self.object_rot_delta=nn.Parameter(torch.zeros(len(p['timestamps']),3))
        self.object_translation=nn.Parameter(self.object_t0.clone())
        self.hbank=GaussianBank(np.arange(len(p['human_colors'])),p['human_colors'],p['human_scale'],1,.005)
        self.obank=GaussianBank(np.arange(len(p['object_anchors'])),p['object_colors'],p['object_scale'],2,.005)
        self.sbank=GaussianBank(np.arange(len(p['background_anchors'])),p['background_colors'],p['background_scale'],0,.15)

    def object_R(self,t):return axis_angle_to_matrix(self.object_rot_delta[t])@self.object_R0[t]

    def components(self,t):
        h,frame=self.human(t);R=self.c2w[:3,:3];tr=self.c2w[:3,3]
        h=h@R.T+tr;frame=R[None]@frame
        O=self.object_R(t);obj=self.object_anchors@O.T+self.object_translation[t]
        bg=self.background_anchors;I=torch.eye(3,device=bg.device).expand(len(bg),3,3)
        return [self.sbank(bg,I),self.hbank(h,frame),self.obank(obj,O.expand(len(obj),3,3))]

    @property
    def banks(self):return [self.sbank,self.hbank,self.obank]

    def optimizer_groups(self):
        groups=[]
        for prefix,bank in [('sbank',self.sbank),('hbank',self.hbank),('obank',self.obank)]:groups+=bank.optimizer_groups(prefix)
        for name,param in self.human.named_parameters():
            if param.requires_grad:
                lr=.0003 if name in ['body_pose','global_orient','transl'] else .0001
                groups.append({'params':[param],'lr':lr,'name':'human.'+name})
        groups.extend([{'params':[self.object_rot_delta],'lr':.0005,'name':'object.rotation'},
                       {'params':[self.object_translation],'lr':.0003,'name':'object.translation'}])
        return groups

    def render(self,t,K,w2c,H,W,only=None,target_frame=None):
        parts=self.components(t);indices=[0,1,2] if only is None else [only]
        gs=[parts[i] for i in indices]
        color=torch.cat([self.banks[i].color_logit.sigmoid() for i in indices])
        if target_frame is None:
            buf=torch.cat([torch.nn.functional.one_hot(torch.full((self.banks[i].n,),i,device=K.device,dtype=torch.long),3).float() for i in indices])
        else:
            dst=self.components(target_frame)
            x=torch.cat([dst[i][0] for i in indices]);buf=x@w2c[:3,:3].T+w2c[:3,3]
        ret=render(gs,H,W,K,w2c,bg_color=[1.,1.,1.],colors_precomp=color,add_buffer=buf)
        return ret,parts

    def resize_for_load(self,state):
        for prefix,bank in [('sbank.',self.sbank),('hbank.',self.hbank),('obank.',self.obank)]:bank.resize_for_load(state,prefix)

    def set_stage(self,stage,only=None):
        for i,bank in enumerate(self.banks):
            for p in bank.parameters():p.requires_grad_(stage!='B' and (only is None or only==i))
        for name,p in self.human.named_parameters():
            if name in ['body_pose','global_orient','transl','betas','canonical_offset_raw','regional_residual_raw']:
                active=(only is None or only==1) and not (stage=='B' and name in ['betas','canonical_offset_raw'])
                p.requires_grad_(active)
        self.object_rot_delta.requires_grad_(only is None or only==2)
        self.object_translation.requires_grad_(only is None or only==2)

    def bank_regularization(self):
        loss=0
        for bank in self.banks:
            # Penalize inflated kernels instead of satisfying silhouette/support by giant blobs.
            bound=.08 if bank.entity==0 else .035
            loss=loss+torch.relu(bank.log_scale.exp()-bound).square().mean()*100
            loss=loss+(bank.local_offset()/bank.offset_bound).square().mean()*.01
        return loss
