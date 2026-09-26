"""Common local candidates and source-excluded descriptors for F0/F1/F2."""
from pathlib import Path
import json,math
import numpy as np,torch
from torch import nn
E=Path(__file__).resolve().parents[1]
QD=68;KD=58;D=32
F2_PARAMS=(QD+1)*D+KD*D
WIDTH=min(range(8,129),key=lambda w:abs((QD+KD+2)*w-F2_PARAMS))
class Scorer(nn.Module):
 def __init__(self,variant):
  super().__init__();self.variant=variant
  if variant=='F1':
   self.mlp=nn.Sequential(nn.Linear(QD+KD,WIDTH),nn.ReLU(),nn.Linear(WIDTH,1,bias=False));nn.init.zeros_(self.mlp[-1].weight)
  if variant=='F2':
   self.q=nn.Linear(QD,D);self.k=nn.Linear(KD,D,bias=False);nn.init.zeros_(self.q.weight);nn.init.zeros_(self.q.bias)
 def forward(self,features,return_weights=False):
  q,k,values,valid,r,old=features;score=r.clamp_min(1e-8).log()
  if self.variant=='F1':score=score+self.mlp(torch.cat((q[:,None,:].expand(-1,8,-1),k),-1)).squeeze(-1)
  elif self.variant=='F2':score=score+(self.q(q)[:,None,:]*self.k(k)).sum(-1)/math.sqrt(D)
  gate=valid.sum(1)>=2
  # Degenerate rows receive safe zeros, then all weights explicitly zero.
  safe=score.masked_fill(~valid,-1e9);safe=torch.where(gate[:,None],safe,torch.zeros_like(safe));weights=torch.softmax(safe,1)*valid*gate[:,None]
  color=torch.where(gate[:,None],(weights[:,:,None]*values).sum(1),old)
  return (color,weights) if return_weights else color

class Memory:
 def __init__(self,dev,device='cuda'):
  p=E/'support'/dev;self.obs=dict(np.load(p/'observations.npz'));self.query=dict(np.load(p/'queries.npz'));self.obj=dict(np.load(p/'B0_object.npz'));self.T=int(self.obs['time'].max()+1);self.device=device;self.cache={};self.selections={}
 def features(self,exclude=None):
  key=-1 if exclude is None else int(exclude)
  if key in self.cache:return self.cache[key]
  a=self.obs;b=self.obj;ids=self.query['candidate_ids'].copy();dd=self.query['distance'].copy();N=len(ids)
  if exclude is not None:ids[:,exclude]=-1 # BEFORE selection and every aggregation
  selected=np.full((N,8),-1,np.int64);distance=np.zeros((N,8),np.float32);counts=(ids>=0).sum(1)
  for i in range(N):
   ts=np.flatnonzero(ids[i]>=0)
   if len(ts)>8:ts=ts[np.rint(np.linspace(0,len(ts)-1,8)).astype(int)]
   selected[i,:len(ts)]=ids[i,ts];distance[i,:len(ts)]=dd[i,ts]
  valid=selected>=0;safe=selected.clip(0);rel=np.clip(a['distance_px'][safe]/5,.1,1)*np.exp(-.5*(distance/.005)**2);rel*=valid
  k=np.concatenate((a['rgb'][safe],a['descriptor'][safe],(a['xyz'][safe]-b['centres_canonical_m'][:,None,:])/.005,a['view'][safe],(a['distance_px'][safe]/5).clip(0,1)[...,None],a['purity'][safe][...,None],rel[...,None],(a['resolution'][safe]/.005)[...,None],valid[...,None]),axis=-1).astype(np.float32);assert k.shape[-1]==KD;k*=valid[...,None]
  median=np.zeros((N,KD),np.float32)
  for i in np.flatnonzero(counts):median[i]=np.median(k[i,valid[i]],axis=0)
  q=np.c_[b['centres_canonical_m'],self.query['normal'],b['colors'],median,counts/12].astype(np.float32);assert q.shape[-1]==QD
  values=a['rgb'][safe].copy();values*=valid[...,None]
  self.cache[key]=tuple(torch.as_tensor(x,device=self.device) for x in (q,k,values,valid,rel,b['colors']))
  self.selections[key]=selected
  return self.cache[key]
 def prepare(self):
  self.features()
  for t in range(self.T):self.features(t)
 def audit(self,scorer):
  features=self.features();color,w=scorer(features,True);w=w.detach().cpu().numpy();ids=self.selections[-1];eligible=np.arange(len(ids))
  ii=eligible[np.rint(np.linspace(0,len(eligible)-1,min(8,len(eligible)))).astype(int)] if len(eligible) else []
  rows=[]
  for i in ii:
   ok=ids[i]>=0;src=ids[i,ok];rows.append(dict(local_id=int(i),stable_id=int(self.obj['stable_id'][i]),face=int(self.query['face'][i]),canonical_xyz_m=self.obj['centres_canonical_m'][i].tolist(),source_indices=src.tolist(),source_times=self.obs['time'][src].tolist(),source_pixels=self.obs['pixel'][src].tolist(),RGB=self.obs['rgb'][src].tolist(),reliability=features[4][i,ok].detach().cpu().tolist(),weights=w[i,ok].tolist(),fallback=bool(ok.sum()<2),fallback_reason=('fewer than two legal source times' if ok.sum()<2 else None)))
  return rows
