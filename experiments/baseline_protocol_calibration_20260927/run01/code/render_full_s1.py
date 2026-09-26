"""Complete original S1 H/O/S loader, time adapter, and frozen forward export.

All original trained banks and original RGB-predicted motion are loaded. No AUX
obank or Ref motion is imported. Native timestamps use exact legacy components;
intermediate timestamps use the previously established prediction interpolation.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation,Slerp
from export_protocol import ROOT,RUN,OLD,identity,sha,save
sys.path.insert(0,str(OLD/'code'))
from gaussian_scene import StructuredScene
from human_lbs import batch_rodrigues,batch_rigid_transform
from lib_render.render_helper import render as joint_render


def linear(values,left,right,alpha):
    values=np.asarray(values);a=alpha.reshape((len(alpha),)+(1,)*(values.ndim-1))
    return values[left]*(1-a)+values[right]*a


class FullS1Scene:
    def __init__(self,dev,query_times,device='cpu'):
        self.dev=dev;self.init_path=OLD/f'{dev}_initialization/initialization.pt'
        self.checkpoint_path=OLD/f'{dev}_S1_v1/checkpoint_008000.pt'
        init=torch.load(self.init_path,map_location='cpu',weights_only=False)
        ck=torch.load(self.checkpoint_path,map_location='cpu',weights_only=False)
        assert init['reference_used'] is False and ck['step']==8000
        assert ck['initialization_sha256']==sha(self.init_path)
        self.scene=StructuredScene(init)
        self.scene.resize_for_load(ck['model'])
        self.scene.load_state_dict(ck['model'],strict=True)
        for param in self.scene.parameters():param.requires_grad_(False)
        self.initialization=init;self.source_times=np.asarray(init['timestamps'],np.float64)
        q=np.asarray(query_times,np.float64)
        assert q.ndim==1 and len(q)>0 and np.isfinite(q).all()
        assert np.all(np.diff(self.source_times)>0)
        if q.min()<self.source_times[0] or q.max()>self.source_times[-1]:
            raise ValueError('Query outside original S1 timestamp range: extrapolation forbidden, no silent clamp')
        right=np.searchsorted(self.source_times,q,side='left');exact=self.source_times[right]==q
        left=np.where(exact,right,right-1);dt=self.source_times[right]-self.source_times[left]
        alpha=np.divide(q-self.source_times[left],dt,out=np.zeros_like(q),where=dt>0)
        self.query_times=q;self.exact_indices=np.where(exact,right,-1)
        self.motion_records=[dict(query_time_seconds=float(t),left_frame=int(l),right_frame=int(r),
            left_time_seconds=float(self.source_times[l]),right_time_seconds=float(self.source_times[r]),
            right_weight=float(a),interpolated=bool(l!=r),extrapolated=False) for t,l,r,a in zip(q,left,right,alpha)]
        with torch.no_grad():
            Rs=self.scene.object_R(torch.arange(len(self.source_times))).numpy().astype(np.float64)
            ts=self.scene.object_translation.numpy().astype(np.float64)
            human=self.scene.human;mean=human.pose_mean.numpy().astype(np.float64)
            angles=np.concatenate([human.global_orient.numpy(),human.body_pose.numpy()],1).reshape(len(self.source_times),22,3).astype(np.float64)+mean[:66].reshape(1,22,3)
            result=np.empty((len(q),22,3))
            for j in range(22):result[:,j]=Slerp(self.source_times,Rotation.from_rotvec(angles[:,j]))(q).as_rotvec()
            result-=mean[:66].reshape(1,22,3)
            htrans=linear(human.transl.numpy(),left,right,alpha)
            region=linear(human.regional_offsets().numpy(),left,right,alpha)
        self.scene=self.scene.to(device)
        self.R=torch.tensor(Slerp(self.source_times,Rotation.from_matrix(Rs))(q).as_matrix(),dtype=torch.float32,device=device)
        self.t=torch.tensor(linear(ts,left,right,alpha),dtype=torch.float32,device=device)
        self.global_orient=torch.tensor(result[:,0],dtype=torch.float32,device=device)
        self.body_pose=torch.tensor(result[:,1:].reshape(len(q),63),dtype=torch.float32,device=device)
        self.htrans=torch.tensor(htrans,dtype=torch.float32,device=device)
        self.region=torch.tensor(region,dtype=torch.float32,device=device)
        self.source_identity=dict(protocol='E0_complete_original_S1_RGB_motion',initialization=identity(self.init_path),checkpoint=identity(self.checkpoint_path),
            original_checkpoint_step=8000,old_obank_loaded=True,all_state_keys_loaded_strict=True,
            object_motion_source='original RGB-predicted S1 object_R0 + learned rotation correction and learned world translation',
            human_motion_source='original RGB-predicted S1 plus original optimized parameters',reference_motion_used=False,
            motion_rule='Exact source timestamp: unchanged legacy forward. Intermediate: object SO(3) SLERP, linear world translation; human actual 22 joint rotations SO(3) SLERP, linear camera translation and actual bounded regional offsets; static shape/canonical offsets unchanged. Extrapolation rejected.',
            interpolation_records=self.motion_records,point_counts={k:getattr(self.scene,k).n for k in ['sbank','hbank','obank']})

    @torch.no_grad()
    def components(self,index,force_interpolation=False):
        if self.exact_indices[index]>=0 and not force_interpolation:
            return self.scene.components(int(self.exact_indices[index]))
        s=self.scene;h=s.human
        pose=torch.cat([self.global_orient[index],self.body_pose[index],torch.zeros(99,dtype=h.body_pose.dtype,device=h.body_pose.device)])+h.pose_mean
        rotations=batch_rodrigues(pose.reshape(-1,3)).reshape(1,55,3,3)
        joints=h.joint_template[None]+torch.einsum('bk,jck->bjc',h.betas,h.joint_shape_directions)
        _,transforms=batch_rigid_transform(rotations,joints,h.parents,dtype=h.body_pose.dtype)
        mixed=torch.einsum('nj,bjxy->bnxy',h.lbs_weights,transforms);affine=mixed[0,:,:3,:3]
        corrective=((rotations[:,1:]-torch.eye(3,dtype=h.body_pose.dtype,device=h.body_pose.device)).reshape(1,486)@h.pose_directions).reshape(h.num_gaussians,3)
        region=torch.einsum('nj,jc->nc',h.body_region_weights,self.region[index])
        canonical=h.canonical_centres()+corrective+region
        centres=torch.einsum('nij,nj->ni',affine,canonical)+mixed[0,:,:3,3]+self.htrans[index]
        R=s.c2w[:3,:3];centres=centres@R.T+s.c2w[:3,3];affine=R[None]@affine
        bg=s.background_anchors;I=torch.eye(3,device=bg.device).expand(len(bg),3,3)
        obj=s.object_anchors@self.R[index].T+self.t[index]
        return [s.sbank(bg,I),s.hbank(centres,affine),s.obank(obj,self.R[index].expand(len(obj),3,3))]

    @torch.no_grad()
    def render(self,index,K,w2c,H,W):
        parts=self.components(index)
        colors=torch.cat([bank.color_logit.sigmoid() for bank in self.scene.banks])
        buffer=torch.cat([torch.nn.functional.one_hot(torch.full((bank.n,),i,device=K.device,dtype=torch.long),3).float() for i,bank in enumerate(self.scene.banks)])
        return joint_render(parts,H,W,K,w2c,bg_color=[1.,1.,1.],colors_precomp=colors,add_buffer=buffer),parts


def native_check(dev,device):
    torch.set_num_threads(2);start=time.perf_counter()
    manifest=json.loads((RUN/'inputs'/f'behave_{dev}'/'manifest.json').read_text())
    frames=manifest['frames'];indices=[0,len(frames)//2,len(frames)-1]
    model=FullS1Scene(dev,[frames[i]['time_seconds'] for i in indices],device)
    checks=[]
    for qi,source_i in enumerate(indices):
        frame=frames[source_i];old=model.scene.components(source_i);new=model.components(qi)
        interp=model.components(qi,force_interpolation=True)
        component_error=max(float((x-y).abs().max()) for a,b in zip(old,new) for x,y in zip(a,b))
        interpolation_error=max(float((x-y).abs().max()) for a,b in zip(old,interp) for x,y in zip(a,b))
        assert component_error==0.0 and interpolation_error<=2e-5
        record=dict(source_frame_index=source_i,time_seconds=frame['time_seconds'],exact_components_max_abs=component_error,
                    forced_interpolation_components_max_abs=interpolation_error)
        if device!='cpu':
            K=torch.tensor(frame['K'],dtype=torch.float32,device=device);w2c=torch.tensor(frame['w2c'],dtype=torch.float32,device=device)
            expected,_=model.scene.render(source_i,K,w2c,frame['height'],frame['width'])
            actual,_=model.render(qi,K,w2c,frame['height'],frame['width'])
            differences={name:float((actual[name]-expected[name]).abs().max()) for name in ['rgb','alpha']}
            assert max(differences.values())<=1e-6
            record['render_max_abs']=differences
        checks.append(record)
    report=dict(status='passed',dev=dev,device=device,backend='native_add3' if device!='cpu' else 'no_render',
                optimization_steps=0,evaluation_RGB_read=False,checks=checks,source_identity=model.source_identity,
                wall_seconds=time.perf_counter()-start,source_code=identity(__file__))
    path=RUN/'protocol'/f'behave_{dev}_native_{"cpu" if device=="cpu" else "render"}_check.json'
    save(path,report);print(json.dumps({k:report[k] for k in ['dev','status','device','checks','wall_seconds']}))


def export_render(dev,freeze):
    torch.set_num_threads(2)
    frozen=json.loads(Path(freeze).read_text())
    assert frozen.get('status')=='all_states_frozen' and frozen.get('scope')=='behave_pair', 'All planned BEHAVE final models must be frozen first'
    for asset in frozen['assets']:
        assert sha(asset['path'])==asset['sha256'],f'Frozen asset changed: {asset["path"]}'
    check=json.loads((RUN/'protocol'/f'behave_{dev}_native_render_check.json').read_text());assert check['status']=='passed'
    base=RUN/'inputs'/f'behave_{dev}'
    evalm=json.loads((base/'evaluation_manifest.json').read_text());train=json.loads((base/'manifest.json').read_text())
    selected=np.unique(np.linspace(0,len(train['frames'])-1,min(16,len(train['frames']))).astype(int)).tolist()
    groups={'camera1_E':evalm['frames'],'camera0_paired_E':evalm['paired_camera0_frames'],
            'camera0_full_training_fit':train['frames']}
    out=RUN/'evaluation'/'E0'/dev
    if out.exists():raise RuntimeError('Refusing to overwrite E0 evaluation export')
    out.mkdir(parents=True);records=[];start=time.perf_counter()
    for label,frames in groups.items():
        scene=FullS1Scene(dev,[f['time_seconds'] for f in frames],'cuda')
        save(out/f'{label}_motion.json',scene.source_identity)
        for i,frame in enumerate(frames):
            K=torch.tensor(frame['K'],dtype=torch.float32,device='cuda');w2c=torch.tensor(frame['w2c'],dtype=torch.float32,device='cuda')
            ret,_=scene.render(i,K,w2c,frame['height'],frame['width'])
            rgb=ret['rgb'].permute(1,2,0).cpu().numpy();alpha=ret['alpha'].squeeze().cpu().numpy()
            dest=out/label/f'{frame["frame_id"]}.npz';dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest,rgb=rgb,alpha=alpha)
            png=dest.with_suffix('.png')
            assert cv2.imwrite(str(png),np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1])
            records.append(dict(group=label,frame_id=frame['frame_id'],time_seconds=frame['time_seconds'],render=identity(dest),
                                png=identity(png),source_frame=frame,visualization_selected=(label!='camera0_full_training_fit' or i in selected)))
        del scene
        torch.cuda.empty_cache()
    report=dict(status='completed',dev=dev,source_checkpoint=identity(OLD/f'{dev}_S1_v1/checkpoint_008000.pt'),
        all_finals_freeze=identity(freeze),native_time_check=identity(RUN/'protocol'/f'behave_{dev}_native_render_check.json'),
        frame_count=len(records),frames=records,wall_seconds=time.perf_counter()-start,optimization_steps=0,
        RGB_GT_read=False,fit_frame_selection='all original 114/98 camera0 frames; preview only uniform at most16; paired E kept separately')
    save(out/'manifest.json',report);print(json.dumps({'dev':dev,'renders':len(records),'wall_seconds':report['wall_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['native-check','render']);p.add_argument('--dev',choices=['dev1','dev2','both'],default='both')
    p.add_argument('--device',default='cpu');p.add_argument('--freeze',type=Path);a=p.parse_args()
    for d in ['dev1','dev2'] if a.dev=='both' else [a.dev]:
        if a.command=='native-check':native_check(d,a.device)
        else:
            if a.freeze is None:raise ValueError('render requires --freeze')
            export_render(d,a.freeze)
