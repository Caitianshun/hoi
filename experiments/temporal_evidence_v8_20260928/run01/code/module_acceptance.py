"""Bounded new-path integration checks; no optimizer calls or development RGB."""
from common import *
import torch,numpy as np,time
from hoi_modules.projected_motion import *
from hoi_modules.temporal_evidence import *
from render_utils import load_model,CalibratedCamera
from restore_state import restore_model,exact
from utils.loss_utils import ssim

class BackwardBudget:
    def __init__(self,label):self.label=label;self.path=RUN/'protocol/extra_backward.jsonl'
    def charge(self):
        n=len(self.path.read_text().splitlines()) if self.path.exists() else 0;assert n<64
        with self.path.open('a') as f:f.write(json.dumps(dict(label=self.label,time_unix=time.time()))+'\n')

def synthetic(budget):
    E=np.eye(4);K=[[32.,0,16.],[0,32.,16.],[0,0,1.]]
    frame=dict(frame_id='synthetic',time=0,width=32,height=32,K=K,w2c=E.tolist());c=CalibratedCamera(frame,0,False)
    xyz=torch.tensor([[0.,0.,2.]],device='cuda',requires_grad=True);q=torch.tensor([[16.,16.]],device='cuda');ps,valid=project(xyz,c)
    assert torch.allclose(ps,q,atol=1e-5) and valid.all()
    E[0,3]=.25;ct=CalibratedCamera(dict(frame,w2c=E.tolist()),1,False);pt,_=project(xyz,ct);assert torch.allclose(pt-ps,torch.tensor([[4.,0.]],device='cuda'),atol=1e-5)
    attrs=(xyz,torch.full((1,3),math.log(.1),device='cuda'),torch.tensor([[1.,0,0,0]],device='cuda'),torch.zeros((1,1),device='cuda'))
    signs=[]
    for dx in [4.,-4.]:
        d=torch.tensor([[dx/32,0.,1.]],device='cuda',requires_grad=True);im=raster_attributes(c,attrs,d,10);s=sample_accumulated(im,q);mu=s[:,:2]/s[:,2:];assert abs(float(mu[0,0]*32)-dx)<1e-4
        budget.charge();mu.sum().backward();assert d.grad is not None and xyz.grad is None;signs.append(float(mu[0,0]*32))
    attrs2=tuple(torch.cat([x.detach(),x.detach()],0) for x in attrs)
    plus=torch.tensor([[.125,0,1],[.125,0,1]],device='cuda');opposite=plus.clone();opposite[1,0]=-.125
    moments=[]
    for colors in [plus,opposite]:
        first=sample_accumulated(raster_attributes(c,attrs2,colors,10),q)
        sq=torch.cat([colors[:,:2].square(),torch.zeros((2,1),device='cuda')],1)
        m2=sample_accumulated(raster_attributes(c,attrs2,sq,10),q)[:,:2]/first[:,2:]
        mean=first[:,:2]/first[:,2:];v=((m2-mean.square()).clamp_min(0)*32**2).sum(-1);moments.append(float(v))
    assert moments[0]<1e-5 and abs(moments[1]-16*8/9)<1e-4
    image=torch.zeros((3,1,2),device='cuda');image[0,0]=torch.tensor([1.,2.],device='cuda');image[2,0]=torch.tensor([1.,4.],device='cuda');sample=sample_accumulated(image,torch.tensor([[.5,0.]],device='cuda'));assert abs(float(sample[0,0]/sample[0,2])-.6)<1e-6
    uv=torch.tensor([[2.,3.]],device='cuda');mu=torch.tensor([[.1,.1]],device='cuda',requires_grad=True);lc=torch.zeros(1,device='cuda',dtype=torch.float64)
    l,_,st,_=weighted_temporal_loss(mu,torch.zeros(1,device='cuda'),lc.float(),uv,uv,lc,32,32,'mixture');assert st['empty_reason']=='no_model_alpha_support';budget.charge();l.backward();assert torch.equal(mu.grad,torch.zeros_like(mu))
    v=torch.tensor([1e8],device='cuda',requires_grad=True);g=mixture_gate(v,uv,uv+1);assert not g.requires_grad and .2<=g.item()<=1
    return dict(status='pass',signed_kernel_translation_pixels=signs,static_point_moving_camera_displacement=[4.,0.],same_displacement_variance=moments[0],opposite_displacement_variance=moments[1],expected_opposite_variance=16*8/9,sample_then_divide=.6,divide_then_sample_wrong=.75,no_alpha_connected_zero=True,gate_detached=True)

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(4);budget=BackwardBudget('module_acceptance');result=dict(status='running');save_json(RUN/'diagnostics/interface_acceptance.json',result)
    result['synthetic']=synthetic(budget)
    model,(ds,h,opt,pipe),state=load_model(V5/'runs/B_U/checkpoint_fine_001000.pt',V5/'runs/B_U/effective_config.json');restore_model(model,opt,state)
    cache=read(RUN/'track_cache_manifest.json');pid=int(torch.load(RUN/'protocol/pair_order.pt',weights_only=False)['order'][0]);p=cache['pairs'][pid];obs=dict(np.load(p['cache']['path']))
    rows={r['frame_id']:r for r in read(OLD/'inputs/hos_backpack/manifest.json')['frames']};cs=CalibratedCamera(rows[p['source_frame']],0,True);ct=CalibratedCamera(rows[p['target_frame']],1,False)
    accum=model._deformation_accum.clone();table=model._deformation_table.clone();params=[('xyz',model._xyz),('SH_dc',model._features_dc),('SH_rest',model._features_rest),('opacity',model._opacity),('scale',model._scaling),('rotation',model._rotation)]+[(n,v) for n,v in model._deformation.named_parameters() if v.requires_grad]
    module=TemporalEvidenceModule('mixture',read(RUN/'configs/v8.json')['scale_bound']);loss,stats,details=module(model,cs,ct,obs);assert stats['supported']>0 and torch.isfinite(loss)
    budget.charge();grads=torch.autograd.grad(loss,[p for n,p in params],allow_unused=True)
    gradrows=[]
    for (name,p),g in zip(params,grads):
        norm=0 if g is None else float(g.double().norm());gradrows.append(dict(name=name,state='None' if g is None else 'nonzero' if norm else 'zero',norm=norm))
        if name in ['SH_dc','SH_rest','opacity','scale','rotation'] or any(s in name for s in ['scales_deform','rotations_deform','opacity_deform','shs_deform']):assert g is None or norm==0,name
    reachable=[x['name'] for x in gradrows if x['norm']>0];assert 'xyz' in reachable and any('pos_deform' in n for n in reachable) and any('grid.grids' in n for n in reachable)
    allowed={n for n,p in motion_parameters(model)};assert all(n in allowed for n in reachable)
    assert torch.equal(accum,model._deformation_accum) and torch.equal(table,model._deformation_table)
    # Identical sampled tensors isolate the gate-off mathematical reduction.
    uv=torch.tensor([[1.,2.],[2.,3.]],device='cuda');mu=torch.ones_like(uv)*.01;alpha=torch.ones(2,device='cuda');v=torch.zeros(2,device='cuda');lc=torch.tensor([-1.,-1000.],device='cuda',dtype=torch.float64)
    a=weighted_temporal_loss(mu,alpha,v,uv,uv+1,lc,1277,718,'plain')[0];b=weighted_temporal_loss(mu,alpha,v,uv,uv+1,lc,1277,718,'mixture')[0];assert torch.equal(a,b)
    render=make_renderer(read(RUN/'configs/v8.json')['scale_bound']);pkg=render(cs,model,pipe,torch.zeros(3,device='cuda'));gt=cs.original_image.cuda();U=(pkg['render']-gt).abs().mean();Q=.8*U+.2*(1-ssim(pkg['render'][None],gt[None]));budget.charge();Q.backward();assert model._features_dc.grad is not None and model._features_dc.grad.norm()>0
    result.update(status='pass',flow_gradients=gradrows,teacher_stats=stats,motion_collection=sorted(allowed),Q_rgb_SH_grad_norm=float(model._features_dc.grad.norm()),Q_formula='.8 mean(abs(raw-GT))+.2(1-SSIM11)',upstream_extra_structure_coefficient=opt.lambda_dssim,no_hidden_deformation_state_write=True,gate_off_reduction_exact=True,optimizations=0,off='returns graph-connected zero and makes no raster/deformation calls; original U renderer preserved',cache_manifest=identity(RUN/'track_cache_manifest.json'))
    save_json(RUN/'diagnostics/interface_acceptance.json',result);print(json.dumps(dict(status='pass',supported=stats['supported'],reachable=len(reachable))))
if __name__=='__main__':main()
