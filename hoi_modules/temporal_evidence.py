"""External correspondence loss with optional detached mixture-motion gate."""
import math
import torch
from .projected_motion import motion_moments

def mixture_gate(variance,source_uv,target_uv):
    with torch.no_grad():
        tau=4+.1*torch.linalg.vector_norm(target_uv-source_uv,dim=-1)
        return .2+.8/(1+variance/tau.square())

def weighted_temporal_loss(mu,alpha,variance,source_uv,target_uv,log_confidence,width,height,mode):
    size=mu.new_tensor([width,height]);diag=math.hypot(width,height)
    predicted=source_uv+mu*size
    delta=predicted-target_uv;error=torch.linalg.vector_norm(delta,dim=-1)
    rho=torch.sqrt((delta/diag).square().sum(-1)+(1/diag)**2)-1/diag
    teacher=torch.isfinite(log_confidence)&torch.isfinite(target_uv).all(-1)&torch.isfinite(source_uv).all(-1)
    a=torch.exp(torch.logsumexp(log_confidence[teacher].double(),dim=0)-math.log(int(teacher.sum()))) if teacher.any() else mu.new_zeros((),dtype=torch.float64)
    supported=teacher&(alpha>.01)
    gate=mixture_gate(variance,source_uv,target_uv) if mode=='mixture' else torch.ones_like(alpha)
    if supported.any():
        logw=log_confidence[supported].double()+gate[supported].double().log()
        weights=torch.exp(logw-logw.max())
        unit=(weights*rho[supported].double()).sum()/weights.sum()
        loss=a*unit
        weighted_error=(weights*error[supported].double()).sum()/weights.sum()
    else:
        unit=mu.sum()*0;loss=unit;weighted_error=unit.detach()
    stats=dict(teacher_valid=int(teacher.sum()),supported=int(supported.sum()),a=float(a),loss=float(loss.detach()),unit=float(unit.detach()),unweighted_error_px=float(error[supported].mean().detach()) if supported.any() else None,weighted_error_px=float(weighted_error.detach()),gate_mean=float(gate[supported].mean()) if supported.any() else None,gate_min=float(gate[supported].min()) if supported.any() else None,variance_mean_px2=float(variance[supported].mean()) if supported.any() else None,alpha_mean=float(alpha[teacher].mean()) if teacher.any() else None,empty_reason=None if supported.any() else ('no_teacher' if not teacher.any() else 'no_model_alpha_support'))
    return loss,unit,stats,dict(predicted=predicted.detach(),teacher_valid=teacher.detach(),supported=supported.detach(),alpha=alpha.detach(),variance=variance.detach(),gate=gate.detach(),unweighted_error=error.detach())

class TemporalEvidenceModule:
    MODES=('off','plain','mixture')
    def __init__(self,mode,bound):
        assert mode in self.MODES;self.mode=mode;self.bound=float(bound)
    def __call__(self,model,source_camera,target_camera,observations):
        if self.mode=='off':return model.get_xyz.sum()*0,dict(empty_reason='module_off'),None
        device=model.get_xyz.device
        # Teacher filtering before raster sampling prevents NaN target arithmetic.
        valid=torch.as_tensor(observations['valid'],device=device,dtype=torch.bool)
        uv=torch.as_tensor(observations['source_uv'],device=device,dtype=torch.float32)[valid]
        target=torch.as_tensor(observations['target_uv'],device=device,dtype=torch.float32)[valid]
        logc=torch.as_tensor(observations['log_confidence'],device=device,dtype=torch.float64)[valid]
        if len(uv)==0:return model.get_xyz.sum()*0,dict(empty_reason='no_teacher',teacher_valid=0,supported=0),None
        mu,alpha,v,info=motion_moments(model,source_camera,target_camera,uv,self.bound,second=self.mode=='mixture')
        loss,unit,stats,details=weighted_temporal_loss(mu,alpha,v,uv,target,logc,source_camera.image_width,source_camera.image_height,self.mode)
        stats.update(info);details['unit_loss']=unit;return loss,stats,details

def motion_parameters(model):
    """Locked 4DGS query_time depends on xyz/grid/feature_out/pos_deform only.

    Reachability is additionally measured by the integration acceptance test.
    """
    result=[('xyz',model._xyz)]
    for name,p in model._deformation.named_parameters():
        if p.requires_grad and any(s in name for s in ['deformation_net.grid.grids.','deformation_net.feature_out.','deformation_net.pos_deform.']):result.append((name,p))
    return result
