"""One fixed eight-pair Q-prefix gradient-scale calibration, no parameter update."""
from common import *
import torch,numpy as np
from module_acceptance import BackwardBudget
from render_utils import load_model,CalibratedCamera
from hoi_modules.temporal_evidence import TemporalEvidenceModule,motion_parameters
from hoi_modules.projected_motion import make_renderer
from restore_state import restore_model,clone_cpu,exact
from utils.loss_utils import ssim

def norm(loss,params,budget):
    budget.charge();grads=torch.autograd.grad(loss,params,allow_unused=True)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    return float(torch.sqrt(sum(g.double().square().sum() for g in grads if g is not None)))

def main():
    torch.set_num_threads(4);cfg=read(RUN/'configs/v8.json');parent=RUN/'runs/B_Q/checkpoint_fine_001000.pt';config=RUN/'runs/B_Q/effective_config.json'
    assert read(RUN/'runs/B_Q/run.json')['status']=='completed' and not (RUN/'evaluation_summary.json').exists()
    model,(ds,h,opt,pipe),state=load_model(parent,config);restore_model(model,opt,state);before=clone_cpu(model.capture());accum=model._deformation_accum.clone()
    render=make_renderer(cfg['scale_bound']);module=TemporalEvidenceModule('plain',cfg['scale_bound']);rows={r['frame_id']:r for r in read(OLD/'inputs/hos_backpack/manifest.json')['frames']}
    cache=read(RUN/'track_cache_manifest.json');order=torch.load(RUN/'protocol/pair_order.pt',weights_only=False)['order'][:8];budget=BackwardBudget('lambda_calibration');params=motion_parameters(model);results=[]
    for i,pid in enumerate(order.tolist()):
        p=cache['pairs'][pid];cams=[CalibratedCamera(rows[p[n]],j,True) for j,n in enumerate(['source_frame','target_frame'])]
        pred=torch.stack([render(c,model,pipe,torch.zeros(3,device='cuda'))['render'] for c in cams]);gt=torch.stack([c.original_image for c in cams]).cuda();Q=.8*(pred-gt).abs().mean()+.2*(1-ssim(pred,gt));qn=norm(Q,[p for n,p in params],budget);del pred,gt,Q
        loss,stats,detail=module(model,*cams,dict(np.load(p['cache']['path'])));assert detail is not None and stats['supported']>0
        un=norm(detail['unit_loss'],[p for n,p in params],budget);assert un>0 and qn>0
        results.append(dict(batch=i,pair_id=pid,Q_frame_ids=[c.image_name for c in cams],Q_gradient_L2=qn,unit_gradient_L2=un,teacher_a=stats['a'],supported=stats['supported']));print(json.dumps(results[-1]),flush=True)
    mq=float(np.median([r['Q_gradient_L2'] for r in results]));mt=float(np.median([r['unit_gradient_L2'] for r in results]));raw=.5*mq/mt;lam=float(np.clip(raw,.001,10.))
    exact(before,model.capture());assert torch.equal(accum,model._deformation_accum)
    result=dict(status='pass',parent=identity(parent),formula='clip(0.5*median(norm grad Q)/median(norm grad plain L_unit BEFORE a), .001, 10)',batches=results,parameters=[n for n,p in params],median_Q=mq,median_unit=mt,unclipped_lambda=raw,lambda0=lam,clipped=lam!=raw,parameters_and_optimizer_unchanged=True,cache=identity(RUN/'track_cache_manifest.json'),pair_order=identity(RUN/'protocol/pair_order.pt'),backwards=16,optimizer_steps=0,development_scores_read=False)
    save_json(RUN/'lambda_calibration.json',result);accept=read(RUN/'module_acceptance.json');accept['lambda_calibration']=identity(RUN/'lambda_calibration.json');accept['lambda_status']='pass';save_json(RUN/'module_acceptance.json',accept)
if __name__=='__main__':main()
