"""Independent one-backward U/F references versus explicit routed derivatives."""
from common import *
import torch,cv2,numpy as np
from render_utils import load_model,CalibratedCamera
from diagnostic_render import make_renderer,diff
from gradient_router import *
from loss_policy import regional_rgb
from budget import CallBudget
def run():
    assert not (RUN/'routing_equivalence.json').exists(), 'Immutable prior diagnostic exists; use a separately authorized isolated run'
    torch.set_num_threads(4);budget=CallBudget('D2_gradients')
    parent=V5/'runs/B_U/checkpoint_fine_001000.pt';model,(_,hidden,_,pipe),state=load_model(parent,V5/'runs/B_U/effective_config.json');del state
    assert hidden.no_do and hidden.no_dshs;bound=read(V5/'runs/B_U/effective_config.json')['scale_bound'];render=make_renderer(bound)
    G,A,partition=partition_current_optimizer_parameters(model);save_json(RUN/'gradient_partition.json',dict(parameters=partition,source=identity(parent),expected_None_source='locked deformation.forward_dynamic bypasses timenet, opacity_deform and shs_deform'))
    by={f['frame_id']:f for f in read(OLD/'inputs/hos_backpack/manifest.json')['frames']};frames=[by[n] for n in ['00001','00041']];cameras=[CalibratedCamera(f,i) for i,f in enumerate(frames)]
    masks=torch.stack([torch.tensor(cv2.imread(f['mask_path'],0)>=128) for f in frames]).cuda();gt=torch.stack([c.original_image for c in cameras]).cuda();bg=torch.zeros(3,device='cuda')
    outputs={};losses={}
    for policy in ['U1','U2','F1','F2','C_route','uniform_all']:
        model.optimizer.zero_grad(set_to_none=True)
        pkgs=[render(c,model,pipe,bg) for c in cameras];pred=torch.stack([p['render'] for p in pkgs]);q=[p['viewspace_points'] for p in pkgs]
        U,_=regional_rgb(pred,gt,masks,'uniform','fine');F,_=regional_rgb(pred,gt,masks,'balanced_fine','fine');R=model.compute_regulation(hidden.time_smoothness_weight,hidden.l1_time_planes,hidden.plane_tv_weight)
        losses[policy]=dict(U=float(U.detach()),F=float(F.detach()),R=float(R.detach()))
        if policy[0] in ['U','F']:
            ((U+R) if policy[0]=='U' else F).backward();gq=[x.grad for x in q]
            validate_gradients(G+A,[p.grad for _,p in G+A])
        else:gq,_,_=compute_routed_gradients(model,U,F,R,q,policy)
        d={row['name']:None if p.grad is None else p.grad.detach().cpu().clone() for row,p in G+A}
        d.update({f'q/{i}':x.detach().cpu().clone() for i,x in enumerate(gq)});outputs[policy]=d
        del pkgs,pred,q,U,F,R;torch.cuda.empty_cache()
    epsilon=float(np.finfo(np.float32).eps);rows=[]
    scopes={r['name']:r['scope'] for r in partition};scopes.update({'q/0':'q','q/1':'q'})
    for policy in ['C_route','uniform_all']:
        for name,g in outputs[policy].items():
            ref='F' if policy=='C_route' and scopes[name]=='A' else 'U';a=outputs[ref+'1'][name];b=outputs[ref+'2'][name]
            if a is None:
                assert b is None and g is None;row=dict(policy=policy,name=name,scope=scopes[name],state='expected_None',passed=True)
            else:
                repeat=diff(a,b);delta=diff(g,a);scale=float(a.abs().max());atol=2*repeat['max_abs']+64*epsilon*scale+1e-10;rtol=2*repeat['relative_L2']+64*epsilon
                passed=delta['finite'] and delta['max_abs']<=atol and (delta['relative_L2']<=rtol if delta['reference_norm']>1e-12 else delta['max_abs']<=atol)
                row=dict(policy=policy,name=name,scope=scopes[name],state='finite_zero' if scale==0 else 'finite_nonzero',repeat=repeat,difference=delta,atol=atol,rtol=rtol,passed=passed)
            rows.append(row)
    save_json(RUN/'routing_equivalence.json',dict(status='gradient_checks_passed' if all(r['passed'] for r in rows) else 'failed',rows=rows,losses=losses,tolerance_rule='declared 2x measured repeat plus 64 float32 eps times reference scale; absolute floor 1e-10; not changed after comparison',optimizer_density_resume='pending',q_source='explicit return from U+R (R independent of q)',batch=[f['frame_id'] for f in frames]))
    assert all(r['passed'] for r in rows),'Gradient mismatch'
    print('D2 independent gradient checks passed',len(rows))
if __name__=='__main__':run()
