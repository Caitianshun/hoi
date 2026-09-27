"""Read-only repeatability evidence after failed D2; never upgrades the gate."""
from common import *
import torch,cv2,numpy as np
from render_utils import load_model,CalibratedCamera
from diagnostic_render import make_renderer,diff
from gradient_router import *
from loss_policy import regional_rgb
from budget import CallBudget
def run():
    torch.set_num_threads(4);budget=CallBudget('D2_localize_only')
    gate=read(RUN/'routing_equivalence.json');assert gate['status']=='failed'
    failed={r['name']:r for r in gate['rows'] if not r['passed']}
    model,(_,h,_,pipe),state=load_model(V5/'runs/B_U/checkpoint_fine_001000.pt',V5/'runs/B_U/effective_config.json');del state
    G,A,rows=partition_current_optimizer_parameters(model);allp=G+A
    before={r['name']:tensor_hash(p) for r,p in allp};render=make_renderer(read(V5/'runs/B_U/effective_config.json')['scale_bound'])
    by={f['frame_id']:f for f in read(OLD/'inputs/hos_backpack/manifest.json')['frames']};frames=[by[n] for n in ['00001','00041']]
    cams=[CalibratedCamera(f,i) for i,f in enumerate(frames)];gt=torch.stack([c.original_image for c in cams]).cuda();mask=torch.stack([torch.tensor(cv2.imread(f['mask_path'],0)>=128) for f in frames]).cuda()
    out=RUN/'diagnostics/gradient_localization';out.mkdir(exist_ok=False);values=[];assets=[]
    for index,mode in enumerate(['original','original','uniform_all','original','uniform_all','original']):
        model.optimizer.zero_grad(set_to_none=True)
        pkgs=[render(c,model,pipe,torch.zeros(3,device='cuda')) for c in cams];pred=torch.stack([p['render'] for p in pkgs]);q=[p['viewspace_points'] for p in pkgs]
        U,_=regional_rgb(pred,gt,mask,'uniform','fine');F,_=regional_rgb(pred,gt,mask,'balanced_fine','fine');R=model.compute_regulation(h.time_smoothness_weight,h.l1_time_planes,h.plane_tv_weight)
        if mode=='original':(U+R).backward();qg=[x.grad for x in q]
        else:qg,_,_=compute_routed_gradients(model,U,F,R,q,'uniform_all')
        grad={r['name']:p.grad.detach().cpu().clone() for r,p in allp if r['name'] in failed};grad.update({f'q/{i}':g.detach().cpu().clone() for i,g in enumerate(qg)})
        values.append(grad);path=out/f'actual_gradients_{index}_{mode}.pt';assets.append(atomic_checkpoint(path,grad))
        assert before=={r['name']:tensor_hash(p) for r,p in allp},'Parameter changed in read-only repeat probe'
        del pkgs,pred,q,U,F,R;torch.cuda.empty_cache()
    comparisons=[];samples={}
    for name,row in failed.items():
        base=values[0][name]
        for i in range(1,len(values)):
            d=diff(values[i][name],base);comparisons.append(dict(name=name,index=i,mode=['original','original','uniform_all','original','uniform_all','original'][i],difference=d,original_declared_atol=row['atol'],original_declared_rtol=row['rtol'],within_original_threshold=d['max_abs']<=row['atol'] and d['relative_L2']<=row['rtol']))
        flat=base.reshape(-1);idx=torch.arange(0,len(flat),max(1,len(flat)//1024));worst=torch.topk((values[4][name]-base).abs().reshape(-1),k=min(128,len(flat))).indices;ids=torch.unique(torch.cat([idx,worst])).numpy()
        key=name.replace('.','_');samples[key+'_indices']=ids
        for i,val in enumerate(values):samples[key+f'_repeat{i}']=val[name].reshape(-1).numpy()[ids]
    sample_path=out/'gradient_float32_samples.npz';np.savez_compressed(sample_path,**samples)
    save_json(out/'localization.json',dict(status='read_only_evidence_only_gate_remains_failed',parameter_hashes_unchanged=before,comparisons=comparisons,full_gradient_assets=assets,small_samples=identity(sample_path),optimizer_steps=0,nonfinite=0,reference='https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.grid_sample.html',interpretation='CUDA grid_sample backward is documented nondeterministic; this does not by itself prove the cause or justify changing the frozen tolerance'))
    print('Read-only localization finished; original gate unchanged')
if __name__=='__main__':run()
