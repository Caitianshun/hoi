"""No-optimizer same-state gradient isolation and one fixed-domain comparison."""
from common import *
import torch,inspect,random,os,argparse
from restore_state import *
from finite_guard import stats
from adapter_v4 import ProtocolScene
from loss_policy import regional_rgb
from scale_domain import bound_log_scale
import gaussian_renderer,train as official
def run(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';torch.set_num_threads(4)
    out=Path(output);out.mkdir(exist_ok=False)
    _,(dataset,hidden,opt,pipe)=official_config(2);dataset.source_path=str(OLD/'inputs/hos_backpack/manifest.json');dataset.model_path=str(out)
    model=official.GaussianModel(dataset.sh_degree,hidden);scene=ProtocolScene(dataset,model);cameras=scene.getTrainCameras()
    extent=scene.cameras_extent;save_json(out/'domain_choice.json',dict(scale_upper_bound=extent,source='Inherited train-only scene_extent',formula='exp(min(final_log_scale, log(scene_extent)))',derivative='original below upper bound; zero above; PyTorch clamp boundary convention',scope='Rendered final scales only; canonical parameters and density rules unchanged'))
    src=inspect.getsource(gaussian_renderer.render);a='scales_final = pc.scaling_activation(scales_final)';assert src.count(a)==1
    snapshot=out/'source_snapshots/render_bounded.py';snapshot.parent.mkdir(exist_ok=True)
    src=src.replace(a,'scales_final = pc.scaling_activation(bound_log_scale(scales_final,extent))');ns=gaussian_renderer.__dict__.copy();ns.update(bound_log_scale=bound_log_scale,extent=extent);exec(compile(src,str(snapshot),'exec'),ns);bounded=ns['render'];snapshot.write_text(src)
    all_rows=[];comparisons=[];ledger=RUN/'protocol/A_probes.jsonl'
    background=torch.tensor([1.,1.,1.],device='cuda')
    for start,label in [('pre_step_002103.pt','failure'),('pre_step_002101.pt','normal')]:
        parent=load_completed(RUN/'runs/A_replay1b'/start)
        references={}
        objectives=['balanced_rgb','uniform_rgb','regularizers'] if label=='failure' else ['balanced_rgb']
        for domain in ['original','bounded']:
            for objective in objectives:
                count=sum(1 for _ in ledger.open()) if ledger.exists() else 0
                assert count<64
                with ledger.open('a') as f:f.write(json.dumps(dict(index=count,case=label,domain=domain,objective=objective))+'\n')
                restore_model(model,opt,parent);stack,temp=restore_sampler(parent,cameras);batch=[]
                for _ in range(opt.batch_size):
                    batch.append(stack.pop(random.randint(0,len(stack)-1)))
                    if not stack:stack=temp.copy()
                render=gaussian_renderer.render if domain=='original' else bounded
                packages=[render(c,model,pipe,background,stage='fine') for c in batch];pred=torch.stack([x['render'] for x in packages]);gt=torch.stack([c.original_image for c in batch]).cuda();masks=torch.stack([c.foreground_mask for c in batch]).cuda()
                if objective=='regularizers':value=model.compute_regulation(hidden.time_smoothness_weight,hidden.l1_time_planes,hidden.plane_tv_weight)
                else:value,_=regional_rgb(pred,gt,masks,'uniform' if objective=='uniform_rgb' else 'balanced_fine','fine')
                named=[(g['name']+'/'+str(i),p) for g in model.optimizer.param_groups for i,p in enumerate(g['params']) if p.requires_grad];grads=torch.autograd.grad(value,[p for _,p in named],allow_unused=True)
                row=dict(case=label,domain=domain,objective=objective,frame_ids=[c.image_name for c in batch],loss=float(value),forward=stats(pred),gradients={name:stats(g) for (name,p),g in zip(named,grads)},optimizer_steps=0,source=identity(RUN/'runs/A_replay1b'/start))
                all_rows.append(row);save_json(out/'summary.json',dict(rows=all_rows,comparisons=comparisons))
                key=objective
                values=dict(pred=pred.detach().cpu(),grads={name:None if g is None else g.detach().cpu() for (name,p),g in zip(named,grads)})
                if domain=='original':references[key]=values
                else:
                    old=references[key];delta=dict(case=label,objective=objective,forward_max_abs=float((old['pred']-values['pred']).abs().max()),gradient_differences={})
                    for name,x in values['grads'].items():
                        y=old['grads'][name]
                        if x is None or y is None:delta['gradient_differences'][name]=dict(state='not_connected');continue
                        finite=torch.isfinite(x)&torch.isfinite(y);d=(x-y)[finite]
                        delta['gradient_differences'][name]=dict(original_nonfinite=int((~torch.isfinite(y)).sum()),modified_nonfinite=int((~torch.isfinite(x)).sum()),common_finite_max_abs=float(d.abs().max()) if d.numel() else None)
                    comparisons.append(delta)
                torch.save(values,out/f'{label}_{domain}_{objective}_values.pt')
                del packages,pred,gt,masks,value,grads,values
    save_json(out/'summary.json',dict(rows=all_rows,comparisons=comparisons,domain_choice=identity(out/'domain_choice.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(RUN/'runs/A_isolation'));run(p.parse_args().output)
