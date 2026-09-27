"""Validate one-Gaussian covariance failure with frozen camera and adjoints."""
from common import *
from restore_state import clone_cpu
from finite_guard import stats
import torch,numpy as np,os
import diff_gaussian_rasterization as raster
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';torch.set_num_threads(4)
    source=RUN/'runs/A_replay1b';a=list(torch.load(source/'raster_forward_last_inputs.pt',map_location='cpu',weights_only=False));b=torch.load(source/'raster_backward_last_inputs.pt',map_location='cpu',weights_only=False)
    assert torch.equal(a[8],b[8]) and torch.equal(a[9],b[9])
    evidence=read(source/'kernel_case_analysis.json');ids=sorted({i for rows in evidence['bad_rows'].values() for i in rows});assert len(ids)==1
    for index in [1,2,3,4,5,7,14]:
        if torch.is_tensor(a[index]) and a[index].numel():a[index]=a[index][ids].clone()
    out=RUN/'minimal_repro';out.mkdir(exist_ok=True);arrays={f'forward_{i}':x.numpy() for i,x in enumerate(a) if torch.is_tensor(x)};arrays.update(grad_RGB=b[12].numpy(),grad_depth=b[13].numpy());np.savez_compressed(out/'one_gaussian_case.npz',**arrays)
    save_json(out/'case.json',dict(forward_scalars={str(i):x for i,x in enumerate(a) if not torch.is_tensor(x)},scale_upper_bound=read(RUN/'protocol/stabilization_proposal.json')['upper_bound'],source_gaussian_ids=ids,full_frame_adjoint_preserved=True,external_requirements='Compatible PyTorch/CUDA and exact rasterizer binary or locked source build; no dataset or full model required'))
    rows=[];ledger=RUN/'protocol/A_probes.jsonl'
    for mode in ['original','bounded']:
        used=sum(1 for _ in ledger.open()) if ledger.exists() else 0;assert used<64
        with ledger.open('a') as f:f.write(json.dumps(dict(index=used,case='one_gaussian',mode=mode))+'\n')
        args=[x.cuda() if torch.is_tensor(x) else x for x in a]
        if mode=='bounded':args[4]=args[4].clamp(max=read(out/'case.json')['scale_upper_bound'])
        outputs=raster._C.rasterize_gaussians(*args)
        n,color,depth,radii,geom,bins,img=outputs
        back=(args[0],args[1],radii,args[2],args[4],args[5],args[6],args[7],args[8],args[9],args[10],args[11],b[12].cuda(),b[13].cuda(),args[14],args[15],args[16],geom,n,bins,img,args[18])
        grads=raster._C.rasterize_gaussians_backward(*back)
        names=['screen','color','opacity','xyz','covariance','SH','scale','rotation'];row=dict(mode=mode,RGB=stats(color),depth=stats(depth),radii=radii.cpu().tolist(),gradients={name:stats(x) for name,x in zip(names,grads)})
        rows.append(row);torch.save(clone_cpu(dict(outputs=outputs,gradients=grads)),out/f'{mode}_kernel_outputs.pt')
    save_json(out/'validation.json',dict(rows=rows,case=identity(out/'one_gaussian_case.npz'),optimization_steps=0))
if __name__=='__main__':run()
