"""One runtime intervention: intersect dynamic seed candidates with input SAM2 FG."""
from pathlib import Path
import importlib.util,json,os,sys,hashlib,shutil
import numpy as np
import torch
ROOT=Path('/home/cai_tianshun/Project/HOI');PREV=ROOT/'experiments/mosca_validation_20260923'
os.environ['GS_BACKEND']='native_add3'
sys.path.insert(0,str(PREV/'code'));sys.path.insert(0,str(PREV/'code/MoSca'))
from validation_utils import tensor_state_hash
spec=importlib.util.spec_from_file_location('frozen_runner',PREV/'code/run_control.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
import mosca_reconstruct as mr
import lib_mosca.photo_recon as photo
model_out=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
mask_path=ROOT/'experiments/mosca_baseline_20260922/segmentation/segmentation.npz'
foreground=np.load(mask_path)['foreground'].astype(bool)
original=mr.DynReconstructionSolver.get_dynamic_model
record={'intervention':'Only dynamic initialization candidate pool intersects original input SAM2 foreground; static initialization, scaffold, count, opacity, losses and steps unchanged.',
        'reference_used':False,'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest()}
def constrained(self,*args,**kwargs):
    s2d=kwargs['s2d'];fg=torch.from_numpy(foreground).to(s2d.dep.device)
    prior=s2d.dyn_mask.bool()&s2d.dep_mask.bool()
    record.update(candidate_count_original=int(prior.sum()),candidate_count_restricted=int((prior&fg).sum()))
    old=kwargs.get('additional_mask');kwargs['additional_mask']=fg if old is None else fg*old
    return original(self,*args,**kwargs)
mr.DynReconstructionSolver.get_dynamic_model=constrained
old_fetch=photo.fetch_leaves_in_world_frame
def trace_fetch(**kw):
    if kw['n_attach']!=30000:return old_fetch(**kw)
    mask=kw['input_mask_list'].detach().cpu().bool();T,H,W=mask.shape
    gen=torch.Generator(device='cpu');gen.set_state(torch.random.get_rng_state())
    flat=mask.flatten().nonzero().flatten();choice=torch.randperm(len(flat),generator=gen)[:30000]
    selected=flat[choice];frame=(selected//(H*W)).numpy();uv=torch.stack([selected%W,(selected//W)%H],-1).numpy()
    result=old_fetch(**kw);assert np.array_equal(frame,result[-1].detach().cpu().numpy())
    order=np.concatenate([np.flatnonzero(frame==t) for t in np.unique(frame)])
    assert foreground[frame,uv[:,1],uv[:,0]].all()
    idx=torch.as_tensor(order,device=result[0].device)
    np.savez_compressed(model_out/'seed_provenance.npz',gaussian_id=np.arange(len(order)),
        raw_source_frame=frame[order],raw_pixel_xy=uv[order],raw_world_xyz_normalized=result[0][idx].detach().cpu().numpy(),
        raw_rgb=result[4][idx].detach().cpu().numpy(),pool_counts_by_frame=mask.flatten(1).sum(1).numpy())
    return result
photo.fetch_leaves_in_world_frame=trace_fetch
if __name__=='__main__':
    try:
        runner.main()
        static=torch.load(model_out/'initial_static.pth',map_location='cpu',weights_only=False)
        original_static=torch.load(PREV/'a_normalized_exact/model/initial_static.pth',map_location='cpu',weights_only=False)
        record['initial_static_tensor_hash_matches_A']=tensor_state_hash(static)==tensor_state_hash(original_static)
        assert record['initial_static_tensor_hash_matches_A']
        record['status']='completed'
    except BaseException as exc:
        record.update(status='failed',error=repr(exc));raise
    finally:
        model_out.parent.mkdir(parents=True,exist_ok=True)
        (model_out.parent/'intervention.json').write_text(json.dumps(record,indent=2)+'\n')
        if model_out.exists():shutil.copy2(__file__,model_out/'foreground_wrapper_snapshot.py')
