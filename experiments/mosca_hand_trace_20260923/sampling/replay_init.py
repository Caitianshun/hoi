"""Replay A initialization exactly, record sample provenance, perform no training."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, sys, time, traceback
import numpy as np
import torch
from omegaconf import OmegaConf
ROOT=Path('/home/cai_tianshun/Project/HOI')
PREV=ROOT/'experiments/mosca_validation_20260923'
A=PREV/'a_normalized_exact/model'
OLD=ROOT/'experiments/mosca_baseline_20260922'
REPO=PREV/'code/MoSca'
os.environ['GS_BACKEND']='native_add3'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(PREV/'code'))
from validation_utils import tensor_state_hash

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'executed_replay.py')
    files=['bundle/bundle.pth','bundle/bundle_cams.pth','mosca/mosca.pth','track_identification.npz']
    for name in files:
        (out/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(A/name,out/name)
    import mosca_reconstruct as mr
    import lib_mosca.photo_recon as photo
    from lib_prior.prior_loading import Saved2D
    cfg=OmegaConf.load(A/'config.yaml')
    scale=float(json.loads((A/'run.json').read_text())['world_scale'])
    mask=np.load(OLD/'segmentation/segmentation.npz')['foreground'].astype(np.float32)
    def load_epi(self,*args,**kwargs):
        self.register_gradfree_buffer('epi',torch.from_numpy(mask.copy()));self.has_epi=True;return self
    Saved2D.load_epi=load_epi
    original_dynamic=mr.DynReconstructionSolver.get_dynamic_model
    def seeded(self,*args,**kwargs):
        mr.seed_everything(mr.SEED);return original_dynamic(self,*args,**kwargs)
    mr.DynReconstructionSolver.get_dynamic_model=seeded
    original_fetch=photo.fetch_leaves_in_world_frame
    trace={}
    def fetch(**kw):
        if kw['n_attach']!=30000:return original_fetch(**kw)
        assert kw.get('subsample',1)==1
        state=torch.random.get_rng_state().clone()
        mask_cpu=kw['input_mask_list'].detach().cpu().bool()
        counts=mask_cpu.flatten(1).sum(1).numpy();T,H,W=mask_cpu.shape
        flat=torch.nonzero(mask_cpu.flatten(),as_tuple=False).flatten()
        gen=torch.Generator(device='cpu');gen.set_state(state)
        choice=torch.randperm(len(flat),generator=gen)[:kw['n_attach']]
        picked=flat[choice];frame=(picked//(H*W)).numpy();uv=torch.stack([picked%W,(picked//W)%H],-1).numpy()
        result=original_fetch(**kw)
        assert np.array_equal(result[-1].detach().cpu().numpy(),frame)
        # append_new_gs concatenates ascending reference-frame groups, preserving
        # randperm order within each group, rather than original randperm order.
        order=np.concatenate([np.flatnonzero(frame==t) for t in np.unique(frame)])
        source_frame=frame[order];source_uv=uv[order]
        idx=torch.as_tensor(order,device=result[0].device)
        raw_world=result[0][idx].detach().cpu().numpy()
        ti=torch.as_tensor(source_frame,device=result[0].device);xy=torch.as_tensor(source_uv,device=result[0].device)
        dep=kw['input_dep_list'][ti,xy[:,1],xy[:,0]].detach().cpu().numpy()
        # Verify the trace against independently re-lifted actual source pixels.
        hom=kw['cams'].get_homo_coordinate_map(H,W)[xy[:,1],xy[:,0]]
        points=kw['cams'].backproject(hom,torch.as_tensor(dep,device=hom.device))
        lifted=[]
        for t in np.unique(source_frame):
            ids=np.flatnonzero(source_frame==t)
            lifted.append(kw['cams'].trans_pts_to_world(int(t),points[ids]).detach().cpu().numpy())
        err=float(np.max(np.abs(np.concatenate(lifted)-raw_world)))
        assert err<1e-6,err
        np.savez_compressed(out/'actual_dynamic_seeds.npz',gaussian_id=np.arange(len(order)),
            raw_source_frame=source_frame,raw_pixel_xy=source_uv,raw_depth_m=dep/scale,
            raw_world_xyz_normalized=raw_world,raw_world_xyz_m=raw_world/scale,
            raw_rgb=result[4][idx].detach().cpu().numpy(),raw_radius_normalized=result[2][idx].detach().cpu().numpy(),
            original_randperm_position=order,pool_counts_by_frame=counts,world_scale=scale)
        np.savez_compressed(out/'first_frame_candidate_mask.npz',dynamic_depth_mask=mask_cpu[0].numpy())
        trace.update(pool_size=len(flat),sample_count=len(order),raw_world=raw_world,frame=source_frame,
                     lift_max_abs_error_normalized=err,pool_counts_by_frame=counts.tolist())
        return result
    photo.fetch_leaves_in_world_frame=fetch
    def inspect_no_training(self,*args,**kw):
        model=kw['d_model'];current=model.state_dict();expected=torch.load(A/'initial_dynamic.pth',map_location='cpu',weights_only=False)
        checks={k:bool(torch.equal(v.detach().cpu(),expected[k])) for k,v in current.items()}
        state_hash=tensor_state_hash(current);reference_hash=json.loads((A/'initialization.json').read_text())['dynamic_state_tensor_sha256']
        assert all(checks.values()) and state_hash==reference_hash, [k for k,v in checks.items() if not v]
        xyz=model.get_xyz().detach().cpu().numpy();err=float(np.max(np.abs(xyz-trace['raw_world'])))
        assert err<1e-6,err
        assert np.array_equal(model.ref_time.detach().cpu().numpy(),trace['frame'])
        trace.update(dynamic_tensor_hash=state_hash,all_dynamic_tensors_exact=True,
            local_storage_roundtrip_max_abs_error_normalized=err,
            static_tensor_hash=tensor_state_hash(kw['s_model'].state_dict()),
            initial_static_hash_matches=tensor_state_hash(kw['s_model'].state_dict())==tensor_state_hash(torch.load(A/'initial_static.pth',map_location='cpu',weights_only=False)))
        # This hook intentionally returns before any optimizer step.
    mr.DynReconstructionSolver.photometric_fit=inspect_no_training
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();status={'status':'running','training_performed':False}
    (out/'status.json').write_text(json.dumps(status,indent=2))
    try:
        mr.photometric_reconstruct(str(OLD/'common_input'),str(out),cfg)
        torch.cuda.synchronize()
        trace.pop('raw_world');trace.pop('frame')
        status.update(status='completed',**trace,world_scale=scale,
            scope='Exact A initialization replay, no optimization; provenance ids valid for initial_dynamic only, not final densified/pruned rows.',
            source_sha256={str(A/f):sha(A/f) for f in files+['initial_dynamic.pth','initial_static.pth','config.yaml']},
            script_sha256=sha(__file__),prediction_provenance_sha256=sha(out/'actual_dynamic_seeds.npz'))
    except BaseException as exc:
        status.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        status.update(wall_seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_allocated_bytes=torch.cuda.max_memory_allocated())
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({k:v for k,v in status.items() if k not in ['pool_counts_by_frame','source_sha256']},indent=2))
if __name__=='__main__':main()
