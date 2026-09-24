"""Frozen-checkpoint dyn_o intervention with identical source Gaussian weights.

No optimization, reference input, checkpoint write, or source-weight reassignment.
"""
from pathlib import Path
import argparse,os,sys,json,time,hashlib
import numpy as np
import torch
import torch.nn.functional as F
import cv2

ROOT=Path('/home/cai_tianshun/Project/HOI')
OLD=ROOT/'experiments/mosca_baseline_20260922'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    os.environ['GS_BACKEND']='native_add3';sys.path.insert(0,str(a.repo))
    from lib_mosca.static_gs import StaticGaussian
    from lib_mosca.dynamic_gs import DynSCFGaussian
    from lib_render.render_helper import render
    from validation_utils import expose_metric_gaussians
    dev='cuda:0';start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    m=json.loads((OLD/'common_input/input_manifest.json').read_text());q=json.loads((OLD/'common_input/queries_first_frame.json').read_text())
    training=json.loads((a.checkpoint_dir/'run.json').read_text());s=float(training.get('world_scale',1.))
    sources={name:a.checkpoint_dir/name for name in ['photometric_s_model_native_add3.pth','photometric_d_model_native_add3.pth']}
    static=StaticGaussian.load_from_ckpt(torch.load(sources['photometric_s_model_native_add3.pth'],map_location=dev,weights_only=False),device=dev).to(dev).eval()
    dynamic=DynSCFGaussian.load_from_ckpt(torch.load(sources['photometric_d_model_native_add3.pth'],map_location=dev,weights_only=False),device=dev).to(dev).eval()
    assert not dynamic.fast_inference_flag and bool(dynamic.dyn_o_flag)
    expose_metric_gaussians(static,dynamic,s)
    h,w=m['height'],m['width'];k=torch.tensor(m['K'],device=dev);c2w=torch.tensor(m['c2w'],device=dev);w2c=torch.linalg.inv(c2w)
    n=q['manual_evaluation_count'];uv=torch.tensor(q['points'][:n],dtype=torch.float32,device=dev)
    grid=torch.stack([2*uv[:,0]/(w-1)-1,2*uv[:,1]/(h-1)-1],-1)[None,None]
    def sample(x):return F.grid_sample(x[None],grid,align_corners=True)[0,:,0].T
    def cpu(x):return x.detach().cpu().numpy()
    with torch.no_grad():
        source_gs=[static(),dynamic(q['query_frame'])]
        source=render(source_gs,h,w,k,w2c,bg_color=[0.,0.,0.])
        source_alpha=sample(source['alpha'])[:,0];valid=cpu(source_alpha>=.05)
        sk=dynamic.scf.get_skinning_weights(query_xyz=dynamic.get_xyz(),query_t=dynamic.ref_time,
                attach_ind=dynamic.attach_ind,skinning_weight_correction=dynamic._skinning_weight if dynamic.w_correction_flag else None)
        gate=sk[2].clamp(0,1);ns=len(source_gs[0][0]);nd=len(source_gs[1][0])
        aux=torch.zeros((ns+nd,3),device=dev);aux[:ns,0]=1.;aux[ns:,1]=1.;aux[ns:,2]=gate
        contribution=sample(render(source_gs,h,w,k,w2c,bg_color=[0.,0.,0.],add_buffer=aux)['buf'])/source_alpha[:,None]
        stats=[]
        for i in range(n):
            f=contribution[i]
            stats.append({'query_id':q['point_ids'][i],'static_fraction':float(f[0]),'dynamic_fraction':float(f[1]),
                          'dynamic_conditional_gate':float(f[2]/f[1].clamp_min(1e-8))})
        (a.output/'source_gate_stats.json').write_text(json.dumps({'world_scale':s,'query_stats':stats,
                 'meaning':'Source pixel alpha-weighted RBF gate; not opacity or calibrated probability.'},indent=2)+'\n')
        for condition,flag in [('gate_on',True),('gate_off',False)]:
            dest=a.output/condition;dest.mkdir();dynamic.dyn_o_flag.fill_(flag)
            pred=[]
            for t in range(len(m['frame_paths'])):
                target=[source_gs[0],dynamic(t)]
                xyz=torch.cat([x[0] for x in target])
                transported=render(source_gs,h,w,k,w2c,bg_color=[0.,0.,0.],add_buffer=xyz)
                pred.append(cpu(sample(transported['buf'])/source_alpha[:,None].clamp_min(1e-8)))
                if t in [0,16,25,31,65,113]:
                    rd=render(target,h,w,k,w2c,bg_color=[1.,1.,1.])
                    im=(cpu(rd['rgb'].permute(1,2,0)).clip(0,1)*255).round().astype(np.uint8)
                    cv2.imwrite(str(dest/f'{t:05d}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
            pred=np.asarray(pred)
            np.savez_compressed(dest/'query_trajectories.npz',predicted=pred,predicted_valid_mask=np.broadcast_to(valid,pred.shape[:2])&np.isfinite(pred).all(-1),
                                frame_times=m['timestamp_seconds'],entity=np.asarray(q['entity'][:n]),query_id=np.asarray(q['point_ids'][:n]),
                                coordinate_frame=np.array('behave_world_k1_color'),units=np.array('m'),source_alpha=cpu(source_alpha))
            record={'status':'completed','condition':condition,'training_performed':False,'global_transform_fitted_on_evaluation':False,
                    'source_weights':'Fixed original gate-on checkpoint at query frame, identical in both interventions',
                    'intervention':'Only destination dyn_o_flag; false uses 1-DQ_EPS=0.999 dynamic weight, not exact 1.',
                    'limitation':'All dynamic Gaussians affected, not an object-only intervention; no claim of learned improvement or physical point identity.',
                    'world_scale_inverted':s,'checkpoint_sha256':{x:sha(y) for x,y in sources.items()},'script_sha256':sha(__file__),
                    'renderer_sha256':sha(a.repo/'lib_render/gauspl_renderer_native_add3.py'),'input_manifest_sha256':sha(OLD/'common_input/input_manifest.json')}
            (dest/'protocol.json').write_text(json.dumps(record,indent=2)+'\n')
    (a.output/'run.json').write_text(json.dumps({'status':'completed','wall_seconds':time.perf_counter()-start,
            'gpu':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'original_checkpoint_unchanged':{x:sha(y) for x,y in sources.items()}},indent=2)+'\n')
    print(a.output)
if __name__=='__main__':main()
