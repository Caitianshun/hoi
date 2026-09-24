"""Input-only frozen body prior. Evaluation references never enter this process."""
from pathlib import Path
import argparse, copy, hashlib, json, os, socket, subprocess, sys, time, traceback
import numpy as np
import cv2
import torch

ROOT=Path('/home/cai_tianshun/Project/HOI')
EXP=ROOT/'experiments/gvhmr_validation_20260923'
REPO=EXP/'code/GVHMR'
INPUT=ROOT/'experiments/mosca_baseline_20260922/common_input/input_manifest.json'
BOXES=ROOT/'experiments/mosca_interface_validation_20260923/human_prior/prepared_input/sam2_person_boxes.npz'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1<<20): h.update(b)
    return h.hexdigest()

def cpu(x):
    if isinstance(x,torch.Tensor): return x.detach().cpu()
    if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); record=dict(status='running',pid=os.getpid(),host=socket.gethostname(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),input_manifest=str(INPUT),input_sha256=sha(INPUT),boxes_sha256=sha(BOXES),reference_used=False,stages={},script_sha256=sha(__file__))
    def save():
        p=out/'run.tmp';p.write_text(json.dumps(record,indent=2)+'\n');p.replace(out/'run.json')
    def stage(name,func):
        t=time.perf_counter();record['current_stage']=name;save();result=func();torch.cuda.synchronize()
        record['stages'][name]=dict(seconds=time.perf_counter()-t,status='completed');save();return result
    save()
    try:
        torch.set_num_threads(8);torch.manual_seed(12345);np.random.seed(12345)
        assert torch.cuda.is_available()
        record['gpu']=torch.cuda.get_device_name();torch.cuda.reset_peak_memory_stats()
        audit=json.loads((EXP/'deployment/weight_audit.json').read_text());assert audit['status']=='passed'
        for f in audit['files']: assert sha(f['path'])==f['sha256']
        record['weight_audit_sha256']=sha(EXP/'deployment/weight_audit.json')
        m=json.loads(INPUT.read_text());assert m['role']=='input_only' and m['camera_id']==0
        for p,h in zip(m['frame_paths'],m['frame_sha256']): assert sha(p)==h
        with np.load(BOXES) as z:
            bbxy=torch.from_numpy(z['bbox_xyxy'].copy())
            assert z['bbox_valid'].all();assert np.array_equal(z['timestamp_seconds'],m['timestamp_seconds'])
        rgb=np.stack([cv2.imread(p)[...,::-1] for p in m['frame_paths']]).copy()
        L=len(rgb);assert L==114
        sys.path.insert(0,str(REPO));os.chdir(REPO)
        from hmr4d.utils.preproc.vitfeat_extractor import get_batch,Extractor
        from hmr4d.utils.preproc.vitpose import VitPoseExtractor
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy,normalize_kp2d
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
        import hmr4d.network.gvhmr.relative_transformer
        import hmr4d.model.gvhmr.utils.endecoder
        from hmr4d.utils.smplx_utils import make_smplx
        from hydra import initialize_config_module,compose
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        bb=get_bbx_xys_from_xyxy(bbxy,base_enlarge=1.2).float()
        imgs,cropbb=get_batch(rgb,bb,img_ds=1.0,path_type='np')
        assert torch.allclose(bb,cropbb,atol=1e-4)
        torch.save(dict(bbox_xyxy=bbxy,bbx_xys=bb,crop_bbx_xys=cropbb),out/'boxes.pt')
        # Preserve the explicit original-image to 256-square crop affine maps.
        aff=[]
        for x,y,s in bb.numpy():
            src=np.array([[x-s/2,y-s/2],[x+s/2,y-s/2],[x,y]],np.float32)
            dst=np.array([[0,0],[255,0],[127.5,127.5]],np.float32)
            aff.append(cv2.getAffineTransform(src,dst))
        np.savez_compressed(out/'crop_geometry.npz',affine_original_to_square=np.stack(aff),bbox_xyxy=bbxy.numpy(),bbx_xys=bb.numpy(),timestamp_seconds=np.asarray(m['timestamp_seconds']),video_frame_index=m['frame_indices'],K=m['K'],c2w=m['c2w'])
        def pose_stage():
            model=VitPoseExtractor();r=model.extract(imgs,bb);del model;torch.cuda.empty_cache();return r
        kp=stage('vitpose_load_and_infer',pose_stage);assert kp.shape==(L,17,3) and torch.isfinite(kp).all();torch.save(kp,out/'vitpose.pt')
        def feature_stage():
            model=Extractor();r=model.extract_video_features(imgs,bb);del model;torch.cuda.empty_cache();return r
        feat=stage('hmr2_load_and_infer',feature_stage);assert feat.shape==(L,1024) and torch.isfinite(feat).all();torch.save(feat,out/'features.pt')
        with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
            cfg=compose(config_name='demo',overrides=['video_name=hoi_quality','static_cam=True'])
        (out/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
        def load_gvhmr():
            model=instantiate(cfg.model,_recursive_=False)
            ck=torch.load(ROOT/'weight/gvhmr_siga24_release.ckpt',map_location='cpu',weights_only=False)['state_dict']
            missing,unexpected=model.load_state_dict(ck,strict=False)
            named=set(dict(model.named_parameters()))
            assert not set(missing)&named,('missing learned weights',missing)
            record['gvhmr_load']=dict(missing=list(missing),unexpected=list(unexpected),missing_learned_parameters=[])
            return model.eval().requires_grad_(False).cuda()
        model=stage('gvhmr_load',load_gvhmr)
        K=torch.tensor(m['K'],dtype=torch.float32).repeat(L,1,1)
        ang=compute_cam_angvel(torch.eye(3).repeat(L,1,1))
        assert torch.allclose(ang,torch.tensor([1.,0.,0.,0.,1.,0.]).expand(L,-1))
        data=dict(length=torch.tensor(L),obs=normalize_kp2d(kp,bb),bbx_xys=bb,K_fullimg=K,cam_angvel=ang,f_imgseq=feat)
        batch={k:v[None].cuda() for k,v in data.items()};torch.save(cpu(batch),out/'network_input.pt')
        predictions={}
        with torch.no_grad():
            for name,pp in [('raw',False),('official_postproc',True)]:
                result=stage('gvhmr_'+name,lambda:model.pipeline.forward(batch,train=False,postproc=pp,static_cam=True))
                predictions[name]=cpu(result)
                torch.save(predictions[name],out/(name+'_outputs.pt'))
            # Export actual SMPL-X vertices, COCO17 and body22. No SMPL conversion for meshes.
            smplx=make_smplx('supermotion').eval().requires_grad_(False).cuda()
            c2w=np.array(m['c2w'],np.float64); k=np.array(m['K'],np.float64)
            for name,pred in predictions.items():
                params={key:v[0].cuda() for key,v in pred['pred_smpl_params_incam'].items()}
                def geometry():
                    small,co=model.pipeline.endecoder.smplx_model(**{key:v[None] for key,v in params.items()})
                    body=model.pipeline.endecoder.fk_v2(**{key:v[None] for key,v in params.items()})
                    verts=[]
                    for i in range(0,L,16): verts.append(smplx(**{key:v[i:i+16] for key,v in params.items()}).vertices.detach().cpu())
                    return torch.cat(verts).numpy(),co[0].cpu().numpy(),body[0].cpu().numpy(),small[0].cpu().numpy()
                verts,co,body,small=stage('geometry_'+name,geometry)
                arrays=dict(vertices_camera_m=verts,joints_coco17_camera_m=co,joints_body22_camera_m=body,vertices437_camera_m=small,faces=smplx.faces,
                            K=k,c2w=c2w,timestamp_seconds=np.array(m['timestamp_seconds']),input_frame_index=np.arange(L),video_frame_index=m['frame_indices'],vitpose=kp.numpy())
                for key,x in [('vertices',verts),('joints_coco17',co),('joints_body22',body)]:
                    arrays[key+'_world_m']=x@c2w[:3,:3].T+c2w[:3,3]
                    h=x@k.T;arrays[key+'_uv']=h[...,:2]/h[...,2:]
                arrays.update({key:v.cpu().numpy() for key,v in params.items()})
                assert all(np.isfinite(x).all() for x in arrays.values())
                np.savez_compressed(out/(name+'_geometry.npz'),**arrays)
        record.update(status='completed',frames=L,weights_frozen=True,parameter_updates=0,world_transform='incam points @ known c2w rotation.T + translation; no fitted alignment, no GV global ground shift',time_limitation='Network uses frame indices, actual irregular timestamps retained but not consumed',primary='raw incam',secondary='official postproc with static_cam=True',input_image_resize='No video reencoding or img_ds downsample; official 256-square body crop then 192-wide feature/pose crop',fx_fy_relative_difference=abs(m['K'][0][0]/m['K'][1][1]-1),source_hashes={str(p.relative_to(REPO)):sha(p) for p in REPO.rglob('*.py')},outputs={p.name:sha(p) for p in out.glob('*') if p.suffix in ['.pt','.npz','.yaml']})
        (out/'executed_inference.py').write_bytes(Path(__file__).read_bytes())
    except BaseException as e:
        record.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        record.update(wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        save()
        print(json.dumps({k:v for k,v in record.items() if k not in ['source_hashes','outputs']},indent=2),flush=True)

if __name__=='__main__':main()
