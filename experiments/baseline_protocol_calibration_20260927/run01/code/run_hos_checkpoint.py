#!/usr/bin/env python3
"""Inference-only harness for the locked official HOSNeRF Backpack checkpoint.

The official test_metrics rendering body is retained. Only unused dataloaders,
path placeholders and metric/output hooks are adapted. No optimizer is created.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import torch


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(2**20),b''):h.update(c)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--official',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--limit',type=int,default=0);p.add_argument('--skip',type=int,default=0);p.add_argument('--chunk',type=int,default=2048)
    a=p.parse_args(); a.official=a.official.resolve();a.data=a.data.resolve();a.checkpoint=a.checkpoint.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(12345);np.random.seed(12345)
    source=a.official/'3rd_Complete_HOSNeRF';os.chdir(source);sys.path.insert(0,str(source))
    from third_parties.yacs import CfgNode as CN
    cfg=CN();cfg.resume=False;cfg.eval_iter=10000000;cfg.render_folder_name='';cfg.ignore_non_rigid_motions=False;cfg.render_skip=1;cfg.render_frames=100;cfg.num_workers=1
    cfg.merge_from_file('configs/default.yaml');cfg.merge_from_file('configs/human_nerf/wild/monocular/adventure.yaml');cfg.basedir=str(a.data);cfg.chunk_bkg=a.chunk
    from core.data.dataset_args import DatasetArgs
    original_get=DatasetArgs.get
    def local_get(c,name):
        v=original_get(c,name);v['dataset_path']=str(a.data);return v
    DatasetArgs.get=staticmethod(local_get)
    import src.model.mipnerf360.model as official_model
    import gin
    gin.bind_parameter('MipNeRF360.opaque_background',True)
    original_loader=official_model.create_dataloader
    official_model.create_dataloader=lambda c,data_type:original_loader(c,data_type) if data_type=='test' else None
    # The complete checkpoint embeds the VGG trunk as well as LPIPS weights;
    # avoiding an unnecessary ImageNet download does not change loaded weights.
    original_lpips=official_model.LPIPS
    official_model.LPIPS=lambda **kw:original_lpips(pnet_rand=True,**kw)
    m=official_model.LitMipNeRF360(cfg=cfg,basedir=str(a.data))
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    result=m.load_state_dict(ckpt['state_dict'],strict=True)
    m.logdir=str(a.output);m.near_bkg=.1;m.far_bkg=1e6;m._trainer=SimpleNamespace(global_step=int(ckpt['global_step']))
    ids=list(m.test_dataloader.dataset.framelist)
    full_ids=list(ids)
    assert 0 <= a.skip < len(ids)
    if a.limit or a.skip:
        stop=min(a.skip+a.limit,len(ids)) if a.limit else len(ids)
        m.test_dataloader=torch.utils.data.DataLoader(torch.utils.data.Subset(m.test_dataloader.dataset,range(a.skip,stop)),batch_size=1,shuffle=False,num_workers=0)
        ids=ids[a.skip:stop]
    identity=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),step=ckpt['global_step'],strict_load=True,
        missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,test_ids=full_ids,render_ids=ids,
        resolution=[1277,718],chunk_bkg=a.chunk,resize_scale=cfg.resize_img_scale,
        historical_split_identity='not embedded in released checkpoint; native loader compatibility verified only',
        official_model_path=str(source/'src/model/mipnerf360/model.py'),official_model_sha256=sha(source/'src/model/mipnerf360/model.py'),
        inference_adjustments=['local dataset path','skip unused progress/movement/freeview/tpose loaders','random VGG constructor followed by strict full checkpoint load','ray chunk only','float RGB/metrics capture'],
        renderer_method='official LitMipNeRF360.test_metrics; sampling/compositing unchanged',
        ssim_note='official code applies SSIM to flattened N×3; harness records conventional H×W×3 SSIM with data_range=1 for comparable evaluation; not an exact paper metric reproduction',
        torch_version=torch.__version__)
    (a.output/'load_identity.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps(identity,indent=2),flush=True)
    if a.check_only:return
    if not torch.cuda.is_available():raise RuntimeError('GPU required by official renderer; use --check-only for CPU validation')
    torch.cuda.reset_peak_memory_stats();m.cuda();m.eval()
    native=textwrap.dedent(inspect.getsource(official_model.LitMipNeRF360.test_metrics))
    native=native.replace('ssim = skimage.metrics.structural_similarity(rendered, truth, channel_axis=True)', 'ssim = self.capture_float(frame_name, rendered, truth, int(height), int(width))')
    native=native.replace('lpipss.append(lpips)','lpipss.append(lpips)\n        self.capture_scalar(frame_name, psnr, ssim, lpips)')
    namespace={};exec(native,vars(official_model),namespace)
    rows=[]
    def capture_float(self,frame_id,pred,gt,h,w):
        from skimage.metrics import structural_similarity
        rgb=pred.reshape(h,w,3);target=gt.reshape(h,w,3)
        np.savez_compressed(a.output/(frame_id+'.npz'),rgb=rgb,gt=target)
        return structural_similarity(rgb,target,data_range=1.0,channel_axis=-1)
    def capture_scalar(self,frame_id,psnr,ssim,lpips):
        row=dict(frame_id=frame_id,PSNR=float(psnr),SSIM=float(ssim),LPIPS=float(lpips));rows.append(row)
        (a.output/'per_frame.json').write_text(json.dumps(rows,indent=2)+'\n');print('H0_FRAME',json.dumps(row),flush=True)
    import types
    m.capture_float=types.MethodType(capture_float,m);m.capture_scalar=types.MethodType(capture_scalar,m);m.test_metrics=types.MethodType(namespace['test_metrics'],m)
    start=time.monotonic()
    with torch.no_grad():m.test_metrics()
    result=dict(status='completed',frames=len(rows),wall_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),mean_metrics={key:float(np.mean([r[key] for r in rows])) for key in ['PSNR','SSIM','LPIPS']},identity=identity)
    (a.output/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
