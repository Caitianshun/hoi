"""Render both frozen independent terminals before quality assessment."""
from common import *
from render_utils import load_model,CalibratedCamera
from hoi_modules.projected_motion import make_renderer
import torch,numpy as np,cv2,time
B=RUN/'tennis'
def run():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(4)
    freeze=read(B/'protocol/finals.json');assert freeze['status']=='both_terminals_frozen'
    for x in freeze['assets']:assert sha(x['path'])==x['sha256']
    selected=set(read(B/'protocol/fixed_examples.json')['training_frame_ids'])
    for arm in ['B_U','B_Q']:
        rd=B/'runs'/arm;r=read(rd/'run.json');out=B/'evaluation'/arm;assert not out.exists();out.mkdir(parents=True)
        model,(ds,h,opt,pipe),state=load_model(r['checkpoint'],rd/'effective_config.json');del state
        render=make_renderer(read(rd/'effective_config.json')['scale_bound']);bg=torch.tensor([1,1,1] if ds.white_background else [0,0,0],device='cuda',dtype=torch.float32);rows=[];start=time.monotonic()
        with torch.no_grad():
            for split,name in [('retained','evaluation_manifest.json'),('train','manifest.json')]:
                for i,f in enumerate(read(B/'inputs/hos_tennis'/name)['frames']):
                    pkg=render(CalibratedCamera(f,i,False),model,pipe,bg,stage='fine');rgb=pkg['render'].permute(1,2,0).cpu().numpy();assert rgb.dtype==np.float32 and np.isfinite(rgb).all()
                    path=out/split/(f['frame_id']+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,rgb=rgb,depth=pkg['depth'].squeeze().cpu().numpy())
                    row=dict(run=arm,split=split,frame_id=f['frame_id'],render=identity(path),source_frame=f)
                    if split=='retained' or f['frame_id'] in selected:
                        png=path.with_suffix('.png');cv2.imwrite(str(png),np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1]);row['display_png']=identity(png)
                    rows.append(row)
        save_json(out/'manifest.json',dict(status='completed',rows=rows,seconds=time.monotonic()-start,checkpoint=identity(r['checkpoint']),freeze=identity(B/'protocol/finals.json'),GT_read=False));del model;torch.cuda.empty_cache()
if __name__=='__main__':run()
