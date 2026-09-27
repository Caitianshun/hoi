"""Render both new terminal states only after their joint freeze."""
from common import *
from render_utils import load_model,CalibratedCamera
import os,torch,numpy as np,cv2,time
from gaussian_renderer import render
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';torch.set_num_threads(4);freeze=read(RUN/'protocol/finals.json');assert freeze['status']=='both_new_terminal_states_frozen'
    for x in freeze['assets']:assert sha(x['path'])==x['sha256'],x['path']
    train=read(OLD/'inputs/hos_backpack/manifest.json');retained=read(OLD/'inputs/hos_backpack/evaluation_manifest.json');examples=read(RUN/'protocol/fixed_examples.json');selected=set(examples['training_frame_ids'])
    for branch in ['W_fine','W_all']:
        out=RUN/'evaluation'/branch;assert not out.exists();out.mkdir()
        rd=RUN/'runs'/branch;r=read(rd/'run.json');model,(dataset,hidden,opt,pipe),state=load_model(r['checkpoint'],rd/'effective_config.json');del state
        bg=torch.tensor([1,1,1] if dataset.white_background else [0,0,0],dtype=torch.float32,device='cuda');rows=[];started=time.monotonic()
        with torch.no_grad():
            for split,frames in [('retained',retained['frames']),('train',train['frames'])]:
                for i,f in enumerate(frames):
                    camera=CalibratedCamera(f,i,False);pkg=render(camera,model,pipe,bg,stage='fine');rgb=pkg['render'].permute(1,2,0).cpu().numpy();assert np.isfinite(rgb).all()
                    path=out/split/(f['frame_id']+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,rgb=rgb,depth=pkg['depth'].squeeze().cpu().numpy())
                    row=dict(branch=branch,split=split,frame_id=f['frame_id'],render=identity(path),source_frame=f)
                    if split=='retained' or f['frame_id'] in selected:
                        png=path.with_suffix('.png');cv2.imwrite(str(png),np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1]);row['display_png']=identity(png)
                    rows.append(row)
                    if (i+1)%32==0:print(branch,split,i+1,flush=True)
        save_json(out/'manifest.json',dict(status='completed',rows=rows,seconds=time.monotonic()-started,checkpoint=identity(r['checkpoint']),freeze=identity(RUN/'protocol/finals.json'),RGB_GT_read=False,optimization_steps=0))
        del model;torch.cuda.empty_cache()
if __name__=='__main__':run()
