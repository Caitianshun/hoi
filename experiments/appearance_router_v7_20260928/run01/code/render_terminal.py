"""One complete M1 terminal rendering; no new D1 probes."""
from common import *
from render_utils import load_model,CalibratedCamera
from diagnostic_render import make_renderer
from budget import CallBudget
import torch,numpy as np,cv2,time
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';torch.set_num_threads(4)
    freeze=read(RUN/'protocol/finals.json');assert freeze['status']=='sole_terminal_frozen'
    for x in freeze['assets']:assert sha(x['path'])==x['sha256']
    rd=Path(freeze['run_directory']);r=read(rd/'run.json');out=RUN/'evaluation/M1';assert not out.exists();out.mkdir(parents=True)
    model,(dataset,hidden,opt,pipe),state=load_model(r['checkpoint'],rd/'effective_config.json');del state
    render=make_renderer(read(rd/'effective_config.json')['scale_bound']);budget=CallBudget('terminal_evaluation');budget.mode='terminal_evaluation'
    bg=torch.tensor([1,1,1] if dataset.white_background else [0,0,0],dtype=torch.float32,device='cuda')
    selected=set(read(RUN/'protocol/fixed_examples.json')['training_frame_ids']);rows=[];started=time.monotonic()
    with torch.no_grad():
        for split,name in [('retained','evaluation_manifest.json'),('train','manifest.json')]:
            for i,f in enumerate(read(OLD/'inputs/hos_backpack'/name)['frames']):
                camera=CalibratedCamera(f,i,False);pkg=render(camera,model,pipe,bg,stage='fine')
                rgb=pkg['render'].permute(1,2,0).cpu().numpy();assert rgb.dtype==np.float32 and np.isfinite(rgb).all()
                path=out/split/(f['frame_id']+'.npz');path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,rgb=rgb,depth=pkg['depth'].squeeze().cpu().numpy())
                row=dict(run='M1',split=split,frame_id=f['frame_id'],render=identity(path),source_frame=f)
                if split=='retained' or f['frame_id'] in selected:
                    png=path.with_suffix('.png');cv2.imwrite(str(png),np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1]);row['display_png']=identity(png)
                rows.append(row)
                if (i+1)%32==0:print(split,i+1,flush=True)
    save_json(out/'manifest.json',dict(status='completed',rows=rows,seconds=time.monotonic()-started,checkpoint=identity(r['checkpoint']),freeze=identity(RUN/'protocol/finals.json'),RGB_GT_read=False,optimization_steps=0,new_N_eff='not_computed_per_V7'))
if __name__=='__main__':run()
