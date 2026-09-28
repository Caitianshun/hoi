"""Render six after-step terminals only after their common freeze event."""
from common import *
import torch,numpy as np,cv2
from state import restore
from render_scene import make_renderer,rgb_background
from adapter_4dgs import CalibratedCamera


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';torch.set_num_threads(4)
    frozen=read(RUN/'protocol/finals.json');assert frozen['status']=='all_six_terminals_frozen'
    for asset in frozen['assets']:assert sha(asset['path'])==asset['sha256']
    for entry in frozen['runs']:
        scene=entry['scene'];mode=entry['mode'];cfg=config()['scenes'][scene]
        sd=scene_dir(scene);out=sd/'evaluation'/mode;assert not out.exists();out.mkdir(parents=True)
        state=torch.load(entry['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert state['completed_updates']==config()['schedule']['fine_updates']
        model,(ds,h,opt,pipe)=restore(state);render=make_renderer(state['metadata']['scale_bound'],mode)
        bg=rgb_background(ds);selected=set(read(ROOT/cfg['historical_dir']/'protocol/fixed_examples.json')['training_frame_ids'])
        start=time.monotonic();rows=[];torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for split,name in [('retained','evaluation_manifest.json'),('train','manifest.json')]:
                for i,f in enumerate(read(ROOT/cfg['input_dir']/name)['frames']):
                    pkg=render(CalibratedCamera(f,i,False),model,pipe,bg,stage='fine')
                    rgb=pkg['render'].permute(1,2,0).cpu().numpy()
                    assert rgb.dtype==np.float32 and np.isfinite(rgb).all()
                    path=out/split/(f['frame_id']+'.npz');path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path,rgb=rgb,depth=pkg['depth'].squeeze().cpu().numpy())
                    row=dict(scene=scene,run=mode,split=split,frame_id=f['frame_id'],render=identity(path),source_frame=f)
                    if split=='retained' or f['frame_id'] in selected:
                        png=path.with_suffix('.png');cv2.imwrite(str(png),np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1])
                        row['display_png']=identity(png)
                    rows.append(row)
                    if (i+1)%32==0:print(scene,mode,split,i+1,flush=True)
        save_json(out/'manifest.json',dict(status='completed',rows=rows,seconds=time.monotonic()-start,
            checkpoint=entry['checkpoint'],freeze=identity(RUN/'protocol/finals.json'),RGB_GT_read=False,
            optimization_updates=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del model,state;torch.cuda.empty_cache()


if __name__=='__main__':run()
