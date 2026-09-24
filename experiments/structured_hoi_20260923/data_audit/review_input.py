"""Input-camera RGB review only; never opens fitted labels or poses."""
import argparse, json, hashlib
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sheet(tiles, path, cols=4):
    while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
    image=np.concatenate([np.concatenate(tiles[i:i+cols],1) for i in range(0,len(tiles),cols)],0)
    assert cv2.imwrite(str(path),image)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['behave','hodome'])
    ap.add_argument('--start',type=float);ap.add_argument('--end',type=float);ap.add_argument('--step',type=float,default=2)
    a=ap.parse_args();out=ROOT/'rgb_review';out.mkdir(exist_ok=True)
    if a.mode=='behave':
        video=ROOT/'sources/Date01_Sub01_chairwood_lift.0.color.mp4'
        tp=ROOT/'sources/Date01_Sub01_chairwood_lift.0.time.json'
        ts=np.asarray(json.loads(tp.read_text())['color'],dtype=np.float64)/1e6
        start=ts[0] if a.start is None else a.start;end=ts[-1] if a.end is None else a.end
        requested=np.arange(start,end+1e-5,a.step);idx=np.unique(np.abs(ts[:,None]-requested).argmin(axis=0))
        cap=cv2.VideoCapture(str(video));assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(ts)
        tiles=[];rows=[]
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,bgr=cap.read();assert ok
            tile=cv2.resize(bgr,(512,384),interpolation=cv2.INTER_AREA)
            cv2.putText(tile,f'{ts[i]:.3f}s / index {i}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            tiles.append(tile);rows.append(dict(video_index=int(i),timestamp_seconds=float(ts[i])))
        cap.release();stem=f'chairwood_cam0_{start:.1f}_{end:.1f}_step{a.step:.1f}'
        sheet(tiles,out/f'{stem}.jpg')
        (out/f'{stem}.json').write_text(json.dumps(dict(source=str(video),source_sha256=sha(video),timestamps_sha256=sha(tp),camera=0,rows=rows,source_time_extent=[float(ts[0]),float(ts[-1])],source_frames=len(ts),selection='RGB only; no evaluation assets'),indent=2))
        print(out/f'{stem}.jpg')
    else:
        base=Path('/home/cai_tianshun/Project/4dsr/data/hoi_v1_independent')
        for seq,start,end in [('subject05_box',224,479),('subject06_trashcan',232,487),('subject09_chair',2112,2367),('subject10_book',3000,3255)]:
            tiles=[];rows=[]
            for i in np.linspace(start,end,16).round().astype(int):
                p=base/seq/'cam14/hr'/f'{i:06d}.png';bgr=cv2.imread(str(p));assert bgr is not None
                tile=cv2.resize(bgr,(480,270),interpolation=cv2.INTER_AREA)
                cv2.putText(tile,f'{seq} frame {i}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),2)
                tiles.append(tile);rows.append(dict(frame_index=int(i),path=str(p),sha256=sha(p)))
            sheet(tiles,out/f'{seq}_cam14.jpg')
            (out/f'{seq}_cam14.json').write_text(json.dumps(dict(camera=14,rows=rows,selection='Reuse predetermined package window; current review uses RGB only, no fitted masks/poses or holdout.'),indent=2))
            print(out/f'{seq}_cam14.jpg')

if __name__=='__main__':main()
