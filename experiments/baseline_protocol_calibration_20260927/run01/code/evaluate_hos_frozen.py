#!/usr/bin/env python3
"""CPU unified HOS metrics from frozen outputs; native H0 and H1 separate tables."""
import argparse,csv,importlib.util,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from adapter_4dgs import ROOT,RUN,sha,save_json
from evaluate_frozen import evaluate_image


def identity(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)


def csvwrite(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(freeze,output):
    frozen=json.loads(freeze.read_text());assert frozen['status']=='all_states_frozen' and frozen['scope']=='hos_backpack'
    for a in frozen['assets']:assert sha(a['path'])==a['sha256'],a['path']
    hm=json.loads((RUN/'evaluation/H1/manifest.json').read_text());assert hm['status']=='completed' and hm['all_finals_freeze']['sha256']==sha(freeze)
    assert hm['frame_count']==284 and len(hm['frames'])==284
    H0={};h0sources=[]
    for tag,count in [('H0_native_first',1),('H0_native_remaining',15)]:
        base=RUN/'runs'/tag;c=json.loads((base/'completion.json').read_text());assert c['status']=='completed' and c['frames']==count
        h0sources.append(identity(base/'completion.json'))
        for fid in c['identity']['render_ids']:
            assert fid not in H0;H0[fid]=base/(fid+'.npz')
    train=json.loads((RUN/'inputs/hos_backpack/manifest.json').read_text());ev=json.loads((RUN/'inputs/hos_backpack/evaluation_manifest.json').read_text())
    assert set(H0)=={r['frame_id'] for r in ev['frames']}
    index={(r['group'],r['frame_id']):r for r in hm['frames']};assert len(index)==284
    assert output.exists() is False,f'Refuse overwrite {output}';output.mkdir(parents=True)
    # Hash all arrays before decoding any prediction or GT; this records complete
    # immutable outputs, distinct from the earlier terminal-model freeze.
    snapshots=[identity(p) for p in H0.values()]
    for row in hm['frames']:
        assert sha(row['render']['path'])==row['render']['sha256'];snapshots.append(row['render'])
    save_json(output/'rendered_states_frozen.json',dict(status='all_rendered_states_frozen',created_unix=time.time(),assets=snapshots,terminal_freeze=identity(freeze)))
    start=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(4)
    legacy=ROOT/'experiments/aux_ref_object_reconstruction_20260924/run01/code/evaluate_aux.py'
    spec=importlib.util.spec_from_file_location('hos_legacy_lpips',legacy);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    lpips,lpips_info=module.existing_lpips('cpu');save_json(output/'lpips.json',lpips_info)
    rows=[];figures=[];raw_ranges=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    for group,frames in [('test',ev['frames']),('input_fit',train['frames'])]:
        for i,f in enumerate(frames):
            fid=f['frame_id'];gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255.0
            # Published soft masks combine all foreground. Only diagnostic regions,
            # not an H/O decomposition or source of H1 test-time supervision.
            mask=cv2.imread(f['mask_path'],cv2.IMREAD_GRAYSCALE)>=128;labels=mask.astype(np.uint8)
            methods=['H0','H1'] if group=='test' else ['H1'];predictions={};metrics={}
            for method in methods:
                p=H0[fid] if method=='H0' else Path(index[group,fid]['render']['path'])
                with np.load(p) as d:
                    pred=d['rgb'].copy()
                    if method=='H0':assert np.max(np.abs(d['gt']-gt))<1e-6
                result,ranges=evaluate_image(pred,gt,labels,lpips);predictions[method]=pred;metrics[method]=result['full']
                for reg in ['full','foreground','background']:
                    rows.append(dict(method=method,group=group,frame_id=fid,source_frame_index=int(fid),region=reg,**result[reg],render_path=str(p),render_sha256=sha(p),GT_sha256=sha(f['image_path'])))
                raw_ranges.append(dict(method=method,group=group,frame_id=fid,**ranges))
            if group=='test':
                w=640;h=round(gt.shape[0]*w/gt.shape[1]);card=Image.new('RGB',(w*3,h+66),'white');draw=ImageDraw.Draw(card)
                for col,name in enumerate(['GT','H0','H1']):
                    arr=gt if name=='GT' else predictions[name]
                    pic=Image.fromarray(np.rint(np.clip(arr,0,1)*255).astype(np.uint8)).resize((w,h),Image.Resampling.LANCZOS)
                    card.paste(pic,(col*w,66));draw.text((col*w+8,7),f'Backpack {fid} | {name}',font=font,fill='black')
                    if name!='GT':draw.text((col*w+8,32),f"PSNR {metrics[name]['psnr_db']:.3f} | SSIM {metrics[name]['ssim']:.3f}",font=font,fill='black')
                p=output/'figures'/f'{fid}.png';p.parent.mkdir(exist_ok=True);card.save(p);figures.append(identity(p))
            if (i+1)%16==0:
                save_json(output/'progress.json',dict(group=group,frames=i+1,metric_rows=len(rows),seconds=time.monotonic()-start));print(group,i+1,'/',len(frames),flush=True)
    csvwrite(output/'metrics_per_frame.csv',[r for r in rows if r['group']=='test']);csvwrite(output/'input_fit.csv',[r for r in rows if r['group']=='input_fit'])
    summaries={}
    for method,group in [('H0','test'),('H1','test'),('H1','input_fit')]:
        summary={}
        for reg in ['full','foreground','background']:
            rs=[r for r in rows if r['method']==method and r['group']==group and r['region']==reg];assert len(rs)==(16 if group=='test' else 268)
            mse=sum(r['sse_rgb_mean'] for r in rs)/sum(r['pixels'] for r in rs)
            summary[reg]=dict(frames=len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in ['psnr_db','ssim','lpips_spatial_mean']},pooled_mse=mse,pooled_psnr_db=float(-10*np.log10(max(mse,1e-12))))
        summaries[f'{method}_{group}']=summary
    # Deliberately no H1-minus-H0 paired delta: native H0 has stronger preprocessing
    # and historical stage split uncertainty, although image IDs align.
    summary=dict(status='completed',results=summaries,H0_identity='official released complete200000 checkpoint; native human/camera/state preprocessing; historical exact frame protocol absent',H1_identity='one official Wu4DGS run; train-RGB-only triangulation and color; published all-video camera source separately allowed',direct_fair_H0_H1_delta=None,
        split='complete-stage16 test IDs; distinct scene-stage split; no scene substitution',
        metric_definition=dict(PSNR='clipped floating RGB MSE; frame mean and pooled shown',SSIM='fullimage7x7 map; region mean; data_range1',LPIPS='historical cached AlexNetv0.1 spatial-map mean, same unified metric as BEHAVE; not native H0VGG scalar',regions='published mask>=128 gives combined foreground and complement background; no individual human/object labels',input_fit='all268 train frames; reconstruction not generalization'),
        lpips=lpips_info,figure_count=len(figures),figures=figures,optimization_steps=0,GPU_used=False,wall_seconds=time.monotonic()-start,sources=h0sources+[identity(freeze),identity(RUN/'evaluation/H1/manifest.json')])
    save_json(output/'raw_ranges.json',raw_ranges);save_json(output/'summary.json',summary);print(json.dumps({k:summary[k] for k in ['status','results','wall_seconds']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,default=RUN/'evaluation/hos_comparison');a=p.parse_args();run(a.freeze,a.output)
