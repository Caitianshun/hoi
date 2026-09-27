"""CPU metrics and compact fixed illustrations; all tables derive from files."""
from common import *
from evaluate_frozen import evaluate_image,InputRegions
import argparse,csv,importlib.util,time,cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
METRICS=['psnr_db','ssim','lpips_spatial_mean']
def csvwrite(path,rows):
    assert rows;keys=list(dict.fromkeys(k for r in rows for k in r));path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def lpips_cpu():
    path=ROOT/'experiments/aux_ref_object_reconstruction_20260924/run01/code/evaluate_aux.py';spec=importlib.util.spec_from_file_location('v4_lpips',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    net,info=m.existing_lpips('cpu');assert info['available'],'Local LPIPS weights must be available';return net,info
def figure(path,arrays,labels,width=420):
    assert len(arrays)==len(labels);height=max(round(x.shape[0]*width/x.shape[1]) for x in arrays);card=Image.new('RGB',(len(arrays)*width,height+36),'white');draw=ImageDraw.Draw(card)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for i,(x,label) in enumerate(zip(arrays,labels)):
        if x.ndim==2:x=np.repeat(x[...,None],3,axis=2)
        pic=Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8));pic.thumbnail((width,height),Image.Resampling.LANCZOS);card.paste(pic,(i*width,36));draw.text((i*width+6,8),label,font=font,fill='black')
    path.parent.mkdir(exist_ok=True,parents=True);card.save(path,quality=90,subsampling=0)
    return identity(path)
def mean_summary(rows):
    valid=[r for r in rows if r['pixels']>0 and r['sse_rgb_mean'] is not None];pixels=sum(r['pixels'] for r in valid);sse=sum(r['sse_rgb_mean'] for r in valid);mse=sse/pixels if pixels else None
    return dict(frames=len(rows),valid_frames=len(valid),pixels=pixels,sse_rgb_mean=sse if valid else None,pooled_mse=mse,pooled_psnr_db=-10*np.log10(max(mse,1e-12)) if mse is not None else None,
        raw_pooled_mse=sum(r['raw_mse']*r['pixels'] for r in valid)/pixels if pixels else None,
        raw_psnr_db=float(np.mean([r['raw_psnr_db'] for r in valid])) if valid else None,
        **{k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in METRICS})
def heat(x):return cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(x,0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)/255.
def hos(net):
    freeze=read(RUN/'protocol/finals.json');assert freeze['status'] in ['both_new_terminal_states_frozen','all_authorized_attempts_closed']
    for a in freeze['assets']:assert sha(a['path'])==a['sha256']
    start=time.monotonic();examples=read(RUN/'protocol/fixed_examples.json');crop={r['frame_id']:r['crop_bounds_xyxy'] for r in examples['hos_crops']};new={}
    for branch in freeze['runs']:
        m=read(RUN/'evaluation'/branch/'manifest.json');assert m['status']=='completed' and m['freeze']['sha256']==sha(RUN/'protocol/finals.json')
        new[branch]={(r['split'],r['frame_id']):r for r in m['rows']};assert len(new[branch])==284
    old=read(OLD/'evaluation/H1/manifest.json');old_index={('train' if r['group']=='input_fit' else 'retained',r['frame_id']):r for r in old['frames']};assert len(old_index)==284
    # Reuse the verified baseline numbers rather than recomputing the old network.
    baseline={}
    for name in ['metrics_per_frame.csv','input_fit.csv']:
        with (OLD/'evaluation/hos_comparison'/name).open() as f:
            for row in csv.DictReader(f):
                if row['method']!='H1':continue
                key=('train' if row['group']=='input_fit' else 'retained',row['frame_id'],row['region']);baseline[key]={k:(None if row[k]=='' else int(row[k]) if k=='pixels' else float(row[k])) for k in ['pixels','pixel_fraction','mse','sse_rgb_mean','psnr_db','ssim','lpips_spatial_mean','full_error_share']}
    rows=[];figs=[];indices=[]
    for split,name in [('retained','evaluation_manifest.json'),('train','manifest.json')]:
        frames=read(OLD/'inputs/hos_backpack'/name)['frames']
        for i,frame in enumerate(frames):
            fid=frame['frame_id'];gt=cv2.imread(frame['image_path'])[...,::-1].astype(np.float64)/255.;labels=(cv2.imread(frame['mask_path'],0)>=128).astype(np.uint8);show=split=='retained' or fid in examples['training_frame_ids'];preds={}
            for branch in ['H1','B_U','B_F']:
                available=branch=='H1' or branch in new
                asset=old_index[split,fid]['render'] if branch=='H1' else new[branch][split,fid]['render'] if available else dict(path=None,sha256=None)
                if available:assert sha(asset['path'])==asset['sha256']
                if branch=='H1':
                    result={reg:baseline[split,fid,reg] for reg in ['full','foreground','background']}
                    pred=np.load(asset['path'])['rgb']
                    if show:preds[branch]=pred
                elif available:
                    pred=np.load(asset['path'])['rgb'];result,_=evaluate_image(pred,gt,labels,net)
                    if show:preds[branch]=pred
                else:
                    result={reg:{**baseline[split,fid,reg],**{k:None for k in ['mse','sse_rgb_mean','psnr_db','ssim','lpips_spatial_mean','full_error_share']}} for reg in ['full','foreground','background']}
                    if show:preds[branch]=np.ones_like(gt)*.85
                for region in ['full','foreground','background']:
                    mask=np.ones(labels.shape,bool) if region=='full' else labels==1 if region=='foreground' else labels==0
                    raw_mse=float(np.square(pred[mask]-gt[mask]).mean()) if available and mask.any() else None
                    result[region]['raw_mse']=raw_mse;result[region]['raw_psnr_db']=None if raw_mse is None else float(-10*np.log10(max(raw_mse,1e-12)))
                    rows.append(dict(run=branch,status='completed' if available else freeze['attempts'][branch]['status'],NA_reason=None if available else freeze['attempts'][branch]['status']+'; no replacement model',split=split,frame_id=fid,source_frame_index=int(fid),time=frame['time'],time_units='normalized nominal source frame index; original index is not measured seconds',extrapolation=frame.get('extrapolation',int(fid)==0),region=region,region_source='published mask >=128 dynamic-subject foreground; includes carried bag but may exclude stationary bag',
                        **result[region],raw_clipped_policy='clip raw float RGB to [0,1] before metrics',raw_render_path=asset['path'],raw_render_sha256=asset['sha256']))
                if available:indices.append(dict(run=branch,split=split,frame_id=fid,**asset))
            if show and new:
                path=RUN/'evaluation/figures'/f'{split}_{fid}_full.jpg';f=figure(path,[gt,*[preds[b] for b in ['H1','B_U','B_F']]],[fid+' GT','H1 uniform','B-U uniform' if 'B_U' in new else 'B-U NA failed','B-F balanced' if 'B_F' in new else 'B-F NA failed'],440);figs.append(dict(**f,split=split,frame_id=fid,kind='full',branch_order=['GT','H1','B_U','B_F'],raw_or_clipped='clipped [0,1]'))
                if split=='retained':
                    x0,y0,x1,y1=crop[fid];path=path.with_name(path.stem.replace('_full','_foreground')+'.jpg');f=figure(path,[x[y0:y1,x0:x1] for x in [gt,*[preds[b] for b in ['H1','B_U','B_F']]]],[fid+' GT crop','H1 crop','B-U crop' if 'B_U' in new else 'B-U NA failed','B-F crop' if 'B_F' in new else 'B-F NA failed'],280);figs.append(dict(**f,split=split,frame_id=fid,kind='foreground_crop',crop_bounds_xyxy=crop[fid],branch_order=['GT','H1','B_U','B_F'],raw_or_clipped='clipped [0,1]'))
            if (i+1)%32==0:print(split,i+1,flush=True)
    csvwrite(RUN/'metrics_per_frame.csv',rows);save_json(RUN/'protocol/raw_render_index.json',dict(assets=indices))
    summaries={}
    for branch in ['H1','B_U','B_F']:
        summaries[branch]={}
        for split in ['train','retained','retained_without_00000']:
            rs=[r for r in rows if r['run']==branch and r['split']==('retained' if split.startswith('retained') else 'train') and (split!='retained_without_00000' or r['frame_id']!='00000')]
            summaries[branch][split]={reg:mean_summary([r for r in rs if r['region']==reg]) for reg in ['full','foreground','background']}
    index={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};assert len(index)==len(rows)==3*284*3
    differences=[];paired={}
    for left,right in [('B_U','H1'),('B_F','H1'),('B_F','B_U')]:
        pair=left+'-'+right;paired[pair]={}
        for split in ['train','retained']:
            paired[pair][split]={}
            for region in ['full','foreground','background']:
                rs=[r for r in rows if r['run']==left and r['split']==split and r['region']==region];dd=[]
                for a in rs:
                    b=index[right,split,a['frame_id'],region];assert a['pixels']==b['pixels'];delta={k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in METRICS}
                    item=dict(comparison=pair,split=split,frame_id=a['frame_id'],region=region,**delta);differences.append(item);dd.append(item)
                paired[pair][split][region]={k:dict(mean=float(np.mean([d[k] for d in dd if d[k] is not None])),median=float(np.median([d[k] for d in dd if d[k] is not None])),valid_frames=sum(d[k] is not None for d in dd)) if any(d[k] is not None for d in dd) else dict(mean=None,median=None,valid_frames=0) for k in METRICS}
        paired[pair]['retained_without_00000']={}
        for region in ['full','foreground','background']:
            dd=[d for d in differences if d['comparison']==pair and d['split']=='retained' and d['frame_id']!='00000' and d['region']==region]
            paired[pair]['retained_without_00000'][region]={k:dict(mean=float(np.mean([d[k] for d in dd if d[k] is not None])),median=float(np.median([d[k] for d in dd if d[k] is not None])),valid_frames=sum(d[k] is not None for d in dd)) if any(d[k] is not None for d in dd) else dict(mean=None,median=None,valid_frames=0) for k in METRICS}
    csvwrite(RUN/'paired_differences.csv',differences);rules=read(RUN/'configs/v5.json')['quality'];gates={}
    if all(b in new for b in ['B_U','B_F']):
        p=paired['B_F-B_U']['retained'];fg=p['foreground'];bg=p['background']
        tests=dict(FG_PSNR_mean=fg['psnr_db']['mean']>=rules['FG_PSNR_mean_min'],FG_PSNR_median=fg['psnr_db']['median']>rules['FG_PSNR_median_strict_min'],FG_LPIPS=fg['lpips_spatial_mean']['mean']<=rules['FG_LPIPS_mean_max'],BG_PSNR=bg['psnr_db']['mean']>=rules['BG_PSNR_mean_min'])
        gates['B_F-B_U']=dict(numerical_rules=tests,numerical_pass=all(tests.values()),visual_quality_gate='pending fixed-frame review')
    else:gates['B_F-B_U']=dict(numerical_rules=None,numerical_pass=None,visual_quality_gate='NA missing terminal')
    save_json(RUN/'evaluation/summary.json',dict(status='completed',summaries=summaries,paired=paired,gates=gates,rows=len(rows),paired_rows=len(differences),figures=figs,seconds=time.monotonic()-start,
        metrics='per-pixel RGB-channel-mean SSE; MSE=SSE/pixels; original 7x7 SSIM and AlexNet0.1 spatial LPIPS with cross-boundary receptive fields',H1_metrics_reused_verified_V3=True,individual_human_bag_metrics=None,individual_region_NA_reason='No independent instance masks',retained_set='16 already-inspected development frames; separate15 supplemental, no replacement of16 main table'))
    save_json(RUN/'figure_manifest.json',dict(protocol=identity(RUN/'protocol/fixed_examples.json'),hos=figs,appearance=[],input_support_reference=identity(V4/'initialization_support.json')))
def run():
    torch.set_num_threads(4);cv2.setNumThreads(4);net,info=lpips_cpu();save_json(RUN/'protocol/evaluation_lpips.json',info);hos(net)
if __name__=='__main__':run()
