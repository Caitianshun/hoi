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
        **{k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in METRICS})
def heat(x):return cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(x,0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)/255.
def appearance(net):
    dest=RUN/'diagnostics/appearance/summary.json'
    if dest.exists():assert read(dest)['status']=='completed';return
    start=time.monotonic();m=read(RUN/'diagnostics/appearance/manifest.json');assert m['status']=='completed';rows=[];figs=[];groups={};regions={d:InputRegions(d) for d in ['dev1','dev2']}
    for record in m['records']:
        key=record['dev'],record['group'],record['frame_id'];groups.setdefault(key,{})[record['mode']]=record
    for (dev,group,fid),modes in groups.items():
        frame=modes['native']['source_frame'];gt=cv2.imread(frame['image_path'])[...,::-1].astype(np.float64)/255.;labels,source,description=regions[dev].get(dict(group=group,frame_id=fid,time_seconds=frame['time_seconds'],source_frame=frame));preds={};alphas={}
        for mode,rec in modes.items():
            assert sha(rec['render']['path'])==rec['render']['sha256'];z=np.load(rec['render']['path']);pred=z['rgb'];alpha=z['alpha'];preds[mode]=pred;alphas[mode]=alpha
            result,_=evaluate_image(pred,gt,labels,net)
            for name,mask in [('full',np.ones(labels.shape,bool)),('human',labels==1),('object',labels==2),('background',labels==0)]:
                pixels=int(mask.sum());a=alpha[mask];v=pred[mask];q=quantiles(v)
                row=dict(dev=dev,group=group,frame_id=fid,time=frame['time_seconds'],time_units='BEHAVE nominal capture seconds, exposure synchronization unverified',mode=mode,region=name,region_source=description,
                    **result[name],raw_clipped_policy='metrics clip raw to [0,1]; raw diagnostics separate',raw_PSNR_db=None if not pixels else float(-10*np.log10(max(float(np.square(v-gt[mask]).mean()),1e-12))),
                    raw_below0_channel_fraction=None if not pixels else float((v<0).mean()),raw_above1_channel_fraction=None if not pixels else float((v>1).mean()),
                    raw_any_out_of_range_pixel_fraction=None if not pixels else float(np.any((v<0)|(v>1),axis=1).mean()),
                    raw_p50=None if q is None else q['p50'],raw_p95=None if q is None else q['p95'],raw_p99=None if q is None else q['p99'],raw_max=None if q is None else q['max'],
                    alpha_p05=None if not pixels else float(np.quantile(a,.05)),alpha_p50=None if not pixels else float(np.quantile(a,.5)),alpha_p95=None if not pixels else float(np.quantile(a,.95)),
                    alpha_below005_fraction=None if not pixels else float((a<.05).mean()),alpha_above095_fraction=None if not pixels else float((a>.95).mean()),
                    above1_high_alpha_channel_fraction=None if not pixels else float(((v>1)&(a[:,None]>.95)).mean()),above1_high_alpha_pixel_fraction=None if not pixels else float((np.any(v>1,axis=1)&(a>.95)).mean()),
                    visible_scale_json=json.dumps(rec['visible_scale']),visible_radius_json=json.dumps(rec['visible_radius']),native_old_max_abs=rec['native_old_max_abs'],depth_max_abs_delta=rec['depth_max_abs_delta'],alpha_max_abs_delta=rec['alpha_max_abs_delta'],background_rgb=json.dumps(rec['background_rgb']),background_identity_error=rec['background_compositing_max_abs'])
                rows.append(row)
        ev=read(OLD/'inputs'/f'behave_{dev}'/'evaluation_manifest.json');ids=[f['frame_id'] for f in ev['frames']];show=group=='camera1_E' or fid in {ids[0],ids[(len(ids)-1)//2],ids[-1]}
        if show:
            path=RUN/'diagnostics/appearance/figures'/f'{dev}_{group}_{fid}.jpg';f=figure(path,[gt,*[preds[x] for x in ['native','dc_only','cam0_direction']]],[fid+' GT','native','DC only','cam0 SH direction'],384)
            figs.append(dict(**f,dev=dev,group=group,frame_id=fid,kind='full comparison',raw_or_clipped='clipped [0,1]'))
            if group=='camera1_E':
                path=path.with_name(path.stem+'_alpha_overflow.jpg');f=figure(path,[alphas['native'],*[heat(np.maximum(preds[x].max(-1)-1,0)) for x in ['native','dc_only','cam0_direction']]],['alpha [0,1]','native max(RGB)-1 [0,1]','DC overflow [0,1]','cam0 overflow [0,1]'],384)
                figs.append(dict(**f,dev=dev,group=group,frame_id=fid,kind='alpha and overflow',color_scale='alpha grayscale [0,1], overflow inferno [0,1] with saturation above1; true values remain in CSV'))
    csvwrite(RUN/'appearance_diagnostics.csv',rows);summary={}
    for dev in ['dev1','dev2']:
        summary[dev]={}
        for group in ['camera1_E','camera0_paired_E']:
            summary[dev][group]={}
            for mode in ['native','dc_only','cam0_direction']:
                summary[dev][group][mode]={reg:mean_summary([r for r in rows if r['dev']==dev and r['group']==group and r['mode']==mode and r['region']==reg]) for reg in ['full','human','object','background']}
    save_json(dest,dict(status='completed',summary=summary,figures=figs,camera_checks=m['camera_checks'],rows=len(rows),seconds=time.monotonic()-start,all_render_parities=m['all_default_and_geometry_parities_passed'],scope='Frozen appearance diagnostic; not a new trained model or geometric accuracy proof'))

def hos(net):
    freeze=read(RUN/'protocol/finals.json');assert freeze['status']=='both_new_terminal_states_frozen'
    for a in freeze['assets']:assert sha(a['path'])==a['sha256']
    start=time.monotonic();examples=read(RUN/'protocol/fixed_examples.json');crop={r['frame_id']:r['crop_bounds_xyxy'] for r in examples['hos_crops']};new={}
    for branch in ['W_fine','W_all']:
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
            for branch in ['H1','W_fine','W_all']:
                asset=old_index[split,fid]['render'] if branch=='H1' else new[branch][split,fid]['render'];assert sha(asset['path'])==asset['sha256']
                if branch=='H1':
                    result={reg:baseline[split,fid,reg] for reg in ['full','foreground','background']}
                    if show:preds[branch]=np.load(asset['path'])['rgb']
                else:
                    pred=np.load(asset['path'])['rgb'];result,_=evaluate_image(pred,gt,labels,net)
                    if show:preds[branch]=pred
                for region in ['full','foreground','background']:
                    rows.append(dict(run=branch,split=split,frame_id=fid,source_frame_index=int(fid),time=frame['time'],time_units='normalized nominal source frame index; original index is not measured seconds',extrapolation=frame.get('extrapolation',int(fid)==0),region=region,region_source='published mask >=128 dynamic-subject foreground; includes carried bag but may exclude stationary bag',
                        **result[region],raw_clipped_policy='clip raw float RGB to [0,1] before metrics',raw_render_path=asset['path'],raw_render_sha256=asset['sha256']))
                indices.append(dict(run=branch,split=split,frame_id=fid,**asset))
            if show:
                path=RUN/'evaluation/figures'/f'{split}_{fid}_full.jpg';f=figure(path,[gt,*[preds[b] for b in ['H1','W_fine','W_all']]],[fid+' GT','H1 uniform','W fine balanced','W all balanced'],440);figs.append(dict(**f,split=split,frame_id=fid,kind='full',branch_order=['GT','H1','W_fine','W_all'],raw_or_clipped='clipped [0,1]'))
                if split=='retained':
                    x0,y0,x1,y1=crop[fid];path=path.with_name(path.stem.replace('_full','_foreground')+'.jpg');f=figure(path,[x[y0:y1,x0:x1] for x in [gt,*[preds[b] for b in ['H1','W_fine','W_all']]]],[fid+' GT crop','H1 crop','W fine crop','W all crop'],280);figs.append(dict(**f,split=split,frame_id=fid,kind='foreground_crop',crop_bounds_xyxy=crop[fid],branch_order=['GT','H1','W_fine','W_all'],raw_or_clipped='clipped [0,1]'))
            if (i+1)%32==0:print(split,i+1,flush=True)
    csvwrite(RUN/'metrics_per_frame.csv',rows);save_json(RUN/'protocol/raw_render_index.json',dict(assets=indices))
    summaries={}
    for branch in ['H1','W_fine','W_all']:
        summaries[branch]={}
        for split in ['train','retained','retained_without_00000']:
            rs=[r for r in rows if r['run']==branch and r['split']==('retained' if split.startswith('retained') else 'train') and (split!='retained_without_00000' or r['frame_id']!='00000')]
            summaries[branch][split]={reg:mean_summary([r for r in rs if r['region']==reg]) for reg in ['full','foreground','background']}
    index={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};assert len(index)==len(rows)==3*284*3
    differences=[];paired={}
    for left,right in [('W_fine','H1'),('W_all','H1'),('W_all','W_fine')]:
        pair=left+'-'+right;paired[pair]={}
        for split in ['train','retained']:
            paired[pair][split]={}
            for region in ['full','foreground','background']:
                rs=[r for r in rows if r['run']==left and r['split']==split and r['region']==region];dd=[]
                for a in rs:
                    b=index[right,split,a['frame_id'],region];assert a['pixels']==b['pixels'];delta={k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in METRICS}
                    item=dict(comparison=pair,split=split,frame_id=a['frame_id'],region=region,**delta);differences.append(item);dd.append(item)
                paired[pair][split][region]={k:dict(mean=float(np.mean([d[k] for d in dd if d[k] is not None])),median=float(np.median([d[k] for d in dd if d[k] is not None])),valid_frames=sum(d[k] is not None for d in dd)) if any(d[k] is not None for d in dd) else dict(mean=None,median=None,valid_frames=0) for k in METRICS}
    csvwrite(RUN/'paired_differences.csv',differences);rules=read(RUN/'configs/v4.json')['validation'];gates={}
    for branch in ['W_fine','W_all']:
        p=paired[branch+'-H1']['retained'];fg=p['foreground'];bg=p['background'];tests=dict(FG_PSNR_mean=fg['psnr_db']['mean']>=rules['FG_PSNR_mean_delta_min'],FG_PSNR_median=fg['psnr_db']['median']>rules['FG_PSNR_paired_median_strict_min'],FG_LPIPS=fg['lpips_spatial_mean']['mean']<=rules['FG_LPIPS_mean_delta_max'],BG_PSNR=bg['psnr_db']['mean']>=rules['BG_PSNR_mean_delta_min']);gates[branch]=dict(numerical_rules=tests,numerical_pass=all(tests.values()),visual_quality_gate='pending full fixed-image review',promotion_status='pending visual review; numeric gate is not sufficient by itself')
    save_json(RUN/'evaluation/summary.json',dict(status='completed',summaries=summaries,paired=paired,gates=gates,rows=len(rows),paired_rows=len(differences),figures=figs,seconds=time.monotonic()-start,
        metrics='per-pixel RGB-channel-mean SSE; MSE=SSE/pixels; original 7x7 SSIM and AlexNet0.1 spatial LPIPS with cross-boundary receptive fields',H1_metrics_reused_verified_V3=True,individual_human_bag_metrics=None,individual_region_NA_reason='No independent instance masks',retained_set='16 already-inspected development frames; separate15 supplemental, no replacement of16 main table'))
    save_json(RUN/'figure_manifest.json',dict(protocol=identity(RUN/'protocol/fixed_examples.json'),hos=figs,appearance=read(RUN/'diagnostics/appearance/summary.json')['figures'],input_support=read(RUN/'initialization_support.json')['figures']))
def run(only_appearance=False):
    torch.set_num_threads(4);cv2.setNumThreads(4);net,info=lpips_cpu();save_json(RUN/'protocol/evaluation_lpips.json',info);appearance(net)
    if not only_appearance:hos(net)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--appearance-only',action='store_true');run(p.parse_args().appearance_only)
