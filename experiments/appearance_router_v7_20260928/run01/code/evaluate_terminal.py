"""Complete fixed metric vectors; reuse baseline measurements and one evaluator."""
from common import *
from evaluate_and_report import csvwrite,lpips_cpu,figure,mean_summary,METRICS
from evaluate_frozen import evaluate_image
import csv,time,cv2,numpy as np,torch
REGIONS=['full','foreground','background']
def run():
    start=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(4)
    freeze=read(RUN/'protocol/finals.json');assert freeze['status']=='sole_terminal_frozen'
    for x in freeze['assets']:assert sha(x['path'])==x['sha256']
    net,info=lpips_cpu();save_json(RUN/'protocol/evaluation_lpips.json',info)
    evaluator=dict(id='V3_V5_clip_SSIM7_AlexNet0.1_spatial',clipping='RGB clipped [0,1] for main metrics; raw MSE/PSNR separate',data_range=1.,SSIM=dict(win_size=7,gaussian_weights=False,use_sample_covariance=True,channel_axis=2,aggregation='RGB channel mean map then region mean, including boundary receptive fields'),LPIPS=dict(backbone='alex',version='0.1',spatial=True,inputs='float32 [0,1], normalize=True maps to [-1,1]',aggregation='region mean of full spatial map'),PSNR='Per-frame dB macro mean; pooled MSE to PSNR separately',mask='unchanged mask >=128',source=identity(OLD/'code/evaluate_frozen.py'))
    save_json(RUN/'protocol/evaluator.json',evaluator)
    with (V5/'metrics_per_frame.csv').open() as f:old=list(csv.DictReader(f))
    number=['pixels','pixel_fraction','mse','sse_rgb_mean','psnr_db','ssim','lpips_spatial_mean','full_error_share','raw_mse','raw_psnr_db','time','source_frame_index']
    rows=[]
    for r in old:
        if r['run'] not in ['B_U','B_F']:continue
        for k in number:r[k]=None if r[k]=='' else int(r[k]) if k in ['pixels','source_frame_index'] else float(r[k])
        r['evaluator_id']=evaluator['id'];r['inherited_baseline']=True;rows.append(r)
    baseline={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows}
    render=read(RUN/'evaluation/M1/manifest.json');assert render['freeze']['sha256']==sha(RUN/'protocol/finals.json')
    new={(x['split'],x['frame_id']):x for x in render['rows']};assert len(new)==284
    ex=read(RUN/'protocol/fixed_examples.json');crop={x['frame_id']:x['crop_bounds_xyxy'] for x in ex['hos_crops']};figs=[];windows=[];raw_index=[]
    out=RUN/'feedback_arrays';out.mkdir(exist_ok=True)
    for split,name in [('retained','evaluation_manifest.json'),('train','manifest.json')]:
        for i,f in enumerate(read(OLD/'inputs/hos_backpack'/name)['frames']):
            fid=f['frame_id'];asset=new[split,fid]['render'];assert sha(asset['path'])==asset['sha256']
            pred=np.load(asset['path'])['rgb'];gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255.;labels=(cv2.imread(f['mask_path'],0)>=128).astype(np.uint8)
            result,_=evaluate_image(pred,gt,labels,net)
            for reg in REGIONS:
                mask=np.ones(labels.shape,bool) if reg=='full' else labels==1 if reg=='foreground' else labels==0
                raw_mse=float(np.square(pred[mask]-gt[mask]).mean())
                rows.append(dict(run='M1',status='completed',NA_reason=None,split=split,frame_id=fid,source_frame_index=int(fid),time=f['time'],time_units='normalized nominal source frame index, not measured seconds',extrapolation=int(fid)==0,region=reg,region_source='unchanged foreground mask may exclude stationary bag',**result[reg],raw_mse=raw_mse,raw_psnr_db=float(-10*np.log10(max(raw_mse,1e-12))),raw_clipped_policy=evaluator['clipping'],raw_render_path=asset['path'],raw_render_sha256=asset['sha256'],evaluator_id=evaluator['id'],inherited_baseline=False))
            raw_index.append(dict(run='M1',split=split,frame_id=fid,**asset))
            show=split=='retained' or fid in ex['training_frame_ids']
            if show:
                oldpics=[]
                for b in ['B_U','B_F']:
                    a=baseline[b,split,fid,'full'];assert sha(a['raw_render_path'])==a['raw_render_sha256'];oldpics.append(np.load(a['raw_render_path'])['rgb'])
                arrays=[gt,*oldpics,pred];labels_=[fid+' GT','B-U uniform','B-F balanced','M1 appearance']
                p=RUN/'evaluation/figures'/f'{split}_{fid}_full.jpg';assetfig=figure(p,arrays,labels_,440)
                figs.append(dict(**assetfig,split=split,frame_id=fid,kind='full',branch_order=['GT','B_U','B_F','M1']))
                if split=='retained':
                    x0,y0,x1,y1=crop[fid];p=p.with_name(p.stem.replace('_full','_foreground')+'.jpg')
                    assetfig=figure(p,[x[y0:y1,x0:x1] for x in arrays],labels_,280)
                    figs.append(dict(**assetfig,split=split,frame_id=fid,kind='foreground_crop',crop_bounds_xyxy=crop[fid],branch_order=['GT','B_U','B_F','M1']))
                    item=next((x for x in ex['small_array_windows'] if x['dataset']=='hos_backpack' and x['frame_id']==fid),None)
                    if item:
                        for reg,bb in item['windows'].items():
                            x0,y0,x1,y1=bb;sl=np.s_[y0:y1,x0:x1]
                            data=dict(GT=gt[sl].astype(np.float32),mask=labels[sl],B_U_raw=oldpics[0][sl],B_F_raw=oldpics[1][sl],M1_raw=pred[sl])
                            assert all(v.dtype==(np.uint8 if k=='mask' else np.float32) for k,v in data.items())
                            p=out/f'Backpack_{fid}_{reg}.npz';np.savez_compressed(p,**data);windows.append(dict(**identity(p),frame_id=fid,region=reg,bounds_xyxy=bb,source_render=asset))
            if (i+1)%32==0:print('metrics',split,i+1,flush=True)
    csvwrite(RUN/'metrics_per_frame.csv',rows);save_json(RUN/'protocol/raw_render_index.json',dict(assets=raw_index))
    summaries={}
    for b in ['B_U','B_F','M1']:
        summaries[b]={}
        for split in ['train','retained','retained_without_00000']:
            rr=[r for r in rows if r['run']==b and r['split']==('train' if split=='train' else 'retained') and (split!='retained_without_00000' or r['frame_id']!='00000')]
            summaries[b][split]={reg:mean_summary([r for r in rr if r['region']==reg]) for reg in REGIONS}
    idx={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};assert len(idx)==len(rows)==3*284*3
    diffs=[];paired={};rules=read(RUN/'configs/v7.json')['quality'];tol=rules['tie_precision']
    for b in ['B_U','B_F']:
        pair='M1-'+b;paired[pair]={}
        for r in rows:
            if r['run']!='M1':continue
            ref=idx[b,r['split'],r['frame_id'],r['region']];assert r['pixels']==ref['pixels']
            diffs.append(dict(comparison=pair,split=r['split'],frame_id=r['frame_id'],region=r['region'],**{k:r[k]-ref[k] for k in METRICS}))
        for split in ['train','retained','retained_without_00000']:
            paired[pair][split]={}
            for reg in REGIONS:
                rr=[d for d in diffs if d['comparison']==pair and d['split']==('train' if split=='train' else 'retained') and d['region']==reg and (split!='retained_without_00000' or d['frame_id']!='00000')]
                values={}
                for k in METRICS:
                    a=np.array([x[k] for x in rr]);oriented=a*(-1 if k=='lpips_spatial_mean' else 1)
                    values[k]=dict(mean=float(a.mean()),median=float(np.median(a)),win=int((oriented>tol[k]).sum()),tie=int((abs(oriented)<=tol[k]).sum()),loss=int((oriented < -tol[k]).sum()),frames=len(a))
                paired[pair][split][reg]=values
    csvwrite(RUN/'paired_differences.csv',diffs)
    relations={}
    for b in ['B_U','B_F']:
        relations[b]={}
        for reg in REGIONS:
            delta={k:paired['M1-'+b]['retained'][reg][k]['mean'] for k in METRICS}
            signed=[delta[k]*(-1 if k=='lpips_spatial_mean' else 1) for k in METRICS]
            directions=[1 if v>tol[k] else -1 if v < -tol[k] else 0 for k,v in zip(METRICS,signed)]
            relation='dominates' if min(directions)>=0 and max(directions)>0 else 'dominated' if max(directions)<=0 and min(directions)<0 else 'equal_at_report_precision' if all(v==0 for v in directions) else 'tradeoff'
            relations[b][reg]=dict(relation=relation,differences=delta,directions=directions)
    bu=relations['B_U'];bf=relations['B_F']
    if bu['full']['relation']=='dominates':decision='retain_quality_candidate_for_future_independent_confirmation'
    elif all(relations[b]['full']['relation']=='dominated' and relations[b]['foreground']['relation']=='dominated' for b in ['B_U','B_F']):decision='close_router_no_quality_basis'
    elif bu['full']['relation']=='dominated':decision='retain_regional_tradeoff_only' if any(x>0 for x in bu['foreground']['directions']) else 'close_router_no_quality_basis'
    else:decision='retain_non_dominated_tradeoff_for_assessment' if bf['full']['relation']!='dominated' else 'existing_B_F_dominates_full_vector_assess_regional_use_only'
    save_json(RUN/'quality_decision.json',dict(status='completed',decision=decision,relations=relations,tie_rule=tol,tie_basis='predeclared six-decimal report precision, not statistical significance',scope='one already-used development sequence; no independent generalization claim',no_auxiliary_veto=True,next_sequence_not_executed=True))
    save_json(RUN/'evaluation_summary.json',dict(status='completed',summaries=summaries,paired=paired,figures=figs,rows=len(rows),paired_rows=len(diffs),seconds=time.monotonic()-start,evaluator=evaluator,individual_human_object_metrics=None,NA_reason='No reliable separate instance masks; no new geometry/track truth',new_N_eff='not_recomputed'))
    save_json(RUN/'figure_manifest.json',dict(protocol=identity(RUN/'protocol/fixed_examples.json'),hos=figs))
    save_json(out/'manifest.json',dict(windows=windows,all_float32=True,region_metrics_from_full_mask=True))
    print('All terminal metric vectors and fixed figures completed',decision)
if __name__=='__main__':run()
