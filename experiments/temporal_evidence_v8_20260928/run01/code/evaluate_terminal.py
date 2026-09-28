"""One unchanged evaluator for all complete V8 terminals and inherited B-U."""
from common import *
from evaluate_and_report import csvwrite,lpips_cpu,figure,mean_summary,METRICS
from evaluate_frozen import evaluate_image
import csv,time,cv2,numpy as np,torch
REGIONS=['full','foreground','background'];ARMS=['B_U','B_Q','T_plain','T_mix']
COMPARISONS=[('B_Q','B_U'),('T_plain','B_Q'),('T_mix','T_plain'),('T_mix','B_Q'),('T_plain','B_U'),('T_mix','B_U')]

def run():
    start=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(4);freeze=read(RUN/'protocol/finals.json');assert freeze['status']=='all_three_terminals_frozen'
    for x in freeze['assets']:assert sha(x['path'])==x['sha256']
    net,info=lpips_cpu();save_json(RUN/'protocol/evaluation_lpips.json',info)
    evaluator=read(ROOT/'experiments/appearance_router_v7_20260928/run01/protocol/evaluator.json');save_json(RUN/'protocol/evaluator.json',evaluator)
    old=list(csv.DictReader((V5/'metrics_per_frame.csv').open()));number=['pixels','pixel_fraction','mse','sse_rgb_mean','psnr_db','ssim','lpips_spatial_mean','full_error_share','raw_mse','raw_psnr_db','time','source_frame_index'];rows=[]
    for r in old:
        if r['run']!='B_U':continue
        for k in number:r[k]=None if r[k]=='' else int(r[k]) if k in ['pixels','source_frame_index'] else float(r[k])
        r['evaluator_id']=evaluator['id'];r['inherited_baseline']=True;rows.append(r)
    baseline={(r['split'],r['frame_id'],r['region']):r for r in rows};render_indices={}
    for arm in ARMS[1:]:
        m=read(RUN/'evaluation'/arm/'manifest.json');assert m['freeze']['sha256']==sha(RUN/'protocol/finals.json');render_indices[arm]={(x['split'],x['frame_id']):x for x in m['rows']};assert len(render_indices[arm])==284
    ex=read(RUN/'protocol/fixed_examples.json');crop={x['frame_id']:x['crop_bounds_xyxy'] for x in ex['hos_crops']};figs=[];windows=[];raw_index=[];out=RUN/'feedback_arrays';out.mkdir(exist_ok=True)
    for split,name in [('retained','evaluation_manifest.json'),('train','manifest.json')]:
        for i,f in enumerate(read(OLD/'inputs/hos_backpack'/name)['frames']):
            fid=f['frame_id'];gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255.;labels=(cv2.imread(f['mask_path'],0)>=128).astype(np.uint8);preds={}
            for arm in ARMS[1:]:
                asset=render_indices[arm][split,fid]['render'];assert sha(asset['path'])==asset['sha256'];pred=np.load(asset['path'])['rgb'];preds[arm]=pred
                result,_=evaluate_image(pred,gt,labels,net)
                for reg in REGIONS:
                    mask=np.ones(labels.shape,bool) if reg=='full' else labels==1 if reg=='foreground' else labels==0;raw_mse=float(np.square(pred[mask]-gt[mask]).mean())
                    rows.append(dict(run=arm,status='completed',NA_reason=None,split=split,frame_id=fid,source_frame_index=int(fid),time=f['time'],time_units='normalized nominal frame index, not seconds',extrapolation=int(fid)==0,region=reg,region_source='unchanged foreground mask may exclude stationary bag',**result[reg],raw_mse=raw_mse,raw_psnr_db=float(-10*np.log10(max(raw_mse,1e-12))),raw_clipped_policy=evaluator['clipping'],raw_render_path=asset['path'],raw_render_sha256=asset['sha256'],evaluator_id=evaluator['id'],inherited_baseline=False))
                raw_index.append(dict(run=arm,split=split,frame_id=fid,**asset))
            if split=='retained' or fid in ex['training_frame_ids']:
                b=baseline[split,fid,'full'];assert sha(b['raw_render_path'])==b['raw_render_sha256'];preds['B_U']=np.load(b['raw_render_path'])['rgb'];arrays=[gt,*[preds[a] for a in ARMS]];labels_=[fid+' GT','B-U','B-Q','T-plain','T-mix'];p=RUN/'evaluation/figures'/f'{split}_{fid}_full.jpg'
                figs.append(dict(**figure(p,arrays,labels_,400),split=split,frame_id=fid,kind='full',branch_order=['GT',*ARMS]))
                if split=='retained':
                    x0,y0,x1,y1=crop[fid];p=p.with_name(p.stem.replace('_full','_foreground')+'.jpg');figs.append(dict(**figure(p,[x[y0:y1,x0:x1] for x in arrays],labels_,260),split=split,frame_id=fid,kind='foreground_crop',crop_bounds_xyxy=crop[fid],branch_order=['GT',*ARMS]))
                    item=next((x for x in ex['small_array_windows'] if x['dataset']=='hos_backpack' and x['frame_id']==fid),None)
                    if item:
                        for reg,bb in item['windows'].items():
                            x0,y0,x1,y1=bb;sl=np.s_[y0:y1,x0:x1];data=dict(GT=gt[sl].astype(np.float32),mask=labels[sl],**{a+'_raw':preds[a][sl] for a in ARMS});p=out/f'Backpack_{fid}_{reg}.npz';np.savez_compressed(p,**data);windows.append(dict(**identity(p),frame_id=fid,region=reg,bounds_xyxy=bb))
            if (i+1)%32==0:print('metrics',split,i+1,flush=True)
    csvwrite(RUN/'metrics_per_frame.csv',rows);save_json(RUN/'protocol/raw_render_index.json',dict(assets=raw_index));summaries={}
    for arm in ARMS:
        summaries[arm]={}
        for split in ['train','retained','retained_without_00000']:
            rr=[r for r in rows if r['run']==arm and r['split']==('train' if split=='train' else 'retained') and (split!='retained_without_00000' or r['frame_id']!='00000')];summaries[arm][split]={reg:mean_summary([r for r in rr if r['region']==reg]) for reg in REGIONS}
    idx={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};assert len(idx)==len(rows)==4*284*3;diffs=[];paired={};tol=read(RUN/'configs/v8.json')['quality']['tie_precision']
    for arm,ref in COMPARISONS:
        pair=arm+'-'+ref;paired[pair]={}
        for r in rows:
            if r['run']!=arm:continue
            other=idx[ref,r['split'],r['frame_id'],r['region']];assert other['pixels']==r['pixels'];diffs.append(dict(comparison=pair,split=r['split'],frame_id=r['frame_id'],region=r['region'],**{k:r[k]-other[k] for k in METRICS}))
        for split in ['train','retained','retained_without_00000']:
            paired[pair][split]={}
            for reg in REGIONS:
                rr=[d for d in diffs if d['comparison']==pair and d['split']==('train' if split=='train' else 'retained') and d['region']==reg and (split!='retained_without_00000' or d['frame_id']!='00000')];vals={}
                for k in METRICS:
                    a=np.array([x[k] for x in rr]);oriented=a*(-1 if k=='lpips_spatial_mean' else 1);vals[k]=dict(mean=float(a.mean()),median=float(np.median(a)),win=int((oriented>tol[k]).sum()),tie=int((abs(oriented)<=tol[k]).sum()),loss=int((oriented < -tol[k]).sum()),frames=len(a))
                paired[pair][split][reg]=vals
    csvwrite(RUN/'paired_differences.csv',diffs)
    # Four consecutive fixed blocks of 4 frames, descriptive rather than independent trials.
    blocks=[];dev_ids=sorted({r['frame_id'] for r in rows if r['split']=='retained'})
    for arm,ref in COMPARISONS:
        for block in range(4):
            ids=dev_ids[4*block:4*block+4]
            for reg in REGIONS:
                rr=[r for r in diffs if r['comparison']==arm+'-'+ref and r['split']=='retained' and r['frame_id'] in ids and r['region']==reg];blocks.append(dict(comparison=arm+'-'+ref,block=block,frame_ids=ids,region=reg,**{k:float(np.mean([x[k] for x in rr])) for k in METRICS}))
    save_json(RUN/'evaluation_summary.json',dict(status='completed',summaries=summaries,paired=paired,time_blocks=blocks,figures=figs,rows=len(rows),paired_rows=len(diffs),seconds=time.monotonic()-start,evaluator=evaluator,individual_human_object_metrics=None,NA_reason='No reliable separate human/object masks or true geometry/tracks'))
    save_json(RUN/'figure_manifest.json',dict(protocol=identity(RUN/'protocol/fixed_examples.json'),hos=figs));save_json(out/'manifest.json',dict(windows=windows,all_float32=True));save_json(RUN/'quality_decision.json',dict(status='requires_complete_vector_assessment',no_scalar_score=True,no_auxiliary_veto=True,next='Assess full and FG vectors and fixed panels; then select permitted Tennis branch or close per attachment section8.'))
    print('All four full metric vectors completed; quality decision needs explicit evidence assessment')
if __name__=='__main__':run()
