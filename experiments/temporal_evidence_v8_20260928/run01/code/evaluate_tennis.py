"""Unchanged CPU three-metric evaluator, U/Q, all Tennis frames and regions."""
from common import *
from evaluate_and_report import csvwrite,lpips_cpu,figure,mean_summary,METRICS
from evaluate_frozen import evaluate_image
import time,cv2,numpy as np,torch
B=RUN/'tennis';ARMS=['B_U','B_Q'];REGIONS=['full','foreground','background']
def run():
    start=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(4)
    freeze=read(B/'protocol/finals.json');assert freeze['status']=='both_terminals_frozen'
    for x in freeze['assets']:assert sha(x['path'])==x['sha256']
    net,info=lpips_cpu();save_json(B/'protocol/evaluation_lpips.json',info)
    evaluator=read(RUN/'protocol/evaluator.json');save_json(B/'protocol/evaluator.json',evaluator)
    ex=read(B/'protocol/fixed_examples.json');indices={};rows=[];figs=[];windows=[];crops=[]
    for arm in ARMS:
        m=read(B/'evaluation'/arm/'manifest.json');assert m['freeze']['sha256']==sha(B/'protocol/finals.json');indices[arm]={(x['split'],x['frame_id']):x for x in m['rows']}
    for split,name in [('retained','evaluation_manifest.json'),('train','manifest.json')]:
        for f in read(B/'inputs/hos_tennis'/name)['frames']:
            fid=f['frame_id'];gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255.;labels=(cv2.imread(f['mask_path'],0)>=128).astype(np.uint8);preds={}
            for arm in ARMS:
                asset=indices[arm][split,fid]['render'];assert sha(asset['path'])==asset['sha256'];pred=np.load(asset['path'])['rgb'];preds[arm]=pred;result,_=evaluate_image(pred,gt,labels,net)
                for reg in REGIONS:
                    mask=np.ones(labels.shape,bool) if reg=='full' else labels==1 if reg=='foreground' else labels==0
                    mse=float(np.square(pred[mask]-gt[mask]).mean()) if mask.any() else None
                    rows.append(dict(run=arm,status='completed',NA_reason=None if mask.any() else 'empty input region',split=split,frame_id=fid,region=reg,time=f['time'],source_frame_index=int(fid),extrapolation=f['extrapolation'],**result[reg],raw_mse=mse,raw_psnr_db=None if mse is None else float(-10*np.log10(max(mse,1e-12))),raw_render_path=asset['path'],raw_render_sha256=asset['sha256'],evaluator_id=evaluator['id']))
            if split=='retained' or fid in ex['training_frame_ids']:
                arrays=[gt,*[preds[a] for a in ARMS]];names=[fid+' GT','B-U','B-Q'];p=B/'evaluation/figures'/f'{split}_{fid}_full.jpg';figs.append(dict(**figure(p,arrays,names,480),split=split,frame_id=fid,kind='full'))
                if split=='retained':
                    yy,xx=np.where(labels==1);h,w=labels.shape
                    # Fixed input-mask bounding box + 24px; no prediction-based selection.
                    bb=[max(0,int(xx.min())-24),max(0,int(yy.min())-24),min(w,int(xx.max())+25),min(h,int(yy.max())+25)] if len(xx) else [0,0,w,h]
                    x0,y0,x1,y1=bb;crops.append(dict(frame_id=fid,bounds_xyxy=bb,source='input mask >=128 + 24px'))
                    figs.append(dict(**figure(p.with_name(p.stem.replace('_full','_foreground')+'.jpg'),[x[y0:y1,x0:x1] for x in arrays],names,320),split=split,frame_id=fid,kind='foreground_crop'))
                    if fid in ex['float_window_frame_ids']:
                        for reg,center in [('foreground',((x0+x1)//2,(y0+y1)//2)),('background',(40,40))]:
                            cx,cy=center;xa=max(0,min(w-80,cx-40));ya=max(0,min(h-80,cy-40));sl=np.s_[ya:ya+80,xa:xa+80]
                            path=B/'feedback_arrays'/f'{fid}_{reg}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,GT=gt[sl].astype(np.float32),mask=labels[sl],**{a+'_raw':preds[a][sl] for a in ARMS});windows.append(dict(**identity(path),frame_id=fid,region=reg,bounds_xyxy=[xa,ya,xa+80,ya+80]))
    assert len(rows)==2*sum(len(read(B/'inputs/hos_tennis'/n)['frames']) for n in ['manifest.json','evaluation_manifest.json'])*3
    csvwrite(B/'metrics_per_frame.csv',rows);idx={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};diffs=[]
    for r in rows:
        if r['run']!='B_Q':continue
        other=idx['B_U',r['split'],r['frame_id'],r['region']];assert other['pixels']==r['pixels'];diffs.append(dict(comparison='B_Q-B_U',split=r['split'],frame_id=r['frame_id'],region=r['region'],**{k:r[k]-other[k] if r[k] is not None and other[k] is not None else None for k in METRICS}))
    csvwrite(B/'paired_differences.csv',diffs);summary={};paired={'B_Q-B_U':{}};tol=read(RUN/'configs/v8.json')['quality']['tie_precision']
    for split in ['train','retained','retained_without_00000']:
        def select(rs):return [r for r in rs if r['split']==('train' if split=='train' else 'retained') and (split!='retained_without_00000' or r['frame_id']!='00000')]
        for arm in ARMS:summary.setdefault(arm,{})[split]={reg:mean_summary([r for r in select(rows) if r['run']==arm and r['region']==reg]) for reg in REGIONS}
        paired['B_Q-B_U'][split]={}
        for reg in REGIONS:
            vals={}
            for k in METRICS:
                v=np.array([r[k] for r in select(diffs) if r['region']==reg and r[k] is not None]);o=v*(-1 if k=='lpips_spatial_mean' else 1)
                vals[k]=dict(mean=float(v.mean()) if len(v) else None,median=float(np.median(v)) if len(v) else None,win=int((o>tol[k]).sum()),tie=int((abs(o)<=tol[k]).sum()),loss=int((o < -tol[k]).sum()),frames=len(v))
            paired['B_Q-B_U'][split][reg]=vals
    blocks=[];ids=sorted({r['frame_id'] for r in rows if r['split']=='retained'})
    for block in range(4):
        for reg in REGIONS:
            dd=[r for r in diffs if r['split']=='retained' and r['frame_id'] in ids[block*4:block*4+4] and r['region']==reg]
            blocks.append(dict(comparison='B_Q-B_U',block=block,frame_ids=ids[block*4:block*4+4],region=reg,**{k:float(np.mean([d[k] for d in dd if d[k] is not None])) if any(d[k] is not None for d in dd) else None for k in METRICS}))
    save_json(B/'evaluation_summary.json',dict(status='completed',summaries=summary,paired=paired,time_blocks=blocks,figures=figs,rows=len(rows),paired_rows=len(diffs),seconds=time.monotonic()-start,evaluator=evaluator,individual_human_object_metrics=None,NA_reason='No separate reliable human/object masks or independent geometry truth'))
    save_json(B/'figure_manifest.json',dict(figures=figs,crops=crops));save_json(B/'feedback_arrays/manifest.json',dict(windows=windows,all_float32=True))
if __name__=='__main__':run()
