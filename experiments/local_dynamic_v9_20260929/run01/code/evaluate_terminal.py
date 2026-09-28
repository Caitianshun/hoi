"""Unchanged V8 image metrics, fixed images, paired vectors, equal-scene means."""
from common import *
from evaluate_and_report import csvwrite,lpips_cpu,figure,mean_summary,METRICS
from evaluate_frozen import evaluate_image
import csv,cv2,numpy as np,torch
REGIONS=['full','foreground','background']
ARMS=['U','Q','Q0','S','SL']
COMPARISONS=[('S','Q0'),('SL','S'),('SL','Q0')]+[(a,b) for a in ['Q0','S','SL'] for b in ['U','Q']]
SPLITS=['train','retained','retained_without_00000']


def select(rows,split):
    return [r for r in rows if r['split']==('train' if split=='train' else 'retained') and (split!='retained_without_00000' or r['frame_id']!='00000')]


def historical_rows(scene):
    path=ROOT/config()['scenes'][scene]['historical_dir']/'metrics_per_frame.csv'
    number=['pixels','pixel_fraction','mse','sse_rgb_mean','psnr_db','ssim','lpips_spatial_mean',
            'full_error_share','raw_mse','raw_psnr_db','time','source_frame_index']
    rows=[]
    with path.open() as f:
        for r in csv.DictReader(f):
            if r['run'] not in ['B_U','B_Q']:continue
            for k in number:
                if k in r:r[k]=None if r[k]=='' else int(r[k]) if k in ['pixels','source_frame_index'] else float(r[k])
            r['scene']=scene;r['run']=r['run'][2:];r['historical']=True
            rows.append(r)
    return rows


def paired_summary(diffs):
    result={};tol=config()['evaluation']['ties']
    for arm,ref in COMPARISONS:
        key=arm+'-'+ref;result[key]={}
        for split in SPLITS:
            result[key][split]={}
            for region in REGIONS:
                rows=[r for r in select(diffs,split) if r['comparison']==key and r['region']==region]
                stats={}
                for metric in METRICS:
                    vals=np.array([r[metric] for r in rows if r[metric] is not None]);oriented=vals*(-1 if metric=='lpips_spatial_mean' else 1)
                    stats[metric]=dict(mean=float(vals.mean()) if len(vals) else None,median=float(np.median(vals)) if len(vals) else None,
                        win=int((oriented>tol).sum()),tie=int((abs(oriented)<=tol).sum()),loss=int((oriented< -tol).sum()),frames=len(vals))
                result[key][split][region]=stats
    return result


def evaluate_scene(scene,net,evaluator):
    start=time.monotonic();cfg=config()['scenes'][scene];sd=scene_dir(scene)
    rows=historical_rows(scene);hist={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows}
    historical=ROOT/cfg['historical_dir'];ex=read(historical/'protocol/fixed_examples.json')
    if scene=='Backpack':crops={r['frame_id']:r['crop_bounds_xyxy'] for r in ex['hos_crops']}
    else:crops={r['frame_id']:r['bounds_xyxy'] for r in read(historical/'figure_manifest.json')['crops']}
    indices={}
    for arm in ['Q0','S','SL']:
        m=read(sd/'evaluation'/arm/'manifest.json');assert m['freeze']['sha256']==sha(RUN/'protocol/finals.json')
        indices[arm]={(r['split'],r['frame_id']):r for r in m['rows']}
    figs=[];windows=[]
    for split,name in [('retained','evaluation_manifest.json'),('train','manifest.json')]:
        for i,f in enumerate(read(ROOT/cfg['input_dir']/name)['frames']):
            fid=f['frame_id'];gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255
            labels=(cv2.imread(f['mask_path'],0)>=128).astype(np.uint8);preds={}
            for arm in ['Q0','S','SL']:
                asset=indices[arm][split,fid]['render'];assert sha(asset['path'])==asset['sha256']
                pred=np.load(asset['path'])['rgb'];preds[arm]=pred
                result,_=evaluate_image(pred,gt,labels,net)
                for reg in REGIONS:
                    mask=np.ones(labels.shape,bool) if reg=='full' else labels==1 if reg=='foreground' else labels==0
                    mse=float(np.square(pred[mask]-gt[mask]).mean()) if mask.any() else None
                    rows.append(dict(scene=scene,run=arm,status='completed',historical=False,split=split,frame_id=fid,region=reg,
                        time=f['time'],source_frame_index=int(fid),extrapolation=f.get('extrapolation',fid=='00000'),
                        **result[reg],raw_mse=mse,raw_psnr_db=None if mse is None else float(-10*np.log10(max(mse,1e-12))),
                        raw_render_path=asset['path'],raw_render_sha256=asset['sha256'],evaluator_id=evaluator['id']))
            if split=='retained' or fid in ex['training_frame_ids']:
                old=hist['Q',split,fid,'full'];assert sha(old['raw_render_path'])==old['raw_render_sha256']
                preds['Q']=np.load(old['raw_render_path'])['rgb']
                arrays=[gt]+[preds[a] for a in ['Q','Q0','S','SL']]
                names=[fid+' GT','historical Q','Q0','S','SL']
                p=sd/'evaluation/figures'/f'{split}_{fid}_full.jpg'
                figs.append(dict(**figure(p,arrays,names,480),scene=scene,split=split,frame_id=fid,kind='full'))
                if split=='retained':
                    x0,y0,x1,y1=crops[fid]
                    p=p.with_name(p.stem.replace('_full','_foreground')+'.jpg')
                    figs.append(dict(**figure(p,[x[y0:y1,x0:x1] for x in arrays],names,320),scene=scene,split=split,frame_id=fid,
                                     kind='foreground_crop',bounds_xyxy=crops[fid]))
                    # First four fixed retained windows are reproducibility excerpts,
                    # never used to select points, poses or training configuration.
                    if i<4:
                        for reg,(cx,cy) in [('foreground',((x0+x1)//2,(y0+y1)//2)),('background',(40,40))]:
                            h,w=labels.shape;xa=max(0,min(w-80,cx-40));ya=max(0,min(h-80,cy-40));sl=np.s_[ya:ya+80,xa:xa+80]
                            path=sd/'feedback_arrays'/f'{fid}_{reg}.npz';path.parent.mkdir(exist_ok=True)
                            np.savez_compressed(path,GT=gt[sl].astype(np.float32),mask=labels[sl],**{a+'_raw':preds[a][sl] for a in ['Q','Q0','S','SL']})
                            windows.append(dict(**identity(path),scene=scene,frame_id=fid,region=reg,bounds_xyxy=[xa,ya,xa+80,ya+80]))
            if (i+1)%32==0:print(scene,split,i+1,flush=True)
    assert len(rows)==5*(cfg['training_frames']+cfg['retained_frames'])*3
    csvwrite(sd/'metrics_per_frame.csv',rows)
    summary={a:{s:{reg:mean_summary([r for r in select(rows,s) if r['run']==a and r['region']==reg]) for reg in REGIONS} for s in SPLITS} for a in ARMS}
    idx={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};diffs=[]
    for arm,ref in COMPARISONS:
        for r in rows:
            if r['run']!=arm:continue
            other=idx[ref,r['split'],r['frame_id'],r['region']];assert r['pixels']==other['pixels']
            diffs.append(dict(scene=scene,comparison=arm+'-'+ref,split=r['split'],frame_id=r['frame_id'],region=r['region'],
                **{k:r[k]-other[k] if r[k] is not None and other[k] is not None else None for k in METRICS}))
    csvwrite(sd/'paired_differences.csv',diffs);paired=paired_summary(diffs)
    blocks=[];ids=sorted({r['frame_id'] for r in rows if r['split']=='retained'})
    for arm,ref in COMPARISONS:
        for block in range(4):
            for reg in REGIONS:
                rr=[r for r in diffs if r['comparison']==arm+'-'+ref and r['split']=='retained' and r['region']==reg and r['frame_id'] in ids[4*block:4*block+4]]
                blocks.append(dict(scene=scene,comparison=arm+'-'+ref,block=block,region=reg,frame_ids=ids[4*block:4*block+4],
                    **{k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in METRICS}))
    result=dict(status='completed',summaries=summary,paired=paired,time_blocks=blocks,rows=len(rows),paired_rows=len(diffs),
                figures=figs,CPU_seconds=time.monotonic()-start,evaluator=evaluator,individual_human_object_metrics=None,
                NA_reason='No reliable separate human/object masks; image metrics do not verify contact or physical geometry')
    save_json(sd/'evaluation_summary.json',result);save_json(sd/'figure_manifest.json',dict(figures=figs,crops=crops))
    save_json(sd/'feedback_arrays/manifest.json',dict(windows=windows))
    return result,rows,diffs


def run():
    started=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(4)
    frozen=read(RUN/'protocol/finals.json');assert frozen['status']=='all_six_terminals_frozen'
    for asset in frozen['assets']:assert sha(asset['path'])==asset['sha256']
    net,info=lpips_cpu();save_json(RUN/'protocol/evaluation_lpips.json',info)
    evaluator=read(V8/'protocol/evaluator.json');save_json(RUN/'protocol/evaluator.json',evaluator)
    scenes={};rows=[];diffs=[]
    for scene in config()['scenes']:
        result,rr,dd=evaluate_scene(scene,net,evaluator);scenes[scene]=result;rows.extend(rr);diffs.extend(dd)
    equal={a:{s:{reg:{k:float(np.mean([v['summaries'][a][s][reg][k] for v in scenes.values()])) for k in METRICS} for reg in REGIONS} for s in SPLITS} for a in ARMS}
    equal_diff={a+'-'+b:{s:{reg:{k:equal[a][s][reg][k]-equal[b][s][reg][k] for k in METRICS} for reg in REGIONS} for s in SPLITS} for a,b in COMPARISONS}
    csvwrite(RUN/'metrics_per_frame.csv',rows);csvwrite(RUN/'paired_differences.csv',diffs)
    save_json(RUN/'evaluation_summary.json',dict(status='completed',scenes=scenes,equal_scene_means=equal,
        equal_scene_differences=equal_diff,CPU_seconds=time.monotonic()-started,rows=len(rows),paired_rows=len(diffs),
        historical_comparison_warning='U/Q differ in initialization and training budget; only S-Q0 and SL-S isolate V9 modules',
        selection='Assess complete vectors; no scalar score, no per-frame oracle, both scenes are development scenes'))


if __name__=='__main__':run()
