"""Independent CSV arithmetic, terminal/source identities and budget accounting."""
from common import *
import csv,cv2,numpy as np,torch


def run():
    start=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(4)
    frozen=read(RUN/'protocol/finals.json');assert frozen['status']=='all_six_terminals_frozen'
    for asset in frozen['assets']:assert sha(asset['path'])==asset['sha256']
    with (RUN/'metrics_per_frame.csv').open() as f:rows=list(csv.DictReader(f))
    with (RUN/'paired_differences.csv').open() as f:diffs=list(csv.DictReader(f))
    idx={(r['scene'],r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};assert len(idx)==len(rows)
    expected=sum(c['training_frames']+c['retained_frames'] for c in config()['scenes'].values())
    assert len(rows)==expected*5*3 and len(diffs)==expected*9*3
    metrics=['psnr_db','ssim','lpips_spatial_mean'];max_pair=0.;max_psnr=0.;checked_psnr=0
    for d in diffs:
        a,b=d['comparison'].split('-');key=(d['scene'],d['split'],d['frame_id'],d['region'])
        x=idx[key[0],a,*key[1:]];y=idx[key[0],b,*key[1:]]
        for k in metrics:
            if d[k]!='':max_pair=max(max_pair,abs(float(x[k])-float(y[k])-float(d[k])))
    assert max_pair<1e-12
    ev=read(RUN/'evaluation_summary.json');max_aggregate=0.
    for scene,sc in config()['scenes'].items():
        for arm in ['U','Q','Q0','S','SL']:
            for split in ['train','retained','retained_without_00000']:
                for reg in ['full','foreground','background']:
                    rr=[r for r in rows if r['scene']==scene and r['run']==arm and r['split']==('train' if split=='train' else 'retained') and r['region']==reg and (split!='retained_without_00000' or r['frame_id']!='00000')]
                    for k in metrics:
                        v=float(np.mean([float(r[k]) for r in rr if r[k]!='']))
                        max_aggregate=max(max_aggregate,abs(v-ev['scenes'][scene]['summaries'][arm][split][reg][k]))
        for arm in ['Q0','S','SL']:
            m=read(scene_dir(scene)/'evaluation'/arm/'manifest.json')
            for rr in m['rows']:
                f=rr['source_frame'];asset=rr['render'];assert sha(asset['path'])==asset['sha256']
                pred=np.clip(np.load(asset['path'])['rgb'].astype(np.float64),0,1)
                gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255
                labels=cv2.imread(f['mask_path'],0)>=128;err=((pred-gt)**2).mean(-1)
                for reg,mask in [('full',np.ones(labels.shape,bool)),('foreground',labels),('background',~labels)]:
                    r=idx[scene,arm,rr['split'],rr['frame_id'],reg]
                    assert int(r['pixels'])==int(mask.sum())
                    if mask.any():
                        value=float(-10*np.log10(max(float(err[mask].mean()),1e-12)))
                        max_psnr=max(max_psnr,abs(value-float(r['psnr_db'])));checked_psnr+=1
    assert max_psnr<1e-10 and max_aggregate<1e-12
    assets=[];completed=0;model_checks=[]
    for scene in config()['scenes']:
        for mode in ['BG','Q0','S','SL']:
            sd=scene_dir(scene);out=sd/'runs'/mode;r=read(out/'run.json')
            s=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=False)
            target=config()['schedule']['coarse_updates' if mode=='BG' else 'fine_updates']
            assert s['completed_updates']==s['optimizer_updates']==target and s['phase']=='after_Adam_and_zero_grad'
            completed+=target
            n=len(s['model'][1]);assert n==len(s['model'][3])==len(s['deformation_accum'])
            assert torch.isfinite(s['model'][1]).all()
            assert s['model'][3].dtype==torch.bool
            if mode=='BG':assert not s['model'][3].any()
            for a in s['metadata']['source_code']:assert sha(a['path'])==a['sha256'],a['path']
            if mode!='BG':
                middle=out/f"checkpoint_fine_{config()['schedule']['middle_snapshot']:06d}.pt"
                assert middle.exists();assets.append(identity(middle))
            assets.extend([r['checkpoint'],identity(out/'effective_config.json')])
            model_checks.append(dict(scene=scene,mode=mode,completed_updates=target,points=n,variable_points=int(s['model'][3].sum())))
    formal=[json.loads(x) for x in (RUN/'protocol/formal_attempts.jsonl').read_text().splitlines()]
    diagnostic=[json.loads(x) for x in (RUN/'protocol/diagnostic_attempts.jsonl').read_text().splitlines()]
    extra=[json.loads(x) for x in (RUN/'protocol/extra_backwards.jsonl').read_text().splitlines()]
    actual_diagnostic_updates=read(RUN/'protocol/module_acceptance.json')['diagnostic_Adam_updates']
    assert completed==config()['budgets']['normal_formal_updates']
    assert completed<=len(formal)<=config()['budgets']['formal_attempts']
    assert actual_diagnostic_updates<=config()['budgets']['integrated_Adam_updates'] and len(extra)<=config()['budgets']['extra_no_update_backwards']
    gpu=[json.loads(x) for x in (RUN/'protocol/GPU_jobs.jsonl').read_text().splitlines()]
    assert sum(x['wall_seconds'] for x in gpu)<=config()['budgets']['GPU_seconds']
    save_json(RUN/'model_index.json',dict(assets=assets,models=model_checks))
    save_json(RUN/'protocol/independent_verification.json',dict(status='passed',CPU_seconds=time.monotonic()-start,
        rows=len(rows),paired_rows=len(diffs),PSNR_recomputed=checked_psnr,max_PSNR_difference=max_psnr,
        max_pair_difference=max_pair,max_aggregate_difference=max_aggregate,formal_updates=completed,
        formal_attempts=len(formal),replay_attempts=len(formal)-completed,diagnostic_attempts=len(diagnostic),
        diagnostic_Adam_updates=actual_diagnostic_updates,
        no_update_backwards=len(extra),SSIM_LPIPS_scope='Original evaluator executed once, independent aggregation and pairing only',
        terminal_identities_and_training_source_hashes=True))


if __name__=='__main__':run()
