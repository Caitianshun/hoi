"""Verify available outcomes, preserve missing rows, and package tiny raw windows."""
from common import *
import csv,time,cv2,numpy as np
from evaluate_frozen import InputRegions

def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def run():
    start=time.monotonic();frozen=read(RUN/'protocol/training_frozen.json');final=read(RUN/'protocol/finals.json')
    checked=[]
    for a in frozen['assets']+frozen['training_sources']+final['assets']:
        assert sha(a['path'])==a['sha256'],a['path'];checked.append(a)
    assert sha(RUN/'code/train_stage_audited.py')==read(RUN/'protocol/failure_resolution.json')['observer']['sha256']
    ex=read(RUN/'protocol/fixed_examples.json');summary=read(RUN/'evaluation/summary.json')
    metrics=rows(RUN/'metrics_per_frame.csv');paired=rows(RUN/'paired_differences.csv');app=rows(RUN/'appearance_diagnostics.csv')
    mindex={(x['run'],x['split'],x['frame_id'],x['region']):x for x in metrics}
    assert len(mindex)==len(metrics)==3*284*3
    frames={};old_index={}
    for split,name in [('train','manifest.json'),('retained','evaluation_manifest.json')]:
        for f in read(OLD/'inputs/hos_backpack'/name)['frames']:frames[split,f['frame_id']]=f
    old=read(OLD/'evaluation/H1/manifest.json')
    for r in old['frames']:old_index['train' if r['group']=='input_fit' else 'retained',r['frame_id']]=r
    # Recompute PSNR/SSE on all retained images from full-precision arrays.
    differences=[];arrays={}
    for key,f in frames.items():
        if key[0]!='retained':continue
        gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255;fg=cv2.imread(f['mask_path'],0)>=128
        for branch in ['H1','W_fine','W_all']:
            r=mindex[branch,*key,'full']
            if r['status']!='completed':
                assert all(mindex[branch,*key,reg]['psnr_db']=='' for reg in ['full','foreground','background']);continue
            assert sha(r['raw_render_path'])==r['raw_render_sha256'];raw=np.load(r['raw_render_path'])['rgb'];assert raw.dtype==np.float32 and np.isfinite(raw).all()
            err=np.square(np.clip(raw.astype(np.float64),0,1)-gt).mean(-1)
            for reg,mask in [('full',np.ones(fg.shape,bool)),('foreground',fg),('background',~fg)]:
                row=mindex[branch,*key,reg];sse=float(err[mask].sum());psnr=-10*np.log10(max(sse/int(mask.sum()),1e-12))
                delta=abs(psnr-float(row['psnr_db']));assert delta<1e-7;assert abs(sse-float(row['sse_rgb_mean']))<1e-6;differences.append(delta)
    for d in paired:
        left,right=d['comparison'].split('-');k=d['split'],d['frame_id'],d['region'];a=mindex[left,*k];b=mindex[right,*k]
        assert a['pixels']==b['pixels']
        for metric in ['psnr_db','ssim','lpips_spatial_mean']:
            if not a[metric] or not b[metric]:assert d[metric]==''
            else:assert abs(float(d[metric])-(float(a[metric])-float(b[metric])))<1e-10
    assert len(app)==54*4 and len({(r['dev'],r['group'],r['frame_id'],r['mode'],r['region']) for r in app})==len(app)
    # Aggregate actual append-only logs, never replace failed attempts with zero cost.
    counts={}
    for filename in ['training_metrics.jsonl','density_events.jsonl']:
        with (RUN/filename).open('w') as out:
            count=0
            for branch in ['W_fine','W_all']:
                for line in (RUN/'runs'/branch/filename).read_text().splitlines():
                    item=json.loads(line);out.write(json.dumps(dict(run=branch,**item))+'\n');count+=1
            counts[filename]=count
    ledger=read(RUN/'protocol/gpu_cost_ledger.json');attempts=[]
    for branch in ['W_fine','W_all']:
        rd=RUN/'runs'/branch;result=read(rd/('run.json' if (rd/'run.json').exists() else 'failure.json'))
        attempts.append(dict(run=branch,**result,terminal_model_bytes=Path(result['checkpoint']).stat().st_size if result.get('checkpoint') else None))
    nominal=len((RUN/'protocol/formal_steps.jsonl').read_text().splitlines());temp=len((RUN/'protocol/temporary_steps.jsonl').read_text().splitlines());gpu_seconds=sum(r['wall_seconds'] for r in ledger)
    assert nominal<=31000 and temp<=40 and gpu_seconds<=10800
    costs=dict(status='available_attempts_accounted',formal_attempts=attempts,formal_attempt_count=len(attempts),formal_nominal_attempts=nominal,formal_optimizer_updates=sum(a['optimizer_updates'] for a in attempts),temporary_nominal_steps=temp,
        gpu_task_wall_seconds=gpu_seconds,gpu_task_hours=gpu_seconds/3600,measurement='Serial GPU process wall time, includes imports/load/export/failed attempts. Not CUDA kernel time.',
        GPU='physical1 RTX3090',peak_PID_MiB=max((r.get('peak_process_nvidia_MiB') or 0) for r in ledger),jobs=[{k:v for k,v in r.items() if k!='samples'} for r in ledger],
        CPU=dict(initial_support=read(RUN/'initialization_support.json')['seconds'],appearance_evaluation=read(RUN/'diagnostics/appearance/summary.json')['seconds'],hos_evaluation=summary['seconds']),
        historical=dict(H1_coarse_steps_reused_for_W_fine=3000,V3_costs=identity(OLD/'costs.json'),note='Historical cost is reused, not zero and not charged again to V4'),limits=dict(formal_attempts=2,nominal_steps=31000,temporary_steps=40,GPU_seconds=10800))
    save_json(RUN/'costs.json',costs)
    # Small windows were fixed before training; preserve float32 values without quantization.
    out=RUN/'feedback_arrays';out.mkdir(exist_ok=True);windows=[];regions={d:InputRegions(d) for d in ['dev1','dev2']}
    am=read(RUN/'diagnostics/appearance/manifest.json')['records']
    for example in ex['small_array_windows']:
        dataset=example['dataset'];fid=example['frame_id'];sources={};missing=[]
        if dataset=='hos_backpack':
            frame=frames['retained',fid];mask=cv2.imread(frame['mask_path'],0);gt=cv2.imread(frame['image_path'])[...,::-1].astype(np.float32)/np.float32(255)
            for branch in ['H1','W_fine','W_all']:
                row=mindex[branch,'retained',fid,'full']
                if row['status']!='completed':missing.append(branch);continue
                path=Path(row['raw_render_path']);z=np.load(path);sources[branch]=dict(rgb=z['rgb'],alpha=np.load(RUN/'evaluation/H1_alpha'/(fid+'.npz'))['alpha'] if branch=='H1' else z['alpha'],identity=identity(path))
        else:
            dev=dataset.removeprefix('behave_');records=[r for r in am if r['dev']==dev and r['group']=='camera1_E' and r['frame_id']==fid];frame=records[0]['source_frame']
            mask,_,_=regions[dev].get(dict(group='camera1_E',frame_id=fid,time_seconds=frame['time_seconds'],source_frame=frame));gt=cv2.imread(frame['image_path'])[...,::-1].astype(np.float32)/np.float32(255)
            for rec in records:
                z=np.load(rec['render']['path']);sources[rec['mode']]=dict(rgb=z['rgb'],alpha=z['alpha'],identity=rec['render'])
        for region,bounds in example['windows'].items():
            x0,y0,x1,y1=bounds;sl=np.s_[y0:y1,x0:x1];data=dict(GT=gt[sl],mask=mask[sl])
            for mode,s in sources.items():data[mode+'_raw']=s['rgb'][sl];data[mode+'_alpha']=s['alpha'][sl]
            for name,a in data.items():assert a.dtype==(np.uint8 if name=='mask' else np.float32),(name,a.dtype)
            path=out/f'{dataset}_{fid}_{region}.npz';np.savez_compressed(path,**data)
            with np.load(path) as saved:
                for name,a in data.items():assert np.array_equal(a,saved[name])
            windows.append(dict(**identity(path),dataset=dataset,frame_id=fid,window=region,bounds_xyxy=bounds,image_size=example['image_size'],missing_modes=missing,sources={k:s['identity'] for k,s in sources.items()},arrays={k:dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for k,a in data.items()}))
    save_json(RUN/'feedback_arrays/manifest.json',dict(status='available_modes_complete',windows=windows,selection=identity(RUN/'protocol/fixed_examples.json'),missing_not_fabricated=True))
    manifest=read(RUN/'run_manifest.json');manifest.update(status='closed_with_failed_branch' if final['failures'] else 'completed',finals=identity(RUN/'protocol/finals.json'),failure_resolution=identity(RUN/'protocol/failure_resolution.json'),costs=identity(RUN/'costs.json'),actual_optimizer_updates=costs['formal_optimizer_updates'],actual_nominal_attempts=nominal,available_terminals=list(final['runs']),failed_branches=final['failures'],observation_only_entry=identity(RUN/'code/train_stage_audited.py'),mask_usage='Published training mask newly used for weighted RGB loss; unchanged pixels')
    save_json(RUN/'run_manifest.json',manifest)
    save_json(RUN/'protocol/verification.json',dict(status='passed_for_available_outcomes_missing_explicit',inherited_and_frozen_assets=len(checked),max_retained_PSNR_recompute_error=max(differences),PSNR_region_rows_recomputed=len(differences),metric_rows=len(metrics),paired_rows=len(paired),appearance_rows=len(app),logs=counts,budget_verified=True,raw_windows=len(windows),seconds=time.monotonic()-start))
if __name__=='__main__':run()
