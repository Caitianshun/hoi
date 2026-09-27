"""One CPU audit of saved vectors, costs and complete logical history."""
from common import *
import csv,math,time,subprocess,collections,numpy as np,cv2
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(path,data):
    keys=list(dict.fromkeys(k for r in data for k in r))
    with Path(path).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def run():
    start=time.monotonic();freeze=read(RUN/'protocol/finals.json');summary=read(RUN/'evaluation_summary.json')
    for x in freeze['assets']:assert sha(x['path'])==x['sha256']
    frozen=read(RUN/'protocol/engineering_protocol_frozen.json')
    for x in frozen['frozen_sources']:assert sha(x['path'])==x['sha256'],x['path']
    frames={}
    for split,name in [('train','manifest.json'),('retained','evaluation_manifest.json')]:
        frames.update({(split,x['frame_id']):x for x in read(OLD/'inputs/hos_backpack'/name)['frames']})
    ms=rows(RUN/'metrics_per_frame.csv');ds=rows(RUN/'paired_differences.csv');idx={(r['run'],r['split'],r['frame_id'],r['region']):r for r in ms}
    expected={(b,s,f,reg) for b in ['B_U','B_F','M1'] for s,f in frames for reg in ['full','foreground','background']}
    assert len(ms)==len(idx) and set(idx)==expected
    assert len(ds)==len({(r['comparison'],r['split'],r['frame_id'],r['region']) for r in ds})==2*284*3
    for d in ds:
        left,right=d['comparison'].split('-');key=d['split'],d['frame_id'],d['region'];a=idx[(left,)+key];b=idx[(right,)+key];assert a['pixels']==b['pixels']
        for k in ['psnr_db','ssim','lpips_spatial_mean']:assert abs(float(d[k])-(float(a[k])-float(b[k])))<1e-10
    errors=[];rawerrors=[];raw_assets=[]
    # Only newly computed M1 PSNR is rederived. Historical full SSIM/LPIPS is not re-evaluated.
    for (split,fid),f in frames.items():
        row=idx['M1',split,fid,'full'];asset=identity(row['raw_render_path']);assert asset['sha256']==row['raw_render_sha256'];raw_assets.append(dict(run='M1',split=split,frame_id=fid,**asset))
        raw=np.load(asset['path'])['rgb'];assert raw.dtype==np.float32 and np.isfinite(raw).all()
        gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255.;fg=cv2.imread(f['mask_path'],0)>=128
        error=np.square(np.clip(raw.astype(np.float64),0,1)-gt).mean(-1);rawerror=np.square(raw.astype(np.float64)-gt).mean(-1)
        for reg,mask in [('full',np.ones(fg.shape,bool)),('foreground',fg),('background',~fg)]:
            r=idx['M1',split,fid,reg];m=float(error[mask].mean());p=-10*math.log10(max(m,1e-12));d=abs(p-float(r['psnr_db']));assert d<1e-7;errors.append(d)
            d=abs(float(rawerror[mask].mean())-float(r['raw_mse']));assert d<1e-9;rawerrors.append(d)
    for b in ['B_U','B_F','M1']:
        for split in ['train','retained','retained_without_00000']:
            for reg in ['full','foreground','background']:
                rr=[r for r in ms if r['run']==b and r['split']==('train' if split=='train' else 'retained') and r['region']==reg and (split!='retained_without_00000' or r['frame_id']!='00000')]
                for k in ['psnr_db','ssim','lpips_spatial_mean']:assert abs(sum(float(x[k]) for x in rr)/len(rr)-summary['summaries'][b][split][reg][k])<1e-10
    D=lines(RUN/'protocol/D_steps.jsonl');C=lines(RUN/'protocol/C_steps.jsonl');calls=lines(RUN/'protocol/optimizer_calls.jsonl');kernels=lines(RUN/'protocol/kernel_calls.jsonl');jobs=lines(RUN/'protocol/gpu_jobs.jsonl');cfg=read(RUN/'configs/v7.json')
    count=collections.Counter((x['purpose'],x['event']) for x in calls)
    assert count['diagnostic','enter']==count['diagnostic','completed']==len(D)<=cfg['budgets']['D_attempts']
    assert count['formal','enter']==count['formal','completed'] and len(C)<=cfg['budgets']['C_attempts']
    logical={x['iteration'] for x in C};assert logical==set(range(1,14001))
    formal_dirs=[p for p in [RUN/'runs/M1',RUN/'runs/M1_resume'] if p.exists()]
    sampling=[];training=[];density=[];bound=[];states=[]
    for p in formal_dirs:
        for name,target in [('sampling_order.jsonl',sampling),('training_metrics.jsonl',training),('density_events.jsonl',density)]:target.extend(dict(process=p.name,**x) for x in lines(p/name))
        bound.extend(rows(p/'bound_events_compact.csv'))
        for x in lines(p/'checkpoint_index.jsonl'):
            asset=x['checkpoint'];exists=Path(asset['path']).exists()
            states.append(dict(iteration=x['iteration'],asset=asset,status='retained' if exists else 'rolling_superseded_by_two_newest'))
    assert len(training)==len(sampling)==len(C) and len(bound)==2*len(C)
    sequence={x['iteration']:x['frame_ids'] for x in sampling};reference={x['iteration']:x['frame_ids'] for x in lines(V5/'runs/B_U/sampling_order.jsonl')}
    differences=[dict(iteration=i,M1=sequence[i],B_U=reference[i]) for i in sorted(sequence) if sequence[i]!=reference[i]]
    save_json(RUN/'sampling_comparison.json',dict(logical_rounds=len(sequence),actual_batches=len(sampling),replayed_batches=len(sampling)-len(sequence),divergent_rounds=len(differences),first_difference=differences[0] if differences else None,all_differences=differences))
    for name,data in [('sampling_order.jsonl',sampling),('training_metrics.jsonl',training),('density_events.jsonl',density)]:
        with (RUN/name).open('w') as f:
            for x in data:f.write(json.dumps(x)+'\n')
    csvwrite(RUN/'bound_events_compact.csv',bound)
    counters=collections.Counter((x['mode'],x['kind']) for x in kernels);seconds=sum(j['wall_seconds'] for j in jobs);dseconds=sum(j['wall_seconds'] for j in jobs if j['category']=='D')
    assert seconds<=cfg['budgets']['total_GPU_seconds'] and dseconds<=cfg['budgets']['D_GPU_seconds']
    assert counters['diagnostic','render']<=64 and counters['diagnostic','backward']<=64 and counters['terminal_evaluation','render']==284
    final=read(RUN/'formal_terminal.json');result=final['run']
    cost=dict(status='all_processes_closed',D_attempts=len(D),D_optimizer_calls=count['diagnostic','completed'],C_actual_attempts=len(C),C_logical_rounds=len(logical),C_optimizer_calls=count['formal','completed'],C_replay_attempts=len(C)-len(logical),terminal_checkpoint_optimizer_updates=result['optimizer_updates'],GPU_task_seconds=seconds,GPU_task_hours=seconds/3600,D_GPU_task_seconds=dseconds,diagnostic_renders=counters['diagnostic','render'],diagnostic_backwards=counters['diagnostic','backward'],terminal_evaluation_renders=counters['terminal_evaluation','render'],CPU_evaluation_seconds=summary['seconds'],jobs=jobs,peak_allocated_bytes=result['peak_allocated_bytes'],peak_reserved_bytes=result['peak_reserved_bytes'],final_points=result['final_points'],peak_points=result['peak_points'],physical_GPU='GPU1 RTX3090',measurement='Process wall time including loading and serialization, not CUDA kernel time',nvidia_continuous_PID_peak=None,limits=cfg['budgets'],historical_reused_costs=identity(V5/'costs.json'))
    save_json(RUN/'costs.json',cost)
    source=[identity(p) for p in sorted((RUN/'code').glob('*.py'))]+[identity(ROOT/'hoi_modules/attribute_gradient_router.py'),identity(RUN/'configs/v7.json')]
    save_json(RUN/'source_manifest.json',dict(start=read(RUN/'protocol/engineering_protocol_frozen.json'),current_source_files=source,end_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    large=[]
    for p in sorted((RUN/'diagnostics').rglob('*.pt')):large.append(identity(p))
    save_json(RUN/'state_manifest.json',dict(parent_assets=read(RUN/'route_protocol.json')['parent_assets'],terminal=identity(result['checkpoint']),checkpoint_history=states,diagnostic_states_and_gradients=large,raw_renders_index=identity(RUN/'protocol/raw_render_index.json'),dataset_index=identity(V6/'protocol/input_hash_verification.json'),environment=identity(RUN/'environment.json')))
    save_json(RUN/'run.json',dict(status='complete_training_evaluation_pending_artifact_QA',formal=final,acceptance=identity(RUN/'module_acceptance.json'),costs=identity(RUN/'costs.json'),quality=identity(RUN/'quality_decision.json'),automatic_new_training=False))
    save_json(RUN/'protocol/independent_verification.json',dict(status='pass',metric_rows=len(ms),paired_rows=len(ds),M1_PSNR_rows_recomputed=len(errors),max_PSNR_difference=max(errors),max_raw_MSE_difference=max(rawerrors),summary_and_pairs_recomputed=True,budget_verified=True,all_NA_accurate=True,source_frozen_hashes_passed=True,seconds=time.monotonic()-start))
    print('Independent CPU verification complete',cost)
if __name__=='__main__':run()
