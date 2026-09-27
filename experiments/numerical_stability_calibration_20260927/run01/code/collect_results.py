"""Independent CPU verification, cost accounting and fixed float32 windows."""
from common import *
import csv,time,cv2,numpy as np,torch,zipfile
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()] if Path(p).exists() else []
def finite_tree(v):
    if torch.is_tensor(v):return bool(torch.isfinite(v).all())
    if isinstance(v,dict):return all(finite_tree(x) for x in v.values())
    if isinstance(v,(tuple,list)):return all(finite_tree(x) for x in v)
    return True
def run():
    started=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(4)
    final=read(RUN/'protocol/finals.json');gate=read(RUN/'protocol/formal_gate.json');summary=read(RUN/'evaluation/summary.json')
    checked=[]
    for a in gate['frozen_source_files']+gate['frozen_assets']+final['assets']:
        assert sha(a['path'])==a['sha256'],a['path'];checked.append(a)
    frames={};source=read(OLD/'protocol/hos_asset_sources.json');archive_verified=False;input_checks=[]
    for split,name in [('train','manifest.json'),('retained','evaluation_manifest.json')]:
        for f in read(OLD/'inputs/hos_backpack'/name)['frames']:
            for kind in ['image','mask']:
                actual=sha(f[kind+'_path']);expected_hash=f.get(kind+'_sha256');origin='frozen train manifest'
                if expected_hash is None:
                    assert split=='retained'
                    if not archive_verified:
                        assert sha(source['archive_path'])==source['archive_sha256'];archive_verified=True
                    member='Backpack/'+('images' if kind=='image' else 'masks')+'/'+Path(f[kind+'_path']).name
                    with zipfile.ZipFile(source['archive_path']) as z:expected_hash=hashlib.sha256(z.read(member)).hexdigest()
                    origin='member of original SHA-frozen V3 archive'
                assert actual==expected_hash
                input_checks.append(dict(path=f[kind+'_path'],sha256=actual,expected_source=origin))
            frames[split,f['frame_id']]=f
    save_json(RUN/'protocol/input_hash_verification.json',dict(inputs=input_checks,original_archive=source['archive_path'],archive_sha256=source['archive_sha256'],archive_rehashed=archive_verified))
    assert len(frames)==284
    metrics=rows(RUN/'metrics_per_frame.csv');paired=rows(RUN/'paired_differences.csv');index={(r['run'],r['split'],r['frame_id'],r['region']):r for r in metrics}
    expected={(branch,split,fid,reg) for branch in ['H1','B_U','B_F'] for split,fid in frames for reg in ['full','foreground','background']}
    assert len(index)==len(metrics)==len(expected) and index.keys()==expected
    errors=[]
    for (split,fid),f in frames.items():
        if split!='retained':continue
        gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255.;fg=cv2.imread(f['mask_path'],0)>=128
        for branch in ['H1','B_U','B_F']:
            row=index[branch,split,fid,'full']
            if row['status']!='completed':
                assert all(index[branch,split,fid,reg]['psnr_db']=='' for reg in ['full','foreground','background']);continue
            assert sha(row['raw_render_path'])==row['raw_render_sha256'];raw=np.load(row['raw_render_path'])['rgb'];assert raw.dtype==np.float32 and np.isfinite(raw).all()
            err=np.square(np.clip(raw.astype(np.float64),0,1)-gt).mean(-1);rawerr=np.square(raw.astype(np.float64)-gt).mean(-1)
            for region,mask in [('full',np.ones(fg.shape,bool)),('foreground',fg),('background',~fg)]:
                r=index[branch,split,fid,region];sse=float(err[mask].sum());mse=sse/int(mask.sum());p=-10*np.log10(max(mse,1e-12));d=abs(p-float(r['psnr_db']));assert d<1e-7;assert abs(sse-float(r['sse_rgb_mean']))<1e-6;assert abs(float(rawerr[mask].mean())-float(r['raw_mse']))<1e-9;errors.append(d)
    assert len(paired)==len({(r['comparison'],r['split'],r['frame_id'],r['region']) for r in paired})==3*284*3
    for d in paired:
        l,r=d['comparison'].split('-');key=d['split'],d['frame_id'],d['region'];a=index[(l,)+key];b=index[(r,)+key];assert a['pixels']==b['pixels']
        for k in ['psnr_db','ssim','lpips_spatial_mean']:
            if not a[k] or not b[k]:assert d[k]==''
            else:assert abs(float(d[k])-(float(a[k])-float(b[k])))<1e-10
    attempts=[];models=[];sampling={};aggregate={k:[] for k in ['training_metrics.jsonl','density_events.jsonl']}
    for branch in ['B_U','B_F']:
        rd=RUN/'runs'/branch;state=final['attempts'][branch];result=read(rd/'run.json') if branch in final['runs'] else read(rd/'failure.json') if (rd/'failure.json').exists() else {}
        events=lines(rd/'replay_events.jsonl');enters=sum(e['phase']=='Adam_enter' for e in events);exits=sum(e['phase']=='after_Adam' for e in events);updates=result.get('optimizer_updates');interval=[updates,updates] if updates is not None else [exits,enters]
        tr=[e for e in events if e['phase']=='scale_bound'];total=sum(e['total_axes'] for e in tr);trigger=sum(e['triggered_axes'] for e in tr)
        attempted=sum(x.get('run')==branch for x in lines(RUN/'protocol/B_steps.jsonl'))
        for name in aggregate:
            aggregate[name].extend(dict(run=branch,**r) for r in lines(rd/name))
        sampling[branch]=lines(rd/'sampling_order.jsonl')
        attempts.append(dict(run=branch,status=state['status'],attempted_rounds=attempted,optimizer_updates=updates,optimizer_updates_interval=interval,result=result,scale_axes_total=total,scale_axes_triggered=trigger,scale_axis_trigger_fraction=trigger/total if total else None,first_bad=identity(rd/'first_bad_tensor.json') if (rd/'first_bad_tensor.json').exists() else None))
        if branch in final['runs']:
            model=torch.load(result['checkpoint'],map_location='cpu',weights_only=False);assert model['iteration']==14000 and model['phase']=='completed_step' and not model['resumable'];assert finite_tree(model['model']) and finite_tree(model['deformation_accum']);assert model['optimizer_updates']==13999
            models.append(dict(run=branch,checkpoint=identity(result['checkpoint']),all_parameters_Adam_buffers_finite=True,resumable=False,reason='Original final iteration backward without final step/zero_grad',points=result['final_points']));del model
    divergence=[]
    for a,b in zip(sampling['B_U'],sampling['B_F']):
        assert a['iteration']==b['iteration']
        if a['frame_ids']!=b['frame_ids']:divergence.append(dict(iteration=a['iteration'],B_U=a['frame_ids'],B_F=b['frame_ids']))
    save_json(RUN/'sampling_comparison.json',dict(common_rounds=min(map(len,sampling.values())),divergent_rounds=len(divergence),first_difference=divergence[0] if divergence else None,all_differences=divergence,explanation='Original density operations can change RNG consumption; seed/shared start do not promise common batches after topology diverges'))
    for name,rs in aggregate.items():
        with (RUN/name).open('w') as f:
            for row in rs:f.write(json.dumps(row)+'\n')
    jobs=lines(RUN/'protocol/gpu_jobs.jsonl');A=len(lines(RUN/'protocol/A_steps.jsonl'));B=len(lines(RUN/'protocol/B_steps.jsonl'));P=len(lines(RUN/'protocol/A_probes.jsonl'));seconds=sum(j['wall_seconds'] for j in jobs);Asec=sum(j['wall_seconds'] for j in jobs if j['category']=='A')
    assert A<=512 and B<=28000 and P<=64 and Asec<=1800 and seconds<=10800
    cost=dict(status='all_attempts_accounted',A_attempts=A,A_no_optimizer_probes=P,A_GPU_seconds=Asec,B_attempts=B,B_optimizer_updates_interval=[sum(a['optimizer_updates_interval'][i] for a in attempts) for i in [0,1]],formal_attempts=attempts,GPU_task_seconds=seconds,GPU_task_hours=seconds/3600,jobs=jobs,measurement='Serial GPU job process wall time includes imports loading serialization and failures; first interrupted launcher uses conservative upper bound; not CUDA kernel time',physical_GPU='GPU1 RTX3090',CPU_evaluation_seconds=summary['seconds'],historical=dict(H1_coarse=identity(OLD/'runs/hos_backpack_formal/checkpoint_coarse_003000.pt'),V3_costs=identity(OLD/'costs.json'),V4_costs=identity(V4/'costs.json'),note='Reused cost, not zero and not recharged'),limits=read(RUN/'configs/v5.json')['budgets'],nvidia_PID_peak_MiB=None,nvidia_PID_peak_limitation='No continuous PID memory trace; torch allocated/reserved peaks recorded in completed run.json')
    save_json(RUN/'costs.json',cost);save_json(RUN/'model_index.json',dict(models=models,shared_initial=read(RUN/'protocol/fine_transition.json'),A_states=identity(RUN/'state_manifest.json')))
    examples=read(RUN/'protocol/fixed_examples.json');out=RUN/'feedback_arrays';out.mkdir(exist_ok=True);windows=[]
    for example in examples['small_array_windows']:
        if example['dataset']!='hos_backpack':continue
        fid=example['frame_id'];f=frames['retained',fid];gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float32)/np.float32(255);mask=cv2.imread(f['mask_path'],0);sources={}
        for branch in ['H1',*final['runs']]:
            row=index[branch,'retained',fid,'full'];z=np.load(row['raw_render_path']);alpha=np.load(V4/'evaluation/H1_alpha'/(fid+'.npz'))['alpha'] if branch=='H1' else z['alpha'];sources[branch]=dict(raw=z['rgb'],alpha=alpha,source=identity(row['raw_render_path']))
        for region,bounds in example['windows'].items():
            x0,y0,x1,y1=bounds;sl=np.s_[y0:y1,x0:x1];data=dict(GT=gt[sl],mask=mask[sl])
            for branch,s in sources.items():data[branch+'_raw']=s['raw'][sl];data[branch+'_alpha']=s['alpha'][sl]
            assert all(v.dtype==(np.uint8 if k=='mask' else np.float32) for k,v in data.items())
            p=out/f'Backpack_{fid}_{region}.npz';np.savez_compressed(p,**data)
            with np.load(p) as z:assert all(np.array_equal(z[k],v) for k,v in data.items())
            windows.append(dict(**identity(p),frame_id=fid,region=region,bounds_xyxy=bounds,sources={k:v['source'] for k,v in sources.items()},arrays={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in data.items()}))
    save_json(out/'manifest.json',dict(windows=windows,selection=identity(RUN/'protocol/fixed_examples.json'),missing_modes=[b for b in ['B_U','B_F'] if b not in final['runs']],full_precision_float32=True))
    save_json(RUN/'protocol/verification.json',dict(status='passed_for_available_outcomes',frozen_assets_rehashed=len(checked),source_RGB_mask_hashes=2*len(frames),metrics_rows=len(metrics),paired_rows=len(paired),retained_PSNR_rows_recomputed=len(errors),max_PSNR_difference=max(errors),all_paired_differences_checked=True,all_available_terminal_states_finite=True,small_windows=len(windows),budget_verified=True,seconds=time.monotonic()-started))
    print('Independent checks complete',cost['GPU_task_hours'])
if __name__=='__main__':run()
