"""One CPU integrity audit: metric vectors, shared history, state and costs."""
from common import *
import csv,time,math,collections,subprocess
import numpy as np,cv2

def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()] if Path(p).exists() else []
def main():
    start=time.monotonic();cfg=read(RUN/'configs/v8.json');freeze=read(RUN/'protocol/finals.json');summ=read(RUN/'evaluation_summary.json')
    for a in freeze['assets']:assert a==identity(a['path'])
    for a in read(RUN/'source_manifest.json')['assets']:assert a==identity(a['path']),a['path']
    rows=list(csv.DictReader((RUN/'metrics_per_frame.csv').open()));diffs=list(csv.DictReader((RUN/'paired_differences.csv').open()));idx={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};assert len(idx)==len(rows)==4*284*3;assert len(diffs)==6*284*3
    for d in diffs:
        left,right=d['comparison'].split('-');key=d['split'],d['frame_id'],d['region'];a=idx[(left,)+key];b=idx[(right,)+key]
        for k in ['psnr_db','ssim','lpips_spatial_mean']:assert abs(float(d[k])-float(a[k])+float(b[k]))<1e-10
    frames={(s,r['frame_id']):r for s,n in [('train','manifest.json'),('retained','evaluation_manifest.json')] for r in read(OLD/'inputs/hos_backpack'/n)['frames']};errors=[]
    for (split,fid),f in frames.items():
        gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255.;fg=cv2.imread(f['mask_path'],0)>=128
        for arm in ['B_Q','T_plain','T_mix']:
            row=idx[arm,split,fid,'full'];assert sha(row['raw_render_path'])==row['raw_render_sha256'];raw=np.load(row['raw_render_path'])['rgb'];err=(np.clip(raw.astype(np.float64),0,1)-gt)**2
            for reg,mask in [('full',np.ones(fg.shape,bool)),('foreground',fg),('background',~fg)]:
                value=-10*math.log10(max(float(err[mask].mean()),1e-12));d=abs(value-float(idx[arm,split,fid,reg]['psnr_db']));assert d<1e-7;errors.append(d)
    for arm in ['B_U','B_Q','T_plain','T_mix']:
        for split in ['train','retained','retained_without_00000']:
            for reg in ['full','foreground','background']:
                rr=[r for r in rows if r['run']==arm and r['split']==('train' if split=='train' else 'retained') and r['region']==reg and (split!='retained_without_00000' or r['frame_id']!='00000')]
                for k in ['psnr_db','ssim','lpips_spatial_mean']:assert abs(np.mean([float(x[k]) for x in rr])-summ['summaries'][arm][split][reg][k])<1e-10
    D=lines(RUN/'protocol/D_steps.jsonl');C=lines(RUN/'protocol/C_steps.jsonl');calls=lines(RUN/'protocol/optimizer_calls.jsonl');jobs=lines(RUN/'protocol/gpu_jobs.jsonl');counts=collections.Counter((x['purpose'],x['event']) for x in calls);assert len(D)==128==counts['diagnostic','completed']==counts['diagnostic','enter'];assert len(C)==40000<=cfg['budgets']['C_attempts'];assert counts['formal','enter']==counts['formal','completed']==39997
    temporal={};sampling={};states=[];runs={}
    baseline={x['iteration']:x['frame_ids'] for x in lines(V5/'runs/B_U/sampling_order.jsonl')};prefix=lines(RUN/'runs/B_Q/sampling_order.jsonl')[:1000]
    for arm in ['B_Q','T_plain','T_mix']:
        rd=RUN/'runs'/arm;runs[arm]=read(rd/'run.json');ss=lines(rd/'sampling_order.jsonl');logical=ss if arm=='B_Q' else prefix+ss;assert len(logical)==14000
        mismatches=[x['iteration'] for x in logical if baseline[x['iteration']]!=x['frame_ids']];assert not mismatches;sampling[arm]=dict(logical_batches=14000,actual_new_batches=len(ss),B_U_RGB_sequence_matches=True)
        temporal[arm]=lines(rd/'temporal_stats.jsonl')
        for r in lines(rd/'checkpoint_index.jsonl'):
            a=r['checkpoint'];states.append(dict(arm=arm,iteration=r['iteration'],asset=a,status='retained' if Path(a['path']).exists() else 'rolling_superseded'))
    assert [(x['iteration'],x['pair_id']) for x in temporal['T_plain']]==[(x['iteration'],x['pair_id']) for x in temporal['T_mix']]
    stats={}
    for arm in ['T_plain','T_mix']:
        rr=temporal[arm];stats[arm]=dict(calls=len(rr),empty_calls=sum(x.get('empty_reason') is not None for x in rr),teacher_valid=quantiles([x['teacher_valid'] for x in rr]),support=quantiles([x['supported'] for x in rr]),unweighted_error_px=quantiles([x['unweighted_error_px'] for x in rr if x.get('unweighted_error_px') is not None]),weighted_error_px=quantiles([x['weighted_error_px'] for x in rr if x.get('weighted_error_px') is not None]),gate_mean=quantiles([x['gate_mean'] for x in rr if x.get('gate_mean') is not None]),variance_mean_px2=quantiles([x['variance_mean_px2'] for x in rr if x.get('variance_mean_px2') is not None]),teacher_a=quantiles([x['a'] for x in rr if 'a' in x]))
    save_json(RUN/'temporal_summary.json',stats);save_json(RUN/'sampling_comparison.json',dict(arms=sampling,T_pair_schedule_exact=True))
    seconds=sum(x['wall_seconds'] for x in jobs);assert seconds<=28800
    extras=lines(RUN/'protocol/extra_backward.jsonl');assert len(extras)<=64
    metriclog=OLD/'runs/hos_backpack_formal/train_metrics.jsonl';oldrows=lines(metriclog)
    coarse=[r for r in oldrows if r.get('stage')=='coarse'];coarse_elapsed=None
    if coarse:coarse_elapsed=coarse[-1].get('seconds')
    costs=dict(status='V8A_jobs_closed',formal_actual_attempts=len(C),formal_Adam=counts['formal','completed'],diagnostic_Adam=len(D),extra_no_update_backward_calls=len(extras),GPU_task_seconds=seconds,GPU_task_hours=seconds/3600,CPU_evaluation_seconds=summ['seconds'],jobs=jobs,runs=runs,shared_prefix=dict(physical_new_Q_prefix=1000,reused_by_two_T=1000,extra_prefix_retraining_steps=0),logical_cost_per_arm=dict(coarse_attempts=3000,fine_attempts=14000,fine_Adam=13999,total_attempts=17000,total_Adam=16999,old_coarse_elapsed_seconds=coarse_elapsed,old_coarse_cost_note='Original V3 train log elapsed at coarse3000 before density/save; not isolated child-process wall time. Coarse was shared and not newly rerun; null means unavailable, not zero.'),old_costs=identity(V5/'costs.json'),measurement='Actual GPU child-process wall time including load/save and all failed jobs; CPU evaluation separate')
    save_json(RUN/'costs.json',costs);save_json(RUN/'state_manifest.json',dict(history=states,parents=read(RUN/'protocol.json')['parents'],Q_prefix=identity(RUN/'runs/B_Q/checkpoint_fine_001000.pt'),terminals=freeze,cache_manifest=identity(RUN/'track_cache_manifest.json'),large_raw=identity(RUN/'protocol/raw_render_index.json')))
    save_json(RUN/'protocol/independent_verification.json',dict(status='pass',metric_rows=len(rows),paired_rows=len(diffs),new_PSNR_rows_recomputed=len(errors),max_PSNR_difference=max(errors),all_summary_and_pair_values_checked=True,SSIM_LPIPS_independently_recomputed=False,independent_SSIM_LPIPS_scope='CSV aggregation only; original evaluator executed once',budgets_and_sampling_verified=True,seconds=time.monotonic()-start))
    save_json(RUN/'run.json',dict(status='V8A_evaluated_and_verified_pending_quality_assessment_and_delivery',costs=identity(RUN/'costs.json'),verification=identity(RUN/'protocol/independent_verification.json'),quality=identity(RUN/'quality_decision.json')))
    print('V8A CPU integrity audit passed')
if __name__=='__main__':main()
