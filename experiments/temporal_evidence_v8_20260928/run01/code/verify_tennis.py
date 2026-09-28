"""CPU-only independent CSV/vector/PSNR, budget and paired-state audit."""
from common import *
import csv,time
import numpy as np,cv2
B=RUN/'tennis'
def run():
    started=time.monotonic();cv2.setNumThreads(4);metrics=['psnr_db','ssim','lpips_spatial_mean']
    rows=list(csv.DictReader((B/'metrics_per_frame.csv').open()));diffs=list(csv.DictReader((B/'paired_differences.csv').open()));ev=read(B/'evaluation_summary.json')
    train=read(B/'inputs/hos_tennis/manifest.json');dev=read(B/'inputs/hos_tennis/evaluation_manifest.json');total=len(train['frames'])+len(dev['frames'])
    assert len(rows)==6*total and len(diffs)==3*total
    idx={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};assert len(idx)==len(rows)
    for d in diffs:
        a=idx['B_Q',d['split'],d['frame_id'],d['region']];b=idx['B_U',d['split'],d['frame_id'],d['region']]
        for k in metrics:
            if d[k]=='':assert a[k]=='' or b[k]==''
            else:assert abs(float(a[k])-float(b[k])-float(d[k]))<1e-12
    for arm in ['B_U','B_Q']:
        for split,regions in ev['summaries'][arm].items():
            for reg,summary in regions.items():
                rr=[r for r in rows if r['run']==arm and r['region']==reg and r['split']==('train' if split=='train' else 'retained') and (split!='retained_without_00000' or r['frame_id']!='00000')]
                for k in metrics:
                    vv=[float(r[k]) for r in rr if r[k]!='']
                    if vv:assert abs(float(np.mean(vv))-summary[k])<1e-12
    max_error=0.;count=0
    for split,manifest in [('train',train),('retained',dev)]:
        for f in manifest['frames']:
            gt=cv2.imread(f['image_path'])[...,::-1].astype(np.float64)/255.;mask=cv2.imread(f['mask_path'],0)>=128
            for arm in ['B_U','B_Q']:
                raw=idx[arm,split,f['frame_id'],'full'];assert sha(raw['raw_render_path'])==raw['raw_render_sha256'];pred=np.clip(np.load(raw['raw_render_path'])['rgb'].astype(np.float64),0,1)
                for reg,sel in [('full',np.ones(mask.shape,bool)),('foreground',mask),('background',~mask)]:
                    row=idx[arm,split,f['frame_id'],reg]
                    if not sel.any():assert row['psnr_db']=='';continue
                    psnr=-10*np.log10(max(float(np.square(pred[sel]-gt[sel]).mean()),1e-12));max_error=max(max_error,abs(psnr-float(row['psnr_db'])));count+=1
    assert max_error<1e-10
    steps=[json.loads(x) for x in (B/'protocol/formal_steps.jsonl').read_text().splitlines()]
    calls=[json.loads(x) for x in (B/'protocol/optimizer_calls.jsonl').read_text().splitlines()];assert len(steps)==31000;assert sum(c['event']=='completed' for c in calls)==30998
    for arm,target,updates in [('coarse_U',3000,3000),('B_U',14000,13999),('B_Q',14000,13999)]:
        ss=[s['iteration'] for s in steps if s['run']==arm];assert ss==list(range(1,target+1));r=read(B/'runs'/arm/'run.json');assert r['attempted_this_process']==target and r['updates_this_process']==updates
    def batches(arm):return [json.loads(x)['frame_ids'] for x in (B/'runs'/arm/'sampling_order.jsonl').read_text().splitlines()]
    assert batches('B_U')==batches('B_Q')
    a=read(B/'runs/B_U/effective_config.json');b=read(B/'runs/B_Q/effective_config.json');assert a['parent']==b['parent'] and a['scale_bound']==b['scale_bound']==train['scene_extent']
    for x in read(B/'protocol/source_manifest.json')['assets']:assert sha(x['path'])==x['sha256']
    for x in read(B/'protocol/finals.json')['assets']:assert sha(x['path'])==x['sha256']
    jobs=[json.loads(s) for s in (B/'protocol/gpu_jobs.jsonl').read_text().splitlines()];seconds=sum(j['wall_seconds'] for j in jobs);assert all(j['returncode']==0 for j in jobs) and seconds<=28800
    save_json(B/'costs.json',dict(status='completed',GPU_task_wall_seconds=seconds,GPU_task_hours=seconds/3600,formal_attempts=len(steps),optimizer_updates=30998,diagnostic_updates=0,extra_backward=0,recovery_replays=0,coarse_shared=True,logical_iterations_each=17000,logical_updates_each=16999,jobs=jobs,CPU_evaluation_seconds=ev['seconds'],preprocessing=read(B/'protocol/preprocessing.json'),model_costs={n:read(B/'runs'/n/'run.json') for n in ['coarse_U','B_U','B_Q']}))
    save_json(B/'protocol/independent_verification.json',dict(status='pass',rows=len(rows),paired_rows=len(diffs),recomputed_PSNR=count,PSNR_max_absolute_error=max_error,all_metric_aggregates_and_differences_verified=True,SSIM_LPIPS_second_implementation=False,paired_RGB_sampling_exact=True,shared_parent_exact=True,source_and_terminal_hashes_unchanged=True,seconds=time.monotonic()-started))
if __name__=='__main__':run()
