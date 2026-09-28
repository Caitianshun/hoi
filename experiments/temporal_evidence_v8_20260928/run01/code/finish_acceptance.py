"""One finite-density regression and exact resume/sampling acceptance."""
from common import *
import numpy as np

def records(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def main():
    main=RUN/'diagnostics/regression';res=RUN/'diagnostics/resume';a=read(main/'run.json');b=read(res/'run.json');assert a['status']==b['status']=='completed';assert a['updates_this_process']==112 and b['updates_this_process']==16
    entry=read(res/'restore_verification.json');assert entry['model_Adam_groups_steps_buffers_exact'] and entry['RNG_exact'] and entry['stack_exact']
    original=records(main/'sampling_order.jsonl');replay=records(res/'sampling_order.jsonl');assert original[-16:]==replay
    origpair=records(main/'temporal_stats.jsonl');respair=records(res/'temporal_stats.jsonl');assert [(x['iteration'],x['pair_id'],x['pair_cursor']) for x in origpair[-4:]]==[(x['iteration'],x['pair_id'],x['pair_cursor']) for x in respair]
    density=records(main/'density_events.jsonl');assert any(x['iteration']==1100 for x in density)
    metrics=records(main/'training_metrics.jsonl');ms=np.array([r['iter_gpu_ms'] for r in metrics]);pairms=[r['iter_gpu_ms'] for r in metrics if r['temporal'] is not None];rgbms=[r['iter_gpu_ms'] for r in metrics if r['temporal'] is None]
    used=sum(r['wall_seconds'] for r in records(RUN/'protocol/gpu_jobs.jsonl'));mean=float(ms.mean())/1000
    # 1.5x mean admits later point growth 240k->~360k; add measured setup/save plus render allowance.
    estimate=used+mean*40000*1.5+900
    cfg=read(RUN/'configs/v8.json');assert estimate<cfg['budgets']['total_GPU_seconds'],f'Estimated matrix exceeds wall budget: {estimate}'
    sources=[identity(p) for p in sorted((RUN/'code').glob('*.py'))]+[identity(ROOT/'hoi_modules/projected_motion.py'),identity(ROOT/'hoi_modules/temporal_evidence.py'),identity(RUN/'configs/v8.json')]
    result=dict(status='pass',interface=identity(RUN/'diagnostics/interface_acceptance.json'),regression=dict(attempts=112,Adam=112,density_events=density),resume=dict(attempts=16,Adam=16,exact_load=True,RGB_sequence_exact=True,pair_sequence_cursor_exact=True,postload_trajectory_bitwise_required=False),cadence=cfg['cadence'],speed=dict(mean_iteration_seconds=mean,mean_temporal_ms=float(np.mean(pairms)),mean_RGB_only_ms=float(np.mean(rgbms)),GPU_jobs_used_seconds=used,conservative_matrix_estimate_seconds=estimate,estimate_not_guarantee=True),lambda_status='pending_real_B_Q1000',frozen_sources=sources)
    save_json(RUN/'module_acceptance.json',result);print(json.dumps(dict(status='pass',estimated_GPU_hours=estimate/3600)))
if __name__=='__main__':main()
