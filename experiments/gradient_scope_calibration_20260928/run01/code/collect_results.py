"""Close the failed D2 prerequisite without inventing a C terminal or quality score."""
from common import *
import csv,time,shutil,collections
import numpy as np
def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def collect():
    parity=read(RUN/'renderer_parity.json');route=read(RUN/'routing_equivalence.json');assert parity['status']=='passed' and route['status']=='failed'
    # Close the necessary gate. Later localization cannot erase the first failed test.
    route['optimizer_density_resume']=dict(status='NA_not_run',reason='Prior necessary uniform_all derivative equivalence gate failed; no diagnostic optimizer attempt or formal C')
    save_json(RUN/'routing_equivalence.json',route)
    fail=[r for r in route['rows'] if not r['passed']]
    inherited=read(RUN/'state_manifest.json')['inherited']
    for a in inherited:assert sha(a['path'])==a['sha256']
    hv=read(V5/'protocol/input_hash_verification.json')['inputs']
    assert len(hv)==568
    for a in hv:assert sha(a['path'])==a['sha256'],a['path']
    save_json(RUN/'protocol/input_hash_verification.json',dict(inputs=hv,source=identity(V5/'protocol/input_hash_verification.json'),status='all_rehashed_equal_to_V5'))
    gate=dict(all_conditions_passed=False,status='closed_D2_equivalence_not_established',D0=True,D1=True,D2=False,C_started=False,failed_derivative_entries=fail,unexecuted=['one-step Adam equivalence','density-buffer equivalence','128-iteration regression','save/resume equivalence','formal C-route','C terminal evaluation'],assets=inherited+[identity(RUN/'renderer_parity.json'),identity(RUN/'routing_equivalence.json')])
    save_json(RUN/'protocol/formal_gate.json',gate)
    config=read(RUN/'configs/v6.json')
    save_json(RUN/'route_protocol.json',dict(question='Does balancing RGB only for canonical SH preserve FG gains while avoiding direct balanced G/opacity/deformation/density updates?',unique_change='A <- balanced; G <- uniform + original regularization; q <- explicit returned uniform',main_comparison='C_route-B_U',auxiliary_comparison='C_route-B_F',parent_assets=inherited,budgets=config['budgets'],quality=config['quality'],status=gate['status'],source=identity(RUN/'protocol/user_guidance_source.md'),historical_results_read_only=True))
    jobs=[json.loads(s) for s in (RUN/'protocol/gpu_jobs.jsonl').read_text().splitlines()];calls=[json.loads(s) for s in (RUN/'protocol/kernel_calls.jsonl').read_text().splitlines()];counts=collections.Counter((x['mode'],x['kind']) for x in calls)
    attempts={k:len((RUN/'protocol'/f'{k}_steps.jsonl').read_text().splitlines()) if (RUN/'protocol'/f'{k}_steps.jsonl').exists() else 0 for k in ['D','C']};assert sum(attempts.values())==0
    seconds=sum(x['wall_seconds'] for x in jobs);assert seconds<config['budgets']['D_GPU_seconds'] and counts['diagnostic','render']<=256 and counts['diagnostic','backward']<=160
    save_json(RUN/'costs.json',dict(status='all_GPU_processes_closed',jobs=jobs,GPU_task_seconds=seconds,GPU_task_hours=seconds/3600,diagnostic_renders=counts['diagnostic','render'],diagnostic_backwards=counts['diagnostic','backward'],diagnostic_optimization_attempts=attempts['D'],formal_optimization_attempts=attempts['C'],optimizer_steps=0,terminal_evaluation_renders=0,physical_GPU=1,hardware='NVIDIA GeForce RTX3090',scope='GPU task process wall-clock including loading, failures, read-only kernels and serialization; not CUDA kernel time',peak_memory=None,peak_memory_NA_reason='This closed read-only diagnostic batch did not continuously sample per-PID GPU memory or export allocator peak; no training',limits=config['budgets']))
    with (V5/'metrics_per_frame.csv').open() as f:old=list(csv.DictReader(f))
    metrics=[dict(r,source_version='V5_inherited') for r in old if r['run'] in ['B_U','B_F']]
    for r in old:
        if r['run']!='B_U':continue
        x=dict(r,run='C_route',status='not_started_D2_gate_failed',NA_reason='No C terminal; prerequisite equivalence unestablished',source_version='V6')
        for key in ['mse','sse_rgb_mean','psnr_db','ssim','lpips_spatial_mean','full_error_share','raw_mse','raw_psnr_db','raw_render_path','raw_render_sha256']:x[key]=''
        metrics.append(x)
    csvwrite(RUN/'metrics_per_frame.csv',metrics)
    pair=[]
    for right in ['B_U','B_F']:
        for r in metrics:
            if r['run']!='C_route':continue
            pair.append(dict(comparison='C_route-'+right,split=r['split'],frame_id=r['frame_id'],region=r['region'],psnr_db=None,ssim=None,lpips_spatial_mean=None,raw_psnr_db=None,NA_reason=r['NA_reason']))
    csvwrite(RUN/'paired_differences.csv',pair)
    ev=read(V5/'evaluation/summary.json');save_json(RUN/'evaluation_summary.json',dict(summaries={**{b:ev['summaries'][b] for b in ['B_U','B_F']},'C_route':{k:None for k in ['train','retained','retained_without_00000']}},comparisons={k:None for k in ['C_route-B_U','C_route-B_F']},quality_gate=None,NA_reason='C not started; not a negative method-quality result',metric_policy=identity(V5/'code/evaluate_and_report.py'),inherited_source=identity(V5/'evaluation/summary.json')))
    save_json(RUN/'sampling_comparison.json',dict(status='NA',reason='No C optimization batches',C_batches=0,inherited_V5=identity(V5/'sampling_comparison.json')))
    csvwrite(RUN/'gradient_audit.csv',[dict(policy=k,**v) for k,v in route['losses'].items()])
    contrib={};time_probes={};summary=[]
    for arm,dirname in [('B_U','B_U_verified'),('B_F','B_F')]:
        folder=RUN/'diagnostics'/dirname;d=read(folder/'contribution_manifest.json');contrib[arm]=d;time_probes[arm]=read(folder/'time_camera_probe.json')
        z=np.load(folder/'effective_time.npz');supported=z['supported'];weight=z['total_contribution'];neff=z['N_eff'];fraction=z['fg_fraction']
        summary.append(dict(arm=arm,points=d['topology_rows'],unsupported=d['unsupported_in_sample'],unsupported_fraction=d['unsupported_in_sample']/d['topology_rows'],N_eff_median=float(np.median(neff[supported])),N_eff_contribution_weighted=float(np.sum(neff*weight,dtype=np.float64)/np.sum(weight,dtype=np.float64)),FG_weighted_N_eff=float(np.sum(neff*weight*fraction,dtype=np.float64)/np.sum(weight*fraction,dtype=np.float64)),mean_FG_contribution_fraction=float(np.mean([r['fg_contribution_fraction'] for r in d['rows']])),bound_contribution_total=sum(r['bound_contribution_total'] for r in d['rows'])))
    save_json(RUN/'contribution_manifest.json',dict(status='passed_with_documented_float32_roundoff',models=contrib,C_route=None,summary=summary,physical_surface_correspondence=None))
    save_json(RUN/'time_camera_probe.json',dict(models=time_probes,C_route=None,cross_has_GT=False))
    # Recover each old bound record's frame from the original renderer's batch order.
    boundrows=[]
    for arm in ['B_U','B_F']:
        offsets=collections.Counter()
        for line in (V5/'runs'/arm/'replay_events.jsonl').open():
            x=json.loads(line)
            if x['phase']!='scale_bound':continue
            i=x['iteration'];slot=offsets[i];assert slot<2 and len(x['batch'])==2
            boundrows.append(dict(arm=arm,iteration=i,frame_id=x['batch'][slot],total_axes=x['total_axes'],triggered_axes=x['triggered_axes'],source='V5 frozen scale_bound event; frame mapped by ordered batch slot',slot=slot));offsets[i]+=1
        assert len(offsets)==14000 and set(offsets.values())=={2}
    csvwrite(RUN/'bound_events_compact.csv',boundrows)
    figures=[];dest=RUN/'inherited_figures';dest.mkdir(exist_ok=True)
    for f in read(V5/'figure_manifest.json')['hos']:
        assert sha(f['path'])==f['sha256'];p=dest/Path(f['path']).name;shutil.copyfile(f['path'],p);figures.append(dict(f,original_source=f['path'],path=str(p),status='V5_historical_no_C'))
    assert len(figures)==40;save_json(RUN/'figure_manifest.json',dict(inherited_V5=figures,D1=[x['figure'] for d in time_probes.values() for x in d['rows']],C=None))
    save_json(RUN/'run.json',dict(status='closed_prerequisite_not_established',formal_C='not_started',quality_gate=None,no_new_nonfinite_observed=True,optimization_attempts=attempts,reason='uniform_all derivative prerequisite exceeded predeclared tolerance; later repeat does not erase failure',completed_unix=time.time()))
    save_json(RUN/'protocol/verification.json',dict(input_hashes_reverified=len(hv),inherited_checkpoint_hashes_reverified=len(inherited),metric_rows=len(metrics),C_rows=sum(r['run']=='C_route' for r in metrics),pair_rows=len(pair),all_C_metrics_NA=True,old_figures=len(figures),budgets_passed=True))
    print('Closed D2 prerequisite; no C optimization',seconds)
if __name__=='__main__':collect()
