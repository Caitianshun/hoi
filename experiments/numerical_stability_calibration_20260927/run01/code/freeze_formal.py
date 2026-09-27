"""Check measured A evidence and freeze the conditional B protocol once."""
from pipeline import RUN,ROOT,OLD,read,save,identity,sha
import time,subprocess
def lines(p):return [__import__('json').loads(s) for s in p.read_text().splitlines()]
def run():
    assert not (RUN/'protocol/formal_gate.json').exists()
    v4=ROOT/'experiments/foreground_stage_calibration_20260927/run01'
    repeats=[read(RUN/f'runs/{n}/first_bad_tensor.json') for n in ['A_replay1b','A_replay2']]
    assert repeats[0]['iteration']==repeats[1]['iteration'] and repeats[0]['phase']==repeats[1]['phase']=='raster_backward_outputs'
    assert repeats[0]['batch']==repeats[1]['batch']
    internal=read(RUN/'runs/A_internal_boundary/first_bad_tensor.json');assert internal['phase']=='forward_internal_geometry'
    isolation=read(RUN/'runs/A_isolation2/summary.json')
    for r in isolation['rows']:
        if r['domain']=='bounded':assert all(x['state'] in ['finite','not_connected'] for x in r['gradients'].values())
    regression=read(RUN/'runs/A_regression128/result.json');assert regression['status']=='window_completed' and regression['attempted']>=128 and regression['optimizer_step_calls']==regression['attempted']
    density=lines(RUN/'runs/A_regression128/density_events.jsonl');assert any(r['operation']=='densify' for r in density)
    assert all(r.get('finite',True) for r in lines(RUN/'runs/A_regression128/replay_events.jsonl'))
    restore=read(RUN/'runs/A_restore_tail/restore_verification.json');assert restore['status']=='passed' and restore['model_Adam_groups_steps_buffers_exact'] and restore['RNG_exact'] and restore['stack_exact']
    tail=read(RUN/'protocol/restore_tail_comparison.json');assert tail['sampling_exact'] and tail['RNG_after_exact']
    transition=read(RUN/'protocol/fine_transition.json');assert transition['status']=='passed' and transition['model_SH_radii_exact'] and transition['Adam_and_statistics_reset'] and transition['full_train_stack']
    used=len(lines(RUN/'protocol/A_steps.jsonl'));probes=len(lines(RUN/'protocol/A_probes.jsonl'));jobs=lines(RUN/'protocol/gpu_jobs.jsonl');seconds=sum(r['wall_seconds'] for r in jobs);assert used<=512 and probes<=64 and seconds<1800
    proposal=read(RUN/'protocol/stabilization_proposal.json');assert proposal['upper_bound']==transition['scene_extent']
    assets=[identity(RUN/p) for p in ['protocol/shared_fine_initial.pt','protocol/fine_transition.json','protocol/stabilization_proposal.json','protocol/fixed_examples.json','configs/v5.json']]
    assets += [identity(transition['parent']['path']),identity(OLD/'inputs/hos_backpack/manifest.json'),identity(OLD/'inputs/hos_backpack/evaluation_manifest.json')]
    env=read(RUN/'environment.json');assets += [identity(x['path']) for x in env['inherited_runtime_identity']['extensions']]
    for x in env['inherited_runtime_identity']['extensions']:assert sha(x['path'])==x['sha256']
    inherited=read(v4/'protocol/training_frozen.json')['training_sources']
    for x in inherited:assert sha(x['path'])==x['sha256']
    source=inherited+[identity(RUN/'code'/n) for n in ['common.py','restore_state.py','finite_guard.py','scale_domain.py','renderer_observer.py','replay_first_failure.py','train_fine_pair.py','run_job.py','pipeline.py']]
    pair=dict(status='frozen_before_B',created_unix=time.time(),arms=dict(B_U='uniform_fine',B_F='balanced_fine'),only_scientific_difference='fine per-image RGB uniform versus foreground/background half-weight region means',shared_initial=transition,stabilization=proposal,config=identity(RUN/'configs/v5.json'),historical_coarse_limitation='Original coarse was not retrained with new domain bound; comparison is conditional on this shared historic coarse state',sampling='Original RNG/stack/density schedule; record actual batches and possible topology-dependent divergence',final_iteration='backward without optimizer.step as in original',no_development_tuning=True,stopping='New numerical failure stops that arm; no new attempt; total B attempted rounds <=28000 including failures and external interruption tails',evaluation='After both attempts close, all268train+16dev, supplemental15, B_F-B_U primary; H1 auxiliary')
    save(RUN/'pair_protocol.json',pair);assets.append(identity(RUN/'pair_protocol.json'))
    gate=dict(all_conditions_passed=True,created_unix=time.time(),conditions=dict(reproducible_first_boundary=True,single_domain_change_defined=True,fault_normal_forward_gradient_actual_differences=True,regression128_crossed_density=True,restore_and_freeze_no_dev_tuning=True),evidence=[identity(RUN/p) for p in ['runs/A_replay1b/first_bad_tensor.json','runs/A_replay2/first_bad_tensor.json','runs/A_internal_boundary/first_bad_tensor.json','runs/A_isolation2/summary.json','runs/A_regression128/result.json','runs/A_regression128/density_events.jsonl','runs/A_restore_tail/restore_verification.json','protocol/restore_tail_comparison.json','minimal_repro/validation.json']],A_attempts=used,A_no_optimizer_probes=probes,A_GPU_job_seconds=seconds,frozen_source_files=source,frozen_assets=assets,git_at_freeze=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),no_full_run_stability_guarantee=True)
    save(RUN/'protocol/formal_gate.json',gate)
    print({k:gate[k] for k in ['all_conditions_passed','A_attempts','A_no_optimizer_probes','A_GPU_job_seconds']})
if __name__=='__main__':run()
