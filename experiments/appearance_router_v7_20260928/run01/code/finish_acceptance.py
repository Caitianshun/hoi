"""Read diagnostic outputs once; require exact restoration and counted schedule."""
from common import *
import torch
from restore_state import exact
from module_acceptance import compare
def run():
    probe=read(RUN/'diagnostics/acceptance_probe.json');consume=read(RUN/'diagnostics/consumer_acceptance.json')
    assert probe['status']=='accepted_numerical_variation' and consume['status']=='pass'
    main=RUN/'diagnostics/regression';resume=RUN/'diagnostics/regression_resume'
    a=read(main/'run.json');b=read(resume/'run.json');v=read(resume/'restore_verification.json')
    assert a['status']==b['status']=='completed' and a['attempted_this_process']==128 and b['attempted_this_process']==24
    assert a['updates_this_process']==128 and b['updates_this_process']==24
    assert v['model_Adam_groups_steps_buffers_exact'] and v['RNG_exact'] and v['stack_exact'] and v['expected_first']==1105
    batches=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
    x=batches(main/'sampling_order.jsonl');y=batches(resume/'sampling_order.jsonl')
    assert [(r['iteration'],r['frame_ids']) for r in x if r['iteration']>=1105]==[(r['iteration'],r['frame_ids']) for r in y]
    density=batches(main/'density_events.jsonl');assert any(r['iteration']==1100 and (r['actual_added'] or r['actual_deleted']) for r in density)
    sa=torch.load(a['checkpoint'],map_location='cpu',weights_only=False);sb=torch.load(b['checkpoint'],map_location='cpu',weights_only=False)
    exact(sa['rng'],sb['rng']);exact(sa['viewpoint_stack'],sb['viewpoint_stack']);exact(sa['temp_list'],sb['temp_list'])
    from finite_guard import walk
    xa=dict(walk(sa['model']));xb=dict(walk(sb['model']));diffs=[]
    for n,t in xa.items():
        if torch.is_tensor(t):
            assert t.shape==xb[n].shape and torch.isfinite(t).all() and torch.isfinite(xb[n]).all()
            diffs.append(dict(name=n,**compare(xb[n].float() if not xb[n].is_floating_point() else xb[n],t.float() if not t.is_floating_point() else t)))
    save_json(RUN/'diagnostics/resume_acceptance.json',dict(status='pass',load_verification=v,next_and_all_24_batch_ids_exact=True,final_RNG_stack_exact=True,terminal_tensor_differences=diffs,trajectory_bitwise_equality_required=False,density_events=density))
    ledger=batches(RUN/'protocol/D_steps.jsonl');assert len(ledger)<=160
    count=batches(RUN/'protocol/optimizer_calls.jsonl');assert sum(x['event']=='completed' for x in count)==sum(x['event']=='enter' for x in count)==len(ledger)
    frozen=read(RUN/'protocol/engineering_protocol_frozen.json')
    for s in frozen['frozen_sources']:assert sha(s['path'])==s['sha256']
    save_json(RUN/'module_acceptance.json',dict(status='accepted_numerical_variation',engineering_protocol='engineering_equivalence_v2',strict_semantics='pass',same_gradient_Adam_density='pass',short_regression='pass',restore='pass',diagnostic_attempts=len(ledger),optimizer_calls=len(ledger),frozen_sources=frozen['frozen_sources'],evidence=[identity(RUN/p) for p in ['diagnostics/acceptance_probe.json','diagnostics/consumer_acceptance.json','diagnostics/resume_acceptance.json']],formal_C_authorized=True))
    print('Integrated acceptance passed; formal M1 authorized without another approval')
if __name__=='__main__':run()
