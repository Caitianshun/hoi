"""Reuse sealed D0/D1; verify parent, source and binary identities only."""
from common import *
import subprocess,shutil
def run():
    config=read(RUN/'configs/v7.json');old=read(V6/'route_protocol.json')
    assets=old['parent_assets'];expected=set(config['parent_hashes'].values())
    assert {x['sha256'] for x in assets}==expected
    for x in assets:assert identity(x['path'])==x
    env=read(V6/'environment.json')
    for key in ['backend','adapter']:assert identity(env[key]['path'])==env[key]
    assert subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()==env['upstream']
    sources=[identity(p) for p in [ROOT/'hoi_modules/attribute_gradient_router.py',RUN/'configs/v7.json',*[RUN/'code'/n for n in ['common.py','budget.py','run_job.py','train_routed_fine.py','diagnostic_render.py','module_acceptance.py']],*[V5/'code'/n for n in ['replay_first_failure.py','restore_state.py','finite_guard.py','scale_domain.py']],*[V4/'code'/n for n in ['train_stage.py','loss_policy.py','adapter_v4.py','render_utils.py']],OLD/'code/adapter_4dgs.py']]
    save_json(RUN/'protocol/engineering_protocol_frozen.json',dict(protocol='engineering_equivalence_v2',config=identity(RUN/'configs/v7.json'),frozen_sources=sources,parent_assets=assets,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),historical_D0=identity(V6/'renderer_parity.json'),historical_D1=identity(V6/'contribution_manifest.json'),input_identity_evidence=identity(V6/'protocol/input_hash_verification.json'),input_manifests=[identity(OLD/'inputs/hos_backpack'/n) for n in ['manifest.json','evaluation_manifest.json']],repeated_D0_D1=False))
    save_json(RUN/'environment.json',env)
    save_json(RUN/'route_protocol.json',dict(status='authorized_V7_execution',unique_change='A canonical SH receives balanced F; G receives U+R; density q receives explicit uniform copy',parent_assets=assets,budgets=config['budgets'],quality=config['quality'],source=identity(RUN/'protocol/user_guidance_source.md'),historical_results_read_only=True))
    shutil.copyfile(V5/'protocol/fixed_examples.json',RUN/'protocol/fixed_examples.json')
    print('Reused D0/D1 and frozen V7 source/asset identities')
if __name__=='__main__':run()
