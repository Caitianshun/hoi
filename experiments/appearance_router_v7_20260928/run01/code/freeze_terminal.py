"""Seal the sole logical terminal before loading development RGB for evaluation."""
from common import *
import torch
from finite_guard import walk
def run():
    final=read(RUN/'formal_terminal.json');rd=Path(final['directory']);r=final['run']
    assert r['status']=='completed' and r['nominal_iterations']==14000
    assert sha(r['checkpoint'])==r['checkpoint_sha256']
    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
    assert state['iteration']==14000 and not state['resumable'] and state['purpose']=='formal'
    assert state['optimizer_updates']==13999
    for _,t in walk(dict(model=state['model'],accum=state['deformation_accum'])):
        if torch.is_tensor(t):assert torch.isfinite(t).all()
    assets=[identity(r['checkpoint']),identity(rd/'effective_config.json'),identity(rd/'run.json'),identity(RUN/'module_acceptance.json'),identity(RUN/'configs/v7.json'),identity(RUN/'protocol/fixed_examples.json')]
    assets.extend(identity(OLD/'inputs/hos_backpack'/n) for n in ['manifest.json','evaluation_manifest.json'])
    save_json(RUN/'protocol/finals.json',dict(status='sole_terminal_frozen',run_directory=str(rd),assets=assets,checkpoint=identity(r['checkpoint']),no_development_model_selection=True,finite_model_Adam_buffers=True))
if __name__=='__main__':run()
