"""Create one shared fine-zero state from the inherited coarse model, no updates."""
from common import *
import torch,os
from restore_state import *
from adapter_v4 import ProtocolScene
import train as official
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';torch.set_num_threads(4);official.setup_seed(12345)
    path=OLD/'runs/hos_backpack_formal/checkpoint_coarse_003000.pt'
    parent=torch.load(path,map_location='cpu',weights_only=False);assert parent['stage']=='coarse' and parent['iteration']==3000
    _,(dataset,hidden,opt,pipe)=official_config(2);dataset.source_path=str(OLD/'inputs/hos_backpack/manifest.json');dataset.model_path=str(RUN/'protocol/fine_initial')
    model=official.GaussianModel(dataset.sh_degree,hidden);scene=ProtocolScene(dataset,model);restore_model(model,opt,parent)
    before=clone_cpu(model.capture());model.training_setup(opt)
    after=model.capture()
    for i in [0,1,2,3,4,5,6,7,8,9,13]:exact(before[i],after[i])
    assert not model.optimizer.state
    for name in ['xyz_gradient_accum','denom','_deformation_accum']:assert torch.count_nonzero(getattr(model,name))==0
    rng_restore(parent['rng']);stack=scene.getTrainCameras().copy();temp=stack.copy()
    state=capture(model,'fine',0,stack,temp,phase='completed_step',resumable=True,shared_coarse_parent=identity(path),optimizer_updates=0)
    output=RUN/'protocol/shared_fine_initial.pt';assert not output.exists();torch.save(state,output)
    save_json(RUN/'protocol/fine_transition.json',dict(status='passed',parent=identity(path),initial_fine=identity(output),model_SH_radii_exact=True,Adam_and_statistics_reset=True,full_train_stack=True,RNG_restored_after_construction=True,optimization_steps=0,scene_extent=scene.cameras_extent))
if __name__=='__main__':run()
