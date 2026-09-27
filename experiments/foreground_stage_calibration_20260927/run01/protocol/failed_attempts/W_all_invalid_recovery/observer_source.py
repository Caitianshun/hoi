"""W_all observation-only entry point: retain recovery and failure evidence.

The original frozen optimizer, loss, sampler, schedules and training loop are
unchanged. Extra serialization consumes wall time, but no RNG or optimizer steps.
"""
from train_stage import *

original_after = Runtime.after_step
original_gradient_check = Runtime.gradient_check

def retain_after_step(self, stage, iteration, model, stack, temp):
    original_after(self, stage, iteration, model, stack, temp)
    if iteration % 100 == 0 and iteration % 1000 != 0 and iteration != self.stage_end:
        self.save_state(stage, iteration, model, stack, temp)
        rolling = self.output / 'recovery_latest.pt'
        self.last_checkpoint.replace(rolling)
        self.last_checkpoint = rolling
        save_json(self.output/'latest_checkpoint.json', dict(path=str(rolling),stage=stage,iteration=iteration))

def retain_failure(self, stage, iteration, model, loss):
    try:
        original_gradient_check(self, stage, iteration, model, loss)
    except BaseException:
        rows=[]
        for group in model.optimizer.param_groups:
            for j,p in enumerate(group['params']):
                rows.append(dict(group=group.get('name'),parameter=j,elements=p.numel(),
                    nonfinite=int((~torch.isfinite(p)).sum()),
                    gradient_nonfinite=None if p.grad is None else int((~torch.isfinite(p.grad)).sum())))
        save_json(self.output/'nonfinite_diagnostic.json',dict(stage=stage,iteration=iteration,
            batch_frame_ids=self.batch,RGB=self.rgb_rows,regularizers=self.reg,parameters=rows,
            last_completed_checkpoint=None if self.last_checkpoint is None else identity(self.last_checkpoint),
            scope='Failure evidence, not a valid terminal or automatic resume target'))
        torch.save(dict(model=model.capture(),rng=rng_capture(),stage=stage,iteration=iteration,
            phase='after_backward_before_density_and_optimizer',deformation_accum=model._deformation_accum),
            self.output/'failure_state_diagnostic_only.pt')
        raise

Runtime.after_step=retain_after_step
Runtime.gradient_check=retain_failure

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--policy',required=True)
    p.add_argument('--branch-from-coarse');p.add_argument('--resume');p.add_argument('--check',action='store_true')
    p.add_argument('--coarse-steps',type=int,default=5);p.add_argument('--fine-steps',type=int,default=10);p.add_argument('--stop-after',type=int)
    run(p.parse_args())
