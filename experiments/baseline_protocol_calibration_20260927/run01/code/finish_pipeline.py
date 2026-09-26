"""Event-driven continuation for HOS rendering after the fixed training matrix.

pidfd waits on process exit, rather than polling for a completion file. Errors
are terminal records; this entry never launches a fourth formal training.
"""
from __future__ import annotations
import fcntl
import json
import os
import select
import subprocess
import time
import traceback
from pathlib import Path
from run_baselines import ROOT,RUN,CODE,PYTHON,S1_PYTHON,gpu_job,used_seconds
from adapter_4dgs import sha,save_json


def main():
    launcher=json.loads((RUN/'protocol/launcher.json').read_text())
    state=RUN/'continuation.json'
    save_json(state,dict(status='waiting_for_training_exit',pid=os.getpid(),training_pid=launcher['pid'],
                         wait_mechanism='Linux pidfd process-exit event'))
    try:
        try:
            descriptor=os.pidfd_open(launcher['pid'])
            select.select([descriptor],[],[]);os.close(descriptor)
        except ProcessLookupError:pass
        pipeline=json.loads((RUN/'pipeline.json').read_text())
        assert pipeline['status']=='formal_and_behave_evaluation_completed',pipeline
        lock=(RUN/'protocol/gpu1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rd=RUN/'runs/hos_backpack_formal';r=json.loads((rd/'run.json').read_text())
        assert r['status']=='completed' and r['nominal_iterations']==17000
        files=[Path(r['checkpoint']),rd/'run.json',rd/'effective_config.json',
               RUN/'inputs/hos_backpack/manifest.json',RUN/'inputs/hos_backpack/evaluation_manifest.json',
               ROOT/'other data/hosnerf/checkpoints/Backpack.ckpt',CODE/'render_hos_4dgs.py',CODE/'run_hos_checkpoint.py']
        frozen=RUN/'protocol/hos_finals.json';assert not frozen.exists()
        save_json(frozen,dict(status='all_states_frozen',scope='hos_backpack',time_unix=time.time(),
                             assets=[dict(path=str(p),sha256=sha(p)) for p in files],
                             H0_identity='official native checkpoint; historical training split not independently embedded',
                             H1_identity='terminal17000 only, trainingRGB triangulation initialization',
                             direct_fair_difference_allowed=False))
        save_json(state,dict(status='rendering_HOS',pid=os.getpid(),freeze=str(frozen)))
        gpu_job('H1_render',[PYTHON,str(CODE/'render_hos_4dgs.py'),'--run-dir',str(rd),'--freeze',str(frozen)])
        gpu_job('H0_native_remaining',[str(ROOT/'envs/hosnerf/bin/python'),str(CODE/'run_hos_checkpoint.py'),
                 '--official',str(ROOT/'third_party/HOSNeRF'),'--data',str(ROOT/'other data/hosnerf/Backpack'),
                 '--checkpoint',str(ROOT/'other data/hosnerf/checkpoints/Backpack.ckpt'),
                 '--output',str(RUN/'runs/H0_native_remaining'),'--skip','1'])
        with (RUN/'logs/evaluate_hos.log').open('w') as log:
            subprocess.run([S1_PYTHON,str(CODE/'evaluate_hos.py'),'--freeze',str(frozen)],cwd=ROOT,
                            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'2'},
                            stdout=log,stderr=subprocess.STDOUT,check=True)
        save_json(state,dict(status='all_training_rendering_evaluation_completed',time_unix=time.time(),
                             gpu_job_wall_seconds=used_seconds(),report_status='pending_interpretation_and_visual_QA'))
    except BaseException:
        save_json(state,dict(status='failed',time_unix=time.time(),traceback=traceback.format_exc(),
                             gpu_job_wall_seconds=used_seconds()))
        raise


if __name__=='__main__':main()
