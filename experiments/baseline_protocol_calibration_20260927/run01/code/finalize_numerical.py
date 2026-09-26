"""Verify completed numerical results immediately on the renderer's exit event."""
import ctypes
import errno
import json
import os
import select
import subprocess
import time
import traceback
from adapter_4dgs import ROOT,RUN,save_json


def main():
    status=RUN/'numerical_completion.json'
    target=json.loads((RUN/'protocol/continuation_launcher.json').read_text())['pid']
    save_json(status,dict(status='waiting_for_render_evaluation_exit',pid=os.getpid(),target_pid=target))
    try:
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.pidfd_open(int(target),0)
        if fd>=0:select.select([fd],[],[]);os.close(fd)
        elif ctypes.get_errno()!=errno.ESRCH:raise OSError(ctypes.get_errno(),'pidfd_open')
        assert json.loads((RUN/'continuation.json').read_text())['status']=='all_training_rendering_evaluation_completed'
        for label,args in [('verify_final',[]),('package_results',['costs'])]:
            with (RUN/'logs'/f'{label}.log').open('w') as log:
                subprocess.run([str(ROOT/'envs/gvhmr/bin/python'),str(RUN/'code'/f'{label}.py'),*args],cwd=ROOT,
                               env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'2'},
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        save_json(status,dict(status='numerical_results_verified_report_pending',time_unix=time.time(),
                             remaining='Interpretation, NEXT_DECISION, embedded DOCX and visual QA, feedback bundle, final logs and source sync'))
    except BaseException:
        save_json(status,dict(status='failed',time_unix=time.time(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
