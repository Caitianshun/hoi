"""Wait on process exit, then perform the one CPU numerical closeout."""
from pathlib import Path
import os,select,subprocess,json,time,traceback
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
def save(name,obj):(RUN/name).write_text(json.dumps(obj,indent=2)+'\n')
def run():
    pid=json.loads((RUN/'pipeline_launcher.json').read_text())['pid']
    try:
        fd=os.pidfd_open(pid);select.select([fd],[],[]);os.close(fd)
    except ProcessLookupError:pass
    state=json.loads((RUN/'pipeline.json').read_text());assert state['stage']=='metrics_completed',state
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2'}
    try:
        for name in ['collect_results.py','write_decision.py']:
            with (RUN/'logs'/('complete_'+name+'.log')).open('w') as f:
                subprocess.run([str(ROOT/'envs/4dgs/bin/python'),str(RUN/'code'/name)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        save('verification_continuation.json',dict(status='numerical_closeout_passed_visual_document_QA_pending',time_unix=time.time(),no_new_training=True))
    except BaseException:
        save('verification_continuation.json',dict(status='CPU_closeout_error',traceback=traceback.format_exc(),time_unix=time.time()));raise
if __name__=='__main__':run()
