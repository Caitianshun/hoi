"""CPU verification triggered by the frozen pipeline process exit, not polling."""
from pipeline import RUN,ROOT,PYTHON,read,save,identity
import os,select,subprocess,traceback,argparse,time
def run(pid):
    start=time.time();out=RUN/'logs/independent_verification.log'
    try:
        try:fd=os.pidfd_open(pid)
        except ProcessLookupError:fd=None
        if fd is not None:
            try:select.select([fd],[],[])
            finally:os.close(fd)
        state=read(RUN/'pipeline.json');assert state['stage']=='numerical_evaluation_completed',state
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
        with out.open('w') as log:subprocess.run([PYTHON,str(RUN/'code/collect_results.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        save(RUN/'continuation.json',dict(status='independent_verification_completed_report_visual_QA_pending',time_unix=time.time(),verification=identity(RUN/'protocol/verification.json'),remaining='Inspect all fixed dev16/train8 images, record final_decision and NEXT_DECISION/MISSING, rebuild final DOCX using existing marker and inspect every page, portable-copy check, package ZIP, logs, sync, pause v5. No further GPU or optimization authorized by completion.',draft_document_QA=identity(RUN/'protocol/document_A_QA.json')))
        save(RUN/'protocol/completion_trigger_verification.json',dict(status='completed',waited_for_pid=pid,started_unix=start,finished_unix=time.time(),GPU_used=False))
    except BaseException:
        failure=dict(status='verification_or_pipeline_failed',time_unix=time.time(),traceback=traceback.format_exc(),training_restart_allowed=False)
        save(RUN/'protocol/completion_trigger_verification.json',failure)
        save(RUN/'continuation.json',dict(status='completion_verification_exception_requires_review',evidence=identity(RUN/'protocol/completion_trigger_verification.json'),no_automatic_training_restart=True));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);run(p.parse_args().pid)
