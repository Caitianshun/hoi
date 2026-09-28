"""Linux pidfd exit event chains one CPU audit; no training polling."""
from pathlib import Path
import os,select,json,subprocess,time,argparse,traceback
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
def save(v):(RUN/'verification_continuation.json').write_text(json.dumps(v,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args();save(dict(status='waiting_for_pipeline_exit',pid=os.getpid(),target=a.pid))
    try:
        fd=os.pidfd_open(a.pid);poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
        state=json.loads((RUN/'pipeline.json').read_text())
        if state['status']!='V8A_evaluated_pending_quality_assessment_and_delivery':save(dict(status='pipeline_not_successful_no_audit',pipeline=state));return
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        with (RUN/'logs/CPU_verification.log').open('w') as f:subprocess.run([str(ROOT/'envs/4dgs/bin/python'),str(RUN/'code/collect_results.py')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env,check=True)
        save(dict(status='CPU_verification_completed_quality_assessment_pending',finished_unix=time.time(),chat_notification_sent=False))
    except BaseException:save(dict(status='verification_failed',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
