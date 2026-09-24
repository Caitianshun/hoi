"""Persistent bounded run: predictions then independent CPU evaluation."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
OUT=Path(__file__).resolve().parent
record=dict(status='running',pid=os.getpid(),start_unix=time.time(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),stages=[])
def save():(OUT/'pipeline_status.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
 for name in ['run_unidepth.py','evaluate_frozen.py']:
  start=time.perf_counter();remaining=1800-(time.time()-record['start_unix'])
  if remaining<=0:raise TimeoutError('30 minute bound exhausted')
  cmd=[sys.executable,str(OUT/name)];subprocess.run(cmd,check=True,timeout=remaining)
  record['stages'].append(dict(script=name,status='completed',seconds=time.perf_counter()-start));save()
 record['status']='completed'
except BaseException as exc:
 record.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
finally:
 record['end_unix']=time.time();record['wall_seconds']=record['end_unix']-record['start_unix'];save()
