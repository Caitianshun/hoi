"""Persistent completion-triggered CPU analysis; no GPU polling or training."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/mosca_interface_validation_20260923'
BRANCH=EXP/'d_exact_geometry';OUT=EXP/'lineage_analysis'
os.environ['CUDA_VISIBLE_DEVICES']=''
OUT.mkdir(exist_ok=True)
if (OUT/'pipeline.json').exists():raise FileExistsError('Keep prior analysis status; use an explicit new run')
status=dict(status='waiting_for_D_inotify',pid=os.getpid(),start_unix=time.time(),gpu_used=False,stages=[])
def save():
    tmp=OUT/'pipeline.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(OUT/'pipeline.json')
save()
try:
    steps=[('wait_D',[sys.executable,str(ROOT/'experiments/mosca_validation_20260923/code/wait_completion.py'),str(BRANCH)]),
           ('analyze',[sys.executable,str(EXP/'code/analyze_lineage_cpu.py'),'--model',str(BRANCH/'model'),'--output',str(OUT)])]
    for name,command in steps:
        if name=='analyze' and json.loads((BRANCH/'pipeline.json').read_text())['status']!='completed':
            raise RuntimeError('D pipeline failed; final checkpoints were not analyzed')
        status['status']='waiting_for_D_inotify' if name=='wait_D' else 'analyzing_CPU';save();t=time.perf_counter()
        with (OUT/f'{name}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        status['stages'].append(dict(name=name,seconds=time.perf_counter()-t,status='completed'));save()
    status['status']='completed_pending_visual_QA'
except BaseException as exc:
    status.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
finally:
    status.update(end_unix=time.time(),wall_seconds=time.time()-status['start_unix']);save()
