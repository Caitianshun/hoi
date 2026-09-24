from pathlib import Path
import subprocess,json,os,time,datetime
E=Path(__file__).resolve().parents[1]
R=E.parents[2]
py=R/'envs/gvhmr/bin/python'
results=[]
for dev in ['dev1','dev2']:
    selection=E/f'pose/{dev}/optimization/selection_frozen.json'
    s=json.loads(selection.read_text())
    if s['status']!='accepted_for_S1_star':
        results.append({'dev':dev,'status':'not_started_input_rejected'});continue
    command=[str(py),str(E/'code/run_reconstruction.py'),'--dev',dev,'--selection',str(selection),'--cpu-evaluator',str(E/'code/evaluate_reconstruction_pair_cpu.py')]
    with (E/f'logs/{dev}_reconstruction.log').open('w') as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'))
    results.append({'dev':dev,'returncode':result.returncode,'command':command,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    (E/'logs/reconstruction_completion.json').write_text(json.dumps(results,indent=2)+'\n')
