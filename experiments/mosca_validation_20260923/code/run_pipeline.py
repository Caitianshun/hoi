"""One bounded controlled training run followed immediately by export/evaluation."""
from pathlib import Path
import argparse,json,os,socket,subprocess,sys,time,traceback,datetime,hashlib
ROOT=Path('/home/cai_tianshun/Project/HOI')
EXP=ROOT/'experiments/mosca_validation_20260923'
OLD=ROOT/'experiments/mosca_baseline_20260922'
CODE=EXP/'code'
PY=ROOT/'envs/mosca/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resume-scaffold-dir',type=Path);a=p.parse_args()
    out=EXP/a.branch;out.mkdir(exist_ok=True)
    if (out/'pipeline.json').exists():raise FileExistsError(out/'pipeline.json')
    record={'status':'running','pid':os.getpid(),'host':socket.gethostname(),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'branch':a.branch,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'config_sha256':sha(a.config),
            'legal_input':str(OLD/'common_input/input_manifest.json'),'input_sha256':sha(OLD/'common_input/input_manifest.json'),
            'reference_used_for_training':False,'stages':[]}
    begin=time.perf_counter()
    def save():
        tmp=out/'pipeline.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out/'pipeline.json')
    def stage(name,command):
        item={'name':name,'command':list(map(str,command)),'status':'running'};record['stages'].append(item);save()
        start=time.perf_counter()
        with (out/f'{name}.log').open('w') as log:
            result=subprocess.run(item['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        item.update(exit_code=result.returncode,wall_seconds=time.perf_counter()-start,status='completed' if result.returncode==0 else 'failed');save()
        if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}; see {out/name}.log')
    save()
    try:
        cmd=[PY,CODE/'run_control.py','--input',OLD/'common_input','--segmentation',OLD/'segmentation/segmentation.npz',
             '--output',out/'model','--config',a.config,'--normalize-world']
        if a.resume_scaffold_dir:cmd.extend(['--resume-scaffold-dir',a.resume_scaffold_dir])
        stage('train',cmd)
        stage('export',[PY,CODE/'export_control.py','--input',OLD/'common_input','--checkpoint-dir',out/'model',
                        '--output',out/'diagnostics','--segmentation',OLD/'segmentation/segmentation.npz'])
        ref=OLD/'evaluation/fixed_rgb_queries'
        stage('trajectory_evaluation',[PY,ROOT/'scripts/diagnostics/evaluate_prediction_reference.py',
                '--prediction',out/'diagnostics/query_trajectories.npz','--prediction-protocol',out/'diagnostics/export_manifest.json',
                '--reference',ref/'same_version_joint_reference_observation_times.npz',
                '--reference-protocol',ref/'same_version_joint_reference_observation_times_protocol.json',
                '--input-manifest',OLD/'common_input/input_manifest.json','--method',a.branch,'--output',out/'evaluation'])
        stage('render_summary',[PY,ROOT/'scripts/diagnostics/summarize_rendered_event.py','--input',OLD/'common_input',
                               '--export',out/'diagnostics','--segmentation',OLD/'segmentation/segmentation.npz','--output',out/'render_summary'])
        stage('heldout',[PY,CODE/'heldout_control.py','--checkpoint-dir',out/'model','--output',out/'heldout'])
        record['status']='completed'
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-begin,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if __name__=='__main__':main()
