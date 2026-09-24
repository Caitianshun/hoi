"""Persistent completion-chained, frozen RGB preprocessing on assigned GPU0."""
from pathlib import Path
import json,os,subprocess,time,traceback
ROOT=Path('/home/cai_tianshun/Project/HOI');E=ROOT/'experiments/structured_hoi_20260923';D=E/'data/dev2'
def main():
    p=ROOT/'envs/mosca/bin/python';g=ROOT/'envs/gvhmr/bin/python'
    commands=[('sam2',[p,ROOT/'scripts/diagnostics/segment_event_rgb.py','--frames',D/'images','--boxes',D/'sam2_boxes.json','--sam2-root',ROOT/'third_party/sam2','--checkpoint',ROOT/'models/SAM2/sam2.1_hiera_small.pt','--output',D/'segmentation','--device','cuda:0','--precision','bfloat16']),
              ('depth',[p,ROOT/'scripts/mosca_event_priors.py','depth','--input',D]),
              ('tracker',[p,ROOT/'scripts/mosca_event_priors.py','tracker','--input',D,'--segmentation',D/'segmentation/segmentation.npz']),
              ('human',[g,E/'code/infer_human_prior.py','--input-manifest',D/'input_manifest.json','--segmentation',D/'segmentation/segmentation.npz','--output',D/'human_prior'])]
    rec=dict(status='running',gpu=0,pid=os.getpid(),stages={},reference_used=False);start=time.perf_counter()
    def save():
        tmp=D/'preprocess_pipeline.tmp';tmp.write_text(json.dumps(rec,indent=2)+'\n');tmp.replace(D/'preprocess_pipeline.json')
    try:
        for name,args in commands:
            rec['current_stage']=name;save();t=time.perf_counter()
            with (D/f'preprocess_{name}.log').open('wb') as out:
                subprocess.run([str(x) for x in args],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'8','PYTHONUNBUFFERED':'1'})
            rec['stages'][name]=dict(status='completed',seconds=time.perf_counter()-t,command=[str(x) for x in args]);save()
        rec['status']='completed'
    except BaseException as ex:rec.update(status='failed',error=repr(ex),traceback=traceback.format_exc());raise
    finally:rec['wall_seconds']=time.perf_counter()-start;save()
if __name__=='__main__':main()
