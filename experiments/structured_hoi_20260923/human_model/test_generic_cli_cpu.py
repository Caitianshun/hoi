"""Variable length acceptance using seven existing legal images; not dev2 inference."""
from pathlib import Path
import json,numpy as np,subprocess,os,hashlib
R=Path('/home/cai_tianshun/Project/HOI');E=R/'experiments/structured_hoi_20260923';O=E/'human_model/generic_cli_fixture7';O.mkdir(exist_ok=False)
m=json.loads((R/'experiments/mosca_baseline_20260922/common_input/input_manifest.json').read_text());idx=[0,16,25,31,55,93,113]
for k in ['frame_paths','frame_sha256','frame_indices','timestamp_seconds','frames']:
 if k in m:m[k]=[m[k][i]for i in idx]
m['sequence']='CPU_fixture7_from_dev1_not_dev2';m['sampling']={'num_frames':7,'method':'CPU CLI fixture from seven existing frozen legal inputs; no model inference'};(O/'input_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
s=np.load(R/'experiments/mosca_baseline_20260922/segmentation/segmentation.npz');np.savez_compressed(O/'segmentation.npz',entity_labels=s['entity_labels'][idx],source_frame_names=s['source_frame_names'][idx],role=s['role'])
cmd=[str(R/'envs/gvhmr/bin/python'),str(E/'code/infer_human_prior.py'),'--input-manifest',str(O/'input_manifest.json'),'--segmentation',str(O/'segmentation.npz'),'--output',str(O/'validation'),'--validate-only'];env=dict(os.environ,CUDA_VISIBLE_DEVICES='');subprocess.run(cmd,env=env,check=True)
out=json.loads((O/'validation/run.json').read_text());assert out['frames']==7 and out['status']=='validated_cpu_only' and out['peak_allocated_bytes']==0
old=np.load(R/'experiments/mosca_interface_validation_20260923/human_prior/prepared_input/sam2_person_boxes.npz');new=np.load(O/'validation/sam2_person_boxes.npz');assert np.array_equal(old['bbox_xyxy'][idx],new['bbox_xyxy'])
full=np.load(E/'human_model/generic_cli_cpu_check/sam2_person_boxes.npz');assert np.array_equal(old['bbox_xyxy'],full['bbox_xyxy'])
record={'status':'passed','fixture_frames':7,'full_validation_frames':114,'boxes_match_original_frozen_arrays':True,'gpu_used':False,'not_a_dev2_prediction':True,'script_sha256':hashlib.sha256((E/'code/infer_human_prior.py').read_bytes()).hexdigest()};(E/'human_model/generic_cli_acceptance_cpu.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
