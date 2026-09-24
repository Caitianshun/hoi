#!/usr/bin/env python3
"""Freeze prediction-only gate intervention, then independently evaluate it.
Uses already computed CPU trajectories; never fits a transform to the reference.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE=ROOT/'experiments/mosca_baseline_20260922'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=HERE);args=parser.parse_args();dest=args.output_dir.resolve();dest.mkdir(parents=True,exist_ok=True)
 if any((dest/f'evaluation_gate_{b}').exists() for b in ['on','off']):raise ValueError('Evaluation output exists; choose a fresh --output-dir under motion_binding for replay.')
 a=np.load(HERE/'frozen_query_motion.npz');q=json.loads((BASE/'common_input/queries_first_frame.json').read_text());validation=json.loads((HERE/'priority_query_gate_results.json').read_text())['validation']
 protocol={'status':'completed','coordinate_frame':'behave_world_k1_color','units':'m','global_alignment_to_reference':'none','reference_used_for_training':False,'source_weight_protocol':'The exact same first-frame ON-model alpha-compositing weights and Gaussian identities are used in both trajectories. Source alpha is not re-rendered after intervention. No nearest-node reassignment. Each Gaussian retains its own original ref_time.','intervention':'Only DQB dyn_o switches from clip(corrected_RBF_sum,0,1) to .999. Source GSs, scaffold curves, quaternions, ref_time, learned skinning correction, camera and static GS remain frozen. No optimization.','interpretation':'Mechanism ablation, not a trained reconstruction or material point tracking guarantee. First positions can shift because GS reference times differ; no reference-based correction is applied.','validation_against_existing_gpu_export':validation,'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'frozen_query_motion.npz',HERE/'audit_motion_binding.py',BASE/'mosca_cotracker/photometric_d_model_native_add3.pth',BASE/'mosca_cotracker/photometric_s_model_native_add3.pth',BASE/'common_input/input_manifest.json',BASE/'common_input/queries_first_frame.json']}}
 # All prediction artifacts are frozen before evaluation reference is opened.
 predpaths=[]
 for name,key in [('gate_on','actual'),('gate_off','dyn_o_off')]:
  p=dest/f'{name}_fixed_source_prediction.npz';x=a[key]
  np.savez_compressed(p,predicted=x,predicted_valid_mask=np.isfinite(x).all(-1),frame_times=a['frame_times'],query_id=a['query_id'],entity=np.array(q['entity'][:6]),coordinate_frame=np.array('behave_world_k1_color'),units=np.array('m'))
  pp=dest/f'{name}_prediction_protocol.json'; pp.write_text(json.dumps({**protocol,'branch':name,'prediction_sha256':sha(p)},ensure_ascii=False,indent=2)+'\n');predpaths.append((name,p,pp))
 frozen={str(p.name):sha(p) for _,p,pp in predpaths for p in [p,pp]};(dest/'prediction_freeze_manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
 sys.path.insert(0,str(ROOT/'scripts/diagnostics'));from evaluate_prediction_reference import main as evaluate
 for name,p,pp in predpaths:
  evaluate(['--prediction',str(p),'--prediction-protocol',str(pp),'--reference',str(BASE/'evaluation/fixed_rgb_queries/same_version_joint_reference_observation_times.npz'),'--reference-protocol',str(BASE/'evaluation/fixed_rgb_queries/same_version_joint_reference_observation_times_protocol.json'),'--input-manifest',str(BASE/'common_input/input_manifest.json'),'--method',f'MoSca_frozen_{name}_fixed_source','--output',str(dest/f'evaluation_{name}')])
 for p,digest in frozen.items():assert sha(dest/p)==digest
if __name__=='__main__':main()
