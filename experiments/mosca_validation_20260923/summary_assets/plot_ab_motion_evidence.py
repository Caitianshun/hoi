#!/usr/bin/env python3
"""CPU-only old/A/B fixed-query comparison, strictly at 14 reference times.

Example now: --branches A
When B is complete: --branches A B --summary /absolute/path/to/summary_ab/summary.json
The old pilot is always included. Missing requested branches are an error, not skipped.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/mosca_validation_20260923'
OLD=ROOT/'experiments/mosca_baseline_20260922'
QUERIES=['object_front_brown','hand_glove_centre']
TITLES=['Box: brown surface','Hand: glove centre']
COLORS={'Old':'#c45b45','A':'#247fa5','B':'#568537'}
MARKERS={'Old':'o','A':'^','B':'s'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--branches',nargs='+',choices=['A','B'],default=['A'])
 p.add_argument('--summary',type=Path,default=EXP/'summary_a/summary.json')
 p.add_argument('--a-dir',type=Path,default=EXP/'a_normalized_exact')
 p.add_argument('--b-dir',type=Path,default=EXP/'b_normalized_static_bg')
 p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
 args=p.parse_args()
 if len(set(args.branches))!=len(args.branches):p.error('Duplicate branches')
 args.output.mkdir(parents=True,exist_ok=True)
 entries=[('Old','old_pilot',OLD/'evaluation/mosca_cotracker_final')]
 for label,directory in [('A',args.a_dir),('B',args.b_dir)]:
  if label in args.branches:
   pipeline=read(directory/'pipeline.json')
   if pipeline['status']!='completed':raise ValueError(f'{label} has not completed: {directory}')
   entries.append((label,pipeline['branch'],directory/'evaluation'))
 summary=read(args.summary)
 if summary['status']!='completed':raise ValueError('Summary must be completed')
 names={row['name']:row for row in summary['runs']}
 input_paths=[args.summary]
 data={};metrics={};ref=None
 for label,name,directory in entries:
  if name not in names:raise ValueError(f'{name} missing from --summary; regenerate completed branch summary')
  bundle=directory/'evaluation_bundle.npz';report_path=directory/'three_dimensional/report.json'
  input_paths.extend([bundle,report_path])
  if sha(report_path)!=names[name]['evaluation_sha256']:raise ValueError(f'{name}: summary/report hash mismatch')
  report=read(report_path)
  if report['input_sha256']!=sha(bundle):raise ValueError(f'{name}: report/bundle hash mismatch')
  with np.load(bundle,allow_pickle=False) as z: d={k:z[k].copy() for k in z.files}
  if d['units'].item()!='m' or d['coordinate_frame'].item()!='behave_world_k1_color':raise ValueError('Unexpected units/coordinate frame')
  if ref is None:ref=d
  else:
   for key in ['reference','valid_mask','frame_times','query_id','entity','visibility']:
    if not np.array_equal(d[key],ref[key]):raise ValueError(f'Not the same frozen reference identity: {key}')
  if d['predicted'].shape!=(14,6,3):raise ValueError('Expected 14 matched times, six frozen queries')
  if not np.all(np.diff(d['frame_times'])>0):raise ValueError('Times must increase')
  if not (np.all(d['valid_mask']) and np.all(d['predicted_valid_mask'])):raise ValueError('This plot requires the declared common full coverage')
  if not (np.isfinite(d['predicted']).all() and np.isfinite(d['reference']).all()):raise ValueError('Nonfinite trajectory')
  # No fit/transform: the absolute error uses stored world coordinates directly.
  error=np.linalg.norm(d['predicted'].astype(float)-d['reference'],axis=-1)
  if abs(error.mean()-names[name]['absolute']['mean_epe_m'])>1e-7:raise ValueError('Summary mean differs from raw bundle')
  data[label]=d;metrics[label]={}
  for query in QUERIES:
   qi=d['query_id'].tolist().index(query)
   xyz=d['predicted'][:,qi].astype(float)
   dis=np.linalg.norm(xyz-xyz[0],axis=-1)
   ee=error[:,qi]
   metrics[label][query]={'displacement_m':dis.tolist(),'absolute_error_m':ee.tolist(),
                          'max_sampled_displacement_m':float(dis.max()),
                          'mean_absolute_error_m':float(ee.mean()),
                          'first_absolute_error_m':float(ee[0]),'last_absolute_error_m':float(ee[-1])}
 before={str(path.resolve()):sha(path) for path in input_paths}
 times=ref['frame_times'];reference_displacement={}
 for query in QUERIES:
  qi=ref['query_id'].tolist().index(query);xyz=ref['reference'][:,qi]
  reference_displacement[query]=np.linalg.norm(xyz-xyz[0],axis=-1)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                      'axes.spines.right':False,'savefig.facecolor':'white'})
 fig,axes=plt.subplots(2,2,figsize=(13,8.7),sharex=True,sharey='col')
 fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.785,wspace=.21,hspace=.32)
 fig.suptitle('Motion and independent 3D error',fontsize=22,fontweight='bold',y=.97)
 fig.text(.5,.921,'Same fixed queries  |  14 matched observation times  |  No alignment',ha='center',fontsize=12)
 maxdisp=max(max(reference_displacement[q].max() for q in QUERIES),max(metrics[k][q]['max_sampled_displacement_m'] for k in data for q in QUERIES))
 maxerror=max(max(metrics[k][q]['absolute_error_m']) for k in data for q in QUERIES)
 for ri,query in enumerate(QUERIES):
  left,right=axes[ri]
  for label in data:
   mm=metrics[label][query]
   left.plot(times,mm['displacement_m'],color=COLORS[label],marker=MARKERS[label],
             markersize=4.5,lw=1.9,label=label,alpha=.94)
   right.scatter(times,mm['absolute_error_m'],color=COLORS[label],marker=MARKERS[label],s=43,label=label,alpha=.94)
  left.scatter(times,reference_displacement[query],color='#222936',marker='D',s=35,label='Fitted ref.',zorder=6)
  left.set_title(TITLES[ri]+' — displacement',fontsize=13,pad=9)
  right.set_title(TITLES[ri]+' — position error',fontsize=13,pad=9)
  left.set_ylabel('Displacement (m)');right.set_ylabel('3D error (m)')
  left.set_ylim(0,maxdisp*1.27);right.set_ylim(0,maxerror*1.27)
  error_text='Mean: '+' | '.join(f'{label} {metrics[label][query]["mean_absolute_error_m"]:.3f}' for label in data)+' m'
  right.text(.025,.955,error_text,transform=right.transAxes,va='top',fontsize=10)
  for ax in [left,right]:
   ax.grid(True,alpha=.17);ax.set_xlim(times[0]-.25,times[-1]+.25);ax.set_xticks(np.arange(18,32,2))
   if ri==1:ax.set_xlabel('Actual observation time (s)')
 handles,labels=axes[0,0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.885),ncol=len(data)+1,frameon=False,fontsize=12)
 fig.text(.08,.064,'Displacement is measured from each trajectory’s own first position; absolute error remains unaligned.',fontsize=10)
 fig.text(.08,.035,'Fitted reference: 14 points, no interpolation. Predictions are sampled at the same times; lines only guide the eye.',fontsize=10)
 stem='ab_motion_old_'+ '_'.join(label.lower() for label in data if label!='Old')
 png=args.output/f'{stem}.png';fig.savefig(png,dpi=175);plt.close(fig)
 provenance={'status':'completed_cpu_only','branch_names':{label:name for label,name,_ in entries},
             'query_id':QUERIES,'frame_times_seconds':times.tolist(),
             'coordinate_frame':'behave_world_k1_color','units':'m','global_alignment':'none',
             'reference_interpolation':False,'prediction_times_plotted':14,
             'plot_reference':'Fitted surfaces, not sensor ground truth or visibility truth.',
             'source_summary':str(args.summary.resolve()),'input_sha256':before,
             'metrics':metrics,'reference_displacement_m':{k:v.tolist() for k,v in reference_displacement.items()},
             'script_sha256':sha(__file__),'png_sha256':sha(png),
             'visual_review':'pending; local figure must be inspected after each branch set is generated'}
 assert before=={str(path.resolve()):sha(path) for path in input_paths},'Input changed while plotting'
 (args.output/f'{stem}.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'figure':str(png),'branch_set':list(data),'means_m':{label:{q:metrics[label][q]['mean_absolute_error_m'] for q in QUERIES} for label in data}},indent=2))

if __name__=='__main__':main()
