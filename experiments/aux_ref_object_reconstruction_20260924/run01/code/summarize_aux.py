"""Read-only aggregation; no optimizer, model selection, or modified training log."""
from pathlib import Path
import hashlib,json
import numpy as np
E=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def main():
 assert read(E/'pipeline.json')['status']=='completed'
 support=E/'diagnostics/step_support';support.mkdir(exist_ok=True)
 fixed=read(E/'diagnostics/fixed_scene/summary.json');report={'protocol_id':'AUX_REF_OBJECT','runs':{},'transmittance':{}}
 for dev in ['dev1','dev2']:
  fr=read(E/'diagnostics/fixed_scene'/dev/'manifest.json')['rows'];byframe={r['frame_index']:r for r in fr}
  for arm in ['Pred','Ref']:
   key=f'{dev}_{arm}';r=read(E/'runs'/key/'run.json');assert r['status']=='completed' and r['step']==8000
   stats=r['object_stats'];rows=[json.loads(s) for s in (E/'runs'/key/'steps.jsonl').read_text().splitlines()]
   assert len(rows)==8000 and [x['step'] for x in rows]==list(range(1,8001))
   fit=read(E/'runs'/key/'camera0_fit.json')['rows'];validfit=[r['object_psnr'] for r in fit if r['object_psnr'] is not None]
   report['runs'][key]={'points':stats['count'],'gradient_steps':r['effective_data_gradient_steps'],'zero_data_steps':r['zero_data_gradient_steps'],'max_scale_mm':stats['scale_m']['1']*1000,'max_offset_mm':stats['offset_norm_m']['1']*1000,'opacity_median':stats['opacity']['0.5'],'scale_median_mm':stats['scale_m']['0.5']*1000,'opacity_p05':stats['opacity']['0.05'],'fraction_alpha_lt002':stats['fraction_alpha_lt002'],'fraction_scale_gt0035':stats['fraction_scale_gt0035'],'wall_seconds':r['wall_seconds'],'peak_allocated_GiB':r['peak_allocated_bytes']/2**30,'peak_reserved_GiB':r['peak_reserved_bytes']/2**30,'camera0_fit_mean_psnr':float(np.mean(validfit)),'camera0_valid_O_frames':len(validfit),'camera0_total_frames':len(fit),'H_S_motion_unchanged':r['H_S_motion_unchanged'],'object_motion_source':'rgb_s1' if arm=='Pred' else 'published_fit','data_gradient_norm_means':{k:float(np.mean([x['data_grad_norm'][k] for x in rows])) for k in rows[0]['data_grad_norm']},'reg_gradient_norm_means':{k:float(np.mean([x['reg_grad_norm'][k] for x in rows])) for k in rows[0]['reg_grad_norm']}}
   a=fixed['devs'][dev]['aggregate']['arms'][arm];report['transmittance'][key]={'valid_samples':a['valid_template_depth_samples'],'low_T_fraction':a['low_T']['0.1']['fraction'],'uncovered_O_fraction':a['all_O_template_uncovered']['fraction'],'mean_T':a['T_HS_before_template']['mean']}
   with (support/f'{key}.jsonl').open('w') as out:
    for x in rows:
     out.write(json.dumps({'protocol_id':'AUX_REF_OBJECT','step':x['step'],'frame':x['frame'],'data_gradient_nonzero':x['data_gradient_nonzero'],'data_grad_norm':x['data_grad_norm'],'reg_grad_norm':x['reg_grad_norm'],'O_pixels':x['O_pixels'],'O_contribution_mean':x['O_contribution_mean'],'fixed_scene':byframe[x['frame']]['arms'][arm]})+'\n')
 report['cost']={'formal_training_seconds':sum(r['wall_seconds'] for r in report['runs'].values()),'pipeline_wall_seconds':read(E/'pipeline.json')['wall_seconds'],'evaluation_seconds':read(E/'evaluation/cross_pose/run.json')['seconds'],'max_training_allocated_GiB':max(r['peak_allocated_GiB'] for r in report['runs'].values()),'fixed_scene_CPU_seconds_sum':sum(r['seconds'] for r in fixed['devs'].values())}
 (E/'numerical_summary.json').write_text(json.dumps(report,indent=2)+'\n')
 (support/'README.md').write_text('Static fixed-scene template-depth T was computed independently during this execution and joined by the frozen input frame index after training. It does not depend on the learned object bank, and is constant at every visit to that frame. These derived records preserve the original training step logs; no loss weight or frame was changed based on this diagnostic. Template surface is a geometric proxy, not the changing Gaussian support depth.\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
