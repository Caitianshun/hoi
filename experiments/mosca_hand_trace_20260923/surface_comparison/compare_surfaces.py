#!/usr/bin/env python3
"""Compare identical A/C surface probes. Does not execute or fit a model."""
from pathlib import Path
import argparse,json,csv,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'experiments/mosca_hand_trace_20260923'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--a',type=Path,default=RUN/'surface_audit/surface_audit.json')
 p.add_argument('--c',type=Path,default=RUN/'surface_audit_c/surface_audit.json')
 p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
 args=p.parse_args();a=read(args.a);c=read(args.c);args.output.mkdir(parents=True,exist_ok=True)
 assert a['status']==c['status']=='completed_cpu_only_evaluation'
 assert a['mapping']==c['mapping'] and a['coordinate_frame']==c['coordinate_frame'] and a['units']==c['units']=='m'
 assert abs(a['normalization_scale']-c['normalization_scale'])<1e-12
 def key(r):return r['query_id'],r['input_frame_index'],r['sample_location']
 aa={key(r):r for r in a['rows']};cc={key(r):r for r in c['rows']};assert len(aa)==len(cc)==56 and aa.keys()==cc.keys()
 identical_fields=['query_id','input_frame_index','reference_index','time_seconds','sample_location',
   'tracker_claims_visible','tracker_uv','reference_uv','tracker_reference_pixel_distance','reference_world_m',
   'reference_camera_z_m','nominal_reference_time_seconds','reference_time_offset_seconds','uv']
 rows=[]
 for k in aa:
  ar,cr=aa[k],cc[k]
  for f in identical_fields:assert ar[f]==cr[f],(k,f,ar[f],cr[f])
  row={f:ar[f] for f in ['query_id','input_frame_index','time_seconds','sample_location','tracker_claims_visible']}
  row['reference_depth_m']=ar['reference_camera_z_m']
  for name,r in [('A',ar),('C',cr)]:
   for f in ['alpha','dynamic_fraction','static_fraction','rendered_depth_bilinear_m','weighted_centroid_camera_z_m',
      'nearest_half_alpha_mean_depth_m','fixed_source_camera_z_m','fixed_source_3d_epe_m',
      'depth_signed_error_to_fitted_point_m','ray_backprojected_epe_to_fitted_point_m']:
    row[name+'_'+f]=r[f]
  rows.append(row)
 with (args.output/'paired_surface_samples.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summaries=[]
 for q in a['mapping']:
  for mode in ['fixed_source','training_tracker','reference_projection']:
   sr={'query_id':q,'mode':mode,'count':14}
   for name,z in [('A',a),('C',c)]:
    rr=[r for r in z['rows'] if r['query_id']==q and r['sample_location']==('reference_projection' if mode=='fixed_source' else mode)]
    errors=[r['fixed_source_3d_epe_m'] if mode=='fixed_source' else r['ray_backprojected_epe_to_fitted_point_m'] for r in rr]
    dz=[r['fixed_source_camera_z_m']-r['reference_camera_z_m'] if mode=='fixed_source' else r['depth_signed_error_to_fitted_point_m'] for r in rr]
    sr[name+'_mean_3d_difference_m']=float(np.mean(errors));sr[name+'_median_3d_difference_m']=float(np.median(errors))
    sr[name+'_mean_absolute_depth_difference_m']=float(np.mean(np.abs(dz)))
    if mode!='fixed_source':
     sr[name+'_alpha_min']=min(r['alpha'] for r in rr);sr[name+'_dynamic_fraction_mean']=float(np.mean([r['dynamic_fraction'] for r in rr]))
   summaries.append(sr)
 result=dict(status='completed_cpu_comparison',same_56_probe_identity_verified=True,same_world_scale=True,
   modes_separate=True,gt_projected_probe_is_not_method_improvement=True,reference_interpolation=False,
   coordinate_frame=a['coordinate_frame'],units='m',summary=summaries,
   sources={'A':str(args.a.resolve()),'C':str(args.c.resolve())},
   input_sha256={str(p.resolve()):sha(p) for p in [args.a,args.c]},script_sha256=sha(__file__))
 (args.output/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 title={'hand_glove_centre':'手套中心','hand_glove_upper':'手套上部'}
 mode_title={'fixed_source':'固定源轨迹','training_tracker':'当前 tracker UV','reference_projection':'当前 fitted UV（仅评价）'}
 lines=['# A／C 同协议表面查询对照','',
 '完成范围：两个固定手套查询 × 14 个独立匹配时刻 × 两类预定像素，共 56 个当前表面查询，另分别报告既有固定源轨迹。仅 CPU；未训练、改模型或改 A 结果。','',
 '**三种量必须分开理解。** 固定源轨迹保留同一首帧高斯贡献身份；tracker UV 分支随输入跟踪坐标重新查询当前模型；fitted UV 分支由独立参考提供每时刻二维位置，只检查该射线上的渲染深度。重新查询改变了高斯身份，特别是 fitted UV 不能当作方法跟踪收益。','',
 '| 查询 | 量 | A 平均三维差 | C 平均三维差 | A 平均绝对深度差 | C 平均绝对深度差 |',
 '| --- | --- | ---: | ---: | ---: | ---: |']
 for s in summaries:
  lines.append('| '+title[s['query_id']]+' | '+mode_title[s['mode']]+' | '+' | '.join(f'{s[k]*100:.2f} cm' for k in ['A_mean_3d_difference_m','C_mean_3d_difference_m','A_mean_absolute_depth_difference_m','C_mean_absolute_depth_difference_m'])+' |')
 lines+=['','全部时刻均保留，未按 C 的好坏或 tracker 可见性删样本。两个 tracker 在 14 时刻中都仅有 5 个声称可见；其坐标可能错误。参考沿用同版拟合人体表面，真实手套边界、材料对应、遮挡与时间偏差仍未解决；高 alpha 不代表正确手几何。','',
 '末帧 fitted UV 上的分层解释如下。近半贡献是按模型深度由近至远累计到总 alpha 的前 50% 后加权平均，不按拟合参考挑高斯：','',
 '| 查询／分支 | 拟合 z | 当前期望 z | 近半贡献 z | 静态比例 | alpha |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
 for q in a['mapping']:
  for name,z in [('A',a),('C',c)]:
   r=next(r for r in z['rows'] if r['query_id']==q and r['sample_location']=='reference_projection' and r['input_frame_index']==113)
   lines.append(f"| {title[q]}／{name} | {r['reference_camera_z_m']:.3f} m | {r['rendered_depth_bilinear_m']:.3f} m | {r['nearest_half_alpha_mean_depth_m']:.3f} m | {r['static_fraction']*100:.2f}% | {r['alpha']:.5f} |")
 lines+=['','手套边缘上的近／远层混合也可能受到拟合轮廓偏差影响；更近的层在遮挡时可能是合理的遮挡者，不能仅凭本表认定新的错误类型。','',
 f"C 全部样本与 CPU 复现检查见 [surface_audit.json]({args.c.resolve()})，A 原结果保留于 [原审计]({args.a.resolve()})。",
 f"逐样本配对结果见 [CSV]({(args.output/'paired_surface_samples.csv').resolve()})；汇总与来源哈希见 [JSON]({(args.output/'comparison.json').resolve()})。",'',
 f"![C 固定源与当前表面对照]({(args.c.parent/'surface_depth_and_reference_difference.png').resolve()})",'',
 f"![C 输入及实际渲染的固定查询位置]({(args.c.parent/'fixed_pixels_input_and_branch_render.png').resolve()})",'',
 '独立复算使用相同的已验收 CPU DQB 与稀疏 alpha 合成器，实际时间／查询身份／参考坐标／像素位置逐项一致，无对齐或插值。图片需要完成本地目检后再作为最终交付；本文件生成不代表应用界面验收。','']
 (args.output/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps(result['summary'],indent=2,ensure_ascii=False))

if __name__=='__main__':main()
