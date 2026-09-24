from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;P=O/'dev2_refined01';E=O.parent;B=E/'data/dev2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=np.load(P/'object_init.npz');d=np.load(P/'input_diagnostics.npz');m=json.loads((B/'input_manifest.json').read_text());rr=json.loads((P/'run.json').read_text());fr=json.loads((O/'dev2_run01/run.json').read_text());times=a['timestamp_seconds'];R=a['R_world'].astype(float);t=a['t_world_m'].astype(float);V=a['centres_m'].astype(float);base=np.linalg.norm(V[1:50]-V[0],axis=1);err=[]
for rot,tr in zip(R,t):w=V[:50]@rot.T+tr;err.append(abs(np.linalg.norm(w[1:]-w[0],axis=1)-base).max())
c2w=a['c2w'];world=V[:50][None]@np.swapaxes(R,1,2)+t[:,None];cam=(world-c2w[:3,3])@c2w[:3,:3];direct=V[:50][None]@np.swapaxes(a['R_camera'].astype(float),1,2)+a['t_camera_m'].astype(float)[:,None];obs=a['observed_mask'];ang=np.rad2deg(d['rotation_increment_rad']);ii=int(ang.argmax())+1
q={'status':'completed_input_screen','final_NPZ_sha256':sha(P/'object_init.npz'),'final_npz_immutable':True,'all_98_RGB_hashes_match':all(sha(p)==h for p,h in zip(m['frame_paths'],m['frame_sha256'])),'source_init_input_hashes_match':all(sha(p)==h for p,h in fr['input_hashes'].items()) and all(sha(p)==h for p,h in fr['depth_input_hashes'].items()),'role':'warmup initialization with explicit uncertainty; not independent pose validation','per_frame_reference_or_alignment_used':False,'scan_is_explicit_geometry_only_input':True,'texture_used':False,'sample_count':len(V),'template_vertices':len(a['canonical_vertices_m']),'template_faces':len(a['faces']),'frames':len(R),'direct_2d_observation_frames':int(obs.sum()),'depth_observation_frames':int(a['depth_observed_mask'].sum()),'insufficient_observation_frames':np.where(~obs)[0].tolist(),'unobserved_reliability_zero':bool((a['reliability'][~obs]==0).all()),'rotation_ambiguity_all_marked':bool(a['rotation_ambiguity'].all()),'max_pairwise_rigidity_error_serialized_float32_m':float(max(err)),'max_world_to_camera_roundtrip_error_m':float(abs(cam-direct).max()),'R_orthogonality_error':float(abs(R@np.swapaxes(R,1,2)-np.eye(3)).max()),'R_determinant_minmax':[float(np.linalg.det(R).min()),float(np.linalg.det(R).max())],'all_R_t_finite':bool(np.isfinite(R).all() and np.isfinite(t).all()),'times_strictly_increasing':bool((np.diff(times)>0).all()),'dt_minmax_s':[float(np.diff(times).min()),float(np.diff(times).max())],'input_mask_IoU_observed_min_median_max':np.percentile(d['visible_silhouette_iou'][obs],[0,50,100]).tolist(),'center_speed_median_max_m_s':np.percentile(d['translation_speed_m_s'],[50,100]).tolist(),'maximum_neighbor_rotation':{'from_frame':ii-1,'to_frame':ii,'degrees':float(ang[ii-1]),'dt_s':float(times[ii]-times[ii-1]),'rate_deg_s':float(ang[ii-1]/(times[ii]-times[ii-1]))},'visual_review':['input_overlay_1.png','input_overlay_2.png','input_overlay_3.png'],'GPU_used':False}
assert q['all_98_RGB_hashes_match'] and q['source_init_input_hashes_match'];assert max(err)<1e-6 and abs(cam-direct).max()<1e-6
(P/'verification.json').write_text(json.dumps(q,indent=2))
stages=[]
for name in ['dev2_run01','dev2_reverse01','dev2_bidirectional01','dev2_refined01']:
 r=json.loads((O/name/'run.json').read_text());stages.append({'stage':name,'status':r['status'],'wall_seconds':r['wall_seconds'],'output_sha256':sha(O/name/'object_init.npz')})
(P/'timing_and_provenance.json').write_text(json.dumps({'stages':stages,'total_main_compute_seconds':sum(x['wall_seconds'] for x in stages),'excludes':'image writing after run.json, source review, QA and image reading','source_template_manifest':str((B/'object_template_manifest.json').resolve()),'source_template_manifest_sha256':sha(B/'object_template_manifest.json'),'source_initialization_records':[str((O/x/'run.json').resolve()) for x in ['dev2_run01','dev2_reverse01','dev2_bidirectional01']], 'final_refinement_record':str((P/'run.json').resolve()),'known_geometry_permission_audit':str((O/'INPUT_PERMISSION_AUDIT.json').resolve())},indent=2))
(O/'dev2_selected_initialization.json').write_text(json.dumps({'adopted_run':'dev2_refined01','path':str((P/'object_init.npz').resolve()),'sha256':q['final_NPZ_sha256'],'world_unit':'meters','canonical_transform':'X_world = X_canonical @ R_world[t].T + t_world_m[t]','same_initialization_for_S0_S1':True,'unobserved_reliability_zero':True,'independent_pose_alignment_absent':True},indent=2))
fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True)
for name,col in [('dev2_run01','0.65'),('dev2_bidirectional01','tab:orange'),('dev2_refined01','tab:blue')]:
 z=np.load(O/name/'input_diagnostics.npz');axs[0].plot(times,z['visible_silhouette_iou'],label=name,color=col);axs[1].plot(times[1:],z['translation_speed_m_s'],label=name,color=col)
for ax in axs:
 ax.grid(alpha=.2);ax.axvspan(times[61],times[67],color='grey',alpha=.15);ax.legend(fontsize=8)
axs[0].set_ylabel('input visible mask IoU');axs[1].set_ylabel('center speed (m/s)');axs[1].set_xlabel('actual acquisition timestamp (s)');fig.suptitle('Chair ordinary initialization diagnostics; grey = insufficient observations\nThese are input-fit and motion diagnostics, not independent accuracy')
fig.tight_layout();fig.savefig(P/'motion_fit_diagnostic.png',dpi=160);plt.close(fig)
md=f'''# dev2 chairwood 已知形状刚体初始化

**已完成供 S0/S1 共享预热使用的初始化，采用 `dev2_refined01/object_init.npz`，输出 98 帧世界米制刚体姿态。** 这是经过输入侧检查的起点；没有使用独立参考或对齐，也没有证据证明全部三维位姿正确。未充分观测的 61–67 帧只保留时序预测，可靠性为零；旋转歧义继续保留，不能把初始化硬锁为真值。

## 输入、拓扑与方法

输入仅为新协议登记的 camera 0 RGB/SAM2/CoTracker/UniDepth、已知 K/c2w，以及 `object_template_geometry.npz` 的 vertices/faces。几何来自官方 `objects/chairwood/chairwood_f2500.ply`，1,609 顶点、2,499 面；其登记清单明确不含颜色、纹理或逐帧姿态。所有输入文件在完成预处理后读取，源哈希与 98 张 RGB 清单均再核对通过。

木椅有杆件与孔洞，不能照搬箱体的凸包轮廓。通用入口按已知模板拓扑预先选择 surface 模式：拟合使用实际模板表面的双向投影距离；诊断使用三角形投影并集，保留孔洞。此选择由形状类型决定，不根据参考误差决定。

执行经历四个有明确作用的阶段：

1. **普通前向刚体跟随**：首端多候选初始化，后续使用输入二维跟踪、估计深度和轮廓。实际图像检查发现遮挡后落入错误椅背方向，故保存为失败起点，没有直接交付训练。
2. **从末端反向跟随**：利用末段真实可见输入独立初始化并反向配准，提供遮挡后更合适的候选；反向结果在前段也可能出错，不能直接整段替换。
3. **输入代价与真实 dt 选择连续候选**：用 `1−可见掩码IoU` 作为每帧代价，并加入按真实 dt 计算的平移/角速度代价。动态规划选择前向 61 个观测帧、反向 30 个观测帧，切换在 68 帧；弱观测间隔仍不计真实观测。
4. **全段普通 SE(3) 连续性细化**：共同优化每帧 R/t，包含同输入轮廓、弱估计深度及按真实时间计算的平移/角加速度正则。采用疏疏 Jacobian 的 CPU 最小二乘，未加入新网络、接触约束或其他视角。该阶段是普通初始化与轨迹正则，不能当作 S2/S3 的方法贡献。

最后阶段仍允许隐藏姿态由时序约束改变，但不会将其转记为观测。`observed_mask` 保留输入侧状态；`selected_direction` 仅记录细化前双向来源，最终姿态不再精确等于某一候选。

## 完成结果与边界

| 项目 | 结果 |
| --- | --- |
| 输出帧数 | 98 |
| 有充分二维观测 / 深度通过帧门 | 91 / 82 |
| 无充分直接观测帧 | 61–67，可靠性均为 0 |
| 最终观测帧输入轮廓IoU 最低 / 中位 / 最高 | {q['input_mask_IoU_observed_min_median_max'][0]:.3f} / {q['input_mask_IoU_observed_min_median_max'][1]:.3f} / {q['input_mask_IoU_observed_min_median_max'][2]:.3f} |
| 双向候选 → 细化后的中心最大速度 | 10.600 → {q['center_speed_median_max_m_s'][1]:.3f} m/s |
| 双向候选 → 细化后的IoU中位 | 0.690 → {q['input_mask_IoU_observed_min_median_max'][1]:.3f} |
| 最大邻帧旋转及真实dt | {q['maximum_neighbor_rotation']['degrees']:.2f}°，42→43，dt={q['maximum_neighbor_rotation']['dt_s']:.6f}s |
| 保存 float32 后最大成对距离变化 | {q['max_pairwise_rigidity_error_serialized_float32_m']:.3e} m |
| 世界到相机回算最大差 | {q['max_world_to_camera_roundtrip_error_m']:.3e} m |
| 最终求解终止 | `{rr['optimizer_message']}` |

中心跳变减小的同时轮廓贴合没有整体牺牲，这是可进入分支预热的输入侧依据；**不能据此认定运动更接近真实值**。最大邻帧角度出现在 0.366667 秒的采样间隔中，不能直接按 0.1 秒解释；临遮挡仍有明显方向变化和不确定性。后续人体、物体、场景组合和独立世界位姿评价仍是必要验证。

实际查看最终 3 张拼图中的 12 帧：首尾椅背方向与可见外形相容，薄杆/座面在快速转动和人手遮挡处仍有偏差，整体输入IoU约0.72不等于充分三维正确。末端椅背错误来自先前单向局部极值，已由输入侧双端初始化纠正；不得把这个工程改动写成独立遮挡恢复创新。

## 成本与复现

| 阶段 | 主计算秒数 |
| --- | ---: |
'''
for stage in stages:md+=f"| {stage['stage']} | {stage['wall_seconds']:.3f} |\n"
md+=f'''| 合计 | {sum(x['wall_seconds'] for x in stages):.3f} |

全部在 CPU 执行，未占用 GPU。上述时间不含写图后的阅读、代码实现、检查与报告；不能当作完整系统训练成本。

最终文件：[object_init.npz]({P}/object_init.npz)，SHA256 `{q['final_NPZ_sha256']}`。API 与箱体相同：`X_world = centres_m @ R_world[t].T + t_world_m[t]`，已应用已知 c2w；D 归一化时只按 s 缩放中心与平移一次。所有高斯共享一帧的单一刚体变换，方向/协方差也需旋转。

[通用入口]({E}/code/object_init_general.py) · [双向选择]({E}/code/object_init_bidirectional.py) · [全段细化]({E}/code/object_init_refine.py) · [验收]({P}/verification.json) · [成本与来源]({P}/timing_and_provenance.json) · [扫描输入许可核查]({O}/INPUT_PERMISSION_AUDIT.json)

![木椅初始化轮廓与运动诊断；不是独立误差]({P}/motion_fit_diagnostic.png)

'''
for k in range(1,4):md+=f'![木椅同输入轮廓叠图 {k}：绿为SAM2，品红为完整mesh投影]({P}/input_overlay_{k}.png)\n\n'
md=md.replace('疏疏 Jacobian','稀疏 Jacobian')
(O/'DEV2_REPORT.md').write_text(md)
print(json.dumps(q,indent=2));print('total_seconds',sum(x['wall_seconds'] for x in stages))
