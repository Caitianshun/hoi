from pathlib import Path
import json,hashlib,shutil
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;P=O/'run03'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=np.load(P/'object_init.npz');d=np.load(P/'input_diagnostics.npz');r=json.loads((P/'run.json').read_text());diag=json.loads((P/'frame_diagnostics.json').read_text());stats=json.loads((P/'fit_diagnostics.json').read_text());R=a['R_world'].astype(float);t=a['t_world_m'].astype(float);V=a['centres_m'].astype(float);origdist=np.linalg.norm(V[1:50]-V[0],axis=1);err=[]
for ri,ti in zip(R,t):w=V[:50]@ri.T+ti;err.append(np.max(np.abs(np.linalg.norm(w[1:]-w[0],axis=1)-origdist)))
c2w=a['c2w'];Rc=a['R_camera'].astype(float);tc=a['t_camera_m'].astype(float);world=V[:50][None]@np.swapaxes(R,1,2)+t[:,None];cam=(world-c2w[:3,3])@c2w[:3,:3];direct=V[:50][None]@np.swapaxes(Rc,1,2)+tc[:,None];obs=a['observed_mask'];times=a['timestamp_seconds'];empty=np.array([x['object_pixels']==0 for x in diag]);iou=d['visible_silhouette_iou']
q={'status':'completed','selected':'run03','NPZ_sha256':sha(P/'object_init.npz'),'serialized_float32_max_pair_distance_error_m':float(max(err)),'serialized_float32_R_orthogonality_max_abs':float(np.max(abs(R@np.swapaxes(R,1,2)-np.eye(3)))),'serialized_float32_determinants_minmax':[float(np.linalg.det(R).min()),float(np.linalg.det(R).max())],'world_to_camera_max_point_difference_m':float(np.max(abs(cam-direct))),'known_camera_roundtrip_implemented_once':True,'all_finite':all(np.isfinite(a[k]).all() for k in ['centres_m','R_world','t_world_m']),'timestamps_strictly_increasing':bool((np.diff(times)>0).all()),'dt_minmax_seconds':[float(np.diff(times).min()),float(np.diff(times).max())],'observed_frames':int(obs.sum()),'depth_observed_frames':int(a['depth_observed_mask'].sum()),'estimated_2d_only_frames':np.where(obs&~a['depth_observed_mask'])[0].tolist(),'interpolated_unreliable_frames':np.where(~obs)[0].tolist(),'SAM2_empty_object_frames':np.where(empty)[0].tolist(),'reliability_zero_on_interpolated_frames':bool((a['reliability'][~obs]==0).all()),'rotation_ambiguity_all_marked':bool(a['rotation_ambiguity'].all()),'input_mask_IoU_on_103_fitted_frames_min_median_max':np.percentile(iou[obs],[0,50,100]).tolist(),'hidden_empty_mask_iou_not_evaluated':True,'median_object_track_pairs_on_fitted_frames':float(np.median([s['track_pairs'] for s in stats if s['observed']])),'GPU_used':False,'original_inputs_hashes_still_match':all(sha(p)==h for p,h in r['input_hashes'].items()) and all(sha(p)==h for p,h in r['depth_input_hashes'].items()),'visual_review':['input_overlay_1.png','input_overlay_2.png','input_overlay_3.png'],'state':'initialization ready for branch warmup, no independent pose accuracy claim'}
assert q['serialized_float32_max_pair_distance_error_m']<1e-6 and q['world_to_camera_max_point_difference_m']<1e-6
assert q['original_inputs_hashes_still_match'];(P/'verification.json').write_text(json.dumps(q,indent=2))
# Permanent explicit adoption index and convenience link; historical runs remain preserved.
link=O/'object_init.npz'
if not link.exists():link.symlink_to('run03/object_init.npz')
(O/'selected_initialization.json').write_text(json.dumps({'adopted_run':'run03','path':str((P/'object_init.npz').resolve()),'sha256':q['NPZ_sha256'],'canonical_transform':'X_world = X_canonical @ R_world[t].T + t_world_m[t]','world_unit':'meters','all_branches_must_reuse_same_file':True,'uncertain_poses_remain_optimizable':True},indent=2))
fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
axs[0].plot(times,d['raw_object_depth_median'],'.-',label='UniDepth visible-mask median (surface estimate)');axs[0].plot(times,tc[:,2],label='Rigid template center camera z');bad=~a['depth_observed_mask'];axs[0].scatter(times[bad],np.asarray(d['raw_object_depth_median'])[bad],c='red',marker='x',label='Depth not admitted');axs[0].set_ylabel('camera z (m)');axs[0].legend(fontsize=8)
for k,name in enumerate(['x','y','z']):axs[1].plot(times,tc[:,k],label=name)
axs[1].legend();axs[1].set_ylabel('center camera coords (m)')
axs[2].plot(times[1:],d['translation_speed_m_s'],label='center speed');axs[2].set_ylabel('m/s');axs[2].set_xlabel('actual acquisition timestamp (s)')
for ax in axs:
 for i in np.where(~obs)[0]:ax.axvspan(times[max(0,i-1)],times[min(len(times)-1,i+1)],alpha=.08,color='grey')
 ax.grid(alpha=.2)
fig.suptitle('Input-only rigid initialization; grey = no sufficient direct observation\nSurface depth and center depth refer to different positions, not an error curve')
fig.tight_layout();fig.savefig(P/'input_depth_motion_diagnostic.png',dpi=160);plt.close(fig)
md=f'''# boxsmall 输入侧刚体初始化

**已完成可接入结构化 S0/S1 的初始化，采用 run03：114 帧统一世界米制 SE(3)、517 个扫描顶点、1,000 个面、4,096 个面积采样中心。** 103 帧使用可见轮廓与 CoTracker 观测拟合；其中 93 帧还通过估计深度门。11 帧只有时序插值初始化并明确标不可靠。本工作未读发布逐帧 R/T、拟合网格、保留相机或传感深度，尚无独立位姿精度结论。

## 输入与信息边界

本轮附件明确允许已知物体扫描形状。唯一新登记资产是 `objects/boxsmall/boxsmall_f1000.ply`，官方来源见 [BEHAVE objects.zip](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/objects.zip)。该文件历史上位于 evaluation_only/templates，现仅将它的 517 个 xyz 和 1,000 个三角面登记为新结构化协议输入；忽略其顶点颜色，输出 PLY 不含颜色属性，原始目录未改动。这不向历史 RGB-only D 分支回填额外形状信息。

使用同一 camera 0 的冻结 RGB 身份/时间戳、SAM2 entity 2、CoTracker（查询时属于物体且目标时仍在可见内部）、UniDepth 及已知 K/c2w。预测掩码、轨迹和深度均为估计。代码没有逐帧参考参数或其他相机路径；输入哈希已再次核对。

## 具体流程

1. 将扫描顶点减去其算术中心作为规范空间，不作尺寸缩放。按面面积固定随机种子采样 4,096 个中心及法线；拟合阶段使用其中 900 个表面点。
2. 对 SAM2 物体掩码腐蚀 1 像素。全段深度中位数及 MAD（中位绝对偏差）形成鲁棒深度范围；帧内再剔除离群深度。只有剩余深度至少 100 像素、占内部像素至少一半且原始帧中位数在全段范围内，才使用该帧深度。深度被拒绝时仍保留有效二维轮廓/轨迹。
3. 首次可见或重新显露时构造 PCA 主轴排列/符号的多姿态候选；相邻已观测帧使用 CoTracker、上一帧预测模板的射线交点、PnP，以及通过深度门的点对刚性配准提供候选。
4. 每帧优化一个旋转和平移。残差包括估计点到模板表面距离、输入物体像素是否落入投影凸包、模板边界落到可见背景的距离、普通二维跟踪和按真实 dt 缩放的弱位姿连续性。人体掩码区域允许遮挡物体，不要求完整物体轮廓等于可见掩码。使用 soft-L1 鲁棒惩罚；具体权重在配置中冻结。
5. 对面积不足 300 像素或物体掩码为空的帧，不假装有充分观测。按两端真实采集时间插值平移，并用 Slerp（旋转球面插值）连接旋转；这些帧不成为新的观测，可靠性为零。属于离线初始化。

这不是另训练物体估姿网络，也不是 S2/S3 新机制；是同输入结构化基线的普通配准前端。纹理未用于识别规范面，因此箱体近似几何对称和遮挡期间的姿态都存在歧义，全部帧保留 `rotation_ambiguity=True`，不能声称唯一真实姿态。

## 实际诊断

| 项目 | run03 结果 |
| --- | --- |
| 主拟合、导出与数值诊断耗时 | {r['wall_seconds']:.2f} 秒，CPU；不含随后图片绘制/人工阅读 |
| 直接二维观测拟合 | 103 / 114 帧 |
| 估计深度通过帧级门 | 93 / 114 帧 |
| 保留二维、拒绝深度的拟合帧 | {q['estimated_2d_only_frames']} |
| 仅不确定插值帧 | {q['interpolated_unreliable_frames']} |
| SAM2 物体像素为零帧 | {q['SAM2_empty_object_frames']} |
| 拟合帧可见轮廓 IoU 最低 / 中位 / 最高 | {q['input_mask_IoU_on_103_fitted_frames_min_median_max'][0]:.3f} / {q['input_mask_IoU_on_103_fitted_frames_min_median_max'][1]:.3f} / {q['input_mask_IoU_on_103_fitted_frames_min_median_max'][2]:.3f} |
| 保存 float32 后 50 个探针的最大成对距离变化 | {q['serialized_float32_max_pair_distance_error_m']:.3e} m |
| 世界到相机回算最大差 | {q['world_to_camera_max_point_difference_m']:.3e} m |
| 中心速度中位 / 最大 | {np.median(d['translation_speed_m_s']):.3f} / {np.max(d['translation_speed_m_s']):.3f} m/s |
| 相邻帧旋转最大变化 | {np.rad2deg(np.max(d['rotation_increment_rad'])):.2f} 度 |

轮廓 IoU 是优化所使用掩码上的内部拟合诊断，**不是独立重建准确率**。空掩码帧的脚本占位 IoU=0 不具备几何评价意义；统计仅列 103 个直接拟合帧。速度/角度只用于发现明显跳变，不对应动作真值，不能据其更平滑断言更正确。

保留三个版本：run01 将深度不足帧整体插值；run02 分离二维与深度，但帧 13 的少量剩余边缘深度仍可误导配准；run03 加入帧级深度支持比例与中位数检查。原图可见提箱阶段深度中位值跳到 3.63–5.40 m，这与箱体轮廓变化不成比例，故依据允许输入纠正深度门。三个版本主拟合累计约 {sum(json.loads((O/x/'run.json').read_text())['wall_seconds'] for x in ['run01','run02','run03']):.2f} 秒；没有以参考误差选择版本或扫参数网格。

实际查看 run03 三张图的 12 帧：可见部分轮廓整体对齐；隐藏期间的完整模板轮廓投影落在人体区域，只表示允许被人遮挡，不等于已确定箱体真实深度。帧 50 局部可见形状仍有偏差；提箱及转身阶段存在深度/朝向歧义，后续应允许图像与普通轨迹继续更新位姿，并用冻结独立参考评价。

## 对接 API

采用 [object_init.npz]({P}/object_init.npz)，SHA256 `{q['NPZ_sha256']}`；顶层 `object_init/object_init.npz` 仅为该文件的相对符号链接。禁止加载 run01/run02 作为部分帧补丁。

- `canonical_vertices_m[517,3]`、`faces[1000,3]`：居中扫描网格。
- `centres_m[4096,3]`、`canonical_normals[4096,3]`：规范空间中心/法线；`sample_face_ids` 与 `sample_barycentric_uv` 保留来源。
- `R_world[114,3,3]`、`t_world_m[114,3]`：按 `X_world = X_canonical @ R_world[t].T + t_world_m[t]` 变换；行向量约定。
- `R_camera`、`t_camera_m`：已知 camera 0 坐标，仅用于核对。世界参数已应用 c2w，不能再乘一次。
- `timestamp_seconds`、`dt_seconds`：实际采集时间；首帧 dt 为 NaN，因为不存在前一帧。
- `observed_mask`：通过最小二维观测门；`depth_observed_mask`：另一个独立的估计深度门。
- `reliability`：基于可见面积与输入轮廓贴合的启发式二维初始化权重，**不等于三维准确概率**；插值帧为零。
- `rotation_ambiguity`：全 True；原点平移记录在 `original_scan_center_m`。

接入 D 的归一化尺度 s 时，将规范中心与世界平移都乘 s，旋转不变；如果先生成世界中心，则只将该世界中心乘一次 s。高斯方向/协方差也需随 R 旋转。不要将规范模板重心又减一次，或把模板形状当作新恢复的未知几何。

[脚本]({O.parent}/code/object_init.py) · [冻结执行脚本]({P}/executed_object_init.py) · [配置]({P}/config.json) · [输入/资源清单]({P}/run.json) · [逐帧诊断]({P}/frame_diagnostics.json) · [保存精度验收]({P}/verification.json)

![输入深度和初始化运动诊断，不是参考误差曲线]({P}/input_depth_motion_diagnostic.png)

'''
for i in range(1,4):md+=f'![物体原图轮廓检查 {i}：绿为可见输入掩码，品红为完整模板投影]({P}/input_overlay_{i}.png)\n\n'
(O/'REPORT.md').write_text(md)
print(json.dumps(q,indent=2))
