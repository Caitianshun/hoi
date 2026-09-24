import json,hashlib,shutil
from pathlib import Path
import numpy as np,cv2
b=Path('/home/cai_tianshun/Project/HOI/experiments/structured_hoi_20260923/object_init');o=b/'fixed_track_pose_run01';base=Path('/home/cai_tianshun/Project/HOI/experiments/mosca_baseline_20260922');init=np.load(b/'run03/object_init.npz');fit=np.load(o/'object_init.npz');lab=np.load(base/'segmentation/segmentation.npz')['entity_labels'];K=init['K'];V=init['canonical_vertices_m'];results={}
for tag,p in [('run03',init),('fixed_track',fit)]:
 rows=[]
 for i in range(114):
  cam=V@p['R_camera'][i].T+p['t_camera_m'][i];uv=cam@K.T;uv=uv[:,:2]/uv[:,2:];poly=cv2.convexHull(np.rint(uv).astype(np.int32));sil=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(sil,poly,1);pred=(sil>0)&(lab[i]!=1);gt=lab[i]==2;union=np.count_nonzero(pred|gt);iou=np.count_nonzero(pred&gt)/max(1,union);rows.append(dict(frame=i,object_pixels=int(gt.sum()),input_visible_iou=float(iou)))
 vals=np.array([r['input_visible_iou'] for r in rows]);mask=init['observed_mask'];direct=fit['observed_mask'];results[tag]=dict(mean_103_original_observed=float(vals[mask].mean()),median_103_original_observed=float(np.median(vals[mask])),mean_61_fixed_track_observed=float(vals[direct].mean()),median_61_fixed_track_observed=float(np.median(vals[direct])),rows=rows)
(o/'input_silhouette_check.json').write_text(json.dumps(results,indent=2)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['fixed_track_pose_fit.py','fixed_track_pose_protocol.json','evaluate_fixed_track_pose.py']:
 target=o/('executed_'+name);shutil.copyfile(b/name,target)
run=json.loads((o/'run.json').read_text());assert sha(o/'object_init.npz')==run['prediction_sha256'];assert sha(o/'executed_fixed_track_pose_fit.py')==run['code_sha256'];assert sha(o/'executed_fixed_track_pose_protocol.json')==run['protocol_sha256']
p=o/'REPORT.md';txt=p.read_text();intro='结论：这一次固定源表面对照失败，不采用为下一轮训练初始化。质心平均误差从 14.76 增至 50.25 cm，源射线查询代理从 17.85 增至 49.18 cm。模板朝向均值虽下降，不能抵消明显的绝对位置退化。保留这次负面结果，不追加拟合或参数扫描。\n\n'
txt=txt.replace('# 一次固定规范点的普通刚体对照\n\n','# 一次固定规范点的普通刚体对照\n\n'+intro)
txt=txt.replace('20–72 帧没有通过固定点直接观测门限，','20–72 帧没有通过这组首帧表面点的直接观测门限，')
txt+='\n补充解释：上述 53 帧是首帧选定表面点不可用，不是“整个物体完全不可见 53 帧”。其间箱子可以露出别的箱面，但这个最简单版本没有把后出现表面加入新锚点，也没有可靠的断裂关联。\n\n'
txt+=f'输入轮廓检查也支持拒绝，不依赖拟合参考：在原 run03 的 103 个直接观测帧，均值可见 IoU 从 {results["run03"]["mean_103_original_observed"]:.3f} 降至 {results["fixed_track"]["mean_103_original_observed"]:.3f}；在本对照 61 个固定点有效帧从 {results["run03"]["mean_61_fixed_track_observed"]:.3f} 降至 {results["fixed_track"]["mean_61_fixed_track_observed"]:.3f}。使用同一 box 凸投影轮廓并排除 SAM2 人体遮挡区域，仅为输入拟合诊断。\n\n'
txt+='为何小重投影仍会失败：首帧 45 个锚点由有偏初始化的射线交点构建，真实表面对应不保证准确；后续 tracker 的 visible 与同实体 mask 只能证明点落在箱子区域，不能证明其仍是同一箱面位置。RANSAC 接受一个能解释局部点集的姿态，也未要求完整已知物体轮廓被解释。这次协议只使用固定轨迹、弱原姿态与时间正则，没有继续使用原深度/完整轮廓约束。因此深度可向远处移动来解释缩小的点集，帧16/113即使重投影中位数约1.91/1.88 px，中心误差仍为78.62/106.28 cm。这是失败机制的合理解释，尚未以消融分别证明各项因果。\n\n'
txt+='本次结果既不证明 S2 有效，也不排除带已知形状轮廓约束、可靠深度或更强对应拒绝的普通位姿优化。后续如再做简单基线，必须在新的输入-only 协议下比较；本轮没有继续拟合。\n\n'
txt+=f'工程交付：规范模板517顶点/1000面、4096原始表面采样点均保持；45个固定轨迹锚点、4个手选源射线代理，114个刚体姿态，61帧有直接固定点约束。与主评价“6个实际高斯查询（4物体+2手）”严格区分：这里没有人体、没有高斯渲染权重，17.85→49.18 cm仅可在这一射线代理内比较。\n\n预测身份：`{run["prediction_sha256"]}`。拟合 {run["wall_seconds"]:.3f} s 为 CPU 计算，不含后续评价与出图。两张叠图已逐张实际目检。\n\n'
txt+=f'[执行拟合脚本快照]({o}/executed_fixed_track_pose_fit.py)、[执行协议快照]({o}/executed_fixed_track_pose_protocol.json)、[评价脚本快照]({o}/executed_evaluate_fixed_track_pose.py)、[输入轮廓检查]({o}/input_silhouette_check.json)。\n'
p.write_text(txt)
verification=dict(status='complete_negative_result_rejected_for_training',prediction_sha256=sha(o/'object_init.npz'),fit_code_sha256=run['code_sha256'],protocol_sha256=run['protocol_sha256'],evaluation_code_sha256=sha(o/'executed_evaluate_fixed_track_pose.py'),report_sha256=sha(p),fixed_anchor_count=45,manual_ray_proxy_count=4,frames=114,direct_observed_frames=61,template_vertices=517,template_faces=1000,canonical_samples=4096,prediction_unchanged=True,visual_inspection={'input_overlay_1.png':'actually viewed','input_overlay_2.png':'actually viewed'},no_additional_fit=True,not_S2=True,no_new_training=True)
(o/'delivery_verification.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in results.items()},indent=2))
