"""Read-only evaluation, run only after fixed_track_pose_run01 predictions freeze."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from post_train_diagnose import angular,plain

ROOT=Path('/home/cai_tianshun/Project/HOI');E=ROOT/'experiments/structured_hoi_20260923';HERE=E/'object_init';O=HERE/'fixed_track_pose_run01';BASE=ROOT/'experiments/mosca_baseline_20260922'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 run=json.loads((O/'run.json').read_text());assert run['status']=='completed'
 predpath=O/'object_init.npz';assert sha(predpath)==run['prediction_sha256']
 # First access to independent references occurs after the completed/hash assertions.
 refpath=ROOT/'research/2026-09-23/evaluation_pointcloud_reference/object_fitted_vertices_world.npz'
 qpath=E/'dev1_S1_v1/evaluation_v2/reference_evaluation/evaluation_bundle.npz'
 ref=np.load(refpath);qr=np.load(qpath);diag=np.load(HERE/'post_train_diagnosis_arrays.npz');ref_R=diag['reference_R_world_analytic_no_fit'];idx=ref['input_frame_indices'];Y=ref['xyz_world'].astype(float);cy=Y.mean(1)
 init=np.load(HERE/'run03/object_init.npz');fit=np.load(predpath);anc=np.load(O/'fixed_anchors.npz');ids=anc['track_ids'];A=anc['canonical_points_m'];sel=np.array([np.where(ids==i)[0][0] for i in range(4)]);Aq=A[sel];assert np.array_equal(qr['query_id'][:4],['object_front_red','object_front_brown','object_left_face','object_top'])
 rows=[];summaries={};arrays={};C=init['c2w'];K=init['K'];V=init['canonical_vertices_m'].astype(float)
 for name,p in [('run03',init),('fixed_track',fit)]:
  R=p['R_world'][idx].astype(float);t=p['t_world_m'][idx].astype(float);X=np.einsum('tij,nj->tni',R,V)+t[:,None];cx=X.mean(1);de=cx-cy;rotation=angular(R,ref_R);predq=np.einsum('tij,nj->tni',R,Aq)+t[:,None];qe=predq-qr['reference'][:,:4]
  verr=X-Y;summaries[name]=dict(centroid_mean_cm=np.linalg.norm(de,axis=-1).mean()*100,centroid_first_cm=np.linalg.norm(de[0])*100,rotation_mean_deg=rotation.mean(),rotation_first_deg=rotation[0],corresponding_vertex_rmse_cm=np.sqrt(np.mean(np.sum(verr**2,-1)))*100,source_ray_query_proxy_mean_cm=np.linalg.norm(qe,axis=-1).mean()*100,source_ray_query_proxy_first_each_cm=np.linalg.norm(qe[0],axis=-1)*100,source_relative_centroid_displacement_mean_excluding_source_cm=np.linalg.norm(de[1:]-de[0],axis=-1).mean()*100)
  for j,f in enumerate(idx):
   rows.append(dict(method=name,frame=int(f),actual_time=float(ref['frame_times'][j]),direct_fixed_track_observation=bool(fit['observed_mask'][f]),centroid_error_cm=float(np.linalg.norm(de[j])*100),rotation_error_deg=float(rotation[j]),source_ray_query_proxy_mean_cm=float(np.linalg.norm(qe[j],axis=-1).mean()*100),camera_delta_cm=(de[j]@C[:3,:3]*100).tolist()))
  arrays[name+'_centroid_delta_world_m']=de;arrays[name+'_query_proxy_delta_world_m']=qe;arrays[name+'_rotation_error_deg']=rotation
 gates=json.loads((O/'frame_gates.json').read_text());reproj=json.loads((O/'reprojection.json').read_text());source=json.loads((O/'source_gate.json').read_text());counts=dict(insufficient_candidates=sum(g['candidate_count']<6 for g in gates),canonical_planar_or_image_degenerate=sum(g['candidate_count']>=6 and not g['nondegenerate'] for g in gates),pnp_or_inlier_gate_failed=sum(g['nondegenerate'] and not g['pnp_success'] for g in gates),passed=sum(g['pnp_success'] for g in gates))
 assert sum(counts.values())==114
 report=dict(status='completed',prediction_sha256=sha(predpath),reference_hashes={str(refpath):sha(refpath),str(qpath):sha(qpath),str(HERE/'post_train_diagnosis_arrays.npz'):sha(HERE/'post_train_diagnosis_arrays.npz')},fit_reference_read=False,evaluation_read_after_prediction_frozen=True,no_alignment=True,gate_counts=counts,summaries=summaries,rows=rows,proxy_note='Same 4 manual source pixels ray-intersect run03 template once. This is a rigid surface-point proxy, not actual trained Gaussian queries. Initialization bias is inherited. No replacement/matching using reference.',time_caveat='same14 fitted mesh references, maximum nominal RGB offset 0.103745s; not exact material ground truth')
 (O/'evaluation.json').write_text(json.dumps(plain(report),indent=2)+'\n');np.savez_compressed(O/'evaluation_arrays.npz',**arrays)
 m=json.loads((BASE/'common_input/input_manifest.json').read_text());tr=np.load(BASE/'common_input/uniform_cotracker_tap.npz');tm=tr['tracks'][:,ids];visible=np.load(O/'fixed_anchor_trajectories.npz')['observed_mask']
 # Same full-resolution RGB/crop for both methods; shape and fixed correspondence only.
 frames=[0,16,31,55,65,113]
 for si in range(2):
  sheets=[]
  for f in frames[si*3:si*3+3]:
   panels=[]
   for name,p in [('run03',init),('fixed track',fit)]:
    im=cv2.imread(m['frame_paths'][f]);R=p['R_camera'][f];t=p['t_camera_m'][f];h=(V@R.T+t)@K.T;uv=h[:,:2]/h[:,2:];hull=cv2.convexHull(np.rint(uv).astype(np.int32));cv2.polylines(im,[hull],True,(255,255,0),1)
    xy=(A@R.T+t)@K.T;xy=xy[:,:2]/xy[:,2:]
    for n,(a,b) in enumerate(zip(xy,tm[f])):
     if not np.isfinite(a).all() or not np.isfinite(b).all():continue
     aa=tuple(np.rint(a).astype(int));bb=tuple(np.rint(b).astype(int));cv2.circle(im,aa,2,(0,100,255),-1)
     if visible[f,n]:cv2.circle(im,bb,3,(0,255,0),1);cv2.line(im,aa,bb,(0,180,0),1)
     if ids[n]<4:cv2.putText(im,str(ids[n]),aa,cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
    cv2.rectangle(im,(0,0),(640,50),(255,255,255),-1);cv2.putText(im,f'{name} f={f} fixed observations={int(visible[f].sum())}',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);cv2.putText(im,'cyan=shape orange=anchors green=eligible tracks',(10,40),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1);panels.append(im)
   sheets.append(np.concatenate(panels,axis=1))
  cv2.imwrite(str(O/f'input_overlay_{si+1}.png'),np.concatenate(sheets,axis=0))
 notes=['# 一次固定规范点的普通刚体对照','', '这是对已知形状初始化的简单检验，不是 S2，不是新的高斯训练。协议在拟合前冻结；拟合仅读取原 run03、camera0 K/时间、SAM2 和原 CoTracker。参考首次读取发生在预测 NPZ 哈希冻结以后。','',f'首帧获得 {source["hit_count"]} 个固定规范点，规范点奇异值比 {source["canonical_rank_ratio"]:.3f}，并非共面退化。每帧要求至少 6 个可见、同实体且非退化点，RANSAC 后仍须至少 6 点；共面或不足时不强行 PnP。114 帧中 {counts["passed"]} 帧通过；不足点数 {counts["insufficient_candidates"]} 帧、共面/图像退化 {counts["canonical_planar_or_image_degenerate"]} 帧、PnP/内点门限失败 {counts["pnp_or_inlier_gate_failed"]} 帧。','',f'20–72 帧没有通过固定点直接观测门限，只有弱原姿态和实际时间加速度先验，可靠性设为零。这段缺测不能解释为跟踪成功。单次拟合耗时 {run["wall_seconds"]:.3f} s，函数评价 {run["nfev"]} 次，求解状态：{run["optimizer_message"]}。','', '| 相同评价 | run03 | 固定对应普通对照 |','| --- | ---: | ---: |']
 for title,key in [('质心平均误差 cm','centroid_mean_cm'),('首帧质心误差 cm','centroid_first_cm'),('模板朝向差均值°','rotation_mean_deg'),('对应模板顶点 RMSE cm','corresponding_vertex_rmse_cm'),('4 源射线查询代理误差均值 cm','source_ray_query_proxy_mean_cm'),('相对首帧位移误差均值 cm','source_relative_centroid_displacement_mean_excluding_source_cm')]:notes.append('| '+title+' | '+' | '.join(f'{summaries[k][key]:.3f}' for k in ['run03','fixed_track'])+' |')
 notes+=['', '源射线代理与冻结训练高斯查询机制不同，不能将其数值直接与 S0/S1 当成模型排名。这里比较的是同一组从 run03 首帧建立的规范表面点，在两套刚体姿态下的运动；两者均继承首帧绝对深度/表面选点偏差。','', '| 输入帧 | 直接固定点观测 | run03 中心 cm | 对照中心 cm | run03 角差° | 对照角差° | run03 查询代理 cm | 对照查询代理 cm |','| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for j,f in enumerate(idx):
  x=rows[j];y=rows[j+14];notes.append(f'| {f} | {y["direct_fixed_track_observation"]} | {x["centroid_error_cm"]:.2f} | {y["centroid_error_cm"]:.2f} | {x["rotation_error_deg"]:.2f} | {y["rotation_error_deg"]:.2f} | {x["source_ray_query_proxy_mean_cm"]:.2f} | {y["source_ray_query_proxy_mean_cm"]:.2f} |')
 notes+=['', '本实验保留全部门限失败和缺测结果，没有参数扫描或参考导向重跑。是否进入下一轮训练须同时考虑未观测段、源深度偏差与输入轮廓；优化器成功退出只表示数值求解结束，不保证三维正确。','',f'![相同原图的 run03 与固定对应对照，帧0/16/31]({O}/input_overlay_1.png)',f'![相同原图的 run03 与固定对应对照，帧55/65/113]({O}/input_overlay_2.png)','',f'[冻结协议]({HERE}/fixed_track_pose_protocol.json)；[输入/运行哈希]({O}/run.json)；[逐帧门限]({O}/frame_gates.json)；[逐帧重投影]({O}/reprojection.json)；[评价全部数值]({O}/evaluation.json)。']
 (O/'REPORT.md').write_text('\n'.join(notes)+'\n');assert sha(predpath)==run['prediction_sha256'];print(json.dumps(plain(report['summaries']),indent=2));print('gates',counts)
if __name__=='__main__':main()
