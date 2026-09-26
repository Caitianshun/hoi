# 人体—物体—场景动态建模

当前目标：从真实视频在统一世界坐标系中共同建模人体、交互物体和周围三维场景，重点评价几何、接触、相对位姿与运动。主要采用三维高斯，NeRF/神经表面/跟踪方法作为技术与实验参照。超分不属于当前默认任务。

GitHub 仓库仅同步项目代码、源配置及必要的上游修改补丁；数据、权重、实验输出和文档成品仅保存在本地。下方指向本机绝对路径的研究记录链接不随代码镜像上传。

- [2026-09-26 V2 最新验证总结（DOCX）](/home/cai_tianshun/Project/HOI/experiments/surface_observation_fusion_20260926/run01/output/V2_supervision_surface_fusion.docx)：两次B1与四次F优化已完成；B1未过统一监督门槛，F2未建立优于同信息MLP的增量，本轮收口。
- [V2 阶段决定](/home/cai_tianshun/Project/HOI/experiments/surface_observation_fusion_20260926/run01/NEXT_DECISION.md) 与 [复算入口](/home/cai_tianshun/Project/HOI/experiments/surface_observation_fusion_20260926/run01/REPRODUCE.md)；源码位于两新实验目录的code，源配置为 `surface_observation_fusion_20260926/run01/code/experiment_config.json`。
- [上一轮 AUX_REF_OBJECT 正式总结（DOCX）](/home/cai_tianshun/Project/HOI/experiments/aux_ref_object_reconstruction_20260924/run01/output/AUX_REF_OBJECT_verification.docx)；旧停止结论保持。
- [2026-09-24 历史表面点与位姿2×2验证（DOCX）：H1有有限运动收益，八臂均未达完整晋级门槛](/home/cai_tianshun/Project/HOI/experiments/surface_pose_joint_20260924/run01/output/Surface_pose_joint_verification.docx)
- [历史方法决定](/home/cai_tianshun/Project/HOI/experiments/surface_pose_joint_20260924/run01/METHOD_DECISION.md) 与 [复算入口](/home/cai_tianshun/Project/HOI/experiments/surface_pose_joint_20260924/run01/REPRODUCE.md)
- [2026-09-24 前轮目标/评分诊断（便携 PDF）：关闭轨迹未纠偏，官方同池排序未通过，不追加训练](/home/cai_tianshun/Project/HOI/experiments/pose_objective_diagnosis_20260924/run01/output/pdf/Pose_objective_diagnosis.pdf)
- [前轮阶段去留决定](/home/cai_tianshun/Project/HOI/experiments/pose_objective_diagnosis_20260924/run01/NEXT_DECISION.md) 与 [复算入口](/home/cai_tianshun/Project/HOI/experiments/pose_objective_diagnosis_20260924/run01/REPRODUCE.md)
- [前轮 P1→S1* 完整配对验证（便携 PDF）：木椅绝对位置改善、箱体退化，暂不统一升级](/home/cai_tianshun/Project/HOI/experiments/object_pose_refinement_20260924/run01/output/pdf/P1_S1star_paired_verification.pdf)
- [项目全局准则及用户后续范围确认](/home/cai_tianshun/Project/HOI/AGENTS.md)
- [2026-09-23 新执行指导落地：两条结构化 S0/S1 完整训练、独立评价与下一关判断](/home/cai_tianshun/Project/HOI/experiments/structured_hoi_20260923/REPORT.md)
- [2026-09-23 GVHMR 实测：三权重齐备、114 帧人体先验与独立质量检查](/home/cai_tianshun/Project/HOI/experiments/gvhmr_validation_20260923/REPORT.md)
- [2026-09-23 D 完整验证：几何接口修复、训练身份链与 SMPL-X 先验入口](/home/cai_tianshun/Project/HOI/experiments/mosca_interface_validation_20260923/REPORT.md)
- [2026-09-23 手部链验证完成：真实种子、节点绑定、A/C对照与模板决策](/home/cai_tianshun/Project/HOI/experiments/mosca_hand_trace_20260923/REPORT.md)
- [2026-09-23 自主验证实测：输入/运动诊断、24次深度推理及A/B重建完成](/home/cai_tianshun/Project/HOI/experiments/mosca_validation_20260923/REPORT.md)
- [2026-09-23 当前判断：时间、分辨率、刚体/非刚体与接触、深度、骨干和最小验证](/home/cai_tianshun/Project/HOI/research/2026-09-23/REPORT.md)
- [14时刻人体/物体评价点云：固定顶点ID、世界坐标及来源边界](/home/cai_tianshun/Project/HOI/research/2026-09-23/evaluation_pointcloud_reference/README.md)
- [2026-09-22 MoSca 实测诊断：训练完成，动态重建质量未通过](/home/cai_tianshun/Project/HOI/experiments/mosca_baseline_20260922/REPORT.md)
- [2026-09-22 调研：遮挡、Point4D/D4RT、注意力、单目协议与代码入口](/home/cai_tianshun/Project/HOI/research/2026-09-22/REPORT.md)
- [2026-09-18 调研主报告：路线、问题、候选与投稿计划](/home/cai_tianshun/Project/HOI/research/2026-09-18/REPORT.md)
- [动态 NeRF / 高斯技术证据](/home/cai_tianshun/Project/HOI/research/2026-09-18/dynamic_routes.md)
- [HOI 与完整场景前沿、源码审计](/home/cai_tianshun/Project/HOI/research/2026-09-18/hoi_frontier.md)
- [接触、位姿近邻与评价审查](/home/cai_tianshun/Project/HOI/research/2026-09-18/contact_prior_audit.md)
- [数据与工程可行性](/home/cai_tianshun/Project/HOI/research/2026-09-18/data_feasibility.md)
- [研究日志](/home/cai_tianshun/Project/HOI/RESEARCH_LOG.md)

当前已依据用户最新附件构建结构化组合基线：人体 SMPL-X 骨骼及受限残差、独立刚体物体、静态背景，共用 MoSca D 准确几何接口与一次遮挡合成。任务输入为标定单路 RGB、通用人体模板、RGB 预测姿态和已知无纹理物体形状；旧 D 的更弱输入分支保留为历史对照，不把新增先验产生的收益算作算法创新。

两条 BEHAVE 开发序列（箱体114帧、木椅98帧）的 S0/S1 四个8000步结果及完整评价均已完成。S0在6000步关闭普通跟踪，S1保留至结束；每一对共享完整6000步检查点、数据、帧安排和优化器。箱体手套/箱体/手物相对查询 S0为7.54/17.20/18.58cm，S1为7.37/17.05/18.42cm；人体全表面代理和保留视角两事件均未呈现S1平均收益。实际高斯保持正确实例隔离与物体刚性，但这些工程性质不能证明位姿或接触准确。

P1→S1*配对已完成：多源短轨迹、官方MegaPose RGB候选与固定形状全段位姿优化使两事件轮廓改善，但箱体最终中心/物体表面误差14.33/6.75→21.60/11.79cm，木椅20.56/10.49→10.65/6.49cm；木椅相邻参考位移误差12.32→13.95cm，不能称为整体运动改善。P1尚不能统一替换旧基线。

按随后修订附件完成固定箱体短轨迹开/关与两事件目标/评分诊断：原开启精确复现，关闭使初始化中心21.680→22.178cm、绝对深度18.623→19.120cm、位移18.179→18.739cm；不支持去项纠偏。同池原评分与官方5帧logit规则重新比较，箱体选路不变，木椅官方选路使中心11.262→13.706cm、位移14.446→15.802cm，B未通过。阶段决定为保留两事件旧S1，拒绝统一关闭和官方排序；无先验回拉证据，不启用C/S2/注意力/接触，不追加高斯训练。9月24日已完成阶段去留，不无限调参。8条BEHAVE正式候选未被本轮消耗，HODome仍为库存审计，未冒称正式基准完成。

内部里程碑：11 月 4 日冻结核心实验；11 月 5—15 日连续 11 天写作；11 月 15 日内部定稿。研究范围扩张不得侵占写作窗口。

最新表面点与位姿原型八臂各300步及冻结后评价已完成：F01相对F00位移改善0.973/1.118cm，但仍复现箱体深度漂移与木椅相邻运动退化，未满足旧初始化护栏；F10/F11也未统一晋级。H1仅有有限相对证据，H2/联合机制尚无统一支持。本轮主训练0次、独立事件0条，保留旧S1，不以二维残差下降或q移动证明三维材料身份恢复。
