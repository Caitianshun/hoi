"""Portable PDF of the completed bounded objective/score diagnosis."""
from pathlib import Path
import importlib.util,json,hashlib,datetime
import numpy as np
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
E=Path(__file__).resolve().parents[1]
O=E.parents[1]/'object_pose_refinement_20260924/run01'
spec=importlib.util.spec_from_file_location('old_report_helpers',O/'code/build_report.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
p,head,page,table,image=B.p,B.head,B.page,B.table,B.image
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meanmotion(v,key):return np.mean([r[key] for r in v['reference_interval_motion'] if r[key] is not None])
def metrics(v):return [v[k]['mean'] for k in ['centroid_error_cm','camera_depth_absolute_error_cm','raw_rotation_error_deg']]+[meanmotion(v,k) for k in ['centre_displacement_error_cm','relative_rotation_error_deg']]
def fmt(x):return f'{x:.3f}'
F=E/'output/figures'
toggle=load(E/'track_toggle/summary.json');a=load(E/'evaluation/track_toggle_dev1/diagnostic_results.json');oracle={d:load(E/f'evaluation/oracle_{d}/diagnostic_results.json') for d in ['dev1','dev2']}
official=load(E/'score_audit/official/base/scored/full_path_scores.json')
summary=load(E/'score_audit/official/summary.json')
p('动态人—物—场景建模\n位姿目标与候选评分诊断','title')
p('2026-09-24 UTC · 两条 BEHAVE 开发事件 · 修订附件第 7–8、14 节','small')
p('阶段决定：保留旧 S1，拒绝本轮两种简单修正，不追加高斯训练。','sub')
p('本轮要分清：箱体深度漂移主要由错误短轨迹约束推动，还是现有目标与候选评分本身偏好错误解。已完成固定路径重放与去项、有限深度/朝向响应、官方重新评分和完整路径独立诊断；后续只执行了同池官方排序分支 B。')
table([['验证','实测结论','执行决定'],['固定箱体短轨迹开/关','关闭后中心、绝对深度和相邻运动略差，原漂移仍在。','拒绝统一关闭，不扩展 A。'],['同池原评分/官方评分','箱体选路相同；木椅官方选路在中心、深度、位移上更差。','B 已完成，拒绝升级。'],['训练先验是否回拉','尚无正确更新被单项先验抵消的证据；输入目标本身可偏好错误几何。','不启动 C 的两次训练。']],[122,258,131])
p('新增完整训练 0 次；新增网络 0 个；独立事件消耗 0 条。S2、注意力和接触没有启用。原两事件 S1、P1/S1*以及失败结果全部保留，未按事件拼最好结果。')
p('为什么这仍是有效进展：排除了“关掉当前短轨迹就会纠偏”的简单行动依据，也实测拒绝了官方重新排序。继续重复训练相同错误初始化，当前没有明确决策价值。')
p('主要限制：两事件已用于开发，参考是有时间匹配误差的拟合网格；本轮只解释有限路径、固定对应和当前权重下的作用。不能据此宣称材料身份正确、精细接触准确或场景在数学上不可恢复。','small')
p('本轮阶段决定已在9月24日完成，早于9月27日界限；11月4日核心结果冻结、11月5–15日连续11天写作保持不变。','small')

page();head('01  控制变量与优化记录')
p('起点为原 path00 优化前全部114帧 R/t；固定103个可见帧、相机、米制模板、真实时间、26查询/365记录。规范点从已保存缓存恢复，未重新跟踪、射线绑定或改变对应门槛。唯一变化是短轨迹残差乘数从1变0，关闭后仍离线计算原轨迹诊断残差。')
table([['项目','开启重放','关闭短轨迹'],['nfev / 实际接受更新','19 / 15','17 / 15'],['全部残差调用（含数值雅可比）','420','418'],['终止条件','ftol 与 xtol','xtol'],['CPU 位姿求解秒',fmt(toggle['branches']['on35']['seconds']),fmt(toggle['branches']['off35']['seconds'])],['固定可见集 mean IoU','0.921316918','0.921340248'],['35上限是否耗尽','否','否']],[224,143,144])
p('原35是 scipy least_squares 的 max_nfev；数值雅可比引起的调用不计入 nfev。开启支目标和旧记录差为0，float32位姿文件哈希与旧 P1 相同；与内部float64的最大分量差约1.19×10⁻⁷ m。')
p('未满足“预算截断且仍不稳定”的预声明条件，所以不续算140。满足步长/目标容差只表示按原规则停止；梯度最优性非零，不等于全局收敛。续算若触发也将重置SciPy信赖域状态，此限制已预先记录。')
image(F/'toggle_optimization.png',maxh=170)
p('图中开/关的活动目标不同，不能把关闭支总损失更低解释为质量更好；质量由冻结后的独立三维评价判断。','small')

page();head('02  箱体：全14参考槽结果')
rows=[['完整路径','中心 cm','绝对深度 cm','旋转 °','位移 cm','相对旋转 °']]
for lab,title in [('old_initialization','旧初始化'),('on35','开启 / P1'),('off35','关闭')]:
 v=a['summaries'][lab];rows.append([title,*map(fmt,metrics(v))])
table(rows,[92,75,86,76,87,95])
p('中心与运动分开：位移误差是相邻有效参考时刻的中心位移向量误差，按真实时间间隔另列速度；不是简单压低模型运动。14/14参考槽、13/13相邻区间均保留，没有剔除困难帧。')
image(E/'evaluation/track_toggle_dev1/paired_pose_curves.png',maxh=335)
p('关闭后中心+0.498 cm、绝对深度+0.497 cm、位移+0.560 cm。带符号深度均值略小但绝对误差增大，不能据带符号抵消宣称改善。单次差异不足以证明轨迹总是有益；它说明当前短轨迹不是该漂移的必要条件。')
p('对应门控原本在求解前固定，因此无需再构造一组“冻结门控”实验。官方/原评分均未读取这些参考槽来调参数。','small')

page();head('03  箱体：什么目标在偏好 P1')
q=load(E/'score_audit/objective/dev1/with_track_off/responses.json')
rows=[['分项（按114帧归一）','旧初始化','P1','关闭']]
for key,name in [('silhouette_forward','轮廓正向'),('silhouette_background','轮廓背景侧'),('estimated_depth_weak','弱估计深度'),('confirmed_tracks','条件规范轨迹'),('actual_time_acceleration','真实时间加速度')]:rows.append([name,*[f"{q['cases'][c]['components'][key]:.5f}" for c in ['old','P1','track_off']]])
table(rows,[202,103,103,103])
p('旧→P1大幅降低轮廓残差，却增加弱深度与轨迹残差；按既定权重合计仍更低。这支持“现有输入目标可接受错误三维解”，不能把IoU改善当成深度纠正。RGB/NCC与覆盖惩罚为路径常数，不提供待优化R/t的梯度；没有绝对位姿先验进入这个CPU求解器。')
image(F/'dev1_depth_reading.png',maxh=395)
p('横轴是模板中心沿相机射线的预设比例平移；旋转、尺度与中心投影固定。图中只含当前帧观测项，邻帧冻结的时间代价另存。P1的5个关键帧在7个采样深度中均以0%观测残差最小，不意味着已知真实深度，也不是全局后验。','small')

page();head('04  木椅：局部响应与低观测区间')
image(F/'dev2_depth_reading.png',maxh=405)
p('木椅旧初始化在多个关键帧偏好更近的有限候选，P1在各关键帧的预设切片均以原位置观测项最小。三个完整路径的条件规范记录均为0，已逐条核查；因此未重复等价的木椅 track-off。')
image(F/'dev2_coverage.png',maxh=119)
p('二维可靠记录数不是正确材料对应数。低于6条的帧和固定低可见区间全部保存。帧64→74是上一轮已暴露的事后失败案例，另做目标切片并保留实际时间；不能称独立验证。')
p('角度响应对模板三轴各±15°，绕同一模板中心旋转并补偿平移；所有样本计数保持。详细分项、角度热图及“观测/时序”分离曲线随实验目录保存。本切片未运行重新优化。','small')

page();head('05  官方评分：接口有效，排序仍会失效')
p('已部署的MegaPose官方粗评分网络对给定位姿渲染并评分。原50个候选已保存分数，生成时据此取每关键帧前5个；P1局部优化、beam路径与最终选择只读取R/t而不使用这些分数。准确结论是“分数未进入后续路径排序”，并非全流程丢弃。')
p('本轮复用原权重、无纹理模板、camera0与EGL。共435次姿态评分：基础370项加箱体关闭支65项；全部有限、全部保存官方crop/render和裁切坐标。50个原候选pose_score逐值重放差0，说明评分入口复现成功。')
rows=[['候选（完整路径）','箱体原分数 ↓','箱体logit ↑','木椅原分数 ↓','木椅logit ↑']]
for name in ['old_initialization','path00_old_start','path01_beam','path02_beam']:
 row=[name.replace('initialization','init').replace('_old_start','').replace('_beam','')]
 for d in ['dev1','dev2']:
  v=next(x for x in load(E/f'score_audit/same_pool_original/{d}_scores.json')['rows'] if x['name']==name)
  s=next(x for x in official if x['dev']==d and x['path_id']==name and x['stage']==('frozen_pose' if name=='old_initialization' else 'after'))
  row += [fmt(v['score']),fmt(s['mean_pose_logit'])]
 rows.append(row)
table(rows,[135,94,94,94,94])
p('两种规则在完全相同4条路径池上重算：旧初始化使用由该原起点产生的path00缓存，其余路径使用各自原缓存；缺项没有默填零。使用原合法性/可见IoU门槛，官方分数在相同5帧等权平均，所有帧都必须有效。')
p('官方分数随有限深度/朝向改变，并非平坦；但有响应不等于几何排序可靠。它保留官方随假设变化的裁切方式，并非校准后的正确概率。50个单帧候选仅箱体frame83的5个有同身份有效参考，其余45个记N/A，没有插值参考造标签。')
p('GPU：物理GPU1 RTX3090，CUDA与实际EGL渲染设备均核验；两次评分总墙钟约22.01秒，含加载与调试图片保存。首次网络部署/权重下载成本沿用旧记录，不重复计算成此次推理。峰值allocated约0.853 GiB。','small')

page();head('06  官方分数的有限深度响应')
image(F/'dev1_official_response.png',maxh=472)
p('箱体三组均有显著响应；分数更高仍可能对应独立三维误差更差的路径。例如关闭支原姿态的5帧均值logit约1.254，高于P1的1.154，但关闭支中心、绝对深度和运动均略差。')
p('这些候选没有据参考进行平移纠正。图中最高点仅是预声明有限切片结果，没有转为新初始化或按帧拼接。木椅对应曲线完整保存在本轮图表资源中；其官方整路径失败直接见下一页。','small')

page();head('07  唯一后续分支 B：同池排序配对')
rows=[['事件 / 选择','中心 cm','深度 cm','旋转 °','位移 cm','相对旋转 °']]
for d,lab,title in [('dev1','original_initialization','箱体旧初值'),('dev1','path00_old_start_after','箱体两种规则'),('dev2','original_initialization','木椅旧初值'),('dev2','path02_beam_after','木椅原规则'),('dev2','path00_old_start_after','木椅官方规则')]:rows.append([title,*map(fmt,metrics(oracle[d]['summaries'][lab]))])
table(rows,[111,74,74,78,82,92])
p('箱体选路完全相同；木椅官方改选path00，中心+2.444 cm、绝对深度+2.424 cm、位移+1.356 cm，角差也变大。相对旋转略好不足以抵消其余退化，更不能按事件选择另一套规则来拼统一方法。')
p('因此B不具备第8.4要求的统一收益；不追加两个8000步重建，也不消耗首个独立事件。这个结论在初始化层面成立，未运行的高斯表面/留出图像指标保持“未运行”，不能声称它们实测退化。')
p('有限完整路径最佳可达诊断（oracle，仅评价）','sub')
table([['事件','可选余地','局限'],['箱体','优化后三条中path00中心/深度最佳；path01角差80.64°较小。','path01中心23.29 cm更差；加入旧初值后其中心14.76 cm更好，但两评分均未选。'],['木椅','path01中心11.00、深度9.06、位移12.88 cm，优于原选path02对应项。','path01角差26.49°略差，官方却选path00。没有一条已证明全面合格的轨迹。']],[53,219,239])
p('oracle只比较每个指标的一条完整轨迹；没有逐帧选最好候选，没有导出参考选择轨迹。优化前/后与旧初始化的重复身份在账本中保留，不当成独立随机实验。不能由当前有限池失败推导数学不可恢复。','small')

page();head('08  去留边界、完整性与复算入口')
p('停止本轮初始化/评分工程扩张。A没有有害短轨迹的纠偏证据；B已实测拒绝；C需要分项梯度和实际更新共同证明单项先验抵消正确修正，当前未满足。训练保留初始化差异不能独自证明先验是唯一原因。')
p('可保留的研究问题是源位姿与规范附着的相互依赖，但“保存多个附着”“更多关键帧”或“加入注意力”尚不是已验证方案。本轮没有证明固定绑定是主因；不在同一轮继续搜索新网络、融合权重、阈值或接触模块。')
table([['产物','本实验根目录下的入口'],['预声明及源附件','protocol/frozen_diagnostic.json；track_toggle_frozen.json；official_*_frozen.json；user_guidance_source.md'],['完整开/关位姿与过程','track_toggle/on35、off35：object_init.npz、pose_float64.npz、全部残差调用/实际迭代及分项'],['有限响应与候选账本','score_audit/objective/；same_pool_original/；official/base/candidate_ledger.json；official/summary.json'],['冻结后独立评价','evaluation/track_toggle_dev1/；oracle_dev1/、oracle_dev2/；same_pool_ranking_comparison.json'],['复现与去留','REPRODUCE.md；NEXT_DECISION.md；protocol/completion_integrity.json；final_artifact_manifest.json']],[130,381])
p('评价口径不变：模板全部顶点均值中心（含历史孤立点）、原始旋转/恒等对称集合、相机带符号和绝对深度、真实dt；无ICP、尺度拟合或逐帧最佳对齐。箱体14/14槽及13/13相邻区间；木椅9/10槽及8/9相邻区间，t=2秒缺失保持。')
p('时间边界：箱体名义参考与输入最大约104 ms；木椅名义偏差约37.10 ms，实际camera1配对约33.489 ms，两者不同。这些不是同步精度保证，不能把相邻运动误差称精细接触误差。')
p('没有新高斯模型，因此实际高斯固定查询、表面代理和留出图像继续引用旧已冻结结果，不复造新表。保留2手套/4物体固定点、14时刻、8相对向量以及旧alpha/稳定ID等评价定义。LPIPS仍N/A。','small')
p('本轮代码和原资产哈希、读取边界、数值增量复算已核查。响应中的track_unweighted_robust_sum字段实际含原始归一化权重，仅未乘开关系数；元数据澄清见独立审查，不解释为原始像素误差。','small')
p('文档内嵌图像与中文/西文字体；交付前逐页渲染与独立复制渲染检查。只声明实际完成的本地验收，不宣称已在另一主机应用中逐项打开。','small')

out=E/'output/pdf/Pose_objective_diagnosis.pdf';out.parent.mkdir(parents=True,exist_ok=True)
def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#bccbd6'));c.line(42,36,553,36);c.setFont('Latin',8);c.setFillColor(colors.HexColor('#405566'));c.drawString(42,24,'HOI | Frozen input diagnosis | 2026-09-24 UTC');c.drawRightString(553,24,str(doc.page))
doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=48,title='HOI 位姿目标与候选评分诊断',author='HOI research project');doc.build(B.story,onFirstPage=footer,onLaterPages=footer)
(E/'protocol/report_assets.json').write_text(json.dumps({'pdf':str(out),'sha256':sha(out),'embedded_images':B.assets,'builder_sha256':sha(__file__)},ensure_ascii=False,indent=2)+'\n')
print(out)
