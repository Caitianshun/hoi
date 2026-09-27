"""Portable V5 document; measured values always come from private result files."""
from pathlib import Path
import json,importlib.util,zipfile
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
V4=ROOT/'experiments/foreground_stage_calibration_20260927/run01'
spec=importlib.util.spec_from_file_location('v4_report_helpers',V4/'code/build_report_docx.py');v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
v3=v4.v3;read=v3.read;num=v3.numeric
class Report(v4.Report):
    def __init__(self,cost):
        v3.Report.__init__(self,RUN,{},cost)
        self.doc.core_properties.title='动态重建数值稳定性与前景监督配对'
        self.doc.core_properties.subject='V5 数值定位和统一稳定设置下的监督对照'
        for run in self.doc.sections[0].footer.paragraphs[0].runs:
            if 'V3' in run.text:run.text=run.text.replace('V3 基线校准','V5 数值稳定与前景配对')
def run():
    a=read(RUN/'A_summary.json');patch=read(RUN/'patch_effects.json');gate=read(RUN/'protocol/formal_gate.json');pair=read(RUN/'pair_protocol.json');domain=patch['domain'];state=read(RUN/'pipeline.json');mini=a['minimal_repro'];kernel=a['kernel_case'];point=str(kernel['all_bad_attribute_row_intersection'][0]);g=kernel['bad_input_geometry'][point];geom=kernel['bad_points'][point]
    ready=(RUN/'protocol/verification.json').exists();ev=read(RUN/'evaluation/summary.json') if ready else None;cost=read(RUN/'costs.json') if ready else {};decision=read(RUN/'protocol/final_decision.json') if (RUN/'protocol/final_decision.json').exists() else None
    r=Report(cost);r.p('动态重建数值稳定性与前景监督配对','Title');r.p('V5  数值故障定位与统一实现下的监督对照','Subtitle')
    r.p(f"已把历史 W_all 的一次数值失效定位到渲染内部：从有限状态恢复后，两次短重放在 fine{a['first_bad']['iteration']} 的同一批次发现坏梯度；额外观察确认，最终图像和损失仍有限时，内部逆二维协方差已经出现 NaN。一个高斯点即可复现这条异常链。")
    r.p('本轮只采用一项有依据的尺度上界，并明确把它当作数值域及优化变化。它通过故障输入、正常输入、跨密度事件的局部回归和完整恢复检查；随后两臂从同一历史 coarse 起点比较 fine 监督，避免把实现变化混成加权收益。')
    if decision:
        for text in decision['summary']:r.p(text)
    elif ready:r.p('训练与数值评价已完成，固定图例的质量判断和最终阶段决定正在核验。下列已测数值不自动构成晋级结论。')
    else:r.p('当前正式配对仍在执行，以下只报告已经完成的 A 数值证据及冻结协议。没有终态质量结果，不能据此判断前景加权有效或无效。此执行状态版本会在两臂收口后更新。')
    tab=[['工作','已完成证据或状态','边界'],['A 数值定位','重复首异常  单点最小样例','原非法访问未追认为同因'],['A 局部稳定',str(patch['regression']['attempted'])+' 次有限更新','不保证完整日程稳定']]
    if ready:
        for x in cost['formal_attempts']:tab.append([x['run'],x['status']+'  '+str(x['attempted_rounds'])+' 轮','不以早期快照代替终态'])
    else:tab.append(['B 正式配对',state['stage'],'等待两臂收口和统一评价'])
    r.table(tab,[1.2,3.39,2.4],9.5)
    r.p('对象固定为 HOSNeRF Backpack。BEHAVE 本轮没有重新训练或重复诊断，Tennis 不看图、不训练。本轮工程校准不作为论文方法创新，也不以图像分数证明人体和物体的三维接触正确。')

    r.page('首个异常与传播链')
    r.p(f"故障批次为 {', '.join(a['first_bad']['batch'])}，高斯行号 {point}。行号只在该次拓扑状态内有效。实际形变后尺度为 "+', '.join(f'{x:.6g}' for x in g['scale'])+f"，相机深度 {geom['depth']:.6g}；这些输入仍是有限数。")
    r.p('每个高斯在三维中是一团有方向和尺度的椭球。渲染器把它投影成图像上的椭圆，再用二维协方差的逆计算像素权重。有限的椭球参数不保证后续平方、矩阵乘法、行列式和求逆仍处于 float32 的安全范围。这里三维协方差仍有限，保存的逆二维协方差前三项已为 NaN，屏幕半径也出现异常极值。')
    r.p('观察到的顺序是：最终形变尺度极端，内部 conic 非有限，反向屏幕/位置/尺度/旋转等梯度非有限。若继续 Adam，参数及其动量状态会受污染。新守卫在更新前停止；历史污染现场保留。第一条已测坏边界是 conic，具体哪次浮点标量运算最先溢出尚未逐项插桩。')
    r.image(RUN/'diagnostics/A_domain_and_gradient.jpg','左图为实际坏点三个轴的尺度数量级，单位沿用场景坐标，未核实米制。右图从同一状态隔离反向：两种 RGB 目标均出现坏参数梯度，独立正则没有；加共同上界后这些探针均有限。',max_height=3.1)
    r.p('向量雅可比积 VJP 是在给定图像上游梯度时，反向求出的参数梯度。此样例 RGB 上游梯度有限，depth 上游梯度全零且深度有限，不支持把未使用的 depth 分支单独认定为根因。球谐颜色梯度有限，说明坏值集中经过几何投影链。','Caption')

    r.page('单项稳定措施及其真实代价')
    r.p(f"唯一修改为 s_render = exp(min(log_scale_final, log(E)))，E = {domain['upper_bound']:.9g}，来自冻结训练输入的 scene extent。log_scale_final 是形变后每轴的对数尺度，s_render 是交给渲染器的正尺度。没有用开发保留图选择 E，也没有修改下界、学习率、正则、增密日程或坏批次处理。")
    r.p('低于上界时保留原导数；高于上界时截断方向的导数为零，恰在边界时沿用 PyTorch clamp 的定义。规范高斯参数并未被截断，但用于图像生成的形状改变了。因此这是限制数值域的优化设置，不是数学等价的 bug 修复，也不是已验证的新研究贡献。')
    tab=[['输入','域','单独目标','损失','坏梯度元素']]
    for x in patch['loss_and_gradient_isolation']:
        tab.append([{'normal':'正常','failure':'故障'}[x['case']],x['domain'],{'balanced_rgb':'平衡 RGB','uniform_rgb':'均匀 RGB','regularizers':'原三项正则'}[x['objective']],num(x['loss'],6),x['nonfinite_gradient_elements']])
    r.table(tab,[.7,1.0,1.6,1.7,1.99],9)
    normal=next(x for x in patch['forward_VJP_comparisons'] if x['case']=='normal');gradmax=max(x.get('common_finite_max_abs') or 0 for x in normal['gradient_differences'].values())
    r.p(f"正常有限输入的原始 RGB 最大变化为 {a['normal_forward_max_abs']:.9g}，参数组梯度差的最大值为 {gradmax:.9g}；故障输入的 RGB 最大变化为 {a['fault_forward_max_abs']:.9g}。完整逐组 VJP 变化保存在 patch_effects.json，不能把正常输出变化隐藏为一个通过标记。")
    r.p('均匀和平衡 RGB 在同一故障状态都失败，说明平衡权重不是此局部异常的必要条件；但不同监督可能影响模型此前如何走到这个状态。因此正式对照仍需共享稳定实现和同一初值。')

    r.page('局部回归与完整恢复')
    r.p(f"从故障前的有限状态实际执行 {patch['regression']['attempted']} 次优化，最后到 fine{patch['regression']['last_iteration']}，跨过原日程的下一次密度事件。密度事件指高斯增密与剪枝，它同时改变点数、参数张量及对应 Adam 状态；本轮检查了操作前后及每次参数更新后的有限性，没有跳过坏批次。")
    r.image(RUN/'diagnostics/A_regression.jpg','上图为实际不同训练批次的总损失，不是固定验证曲线；下图为每次渲染触及尺度上界的轴比例。虚线表示原日程的密度事件。',max_height=3.55)
    r.p(f"局部回归累计触及 {patch['regression_triggered_axes']} / {patch['regression_total_axes']} 个渲染轴，比例 {100*patch['regression_axis_trigger_fraction']:.6g}%。分母跨帧和迭代重复计数，不是独立点的比例。全部局部 loss、梯度、参数和 Adam 检查有限，只支持这一窗口。")
    r.p(f"从保存状态恢复时，模型、Adam 组及 moments、缓冲、RNG 与剩余采样栈逐项精确一致；恢复后两批 ID 和最终 RNG 也一致。CUDA 后续张量最大差为 {a['restored_tail_max_tensor_abs']:.9g}，说明恢复语义正确不等于并行原子累加轨迹逐位一致。保存记录使用 stage 元数据，测试路径不依赖文件名包含 fine。")
    r.p(f"A 实际使用 {gate['A_attempts']} 个尝试轮和 {gate['A_no_optimizer_probes']} 个无优化探针，GPU 任务墙钟 {gate['A_GPU_job_seconds']:.3f} 秒；失败导入、接口检查及启动保守上界均计入。局部门槛允许开始 B，不保证两条完整训练都会成功。")

    r.page('正常与故障输入的前向对照')
    r.image(RUN/'diagnostics/A_normal_forward.jpg','正常有限输入的同一模型与两张训练图。每行依次为 GT、原实现、尺度上界、raw RGB 最大通道差热图。图像显示裁剪到 [0,1]，热图按原浮点差计算；色标范围来自本例。',max_height=3.05)
    r.image(RUN/'diagnostics/A_failure_forward.jpg','故障前状态的同一批次，列顺序相同。原前向图像仍有限，不能据此认定梯度安全。此图只展示输入域修改的作用，不作为模型质量或开发集收益。',max_height=3.05)
    r.p(f"单点复现数值文件仅 {mini['case']['bytes']} 字节，保留真实相机和上游 RGB/depth 梯度。原调用与有界调用的输出和反向计数保存在 minimal_repro/validation.json。数据样例可独立搬运，但仍须匹配 Torch/CUDA 与 rasterizer 扩展；完整训练轨迹还需本机大检查点。")

    r.page('正式配对的冻结协议')
    r.p('B-U 与 B-F 使用同一 H1 coarse3000 的模型、球谐阶数与屏幕半径。按官方 fine 入口重置 Adam、位置梯度累计、分母及形变累计，重建完整采样栈后恢复父 RNG；准备出的完整 fine0 状态被两臂共同加载。自己的 fine 续跑必须恢复自己的优化器和剩余栈，不再次重置。')
    r.table([['因素','B U','B F'],['fine RGB','每图全图均匀 L1','每图前景与背景均值各半'],['初始化','同一完整 fine0','完全相同'],['共同稳定设置','训练 scene extent 尺度上界','完全相同'],['训练日程','fine14000  batch2  seed12345','完全相同'],['其他设置','原相机 时间 正则 SH 形变 密度规则','完全相同']],[1.29,2.85,2.85],9.5)
    r.p('平衡 RGB 先对每像素 RGB 通道绝对差取均值，再分别对前景与背景区域平均，各乘 0.5，最后 batch 中图像等权。mask 采用原 uint8≥128；只有一个非空区域时它取全部权重，预测颜色不新增裁剪。原始三项 fine 正则不变，没有新增 LPIPS、深度、alpha 或接触训练损失。')
    r.p('共享历史 coarse 未按新尺度域重训。因此主比较回答这个固定起点下 fine 监督的作用，不是有界版本从零完整训练，也不能判断最佳 coarse 策略。密度操作可能消耗不同随机数，两臂后续采样是否仍一致由实际 ID 记录确认，不临时更换采样器强求对齐。')
    r.p('两臂均关闭后才评价可用终态。完整协议为原 268 训练帧和 16 开发保留帧，00000 是外推帧，仍留在主表，另列其余 15 帧补充。开发帧已经看过，不构成独立序列确认。静放背包部分时间不在合并前景 mask 内，独立人体/物体指标均为 NA。')
    r.p('每步检查 loss、参数/屏幕梯度、更新后模型和 Adam；每100步保留最近有限完成状态，1000步与终态不可变归档。末轮沿用原实现只反传不 optimizer.step。新数值异常即关闭该臂，不以早期快照代替终态，不另开新尝试。')

    if ready:
        r.page('完整数值结果与配对门槛')
        table=[['集合 区域','H1 PSNR','B U PSNR','B F PSNR','B F 减 B U']]
        for split,label in [('train','训练'),('retained','开发16'),('retained_without_00000','补充15')]:
            for region,reglabel in [('foreground','前景'),('background','背景'),('full','全图')]:
                vals=[ev['summaries'][b][split][region]['psnr_db'] for b in ['H1','B_U','B_F']];table.append([label+' '+reglabel,*map(num,vals),num(None if vals[1] is None or vals[2] is None else vals[2]-vals[1],signed=True)])
        r.table(table,[1.69,1.3,1.3,1.3,1.4],9)
        table=[['开发16 配对','前景均值 dB','前景中位 dB','前景 LPIPS 差','背景均值 dB']]
        for key in ['B_F-B_U','B_U-H1','B_F-H1']:
            d=ev['paired'][key]['retained'];table.append([key,num(d['foreground']['psnr_db']['mean'],signed=True),num(d['foreground']['psnr_db']['median'],signed=True),num(d['foreground']['lpips_spatial_mean']['mean'],signed=True),num(d['background']['psnr_db']['mean'],signed=True)])
        r.table(table,[1.39,1.4,1.4,1.4,1.4],9)
        r.p('主门槛要求 B-F−B-U 的开发前景 PSNR 均值至少 +1 dB、中位数大于 0、LPIPS 至少降低 0.05、背景 PSNR 下降不超过 0.3 dB，并检查全部固定图例有无新断肢、漂浮或背景破碎。缺少任一终态时配对门槛 NA，不能称质量门槛失败。')
        r.p('PSNR/SSIM 越高越好，LPIPS 越低越好。主指标按原口径把 raw RGB 裁剪到 [0,1]；另存 raw PSNR/MSE。区域 SSIM 使用原 7×7 设置，AlexNet 0.1 spatial LPIPS 保留跨边界感受野。逐帧平均与总 SSE/总像素的 pooled PSNR 分开报告。','Caption')
        r.page('整体质量 代价与采样差异')
        table=[['分支 集合','全图 SSIM','前景 SSIM','全图 LPIPS','前景 LPIPS']]
        for b in ['H1','B_U','B_F']:
            for split,label in [('train','训练'),('retained','开发16')]:
                q=ev['summaries'][b][split];table.append([b+' '+label,num(q['full']['ssim']),num(q['foreground']['ssim']),num(q['full']['lpips_spatial_mean']),num(q['foreground']['lpips_spatial_mean'])])
        r.table(table,[1.39,1.4,1.4,1.4,1.4],9)
        table=[['分支','尝试轮','实际更新','终态点数','训练进程秒']]
        for x in cost['formal_attempts']:
            q=x['result'];table.append([x['run'],x['attempted_rounds'],str(x['optimizer_updates_interval']),q.get('final_points','NA'),num(q.get('seconds'))])
        r.table(table,[.89,1.1,1.7,1.6,1.7],9)
        r.p(f"全部 GPU 任务墙钟累计 {cost['GPU_task_seconds']:.3f} 秒，即 {cost['GPU_task_hours']:.6f} 小时，包含 A、正式训练、渲染、加载和失败。物理 GPU1 RTX3090。A {cost['A_attempts']} 尝试、{cost['A_no_optimizer_probes']} 探针，B {cost['B_attempts']} 尝试；历史 H1 coarse 成本属于复用，另留 V3/V4 成本索引。")
        sampling=read(RUN/'sampling_comparison.json');r.p(f"共有 {sampling['common_rounds']} 轮可对照采样，其中 {sampling['divergent_rounds']} 轮批次不同，首次差异为 {sampling['first_difference']}。采样差异是原随机数和密度操作的实际结果，未为对齐而改训练器。")
        r.p('completed run.json 记录 PyTorch allocated/reserved 峰值、终态及峰值点数；未连续采样 nvidia-smi PID 峰值，故该项 NA。计时是 GPU 任务进程墙钟，不是纯 CUDA 核耗时。检查点写盘成本包含在进程中。')
        r.page('固定图例中的收益与残留错误')
        review=read(RUN/'protocol/visual_review.json')
        for text in review['report_paragraphs']:r.p(text)
        r.p('后附全部固定开发全图、相同窗口裁剪及训练图，顺序均为 GT、H1、B-U、B-F。逐帧审阅记录与图像身份保存在 protocol/visual_review.json；图像选择和裁剪窗口没有根据本次结果重选。')

    r.page('阶段决定与尚未解决的问题')
    if decision:
        for text in decision['decision']:r.p(text)
    else:r.p('正式配对尚未完成最终核验，当前只确认局部数值门槛满足，研究路线决定待两臂收口后给出。不会自动扩大训练矩阵或引入新模块。')
    r.p('尚未解决的问题包括：最终形变为什么在长优化中走向极端尺度；V4 的早期 CUDA 非法访问是否同因；HOS 相机可能使用全视频、动态点云短配对三角化近似静态所带来的偏差。尺度上界不能自动修复错误表面、相对运动或未观测区域，也不能把普通图像指标变成三维几何评价。')
    r.p('9月29日前收口、最迟10月1日路线决定、10月7日收敛主问题、11月4日核心结果冻结；11月5日至15日连续11个完整自然日保留给集中写作。本轮不自动扩展独立事件、权重扫描或新方法。')
    r.page('证据、复现与同步边界')
    r.p('实现过程中保留了启动、唯一源码匹配、requires_grad 过滤、前向/反向视图错配等接口失败。它们没有被抹为零成本。最小样例从同一次失败 backward 的相机和几何恢复 opacity，避免把 batch 最后一帧 forward 与第一帧 backward 混用。')
    if ready:
        verify=read(RUN/'protocol/verification.json')
        r.p(f"独立核验复查 {verify['frozen_assets_rehashed']} 项冻结身份、{verify['source_RGB_mask_hashes']} 个 RGB/mask 哈希、{verify['metrics_rows']} 指标行及 {verify['paired_rows']} 配对行；开发 PSNR/SSE 独立重算 {verify['retained_PSNR_rows_recomputed']} 行，PSNR 最大差 {verify['max_PSNR_difference']}。SSIM/LPIPS 沿冻结评价器计算，没有宣称第二套实现全部重算。")
        r.p('CPU核验曾因开发清单没有逐帧 image_sha256 字段停止。修复后先核对原 V3 冻结 Backpack 压缩包 SHA256，再将开发 RGB/mask 与包内成员逐个比较。原错误和旧源码留存，训练、渲染与指标定义均未改变；修复没有新增 GPU 或优化工作。')
    r.p('同步例外需明确：两份上游派生渲染快照曾因目录位置进入自动同步，已从远端当前版本移除，Git 历史仍保留。AGENTS 中部分历史研究摘要也曾随既有白名单公开，不能笼统称历史摘要全私有；本轮后续入口只记录执行边界和私有路径。原始图像、模型及完整报告没有随这两份快照上传。')
    r.p('ROOT_CAUSE.md、patch_effects.json、first_bad_tensor.json、state_manifest.json、environment.json 与 pair_protocol.json 分别给出异常链、数学域变化、首坏索引、状态身份、运行后端和配对边界。REPRODUCE.md 区分只读复算与需预算的优化命令。最小 ZIP 包含固定浮点小块和关键图，完整模型及环境仍留训练机。')

    if ready and read(RUN/'protocol/finals.json')['runs']:
        figs=read(RUN/'figure_manifest.json')['hos'];ex=read(RUN/'protocol/fixed_examples.json')
        for start in range(0,len(ex['hos_retained_frame_ids']),4):
            ids=ex['hos_retained_frame_ids'][start:start+4]
            r.page('全部开发全图 '+str(start//4+1));r.p('每行从左到右为 GT、历史 H1、B-U、B-F；全部使用同一显示规则。缺失终态明确标 NA。','Caption')
            items=[x for fid in ids for x in figs if x['frame_id']==fid and x['kind']=='full'];r.image(r.sheet('dev_full_'+str(start),items,1,True),'完整画面 '+', '.join(ids)+'；00000 为时间外推，仍纳入主表。',max_height=7.7)
            r.page('开发前景固定裁剪 '+str(start//4+1));r.p('每组 GT、H1、B-U、B-F。各方法采用训练前固定的相同像素窗口；指标来自完整 mask，不是裁剪矩形。','Caption')
            items=[x for fid in ids for x in figs if x['frame_id']==fid and x['kind']=='foreground_crop'];r.image(r.sheet('dev_crop_'+str(start),items,2,True),'固定裁剪 '+', '.join(ids)+'。区域标签不能把人体与静放背包可靠分开。',max_height=7.6)
        for start in range(0,len(ex['training_frame_ids']),4):
            ids=ex['training_frame_ids'][start:start+4];r.page('固定训练图像 '+str(start//4+1));r.p('每行 GT、H1、B-U、B-F。仅用原先冻结的时间等间隔训练帧解释拟合，不重新选好例。','Caption')
            items=[x for fid in ids for x in figs if x['frame_id']==fid and x['split']=='train' and x['kind']=='full'];r.image(r.sheet('train_full_'+str(start),items,1,True),'固定训练帧 '+', '.join(ids)+'。完整268帧数值见机器表。',max_height=7.7)
    r.output.mkdir(exist_ok=True);path=r.output/'V5_numerical_stability_and_fine_pair.docx';r.doc.save(path)
    with zipfile.ZipFile(path) as z:
        media=[n for n in z.namelist() if n.startswith('word/media/')];external=[n for n in z.namelist() if n.endswith('.rels') and b'TargetMode="External"' in z.read(n)];assert media and not external
    (RUN/'protocol/document_build.json').write_text(json.dumps(dict(document=v3.ident(path),embedded_media=len(media),external_relationships=external,section_titles=r.section_pages,B_complete_evidence_loaded=ready,final_decision_loaded=decision is not None,visual_QA='pending canonical rendering and every-page inspection'),indent=2,ensure_ascii=False)+'\n');print(path)
if __name__=='__main__':run()
