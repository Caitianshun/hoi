"""Portable V6 DOCX from frozen evidence, using bundled document runtime."""
from pathlib import Path
import json,importlib.util,zipfile
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
V4=ROOT/'experiments/foreground_stage_calibration_20260927/run01'
spec=importlib.util.spec_from_file_location('v4_document_helpers',V4/'code/build_report_docx.py');v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
read=v4.read;num=v4.num
class Report(v4.Report):
    def __init__(self,cost):
        v4.v3.Report.__init__(self,RUN,{},cost)
        self.doc.core_properties.title='几何与外观梯度作用范围验证'
        self.doc.core_properties.subject='V6 前置验证关闭报告'
        for run in self.doc.sections[0].footer.paragraphs[0].runs:
            if 'V3' in run.text:run.text=run.text.replace('V3 基线校准','V6 梯度作用范围')
def run():
    cost=read(RUN/'costs.json');gate=read(RUN/'protocol/formal_gate.json');route=read(RUN/'routing_equivalence.json');bad=[x for x in route['rows'] if not x['passed']]
    contrib=read(RUN/'contribution_manifest.json');probe=read(RUN/'time_camera_probe.json');loc=read(RUN/'diagnostics/gradient_localization/localization.json');audit=read(RUN/'camera_time_audit.json');parity=read(RUN/'renderer_parity.json')
    r=Report(cost);r.p('几何与外观梯度作用范围验证','Title');r.p('V6  前置验证结果与未启动正式训练的原因','Subtitle')
    r.p('本轮以 D2 必需前置验证未建立收口。输入和渲染一致性 D0 通过，冻结模型诊断 D1 完成；唯一正式分支 C-route 没有启动。因此 C 相对 B-U/B-F 的质量变化、晋级门槛和科学路线结论均为 NA。NA 表示没有可比较的新终态，不表示零提升或方法无效。')
    r.p('要检验的问题是：此前把前景 RGB 误差加大后，训练人物细节改善却伴随开发观测和背景代价。若只让这部分额外梯度直接更新球谐颜色 SH，其余几何、透明度、形变及增密仍接受全图均匀监督，能否保留收益并减轻代价？这需要先证明梯度确实被送到约定的参数组。')
    r.p(f"分组梯度检查中，C-route 的全部条目通过；uniform_all 的 {len(bad)} 个网格参数超出执行前声明的等价容差。所有导数仍有限。后续交错重复的差值落回原范围，但不以有利重复覆盖首次未通过，也不扩大容差。因此未继续 Adam、密度、恢复和128轮回归，更未运行正式 C。")
    r.table([['阶段','实际状态','可支持的结论'],['D0','通过','未发现本轮检查范围内的绑定或渲染错误'],['D1','两模型各八训练帧完成','仅描述模型合成贡献与敏感性'],['D2','必需等价门槛未建立','实现仍缺完整优化及状态验收'],['C 与终态评价','未启动','质量与晋级门槛 NA']],[.75,2.3,3.94],10)
    r.p(f"实际使用物理 GPU1 RTX3090：{cost['diagnostic_renders']} 次附加渲染、{cost['diagnostic_backwards']} 次无优化 rasterizer 反向，GPU 任务墙钟 {cost['GPU_task_seconds']:.3f} 秒。新增优化尝试和 optimizer.step 均为 {cost['optimizer_steps']}。失败和重复检查全部计账。")
    r.p('旧 V5 终态、负面质量结果和历史数值故障保持不变。本轮不新增第二正式臂、不扫权重、属性组合或新模块。11月4日核心冻结与11月5—15日完整写作窗口保持。')

    r.page('输入相机时间与渲染路径')
    r.p(f"完整导出 {len(audit['rows'])} 行实际输入：268训练与16开发。图像、掩码与相机均按源帧 stem 绑定；K/w2c 与发布相机文件逐元素一致，尺寸1277×718、不resize，时间为（源帧号−1）/282。00000为外推，仍保留在16帧主协议。568个输入文件与V5冻结索引重新哈希比较一致。")
    r.p('时间字段是名义帧号，不是测得的曝光秒数。相机为发布的全视频预处理结果，原始 SfM 输入清单未知；本轮没有重新估计相机，也不把固定相机文件解释为没有开发信息。以下距离除以训练场景尺度 E，角度为相对相机旋转。')
    tab=[['开发帧','左训练帧','右训练帧','左右距离除E','左右旋转度']]
    for x in audit['coverage']:
        a=x['neighbors']['left'];b=x['neighbors']['right']
        tab.append([x['frame_id'],a['frame_id'] if a else '无',b['frame_id'] if b else '无',' / '.join('NA' if v is None else num(v['camera_distance_over_extent'],4) for v in [a,b]),' / '.join('NA' if v is None else num(v['rotation_degrees'],3) for v in [a,b])])
    r.table(tab,[.9,1.1,1.1,1.9,1.99],8.7)
    m=max(v['max_abs'] for row in parity['rows'] for ds in row['comparisons'].values() for v in ds.values())
    r.p(f"B-U/B-F各用固定00001、00041训练帧检查。同路径重复、训练bound路径、历史评价monkeypatch路径和新最终尺度接口的 RGB/depth/radii 及可比最终属性最大绝对差均为 {m:g}。新接口单独输出原始规范log尺度、形变log尺度和最终有界尺度，不复制全局activation替换到增密。")
    r.p('该核查支持这些已测接口一致，不能证明相机真实准确、遮挡层正确，或覆盖所有renderer配置。完整矩阵、源哈希和邻帧间隔见 camera_time_audit.csv/json。','Caption')

    r.page('梯度路由的实现与验证范围')
    r.p('SH是用低阶基函数表达随观察方向变化的颜色；本轮A组只包含规范 f_dc/f_rest。G组包含位置、对数尺度、旋转、透明度以及全部形变网络和网格参数。opacity改变遮挡，归G。q是每张图的屏幕位置代理张量，其梯度进入增密统计，不是另一个优化器参数。')
    r.table([['对象','送入的梯度','实现保护'],['G','全图均匀 RGB U 加原正则 R','保留SH方向对位置的合法导数'],['A','原前景背景各半 RGB F','仅显式赋给规范SH叶子'],['q','U的显式返回梯度','不用retain_grad留下的混合缓存'],['优化器','每个有效迭代一次Adam','候选实现 未完成一步验收'],['增密','沿原批内有符号求和','每轮重新分组防止缓存旧Parameter']],[.7,2.7,3.59],10)
    r.p('一次batch前向后，先对G与q求 U+R 的向量雅可比积 VJP，再对A求F的VJP。VJP就是给定输出损失的上游梯度后，把它反向传播到指定输入。两次求导之间没有更新参数或增密；G/A叶子梯度分别赋值，密度接口只接收单独保存的q_uniform。')
    r.p('这是按参数组指定的更新规则，不声称存在一个普通共同标量loss同时产生所有这些偏导。颜色改变还会影响未来迭代的残差，因此只限制直接梯度通道，不意味着几何长期保持不动。U/F/R分别记录。')
    part=read(RUN/'gradient_partition.json')['parameters'];inactive=[p for p in part if p['expected_None']]
    r.p(f"当前优化器覆盖 {len(part)} 个参数张量；{len(inactive)} 个张量按源码预期不连接，包括被no_do/no_dshs关闭的head和未调用的timenet。None、有限零梯度、非零梯度和非有限值分开检查。位置/颜色等参数引用每轮从实际优化器重取，避免增密替换Parameter后误写旧张量。")
    r.p('已完成的是同状态、固定批次的导数检查。一步Adam、密度缓冲、保存恢复以及跨fine1100事件的128轮回归均未执行。train_routed_fine.py虽然已写出硬账本、原子保存及守卫，但不能称这些运行时语义已经验收。')

    r.page('前置等价检查为什么没有放行')
    r.p('uniform_all把A/G/q全部恢复为uniform，用于判断新路由接口能否复现原单backward。参照是同一有限B-U fine1000状态、同一00001/00041批次；U与F各重复两次，以同后端重复误差和float32精度声明容差，再检查C-route和uniform_all。')
    r.p('逐参数绝对容差为两倍参考重复最大差，加64个float32机器精度乘参考梯度最大幅值，再加1e−10；相对L2采用对应的重复差加精度项。该规则在候选差异判定前固定。小范数、零范数与None单列；没有看到未过条目后增大阈值。')
    tab=[['未通过网格','最大绝对差','原绝对容差','相对L2差']]
    for x in bad:tab.append([x['name'].split('grid.')[-1],f"{x['difference']['max_abs']:.3e}",f"{x['atol']:.3e}",f"{x['difference']['relative_L2']:.3e}"])
    r.table(tab,[2.19,1.6,1.6,1.6],9.5)
    r.image(RUN/'output/report_figures/routing_gate.png','虚线为原门槛。超过1的条目未通过；这是实现验收结果，不是重建质量对比。',max_height=2.9)
    r.p('追加交错的原backward与uniform_all检查中，参数哈希始终不变，差值均回到原容差内。PyTorch官方文档明确CUDA grid_sample反向可能非确定；本例与浮点并行累加差异相容，但没有确定性替换来证明唯一根因。两次参考重复不足以覆盖完整变动范围，也是本次验收方案的局限。')
    r.p('保留首次失败及全部真实梯度，不把后来的有利重复当撤销条件。停止点是“必要等价证据未建立”，不是新NaN或已证实路由数学错误。证据：routing_equivalence.json 与 diagnostics/gradient_localization。','Caption')

    r.page('实际合成贡献的定义与数值核对')
    r.p('渲染一个像素时，前方高斯的透明度决定后方还剩多少透射光。把所有形变后几何、透明度和遮挡顺序固定，背景设黑，为每个高斯指定可求导的预计算颜色；对输出一个颜色通道求和并反向到该点颜色，得到它在这些像素中的实际合成权重之和。对FG mask内求和得到前景贡献，全图减前景得到背景贡献。')
    r.p('新路径明确传 shs=None，避免原override_color同时传SH被后端拒绝。没有删除点或置零opacity，原遮挡顺序保留。颜色全1时三个输出通道相同并表示累计alpha；点贡献之和与区域alpha之和相符。几何和radii与原路径一致。')
    check=read(RUN/'diagnostics/B_U_repeat_check/00001_contribution_checks.json')
    r.p(f"初版使用固定绝对阈值时，原始背景差最小为 {check['min_bg']:.9g}；同一路径重复的全图贡献最大差为 {check['total_repeat_max_abs']:.9g}。因此单一绝对阈值不足以描述颜色反向的浮点累加。保留两次失败后，在逐点重复误差加float32精度范围内核对；原始有符号值没有截零，完整误差记录留存。这是D1诊断接口修正，没有改D2的冻结容差。")
    maxerr=max(x['checks']['sum_total_relative_error'] for m in contrib['models'].values() for x in m['rows']);maxfg=max(x['checks']['sum_fg_relative_error'] for m in contrib['models'].values() for x in m['rows'])
    r.p(f"两模型各八帧最终全图守恒的最大相对误差为 {maxerr:.3e}，FG为 {maxfg:.3e}。贡献数组以float32稀疏无损保存，缺省行恰为零；每个模型的行号只在自身固定拓扑中有意义，不跨臂一一对齐。")
    r.p('对每个点，把这八帧贡献归一化成p，再计算 N_eff = 1 / ∑ p²。若贡献集中在一个采样时刻，N_eff接近1；若均匀分散在多个时刻，数值增大。总贡献为零标unsupported_in_sample，不除零。它不是实际可见帧数，更不是物理表面身份。')
    r.p('现有合并mask会把部分静放背包归为背景，没有独立track/flow来验证材料点对应。本轮 physical_surface_correspondence 为 NA。不能因为贡献跨帧较多就认定某个高斯跟住同一只手或同一块衣服。')

    r.page('冻结模型的贡献分布')
    tab=[['模型','点数','八帧无贡献比例','N_eff中位','FG加权N_eff']]
    for x in contrib['summary']:tab.append([x['arm'],x['points'],f"{100*x['unsupported_fraction']:.2f}%",num(x['N_eff_median']),num(x['FG_weighted_N_eff'])])
    r.table(tab,[.8,1.2,2.,1.4,1.59],10)
    r.image(RUN/'output/report_figures/contribution_summary.png','左图只统计在八帧至少有一次贡献的点。右图按真实合成贡献加权，减少完全无支持点数量的影响；两臂拓扑不同，不把行号匹配。',max_height=3.5)
    r.p('B-F的采样内贡献更集中，与其表示更依赖部分训练观测的解释相容。但相机位置、遮挡和时间同时变化，点拓扑及总点数也不同；不能把差异单独认定为时间过拟合，或认为已有独立表面证据。')
    r.p('触及尺度上界的点也按实际贡献单列。历史bound事件CSV逐次给出重复渲染轴分母；D1给出的点贡献是另一种口径，二者不可相除当受影响像素比例。canonical原exp尺度、形变后raw尺度与最终bounded尺度分开保存。')
    r.p('后附有限时间相机矩阵使用同一固定八训练帧及最近的另一训练帧；同距取较小源帧号。对角项有真实训练RGB，可报告对角拟合；交叉项只有模型输出，不与错误的GT计算新视角PSNR。')

    for start in range(0,8,2):
        ids=[x['frame_id'] for x in probe['models']['B_U']['rows']][start:start+2]
        r.page('时间与相机敏感性 '+str(start//2+1))
        r.p('每行四列依次为 Ci ti、Ci tj、Cj ti、Cj tj；中间两列没有GT。下面只展示模型对时间和相机变化的响应，不能分离两类泛化因果。','Caption')
        items=[]
        for fid in ids:
            for arm in ['B_U','B_F']:
                x=next(x for x in probe['models'][arm]['rows'] if x['frame_id']==fid)
                items.append(dict(x['figure'],label=f"{arm}  i={fid}  j={x['neighbor']}  dt={x['time_delta']:.7f}"))
        r.image(r.sheet('time_camera_'+str(start),items,1,False),'固定训练帧 '+', '.join(ids)+'。完整对角指标、交叉像素敏感性与固定canonical行号数组在 time_camera_probe.json。',max_height=7.7)

    r.page('成本验收和阶段决定')
    r.table([['预算项','实际','上限'],['诊断优化尝试',cost['diagnostic_optimization_attempts'],cost['limits']['D_attempts']],['正式优化尝试',cost['formal_optimization_attempts'],cost['limits']['C_attempts']],['附加诊断渲染',cost['diagnostic_renders'],cost['limits']['diagnostic_renders']],['无优化rasterizer反向',cost['diagnostic_backwards'],cost['limits']['diagnostic_backwards']],['GPU任务秒',num(cost['GPU_task_seconds']),cost['limits']['total_GPU_seconds']]],[3.,1.995,1.995],10)
    r.p('所有GPU进程均已退出。计时包含加载、错误、重复和序列化，是GPU任务进程墙钟；没有连续PID显存峰值，故该项NA。初次D2结果写JSON时遇到numpy bool_序列化错误，错误和调用成本保留，修复序列化后才生成完整门槛表。没有优化、失败训练恢复或尾段重放。')
    r.p('正式C未启动，完整268训练、16开发和排除00000的15帧补充均没有C新分数。逐帧机器表保留B-U/B-F旧值及C的全部NA行，配对C−B-U、C−B-F均留空并写明原因；没有用早期状态或旧终态补缺。')
    r.p('阶段决定是技术前置验证关闭，科学问题仍未回答。不能套用附件中“相对B-F恢复背景”“开发仍碎片化”或“C更差”等需要C终态的分支。若继续，首先需要事前固定、能覆盖后端非确定性的等价验收方案，再做原定Adam/密度/恢复/128轮检查；本轮不自动换阈值或反复重试到过关。')
    r.p('旧V5关于50/50监督不升级的结论不变。不新增权重/属性组合、注意力、接触、新数据或骨干。新的独立对应证据仍是未来结构问题的前提，不能用本次贡献图替代。最迟10月1日路线决定，10月7日主问题收敛；11月4日核心结果冻结，11月5—15日完整写作。')
    r.p('文档附录保留V5全部16开发全图和同窗口裁剪、8训练图，明确为历史参照。它们没有C列，不能当本轮方法质量证据。完整原图、模型、全图raw与真实完整梯度留训练机；反馈包提供索引、源代码、贡献数组、固定浮点小块与全部图例。')
    r.p('参考：PyTorch 2.7 grid_sample官方说明 https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.grid_sample.html 。这里只引用其CUDA反向可能非确定的说明，不以文档代替本例根因证明。','Caption')

    figs=read(RUN/'figure_manifest.json')['inherited_V5'];ex=read(RUN/'protocol/fixed_examples.json')
    for start in range(0,16,4):
        ids=ex['hos_retained_frame_ids'][start:start+4];r.page('历史开发全图 '+str(start//4+1));r.p('V5冻结参照。每行依次GT、H1、B-U、B-F；没有C终态。本轮未用这些开发图挑选路由、容差或训练时刻。','Caption')
        items=[x for fid in ids for x in figs if x['frame_id']==fid and x['kind']=='full'];r.image(r.sheet('old_dev_full_'+str(start),items,1,True),'全部固定开发全图 '+', '.join(ids)+'。显示clip到[0,1]；质量结论沿用V5记录。',max_height=7.7)
        r.page('历史开发固定裁剪 '+str(start//4+1));r.p('V5冻结参照。GT、H1、B-U、B-F采用原相同像素窗口；指标来自完整mask，并非此裁剪矩形。','Caption')
        items=[x for fid in ids for x in figs if x['frame_id']==fid and x['kind']=='foreground_crop'];r.image(r.sheet('old_dev_crop_'+str(start),items,2,True),'原固定裁剪 '+', '.join(ids)+'；静放包mask语义限制保持。',max_height=7.6)
    for start in range(0,8,4):
        ids=ex['training_frame_ids'][start:start+4];r.page('历史固定训练图 '+str(start//4+1));r.p('V5原固定训练图。每行GT、H1、B-U、B-F，不是新训练输出。与D1完全相同的源帧选择，未重新挑好例。','Caption')
        items=[x for fid in ids for x in figs if x['frame_id']==fid and x['kind']=='full' and x['split']=='train'];r.image(r.sheet('old_train_'+str(start),items,1,True),'固定训练帧 '+', '.join(ids)+'。独立物理表面对应仍为NA。',max_height=7.7)
    r.page('未完成验证与可复算范围')
    r.p('本轮没有完成：Adam一步等价、密度缓冲等价、保存恢复、128轮跨事件回归、正式C训练、C全量终态评价、C图例审阅及质量晋级。不得把这些NA表述为没有问题。没有新非有限张量不等于完整训练稳定。')
    r.p('只读CPU核验已复查输入和继承模型身份、完整NA覆盖、真实调用预算、贡献稀疏数组及N_eff复算。它不重新运行CUDA，也不复算历史全部SSIM/LPIPS。真值几何、独立材料点对应和人体/物体独立mask不具备；当前只支持图像模型行为分析。')
    r.p('DOCX图片直接内嵌。文档验收为本机bundled LibreOffice渲染逐页检查及结构性外链检查，不称已在另一主机Word逐项验证。不额外交付PDF。')
    r.p('仅自产代码和源配置进入Git同步白名单。附件、相机矩阵、图像、贡献数组、模型、指标、日志与文档留私有目录。历史V5两份上游快照及AGENTS摘要曾随同步公开，当前版本已按原记录处理，Git历史保留；不能把历史重新描述为全私有。')
    r.p('复算入口为REPRODUCE.md；原因分级为CAUSE_ASSESSMENT.md，阶段决定为NEXT_DECISION.md。大文件见state_manifest.json和environment.json。GPU复现另需明确输出与预算，不重跑原脚本覆盖首次失败。')
    path=RUN/'output/V6_gradient_scope_prerequisite_report.docx';r.doc.save(path)
    with zipfile.ZipFile(path) as z:
        media=[n for n in z.namelist() if n.startswith('word/media/')];assert media
        external_images=[]
        import lxml.etree as ET
        for n in z.namelist():
            if n.endswith('.rels'):
                for rel in ET.fromstring(z.read(n)):
                    if rel.get('Type','').endswith('/image') and rel.get('TargetMode')=='External':external_images.append(n)
        assert not external_images
    (RUN/'protocol/document_build.json').write_text(json.dumps(dict(path=str(path),embedded_images=len(media),external_image_relationships=external_images),indent=2)+'\n')
    print(path)
if __name__=='__main__':run()
