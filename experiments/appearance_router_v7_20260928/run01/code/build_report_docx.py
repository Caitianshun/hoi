"""Result-first portable report. Author with the bundled document runtime."""
from pathlib import Path
import json,importlib.util,zipfile
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
V4=ROOT/'experiments/foreground_stage_calibration_20260927/run01'
sp=importlib.util.spec_from_file_location('report_helpers',V4/'code/build_report_docx.py');h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
read=h.read;num=h.num
from write_decision import LABELS
REGIONS=[('full','全图'),('foreground','前景'),('background','背景')]
KEYS=['psnr_db','ssim','lpips_spatial_mean']
class Report(h.Report):
    def __init__(self,cost):
        h.v3.Report.__init__(self,RUN,{},cost)
        self.doc.core_properties.title='外观梯度路由完整实验结果'
        self.doc.core_properties.subject='V7 单模块对照与三项图像质量评价'
        for x in self.doc.sections[0].footer.paragraphs[0].runs:
            if 'V3' in x.text:x.text=x.text.replace('V3 基线校准','V7 外观梯度路由')
def run():
    ev=read(RUN/'evaluation_summary.json');q=read(RUN/'quality_decision.json');cost=read(RUN/'costs.json');acc=read(RUN/'module_acceptance.json');var=read(RUN/'diagnostics/continuation_variation.json');ex=read(RUN/'protocol/fixed_examples.json')
    r=Report(cost);r.p('外观梯度路由完整实验结果','Title');r.p('V7  唯一 M1 终态与开发图像质量取舍','Subtitle')
    r.p(f"M1已完整执行fine1至14000，并完成268训练帧、16开发保留帧和排除00000的15帧补充评价。本轮决定：{LABELS[q['decision']]}。下表为16帧逐帧宏平均；PSNR和SSIM越高越好，LPIPS越低越好。")
    tab=[['区域','设置','PSNR dB ↑','SSIM ↑','LPIPS ↓']]
    for reg,label in REGIONS[:2]:
        for b in ['B_U','B_F','M1']:
            x=ev['summaries'][b]['retained'][reg];tab.append([label,b,*[num(x[k],6) for k in KEYS]])
    r.table(tab,[.8,1.,1.73,1.73,1.73],10)
    relnames={'dominates':'三项支配参照','dominated':'被参照三项支配','tradeoff':'三项之间存在取舍','equal_at_report_precision':'在报告精度下持平'}
    for b in ['B_U','B_F']:
        r.p(f"相对{b}：全图{relnames[q['relations'][b]['full']['relation']]}，前景{relnames[q['relations'][b]['foreground']['relation']]}。所有配对差均为M1减参照，LPIPS负差代表改善。未把三种不同单位加成总分，也未使用旧背景或几何代理的一票否决。")
    r.p('本轮要检验的是：将前景平衡RGB监督的直接作用限制到颜色，能否保留其感知收益并减少全图代价。原B-U是全图均匀监督，B-F对所有参数采用前景背景各半，M1仅让规范球谐颜色接收后者；其余条件及原尺度保护相同。')
    r.p(f"诊断实际{cost['D_attempts']}次优化尝试；正式{cost['C_actual_attempts']}次尝试、{cost['C_optimizer_calls']}次Adam调用，额外正式重放{cost['C_replay_attempts']}轮。GPU任务墙钟{cost['GPU_task_hours']:.4f}小时，仅物理GPU1 RTX3090。终态已冻结，未选择中间开发最优。")
    r.p('16帧来自同一已反复使用的开发序列，不是16个独立场景。本轮不自动扩数据或新增网络，也不据图像分数声称真实几何、接触与材料点运动已正确。')

    r.page('模块契约和必要工程验收')
    r.p('输入是同批raw预测RGB、GT、原FG mask、原正则R、当前优化器参数及每图屏幕代理q。U为逐图全图L1再按batch平均，F为逐图前景和背景L1均值各半；空区域沿用既有处理。球谐SH用基函数表达随观察方向变化的颜色。')
    r.table([['模块接口','实际作用'],['off 关闭','原U+R单次backward，保留B-U语义'],['appearance_only 开启','规范SH接F；几何、opacity、形变及grid接U+R'],['q增密','只消费显式返回的uniform副本，保持批内有符号求和'],['共同日程','先density/prune后Adam；原最后一轮不step']],[2.1,4.89],10)
    r.p('同次前向后分别对指定参数求向量雅可比积VJP，即将损失导数反传到指定输入。没有整体detach SH分支，保留观察方向对几何的合法导数；两次VJP之间没有参数更新。这是分组更新规则，不声称存在一个共同标量loss产生全部偏导。')
    r.table([['必要验收','实际状态'],['完整参数覆盖 含固定aabb','pass'],['同次G A q来源与None 形状 dtype','pass'],['六次固定数值比较','accepted numerical variation'],['同保存梯度的Adam与增密消费','pass'],['128轮跨拓扑与24轮加载恢复','pass  后续轨迹差异另列']],[4.6,2.39],9.5)
    r.p(f"V7事前数值容限为张量相对L2及归一化最大差各≤1e−3，这是工程选择。诊断总计{acc['diagnostic_attempts']}次真实Adam调用，包含四个局部消费副本；V6原失败保持。恢复加载与后24批ID精确，但后续形变参数最大相对L2差{var['deformation_parameter_max_relative_L2']:.5f}，不能宣称轨迹近似或逐位相同；U/F批损失最大差{var['uniform_loss_max_difference']:.3g}/{var['balanced_loss_max_difference']:.3g}。")

    for split,title in [('retained','开发十六帧完整主结果'),('train','训练二百六十八帧拟合结果'),('retained_without_00000','去除外推帧后的十五帧补充')]:
        r.page(title)
        r.p('全部方法使用相同RGB、相机、mask和评价版本。以下主列为逐帧dB或指标的宏平均；池化PSNR先合并像素MSE再转dB，不能与宏平均混用。')
        tab=[['区域','设置','PSNR','SSIM','LPIPS','池化PSNR']]
        for reg,label in REGIONS:
            for b in ['B_U','B_F','M1']:
                x=ev['summaries'][b][split][reg];tab.append([label,b,*[num(x[k],6) for k in KEYS],num(x['pooled_psnr_db'],6)])
        r.table(tab,[.7,.85,1.36,1.36,1.36,1.36],9)
        tab=[['区域','M1−B-U PSNR','SSIM差','LPIPS差','M1 raw PSNR']]
        for reg,label in REGIONS:
            d=ev['paired']['M1-B_U'][split][reg];x=ev['summaries']['M1'][split][reg]
            tab.append([label,*[num(d[k]['mean'],6,True) for k in KEYS],num(x['raw_psnr_db'],6)])
        r.table(tab,[.75,1.56,1.56,1.56,1.56],9)
        r.p('主指标先把raw RGB裁剪到[0,1]；raw PSNR另列。SSIM为7×7窗口、data_range=1、RGB通道均值；LPIPS为AlexNet 0.1 spatial，输入[0,1]经normalize=True映射到[−1,1]，再按完整区域聚合。边界感受野保留，区域LPIPS不能直接当其他论文的标准全图指标。')
        r.p('发布FG可能排除静放背包，不能将它视为完整的人体加物体实例标注。独立人体/物体分数为NA。相机来自发布的全视频预处理，原始SfM输入范围未知，不声称没有开发信息。')

    for b in ['B_U','B_F']:
        r.page('相对'+b.replace('_',' ')+'的开发逐帧配对')
        r.p('以下差值均为M1减参照。胜负按指标方向判断；持平仅使用事前六位小数报告精度。均值和中位并列，避免只报少数改善帧。')
        tab=[['区域','指标','均值差','中位差','胜 平 负']]
        for reg,label in REGIONS:
            for k,labelk in zip(KEYS,['PSNR','SSIM','LPIPS']):
                d=ev['paired']['M1-'+b]['retained'][reg][k]
                tab.append([label,labelk,num(d['mean'],6,True),num(d['median'],6,True),f"{d['win']}  {d['tie']}  {d['loss']}"])
        r.table(tab,[.8,1.,1.73,1.73,1.73],9.5)
        r.p('非支配只表示这个三维向量中有得有失，不自动成为整个候选集合的最优方案；需同时检查另一个参照。不得以任意两项改善掩盖第三项代价。统计单位仍是同一序列的相关帧，不作跨场景显著性主张。')

    figs=read(RUN/'figure_manifest.json')['hos']
    for start in range(0,16,4):
        ids=ex['hos_retained_frame_ids'][start:start+4]
        r.page('开发固定全图 '+str(start//4+1));r.p('每行从左至右为GT、B-U、B-F、M1。全部开发帧按原固定顺序显示，均由完整日程终态生成；展示clip到[0,1]。','Caption')
        items=[x for fid in ids for x in figs if x['split']=='retained' and x['frame_id']==fid and x['kind']=='full'];r.image(r.sheet('dev_full_'+str(start),items,1,True),'完整开发图 '+', '.join(ids)+'；分数来自原分辨率，不从缩略图计算。',max_height=7.7)
        r.page('开发原固定裁剪 '+str(start//4+1));r.p('每个图组从左至右为GT、B-U、B-F、M1。沿用原相同像素窗口；区域指标仍由完整mask计算，未按结果改裁剪。','Caption')
        items=[x for fid in ids for x in figs if x['split']=='retained' and x['frame_id']==fid and x['kind']=='foreground_crop'];r.image(r.sheet('dev_crop_'+str(start),items,2,True),'固定裁剪 '+', '.join(ids)+'。',max_height=7.6)
    for start in range(0,8,4):
        ids=ex['training_frame_ids'][start:start+4];r.page('原固定训练图 '+str(start//4+1));r.p('每行GT、B-U、B-F、M1；以下只反映输入拟合，不能替代开发或独立序列证据。','Caption')
        items=[x for fid in ids for x in figs if x['split']=='train' and x['frame_id']==fid and x['kind']=='full'];r.image(r.sheet('train_'+str(start),items,1,True),'固定训练帧 '+', '.join(ids)+'。',max_height=7.7)

    r.page('原因分级与下一决定')
    r.p('已确认的工程事实：关闭模式保留原uniform单backward，开启模式满足A/G/q直接来源契约；旧输入和渲染核验复用，尺度保护与对照相同。本轮没有混入新损失、相机、深度、track或训练先验。')
    r.p('与设计相容的机制：限制前景平衡监督直接更新的属性范围，可减少该额外压力直接改变几何、透明度与增密的通道。但SH会影响下一次残差，几何仍可间接改变；图像指标差异不能唯一分解到几何、纹理或遮挡。')
    visual=read(RUN/'protocol/visual_review.json') if (RUN/'protocol/visual_review.json').exists() else {'findings':['固定图视觉审阅尚未写入，交付前须补齐。']}
    for p in visual['findings']:r.p(p)
    r.p('明确未验证：独立序列泛化、真实材料点对应、几何/运动/接触准确性。V6贡献只是模型行为，不是真实表面对应；M1的N_eff本轮按协议未重复计算。三项图像主指标之间的取舍如实保留。')
    r.p('阶段决定：'+LABELS[q['decision']]+'。本轮不自动运行独立确认或时序证据模块，不扫描权重来翻转结论。若后续投入，需固定新序列、同协议参照和预算。')

    r.page('真实成本与交付边界')
    tab=[['项目','实际','上限或口径'],['诊断真实尝试及Adam',cost['D_attempts'],'160'],['正式真实尝试',cost['C_actual_attempts'],'14256'],['正式Adam调用',cost['C_optimizer_calls'],'末轮不step'],['正式尾段重放',cost['C_replay_attempts'],'只限一次外部中断'],['附加无优化render backward',f"{cost['diagnostic_renders']}  {cost['diagnostic_backwards']}",'各64'],['GPU任务小时',num(cost['GPU_task_hours'],6),'3'],['终态点数',cost['final_points'],'代价 不是veto'],['峰值allocated GiB',num(cost['peak_allocated_bytes']/2**30),'PyTorch分配器']]
    r.table(tab,[3.0,1.9,2.09],9.5)
    r.p('GPU计时为各任务进程墙钟，包含加载、诊断、保存、训练、终态渲染与失败开销；CPU指标耗时单列，不冒称CUDA核计时。未连续采集nvidia-smi进程显存峰值。模型体积、检查点身份与历史复用成本在机器表中。')
    verify=read(RUN/'protocol/independent_verification.json')
    r.p(f"独立CPU核验覆盖{verify['metric_rows']}指标行、{verify['paired_rows']}配对行、新M1的{verify['M1_PSNR_rows_recomputed']}区域PSNR复算及预算；PSNR最大差{verify['max_PSNR_difference']:.3g}。未以第二套实现复算全部SSIM/LPIPS。完整模型、全图raw与完整梯度留训练机，反馈包提供索引和固定float32窗口，不称自包含完整重训。")
    r.p('图片内嵌DOCX，采用bundled LibreOffice本机渲染和逐页检查，不宣称已在另一主机Word逐项验收；不另交PDF。自产代码和源配置按白名单同步，输入/指标/图片/模型/日志/文档保持私有。历史V5上游快照及AGENTS摘要的公开例外继续披露，Git历史不擅自改写。')
    r.p('10月1日前按实测结果作路线决定，10月7日主问题收敛；11月4日核心方法和实验冻结，11月5—15日保留连续11个完整自然日写作。')
    path=RUN/'output/V7_appearance_router_results.docx';r.doc.save(path)
    with zipfile.ZipFile(path) as z:
        from lxml import etree as E
        media=[n for n in z.namelist() if n.startswith('word/media/')];external=[]
        for n in z.namelist():
            if n.endswith('.rels'):external.extend(dict(x.attrib) for x in E.fromstring(z.read(n)) if x.get('Type','').endswith('/image') and x.get('TargetMode')=='External')
        assert media and not external
    (RUN/'protocol/document_build.json').write_text(json.dumps(dict(path=str(path),embedded_images=len(media),external_image_relationships=external),indent=2)+'\n');print(path)
if __name__=='__main__':run()
