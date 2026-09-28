"""Portable V8 report from completed result files, using bundled docx runtime.

Run only after research_decision_review.json and visual_review.json are complete.
The caller must run the Documents skill create marker exactly once before the
first actual authoring invocation. This file does not train or evaluate models.
"""
from pathlib import Path
import json,importlib.util,zipfile
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
sp=importlib.util.spec_from_file_location('report_helpers',ROOT/'experiments/foreground_stage_calibration_20260927/run01/code/build_report_docx.py');h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
read=h.read;num=h.num
ARMS=['B_U','B_Q','T_plain','T_mix'];LABELS={'B_U':'B U','B_Q':'B Q','T_plain':'T plain','T_mix':'T mix'}
REGIONS=[('full','全图'),('foreground','前景'),('background','背景')];KEYS=['psnr_db','ssim','lpips_spatial_mean']
PAIRS=['B_Q-B_U','T_plain-B_Q','T_mix-T_plain','T_mix-B_Q','T_plain-B_U','T_mix-B_U']
class Report(h.Report):
    def __init__(self,cost):
        h.v3.Report.__init__(self,RUN,{},cost)
        self.doc.core_properties.title='时序证据模块完整对照结果'
        self.doc.core_properties.subject='V8 图像目标与投影运动监督的质量取舍'
        for x in self.doc.sections[0].footer.paragraphs[0].runs:
            if 'V3' in x.text:x.text=x.text.replace('V3 基线校准','V8 时序证据模块')

def run():
    ev=read(RUN/'evaluation_summary.json');decision=read(RUN/'quality_decision.json');review=read(RUN/'protocol/research_decision_review.json');visual=read(RUN/'protocol/visual_review.json');cost=read(RUN/'costs.json');acc=read(RUN/'module_acceptance.json');cal=read(RUN/'lambda_calibration.json');cache=read(RUN/'track_cache_manifest.json');ex=read(RUN/'protocol/fixed_examples.json');verify=read(RUN/'protocol/independent_verification.json');temporal=read(RUN/'temporal_summary.json')
    assert decision['status']=='completed' and review['status']=='completed' and visual['status']=='completed' and verify['status']=='pass'
    r=Report(cost);r.p('时序证据模块完整对照结果','Title');r.p('V8  标准图像目标 普通时序监督 混合运动分歧加权','Subtitle')
    r.p(review['summary']);r.p('主表为Backpack同一16帧开发集的逐帧宏平均。PSNR与SSIM越高越好，LPIPS越低越好；不能把三种单位相加作为总分。B U复用旧终态，三条新分支全部完成后统一冻结、评价。')
    tab=[['区域','设置','PSNR dB ↑','SSIM ↑','LPIPS ↓']]
    for reg,label in REGIONS[:2]:
        for arm in ARMS:
            x=ev['summaries'][arm]['retained'][reg];tab.append([label,LABELS[arm],*[num(x[k],6) for k in KEYS]])
    r.table(tab,[.8,1.1,1.7,1.7,1.69],9.5)
    for text in review['increment_summary']:r.p(text)
    r.p(f"本轮新增正式{cost['formal_actual_attempts']}次尝试、{cost['formal_Adam']}次Adam更新；三臂共享前1000轮但各自逻辑fine日程均为14000轮。V8A GPU任务进程墙钟{cost['GPU_task_hours']:.4f}小时，诊断{cost['diagnostic_Adam']}次Adam。")
    r.p('16帧来自一条已反复使用的开发序列，不是16个独立场景。前景掩码可能漏掉静放背包；图像改善本身不能证明材料点、深度或接触几何正确。')

    for split,title in [('retained','开发十六帧完整结果'),('train','训练二百六十八帧拟合结果'),('retained_without_00000','去除外推帧后的十五帧补充')]:
        r.page(title);r.p('主列PSNR为逐帧dB宏平均。池化PSNR先合并像素平方误差再转dB；raw PSNR在未裁剪预测上计算，仅作补充。训练输入拟合不能替代开发或独立序列证据。')
        tab=[['区域','设置','PSNR','SSIM','LPIPS','池化PSNR','raw PSNR']]
        for reg,label in REGIONS:
            for arm in ARMS:
                x=ev['summaries'][arm][split][reg];tab.append([label,LABELS[arm],*[num(x[k],6) for k in KEYS],num(x['pooled_psnr_db'],6),num(x['raw_psnr_db'],6)])
        r.table(tab,[.65,.9,1.09,1.02,1.05,1.12,1.16],8.5)
        r.p('主评价保持历史实现：RGB裁剪到[0,1]；SSIM使用7×7非高斯窗口、样本协方差，先对RGB通道求均值再按完整区域聚合；LPIPS使用AlexNet 0.1 spatial，输入[0,1]由normalize=True映射到[−1,1]，边界感受野保留。区域LPIPS不等同于其他论文的全图排行榜。')
        r.p('训练Q中的SSIM另为原上游11×11高斯窗口、sigma 1.5、padding 5，在原始预测上对batch、通道和像素取均值。改变训练目标没有改变评价函数。没有可靠独立人体/物体mask，二者指标均为NA。发布相机的原始SfM输入范围未知，保留历史信息条件披露。')

    for offset in range(0,len(PAIRS),2):
        r.page('开发逐帧配对差 '+str(offset//2+1));r.p('差值均为候选减参照；LPIPS负差表示改善。胜平负使用预声明报告精度，不能解释为统计显著性。0.10dB、0.005 SSIM或LPIPS只辅助阅读微小变化，不是晋级硬阈值。')
        for pair in PAIRS[offset:offset+2]:
            r.p(pair.replace('_',' '),'Heading 2');tab=[['区域','指标','均值差','中位差','胜 平 负']]
            for reg,label in REGIONS:
                for k,labelk in zip(KEYS,['PSNR','SSIM','LPIPS']):
                    d=ev['paired'][pair]['retained'][reg][k];tab.append([label,labelk,num(d['mean'],6,True),num(d['median'],6,True),f"{d['win']}  {d['tie']}  {d['loss']}"])
            r.table(tab,[.8,1.,1.73,1.73,1.73],9)
        r.p('完整训练和15帧补充的配对均值、中位、胜平负，以及连续4帧时间块的方向，一并保存在evaluation_summary.json与paired_differences.csv。固定时间块仅作描述，不作为独立场景检验。')

    r.page('四种设置与监督如何传递')
    r.table([['设置','图像项','时序项','新增实际fine尝试'],['B U','原全图均匀L1','无','复用历史'],['B Q','0.8 L1 加0.2结构项','无','14000'],['T plain','同B Q','外部可见性及软cycle权重','13000'],['T mix','同B Q','仅再乘混合运动分歧gate','13000']],[1.0,2.2,2.59,1.2],9)
    r.p('问题是怎样把训练RGB给出的二维对应传递到统一动态高斯表示。比如同一个源像素可能同时混合人体边缘与后方背景：两个高斯在目标相机里的位移不同，源像素的平均位移未必对应单一材料点。普通时序监督直接约束这个平均位移；mixture只按混合位移的分歧重新分配监督权重。这个解释是动机，不能单凭模块名称当作遮挡正确性证明。')
    r.p('对同一组当前高斯，在源时间和目标时间分别查询形变位置，再用各自相机投影。将两个投影的像素位移除以宽高，作为允许正负值的颜色输入实际CUDA rasterizer。辅助渲染的源位置、尺度、旋转、透明度及screen张量detach，只让位移颜色保留位置梯度。先在轨迹源坐标采样累计一阶量和alpha，再相除获得预测平均位移；预测终点是源查询坐标加此位移。')
    r.p('T mix还以完全相同源alpha权重、无梯度地渲染位移平方。二阶矩减均值平方得到像素单位的混合方差v。尺度tau为4像素加教师位移长度的0.1倍；gate为0.2加0.8除以(1+v/tau平方)，停止梯度。方差描述同一源像素的双时刻位移混合，不是单高斯的跨时间方差，不是接触概率。真实边界、视差或非刚体运动也可产生高方差，一致但错误的运动可能低方差。')
    r.p('教师权重c由两端及返回可见、坐标合法和循环返回误差产生，采用3像素尺度的高斯软权重。先在固定教师候选上计算a为c均值，再仅在alpha大于0.01的模型支持处求c×gate加权鲁棒误差均值，最后乘a。按最终权重归一，避免常量gate只等同缩小lambda；a则保留整组教师可靠性低时应减弱约束的作用。')

    r.page('一次工程验收与训练数据尺度校准')
    r.p(f"训练专用缓存含{len(cache['centers'])}个17帧窗口、{len(cache['pairs'])}组配对；每个中心192个全图网格查询与192个固定前景分层查询。输入上下文在跟踪前就从268个训练帧中筛定，没有先运行全视频再删除开发结果。modeltime沿用原帧号归一化；time_seconds字段实际是名义帧号，不是物理秒。")
    r.p('实际渲染核通过正负平移、静态点加移动相机、无alpha支撑、同位移零方差与相反位移非零方差检查；验证先采样后归一、gate detach及关闭gate退化一致。真实flow反向可达规范xyz、位置形变头和共享grid/backbone，未直接连接专属SH、opacity、尺度、旋转参数。共享骨干可间接改变其他属性，不能声称它们始终不变。')
    r.p('112轮连续短回归跨过原1100密度事件，1096完整检查点另续16轮。加载模型、Adam、缓冲、RNG、剩余RGB相机栈以及后续RGB/pair顺序核对通过；不要求独立CUDA续算之后的累计轨迹逐位一致。诊断共128次Adam，后续不追加优化副本。辅助flow screen与RGB screen独立，增密只读取Q的屏幕梯度，仍先density后Adam。')
    r.p(f"B Q完成后，在其after-step Q1000固定前8组训练pair上计算Q与未乘a的plain单位项梯度范数。共同运动参数集合的范数中位数分别为{cal['median_Q']:.8g}与{cal['median_unit']:.8g}；0.5倍比值为{cal['unclipped_lambda']:.8g}，按事前[0.001,10]截断得到共同lambda={cal['lambda0']:.8g}。截断状态为{cal['clipped']}。没有看开发分数或单独为mixture校准。")
    r.p('fine前1000轮为共享无时序前缀；1001至2000线性升权，之后固定。每4个逻辑迭代调用一个固定顺序pair，不额外乘4。独立CPU Generator不消费RGB随机流，两T从同一Q1000完整状态分叉。alpha无支撑时跳过对应，不以零位移伪标签替代；非有限源几何属于技术错误。')

    figs=read(RUN/'figure_manifest.json')['hos']
    for start in range(0,16,4):
        ids=ex['hos_retained_frame_ids'][start:start+4];r.page('开发固定全图 '+str(start//4+1));r.p('每行从左至右GT、B U、B Q、T plain、T mix。全部开发图固定顺序、统一显示裁剪，无中间最优检查点。','Caption');items=[x for fid in ids for x in figs if x['split']=='retained' and x['frame_id']==fid and x['kind']=='full'];r.image(r.sheet('dev_full_'+str(start),items,1,True),'完整开发帧 '+', '.join(ids)+'；指标在原分辨率计算。',max_height=7.7)
        r.page('开发固定前景裁剪 '+str(start//4+1));r.p('每组从左至右GT、B U、B Q、T plain、T mix。沿用相同原固定像素窗口；区域分数仍来自完整mask。','Caption');items=[x for fid in ids for x in figs if x['split']=='retained' and x['frame_id']==fid and x['kind']=='foreground_crop'];r.image(r.sheet('dev_crop_'+str(start),items,2,True),'原固定裁剪 '+', '.join(ids)+'。',max_height=7.6)
    for start in range(0,8,4):
        ids=ex['training_frame_ids'][start:start+4];r.page('原固定训练图 '+str(start//4+1));r.p('每行GT、B U、B Q、T plain、T mix。输入拟合图不替代开发和独立确认。','Caption');items=[x for fid in ids for x in figs if x['split']=='train' and x['frame_id']==fid and x['kind']=='full'];r.image(r.sheet('train_'+str(start),items,1,True),'固定训练帧 '+', '.join(ids)+'。',max_height=7.7)
    explanations=read(RUN/'evaluation/temporal_explanations/manifest.json')
    for j in range(0,len(explanations['figures']),2):
        r.page('固定训练轨迹与运动混合解释 '+str(j//2+1));r.p('每图从上至下为B Q、T plain、T mix；从左至右为教师位移、预测位移、源有效alpha、log10(1+像素方差)、gate。三个模型都用mixture公式作终态解释，但只有T mix训练时使用gate。以下是同一冻结训练缓存的估计观测，不新增开发track教师。','Caption')
        for fig in explanations['figures'][j:j+2]:r.image(h.v3.asset(fig),'固定pair顺序中的训练观测；散点颜色反映查询位置，未覆盖的像素没有显示指标。',max_height=3.4)
    r.page('确认事实 相容解释和未知')
    for text in review['confirmed_facts']:r.p(text)
    for text in visual['findings']:r.p(text)
    for text in review['compatible_explanations']:r.p(text)
    for text in review['unknowns']:r.p(text)
    tab=[['分支','有效调用','空证据调用','未加权误差中位 px','gate均值中位']]
    for arm in ['T_plain','T_mix']:
        x=temporal[arm];tab.append([LABELS[arm],x['calls']-x['empty_calls'],x['empty_calls'],num(None if x['unweighted_error_px'] is None else x['unweighted_error_px']['p50'],3),num(None if x['gate_mean'] is None else x['gate_mean']['p50'],4)])
    r.table(tab,[1.1,1.1,1.1,2.1,1.59],9);r.p('这些轨迹误差、gate、alpha与点数用于解释，不是质量veto。各次调用时刻和模型不同，不能把过程统计当独立真实轨迹精度。')

    r.page('独立确认决定与研究边界')
    independent=read(RUN/'independent_confirmation.json') if (RUN/'independent_confirmation.json').exists() else {}
    for text in review['independent_confirmation']:
        if text not in independent.get('interpretation',[]):r.p(text)
    if (RUN/'independent_confirmation.json').exists():
        independent=read(RUN/'independent_confirmation.json')
        if independent.get('status')=='completed':
            other=read(independent['evaluation_summary_path']);r.p('以下Tennis结果单独报告，不与Backpack合并。')
            tab=[['区域','设置','PSNR','SSIM','LPIPS']]
            for reg,label in REGIONS[:2]:
                for arm,s in other['summaries'].items():
                    x=s['retained'][reg];tab.append([label,arm,*[num(x[k],6) for k in KEYS]])
            r.table(tab,[.8,1.1,1.7,1.7,1.69],9)
            for text in independent['interpretation']:r.p(text)
            other_cost=read(RUN/'tennis/costs.json');other_verify=read(RUN/'tennis/protocol/independent_verification.json');prep=read(RUN/'tennis/protocol/preprocessing.json')
            r.p(f"Tennis使用自身{prep['training_count']}个训练帧与{prep['development_count']}个保留帧，原尺寸{prep['resolution'][0]}乘{prep['resolution'][1]}；时间归一分母{prep['time_normalization']['denominator']}，训练输入确定的共同尺度上界{prep['scene_extent']:.8f}。训练点云来自相同固定SIFT双视图规则与90k背景加10k前景上限；不是发布人体拟合或开发RGB初始化。相机原SfM来源范围未知的限制保持。")
            for split,title in [('retained','独立序列开发完整结果'),('train','独立序列训练拟合结果'),('retained_without_00000','独立序列十五帧补充结果')]:
                r.page(title);tab=[['区域','设置','PSNR','SSIM','LPIPS','池化PSNR','raw PSNR']]
                for reg,label in REGIONS:
                    for arm,s in other['summaries'].items():
                        x=s[split][reg];tab.append([label,arm.replace('_',' '),*[num(x[k],6) for k in KEYS],num(x['pooled_psnr_db'],6),num(x['raw_psnr_db'],6)])
                r.table(tab,[.65,.9,1.09,1.02,1.05,1.12,1.16],8.5)
                tab=[['区域','指标','均值差','中位差','胜 平 负']]
                for reg,label in REGIONS:
                    for k,labelk in zip(KEYS,['PSNR','SSIM','LPIPS']):
                        d=other['paired']['B_Q-B_U'][split][reg][k];tab.append([label,labelk,num(d['mean'],6,True),num(d['median'],6,True),f"{d['win']}  {d['tie']}  {d['loss']}"])
                r.table(tab,[.8,1.,1.73,1.73,1.73],9)
                r.p('本页差值均为Tennis Q减U；LPIPS负数表示改善。独立确认采用同一原评价函数，两个终态冻结后才读取开发质量。完整时间块与原数值另存Tennis数字文件，不把逐帧胜率当独立场景显著性。')
            other_ex=read(RUN/'tennis/protocol/fixed_examples.json');other_figs=read(RUN/'tennis/figure_manifest.json')['figures']
            for start in range(0,16,4):
                ids=other_ex['development_frame_ids'][start:start+4]
                for kind,title,cols in [('full','独立序列开发全图',1),('foreground_crop','独立序列前景裁剪',2)]:
                    r.page(title+' '+str(start//4+1));r.p('每组左至右GT、B U、B Q。全部保留帧按事前时间顺序显示；裁剪只使用输入mask边框加24像素，不依预测选择。','Caption')
                    items=[x for fid in ids for x in other_figs if x['split']=='retained' and x['frame_id']==fid and x['kind']==kind]
                    r.image(r.sheet('tennis_'+kind+'_'+str(start),items,cols,True),'Tennis '+', '.join(ids)+'。',max_height=7.6)
            for start in range(0,8,4):
                ids=other_ex['training_frame_ids'][start:start+4];r.page('独立序列固定训练图 '+str(start//4+1))
                items=[x for fid in ids for x in other_figs if x['split']=='train' and x['frame_id']==fid and x['kind']=='full'];r.image(r.sheet('tennis_train_'+str(start),items,1,True),'GT、B U、B Q；固定训练帧 '+', '.join(ids)+'。',max_height=7.6)
            r.page('独立确认成本与核验');tab=[['阶段','新尝试','新Adam','终态点数','峰值allocated GiB']]
            for arm,x in other_cost['model_costs'].items():tab.append([arm.replace('_',' '),x['attempted_this_process'],x['updates_this_process'],x['final_points'],num(x['peak_allocated_bytes']/2**30,3)])
            r.table(tab,[1.1,1.2,1.2,1.69,1.8],9.5)
            r.p(f"Tennis实际新增{other_cost['formal_attempts']}次尝试、{other_cost['optimizer_updates']}次Adam。共享coarse实际执行一次3000轮且末轮仍更新；U/Q各fine14000轮、13999次更新。每个独立逻辑路径为17000次尝试、16999次更新。GPU任务进程墙钟{other_cost['GPU_task_wall_seconds']:.3f}秒；CPU评价{other_cost['CPU_evaluation_seconds']:.3f}秒；训练点云三角化CPU墙钟{prep['triangulation_CPU_wall_seconds']:.3f}秒。点云选样独立进程未计时，标NA，不能当零成本。")
            r.p(f"Tennis一次CPU核验覆盖{other_verify['rows']}指标行与{other_verify['paired_rows']}配对行，独立重算{other_verify['recomputed_PSNR']}项PSNR，最大差{other_verify['PSNR_max_absolute_error']:.3g}；SSIM/LPIPS仅独立核对聚合和差值。U/Q全部fine RGB批次完全相同，共同父状态与终态SHA通过核验。")
            r.p(f"V8A加V8B的GPU任务墙钟合计{cost['GPU_task_hours']+other_cost['GPU_task_hours']:.6f}小时；两阶段各自硬预算为8小时，没有借用另一阶段余量。")
    r.p(review['next_decision']);r.p('本轮最多检验具体时序监督配置对图像质量的作用及取舍。不提供人/物实例身份保证，不输出真实接触几何，不消除单目深度歧义，也没有证明完全遮挡区域恢复正确。已有相似机制不自动否定研究价值；已复用内容如实归属，后续贡献应由实际适配、监督传递改造和统一协议质量收益支撑。')
    r.p('10月7日前收敛主问题；11月4日前冻结核心方法与主要实验，11月5至15日保留连续11个完整自然日集中写作。当前独立确认即使成立，也不是完整多场景论文基准。')

    r.page('实际成本与可复算范围')
    tab=[['设置','新尝试','新Adam','终态点数','峰值allocated GiB']]
    for arm in ARMS[1:]:
        x=cost['runs'][arm];tab.append([LABELS[arm],x['attempted_this_process'],x['updates_this_process'],x['final_points'],num(x['peak_allocated_bytes']/2**30,3)])
    r.table(tab,[1.1,1.2,1.2,1.69,1.8],9.5)
    r.p(f"V8A累计GPU任务进程墙钟{cost['GPU_task_seconds']:.3f}秒，约{cost['GPU_task_hours']:.6f}小时，含缓存、验收、训练、加载/保存和渲染；CPU指标计算{cost['CPU_evaluation_seconds']:.3f}秒单列。额外无优化backward调用{cost['extra_no_update_backward_calls']}次。显存为PyTorch分配器峰值，不是全卡连续测量峰值。")
    r.p(f"三条新臂每条逻辑路径均含旧coarse3000与fine14000，共17000次尝试、16999次Adam；实际新计算通过共享Q前1000轮减少2000次重复。旧coarse3000日志时间字段为{num(cost['logical_cost_per_arm']['old_coarse_elapsed_seconds'],3)}秒，记录在末轮density/save之前，不是独立子进程墙钟；历史完整成本另由索引提供。模型体积、配置、输入身份、缓存与终态SHA见state_manifest和各run记录。")
    r.p(f"一次独立CPU核对覆盖{verify['metric_rows']}指标行、{verify['paired_rows']}配对行，重新计算{verify['new_PSNR_rows_recomputed']}个新区域PSNR，最大差{verify['max_PSNR_difference']:.3g}。全部SSIM/LPIPS只核对CSV聚合和差值，没有用第二套实现重算。")
    r.p('DOCX内嵌全部图像，本机使用bundled LibreOffice渲染并逐页检查，不声称跨主机Word应用实测；不另交PDF。反馈ZIP保留代码、源配置、冻结协议、数字、原精度窗口与固定图；完整模型、原图和raw输出仍在训练机，提供大小与SHA索引，不称自包含的完整重训包。自产代码和源配置按白名单同步，结果、数据、缓存、日志及文档不新增公开；旧公开历史例外保留披露。')
    path=RUN/'output/V8_temporal_evidence_results.docx';r.doc.save(path)
    from lxml import etree as E
    with zipfile.ZipFile(path) as z:
        media=[n for n in z.namelist() if n.startswith('word/media/')];external=[]
        for n in z.namelist():
            if n.endswith('.rels'):external.extend(dict(x.attrib) for x in E.fromstring(z.read(n)) if x.get('Type','').endswith('/image') and x.get('TargetMode')=='External')
        assert media and not external
    (RUN/'protocol/document_build.json').write_text(json.dumps(dict(path=str(path),embedded_images=len(media),external_image_relationships=external),indent=2)+'\n');print(path)
if __name__=='__main__':run()
