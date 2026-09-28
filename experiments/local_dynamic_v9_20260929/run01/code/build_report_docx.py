"""Embedded V9 report from private completed evidence, bundled Python only.

Caller: run Documents create marker exactly once before first actual authoring.
Requires agent-written research_decision_review.json and visual_review.json.
This builder performs no training, inference or quality-based sample selection.
"""
from pathlib import Path
import json,importlib.util,zipfile,hashlib
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
sp=importlib.util.spec_from_file_location('v9_doc_helpers',ROOT/'experiments/foreground_stage_calibration_20260927/run01/code/build_report_docx.py')
h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
read=h.read;num=h.num
KEYS=['psnr_db','ssim','lpips_spatial_mean']
REGIONS=[('full','全图'),('foreground','前景'),('background','背景')]
ARMS=['U','Q','Q0','S','SL'];PAIRS=['S-Q0','SL-S','SL-Q0']


class Report(h.Report):
    def __init__(self,cost):
        h.v3.Report.__init__(self,RUN,{},cost)
        self.doc.core_properties.title='静态核心与局部时空输入完整对照结果'
        self.doc.core_properties.subject='V9 逐场景重建质量与资源取舍'
        for x in self.doc.sections[0].footer.paragraphs[0].runs:
            x.text=x.text.replace('V3 基线校准','V9 完整对照')


def vector_table(r,ev,split,regions=REGIONS):
    rows=[['区域','设置','PSNR dB ↑','SSIM ↑','LPIPS ↓']]
    for region,label in regions:
        for arm in ARMS:
            v=ev['summaries'][arm][split][region]
            rows.append([label,arm,*[num(v[k],6) for k in KEYS]])
    r.table(rows,[.8,.7,1.83,1.83,1.83],10)


def run():
    ev=read(RUN/'evaluation_summary.json');cost=read(RUN/'costs.json')
    review=read(RUN/'protocol/research_decision_review.json');visual=read(RUN/'protocol/visual_review.json')
    verification=read(RUN/'protocol/independent_verification.json');accept=read(RUN/'protocol/module_acceptance.json')
    assert review['status']==visual['status']=='completed' and verification['status']=='passed'
    r=Report(cost)
    r.p('静态核心与局部时空输入完整对照结果','Title')
    r.p('V9  Backpack 和 Tennis  逐场景优化','Subtitle')
    for p in review['summary_paragraphs']:r.p(p)
    r.p('两场景各自从训练数据建立Q0、S、SL，共六个终态；统一完成后再评价。主表为16帧开发保留集的逐帧宏平均，另给完整训练集与排除00000后的15帧。PSNR和SSIM越高越好，LPIPS越低越好，不把不同单位相加。')
    tab=[['场景','设置','全图 PSNR','全图 SSIM','全图 LPIPS']]
    for scene,s in ev['scenes'].items():
        for arm in ARMS:
            v=s['summaries'][arm]['retained']['full'];tab.append([scene,arm,*[num(v[k],6) for k in KEYS]])
    r.table(tab,[1.15,.7,1.71,1.71,1.72],10)
    r.p('U和Q为历史终态。它们与新Q0的差异包含背景预热、可变种子保留、零残差初态及更长预算；只有S−Q0与SL−S分别回答本轮两个模块的增量。')

    r.page('表示分工要解决的问题')
    r.p('历史Q改善了大面积背景的图像质量，但人物、携带物和球拍等内容在输入训练图上仍可能缺失。待验证的解释是：全场静态背景占用形变表示，而跨时刻前景点并集在静态预热中可能被削弱。该解释是设计动机，不能预先认定为唯一原因。')
    r.p('新共同初态先对完整100k来源点计算一次高斯属性，保留原90k背景和10k前景标签。以训练前景点的空间分位盒为依据，从背景选出距离最近的固定5000点作为可变容量余量；平距使用独立种子与原索引的哈希排序。原标签与角色分别保存，这些余量点不是新观测，也不保证包含漏标物体。')
    r.p('背景预热只渲染和优化静态核心。训练mask按8-bit阈值128二值化，以7×7方形核膨胀，图外为零；其补集定义合法背景像素。目标是在这些像素上的RGB L1平均值，分母为有效像素数乘3，前景RGB没有被替换成黑色或白色。')
    r.p('3000次预热完成后，将背景完整终态与未动可变种子拼接，包括位置、全部球谐颜色系数、log-scale、旋转及logit-opacity。可变种子不参与预热透明度学习或剪枝。共同fine0继承背景的球谐阶数，重置Adam与密度统计，并将位置、尺度、旋转的残差头置零。')
    r.p('Q0让所有点进入原全场形变网络。S只让可变点查询该网络；静态核心位置、形状、透明度和颜色仍由全图RGB更新，只是不随时间变化。两类点在世界坐标中合并后通过一次raster完成深度排序和透明度合成，未分别贴图。clone和split的子点继承父点角色，prune同步删除角色，禁止按运动量重新分类。')
    r.p('例如人物经过静置背包时，静态核心可以专门拟合场地，而可变点保留解释人体和候选背包区域的能力。若真正移动的物体错放入静态核心，S仍可能失败；它是粗来源先验下的容量分配，不是已知真实动静分割。')

    r.page('局部输入如何连接到形变网络')
    r.p('SL在S的可变支中冻结一个训练来源局部域。c为可变种子逐轴1%和99%分位的中心；r为分位跨度的0.625倍，并以场景extent的0.001倍设置最小值。对规范世界位置x计算z = (x−c)/r，网格查询使用u = −tanh(z)，不再次通过全场AABB归一化。')
    r.p('模型时间t仍由原帧号产生，只在SL内部变成τ = 2t−1。48维HexPlane特征再拼接三维softsign(z)及τ，共52维送入第一层。该层从48→128扩为52→128，额外输入列初值为零，其余参数来自共同初态。位置、尺度、相机及形变输出的世界单位均未改变。')
    r.p('局部空间、完整时间查询与低频坐标直连共同构成本轮L模块；SL−S检验这整个组合，不分别证明三个因素的独立作用。tanh压缩离群点，不能创造缺失几何或修复所有错误三角化；单规范云对长时大变形的限制也仍存在。')
    for scene in ev['scenes']:
        init=read(RUN/'scenes'/scene/'protocol/initialization.json');domain=init['local_domain']
        r.p(scene,'Heading 2')
        r.p('局部中心c为 '+', '.join(num(x,6) for x in domain['center'])+'；半径r为 '+', '.join(num(x,6) for x in domain['radius'])+'。')
        table=[['分辨率','全场各轴近似格距','局部中心各轴近似格距']]
        for res in ['64','128','256']:
            table.append([res,', '.join(num(x,4) for x in init['global_grid_spacing'][res]),', '.join(num(x,4) for x in domain['center_approx_grid_spacing'][res])])
        r.table(table,[1.0,3.0,2.99],9)
        r.p('种子的各轴饱和比例为 '+', '.join(num(x,6) for x in domain['saturation_fraction_per_axis'])+'，这里饱和指abs(tanh(z))≥0.99。格距仅描述局部中心附近，非全空间恒定体素大小。','Caption')

    for scene,s in ev['scenes'].items():
        for split,label in [('retained','开发16帧'),('train','完整训练集'),('retained_without_00000','补充15帧')]:
            r.page(scene+' '+label+' 完整质量向量')
            vector_table(r,s,split)
            if split=='retained':
                for p in review['scene_interpretation'][scene]:r.p(p)
            r.p('前景沿用发布合并mask，可能漏掉静放交互物。没有可靠独立人和物mask，相应指标为NA；本表不能证明接触、真实深度或材料点运动正确。')
        r.page(scene+' 三组核心配对差')
        tab=[['比较','区域','指标','均值差','中位差','胜 平 负']]
        for pair in PAIRS:
            for region,label in REGIONS:
                for k,kl in zip(KEYS,['PSNR','SSIM','LPIPS']):
                    d=s['paired'][pair]['retained'][region][k]
                    tab.append([pair,label,kl,num(d['mean'],6,True),num(d['median'],6,True),f"{d['win']} {d['tie']} {d['loss']}"])
        r.table(tab,[.85,.65,.65,1.55,1.55,1.74],9)
        r.p('LPIPS负差表示改善。胜平负已统一为有利方向；同视频16帧不是16个独立场景。全部训练、15帧补充、历史U/Q配对与连续时间块在CSV/JSON中保留。')

    r.page('两个场景等权汇总')
    tab=[['区域','设置','PSNR dB','SSIM','LPIPS']]
    for region,label in REGIONS:
        for arm in ARMS:
            x=ev['equal_scene_means'][arm]['retained'][region];tab.append([label,arm,*[num(x[k],6) for k in KEYS]])
    r.table(tab,[.8,.7,1.83,1.83,1.83],10)
    r.p('先分别计算每场景逐帧宏平均，再将两个场景等权平均，不按视频帧数或总像素数加权。场景级配置选择允许存在，但须报告原始三臂，不按单帧拼接最好结果。')
    for p in review['selection_paragraphs']:r.p(p)

    r.page('工程验收与实际状态恢复')
    r.p(f"一次集成验收累计{accept['diagnostic_Adam_updates']}次实际Adam，额外无更新反向{accept['extra_no_update_backwards']}次；检查中的全部失败和已用成本保留。三臂零形变图像及世界属性一致，静态RGB梯度、局部网格和新增输入列路径，以及clone、split、prune角色继承已检查。")
    for p in review['engineering_incidents']:r.p(p)
    r.p('完整检查点保存模型和网络、角色、局部c/r、全场AABB、场景尺度、Adam、LR位置、球谐阶数、密度缓冲、完成更新数、RNG和RGB剩余栈。恢复的起始张量和状态精确相同，后续CUDA优化尾段的差异另存，未把尾段逐位相等设为本轮研究晋级要求。')
    r.p('正式fine每轮都执行Adam，包括最后一轮；fine30000是完成30000次更新后的终态。fine14000中间快照仅留档，没有用开发评价挑选终态。共享背景前缀的实际资源只计一次，同时报告每条逻辑路径包含前缀的成本。')
    tab=[['场景','背景终态点数','保存的可变种子','共同fine0点数','种子属性一致']]
    for scene in ev['scenes']:
        x=read(RUN/'scenes'/scene/'protocol/fine_transition.json')
        tab.append([scene,x['background_points'],x['variable_points'],x['fine0_points'],str(x['variable_seed_identity_before']==x['variable_seed_identity_after'])])
    r.table(tab,[1.15,1.65,1.55,1.45,1.19],9)

    r.page('预算 点数与复算范围')
    v=cost['verification']
    r.p(f"正式有效更新{v['formal_updates']}，尝试{v['formal_attempts']}，重放{v['replay_attempts']}。GPU任务进程累计{num(cost['GPU_task_hours'],6)}小时，包含加载、保存、失败和最终渲染，不是CUDA核计时；CPU任务累计{num(cost['CPU_job_seconds'],3)}秒。仅使用物理GPU1 RTX3090。")
    tab=[['场景','设置','终态点数','可变点数','进程秒','峰值分配 GiB']]
    for b in cost['branches']:
        tab.append([b['scene'],b['mode'],b['final_points'],b['variable_points'],num(b['seconds'],2),num(b['peak_allocated_bytes']/(1<<30),3)])
    r.table(tab,[1.05,.65,1.25,1.25,1.4,1.39],9)
    r.p(f"独立核验覆盖{verification['rows']}指标行与{verification['paired_rows']}配对行，重算{verification['PSNR_recomputed']}个新PSNR，最大差{verification['max_PSNR_difference']:.3g}。SSIM和LPIPS沿用原评价执行，独立核验其聚合与配对，未宣称第二实现重算。")
    r.p('训练目标为raw RGB上的0.8 L1加0.2乘以1−SSIM11，原plane/time正则不变。评价先裁剪RGB到[0,1]，PSNR为逐帧dB宏平均；SSIM用7×7非高斯窗口和样本协方差；LPIPS用AlexNet 0.1空间图、normalize=True并按完整区域平均。pooled与raw为补充值。')
    r.p('反馈ZIP包含源码、配置、逐帧指标、配对差、固定图、状态与身份索引。完整模型、原始RGB、全部浮点渲染和大缓存留在训练机，路径、大小和SHA256见model_index及输入清单；该ZIP不是自包含重训数据包。')

    # All 80 frozen panels, same order and crops as V8; full resolution metric
    # sources remain in the feedback package and local model/render indices.
    for scene in ev['scenes']:
        figs=read(RUN/'scenes'/scene/'figure_manifest.json')['figures']
        retained=sorted({x['frame_id'] for x in figs if x['split']=='retained'})
        train=sorted({x['frame_id'] for x in figs if x['split']=='train'})
        for start in range(0,len(retained),4):
            ids=retained[start:start+4]
            for kind,label,cols in [('full','开发固定全图',1),('foreground_crop','开发固定前景裁剪',2)]:
                r.page(scene+' '+label+' '+str(start//4+1))
                r.p('每组从左至右为GT、历史Q、Q0、S、SL。全部固定帧完整展示，未按输出误差重选样例或窗口。','Caption')
                items=[x for fid in ids for x in figs if x['split']=='retained' and x['frame_id']==fid and x['kind']==kind]
                r.image(r.sheet(scene+'_'+kind+'_'+str(start),items,cols,True),'固定帧 '+', '.join(ids)+'；图像仅作统一显示裁剪，指标按原分辨率计算。',max_height=7.7)
        for start in range(0,len(train),4):
            ids=train[start:start+4];r.page(scene+' 原固定训练图 '+str(start//4+1))
            r.p('从左至右为GT、历史Q、Q0、S、SL。输入拟合用于判断可见训练观测是否得到重建，不能替代开发集结果。','Caption')
            items=[x for fid in ids for x in figs if x['split']=='train' and x['frame_id']==fid and x['kind']=='full']
            r.image(r.sheet(scene+'_train_'+str(start),items,1,True),'固定训练帧 '+', '.join(ids)+'。',max_height=7.7)

    r.page('保留配置 适用边界与下一步')
    for p in visual['findings']:r.p(p)
    for p in review['limitations']:r.p(p)
    for p in review['next_step']:r.p(p)
    r.p('11月4日前冻结核心证据，11月5日至15日保留连续11个完整自然日用于写作、图表和审阅。后续扩展只围绕实测有效的配置；新的表示替代需要另立问题、对照和预算，本轮未自动授权权重细扫或新强先验。')
    path=RUN/'output/V9_static_local_results.docx';r.doc.save(path)
    with zipfile.ZipFile(path) as z:
        media=[n for n in z.namelist() if n.startswith('word/media/')]
        external=[n for n in z.namelist() if n.endswith('.rels') and b'TargetMode="External"' in z.read(n)]
        assert media and not external
    meta=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              embedded_images=len(media),external_relationship_files=external,status='authored_pending_render_QA')
    (RUN/'protocol/document_authored.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))


if __name__=='__main__':run()
