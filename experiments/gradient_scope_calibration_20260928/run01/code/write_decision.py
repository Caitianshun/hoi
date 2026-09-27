"""Private evidence-grounded closeout text; measured values read from JSON."""
from common import *
def run():
    route=read(RUN/'routing_equivalence.json');bad=[x for x in route['rows'] if not x['passed']];cost=read(RUN/'costs.json');loc=read(RUN/'diagnostics/gradient_localization/localization.json')
    maxabs=max(x['difference']['max_abs'] for x in bad);maxrel=max(x['difference']['relative_L2'] for x in bad)
    cause=f'''# V6 原因分级与证据边界

本轮在 D2 必需的梯度等价性检查处关闭，没有启动 C，质量门槛 NA。不是训练数值爆炸，也不是 C 质量失败。

## 已证事实

- D0 的 284 帧 stem、原图/mask、K/w2c、尺寸、时间核查通过；568 个输入哈希与 V5 冻结索引重新比较。camera_time_audit.json、protocol/input_hash_verification.json。
- 两冻结模型各两帧的实际训练、原评价、新接口在 RGB/depth/radii 和最终属性上与重复性基准一致。renderer_parity.json。全视频相机来源保留，原 SfM 输入范围未知，不能称无开发信息。
- C_route 的全部分组导数条目通过，但 uniform_all 有 {len(bad)} 个 grid 参数条目超出本轮预先声明规则；最大绝对差 {maxabs:.12g}，最大相对 L2 差 {maxrel:.12g}。全部有限；A/G/q 的 None 分类及 q 显式返回机制已检查。routing_equivalence.json。
- 后续交错重复只读检查的模型参数哈希不变，追加比较全部处于原容差内；这是实际差异会随 CUDA 调度变化的证据，但不能选择性删除首次未通过。diagnostics/gradient_localization/localization.json。首次超限仅保存逐参数统计，未导出当次完整梯度；后续定位的完整真实梯度留本机并索引，包内有原精度抽样，不能冒充首现场。
- D1 两冻结终态各八训练帧贡献/时间相机诊断完成。贡献接口绕开 SH，只对预计算颜色求导，几何和遮挡固定，radii/几何不变。独立原始 float32 背景差可能轻微为负；重复反向及精度范围核对后保留有符号值，不截零。contribution_manifest.json。

## 相容解释与不能下的结论

PyTorch 文档说明 CUDA grid_sample 反向可能非确定。当前差异集中在网格、量级很小、重复检查会变化，与浮点并行累加解释相容；未对每一个 CUDA 累加算子作确定性替换，故不能称已定位为唯一原因。原地址：https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.grid_sample.html 。

两次参考重复不足以描述完整误差分布，是本次门槛设计的局限。不能事后扩大阈值或只采用有利重复来宣布通过。未执行的一步 Adam、密度缓冲、128轮回归与恢复检查均为 NA，已写出的训练入口不能称经过完整验收。

D1 的有效时间数仅是八帧中贡献分散程度；不同模型行号不能对应。贡献少不等于真实物理表面不可见，贡献跨时刻也不证明同一材料点。缺独立对应、几何和实例标签，physical_surface_correspondence 为 NA。

## 成本与工程事件

全部 GPU 任务 {cost['GPU_task_seconds']:.9f} 秒，{cost['diagnostic_renders']} 次附加 rasterizer 前向、{cost['diagnostic_backwards']} 次无优化 rasterizer 反向，诊断/正式优化均为0。包括贡献首次绝对容差检查、重复检查以及 JSON bool_ 序列化失败；没有把失败成本抹除。贡献检查初版的固定绝对阈值不适配 CUDA 累加，新增重复量化后改为逐点重复误差加 float32 精度范围；这是诊断接口修正，未修训练或改变 D2 冻结门槛。源和错误见 logs、code、protocol/kernel_calls.jsonl。
'''
    (RUN/'CAUSE_ASSESSMENT.md').write_text(cause)
    decision='''# V6 阶段决定

**以 D2 必需前置验证未建立收口。C_route 未启动，所有新终态质量与晋级门槛 NA。**

D0通过；D1完成。分组求导本身已有局部正证据，但 uniform_all 首次等价检查超限，后续重复不足以撤销。保持预声明容差，不用剩余预算自动重复直至成功，不追加属性组合、权重扫描或网络。没有 C 终态，不能选择第9节任何基于 C 质量的科学路线判决，不能据此宣布颜色路由充分或无效。

保留 V5 两终态、全部负面结果和本轮定位。若继续，唯一必要工程入口是另行明确一种事前固定、足够覆盖后端随机性的等价验收方案，并完成原要求的一步 Adam/密度/恢复/128轮验证；它不是新研究模块。本轮不自动续跑或扩大容差，也不把本轮闲置预算挪作下一轮。

研究层面没有新几何真值。D1只描述模型的像素解释，不替代独立轨迹或表面对应。V5关于50/50监督不升级的结论保留；C的科学问题保持未回答。

本轮已同日收口。最迟10月1日路线决定、10月7日主问题收敛、11月4日核心冻结、11月5—15连续11个完整自然日写作窗口保持。
'''
    (RUN/'NEXT_DECISION.md').write_text(decision)
    (RUN/'MISSING_ASSETS.md').write_text('''# 资产与未完成验证

本机必需资产均已找到并核验：shared fine0、B-U/B-F终态、B-U fine1000、原284帧、相机与掩码。不存在用近似初值替代的情况。

反馈包不包含完整模型、全部源RGB/mask、全图raw渲染、环境与CUDA二进制、上游源码；state_manifest.json和environment.json提供路径、大小与SHA。真实完整梯度留在diagnostics/gradient_localization，各文件身份在localization.json；包内保留原float32抽样。C检查点不存在，因为D2未放行。

一步Adam对照、密度及恢复、128轮回归、C正式与C终态评价未执行。峰值显存未导出，NA。独立材料点对应、几何准确、接触评价及跨主机Word应用验收NA。不能称反馈ZIP自包含完整重训。
''')
    (RUN/'REPRODUCE.md').write_text('''# 复算入口与执行边界

## 无GPU只读核验

在原项目运行 `envs/4dgs/bin/python experiments/gradient_scope_calibration_20260928/run01/code/verify_saved_results.py`。核对输入/模型身份、贡献稀疏数组和有效时间数、D2首次门槛、NA行和预算。ZIP可用 `--zip /绝对路径/V6_feedback.zip` 检查CRC与全部成员哈希。

报告与图表由collect_results.py、plot_evidence.py、build_report_docx.py读取冻结文件生成。build_report_docx使用bundled Python；DOCX渲染使用bundled LibreOffice。重建报告不触发训练。不要重跑collect_results覆盖现有收口文件作为独立验证。

## 需要GPU且消耗新预算的复现

必须另立输出目录和预算。本轮已关闭，不将下列说明当自动续跑授权。verify_gradient_router.py展示同一B-U fine1000、固定00001/00041批次的原U/F单backward和C/uniform_all求导对照；localize_gradient_difference.py给出交错重复与参数哈希。完整梯度及抽样可先作CPU核验。CUDA非确定性可能使复现的差值改变；不得用一次有利重复覆盖首次记录。

train_routed_fine.py是尚未完成D2优化验收的候选入口。正式分支由formal_gate.json阻止启动，不能仅改passed字段来绕过验收。没有合法C终态，不存在C评价命令。

所有源码只公开自产薄适配及源配置；数据、模型、图像、诊断数组、报告与日志不上传。历史V5已披露的上游快照/AGENTS同步例外仍在Git历史，不宣称历史全私有。
''')
    print('Private conclusions saved')
if __name__=='__main__':run()
