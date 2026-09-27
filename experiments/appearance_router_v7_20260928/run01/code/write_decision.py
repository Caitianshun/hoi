"""State conclusions from complete metric vectors, with explicit evidence limits."""
from pathlib import Path
import json
RUN=Path(__file__).resolve().parents[1]
def read(name):return json.loads((RUN/name).read_text())
LABELS={'retain_quality_candidate_for_future_independent_confirmation':'保留为开发序列上的质量候选，后续需独立序列确认',
 'close_router_no_quality_basis':'关闭当前路由方向，缺乏继续投入的质量依据',
 'retain_regional_tradeoff_only':'仅保留为区域取舍模式或消融，不替代统一全图基线',
 'retain_non_dominated_tradeoff_for_assessment':'保留为非支配的质量取舍，是否投入独立确认需结合代价决定',
 'existing_B_F_dominates_full_vector_assess_regional_use_only':'全图被已有B-F支配，仅评估是否具有区域用途'}
def run():
    q=read('quality_decision.json');ev=read('evaluation_summary.json');cost=read('costs.json');var=read('diagnostics/continuation_variation.json')
    decision=LABELS[q['decision']];full=ev['summaries']['M1']['retained']['full'];fg=ev['summaries']['M1']['retained']['foreground']
    vec=lambda x:f"PSNR {x['psnr_db']:.6f} dB、SSIM {x['ssim']:.6f}、LPIPS {x['lpips_spatial_mean']:.6f}"
    relation='；'.join(f"相对{b}全图{q['relations'][b]['full']['relation']}、前景{q['relations'][b]['foreground']['relation']}" for b in ['B_U','B_F'])
    (RUN/'NEXT_DECISION.md').write_text(f'''# V7 阶段决定

M1完整执行fine1→14000并完成268训练、16开发及15帧补充评价。开发全图：{vec(full)}；开发前景：{vec(fg)}。

**{decision}。** {relation}。关系由开发均值的完整三项向量分别计算，不把不同单位相加、不使用旧BG或几何代理veto。LPIPS候选减基线为负代表改善；持平使用事前六位小数报告精度，并非显著性检验。

本轮不追加权重、种子、数据或网络。后续独立序列确认或结构模块均属于新协议，未在本轮自动运行。当前只有一条已反复使用的开发序列，不能据16帧声称跨场景泛化或真实交互几何已正确。11月4日核心冻结与11月5—15日连续11天写作保持。
''')
    (RUN/'CAUSE_ASSESSMENT.md').write_text(f'''# V7 原因与证据边界

已确认：M1是可开关的参数组更新模块；关闭使用原U+R单backward，开启使G接U+R、canonical SH接F、增密接显式uniform q。完整参数覆盖和同次来源赋值严格核对；新数值比较接受有限CUDA波动，旧V6失败未改写。见module_acceptance.json及源码。

已确认：唯一正式日程完整结束；主指标为{vec(full)}，前景为{vec(fg)}。收益和代价必须逐项看quality_decision.json、evaluation_summary.json及全部固定图，不将区域LPIPS改善自动解释成几何改善。

机制假设：限制前景平衡误差的直接梯度作用范围可以改变其对几何、opacity及增密的直接影响；但SH会改变后续残差，M1的几何轨迹不必等于B-U。当前差异只能支持实际图像质量关系，不能唯一归因于某个几何、遮挡或表面因素。

恢复限制：加载状态和24批顺序精确，但后续轨迹不逐位相同。形变参数最大张量相对L2差{var['deformation_parameter_max_relative_L2']:.9g}，最大绝对差{var['deformation_parameter_max_abs']:.9g}；批次U/F最大差分别{var['uniform_loss_max_difference']:.9g}/{var['balanced_loss_max_difference']:.9g}。不能把这些累积差异都称作单次梯度1e-3内的小误差，也未唯一归因。工程验收把精确加载、同保存梯度消费和独立轨迹差异分开；正式训练是否发生外部恢复见costs.json。

未验证：独立序列泛化、真实材料点对应、几何、运动/接触准确性，以及无发布相机先验条件。静放背包可能被原FG mask排除；无可靠人体/物体分别mask，不新增伪实例指标。旧D0/D1仅复用，本轮未重新计算M1 N_eff。

成本：诊断{cost['D_attempts']}尝试，正式{cost['C_actual_attempts']}真实尝试、{cost['C_optimizer_calls']}实际Adam调用，GPU任务进程墙钟{cost['GPU_task_seconds']:.6f}秒。所有计入，不将历史复用成本称作零。
''')
    (RUN/'REPRODUCE.md').write_text('''# 复算与执行边界

在原项目运行envs/4dgs/bin/python experiments/appearance_router_v7_20260928/run01/code/verify_feedback.py --zip /绝对路径/V7_feedback.zip，可只读核对全部索引成员哈希。逐帧宽表的M1−B-U、M1−B-F直接逐键相减；宏平均与pooled MSE分列，LPIPS负差是改善。protocol/independent_verification.json记录一次独立CPU核验；不重复重算历史全部SSIM/LPIPS。

GPU执行入口为code/run_v7.py --config configs/v7.json，但已有流水线目录禁止原地重启。要重现实验，需另立明确输出和预算、保留配置/源/父状态身份，并保留真实尝试和所有失败；不能删除gate或日志来绕过一次执行约束。正式自身恢复要求purpose=formal、start-kind=own_checkpoint、相同谱系/科学配置/完整Adam/RNG/剩余栈及外部中断记录；诊断恢复不能作正式初值。

正式文档由build_report_docx.py读取冻结结果生成，使用bundled Python，render_docx.py与bundled LibreOffice做排版核验；这不触发训练。公共仓库仅含自产模块、薄适配与源配置；数据、模型、指标、图片、日志、原附件及文档不上传。
''')
    (RUN/'MISSING_ASSETS.md').write_text('''# 反馈包边界

本轮必需训练资产已实际找到并核验。反馈ZIP不包含完整模型、诊断完整梯度、284帧原RGB/mask、全图raw渲染、第三方源码、虚拟环境和CUDA二进制。state_manifest.json、environment.json及source_manifest.json提供路径、字节数和SHA；不是自包含完整重训包。

包内数字表、有限梯度原精度抽样和固定float32图像窗口用于只读复核，不能代替全量重算。全部固定图保留GT与模型对照，属于本地私有反馈。独立几何、track和实例mask未具备，本轮不用于放行图像质量实验；M1新N_eff未重复计算。跨主机Word应用逐项验收未执行，DOCX按图片内嵌与本机渲染检查交付。
''')
    print(decision)
if __name__=='__main__':run()
