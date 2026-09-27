"""Derive root-cause evidence and state identities from measured local files."""
from pipeline import RUN,ROOT,OLD,read,save,identity
import json,shutil,difflib
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def run():
    first=read(RUN/'runs/A_internal_boundary/first_bad_tensor.json');kernel=read(RUN/'runs/A_replay1b/kernel_case_analysis.json');iso=read(RUN/'runs/A_isolation2/summary.json');reg=read(RUN/'runs/A_regression128/result.json');tail=read(RUN/'protocol/restore_tail_comparison.json');domain=read(RUN/'protocol/stabilization_proposal.json')
    effects=[]
    for row in iso['rows']:
        effects.append({k:row[k] for k in ['case','domain','objective','frame_ids','loss']}|dict(nonfinite_gradient_elements=sum(x.get('nan',0)+x.get('posinf',0)+x.get('neginf',0) for x in row['gradients'].values())))
    ev=lines(RUN/'runs/A_regression128/replay_events.jsonl');triggers=[r for r in ev if r['phase']=='scale_bound'];axes=sum(r['total_axes'] for r in triggers);clipped=sum(r['triggered_axes'] for r in triggers)
    patch=dict(type=domain['type'],domain=domain,loss_and_gradient_isolation=effects,forward_VJP_comparisons=iso['comparisons'],regression=reg,density_events=lines(RUN/'runs/A_regression128/density_events.jsonl'),regression_triggered_axes=clipped,regression_total_axes=axes,regression_axis_trigger_fraction=clipped/axes,render_calls=len(triggers),restore=tail,derivative='Identity below frozen log extent; zero above; torch.clamp boundary convention at equality',not_equivalent=True,no_lower_bound_added=True,no_bad_gradient_replacement=True,normal_case='Pre-fine2101 exact checkpoint; original finite does not imply bound inactive')
    save(RUN/'patch_effects.json',patch);save(RUN/'first_bad_tensor.json',first)
    with (RUN/'replay_events.jsonl').open('w') as f:
        for name in ['A_replay1b','A_replay2','A_internal_boundary','A_regression128','A_restore_tail']:
            for line in (RUN/'runs'/name/'replay_events.jsonl').read_text().splitlines():f.write(line+'\n')
    old='scales_final = pc.scaling_activation(scales_final)\n';new='scales_final = pc.scaling_activation(bound_log_scale(scales_final, frozen_train_scene_extent))\n'
    diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/gaussian_renderer/__init__.py scaling activation',tofile='observed/render scaling activation'))
    diff+='\n# Own helper scale_domain.py\n'+(RUN/'code/scale_domain.py').read_text()
    (RUN/'patch.diff').write_text(diff)
    v4=ROOT/'experiments/foreground_stage_calibration_20260927/run01';states=[]
    specs=[(v4/'runs/W_all/recovery_fine_latest.pt','fine2100 finite completed',True),(v4/'runs/W_all/failure_state.pt','historical contaminated failure',False)]
    # Historical filename varies; the primary source index remains authoritative.
    if not specs[-1][0].exists():specs=specs[:-1]
    specs += [(RUN/'runs/A_replay1b/pre_step_002101.pt','fine2100 finite normal pre-state',True),(RUN/'runs/A_replay1b/pre_step_002103.pt','fine2102 finite fault pre-state',True),(RUN/'runs/A_replay1b/failure_state_with_gradients.pt','fine2103 interrupted backward with explicit gradients',False),(RUN/'runs/A_internal_boundary/failure_state_with_gradients.pt','fine2103 first internal conic boundary',False),(RUN/'runs/A_regression128/completed.pt','fine2230 finite diagnostic completed',True),(RUN/'runs/A_restore_tail/completed.pt','fine2230 finite restored diagnostic completed',True),(RUN/'protocol/shared_fine_initial.pt','shared fine0 after original fine reset',True)]
    for p,phase,resumable in specs:
        if p.exists():states.append(dict(**identity(p),phase=phase,resumable=resumable,diagnostic_states_excluded_from_B='shared fine0' not in phase))
    save(RUN/'state_manifest.json',dict(states=states,large_assets_local_only=True,checkpoint_fields=['model parameters','Adam groups moments steps','active SH','RNG Python NumPy Torch CPU CUDA','remaining viewpoint stack and temp list','density accumulators and radii','stage iteration phase config identity'],failure_gradients_saved_separately=True))
    shutil.copyfile(RUN/'code/reproduce_one_gaussian.py',RUN/'minimal_repro/reproduce.py')
    mini=read(RUN/'minimal_repro/validation.json')
    (RUN/'minimal_repro/README.md').write_text('''# 单高斯点前后向复现

此目录保存一个故障点、真实相机、全图 RGB 与 depth 的上游梯度。它复现该点的 CUDA 梯度异常，不等于复现完整优化轨迹。无需数据集或完整模型，仍需匹配的 PyTorch CUDA 环境及 rasterizer 扩展，实际身份见 environment.json。标准 PyTorch Python 包不足以替代此本地扩展。

```bash
CUDA_VISIBLE_DEVICES=1 /home/cai_tianshun/Project/HOI/envs/4dgs/bin/python minimal_repro/reproduce.py --case-dir minimal_repro
```

外部主机请替换解释器路径，并显式选择空闲 GPU。此命令有两次无优化 GPU 前反向；本轮授权探针账本已记录实际验证，交付时不要自动追加运行。case.json 是标量及形状信息，npz 保留 float32。原始和尺度有界各一次，预期结果见 validation.json。提取时从同一次失败 backward 保存的几何缓冲恢复 opacity，避免 batch 中前向末帧与反向首帧错配。完整训练重放仍需要 state_manifest 指向的本地大文件。
''')
    save(RUN/'minimal_repro/index.json',dict(assets=[identity(RUN/'minimal_repro'/p) for p in ['case.json','one_gaussian_case.npz','reproduce.py','README.md','validation.json']],self_contained_numeric_case=True,requires_matching_external_CUDA_extension=True,full_training_replay_self_contained=False))
    point=str(kernel['all_bad_attribute_row_intersection'][0]);bad=kernel['bad_points'][point];geometry=kernel['bad_input_geometry'][point];normal=next(r for r in iso['comparisons'] if r['case']=='normal');fault=next(r for r in iso['comparisons'] if r['case']=='failure' and r['objective']=='balanced_rgb')
    text=f'''# V5 数值故障链与证据边界

从 W_all 有限 fine2100 恢复的两次独立短重放均在 fine{first['iteration']}、批次 {first['batch']} 触发同一高斯行 {point} 的 rasterizer 反向非有限。额外内部缓冲观察将最早已测边界前移到 forward_internal_geometry/conic_opacity：其前三个逆二维协方差系数为 NaN。输入、三维协方差、最终 RGB/depth 和当轮总损失此前仍有限。RGB 看上去有限不足以认定渲染内部和梯度安全。

该点形变后实际尺度为 {geometry['scale']}，深度 {bad['depth']}，投影半径 {bad['radii']}。规范尺度本身没有相同数量级，爆大来自最终形变尺度。锁定 CUDA forward.cu 先构造三维协方差，再以 float32 投影、求行列式和逆协方差；巨大的有限尺度进入这个链，会超过后续乘法与差值的有效数值范围。数据确认 conic 已坏，源码与尺度支持溢出/消减解释，但未逐标量插桩断言是哪一次乘法最先坏。

传播链为 形变后尺度极端 → 内部 conic 非有限 → 屏幕/位置/尺度/旋转等 VJP 非有限 → 若继续 Adam 则参数及 moments 污染。新守卫在更新前停止；V4 的污染现场保留。VJP 表示给定图像上游梯度时对渲染输入的向量雅可比积。实际 depth 上游梯度全零，深度有限；当前样例不支持把未用 depth 梯度单独认定为根因。

同一故障前状态分别只反传均匀 RGB、平衡 RGB、正则。两种 RGB 都有异常，正则独立有限；所以平衡监督并非该状态下异常出现的必要条件。这不能排除不同监督在较早训练中影响模型走向极端状态。完整隔离值与每参数组差异见 patch_effects.json。

唯一稳定修改是把最终形变 log scale 的上界设为冻结训练 scene extent 的对数，上界对应 {domain['upper_bound']} 个继承场景单位。未核实米制，不写作米。它限制进入投影的尺度，未改规范参数、密度规则或其他损失；数学域与梯度已经改变，不是严格等价的实现修复。正常输入实际 forward 最大差 {normal['forward_max_abs']}，故障输入 {fault['forward_max_abs']}，不能用一个通过字段掩盖正常结果也改变。上下游梯度差异完整保留。

从故障前有限状态执行 {reg['attempted']} 次真实参数更新，最后 fine{reg['last_iteration']}，跨过原下一次增密/剪枝，无跳批及坏参数/Adam。尺度上界在 {len(triggers)} 次渲染中累计触及 {clipped}/{axes} 个轴（比例 {clipped/axes:.9g}，不是独立高斯比例）。局部回归不保证完整日程稳定。完整状态恢复前逐张量精确、后续采样和 RNG 一致；两轮 CUDA 轨迹最大张量差 {tail['max_tensor_abs']}，不是逐位重现。

最小样例只保留一个点，数值包 {mini['case']['bytes']} 字节，原始后端复现坏梯度、同输入尺度有界后全部有限；相机和上游梯度不改。它隔离了该几何链，尚不能解释该点为什么在长优化中走到极端，也不能证明所有数值故障已解决。V4 fine1167 的 CUDA 非法访问和 W_fine fine675 没有被追认成同一原因。

证据入口：runs/A_internal_boundary/first_bad_tensor.json、runs/A_replay1b/kernel_case_analysis.json、runs/A_isolation2/summary.json、runs/A_regression128、protocol/restore_tail_comparison.json、minimal_repro。后续 B 的任何新失败须另存首异常，不用这条已定位链替代新取证。
'''
    (RUN/'ROOT_CAUSE.md').write_text(text)
    save(RUN/'A_summary.json',dict(first_bad=first,kernel_case=kernel,patch_effects=identity(RUN/'patch_effects.json'),minimal_repro=mini,regression=reg,normal_forward_max_abs=normal['forward_max_abs'],fault_forward_max_abs=fault['forward_max_abs'],restored_tail_max_tensor_abs=tail['max_tensor_abs'],gate=identity(RUN/'protocol/formal_gate.json')))
    print('A evidence collected')
if __name__=='__main__':run()
