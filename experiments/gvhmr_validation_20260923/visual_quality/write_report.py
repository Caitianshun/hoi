from pathlib import Path
import json,hashlib,datetime
import numpy as np
O=Path(__file__).resolve().parent;E=O.parent
s=json.loads((O/'scores.json').read_text());a=np.load(E/'run01/raw_geometry.npz')['joints_coco17_uv'];b=np.load(E/'run01/official_postproc_geometry.npz')['joints_coco17_uv']
md='''# GVHMR 可见关节的独立粗检查

**三种输出都在冻结的 8 个可见肘腕粗代理中有 7 个落入逐轴容差框；这只通过了有限的二维位置粗检查，不能证明人体三维、隐藏手腕或接触几何准确。** 输入侧助手粗代理在读取预测前已冻结，文件哈希未变化。它不是专家人工标注或数据集真值，以下差值应读作“与粗读点的差异”，不能直接当真实预测误差。

只在 `run01/run.json` 为 completed 后读取模型输出；逐项核对该记录中 ViTPose、raw 和 official_postproc 几何文件哈希。使用原始 640×480 坐标和固定 COCO17 解剖左右身份，不进行对齐、左右调换、置信度筛选或训练。8 个点的选择与容差来自输入侧冻结记录，没有根据预测好坏删点。

| 输出 | 落入逐轴容差框的粗点数 | 共同未落入者 |
| --- | ---: | --- |
| ViTPose 二维输入 | 7 / 8 | 帧 113 右肘 |
| GVHMR raw 相机投影 | 7 / 8 | 帧 113 右肘 |
| GVHMR official_postproc 相机投影 | 7 / 8 | 帧 113 右肘 |

这不是正式 PCK（关键点正确比例）。样本少、容差来自助手粗读；不能把 7/8 当作人体重建准确率。帧 113 右肘三种输出的差异均为约 11 像素，超过粗代理的 ±5 像素框，但仍可能包含图像读点或关节中心定义差异，应由研究者复核原图。此处保留冻结坐标，不根据模型输出反向调整标注。

下面 dx、dy 为“预测减粗读点”，d 为二维欧氏距离，单位均为原始像素。“框内”要求 dx、dy 的绝对值各自不超过容差，故不能把欧氏距离 d 与单轴容差直接混用。

| 帧 | 解剖关节 | 输出 | dx | dy | d | 每轴容差 | 框内 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
'''
for r in s['rows']:
 name={'ViTPose':'ViTPose','raw':'raw','official_postproc':'postproc'}[r['method']];joint=('左' if r['anatomical_side']=='left' else '右')+('肘' if r['joint']=='elbow' else '腕')
 md+=f"| {r['frame']} | {joint} | {name} | {r['dx_pixels']:.2f} | {r['dy_pixels']:.2f} | {r['euclidean_pixels']:.2f} | ±{r['tolerance_per_axis_pixels']} | {'是' if r['within_both_axis_tolerance'] else '否'} |\n"
md+='''
实际查看全部 8 帧、4 张四列拼图后的定性观察：

- 帧 25、55 背身时，可见肘投影与粗代理相容；模型同时给出了被躯干/箱子挡住的手腕位置，但原图不能检验它们。不能因为手腕投影落在衣物区域，就断言三维穿插或遮挡恢复正确。
- 帧 0、16、31、65 侧身时，输出保留了骨架结构，近侧手臂的大体方向与图像相符；两侧骨架投影经常重叠。输入侧未可靠判断左右身份，因此不进行事后匹配评分。
- 帧 93 弯腰时两臂仍有结构化输出；手部模糊、肩部遮挡和关节定义不确定，保持定性观察，不能由该叠图确认腕深度。
- 帧 113 双臂下垂时，双腕与左肘落在冻结容差内；右肘存在上述共同差异。这是后续复核可见关节定位的明确位置。

'''
md+=f"raw 与 postproc 的全部 114×17 个相机投影的最大逐轴差为 {np.max(np.abs(a-b)):.6f} 像素，因而叠图看起来接近，但不能表述为完全相同。这项检查不比较官方全局地面坐标，也不评价其物理后处理效果。\n\n"
md+='''图中所有帧均裁取原图固定范围 x=160:380、y=130:440，并作同样的 2 倍最近邻放大。第一列为无标记 RGB；后面依次是 ViTPose、raw、postproc。绿色表示模型的解剖左侧，品红色表示右侧，橙色框仅显示 8 个合格粗代理及其容差。模型预测的遮挡关节也绘出，但不等于这些关节在输入中可见。

'''
for k in range(1,5):md+=f"![固定帧叠图 {k}：原图、ViTPose、GVHMR raw、official postproc；绿色左侧、品红右侧、橙框为粗代理容差]({O}/overlay_sheet_{k}.png)\n\n"
md+=f'''[逐点机器可读记录]({O}/scores.json) · [逐点 CSV]({O}/per_point.csv) · [输入侧冻结代理]({E}/input_quality/assistant_visual_proxy.json) · [复算脚本]({O}/score_and_plot.py)

复算仅需 CPU，使用项目现有 mml 环境运行 `score_and_plot.py`。实际图片核查通过本地图片工具完成，没有声称在 VS Code 与 Codex 两个应用界面分别验收。
'''
(O/'REPORT.md').write_text(md)
s['status']='completed_with_visual_review';s['visual_review']={'images_viewed':[f'overlay_sheet_{k}.png' for k in range(1,5)],'all_eight_fixed_frames_viewed':True,'raw_postproc_max_abs_uv_difference_pixels':float(np.max(np.abs(a-b))),'expert_ground_truth':False,'GPU_used':False};(O/'scores.json').write_text(json.dumps(s,indent=2))
q={'status':'completed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'file_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.iterdir()) if p.is_file() and p.name!='verification.json'}}
(O/'verification.json').write_text(json.dumps(q,indent=2));print(O/'REPORT.md')
