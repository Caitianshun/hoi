"""Create the portable, image/font-embedded PDF from frozen paired evidence."""
from pathlib import Path
import json, hashlib, datetime, html
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab import rl_config
E=Path(__file__).resolve().parents[1];OLD=E.parents[1]/'structured_hoi_20260923'
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v,d=2): return 'N/A' if v is None else f'{v:.{d}f}'
fontpath='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
pdfmetrics.registerFont(TTFont('CJK',fontpath));pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
rl_config.canvas_basefontname='Latin'
latin_cmap=pdfmetrics.getFont('Latin').face.charToGlyph
cjk_cmap=pdfmetrics.getFont('CJK').face.charToGlyph
def markup(text):
 # DroidSansFallback intentionally contains CJK only. Use an embedded Latin
 # font for all supported Latin/numerical/math characters, not blank glyphs.
 runs=[];last=None;chunk=''
 for ch in str(text):
  if ch=='\n':
   if chunk:runs.append(f'<font name="{last}">{html.escape(chunk)}</font>');chunk=''
   runs.append('<br/>');last=None;continue
  face='Latin' if ord(ch) in latin_cmap else 'CJK'
  assert ord(ch) in (latin_cmap if face=='Latin' else cjk_cmap),f'Unsupported glyph {ch!r}'
  if face!=last and chunk:runs.append(f'<font name="{last}">{html.escape(chunk)}</font>');chunk=''
  chunk+=ch;last=face
 if chunk:runs.append(f'<font name="{last}">{html.escape(chunk)}</font>')
 return ''.join(runs)
styles={
 'title':ParagraphStyle('title',fontName='CJK',fontSize=23,leading=32,textColor=colors.HexColor('#123451'),spaceAfter=18,wordWrap='CJK'),
 'h':ParagraphStyle('h',fontName='CJK',fontSize=16,leading=23,textColor=colors.HexColor('#123451'),spaceAfter=12,wordWrap='CJK'),
 'sub':ParagraphStyle('sub',fontName='CJK',fontSize=11.5,leading=18,textColor=colors.HexColor('#226a82'),spaceBefore=8,spaceAfter=7,wordWrap='CJK'),
 'p':ParagraphStyle('p',fontName='CJK',fontSize=10,leading=16,spaceAfter=8,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='CJK',fontSize=8.3,leading=12,spaceAfter=6,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='CJK',fontSize=8.3,leading=12,wordWrap='CJK'),
}
story=[];assets=[]
def p(text,style='p'):story.append(Paragraph(markup(text),styles[style]))
def head(text):p(text,'h')
def page():story.append(PageBreak())
def table(rows,widths=None):
 data=[[Paragraph(markup(v),styles['cell']) for v in row] for row in rows]
 t=Table(data,colWidths=widths or [511/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Latin'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7594aa')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dddddd')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.extend([t,Spacer(1,10)])
def image(path,maxh=510,w=511):
 path=Path(path);im=PILImage.open(path);ww,hh=im.size;factor=min(w/ww,maxh/hh);story.append(Image(str(path),width=ww*factor,height=hh*factor));story.append(Spacer(1,9));assets.append({'path':str(path),'sha256':sha(path)})
def pose_summary(pair,kind,method,key):return pair[kind]['summaries'][method][key]['mean']
def query(pair,method,group):
 a=pair['fixed_query_results'][method]['metrics']['results']
 return next(x for x in a['summaries'] if x['group']=='entity:'+group and x['visibility']=='all')['mean_epe_m']*100
def relative(pair,method):
 a=pair['fixed_query_results'][method]['metrics']['results']['hand_object_relative']
 return a
def make_heldout(dev,pair):
 # Crops use the pre-existing evaluation mask only; identical crop for old/new.
 # All five predeclared heldout slots are displayed, including missing-mask slot.
 rows=load(OLD/f'data_audit/heldout_regions/{dev}/mask_manifest.json')['rows'];old=Path(pair['old_export']);new=Path(pair['new_export']);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
 sheet=PILImage.new('RGB',(1260,5*262+52),'white');draw=ImageDraw.Draw(sheet)
 for j,title in enumerate(['Heldout camera1 RGB','Old S1','S1*']):draw.text((j*420+10,10),title,font=font,fill='#123451')
 for i,row in enumerate(rows):
  imgs=[PILImage.open(row['gt_path']).convert('RGB'),PILImage.open(old/'heldout'/row['filename']).convert('RGB'),PILImage.open(new/'heldout'/row['filename']).convert('RGB')]
  box=(0,0,640,480)
  if row.get('mask_status')=='available':
   q=np.load(row['mask_path']);labels=q['entity_labels'] if 'entity_labels' in q else q[q.files[0]];yy,xx=np.where((labels==1)|(labels==2))
   if len(xx):
    cx=(xx.min()+xx.max())/2;cy=(yy.min()+yy.max())/2;h=min(480,max(130,yy.max()-yy.min()+45));w=min(640,max(h*1.72,xx.max()-xx.min()+45));h=min(480,w/1.72);x=max(0,min(640-w,cx-w/2));y=max(0,min(480-h,cy-h/2));box=(int(x),int(y),int(x+w),int(y+h))
  for j,im in enumerate(imgs):
   crop=im.crop(box);crop.thumbnail((414,235));sheet.paste(crop,(j*420+(420-crop.width)//2,i*262+52));draw.text((j*420+7,i*262+52+236),f't={row["nominal_time"]:.1f}s  f{row["input_index"]}'+('  mask N/A' if row.get('mask_status')!='available' else ''),font=font,fill='#333333')
 out=E/f'output/figures/{dev}_heldout_five_cases.png';sheet.save(out);return out

def main():
 decision=load(E/'evaluation/decision.json');pairs={d:load(E/f'evaluation/{d}_reconstruction_pair/paired_reconstruction_results.json') for d in ['dev1','dev2']};sels={d:load(E/f'pose/{d}/optimization/selection_frozen.json') for d in pairs};train={d:load(E/f'reconstruction/{d}_S1_star/run.json') for d in pairs}
 p('动态人—物—场景建模\nP1 → S1* 配对验证','title')
 p('CVPR 2027 开发实验 · 2026-09-24 UTC · 两条 BEHAVE 开发序列 · 单种子','small')
 p(decision['headline'],'sub')
 for text in decision['summary']:p(text)
 rows=[['主要指标','箱体：旧 S1 → S1*','木椅：旧 S1 → S1*']]
 for name,key in [('物体中心误差 / cm','centroid_error_cm'),('原始旋转误差 / 度','raw_rotation_error_deg')]:
  rows.append([name,*[f"{fmt(pose_summary(pairs[d],'pose', 'old_S1',key))} → {fmt(pose_summary(pairs[d],'pose','S1_star',key))}" for d in pairs]])
 rows.append(['实际物体表面代理 / cm',*[f"{fmt(pairs[d]['surface']['old_S1']['summary']['object']['symmetric_proxy_mean_cm'])} → {fmt(pairs[d]['surface']['S1_star']['summary']['object']['symmetric_proxy_mean_cm'])}" for d in pairs]])
 table(rows,[165,173,173]);p(decision['action'],'sub');p('本轮所有候选先按输入侧规则选定并保存哈希，再读取独立参考。原四分支、缓存和历史评价保留。S2、注意力和接触模块均未启用。','small')
 p('误差参考为数据集拟合网格，包含拟合与时间配对不确定性；本文档不将这些结果解释为真实材料标签、精细接触准确率或盲测基准成绩。','small')
 page();head('01  本轮到底改变了什么')
 p('要检验的问题是：物体初始位置、朝向和运动是否限制最终的结构化重建。以手握箱体为例，即使图像轮廓贴合，箱体仍可能沿视线偏远或转到错误一面；高斯颜色和小幅位置更新可以补偿部分像素误差，不能据此认定运动正确。')
 table([['项目','固定或变化'],['合法输入','camera0 标定单目 RGB，640×480，真实不等间隔时间；输入侧分割、估计深度、RGB人体姿态；通用 SMPL-X；已知实例无纹理米制几何。'],['本轮新增','每事件5个互补关键帧；短 CoTracker 片段、前后向循环及描述子筛选；MegaPose RGB候选；最多3条全段候选路径。'],['P1 优化','只更新每帧刚体旋转/平移。固定形状、尺度、相机、外观、人体和背景。'],['S1* 实际差异','物体 R/t 初值、pose soft prior目标和修正量时序基准一起更新；object_init来源指针随之变化。'],['训练控制','全部从0到8000步；原源码、种子12345、采样日程、点数预算、增密规则和损失权重一致；旧物体颜色/规范采样、人体/背景原样复制。'],['仅评价的信息','发布逐帧姿态、拟合网格/拟合掩码、camera1、传感深度。它们不参与P1或训练。']],[103,408])
 p('新的短轨迹没有进入 S1* 重建损失；原普通 track 缓存和频率保持不变。因而本轮评价的是“初始化流程”，不能把全部变化归于单独的旋转/平移初值，也不是 S2 或新的注意力机制。')
 p('真实时间：所有几何轨迹仍对应原视频帧及实际时间戳；速度与加速度用相邻真实秒差计算。中间缺观测帧是预测状态，不累加成观测。')
 page();head('02  输入侧筛选、对应与成本')
 table([['项目','箱体 dev1','木椅 dev2'],['全部帧 / 固定可见帧','114 / 103','98 / 91'],['关键帧','2, 40, 43, 83, 88','3, 24, 49, 71, 77'],['通过二维可靠性门槛的时序记录','1794 / 5889','1895 / 8472'],['至少6条可靠二维记录的帧','53 / 114','52 / 98'],['选中全段路径',sels['dev1']['selected']['path'],sels['dev2']['selected']['path']],['被采用的条件性规范时序记录',str(sels['dev1']['selected']['used_temporal_observations']),str(sels['dev2']['selected']['used_temporal_observations'])],['固定可见集合 mean IoU',*[f"{sels[d]['selected']['old_mean_visible_IoU']:.4f} → {sels[d]['selected']['new_mean_visible_IoU']:.4f}" for d in pairs]],['CPU 位姿流程 / 秒',*[fmt(sels[d]['wall_seconds']) for d in pairs]]],[237,137,137])
 p('采用规则：真实可见、5×5同实例内区、返回查询可见、循环误差不超过2像素，并通过RGB/纹理门槛。规范点还须由两个不同源时刻的重叠片段、至少两个共同目标时刻、外观与候选规范距离共同支持。箱体选中路径有26个条件确认查询；木椅为0，未放宽阈值。')
 p('每个规范附着都依赖候选源位姿，仍可能整体绑错表面。单源自己的回投影没有加入强轨迹残差。RGB描述子主要作为对应门控，评分中的RGB常数项不能单独选择正确全局朝向。不同路径采用的对应集合不同，覆盖数量和残差同时保留。')
 p('MegaPose 共生成50个候选，成功推理流程47.98秒（纯模型41.03秒）；权重173.07 MB，下载49.32秒。短轨迹推理19.73秒。以上为本机实测，不包含首次环境接入与失败尝试的人工排查时间。')
 p('输入门槛：同一掩码、同一103/91帧，旧/新均使用真实三角面投影并排除人体与未知区；mean IoU下降超过0.02拒绝晋级。两事件均通过。二维门槛仅用于止损，不是三维正确性判据。','small')
 for d,name in [('dev1','箱体'),('dev2','木椅')]:
  page();head('03  '+name+'：输入侧拟合与完整运动')
  image(E/f'output/figures/{d}_input_pose_comparison.png',maxh=322)
  image(E/f'output/figures/{d}_input_score_curves.png',maxh=310)
  p('固定关键帧和全部时间区间均保留。灰区为原协议缺少可靠物体观测的时刻，曲线输出仍属预测；高 IoU、低速度或轨迹连续均不自动证明三维正确。','small')
 page();head('04  初始化与训练后：独立三维误差')
 for d,name in [('dev1','箱体'),('dev2','木椅')]:
  p(name+'：固定参考槽 '+('14/14' if d=='dev1' else '9/10（t=2秒缺失保留）'),'sub')
  rows=[['阶段','中心误差 cm','深度绝对误差 cm','原始角差 度','顶点 RMSE cm','相邻参考位移误差 cm']]
  for kind,lab,label in [('initialization_pose','old_init','旧初始化'),('initialization_pose','P1','P1'),('pose','old_S1','旧 S1'),('pose','S1_star','S1*')]:
   v=pairs[d][kind]['summaries'][lab];motion=np.mean([r['centre_displacement_error_cm'] for r in v['reference_interval_motion'] if r['centre_displacement_error_cm'] is not None]);rows.append([label,*[fmt(v[k]['mean']) for k in ['centroid_error_cm','camera_depth_absolute_error_cm','raw_rotation_error_deg','corresponding_vertex_rmse_cm']],fmt(motion)])
  table(rows,[65,82,94,84,88,98])
 p('中心定义沿用同一版本模板全部顶点的均值（含历史孤立顶点），未调整评价口径。旋转参考从对应模板边解析读出，并在整张参考网格上校验；不拟合预测，不做 ICP、尺度或坐标对齐。')
 p('对称集合在新评分前由无纹理三角面几何预声明。保守容差下两模板只有恒等变换通过，因此本轮对称角差等于原始角差。保留完整轨迹角速度、超过90度的相邻旋转记录以及整段统一规范约定；不逐帧挑对称变换来抹掉身份翻转。未实现正式 BOP MSSD/MSPD。')
 for d,name in [('dev1','箱体'),('dev2','木椅')]:
  page();head('05  '+name+'：参考误差随时间变化')
  image(E/f'evaluation/{d}_reconstruction_pair/initialization_pose/paired_pose_curves.png',maxh=302)
  image(E/f'evaluation/{d}_reconstruction_pair/pose/paired_pose_curves.png',maxh=302)
  p('上：旧初始化与P1。下：旧S1与S1*。缺失参考保持空缺；未按可见性删掉困难参考槽。深度曲线为相机坐标带符号偏差，其他距离单位在图内注明。','small')
 page();head('06  实际高斯表面、查询及新视角')
 rows=[['指标','箱体 旧 S1 → S1*','木椅 旧 S1 → S1*']]
 for ent,cn in [('human','人体'),('object','物体')]:rows.append([cn+'表面代理 / cm',*[f"{fmt(pairs[d]['surface']['old_S1']['summary'][ent]['symmetric_proxy_mean_cm'])} → {fmt(pairs[d]['surface']['S1_star']['summary'][ent]['symmetric_proxy_mean_cm'])}" for d in pairs]])
 for ent,cn in [('human','人体'),('object','物体'),('full','整图')]:
  for metric,suf in [('mean_frame_psnr_db','PSNR/dB'),('mean_frame_ssim','SSIM')]:rows.append(['留出'+cn+' '+suf,*[f"{fmt(pairs[d]['heldout']['old_S1']['summary'][ent][metric],3)} → {fmt(pairs[d]['heldout']['S1_star']['summary'][ent][metric],3)}" for d in pairs]])
 table(rows,[203,154,154])
 q=pairs['dev1'];p(f"箱体事件的实际高斯固定查询：手部 {query(q,'old_S1','hand'):.3f} → {query(q,'S1_star','hand'):.3f} cm，物体 {query(q,'old_S1','object'):.3f} → {query(q,'S1_star','object'):.3f} cm。手—物相对向量及覆盖率见下方决策说明。木椅没有对应材料查询参考，记 N/A。")
 for txt in decision['surface_and_query_notes']:p(txt,'small')
 p('表面代理采用 opacity≥0.05、每实例最多10000个稳定ID高斯中心，并与参考三角面双向比较；它不是完整实体表面距离。保留相机每事件5帧，木椅t=2秒无区域掩码，区域指标只按其余4帧；整图仍含5帧。LPIPS保持N/A。','small')
 for d,name in [('dev1','箱体'),('dev2','木椅')]:
  page();head('07  '+name+'：全部固定留出案例')
  image(make_heldout(d,pairs[d]),maxh=610)
  p('三列使用相同裁切，裁切仅来自原先冻结的评价掩码；五个留出时刻全部展示，未按结果选有利帧。木椅缺掩码帧展示整图并明确标注。预测白色区域也包括未覆盖背景与背景深度错误，不能全部解释为场景未被拍摄。','small')
 page();head('08  决策、局限与下一步')
 for text in decision['reasoning']:p(text)
 p('本轮不进入 S2 / 注意力 / 接触模块','sub')
 for text in decision['next_steps']:p(text)
 p('评价边界','sub')
 p('这两条序列此前已经用于开发诊断，本轮冻结不使它们变成盲测。单个种子的细微变化不足以支持稳定领先。箱体输入与标称参考时间最大偏差约104 ms；木椅与标称槽约37.10 ms，与实际camera1帧匹配约33.489 ms。三者含义不同，均不是同步精度保证，也不支持精细接触正确的结论。')
 p('物体RGB描述子只能对当前观测片段提供有限关联；无纹理几何输入没有提供真实材料身份。P1并未恢复所有不可见表面或长时间完全遮挡运动，缺观测区仍受先验约束。')
 p('时限保持：核心结果最晚11月4日冻结；11月5—15日连续11个完整自然日用于写作、图表和审阅。下一验证应限定一个明确问题，不扩展骨干或大型网络搜索。','small')
 page();head('09  资源、完整性与复算入口')
 rows=[['实测项目','箱体','木椅']]
 for label,key in [('完整8000步训练 / s','wall_seconds')]:rows.append([label,*[fmt(train[d].get(key,train[d].get('total_seconds'))) for d in pairs]])
 rows.append(['峰值训练分配 / GiB',*[fmt(train[d]['peak_allocated_bytes']/2**30,3) for d in pairs]])
 rows.append(['从零开始 / 最终步数',*[str(train[d]['start_step'])+' / '+str(train[d]['last_checkpoint_step']) for d in pairs]])
 rows.append(['最终高斯数：背景/人体/物体',*[' / '.join(map(str,pairs[d]['cost']['new_export_record']['points_by_entity'])) for d in pairs]])
 table(rows,[203,154,154])
 p('重建均在RTX3090执行，明确绑定物理GPU1。训练成功后自动导出与评价；最终核查包括子进程返回值、训练检查点、数据/源码身份、输出帧数以及评价状态。GPU0已有其他任务，未调度本轮训练到该卡。MegaPose最初GLX曾短暂创建GPU0图形上下文，发现后终止己方进程；最终用EGL及GPU型号断言固定3090，异常与失败日志保留。','small')
 p('可复算入口（均位于本实验目录）','sub')
 table([['文件','作用'],['REPRODUCE.md','完整命令、旧新数据边界、运行顺序与状态核验'],['protocol/audit_inputs_manifest.json','输入、评价资产、旧模型、前端与源码哈希'],['protocol/pose_config_v1.json + pre_solver_clarification.json','运行前配置、有限预算与预运行代码审查修正'],['pose/dev*/optimization/selection_frozen.json','唯一选中路径、其他候选分数/拒绝原因及哈希'],['reconstruction/dev*_initialization/controlled_difference.json','旧/新初始化仅三字段变化，其余逐项相同'],['evaluation/dev*_reconstruction_pair/','旧/新初始化与最终重建完整配对、逐参考槽数据'],['megapose/status.json + compatibility_patch.diff','官方来源、权重、部署兼容修复、成本与失败记录']],[239,272])
 p('本PDF的中文字体、关键图片和图表全部内嵌。Markdown/JSON/脚本是可编辑与可复算来源，不是阅读本PDF的依赖。已执行本地渲染与资源嵌入核验；未宣称在其他主机应用中逐项测试。','small')
 out=E/'output/pdf/P1_S1star_paired_verification.pdf';out.parent.mkdir(parents=True,exist_ok=True)
 def footer(canvas,doc):
  canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d9e2eb'));canvas.line(42,39,A4[0]-42,39);canvas.setFillColor(colors.HexColor('#607080'));f=Paragraph(markup('HOI · 输入冻结后独立评价 · 2026-09-24 UTC'),styles['small']);f.wrap(470,15);f.drawOn(canvas,42,24);canvas.setFont('Latin',8);canvas.drawRightString(A4[0]-42,26,str(doc.page));canvas.restoreState()
 SimpleDocTemplate(str(out),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=52,title='HOI P1与S1*配对验证',author='HOI research').build(story,onFirstPage=footer,onLaterPages=footer)
 (E/'output/pdf/report_assets.json').write_text(json.dumps({'pdf':str(out),'pdf_sha256':sha(out),'embedded_source_images':assets,'generator_sha256':sha(__file__),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},ensure_ascii=False,indent=2)+'\n');print(out)
if __name__=='__main__':main()
