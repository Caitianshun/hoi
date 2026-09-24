from pathlib import Path
import json,hashlib,datetime
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent
src=json.loads((O/'source_manifest.json').read_text())
# Assistant image-only coarse reading on original 640x480 coordinates.
# These are not dataset/manual-expert ground truth.
rows=[]
def add(f, side, joint, xy, uncertainty, visibility, note, anatomical=None, side_confidence='unknown', eligible=False):
 rows.append({'frame':f,'arm_identity_in_image':side,'joint':joint,'anatomical_side':anatomical,'anatomical_side_confidence':side_confidence,'xy_original_640x480':xy,'uncertainty_plusminus_pixels_per_axis':uncertainty,'uncertainty_interpretation':'assistant conservative localization range, not calibrated confidence interval','visibility':visibility,'numeric_comparison_eligible':eligible,'note_zh':note})
for f,ps in [(0,[(280,280),(290,322),(278,355)]),(16,[(244,222),(255,250),(240,261)]),(31,[(287,216),(291,241),(320,236)]),(65,[(277,219),(281,243),(309,238)])]:
 for j,p,u in zip(['shoulder','elbow','wrist'],ps,[8,5,4]):
  add(f,'near_visible_arm',j,list(p),u,'clothing_inferred' if j=='shoulder' else 'visible_coarse','可见近侧手臂；侧身下左右身份未独立确认，不进行带左右标签的数值误差比较。肩中心只能依据衣袖估计。')
 for j in ['shoulder','elbow','wrist']:
  add(f,'far_arm',j,None,None,'unknown_occluded','远侧手臂与躯干/箱子重叠，当前单幅 RGB 无法可靠定位。')
for f,ps in [(25,{'left':[(225,212),(218,244),None],'right':[(276,215),(287,245),None]}),(55,{'left':[(215,211),(213,246),None],'right':[(269,211),(279,244),None]})]:
 for side,pts in ps.items():
  for j,p in zip(['shoulder','elbow','wrist'],pts):
   add(f,'image_left_arm' if side=='left' else 'image_right_arm',j,list(p) if p else None,8 if j=='shoulder' else (6 if j=='elbow' else None),'clothing_inferred' if j=='shoulder' else ('visible_coarse' if j=='elbow' else 'unknown_occluded'),'背身，左右身份可判断；肩中心位于黑色上衣内、仅粗估；肘部可见；腕在躯干/箱子前侧无法定位。',side,'high',j=='elbow')
for side,pts in [('image_right_near_arm',[(283,283),(284,322),(272,352)]),('image_left_far_arm',[None,(262,321),(259,348)])]:
 for j,p in zip(['shoulder','elbow','wrist'],pts):
  add(93,side,j,list(p) if p else None,(8 if j=='shoulder' else (6 if j=='elbow' else 5)) if p else None,'unknown_occluded' if p is None else ('clothing_inferred' if j=='shoulder' else 'visible_blurred'),'弯腰、两臂重叠且手部模糊；图像侧别可分，解剖左右身份不作可靠结论。')
for side,pts in [('right',[(226,219),(217,250),(205,269)]),('left',[(269,219),(279,252),(279,282)])]:
 for j,p in zip(['shoulder','elbow','wrist'],pts):
  add(113,'image_left_arm' if side=='right' else 'image_right_arm',j,list(p),8 if j=='shoulder' else 5 if j=='elbow' else 4,'clothing_inferred' if j=='shoulder' else 'visible_coarse','正面略偏转，双臂下垂。肩中心由上衣轮廓估计；肘部与手腕袖口可独立定位。',side,'high',j in ['elbow','wrist'])
rows.sort(key=lambda r:(r['frame'],r['arm_identity_in_image'],['shoulder','elbow','wrist'].index(r['joint'])))
data={'schema':'assistant_image_only_upper_limb_proxy_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixed_frames':src['fixed_frames'],'source_manifest_sha256':hashlib.sha256((O/'source_manifest.json').read_bytes()).hexdigest(),'method':'Assistant visually inspected unaltered frozen input RGB shown in crops; manually read grid coordinates; no model prediction, dataset fitted body/mesh, other camera, or depth read. Image crops use nearest-neighbor enlargement only.','is_ground_truth':False,'is_expert_annotation':False,'intended_use':'Pre-prediction frozen coarse sanity check only. Never supervise training, choose optimization hyperparameters, or claim reconstruction accuracy gains with these labels.','evaluation_rules':['Only numeric_comparison_eligible points may be used for a coarse pixel discrepancy check.','Use the declared uncertainty box; inside uncertainty is compatible, not proof of correctness.','Do not infer 3D accuracy/contact, metric camera depth or occluded joint validity from 2D compatibility.','Do not report these as manual human annotation or dataset GT.','For side-ambiguous frames retain qualitative overlays; do not choose the matching model left/right post hoc.','Shoulder coordinates are inferred through clothing and excluded from numerical check.'],'rows':rows,'counts':{'total':len(rows),'numerical_eligible':sum(r['numeric_comparison_eligible'] for r in rows),'unknown':sum(r['xy_original_640x480'] is None for r in rows)}}
(O/'assistant_visual_proxy.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
frames=[]
for f in src['frames']:
 im=Image.open(f['source']).convert('RGB').crop((160,130,380,440)).resize((660,930),Image.Resampling.NEAREST);d=ImageDraw.Draw(im)
 for r in rows:
  if r['frame']!=f['input_frame'] or r['xy_original_640x480'] is None:continue
  x,y=r['xy_original_640x480'];x=(x-160)*3;y=(y-130)*3;u=r['uncertainty_plusminus_pixels_per_axis']*3;c='lime' if r['numeric_comparison_eligible'] else 'orange'
  d.rectangle((x-u,y-u,x+u,y+u),outline=c,width=2);d.line((x-4,y,x+4,y),fill=c,width=2);d.line((x,y-4,x,y+4),fill=c,width=2)
  lab=(r['anatomical_side'][0].upper() if r['anatomical_side'] else '?')+r['joint'][0].upper();d.text((x+6,y+6),lab,font=font,fill=c,stroke_width=1,stroke_fill='black')
 canvas=Image.new('RGB',(660,970),'white');canvas.paste(im,(0,40));ImageDraw.Draw(canvas).text((7,10),f'Input {f["input_frame"]:03d}: green=coarse check, orange=qualitative',font=font,fill='black');frames.append(canvas);canvas.save(O/f'annotated_{f["input_frame"]:03d}.png')
for k in range(2):
 sheet=Image.new('RGB',(1320,1940),'white')
 for i,im in enumerate(frames[k*4:(k+1)*4]):sheet.paste(im,((i%2)*660,(i//2)*970))
 sheet.save(O/f'annotated_sheet_{k+1}.png')
print(data['counts'])
