from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np,json,hashlib
R=Path('/home/cai_tianshun/Project/HOI')
O=Path(__file__).resolve().parent
M=R/'experiments/mosca_interface_validation_20260923/human_prior/prepared_input/input_manifest.json'
m=json.loads(M.read_text()); fs=[0,16,25,31,55,65,93,113]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
rows=[]
for f in fs:
 p=Path(m['frame_paths'][f]); im=Image.open(p).convert('RGB'); h=hashlib.sha256(p.read_bytes()).hexdigest(); assert h==m['frame_sha256'][f]; assert im.size==(640,480)
 crop=(160,130,380,440); roi=im.crop(crop).resize((660,930),Image.Resampling.NEAREST)
 canvas=Image.new('RGB',(660,970),'white');canvas.paste(roi,(0,40));ImageDraw.Draw(canvas).text((8,10),f'input {f:03d} | t={m["timestamp_seconds"][f]:.3f}s | crop x160:380 y130:440',font=font,fill='black')
 canvas.save(O/f'input_{f:03d}_roi.png'); rows.append(canvas)
 rowspec={'input_frame':f,'source':str(p),'sha256':h,'video_frame':m['frame_indices'][f],'timestamp_seconds':m['timestamp_seconds'][f],'bbox_xyxy':m['person_bbox_xyxy'][f],'inspection_crop_xyxy':crop,'image_width':640,'image_height':480}
 if f==fs[0]:meta=[]
 meta.append(rowspec)
for k in range(2):
 sheet=Image.new('RGB',(1320,1940),'white')
 for i,im in enumerate(rows[k*4:(k+1)*4]): sheet.paste(im,((i%2)*660,(i//2)*970))
 sheet.save(O/f'input_roi_sheet_{k+1}.png')
full=Image.new('RGB',(1280,4*510),'white')
for i,f in enumerate(fs):
 im=Image.open(m['frame_paths'][f]);x=(i%2)*640;y=(i//2)*510;full.paste(im,(x,y+30));ImageDraw.Draw(full).text((x+8,y+5),f'input frame {f} / original frozen 640 x 480',font=font,fill='black')
full.save(O/'input_full_frames.png')
(O/'source_manifest.json').write_text(json.dumps({'protocol':'input-only assistant visual inspection; no predictor or reference geometry read','source_manifest':str(M),'source_manifest_sha256':hashlib.sha256(M.read_bytes()).hexdigest(),'fixed_frames':fs,'panels_transform':'Nearest-neighbor enlargement for inspection only; no training input change','frames':meta},indent=2))
print(O)
