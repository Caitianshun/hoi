from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json
O=Path(__file__).resolve().parent;m=json.loads((O/'source_manifest.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
for fr in m['frames']:
 f=fr['input_frame'];im=Image.open(fr['source']).crop((180,180,345,380)).resize((660,800),Image.Resampling.NEAREST);d=ImageDraw.Draw(im,'RGBA')
 for u in range(180,346,10):
  x=(u-180)*4;d.line((x,0,x,800), fill=(100,180,255,65),width=1);d.text((x+1,0),str(u),font=font,fill=(255,255,0,255),stroke_width=1,stroke_fill='black')
 for v in range(180,381,10):
  y=(v-180)*4;d.line((0,y,660,y), fill=(100,180,255,65),width=1);d.text((0,y+1),str(v),font=font,fill=(255,255,0,255),stroke_width=1,stroke_fill='black')
 im.save(O/f'grid_{f:03d}.png')
