"""Export all fixed evaluation frames and source audits; no selection by gain."""
from pathlib import Path
import json
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
E=Path(__file__).resolve().parents[1];A=E.parents[2]/'experiments/aux_ref_object_reconstruction_20260924/run01';out=E/'output/figures';out.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
manifest=json.loads((A/'evaluation/regions/manifest.json').read_text());metrics=json.loads((E/'evaluation/per_frame.json').read_text());lookup={(r['dev'],r['time'],r['variant']):r for r in metrics}
for dev in ['dev1','dev2']:
 erows=[r for r in manifest['rows'] if r['dev']==dev];names=['GT','B0','B1','F0','F1','F2','HS'];cw=220;ch=235;grid=Image.new('RGB',(cw*7,ch*len(erows)+35),'white');dr=ImageDraw.Draw(grid)
 for j,n in enumerate(names):dr.text((j*cw+10,6),n,font=font,fill='black')
 full=Image.new('RGB',(640*7,510*len(erows)+35),'white');df=ImageDraw.Draw(full)
 for j,n in enumerate(names):df.text((j*640+15,6),n,font=font,fill='black')
 for i,e in enumerate(erows):
  lab=np.load(e['regions']['path'])['entity_labels'];O=lab==2;yy,xx=np.where(O);bounds=(max(0,xx.min()-60),max(0,yy.min()-60),min(640,xx.max()+61),min(480,yy.max()+61));x0,y0,x1,y1=bounds
  for j,n in enumerate(names):
   p=E/'evaluation'/dev/e['frame_id']/f'{n}.png';img=cv2.imread(str(p));assert img is not None
   contour,_=cv2.findContours(O.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(img,contour,-1,(255,255,0),1)
   im=Image.fromarray(img[...,::-1]);full.paste(im,(j*640,i*510+35));df.text((j*640+10,i*510+516),e['frame_id'],font=font,fill='black')
   crop=im.crop(bounds);crop.thumbnail((210,185));grid.paste(crop,(j*cw+(cw-crop.width)//2,i*ch+40));dr.text((j*cw+8,i*ch+222),e['frame_id'],font=small,fill='black')
   if n in names[1:-1]:
    val=lookup[dev,e['query_time_seconds'],n]['metrics']['object']['psnr_db'];dr.text((j*cw+82,i*ch+222),f'{val:.2f}dB',font=small,fill='black')
 grid.save(out/f'{dev}_crop.png');full.save(out/f'{dev}_full.png')
 # Support plot uses every fixed O pixel; zero=no local mapping, not unseen truth.
 sg=Image.new('RGB',(320*len(erows),290),'white');ds=ImageDraw.Draw(sg);pal=np.array([[190,190,190],[175,100,210],[245,177,45],[60,185,100]],np.uint8)
 for j,e in enumerate(erows):
  data=np.load(E/'evaluation'/dev/e['frame_id']/'support.npz');lab=np.load(e['regions']['path'])['entity_labels'];img=np.full((480,640,3),255,np.uint8);img[lab==2]=pal[data['support_state'][lab==2]];sg.paste(Image.fromarray(img).resize((320,240)),(j*320,25));ds.text((j*320+10,5),e['frame_id'],font=small,fill='black')
 sg.save(out/f'{dev}_support.png')
 # Canonical IDs uniformly sampled, at most8, including fallbacks.
 audit=json.loads((E/'runs'/f'{dev}_F2/source_audit.json').read_text());meta=json.loads((A/'inputs'/dev/'input_manifest.json').read_text());canvas=Image.new('RGB',(1150,185*len(audit)),'white');dc=ImageDraw.Draw(canvas)
 for rowi,r in enumerate(audit):
  base=rowi*185;dc.text((4,base+4),f"ID {r['local_id']} face {r['face']}"+(' FALLBACK' if r['fallback'] else ''),font=font,fill='black')
  for j,(t,pix,w) in enumerate(zip(r['source_times'],r['source_pixels'],r['weights'])):
   rgb=cv2.imread(meta['frame_paths'][t])[...,::-1];y,x=divmod(pix,640);patch=Image.fromarray(rgb).crop((x-12,y-12,x+13,y+13)).resize((112,112));canvas.paste(patch,(j*140+6,base+32));dr2=ImageDraw.Draw(canvas);dr2.line((j*140+56,base+88,j*140+68,base+88),fill='cyan',width=2);dr2.line((j*140+62,base+82,j*140+62,base+94),fill='cyan',width=2);dr2.text((j*140+5,base+150),f"t{meta['timestamp_seconds'][t]:g} w{w:.2f}",font=small,fill='black')
 canvas.save(out/f'{dev}_sources.png')
print(out)
