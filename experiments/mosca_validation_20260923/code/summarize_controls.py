"""Aggregate completed controlled runs and matched fixed crops; no alignment."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import cv2
ROOT=Path('/home/cai_tianshun/Project/HOI');OLD=ROOT/'experiments/mosca_baseline_20260922'
EXP=ROOT/'experiments/mosca_validation_20260923'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--branches',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 rows=[];renders=[]
 cases=[('old_pilot',OLD/'evaluation/mosca_cotracker_final',OLD/'render_summary',OLD/'mosca_cotracker',OLD/'mosca_cotracker/diagnostics')]
 for name in a.branches:
  b=EXP/name
  assert read(b/'pipeline.json')['status']=='completed'
  cases.append((name,b/'evaluation',b/'render_summary',b/'model',b/'diagnostics'))
 reference_identity=None
 for name,e,r,t,d in cases:
  report=read(e/'three_dimensional/report.json');rgb=read(r/'summary.json');train=read(t/'run.json');result=report['results']
  with np.load(e/'evaluation_bundle.npz') as z:
   identity=(z['reference'].tobytes(),z['frame_times'].tobytes(),z['query_id'].tobytes())
   if reference_identity is None:reference_identity=identity
   assert reference_identity==identity
  def stat(key,group):return next(x for x in result[key] if x.get('group')==group and x.get('visibility')=='all')
  row={'name':name,'absolute':stat('summaries','all'),'displacement':stat('displacement_summaries','all'),
       'object_absolute':stat('summaries','entity:object'),'object_displacement':stat('displacement_summaries','entity:object'),
       'hand_displacement':stat('displacement_summaries','entity:hand'),'relative':result['relative_pair_summaries'][0],
       'rgb':rgb['summary'],'stages':train['stages'],'peak_allocated_bytes':train['peak_allocated_bytes'],
       'camera_changes':train['camera_max_abs_changes'],'evaluation_sha256':sha(e/'three_dimensional/report.json'),
       'render_sha256':sha(r/'summary.json'),'model_path':str(t),'diagnostic_path':str(d)}
  with np.load(d/'query_trajectories.npz') as z:
   row['source_dynamic_fraction']=z['source_dynamic_fraction'].tolist()
   row['query_max_displacement_m']=np.linalg.norm(z['predicted']-z['predicted'][:1],axis=-1).max(0).tolist()
  rows.append(row);renders.append((name,d/'rgb'))
 masks=np.load(OLD/'segmentation/segmentation.npz')['entity_labels'];m=read(OLD/'common_input/input_manifest.json')
 panels=[]
 for frame in [0,16,31,113]:
  rgb=cv2.imread(m['frame_paths'][frame]);yy,xx=np.where(masks[frame]>0)
  box=(max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(rgb.shape[1],int(xx.max())+13),min(rgb.shape[0],int(yy.max())+13))
  x0,y0,x1,y1=box;tiles=[]
  for name,im in [('Input',rgb)]+[(name,cv2.imread(str(folder/f'{frame:05d}.png'))) for name,folder in renders]:
   crop=im[y0:y1,x0:x1];scale=min(280/crop.shape[1],330/crop.shape[0]);crop=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)))
   tile=np.full((375,300,3),255,np.uint8);x=(300-crop.shape[1])//2;y=43+(330-crop.shape[0])//2
   tile[y:y+crop.shape[0],x:x+crop.shape[1]]=crop
   cv2.putText(tile,name,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
   cv2.putText(tile,f'frame {frame}; {m["timestamp_seconds"][frame]:.3f}s',(5,33),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1);tiles.append(tile)
  panels.append(np.concatenate(tiles,axis=1))
 cv2.imwrite(str(a.output/'matched_foreground_crops.png'),np.concatenate(panels,axis=0))
 (a.output/'summary.json').write_text(json.dumps({'status':'completed','same_reference_identity_verified':True,'runs':rows,'script_sha256':sha(Path(__file__))},indent=2)+'\n')
 print(json.dumps([{'name':x['name'],'absolute_cm':100*x['absolute']['mean_epe_m'],'displacement_cm':100*x['displacement']['mean_epe_m'],
       'relative_cm':100*x['relative']['mean_epe_m'],'object_psnr':x['rgb']['object']['pooled_psnr_db']} for x in rows],indent=2))
if __name__=='__main__':main()
