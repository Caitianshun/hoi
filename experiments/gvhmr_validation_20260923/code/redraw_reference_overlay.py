"""Layout-only redraw from already frozen evaluated joints; no scoring change."""
from pathlib import Path
import hashlib,json,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/cai_tianshun/Project/HOI');E=R/'experiments/gvhmr_validation_20260923';O=E/'reference_quality';a=np.load(O/'joint_comparison.npz');m=json.loads((R/'experiments/mosca_baseline_20260922/common_input/input_manifest.json').read_text());C=np.array(m['c2w']);K=np.array(m['K']);idx=a['reference_input_indices'];body=range(5,17)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def project(x):
 cam=(x-C[:3,3])@C[:3,:3];h=cam@K.T;return h[:,:2]/h[:,2:]
edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)];fig,axes=plt.subplots(2,4,figsize=(14,10),dpi=140)
for ax,i in zip(axes.flat,[0,16,25,31,55,65,93,113]):
 ri=int(np.where(idx==i)[0][0]);im=Image.open(m['frame_paths'][i]);ax.imshow(im)
 for pts,color,style in [(a['reference_coco17_world_m'][ri],'#0b8b3e','-'),(a['raw_coco17_world_m'][i],'#2865bd','-'),(a['official_postproc_coco17_world_m'][i],'#aa4db7','--')]:
  uv=project(pts)
  for x,y in edges:ax.plot(uv[[x,y],0],uv[[x,y],1],color=color,lw=1.1,ls=style,alpha=.9)
  ax.scatter(uv[list(body),0],uv[list(body),1],s=8,c=color)
 ax.set_xlim(170,470);ax.set_ylim(460,110);ax.set_title(f'Frame {i} | RGB {a["matched_rgb_seconds"][ri]:.3f}s\nFit nominal {a["reference_nominal_seconds"][ri]:.0f}s',fontsize=11,pad=9);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Input-camera overlay: green=fitted reference, blue=raw, purple=official postproc\nFixed crop for display only; raw and postproc nearly overlap',fontsize=12,y=.98);fig.subplots_adjust(left=.02,right=.99,top=.90,bottom=.025,hspace=.22,wspace=.065);oldhash=sha(O/'reference_overlay.png');fig.savefig(O/'reference_overlay.png');plt.close(fig)
s=json.loads((O/'summary.json').read_text());s['outputs']['reference_overlay.png']=sha(O/'reference_overlay.png');s['visual_layout_revision']={'script':str(Path(__file__).resolve()),'script_sha256':sha(__file__),'prior_overlay_sha256':oldhash,'reason':'increase row spacing to avoid lower-row title overlap; numerical scores unchanged'};(O/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
