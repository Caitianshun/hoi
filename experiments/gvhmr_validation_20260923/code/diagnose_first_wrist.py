"""Frozen-output diagnostic only: first-frame wrist 2D and camera-space disagreement."""
from pathlib import Path
import hashlib,json,numpy as np,torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/cai_tianshun/Project/HOI');E=R/'experiments/gvhmr_validation_20260923';O=E/'reference_quality';g=np.load(E/'run01/raw_geometry.npz');a=np.load(O/'joint_comparison.npz');C=g['c2w'];K=g['K'];k=torch.load(E/'run01/vitpose.pt',map_location='cpu',weights_only=True).numpy()
def uv(x):h=x@K.T;return h[:2]/h[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows={}
for j,name in [(9,'left_wrist'),(10,'right_wrist')]:
 ref=(a['reference_coco17_world_m'][0,j]-C[:3,3])@C[:3,:3];pred=(a['raw_coco17_world_m'][0,j]-C[:3,3])@C[:3,:3]; pu,ru=uv(pred),uv(ref);delta=pred-ref
 rows[name]={'coco_joint_index':j,'raw_camera_xyz_m':pred.tolist(),'reference_camera_xyz_m':ref.tolist(),'raw_uv_px':pu.tolist(),'reference_uv_px':ru.tolist(),'vitpose_uv_conf':k[0,j].tolist(),'raw_minus_reference_xyz_cm':(100*delta).tolist(),'raw_reference_3d_error_cm':float(100*np.linalg.norm(delta)),'raw_reference_xy_plane_error_cm':float(100*np.linalg.norm(delta[:2])),'raw_reference_uv_error_px':float(np.linalg.norm(pu-ru)),'vitpose_reference_uv_error_px':float(np.linalg.norm(k[0,j,:2]-ru)),'raw_vitpose_uv_error_px':float(np.linalg.norm(k[0,j,:2]-pu))}
rec={'status':'frozen_evaluation_diagnostic_only','inference_updated':False,'frame':0,'actual_timestamp_seconds':float(g['timestamp_seconds'][0]),'reference_nominal_seconds':float(a['reference_nominal_seconds'][0]),'rows':rows,'script_sha256':sha(__file__),'visibility_interpretation':'Only one lowered glove/forearm is clearly visible in this RGB; COCO left wrist tracks that side. The other wrist is strongly overlapped with body/arm, anatomical side and exact location cannot be certified from RGB. Reference far wrist projected near right torso/thigh is registration hypothesis, not visible 2D ground truth.','cause_limit':'Nonzero2D and Z disagreement both contribute. Cannot label this pure depth failure or prove reference exact. XY/Z are camera-coordinate components, not separate causal sources.'};(O/'first_wrist_diagnostic.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
im=Image.open(R/'experiments/mosca_baseline_20260922/common_input/images/00000.png');fig,axes=plt.subplots(1,2,figsize=(9.2,5.5),dpi=150)
for ax,name in zip(axes,['left_wrist','right_wrist']):
 ax.imshow(im);row=rows[name]
 for label,key,color,marker in [('Raw projection','raw_uv_px','#2369be','s'),('ViTPose2D','vitpose_uv_conf','#e99317','x'),('Fitted reference','reference_uv_px','#138e49','o')]:
  p=row[key];ax.scatter(p[0],p[1],s=80,c=color,marker=marker,linewidths=2,label=label,zorder=5)
 ax.set_xlim(215,350);ax.set_ylim(400,235);ax.set_title(name.replace('_',' ').title()+f'\nViTPose score {row["vitpose_uv_conf"][2]:.3f}');ax.set_xlabel('Input pixel u');ax.set_ylabel('Input pixel v');ax.legend(loc='lower left',fontsize=8)
fig.suptitle('Frame0 wrist disagreement | fitted reference is not visible-keypoint ground truth',fontsize=10);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(O/'first_wrist_diagnostic.png');plt.close(fig)
s=json.loads((O/'summary.json').read_text());s['first_wrist_followup']={'script_sha256':sha(__file__),'json_sha256':sha(O/'first_wrist_diagnostic.json'),'inference_modified':False};s['outputs']['first_wrist_diagnostic.png']=sha(O/'first_wrist_diagnostic.png');(O/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');print(json.dumps(rows,indent=2))
