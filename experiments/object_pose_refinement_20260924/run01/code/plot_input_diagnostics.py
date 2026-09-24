#!/usr/bin/env python3
"""Portable PNG evidence from legal input and ONE frozen P1 prediction.

No evaluation-reference loader. Two-dimensional mask fitting is not 3D accuracy.
If the selection rejects every P1 path, show the old result and explicit N/A;
never substitute an unaccepted candidate merely to populate a figure.
"""
from pathlib import Path
import argparse,datetime,hashlib,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.spatial.transform import Rotation
ROOT=Path('/home/cai_tianshun/Project/HOI')
OLD=ROOT/'experiments/structured_hoi_20260923'
E=ROOT/'experiments/object_pose_refinement_20260924/run01'
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'font.size':10,'savefig.facecolor':'white'})
OLD_COLOR=np.array([189,49,178],np.uint8)
P1_COLOR=np.array([0,174,194],np.uint8)
MASK_COLOR=np.array([85,210,87],np.uint8)
OCC_COLOR=np.array([240,158,34],np.uint8)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,record):Path(path).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
def project(points,K):
 h=points@K.T
 return h[:,:2]/np.maximum(h[:,2:],.01)
def raster(V,F,R,t,K,hw):
 uv=project(np.asarray(V,dtype=float)@np.asarray(R,dtype=float).T+np.asarray(t,dtype=float),K);out=np.zeros(hw,np.uint8)
 for triangle in np.rint(np.clip(uv[F],-2000,3000)).astype(np.int32):cv2.fillConvexPoly(out,triangle,1)
 return out
def iou(pred,label):
 p=(pred>0)&((label==0)|(label==2));target=label==2
 return float((p&target).sum()/max(1,(p|target).sum()))
def draw_overlay(rgb,mask,label,color):
 image=rgb.copy();target=(label==2).astype(np.uint8)
 for c in cv2.findContours(target,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]:cv2.polylines(image,[c],True,tuple(map(int,MASK_COLOR)),1,cv2.LINE_AA)
 # Full predicted contour; human/unknown pixels are estimated occlusion only.
 for contour in cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]:
  pts=contour[:,0]
  for j,(a,b) in enumerate(zip(pts,np.roll(pts,-1,axis=0))):
   mid=np.rint((a+b)/2).astype(int);occluded=label[mid[1],mid[0]] not in (0,2)
   if not occluded or j%8<4:cv2.line(image,tuple(a),tuple(b),tuple(map(int,OCC_COLOR if occluded else color)),2,cv2.LINE_AA)
 return image
def crop_box(label,masks):
 union=label==2
 for mask in masks:
  if mask is not None:union|=mask>0
 y,x=np.where(union);H,W=label.shape
 if not len(x):return [0,0,W,H]
 x0,x1=int(x.min()),int(x.max()+1);y0,y1=int(y.min()),int(y.max()+1)
 pad=max(36,int(max(x1-x0,y1-y0)*.25));cx=(x0+x1)/2;cy=(y0+y1)/2;bw=max(180,x1-x0+2*pad);bh=max(180,y1-y0+2*pad)
 return [max(0,int(cx-bw/2)),max(0,int(cy-bh/2)),min(W,int(cx+bw/2)),min(H,int(cy+bh/2))]
def shade_unknown(ax,t,visible):
 edges=np.r_[t[0],(t[:-1]+t[1:])/2,t[-1]];ii=np.flatnonzero(~visible)
 if not len(ii):return
 groups=np.split(ii,np.flatnonzero(np.diff(ii)>1)+1)
 for g in groups:ax.axvspan(edges[g[0]],edges[g[-1]+1],color='#777777',alpha=.13,zorder=0)
def kinematics(a):
 R=a['R_camera'].astype(float);t=a['t_camera_m'].astype(float);ts=a['timestamp_seconds'];dt=np.diff(ts);center=a['canonical_vertices_m'].mean(0)@R.transpose(0,2,1)+t
 # The broadcast dot above yields [T,3]; no world-reference alignment.
 return center[:,2],np.linalg.norm(np.diff(t,axis=0),axis=1)/dt,Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).magnitude()/dt

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dev',required=True,choices=['dev1','dev2']);p.add_argument('--selection',required=True,type=Path);args=p.parse_args()
 selection_path=args.selection.resolve();selected=json.loads(selection_path.read_text());assert selected.get('dev')==args.dev
 assert selected.get('no_independent_reference_used') is True
 keypath=E/'pose'/args.dev/'keyframes.json';keys=json.loads(keypath.read_text());mp=Path(keys['source_manifest']);m=json.loads(mp.read_text());assert m['role']=='input_only' and m['camera_id']==0
 segpath=Path(keys['segmentation']);labels=np.load(segpath)['entity_labels'];ts=np.array(m['timestamp_seconds']);K=np.array(m['K']);H,W=labels.shape[1:]
 oldpath=OLD/('object_init/run03/object_init.npz' if args.dev=='dev1' else 'object_init/dev2_refined01/object_init.npz');old=dict(np.load(oldpath));assert np.array_equal(old['timestamp_seconds'],ts) and np.allclose(old['K'],K)
 new=None;newpath=None;choice=selected.get('selected')
 if choice is not None:
  assert selected['status']=='accepted_for_S1_star','Only explicitly frozen accepted P1 may be used'
  newpath=Path(choice['pose_file']);assert sha(newpath)==choice['pose_sha256'];new=dict(np.load(newpath));assert np.array_equal(new['timestamp_seconds'],ts)
  for name in ['canonical_vertices_m','faces','K','c2w','observed_mask']:assert np.array_equal(old[name],new[name]),f'Changed fixed input {name}'
 visible=old['observed_mask'].astype(bool);V=old['canonical_vertices_m'];F=old['faces'];predictions={'old':[], 'new':[]};scores={'old':[], 'new':[]}
 for name,a in [('old',old),('new',new)]:
  if a is None:continue
  for i in range(len(ts)):
   mask=raster(V,F,a['R_camera'][i],a['t_camera_m'][i],K,(H,W));predictions[name].append(mask);scores[name].append(iou(mask,labels[i]))
  scores[name]=np.array(scores[name])
 # Same source/prediction crop per column; rows never silently use different zoom.
 kf=[x['frame_index'] for x in keys['keyframes']];fig,axes=plt.subplots(3,len(kf),figsize=(15.5,9),squeeze=False)
 crops=[]
 for col,f in enumerate(kf):
  assert sha(m['frame_paths'][f])==m['frame_sha256'][f]
  rgb=cv2.cvtColor(cv2.imread(m['frame_paths'][f]),cv2.COLOR_BGR2RGB);pm=predictions['new'][f] if new is not None else None;box=crop_box(labels[f],[predictions['old'][f],pm]);crops.append(box);x0,y0,x1,y1=box
  panels=[rgb,draw_overlay(rgb,predictions['old'][f],labels[f],OLD_COLOR),draw_overlay(rgb,pm,labels[f],P1_COLOR) if pm is not None else np.full_like(rgb,247)]
  for row,image in enumerate(panels):
   ax=axes[row,col];ax.imshow(image[y0:y1,x0:x1]);ax.set_xticks([]);ax.set_yticks([])
   for spine in ax.spines.values():spine.set_visible(False)
   if row==0:ax.set_title(f'帧 {f}  /  {ts[f]:.3f} s',fontproperties=FONT,fontsize=11)
   if row==1:ax.text(.03,.96,f'IoU {scores["old"][f]:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.88,edgecolor='none'))
   if row==2:
    if new is not None:ax.text(.03,.96,f'IoU {scores["new"][f]:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.88,edgecolor='none'))
    else:ax.text(.5,.5,'无合格 P1\n不替代展示拒绝候选',transform=ax.transAxes,ha='center',va='center',fontproperties=FONT,fontsize=11,color='#555555')
  for row,label in enumerate(['输入 RGB','旧物体初值','冻结选定 P1']):
   if col==0:axes[row,col].set_ylabel(label,fontproperties=FONT,fontsize=11)
 title='箱体事件' if args.dev=='dev1' else '木椅事件';fig.suptitle(f'{title}：固定关键帧的输入与物体投影',fontproperties=FONT,fontsize=17,y=.985)
 fig.text(.5,.036,'绿色：输入 SAM2 边界；紫色：旧初值；青色：P1；橙色虚线：落于人体/未知标签的完整投影轮廓。',ha='center',fontproperties=FONT,fontsize=10)
 fig.text(.5,.013,'同列使用完全相同近景范围。IoU 仅评价输入二维拟合，不证明三维位姿或材料身份正确；人体标签也是估计。',ha='center',fontproperties=FONT,fontsize=10,color='#333333')
 fig.subplots_adjust(left=.065,right=.995,top=.93,bottom=.075,hspace=.12,wspace=.07)
 out=E/'output/figures';out.mkdir(exist_ok=True,parents=True);comparison=out/f'{args.dev}_input_pose_comparison.png';fig.savefig(comparison,dpi=180);plt.close(fig)
 # Entire fixed input sequence, including every weak observation interval.
 fig,ax=plt.subplots(4,1,figsize=(12.5,10),sharex=True);mid=(ts[:-1]+ts[1:])/2
 for name,a,color,label in [('old',old,OLD_COLOR/255,'旧初值'),('new',new,P1_COLOR/255,'冻结选定 P1')]:
  if a is None:continue
  z,v,w=kinematics(a);ax[0].plot(ts,scores[name],color=color,lw=1.5,label=label);ax[1].plot(ts,z,color=color,lw=1.5);ax[2].plot(mid,v,color=color,lw=1.4);ax[3].plot(mid,w,color=color,lw=1.4)
 for a in ax:
  shade_unknown(a,ts,visible);a.grid(alpha=.2);a.set_xlim(ts[0],ts[-1]);a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
 ax[0].set_ylim(0,1.03);ax[0].set_ylabel('输入轮廓 IoU',fontproperties=FONT);ax[1].set_ylabel('模板均值中心深度 (m)',fontproperties=FONT);ax[2].set_ylabel('平移速度 (m/s)',fontproperties=FONT);ax[3].set_ylabel('角速度 (rad/s)',fontproperties=FONT);ax[3].set_xlabel('实际输入时间 (s)',fontproperties=FONT);ax[0].legend(prop=FONT,loc='lower right',ncol=2)
 oldmean=float(np.mean(scores['old'][visible]));newtxt=f'{np.mean(scores["new"][visible]):.3f}' if new is not None else 'N/A（无合格 P1）'
 fig.suptitle(f'{title}：全段输入侧诊断\n固定 {int(visible.sum())} 个有效帧 mean IoU：旧 {oldmean:.3f}，P1 {newtxt}',fontproperties=FONT,fontsize=15,y=.98)
 fig.text(.5,.045,'灰色区间：旧协议预先固定的低观测帧；未删除。速度使用真实时间差。',ha='center',fontproperties=FONT,fontsize=10)
 fig.text(.5,.022,'这些曲线仅为输入拟合、估计运动和稳定性诊断；深度/速度不是独立参考误差，平滑也不证明运动准确。',ha='center',fontproperties=FONT,fontsize=10,color='#333333')
 fig.subplots_adjust(left=.11,right=.985,top=.895,bottom=.12,hspace=.11);curves=out/f'{args.dev}_input_score_curves.png';fig.savefig(curves,dpi=180);plt.close(fig)
 metadata=dict(dev=args.dev,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_path=str(selection_path),selection_sha256=sha(selection_path),selected_pose_path=str(newpath) if newpath else None,selected_pose_sha256=sha(newpath) if newpath else None,old_pose_path=str(oldpath),old_pose_sha256=sha(oldpath),input_manifest_sha256=sha(mp),segmentation_sha256=sha(segpath),keyframes=kf,crops_xyxy=crops,script_sha256=sha(__file__),fixed_visible_frames=np.flatnonzero(visible).tolist(),mean_visible_iou_old=oldmean,mean_visible_iou_new=float(scores['new'][visible].mean()) if new is not None else None,evaluation_references_read=False,prediction_selection_modified=False,outputs={str(p):sha(p) for p in [comparison,curves]},caption='Input 2D fit and estimated motion only; not independent 3D accuracy or material identity.')
 dump(out/f'{args.dev}_input_figures.json',metadata);print(json.dumps(metadata,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
