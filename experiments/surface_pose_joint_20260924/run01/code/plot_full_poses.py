from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).resolve().parents[1]
colors=['black','#999999','#2766a2','#d16a08','#40823d','#9058a7']
a=json.loads((E/'evaluation/all_predictions_evaluation_freeze.json').read_text())
for dev in ['dev1','dev2']:
 fig,ax=plt.subplots(2,2,figsize=(10,6),layout='constrained')
 for rec,col in zip([x for x in a['frozen_predictions'] if x['dev']==dev],colors):
  p=np.load(rec['path']);R=p['R_camera'];t=p['t_camera_m'];ts=p['timestamp_seconds'];center=np.einsum('tij,j->ti',R,p['canonical_vertices_m'].mean(0))+t
  for j in range(3):ax.flat[j].plot(ts,center[:,j],color=col,label=rec['label'],lw=1.2)
  omega=np.rad2deg(Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).magnitude())/np.diff(ts);ax.flat[3].plot(ts[1:],omega,color=col,lw=1.2)
 for j,label in enumerate(['Camera centroid X (m)','Camera centroid Y (m)','Camera centroid Z (m)','Angular speed (deg/s)']):ax.flat[j].set_title(label);ax.flat[j].set_xlabel('Actual time (s)');ax.flat[j].grid(alpha=.2)
 ax.flat[0].legend(fontsize=7,ncol=2);fig.suptitle(dev+' full predictions; no reference-based selection, all frames')
 fig.savefig(E/f'visualizations/{dev}_full_pose_timeline.png',dpi=160);plt.close(fig)
