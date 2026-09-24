"""Plot only the six frozen frames from stored audit arrays; no rerun of losses."""
from pathlib import Path
import json,numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).resolve().parents[1]
cfg=json.loads((E/'protocol/frozen_visibility.json').read_text());sel=json.loads((E/'protocol/frame_selection.json').read_text())
for dev in ['dev1','dev2']:
    manifest=json.loads((Path(cfg['inputs'][dev]['base'])/'input_manifest.json').read_text())
    fig,axs=plt.subplots(3,2,figsize=(8.8,8.2),layout='constrained')
    for row,i in enumerate(sel['sequences'][dev]['frames']):
        a=np.load(E/f'mask_ledger/{dev}_{i:03d}.npz');lab=a['entity_labels'];v=a['complete_vertices_projection'];s=a['complete_samples_projection'];uv=a['positive_uv'];buv=a['background_query_uv']
        rgb=cv2.imread(manifest['frame_paths'][i])[...,::-1]/255
        overlay=rgb.copy()
        for val,color in [(1,np.array([.88,.3,.45])),(2,np.array([.1,.8,.85]))]:
            z=lab==val;overlay[z]=.72*overlay[z]+.28*color
        left,right=axs[row];left.imshow(overlay)
        if dev=='dev1':
            h=cv2.convexHull(v.astype(np.float32))[:,0];h=np.vstack([h,h[0]]);left.plot(h[:,0],h[:,1],c='#ffff33',lw=1.1)
        else:left.scatter(s[:,0],s[:,1],s=.7,c='#ffff33',alpha=.5)
        pos=left.scatter(uv[:,0],uv[:,1],s=11,c=a['positive_residual_px'],cmap='viridis',vmin=0,vmax=5,edgecolor='white',linewidth=.3)
        left.set_title(f'{dev} frame {i}   RGB and positive queries',fontsize=10)
        right.imshow(a['background_distance_px'],cmap='Greys',vmin=0,vmax=20)
        for val,color in [(1,'#d84670'),(2,'#20adbe')]:right.contour(lab==val,levels=[.5],colors=[color],linewidths=.8)
        neg=right.scatter(buv[:,0],buv[:,1],s=9 if dev=='dev1' else 2,c=a['background_raw_distance_px'],cmap='inferno',vmin=0,vmax=5)
        right.set_title('EDT of background and negative queries',fontsize=10)
        ys,xs=np.nonzero(lab==2);xs=np.r_[xs,buv[:,0]];ys=np.r_[ys,buv[:,1]]
        low=np.array([xs.min(),ys.min()])-15;high=np.array([xs.max(),ys.max()])+15
        center=(low+high)/2;half=np.maximum((high-low)/2,35);half[0]=max(half[0],half[1]*1.25);half[1]=max(half[1],half[0]/1.25)
        for ax in [left,right]:ax.set_xlim(max(0,center[0]-half[0]),min(639,center[0]+half[0]));ax.set_ylim(min(479,center[1]+half[1]),max(0,center[1]-half[1]));ax.set_aspect('equal');ax.tick_params(labelsize=8)
    fig.colorbar(pos,ax=axs[:,0],orientation='horizontal',fraction=.025,pad=.015,label='Positive distance  pixels   5 includes larger values')
    fig.colorbar(neg,ax=axs[:,1],orientation='horizontal',fraction=.025,pad=.015,label='Background distance  pixels   5 includes larger values')
    fig.savefig(E/f'mask_ledger/{dev}_frozen_frames.png',dpi=185);plt.close(fig)
