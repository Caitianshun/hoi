"""Read inherited sources and masks; freeze fixed illustrations before training."""
from common import *
import cv2,numpy as np,time,csv,subprocess
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def indices(n,k):return [int(i*(n-1)//(k-1)) for i in range(k)]
def window(mask,foreground,size=128):
    h,w=mask.shape;inner=np.zeros_like(mask,np.uint8);inner[size//2:h-size//2,size//2:w-size//2]=1
    if foreground:
        y,x=np.nonzero(mask);assert len(x);cx,cy=np.median(x),np.median(y);j=np.argmin((x-cx)**2+(y-cy)**2);cx,cy=int(x[j]),int(y[j])
    else:
        distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5);distance[inner==0]=-1;cy,cx=np.unravel_index(np.argmax(distance),distance.shape)
    x0=int(np.clip(cx-size//2,0,w-size));y0=int(np.clip(cy-size//2,0,h-size));return [x0,y0,x0+size,y0+size]
def run():
    started=time.monotonic();cv2.setNumThreads(2);base=OLD/'inputs/hos_backpack';m=read(base/'manifest.json');ev=read(base/'evaluation_manifest.json')
    if (RUN/'initialization_support.json').exists():
        finalize_examples(m,ev)
        return
    completion=read(OLD/'completion.json');assert completion['status']=='completed'
    selected=np.load(m['point_cloud']['npz_path']);raw=np.load(m['point_cloud']['raw_path']);choice=np.load(m['point_cloud']['selected_indices'])
    assert sha(m['point_cloud']['npz_path'])==m['point_cloud']['npz_sha256'];assert np.array_equal(selected['raw_selected_indices'],choice)
    for k in selected.files:
        if k!='raw_selected_indices':assert np.array_equal(selected[k],raw[k][choice]),k
    frames=m['frames'];train_ids={int(f['frame_id']) for f in frames};assert set(selected['source_frame_ids'].reshape(-1)).issubset(train_ids)
    fixed=indices(len(frames),8);masks=[];rows=[];figures=[]
    image_dir=RUN/'diagnostics/input_support';image_dir.mkdir(exist_ok=True)
    support_sources={}
    for i,f in enumerate(frames):
        assert sha(f['image_path'])==f['image_sha256'] and sha(f['mask_path'])==f['mask_sha256']
        mask=cv2.imread(f['mask_path'],cv2.IMREAD_GRAYSCALE);assert mask is not None and mask.dtype==np.uint8 and mask.shape==(f['height'],f['width'])
        fg=mask>=128;masks.append(fg);observations={};coverage={}
        for label,z in [('raw',raw),('selected',selected)]:
            points=[]
            for side in [0,1]:
                good=(z['label']==1)&(z['source_frame_ids'][:,side]==int(f['frame_id']));points.append(z['source_keypoints'][good,side])
            points=np.concatenate(points);observations[label]=len(points);canvas=np.zeros_like(mask)
            for x,y in np.rint(points).astype(int):cv2.circle(canvas,(x,y),8,1,-1)
            coverage[label]=float(((canvas>0)&fg).sum()/fg.sum()) if fg.any() else None
            if label=='selected':support_sources[i]=points
        rows.append(dict(frame_id=f['frame_id'],source_frame_index=int(f['frame_id']),time_units='nominal frame index, not seconds',width=f['width'],height=f['height'],threshold=128,
            fg_pixels=int(fg.sum()),bg_pixels=int((~fg).sum()),fg_fraction=float(fg.mean()),empty_fg=not bool(fg.any()),empty_bg=bool(fg.all()),
            raw_source_keypoints=observations['raw'],selected_source_keypoints=observations['selected'],raw_support_within_8px=coverage['raw'],selected_support_within_8px=coverage['selected']))
    extra=sorted(set([int(np.argmin([r['fg_pixels'] for r in rows])),int(np.argmax([r['fg_pixels'] for r in rows]))])-set(fixed))
    for i in fixed+extra:
        f=frames[i];rgb=cv2.imread(f['image_path'])[...,::-1];fg=masks[i];overlay=rgb.copy();overlay[fg]=(overlay[fg]*.65+np.array([255,60,60])*.35).astype(np.uint8)
        for x,y in np.rint(support_sources[i]).astype(int):cv2.circle(overlay,(x,y),3,(30,255,70),-1)
        panels=[Image.fromarray(rgb),Image.fromarray((fg*255).astype(np.uint8)).convert('RGB'),Image.fromarray(overlay)]
        w=420;h=round(rgb.shape[0]*w/rgb.shape[1]);card=Image.new('RGB',(w*3,h+36),'white');draw=ImageDraw.Draw(card)
        for col,pic in enumerate(panels):card.paste(pic.resize((w,h)),(col*w,36));draw.text((col*w+6,8),f"{f['frame_id']} | {['RGB','mask >=128','accepted selected sources'][col]}",fill='black')
        path=image_dir/f"mask_source_{f['frame_id']}.jpg";card.save(path,quality=94,subsampling=0);figures.append(dict(**identity(path),frame_id=f['frame_id'],selection='uniform8' if i in fixed else 'training mask area extremum',source_keypoints='Only observations actually accepted from this frame; no all-time point projection'))
    subset=selected['label']==1;xyz=selected['xyz'][subset];times=selected['source_frame_ids'][subset].mean(1)
    fig=plt.figure(figsize=(10,4));ax=fig.add_subplot(121,projection='3d');sc=ax.scatter(*xyz.T,c=times,s=1,cmap='viridis');ax.set(xlabel='X',ylabel='Y',zlabel='Z',title='Selected FG points by mean source frame')
    ax=fig.add_subplot(122);sc=ax.scatter(xyz[:,0],xyz[:,2],c=times,s=2,cmap='viridis');ax.set(xlabel='X',ylabel='Z',title='World-space XZ projection');fig.colorbar(sc,ax=ax,label='Mean nominal source frame index');fig.tight_layout();path=image_dir/'foreground_source_time.png';fig.savefig(path,dpi=170);plt.close(fig);figures.append(identity(path))
    stats={}
    for label,z in [('raw',raw),('selected',selected)]:
        stats[label]={}
        for region,val in [('foreground',1),('background',0)]:
            use=z['label']==val
            stats[label][region]=dict(points=int(use.sum()),reprojection_px=quantiles(z['reprojection_px'][use]),parallax_deg=quantiles(z['parallax_deg'][use]),
                source_frame_gap=quantiles(np.abs(np.diff(z['source_frame_ids'][use],axis=1))),source_mean_frame_index=quantiles(z['source_frame_ids'][use].mean(1)))
    save_json(RUN/'initialization_support.json',dict(status='completed',source_manifest=identity(base/'manifest.json'),source_policy=read(OLD/'protocol/hos_triangulation_policy.json'),
        inherited_triangulation=read(base/'initialization.json'),inherited_sampling={k:v for k,v in read(base/'sampling_and_support.json').items() if k!='foreground_source_support'},
        source_identity=[identity(base/n) for n in ['initial_points.npz','initial_points_100k.npz','selected_raw_indices.npy','sampling_and_support.json']],
        stats=stats,frames=rows,figures=figures,all_source_ids_training_only=True,selected_matches_raw_indices=True,mask_threshold=128,uniform_training_indices=fixed,additional_area_extrema_indices=extra,
        no_masks_modified=True,scope='8px circular source-observation support proxy; quasi-static triangulation union across time, not canonical dynamic geometry or material correspondence',seconds=time.monotonic()-started))
    finalize_examples(m,ev)
def finalize_examples(m,ev):
    # Reuse the completed source audit when only a downstream manifest failed.
    base=OLD/'inputs/hos_backpack';frames=m['frames'];audit=read(RUN/'initialization_support.json')
    fixed=audit['uniform_training_indices'];extra=audit['additional_area_extrema_indices']
    assert audit['source_manifest']['sha256']==sha(base/'manifest.json')
    # Freeze illustration IDs and tiny-array windows from masks, without model scores.
    chosen=[]
    for i in indices(len(ev['frames']),4):
        f=ev['frames'][i];mask=cv2.imread(f['mask_path'],0)>=128;chosen.append(dict(dataset='hos_backpack',frame_id=f['frame_id'],image_size=[f['width'],f['height']],windows={'foreground':window(mask,True),'background':window(mask,False)}))
    for dev in ['dev1','dev2']:
        e=read(OLD/'inputs'/f'behave_{dev}'/'evaluation_manifest.json');f=e['frames'][(len(e['frames'])-1)//2]
        z=np.load(f['regions']['path']);mask=z['entity_labels']>0
        chosen.append(dict(dataset='behave_'+dev,frame_id=f['frame_id'],image_size=[f['width'],f['height']],windows={'foreground':window(mask,True),'background':window(mask,False)}))
    crops=[]
    for f in ev['frames']:
        fg=cv2.imread(f['mask_path'],0)>=128;y,x=np.nonzero(fg);margin=32
        crops.append(dict(frame_id=f['frame_id'],crop_bounds_xyxy=[max(0,int(x.min())-margin),max(0,int(y.min())-margin),min(f['width'],int(x.max())+margin+1),min(f['height'],int(y.max())+margin+1)]))
    save_json(RUN/'protocol/fixed_examples.json',dict(status='frozen_before_training',created_unix=time.time(),training_frame_ids=[frames[i]['frame_id'] for i in fixed],
        gradient_probe_frame_ids=[frames[i]['frame_id'] for i in indices(len(frames),4)],hos_retained_frame_ids=[f['frame_id'] for f in ev['frames']],hos_crops=crops,
        small_array_windows=chosen,window_rule='128 square centered on FG pixel nearest FG median or maximum BG distance inside image margins; fixed before new training outcomes',
        additional_mask_extrema_frame_ids=[frames[i]['frame_id'] for i in extra]))
    assets=[base/'manifest.json',base/'evaluation_manifest.json',base/'initial_points_100k.npz',base/'selected_raw_indices.npy',OLD/'runs/hos_backpack_formal/effective_config.json']
    assets += [OLD/'runs/hos_backpack_formal'/n for n in ['checkpoint_coarse_003000.pt','checkpoint_fine_001000.pt','checkpoint_fine_014000.pt']]
    assets += [Path(f[k]) for f in frames for k in ['image_path','mask_path']]
    save_json(RUN/'protocol/inherited_assets.json',dict(status='verified',assets=[identity(p) for p in assets],V3_completion=identity(OLD/'completion.json')))
    (RUN/'MISSING_ASSETS.md').write_text('# V4 资产检查\n\n当前两个训练分支所需粗阶段、fine1000和终态检查点、初始点云、采样索引、训练RGB及mask均已找到并核对。未修改任何旧资产。原H1历史日志缺批次ID及独立纯RGB值，首次fine历史总损失不能直接严格对齐；这不阻塞合法分叉。\n')
    print('Input support, mask audit and fixed-example manifest complete',flush=True)
if __name__=='__main__':run()
