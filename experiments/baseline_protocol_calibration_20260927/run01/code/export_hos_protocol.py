#!/usr/bin/env python3
"""Freeze HOSNeRF Backpack split/cameras and build train-RGB-only triangulation.

No test RGB is opened for feature extraction, matching, point geometry or color.
Published camera poses are an explicitly separate all-video-derived input path.
The foreground two-view points are approximate initial geometry, not motion GT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time

import cv2
import numpy as np
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(2**20),b''): h.update(c)
    return h.hexdigest()


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def camera_record(data, stem, first, last, size):
    v=data[stem]; K=np.asarray(v['intrinsics'],np.float64)
    E=np.asarray(v['scaleworld_to_camera'],np.float64)
    return dict(frame_id=stem,frame_number=int(stem),physical_time=int(stem),time_seconds=int(stem),
                time=(int(stem)-first)/(last-first),width=size[0],height=size[1],
                K=K.tolist(),w2c=E.tolist(),c2w=np.linalg.inv(E).tolist(),
                time_kind='nominal source frame index; exposure timestamps/fps unpublished',
                extrapolation=not first<=int(stem)<=last)


def export(args):
    data=args.data.resolve(); output=args.output.resolve(); official=args.official.resolve()
    dst=output/'inputs'/'hos_backpack'; dst.mkdir(parents=True,exist_ok=True)
    images=sorted((data/'images').glob('*.png')); ids=[p.stem for p in images]
    assert ids==[f'{i:05d}' for i in range(len(ids))]
    n=len(ids); tests=list(range(0,n,n//16))[:16]; train=[i for i in range(n) if i not in tests]
    # Header inspection only. Evaluation RGB content is not read here.
    size=Image.open(images[train[0]]).size
    camera_path=data/'cameras_scaleworld.pkl'
    with open(camera_path,'rb') as f: cameras=pickle.load(f)
    assert sorted(cameras)==ids
    first,last=min(train),max(train)
    common=dict(schema_version=1,dataset='HOSNeRF',scene='Backpack',
        protocol='official complete-stage human-object split',
        resolution=list(size),resize='none; native 1277x718',
        camera_convention='OpenCV +x right +y down +z forward; published scaleworld',
        camera_source=dict(path=str(camera_path),sha256=sha(camera_path),
            status='published full-video cameras; raw SfM image provenance absent; not asserted train-only'),
        time_normalization=dict(origin_frame=first,last_train_frame=last,denominator=last-first,
            formula='(source_frame_number - 1) / 282',train_only_bounds=True,
            first_test_frame_outside_train_range=True))
    rows=[]
    for i in train:
        p=images[i]; m=data/'masks'/p.name
        r=camera_record(cameras,ids[i],first,last,size)
        r.update(rgb_path=str(p),image=str(p),image_path=str(p),image_sha256=sha(p),rgb_sha256=sha(p),mask_path=str(m),mask_sha256=sha(m))
        rows.append(r)
    train_manifest={**common,'role':'training_only','split':'train','count':len(rows),'frames':rows}
    dump(dst/'manifest.json',train_manifest)
    evals=[]
    for i in tests:
        r=camera_record(cameras,ids[i],first,last,size)
        r.update(rgb_path=str(images[i]),image=str(images[i]),image_path=str(images[i]),mask_path=str(data/'masks'/images[i].name))
        evals.append(r)
    dump(dst/'evaluation_manifest.json',{**common,'split':'test','count':len(evals),'frames':evals})
    commit=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    split=dict(official_repository='https://github.com/TencentARC/HOSNeRF',official_commit=commit,
        total_frames=n,complete_human_train_ids=[ids[i] for i in train],
        complete_human_test_ids=[ids[i] for i in tests],
        scene_train_ids=[ids[i] for i in range(n) if i%10 not in (0,5)],
        scene_validation_ids=[ids[i] for i in range(5,n,10)],
        scene_test_ids=[ids[i] for i in range(0,n,10)],
        split_mismatch=True,
        human_loader='3rd_Complete_HOSNeRF/core/data/human_nerf/train.py:113-124',
        human_test_factory='3rd_Complete_HOSNeRF/core/data/create_dataset.py:47-50',
        scene_loader='3rd_Complete_HOSNeRF/src/data/data_util/nerf_360_v2.py:387-399',
        official_complete_test_metrics='3rd_Complete_HOSNeRF/src/model/mipnerf360/model.py:test_metrics')
    dump(output/'protocol'/'hos_split_manifest.json',split)
    dump(output/'protocol'/'hos_asset_sources.json',dict(
        archive_path=str(args.archive.resolve()),archive_sha256=sha(args.archive),
        official_dataset_folder='https://drive.google.com/drive/folders/1viuXcihwFpLIjl6TmLyF5VARB7GxEfEv',
        official_archive_id='1GGDy2ixG8gZydigNy5qWiL35tc4Kiq8Y',
        archive_remote_byte_identity='local user-supplied archive; remote byte equality not yet asserted',
        checkpoint_url='https://drive.google.com/file/d/125b2-_zJueftf-b-fnUkVSg2wdtvRRZt/view',
        official_commit=commit,
        assets=[dict(path=str(p),sha256=sha(p)) for p in sorted(data.iterdir()) if p.is_file() and p.name!='.DS_Store'],
        missing_published_point_tracks=True,published_pointcloud_present=False,
        no_H1_use=['mesh_infos.pkl','canonical_joints.pkl','transitions_times.json','test RGB','test masks','H0 checkpoint'],
        H1_camera_input='published scaleworld camera matrices and intrinsics only'))
    return train_manifest


def triangulate(args,manifest):
    start=time.monotonic(); cv2.setNumThreads(4); cv2.setRNGSeed(12345)
    dst=args.output.resolve()/'inputs'/'hos_backpack'
    policy=dict(seed=12345,sift_max_features=6000,train_index_pair_offsets=[1,2,4,8],
        flann_trees=5,flann_checks=64,mutual_ratio=0.75,max_reprojection_px=1.5,
        min_parallax_degrees=0.5,min_positive_depth=0.01,
        max_distance_train_camera_r95_multiplier=10.0,mask_boundary_erosion_px=3,
        background_mask_max=31,foreground_mask_min=224,
        point_identity='each accepted mutual two-view match; duplicates retained',
        color='mean of source RGB samples at both training keypoints, 0..1',
        geometry='fixed published camera triangulation; no bundle adjustment',
        dynamic_assumption='foreground treated quasi-static only within each selected pair; approximate initialization, not correspondence or geometry ground truth',
        no_test_RGB_or_mask_reads=True)
    dump(args.output/'protocol'/'hos_triangulation_policy.json',policy)
    rows=manifest['frames']; centres=np.array([np.array(r['c2w'])[:3,3] for r in rows]);center=np.median(centres,axis=0)
    bound=10*np.percentile(np.linalg.norm(centres-center,axis=1),95)
    sift=cv2.SIFT_create(nfeatures=policy['sift_max_features'])
    cache=dst/'features'; cache.mkdir(exist_ok=True)
    feature_meta=[]
    for index,r in enumerate(rows):
        p=cache/(r['frame_id']+'.npz')
        rgb=cv2.cvtColor(cv2.imread(r['rgb_path']),cv2.COLOR_BGR2RGB)
        mask=cv2.imread(r['mask_path'],cv2.IMREAD_GRAYSCALE)
        assert rgb.shape[:2]==mask.shape==(r['height'],r['width'])
        kp,desc=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None)
        xy=np.array([x.pt for x in kp],np.float32); ij=np.round(xy).astype(int)
        fg=cv2.erode((mask>=224).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        bg=cv2.erode((mask<=31).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        cls=np.full(len(kp),-1,np.int8);cls[bg[ij[:,1],ij[:,0]]]=0;cls[fg[ij[:,1],ij[:,0]]]=1
        color=rgb[ij[:,1],ij[:,0]].astype(np.float32)/255
        np.savez_compressed(p,xy=xy,descriptor=desc,label=cls,rgb=color)
        feature_meta.append(dict(frame_id=r['frame_id'],features=len(kp),foreground=int((cls==1).sum()),background=int((cls==0).sum())))
        if index%32==0: print('features',index+1,'/',len(rows),flush=True)
    # All caching is train only. Read back descriptor arrays once for matching.
    feats=[dict(np.load(cache/(r['frame_id']+'.npz'))) for r in rows]
    pts=[];colors=[];labels=[];src=[];coords=[];reprojs=[];angles=[];pair_stats=[]
    for ia,a in enumerate(rows):
        for off in policy['train_index_pair_offsets']:
            ib=ia+off
            if ib>=len(rows):continue
            b=rows[ib];fa,fb=feats[ia],feats[ib]
            for label in [0,1]:
                inda=np.where(fa['label']==label)[0];indb=np.where(fb['label']==label)[0]
                if min(len(inda),len(indb))<2:continue
                da,db=fa['descriptor'][inda],fb['descriptor'][indb]
                match=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64))
                forward=match.knnMatch(da,db,k=2);backward=match.knnMatch(db,da,k=2)
                rev={p.queryIdx:p.trainIdx for p,q in backward if p.distance<.75*q.distance}
                pairs=np.array([(inda[p.queryIdx],indb[p.trainIdx]) for p,q in forward if p.distance<.75*q.distance and rev.get(p.trainIdx)==p.queryIdx],int)
                if len(pairs)==0:continue
                ua,ub=fa['xy'][pairs[:,0]],fb['xy'][pairs[:,1]]
                Ka,Kb=np.array(a['K']),np.array(b['K']); Ea,Eb=np.array(a['w2c']),np.array(b['w2c'])
                Pa,Pb=Ka@Ea[:3],Kb@Eb[:3]
                X=cv2.triangulatePoints(Pa,Pb,ua.T,ub.T).T
                X=X[:,:3]/X[:,3:4]
                ca=X@Ea[:3,:3].T+Ea[:3,3];cb=X@Eb[:3,:3].T+Eb[:3,3]
                xa,xb=ca@Ka.T,cb@Kb.T
                err=np.maximum(np.linalg.norm(xa[:,:2]/xa[:,2:]-ua,axis=1),np.linalg.norm(xb[:,:2]/xb[:,2:]-ub,axis=1))
                ra=X-centres[ia];rb=X-centres[ib]
                cosine=np.sum(ra*rb,axis=1)/np.maximum(np.linalg.norm(ra,axis=1)*np.linalg.norm(rb,axis=1),1e-15)
                angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
                good=np.isfinite(X).all(1)&(ca[:,2]>.01)&(cb[:,2]>.01)&(err<=1.5)&(angle>=.5)&(np.linalg.norm(X-center,axis=1)<=bound)
                num=int(good.sum());pair_stats.append(dict(frame_a=a['frame_id'],frame_b=b['frame_id'],label=label,mutual_matches=len(pairs),accepted=num))
                if num:
                    pts.append(X[good]);colors.append((fa['rgb'][pairs[good,0]]+fb['rgb'][pairs[good,1]])/2)
                    labels.extend([label]*num);src.extend([(int(a['frame_id']),int(b['frame_id']))]*num)
                    coords.append(np.stack([ua[good],ub[good]],axis=1));reprojs.append(err[good]);angles.append(angle[good])
        if ia%32==0:print('matching',ia+1,'/',len(rows),'accepted',sum(map(len,pts)),flush=True)
    if not pts:raise RuntimeError('No training-only triangulated points; H1 blocked')
    xyz=np.concatenate(pts).astype(np.float32);rgb=np.concatenate(colors).astype(np.float32);label=np.array(labels,np.int8);source_ids=np.array(src,np.int32)
    np.savez_compressed(dst/'initial_points.npz',xyz=xyz,rgb=rgb,label=label,source_frame_ids=source_ids,source_keypoints=np.concatenate(coords),reprojection_px=np.concatenate(reprojs).astype(np.float32),parallax_deg=np.concatenate(angles).astype(np.float32))
    centre=np.median(xyz,axis=0);radius=1.1*np.percentile(np.linalg.norm(xyz-centre,axis=1),95)
    stats=dict(points=len(xyz),foreground_points=int((label==1).sum()),background_points=int((label==0).sum()),
        foreground_source_frames=np.unique(source_ids[label==1]).tolist(),
        xyz_min=xyz.min(0).tolist(),xyz_max=xyz.max(0).tolist(),scene_center=centre.tolist(),extent=float(radius),
        train_camera_distance_bound=float(bound),
        reprojection_px_quantiles=np.percentile(np.concatenate(reprojs),[0,50,95,100]).tolist(),
        foreground_reprojection_px_quantiles=np.percentile(np.concatenate(reprojs)[label==1],[0,50,95,100]).tolist() if (label==1).any() else None,
        wall_seconds=time.monotonic()-start,pointcloud_sha256=sha(dst/'initial_points.npz'),policy=policy,
        features=feature_meta,pairs=pair_stats,
        caveat='Foreground mask affiliation and two-view reprojection are support proxies, not proof of true material tracks; union spans moving foreground states.')
    dump(dst/'initialization.json',stats)
    manifest['initialization']=dict(path=str(dst/'initial_points.npz'),sha256=stats['pointcloud_sha256'],points=stats['points'],extent=stats['extent'],source='fixed-camera triangulation exclusively from frozen training RGB and masks')
    manifest['point_cloud']={**manifest['initialization'],'npz_path':str(dst/'initial_points.npz')}
    manifest['scene_extent']=stats['extent'];manifest['scene_center']=stats['scene_center'];manifest['aabb']=[stats['xyz_min'],stats['xyz_max']]
    dump(dst/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in stats.items() if k not in ['features','pairs','policy','foreground_source_frames']},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--official',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--triangulate',action='store_true')
    args=p.parse_args(); manifest=export(args)
    if args.triangulate:triangulate(args,manifest)

if __name__=='__main__':main()
