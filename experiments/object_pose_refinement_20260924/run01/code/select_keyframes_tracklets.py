#!/usr/bin/env python3
"""Frozen input-only selection. No object poses, references or heldout inputs."""
from pathlib import Path
import argparse,hashlib,json,time,datetime
import cv2,numpy as np
ROOT=Path('/home/cai_tianshun/Project/HOI'); E=ROOT/'experiments/object_pose_refinement_20260924/run01'
SOURCES={'dev1':(ROOT/'experiments/mosca_baseline_20260922/common_input/input_manifest.json',ROOT/'experiments/mosca_baseline_20260922/segmentation/segmentation.npz'),'dev2':(ROOT/'experiments/structured_hoi_20260923/data/dev2/input_manifest.json',ROOT/'experiments/structured_hoi_20260923/data/dev2/segmentation/segmentation.npz')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def candidates(gray,mask,limit=128):
    er=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))
    corners=cv2.goodFeaturesToTrack(gray,80,.01,4,mask=er)
    seeds=[] if corners is None else corners[:,0].astype(np.float32).tolist()
    yy,xx=np.where(er);pool=np.stack([xx,yy],1)[::4].astype(np.float32)
    while len(seeds)<limit and len(pool):
        if seeds:
            distances=((pool[:,None]-np.asarray(seeds)[None])**2).sum(-1).min(1)
            best=int(np.argmax(distances))
            if distances[best]<16:break
        else:best=0
        seeds.append(pool[best].tolist());pool=np.delete(pool,best,0)
    return np.array(seeds,dtype=np.float32).reshape(-1,2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--dev',choices=list(SOURCES),required=True);a=p.parse_args();start=time.perf_counter()
    out=E/'pose'/a.dev;out.mkdir(exist_ok=True);mp,sp=SOURCES[a.dev];m=json.loads(mp.read_text()); assert m['role']=='input_only' and m['camera_id']==0
    masks=np.load(sp)['entity_labels']==2;ts=np.array(m['timestamp_seconds']);metrics=[];rgb=[];descs=[]
    for i,path in enumerate(m['frame_paths']):
        assert sha(path)==m['frame_sha256'][i]
        im=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB);rgb.append(im);mask=masks[i];er=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8));gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        area=int(mask.sum());interior=int(er.sum());yy,xx=np.where(mask);bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if area else [0,0,0,0]
        lap=cv2.Laplacian(gray,cv2.CV_32F);sharp=float(lap[er>0].var()) if interior else 0
        hist=np.concatenate([np.histogram(im[...,c][mask],bins=16,range=(0,256))[0] for c in range(3)]).astype(float);hist/=max(np.linalg.norm(hist),1e-12)
        occupancy=cv2.resize(mask[bbox[1]:bbox[3],bbox[0]:bbox[2]].astype(float),(8,8),interpolation=cv2.INTER_AREA).ravel() if area else np.zeros(64)
        descs.append((hist,occupancy)); metrics.append(dict(frame_index=i,timestamp_seconds=float(ts[i]),object_pixels=area,interior_pixels=interior,interior_fraction=interior/max(area,1),masked_laplacian_variance=sharp,bbox_xyxy=bbox,source_rgb_path=path,source_rgb_sha256=m['frame_sha256'][i]))
    pa=max(float(np.percentile([q['object_pixels'] for q in metrics],95)),1);ps=max(float(np.percentile([q['masked_laplacian_variance'] for q in metrics],95)),1);chosen=[];missing=[]
    for b in range(5):
        lo=ts[0]+(ts[-1]-ts[0])*b/5;hi=ts[0]+(ts[-1]-ts[0])*(b+1)/5
        for q in metrics:
            i=q['frame_index'];eligible=lo<=ts[i] and (ts[i]<hi or (b==4 and ts[i]<=hi))
            if not eligible:continue
            novelty=min([.5*(1-float(descs[i][0]@descs[j][0]))+.5*float(np.abs(descs[i][1]-descs[j][1]).mean()) for j in chosen]) if chosen else 0.
            q.update(time_bin=b,appearance_novelty=novelty,eligible=bool(q['object_pixels']>=128 and q['interior_pixels']>=32 and q['object_pixels']>=.3*pa));q['selection_score']=.35*min(q['object_pixels']/pa,1)+.35*min(q['masked_laplacian_variance']/ps,1)+.15*q['interior_fraction']+.15*novelty
        opts=[q for q in metrics if q.get('time_bin')==b and q['eligible']]
        if opts:chosen.append(max(opts,key=lambda q:(q['selection_score'],-q['frame_index']))['frame_index'])
        else:missing.append(b)
    summaries=[];panels=[]
    for n,i in enumerate(chosen):
        q=metrics[i].copy();queries=candidates(cv2.cvtColor(rgb[i],cv2.COLOR_RGB2GRAY),masks[i]);w=np.flatnonzero(np.abs(ts-ts[i])<=1.2);np.save(out/f'keyframe_{i:05d}_queries.npy',queries)
        q.update(keyframe_id=n,query_count=len(queries),query_path=str(out/f'keyframe_{i:05d}_queries.npy'),window_frame_indices=w.tolist(),reason=f"Highest frozen input quality+novelty score in physical-time bin {q['time_bin']+1}/5; chronological coverage only, turn/reappearance semantics not assumed.",canonical_attachment='unknown');summaries.append(q)
        panel=rgb[i].copy();overlay=np.zeros_like(panel);overlay[masks[i]]=[0,220,0];panel=cv2.addWeighted(panel,.8,overlay,.2,0)
        for x,y in queries:cv2.circle(panel,(round(float(x)),round(float(y))),2,(255,60,0),-1)
        panel=cv2.copyMakeBorder(panel,48,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(panel,f'{a.dev} f{i:03d} t={ts[i]:.3f}s | area={q["object_pixels"]} n={len(queries)}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1);cv2.putText(panel,f'score={q["selection_score"]:.3f}, bin={q["time_bin"]+1}, patch queries shown',(8,40),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
        cv2.imwrite(str(out/f'keyframe_{i:05d}.png'),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR));panels.append(panel)
    cv2.imwrite(str(out/'keyframe_contact_sheet.png'),cv2.cvtColor(np.concatenate(panels,0),cv2.COLOR_RGB2BGR))
    rule=E/'protocol/tracklet_rules_v1.json';record=dict(schema_version=1,dev=a.dev,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest=str(mp),source_manifest_sha256=sha(mp),segmentation=str(sp),segmentation_sha256=sha(sp),selection_rules=str(rule),selection_rules_sha256=sha(rule),script_sha256=sha(__file__),keyframes=summaries,missing_bins=missing,all_frame_metrics=metrics,area_p95=pa,sharpness_p95=ps,wall_seconds=time.perf_counter()-start,evaluation_inputs_used=False)
    dump(out/'keyframes.json',record);print(json.dumps({"dev":a.dev,"selected_frames":chosen,"query_counts":[x['query_count'] for x in summaries],"path":str(out/'keyframes.json')}))
if __name__=='__main__':main()
