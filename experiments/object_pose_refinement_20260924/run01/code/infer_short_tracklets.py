#!/usr/bin/env python3
"""Frozen multi-seed input observations; no canonical positions or reference reads."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time,traceback
import cv2,numpy as np,torch
from scipy.ndimage import distance_transform_edt
ROOT=Path('/home/cai_tianshun/Project/HOI');E=ROOT/'experiments/object_pose_refinement_20260924/run01'
sys.path[:0]=[str(ROOT/'third_party/co-tracker'),str(ROOT/'third_party/MoSca/lib_prior/tracking')]
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker_wrapper import online_track_point,__online_inference_one_pass__
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
REASONS={'outside_image':1,'network_not_visible':2,'not_5x5_entity2_interior':4,'return_not_visible':8,'cycle_exceeds_2px':16,'low_texture':32,'patch_ncc_low':64,'rgb_mean_difference':128,'source_definition_not_temporal_evidence':256,'nonfinite':512}
def patches(image,uv):
    vals=[];stds=[];grayzs=[];means=[]
    for x,y in uv:
        if not np.isfinite([x,y]).all():p=np.zeros((9,9,3),np.float32)
        else:p=cv2.getRectSubPix(image,(9,9),(float(np.clip(x,0,image.shape[1]-1)),float(np.clip(y,0,image.shape[0]-1)))).astype(np.float32)
        g=cv2.cvtColor(p,cv2.COLOR_RGB2GRAY);std=float(g.std());grayzs.append((g-g.mean()).ravel()/max(std*9,1e-6));stds.append(std);means.append(p.mean((0,1)));vals.append((cv2.resize(p,(3,3),interpolation=cv2.INTER_AREA)/255).ravel())
    return np.asarray(vals),np.asarray(stds),np.asarray(grayzs),np.asarray(means)
def run(dev,model):
    out=E/'pose'/dev;kpath=out/'keyframes.json';kdata=json.loads(kpath.read_text());m=json.loads(Path(kdata['source_manifest']).read_text());s=np.load(kdata['segmentation'])['entity_labels'];ts=np.array(m['timestamp_seconds']);images=[]
    for p,h in zip(m['frame_paths'],m['frame_sha256']):assert sha(p)==h;images.append(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB))
    dist=np.array([distance_transform_edt(x==2) for x in s]);interior=np.array([cv2.erode((x==2).astype(np.uint8),np.ones((5,5),np.uint8)) for x in s],dtype=bool)
    all_records={};runs=[];track_offset=0;start=time.perf_counter();device=torch.device('cuda:0')
    for k in kdata['keyframes']:
        t0=time.perf_counter();src=k['frame_index'];idx=np.asarray(k['window_frame_indices']);q=np.load(k['query_path']);N=len(q);L=len(idx);src_local=int(np.where(idx==src)[0][0]);H,W=images[0].shape[:2]
        assert N<=128 and L>=9, 'Current official streaming wrapper requires at least9frames; no synthetic frames allowed'
        video=torch.from_numpy(np.stack([images[x] for x in idx])).permute(0,3,1,2)[None].float();queries=np.concatenate([np.full((N,1),src_local,np.float32),q],1)
        with torch.inference_mode():tracks,vis=online_track_point(video,torch.from_numpy(queries)[None],model,device)
        uv=tracks.numpy();visible=vis.numpy().astype(bool);assert uv.shape==(L,N,2)
        uv[src_local]=q;visible[src_local]=True
        finite=np.isfinite(uv).all(-1);inbounds=finite&(uv[...,0]>=0)&(uv[...,0]<W)&(uv[...,1]>=0)&(uv[...,1]<H)
        rounduv=np.rint(np.nan_to_num(uv,nan=-1000)).astype(int);rounduv[...,0]=np.clip(rounduv[...,0],0,W-1);rounduv[...,1]=np.clip(rounduv[...,1],0,H-1)
        dint=np.array([interior[t][u[:,1],u[:,0]] for t,u in zip(idx,rounduv)]);bd=np.array([dist[t][u[:,1],u[:,0]] for t,u in zip(idx,rounduv)])
        cycle=np.full((L,N),np.nan,np.float32);return_uv=np.full((L,N,2),np.nan,np.float32);return_vis=np.zeros((L,N),bool);cycle_tested=np.zeros((L,N),bool)
        possible=inbounds&visible&dint;possible[src_local]=False
        # Re-seeded return calls contain every eligible target query. A target later
        # than source returns on reversed original sequence; earlier uses forward.
        for reverse in (False,True):
            jj,nn=np.where(possible&((np.arange(L)[:,None]>src_local) if reverse else (np.arange(L)[:,None]<src_local)))
            for first in range(0,len(jj),512):
                j=jj[first:first+512];n=nn[first:first+512];qt=(L-1-j if reverse else j).astype(np.float32);rq=np.c_[qt,uv[j,n]].astype(np.float32)
                with torch.inference_mode():rt,rv=__online_inference_one_pass__(video.flip(1) if reverse else video,torch.from_numpy(rq)[None],model,device,True)
                rsrc=L-1-src_local if reverse else src_local;back=rt[0,rsrc].numpy();bv=rv[0,rsrc].numpy().astype(bool)
                return_uv[j,n]=back;return_vis[j,n]=bv;cycle[j,n]=np.linalg.norm(back-q[n],axis=1);cycle_tested[j,n]=True
        psrc,ssrc,zsrc,msrc=patches(images[src],q);pt=[];st=[];ncc=[];meanerr=[]
        for j,t in enumerate(idx):
            p,sd,z,mn=patches(images[t],uv[j]);pt.append(p);st.append(sd);ncc.append((z*zsrc).sum(-1));meanerr.append(np.abs(mn-msrc).mean(-1))
        pt=np.asarray(pt);st=np.asarray(st);ncc=np.asarray(ncc);meanerr=np.asarray(meanerr)
        reasons=np.zeros((L,N),np.uint16);reasons[~inbounds]|=1;reasons[~visible]|=2;reasons[~dint]|=4;reasons[~return_vis]|=8;reasons[(~cycle_tested)|(cycle>2)]|=16;reasons[(st<5)|(ssrc[None]<5)]|=32;reasons[ncc<.35]|=64;reasons[meanerr>50]|=128;reasons[src_local]|=256;reasons[~finite]|=512
        adopted=reasons==0;states=np.full((L,N),4,np.uint8);states[(~visible)&finite]=2;states[adopted]=0;states[(reasons!=0)&((reasons&~(32|64))==0)]=1;states[src_local]=1
        # 0 reliable observed 2D (canonical UNKNOWN),1 candidate,2 dormant,4 rejected.
        records=dict(track_id=np.broadcast_to(np.arange(N)+track_offset,(L,N)),keyframe_id=np.full((L,N),k['keyframe_id']),source_frame=np.full((L,N),src),target_frame=np.broadcast_to(idx[:,None],(L,N)),source_time=np.full((L,N),ts[src]),target_time=np.broadcast_to(ts[idx,None],(L,N)),source_uv=np.broadcast_to(q[None],(L,N,2)),target_uv=uv,entity=np.full((L,N),2,np.uint8),visibility=visible,return_source_uv=return_uv,return_visibility=return_vis,cycle_tested=cycle_tested,cycle_error_px=cycle,target_boundary_distance_px=bd,source_boundary_distance_px=np.broadcast_to(dist[src][np.rint(q[:,1]).astype(int),np.rint(q[:,0]).astype(int)][None],(L,N)),target_5x5_interior=dint,source_descriptor_rgb3x3=np.broadcast_to(psrc[None],(L,N,27)),target_descriptor_rgb3x3=pt,source_gray_patch_std=np.broadcast_to(ssrc[None],(L,N)),target_gray_patch_std=st,patch_ncc=ncc,rgb_mean_abs_difference=meanerr,reason_bits=reasons,observation_state=states,adopted=adopted,canonical_confirmed=np.zeros((L,N),bool))
        for name,v in records.items():all_records.setdefault(name,[]).append(v.reshape((-1,)+v.shape[2:]))
        np.savez_compressed(out/f'tracklet_keyframe_{src:05d}.npz',**records)
        # Diagnostic contact sheet: all queried RGB, green accepted/red rejected.
        panels=[]
        for jj2 in sorted(set([0,src_local,L//2,L-1])):
            panel=images[idx[jj2]].copy()
            for n,(x,y) in enumerate(uv[jj2]):
                if inbounds[jj2,n]:cv2.circle(panel,(int(round(x)),int(round(y))),2,(0,240,0) if adopted[jj2,n] else (255,45,0),-1)
            panel=cv2.copyMakeBorder(panel,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(panel,f'{dev} source {src} target {idx[jj2]} accepted {adopted[jj2].sum()}/{N}',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1);panels.append(panel)
        cv2.imwrite(str(out/f'tracklet_keyframe_{src:05d}.png'),cv2.cvtColor(np.concatenate(panels,0),cv2.COLOR_RGB2BGR))
        torch.cuda.synchronize();run=dict(source_frame=src,source_time=float(ts[src]),window_frames=idx.tolist(),queries=N,total_record_count=L*N,nonself_observations=int(adopted.sum()),frames_with_6_adopted=int((adopted.sum(-1)>=6).sum()),cycle_tested=int(cycle_tested.sum()),wall_seconds=time.perf_counter()-t0)
        runs.append(run);track_offset+=N;print(json.dumps(run),flush=True)
    merged={name:np.concatenate(v) for name,v in all_records.items()};np.savez_compressed(out/'tracklets.npz',**merged)
    adopted=merged['adopted'];frame_counts=np.bincount(merged['target_frame'][adopted],minlength=len(ts));reason_counts={k:int(((merged['reason_bits']&v)!=0).sum()) for k,v in REASONS.items()};covered=set(merged['target_frame'].tolist())
    summary=dict(dev=dev,keyframes_sha256=sha(kpath),tracklets_sha256=sha(out/'tracklets.npz'),record_count=len(adopted),adopted_nonself_2d_observations=int(adopted.sum()),unique_source_tracks=int(track_offset),frames_with_6_adopted=int((frame_counts>=6).sum()),frames_with_any_adopted=int((frame_counts>0).sum()),total_frames=len(ts),outside_all_windows=[i for i in range(len(ts)) if i not in covered],adopted_per_frame=frame_counts.tolist(),reason_counts=reason_counts,keyframe_runs=runs,wall_seconds=time.perf_counter()-start,canonical_confirmed_count=0,notes=['2D reliable observations remain estimated, not ground truth.', 'No cross-keyframe identity link or canonical attachment is inferred by this stage.', 'Cycle consistency reseeds predicted UV; it rejects inconsistency but is not independent proof of true material identity.', 'All source self-observations are query definitions and excluded from adopted temporal evidence.', 'Forward model uses frame order, not physical dt; clip selection and saved timestamps use actual input time.'])
    dump(out/'tracklets_summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--dev',choices=['dev1','dev2','both'],default='both');a=p.parse_args();record=E/'logs/tracklets_run.json';start=time.perf_counter();r=dict(status='running',pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),script_sha256=sha(__file__),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=12345,evaluation_inputs_used=False);dump(record,r)
    try:
        torch.manual_seed(12345);np.random.seed(12345);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();r['device']=torch.cuda.get_device_name(0);ck=ROOT/'models/CoTracker3/scaled_online.pth';r['checkpoint_sha256']=sha(ck);assert r['checkpoint_sha256']=='205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218';model=CoTrackerOnlinePredictor(checkpoint=str(ck),window_len=16,v2=False).cuda().eval();r['devs']={}
        for dev in ['dev1','dev2'] if a.dev=='both' else [a.dev]:r['devs'][dev]=run(dev,model);dump(record,r)
        r['status']='completed'
    except BaseException as ex:r.update(status='failed',error=repr(ex),traceback=traceback.format_exc());raise
    finally:
        r.update(wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());dump(record,r)
if __name__=='__main__':main()
