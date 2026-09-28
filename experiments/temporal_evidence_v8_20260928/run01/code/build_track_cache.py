"""Fixed train-only CoTracker3 queries and re-seeded return observations."""
from common import *
import numpy as np,torch,cv2,time,traceback,subprocess
sys.path[:0]=[str(ROOT/'third_party/co-tracker'),str(ROOT/'third_party/MoSca/lib_prior/tracking')]
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker_wrapper import online_track_point,__online_inference_one_pass__

def queries(mask,rng):
    h,w=mask.shape
    # 16 x 12 =192 equally spaced cell centers, integer pixel centers.
    grid=np.array([(round((x+.5)*w/16-.5),round((y+.5)*h/12-.5)) for y in range(12) for x in range(16)],np.float32)
    seen={tuple(p) for p in grid};extra=[]
    # Spatially stratify the foreground into the same cells, cycle strata until full.
    ys,xs=np.where(mask>=128);bins=[[] for _ in range(192)]
    for x,y in zip(xs,ys):
        if (x,y) not in seen:bins[min(y*12//h,11)*16+min(x*16//w,15)].append((x,y))
    for b in bins:rng.shuffle(b)
    while len(extra)<192 and any(bins):
        for b in bins:
            if b and len(extra)<192:extra.append(b.pop())
    if len(extra)<192:
        seen.update(extra)
        for j in rng.permutation(h*w):
            p=(int(j%w),int(j//w))
            if p not in seen:extra.append(p);seen.add(p)
            if len(extra)==192:break
    return np.concatenate([grid,np.asarray(extra,np.float32)])

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(4);torch.manual_seed(12345);np.random.seed(12345)
    cfg=read(RUN/'configs/v8.json');mp=OLD/'inputs/hos_backpack/manifest.json';m=read(mp)
    frames=sorted(m['frames'],key=lambda x:int(x['frame_id']));assert len(frames)==268 and m['role']=='training_only'
    heldout={x['frame_id'] for x in read(OLD/'inputs/hos_backpack/evaluation_manifest.json')['frames']}
    assert heldout.isdisjoint({r['frame_id'] for r in frames})
    centers=sorted(set([*range(8,len(frames)-8,16),len(frames)-9]));ck=ROOT/cfg['track']['checkpoint'];assert sha(ck)==cfg['track']['checkpoint_sha256']
    model=CoTrackerOnlinePredictor(checkpoint=str(ck),window_len=16,v2=False).cuda().eval();rng=np.random.default_rng(12345)
    start=time.monotonic();pairs=[];wins=[]
    for ci,center in enumerate(centers):
        idx=list(range(center-8,center+9));rows=[frames[j] for j in idx];assert not any(r['frame_id'] in heldout for r in rows)
        images=[]
        for r in rows:
            assert sha(r['rgb_path'])==r['rgb_sha256'];im=cv2.cvtColor(cv2.imread(r['rgb_path']),cv2.COLOR_BGR2RGB);assert im.shape[:2]==(r['height'],r['width']);images.append(im)
        src=frames[center];assert sha(src['mask_path'])==src['mask_sha256'];mask=cv2.imread(src['mask_path'],0);q=queries(mask,rng);h,w=mask.shape
        video=torch.from_numpy(np.stack(images)).permute(0,3,1,2)[None].float();qt=torch.from_numpy(np.c_[np.full(len(q),8,np.float32),q])[None]
        t=time.monotonic()
        with torch.inference_mode():uv,vis=online_track_point(video,qt,model,torch.device('cuda:0'))
        uv=uv.numpy();vis=vis.numpy().astype(bool)
        for off in cfg['track']['target_offsets']:
            j=8+off;target=frames[center+off];target_uv=uv[j].copy();finite=np.isfinite(target_uv).all(-1)
            inbounds=finite&(target_uv[:,0]>=0)&(target_uv[:,0]<w)&(target_uv[:,1]>=0)&(target_uv[:,1]<h)
            back=np.full_like(q,np.nan);rv=np.zeros(len(q),bool);sel=np.flatnonzero(inbounds&vis[j])
            if len(sel):
                reverse=off>0;reseed=np.c_[np.full(len(sel),16-j if reverse else j,np.float32),target_uv[sel]].astype(np.float32)
                with torch.inference_mode():ret,rvis=__online_inference_one_pass__(video.flip(1) if reverse else video,torch.from_numpy(reseed)[None],model,torch.device('cuda:0'),True)
                back[sel]=ret[0,8].numpy();rv[sel]=rvis[0,8].numpy().astype(bool)
            cycle=np.linalg.norm(back-q,axis=-1).astype(np.float64)
            returned_finite=np.isfinite(back).all(-1)
            returned_inside=returned_finite&(back[:,0]>=0)&(back[:,0]<w)&(back[:,1]>=0)&(back[:,1]<h)
            valid=inbounds&vis[j]&rv&returned_inside
            reason=np.zeros(len(q),np.uint16);reason[~finite]|=1;reason[~inbounds]|=2;reason[~vis[j]]|=4;reason[~rv]|=8;reason[~returned_inside]|=16
            logc=np.full(len(q),-np.inf,np.float64);logc[valid]=-.5*(cycle[valid]/3)**2
            # Double precision and log weights retained for stable formal normalization.
            path=RUN/'cache'/f'pair_{len(pairs):04d}.npz'
            np.savez_compressed(path,source_uv=q,target_uv=target_uv,forward_visibility=vis[j],return_source_uv=back,return_visibility=rv,cycle_error_px=cycle,valid=valid,reason_bits=reason,log_confidence=logc,confidence=np.exp(logc))
            pairs.append(dict(pair_id=len(pairs),window_id=ci,source_frame=src['frame_id'],target_frame=target['frame_id'],source_train_index=center,target_train_index=center+off,source_time=src['time'],target_time=target['time'],window_frame_ids=[r['frame_id'] for r in rows],offset=off,cache=identity(path),valid=int(valid.sum()),queries=len(q),a=float(np.exp(logc[valid]).mean()) if valid.any() else 0,cycle_px=quantiles(cycle[valid]),reason_counts={str(bit):int(((reason&bit)!=0).sum()) for bit in [1,2,4,8,16]}))
        row=dict(window_id=ci,center_train_index=center,center_frame=src['frame_id'],seconds=time.monotonic()-t,pairs=6);wins.append(row);print(json.dumps(row),flush=True)
        save_json(RUN/'cache/progress.json',dict(windows=wins,pairs=len(pairs)))
    assert sum(p['valid'] for p in pairs)>0,'No valid temporal teacher observations'
    gen=torch.Generator(device='cpu').manual_seed(12345);order=torch.randint(len(pairs),(5000,),generator=gen)
    orderfile=RUN/'protocol/pair_order.pt';torch.save(dict(order=order,generator_state=gen.get_state(),seed=12345),orderfile)
    save_json(RUN/'track_cache_manifest.json',dict(status='frozen',manifest=identity(mp),checkpoint=identity(ck),cotracker_commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/co-tracker'),'rev-parse','HEAD'],text=True).strip(),source_files=[identity(ROOT/'third_party/co-tracker/cotracker/predictor.py'),identity(ROOT/'third_party/MoSca/lib_prior/tracking/cotracker_wrapper.py')],train_ids=[r['frame_id'] for r in frames],heldout_ids_excluded=sorted(heldout),centers=centers,pairs=pairs,pair_order=identity(orderfile),windows=wins,resize=dict(native=[w,h],model=list(model.interp_shape),mode='bilinear align_corners=True; query and prediction coordinates scaled with (W-1)/(modelW-1)',sampling_after_raster='bilinear align_corners=False using 2*(x+.5)/W-1'),time_normalization=m['time_normalization'],reason_bits={'1':'nonfinite_target','2':'target_outside','4':'forward_not_visible','8':'return_not_visible_or_untested','16':'return_outside_or_nonfinite'},confidence='valid * exp(-0.5*(cycle_error_px/3)^2), float64 log weights retained; official predictor boolean visibility only',all_cost_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),images_context='training rows selected before any tracking; no heldout RGB decoded'))
if __name__=='__main__':main()
