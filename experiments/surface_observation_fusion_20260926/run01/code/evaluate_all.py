"""Evaluation-only entry. All final states and cache hashes required before reading E."""
from pathlib import Path
import sys,os,json,time,csv
import numpy as np,cv2,torch
from skimage.metrics import structural_similarity
from build_surface_observations import E,A,sha_file,save,SurfaceIndex,support_query,mesh_maps
R=Path('/home/cai_tianshun/Project/HOI');B=R/'experiments/fixed_motion_reconstruction_20260926/run01';sys.path.insert(0,str(B/'code'))
from train_visible_object import render_parts
from aux_scene import AuxObjectScene,state_identity
import evaluate_aux as legacy

def evaluate():
 freeze=json.loads((E/'protocol/finals.json').read_text());assert freeze['status']=='all_states_frozen'
 for r in freeze['assets']:assert sha_file(r['path'])==r['sha256'],r['path']
 out=E/'evaluation';assert not (out/'per_frame.json').exists();start=time.perf_counter();m=json.loads((A/'evaluation/regions/manifest.json').read_text());rows=[];representation={};hsrows=[]
 torch.set_num_threads(2);model,lpinfo=legacy.existing_lpips('cpu');save(out/'lpips.json',lpinfo)
 K=torch.tensor(m['camera']['K'],dtype=torch.float32,device='cuda');w2c=torch.tensor(m['camera']['w2c'],dtype=torch.float32,device='cuda');H,W=m['camera']['height'],m['camera']['width'];C=np.asarray(m['camera']['c2w'])
 with torch.no_grad():
  for dev in ['dev1','dev2']:
   erows=[r for r in m['rows'] if r['dev']==dev];qt=[r['query_time_seconds'] for r in erows];meta=json.loads((A/'inputs'/dev/'input_manifest.json').read_text());ref=np.load(meta['reference_object_motion']);p=E/'support'/dev;manifest=json.loads((p/'manifest.json').read_text());g=np.load(manifest['geometry']['path']);v=g['canonical_vertices_m'];faces=g['faces'];ix=SurfaceIndex(dict(np.load(p/'observations.npz')));T=len(meta['timestamp_seconds'])
   scene=AuxObjectScene(dev,qt,device='cuda');sources={'B0':A/'runs'/f'{dev}_Ref/checkpoint_008000.pt','B1':B/'runs'/dev/'checkpoint_008000.pt'}
   for variant in ['F0','F1','F2']:
    path=E/'runs'/f'{dev}_{variant}/baked_model.pt'
    if path.is_file():sources[variant]=path
   partitions=[];labels_list=[];gt_list=[]
   for qi,e in enumerate(erows):
    ti=int(np.flatnonzero(ref['times']==qt[qi])[0]);lab=np.load(e['regions']['path'])['entity_labels'];gt=cv2.imread(e['rgb']['path'])[...,::-1].astype(np.float64)/255;O=lab==2;dep,fi,bc=mesh_maps(v,faces,ref['R_world'][ti],ref['t_world'][ti],np.asarray(m['camera']['K']),C,H,W);valid=O&(fi>=0);yy,xx=np.where(valid);fid=fi[yy,xx];pts=(v[faces[fid]]*bc[yy,xx,:,None]).sum(1);_,_,n,state=support_query(ix,pts,fid,T);st=np.zeros((H,W),np.uint8);st[yy,xx]=state;count=np.zeros((H,W),np.uint8);count[yy,xx]=n
    anchor_tree={};anchor_ok=np.zeros(len(pts),bool)
    from scipy.spatial import cKDTree
    for f in np.unique(fid):
     qs=np.flatnonzero(fid==f);aa=np.flatnonzero(g['sample_face_ids']==f)
     if len(aa):anchor_ok[qs]=np.isfinite(cKDTree(g['centres_m'][aa]).query(pts[qs],distance_upper_bound=.005)[0])
    masks=legacy.region_masks(lab);masks.update(supported=O&(st==3),single_view=O&(st==2),no_positive_evidence=O&(st==1),unmapped=O&(st==0),no_template_depth=O&(fi<0),source_ge3=O&(count>=3))
    partitions.append(masks);labels_list.append(lab);gt_list.append(gt);d=out/dev/e['frame_id'];d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'support.npz',support_state=st,source_count=count,template_depth=dep.astype(np.float32),face=fi)
    save(d/'support.json',dict(O_pixels=int(O.sum()),counts={k:int(m.sum()) for k,m in masks.items()},positive_without_anchor=int(((n>0)&~anchor_ok).sum()),meaning='conditional geometry/mask support proxy, no material truth',time=qt[qi]))
    hs=scene.render_hs_only(qi,K,w2c,H,W);h=hs['rgb'].cpu().numpy().transpose(1,2,0).clip(0,1);cv2.imwrite(str(d/'HS.png'),np.rint(h[...,::-1]*255).astype(np.uint8));cv2.imwrite(str(d/'GT.png'),np.rint(gt[...,::-1]*255).astype(np.uint8));hsrows.append({'dev':dev,'time':qt[qi],'metrics':legacy.region_metrics(h,gt,lab)})
   for variant,source in sources.items():
    bank=torch.load(source,map_location='cpu',weights_only=False)['obank'];scene.obank.resize_for_load(bank,'');scene.obank.load_state_dict(bank);scene.obank.to('cuda');ident=state_identity(scene.obank.state_dict());exp=scene.object_canonical_export();representation[dev+'_'+variant]=legacy.object_statistics(exp)
    if variant.startswith('F'):
     b0=torch.load(sources['B0'],map_location='cpu',weights_only=False)['obank'];assert all(torch.equal(bank[k],b0[k]) for k in bank if k!='color_logit')
    for qi,e in enumerate(erows):
     ti=int(np.flatnonzero(ref['times']==qt[qi])[0]);parts=scene.components(qi,ref['R_world'][ti],ref['t_world'][ti]);ret=render_parts(scene,parts,K,w2c,H,W);pred=ret['rgb'].cpu().numpy().transpose(1,2,0).clip(0,1);contrib=ret['buf'][2].cpu().numpy();gt=gt_list[qi];lab=labels_list[qi];masks=partitions[qi];se=((pred.astype(float)-gt)**2).mean(-1);_,ss=structural_similarity(gt,pred.astype(float),data_range=1.,channel_axis=2,full=True);ss=ss.mean(-1)
     lm=None
     if model is not None:lm=model(torch.tensor(pred.transpose(2,0,1).copy())[None],torch.tensor(gt.transpose(2,0,1).copy(),dtype=torch.float32)[None],normalize=True).numpy().squeeze()
     metrics={}
     for name,mask in masks.items():
      n=int(mask.sum());mse=float(se[mask].mean()) if n else None;metrics[name]=dict(pixels=n,fraction_O=n/int((lab==2).sum()) if name in ['supported','single_view','no_positive_evidence','unmapped','no_template_depth','source_ge3'] else None,psnr_db=float(-10*np.log10(max(mse,1e-12))) if n else None,ssim=float(ss[mask].mean()) if n else None,lpips_spatial_mean=float(lm[mask].mean()) if n and lm is not None else None)
     d=out/dev/e['frame_id'];cv2.imwrite(str(d/f'{variant}.png'),np.rint(pred[...,::-1]*255).astype(np.uint8));np.savez_compressed(d/f'{variant}.npz',rgb=pred,contribution=contrib)
     rr=dict(dev=dev,variant=variant,time=qt[qi],sample_id=e['sample_id'],metrics=metrics,O_contribution_fraction_gt005=float((contrib[lab==2]>.05).mean()),O_contribution_mean=float(contrib[lab==2].mean()))
     if variant=='B0':
      historical=np.load(A/'evaluation/cross_pose'/dev/e['frame_id']/'RR_render_arrays.npz')['rgb_clipped'];rr['historical_B0_render_max_difference']=float(abs(pred-historical).max());assert rr['historical_B0_render_max_difference']<1e-6
     rows.append(rr)
    assert state_identity(scene.obank.state_dict())==ident and scene._frozen_snapshot()==scene.frozen_identity
   # B1 training-camera metrics, never used for selection.
   from train_fusion import setup,old
   s,meta,mp,rgb,labels,masks,interior,K0,w0=setup(dev);st=torch.load(sources['B1'],map_location='cpu',weights_only=False)['obank'];s.obank.resize_for_load(st,'');s.obank.load_state_dict(st);s.obank.to('cuda');motion=old.ref_motion(meta,'Ref');fits=[]
   for t in range(len(rgb)):
    parts=s.components(t,motion[0][t],motion[1][t]);ret=render_parts(s,parts,K0,w0,H,W);err=(ret['rgb']-rgb[t]).square().mean(0);O=labels[t]==2;fits.append(dict(frame=t,O_pixels=int(O.sum()),O_PSNR=float(-10*torch.log10(err[O].mean().clamp_min(1e-12))) if O.any() else None,full_PSNR=float(-10*torch.log10(err.mean().clamp_min(1e-12)))))
   save(B/'runs'/dev/'camera0_fit.json',fits)
 save(out/'per_frame.json',rows);save(out/'representation.json',representation);save(out/'HS_metrics.json',hsrows)
 flat=[dict(dev=r['dev'],variant=r['variant'],time=r['time'],region=name,**met) for r in rows for name,met in r['metrics'].items()]
 with (out/'per_frame.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 summaries={};paired=[]
 for dev in ['dev1','dev2']:
  dr=[r for r in rows if r['dev']==dev];times=sorted(set(r['time'] for r in dr));lookup={(r['variant'],r['time']):r for r in dr};s={}
  for variant in ['B0','B1','F0','F1','F2']:
   vals=[r for r in dr if r['variant']==variant];s[variant]={}
   if not vals:continue
   for reg in vals[0]['metrics']:
    s[variant][reg]={}
    for metric in ['psnr_db','ssim','lpips_spatial_mean']:
     vv=[r['metrics'][reg][metric] for r in vals if r['metrics'][reg][metric] is not None];s[variant][reg][metric]=dict(mean=float(np.mean(vv)) if vv else None,median=float(np.median(vv)) if vv else None,valid_frames=len(vv))
  differences={}
  for va,vb in [('B1','B0'),('F0','B0'),('F1','F0'),('F2','F0'),('F2','F1'),('F2','B0'),('F2','B1')]:
   name=f'{va}-{vb}';differences[name]={}
   for reg in ['object','full','supported','human','background','single_view','no_positive_evidence','unmapped']:
    differences[name][reg]={}
    for metric in ['psnr_db','ssim','lpips_spatial_mean']:
     vv=[]
     for t in times:
      a=lookup.get((va,t));b=lookup.get((vb,t));diff=None
      if a and b:
       av=a['metrics'][reg][metric];bv=b['metrics'][reg][metric]
       if av is not None and bv is not None:diff=av-bv;vv.append(diff)
      paired.append(dict(dev=dev,time=t,pair=name,region=reg,metric=metric,value=diff))
     differences[name][reg][metric]=dict(mean=float(np.mean(vv)) if vv else None,median=float(np.median(vv)) if vv else None,n=len(vv))
  b=differences['B1-B0'];ds=[lookup['B1',t]['O_contribution_fraction_gt005']-lookup['B0',t]['O_contribution_fraction_gt005'] for t in times];supportvals=[lookup['B1',t]['metrics']['supported']['psnr_db']-lookup['B0',t]['metrics']['supported']['psnr_db'] for t in times if lookup['B0',t]['metrics']['supported']['pixels']>=100];support_pass=np.mean(supportvals)>=-.2 if len(supportvals)>=2 else None
  bgates=dict(O_PSNR_mean=b['object']['psnr_db']['mean']>=.5,O_PSNR_median=b['object']['psnr_db']['median']>0,O_SSIM=b['object']['ssim']['mean']>=-.005,full_PSNR=b['full']['psnr_db']['mean']>=-.2,contribution=float(np.mean(ds))>=-.05,supported=support_pass,bounds=representation[dev+'_B1']['canonical_offset_at_most_5mm'] and representation[dev+'_B1']['gaussian_count']<=6000)
  fgates={}
  if s['F2'] and s['F1']:
   f=differences['F2-F1'];fgates=dict(O_PSNR_mean=f['object']['psnr_db']['mean']>=.2,O_PSNR_median=f['object']['psnr_db']['median']>0,vsF0=differences['F2-F0']['object']['psnr_db']['mean']>=0,vsB0=differences['F2-B0']['object']['psnr_db']['mean']>=0,O_SSIM=f['object']['ssim']['mean']>=-.005,O_LPIPS=f['object']['lpips_spatial_mean']['mean']<=.005,full_PSNR=f['full']['psnr_db']['mean']>=-.1)
  summaries[dev]=dict(variants=s,differences=differences,B1_gates={k:bool(v) if v is not None else None for k,v in bgates.items()},B1_numeric_pass=all(v for v in bgates.values() if v is not None),B1_supported_guard_frames=len(supportvals),B1_contribution_delta=float(np.mean(ds)),F2_gates={k:bool(v) for k,v in fgates.items()},F2_numeric_pass=bool(fgates) and all(fgates.values()))
 save(out/'summary.json',summaries);save(out/'paired_differences.json',paired)
 for r in freeze['assets']:assert sha_file(r['path'])==r['sha256']
 save(out/'run.json',dict(status='completed',seconds=time.perf_counter()-start,rows=len(rows),all_states_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 print(json.dumps({d:{'B1_pass':s['B1_numeric_pass'],'F2_pass':s['F2_numeric_pass']} for d,s in summaries.items()}),flush=True)
if __name__=='__main__':assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';evaluate()
