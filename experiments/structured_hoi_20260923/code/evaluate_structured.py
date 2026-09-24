"""Frozen structured-scene export. No fitting, alignment, or parameter updates.

Input-only RGB/geometry/query export completes before optional held-out/reference
scoring. Caller allocates CUDA; --validate-only is CPU. --score-only reuses export.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, traceback, subprocess
import numpy as np
ROOT=Path('/home/cai_tianshun/Project/HOI')
E=ROOT/'experiments/structured_hoi_20260923'
OLD=ROOT/'experiments/mosca_baseline_20260922'
D=ROOT/'experiments/mosca_interface_validation_20260923/d_exact_geometry'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()

def save_json(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def cpu(x):return x.detach().cpu().numpy()

def image_metrics(pred,gt,labels=None):
    from skimage.metrics import structural_similarity
    pred=np.clip(pred,0,1).astype(np.float64);gt=np.asarray(gt,dtype=np.float64)
    _,sm=structural_similarity(gt,pred,data_range=1.,channel_axis=2,full=True)
    se=((pred-gt)**2).mean(-1);sm=sm.mean(-1)
    out={}
    for name,mask in [('full',np.ones(se.shape,bool))]+([] if labels is None else [(n,labels==i) for i,n in enumerate(['background','human','object'])]):
        count=int(mask.sum());mse=float(se[mask].mean()) if count else None
        out[name]={'pixels':count,'mse':mse,'psnr_db':float(-10*np.log10(max(mse,1e-12))) if count else None,'ssim':float(sm[mask].mean()) if count else None}
    return out

def summarize(rows):
    out={}
    if not rows:return out
    for region in rows[0]['metrics']:
        rr=[r['metrics'][region] for r in rows if r['metrics'][region]['pixels']]
        out[region]={k:float(np.mean([r[k] for r in rr])) if rr else None for k in ['psnr_db','ssim']}
        out[region]['frames']=len(rr)
    return out

def score_reference(out,meta_path):
    """Invoke unchanged legacy evaluator only after prediction identity is frozen."""
    manifest=json.loads((out/'export_manifest.json').read_text())
    assert manifest['status']=='completed'
    if manifest['query_count']!=6 or sha(meta_path)!=sha(OLD/'common_input/input_manifest.json'):
        save_json(out/'reference_status.json',{'status':'not_available','reason':'No matching predeclared six-query reference protocol for this input. Actual Gaussian export is complete.'});return
    ref=OLD/'evaluation/fixed_rgb_queries'
    cmd=[sys.executable,str(ROOT/'scripts/diagnostics/evaluate_prediction_reference.py'),'--prediction',str(out/'query_trajectories.npz'),'--prediction-protocol',str(out/'export_manifest.json'),'--reference',str(ref/'same_version_joint_reference_observation_times.npz'),'--reference-protocol',str(ref/'same_version_joint_reference_observation_times_protocol.json'),'--input-manifest',str(meta_path),'--method',out.parent.name,'--output',str(out/'reference_evaluation')]
    with (out/'reference_evaluation.log').open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    save_json(out/'reference_status.json',{'status':'completed' if result.returncode==0 else 'failed','returncode':result.returncode,'command':cmd})
    if result.returncode:raise RuntimeError('Independent reference evaluator failed; see reference_evaluation.log')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--init',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--queries',type=Path);p.add_argument('--heldout-manifest',type=Path);p.add_argument('--validate-only',action='store_true');p.add_argument('--score-only',action='store_true');p.add_argument('--skip-reference',action='store_true')
    a=p.parse_args();import torch,cv2
    torch.set_num_threads(4);start=time.perf_counter()
    init=torch.load(a.init,map_location='cpu',weights_only=False);meta_path=Path(init['input_dir'])/'input_manifest.json';meta=json.loads(meta_path.read_text())
    if a.score_only:score_reference(a.output,meta_path);return
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['initialization_sha256']==sha(a.init)
    for name,digest in checkpoint['training_code_identity'].items():assert sha(E/'code'/name)==digest,('training source changed',name)
    for path,digest in checkpoint['data_identity'].items():assert sha(path)==digest,('input changed',path)
    for path,digest in zip(meta['frame_paths'],meta['frame_sha256']):assert sha(path)==digest
    assert meta['role']=='input_only' and init['reference_used'] is False
    query_path=a.queries or Path(init['input_dir'])/'queries_first_frame.json'
    q=json.loads(query_path.read_text()) if query_path.exists() else None
    if q and 'manual_queries' in q:
        q={**q,'query_frame':q['frame_index'],'points':[v['xy'] for v in q['manual_queries']],'point_ids':[v['name'] for v in q['manual_queries']],'entity':['object' if v['name'].startswith('chair_') else ('hand' if 'glove' in v['name'] else 'human') for v in q['manual_queries']],'manual_evaluation_count':q['manual_count']}
    nq=int(q.get('manual_evaluation_count',len(q['points']))) if q else 0
    if q:assert q['query_frame']==0,'Initial implementation requires predeclared source frame zero'
    a.output.mkdir(parents=True,exist_ok=False)
    rec=dict(status='validated_cpu' if a.validate_only else 'running',checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),initialization_sha256=sha(a.init),script_sha256=sha(__file__),training_code_identity=checkpoint['training_code_identity'],input_manifest=str(meta_path),input_manifest_sha256=sha(meta_path),query_file=str(query_path) if q else None,query_sha256=sha(query_path) if q else None,query_count=nq,reference_used_for_training=False,global_transform_fitted_on_evaluation=False,global_alignment_to_reference='none',frame_count=len(meta['frame_paths']),checkpoint_step=checkpoint['step'],units='metres',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),lpips={'status':'not_available','reason':'LPIPS not computed by this exporter; no unverified download or metric substitution.'})
    save_json(a.output/'export_manifest.json',rec)
    if a.validate_only:print(json.dumps(rec));return
    try:
        from gaussian_scene import StructuredScene
        sys.path.insert(0,str(ROOT/'experiments/mosca_validation_20260923/analysis_a'));from sparse_raster_exact import sparse_raster
        from lib_render.render_helper import render
        scene=StructuredScene(init);scene.resize_for_load(checkpoint['model']);scene.load_state_dict(checkpoint['model'],strict=True);scene=scene.cuda().eval()
        for v in scene.parameters():v.requires_grad_(False)
        del checkpoint
        torch.cuda.reset_peak_memory_stats();H,W=meta['height'],meta['width'];K=torch.tensor(init['K'],device='cuda',dtype=torch.float32);C=np.array(init['c2w']);w2c=torch.linalg.inv(scene.c2w)
        times=np.asarray(meta['timestamp_seconds'],dtype=np.float64);labels=np.load(init['segmentation'])['entity_labels']
        for d in ['rgb','buffers','geometry','query_overlay','components']:(a.output/d).mkdir()
        entity=np.concatenate([np.full(b.n,b.entity,np.int8) for b in scene.banks]);stable=np.concatenate([cpu(b.stable_id) for b in scene.banks]);anchor=np.concatenate([cpu(b.anchor_id) for b in scene.banks]);parent=np.concatenate([cpu(b.parent_id) for b in scene.banks])
        ns=scene.sbank.n;counts=[b.n for b in scene.banks]
        np.savez_compressed(a.output/'gaussian_identity.npz',entity=entity,stable_id=stable,anchor_id=anchor,parent_id=parent,bank_sizes=counts,identity_rule=np.array('entity plus stable_id; unique within frozen checkpoint; no reference reassignment'))
        with torch.no_grad():
            source=scene.components(0);cat=lambda k:torch.cat([v[k] for v in source])
            xyz0,fr0,scale0,opacity0=[cpu(cat(k)) for k in range(4)]
            np.savez_compressed(a.output/'gaussian_static_attributes.npz',scale=scale0,opacity=opacity0,colors=cpu(torch.cat([b.color_logit.sigmoid() for b in scene.banks])),background_centres_world_m=xyz0[:ns],background_affine_frames=fr0[:ns])
            uv=np.asarray(q['points'][:nq],dtype=np.float64) if q else np.empty((0,2))
            if q and not np.all(uv==np.round(uv)):raise ValueError('Sparse-raster query contract requires integer source pixels')
            bindings=sparse_raster(xyz0,fr0,scale0,opacity0,np.asarray(init['K']),C,uv,W,H) if q else []
            source_alpha=np.array([w.sum() for _,w in bindings]);fractions=[];bind={}
            for i,(ids,weights) in enumerate(bindings):
                fractions.append([float(weights[entity[ids]==e].sum()/max(weights.sum(),1e-12)) for e in range(3)])
                bind.update({f'indices_{i}':ids,f'weights_{i}':weights,f'entity_{i}':entity[ids],f'stable_ids_{i}':stable[ids]})
            np.savez_compressed(a.output/'frozen_source_bindings.npz',**bind,query_uv=uv,source_alpha=source_alpha,source_entity_fractions=np.asarray(fractions))
            # Validate CPU sparse composition against this frozen CUDA checkpoint.
            if q:
                color=torch.cat([b.color_logit.sigmoid() for b in scene.banks])
                check=render(source,H,W,K,w2c,bg_color=[1.,1.,1.],colors_precomp=color,add_buffer=cat(0))
                x,y=uv.astype(int).T;g_alpha=cpu(check['alpha'])[0,y,x];g_xyz=cpu(check['buf'])[:,y,x].T/np.maximum(g_alpha[:,None],1e-8)
                s_xyz=np.array([(xyz0[ids]*w[:,None]).sum(0)/max(w.sum(),1e-8) for ids,w in bindings])
                alpha_error=float(np.max(abs(g_alpha-source_alpha)));xyz_error=float(np.max(np.linalg.norm(g_xyz-s_xyz,axis=-1)))
                rec['sparse_cuda_check']={'source_alpha_max_error':alpha_error,'source_xyz_max_error_m':xyz_error,'tolerance_alpha':.002,'tolerance_xyz_m':.003}
                assert alpha_error<.002 and xyz_error<.003,rec['sparse_cuda_check']
            predicted=[];rows=[];objR=[];objT=[];querysupport=[]
            keyframes=sorted(set([0,len(times)//4,len(times)//2,3*len(times)//4,len(times)-1]));rec['component_keyframes']=keyframes
            render_start=time.perf_counter()
            for t in range(len(times)):
                result,parts=scene.render(t,K,w2c,H,W);xyz=np.concatenate([cpu(v[0]) for v in parts]);aff=np.concatenate([cpu(v[1]) for v in parts])
                image=cpu(result['rgb']).transpose(1,2,0).clip(0,1);alpha=cpu(result['alpha']);buf=cpu(result['buf']);dep=cpu(result['dep'])
                cv2.imwrite(str(a.output/'rgb'/f'{t:05d}.png'),np.rint(image[...,::-1]*255).astype(np.uint8))
                np.savez_compressed(a.output/'buffers'/f'{t:05d}.npz',alpha=alpha,entity_visible_contribution=buf,depth_camera_z_m=dep)
                np.savez_compressed(a.output/'geometry'/f'{t:05d}.npz',dynamic_centres_world_m=xyz[ns:],dynamic_affine_frames=aff[ns:],timestamp_seconds=times[t],first_dynamic_index=ns)
                traj=np.array([(xyz[ids]*weight[:,None]).sum(0)/max(weight.sum(),1e-8) for ids,weight in bindings]).reshape(nq,3);predicted.append(traj)
                cam=(traj-C[:3,3])@C[:3,:3];proj=cam@np.asarray(init['K']).T;puv=proj[:,:2]/np.maximum(proj[:,2:],1e-8)
                overlay=image.copy()
                for qi,pixel in enumerate(puv):
                    if source_alpha[qi]>=.05 and np.isfinite(pixel).all() and cam[qi,2]>0:
                        u,v=np.rint(pixel).astype(int)
                        if 0<=u<W and 0<=v<H:cv2.circle(overlay,(u,v),4,(1.,.2,.1),1);cv2.putText(overlay,str(qi),(u+5,v),cv2.FONT_HERSHEY_SIMPLEX,.35,(1.,.2,.1),1)
                cv2.imwrite(str(a.output/'query_overlay'/f'{t:05d}.png'),np.rint(overlay[...,::-1]*255).astype(np.uint8))
                if t in keyframes:
                    panels=[]
                    for ent,name in enumerate(['background','human','object']):
                        alone,_=scene.render(t,K,w2c,H,W,only=ent);panel=np.ascontiguousarray(cpu(alone['rgb']).transpose(1,2,0).clip(0,1));cv2.putText(panel,name,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0.,0.,0.),1);panels.append(panel)
                    panels.append(image);cv2.imwrite(str(a.output/'components'/f'{t:05d}.png'),np.rint(np.concatenate(panels,axis=1)[...,::-1]*255).astype(np.uint8))
                gt=cv2.imread(meta['frame_paths'][t])[...,::-1]/255.;rows.append({'frame':t,'time_s':float(times[t]),'metrics':image_metrics(image,gt,labels[t])})
                objR.append(cpu(scene.object_R(t)));objT.append(cpu(scene.object_translation[t]))
                if t%20==0:print(json.dumps({'exported_frame':t,'frames':len(times)}),flush=True)
            torch.cuda.synchronize();rec['input_export_seconds']=time.perf_counter()-render_start
            sys.path.insert(0,str(ROOT/'experiments/mosca_validation_20260923/code'))
            from export_control import write_video
            rec['videos']=[write_video([a.output/'rgb'/f'{t:05d}.png' for t in range(len(times))],times,a.output/'combined_rgb.mp4'),write_video([a.output/'query_overlay'/f'{t:05d}.png' for t in range(len(times))],times,a.output/'fixed_query_overlay.mp4')]
            rec['component_note']='Selected frames show isolated background/human/object and jointly composited RGB. Isolated views reveal hidden model parts and are diagnostic, not visible-mask predictions.'
            np.savez_compressed(a.output/'object_pose.npz',R_world=np.array(objR),translation_world_m=np.array(objT),timestamp_seconds=times)
            pred=np.asarray(predicted);valid=np.broadcast_to(source_alpha[None]>=.05,(len(times),nq)).copy()&np.isfinite(pred).all(-1)
            np.savez_compressed(a.output/'query_trajectories.npz',predicted=pred,predicted_valid_mask=valid,frame_times=times,query_id=np.asarray(q['point_ids'][:nq] if q else [],dtype=str),entity=np.asarray(q['entity'][:nq] if q else [],dtype=str),query_uv=uv,source_alpha=source_alpha,source_entity_fractions=np.asarray(fractions),coordinate_frame=np.array('behave_world_k1_color'),units=np.array('m'),role=np.array('prediction_only_no_reference_alignment'))
            save_json(a.output/'input_image_metrics.json',{'rows':rows,'summary':summarize(rows),'metric_note':'Training RGB fit; full/estimated-SAM2 regions. SSIM standard skimage 7-pixel uniform window, data_range1; region average of local full-image SSIM map. Not independent geometric accuracy. LPIPS unavailable.'})
            rec.update(status='completed',points_by_entity=counts,trajectory_sha256=sha(a.output/'query_trajectories.npz'),source_alpha_min=.05,query_source_alpha=source_alpha.tolist(),query_source_entity_fractions=fractions,query_mechanism='CPU exact source-frame joint alpha contributions frozen once, verified against CUDA XYZ/alpha; fixed actual Gaussian identity and weights propagate current centres; no GT selection or per-frame reassignment.',query_limitation='Weighted actual Gaussian centres can mix entities/surfaces; glove surface queries are not wrist joints. Coverage threshold is source-only.',prediction_geometry_frozen_before_reference=True)
            save_json(a.output/'export_manifest.json',rec)
            # Evaluation-only camera/images are first read after all prediction queries freeze.
            hp=a.heldout_manifest
            if hp is None and sha(meta_path)==sha(OLD/'common_input/input_manifest.json'):hp=D/'heldout/heldout_manifest.json'
            if hp is None and Path(init['input_dir']).resolve()==(E/'data/dev2').resolve():hp=E/'data_audit/dev2_evaluation/heldout_manifest.json'
            if hp:
                hm=json.loads(hp.read_text())
                if 'records' in hm and 'camera' in hm:
                    hm={**hm,'K_rectified':hm['camera']['K'],'c2w':hm['camera']['c2w'],'output_hw':[hm['camera']['height'],hm['camera']['width']],'pairs':[{'input_index':r['matched_input_frame_index'],'heldout_gt':r['path'],'heldout_gt_sha256':r['sha256'],'filename':Path(r['path']).name,'nominal_heldout_time_s':r['nominal_time_seconds'],'input_minus_nominal_s':r['input_minus_camera1_time_seconds'],'actual_camera1_timestamp_seconds':r['actual_camera1_timestamp_seconds']} for r in hm['records']],'time_limit':'Actual camera1 video timestamps matched to nearest input state; no interpolation; time offset recorded; no guaranteed simultaneous exposure.'}
                hk=torch.tensor(hm['K_rectified'],device='cuda',dtype=torch.float32);hc=torch.tensor(hm['c2w'],device='cuda',dtype=torch.float32);hw=torch.linalg.inv(hc);hrows=[];(a.output/'heldout').mkdir()
                hh,ww=hm['output_hw']
                for pair in hm['pairs']:
                    t=pair['input_index'];gtp=Path(pair['heldout_gt']);assert sha(gtp)==pair['heldout_gt_sha256']
                    r,_=scene.render(t,hk,hw,hh,ww);im=cpu(r['rgb']).transpose(1,2,0).clip(0,1);gt=cv2.imread(str(gtp))[...,::-1]/255.
                    cv2.imwrite(str(a.output/'heldout'/pair['filename']),np.rint(im[...,::-1]*255).astype(np.uint8));cv2.imwrite(str(a.output/'heldout'/('compare_'+pair['filename'])),np.rint(np.concatenate([gt,im],1)[...,::-1]*255).astype(np.uint8))
                    hrows.append({'input_index':t,'nominal_time_s':pair['nominal_heldout_time_s'],'input_minus_nominal_s':pair['input_minus_nominal_s'],'gt_path':str(gtp),'metrics':image_metrics(im,gt)})
                save_json(a.output/'heldout_metrics.json',{'source_manifest':str(hp),'source_manifest_sha256':sha(hp),'rows':hrows,'summary':summarize(hrows),'limitation':hm.get('time_limit','Nominal archive time pairing; synchronization unverified. No alignment or photometric adjustment.'),'region_metrics':'not_available: held-out independent instance masks absent','lpips':'not_available'})
        if not a.skip_reference:score_reference(a.output,meta_path)
        rec.update(evaluation_status='completed',wall_seconds=time.perf_counter()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        save_json(a.output/'export_manifest.json',rec);print(json.dumps(rec),flush=True)
    except BaseException as e:
        rec.update(status='failed',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start);save_json(a.output/'export_manifest.json',rec);raise

if __name__=='__main__':main()
