"""One integrated engineering acceptance. Short outputs never choose a method."""
from common import *
from state import *
from train_scene import engine,initial
from render_scene import make_renderer,rgb_background
from hoi_modules.projected_motion import make_renderer as original_renderer
from hoi_modules.pose_support_gaussians import PoseSupportGaussians
import gc
def source_identity():
    paths=[RUN/'code'/name for name in ('common.py','state.py','render_scene.py','train_scene.py')]
    paths += [ROOT/'hoi_modules'/name for name in ('pose_prior_adapter.py','pose_support_gaussians.py','region_reconstruction.py')]
    paths += [RUN/'configs/v10.json']
    paths += [ROOT/'experiments/baseline_protocol_calibration_20260927/run01/code/adapter_4dgs.py',ROOT/'hoi_modules/projected_motion.py']
    paths += [p for p in (UPSTREAM/'scene').glob('*.py')]+[UPSTREAM/'utils/general_utils.py',UPSTREAM/'gaussian_renderer/__init__.py']
    files=[identity(p) for p in paths]
    save_json(RUN/'protocol/source_identity.json',dict(status='scientific_source_frozen',files=files,
        project_git_head=os.popen('git rev-parse HEAD').read().strip(),upstream_commit=OFFICIAL_COMMIT,time_unix=time.time()))
    return files
def hashes(module):return {name:tensor_hash(p) for name,p in module.named_parameters()}
def run():
    torch.set_num_threads(4);source_identity();start=time.monotonic();checks={};records=[]
    initial_support={}
    for scene in config()['scenes']:
        base,support,opt,objs,meta,prior=initial(scene,'P');initial_support[scene]=hashes(support)
        cams,frames=load_scene_cameras(scene);fixed=read(ROOT/config()['scenes'][scene]['historical_dir']/'protocol/fixed_examples.json')['training_frame_ids']
        fixed=set(fixed);selected=[c for c in cams if c.image_name in fixed];assert len(selected)==8
        render=make_renderer(meta['scale_bound']);old=original_renderer(meta['scale_bound']);bg=rgb_background(objs[0]);rows=[]
        with torch.no_grad():
            for cam in selected:
                a=render(cam,base,objs[3],bg)['render'];b=old(cam,base,objs[3],bg,stage='fine')['render']
                difference=(a-b).abs();row=dict(frame_id=cam.image_name,max_abs=float(difference.max()),mean_abs=float(difference.mean()),
                    rmse=float((a-b).square().mean().sqrt()))
                rows.append(row)
                # Different covariance multiplication paths have floating-point
                # raster rounding. This gate concerns implementation equivalence.
                assert row['mean_abs']<2e-6 and row['max_abs']<2e-3,row
            cache=torch.load(scene_dir(scene)/'protocol/support_seed.pt',map_location='cpu',weights_only=False)
            empty={**cache,**{k:cache[k][:0] for k in ('u','rgb','knn_dist2','bone_indices','bone_weights','source')}}
            bank=PoseSupportGaussians(empty,meta['scene_extent'],meta['scale_bound']).cuda()
            a=render(selected[0],base,objs[3],bg)['render'];b=render(selected[0],base,objs[3],bg,support=bank,bones=prior.bone_transforms(selected[0].image_name,device='cuda'))['render']
            assert torch.equal(a,b),'empty support renderer'
        checks[scene]={'parent_equivalence_8_training_frames':rows,'empty_support_exact':True,'P_PQ_initial_support':initial_support[scene]}
        del base,support,opt,bank,a,b,cams;gc.collect();torch.cuda.empty_cache()
        gradient_records=[];topology_records=[]
        def probe(phase,k,base,support,optimizer,packages,terms):
            if phase=='after_backward' and k==1:
                if support is not None:
                    if arm=='PQ':assert hashes(support)==initial_support[scene],'P/PQ initial parameters differ'
                    values={name:float(getattr(support,name).grad.abs().max()) for name in ('_xyz','_features_dc','_opacity')}
                    assert values['_features_dc']>0 and values['_opacity']>0 and values['_xyz']>0,values
                    gradient_records.append(dict(arm=arm,update=k,max_gradient=values,high_SH_grad=float(support._features_rest.grad.abs().max())))
                    assert not bool(support._features_rest.grad.any()),'inactive SH learned'
            if phase=='after_update' and arm=='P' and k==4:
                # Deliberately force all three topology operations only in this
                # engineering scratch run; formal density stats remain RGB-only.
                with torch.no_grad():
                    support.gradient_accum.zero_();support.visible_count.zero_()
                    support.gradient_accum[1:3]=.001;support.visible_count[1:3]=1
                    support._scaling[1].fill_(-9);support._scaling[2].fill_(-1);support._opacity[0].fill_(-10)
                    old_ids=support.point_id.clone();before=hashes(base._deformation)
                    event=support.after_adam_topology(optimizer,1100)
                    assert event['clone_parents']==1 and event['split_parents']==1 and event['opacity_pruned']==1,event
                    child=support.point_id>=int(old_ids.max())+1
                    assert int(child.sum())==3
                    for group in optimizer.param_groups:
                        if group['name'].startswith('support.') and group['name']!='support.residual':
                            state=optimizer.state[group['params'][0]]
                            assert not bool(state['exp_avg'][child].any()) and not bool(state['exp_avg_sq'][child].any())
                    assert before==hashes(base._deformation),'support topology modified base'
                    topology_records.append(dict(event=event,actual_update=k,engineering_forced_density_label=1100,
                        children_zero_Adam=True,unique_ids=True,base_unchanged=True))
        for arm,target in [('C',2),('P',8),('PQ',2)]:
            out=RUN/'acceptance'/scene/arm
            base,support,opt,state=engine(scene,arm,target,out,purpose='acceptance',probe=probe)
            records.append(dict(scene=scene,arm=arm,updates=target,receipt=identity(out/'run.json')))
            if arm=='P':
                recovered_base,recovered_support,recovered_optimizer,recovered_objs,recovered_meta=restore(state)
                recovered=capture(recovered_base,recovered_support,recovered_optimizer,scene,arm,target,recovered_meta)
                exact(state,recovered,'complete_checkpoint_roundtrip')
                checks[scene]['complete_roundtrip_exact']=True
                del recovered_base,recovered_support,recovered_optimizer,recovered
            del base,support,opt,state;gc.collect();torch.cuda.empty_cache()
        arm='P';out=RUN/'acceptance'/scene/'P_resume'
        base,support,opt,state=engine(scene,arm,10,out,purpose='acceptance',resume=RUN/'acceptance'/scene/'P/checkpoint_000008.pt')
        records.append(dict(scene=scene,arm='P_resume',updates=2,receipt=identity(out/'run.json')))
        checks[scene]['gradients']=gradient_records;checks[scene]['forced_topology']=topology_records
        del base,support,opt,state;gc.collect();torch.cuda.empty_cache()
    calls=len((RUN/'protocol/diagnostic_attempts.jsonl').read_text().splitlines())
    cpu_calls=read(RUN/'protocol/support_cpu_acceptance.json')['actual_CPU_Adam_updates']
    assert calls+cpu_calls<=96
    receipt=dict(status='passed',checks=checks,runs=records,actual_GPU_Adam_updates=calls,actual_CPU_Adam_updates=cpu_calls,
        actual_Adam_updates=calls+cpu_calls,extra_no_update_backwards=0,seconds=time.monotonic()-start,
        short_quality_evaluated=False,scientific_source=identity(RUN/'protocol/source_identity.json'))
    save_json(RUN/'protocol/module_acceptance.json',receipt)
    save_json(RUN/'protocol/external_GPU_costs.json',dict(GPU_seconds=receipt['seconds'],scope='integrated engineering acceptance incl parent forwards'))
if __name__=='__main__':run()
