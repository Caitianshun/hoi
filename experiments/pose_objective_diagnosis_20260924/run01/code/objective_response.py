"""Finite input-only objective slices. No evaluation references are loaded."""
from pathlib import Path
import sys,json,hashlib,datetime,argparse,time,csv
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/cai_tianshun/Project/HOI')
E=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/object_pose_refinement_20260924/run01'
sys.path.insert(0,str(OLD/'code'))
from solve_pose import Problem,robust,project,plain

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def records_from_cache(P,path):
    meta=json.loads((path/'canonical_correspondences.json').read_text())
    records=[]
    for f,tr in zip(P.kf,P.tracklets):
        b=[b for b in meta['bindings'] if b['source_frame']==f and b['state']=='reliable_observed_conditional']
        for b0 in sorted(b,key=lambda b:(-int(tr['adopted'][:,b['query']].sum()),b['query'])):
            q=b0['query'];valid=np.flatnonzero(tr['adopted'][:,q]);weight=.5/(3*np.sqrt(max(1,len(valid))*max(1,len(b))))
            for idx in valid:
                records.append((int(tr['target_frame'][idx,q]),np.array(b0['canonical_candidate']),tr['target_uv'][idx,q],weight,f,q))
    assert len(records)==meta['temporal_records']+meta['self_definition_records']
    return records,meta

def names():
    out=[('depth',float(e),None) for e in [-.1,-.05,-.02,0,.02,.05,.1]]
    out += [('rotation',float(d),a) for a in range(3) for d in [-15,15]]
    return out

def perturbed(R,t,c0,kind,value,axis):
    c=R@c0+t
    if kind=='depth':return R.copy(),(1+value)*c-R@c0
    rv=np.zeros(3);rv[axis]=np.radians(value);rr=R@Rotation.from_rotvec(rv).as_matrix()
    return rr,c-rr@c0

def score(P,R,t,records,meta,track_scale=1.):
    b=P.blocks(R,t,records)
    comp={k:float(robust(v*(track_scale if k=='confirmed_tracks' else 1)).sum()/P.L) for k,v in b.items()}
    cross=meta['cross_source_checks'];rgb=float(np.mean([max(0,1-x['patch_ncc']) for x in cross])) if cross else 1.
    used=sum(int(x[0]!=x[4]) for x in records);raw=sum(int(tr['adopted'].sum()) for tr in P.tracklets)
    const={'rgb_constant':.1*rgb,'coverage_constant':.1*(1-used/max(1,raw))}
    return b,comp,const,sum(comp.values())+sum(const.values())

def run(dev,off=None):
    start=time.perf_counter();P=Problem(dev);base=OLD/f'pose/{dev}/optimization'
    selection=json.loads((base/'selection_frozen.json').read_text());selected=selection['selected']['path']
    cases=[('old',P.initpath,base/'path00_old_start',1),('P1',base/selected/'object_init.npz',base/selected,1)]
    if off:cases.append(('track_off',off,base/'path00_old_start',0))
    out=E/f'score_audit/objective/{dev}'+Path('') if False else E/f'score_audit/objective/{dev}'
    if off:out=out/'with_track_off'
    out.mkdir(parents=True,exist_ok=False)
    freeze={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dev':dev,'no_reference_used':True,'code_sha256':sha(__file__),'original_solver_sha256':sha(OLD/'code/solve_pose.py'),'cases':[{'name':n,'path':str(p),'sha256':sha(p),'correspondence_path':str(q/'canonical_correspondences.json'),'correspondence_sha256':sha(q/'canonical_correspondences.json'),'track_scale':s} for n,p,q,s in cases],'keyframes':P.kf,'diagnostic_extra_frames':[64,74] if dev=='dev2' else [],'extra_frames_status':'posthoc known failure case, not independent validation','depth_factors':[-.1,-.05,-.02,0,.02,.05,.1],'rotation_axes':[0,1,2],'rotation_degrees':[-15,15],'center':'all historical template vertices mean','center_preserving':True,'scale':1,'observations':'frozen frame-visible, depth-good, track record sets; no visibility reselection','prior_in_pose_solver':'absent; temporal acceleration only, old pose tether is zero'}
    save(out/'input_freeze.json',freeze)
    rows=[];case_data={};maxproj=0.;maxcenter=0.
    for name,path,cpath,scale in cases:
        a=np.load(path);R=a['R_camera'].astype(float);t=a['t_camera_m'].astype(float);records,meta=records_from_cache(P,cpath)
        blocks,basecomp,const,total=score(P,R,t,records,meta,scale)
        case_data[name]={'components':basecomp,'constants':const,'total':total,'track_count':len(records),'query_count':meta['accepted_queries'],'visible_frames':int(P.visible.sum()),'track_scale':scale}
        for f in sorted(set(P.kf+([64,74] if dev=='dev2' else []))):
            fr=[x for x in records if x[0]==f];baseframe=dict(zip(['silhouette_forward','silhouette_background','estimated_depth_weak'],P.frame_blocks(f,R[f],t[f])))
            baseframe['confirmed_tracks']=np.array([(project((ca@R[f].T+t[f])[None],P.K)[0]-uv)*w for _,ca,uv,w,_,_ in fr]).reshape(-1,2)
            baseframecost={k:float(robust(v*(scale if k=='confirmed_tracks' else 1)).sum()) for k,v in baseframe.items()}
            c0=P.V.mean(0);c=R[f]@c0+t[f]
            for kind,value,axis in names():
                rr,tt=perturbed(R[f],t[f],c0,kind,value,axis)
                cc=rr@c0+tt
                if kind=='depth':maxproj=max(maxproj,float(np.max(abs(project(cc[None],P.K)-project(c[None],P.K)))))
                else:maxcenter=max(maxcenter,float(np.linalg.norm(cc-c)))
                bb=dict(zip(['silhouette_forward','silhouette_background','estimated_depth_weak'],P.frame_blocks(f,rr,tt)))
                bb['confirmed_tracks']=np.array([(project((ca@rr.T+tt)[None],P.K)[0]-uv)*w for _,ca,uv,w,_,_ in fr]).reshape(-1,2)
                current={k:float(robust(v*(scale if k=='confirmed_tracks' else 1)).sum()) for k,v in bb.items()}
                dr={k:(current[k]-baseframecost[k])/P.L for k in current}
                Rnew=R.copy();tnew=t.copy();Rnew[f]=rr;tnew[f]=tt
                temporal=float(robust(P.temporal(Rnew,tnew)).sum()/P.L)
                temporal_delta=temporal-basecomp['actual_time_acceleration']
                rows.append({'dev':dev,'case':name,'frame':f,'time_seconds':float(P.ts[f]),'is_keyframe':f in P.kf,'posthoc_failure_case':f in [64,74] and dev=='dev2','kind':kind,'value':value,'axis':axis,'visible':bool(P.visible[f]),'mask_sample_count':64 if P.visible[f] else 0,'depth_good_count':int(P.obs[f][3].sum()) if P.visible[f] else 0,'track_record_count':len(fr),'frame_components_robust_sum':current,'frame_observation_sum':sum(current.values()),'whole_components':{**{k:basecomp[k]+dr[k] for k in dr},'actual_time_acceleration':temporal},'whole_observation_delta':sum(dr.values()),'whole_temporal_delta':temporal_delta,'whole_objective_delta':sum(dr.values())+temporal_delta,'whole_score':total+sum(dr.values())+temporal_delta,'constants':const,'counts_changed':False,'track_unweighted_robust_sum':float(robust(bb['confirmed_tracks']).sum()),'camera_center_depth_m':float(cc[2]),'scale':1})
    # All frame observation counts retain low reliability intervals, not only keyframes.
    trcount=np.zeros(P.L,dtype=int)
    for tr in P.tracklets:
        for i in range(len(tr['adopted'])):
            for q in np.flatnonzero(tr['adopted'][i]):trcount[int(tr['target_frame'][i,q])]+=1
    states=[{'frame':i,'time_seconds':float(P.ts[i]),'direct_visible':bool(P.visible[i]),'reliable_2d_record_count':int(trcount[i]),'fewer_than_6_reliable_records':bool(trcount[i]<6),'estimated_depth_count':int(P.obs[i][3].sum()) if P.visible[i] else 0} for i in range(P.L)]
    save(out/'responses.json',{'freeze':freeze,'cases':case_data,'rows':rows,'all_frame_observation_states':states,'validation':{'max_depth_center_projection_change_px':maxproj,'max_rotation_center_change_m':maxcenter,'frozen_effective_samples':True},'wall_seconds':time.perf_counter()-start})
    fields=['dev','case','frame','time_seconds','kind','value','axis','visible','mask_sample_count','depth_good_count','track_record_count','frame_observation_sum','whole_observation_delta','whole_temporal_delta','whole_objective_delta','whole_score','camera_center_depth_m']
    with (out/'responses.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps({'dev':dev,'output':str(out),'rows':len(rows),'cases':case_data,'seconds':time.perf_counter()-start},ensure_ascii=False))

def same_pool():
    out=E/'score_audit/same_pool_original';out.mkdir(parents=True,exist_ok=False)
    for dev in ['dev1','dev2']:
        P=Problem(dev);base=OLD/f'pose/{dev}/optimization';paths=sorted(base.glob('path*/object_init.npz'));cases=[('old_initialization',P.initpath,base/'path00_old_start')]+[(p.parent.name,p,p.parent) for p in paths]
        save(out/f'{dev}_freeze.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_reference_used':True,'cases':[{'name':n,'path':str(p),'sha256':sha(p),'correspondence_sha256':sha(c/'canonical_correspondences.json')} for n,p,c in cases],'rule':'original exact input sum and same original legality/visible IoU gate; old has path00 frozen records derived from that very old initialization; deterministic score then path ID'})
        rows=[];oldiou=P.ious(P.R0,P.t0)
        for name,path,cpath in cases:
            a=np.load(path);R=a['R_camera'].astype(float);t=a['t_camera_m'].astype(float);records,meta=records_from_cache(P,cpath);_,comp,const,total=score(P,R,t,records,meta);iou=P.ious(R,t);reasons=[]
            if not np.isfinite(R).all() or not np.isfinite(t).all():reasons.append('nonfinite')
            if not np.allclose(R@R.transpose(0,2,1),np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(R),1,atol=1e-5):reasons.append('illegal_SO3')
            if min(float(np.min((P.V[np.unique(P.F)]@R[i].T+t[i])[:,2])) for i in range(P.L))<=.05:reasons.append('surface_behind_camera')
            if float(iou[P.visible].mean()-oldiou[P.visible].mean())<-.02:reasons.append('mean_visible_IoU_drop_exceeds_0.02')
            rows.append({'name':name,'path':str(path),'sha256':sha(path),'score':total,'components':comp,'constants':const,'query_count':meta['accepted_queries'],'record_count':len(records),'raw_reliable_records':meta['raw_track_observation_count'],'fixed_visible_count':int(P.visible.sum()),'mean_visible_iou':float(iou[P.visible].mean()),'accepted':not reasons,'rejections':reasons})
        valid=sorted((r for r in rows if r['accepted']),key=lambda r:(r['score'],r['name']))
        save(out/f'{dev}_scores.json',{'rows':rows,'selected':valid[0]['name'] if valid else None,'same_pool_complete':True,'pose_float32_serialization_may_make_small_score_changes':True})
        print(dev,'same pool original:',valid[0]['name'] if valid else None)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dev',choices=['dev1','dev2']);ap.add_argument('--off',type=Path);ap.add_argument('--same-pool',action='store_true');args=ap.parse_args()
    if args.same_pool:same_pool()
    else:run(args.dev,args.off)
