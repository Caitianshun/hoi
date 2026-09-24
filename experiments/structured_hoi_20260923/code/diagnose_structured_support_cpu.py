"""Input-only support diagnostic on frozen actual rendered Gaussian identities."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import cv2
ROOT=Path('/home/cai_tianshun/Project/HOI')
sys.path.insert(0,str(ROOT/'experiments/mosca_validation_20260923/analysis_a'))
from sparse_raster_exact import sparse_raster

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return {'count':len(x),'mean':float(x.mean()) if len(x) else None,'median':float(np.median(x)) if len(x) else None,'p90':float(np.percentile(x,90)) if len(x) else None}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path,required=True);p.add_argument('--segmentation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.perf_counter()
    ex=json.loads((a.export/'export_manifest.json').read_text());assert ex['status']=='completed' and ex.get('evaluation_status')=='completed'
    meta_path=Path(ex['input_manifest']);meta=json.loads(meta_path.read_text());ws=meta_path.parent
    assert ex['input_manifest_sha256']==sha(meta_path)
    a.output.mkdir(parents=True,exist_ok=False)
    protocol={'role':'input_only_posthoc_model_diagnostic','export_manifest_sha256':sha(a.export/'export_manifest.json'),'script_sha256':sha(__file__),'tracker_sha256':sha(ws/'uniform_cotracker_tap.npz'),'segmentation_sha256':sha(a.segmentation),'source_bindings_sha256':sha(a.export/'frozen_source_bindings.npz'),'reference_geometry_read':False,'gating':['Original CoTracker visibility true at both source and target','Finite target xy, rounded integer pixel inside image','SAM2 label matches predeclared query entity at source and target','5x5 eroded same-entity mask contains both source and target pixels'],'gating_limit':'Tracker Boolean visibility and estimated mask are reliability proxies, not verified ground truth or forward/backward consistency. No depth, geometry or evaluation-error gate.','source_alpha_min':.05,'source_clean_fraction':.8,'low_absolute_alpha_contribution_threshold':.05,'small_projection_error_px':5.,'threshold_role':'Predeclared descriptive triage only; not scientific success criteria or training weights','identity_rule':'Original export source-frame Gaussian IDs and weights remain frozen; no framewise or reference reassignment','surface_support_rule':'Actual source-family alpha contribution at original tracker pixel in whole-scene CPU sparse alpha compositor; separately report full intended-entity contribution','coordinate_rule':'Known fixed K/c2w; metres; no alignment'}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    z=np.load(a.export/'query_trajectories.npz');bind=np.load(a.export/'frozen_source_bindings.npz');identity=np.load(a.export/'gaussian_identity.npz');attrs=np.load(a.export/'gaussian_static_attributes.npz')
    tr=np.load(ws/'uniform_cotracker_tap.npz');labels=np.load(a.segmentation)['entity_labels'];T,Q=z['predicted'].shape[:2];ids=[]
    for uv in z['query_uv']:
        match=np.flatnonzero(np.all(tr['query_points']==np.r_[0.,uv],axis=1));assert len(match)==1;ids.append(int(match[0]))
    tracks=tr['tracks'][:,ids];vis=tr['visibility'][:,ids];entity=identity['entity'];ns=int(identity['bank_sizes'][0]);intended=np.array([2 if str(e)=='object' else 1 for e in z['entity']]);K=np.array(meta['K']);C=np.array(meta['c2w']);W,H=meta['width'],meta['height'];rows=[]
    source_ok=[]
    for qi,uv in enumerate(z['query_uv'].astype(int)):
        x,y=uv;mask=cv2.erode((labels[0]==intended[qi]).astype(np.uint8),np.ones((5,5),np.uint8));source_ok.append(bool(vis[0,qi] and mask[y,x]))
    families=[bind[f'indices_{qi}'] for qi in range(Q)];source_fraction=np.array([bind['source_entity_fractions'][qi,intended[qi]] for qi in range(Q)])
    for t in range(T):
        path=a.export/'geometry'/f'{t:05d}.npz';geo=np.load(path);xyz=np.concatenate([attrs['background_centres_world_m'],geo['dynamic_centres_world_m']]);aff=np.concatenate([attrs['background_affine_frames'],geo['dynamic_affine_frames']]);assert len(xyz)==len(entity)
        projected=(z['predicted'][t]-C[:3,3])@C[:3,:3];proj=projected@K.T;puv=proj[:,:2]/np.maximum(proj[:,2:],1e-8)
        safe=np.isfinite(tracks[t]).all(-1);pix=np.zeros((Q,2),int);pix[safe]=np.rint(tracks[t,safe]).astype(int);inside=safe&(pix[:,0]>=0)&(pix[:,0]<W)&(pix[:,1]>=0)&(pix[:,1]<H)
        current=sparse_raster(xyz,aff,attrs['scale'],attrs['opacity'],K,C,pix,W,H)
        masks={e:cv2.erode((labels[t]==e).astype(np.uint8),np.ones((5,5),np.uint8)) for e in [1,2]}
        for qi,(ii,weights) in enumerate(current):
            x,y=pix[qi];gate=bool(source_ok[qi] and vis[t,qi] and inside[qi] and masks[intended[qi]][y,x])
            total=float(weights.sum());correct=float(weights[entity[ii]==intended[qi]].sum());family=float(weights[np.isin(ii,families[qi])].sum());correct_family=float(weights[np.isin(ii,families[qi])&(entity[ii]==intended[qi])].sum())
            error=float(np.linalg.norm(puv[qi]-tracks[t,qi])) if safe[qi] and projected[qi,2]>0 else None
            source_clean=bool(source_fraction[qi]>=.8)
            if not gate:category='unreliable_input_gate'
            elif bind['source_alpha'][qi]<.05:category='source_support_below_export_threshold'
            elif not source_clean:category='source_entity_mixture'
            elif error is None or error>5:category='projection_motion_error'
            elif correct<.05:category='small_projection_error_but_low_entity_support'
            elif correct_family<.05:category='small_projection_error_but_low_frozen_family_support'
            else:category='projection_and_support_present'
            rows.append({'frame':t,'time_s':float(z['frame_times'][t]),'query_index':qi,'query_id':str(z['query_id'][qi]),'intended_entity':int(intended[qi]),'input_reliable_gate':gate,'source_gate':source_ok[qi],'source_alpha':float(bind['source_alpha'][qi]),'source_intended_fraction':float(source_fraction[qi]),'tracker_visible':bool(vis[t,qi]),'tracker_uv':tracks[t,qi].tolist(),'model_query_uv':puv[qi].tolist(),'query_projection_error_px':error,'total_alpha':total,'intended_entity_alpha':correct,'frozen_family_alpha':family,'intended_frozen_family_alpha':correct_family,'family_fraction_of_total':family/max(total,1e-12),'category':category})
    summary=[]
    for qi in range(Q):
        rr=[r for r in rows if r['query_index']==qi and r['input_reliable_gate']];cats={c:sum(r['category']==c for r in rr) for c in sorted(set(r['category'] for r in rows if r['query_index']==qi))}
        summary.append({'query_id':str(z['query_id'][qi]),'source_gate':source_ok[qi],'source_alpha':float(bind['source_alpha'][qi]),'source_intended_fraction':float(source_fraction[qi]),'reliable_observations':len(rr),'projection_error_px':stat([r['query_projection_error_px'] for r in rr if r['query_projection_error_px'] is not None]),'intended_entity_alpha':stat([r['intended_entity_alpha'] for r in rr]),'frozen_family_alpha':stat([r['frozen_family_alpha'] for r in rr]),'categories':cats})
    result={'status':'completed','summary':summary,'rows':rows,'wall_seconds':time.perf_counter()-start,'conclusion_boundary':'Descriptive input-only evidence. Low family support with good projection is a candidate symptom, not proof of correct 3D geometry or benefit from S2. Wrong initial entity assignment and motion error are explicitly separated.'}
    (a.output/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'completed','summary':summary,'wall_seconds':result['wall_seconds']}),flush=True)
if __name__=='__main__':main()
