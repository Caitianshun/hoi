"""CPU independent fitted-surface proxy for an already frozen Gaussian export.

No model parameters, alignment, or time interpolation. Gaussian centres are
samples, not a triangle surface. Thus asymmetric directions are kept separate:
centres -> fitted triangles, and fitted area samples -> nearest retained centre.
"""
from pathlib import Path
import argparse, hashlib, json, time, sys
import numpy as np
from scipy.spatial import cKDTree
import trimesh

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()

def area_sample(vertices,faces,count=10000,seed=7123):
    triangles=vertices[faces];area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=-1)*.5
    if not np.isfinite(area).all() or area.sum()<=0:raise ValueError('invalid reference surface')
    rng=np.random.default_rng(seed);idx=rng.choice(len(faces),count,p=area/area.sum());r=rng.random((count,2));s=np.sqrt(r[:,0]);bary=np.c_[1-s,s*(1-r[:,1]),s*r[:,1]]
    return np.einsum('ni,nij->nj',bary,triangles[idx])

def centre_to_surface(points,vertices,faces):
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False);dist=[]
    for part in np.array_split(points,max(1,int(np.ceil(len(points)/256)))):
        if len(part):dist.append(trimesh.proximity.closest_point(mesh,part)[1])
    return np.concatenate(dist) if dist else np.empty(0)

def numbers(x):
    return {'mean_cm':float(x.mean()*100),'median_cm':float(np.median(x)*100),'p90_cm':float(np.percentile(x,90)*100)} if len(x) else {'mean_cm':None,'median_cm':None,'p90_cm':None}

def pair_metrics(points,vertices,faces):
    reference_samples=area_sample(vertices,faces)
    forward=centre_to_surface(points,vertices,faces)
    reverse=cKDTree(points).query(reference_samples,workers=1)[0] if len(points) else np.full(len(reference_samples),np.inf)
    r={'centre_to_fitted_triangles':numbers(forward),'fitted_area_to_centres':numbers(reverse) if len(points) else {'mean_cm':None,'median_cm':None,'p90_cm':None},'symmetric_proxy_mean_cm':float(.5*(forward.mean()+reverse.mean())*100) if len(points) else None}
    for threshold in [.05,.10]:
        precision=float(np.mean(forward<=threshold)) if len(forward) else 0.;recall=float(np.mean(reverse<=threshold))
        r[f'within_{int(threshold*100)}cm']={'predicted_centre_precision':precision,'reference_surface_coverage':recall,'fscore':2*precision*recall/(precision+recall) if precision+recall else 0.}
    return r,forward,reverse

def selftest():
    mesh=trimesh.creation.box(extents=[2,2,2]);pts=np.array([[1.,0,0],[1.25,0,0],[0.,0,0]])
    d=centre_to_surface(pts,mesh.vertices,mesh.faces);assert np.allclose(d,[0,.25,1],atol=1e-10)
    s=area_sample(mesh.vertices,mesh.faces);assert len(s)==10000 and np.allclose(np.max(abs(s),axis=1),1)
    assert np.array_equal(s,area_sample(mesh.vertices,mesh.faces))
    assert np.allclose(cKDTree(pts).query(pts)[0],0)
    print(json.dumps({'CPU_selftest':'passed','known_cube_distances_m':d.tolist(),'reference_samples':len(s),'reference_sampling_reproducible':True}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:selftest();return
    if not all([a.export,a.reference,a.output]):p.error('--export --reference --output required')
    start=time.perf_counter();ex=json.loads((a.export/'export_manifest.json').read_text());assert ex['status']=='completed','Only frozen completed prediction export may be evaluated'
    assert ex['global_transform_fitted_on_evaluation'] is False
    a.output.mkdir(parents=True,exist_ok=False)
    # Freeze evaluation protocol before opening independent geometry.
    protocol={'role':'evaluation_only','prediction_manifest':str(a.export/'export_manifest.json'),'prediction_manifest_sha256':sha(a.export/'export_manifest.json'),'reference':str(a.reference),'reference_sha256':sha(a.reference),'script_sha256':sha(__file__),'opacity_threshold':.05,'maximum_predicted_centres_per_entity':10000,'predicted_sample_seed':419,'reference_area_samples_per_entity':10000,'reference_sample_seed':7123,'coverage_thresholds_m':[.05,.10],'alignment':'none','time_interpolation':False,'gaussian_identity':'entity + stable_id from completed model export','missing_reference_rule':'Retain requested slot as unavailable; no replacement','point_rule':'Fixed per-entity deterministic subset of opacity>=0.05 centres, no reference-dependent selection; same frozen identity subset at all times','limitations':['This is an unsigned geometric proxy, not material correspondence or contact accuracy.','Prediction is an opacity-filtered Gaussian-centre sample cloud, not an extracted surface mesh; reverse distance depends on Gaussian coverage/density.','Forward is exact point-to-triangle distance; reverse uses 10000 fixed-seed area-uniform reference samples to nearest retained Gaussian centre.','Whole fitted surface includes occluded and unobserved regions; fitted SMPL/clothing discrepancy and archive timing offset remain.','Symmetric proxy mean equally averages the two directions; no rigid, scale, or per-frame alignment.']}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    identity=np.load(a.export/'gaussian_identity.npz');attrs=np.load(a.export/'gaussian_static_attributes.npz');entities=identity['entity'];opacity=attrs['opacity'].reshape(-1);ns=int(identity['bank_sizes'][0]);selection={}
    for ent,name in [(1,'human'),(2,'object')]:
        eligible=np.flatnonzero((entities==ent)&(opacity>=.05));rng=np.random.default_rng(419+ent);selected=np.sort(rng.choice(eligible,min(10000,len(eligible)),replace=False)) if len(eligible) else eligible
        selection[name]=selected
    np.savez_compressed(a.output/'selected_prediction_identities.npz',**{f'{name}_global_index':ids for name,ids in selection.items()},**{f'{name}_stable_id':identity['stable_id'][ids] for name,ids in selection.items()})
    ref=np.load(a.reference);rows=[];distances={}
    for slot,nominal in enumerate(ref['reference_nominal_times_seconds']):
        t=int(ref['matched_input_indices'][slot]);available=bool(ref['reference_available'][slot]);base={'slot':slot,'nominal_time_s':float(nominal),'input_index':t,'input_time_s':float(ref['matched_input_times_seconds'][slot]),'input_minus_nominal_s':float(ref['matched_time_minus_nominal_seconds'][slot]),'reference_available':available}
        if not available:
            rows.append({**base,'status':'missing_reference','metrics':None});continue
        path=a.export/'geometry'/f'{t:05d}.npz';geo=np.load(path);assert int(geo['first_dynamic_index'])==ns
        assert abs(float(geo['timestamp_seconds'])-base['input_time_s'])<1e-6
        for ent,name in [(1,'human'),(2,'object')]:
            selected=selection[name];points=geo['dynamic_centres_world_m'][selected-ns];vertices=ref[name+'_vertices_world_m'][slot];faces=ref[name+'_faces']
            assert np.isfinite(vertices).all() and np.isfinite(points).all()
            metrics,fwd,rev=pair_metrics(points,vertices,faces)
            row={**base,'status':'scored','entity':name,'total_model_entity_points':int((entities==ent).sum()),'opaque_eligible_points':int(((entities==ent)&(opacity>=.05)).sum()),'scored_points':len(selected),'geometry_sha256':sha(path),'metrics':metrics};rows.append(row)
            distances[f'slot{slot}_{name}_centre_to_triangles_m']=fwd;distances[f'slot{slot}_{name}_reference_to_centres_m']=rev
            print(json.dumps({'slot':slot,'entity':name,'symmetric_proxy_mean_cm':metrics['symmetric_proxy_mean_cm']}),flush=True)
    summary={}
    for name in ['human','object']:
        rr=[r for r in rows if r.get('entity')==name and r['status']=='scored'];valid=[r for r in rr if r['metrics']['symmetric_proxy_mean_cm'] is not None]
        summary[name]={'reference_slots_available':len(rr),'slots_with_predicted_points':len(valid),'symmetric_proxy_mean_cm':float(np.mean([r['metrics']['symmetric_proxy_mean_cm'] for r in valid])) if valid else None,'mean_reference_coverage_5cm':float(np.mean([r['metrics']['within_5cm']['reference_surface_coverage'] for r in rr])) if rr else None,'mean_reference_coverage_10cm':float(np.mean([r['metrics']['within_10cm']['reference_surface_coverage'] for r in rr])) if rr else None}
    np.savez_compressed(a.output/'distance_arrays.npz',**distances)
    result={'status':'completed','summary':summary,'rows':rows,'protocol_sha256':sha(a.output/'protocol.json'),'wall_seconds':time.perf_counter()-start,'evaluation_cannot_update_model':True}
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary']))

if __name__=='__main__':main()
