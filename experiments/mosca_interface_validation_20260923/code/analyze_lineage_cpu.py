"""CPU-only computational lineage and joint-pixel contribution audit for D.

Roots are selected ONLY from initial source RGB/SAM/depth, before final state is
read. Every zero-support/extinct set is retained. No reference geometry is read.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'experiments/mosca_baseline_20260922'
PREV = ROOT/'experiments/mosca_validation_20260923'
sys.path.insert(0, str(PREV/'analysis_a'))
from sparse_raster_exact import sparse_raster
sys.path.insert(0, str(PREV/'motion_binding'))
from audit_motion_binding import get_weights, warp, project


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''): h.update(part)
    return h.hexdigest()
def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def state(path): return torch.load(path, map_location='cpu', weights_only=False)
def frac(a, b): return float(a/b) if b > 1e-12 else None
def quant(a):
    a = np.asarray(a)
    return dict(zip(['min', 'p25', 'median', 'p75', 'max'], map(float, np.quantile(a, [0,.25,.5,.75,1])))) if a.size else None
def weights_by_node(indices,weights):
    answer={}
    for index,weight in zip(indices,weights):
        answer[int(index)]=answer.get(int(index),0.)+float(weight)
    return answer


def bilinear_weights(xyz, rotation, scales, opacity, K, C, queries, W=640, H=480):
    """Actual sparse compositor, followed by bilinear premultiplied weights."""
    positions, coeffs, valid = [], [], []
    for q in np.asarray(queries):
        ok = np.isfinite(q).all() and 0 <= q[0] <= W-1 and 0 <= q[1] <= H-1
        valid.append(bool(ok))
        if not ok:
            positions.extend([[0,0]]*4); coeffs.append([0]*4); continue
        x,y = np.floor(q).astype(int); dx,dy = q-[x,y]
        positions.extend([[x,y],[min(x+1,W-1),y],[x,min(y+1,H-1)],[min(x+1,W-1),min(y+1,H-1)]])
        coeffs.append([(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy])
    unique, inverse = np.unique(positions, axis=0, return_inverse=True)
    raster = sparse_raster(xyz, rotation, scales, opacity, K, C, unique, W, H)
    answer = []
    for j in range(len(queries)):
        ids, weights = [], []
        for k, coefficient in enumerate(coeffs[j]):
            if coefficient <= 0: continue
            ii, ww = raster[inverse[4*j+k]]
            ids.extend(ii.tolist()); weights.extend((ww*coefficient).tolist())
        if ids:
            ii, inv = np.unique(ids, return_inverse=True); ww = np.zeros(len(ii)); np.add.at(ww, inv, weights)
        else:
            ii, ww = np.empty(0, dtype=np.int64), np.empty(0)
        answer.append((ii, ww, valid[j]))
    return answer


def kernel_at_integer(xyz, rotation, scales, opacity, K, C, q, W=640, H=480):
    """Same native projected kernel as validated sparse_raster, without compositing.
    Used only at integer first-frame input queries to separate clipping from
    transmittance. Geometry/opacity remain frozen; no renderer threshold changes.
    """
    xyz = np.asarray(xyz, np.float64); camera = (xyz-C[:3,3])@C[:3,:3]; z = camera[:,2]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    uv = camera[:,:2]/(z[:,None]+1e-7)*[fx,fy]+[cx,cy]
    rc = C[:3,:3].T[None]@rotation
    cov = (rc*scales[:,None,:]**2)@rc.transpose(0,2,1)
    zz = np.maximum(z,1e-8)
    J = np.zeros((len(z),2,3)); J[:,0,0]=fx/zz; J[:,1,1]=fy/zz
    J[:,0,2] = -fx*np.clip(camera[:,0]/zz,-1.3*W/(2*fx),1.3*W/(2*fx))/zz
    J[:,1,2] = -fy*np.clip(camera[:,1]/zz,-1.3*H/(2*fy),1.3*H/(2*fy))/zz
    cov2 = J@cov@J.transpose(0,2,1); cov2[:,0,0]+=.3; cov2[:,1,1]+=.3
    aa,bb,cc = cov2[:,0,0],cov2[:,0,1],cov2[:,1,1]; det=aa*cc-bb**2
    conic=np.stack([cc,-bb,aa],-1)/np.maximum(det[:,None],1e-20)
    mid=.5*(aa+cc); rad=np.ceil(3*np.sqrt(np.maximum(mid+np.sqrt(np.maximum(.1,mid**2-det)),0)))
    grid=np.array([(W+15)//16,(H+15)//16]); tile=np.floor(q/16).astype(int)
    lo=np.clip(np.trunc((uv-rad[:,None])/16),0,grid).astype(int)
    hi=np.clip(np.trunc((uv+rad[:,None]+15)/16),0,grid).astype(int)
    eligible=(z>.02)&(det>0)&((lo<=tile)&(hi>tile)).all(-1)
    delta=uv-q; power=-.5*(conic[:,0]*delta[:,0]**2+conic[:,2]*delta[:,1]**2)-conic[:,1]*delta[:,0]*delta[:,1]
    alpha=np.minimum(.99,opacity.ravel()*np.exp(np.minimum(power,0)))
    threshold_pass=eligible&(power<=0)&(alpha>=1/255)
    return dict(projected_uv=uv, camera_z_m=z, raw_kernel_alpha=alpha,
                tile_and_depth_eligible=eligible, kernel_threshold_pass=threshold_pass)


def prepare(s, d, scale):
    assert bool(d['leaf_local_flag'])
    g=get_weights(d); sx=s['_xyz'].numpy()/scale
    sr=quaternion_to_matrix(torch.nn.functional.normalize(s['_rotation'],dim=-1)).numpy()
    scales=torch.cat([v['min_scale']+torch.sigmoid(v['_scaling'])*(v['max_scale']-v['min_scale']) for v in [s,d]]).numpy()/scale
    opacity=torch.cat([torch.sigmoid(v['_opacity']) for v in [s,d]]).numpy().ravel()
    return g,sx,sr,scales,opacity


def combine(d, g, sx, sr, scale, frame):
    dx,dr=warp(d,g,int(frame),dyn_off=not bool(d['dyn_o_flag']))
    return np.concatenate([sx,dx.numpy()/scale]),np.concatenate([sr,dr.numpy()])


def select_roots(model, queries, depth, out):
    p=load(model/'seed_provenance.npz'); n=len(p['gaussian_id'])
    assert np.array_equal(p['gaussian_id'],np.arange(n))
    sets=[]
    for qi,(name,uv) in enumerate(zip(queries['point_ids'][:6], queries['points'][:6])):
        uv=np.asarray(uv); expected_label=1 if name.startswith('hand') else 2
        input_depth=float(depth[int(uv[1]),int(uv[0])])
        for radius in [4,8]:
            mask=(p['raw_source_frame']==0)&(np.max(np.abs(p['raw_pixel_xy']-uv),axis=-1)<=radius)
            mask &= (np.abs(p['raw_depth_m']-input_depth)<=.1)&(p['input_sam2_label']==expected_label)
            ids=p['gaussian_id'][mask]
            sets.append(dict(query_id=name,query_index=qi,radius_pixels=radius,source_uv=uv.tolist(),
                input_depth_m=input_depth,input_sam2_label=expected_label,root_ids=ids.tolist(),
                selected_count=int(len(ids)),roots=[dict(root_id=int(i),raw_pixel_xy=p['raw_pixel_xy'][i].tolist(),
                    raw_depth_m=float(p['raw_depth_m'][i]),raw_world_xyz_m=p['raw_world_xyz_m'][i].tolist()) for i in ids]))
    result=dict(protocol='source frame0; source L-infinity radius4 primary / radius8 prespecified sensitivity; source depth within +/-0.1m of query input depth; input SAM2 entity label',
        selected_before_reading_final_checkpoint=True,reference_geometry_read=False,
        first_frame_query_selection=queries['source'],source_hash=sha(model/'seed_provenance.npz'),sets=sets)
    target=out/'selected_roots.json'
    if target.exists():
        assert read(target)==result, 'Frozen root selection must not change'
    else: dump(target,result)
    return sets,p


def select_effective_roots(model, queries, depth, manifest, provenance, out):
    """Additional initial-state selection, frozen before final model analysis.
    Prompted by the observed empty centre r4 set and other-birth-frame support.
    Distinct from source-frame0 radius groups; never a replacement for them.
    """
    target=out/'selected_initial_effective_roots.json'
    if target.exists():
        recorded=read(target)
        assert recorded['seed_provenance_sha256']==sha(model/'seed_provenance.npz')
        for source,expected in recorded['source_snapshot_sha256'].items():
            assert sha(source)==expected, 'Initial effective-root selection source changed'
        return recorded['sets']
    folder=model/'lineage/snapshots/00000'
    s=state(folder/'static_state.pth');d=state(folder/'dynamic_state.pth');identity=load(folder/'identity.npz')
    scale=float(provenance['world_scale']);K=np.array(manifest['K']);C=np.array(manifest['c2w'])
    g,sx,sr,scales,opacity=prepare(s,d,scale);ns=len(sx);xyz,rot=combine(d,g,sx,sr,scale,0)
    pixels=np.array(queries['points'][:6]);weights=bilinear_weights(xyz,rot,scales,opacity,K,C,pixels)
    _,z=project(xyz,K,C);sets=[]
    for qi in [4,5]:
        uv=pixels[qi];iz=float(depth[int(uv[1]),int(uv[0])]);ii,ww,_=weights[qi]
        eligible=ii>=ns;rows=ii[eligible]-ns;w=ww[eligible]
        near=(np.abs(z[ns+rows]-iz)<=.1)&(provenance['input_sam2_label'][rows]==1)&(w>0)
        rows=rows[near];w=w[near];root_ids=identity['root_id'][rows]
        assert np.array_equal(root_ids,rows)  # initial rows map to initial roots
        sets.append(dict(query_id=queries['point_ids'][qi],query_index=qi,radius_pixels=-1,
            selection_kind='initial_effective_near_human_contributors',source_uv=uv.tolist(),input_depth_m=iz,
            input_sam2_label=1,root_ids=root_ids.tolist(),selected_count=len(rows),
            initial_selected_joint_weight=float(w.sum()),initial_selected_joint_fraction=frac(w.sum(),ww.sum()),
            roots=[dict(root_id=int(row),raw_source_frame=int(provenance['raw_source_frame'][row]),
                raw_pixel_xy=provenance['raw_pixel_xy'][row].tolist(),raw_depth_m=float(provenance['raw_depth_m'][row]),
                raw_world_xyz_m=provenance['raw_world_xyz_m'][row].tolist(),
                initial_world_at_query_frame_m=xyz[ns+row].tolist(),initial_joint_weight=float(weight)) for row,weight in zip(rows,w)]))
    dump(target,dict(decision_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol='Snapshot0 actual first-frame joint weight >0; warped camera z within input depth +/-0.1m; original source SAM2 human; all eligible roots, no top-K and no birth-frame restriction',
        rationale='Initial source-frame0 centre radius4 set was empty despite near-depth pixel support; distinguish roots from other birth frames.',
        added_after_observing_snapshot0_before_final_model_read=True,reference_geometry_used=False,
        seed_provenance_sha256=sha(model/'seed_provenance.npz'),
        source_snapshot_sha256={str(folder/f):sha(folder/f) for f in ['static_state.pth','dynamic_state.pth','identity.npz']},sets=sets))
    return sets


def check_ledger(lineage, summary):
    initial=load(lineage/'initial_identity.npz'); leaf=initial['leaf_id']; node=initial['node_id']
    with (lineage/'events.csv').open() as f: events=list(csv.DictReader(f))
    snapshots=sorted(summary['snapshots'],key=lambda x:x['completed_steps']); cursor=0
    for snap in snapshots:
        while cursor<len(events) and int(events[cursor]['completed_step_of_update'])<=snap['completed_steps']:
            row=events[cursor]; a=load(lineage/row['npz']); current=leaf if row['entity']=='leaf' else node
            assert len(current)==int(row['count_before'])
            if row['operation']=='prune':
                assert np.array_equal(current[a['old_row']],a['id'])
                current=np.delete(current,a['old_row'])
            elif row['operation']!='prune_skipped':
                assert np.array_equal(a['new_row'],np.arange(len(current),len(current)+len(a['id'])))
                if 'parent_row_before_append' in a:
                    assert np.array_equal(current[a['parent_row_before_append']],a['parent_id'])
                current=np.concatenate([current,a['id']])
            assert len(current)==int(row['count_after'])
            if row['entity']=='leaf': leaf=current
            else: node=current
            cursor+=1
        ids=load(Path(snap['directory'])/'identity.npz')
        assert np.array_equal(ids['leaf_id'],leaf) and np.array_equal(ids['node_id'],node)
    assert cursor==len(events)
    final=load(lineage/'leaf_lineage.npz'); finalnode=load(lineage/'node_lineage.npz')
    assert np.array_equal(final['live_id'],leaf) and np.array_equal(finalnode['live_id'],node)
    return dict(events=len(events),snapshots=len(snapshots),row_order_replayed_exactly=True),final


def inspect_queries(xyz, rot, scales, opacity, ns, K, C, pixels, W, H):
    joint=bilinear_weights(xyz,rot,scales,opacity,K,C,pixels,W,H)
    dynamic=bilinear_weights(xyz[ns:],rot[ns:],scales[ns:],opacity[ns:],K,C,pixels,W,H)
    return joint,dynamic


def support_row(spec, ids, xyz, ns, weights, dynamic, K, C, input_depth, camera_depth=None):
    row_ids=np.flatnonzero(np.isin(ids['root_id'],spec['root_ids']))
    ii,ww,valid=weights; di,dw,_=dynamic
    all_alpha=float(ww.sum()); dyn_alpha=float(ww[ii>=ns].sum())
    selected=np.isin(ii,ns+row_ids); selected_dynamic=np.isin(di,row_ids)
    z=project(xyz,K,C)[1] if camera_depth is None else camera_depth
    return dict(query_id=spec['query_id'],radius_pixels=spec['radius_pixels'],
        selection_kind=spec.get('selection_kind','source_frame0_radius'),initial_root_count=spec['selected_count'],
        surviving_descendant_count=int(len(row_ids)),surviving_root_count=int(len(np.unique(ids['root_id'][row_ids]))),
        extinct_selected_root_count=int(len(set(spec['root_ids'])-set(ids['root_id'][row_ids].tolist()))),
        sample_in_image=bool(valid),joint_alpha=all_alpha,joint_dynamic_alpha=dyn_alpha,
        descendant_joint_weight=float(ww[selected].sum()),descendant_fraction_of_joint_alpha=frac(ww[selected].sum(),all_alpha),
        descendant_fraction_of_joint_dynamic_alpha=frac(ww[selected].sum(),dyn_alpha),
        dynamic_only_alpha=float(dw.sum()),descendant_fraction_of_dynamic_only_alpha=frac(dw[selected_dynamic].sum(),dw.sum()),
        joint_dynamic_fraction=frac(dyn_alpha,all_alpha),joint_static_fraction=frac(all_alpha-dyn_alpha,all_alpha),
        input_depth_m=float(input_depth) if np.isfinite(input_depth) else None,
        front_of_input_fraction=frac(ww[z[ii]<input_depth-.1].sum(),all_alpha) if np.isfinite(input_depth) else None,
        near_input_fraction=frac(ww[np.abs(z[ii]-input_depth)<=.1].sum(),all_alpha) if np.isfinite(input_depth) else None,
        behind_input_fraction=frac(ww[z[ii]>input_depth+.1].sum(),all_alpha) if np.isfinite(input_depth) else None,
        weighted_centroid_world_m=((xyz[ii]*ww[:,None]).sum(0)/all_alpha).tolist() if all_alpha>1e-12 else None,
        weighted_camera_z_m=float(np.dot(ww,z[ii])/all_alpha) if all_alpha>1e-12 else None)


def snapshot_analysis(model,out,m,sets,p,summary):
    K=np.array(m['K']); C=np.array(m['c2w']); scale=summary['world_scale']; W,H=m['width'],m['height']
    pixels=np.array([next(v['source_uv'] for v in sets if v['query_index']==qi) for qi in range(6)])
    results=[]; root_rows=[]; hashes={}; checks={}; prior_bindings={}
    initial_folder=model/'lineage/snapshots/00000'
    initial_d=state(initial_folder/'dynamic_state.pth');initial_g=get_weights(initial_d)
    initial_source=warp(initial_d,initial_g,0,dyn_off=not bool(initial_d['dyn_o_flag']))[0].numpy()/scale
    (out/'descendants').mkdir(exist_ok=True)
    for snap in sorted(summary['snapshots'],key=lambda x:x['completed_steps']):
        folder=Path(snap['directory']); step=int(snap['completed_steps'])
        s=state(folder/'static_state.pth'); d=state(folder/'dynamic_state.pth'); ids=load(folder/'identity.npz')
        assert sha(folder/'dynamic_state.pth')==snap['state_sha256']
        assert sha(folder/'static_state.pth')==snap['static_state_sha256']
        assert sha(folder/'identity.npz')==snap['identity_sha256']
        hashes[str(folder)]=dict(dynamic=snap['state_sha256'],static=snap['static_state_sha256'],identity=snap['identity_sha256'])
        g,sx,sr,scales,opacity=prepare(s,d,scale); ns=len(sx); xyz,rot=combine(d,g,sx,sr,scale,0)
        joint,dynamic=inspect_queries(xyz,rot,scales,opacity,ns,K,C,pixels,W,H)
        camera_depth=project(xyz,K,C)[1]
        for spec in sets:
            qi=spec['query_index']; selected=np.flatnonzero(np.isin(ids['root_id'],spec['root_ids'])); physical=selected+ns
            result=support_row(spec,ids,xyz,ns,joint[qi],dynamic[qi],K,C,spec['input_depth_m'],camera_depth)
            result.update(completed_steps=step,input_frame_index=0,time_seconds=m['timestamp_seconds'][0])
            det=kernel_at_integer(xyz[physical],rot[physical],scales[physical],opacity[physical],K,C,pixels[qi],W,H)
            weight_by_row=np.zeros(len(xyz)); weight_by_row[joint[qi][0]]=joint[qi][1]
            dyn_weight_by_row=np.zeros(len(d['_xyz'])); dyn_weight_by_row[dynamic[qi][0]]=dynamic[qi][1]
            op=opacity[physical]; distance=np.linalg.norm(det['projected_uv']-pixels[qi],axis=-1)
            drift=np.linalg.norm(xyz[physical]-initial_source[ids['root_id'][selected]],axis=-1)
            count=dict(opacity_below_renderer_threshold=int((op<1/255).sum()),
                eligible_but_kernel_below_threshold=int((det['tile_and_depth_eligible']&(op>=1/255)&~det['kernel_threshold_pass']).sum()),
                kernel_pass_but_zero_joint_weight=int((det['kernel_threshold_pass']&(weight_by_row[physical]==0)).sum()),
                nonzero_joint_contributors=int((weight_by_row[physical]>0).sum()),
                outside_diagnostic_radius4=int((np.max(np.abs(det['projected_uv']-pixels[qi]),axis=-1)>4).sum()),
                not_tile_or_depth_eligible=int((~det['tile_and_depth_eligible']).sum()))
            result.update(suppression_counts=count,descendant_opacity=quant(op),
                          projected_distance_pixels=quant(distance),source_world_drift_m=quant(drift))
            file=out/'descendants'/f'{step:05d}_{spec["query_id"]}_r{spec["radius_pixels"]}.npz'
            np.savez_compressed(file,leaf_id=ids['leaf_id'][selected],parent_id=ids['parent_id'][selected],
                root_id=ids['root_id'][selected],world_xyz_m=xyz[physical],opacity=op,
                attached_node_id=ids['attached_node_id'][selected],ref_time=ids['ref_time'][selected],
                skinning_node_id=ids['node_id'][g['idx'][selected].numpy()],skinning_weight=g['w'][selected].numpy(),
                scales_m=scales[physical],source_world_drift_m=drift,projected_distance_pixels=distance,
                joint_alpha_weight=weight_by_row[physical],dynamic_only_alpha_weight=dyn_weight_by_row[selected],**det)
            result['descendant_file']=str(file.resolve()); results.append(result)
            for root_id in spec['root_ids']:
                mask=ids['root_id'][selected]==root_id
                leaf_ids=ids['leaf_id'][selected][mask]
                attach_ids=ids['attached_node_id'][selected][mask]
                ref_times=ids['ref_time'][selected][mask]
                neighbor_ids=ids['node_id'][g['idx'][selected].numpy()][mask]
                neighbor_weights=g['w'][selected].numpy()[mask]
                binding_key=(spec['query_id'],spec['radius_pixels'],root_id)
                before=prior_bindings.get(binding_key,{})
                now={int(lid):(int(att),int(rt),weights_by_node(nn,nw))
                     for lid,att,rt,nn,nw in zip(leaf_ids,attach_ids,ref_times,neighbor_ids,neighbor_weights)}
                shared=sorted(set(now)&set(before)); changed_attach=[]; changed_neighbors=[];weight_changes=[]
                for lid in shared:
                    old,new=before[lid],now[lid]
                    if old[0]!=new[0]:changed_attach.append(dict(leaf_id=lid,before=old[0],after=new[0]))
                    if set(old[2])!=set(new[2]):changed_neighbors.append(lid)
                    weight_changes.append(sum(abs(old[2].get(k,0)-new[2].get(k,0)) for k in set(old[2])|set(new[2])))
                prior_bindings[binding_key]=now
                kernel=det['kernel_threshold_pass'][mask]
                effective=weight_by_row[physical][mask]>0
                root_rows.append(dict(completed_steps=step,query_id=spec['query_id'],radius_pixels=spec['radius_pixels'],
                    selection_kind=spec.get('selection_kind','source_frame0_radius'),
                    root_id=root_id,descendants=int(mask.sum()),joint_weight=float(weight_by_row[physical][mask].sum()),
                    joint_fraction=frac(weight_by_row[physical][mask].sum(),joint[qi][1].sum()),
                    min_pixel_distance=float(distance[mask].min()) if mask.any() else None,
                    min_source_world_drift_m=float(drift[mask].min()) if mask.any() else None,
                    max_opacity=float(op[mask].max()) if mask.any() else None,
                    opacity_below_threshold_count=int((op[mask]<1/255).sum()),
                    kernel_threshold_pass_count=int(kernel.sum()),positive_joint_contributor_count=int(effective.sum()),
                    kernel_pass_but_zero_joint_weight_count=int((kernel&~effective).sum()),
                    within4_and_input_depth_count=int(((np.max(abs(det['projected_uv'][mask]-pixels[qi]),axis=-1)<=4)&
                        (abs(det['camera_z_m'][mask]-spec['input_depth_m'])<=.1)).sum()),
                    dynamic_only_weight=float(dyn_weight_by_row[selected][mask].sum()),
                    attached_node_ids=sorted(set(attach_ids.tolist())),reference_times=sorted(set(ref_times.tolist())),
                    shared_leaf_count_from_previous_snapshot=len(shared),
                    same_leaf_attachment_changes=changed_attach,
                    same_leaf_neighbor_id_change_count=len(changed_neighbors),
                    same_leaf_neighbor_weight_l1_change=quant(weight_changes)))
        if step==0:
            checks['initial_raw_world_vs_recovered_source_max_m']=float(np.max(np.abs(g['src'].numpy()/scale-p['raw_world_xyz_m'])))
            assert checks['initial_raw_world_vs_recovered_source_max_m']<2e-5
        print('lineage checkpoint',step,flush=True)
    return results,root_rows,hashes,checks


def final_tracker_analysis(model,out,m,sets,summary,queries):
    branch=model.parent; scale=summary['world_scale']; K=np.array(m['K']); C=np.array(m['c2w']); W,H=m['width'],m['height']
    final_folder=Path(sorted(summary['snapshots'],key=lambda x:x['completed_steps'])[-1]['directory'])
    s=state(model/'photometric_s_model_native_add3.pth'); d=state(model/'photometric_d_model_native_add3.pth')
    ids=load(final_folder/'identity.npz'); assert len(ids['leaf_id'])==len(d['_xyz'])
    for fn,key in [('dynamic_state.pth',d),('static_state.pth',s)]:
        checkpoint=state(final_folder/fn)
        assert all(torch.equal(v,key[k]) for k,v in checkpoint.items() if isinstance(v,torch.Tensor))
    tr=load(BASE/'common_input/uniform_cotracker_tap.npz'); mapping=[]
    assert np.allclose(tr['timestamp_seconds'],m['timestamp_seconds'],rtol=0,atol=1e-9)
    for uv in queries['points'][:6]:
        matches=np.flatnonzero(np.all(tr['query_points']==np.r_[0.,uv],axis=-1)); assert len(matches)==1
        mapping.append(int(matches[0]))
    g,sx,sr,scales,opacity=prepare(s,d,scale); ns=len(sx); rows=[]; checks={}
    predictions=load(branch/'diagnostics/query_trajectories.npz')
    for t in range(len(m['frame_paths'])):
        xyz,rot=combine(d,g,sx,sr,scale,t); pixels=tr['tracks'][t,mapping]
        joint,dynamic=inspect_queries(xyz,rot,scales,opacity,ns,K,C,pixels,W,H)
        camera_depth=project(xyz,K,C)[1]
        depth=load(BASE/f'common_input/unidepth_depth/{t:05d}.npz')['dep']
        for spec in sets:
            qi=spec['query_index']; uv=pixels[qi]; inside=joint[qi][2]
            iz=float(depth[int(round(uv[1])),int(round(uv[0]))]) if inside else float('nan')
            r=support_row(spec,ids,xyz,ns,joint[qi],dynamic[qi],K,C,iz,camera_depth)
            r.update(input_frame_index=t,time_seconds=m['timestamp_seconds'][t],tracker_uv=uv.tolist(),
                tracker_claims_visible=bool(tr['visibility'][t,mapping[qi]]),tracker_index=mapping[qi],
                input_depth_sample='nearest pixel, estimated UniDepth; may belong to occluder')
            rows.append(r)
        if t==0:
            # CoTracker can move even its source-time coordinate by a fraction
            # of a pixel. Export validation must use the ORIGINAL manual UV,
            # while tracker diagnostics above retain actual tracker outputs.
            source=bilinear_weights(xyz,rot,scales,opacity,K,C,np.array(queries['points'][:6]),W,H)
            alpha=np.array([w.sum() for _,w,_ in source])
            fraction=np.array([w[ii>=ns].sum()/w.sum() for ii,w,_ in source])
            xyz_query=np.array([(xyz[ii]*w[:,None]).sum(0)/w.sum() for ii,w,_ in source])
            checks.update(source_alpha_max_error=float(np.max(abs(alpha-predictions['source_alpha']))),
                source_dynamic_fraction_max_error=float(np.max(abs(fraction-predictions['source_dynamic_fraction']))),
                source_xyz_max_error_m=float(np.max(abs(xyz_query-predictions['predicted'][0]))))
        keyframe=branch/f'diagnostics/keyframes/{t:05d}.npz'
        if keyframe.exists():
            k=load(keyframe); checks[f'keyframe_{t}_xyz_max_error_m']=float(np.max(abs(xyz-k['xyz_world'])))
        if t%20==0: print('lineage final tracker',t,flush=True)
    assert max(checks.values())<2e-4,checks
    return rows,checks


def csv_flat(path,rows):
    nested={k for r in rows for k,v in r.items() if isinstance(v,(list,dict))}
    flattened=[{k:v for k,v in r.items() if k not in nested} for r in rows]
    fields=list(dict.fromkeys(k for r in flattened for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(flattened)


def figures(out,rows,tracker):
    names=['hand_glove_centre','hand_glove_upper','object_front_brown']
    labels=['Glove centre','Glove upper','Box brown']; colors=['#be4b3d','#336ea0','#8a6839']
    fig,ax=plt.subplots(2,2,figsize=(13,8));fig.subplots_adjust(top=.87,bottom=.13,hspace=.3,wspace=.24)
    fig.suptitle('D seed lineage: survival is not pixel support',fontsize=17)
    for name,label,color in zip(names,labels,colors):
        rr=[r for r in rows if r['query_id']==name and r['radius_pixels']==4]
        x=[r['completed_steps'] for r in rr]
        ax[0,0].plot(x,[r['surviving_descendant_count'] for r in rr],'-o',ms=3,label=label,color=color)
        ax[0,1].plot(x,[r['descendant_fraction_of_joint_alpha'] for r in rr],'-o',ms=3,color=color)
        ax[1,0].plot(x,[r['projected_distance_pixels']['min'] if r['projected_distance_pixels'] else np.nan for r in rr],'-o',ms=3,color=color)
        tt=[r for r in tracker if r['query_id']==name and r['radius_pixels']==4]
        ax[1,1].plot([r['time_seconds'] for r in tt],[r['descendant_fraction_of_joint_alpha'] for r in tt],'-',color=color)
        vv=[r for r in tt if r['tracker_claims_visible']]
        ax[1,1].scatter([r['time_seconds'] for r in vv],[r['descendant_fraction_of_joint_alpha'] for r in vv],s=9,color=color)
        # Prespecified radius8 sensitivity is shown explicitly, not substituted
        # for empty primary sets. It remains the same input-only selection.
        sensitivity=[r for r in rows if r['query_id']==name and r['radius_pixels']==8]
        sx=[r['completed_steps'] for r in sensitivity]
        ax[0,0].plot(sx,[r['surviving_descendant_count'] for r in sensitivity],'--',color=color,alpha=.65)
        ax[0,1].plot(sx,[r['descendant_fraction_of_joint_alpha'] for r in sensitivity],'--',color=color,alpha=.65)
        ax[1,0].plot(sx,[r['projected_distance_pixels']['min'] if r['projected_distance_pixels'] else np.nan for r in sensitivity],'--',color=color,alpha=.65)
        st=[r for r in tracker if r['query_id']==name and r['radius_pixels']==8]
        ax[1,1].plot([r['time_seconds'] for r in st],[r['descendant_fraction_of_joint_alpha'] for r in st],'--',color=color,alpha=.65)
        effective=[r for r in rows if r['query_id']==name and r['radius_pixels']==-1]
        if effective:
            ex=[r['completed_steps'] for r in effective]
            ax[0,0].plot(ex,[r['surviving_descendant_count'] for r in effective],':',color=color,lw=2)
            ax[0,1].plot(ex,[r['descendant_fraction_of_joint_alpha'] for r in effective],':',color=color,lw=2)
            ax[1,0].plot(ex,[r['projected_distance_pixels']['min'] if r['projected_distance_pixels'] else np.nan for r in effective],':',color=color,lw=2)
            et=[r for r in tracker if r['query_id']==name and r['radius_pixels']==-1]
            ax[1,1].plot([r['time_seconds'] for r in et],[r['descendant_fraction_of_joint_alpha'] for r in et],':',color=color,lw=2)
    titles=['Living descendants','Source pixel: joint-alpha fraction','Closest descendant to source query','Final model: fraction at tracker UV']
    ylabels=['Count','Fraction','Pixels','Fraction']
    for a,title,yl in zip(ax.flat,titles,ylabels): a.set_title(title);a.set_ylabel(yl);a.grid(alpha=.2)
    for a in [ax[0,0],ax[0,1],ax[1,0]]:a.set_xlabel('Completed optimization steps')
    ax[1,1].set_xlabel('Actual observation time (s)');ax[0,0].legend(frameon=False,fontsize=9)
    fig.text(.08,.04,'Source frame 0, input depth +/-0.1 m, SAM2 entity. Solid: radius4; dashed: prespecified radius8 sensitivity.',fontsize=10)
    fig.text(.08,.016,'Dotted: initially effective near-human contributors (all birth frames). Tracker-panel dots: declared visibility, not GT.',fontsize=9)
    fig.savefig(out/'lineage_support.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--select-only',action='store_true');a=p.parse_args();begin=time.perf_counter();torch.set_num_threads(4)
    model=a.model.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    m=read(BASE/'common_input/input_manifest.json');q=read(BASE/'common_input/queries_first_frame.json')
    depth=load(BASE/'common_input/unidepth_depth/00000.npz')['dep']
    sets,provenance=select_roots(model,q,depth,out)
    sets += select_effective_roots(model,q,depth,m,provenance,out)
    if a.select_only:
        print(json.dumps({v['query_id']+f'_r{v["radius_pixels"]}':v['root_ids'] for v in sets},indent=2));return
    lineage=model/'lineage';summary=read(lineage/'summary.json')
    assert summary['status']=='completed' and summary['completed_steps']==8000
    ledger,final_lineage=check_ledger(lineage,summary)
    rows,root_rows,hashes,checks=snapshot_analysis(model,out,m,sets,provenance,summary)
    tracker,tcheck=final_tracker_analysis(model,out,m,sets,summary,q);checks.update(tcheck)
    # Save all allocated members of every selected root, including pruned ones.
    selected_union=np.unique([i for s in sets for i in s['root_ids']]).astype(np.int64)
    allocated=np.flatnonzero(np.isin(final_lineage['root_id'],selected_union))
    np.savez_compressed(out/'selected_root_full_history.npz',leaf_id=allocated,
        **{k:v[allocated] for k,v in final_lineage.items() if k!='live_id'},
        survives_final=np.isin(allocated,final_lineage['live_id']))
    death_events={}
    with (lineage/'events.csv').open() as f:
        for event in csv.DictReader(f):
            if event['entity']=='leaf' and event['operation']=='prune':
                removed=load(lineage/event['npz'])['id']
                for leaf_id in removed:
                    death_events[int(leaf_id)]=dict(event=int(event['event']),
                        update=int(event['completed_step_of_update']),caller=event['caller'])
    root_lifetimes=[]
    for root_id in selected_union:
        members=np.flatnonzero(final_lineage['root_id']==root_id)
        survivors=members[np.isin(members,final_lineage['live_id'])]
        root_lifetimes.append(dict(root_id=int(root_id),allocated_family_size=len(members),
            initial_particle_death_event=death_events.get(int(root_id)),
            surviving_family_members=survivors.tolist(),
            entire_lineage_extinction_update=None if len(survivors) else int(final_lineage['death_step'][members].max()),
            note='Initial-parent pruning during split is replacement, not whole-lineage extinction.'))
    dump(out/'root_lifetimes.json',root_lifetimes)
    effective_diagnosis=[]
    for spec in sets:
        if spec.get('selection_kind')!='initial_effective_near_human_contributors':continue
        for root_id in spec['root_ids']:
            rr=[r for r in root_rows if r['query_id']==spec['query_id'] and r['radius_pixels']==-1 and r['root_id']==root_id]
            rr.sort(key=lambda r:r['completed_steps']);assert rr[0]['joint_weight']>0
            zero_indices=[i for i,r in enumerate(rr) if r['joint_weight']==0]
            first_zero=zero_indices[0] if zero_indices else None
            interval=[rr[first_zero-1]['completed_steps'],rr[first_zero]['completed_steps']] if first_zero is not None else None
            lifetime=next(z for z in root_lifetimes if z['root_id']==root_id)
            effective_diagnosis.append(dict(query_id=spec['query_id'],root_id=root_id,
                first_observed_zero_contribution_interval=interval,
                interval_convention='Snapshot-bracketed transition: last sampled positive at left, first sampled zero at right. Earlier unobserved zero/recovery cannot be excluded. No exact within-interval timing claim.',
                later_sampled_contribution_recovered=any(r['joint_weight']>0 for r in rr[first_zero+1:]) if first_zero is not None else None,
                **{k:v for k,v in lifetime.items() if k!='root_id'},snapshots=rr))
    dump(out/'initial_effective_root_diagnosis.json',effective_diagnosis)
    csv_flat(out/'snapshot_support.csv',rows);csv_flat(out/'per_root_support.csv',root_rows);csv_flat(out/'final_tracker_support.csv',tracker)
    result=dict(status='completed_cpu_only',reference_geometry_used=False,checkpoint_summary=summary,
        root_selection=str((out/'selected_roots.json').resolve()),ledger_validation=ledger,numerical_validation=checks,
        snapshots=rows,final_tracker_samples=tracker,
        source_sha256={str(BASE/'common_input/input_manifest.json'):sha(BASE/'common_input/input_manifest.json'),
            str(BASE/'common_input/uniform_cotracker_tap.npz'):sha(BASE/'common_input/uniform_cotracker_tap.npz'),
            str(BASE/'common_input/queries_first_frame.json'):sha(BASE/'common_input/queries_first_frame.json'),
            str(PREV/'analysis_a/sparse_raster_exact.py'):sha(PREV/'analysis_a/sparse_raster_exact.py'),
            str(PREV/'motion_binding/audit_motion_binding.py'):sha(PREV/'motion_binding/audit_motion_binding.py'),
            str(ROOT/'third_party/MoSca/lib_mosca/scaffold_utils/dualquat_helper.py'):sha(ROOT/'third_party/MoSca/lib_mosca/scaffold_utils/dualquat_helper.py'),
            str(model/'seed_provenance.npz'):sha(model/'seed_provenance.npz'),
            str(lineage/'summary.json'):sha(lineage/'summary.json'),str(lineage/'leaf_lineage.npz'):sha(lineage/'leaf_lineage.npz')},
        snapshot_sha256=hashes,script_sha256=sha(__file__),elapsed_cpu_seconds=time.perf_counter()-begin,
        input_depth_sha256={str(BASE/f'common_input/unidepth_depth/{t:05d}.npz'):sha(BASE/f'common_input/unidepth_depth/{t:05d}.npz') for t in range(len(m['frame_paths']))},
        limitations=['Computational ancestry is not material identity. A newly observed root can replace a deleted lineage.',
            'SAM2 labels, UniDepth, and tracker coordinates/visibility are estimated input evidence, not independent truth.',
            'Kernel suppression, opacity clipping, displacement, and front-layer transmittance are separate descriptions; no single causal claim from a final state.',
            'Dynamic-only alpha removes static competition and is a diagnostic counterfactual, not the actual rendered surface.',
            'Source-pixel snapshots probe time 0 at different optimizer states; final tracker samples probe actual video times at one frozen final model.',
            'Empty selected sets, extinct roots, and zero contributions are retained. Radius8 is predetermined sensitivity, not a rescue selection.',
            'All input-depth +/-0.1m layer summaries are relative to an estimate that can belong to an occluder; they are not geometric accuracy.'])
    dump(out/'lineage_analysis.json',result);figures(out,rows,tracker)
    lines=['# D 初始近手／物种子的计算谱系与像素支持','',
        '本报告沿 D 自己的初始输入种子追踪，不用拟合真值挑种子；计算上的父子关系不自动等于真实手套材料身份。首帧联合渲染和最终 tracker 像素是两种不同诊断。','',
        '| 初始查询，半径4像素 | 初始根数 | 末期存活根数 | 末期后代数 | 初始→末期首帧联合贡献占比 |','| --- | ---: | ---: | ---: | ---: |']
    for spec in sets:
        if spec['radius_pixels']!=4:continue
        rr=[r for r in rows if r['query_id']==spec['query_id'] and r['radius_pixels']==4]
        first,last=rr[0],rr[-1]
        lines.append(f'| {spec["query_id"]} | {spec["selected_count"]} | {last["surviving_root_count"]} | {last["surviving_descendant_count"]} | {100*(first["descendant_fraction_of_joint_alpha"] or 0):.3f}% → {100*(last["descendant_fraction_of_joint_alpha"] or 0):.3f}% |')
    lines += ['', '初始实际近手贡献者的逐根结果如下。消失时刻是事件账本中的更新数；贡献丢失只能按快照给区间，不能伪造精确训练步。所有数字均为完成的更新次数；例如1801对应零基训练日志step1800。','',
        '| 查询／root | 原粒子删除事件 | 首次观察到零贡献的区间 | 后来采样恢复 | 最终活后代 | 整家族消失更新 | 同一leaf attach改变次数 |',
        '| --- | --- | --- | --- | ---: | --- | ---: |']
    for diagnosis in effective_diagnosis:
        event=diagnosis['initial_particle_death_event']
        death=f'{event["caller"]} @ {event["update"]}' if event else '未删除'
        interval=diagnosis['first_observed_zero_contribution_interval']
        loss=f'({interval[0]}, {interval[1]}]' if interval else '未观察到'
        changed=sum(len(r['same_leaf_attachment_changes']) for r in diagnosis['snapshots'])
        lines.append(f'| {diagnosis["query_id"]} / {diagnosis["root_id"]} | {death} | {loss} | {diagnosis["later_sampled_contribution_recovered"]} | {len(diagnosis["surviving_family_members"])} | {diagnosis["entire_lineage_extinction_update"]} | {changed} |')
    lines += ['', '绑定统计只对相邻快照均存活的同一leaf ID比较。附着节点ID不变不代表变形场不变：scaffold位置、旋转、近邻集合和混合权重仍可改变；逐根JSON同时保存近邻ID变化与按稳定节点ID聚合的权重L1差。split父被删除而后代保留，与整条谱系消失是不同事件。', '']
    lines += ['',f'![计算谱系、首帧贡献与最终 tracker 支持]({out}/lineage_support.png)','',
        '删除由事件账本给出确切更新编号；空间漂移由相同 root 后代的首帧投影与源位置描述；opacity 低于 1/255、像素核降到阈值以下、通过核阈值却没有联合权重分别记录。不能把所有零贡献统一称为“被删掉”。原 root 消失也不表示模型没有从别帧新增另一组表面。','',
        '首帧贡献以静态与动态的真实组合 alpha 为分母；另存“在联合动态贡献内的比例”和去掉静态后的 dynamic-only 对照，三者不能混用。最终 tracker 诊断保留所有114时刻，单独标出 tracker 自报可见；坐标或可见性可能错误，不是 GT。','',
        '另有独立命名的“初始实际近手贡献者”集合：在训练前快照0，收集两手查询处所有联合权重>0、warp0深度距输入±0.1米、原source SAM2为人的root，不限出生帧、不选topK。这个补充是在看到首帧radius4空集但存在近层支持后作出的诊断决策，冻结时间与原快照哈希记录在独立JSON；不替换radius4/radius8，也不能据此认定材料身份。图中点线表示该集合。','',
        '半径8像素敏感性与所有零贡献保存在 JSON/CSV；后代 NPZ 保留位置、投影、opacity、尺度、像素核、父与 root ID，完整生灭表包含已删除成员。此处无接触、穿透或材料对应正确性的结论。','',
        f'[固定种子集合]({out}/selected_roots.json) · [初始实际近手贡献者]({out}/selected_initial_effective_roots.json) · [逐根完整诊断与节点绑定]({out}/initial_effective_root_diagnosis.json) · [根粒子死亡与整条谱系消失]({out}/root_lifetimes.json) · [逐快照表]({out}/snapshot_support.csv) · [逐根表]({out}/per_root_support.csv) · [最终输入 tracker 表]({out}/final_tracker_support.csv) · [完整可追踪结果与数值验收]({out}/lineage_analysis.json)','',
        f'CPU耗时 {result["elapsed_cpu_seconds"]:.2f} 秒；事件与快照行序已重放验证。图片仍需人工视觉检查；不得把此自动生成说明当作完整归因结论。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],validation=checks,seconds=result['elapsed_cpu_seconds']),indent=2))


if __name__=='__main__':
    with torch.no_grad():main()
