"""Evaluation-only: frozen box SE(3) versus same-version fitted mesh, no alignment.

This script never writes inputs, predictions, templates, or reference arrays.
Analytic reference orientation extraction uses three canonical-only selected
vertex IDs, and is enabled only after exact full-mesh correspondence validation.
It does not solve an alignment objective or transform any prediction to reference.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/cai_tianshun/Project/HOI')
E = ROOT / 'experiments/structured_hoi_20260923'
OUT = E / 'object_init'
REF = ROOT / 'research/2026-09-23/evaluation_pointcloud_reference'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [plain(v) for v in x]
    return x


def angular(a, b):
    rel = a @ np.swapaxes(b, -1, -2)
    # atan2 avoids a spurious ~0.02 deg self-angle from float32 trace rounding.
    skew=np.stack([rel[...,2,1]-rel[...,1,2],rel[...,0,2]-rel[...,2,0],rel[...,1,0]-rel[...,0,1]],-1)
    return np.degrees(np.arctan2(np.linalg.norm(skew,axis=-1)/2,(np.trace(rel,axis1=-2,axis2=-1)-1)/2))


def main():
    start = time.perf_counter()
    inp = OUT / 'run03/object_init.npz'
    refp = REF / 'object_fitted_vertices_world.npz'
    inputs = [inp, refp, REF/'manifest.json', OUT/'run03/template_input_manifest.json',
              E/'code/prepare_initialization.py', E/'code/gaussian_scene.py', E/'code/evaluate_structured.py']
    for label in ['S0', 'S1']:
        base = E/f'dev1_{label}_v1/evaluation_v2'
        inputs += [base/n for n in ['object_pose.npz', 'export_manifest.json',
            'reference_evaluation/evaluation_bundle.npz', 'query_trajectories.npz',
            'surface_evaluation/metrics.json', 'support_diagnostic/diagnostic.json']]
    hashes = {str(p): sha(p) for p in inputs}
    initial = np.load(inp); ref = np.load(refp)
    manifest = json.loads((REF/'manifest.json').read_text())
    assert hashes[str(refp)] == manifest['entities']['object']['sha256']
    V = initial['canonical_vertices_m'].astype(np.float64)
    Y = ref['xyz_world'].astype(np.float64)
    frames = ref['input_frame_indices']; times = ref['frame_times']
    assert np.array_equal(times, initial['timestamp_seconds'][frames])
    C = initial['c2w']; K = initial['K']; cy = Y.mean(1)
    to_cam = lambda x: (x-C[:3,3]) @ C[:3,:3]
    project = lambda x: (to_cam(x) @ K.T)[..., :2] / to_cam(x)[..., 2:]

    # Gauge verification uses all vertex pairs; three indices selected from V only.
    # 0.1 mm is a fixed serialization tolerance, not optimized to prediction error.
    tolerance_m = 1e-4
    face_equal = np.array_equal(initial['faces'], ref['faces'])
    vertex_id_equal = np.array_equal(ref['vertex_id'], np.arange(len(V)))
    reference_rigid_error = max(float(np.max(abs(pdist(v)-pdist(V)))) for v in Y)
    a = 0
    b = int(np.argmax(np.linalg.norm(V-V[a], axis=-1)))
    c = int(np.argmax(np.linalg.norm(np.cross(V-V[a], V[b]-V[a]), axis=-1)))
    def basis(x):
        u = x[b]-x[a]; u /= np.linalg.norm(u)
        v = x[c]-x[a]; v -= u*(u@v); v /= np.linalg.norm(v)
        return np.stack([u, v, np.cross(u, v)], -1)
    B = basis(V)
    reference_R = np.array([basis(v) @ B.T for v in Y])
    reference_recovered = np.einsum('tij,nj->tni', reference_R, V-V.mean(0)) + cy[:, None]
    reconstruction_error = float(np.max(np.linalg.norm(reference_recovered-Y, axis=-1)))
    exact_gauge = bool(face_equal and vertex_id_equal and reference_rigid_error < tolerance_m and reconstruction_error < tolerance_m)

    summaries = {}; rows = []; arrays = dict(reference_centroid_world_m=cy,
        reference_centroid_camera_m=to_cam(cy), input_frame_indices=frames,
        timestamp_seconds=times, nominal_timestamp_seconds=ref['nominal_frame_times'])
    if exact_gauge: arrays['reference_R_world_analytic_no_fit'] = reference_R
    all_curves = {}
    for label in ['init', 'S0', 'S1']:
        base = E/f'dev1_{label}_v1/evaluation_v2'
        p = initial if label == 'init' else np.load(base/'object_pose.npz')
        if label != 'init':
            ex = json.loads((base/'export_manifest.json').read_text())
            assert ex['status'] == ex['evaluation_status'] == 'completed'
            assert ex['prediction_geometry_frozen_before_reference']
            assert ex['global_alignment_to_reference'] == 'none'
            assert not ex['reference_used_for_training']
        assert np.array_equal(p['timestamp_seconds'], initial['timestamp_seconds'])
        R = p['R_world'][frames].astype(np.float64)
        t = p['t_world_m' if label == 'init' else 'translation_world_m'][frames].astype(np.float64)
        X = np.einsum('tij,nj->tni', R, V) + t[:,None]
        cx = X.mean(1)  # actual template vertex centroid, never Gaussian-bank mean
        delta = cx-cy; delta_camera = delta@C[:3,:3]
        norm = np.linalg.norm(delta, axis=-1)
        displacement_error = delta-delta[0]
        # Algebraic error decomposition, not adjusted/aligned prediction metrics.
        vertex_error = X-Y
        orientation_residual = vertex_error-delta[:,None]
        mse_vertex = np.mean(np.sum(vertex_error**2,-1),-1)
        mse_orientation = np.mean(np.sum(orientation_residual**2,-1),-1)
        assert np.allclose(mse_vertex, norm**2+mse_orientation, atol=1e-12)
        theta = angular(R, reference_R) if exact_gauge else np.full(len(frames),np.nan)
        change_t = cx - (np.einsum('tij,j->ti',initial['R_world'][frames],V.mean(0))+initial['t_world_m'][frames])
        change_R = angular(R,initial['R_world'][frames])
        rmse_axes = np.sqrt(np.mean(delta_camera**2,axis=0))
        summary = dict(centroid_error_mean_cm=norm.mean()*100,
            centroid_error_median_cm=np.median(norm)*100,centroid_error_first_cm=norm[0]*100,
            centroid_error_max_cm=norm.max()*100,
            camera_delta_first_cm=delta_camera[0]*100,camera_delta_mean_cm=delta_camera.mean(0)*100,
            camera_axis_rmse_cm=rmse_axes*100,
            camera_z_share_centroid_squared_error=float(np.sum(delta_camera[:,2]**2)/np.sum(delta**2)),
            mean_source_relative_displacement_error_excluding_source_cm=np.linalg.norm(displacement_error[1:],axis=-1).mean()*100,
            adjacent_reference_slot_displacement_error_mean_cm=np.linalg.norm(np.diff(delta,axis=0),axis=-1).mean()*100,
            translation_update_from_init_mean_cm=np.linalg.norm(change_t,axis=-1).mean()*100,
            translation_update_from_init_max_cm=np.linalg.norm(change_t,axis=-1).max()*100,
            rotation_update_from_init_mean_deg=change_R.mean(),rotation_update_from_init_max_deg=change_R.max(),
            template_frame_rotation_error_mean_deg=theta.mean() if exact_gauge else None,
            template_frame_rotation_error_first_deg=theta[0] if exact_gauge else None,
            corresponding_vertex_rmse_cm=np.sqrt(mse_vertex.mean())*100,
            centroid_rmse_cm=np.sqrt(np.mean(norm**2))*100,
            orientation_vertex_residual_rmse_cm=np.sqrt(mse_orientation.mean())*100,
            mean_template_centroid_projection_error_px=np.linalg.norm(project(cx)-project(cy),axis=-1).mean(),
            unobserved_slots=[int(i) for i in frames if not initial['observed_mask'][i]])
        arrays.update({f'{label}_centroid_world_m':cx,f'{label}_centroid_camera_m':to_cam(cx),
            f'{label}_delta_world_m':delta,f'{label}_delta_camera_m':delta_camera,
            f'{label}_source_relative_displacement_error_world_m':displacement_error,
            f'{label}_reference_template_rotation_error_deg':theta,
            f'{label}_corresponding_vertex_mse_m2':mse_vertex,
            f'{label}_orientation_vertex_residual_mse_m2':mse_orientation})
        query_norm = None
        if label != 'init':
            q=np.load(base/'reference_evaluation/evaluation_bundle.npz')
            assert np.array_equal(q['frame_times'],times)
            select=q['entity']=='object'; valid=q['valid_mask'][:,select]
            qe=q['predicted'][:,select]-q['reference'][:,select]
            qe_cam=qe@C[:3,:3]; qoffset=qe-delta[:,None]
            qnorm=np.linalg.norm(qe,axis=-1); query_norm=qnorm.mean(1)
            qp=np.load(base/'query_trajectories.npz')
            sq=json.loads((base/'surface_evaluation/metrics.json').read_text())
            support=json.loads((base/'support_diagnostic/diagnostic.json').read_text())
            q_squared=np.sum(qe**2,-1); center_squared=np.broadcast_to(norm[:,None]**2,q_squared.shape)
            offset_squared=np.sum(qoffset**2,-1); cross=2*np.sum(delta[:,None]*qoffset,-1)
            assert np.allclose(q_squared,center_squared+offset_squared+cross,atol=1e-12)
            summary['queries']=dict(query_ids=q['query_id'][select],count=int(valid.sum()),
                mean_cm=qnorm[valid].mean()*100,first_each_cm=qnorm[0]*100,
                first_each_camera_delta_cm=qe_cam[0]*100,
                mean_centroid_relative_offset_error_cm=np.linalg.norm(qoffset,axis=-1)[valid].mean()*100,
                first_centroid_relative_offset_error_cm=np.linalg.norm(qoffset[0],axis=-1)*100,
                min_source_object_contribution=qp['source_entity_fractions'][select,2].min(),
                mse_decomposition_cm2=dict(total=q_squared[valid].mean()*1e4,
                    centroid=center_squared[valid].mean()*1e4,centroid_relative_offset=offset_squared[valid].mean()*1e4,
                    cross=cross[valid].mean()*1e4),
                caveat='Query offset combines orientation, source-ray surface selection, weighted Gaussian centres, and reference definition; terms do not add as Euclidean distances.')
            summary['surface_proxy']=sq['summary']['object']
            summary['input_support_objects']=[r for r in support['summary'] if r['query_id'].startswith('object')]
            arrays[f'{label}_query_error_world_m']=qe
            arrays[f'{label}_query_centroid_relative_offset_error_world_m']=qoffset
        for j,f in enumerate(frames):
            row=dict(method=label,slot=j,input_frame=int(f),nominal_s=float(ref['nominal_frame_times'][j]),
                input_s=float(times[j]),input_minus_nominal_s=float(times[j]-ref['nominal_frame_times'][j]),
                init_observed=bool(initial['observed_mask'][f]),init_depth_observed=bool(initial['depth_observed_mask'][f]),
                centroid_error_cm=norm[j]*100,rotation_error_deg=theta[j] if exact_gauge else None,
                source_relative_displacement_error_cm=np.linalg.norm(displacement_error[j])*100,
                template_centroid_projection_error_px=np.linalg.norm(project(cx[j])-project(cy[j])),
                translation_update_from_init_cm=np.linalg.norm(change_t[j])*100,
                rotation_update_from_init_deg=change_R[j],
                corresponding_vertex_rmse_cm=np.sqrt(mse_vertex[j])*100,
                orientation_vertex_residual_rmse_cm=np.sqrt(mse_orientation[j])*100)
            for coord,arr in [('world',delta),('camera',delta_camera)]:
                for axis,z in zip('xyz',arr[j]): row[f'delta_{coord}_{axis}_cm']=float(z*100)
            rows.append(row)
        summary['event_displacements']=[]
        # Prespecified input-mask gaps; reference only supplies sparse bracketing endpoints.
        for before,after in [(16,31),(46,65),(0,113)]:
            j=int(np.where(frames==before)[0][0]);k=int(np.where(frames==after)[0][0])
            pd=cx[k]-cx[j];rd=cy[k]-cy[j];ed=pd-rd
            summary['event_displacements'].append(dict(start=before,end=after,
                actual_input_dt_seconds=times[k]-times[j],predicted_world_m=pd,reference_world_m=rd,
                delta_world_cm=ed*100,delta_camera_cm=ed@C[:3,:3]*100,error_cm=np.linalg.norm(ed)*100,
                endpoint_reference_time_offsets_s=[times[j]-ref['nominal_frame_times'][j],times[k]-ref['nominal_frame_times'][k]]))
        summaries[label]=summary
        all_curves[label]=(delta_camera*100,norm*100,np.linalg.norm(displacement_error,axis=-1)*100,theta,query_norm)

    # Writing diagnostic-only outputs after input and prediction hashes were frozen.
    result=dict(status='completed',role='evaluation_only_post_training_no_updates',
        sources=hashes,reference_type=manifest['reference_type'],coordinate_frame=manifest['coordinate_frame'],
        time_caveat=manifest['time_caveat'],maximum_time_offset_s=manifest['max_absolute_nominal_rgb_offset_seconds'],
        centroid_definition='Arithmetic mean of same 517 indexed mesh vertices after each frozen rigid transform; not t assumed blindly, not area/volume/Gaussian centroid.',
        error_sign='prediction minus fitted reference; camera x right, y down, z forward',
        no_alignment_or_refit=True,no_model_or_init_updates=True,
        gauge_verification=dict(exact_gauge_verified=exact_gauge,faces_identical=face_equal,vertex_ids_identical=vertex_id_equal,
            all_pair_distance_max_discrepancy_m=reference_rigid_error,canonical_only_selected_basis_vertex_ids=[a,b,c],
            full_mesh_analytic_reconstruction_max_error_m=reconstruction_error,fixed_numerical_tolerance_m=tolerance_m,
            method='Construct orthonormal basis from fixed three corresponding edge vectors. No SVD, ICP, optimization, nearest-neighbour reassociation, or prediction alignment.',
            material_identity_limit='Exact fitted-template vertex gauge is not independently verified real material-point identity; no symmetry-minimized rotation is reported.'),
        summaries=summaries,rows=rows)
    result['source_hashes_unchanged_after_read'] = all(sha(p)==hashes[str(p)] for p in inputs)
    assert result['source_hashes_unchanged_after_read']
    result['wall_seconds_before_plot']=time.perf_counter()-start
    (OUT/'post_train_diagnosis.json').write_text(json.dumps(plain(result),ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(OUT/'post_train_diagnosis_arrays.npz',**arrays)
    with (OUT/'post_train_diagnosis_per_slot.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
    for label,color in [('init','0.55'),('S0','#dc8037'),('S1','#1266b1')]:
        dc,norm,dis,rot,qn=all_curves[label]
        ax[0,1].plot(times,norm,'o-',label=label,color=color)
        ax[1,0].plot(times,dis,'o-',label=label,color=color)
        ax[1,1].plot(times,rot,'o-',label=label,color=color)
    for j,color in enumerate(['#377eb8','#4daf4a','#e41a1c']):
        ax[0,0].plot(times,all_curves['S1'][0][:,j],'o-',color=color,label='camera '+['X','Y','Z'][j])
    ax[0,0].axhline(0,color='0.6',lw=.7)
    ax[0,0].set_title('S1 centroid delta: prediction - fitted reference')
    ax[0,1].plot(times,all_curves['S1'][4]*100,'--',color='#6a3d9a',label='S1 query mean (different quantity)')
    ax[0,1].set_title('Absolute centroid / query errors')
    ax[1,0].set_title('Displacement error relative to source frame 0')
    ax[1,1].set_title('Exact template-frame rotation difference (no symmetry fit)')
    for axes in ax.flat:
        axes.set_xlabel('Actual camera0 timestamp (s)');axes.set_ylabel('cm');axes.grid(alpha=.22);axes.legend(fontsize=8)
        for j in [25,55]:
            axes.axvline(initial['timestamp_seconds'][j],color='0.6',ls=':',lw=1)
    ax[1,1].set_ylabel('degrees');ax[1,1].set_ylim(0,185)
    fig.suptitle('14 frozen sparse fitted-reference slots; no alignment, no refitting. Dotted: unobserved input slots 25/55.',fontsize=11)
    fig.savefig(OUT/'post_train_diagnosis.png',dpi=160);plt.close(fig)
    print(json.dumps(plain({k:{kk:v for kk,v in s.items() if kk not in ['input_support_objects','event_displacements']} for k,s in summaries.items()}),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
