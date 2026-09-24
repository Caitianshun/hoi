"""Score only original candidate frames having an identical frozen reference slot."""
import numpy as np
import evaluate_diagnosis as D


def main():
    freeze_path=D.ROOT/'evaluation/historical_pool_freeze.json'
    freeze=D.read(freeze_path)
    D.validate_freeze(freeze)
    audit=D.read(D.OLD/'protocol/audit_inputs_manifest.json')
    rows=[]
    for dev in ('dev1','dev2'):
        ini=np.load(audit['old_S1'][dev]['object_init'])
        v=ini['canonical_vertices_m'].astype(float);C=ini['c2w'].astype(float)
        rp=next(r for r in audit['evaluation_only'][dev]['files'] if r['path'].endswith('reference_meshes_world.npz'))
        assert D.E.sha(rp['path'])==rp['sha256']
        ref=np.load(rp['path']);Rref,tref,verification=D.E.gauge(v,ini['faces'],ref)
        assert verification['verified']
        # Exact existing input-frame matching only; do not create new temporal labels.
        slots={int(f):j for j,f in enumerate(ref['matched_input_indices']) if ref['reference_available'][j]}
        a=np.load(freeze['dev'][dev]['single_keyframe_candidates']['file']['path'])
        for i,frame in enumerate(a['frame_indices']):
            slot=slots.get(int(frame))
            for k in range(a['R_camera'].shape[1]):
                row=dict(dev=dev,input_frame=int(frame),candidate_index=k,input_s=float(a['timestamp_seconds'][i]),
                         historical_official_score=float(a['scores'][i,k]),reference_slot=slot,
                         status='scored_same_original_slot' if slot is not None else 'N/A_no_same_original_valid_slot',
                         centroid_error_cm=None,depth_bias_cm=None,depth_absolute_error_cm=None,
                         raw_rotation_error_deg=None,corresponding_vertex_rmse_cm=None)
                if slot is not None:
                    R=C[:3,:3]@a['R_camera'][i,k];t=C[:3,:3]@a['t_camera_m'][i,k]+C[:3,3]
                    pred=v@R.T+t;truth=ref['object_vertices_world_m'][slot]
                    delta=pred.mean(0)-truth.mean(0);depth=float((delta@C[:3,:3])[2]*100)
                    row.update(centroid_error_cm=float(np.linalg.norm(delta)*100),depth_bias_cm=depth,
                               depth_absolute_error_cm=abs(depth),raw_rotation_error_deg=float(D.E.angle(R,Rref[slot])),
                               corresponding_vertex_rmse_cm=float(np.sqrt(np.mean(np.sum((pred-truth)**2,axis=1)))*100))
                rows.append(row)
    D.validate_freeze(freeze)
    output=dict(status='completed',freeze=D.identity(freeze_path),script=D.identity(__file__),rows=rows,
                requested=50,scored=sum(r['reference_slot'] is not None for r in rows),
                no_reference_interpolation=True,no_new_nearest_time_matching=True,
                no_path_or_frame_selection_from_reference=True,
                note='Only dev1 frame83 has an exact original valid reference slot; other 45 candidates remain N/A.')
    D.write(D.ROOT/'evaluation/single_keyframe_candidate_diagnostic.json',output)
    D.csvwrite(D.ROOT/'evaluation/single_keyframe_candidate_diagnostic.csv',rows)
    print('Single-keyframe coverage:',output['scored'],'of',output['requested'])


if __name__=='__main__':main()
