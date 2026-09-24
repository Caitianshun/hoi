"""Evaluation-only conversion of existing same-version fitted meshes, CPU only."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/cai_tianshun/Project/HOI')
EXP=ROOT/'experiments/mosca_baseline_20260922'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/diagnostics'))
from mesh_query_reference import load_mesh
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest_path=EXP/'data/behave_event_reference_manifest.json'
time_path=EXP/'evaluation/fixed_rgb_queries/same_version_joint_reference_observation_times_protocol.json'
m=json.loads(manifest_path.read_text());p=json.loads(time_path.read_text())
assert m['role']=='evaluation_only' and p['reference_used_for_training'] is False
assert p['coordinate_frame']=='behave_world_k1_color' and p['units']=='m'
nominal=np.asarray(p['nominal_frame_times_seconds'])
actual=np.asarray(p['observation_times_seconds'])
input_indices=np.asarray(p['prediction_frame_indices'])
assert len(nominal)==14
outputs={};entities={};records=[]
for label,suffix,entity_code in [('person','person/fit02/person_fit.ply',1),('object','boxsmall/fit01/boxsmall_fit.ply',2)]:
    verts=[];faces0=None;hashes=[];paths=[]
    for ni in nominal:
        member=f'Date01_Sub01_boxsmall_hand/t{ni:08.3f}/{suffix}'
        rr=[r for r in m['records']if r['member']==member]
        assert len(rr)==1
        rec=rr[0];mesh=Path(rec['path']);digest=sha(mesh)
        assert digest==rec['sha256']
        v,f=load_mesh(mesh)
        if faces0 is None:faces0=f.copy()
        assert np.array_equal(f,faces0),'topology/face ordering changed'
        assert len(v)==len(verts[0]) if verts else True
        verts.append(v.astype(np.float32));hashes.append(digest);paths.append(str(mesh))
        records.append(dict(entity=label,nominal_seconds=float(ni),path=str(mesh),sha256=digest,source_url=rec['url'],archive_member=member))
    xyz=np.stack(verts);vertex_id=np.arange(xyz.shape[1],dtype=np.int32)
    dst=OUT/f'{label}_fitted_vertices_world.npz'
    np.savez_compressed(dst,xyz_world=xyz,vertex_id=vertex_id,faces=faces0.astype(np.int32),
        nominal_frame_times=nominal,frame_times=actual,input_frame_indices=input_indices,
        source_paths=np.asarray(paths),source_sha256=np.asarray(hashes),
        entity=np.array(label),entity_code=np.array(entity_code),coordinate_frame=np.array('behave_world_k1_color'),
        units=np.array('m'),role=np.array('evaluation_only_registered_fit_not_sensor_ground_truth'),
        reference_used_for_training=np.array(False),interpolated=np.array(False))
    reload=np.load(dst,allow_pickle=False)
    assert np.array_equal(reload['xyz_world'],xyz) and np.array_equal(reload['vertex_id'],vertex_id)
    entities[label]=dict(xyz=xyz,entity_code=entity_code)
    outputs[label]=dict(path=str(dst),sha256=sha(dst),shape=list(xyz.shape),face_shape=list(faces0.shape),
        identical_face_indices_all_14_frames=True,face_indices_sha256=hashlib.sha256(faces0.astype('<i4').tobytes()).hexdigest(),
        point_identity='entity plus original template vertex_id; no nearest-neighbour reassociation')

selected=[0,3,4] # nominal18,21,22, covers anchor/severe event/reappearance
plyrows=[]
for ti in selected:
    xyz=np.concatenate([entities[e]['xyz'][ti]for e in ['person','object']])
    eid=np.concatenate([np.full(entities[e]['xyz'].shape[1],entities[e]['entity_code'])for e in ['person','object']])
    vid=np.concatenate([np.arange(entities[e]['xyz'].shape[1])for e in ['person','object']])
    arr=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('entity_id','u1'),('vertex_id','<i4')])
    for k,c in enumerate('xyz'):arr[c]=xyz[:,k]
    colors=np.where((eid==1)[:,None],np.array([70,160,225]),np.array([240,140,50]))
    for k,c in enumerate(['red','green','blue']):arr[c]=colors[:,k]
    arr['entity_id']=eid;arr['vertex_id']=vid
    dst=OUT/f'combined_nominal_{int(nominal[ti]):02d}s.ply'
    PlyData([PlyElement.describe(arr,'vertex')],text=False,byte_order='<',comments=[
        'EVALUATION ONLY - fitted registration, not measured sensor GT',
        f'world Kinect1 color; metres; nominal {nominal[ti]:.3f}; matched RGB {actual[ti]:.6f}',
        'entity_id1 person, entity_id2 object; vertex_id stable within entity']).write(dst)
    reread=PlyData.read(dst)['vertex'].data
    assert np.array_equal(reread['vertex_id'],vid) and np.array_equal(reread['entity_id'],eid)
    plyrows.append(dict(path=str(dst),sha256=sha(dst),nominal_s=float(nominal[ti]),actual_rgb_s=float(actual[ti]),input_index=int(input_indices[ti]),point_count=len(xyz)))

# Coordinates displayed as X,Z,-Y only so the upright human reads naturally;
# data are NOT transformed or aligned in the saved NPZ/PLY.
allxyz=np.concatenate([entities[e]['xyz'][selected].reshape(-1,3)for e in ['person','object']])
display=allxyz[:,[0,2,1]]*np.array([1,1,-1]);center=(display.min(0)+display.max(0))/2;radius=float(np.ptp(display,axis=0).max()/2*1.08)
fig=plt.figure(figsize=(12,4.6),dpi=140)
for j,ti in enumerate(selected):
    ax=fig.add_subplot(1,3,j+1,projection='3d')
    for entity,color,size in [('person','#469fe0',1.4),('object','#ef8d33',5)]:
        pts=entities[entity]['xyz'][ti];pts=pts[:,[0,2,1]]*np.array([1,1,-1])
        ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=size,c=color,depthshade=False,label=entity)
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),
        xlabel='World X (m)',ylabel='World Z (m)',zlabel='-World Y (m)',
        title=f'Fit t={nominal[ti]:.0f}s | RGB {actual[ti]:.3f}s')
    ax.view_init(elev=12,azim=-65);ax.set_box_aspect((1,1,1));ax.tick_params(labelsize=7)
    if j==0:ax.legend(loc='upper left',fontsize=8)
fig.suptitle('Evaluation-only fitted vertices | stable IDs | no temporal interpolation',fontsize=12)
fig.subplots_adjust(left=.01,right=.98,bottom=.06,top=.87,wspace=.05)
preview=OUT/'fitted_pointcloud_preview.png';fig.savefig(preview);plt.close(fig)
record=dict(status='completed_cpu_only',role='evaluation_only',reference_used_for_training=False,
    reference_type='official_Date01_1fps_same_version_fitted_meshes_not_sensor_ground_truth',
    coordinate_frame='behave_world_k1_color',units='m',inputs=[dict(path=str(q),sha256=sha(q))for q in [manifest_path,time_path]],
    original_meshes=records,entities=outputs,preview_ply=plyrows,preview_png=str(preview),
    nominal_frame_times_seconds=nominal.tolist(),matched_camera0_rgb_times_seconds=actual.tolist(),
    input_frame_indices=input_indices.tolist(),max_absolute_nominal_rgb_offset_seconds=float(abs(nominal-actual).max()),
    time_caveat=p['time_warning'],interpolated=False,no_predictions_loaded=True,
    limitations=['Fitted full surfaces include unobserved regions supplied by templates and registration.',
        'No scene/background point cloud is contained in this human/object reference.',
        'Fixed template vertex IDs are correspondence of fitted surfaces, not proof of real material identity.',
        'No exact contact, penetration, glove surface, visibility or multi-camera synchronization ground truth.',
        'Do not mix with the differing official30fps object parameter fit.',
        'Do not feed these vertices, silhouettes, labels or derived losses to RGB-only baseline training.'])
(OUT/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'entities':outputs,'ply':plyrows,'preview':str(preview)},ensure_ascii=False,indent=2))
