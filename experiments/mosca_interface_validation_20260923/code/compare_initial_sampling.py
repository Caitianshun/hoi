"""Input-only sampling coverage and first-frame rendering, A versus C."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
from pytorch3d.transforms import quaternion_to_matrix
ROOT=Path('/home/cai_tianshun/Project/HOI');EXP=ROOT/'experiments/mosca_interface_validation_20260923';PREV=ROOT/'experiments/mosca_validation_20260923';BASE=ROOT/'experiments/mosca_baseline_20260922'
sys.path.insert(0,str(PREV/'analysis_a'));from sparse_raster_exact import sparse_raster
sys.path.insert(0,str(PREV/'motion_binding'));from audit_motion_binding import get_weights,warp
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=EXP/'initial_sampling_ad.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(4)
    meta=json.loads((BASE/'common_input/input_manifest.json').read_text());q=json.loads((BASE/'common_input/queries_first_frame.json').read_text())
    labels=np.load(BASE/'segmentation/segmentation.npz')['entity_labels'];tr=np.load(BASE/'common_input/uniform_cotracker_tap.npz')
    scale=json.loads((PREV/'a_normalized_exact/model/run.json').read_text())['world_scale'];C=np.array(meta['c2w']);K=np.array(meta['K'])
    record={'reference_read':False,'patch_definition':'L-infinity radius 2,4,8 pixels around the six original queries/visible input CoTracker correspondences; not anatomical ground-truth regions.','cases':{}}
    cases={'A':(ROOT/'experiments/mosca_hand_trace_20260923/sampling/run01/actual_dynamic_seeds.npz',PREV/'a_normalized_exact/model'),
           'D':(EXP/'d_exact_geometry/model/seed_provenance.npz',EXP/'d_exact_geometry/model')}
    for name,(path,folder) in cases.items():
        seeds=np.load(path);f=seeds['raw_source_frame'];uv=seeds['raw_pixel_xy'];lab=labels[f,uv[:,1],uv[:,0]]
        s=torch.load(folder/'initial_static.pth',map_location='cpu',weights_only=False);d=torch.load(folder/'initial_dynamic.pth',map_location='cpu',weights_only=False)
        g=get_weights(d);dx,dr=warp(d,g,0);ns=len(s['_xyz'])
        xyz=torch.cat([s['_xyz'],dx]).numpy()/scale
        rot=torch.cat([quaternion_to_matrix(torch.nn.functional.normalize(s['_rotation'],dim=-1)),dr]).numpy()
        sizes=torch.cat([v['min_scale']+torch.sigmoid(v['_scaling'])*(v['max_scale']-v['min_scale']) for v in [s,d]]).numpy()/scale
        opacity=torch.cat([torch.sigmoid(v['_opacity']) for v in [s,d]]).numpy()
        src=sparse_raster(xyz,rot,sizes,opacity,K,C,np.array(q['points'][:6]));camz=((xyz-C[:3,3])@C[:3,:3])[:,2]
        rows=[]
        for j in range(6):
            candidates=np.flatnonzero(f==0);dist=np.linalg.norm(uv[candidates]-q['points'][j],axis=1)
            nearest=candidates[np.argmin(dist)];neighborhood=np.max(abs(uv[candidates]-q['points'][j]),axis=1)
            tracker_idx=np.flatnonzero((tr['query_points'][:,0]==0)&(tr['query_points'][:,1:]==np.array(q['points'][j])[None]).all(1))
            assert len(tracker_idx)==1;ti=int(tracker_idx[0]);times=np.flatnonzero(tr['visibility'][:,ti])
            coverage={str(r):[] for r in [2,4,8]}
            for t in times:
                delta=np.max(abs(uv[f==t]-tr['tracks'][t,ti]),axis=1)
                for r in [2,4,8]:coverage[str(r)].append(int((delta<=r).sum()))
            ids,w=src[j];dyn=ids>=ns;total=w.sum();mean=float(np.dot(w,camz[ids])/total)
            dep=float(np.load(BASE/'common_input/unidepth_depth/00000.npz')['dep'][q['points'][j][1],q['points'][j][0]])
            rows.append({'query_id':q['point_ids'][j],'nearest_first_frame_seed_id':int(nearest),
                'nearest_first_frame_seed_pixel':uv[nearest].tolist(),'nearest_first_frame_pixel_distance':float(dist.min()),
                'first_frame_patch_seed_counts':{str(r):int((neighborhood<=r).sum()) for r in [2,4,8]},
                'input_tracker_visible_frames':len(times),'visible_track_patch_coverage':{r:{'frames_with_seed':int((np.array(v)>0).sum()),'total_frames':len(v),'total_seeds':sum(v)} for r,v in coverage.items()},
                'initial_source_alpha':float(total),'initial_dynamic_fraction':float(w[dyn].sum()/total),
                'initial_expected_depth_m':mean,'input_pixel_depth_m':dep,
                'initial_near_input_10cm_alpha_fraction':float(w[np.abs(camz[ids]-dep)<=.1].sum()/total),
                'initial_far_behind_25cm_alpha_fraction':float(w[camz[ids]-dep>.25].sum()/total)})
        record['cases'][name]={'sample_count':len(f),'seed_source_label_counts':{str(k):int((lab==k).sum()) for k in [0,1,2]},
            'candidate_count':int(seeds['pool_counts_by_frame'].sum()),'first_frame_sample_count':int((f==0).sum()),'rows':rows,
            'measured_source_pixel_roundtrip_max_px':float(np.linalg.norm((((seeds['raw_world_xyz_m']-C[:3,3])@C[:3,:3])@K.T)[:,:2]/((((seeds['raw_world_xyz_m']-C[:3,3])@C[:3,:3])@K.T)[:,2:])-uv,axis=-1).max()),'provenance_sha256':sha(path),'dynamic_checkpoint_sha256':sha(folder/'initial_dynamic.pth')}
    out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({name:{'labels':r['seed_source_label_counts'],'hands':r['rows'][4:]} for name,r in record['cases'].items()},indent=2))
if __name__=='__main__':
    with torch.no_grad():main()
