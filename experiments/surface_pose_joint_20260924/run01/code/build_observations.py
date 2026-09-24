"""Frozen candidate surface observation graph, built from permitted input assets only.
No reference loader, new tracker, candidate-pose selector, or optimization.
"""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path('/home/cai_tianshun/Project/HOI');OLD=ROOT/'experiments/object_pose_refinement_20260924/run01';NEW=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v) for v in x]
 return x

def save(p,a):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(plain(a),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def local_quality(texture,boundary):return float(texture/(texture+10)*boundary/(boundary+3))

def ray_hit(uv,R,t,V,F,K):
 ray=np.linalg.solve(K,np.r_[uv,1.]);tri=(V@R.T+t)[F];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];h=np.cross(np.broadcast_to(ray,e2.shape),e2);det=np.sum(e1*h,1);ok=abs(det)>1e-9;inv=np.divide(1.,det,out=np.zeros_like(det),where=ok);sv=-tri[:,0];u=inv*np.sum(sv*h,1);cross=np.cross(sv,e1);v=inv*(cross@ray);depth=inv*np.sum(e2*cross,1);good=ok&(u>=0)&(v>=0)&(u+v<=1)&(depth>0);depth[~good]=np.inf;i=int(depth.argmin())
 if not np.isfinite(depth[i]):return None
 bary=np.array([1-u[i]-v[i],u[i],v[i]]);q=bary@V[F[i]];return q,i,bary

def time_runs(frames,ts):
 runs=[]
 for f in frames:
  if not runs or f!=runs[-1][-1]+1:runs.append([f])
  else:runs[-1].append(f)
 return [dict(first_frame=r[0],last_frame=r[-1],first_time=float(ts[r[0]]),last_time=float(ts[r[-1]]),span_seconds=float(ts[r[-1]]-ts[r[0]]),frames=len(r)) for r in runs]

def components(nframes,ntracks,edges,split=None):
 parent=np.arange(nframes+ntracks)
 def find(x):
  while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):parent[find(a)]=find(b)
 for i,e in enumerate(edges):
  if split is None or split[i]==0:union(int(e['frame']),nframes+int(e['track']))
 groups=collections.defaultdict(lambda:dict(frames=[],tracks=[]))
 for f in range(nframes):groups[find(f)]['frames'].append(f)
 for k in range(ntracks):groups[find(nframes+k)]['tracks'].append(k)
 return list(groups.values())

def build(dev):
 out=NEW/'observations'/dev;out.mkdir(exist_ok=False)
 oldinit=ROOT/('experiments/structured_hoi_20260923/object_init/run03/object_init.npz' if dev=='dev1' else 'experiments/structured_hoi_20260923/object_init/dev2_refined01/object_init.npz')
 a=dict(np.load(oldinit));V=a['canonical_vertices_m'].astype(float);F=a['faces'].astype(int);R=a['R_camera'].astype(float);t=a['t_camera_m'].astype(float);K=a['K'].astype(float);ts=a['timestamp_seconds'];L=len(ts)
 basedir=ROOT/('experiments/mosca_baseline_20260922/common_input' if dev=='dev1' else 'experiments/structured_hoi_20260923/data/dev2');manifest=basedir/'input_manifest.json';m=json.loads(manifest.read_text());assert m['role']=='input_only';seg=ROOT/'experiments/mosca_baseline_20260922/segmentation/segmentation.npz' if dev=='dev1' else basedir/'segmentation/segmentation.npz';labels=np.load(seg)['entity_labels'];H,W=labels.shape[-2:]
 files=sorted((OLD/'pose'/dev).glob('tracklet_keyframe_*.npz'));ledger=[];candidates=[];allreasons=collections.Counter();seen_queries=set();duplicates=[];raw_reliable=0;raw_records=0;reliable_perframe=np.zeros(L,int);hist={}
 for file in files:
  tr=dict(np.load(file));N=tr['adopted'].shape[1];raw_reliable+=int(tr['adopted'].sum());raw_records+=tr['adopted'].size
  for f in range(L):reliable_perframe[f]+=int(tr['adopted'][tr['target_frame']==f].sum())
  for qidx in range(N):
   oldsource=int(tr['source_frame'][0,qidx]);olduv=tr['source_uv'][0,qidx].astype(float);oldid=int(tr['track_id'][0,qidx]);key=(oldsource,*olduv.tolist());entry={'file':str(file),'file_sha256':sha(file),'original_track_id':oldid,'original_source_frame':oldsource,'original_query':qidx,'original_source_uv':olduv.tolist(),'candidate_material_correspondence':True,'confirmed_material_identity':False,'accepted_for_graph':False}
   if key in seen_queries:entry['rejection']='duplicate_original_source_query';ledger.append(entry);duplicates.append(entry);continue
   seen_queries.add(key);valid=np.flatnonzero(tr['adopted'][:,qidx]);entry['reliable_nonself_records']=len(valid)
   hist[str(len(valid))]=hist.get(str(len(valid)),0)+1
   if len(valid)<6:entry['rejection']='fewer_than_6_adopted_nonself_records';ledger.append(entry);continue
   observations={}
   # Original query is a measured source pixel; now explicitly retained, not
   # counted as an additional independent temporal measurement.
   observations[oldsource]={'frame':oldsource,'uv':olduv,'quality':local_quality(float(tr['source_gray_patch_std'][0,qidx]),float(tr['source_boundary_distance_px'][0,qidx])),'source_rank_quality':local_quality(float(tr['source_gray_patch_std'][0,qidx]),float(tr['source_boundary_distance_px'][0,qidx])),'original_query_measurement':True,'cycle_error_px':None,'original_row':-1,'texture_std':float(tr['source_gray_patch_std'][0,qidx]),'boundary_px':float(tr['source_boundary_distance_px'][0,qidx])}
   for j in valid:
    frame=int(tr['target_frame'][j,qidx]);uv=tr['target_uv'][j,qidx].astype(float);assert np.isfinite(uv).all() and tr['target_5x5_interior'][j,qidx] and tr['cycle_tested'][j,qidx] and tr['cycle_error_px'][j,qidx]<=2 and tr['visibility'][j,qidx] and tr['return_visibility'][j,qidx]
    quality=local_quality(float(tr['target_gray_patch_std'][j,qidx]),float(tr['target_boundary_distance_px'][j,qidx]));edge={'frame':frame,'uv':uv,'quality':quality/(1+(float(tr['cycle_error_px'][j,qidx])/2)**2),'source_rank_quality':quality,'original_query_measurement':False,'cycle_error_px':float(tr['cycle_error_px'][j,qidx]),'original_row':int(j),'texture_std':float(tr['target_gray_patch_std'][j,qidx]),'boundary_px':float(tr['target_boundary_distance_px'][j,qidx])}
    if frame in observations:
     duplicates.append(dict(original_track_id=oldid,frame=frame,type='duplicate_same_track_frame'))
     if observations[frame]['original_query_measurement'] or observations[frame]['quality']>=edge['quality']:continue
    observations[frame]=edge
   obs=sorted(observations.values(),key=lambda x:x['frame'])
   if len(obs)<3:entry['rejection']='fewer_than_3_distinct_times';ledger.append(entry);continue
   source=min(obs,key=lambda x:(-x['source_rank_quality'],x['frame']));sf=source['frame'];suv=source['uv'];entry.update(source_frame=sf,source_uv=suv.tolist(),source_input_local_quality=source['source_rank_quality'])
   # Source identity is selected before any ray hit; failure rejects the track,
   # never switches to another source for easier geometry.
   hit=ray_hit(suv,R[sf],t[sf],V,F,K)
   if hit is None:entry['rejection']='selected_source_no_frontmost_template_triangle_hit';ledger.append(entry);continue
   q0,face,bary=hit;non_source=[x for x in obs if x['frame']!=sf];nhold=max(1,int(np.floor(.2*len(non_source)+.5)));nhold=min(nhold,len(non_source)-2)
   if nhold<1:entry['rejection']='cannot_keep_source_plus2targets_and_holdout';ledger.append(entry);continue
   # Mid-bin deterministic stratified indices in chronological non-source list.
   holdindices=set(np.floor((np.arange(nhold)+.5)*len(non_source)/nhold).astype(int).tolist());holdframes={non_source[j]['frame'] for j in holdindices}
   for e in obs:e['split']=int(e['frame'] in holdframes);e['is_source']=e['frame']==sf
   assert sum(e['split']==0 and not e['is_source'] for e in obs)>=2 and sum(e['is_source'] for e in obs)==1 and not source['split']
   maskxy=np.argwhere(labels[sf]==2)[:,::-1];mn=maskxy.min(0);span=np.maximum(1,maskxy.max(0)-mn+1);grid=np.minimum(3,np.maximum(0,((suv-mn)/span*4).astype(int)));tb=min(4,int(5*(ts[sf]-ts[0])/max(1e-9,ts[-1]-ts[0])));cell=(tb,int(grid[1]),int(grid[0]));entry.update(rejection=None,initializable=True,q0=q0.tolist(),face0=face,bary0=bary.tolist(),sampling_cell=cell,optimization_records=sum(e['split']==0 for e in obs),heldout_records=sum(e['split']==1 for e in obs),time_span_seconds=float(ts[obs[-1]['frame']]-ts[obs[0]['frame']]))
   cand={'ledger_index':len(ledger),'entry':entry,'obs':obs,'source':source,'q0':q0,'face0':face,'bary0':bary,'cell':cell};ledger.append(entry);candidates.append(cand)
  for bit in [1,2,4,8,16,32,64,128,256,512]:allreasons[str(bit)]+=int(((tr['reason_bits']&bit)!=0).sum())
 cells=collections.defaultdict(list)
 for c in candidates:cells[c['cell']].append(c)
 for cell in cells:cells[cell].sort(key=lambda c:(-c['entry']['reliable_nonself_records'],-c['source']['source_rank_quality'],c['entry']['original_source_frame'],c['entry']['original_query']))
 selected=[]
 while len(selected)<128:
  any_added=False
  for cell in sorted(cells):
   if cells[cell] and len(selected)<128:selected.append(cells[cell].pop(0));any_added=True
  if not any_added:break
 selected.sort(key=lambda c:(c['entry']['original_source_frame'],c['entry']['original_query']));selected_idx={c['ledger_index'] for c in selected}
 for c in candidates:
  c['entry']['accepted_for_graph']=c['ledger_index'] in selected_idx
  if not c['entry']['accepted_for_graph']:c['entry']['rejection']='deterministic_time_mask_grid_budget128'
 edges=[]
 for k,c in enumerate(selected):
  c['entry']['track_index']=k
  for e in c['obs']:edges.append({**e,'track':k,'original_track_id':c['entry']['original_track_id']})
 E=len(edges);T=len(selected);edge_track=np.array([e['track'] for e in edges],np.int64);edge_frame=np.array([e['frame'] for e in edges],np.int64);split=np.array([e['split'] for e in edges],np.uint8);train=split==0;quality=np.array([e['quality'] for e in edges]);nt=np.bincount(edge_track[train],minlength=T);nf=np.bincount(edge_frame[train],minlength=L)
 weight=quality/np.sqrt(np.maximum(1,nt[edge_track])*np.maximum(1,nf[edge_frame]));source=np.array([e['is_source'] for e in edges]);npz=dict(edge_track=edge_track,edge_frame=edge_frame,edge_uv=np.array([e['uv'] for e in edges],float).reshape(-1,2),edge_weight=weight,edge_split=split,edge_is_source=source,edge_quality=quality,edge_original_row=np.array([e['original_row'] for e in edges]),edge_original_query_measurement=np.array([e['original_query_measurement'] for e in edges]),track_source_frame=np.array([c['source']['frame'] for c in selected],np.int64),track_source_uv=np.array([c['source']['uv'] for c in selected],float).reshape(-1,2),q0=np.array([c['q0'] for c in selected],float).reshape(-1,3),face0=np.array([c['face0'] for c in selected],np.int64),bary0=np.array([c['bary0'] for c in selected],float).reshape(-1,3),track_original_id=np.array([c['entry']['original_track_id'] for c in selected],np.int64),track_original_source_frame=np.array([c['entry']['original_source_frame'] for c in selected],np.int64),track_original_query=np.array([c['entry']['original_query'] for c in selected],np.int64),timestamps=ts,K=K,c2w=a['c2w'],vertices=V,faces=F,R0=R,t0=t,train_weight_sum=np.array(weight[train].sum()),image_train_weight_sum=np.array(weight[train&~source].sum()))
 np.savez_compressed(out/'observations.npz',**npz)
 coverage=[];grids=[]
 for f in range(L):
  mask=edge_frame==f;xy=npz['edge_uv'][mask];coords=np.minimum([7,5],np.maximum(0,(xy/[W,H]*[8,6]).astype(int)));grid=np.zeros((6,8),int)
  for x,y in coords:grid[y,x]+=1
  coverage.append(dict(frame=f,time=float(ts[f]),raw_adopted_count=int(reliable_perframe[f]),selected_all_count=int(mask.sum()),optimization_count=int((mask&train).sum()),heldout_count=int((mask&~train).sum()),source_count=int((mask&source).sum()),occupied_full_image_grid_cells=int((grid>0).sum()),near_collinearity_small_to_large_cov_eigenvalue=(float(np.linalg.eigvalsh(np.cov(xy.T)).min()/max(np.linalg.eigvalsh(np.cov(xy.T)).max(),1e-12)) if len(xy)>=3 else None)));grids.append(grid)
 np.savez_compressed(out/'image_space_histograms.npz',grid_counts=np.array(grids),grid_shape=np.array([6,8]),image_shape=np.array([H,W]))
 c_all=components(L,T,edges);c_train=components(L,T,edges,split);rejection_counts=collections.Counter(e['rejection'] for e in ledger if e['rejection']);noedge=np.flatnonzero(nf==0).tolist();projerr=[];baryerr=[]
 for c in selected:
  f=c['source']['frame'];xyz=R[f]@c['q0']+t[f];p=K@xyz;projerr.append(float(np.linalg.norm(p[:2]/p[2]-c['source']['uv'])));baryerr.append(float(np.linalg.norm(c['bary0']@V[F[c['face0']]]-c['q0'])))
 summary={'dev':dev,'frozen_utc':utc(),'status':'completed' if T else 'input_bottleneck_no_candidates','old_initialization':str(oldinit),'old_initialization_sha256':sha(oldinit),'manifest':str(manifest),'segmentation':str(seg),'raw_queries':len(ledger),'raw_records_including_rejected_and_query_definition':raw_records,'raw_adopted_nonself_records':raw_reliable,'queries_with_at_least6_adopted':sum(e['reliable_nonself_records']>=6 for e in ledger if 'reliable_nonself_records' in e),'initializable_candidates':len(candidates),'selected_tracks':T,'total_edges':E,'optimization_edges':int(train.sum()),'heldout_edges':int((~train).sum()),'source_edges':int(source.sum()),'image_optimization_edges':int((train&~source).sum()),'train_weight_sum':float(weight[train].sum()),'image_train_weight_sum':float(weight[train&~source].sum()),'heldout_non_source_fraction':float((~train).sum()/max(1,(~source).sum())),'rejection_counts':dict(rejection_counts),'raw_rejection_bit_counts':dict(allreasons),'duplicate_count':len(duplicates),'reliable_count_histogram':hist,'all_observation_graph_components':c_all,'optimization_observation_graph_components':c_train,'number_nontrivial_optimization_components':sum(bool(c['tracks']) for c in c_train),'no_optimization_observation_frame_gaps':time_runs(noedge,ts),'gap_interpretation':'These frames lack this new observation graph and are linked to it only by pose temporal terms; original silhouette/depth may still observe them. Not claimed fully unobserved.','source_reprojection_max_px':max(projerr) if projerr else None,'q0_barycentric_reconstruction_max_m':max(baryerr) if baryerr else None,'source_edge_always_optimization':bool(np.all(split[source]==0)),'canonical_material_identity':'candidate only; no cross-tracklet identity merging','input_only':True,'reference_accessed':False,'holdout_boundary':'Heldout edge UV/descriptor never used in q initialization or source descriptor aggregation; source was chosen by its own input quality before chronological split. Holdout remains same video and is not an independent test set.','files':{'observations':str(out/'observations.npz'),'observations_sha256':sha(out/'observations.npz')},'input_hashes':{str(p):sha(p) for p in [oldinit,manifest,seg,OLD/'protocol/tracklet_rules_v1.json',OLD/'pose'/dev/'keyframes.json',*files]},'source_rgb_identities':[{ 'frame':int(f),'path':m['frame_paths'][int(f)],'manifest_sha256':m['frame_sha256'][int(f)]} for f in sorted(set(npz['track_source_frame'].tolist()))]}
 save(out/'summary.json',summary);save(out/'track_ledger.json',ledger);save(out/'edges.json',edges);save(out/'frame_coverage.json',coverage);save(out/'duplicates.json',duplicates);return summary

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dev',choices=['dev1','dev2','both'],default='both');args=parser.parse_args();config={'frozen_utc':utc(),'seed':12345,'input_only':True,'source_code_sha256':sha(__file__),'max_tracks_per_dev':128,'eligible':'at least6 original adopted nonself records; >=3 distinct frame times; add original measured query source but never count it as new temporal evidence','source_rule':'Pick maximum texture/(texture+10) * boundary/(boundary+3) over original source + adopted measurements; tie earliest frame. No pose/reference/hit criterion influences source choice.','q_init':'Exactly once at selected source, frontmost positive ray-triangle hit under own old R_camera/t_camera_m. No hit rejects track; no fallback source or free depth.','sampling':'Five equal full-sequence physical-time bins ×4x4 normalized source object-mask bbox cells; round-robin occupied cells sorted(time,y,x); within-cell descending adopted_count then local_source_quality then original(source_frame,query); selected<=128. No cross-source identity merging.','holdout':'Chronological non-source edges; n_hold=max(1,round-half-up(0.2*n_non_source)) capped to keep2targets; hold indices floor((j+0.5)*n_non_source/n_hold). Source train; no heldout q initialization/descriptors/loss.','quality':'Local texture/(texture+10)*boundary/(boundary+3); adopted edges also multiply1/(1+(cycle_error_px/2)^2). Original query has no fabricated cycle test and uses local quality. All values frozen.','edge_weight':'quality/sqrt(n_train_edges_for_track * n_train_edges_for_frame); train counts frozen from optimization split only; heldout count denominator uses same count tables with minimum1. Source included in common2D; image term excludes source.','normalizers':'W=sum(edge_weight for split==0); W_I=sum(edge_weight for split==0 and not edge_is_source); weights never gate away predicted difficulties.','sigma_u_px':3.0,'identity':'Original tracklet query identity only, candidate material correspondence never asserted true; duplicate original source UV/query and duplicate(k,t) rejected/deduplicated','schema':{'edge_split':'0 optimization,1 heldout','q0':'canonical metric surface point Kx3','face0':'original triangle index','bary0':'weights for original vertices[faces[face0]], columns sum1','R0_t0':'old camera0 object pose; do not multiply c2w again for image projection'},'all_arms_share_exact_file':True,'GPU_used':False}
 save(NEW/'protocol/observations_rules.json',config);start=time.perf_counter();summary={dev:build(dev) for dev in ['dev1','dev2'] if args.dev in [dev,'both']};save(NEW/'protocol/observations_complete.json',{'status':'completed','utc':utc(),'seconds':time.perf_counter()-start,'code_sha256':sha(__file__),'rules_sha256':sha(NEW/'protocol/observations_rules.json'),'devs':summary});print(json.dumps({d:{k:x[k] for k in ['selected_tracks','total_edges','optimization_edges','heldout_edges','initializable_candidates','rejection_counts']} for d,x in summary.items()},indent=2))
if __name__=='__main__':main()
