"""Read-only CPU audit of local source identities and B1 shared initialization."""
from pathlib import Path
import json,sys
import numpy as np
from build_surface_observations import E,A,sha_file,save
B=E.parents[2]/'experiments/fixed_motion_reconstruction_20260926/run01'
checks={}
for d in ['dev1','dev2']:
 p=E/'support'/d;a=dict(np.load(p/'observations.npz'));q=dict(np.load(p/'queries.npz'));x=dict(np.load(p/'B0_object.npz'));ids=q['candidate_ids'];ii,tt=np.where(ids>=0);hit=ids[ii,tt];dist=np.linalg.norm(a['xyz'][hit]-x['centres_canonical_m'][ii],axis=1)
 assert a['positive'][hit].all();assert np.array_equal(a['time'][hit],tt);assert np.array_equal(a['face'][hit],q['face'][ii]);assert (dist<=.005000001).all();assert np.allclose(dist,q['distance'][ii,tt],atol=1e-8)
 assert np.array_equal((ids>=0).sum(1),q['support_count']);assert len(np.unique(x['stable_id']))==6000
 old=json.loads((A/'runs'/f'{d}_Ref/initial_identity.json').read_text());new=json.loads((B/'runs'/d/'initial_identity.json').read_text());final=json.loads((B/'runs'/d/'final_identity.json').read_text());assert old['object']==new['object'] and old['frozen']==new['frozen']==final['frozen']
 b1=[json.loads(line) for line in (B/'runs'/d/'steps.jsonl').read_text().splitlines()];sched=np.load(A/'frozen_aux'/f'{d}_frame_schedule.npy');assert len(b1)==8000 and np.array_equal([r['frame'] for r in b1],sched);assert all(r['nonzero'] for r in b1)
 checks[d]=dict(all_candidates_same_face=True,unique_native_source_times=True,positive_interiors_only=True,max_distance_m=float(dist.max()),B1_same_untrained_initialization=True,B1_frozen_HS_motion=True,B1_original_schedule=True,B1_effective_steps=8000)
save(E/'protocol/input_integrity.json',{'status':'passed','checks':checks});print(json.dumps(checks))
