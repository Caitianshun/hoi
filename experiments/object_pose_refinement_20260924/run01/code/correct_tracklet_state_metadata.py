#!/usr/bin/env python3
"""Metadata-only CPU repair. No UV, adopted gates, scores or model inference change."""
from pathlib import Path
import json,hashlib,shutil,datetime
import numpy as np
E=Path('/home/cai_tianshun/Project/HOI/experiments/object_pose_refinement_20260924/run01')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
for dev in ['dev1','dev2']:
    out=E/'pose'/dev;raw=out/'raw_state_metadata_v1';raw.mkdir(exist_ok=False)
    for p in sorted(out.glob('tracklet*.npz')):
        shutil.copy2(p,raw/p.name);d=dict(np.load(p));reason=d['reason_bits'];states=np.full(reason.shape,4,np.uint8);states[(~d['visibility'])&np.isfinite(d['target_uv']).all(-1)]=2;states[d['adopted']]=0;states[(reason!=0)&((reason&~(32|64))==0)]=1;states[d['source_frame']==d['target_frame']]=1
        changed=int((states!=d['observation_state']).sum());d['observation_state']=states
        temp=p.with_suffix('.tmp.npz');np.savez_compressed(temp,**d);temp.replace(p)
        old=dict(np.load(raw/p.name));assert all(np.array_equal(v,old[k],equal_nan=True) for k,v in d.items() if k!='observation_state');assert np.array_equal(d['adopted'],reason==0);assert np.all(d['observation_state'][d['adopted']]==0);assert not d['canonical_confirmed'].any()
        checks.append(dict(path=str(p),before_sha256=sha(raw/p.name),after_sha256=sha(p),changed_state_labels=changed,all_other_arrays_unchanged=True))
    p=out/'tracklets_summary.json';s=json.loads(p.read_text());shutil.copy2(p,raw/p.name);s['tracklets_sha256']=sha(out/'tracklets.npz');s['metadata_repair']='Only reliable observation_state=0 overwritten by uncertain label was corrected; adopted bool and all observations unchanged.';p.write_text(json.dumps(s,indent=2)+'\n')
r=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),explanation='Boolean operator accidentally relabelled zero-reason observations as uncertain. Adopted boolean was correct. Metadata-only fix excludes reasons==0 from uncertain assignment; raw arrays preserved.',checks=checks);(E/'logs/tracklet_state_metadata_repair.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS',len(checks),'files; all observation arrays/adoption unchanged')
