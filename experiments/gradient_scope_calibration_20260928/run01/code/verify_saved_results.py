"""Independent CPU-only checks, no optimization, rendering, or scientific writes."""
from pathlib import Path
import json,hashlib,csv,zipfile,argparse,collections
import numpy as np
RUN=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def run():
    p=argparse.ArgumentParser();p.add_argument('--zip');a=p.parse_args()
    if a.zip:
        with zipfile.ZipFile(a.zip) as z:
            assert z.testzip() is None;idx=json.loads(z.read('PACKAGE_INDEX.json'))
            for f in idx['files']:assert hashlib.sha256(z.read(f['name'])).hexdigest()==f['sha256']
        print('ZIP all member hashes passed',len(idx['files']));return
    inherited=read(RUN/'state_manifest.json')['inherited'];inputs=read(RUN/'protocol/input_hash_verification.json')['inputs']
    for asset in inherited+inputs:assert sha(asset['path'])==asset['sha256'],asset['path']
    audit=read(RUN/'camera_time_audit.json');assert len(audit['rows'])==284
    for r in audit['rows']:assert r['timestamp']==(int(r['frame_id'])-1)/282
    gate=read(RUN/'protocol/formal_gate.json');assert not gate['all_conditions_passed'] and not gate['C_started']
    route=read(RUN/'routing_equivalence.json');failed=[r for r in route['rows'] if not r['passed']]
    assert failed and all(r['policy']=='uniform_all' for r in failed)
    for r in route['rows']:
        if 'difference' not in r:continue
        x=r['difference'];ok=x['finite'] and x['max_abs']<=r['atol'] and (x['relative_L2']<=r['rtol'] if x['reference_norm']>1e-12 else x['max_abs']<=r['atol'])
        assert ok==r['passed']
    manifest=read(RUN/'contribution_manifest.json');arrays=0;max_neff_difference=0
    for arm,m in manifest['models'].items():
        ts=[]
        for r in m['rows']:
            asset=r['arrays'];assert sha(asset['path'])==asset['sha256'];z=np.load(asset['path']);n=int(z['total_rows']);assert n==m['topology_rows']
            ids=z['row_ids'];assert len(ids)==len(np.unique(ids)) and ids.min()>=0 and ids.max()<n
            assert z['fg'].dtype==z['bg'].dtype==np.float32 and np.isfinite(z['fg']).all() and np.isfinite(z['bg']).all()
            t=np.zeros(n,np.float32);t[ids]=z['fg']+z['bg'];ts.append(t);arrays+=1
        T=np.stack(ts);d=T.sum(0,dtype=np.float64);valid=d>0;P=np.zeros_like(T);P[:,valid]=T[:,valid]/d[valid]
        neff=np.zeros(len(d),np.float32);neff[valid]=1/(P[:,valid]**2).sum(0)
        asset=m['summary_arrays'];assert sha(asset['path'])==asset['sha256'];z=np.load(asset['path']);assert np.array_equal(z['supported'],valid)
        delta=float(np.max(np.abs(neff-z['N_eff'])));max_neff_difference=max(max_neff_difference,delta);assert delta==0
    with (RUN/'metrics_per_frame.csv').open() as f:metrics=list(csv.DictReader(f))
    keys={(r['run'],r['split'],r['frame_id'],r['region']) for r in metrics};assert len(metrics)==len(keys)==3*284*3
    C=[r for r in metrics if r['run']=='C_route'];assert len(C)==284*3
    assert all(not r[k] for r in C for k in ['psnr_db','ssim','lpips_spatial_mean','raw_psnr_db','raw_render_path'])
    with (RUN/'paired_differences.csv').open() as f:pairs=list(csv.DictReader(f))
    assert len(pairs)==2*284*3 and all(not r[k] for r in pairs for k in ['psnr_db','ssim','lpips_spatial_mean'])
    calls=collections.Counter((r['mode'],r['kind']) for r in map(json.loads,(RUN/'protocol/kernel_calls.jsonl').read_text().splitlines()));cost=read(RUN/'costs.json')
    assert calls['diagnostic','render']==cost['diagnostic_renders']<=256 and calls['diagnostic','backward']==cost['diagnostic_backwards']<=160
    assert not calls['optimization','render'] and cost['optimizer_steps']==0
    assert abs(sum(x['wall_seconds'] for x in cost['jobs'])-cost['GPU_task_seconds'])<1e-9
    print(json.dumps(dict(status='passed',inherited_hashes=len(inherited),source_hashes=len(inputs),contribution_arrays=arrays,max_N_eff_difference=max_neff_difference,metric_rows=len(metrics),C_NA_rows=len(C),paired_NA_rows=len(pairs),failed_gate_preserved=True,optimizer_steps=0)))
if __name__=='__main__':run()
