from pathlib import Path
import json, hashlib, time
import torch
ROOT = Path('/home/cai_tianshun/Project/HOI')
E = ROOT/'experiments/gvhmr_validation_20260923'
torch.set_num_threads(4)
expected = {'gvhmr_siga24_release.ckpt':163508011, 'epoch=10-step=25000.ckpt':2709494041, 'vitpose-h-multi-coco.pth':2549075546}
start=time.perf_counter(); rows=[]
for name,size in expected.items():
    p=ROOT/'weight'/name
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(1<<20): h.update(chunk)
    digest=h.hexdigest()
    assert p.stat().st_size==size,(name,p.stat().st_size,size)
    data=torch.load(p,map_location='cpu',weights_only=False)
    state=data['state_dict']; tensors={k:v for k,v in state.items() if isinstance(v,torch.Tensor)}
    bad=[k for k,v in tensors.items() if (v.is_floating_point() or v.is_complex()) and not torch.isfinite(v).all()]
    assert not bad,bad
    rows.append(dict(path=str(p),bytes=size,sha256=digest,official_size_match=True,top_keys=list(data),state_tensor_count=len(tensors),tensor_elements=sum(v.numel() for v in tensors.values()),nonfinite_keys=bad,example_shapes={k:list(v.shape) for k,v in list(tensors.items())[:8]}))
    del data,state,tensors
result=dict(status='passed',source='User-provided files; size matches official Drive metadata, SHA256 computed locally (not published author hash)',files=rows,seconds=time.perf_counter()-start)
(E/'deployment/weight_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
