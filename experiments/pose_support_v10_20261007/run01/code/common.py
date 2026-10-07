"""V10 private paths and atomic evidence; historical implementations are read-only."""
from pathlib import Path
import os, sys, json, hashlib, time, importlib.util
ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/4DGaussians'
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location('v10_original_adapter', ROOT/'experiments/baseline_protocol_calibration_20260927/run01/code/adapter_4dgs.py')
adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
CalibratedCamera, official_config, OFFICIAL_COMMIT = adapter.CalibratedCamera, adapter.official_config, adapter.OFFICIAL_COMMIT
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def identity(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def append_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',buffering=1) as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def atomic_checkpoint(path,state):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    return identity(path)
def config():return read(RUN/'configs/v10.json')
def scene_dir(scene):
    assert scene in config()['scenes'];return RUN/'scenes'/scene
def clone_cpu(value):
    import torch,numpy as np
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:clone_cpu(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(clone_cpu(v) for v in value)
    if isinstance(value,list):return [clone_cpu(v) for v in value]
    if isinstance(value,np.ndarray):return value.copy()
    return value
def exact(a,b,label='state'):
    import torch,numpy as np
    if torch.is_tensor(a):assert torch.equal(a.detach().cpu(),b.detach().cpu()),label
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),label
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),label
        for k in a:exact(a[k],b[k],label+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),label
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,label+'/'+str(i))
    else:assert a==b,label
