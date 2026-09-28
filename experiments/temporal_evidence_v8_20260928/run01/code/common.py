"""V8 isolated output paths with read-only inherited implementation dependencies."""
from pathlib import Path
import sys,json,hashlib,os
ROOT=Path(__file__).resolve().parents[4]
RUN=Path(__file__).resolve().parents[1]
V6=ROOT/'experiments/gradient_scope_calibration_20260928/run01'
V5=ROOT/'experiments/numerical_stability_calibration_20260927/run01'
V4=ROOT/'experiments/foreground_stage_calibration_20260927/run01'
OLD=ROOT/'experiments/baseline_protocol_calibration_20260927/run01'
for i,p in enumerate([V6,V5,V4,OLD],1):sys.path.insert(i,str(p/'code'))
sys.path.insert(0,str(ROOT))
from adapter_4dgs import UPSTREAM,OFFICIAL_COMMIT,official_config,sha
def read(p):return json.loads(Path(p).read_text())
def save_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def identity(p):
    p=Path(p).absolute();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def quantiles(t):
    import numpy as np
    a=t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)
    a=a.reshape(-1);a=a[np.isfinite(a)]
    return dict(zip(['min','p05','p50','p95','p99','max'],map(float,np.quantile(a,[0,.05,.5,.95,.99,1])))) if len(a) else None
def atomic_checkpoint(path,state):
    import torch
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    return identity(path)
