"""V5 local paths; all previous experiments are read-only dependencies."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[4]
RUN=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/baseline_protocol_calibration_20260927/run01'
V4=ROOT/'experiments/foreground_stage_calibration_20260927/run01'
sys.path.insert(1,str(V4/'code'));sys.path.insert(2,str(OLD/'code'))
from adapter_4dgs import UPSTREAM,OFFICIAL_COMMIT,official_config,sha
def read(p):return json.loads(Path(p).read_text())
def save_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def identity(p):
    p=Path(p).absolute();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def quantiles(t):
    import numpy as np
    a=t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)
    a=a.reshape(-1);a=a[np.isfinite(a)]
    return dict(zip(['min','p05','p50','p95','p99','max'],map(float,np.quantile(a,[0,.05,.5,.95,.99,1])))) if len(a) else None
