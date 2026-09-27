"""V4 paths and read-only access to the locked V3 interfaces."""
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parents[4]
RUN=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/baseline_protocol_calibration_20260927/run01'
sys.path.insert(0,str(OLD/'code'))
from adapter_4dgs import UPSTREAM,OFFICIAL_COMMIT,sha,save_json,official_config
def read(p):return json.loads(Path(p).read_text())
def identity(p):
    p=Path(p).absolute();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def quantiles(t):
    import numpy as np
    a=t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)
    a=a.reshape(-1);a=a[np.isfinite(a)]
    return dict(zip(['min','p05','p50','p95','p99','max'],map(float,np.quantile(a,[0,.05,.5,.95,.99,1])))) if len(a) else None
