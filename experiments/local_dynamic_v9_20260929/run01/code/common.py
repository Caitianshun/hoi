"""V9 paths and atomic private evidence helpers; inherited code is read-only."""
from pathlib import Path
import os, sys, json, hashlib, time
ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
V8 = ROOT / 'experiments/temporal_evidence_v8_20260928/run01'
V6 = ROOT / 'experiments/gradient_scope_calibration_20260928/run01'
V5 = ROOT / 'experiments/numerical_stability_calibration_20260927/run01'
V4 = ROOT / 'experiments/foreground_stage_calibration_20260927/run01'
OLD = ROOT / 'experiments/baseline_protocol_calibration_20260927/run01'
for i, p in enumerate((V8, V6, V5, V4, OLD), 1):
    sys.path.insert(i, str(p / 'code'))
sys.path.insert(0, str(ROOT))
from adapter_4dgs import UPSTREAM, OFFICIAL_COMMIT, official_config, sha


def read(p):
    return json.loads(Path(p).read_text())


def save_json(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def identity(p):
    p = Path(p).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_checkpoint(p, state):
    import torch
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    fd = os.open(p.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)
    return identity(p)


def append_json(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', buffering=1) as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def config():
    return read(RUN / 'configs/v9.json')


def scene_dir(scene):
    assert scene in config()['scenes']
    return RUN / 'scenes' / scene


def quantiles(a):
    import numpy as np
    a = a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
    a = a.reshape(-1); a = a[np.isfinite(a)]
    return dict(zip(('min', 'p01', 'p05', 'p50', 'p95', 'p99', 'max'),
                    map(float, np.quantile(a, [0, .01, .05, .5, .95, .99, 1])))) if len(a) else None
