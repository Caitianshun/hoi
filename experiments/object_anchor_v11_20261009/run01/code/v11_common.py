"""V11 paths, atomic JSON/identity helpers and read-only access to frozen V10 code.

V11 modules use a v11_ prefix so that V10's ``common``/``state``/``render_scene``
modules can be imported unchanged from their own directory without shadowing.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
V10 = ROOT / "experiments/pose_support_v10_20261007/run01"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def jsonl(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def config():
    return read(RUN / "configs/v11.json")


def scene_dir(scene):
    assert scene in config()["scenes"], scene
    return RUN / "scenes" / scene


def v10_modules():
    """Import frozen V10 modules (common, state, render_scene) from the V10 code directory."""
    code = str(V10 / "code")
    if code not in sys.path:
        sys.path.insert(0, code)
    import state as v10_state  # noqa: E402  (V10 module, read-only)
    import render_scene as v10_render  # noqa: E402
    return v10_state, v10_render


def v10_prepare_inputs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("v10_prepare_inputs", V10 / "code/prepare_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
