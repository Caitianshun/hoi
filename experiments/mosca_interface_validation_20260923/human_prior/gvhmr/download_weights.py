"""Download only the three official public GVHMR inference checkpoints."""
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import gdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'models' / 'gvhmr_public'
SOURCES = [
    ('gvhmr/gvhmr_siga24_release.ckpt', '1c9iCeKFN4Kr6cMPJ9Ss6Jdc3SZFnO5NP', 163508011),
    ('hmr2/epoch=10-step=25000.ckpt', '1X5hvVqvqI9tvjUCb2oAlZxtgIKD9kvsc', 2709494041),
    ('vitpose/vitpose-h-multi-coco.pth', '1sR8xZD9wrZczdDVo6zKscNLwvarIRhP5', 2549075546),
]

def run(item):
    name, ident, expected = item
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    record = dict(name=name, path=str(target), expected_bytes=expected,
                  source='https://drive.google.com/file/d/' + ident + '/view',
                  upstream_install='https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md',
                  expected_size_source='official Drive file page metadata; not a published cryptographic checksum')
    try:
        if not target.exists() or target.stat().st_size != expected:
            gdown.download(id=ident, output=str(target), quiet=True, use_cookies=False,
                           resume=True, timeout=(15, 60), retries=2)
        record['size_bytes'] = target.stat().st_size
        assert record['size_bytes'] == expected, (record['size_bytes'], expected)
        h = hashlib.sha256()
        with target.open('rb') as f:
            for buf in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(buf)
        record.update(status='complete', sha256=h.hexdigest())
    except Exception as e:
        record.update(status='failed', error=type(e).__name__ + ': ' + str(e))
    record['seconds'] = time.perf_counter() - start
    (HERE / (name.split('/')[0] + '_download.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return record

if __name__ == '__main__':
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(run, SOURCES))
    result = dict(status='complete' if all(r['status']=='complete' for r in records) else 'incomplete',
                  seconds=time.perf_counter()-started, files=records,
                  no_gpu=True, no_private_model_download=True)
    (HERE / 'downloads.json').write_text(json.dumps(result, indent=2)+'\n')
    raise SystemExit(0 if result['status']=='complete' else 1)
