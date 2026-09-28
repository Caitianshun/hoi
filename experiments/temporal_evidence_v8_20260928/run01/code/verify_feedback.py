"""Independent stdlib ZIP integrity check, no training or metric recomputation."""
from pathlib import Path
import zipfile,json,hashlib
RUN=Path(__file__).resolve().parents[1]
def main():
    path=RUN/'output/V8_feedback.zip'
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None;index=json.loads(z.read('PACKAGE_INDEX.json'));names=set(z.namelist());assert len(names)==len(z.namelist())
        assert {x['name'] for x in index['files']}==names-{'PACKAGE_INDEX.json'}
        for x in index['files']:
            data=z.read(x['name']);assert len(data)==x['bytes'];assert hashlib.sha256(data).hexdigest()==x['sha256']
    print(json.dumps(dict(status='pass',members=len(names),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
if __name__=='__main__':main()
