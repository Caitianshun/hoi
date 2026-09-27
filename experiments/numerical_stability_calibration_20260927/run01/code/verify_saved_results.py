"""Read-only local/package checks. No rendering, optimizer or output writes."""
from pathlib import Path
import argparse,hashlib,json,csv,zipfile
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def run():
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path);a=p.parse_args()
    if a.zip:
        with zipfile.ZipFile(a.zip) as z:
            assert z.testzip() is None;index=json.loads(z.read('PACKAGE_INDEX.json'))
            for x in index['files']:assert hashlib.sha256(z.read(x['name'])).hexdigest()==x['sha256'],x['name']
        print('ZIP CRC and all member hashes passed',len(index['files']));return
    root=Path(__file__).resolve().parents[1];gate=read(root/'protocol/formal_gate.json');checked=0
    assets=gate['frozen_source_files']+gate['frozen_assets']
    if (root/'protocol/finals.json').exists():assets+=read(root/'protocol/finals.json')['assets']
    for x in assets:assert sha(x['path'])==x['sha256'],x['path'];checked+=1
    if (root/'metrics_per_frame.csv').exists():
        with (root/'metrics_per_frame.csv').open() as f:rows=list(csv.DictReader(f))
        idx={(r['run'],r['split'],r['frame_id'],r['region']):r for r in rows};assert len(idx)==len(rows)==3*284*3
        with (root/'paired_differences.csv').open() as f:
            for r in csv.DictReader(f):
                left,right=r['comparison'].split('-');key=r['split'],r['frame_id'],r['region'];x=idx[(left,)+key];y=idx[(right,)+key]
                for name in ['psnr_db','ssim','lpips_spatial_mean']:
                    if not x[name] or not y[name]:assert not r[name]
                    else:assert abs(float(r[name])-(float(x[name])-float(y[name])))<1e-10
    print('Frozen identities and available pair arithmetic passed',checked)
if __name__=='__main__':run()
