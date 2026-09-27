"""Read-only SHA verification of the bounded private feedback package."""
import argparse,zipfile,json,hashlib
def run():
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);a=p.parse_args()
    with zipfile.ZipFile(a.zip) as z:
        assert z.testzip() is None
        idx=json.loads(z.read('PACKAGE_INDEX.json'))
        for x in idx['files']:assert hashlib.sha256(z.read(x['name'])).hexdigest()==x['sha256'],x['name']
        assert len(idx['files'])==len(z.infolist())-1
    print('CRC and every indexed member SHA verified',len(idx['files']))
if __name__=='__main__':run()
