"""Private V9 feedback bundle; explicit exclusions, content-addressed members."""
from common import *
import zipfile,argparse


def members():
    files={}
    def add(path,name=None):
        path=Path(path)
        if path.is_file():files[name or path.relative_to(RUN).as_posix()]=path
    for ext in ('*.json','*.csv','*.md'):
        for p in RUN.glob(ext):
            if p.name not in ['feedback_verification.json','feedback_identity.json']:add(p)
    for base in ('protocol','scenes','diagnostics','logs'):
        for p in (RUN/base).rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(RUN);parts=rel.parts
            if '__pycache__' in parts:continue
            if p.suffix in ['.json','.jsonl','.csv','.md','.log','.txt']:
                add(p)
            elif p.suffix in ['.jpg','.png'] and 'figures' in parts:
                add(p)
            elif p.suffix=='.npz' and 'feedback_arrays' in parts:
                add(p)
    for p in (RUN/'code').glob('*.py'):add(p)
    for p in (RUN/'configs').glob('*.json'):add(p)
    for name in ['static_dynamic_gaussians.py','local_spacetime_input.py','projected_motion.py']:
        p=ROOT/'hoi_modules'/name;add(p,'source/hoi_modules/'+name)
    dependencies={
        OLD:['adapter_4dgs.py','train_official.py','evaluate_frozen.py','build_report_docx.py'],
        V4:['build_report_docx.py'],V5:['restore_state.py','evaluate_and_report.py']}
    dependencies[ROOT/'experiments/aux_ref_object_reconstruction_20260924/run01']=['evaluate_aux.py']
    for base,names in dependencies.items():
        for name in names:
            p=base/'code'/name;add(p,'source/'+p.relative_to(ROOT).as_posix())
    # Input metadata has the original identity/path; image and point data stay local.
    for scene,c in config()['scenes'].items():
        for name in ['manifest.json','evaluation_manifest.json']:
            p=ROOT/c['input_dir']/name;add(p,'inputs/'+scene+'/'+name)
        p=ROOT/c['historical_dir']/'protocol/fixed_examples.json';add(p,'inputs/'+scene+'/fixed_examples.json')
    add(RUN/'output/V9_static_local_results.docx','V9_static_local_results.docx')
    return files


def run():
    assert read(RUN/'protocol/document_final_QA.json')['status']=='passed'
    assert read(RUN/'protocol/research_decision_review.json')['status']=='completed'
    files=members();index={name:dict(sha256=sha(path),bytes=path.stat().st_size) for name,path in sorted(files.items())}
    output=RUN/'output/V9_feedback.zip';tmp=output.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        z.writestr('feedback_index.json',json.dumps(dict(members=index,excluded=['Full models','Original RGB and masks','Large point and float-render caches','Third-party source and environments'],
            reproduction_scope='Auditable evidence and code bundle, not a self-contained retraining dataset'),ensure_ascii=False,indent=2))
    os.replace(tmp,output);save_json(RUN/'feedback_identity.json',identity(output));verify(output)


def verify(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        names=z.namelist();assert len(names)==len(set(names))
        idx=json.loads(z.read('feedback_index.json'))['members']
        assert set(names)==set(idx)|{'feedback_index.json'}
        for name,record in idx.items():
            data=z.read(name);assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
    save_json(RUN/'feedback_verification.json',dict(status='passed',members=len(names),content_hashes_checked=len(idx),zip=identity(path),CRC_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify');a=p.parse_args()
    verify(a.verify) if a.verify else run()
