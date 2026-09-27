"""Small private feedback with original-precision samples and complete figures."""
from pathlib import Path
import json,hashlib,zipfile,torch,numpy as np,shutil,subprocess
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def identity(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def run():
    qa=read(RUN/'protocol/document_final_QA.json');assert qa['all_pages_visually_checked']
    assert qa['document_sha256']==sha(qa['document'])
    dst=RUN/'code/attribute_gradient_router.py.snapshot';shutil.copyfile(ROOT/'hoi_modules/attribute_gradient_router.py',dst)
    sample=RUN/'feedback_arrays/gradient_samples.npz'
    if not sample.exists():
        arrays={};meta=[]
        for index,asset in enumerate(read(RUN/'diagnostics/acceptance_probe.json')['gradient_assets']):
            assert sha(asset['path'])==asset['sha256'];data=torch.load(asset['path'],map_location='cpu',weights_only=False)
            for name,t in data.items():
                if t is None:meta.append(dict(index=index,name=name,state='None'));continue
                flat=t.reshape(-1);ids=np.unique(np.linspace(0,len(flat)-1,min(512,len(flat)),dtype=np.int64))
                key=f'probe{index}__'+name.replace('.','_').replace('/','_');arrays[key+'_indices']=ids;arrays[key+'_values']=flat.numpy()[ids]
                meta.append(dict(index=index,name=name,key=key,shape=list(t.shape),dtype=str(t.dtype),source=asset))
        np.savez_compressed(sample,**arrays);save(RUN/'feedback_arrays/gradient_samples_manifest.json',dict(samples=meta,selection='Deterministic uniform flattened indices, no precision change',full_gradients_remain_on_host=True))
    # Include the whole authorized source change, including resume and accounting.
    # Results, attachments and historical research notes are deliberately omitted.
    base='e69c435adfe600fdf31f5012041f8c5876fbc6e0'
    paths=[ROOT/'hoi_modules/attribute_gradient_router.py',ROOT/'scripts/repo_sync.py',RUN/'configs/v7.json',*sorted((RUN/'code').glob('*.py'))]
    relative=[str(p.relative_to(ROOT)) for p in paths]
    tracked=set(subprocess.check_output(['git','ls-files','--',*relative],cwd=ROOT,text=True).splitlines())
    assert tracked==set(relative),'Synchronize authorized source files before packaging'
    patch=subprocess.check_output(['git','diff','--no-ext-diff','--binary',base,'--',*relative],cwd=ROOT)
    assert patch and b'attribute_gradient_router.py' in patch and b'train_routed_fine.py' in patch
    (RUN/'patch.diff').write_bytes(patch)
    save(RUN/'protocol/patch_identity.json',dict(base_commit=base,end_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_paths=relative,**identity(RUN/'patch.diff')))
    required=['run.json','route_protocol.json','module_acceptance.json','gradient_audit.csv','metrics_per_frame.csv','paired_differences.csv','evaluation_summary.json','quality_decision.json','training_metrics.jsonl','sampling_order.jsonl','density_events.jsonl','bound_events_compact.csv','costs.json','sampling_comparison.json','state_manifest.json','source_manifest.json','environment.json','CAUSE_ASSESSMENT.md','NEXT_DECISION.md','REPRODUCE.md','MISSING_ASSETS.md','figure_manifest.json','patch.diff']
    chosen={RUN/n for n in required}
    for pattern in ['configs/*.json','code/*.py','code/*.snapshot','protocol/*.json','protocol/*steps.jsonl','protocol/optimizer_calls.jsonl','protocol/gpu_jobs.jsonl','protocol/user_guidance_source.md','diagnostics/*.json','diagnostics/regression/*.json','diagnostics/regression_resume/*.json','runs/*/*.json','logs/*/attempt.json','feedback_arrays/*.npz','feedback_arrays/*.json','evaluation/figures/*.jpg']:
        chosen.update(RUN.glob(pattern))
    chosen={p for p in chosen if p.is_file() and p.name not in ['package_verification.json','final_sync.json']}
    assert all('source_snapshots' not in p.parts and p.suffix not in ['.pt','.pth','.so','.docx'] for p in chosen)
    rows=[dict(name=str(p.relative_to(RUN)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(chosen)]
    output=RUN/'output/V7_feedback.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for x in rows:z.write(RUN/x['name'],x['name'])
        z.writestr('PACKAGE_INDEX.json',json.dumps(dict(files=rows,scope='Private feedback: complete fixed GT/B_U/B_F/M1 comparisons, numeric vectors, lossless float32 windows and gradient samples. No models or full dataset.',DOCX_separate=True,self_contained_training_reproduction=False),indent=2))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for x in rows:assert hashlib.sha256(z.read(x['name'])).hexdigest()==x['sha256']
    result=dict(**identity(output),members=len(rows)+1,CRC_and_member_SHA_passed=True,target_bytes=25_000_000,under_target=output.stat().st_size<=25_000_000)
    save(RUN/'protocol/package_verification.json',result);print(json.dumps(result))
if __name__=='__main__':run()
