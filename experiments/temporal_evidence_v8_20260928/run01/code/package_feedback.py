"""Private feedback bundle with complete fixed figures and original precision."""
from pathlib import Path
import json,hashlib,zipfile,subprocess
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def identity(p):p=Path(p).absolute();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def run():
    qa=read(RUN/'protocol/document_final_QA.json');assert qa['all_pages_visually_checked'] and qa['document_sha256']==sha(qa['document'])
    assert read(RUN/'quality_decision.json')['status']=='completed' and read(RUN/'protocol/independent_verification.json')['status']=='pass'
    base=read(RUN/'protocol/initial_source_sync.json').get('base_commit','83103f6cff9483c4242fb2d0f8bc9457a2292522')
    paths=[ROOT/'hoi_modules/projected_motion.py',ROOT/'hoi_modules/temporal_evidence.py',ROOT/'scripts/repo_sync.py',RUN/'configs/v8.json',*sorted((RUN/'code').glob('*.py'))];relative=[str(p.relative_to(ROOT)) for p in paths]
    tracked=set(subprocess.check_output(['git','ls-files','--',*relative],cwd=ROOT,text=True).splitlines());assert tracked==set(relative)
    patch=subprocess.check_output(['git','diff','--no-ext-diff','--binary',base,'--',*relative],cwd=ROOT);assert b'temporal_evidence.py' in patch and b'train_temporal.py' in patch;(RUN/'patch.diff').write_bytes(patch)
    save(RUN/'protocol/patch_identity.json',dict(base_commit=base,end_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_paths=relative,**identity(RUN/'patch.diff')))
    save(RUN/'current_source_manifest.json',dict(assets=[identity(p) for p in paths],frozen_start_manifest=identity(RUN/'source_manifest.json')))
    required=['run.json','protocol.json','module_acceptance.json','track_cache_manifest.json','lambda_calibration.json','metrics_per_frame.csv','paired_differences.csv','evaluation_summary.json','quality_decision.json','costs.json','sampling_comparison.json','temporal_summary.json','state_manifest.json','source_manifest.json','current_source_manifest.json','environment.json','CAUSE_ASSESSMENT.md','NEXT_DECISION.md','REPRODUCE.md','MISSING_ASSETS.md','figure_manifest.json','patch.diff']
    chosen={n:RUN/n for n in required};assert all(p.is_file() for p in chosen.values())
    for pattern in ['configs/*.json','code/*.py','protocol/*.json','protocol/*steps.jsonl','protocol/optimizer_calls.jsonl','protocol/extra_backward.jsonl','protocol/gpu_jobs.jsonl','protocol/pair_order.pt','protocol/user_guidance_source.md','diagnostics/*.json','diagnostics/regression/*.json','diagnostics/resume/*.json','runs/*/*.json','runs/*/temporal_stats.jsonl','runs/*/sampling_order.jsonl','runs/*/training_metrics.jsonl','runs/*/density_events.jsonl','runs/*/checkpoint_index.jsonl','logs/*/attempt.json','feedback_arrays/*.npz','feedback_arrays/*.json','evaluation/figures/*.jpg','evaluation/temporal_explanations/*.json','evaluation/temporal_explanations/*.jpg','evaluation/temporal_explanations/*.npz']:
        for p in RUN.glob(pattern):
            if p.is_file():chosen[str(p.relative_to(RUN))]=p
    chosen['module_sources/projected_motion.py']=ROOT/'hoi_modules/projected_motion.py';chosen['module_sources/temporal_evidence.py']=ROOT/'hoi_modules/temporal_evidence.py'
    for n in ['package_verification.json','final_sync.json']:chosen.pop('protocol/'+n,None)
    for name,p in chosen.items():assert 'source_snapshots' not in p.parts and (p.suffix not in ['.pt','.pth','.so','.docx'] or p.name=='pair_order.pt')
    if (RUN/'independent_confirmation.json').exists():
        chosen['independent_confirmation.json']=RUN/'independent_confirmation.json';ind=read(RUN/'independent_confirmation.json')
        if ind.get('status')=='completed':
            # Independent run prepares an explicit private feedback include list.
            assert 'feedback_files' in ind
            for a in ind['feedback_files']:
                p=Path(a['path']);assert sha(p)==a['sha256'];chosen['Tennis/'+a['name']]=p
    rows=[dict(name=n,bytes=p.stat().st_size,sha256=sha(p)) for n,p in sorted(chosen.items())]
    path=RUN/'output/V8_feedback.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,p in sorted(chosen.items()):z.write(p,n)
        z.writestr('PACKAGE_INDEX.json',json.dumps(dict(files=rows,DOCX_separate=True,full_training_models_and_raw_on_host=True,no_precision_reduction=True,scope='Frozen protocol and numeric vectors, all fixed comparison images, float32 windows, compact temporal evidence and full per-call statistics. Full cache/checkpoints/data indexed on training host.'),indent=2))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for a in rows:assert hashlib.sha256(z.read(a['name'])).hexdigest()==a['sha256']
    result=dict(**identity(path),members=len(rows)+1,CRC_and_member_SHA_passed=True,suggested_size_range_bytes=[25_000_000,40_000_000],within_suggested_range=25_000_000<=path.stat().st_size<=40_000_000)
    save(RUN/'protocol/package_verification.json',result);print(json.dumps(result))
if __name__=='__main__':run()
