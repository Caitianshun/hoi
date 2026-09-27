"""Small private feedback bundle; checkpoints and third-party snapshots excluded."""
from pathlib import Path
import json,hashlib,zipfile
RUN=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def run():
    required=['ROOT_CAUSE.md','patch.diff','patch_effects.json','first_bad_tensor.json','state_manifest.json','environment.json','pair_protocol.json','A_summary.json','metrics_per_frame.csv','paired_differences.csv','training_metrics.jsonl','density_events.jsonl','sampling_comparison.json','model_index.json','costs.json','NEXT_DECISION.md','MISSING_ASSETS.md','REPRODUCE.md']
    selected={RUN/n for n in required};assert all(p.exists() for p in selected)
    patterns=['code/*.py','configs/*.json','protocol/*.json','protocol/*.jsonl','protocol/user_guidance_source.md','minimal_repro/*.json','minimal_repro/*.md','minimal_repro/*.py','minimal_repro/*.npz','diagnostics/*.jpg','diagnostics/*.json','evaluation/summary.json','evaluation/figures/*.jpg','feedback_arrays/*.npz','feedback_arrays/*.json','runs/*/result.json','runs/*/run.json','runs/*/failure.json','runs/*/effective_config.json','runs/*/first_bad_tensor.json','runs/*/restore_verification.json','runs/*/kernel_case_analysis.json','runs/*/summary.json','runs/*/domain_choice.json','runs/B_*/sampling_order.jsonl','logs/*/attempt.json','replay_events.jsonl']
    for pattern in patterns:selected.update(RUN.glob(pattern))
    selected.update(RUN.glob('runs/*/source_adaptation.json'))
    selected.add(RUN/'run_manifest.json')
    selected.add(RUN/'figure_manifest.json')
    selected.update(RUN.glob('protocol/verification_interface_failure/*'))
    selected.update(RUN.glob('logs/independent_verification*.log'))
    selected={p for p in selected if p.is_file() and p.name not in ['package_verification.json','final_sync.json','delivery_audit.json']}
    for p in selected:assert p.suffix not in ['.pt','.pth','.so'] and 'source_snapshots' not in p.parts
    content=[dict(name=str(p.relative_to(RUN)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)]
    out=RUN/'output/V5_feedback.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for x in content:z.write(RUN/x['name'],x['name'])
        z.writestr('PACKAGE_INDEX.json',json.dumps(dict(files=content,scope='Private feedback. Includes compressed comparison panels containing GT, fixed float32 windows and one-Gaussian CUDA case. Full model checkpoints, raw dataset, large arrays, environment and third-party source snapshots excluded. DOCX separate.',self_contained_full_training_reproduction=False,minimal_case_requires_matching_CUDA_extension=True),indent=2,ensure_ascii=False))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for x in content:assert hashlib.sha256(z.read(x['name'])).hexdigest()==x['sha256']
    assert out.stat().st_size<=25_000_000,'Reduce optional display encodings, not numeric precision or required coverage'
    result=dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size,members=len(content)+1,CRC_passed=True,all_member_hashes_passed=True,DOCX_separate=True)
    (RUN/'protocol/package_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':run()
