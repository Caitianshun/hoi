"""Small private feedback bundle with an exact member identity manifest."""
from pathlib import Path
import json,hashlib,zipfile
RUN=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while data:=f.read(1<<20):h.update(data)
    return h.hexdigest()
def run():
    required=['run_manifest.json','metrics_per_frame.csv','paired_differences.csv','training_metrics.jsonl','density_events.jsonl','state_probes.jsonl','initialization_support.json','appearance_diagnostics.csv','figure_manifest.json','costs.json','MISSING_ASSETS.md','NEXT_DECISION.md','REPRODUCE.md','report_content.json','PROTOCOL.md','model_index.json']
    selected={RUN/f for f in required};assert all(p.exists() for p in selected)
    for pattern in ['code/*.py','configs/*.json','protocol/*.json','protocol/*.jsonl','protocol/user_guidance_source.md','evaluation/summary.json','evaluation/figures/*.jpg','diagnostics/appearance/summary.json','diagnostics/appearance/manifest.json','diagnostics/appearance/figures/*.jpg','diagnostics/input_support/*.jpg','diagnostics/input_support/*.png','diagnostics/state_probes/manifest.json','diagnostics/state_probes/*/*/*.jpg','diagnostics/*summary.jpg','feedback_arrays/*.json','feedback_arrays/*.npz','runs/*/effective_config.json','runs/*/failure.json','runs/*/branch_validation.json','runs/*/fine_transition.json','runs/*/upstream_adaptation.json','runs/*/run.json','runs/*/verified_restore.json','runs/*/resume_check.json','protocol/failed_attempts/*/*.json','protocol/failed_attempts/*/*.jsonl','logs/*/attempt.json']:
        selected.update(RUN.glob(pattern))
    selected.update(RUN.glob('protocol/failed_attempts/*/observer_source.py'))
    selected={p for p in selected if p.is_file() and p.name!='package_verification.json'}
    content=[]
    for p in sorted(selected):
        assert p.suffix not in ['.pt','.pth','.so'];content.append(dict(name=str(p.relative_to(RUN)),sha256=sha(p),bytes=p.stat().st_size))
    package=RUN/'output/V4_feedback.zip';package.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for item in content:z.write(RUN/item['name'],item['name'])
        z.writestr('PACKAGE_INDEX.json',json.dumps(dict(files=content,scope='Private feedback; original raw datasets, large full-frame arrays, model checkpoints, environments and third-party code excluded. DOCX delivered separately. Tiny float32 windows and compressed GT comparison panels included.'),indent=2))
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for item in content:assert hashlib.sha256(z.read(item['name'])).hexdigest()==item['sha256']
    assert package.stat().st_size<=25_000_000,'Keep all necessary numeric evidence; reduce optional derived previews, never raw precision'
    result=dict(path=str(package),sha256=sha(package),bytes=package.stat().st_size,members=len(content)+1,CRC_checked=True,all_index_hashes_checked=True,DOCX_separate=True)
    (RUN/'protocol/package_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':run()
