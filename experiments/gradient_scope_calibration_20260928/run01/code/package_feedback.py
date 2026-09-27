"""Private bounded feedback package; include numeric contribution data, not weights."""
from pathlib import Path
import json,hashlib,zipfile,shutil,subprocess,math,csv
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
V5=ROOT/'experiments/numerical_stability_calibration_20260927/run01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def identity(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def run():
    assert read(RUN/'protocol/document_final_QA.json')['all_pages_visually_checked']
    fb=RUN/'feedback_arrays';fb.mkdir(exist_ok=True)
    for p in (V5/'feedback_arrays').glob('*'):
        if p.is_file():shutil.copyfile(p,fb/p.name)
    save(fb/'inheritance.json',dict(source=identity(V5/'feedback_arrays/manifest.json') if (V5/'feedback_arrays/manifest.json').exists() else None,scope='Unchanged V5 fixed float32 image windows; not new C output'))
    state=read(RUN/'state_manifest.json');state['local_diagnostic_large_assets']=[identity(p) for pat in ['diagnostics/*/effective_time.npz','diagnostics/*/*_probe_raw.npz','diagnostics/gradient_localization/*.pt'] for p in RUN.glob(pat)]
    state['history_raw_index']=identity(V5/'protocol/raw_render_index.json');save(RUN/'state_manifest.json',state)
    shutil.copyfile(V5/'protocol/raw_render_index.json',RUN/'protocol/inherited_raw_render_index.json')
    shutil.copyfile(V5/'protocol/public_sync_boundary_review.json',RUN/'protocol/inherited_public_sync_boundary_review.json')
    shutil.copyfile(V5/'environment.json',RUN/'protocol/inherited_environment.json')
    routes=read(RUN/'routing_equivalence.json');ga=[]
    for policy in ['C_route','uniform_all']:
        for scope in ['G','A','q']:
            values=[x for x in routes['rows'] if x['policy']==policy and x['scope']==scope and 'difference' in x]
            ga.append(dict(policy=policy,scope=scope,batch='00001/00041',**routes['losses'][policy],reference_gradient_norm=math.sqrt(sum(x['difference']['reference_norm']**2 for x in values)),norm_basis='independent U+R reference for G, F reference for C A, otherwise U; from saved per-tensor norms',routed_actual_norm=None,routed_norm_NA_reason='not serialized in first check; comparisons retained',all_finite=all(x['difference']['finite'] for x in values),max_abs_difference=max(x['difference']['max_abs'] for x in values),expected_None=sum(x.get('state')=='expected_None' for x in routes['rows'] if x['policy']==policy and x['scope']==scope)))
    with (RUN/'gradient_audit.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(ga[0]));w.writeheader();w.writerows(ga)
    # Exact generated source diffs are private; no third-party full snapshots shipped.
    source=RUN/'diagnostics/source_snapshots/render_final_bound.py'
    import difflib
    original=(ROOT/'third_party/4DGaussians/gaussian_renderer/__init__.py').read_text();generated=source.read_text();start=original.index('def render(')
    patch=''.join(difflib.unified_diff(original[start:].splitlines(True),generated.splitlines(True),fromfile='locked_upstream/render_function',tofile='V6/final_bound_precomputed_color'))
    patch+='\n# Gradient router is an added independent source file; see code/gradient_router.py.\n'
    (RUN/'patch.diff').write_text(patch)
    save(RUN/'protocol/source_manifest.json',dict(files=[identity(p) for p in sorted((RUN/'code').glob('*.py'))],configuration=identity(RUN/'configs/v6.json'),scope='Self-authored source; scientific outputs remain private'))
    required=['route_protocol.json','camera_time_audit.csv','camera_time_audit.json','renderer_parity.json','gradient_partition.json','routing_equivalence.json','gradient_audit.csv','contribution_manifest.json','time_camera_probe.json','bound_events_compact.csv','metrics_per_frame.csv','paired_differences.csv','run.json','costs.json','sampling_comparison.json','CAUSE_ASSESSMENT.md','NEXT_DECISION.md','state_manifest.json','environment.json','patch.diff','REPRODUCE.md','MISSING_ASSETS.md','figure_manifest.json','evaluation_summary.json']
    selected={RUN/p for p in required}
    for pattern in ['code/*.py','configs/*.json','protocol/*.json','protocol/*.jsonl','protocol/user_guidance_source.md','logs/*/*.json','logs/*/console.log','diagnostics/*/*.json','diagnostics/B_U_verified/*_contributions.npz','diagnostics/B_F/*_contributions.npz','diagnostics/B_U_verified/*_fixed_rows.npz','diagnostics/B_F/*_fixed_rows.npz','diagnostics/B_U_verified/*_time_camera.png','diagnostics/B_F/*_time_camera.png','diagnostics/gradient_localization/*samples.npz','inherited_figures/*.jpg','feedback_arrays/*.npz','feedback_arrays/*.json','output/report_figures/routing_gate.png','output/report_figures/contribution_summary.png']:
        selected.update(RUN.glob(pattern))
    selected={p for p in selected if p.is_file() and p.name not in ['package_verification.json','final_sync.json']}
    assert all('source_snapshots' not in p.parts and p.suffix not in ['.pt','.pth','.so','.docx'] for p in selected)
    content=[dict(name=str(p.relative_to(RUN)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)]
    output=RUN/'output/V6_feedback.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for row in content:z.write(RUN/row['name'],row['name'])
        z.writestr('PACKAGE_INDEX.json',json.dumps(dict(files=content,self_contained_training_reproduction=False,DOCX_separate=True,scope='Private feedback including GT comparison panels and unchanged V5 float32 crops. Per-frame contribution arrays lossless float32. Effective-time derived arrays and full gradients indexed on host; can recompute N_eff from included contributions. No C terminal.'),indent=2))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for row in content:assert hashlib.sha256(z.read(row['name'])).hexdigest()==row['sha256']
    result=dict(**identity(output),members=len(content)+1,CRC_and_member_SHA_passed=True,target_bytes=25_000_000,under_target=output.stat().st_size<=25_000_000);save(RUN/'protocol/package_verification.json',result);print(json.dumps(result))
if __name__=='__main__':run()
