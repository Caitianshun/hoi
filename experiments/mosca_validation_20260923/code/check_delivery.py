"""Read-only validation of completed controls and local Markdown delivery."""
from pathlib import Path
import argparse, hashlib, json, re
import numpy as np
ROOT=Path('/home/cai_tianshun/Project/HOI')
EXP=ROOT/'experiments/mosca_validation_20260923'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    runs={b:EXP/b for b in ['a_normalized_exact','b_normalized_static_bg']}
    records={};identities=[];dynamic_hash=[]
    for b,d in runs.items():
        flow=read(d/'pipeline.json');r=read(d/'model/run.json');init=read(d/'model/initialization.json')
        assert flow['status']==r['status']=='completed'
        assert all(s['status']=='completed' and s['exit_code']==0 for s in flow['stages'])
        assert max(r['camera_max_abs_changes'].values())==0
        assert len(list((d/'diagnostics/rgb').glob('*.png')))==114
        assert len(list((d/'heldout/heldout_cam1_render').glob('*.png')))==5
        assert r['input_manifest_sha256']==sha(ROOT/'experiments/mosca_baseline_20260922/common_input/input_manifest.json')
        assert r['segmentation_sha256']==sha(ROOT/'experiments/mosca_baseline_20260922/segmentation/segmentation.npz')
        with np.load(d/'evaluation/evaluation_bundle.npz') as z:
            identities.append({k:z[k].copy() for k in ['reference','valid_mask','frame_times','entity','query_id','coordinate_frame','units']})
            assert z['predicted_valid_mask'].all()
        dynamic_hash.append(init['dynamic_state_tensor_sha256'])
        records[b]={'flow_sha256':sha(d/'pipeline.json'),'training_sha256':sha(d/'model/run.json'),
                    'dynamic_initialization_tensor_sha256':dynamic_hash[-1],
                    'checkpoint_sha256':{f:sha(d/'model'/f) for f in ['photometric_s_model_native_add3.pth','photometric_d_model_native_add3.pth']}}
    assert dynamic_hash[0]==dynamic_hash[1]
    assert all(np.array_equal(identities[0][k],identities[1][k]) for k in identities[0])
    b=read(runs['b_normalized_static_bg']/'model/run.json')
    assert b['allowed_photo_only_config_differences']=={'gs_include_fg_in_static':[True,False]}
    for name,digest in b['reused_scaffold']['files'].items():
        assert sha(Path(b['reused_scaffold']['source'])/name)==digest
        assert sha(runs['b_normalized_static_bg']/'model'/name)==digest
    documents=[EXP/'REPORT.md',EXP/'REPRODUCE.md',EXP/'PLAN.md',ROOT/'README.md',ROOT/'RESEARCH_LOG.md']
    documents.extend(p for p in EXP.rglob('*.md') if '/code/MoSca/' not in str(p) and p not in documents)
    links=[]
    for path in documents:
        body=path.read_text()
        assert '\x00' not in body
        if path not in [ROOT/'RESEARCH_LOG.md']:
            assert '$$' not in body and '\\begin{aligned}' not in body, path
        for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',body):
            target=raw.strip('<>')
            if target.startswith(('https://','http://','#','app://')):continue
            target=target.split('#')[0];target=re.sub(r':\d+$','',target)
            pth=Path(target)
            assert pth.is_absolute(),(path,target)
            assert pth.exists(),(path,target)
            links.append({'document':str(path),'target':target})
    result={'status':'passed','runs':records,'same_reference_verified':True,'same_dynamic_initialization_verified':True,
            'only_config_difference':'gs_include_fg_in_static: true -> false',
            'documents':{str(p):sha(p) for p in documents},'local_links_checked':len(links),
            'scope':'Completed pipeline artifacts, shared input identities, same reference and initialization, links and math-source scan. Not VS Code/Codex UI verification.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','documents':len(documents),'local_links_checked':len(links)},indent=2))
if __name__=='__main__':main()
