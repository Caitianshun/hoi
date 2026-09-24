#!/usr/bin/env python3
"""Bounded CPU-only HMR2 asset gate and legal RGB input preparation; no inference."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/cai_tianshun/Project/HOI'))
    p.add_argument('--output',type=Path)
    args=p.parse_args();root=args.root.resolve()
    work=root/'experiments/mosca_interface_validation_20260923/human_prior'
    output=(args.output or work/'prerequisite_check').resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'audit.json').exists():raise RuntimeError('Use a fresh output directory')
    start=time.perf_counter()
    (output/'executed_audit.py').write_bytes(Path(__file__).read_bytes())
    roots=[Path('/home/cai_tianshun/Project'),Path('/home/cai_tianshun/.cache'),Path('/home/cai_tianshun/下载')]
    patterns=['*neutral*','*NEUTRAL*','*basicModel*','*mpips*','*smpl*.zip','*SMPL*.zip','*smpl*.tar*','*SMPL*.tar*']
    exclusions=['!**/envs/**','!**/site-packages/**','!**/.git/**','!**/node_modules/**']
    command=['rg','--files','--hidden','--no-ignore',*[str(x) for x in roots if x.exists()]]
    for pat in patterns+exclusions:command+=['-g',pat]
    result=subprocess.run(command,check=False,text=True,capture_output=True)
    assert result.returncode in (0,1),result.stderr
    found=sorted(result.stdout.splitlines())
    # No evaluation registrations/subject pose files are opened by this filename inventory.
    expected={'SMPL_NEUTRAL.pkl','basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'}
    compatible_names=[s for s in found if Path(s).name in expected]
    source=work/'evidence/hmr2/models/__init__.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_smpl_exists')
    (output/'official_check_smpl_exists.py').write_text(ast.get_source_segment(source.read_text(),node)+'\n')
    namespace={'CACHE_DIR_4DHUMANS':'/home/cai_tianshun/.cache/4DHumans'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    original_cwd=Path.cwd();os.chdir(work)
    try:
        try:
            namespace['check_smpl_exists']()
            official_check={'status':'passed'}
        except FileNotFoundError as e:
            official_check={'status':'missing_asset','exception_type':type(e).__name__,'message':str(e)}
    finally:os.chdir(original_cwd)
    input_path=root/'experiments/mosca_baseline_20260922/common_input/input_manifest.json'
    inp=json.loads(input_path.read_text())
    assert inp['role']=='input_only',inp['role']
    assert len(inp['frames'])==114
    expected_parent=(root/'experiments/mosca_baseline_20260922/common_input/images').resolve()
    failures=[]
    for f in inp['frames']:
        fp=Path(f['path']).resolve()
        assert fp.parent==expected_parent,fp
        if not fp.is_file() or sha(fp)!=f['sha256']:failures.append(str(fp))
    assert not failures,failures
    timestamps=inp['timestamp_seconds'];assert all(b>a for a,b in zip(timestamps,timestamps[1:]))
    future_input={'status':'input_ready_model_blocked' if not compatible_names else 'asset_candidate_requires_provenance_validation',
        'schema_version':1,'source_manifest':str(input_path),'source_manifest_sha256':sha(input_path),
        'role':'frozen_input_only_for_human_prior','frame_count':114,'frame_paths':inp['frame_paths'],
        'frames':inp['frames'],'timestamp_seconds':timestamps,'K':inp['K'],'c2w':inp['c2w'],
        'width':inp['width'],'height':inp['height'],'coordinate_system':inp['coordinate_system'],
        'input_rules':['No evaluation registrations, meshes, reference points, fitted body poses or hidden views.',
            'Do not re-encode irregular frames to 30 fps or replace timestamps by frame indices.',
            'HMR2 per-image inference must preserve all original input frame identities.',
            'Known calibrated K/c2w and original pixels remain distinct from the model crop camera.'],
        'prediction_files_created':False,'GPU_slot':'not_requested_while_asset_missing'}
    dump(output/'human_prior_input_manifest.json',future_input)
    report={'status':'blocked_missing_licensed_neutral_SMPL' if not compatible_names else 'asset_candidate_found_not_yet_accepted',
        'search_roots':[str(x) for x in roots],'search_patterns':patterns,'excluded_patterns':exclusions,
        'search_command':command,'filename_matches':found,'compatible_filename_candidates':compatible_names,
        'search_limits':'Filename/standard model asset directories only; no credential search, external disks, renamed opaque archives or environments. Presence is not license validation.',
        'official_HMR2_SMPL_check':official_check,'input_frames_hash_verified':114,'input_hash_failures':failures,
        'SMPLX_is_drop_in_HMR2_replacement':False,'needed_model':'neutral SMPL, not SMPL-X/SMPL-H/MANO or fitted pose arrays',
        'official_asset_source':'https://smplify.is.tue.mpg.de/download.php',
        'official_download_requires_account':'2026-09-23 read-only browser check redirected to login.php',
        'GPU_used':False,'weights_downloaded':False,'environment_installed':False,'inference_executed':False,
        'model_sources_sha256':{str(source):sha(source),str(work/'evidence/hmr2/models/smpl_wrapper.py'):sha(work/'evidence/hmr2/models/smpl_wrapper.py')},
        'script_sha256':sha(Path(__file__)),'wall_seconds':time.perf_counter()-start}
    dump(output/'audit.json',report)
    print(json.dumps({'status':report['status'],'matches':found,'official_check':official_check,'input_frames':114,'wall_seconds':report['wall_seconds']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
