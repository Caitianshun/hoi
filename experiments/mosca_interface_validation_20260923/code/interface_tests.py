#!/usr/bin/env python3
"""CPU acceptance for isolated MoSca geometry interface v2; no GPU or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import ast
import copy
import difflib
import hashlib
import io
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
REPO = HERE / 'MoSca'
SOURCE = ROOT / 'experiments/mosca_validation_20260923/code/MoSca'
sys.path.insert(0, str(REPO))
from lib_moca.camera import MonocularCameras, __backproject__, __project__, __get_homo_coordinate_map__
from lib_moca.pixel_geometry import pixel_to_homo, homo_to_pixel, get_homo_coordinate_map, normalized_principal
from lib_prior.prior_loading import get_homo_coordinate_map as prior_grid
from lib_mosca.dynamic_solver_utils import prepare_track_buffers, get_world_points


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_functions(relative_path, names, namespace):
    """Run actual small pure functions without importing unrelated CUDA renderers."""
    path = REPO / relative_path
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert len(nodes) == len(names), (path, names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return [namespace[name] for name in names]


def oracle_lift(pixel, depth, K):
    homogeneous = torch.cat([pixel, torch.ones_like(pixel[..., :1])], dim=-1)
    ray = homogeneous @ torch.linalg.inv(K).T
    return ray * (depth / ray[..., 2])[..., None]


def oracle_project(xyz, K):
    p = xyz @ K.T
    return p[..., :2] / p[..., 2:]


def error(a, b):
    return float((a - b).abs().max().detach())


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    logging.getLogger().setLevel(logging.ERROR)
    baseline = json.loads((HERE/'source_copy_manifest.json').read_text())
    assert Path(baseline['source']) == SOURCE
    assert all(sha(SOURCE / row['relative_path']) == row['sha256'] for row in baseline['files'])
    manifest = json.loads((ROOT/'experiments/mosca_baseline_20260922/common_input/input_manifest.json').read_text())
    K_real = torch.tensor(manifest['K'], dtype=torch.float64)
    run = json.loads((ROOT/'experiments/mosca_validation_20260923/a_normalized_exact/model/run.json').read_text())
    scale = run['world_scale']
    namespace = {'torch': torch, 'np': np, 'pixel_to_homo': pixel_to_homo, 'pixel_homo_grid': get_homo_coordinate_map}
    bundle_grid, bundle_track = extract_functions('lib_moca/bundle.py', ['get_homo_coordinate_map', 'track2undistroed_homo'], namespace.copy())
    dynamic_track, = extract_functions('lib_mosca/dynamic_solver.py', ['__int2homo_coord__'], namespace.copy())
    intrinsic_track, = extract_functions('lib_moca/intrinsic_helpers.py', ['track2undistroed_homo'], namespace.copy())
    helper_namespace = {'torch':torch, 'logging':logging, 'normalized_principal':normalized_principal}
    intrinsic_lift, intrinsic_project = extract_functions('lib_moca/intrinsic_helpers.py', ['backproject','project'], helper_namespace.copy())
    fov_lift, fov_project = extract_functions('lib_mosca/scaffold_utils/fov_helper.py', ['backproject','project'], helper_namespace.copy())
    cases = []
    for H, W in [(480, 640), (640, 480), (513, 513), (3, 7), (1, 7), (7, 1)]:
        K = K_real.clone() if (H,W)==(480,640) else torch.tensor([[0.83*min(H,W),0,0.43*W],[0,0.91*min(H,W),0.58*H],[0,0,1]],dtype=torch.float64)
        cam = MonocularCameras(2,H,W,K=K,delta_flag=False).double()
        # Constructor storage is float32; compare to the actual serialized K.
        K_stored = cam.K()
        yy, xx = torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
        pixel = torch.stack([xx,yy],-1).reshape(-1,2).double()
        grid = torch.from_numpy(get_homo_coordinate_map(H,W)).reshape(-1,2)
        for producer in [prior_grid,bundle_grid,__get_homo_coordinate_map__]:
            assert np.array_equal(producer(H,W),get_homo_coordinate_map(H,W))
        assert error(cam.get_homo_coordinate_map().reshape(-1,2),grid)<1e-12
        assert error(pixel_to_homo(pixel,H,W),grid)<1e-12
        assert error(homo_to_pixel(grid,H,W),pixel)<1e-10
        depth_results=[]
        for z in [0.5,2.1282360553741455,5.0]:
            depth=torch.full((len(pixel),),z,dtype=torch.float64)
            xyz=cam.backproject(grid,depth)
            true_xyz=oracle_lift(pixel,depth,K_stored)
            projected=oracle_project(xyz,K_stored)
            assert error(xyz,true_xyz)<1e-10
            assert error(projected,pixel)<1e-9
            assert error(cam.project_pixel(xyz),pixel)<1e-9
            assert error(cam.backproject_pixel(pixel,depth),true_xyz)<1e-10
            for lift,proj in [(intrinsic_lift,intrinsic_project),(fov_lift,fov_project)]:
                assert error(lift(grid,depth,cam),true_xyz)<1e-10
                assert error(homo_to_pixel(proj(xyz,cam),H,W),pixel)<1e-9
            # Actual training float32 path, including A's native world scale.
            cam32=copy.deepcopy(cam).float()
            xyz32=cam32.backproject(grid.float(),depth.float()*scale)
            u32=oracle_project(xyz32.double()/scale,K_stored)
            max32=error(u32,pixel)
            assert max32<0.001
            depth_results.append({'camera_z_m':z,'max_abs_lift_error_m':error(xyz,true_xyz),'max_abs_K_reprojection_px':error(projected,pixel),'max_abs_float32_scaled_reprojection_px':max32})
        sparse=torch.tensor([[0.,0.],[W-1.,H-1.],[W*.37-.13,H*.41+.21],[-.25,H+.75]],dtype=torch.float64)
        normal=pixel_to_homo(sparse,H,W)
        for fn in [bundle_track,dynamic_track,intrinsic_track]:
            assert error(fn(sparse,H,W),normal)<1e-12
        sparse_z=torch.tensor([.8,1.2,2.3,5.1],dtype=torch.float64)
        sparse_xyz=cam.backproject(normal,sparse_z)
        assert error(oracle_project(sparse_xyz,K_stored),sparse)<1e-9
        # Geometry and K at an explicitly requested alternate image shape.
        HH,WW=H+2,W+3
        alt_xyz=cam.backproject_pixel(sparse,sparse_z,H=HH,W=WW)
        assert error(alt_xyz,oracle_lift(sparse,sparse_z,cam.K(HH,WW)))<1e-10
        assert error(cam.project_pixel(alt_xyz,H=HH,W=WW),sparse)<1e-9
        cases.append({'H':H,'W':W,'full_grid_pixels':H*W,'noncentral_K':K_stored.detach().tolist(),'depths':depth_results,'sparse_fractional_and_outside_unclipped':'passed','alternate_H_W':'passed'})

    # Real sparse-track buffer integration: preserve original rounded nearest-depth
    # sampling, and verify its chosen pixel produces that pixel's K ray.
    H,W=480,640
    track=torch.tensor([[[267.,369.],[272.2,363.1],[235.7,381.6]],[[269.,371.],[273.2,365.1],[236.7,382.6]]])
    depth_map=(torch.arange(H)[:,None]*0.001+torch.arange(W)[None,:]*0.0001+2.)
    s2d=SimpleNamespace(H=H,W=W,dep=torch.stack([depth_map,depth_map+.1]),homo_map=torch.from_numpy(prior_grid(H,W)).float(),rgb=torch.zeros(2,H,W,3))
    valid=torch.tensor([[True,True,False],[True,True,True]])
    h,d,rgb=prepare_track_buffers(s2d,track,valid,[0,1])
    c2w=torch.tensor(manifest['c2w'],dtype=torch.float32);c2w[:3,3]*=scale
    cam=MonocularCameras(2,H,W,K=K_real.float(),delta_flag=False,init_camera_pose=c2w.repeat(2,1,1))
    world=get_world_points(h,d*scale,cam)
    sparse_errors=[]
    for t in range(2):
        p=track[t].round()
        expected=oracle_lift(p,d[t],cam.K())*scale
        expected=expected @ c2w[:3,:3].T+c2w[:3,3]
        e=error(world[t,valid[t]],expected[valid[t]])
        assert e<2e-6
        back=cam.trans_pts_to_cam(t,world[t,valid[t]])
        proj=cam.project_pixel(back)
        assert error(proj,p[valid[t]])<.001
        sparse_errors.append({'frame':t,'world_native_max_abs':e,'pixel_roundtrip_max_abs':error(proj,p[valid[t]])})
    assert (d[~valid]==-1).all() and (h[~valid]==0).all()

    # Meaningful derivatives: gradcheck each nontrivial lift/project operation,
    # including uv/z/focal/principal. Check principal gradients aren't detached.
    H,W=480,640
    uv=torch.tensor([[-.8,-.1],[.4,.6]],dtype=torch.float64,requires_grad=True)
    dep=torch.tensor([1.3,2.8],dtype=torch.float64,requires_grad=True)
    focal=torch.tensor([1.33,1.4],dtype=torch.float64,requires_grad=True)
    ratio=torch.tensor([.43,.56],dtype=torch.float64,requires_grad=True)
    def lift_fn(uv,dep,focal,ratio):
        c=SimpleNamespace(rel_focal=focal,cxcy_ratio=ratio,default_H=H,default_W=W)
        return __backproject__(uv,dep,c)
    xyz=torch.tensor([[-.2,.3,1.3],[.5,-.4,2.8]],dtype=torch.float64,requires_grad=True)
    def project_fn(xyz,focal,ratio):
        c=SimpleNamespace(rel_focal=focal,cxcy_ratio=ratio,default_H=H,default_W=W)
        return __project__(xyz,c)
    assert torch.autograd.gradcheck(lift_fn,(uv,dep,focal,ratio),eps=1e-6,atol=1e-6,rtol=1e-4)
    assert torch.autograd.gradcheck(project_fn,(xyz,focal,ratio),eps=1e-6,atol=1e-6,rtol=1e-4)
    pgrad=torch.autograd.grad(project_fn(xyz,focal,ratio).sum(),ratio)[0]
    expected_grad=torch.tensor([2*W/min(H,W)*len(xyz),2*H/min(H,W)*len(xyz)],dtype=torch.float64)
    assert error(pgrad,expected_grad)<1e-12

    # Version boundary: unknown/legacy checkpoints fail unless conversion is explicit.
    state=cam.state_dict()
    buf=io.BytesIO();torch.save(state,buf);buf.seek(0)
    loaded=MonocularCameras.load_from_ckpt(torch.load(buf,weights_only=False))
    assert int(loaded.geometry_interface_version)==2
    assert error(loaded.K(),cam.K())==0
    legacy={k:v for k,v in state.items() if k!='geometry_interface_version'}
    try:MonocularCameras.load_from_ckpt(legacy)
    except ValueError:pass
    else:raise AssertionError('Unmarked checkpoint must not silently load as v2')
    converted=MonocularCameras.load_from_ckpt(legacy,allow_legacy_geometry=True)
    assert int(converted.geometry_interface_version)==2 and 'geometry_interface_version' not in legacy
    bad=dict(state);bad['geometry_interface_version']=torch.tensor(99)
    try:MonocularCameras.load_from_ckpt(bad)
    except ValueError:pass
    else:raise AssertionError('Unknown version must be rejected')
    # Verify pixel loss caller uses the accepted API, not a stale manual decoder.
    photo=ast.parse((REPO/'lib_mosca/photo_recon.py').read_text())
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='project_pixel' for n in ast.walk(photo))

    changes=[];diffs=[]
    candidates={r['relative_path'] for r in baseline['files']}|{'lib_moca/pixel_geometry.py'}
    for relative in sorted(candidates):
        old,new=SOURCE/relative,REPO/relative
        if not new.exists():continue
        old_hash=sha(old) if old.exists() else None
        new_hash=sha(new)
        if old_hash!=new_hash:
            ast.parse(new.read_text()) if new.suffix=='.py' else None
            changes.append({'path':relative,'base_sha256':old_hash,'new_sha256':new_hash})
            diffs.extend(difflib.unified_diff(old.read_text().splitlines(keepends=True) if old.exists() else [],new.read_text().splitlines(keepends=True),fromfile='previous/'+relative,tofile='interface_v2/'+relative))
    source_unchanged=all(sha(SOURCE/r['relative_path'])==r['sha256'] for r in baseline['files'])
    assert source_unchanged
    (HERE/'interface_patch.diff').write_text(''.join(diffs))
    report={'status':'passed','device':'cpu','no_gpu_or_training':True,'geometry_interface_version':2,'definition':'homo=(2*pixel-[W,H])/min(H,W); pixel centres are integer indices; depth is camera z','full_grid_cases':cases,'sparse_track_buffer_integration':sparse_errors,'sparse_sampling_policy':'Existing rounded/clamped nearest-depth lookup retained, not silently changed to bilinear or raw fractional depth sampling. Direct fractional pixel API tested independently.','geometry_gradcheck':{'lift_uv_depth_focal_principal':'passed','project_xyz_focal_principal':'passed','principal_gradient':pgrad.tolist()},'checkpoint_version_tests':'v2 roundtrip, legacy rejection, explicit conversion, unknown version rejection passed','base_sources_unchanged':source_unchanged,'modified_files':changes,'patch_sha256':sha(HERE/'interface_patch.diff'),'test_script_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-started,'limits':'CPU geometry/loss-coordinate acceptance, not renderer GPU runtime or trained quality. Pretrained tracker/flow internals and their own grid_sample coordinates are untouched.'}
    (HERE/'interface_tests.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'full_grid_cases':len(cases),'grid_depth_combinations':len(cases)*3,'modified_files':len(changes),'seconds':report['elapsed_seconds'],'source_unchanged':source_unchanged}))


if __name__=='__main__':
    main()
