"""Explicit, training-only data interface for the locked official Wu 4DGS.

No official model/SH/density/loss implementation is replaced by this adapter.
Pixel centres follow OpenCV: rasterizer NDC to pixel includes its -0.5 shift.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/4DGaussians'
OFFICIAL_COMMIT = '843d5ac636c37e4b611242287754f3d4ed150144'
sys.path.insert(0, str(UPSTREAM))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def official_config(batch_size=2):
    from arguments import ModelParams, ModelHiddenParams, OptimizationParams, PipelineParams
    parser = argparse.ArgumentParser()
    groups = [ModelParams(parser), ModelHiddenParams(parser), OptimizationParams(parser), PipelineParams(parser)]
    args = parser.parse_args([])
    cfg = runpy.run_path(str(UPSTREAM / 'arguments/hypernerf/default.py'))
    for group in ['ModelHiddenParams', 'OptimizationParams']:
        for key, value in cfg[group].items():
            if not hasattr(args, key):
                raise KeyError(key)
            setattr(args, key, value)
    # render_process invokes test RGB in upstream: explicitly disable it.
    args.render_process = False
    args.batch_size = batch_size
    args.data_device = 'cpu'
    args.source_path = ''
    args.model_path = ''
    return args, [g.extract(args) for g in groups]


class CalibratedCamera:
    def __init__(self, frame, uid, load_rgb=True):
        self.uid = uid; self.colmap_id = uid
        self.image_name = str(frame['frame_id'])
        self.time = float(frame['time'])
        self.image_width = int(frame['width']); self.image_height = int(frame['height'])
        K = np.asarray(frame['K'], dtype=np.float64)
        w2c = np.asarray(frame['w2c'], dtype=np.float64)
        assert abs(K[0, 1]) < 1e-8 and abs(K[1, 0]) < 1e-8, 'Skew needs separate raster covariance adaptation'
        self.K = K; self.R = w2c[:3, :3].T; self.T = w2c[:3, 3]
        self.FoVx = 2 * math.atan(self.image_width / (2 * K[0, 0]))
        self.FoVy = 2 * math.atan(self.image_height / (2 * K[1, 1]))
        self.znear = 0.01; self.zfar = 100.0
        p = np.zeros((4, 4), np.float64)
        p[0, 0] = 2 * K[0, 0] / self.image_width
        p[1, 1] = 2 * K[1, 1] / self.image_height
        p[0, 2] = (2 * K[0, 2] + 1) / self.image_width - 1
        p[1, 2] = (2 * K[1, 2] + 1) / self.image_height - 1
        p[3, 2] = 1
        p[2, 2] = self.zfar / (self.zfar - self.znear)
        p[2, 3] = -self.zfar * self.znear / (self.zfar - self.znear)
        self.world_view_transform = torch.from_numpy(w2c.T.copy()).float()
        self.projection_matrix = torch.from_numpy(p.T.copy()).float()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.mask = None; self.depth = None
        if load_rgb:
            image = cv2.imread(frame['image_path'], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(frame['image_path'])
            assert image.shape[:2] == (self.image_height, self.image_width)
            self.original_image = torch.from_numpy(image[..., ::-1].copy()).permute(2, 0, 1).float() / 255


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    assert m['role'] == 'training_only'
    assert len(m['frames']) and m['scene_extent'] > 0
    assert all(0 <= f['time'] <= 1 for f in m['frames'])
    times = [f['time_seconds'] for f in m['frames']]
    assert times == sorted(times) and len(times) == len(set(times))
    for f in m['frames']:
        if f.get('image_sha256'):
            assert sha(f['image_path']) == f['image_sha256']
        assert np.allclose(np.array(f['w2c']) @ np.array(f['c2w']), np.eye(4), atol=1e-6)
    return m


class ProtocolScene:
    def __init__(self, args, gaussians, **kwargs):
        from utils.graphics_utils import BasicPointCloud
        self.manifest = load_manifest(args.source_path)
        self.model_path = args.model_path
        self.gaussians = gaussians; self.loaded_iter = None
        self.dataset_type = 'explicit_protocol'; self.maxtime = 1.0
        self.cameras_extent = self.manifest['scene_extent']
        self.train_camera = [CalibratedCamera(f, i) for i, f in enumerate(self.manifest['frames'])]
        self.test_camera = []; self.video_camera = []
        p = np.load(self.manifest['point_cloud']['npz_path'])
        xyz = p['xyz'].astype(np.float32); rgb = p['rgb'].astype(np.float32)
        assert np.isfinite(xyz).all() and np.isfinite(rgb).all()
        assert rgb.min() >= 0 and rgb.max() <= 1
        gaussians._deformation.deformation_net.set_aabb(xyz.max(0), xyz.min(0))
        gaussians.create_from_pcd(BasicPointCloud(xyz, rgb, np.zeros_like(xyz)), self.cameras_extent, 1)

    def getTrainCameras(self, scale=1): return self.train_camera
    def getTestCameras(self, scale=1): return self.test_camera
    def getVideoCameras(self, scale=1): return self.video_camera

    def save(self, iteration, stage):
        p = Path(self.model_path) / 'point_cloud' / f'{stage}_iteration_{iteration}'
        self.gaussians.save_ply(str(p / 'point_cloud.ply'))
        self.gaussians.save_deformation(str(p))


def projection_check(m):
    xyz = np.load(m['point_cloud']['npz_path'])['xyz'][::17].astype(np.float64)
    ones = np.ones((len(xyz), 1))
    max_error = 0.; roundtrip = 0.; count = 0
    for i in [0, len(m['frames']) // 2, len(m['frames']) - 1]:
        f = m['frames'][i]; camera = CalibratedCamera(f, i, load_rgb=False)
        cv = np.c_[xyz, ones] @ np.array(f['w2c']).T
        good = cv[:, 2] > camera.znear
        projected = cv[good, :3] @ np.array(f['K']).T
        uv = projected[:, :2] / projected[:, 2:]
        clip = np.c_[xyz[good], ones[good]] @ camera.full_proj_transform.numpy()
        ndc = clip[:, :2] / clip[:, 3:4]
        pix = ((ndc + 1) * [camera.image_width, camera.image_height] - 1) * .5
        max_error = max(max_error, float(np.abs(pix - uv).max()))
        restored = cv @ np.array(f['c2w']).T
        roundtrip = max(roundtrip, float(np.abs(restored[:, :3] - xyz).max()))
        count += len(uv)
    assert max_error < .5
    return {'tested_projections': count, 'max_pixel_error': max_error,
            'world_roundtrip_max_m': roundtrip, 'scene_extent': m['scene_extent'],
            'aabb': m['aabb'], 'passed': True, 'test_RGB_loaded': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('manifest'); p.add_argument('--output', required=True)
    a = p.parse_args(); save_json(a.output, projection_check(load_manifest(a.manifest)))
