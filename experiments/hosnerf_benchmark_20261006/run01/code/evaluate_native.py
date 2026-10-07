#!/usr/bin/env python3
"""Evaluate one final native HOSNeRF checkpoint on the frozen 16 test frames.

The official test_metrics sampling/compositing body is retained. Capture hooks
save floating RGB and PNGs and correct flattened SSIM to image-shaped SSIM.
No optimizer, all-frame render, freeview render, or checkpoint selection runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import traceback
import types


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official", type=Path, default=ROOT / "third_party/HOSNeRF")
    parser.add_argument("--data-manifest", type=Path, default=RUN / "protocol/dataset_manifest.json")
    parser.add_argument("--source-kind", choices=["local_retrained", "official_reference"],
                        default="local_retrained")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true",
                        help="Strict CPU load and data identity check, without rendering")
    args = parser.parse_args()
    for name in ["data_root", "checkpoint", "output", "official", "data_manifest"]:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.chunk < 1 or args.threads < 1:
        parser.error("chunk and threads must be positive")
    return args


def main(args):
    import numpy as np
    from PIL import Image
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.json").exists():
        prior = json.loads((args.output / "metrics.json").read_text())
        if prior.get("status") == "completed":
            if prior.get("source_kind") != args.source_kind or prior.get("scene") != args.scene:
                raise RuntimeError("Completed evaluation has a different scene/source identity")
            if prior["identity"]["checkpoint_sha256"] != sha256(args.checkpoint):
                raise RuntimeError("Completed evaluation exists for a different checkpoint")
            if prior.get("frames") != 16 or len(prior.get("per_frame", [])) != 16 or len(prior.get("artifacts", [])) != 48:
                raise RuntimeError("Completed evaluation has incomplete frame/artifact records")
            for resource in prior["identity"]["test_resources"]:
                if sha256(args.data_root / args.scene / resource["path"]) != resource["sha256"]:
                    raise RuntimeError("Completed evaluation test input identity changed")
            for artifact in prior["artifacts"]:
                if sha256(args.output / artifact["path"]) != artifact["sha256"]:
                    raise RuntimeError("Completed evaluation output missing or changed")
            print("HOS_EVALUATION_ALREADY_COMPLETE", args.output, flush=True)
            return
    data = args.data_root / args.scene
    files = sorted(data.glob("images/*.png"))
    if len(files) < 16:
        raise ValueError("Native uniform-16 evaluation needs at least 16 PNG frames")
    indices = list(range(0, len(files), len(files) // 16))[:16]
    expected_ids = [files[i].stem for i in indices]
    manifest = json.loads(args.data_manifest.read_text())
    scene_manifest = manifest["scenes"][args.scene]
    if expected_ids != scene_manifest["test_ids"]:
        raise RuntimeError("Dataset manifest and actual uniform-16 frame IDs disagree")
    file_hashes = {row["path"]: row["sha256"] for row in scene_manifest["files"]}
    test_resources = []
    for frame_id in expected_ids:
        for folder in ["images", "masks"]:
            relative = f"{folder}/{frame_id}.png"
            actual = sha256(data / relative)
            if actual != file_hashes[relative]:
                raise RuntimeError(f"Evaluation input changed after freeze: {relative}")
            test_resources.append(dict(path=relative, sha256=actual))
    with Image.open(files[0]) as im:
        resolution = list(im.size)
    if resolution != [scene_manifest["resolution"]["width"], scene_manifest["resolution"]["height"]]:
        raise RuntimeError("Manifest resolution does not match native RGB")

    source = args.official / "3rd_Complete_HOSNeRF"
    os.chdir(source)
    sys.path.insert(0, str(source))
    if "bool" not in np.__dict__:
        np.bool = np.bool_
    original_load = torch.load
    @functools.wraps(original_load)
    def legacy_load(*positional, **keywords):
        keywords.setdefault("weights_only", False)
        return original_load(*positional, **keywords)
    torch.load = legacy_load
    torch.set_num_threads(args.threads)
    torch.manual_seed(777)
    np.random.seed(777)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = int(checkpoint["global_step"])
    training_identity = checkpoint.get("benchmark_identity")
    if args.source_kind == "local_retrained":
        if training_identity is None:
            raise RuntimeError("Local retrained evaluation requires benchmark_identity")
        if not (training_identity["stage"] == 3 and training_identity["scene"] == args.scene
                and training_identity["formal"] and step == training_identity["max_steps"] == 200000):
            raise RuntimeError("Checkpoint is not the prescribed formal stage-3 endpoint")
        if training_identity["test_ids"] != expected_ids:
            raise RuntimeError("Checkpoint test IDs disagree with frozen evaluation IDs")
        for name, expected in training_identity["metadata_sha256"].items():
            if sha256(data / name) != expected:
                raise RuntimeError(f"Metadata changed after training freeze: {name}")

    from third_parties.yacs import CfgNode as CN
    cfg = CN()
    cfg.resume = False
    cfg.eval_iter = 10000000
    cfg.render_folder_name = ""
    cfg.ignore_non_rigid_motions = False
    cfg.render_skip = 1
    cfg.render_frames = 100
    cfg.num_workers = 1
    cfg.merge_from_file(str(source / "configs/default.yaml"))
    cfg.merge_from_file(str(source / "configs/human_nerf/wild/monocular/adventure.yaml"))
    cfg.basedir = str(data)
    cfg.chunk_bkg = args.chunk
    from core.data.dataset_args import DatasetArgs
    original_get = DatasetArgs.get
    def local_get(configuration, name):
        result = original_get(configuration, name)
        result["dataset_path"] = str(data)
        return result
    DatasetArgs.get = staticmethod(local_get)
    import src.model.mipnerf360.model as official_model
    import gin
    gin.bind_parameter("MipNeRF360.opaque_background", True)
    original_loader = official_model.create_dataloader
    official_model.create_dataloader = lambda c, data_type: original_loader(c, data_type) if data_type == "test" else None
    original_lpips = official_model.LPIPS
    # Every VGG trunk and LPIPS parameter is subsequently loaded strictly.
    # A random constructor avoids a redundant ImageNet download, not an
    # evaluation with random perceptual weights.
    official_model.LPIPS = lambda **keywords: original_lpips(pnet_rand=True, **keywords)
    model = official_model.LitMipNeRF360(cfg=cfg, basedir=str(data))
    loaded = model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.logdir = str(args.output)
    model.near_bkg = .1
    model.far_bkg = 1e6
    model._trainer = types.SimpleNamespace(global_step=step)
    if list(model.test_dataloader.dataset.framelist) != expected_ids:
        raise RuntimeError("Official test loader differs from the frozen 16 IDs")
    revision = subprocess.check_output(["git", "-C", str(args.official), "rev-parse", "HEAD"], text=True).strip()
    model_path = source / "src/model/mipnerf360/model.py"
    if training_identity is not None:
        if training_identity["official_revision"] != revision or training_identity["official_model_sha256"] != sha256(model_path):
            raise RuntimeError("Official model code identity changed since training")
    identity = dict(scene=args.scene, source_kind=args.source_kind, checkpoint=str(args.checkpoint),
                    checkpoint_sha256=sha256(args.checkpoint), global_step=step,
                    strict_load=True, missing_keys=loaded.missing_keys, unexpected_keys=loaded.unexpected_keys,
                    checkpoint_benchmark_identity=training_identity,
                    official_revision=revision, official_model_sha256=sha256(model_path),
                    evaluator_sha256=sha256(Path(__file__)), data_path=str(data),
                    dataset_manifest=str(args.data_manifest), dataset_manifest_sha256=sha256(args.data_manifest),
                    test_ids=expected_ids, test_resources=test_resources, resolution=resolution,
                    resize_img_scale=float(cfg.resize_img_scale), chunk_bkg=args.chunk,
                    renderer="official LitMipNeRF360.test_metrics; sample/composite unchanged",
                    LPIPS="official third_parties.lpips.LPIPS(net=vgg), scalar full image, input 2*RGB-1",
                    SSIM="HWC image SSIM, channel_axis=2, data_range=1; fixes vendor flattened Nx3 metric",
                    aggregation="arithmetic mean of exactly 16 per-frame scalar values",
                    adjustment="local paths; unused loaders skipped; random VGG constructor followed by strict full state load; output hooks; spatial SSIM",
                    torch_version=torch.__version__, numpy_version=np.__version__)
    write_json(args.output / "load_identity.json", identity)
    print("HOS_EVALUATION_LOAD_OK", args.scene, step, flush=True)
    if args.check_only:
        write_json(args.output / "construction_check.json", dict(status="passed", identity=identity))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Native renderer needs CUDA; CPU validation uses --check-only")
    torch.cuda.reset_peak_memory_stats()
    model.cuda()
    model.eval()
    native = textwrap.dedent(inspect.getsource(official_model.LitMipNeRF360.test_metrics))
    ssim_source = "ssim = skimage.metrics.structural_similarity(rendered, truth, channel_axis=True)"
    scalar_source = "lpipss.append(lpips)"
    if native.count(ssim_source) != 1 or native.count(scalar_source) != 1:
        raise RuntimeError("Vendor test_metrics changed; capture hooks require a new audit")
    native = native.replace(ssim_source, "ssim = self.capture_float(frame_name, rendered, truth, int(height), int(width))")
    native = native.replace(scalar_source, "lpipss.append(lpips)\n        self.capture_scalar(frame_name, psnr, ssim, lpips)")
    namespace = {}
    exec(native, vars(official_model), namespace)
    rows = []
    for name in ["floats", "images", "gt"]:
        (args.output / name).mkdir(exist_ok=True)
    def capture_float(self, frame_id, prediction, truth, height, width):
        from skimage.metrics import structural_similarity
        rgb = prediction.reshape(height, width, 3)
        target = truth.reshape(height, width, 3)
        if not np.isfinite(rgb).all() or not np.isfinite(target).all():
            raise FloatingPointError(f"Nonfinite render at frame {frame_id}")
        np.savez_compressed(args.output / "floats" / (frame_id + ".npz"), rgb=rgb, gt=target)
        for name, array in [("images", rgb), ("gt", target)]:
            Image.fromarray((np.clip(array, 0., 1.) * 255.).astype(np.uint8)).save(args.output / name / (frame_id + ".png"))
        return structural_similarity(rgb, target, data_range=1., channel_axis=2)
    def capture_scalar(self, frame_id, psnr, ssim, lpips):
        row = dict(frame_id=frame_id, PSNR=float(psnr), SSIM=float(ssim), LPIPS=float(lpips))
        if frame_id not in expected_ids or any(r["frame_id"] == frame_id for r in rows):
            raise RuntimeError(f"Unexpected or repeated evaluation frame {frame_id}")
        if not all(np.isfinite(row[key]) for key in ["PSNR", "SSIM", "LPIPS"]):
            raise FloatingPointError(f"Nonfinite metrics at frame {frame_id}")
        rows.append(row)
        write_json(args.output / "per_frame.json", rows)
        print("HOS_FRAME", json.dumps(row), flush=True)
    model.capture_float = types.MethodType(capture_float, model)
    model.capture_scalar = types.MethodType(capture_scalar, model)
    model.test_metrics = types.MethodType(namespace["test_metrics"], model)
    started = time.monotonic()
    with torch.no_grad():
        model.test_metrics()
    if [row["frame_id"] for row in rows] != expected_ids:
        raise RuntimeError("Endpoint evaluation did not emit exactly the frozen 16 frames")
    artifacts = []
    for frame_id in expected_ids:
        for name, suffix in [("floats", ".npz"), ("images", ".png"), ("gt", ".png")]:
            artifact = args.output / name / (frame_id + suffix)
            artifacts.append(dict(path=str(artifact.relative_to(args.output)), sha256=sha256(artifact)))
    result = dict(status="completed", source_kind=args.source_kind, scene=args.scene,
                  frames=len(rows), completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.monotonic() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  mean_metrics={key: float(np.mean([row[key] for row in rows])) for key in ["PSNR", "SSIM", "LPIPS"]},
                  per_frame=rows, identity=identity, artifacts=artifacts)
    write_json(args.output / "metrics.json", result)
    write_json(args.output / "completion.json", result)
    print("HOS_EVALUATION_COMPLETE", json.dumps(result["mean_metrics"]), flush=True)


if __name__ == "__main__":
    parsed = arguments()
    try:
        main(parsed)
    except BaseException as error:
        parsed.output.mkdir(parents=True, exist_ok=True)
        write_json(parsed.output / "evaluation_failure.json", dict(status="failed", error=repr(error), traceback=traceback.format_exc()))
        raise
