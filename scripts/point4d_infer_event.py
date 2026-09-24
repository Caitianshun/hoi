#!/usr/bin/env python3
"""Freeze Point4D predictions for one explicitly enumerated RGB-only event.

Manifest schema: {"frames": [{"path": "rgb/000001.jpg", "frame_id": "1",
"timestamp": 0.033}, ...]}. Paths are relative to the manifest. No depth,
calibration, fit, or held-out image fields are read. All queries start at the
first selected input frame. This wrapper does not infer visibility from conf.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "third_party/Point4D"
OFFICIAL_COMMIT = "4da8153711a4d37afbdea9cdc789db20ac9d3f4a"
EXPECTED_CHECKPOINT_SHA256 = "dacdabdd8678da9f247f6ac51aefcd1771a94af4b677166a5683b6971c4950cb"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--frames-dir", type=Path)
    source.add_argument("--video", type=Path, help="For setup smoke tests; exact decoded frames are hashed.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-points", type=Path, help="N x 2 .npy or JSON list in original input pixel coordinates.")
    parser.add_argument("--query-mask", type=Path, help="Optional first-frame mask derived from legal RGB only.")
    parser.add_argument("--query-stride", type=int, default=32, help="Grid spacing at model input resolution.")
    parser.add_argument("--height", type=int, default=294)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--chunk-size", type=int, default=48)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--geometry-stride", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "checkpoints/point4d/point4d_final.pt")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and hash RGB/queries without importing torch.")
    parser.add_argument("--fp32", action="store_true", help="Disable official BF16 autocast.")
    args = parser.parse_args()
    if args.height % 14 or args.width % 14:
        parser.error("Model height and width must be multiples of the 14-pixel patch size.")
    if not 0 <= args.overlap < args.chunk_size or args.chunk_size < 2:
        parser.error("Require 0 <= overlap < chunk-size and chunk-size >= 2.")
    if min(args.frame_stride, args.query_stride, args.geometry_stride, args.query_chunk_size) < 1 or args.start < 0:
        parser.error("Strides and query chunk size must be positive; start must be nonnegative.")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("max-frames must be at least 2.")
    return args


def load_rgb(args):
    import cv2
    import numpy as np

    source_meta = {}
    if args.video:
        video = args.video.resolve()
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        selected, records, frame_index = [], [], 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_index >= args.start and (frame_index - args.start) % args.frame_stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                selected.append(rgb)
                records.append({"path": str(video), "frame_id": str(frame_index),
                                "timestamp": frame_index / fps if fps > 0 else None,
                                "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
                if args.max_frames and len(selected) >= args.max_frames:
                    break
            frame_index += 1
        cap.release()
        source_meta = {"video_sha256": sha256(video), "fps": fps}
    else:
        if args.manifest:
            manifest = args.manifest.resolve()
            data = json.loads(manifest.read_text())
            items = data["frames"]
            if not isinstance(items, list):
                raise ValueError("Manifest frames must be an ordered list.")
            paths = []
            for item in items:
                item = {"path": item} if isinstance(item, str) else item
                path = Path(item["path"])
                if not path.is_absolute():
                    path = manifest.parent / path
                paths.append((path.resolve(), item))
            source_meta = {"manifest_path": str(manifest), "manifest_sha256": sha256(manifest)}
        else:
            paths = [(p.resolve(), {"frame_id": p.stem}) for p in sorted(args.frames_dir.iterdir())
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        paths = paths[args.start::args.frame_stride]
        if args.max_frames:
            paths = paths[:args.max_frames]
        selected, records = [], []
        for path, item in paths:
            file_hash = sha256(path)
            if item.get("sha256") and item["sha256"] != file_hash:
                raise ValueError(f"Manifest SHA-256 mismatch for RGB image: {path}")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot decode RGB image: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            selected.append(rgb)
            records.append({"path": str(path), "frame_id": str(item.get("frame_id", item.get("video_frame_index", item.get("index", path.stem)))),
                            "timestamp": item.get("timestamp", item.get("timestamp_seconds")),
                            "video_frame_index": item.get("video_frame_index"), "sha256": file_hash,
                            "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
    if len(selected) < 2:
        raise ValueError("At least two selected RGB frames are required.")
    if len({rgb.shape for rgb in selected}) != 1:
        raise ValueError("Input RGB dimensions change; explicit upstream resampling is required.")
    return np.stack(selected), records, source_meta


def make_queries(args, source_hw):
    import cv2
    import numpy as np

    sh, sw = source_hw
    scale = np.array([args.width / sw, args.height / sh], dtype=np.float32)
    if args.query_points:
        original = (np.load(args.query_points, allow_pickle=False) if args.query_points.suffix == ".npy"
                    else np.asarray(json.loads(args.query_points.read_text()), dtype=np.float32))
        original = np.asarray(original, dtype=np.float32)
        if original.ndim != 2 or original.shape[1] != 2 or not np.isfinite(original).all():
            raise ValueError("Query file must contain a finite N x 2 array.")
        if ((original < 0).any() or (original[:, 0] >= sw).any() or (original[:, 1] >= sh).any()):
            raise ValueError("A query lies outside the first original image.")
        model = (original + 0.5) * scale - 0.5
    else:
        yy, xx = np.meshgrid(np.arange(args.query_stride // 2, args.height, args.query_stride),
                             np.arange(args.query_stride // 2, args.width, args.query_stride), indexing="ij")
        model = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        original = (model + 0.5) / scale - 0.5
    query_ids = np.arange(len(model), dtype=np.int64)
    if args.query_mask:
        mask = cv2.imread(str(args.query_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None or tuple(mask.shape) != (sh, sw):
            raise ValueError("Query mask must have exactly the original input height and width.")
        xy = np.rint(original).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, sw - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, sh - 1)
        keep = mask[xy[:, 1], xy[:, 0]] != 0
        original, model = original[keep], model[keep]
        query_ids = query_ids[keep]
    if len(model) == 0:
        raise ValueError("No valid first-frame queries remain.")
    # The official decoder expects bounded pixel queries. Endpoint resizing can
    # place a corner pixel fractionally outside the destination center range.
    model[:, 0] = np.clip(model[:, 0], 0, args.width - 1)
    model[:, 1] = np.clip(model[:, 1], 0, args.height - 1)
    return original, model, query_ids


def strict_load_model(checkpoint, device):
    import torch
    from point4d.loader import build_model

    model = build_model()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_buffers = ("_resnet_mean", "_resnet_std")
    bad_missing = [key for key in missing if not key.endswith(allowed_buffers)]
    if bad_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={bad_missing}; unexpected={unexpected}")
    del state
    return model.eval().to(device), {"allowed_missing_buffers": missing, "unexpected_keys": unexpected}


def run(args):
    import numpy as np

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "predictions.npz").exists():
        raise FileExistsError("Output already contains predictions; choose a new run directory.")
    wall_start = time.perf_counter()
    rgb, records, source_meta = load_rgb(args)
    original_queries, model_queries, query_ids = make_queries(args, rgb.shape[1:3])
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if version != OFFICIAL_COMMIT:
        raise ValueError(f"Unexpected Point4D commit {version}; expected {OFFICIAL_COMMIT}.")
    metadata = {
        "status": "prepared", "script_sha256": sha256(__file__), "git_commit": version,
        "git_tracked_diff": subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=REPO, text=True),
        "command": sys.argv, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "frames": records, "source_metadata": source_meta, "input_shape": list(rgb.shape),
        "query_count": len(model_queries), "query_source_frame": 0,
        "query_initialization_scope": "Only points selected on the first input RGB frame; no later point births.",
        "input_information": "Selected RGB images and optional RGB-derived first-frame query mask/points only.",
        "coordinate_system": "Predicted frame-0 camera coordinates; scale is model-predicted, not GT-aligned or certified metric.",
        "camera_convention": "cam_R and cam_t map each predicted camera to the predicted global frame-0 coordinates.",
        "confidence_semantics": "Raw model confidence, not visibility, occlusion truth, or a calibrated probability.",
        "visibility_available": False, "resize_convention": "OpenCV linear resize with pixel-center coordinate mapping.",
        "resize_default_deviation": {"official_default_hw": [294, 518], "actual_hw": [args.height, args.width],
                                     "reason_if_changed": "Explicit experiment setting; common 4:3 input should retain approximately 4:3 aspect ratio."},
        "time_model": "Official model receives ordered images, not physical timestamps. Recorded timestamps must be used for downstream velocity evaluation.",
        "hostname": platform.node(), "python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for name, file in (("query_points", args.query_points), ("query_mask", args.query_mask)):
        if file:
            metadata[name + "_sha256"] = sha256(file)
    np.savez_compressed(args.output / "queries.npz", query_uv_original=original_queries, query_uv_model=model_queries, query_ids=query_ids)
    metadata["input_prepare_seconds"] = time.perf_counter() - wall_start
    write_json(args.output / "run.json", metadata)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "frames": len(rgb), "queries": len(model_queries), "output": str(args.output)}))
        return

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly to the allocated GPU before inference.")
    checksum_start = time.perf_counter()
    actual_hash = sha256(args.checkpoint)
    metadata["checkpoint_verification_seconds"] = time.perf_counter() - checksum_start
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint hash does not match the official frozen checkpoint.")
    sys.path.insert(0, str(REPO))
    import torch
    from point4d.inference import LongTrackConfig, track_long_video

    if not torch.cuda.is_available():
        raise RuntimeError("The allocated CUDA device is unavailable.")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda:0")
    cfg = LongTrackConfig(chunk_size=args.chunk_size, overlap=args.overlap, image_hw=(args.height, args.width),
                          query_chunk_size=args.query_chunk_size, geometry_stride=args.geometry_stride, bf16=not args.fp32)
    metadata.update(status="loading_model", configuration=dataclasses.asdict(cfg), checkpoint_sha256=actual_hash,
                    checkpoint_size=args.checkpoint.stat().st_size, gpu_name=torch.cuda.get_device_name(device),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    torch_version_deviation="Official requires ~=2.6.0; using 2.7.1+cu128 for Blackwell support.")
    write_json(args.output / "run.json", metadata)
    torch.cuda.reset_peak_memory_stats(device)
    load_start = time.perf_counter()
    model, state_check = strict_load_model(args.checkpoint, device)
    torch.cuda.synchronize(device)
    metadata["load_seconds"] = time.perf_counter() - load_start
    metadata["state_dict_check"] = state_check
    metadata["status"] = "running"
    write_json(args.output / "run.json", metadata)
    infer_start = time.perf_counter()
    with torch.inference_mode():
        trajectory, confidence, geometry = track_long_video(model, rgb, model_queries, cfg, device=device, return_geometry=True)
    torch.cuda.synchronize(device)
    metadata["inference_seconds"] = time.perf_counter() - infer_start
    arrays = {"trajectories": trajectory, "confidence": confidence,
              "query_uv_original": original_queries, "query_uv_model": model_queries,
              "query_ids": query_ids,
              "input_hw": np.array(rgb.shape[1:3]), "model_hw": np.array(cfg.image_hw),
              "frame_ids": np.asarray([frame["frame_id"] for frame in records]),
              "timestamps_seconds": np.asarray([frame["timestamp"] if frame["timestamp"] is not None else np.nan for frame in records], dtype=np.float64)}
    arrays.update({"geometry_" + key: np.asarray(value) for key, value in geometry.items()})
    # Derived projections are diagnostic, never independent evidence or GT.
    camera_points = np.einsum("tni,tij->tnj", trajectory - geometry["cam_t"][:, None, :], geometry["cam_R"])
    projected = np.einsum("tij,tnj->tni", geometry["intrinsics"], camera_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv_model = projected[..., :2] / projected[..., 2:3]
    arrays["projected_uv_model"] = uv_model
    resize_scale = np.array([args.width / rgb.shape[2], args.height / rgb.shape[1]], dtype=np.float32)
    arrays["projected_uv_original"] = (uv_model + 0.5) / resize_scale - 0.5
    finite = {key: bool(np.isfinite(value).all()) for key, value in arrays.items() if value.dtype.kind == "f"}
    save_start = time.perf_counter()
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    output_hash = sha256(args.output / "predictions.npz")
    required_finite = ("trajectories", "confidence", "geometry_points", "geometry_cam_R", "geometry_cam_t", "geometry_intrinsics")
    valid_core = all(finite[key] for key in required_finite)
    metadata.update(status="complete" if valid_core else "failed", array_shapes={key: list(value.shape) for key, value in arrays.items()},
                    all_finite=finite, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device), wall_seconds=time.perf_counter() - wall_start,
                    serialization_seconds=time.perf_counter() - save_start, predictions_sha256=output_hash)
    write_json(args.output / "run.json", metadata)
    if not valid_core:
        raise RuntimeError("Non-finite core tracker outputs; saved output must not be used as valid prior.")
    print(json.dumps({key: metadata[key] for key in ("status", "query_count", "inference_seconds", "peak_gpu_allocated_bytes", "predictions_sha256")}, indent=2))


if __name__ == "__main__":
    cli_args = parse_args()
    if (cli_args.output / "predictions.npz").exists():
        raise FileExistsError("Output already contains predictions; choose a new run directory.")
    try:
        run(cli_args)
    except Exception as error:
        cli_args.output.mkdir(parents=True, exist_ok=True)
        write_json(cli_args.output / "error.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        record_path = cli_args.output / "run.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            record.update(status="failed", error=str(error))
            write_json(record_path, record)
        raise
