#!/usr/bin/env python3
"""SAM2 video masks from manual first-frame RGB boxes; these are estimated priors.

No package installation or automatic device selection. Run --validate-only on CPU
to validate inputs without importing torch/SAM2 or loading model weights.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""): h.update(block)
    return h.hexdigest()


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def validate_inputs(frame_dir, boxes_path):
    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")], key=natural_key)
    if not frames: raise ValueError("No RGB frames found")
    prompt = json.loads(boxes_path.read_text())
    if prompt.get("source") != "manual_first_frame_rgb":
        raise ValueError("boxes source must be manual_first_frame_rgb; no reference geometry prompts")
    if prompt.get("frame_index", 0) != 0: raise ValueError("This adapter supports first-frame boxes only")
    with Image.open(frames[0]) as image: size = image.size
    if list(size) != prompt.get("image_size"):
        raise ValueError("image_size must be [width,height] in the supplied processed RGB frame coordinates")
    if prompt.get("first_frame_sha256") != digest(frames[0]):
        raise ValueError("first_frame_sha256 must match the actual first RGB file used to draw boxes")
    if set(prompt.get("boxes", {})) != {"person", "object"}:
        raise ValueError("boxes must contain exactly person and object [x0,y0,x1,y1]")
    for entity, box in prompt["boxes"].items():
        x = np.asarray(box, dtype=float)
        if x.shape != (4,) or not np.isfinite(x).all(): raise ValueError(f"invalid {entity} box")
        if not (0 <= x[0] < x[2] <= size[0] and 0 <= x[1] < x[3] <= size[1]):
            raise ValueError(f"{entity} box outside first frame")
    for path in frames:
        with Image.open(path) as image:
            if image.size != size: raise ValueError(f"frame dimensions differ: {path}")
    return frames, prompt, size


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", required=True, type=Path)
    p.add_argument("--boxes", required=True, type=Path)
    p.add_argument("--sam2-root", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", required=True, help="explicit cpu or cuda:0; caller owns GPU scheduling")
    p.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args(argv)
    frames, prompt, (width, height) = validate_inputs(args.frames, args.boxes)
    root = args.sam2_root.resolve()
    config_path = root / "sam2" / args.model_config
    if not config_path.is_file(): raise ValueError(f"model config missing: {config_path}")
    if args.validate_only:
        print(json.dumps({"status": "CPU input validation passed", "frames": len(frames),
                          "image_size": [width, height], "checkpoint_exists": args.checkpoint.is_file()}))
        return 0
    if not args.checkpoint.is_file(): raise ValueError("checkpoint missing; no automatic downloads")
    if args.output.exists() and any(args.output.iterdir()): raise ValueError("use a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    # SAM2's folder loader accepts numeric JPEG filenames. Preserve source RGB and
    # stage only SAM2 inputs; PNG conversion is explicit and recorded.
    staging = args.output / "sam2_frames"; staging.mkdir()
    manifest = []
    for i, path in enumerate(frames):
        target = staging / f"{i:06d}.jpg"
        if path.suffix.lower() in (".jpg", ".jpeg"):
            target.symlink_to(path.resolve()); conversion = "symlink_original_jpeg"
        else:
            with Image.open(path) as image: image.convert("RGB").save(target, quality=100, subsampling=0)
            conversion = "SAM2_only_JPEG_quality100_subsampling0"
        manifest.append({"index": i, "source": str(path.resolve()), "sha256": digest(path),
                         "staged_sha256": digest(target), "conversion": conversion})
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()
    source_files = [root / "sam2" / x for x in ("build_sam.py", "sam2_video_predictor.py", "utils/misc.py")]
    record = {"schema_version": 1, "status": "running", "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "sam2_commit": commit, "source_sha256": {str(x): digest(x) for x in source_files},
              "adapter_sha256": digest(__file__), "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": digest(args.checkpoint), "config_sha256": digest(config_path),
              "config": args.model_config, "device": args.device, "precision": args.precision,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "prompts": prompt,
              "frames": manifest, "entity_ids": {"background": 0, "person": 1, "object": 2},
              "mask_role": "estimated_segmentation_prior_not_reference_visibility",
              "overlap_rule": "largest positive logit; equal logits resolved by lower entity ID",
              "visibility_note": "Empty object masks permitted; predicted presence/absence is not ground-truth visibility",
              "reference_geometry_used": False, "optimized_predictor": False,
              "timing_note": "load_seconds includes model/state loading; inference_seconds excludes model load and final file writes"}
    record_path = args.output / "segmentation_run.json"
    def save_record(): record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    save_record(); total_start = time.perf_counter()
    try:
        sys.path.insert(0, str(root))
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("requested CUDA unavailable")
        if args.precision == "bfloat16" and device.type != "cuda": raise ValueError("bfloat16 mode is CUDA-only here")
        def sync():
            if device.type == "cuda": torch.cuda.synchronize(device)
        sync(); start = time.perf_counter()
        predictor = build_sam2_video_predictor(args.model_config, str(args.checkpoint.resolve()), device=args.device,
                                              apply_postprocessing=True, vos_optimized=False)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bfloat16" else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            state = predictor.init_state(str(staging.resolve()), offload_video_to_cpu=True, offload_state_to_cpu=True)
            sync(); record["load_seconds"] = time.perf_counter() - start; start = time.perf_counter()
            for name, obj_id in [("person", 1), ("object", 2)]:
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id,
                                                box=np.asarray(prompt["boxes"][name], dtype=np.float32))
            labels = np.zeros((len(frames), height, width), dtype=np.uint8)
            seen = np.zeros(len(frames), bool); areas = []
            for frame_idx, obj_ids, logits in predictor.propagate_in_video(state):
                scores = logits.detach().float().cpu().numpy().reshape(len(obj_ids), height, width)
                ordered = sorted(range(len(obj_ids)), key=lambda i: obj_ids[i])
                scores = scores[ordered]; ids = np.asarray(obj_ids)[ordered]
                if not np.isfinite(scores).all(): raise ValueError("SAM2 produced nonfinite logits")
                winner = scores.argmax(axis=0); positive = scores.max(axis=0) > 0
                labels[frame_idx] = np.where(positive, ids[winner], 0).astype(np.uint8)
                seen[frame_idx] = True
                areas.append({"frame": frame_idx, "person_pixels": int((labels[frame_idx] == 1).sum()),
                              "object_pixels": int((labels[frame_idx] == 2).sum())})
            sync(); record["inference_seconds"] = time.perf_counter() - start
        if not seen.all(): raise RuntimeError(f"propagation omitted frames: {np.flatnonzero(~seen).tolist()}")
        foreground = labels != 0
        np.savez_compressed(args.output / "segmentation.npz", foreground=foreground, entity_labels=labels,
                            source_frame_names=np.array([x.name for x in frames]),
                            role=np.array("estimated_segmentation_prior_not_reference_visibility"))
        for subdir in ("foreground", "entity_labels"): (args.output / subdir).mkdir()
        for i, path in enumerate(frames):
            Image.fromarray(foreground[i].astype(np.uint8)*255).save(args.output / "foreground" / f"{path.stem}.png")
            Image.fromarray(labels[i]).save(args.output / "entity_labels" / f"{path.stem}.png")
        record.update(status="completed", area_by_frame=areas, torch_version=torch.__version__,
                      output_shape=list(labels.shape), inference_fps=len(frames)/record["inference_seconds"])
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(total_model_and_output_seconds=time.perf_counter()-total_start,
                      end_utc=dt.datetime.now(dt.timezone.utc).isoformat()); save_record()
    print(str(record_path.resolve()))
    return 0


if __name__ == "__main__": raise SystemExit(main())
