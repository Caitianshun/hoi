"""E0b: SAM2 entity labels (0 background, 1 person, 2.. objects) from training-frame prompts.

Sequence 'train' contains training frames only and is the only source of training supervision.
Sequence 'full' adds retained frames; its retained-frame labels are used only for region scoring.
"""
import argparse
import sys
import time

import numpy as np
import torch
from PIL import Image

from v11_common import ROOT, RUN, config, identity, read, save_json


def scene_frames(scene, sequence):
    inputs = ROOT / config()["scenes"][scene]["input_dir"]
    frames = read(inputs / "manifest.json")["frames"]
    if sequence == "full":
        frames = frames + read(inputs / "evaluation_manifest.json")["frames"]
    return sorted(frames, key=lambda f: int(f["frame_id"]))


def fg_bbox(frame):
    mask = np.asarray(Image.open(frame["mask_path"]).convert("L")) >= 128
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def segment(scene, sequence, device="cuda"):
    prompts = read(RUN / "protocol/sam2_prompts.json")
    spec = prompts["scenes"][scene]
    frames = scene_frames(scene, sequence)
    index = {f["frame_id"]: i for i, f in enumerate(frames)}
    train_ids = {f["frame_id"] for f in scene_frames(scene, "train")}
    out = RUN / "scenes" / scene / "sam2" / sequence
    assert not (out / "segmentation_run.json").exists(), "use a fresh output"
    staging = out / "frames"
    staging.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.open(f["image_path"]).convert("RGB").save(staging / f"{i:06d}.jpg", quality=100, subsampling=0)
    width, height = Image.open(frames[0]["image_path"]).size
    record = dict(status="running", scene=scene, sequence=sequence, frames=[f["frame_id"] for f in frames],
                  prompts=spec, model={k: (identity(ROOT / v) if k == "checkpoint" else v) for k, v in prompts["model"].items()},
                  staging="JPEG quality 100, subsampling 0 (SAM2 folder loader only)",
                  role="estimated segmentation prior, not reference visibility or alpha",
                  retained_frames_prompted=False, start_unix=time.time())
    save_json(out / "segmentation_run.json", record)
    sys.path.insert(0, str(ROOT / "third_party/sam2"))
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(prompts["model"]["config"], str(ROOT / prompts["model"]["checkpoint"]),
                                           device=device, apply_postprocessing=True, vos_optimized=False)
    used_prompts = []
    with torch.inference_mode():
        state = predictor.init_state(str(staging), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for p in spec["prompts"]:
            assert p["frame_id"] in train_ids, "prompts only on training frames"
            frame = frames[index[p["frame_id"]]]
            kwargs = dict(inference_state=state, frame_idx=index[p["frame_id"]], obj_id=int(p["obj_id"]))
            box = p.get("box")
            if box == "fg_bbox":
                box = fg_bbox(frame)
            if box is not None:
                kwargs["box"] = np.asarray(box, dtype=np.float32)
            points = [*p.get("points", []), *p.get("negative", [])]
            if points:
                kwargs["points"] = np.asarray(points, dtype=np.float32)
                kwargs["labels"] = np.asarray([1] * len(p.get("points", [])) + [0] * len(p.get("negative", [])), dtype=np.int32)
            predictor.add_new_points_or_box(**kwargs)
            used_prompts.append(dict(p, resolved_box=box))
        first = min(index[p["frame_id"]] for p in spec["prompts"])
        scores = {}
        for direction in (False, True):
            for frame_idx, obj_ids, logits in predictor.propagate_in_video(state, reverse=direction):
                if direction and frame_idx >= first:
                    continue
                values = logits.detach().float().cpu().numpy().reshape(len(obj_ids), height, width)
                order = np.argsort(np.asarray(obj_ids))
                scores[frame_idx] = (np.asarray(obj_ids)[order], values[order])
    assert sorted(scores) == list(range(len(frames))), "propagation omitted frames"
    labels_dir = out / "labels"
    labels_dir.mkdir(exist_ok=True)
    areas = []
    for i, f in enumerate(frames):
        ids, values = scores[i]
        assert np.isfinite(values).all()
        label = np.where(values.max(0) > 0, ids[values.argmax(0)], 0).astype(np.uint8)
        Image.fromarray(label).save(labels_dir / f"{f['frame_id']}.png")
        areas.append({"frame_id": f["frame_id"], **{str(int(k)): int((label == k).sum()) for k in ids}})
    record.update(status="completed", resolved_prompts=used_prompts, areas=areas, end_unix=time.time(),
                  seconds=time.time() - record["start_unix"], torch=torch.__version__,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith("cuda") else None)
    save_json(out / "segmentation_run.json", record)
    for path in staging.glob("*.jpg"):
        path.unlink()
    staging.rmdir()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default="Backpack,Tennis")
    parser.add_argument("--sequences", default="train,full")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == config()["hardware"]["name"]
    for scene in args.scenes.split(","):
        for sequence in args.sequences.split(","):
            r = segment(scene, sequence, args.device)
            print(scene, sequence, len(r["frames"]), "frames", round(r["seconds"], 1), "s", flush=True)
