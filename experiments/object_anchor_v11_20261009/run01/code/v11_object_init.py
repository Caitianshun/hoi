"""CPU object initialisation from training frames only.

1. Contact states: per object and training frame, the distance from each projected SMPL hand
   joint (22 left, 23 right) to the SAM2 object mask decides world / left / right; a median
   filter and minimum segment length give state segments over the full frame axis.
2. Switch anchors: rigid hand frames at the switch frame (training poses).
3. Anchor-frame silhouette carving: voxels in each segment's anchor frame are kept when their
   projections fall inside the object mask far more often than on labelled background
   (person pixels are ignored as possible occluders); colours are mask-pixel means.
4. Keyframe initialisation for the world-trajectory arm, from training-frame poses only.
"""
import argparse
import json
import math

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, median_filter

from v11_common import ROOT, RUN, V10, config, identity, read, save_json, scene_dir
from hoi_modules.pose_prior_adapter import PosePriorAdapter

HAND = {"left": 22, "right": 23}


def rigid_anchor(query, joint):
    body = query["body_to_world"]
    scale = float(np.cbrt(np.linalg.det(body[:3, :3])))
    R = (body[:3, :3] / scale) @ query["global_pose"][joint][:3, :3]
    U, _, Vt = np.linalg.svd(R)
    T = np.eye(4)
    T[:3, :3] = U @ Vt
    T[:3, 3] = query["world_joints"][joint]
    return T


def project(points, frame):
    w2c, K = np.asarray(frame["w2c"], float), np.asarray(frame["K"], float)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    uvw = cam @ K.T
    return uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-9), cam[:, 2]


def segments_from_states(states, frame_numbers, n_frames, min_len):
    """states: per training frame label; returns segments tiling 0..n_frames-1 (switches at training frames)."""
    labels = list(states)
    changed = True
    while changed:  # absorb short runs into the longer neighbour
        changed = False
        runs, start = [], 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                runs.append((start, i - 1, labels[start]))
                start = i
        for r, (a, b, lab) in enumerate(runs):
            if b - a + 1 < min_len and len(runs) > 1:
                left = runs[r - 1] if r > 0 else None
                right = runs[r + 1] if r + 1 < len(runs) else None
                neighbour = max([x for x in (left, right) if x], key=lambda x: x[1] - x[0])
                for i in range(a, b + 1):
                    labels[i] = neighbour[2]
                changed = True
                break
    segments, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segments.append(dict(anchor=labels[start], first_train_index=start))
            start = i
    out = []
    for k, s in enumerate(segments):
        begin = 0 if k == 0 else int(frame_numbers[s["first_train_index"]])
        end = n_frames - 1 if k == len(segments) - 1 else int(frame_numbers[segments[k + 1]["first_train_index"]]) - 1
        out.append(dict(start=begin, end=end, anchor=s["anchor"]))
    return out, labels


def carve(points_anchor, anchors_per_frame, frames, labels, obj_id, images, min_hits=3, ratio=0.7):
    n_obj = np.zeros(len(points_anchor))
    n_bg = np.zeros(len(points_anchor))
    colour = np.zeros((len(points_anchor), 3))
    disk = lambda r: (np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2) <= r * r)
    for A, frame, label, rgb in zip(anchors_per_frame, frames, labels, images):
        obj = label == obj_id
        if obj.sum() < 50:
            continue
        inside = binary_dilation(obj, structure=disk(3))
        far_bg = (label == 0) & ~binary_dilation(obj, structure=disk(7))
        world = points_anchor @ A[:3, :3].T + A[:3, 3]
        uv, z = project(world, frame)
        u, v = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
        ok = (z > 0) & (u >= 0) & (v >= 0) & (u < label.shape[1]) & (v < label.shape[0])
        idx = np.flatnonzero(ok)
        hit = inside[v[idx], u[idx]]
        n_obj[idx[hit]] += 1
        n_bg[idx[far_bg[v[idx], u[idx]]]] += 1
        colour[idx[hit]] += rgb[v[idx[hit]], u[idx[hit]]]
    keep = (n_obj >= min_hits) & (n_obj >= ratio * (n_obj + n_bg))
    return keep, colour / np.maximum(n_obj, 1)[:, None], n_obj, n_bg


def build(scene, grid=56, half_size_body=0.45, contact_ratio=0.35, min_len=6, max_seeds=20000, seed=20261009):
    cfg = config()["scenes"][scene]
    prompts = read(RUN / "protocol/sam2_prompts.json")["scenes"][scene]
    objects = {int(k): v for k, v in prompts["entities"].items() if int(k) != 1}
    frames = sorted(read(ROOT / cfg["input_dir"] / "manifest.json")["frames"], key=lambda f: int(f["frame_id"]))
    n_frames = cfg["training_frames"] + cfg["retained_frames"]
    numbers = np.asarray([int(f["frame_id"]) for f in frames])
    adapter = PosePriorAdapter(torch.load(V10 / "scenes" / scene / "protocol/pose_cache.pt", map_location="cpu", weights_only=False))
    queries = [adapter.query(f["frame_id"]) for f in frames]
    anchors = {h: np.stack([rigid_anchor(q, j) for q in queries]) for h, j in HAND.items()}
    sam = scene_dir(scene) / "sam2/train"
    assert read(sam / "segmentation_run.json")["status"] == "completed"
    labels = [np.asarray(Image.open(sam / "labels" / f"{f['frame_id']}.png")) for f in frames]
    images = [np.asarray(Image.open(f["image_path"]).convert("RGB"), dtype=np.float64) / 255 for f in frames]
    joints = np.stack([q["world_joints"] for q in queries])
    body_height = float(np.median(np.linalg.norm(joints[:, 15] - 0.5 * (joints[:, 7] + joints[:, 8]), axis=1))) * 1.15
    half = half_size_body * body_height
    rng = np.random.default_rng(seed)
    result = dict(scene=scene, body_height_world=body_height, carve_half_size=half, grid=grid, objects={})
    seeds_xyz, seeds_rgb, seeds_scale, seeds_obj = [], [], [], []
    per_object = {}
    names = []
    for oi, (obj_id, name) in enumerate(sorted(objects.items())):
        names.append(name)
        # ---- contact states on training frames
        dist = {h: np.full(len(frames), np.inf) for h in HAND}
        visible = np.zeros(len(frames), bool)
        for i, (f, label) in enumerate(zip(frames, labels)):
            mask = label == obj_id
            if mask.sum() < 50:
                continue
            visible[i] = True
            ys, xs = np.nonzero(mask)
            diameter = 2 * math.sqrt(mask.sum() / math.pi)
            for h in HAND:
                uv, _ = project(joints[i, HAND[h]][None], f)
                d = np.sqrt(((xs - uv[0, 0]) ** 2 + (ys - uv[0, 1]) ** 2).min())
                dist[h][i] = d / diameter
        raw = []
        for i in range(len(frames)):
            best = min(HAND, key=lambda h: dist[h][i])
            raw.append(best if visible[i] and dist[best][i] <= contact_ratio else ("world" if visible[i] else None))
        known = [i for i, s in enumerate(raw) if s is not None]
        filled = [raw[min(known, key=lambda k: abs(k - i))] for i in range(len(frames))]
        code = {"world": 0, "left": 1, "right": 2}
        inverse = {v: k for k, v in code.items()}
        smoothed = median_filter(np.asarray([code[s] for s in filled]), size=9, mode="nearest")
        segs, final_labels = segments_from_states([inverse[int(c)] for c in smoothed], numbers, n_frames, min_len)
        # ---- switch anchors and chain (S1 = I, Delta = I at initialisation)
        def anchor_at(name_, index):
            return np.eye(4) if name_ == "world" else anchors[name_][index]
        switch_pairs, chain = [], [np.eye(4)]
        for k in range(len(segs) - 1):
            tau = segs[k + 1]["start"]
            index = int(np.flatnonzero(numbers == tau)[0])
            before, after = anchor_at(segs[k]["anchor"], index), anchor_at(segs[k + 1]["anchor"], index)
            switch_pairs.append(np.stack([before, after]))
            chain.append(np.linalg.inv(after) @ before @ chain[-1])
        # ---- carve in every segment's anchor frame, map to canonical with the initial chain
        object_points, object_colours, carve_log = [], [], []
        for k, s in enumerate(segs):
            idx = [i for i, num in enumerate(numbers) if s["start"] <= num <= s["end"] and visible[i]]
            if len(idx) < 5:
                carve_log.append(dict(segment=k, frames=len(idx), kept=0, note="too few visible frames"))
                continue
            if s["anchor"] == "world":
                boundary = s["start"] if k > 0 else segs[k + 1]["start"] if len(segs) > 1 else int(numbers[idx[0]])
                bi = int(np.argmin(np.abs(numbers - boundary)))
                hands = [h for h in HAND if any(x["anchor"] == h for x in segs)] or list(HAND)
                center = np.mean([joints[bi, HAND[h]] for h in hands], axis=0)
                A_frames = [np.eye(4) for _ in idx]
                axes = np.linspace(-half, half, grid)
                local = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), -1).reshape(-1, 3) + center
            else:
                A_frames = [anchors[s["anchor"]][i] for i in idx]
                axes = np.linspace(-half, half, grid)
                local = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), -1).reshape(-1, 3)
            keep, colour, n_obj, n_bg = carve(local, A_frames, [frames[i] for i in idx], [labels[i] for i in idx], obj_id, [images[i] for i in idx])
            pts = local[keep]
            canon = (np.linalg.inv(chain[k])[:3, :3] @ pts.T).T + np.linalg.inv(chain[k])[:3, 3]
            object_points.append(canon)
            object_colours.append(colour[keep])
            carve_log.append(dict(segment=k, anchor=s["anchor"], frames=len(idx), kept=int(keep.sum())))
        if not object_points or sum(len(p) for p in object_points) < 100:
            raise RuntimeError(f"{scene}/{name}: carving produced too few seeds {carve_log}")
        pts, cols = np.concatenate(object_points), np.concatenate(object_colours)
        if len(pts) > max_seeds:
            choice = rng.choice(len(pts), max_seeds, replace=False)
            pts, cols = pts[choice], cols[choice]
        voxel = 2 * half / (grid - 1)
        pts = pts + rng.uniform(-0.5, 0.5, pts.shape) * voxel
        seeds_xyz.append(pts)
        seeds_rgb.append(cols)
        seeds_scale.append(np.full(len(pts), 0.7 * voxel))
        seeds_obj.append(np.full(len(pts), oi))
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        # ---- keyframe initialisation for the world-trajectory arm (training poses only)
        step = 4
        keyframes = list(range(0, n_frames, step))
        if keyframes[-1] != n_frames - 1:
            keyframes.append(keyframes[-1] + step)
        kf = []
        for fnum in keyframes:
            i = int(np.argmin(np.abs(numbers - min(fnum, n_frames - 1))))
            k = next(j for j, s in enumerate(segs) if s["start"] <= numbers[i] <= s["end"])
            kf.append(anchor_at(segs[k]["anchor"], i) @ chain[k])
        release = read(RUN / "protocol/sam2_prompts.json")["scenes"][scene]["states_from_release_transitions"]
        per_object[name] = dict(segments=segs, S1=np.eye(4).tolist(), switch_anchors=np.asarray(switch_pairs).reshape(-1, 2, 4, 4).tolist(),
                                keyframe_init=np.asarray(kf).tolist(), keyframe_frames=keyframes)
        result["objects"][name] = dict(obj_id=obj_id, segments=segs, visible_training_frames=int(visible.sum()),
                                       carve=carve_log, seeds=int(len(pts)), extent=extent, release_note=release,
                                       state_by_training_frame=dict(zip([f["frame_id"] for f in frames], final_labels)),
                                       hand_distance_ratio={h: [None if not np.isfinite(x) else round(float(x), 4) for x in dist[h]] for h in HAND})
    motion = dict(objects=names, extent={n: result["objects"][n]["extent"] for n in names}, per_object=per_object,
                  residual_knot_spacing=8, keyframe_step=4, anchor_joints=HAND, frames=n_frames)
    out = scene_dir(scene) / "objects"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(xyz=np.concatenate(seeds_xyz).astype(np.float32), rgb=np.concatenate(seeds_rgb).astype(np.float32),
                    scale=np.concatenate(seeds_scale).astype(np.float32), object_id=np.concatenate(seeds_obj).astype(np.int64)),
               out / "seeds.pt")
    (out / "motion.json").write_text(json.dumps(motion))
    result.update(seeds=identity(out / "seeds.pt"), motion=identity(out / "motion.json"), sam2_labels=identity(sam / "segmentation_run.json"),
                  training_frames_only=True)
    save_json(out / "init_report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="Backpack,Tennis")
    for s in p.parse_args().scenes.split(","):
        r = build(s)
        for name, o in r["objects"].items():
            print(s, name, "segments", [(x["start"], x["end"], x["anchor"]) for x in o["segments"]],
                  "seeds", o["seeds"], "carve", [(c["segment"], c.get("kept")) for c in o["carve"]])
