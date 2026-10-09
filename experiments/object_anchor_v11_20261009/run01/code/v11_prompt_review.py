"""Draw review sheets for choosing SAM2 prompts on training frames only.

Each sheet shows the training RGB with an original-pixel grid, the published combined
foreground outline (red) and projected training-pose wrists (cyan left, magenta right).
"""
import argparse

import cv2
import numpy as np
import torch

from v11_common import ROOT, RUN, V10, config, read
from hoi_modules.pose_prior_adapter import PosePriorAdapter


def sheet(scene, frame_id, width=960):
    manifest = read(ROOT / config()["scenes"][scene]["input_dir"] / "manifest.json")
    frames = {f["frame_id"]: f for f in manifest["frames"]}
    assert frame_id in frames, f"{frame_id} is not a training frame"
    f = frames[frame_id]
    rgb = cv2.imread(f["image_path"])
    mask = cv2.imread(f["mask_path"], 0) >= 128
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, contours, -1, (0, 0, 255), 2)
    h, w = rgb.shape[:2]
    for x in range(0, w, 100):
        cv2.line(rgb, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(rgb, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    for y in range(0, h, 100):
        cv2.line(rgb, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(rgb, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    adapter = PosePriorAdapter(torch.load(V10 / "scenes" / scene / "protocol/pose_cache.pt", map_location="cpu", weights_only=False))
    joints = adapter.query(frame_id)["world_joints"]
    w2c, K = np.asarray(f["w2c"]), np.asarray(f["K"])
    cam = joints @ w2c[:3, :3].T + w2c[:3, 3]
    pix = (cam @ K.T)[:, :2] / (cam @ K.T)[:, 2:3]
    for j, colour in ((20, (255, 255, 0)), (21, (255, 0, 255)), (0, (0, 255, 0))):
        u, v = pix[j]
        cv2.circle(rgb, (int(round(u)), int(round(v))), 7, colour, 2)
    cv2.putText(rgb, f"{scene} {frame_id}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    scale = width / w
    out = RUN / "protocol/prompt_review" / f"{scene}_{frame_id}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(rgb, (width, round(h * scale)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out, {int(j): [float(pix[j][0]), float(pix[j][1])] for j in (0, 20, 21)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--frames", required=True)
    a = p.parse_args()
    for fid in a.frames.split(","):
        path, wrists = sheet(a.scene, fid)
        print(path.name, {k: [round(x) for x in v] for k, v in wrists.items()})
