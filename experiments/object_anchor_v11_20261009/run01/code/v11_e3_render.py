"""E3: zero-training re-render of frozen V10 P/PQ terminals at retained frames.

Protocol B swaps only the support-bank bone transforms for the published retained-frame
SMPL; P is also re-rendered under protocol A and compared with V10's saved renders to
prove that this read-only path reproduces V10. No ground-truth RGB is read here.
"""
import time

import numpy as np
import torch

from v11_common import RUN, V10, config, identity, read, save_json, scene_dir, v10_modules
from hoi_modules.pose_prior_adapter import PosePriorAdapter


def main():
    state, renderer = v10_modules()
    assert torch.cuda.get_device_name(0) == config()["hardware"]["name"]
    torch.set_num_threads(config()["hardware"]["CPU_threads"])
    out_root = RUN / "evaluation/E3"
    rows, checkpoints, start = [], [], time.monotonic()
    for scene in config()["scenes"]:
        adapters = {"A": PosePriorAdapter.from_cache(V10 / "scenes" / scene / "protocol/pose_cache.pt"),
                    "B": PosePriorAdapter.from_cache(scene_dir(scene) / "protocol/pose_query_cache_B.pt")}
        for arm in ("P", "PQ"):
            asset = read(V10 / "scenes" / scene / "runs" / arm / "run.json")["checkpoint"]
            assert identity(asset["path"])["sha256"] == asset["sha256"]
            checkpoints.append(dict(scene=scene, arm=arm, checkpoint=asset))
            base, support, optimizer, objs, meta = state.restore(torch.load(asset["path"], map_location="cpu", weights_only=False))
            render, bg = renderer.make_renderer(meta["scale_bound"]), renderer.rgb_background(objs[0])
            base._deformation.eval()
            support.eval()
            cams, frames = state.load_scene_cameras(scene, "retained", load_rgb=False)
            with torch.no_grad():
                for protocol in (("A", "B") if arm == "P" else ("B",)):
                    for cam, frame in zip(cams, frames):
                        bones = adapters[protocol].bone_transforms(frame["frame_id"], device="cuda")
                        pkg = render(cam, base, objs[3], bg, support=support, bones=bones, scene_extent=meta["scene_extent"])
                        rgb = pkg["render"].permute(1, 2, 0).cpu().numpy()
                        assert rgb.dtype == np.float32 and np.isfinite(rgb).all()
                        path = out_root / scene / arm / protocol / (frame["frame_id"] + ".npz")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, rgb=rgb)
                        row = dict(scene=scene, arm=arm, protocol=protocol, frame_id=frame["frame_id"], render=identity(path))
                        if protocol == "A":
                            saved = np.load(V10 / "evaluation" / scene / arm / "retained" / (frame["frame_id"] + ".npz"))["rgb"]
                            row["max_abs_diff_vs_v10_saved_render"] = float(np.abs(saved - rgb).max())
                        rows.append(row)
            del base, support, optimizer
            torch.cuda.empty_cache()
    save_json(out_root / "manifest.json", dict(status="completed", rows=rows, checkpoints=checkpoints, GT_read=False,
              seconds=time.monotonic() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    worst = max(r.get("max_abs_diff_vs_v10_saved_render", 0.0) for r in rows)
    print("E3 renders", len(rows), "max |A - V10 saved|", worst)


if __name__ == "__main__":
    main()
