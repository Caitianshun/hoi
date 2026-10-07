"""V10 CPU evaluation using the unchanged V9 pixel and region protocol.

Every new render is scored once after the common six-terminal freeze. Historical
U/Q/Q0 rows are explicitly reused, never represented as a new model run. The
independent verifier recomputes PSNR from raw arrays and checks all aggregates;
it does not claim a second SSIM/LPIPS implementation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

RUN = Path(__file__).resolve().parents[1]; ROOT = RUN.parents[2]
METRICS = ["psnr_db", "ssim", "lpips_spatial_mean"]
REGIONS = ["full", "foreground", "background"]
ARMS = ["U", "Q", "Q0", "PARENT", "C", "P", "PQ"]
NEW_ARMS = ["PARENT", "C", "P", "PQ"]
SPLITS = ["train", "retained", "retained_without_00000"]
COMPARISONS = [("P", "C"), ("PQ", "P"), ("PQ", "C")] + [(a, b) for a in ["C", "P", "PQ"] for b in ["PARENT", "U", "Q", "Q0"]]


def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def identity(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""): h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def checked(asset):
    value = identity(asset["path"]); assert value["sha256"] == asset["sha256"], asset
    return Path(value["path"])


def config(): return read(RUN / "configs/v10.json")


def csvwrite(path, rows):
    assert rows; path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def select(rows, split):
    return [r for r in rows if r["split"] == ("train" if split == "train" else "retained") and (split != "retained_without_00000" or r["frame_id"] != "00000")]


def mean_summary(rows):
    import numpy as np
    valid = [r for r in rows if r["pixels"] > 0 and r["sse_rgb_mean"] is not None]
    pixels = sum(r["pixels"] for r in valid); sse = sum(r["sse_rgb_mean"] for r in valid)
    mse = sse / pixels if pixels else None
    return dict(frames=len(rows), valid_frames=len(valid), pixels=pixels,
        sse_rgb_mean=sse if valid else None, pooled_mse=mse,
        pooled_psnr_db=float(-10*np.log10(max(mse, 1e-12))) if mse is not None else None,
        raw_pooled_mse=sum(r["raw_mse"]*r["pixels"] for r in valid)/pixels if pixels else None,
        raw_psnr_db=float(np.mean([r["raw_psnr_db"] for r in valid])) if valid else None,
        **{k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in METRICS})


def paired_summary(diffs):
    import numpy as np
    result = {}; tol = config()["evaluation"]["ties"]
    for arm, ref in COMPARISONS:
        key = arm + "-" + ref; result[key] = {}
        for split in SPLITS:
            result[key][split] = {}
            for region in REGIONS:
                rows = [r for r in select(diffs, split) if r["comparison"] == key and r["region"] == region]
                stats = {}
                for metric in METRICS:
                    vals = np.array([r[metric] for r in rows if r[metric] is not None])
                    oriented = vals*(-1 if metric == "lpips_spatial_mean" else 1)
                    stats[metric] = dict(mean=float(vals.mean()) if len(vals) else None,
                        median=float(np.median(vals)) if len(vals) else None,
                        win=int((oriented>tol).sum()), tie=int((abs(oriented)<=tol).sum()),
                        loss=int((oriented < -tol).sum()), frames=len(vals))
                result[key][split][region] = stats
    return result


def metric_functions():
    # These imported functions have already defined the V8/V9 evaluator. They
    # retain its full-image receptive fields, SSIM edges and region denominators.
    paths = [ROOT / "experiments" / name / "run01/code" for name in [
        "numerical_stability_calibration_20260927", "foreground_stage_calibration_20260927",
        "baseline_protocol_calibration_20260927", "aux_ref_object_reconstruction_20260924"]]
    for path in reversed(paths): sys.path.insert(0, str(path))
    import evaluate_and_report as helper
    from evaluate_frozen import evaluate_image
    return helper.lpips_cpu, helper.figure, evaluate_image


def numeric_csv(path):
    integers = {"pixels", "source_frame_index"}
    numbers = integers | {"pixel_fraction", "mse", "sse_rgb_mean", "psnr_db", "ssim", "lpips_spatial_mean", "full_error_share", "raw_mse", "raw_psnr_db", "time"}
    rows = []
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            for key in numbers:
                if key in row: row[key] = None if row[key] == "" else int(row[key]) if key in integers else float(row[key])
            rows.append(row)
    return rows


def historical_rows(scene):
    path = ROOT / config()["scenes"][scene]["v9_dir"] / "metrics_per_frame.csv"
    source = identity(path)
    rows = [r for r in numeric_csv(path) if r["run"] in ["U", "Q", "Q0"]]
    for row in rows:
        row.update(scene=scene, historical=True, historical_metrics_reused=True,
            metric_source_path=str(path), metric_source_sha256=source["sha256"])
    return rows


def freeze_gate():
    path = RUN / "protocol/terminal_freeze.json"; freeze = read(path)
    assert freeze["status"] == "all_six_terminals_frozen" and len(freeze["runs"]) == 6
    assert {(r["scene"], r.get("arm", r.get("mode"))) for r in freeze["runs"]} == {(s, a) for s in config()["scenes"] for a in config()["arms"]}
    for asset in freeze["assets"]: checked(asset)
    return identity(path)


def render_index(scene, arm, freeze):
    path = RUN / "evaluation" / scene / arm / "manifest.json"; manifest = read(path)
    assert manifest["status"] == "completed"
    reference = manifest.get("freeze", manifest.get("all_finals_freeze"))
    assert reference["sha256"] == freeze["sha256"]
    assert identity(checked(reference))["sha256"] == freeze["sha256"]
    state_asset = manifest["checkpoint"]; checked(state_asset)
    frozen_assets = read(freeze["path"])["assets"]
    assert any(a["sha256"] == state_asset["sha256"] and Path(a["path"]).resolve() == Path(state_asset["path"]).resolve() for a in frozen_assets), "Rendered checkpoint was not frozen"
    assert manifest.get("optimization_updates", 0) == 0
    rows = manifest.get("rows", manifest.get("frames")); index = {(r["split"], r["frame_id"]):r for r in rows}
    cfg = config()["scenes"][scene]
    assert len(index) == len(rows) == cfg["training_frames"] + cfg["retained_frames"]
    for split, count in [("train", cfg["training_frames"]), ("retained", cfg["retained_frames"])]:
        assert sum(key[0] == split for key in index) == count
    return index, identity(path)


def evaluate_scene(scene, net, evaluator, figure, evaluate_image, freeze):
    import cv2
    import numpy as np
    start = time.monotonic(); cfg = config()["scenes"][scene]
    rows = historical_rows(scene); historical = ROOT / cfg["historical_dir"]
    examples = read(historical / "protocol/fixed_examples.json")
    if scene == "Backpack": crops = {r["frame_id"]:r["crop_bounds_xyxy"] for r in examples["hos_crops"]}
    else: crops = {r["frame_id"]:r["bounds_xyxy"] for r in read(historical / "figure_manifest.json")["crops"]}
    indices = {}; manifests = []
    for arm in NEW_ARMS:
        indices[arm], asset = render_index(scene, arm, freeze); manifests.append(asset)
    figures = []; windows = []; assets = []
    for split, name in [("retained", "evaluation_manifest.json"), ("train", "manifest.json")]:
        frames = read(ROOT / cfg["input_dir"] / name)["frames"]
        for i, frame in enumerate(frames):
            fid = frame["frame_id"]
            gt = cv2.imread(frame["image_path"])[..., ::-1].astype(np.float64)/255
            labels = (cv2.imread(frame["mask_path"], 0) >= config()["mask_threshold"]).astype(np.uint8)
            predictions = {}
            for arm in NEW_ARMS:
                entry = indices[arm][split, fid]; asset = entry["render"]
                assert entry.get("source_frame", frame) == frame, (scene, arm, fid, "Changed frame identity")
                with np.load(checked(asset), allow_pickle=False) as data: pred = data["rgb"]
                predictions[arm] = pred; result, raw_info = evaluate_image(pred, gt, labels, net)
                assets.append(dict(scene=scene, run=arm, split=split, frame_id=fid, **asset))
                for region in REGIONS:
                    mask = np.ones(labels.shape, bool) if region == "full" else labels == 1 if region == "foreground" else labels == 0
                    raw_mse = float(np.square(pred[mask]-gt[mask]).mean()) if mask.any() else None
                    rows.append(dict(scene=scene, run=arm, status="completed", historical=False,
                        historical_metrics_reused=False, split=split, frame_id=fid, region=region,
                        source_frame_index=int(fid), time=frame["time"], extrapolation=frame.get("extrapolation", fid == "00000"),
                        **result[region], raw_mse=raw_mse,
                        raw_psnr_db=None if raw_mse is None else float(-10*np.log10(max(raw_mse, 1e-12))),
                        raw_render_path=asset["path"], raw_render_sha256=asset["sha256"], evaluator_id=evaluator["id"],
                        raw_min=raw_info["min"], raw_max=raw_info["max"], clipped_channel_fraction=raw_info["clipped_channel_fraction"]))
            if split == "retained" or fid in examples["training_frame_ids"]:
                arrays = [gt] + [predictions[a] for a in NEW_ARMS]
                names = [fid + " GT", "parent", "C", "P", "PQ"]
                path = RUN / "scenes" / scene / "evaluation/figures" / (split + "_" + fid + "_full.jpg")
                figures.append(dict(**figure(path, arrays, names, 480), scene=scene,
                    split=split, frame_id=fid, kind="full", branch_order=["GT", *NEW_ARMS], display="RGB clipped [0,1]; uniformly resized"))
                if split == "retained":
                    x0, y0, x1, y1 = crops[fid]
                    crop_path = path.with_name(path.stem.replace("_full", "_foreground") + ".jpg")
                    figures.append(dict(**figure(crop_path, [x[y0:y1,x0:x1] for x in arrays], names, 320),
                        scene=scene, split=split, frame_id=fid, kind="foreground_crop", bounds_xyxy=crops[fid], branch_order=["GT", *NEW_ARMS]))
                    if i < 4:
                        for region, (cx, cy) in [("foreground", ((x0+x1)//2, (y0+y1)//2)), ("background", (40,40))]:
                            height, width = labels.shape; xa=max(0,min(width-80,cx-40)); ya=max(0,min(height-80,cy-40))
                            sl = np.s_[ya:ya+80, xa:xa+80]
                            window = RUN / "scenes" / scene / "feedback_arrays" / (fid + "_" + region + ".npz")
                            window.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(window, GT=gt[sl].astype(np.float32), mask=labels[sl],
                                **{a+"_raw":predictions[a][sl] for a in NEW_ARMS})
                            windows.append(dict(**identity(window), scene=scene, frame_id=fid, region=region, bounds_xyxy=[xa,ya,xa+80,ya+80]))
            if (i+1) % 32 == 0: print(scene, split, i+1, flush=True)
    assert len(rows) == len(ARMS)*(cfg["training_frames"]+cfg["retained_frames"])*len(REGIONS)
    summaries = {a:{s:{reg:mean_summary([r for r in select(rows,s) if r["run"]==a and r["region"]==reg]) for reg in REGIONS} for s in SPLITS} for a in ARMS}
    idx={(r["run"],r["split"],r["frame_id"],r["region"]):r for r in rows}; assert len(idx)==len(rows)
    diffs=[]
    for arm,ref in COMPARISONS:
        for r in rows:
            if r["run"] != arm: continue
            other=idx[ref,r["split"],r["frame_id"],r["region"]]; assert r["pixels"]==other["pixels"]
            diffs.append(dict(scene=scene,comparison=arm+"-"+ref,split=r["split"],frame_id=r["frame_id"],region=r["region"],
                **{k:r[k]-other[k] if r[k] is not None and other[k] is not None else None for k in METRICS}))
    blocks=[]; ids=sorted({r["frame_id"] for r in rows if r["split"]=="retained"})
    for arm,ref in COMPARISONS:
        for block in range(4):
            for region in REGIONS:
                selected=[r for r in diffs if r["comparison"]==arm+"-"+ref and r["split"]=="retained" and r["region"]==region and r["frame_id"] in ids[4*block:4*block+4]]
                blocks.append(dict(scene=scene,comparison=arm+"-"+ref,block=block,region=region,
                    frame_ids=ids[4*block:4*block+4],**{k:float(np.mean([r[k] for r in selected if r[k] is not None])) if any(r[k] is not None for r in selected) else None for k in METRICS}))
    output=RUN/"scenes"/scene
    csvwrite(output/"metrics_per_frame.csv",rows); csvwrite(output/"paired_differences.csv",diffs)
    result=dict(status="completed",summaries=summaries,paired=paired_summary(diffs),time_blocks=blocks,
        rows=len(rows),paired_rows=len(diffs),figures=figures,CPU_seconds=time.monotonic()-start,evaluator=evaluator,
        individual_human_object_metrics=None,NA_reason="No reliable separate human/object masks; image metrics do not verify contact or geometry")
    save(output/"evaluation_summary.json",result);save(output/"figure_manifest.json",dict(figures=figures,crops=crops,fixed_example_source=identity(historical/"protocol/fixed_examples.json")))
    save(output/"feedback_arrays/manifest.json",dict(windows=windows));save(output/"raw_render_index.json",dict(assets=assets,manifests=manifests))
    return result, rows, diffs


def run():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CPU metric process must hide CUDA"
    import numpy as np
    import torch
    import cv2
    started=time.monotonic(); torch.set_num_threads(config()["hardware"]["CPU_threads"]);cv2.setNumThreads(config()["hardware"]["CPU_threads"])
    freeze=freeze_gate(); lpips_cpu,figure,evaluate_image=metric_functions();net,info=lpips_cpu()
    assert not next(net.parameters()).is_cuda
    save(RUN/"protocol/evaluation_lpips.json",info)
    evaluator=read(ROOT/"experiments/temporal_evidence_v8_20260928/run01/protocol/evaluator.json")
    save(RUN/"protocol/evaluator.json",dict(**evaluator,V10_metric_source=identity(ROOT/"experiments/baseline_protocol_calibration_20260927/run01/code/evaluate_frozen.py")))
    scenes={};rows=[];diffs=[]
    for scene in config()["scenes"]:
        scenes[scene],rr,dd=evaluate_scene(scene,net,evaluator,figure,evaluate_image,freeze);rows.extend(rr);diffs.extend(dd)
    equal={a:{s:{reg:{k:float(np.mean([v["summaries"][a][s][reg][k] for v in scenes.values()])) if all(v["summaries"][a][s][reg][k] is not None for v in scenes.values()) else None for k in METRICS} for reg in REGIONS} for s in SPLITS} for a in ARMS}
    delta={a+"-"+b:{s:{reg:{k:equal[a][s][reg][k]-equal[b][s][reg][k] if equal[a][s][reg][k] is not None and equal[b][s][reg][k] is not None else None for k in METRICS} for reg in REGIONS} for s in SPLITS} for a,b in COMPARISONS}
    csvwrite(RUN/"metrics_per_frame.csv",rows);csvwrite(RUN/"paired_differences.csv",diffs)
    save(RUN/"evaluation_summary.json",dict(status="completed",scenes=scenes,equal_scene_means=equal,
        equal_scene_differences=delta,CPU_seconds=time.monotonic()-started,rows=len(rows),paired_rows=len(diffs),
        freeze=freeze,historical_comparison_warning="U/Q/Q0 differ in initialization and budget; historical scores reused with source SHA. C/P/PQ use one same-scene warm-start and fresh Adam. C-to-P adds training-pose input and support capacity.",
        selection="Complete vectors and fixed visual evidence; no scalar score, no per-frame oracle; developed scenes"))


def verify():
    import cv2
    import numpy as np
    start=time.monotonic();freeze=freeze_gate();ev=read(RUN/"evaluation_summary.json")
    assert ev["freeze"]["sha256"]==freeze["sha256"]
    rows=numeric_csv(RUN/"metrics_per_frame.csv");diffs=numeric_csv(RUN/"paired_differences.csv")
    expected=sum((c["training_frames"]+c["retained_frames"])*len(ARMS)*len(REGIONS) for c in config()["scenes"].values())
    assert len(rows)==ev["rows"]==expected
    assert len(diffs)==ev["paired_rows"]==sum((c["training_frames"]+c["retained_frames"])*len(COMPARISONS)*len(REGIONS) for c in config()["scenes"].values())
    index={(r["scene"],r["run"],r["split"],r["frame_id"],r["region"]):r for r in rows};assert len(index)==len(rows)
    max_psnr=0.;count=0;max_summary=0.;max_pair=0.;max_equal=0.
    for scene,cfg in config()["scenes"].items():
        for split,name in [("retained","evaluation_manifest.json"),("train","manifest.json")]:
            for frame in read(ROOT/cfg["input_dir"]/name)["frames"]:
                gt=cv2.imread(frame["image_path"])[...,::-1].astype(np.float64)/255
                labels=cv2.imread(frame["mask_path"],0)>=config()["mask_threshold"]
                for arm in NEW_ARMS:
                    record=index[scene,arm,split,frame["frame_id"],"full"]
                    asset=dict(path=record["raw_render_path"],sha256=record["raw_render_sha256"])
                    with np.load(checked(asset),allow_pickle=False) as z: rgb=np.clip(z["rgb"],0,1).astype(np.float64)
                    for region in REGIONS:
                        mask=np.ones(labels.shape,bool) if region=="full" else labels if region=="foreground" else ~labels
                        r=index[scene,arm,split,frame["frame_id"],region]
                        assert int(mask.sum())==r["pixels"]
                        if mask.any():
                            value=float(-10*np.log10(max(float(np.square(rgb[mask]-gt[mask]).mean()),1e-12)))
                            max_psnr=max(max_psnr,abs(value-r["psnr_db"]));count+=1
        for arm in ARMS:
            for split in SPLITS:
                for region in REGIONS:
                    calculated=mean_summary([r for r in select(rows,split) if r["scene"]==scene and r["run"]==arm and r["region"]==region])
                    for key,value in calculated.items():
                        recorded=ev["scenes"][scene]["summaries"][arm][split][region][key]
                        assert (value is None)==(recorded is None)
                        if value is not None:max_summary=max(max_summary,abs(value-recorded))
        calculated=paired_summary([d for d in diffs if d["scene"]==scene])
        assert calculated==ev["scenes"][scene]["paired"]
    for r in diffs:
        a,b=r["comparison"].split("-")
        x=index[r["scene"],a,r["split"],r["frame_id"],r["region"]];y=index[r["scene"],b,r["split"],r["frame_id"],r["region"]]
        for key in METRICS:
            assert (r[key] is None)==(x[key] is None or y[key] is None)
            if r[key] is not None:max_pair=max(max_pair,abs((x[key]-y[key])-r[key]))
    for arm in ARMS:
        for split in SPLITS:
            for region in REGIONS:
                for key in METRICS:
                    vals=[ev["scenes"][s]["summaries"][arm][split][region][key] for s in config()["scenes"]]
                    if all(v is not None for v in vals):max_equal=max(max_equal,abs(float(np.mean(vals))-ev["equal_scene_means"][arm][split][region][key]))
    assert max_psnr<1e-10 and max_summary<1e-10 and max_pair<1e-10 and max_equal<1e-10
    save(RUN/"protocol/independent_verification.json",dict(status="passed",rows=len(rows),paired_rows=len(diffs),
        PSNR_recomputed=count,max_PSNR_difference=max_psnr,max_summary_difference=max_summary,
        max_pair_difference=max_pair,max_equal_scene_difference=max_equal,CPU_seconds=time.monotonic()-start,
        SSIM_LPIPS_scope="Reused exact evaluator once; independently checked aggregation and pair arithmetic, not recomputed with another implementation"))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true");args=parser.parse_args()
    verify() if args.verify else run()
