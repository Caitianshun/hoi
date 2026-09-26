"""CPU metrics and figures after all BEHAVE terminal models and renders freeze.

The explicit freeze gate is checked before reading any GT pixels or renders.
Only final E0 and official Wu 4DGS adapted runs enter this comparison. Historical
Ref-conditioned B/F results remain in summarize_existing.py's separate table.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from summarize_existing import identity, psnr, sha256

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
AUX = ROOT / "experiments/aux_ref_object_reconstruction_20260924/run01"
METHODS = ("E0", "4DGS")
GROUPS = ("camera1_E", "camera0_paired_E", "camera0_full_training_fit")
REGIONS = ("full", "human", "object", "background", "foreground")
METRICS = ("psnr_db", "ssim", "lpips_spatial_mean")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def csv_save(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked(asset):
    assert sha256(asset["path"]) == asset["sha256"], f"Changed asset: {asset['path']}"
    return Path(asset["path"])


def gate(freeze_path):
    """Metadata/hashes only; no GT image, labels or prediction arrays are read."""
    freeze = read(freeze_path)
    assert freeze.get("status") == "all_states_frozen", "Final states are not frozen"
    assert freeze.get("scope") == "behave_pair", "Require the complete BEHAVE pair"
    assert freeze.get("assets"), "Freeze cannot be an empty marker"
    for asset in freeze["assets"]:
        checked(asset)
    frozen_assets = {str(Path(a["path"]).resolve()): a["sha256"] for a in freeze["assets"]}
    freeze_sha = sha256(freeze_path)
    manifests = {}
    for dev in ("dev1", "dev2"):
        expected = {"camera1_E": 5 if dev == "dev1" else 4,
                    "camera0_paired_E": 5 if dev == "dev1" else 4,
                    "camera0_full_training_fit": 114 if dev == "dev1" else 98}
        for method in METHODS:
            path = RUN / "evaluation" / method / dev / "manifest.json"
            m = read(path)
            assert m["status"] == "completed" and m["dev"] == dev
            assert m["all_finals_freeze"]["sha256"] == freeze_sha
            assert checked(m["all_finals_freeze"]).resolve() == Path(freeze_path).resolve()
            checkpoint = m["source_checkpoint"]
            assert frozen_assets.get(str(checked(checkpoint).resolve())) == checkpoint["sha256"], "Rendered checkpoint was not frozen"
            assert m["frame_count"] == len(m["frames"])
            assert m.get("optimization_steps", 0) == 0
            index = {(r["group"], r["frame_id"]): r for r in m["frames"]}
            assert len(index) == len(m["frames"]), "Duplicate render frame"
            for group, count in expected.items():
                assert sum(r["group"] == group for r in m["frames"]) == count, (method, dev, group)
            assert set(r["group"] for r in m["frames"]) == set(GROUPS)
            manifests[method, dev] = (m, index, identity(path))
        a, b = manifests["E0", dev][1], manifests["4DGS", dev][1]
        assert set(a) == set(b)
        for key in a:
            assert a[key]["time_seconds"] == b[key]["time_seconds"]
            assert a[key]["source_frame"] == b[key]["source_frame"], key
            assert a[key]["visualization_selected"] == b[key]["visualization_selected"]
    return manifests


class InputRegions:
    """Use each input set's existing SAM2 regions, without substituting E labels."""
    def __init__(self, dev):
        assets_path = RUN / "protocol" / f"behave_{dev}_assets.json"
        assets = read(assets_path)
        segment = next(x["asset"] for x in assets["training_assets"]
                       if x["category"] == "input_RGB_derived_segmentation")
        with np.load(checked(segment), allow_pickle=False) as data:
            self.training = data["entity_labels"].copy()
        native_path = AUX / "inputs" / dev / "input_manifest.json"
        native = read(native_path)
        assert native["camera_id"] == 0
        native_segment = {"path": native["segmentation"], "sha256": native["segmentation_sha256"]}
        with np.load(checked(native_segment), allow_pickle=False) as data:
            self.native = data["entity_labels"].copy()
        self.native_index = {float(t): i for i, t in enumerate(native["timestamp_seconds"])}
        self.train_source = segment
        self.native_source = native_segment
        self.sources = [identity(assets_path), segment, identity(native_path), native_segment]
        assert len(self.training) == (114 if dev == "dev1" else 98)
        assert len(self.native) == len(self.native_index)

    def get(self, row):
        group = row["group"]
        if group == "camera0_full_training_fit":
            return self.training[int(row["frame_id"])], self.train_source, "original S1 training SAM2 labels"
        if group == "camera0_paired_E":
            return self.native[self.native_index[float(row["time_seconds"])]], self.native_source, "native camera0 AUX SAM2 labels; independent input-view diagnostic"
        asset = row["source_frame"]["regions"]
        with np.load(checked(asset), allow_pickle=False) as data:
            labels = data["entity_labels"].copy()
        return labels, asset, "fixed historical camera1 fit-derived H/O/S; unchanged denominator"


def evaluate_image(pred_raw, gt, labels, lpips_model):
    assert pred_raw.shape == gt.shape and labels.shape == gt.shape[:2]
    assert np.isfinite(pred_raw).all() and np.isfinite(gt).all()
    assert set(np.unique(labels).tolist()).issubset({0, 1, 2})
    pred = np.clip(pred_raw, 0, 1).astype(np.float64)
    error = np.square(pred - gt).mean(-1)
    _, ssim_map = structural_similarity(gt, pred, data_range=1.0, channel_axis=2,
                                        win_size=7, gaussian_weights=False,
                                        use_sample_covariance=True, full=True)
    ssim_map = ssim_map.mean(-1)
    lpips_map = None
    if lpips_model is not None:
        with torch.no_grad():
            p = torch.from_numpy(pred.transpose(2, 0, 1).copy()).float()[None]
            g = torch.from_numpy(gt.transpose(2, 0, 1).copy()).float()[None]
            lpips_map = lpips_model(p, g, normalize=True).numpy().squeeze()
        assert lpips_map.shape == labels.shape and np.isfinite(lpips_map).all()
    masks = {"full": np.ones(labels.shape, bool), "human": labels == 1,
             "object": labels == 2, "background": labels == 0, "foreground": labels > 0}
    result = {}
    total_sse = float(error.sum())
    for region, mask in masks.items():
        n = int(mask.sum())
        sse = float(error[mask].sum()) if n else None
        mse = sse / n if n else None
        result[region] = dict(pixels=n, pixel_fraction=n / labels.size, mse=mse,
                              sse_rgb_mean=sse, psnr_db=psnr(mse) if n else None,
                              ssim=float(ssim_map[mask].mean()) if n else None,
                              lpips_spatial_mean=float(lpips_map[mask].mean()) if n and lpips_map is not None else None,
                              full_error_share=sse / total_sse if n and total_sse else None)
    assert sum(result[r]["pixels"] for r in ("human", "object", "background")) == labels.size
    assert np.isclose(sum(result[r]["sse_rgb_mean"] or 0 for r in ("human", "object", "background")), total_sse)
    return result, dict(min=float(pred_raw.min()), max=float(pred_raw.max()),
                        clipped_channel_fraction=float(((pred_raw < 0) | (pred_raw > 1)).mean()))


def stats(values):
    valid = [v for v in values if v is not None]
    return dict(mean=float(np.mean(valid)) if valid else None,
                median=float(np.median(valid)) if valid else None,
                valid_frames=len(valid), requested_frames=len(values))


def aggregate(rows):
    groups, paired_rows = {}, []
    for dev in ("dev1", "dev2"):
        groups[dev] = {}
        for group in GROUPS:
            selected = [r for r in rows if r["dev"] == dev and r["group"] == group]
            result = {"methods": {}, "paired_4DGS_minus_E0": {}}
            for method in METHODS:
                result["methods"][method] = {}
                for region in REGIONS:
                    rr = [r for r in selected if r["method"] == method and r["region"] == region]
                    valid = [r for r in rr if r["mse"] is not None]
                    n = sum(r["pixels"] for r in valid)
                    se = sum(r["sse_rgb_mean"] for r in valid)
                    result["methods"][method][region] = {
                        **{metric: stats([r[metric] for r in rr]) for metric in METRICS},
                        "pooled_mse": se / n if n else None,
                        "pooled_psnr_db": psnr(se / n) if n else None,
                        "summed_pixels": n, "summed_sse_rgb_mean": se if valid else None,
                        "times_seconds": [r["time_seconds"] for r in rr],
                    }
            for region in REGIONS:
                lookup = {(r["method"], r["frame_id"]): r for r in selected if r["region"] == region}
                frames = sorted({r["frame_id"] for r in selected})
                result["paired_4DGS_minus_E0"][region] = {}
                for metric in METRICS:
                    values = []
                    for frame in frames:
                        a, b = lookup["4DGS", frame], lookup["E0", frame]
                        assert a["pixels"] == b["pixels"] and a["time_seconds"] == b["time_seconds"]
                        delta = a[metric] - b[metric] if a[metric] is not None and b[metric] is not None else None
                        values.append(delta)
                        paired_rows.append(dict(dev=dev, group=group, frame_id=frame,
                                                time_seconds=a["time_seconds"], region=region, metric=metric,
                                                difference="4DGS-E0", value=delta))
                    result["paired_4DGS_minus_E0"][region][metric] = stats(values)
                a = result["methods"]["4DGS"][region]["pooled_psnr_db"]
                b = result["methods"]["E0"][region]["pooled_psnr_db"]
                result["paired_4DGS_minus_E0"][region]["pooled_psnr_difference_db"] = a - b if a is not None and b is not None else None
            groups[dev][group] = result
    return groups, paired_rows


def display_image(rgb):
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def fonts():
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, 20), ImageFont.truetype(path, 16)


def make_figure(output, dev, row, gt, predictions, metrics, labels):
    """Full images always present; crop bounds depend only on fixed GT O."""
    font, small = fonts()
    height, width = gt.shape[:2]
    canvas = Image.new("RGB", (3 * width, height + 80), "white")
    draw = ImageDraw.Draw(canvas)
    images = [gt, predictions["E0"], predictions["4DGS"]]
    names = ["GT", "E0 complete S1", "Wu 4DGS shared-prior init"]
    for j, (name, rgb) in enumerate(zip(names, images)):
        draw.text((j * width + 10, 8), name, fill="black", font=font)
        canvas.paste(display_image(rgb), (j * width, 36))
        caption = f"{row['frame_id']}  t={row['time_seconds']:.6f}  O pixels={int((labels == 2).sum())}"
        if j:
            m = metrics[METHODS[j - 1]]
            object_psnr = m["object"]["psnr_db"]
            oval = "NA" if object_psnr is None else f"{object_psnr:.3f}"
            caption = f"full {m['full']['psnr_db']:.3f} dB   O {oval} dB"
        draw.text((j * width + 10, height + 44), caption, fill="black", font=small)
    folder = output / "figures" / dev / row["group"]
    folder.mkdir(parents=True, exist_ok=True)
    full = folder / f"{row['frame_id']}_full.png"
    canvas.save(full)
    record = dict(dev=dev, group=row["group"], frame_id=row["frame_id"], time_seconds=row["time_seconds"],
                  full=identity(full), methods=["GT", *METHODS], displayed_clipped_to_unit_range=True)
    if row["group"] == "camera1_E":
        yy, xx = np.where(labels == 2)
        assert len(xx), "Historical E object region must remain nonempty"
        box = (max(0, int(xx.min()) - 60), max(0, int(yy.min()) - 60),
               min(width, int(xx.max()) + 61), min(height, int(yy.max()) + 61))
        crop = Image.new("RGB", (3 * 320, 300), "white")
        dc = ImageDraw.Draw(crop)
        for j, (name, rgb) in enumerate(zip(names, images)):
            im = display_image(rgb).crop(box)
            im.thumbnail((310, 250), Image.Resampling.LANCZOS)
            crop.paste(im, (j * 320 + (320 - im.width) // 2, 36))
            dc.text((j * 320 + 6, 8), name, font=small, fill="black")
        path = folder / f"{row['frame_id']}_crop.png"
        crop.save(path)
        record.update(crop=identity(path), crop_bounds_xyxy=box,
                      crop_rule="fixed O bounding box plus 60px, same bounds for every method, no metric cropping")
    return record


def make_contact_sheets(output, figures):
    records = []
    for dev in ("dev1", "dev2"):
        for group in GROUPS:
            selected = sorted([r for r in figures if r["dev"] == dev and r["group"] == group],
                              key=lambda x: x["time_seconds"])
            assert selected
            thumbs = []
            for row in selected:
                with Image.open(row["full"]["path"]) as im:
                    thumb = im.copy()
                thumb.thumbnail((960, 300), Image.Resampling.LANCZOS)
                thumbs.append(thumb)
            sheet = Image.new("RGB", (max(im.width for im in thumbs), sum(im.height for im in thumbs)), "white")
            y = 0
            for im in thumbs:
                sheet.paste(im, (0, y)); y += im.height
            path = output / "figures" / f"{dev}_{group}_all_selected_full.png"
            sheet.save(path)
            records.append(dict(dev=dev, group=group, frames=[r["frame_id"] for r in selected], sheet=identity(path)))
    return records


def run(freeze_path, output, threads=4):
    # Gate before even importing metric-weight loader or reading image arrays.
    manifests = gate(freeze_path)
    if (output / "evaluation_run.json").exists() or (output / "summary.json").exists():
        raise FileExistsError("Evaluation output already exists; retain original evaluation")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    legacy_path = AUX / "code/evaluate_aux.py"
    spec = importlib.util.spec_from_file_location("v3_legacy_metrics", legacy_path)
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    model, lpips_info = legacy.existing_lpips("cpu")
    save(output / "lpips.json", lpips_info)
    rows, figures, sources, raw_ranges = [], [], [identity(freeze_path), identity(__file__), identity(legacy_path)], []
    try:
        for dev in ("dev1", "dev2"):
            region_source = InputRegions(dev)
            sources += region_source.sources
            for method in METHODS:
                sources.append(manifests[method, dev][2])
            order = sorted(manifests["E0", dev][1], key=lambda k: (GROUPS.index(k[0]), manifests["E0", dev][1][k]["time_seconds"]))
            for index, key in enumerate(order):
                row = manifests["E0", dev][1][key]
                f = row["source_frame"]
                gt_path = checked({"path": f["image_path"], "sha256": f["image_sha256"]})
                gt_bgr = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
                assert gt_bgr is not None
                gt = gt_bgr[..., ::-1].astype(np.float64) / 255.0
                labels, region_asset, label_meaning = region_source.get(row)
                prediction, results = {}, {}
                for method in METHODS:
                    render_row = manifests[method, dev][1][key]
                    with np.load(checked(render_row["render"]), allow_pickle=False) as data:
                        raw = data["rgb"].copy()
                    results[method], ranges = evaluate_image(raw, gt, labels, model)
                    prediction[method] = raw
                    raw_ranges.append(dict(dev=dev, method=method, group=row["group"], frame_id=row["frame_id"], **ranges))
                    for region, metrics in results[method].items():
                        rows.append(dict(dev=dev, method=method, group=row["group"], frame_id=row["frame_id"],
                                         time_seconds=row["time_seconds"], camera_id=f["camera_id"], region=region,
                                         **metrics, GT_path=str(gt_path), GT_sha256=f["image_sha256"],
                                         render_path=render_row["render"]["path"], render_sha256=render_row["render"]["sha256"],
                                         region_path=region_asset["path"], region_sha256=region_asset["sha256"],
                                         region_definition=label_meaning))
                if row["visualization_selected"]:
                    figures.append(make_figure(output, dev, row, gt, prediction, results, labels))
                if index % 20 == 0:
                    save(output / "evaluation_progress.json", dict(dev=dev, processed_frames=index + 1,
                                                                    metric_rows=len(rows), seconds=time.perf_counter() - start))
        main = [r for r in rows if r["group"] == "camera1_E"]
        fit = [r for r in rows if r["group"] != "camera1_E"]
        assert len(main) == 9 * 2 * len(REGIONS)
        assert len(fit) == (114 + 98 + 9) * 2 * len(REGIONS)
        csv_save(output / "metrics_per_frame.csv", main)
        csv_save(output / "input_fit.csv", fit)
        grouped, paired = aggregate(rows)
        csv_save(output / "paired_differences.csv", paired)
        save(output / "raw_render_ranges.json", raw_ranges)
        sheets = make_contact_sheets(output, figures)
        save(output / "figure_manifest.json", dict(figures=figures, contact_sheets=sheets,
             selection="All nine E frames and all nine paired input frames; original training preview uses uniform at most16 indices per event, metrics use all114/98.",
             interpretation="Full images always included. Object crops supplement full images; pixel metrics use original full fixed-region masks."))
        summary = {
            "status": "completed", "protocol": "BEHAVE camera0 full S1 input to fixed camera1 E",
            "comparison": "E0 complete original S1 RGB-predicted motion versus official Wu 4DGS adaptation with shared-prior initialization",
            "same_input_system_comparison_not_single_variable_ablation": True,
            "ref_conditioned_B_F_excluded": True, "results": grouped,
            "metric_definition": {
                "prediction": "raw floating RGB retained; metrics and displays clamp to [0,1], no color fit/alignment/output-based selection",
                "GT": "frozen uint8 rectified RGB /255, unchanged 640x480",
                "PSNR": "-10 log10(mean per-pixel RGB squared error); floor1e-12; equal-frame mean and pixel-pooled score both reported",
                "SSIM": "mean of full-image standard7x7 uniform SSIM map inside the fixed region; sample covariance, data_range1",
                "LPIPS": "AlexNet v0.1 full-image spatial-map region mean, lower better; historical locally cached weights only",
                "regions": "camera1 historical fit-derived H/O/S exhaustive partition; foreground=H union O. camera0 uses own frozen SAM2 input labels, with empty regions NA.",
                "paired_difference": "4DGS minus E0 on identical frame/region; means and medians per event. Nine frames are not nine independent scenes.",
                "SSE": "sum over pixels of RGB-mean squared error, not RGB channel sum",
            },
            "input_fit_caveats": [
                "camera0_full_training_fit uses every original S1 training image and is training reconstruction, not generalization",
                "camera0_paired_E is native camera0 at the same nominal capture groups as E; these are not exact original training frames",
                "recording timestamps and official nominal capture times have no verified exact exposure synchronization",
                "camera0 input masks differ from camera1 fit-derived regions; view differences are descriptive, not isolated view-loss attribution",
            ],
            "lpips": lpips_info, "metric_rows": len(main), "input_fit_rows": len(fit),
            "evaluation_sources": sources, "optimization_steps": 0, "GPU_used": False,
            "seconds": time.perf_counter() - start,
        }
        save(output / "summary.json", summary)
        save(output / "evaluation_run.json", dict(status="completed", seconds=time.perf_counter() - start,
             optimization_steps=0, GPU_used=False, freeze=identity(freeze_path),
             metrics=identity(output / "metrics_per_frame.csv"), input_fit=identity(output / "input_fit.csv"),
             figures=identity(output / "figure_manifest.json"), summary=identity(output / "summary.json")))
        print(json.dumps({"status": "completed", "metric_rows": len(main), "input_fit_rows": len(fit),
                          "seconds": time.perf_counter() - start, "output": str(output)}), flush=True)
    except BaseException:
        save(output / "evaluation_failure.json", dict(status="failed", metric_rows=len(rows),
             seconds=time.perf_counter() - start, traceback=traceback.format_exc(), optimization_steps=0, GPU_used=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN / "evaluation/comparison")
    parser.add_argument("--threads", type=int, default=4)
    options = parser.parse_args()
    run(options.freeze, options.output, options.threads)
