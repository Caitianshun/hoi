"""Independent CPU-only V3 integrity and PSNR verification, after completion.

No renderer, evaluator or official training module is imported. Completion is
required before reading model states, test GT pixels or floating predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
DATASETS = ("behave_dev1", "behave_dev2", "hos_backpack")
REGIONS = ("full", "human", "object", "background", "foreground")
GROUPS = ("camera1_E", "camera0_paired_E", "camera0_full_training_fit")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def check_asset(asset):
    path = Path(asset["path"])
    assert path.is_file(), f"Missing asset: {path}"
    assert digest(path) == asset["sha256"], f"Changed frozen asset: {path}"
    return path


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def csv_read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def number(value):
    return None if value is None or value == "" else float(value)


def close(a, b, atol=1e-9, rtol=1e-10):
    a, b = number(a), number(b)
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert np.isfinite(a) and np.isfinite(b) and np.isclose(a, b, atol=atol, rtol=rtol), (a, b)


def psnr(mse):
    return -10 * math.log10(max(float(mse), 1e-12))


def completion_gate():
    """Metadata-only gate, before any model/prediction/test-pixel read."""
    assert read(RUN / "pipeline.json")["status"] == "formal_and_behave_evaluation_completed"
    assert read(RUN / "continuation.json")["status"] == "all_training_rendering_evaluation_completed"
    for dataset in DATASETS:
        r = read(RUN / "runs" / f"{dataset}_formal/run.json")
        assert r["status"] == "completed"
    for folder in ("comparison", "hos_comparison"):
        assert read(RUN / "evaluation" / folder / "summary.json")["status"] == "completed"
    assert read(RUN / "evaluation/comparison/evaluation_run.json")["status"] == "completed"


def verify_protocol():
    f = read(RUN / "protocol/training_frozen.json")
    assert f["status"] == "training_configuration_frozen" and f["datasets"] == list(DATASETS)
    assert (f["seed"], f["coarse_iterations"], f["fine_iterations"]) == (12345, 3000, 14000)
    assert (f["max_formal_runs"], f["max_gpu_seconds"], f["max_temporary_steps"]) == (3, 43200, 200)
    for a in f["sources"] + f["inputs"] + f["checks"]:
        check_asset(a)
    frozen = {}
    for name, scope in (("finals.json", "behave_pair"), ("hos_finals.json", "hos_backpack")):
        path = RUN / "protocol" / name
        data = read(path)
        assert data["status"] == "all_states_frozen" and data["scope"] == scope and data["assets"]
        for a in data["assets"]:
            check_asset(a)
            key = str(Path(a["path"]).resolve())
            assert key not in frozen or frozen[key] == a["sha256"]
            frozen[key] = a["sha256"]
    train_manifests, eval_manifests, boundaries = {}, {}, {}
    for dataset in DATASETS:
        path = RUN / "inputs" / dataset / "manifest.json"
        text = path.read_text(); train = json.loads(text)
        evaluation = read(path.with_name("evaluation_manifest.json"))
        assert train["role"] == "training_only"
        assert len(train["frames"]) == {"behave_dev1": 114, "behave_dev2": 98, "hos_backpack": 268}[dataset]
        assert len({r["frame_id"] for r in train["frames"]}) == len(train["frames"])
        test_paths = [str(Path(r["image_path"]).resolve()) for r in evaluation["frames"]]
        assert all(p not in text for p in test_paths), "Test RGB path leaked into train manifest"
        train_paths = {str(Path(r["image_path"]).resolve()) for r in train["frames"]}
        assert not set(test_paths) & train_paths
        times = [r["time"] for r in train["frames"]]
        assert min(times) == 0 and max(times) == 1
        assert all(a < b for a, b in zip(times, times[1:]))
        pc = train["point_cloud"]
        pc_asset = {"path": pc["npz_path"], "sha256": pc.get("npz_sha256", pc["sha256"])}
        check_asset(pc_asset)
        for row in train["frames"]:
            check_asset({"path": row["image_path"], "sha256": row["image_sha256"]})
            if "mask_sha256" in row:
                check_asset({"path": row["mask_path"], "sha256": row["mask_sha256"]})
        train_manifests[dataset], eval_manifests[dataset] = train, evaluation
        boundaries[dataset] = dict(train_frames=len(train["frames"]), test_frames=len(evaluation["frames"]),
                                   no_test_RGB_paths_in_training=True, point_cloud=pc_asset)
    hos = train_manifests["hos_backpack"]
    test0 = next(x for x in eval_manifests["hos_backpack"]["frames"] if x["frame_id"] == "00000")
    assert test0["time"] == -1 / 282 and test0["extrapolation"] is True
    assert "nominal" in hos["time_seconds_field_units"]
    audit = read(RUN / "protocol/hos_independent_source_review.json")
    assert audit["status"] == "passed_source_and_initialization_audit_with_declared_limits"
    assert not audit["actionable_issues"]
    assert audit["source_verification"]["selected_color_equals_two_training_RGB_mean_exactly"]
    source_hashes = {str(Path(a["path"]).resolve()): a["sha256"] for a in audit["sources"]}
    assert source_hashes[str(Path(hos["point_cloud"]["npz_path"]).resolve())] == hos["point_cloud"]["npz_sha256"]
    return f, frozen, train_manifests, eval_manifests, boundaries


def verify_budget(training_freeze):
    ledger = [json.loads(x) for x in (RUN / "protocol/temporary_steps.jsonl").read_text().splitlines()]
    counts = Counter()
    for row in ledger:
        dataset = row.get("dataset", row["run"].removesuffix("_check"))
        assert dataset in DATASETS
        counts[dataset] += 1
    assert len(ledger) == training_freeze["temporary_steps_used"] == 200
    assert dict(counts) == {"behave_dev1": 50, "behave_dev2": 50, "hos_backpack": 100}
    assert len({(r["run"], r["stage"], r["iteration"]) for r in ledger}) == 200
    costs = read(RUN / "protocol/gpu_cost_ledger.json")
    assert all(np.isfinite(r["wall_seconds"]) and r["wall_seconds"] >= 0 for r in costs)
    total = sum(r["wall_seconds"] for r in costs)
    assert total < 43200, f"Cumulative GPU-job budget exceeded: {total}"
    labels = Counter(r["label"] for r in costs)
    for dataset in DATASETS:
        assert labels[dataset + "_formal"] == 1
    for attempt in (RUN / "logs").glob("*/attempt.json"):
        a = read(attempt)
        assert a["status"] in ("completed", "failed") and "wall_seconds" in a
        assert any(c.get("pid") == a.get("pid") and c["label"] == a["label"] for c in costs), attempt
    return dict(temporary_total=200, temporary_by_dataset=dict(counts),
                formal_runs=3, nominal_iterations_total=51000, optimizer_updates_total=50997,
                gpu_job_wall_seconds=total, budget_seconds=43200, ledger_entries=len(costs),
                failed_cost_entries=[r["label"] for r in costs if r["status"] == "failed"],
                measurement_limit="GPU-job wall ledger includes subprocess time. Early checks/native checks used measured blocks and exclude some import overhead; CPU work is not GPU time.")


def tensors_finite(obj):
    counts = Counter()
    def visit(x, key):
        if torch.is_tensor(x):
            counts["tensors"] += 1; counts["tensor_elements"] += x.numel()
            if x.is_floating_point() or x.is_complex():
                counts["floating_tensors"] += 1
                assert bool(torch.isfinite(x).all()), f"Nonfinite model tensor: {key}"
        elif isinstance(x, np.ndarray):
            counts["numpy_arrays"] += 1
            if np.issubdtype(x.dtype, np.number):
                assert np.isfinite(x).all(), f"Nonfinite checkpoint array: {key}"
        elif isinstance(x, dict):
            for k, v in x.items(): visit(v, f"{key}.{k}")
        elif isinstance(x, (tuple, list)):
            for k, v in enumerate(x): visit(v, f"{key}[{k}]")
        elif isinstance(x, float):
            assert math.isfinite(x), f"Nonfinite checkpoint scalar: {key}"
    visit(obj, "state")
    return dict(counts, all_finite=True)


def inspect_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    result = tensors_finite(obj)
    del obj; gc.collect()
    return result


def verify_models(frozen):
    models = []
    for dataset in DATASETS:
        rd = RUN / "runs" / f"{dataset}_formal"
        run, cfg = read(rd / "run.json"), read(rd / "effective_config.json")
        assert run["status"] == "completed" and run["nominal_iterations"] == 17000 and run["optimizer_updates"] == 16999
        assert run["test_RGB_loaded"] is False and run["seed"] == 12345
        assert cfg["check"] is False and cfg["seed"] == 12345
        assert cfg["optimization"]["iterations"] == 14000 and cfg["optimization"]["coarse_iterations"] == 3000
        assert run["source_config_sha256"] == digest(rd / "effective_config.json")
        assert cfg["manifest_sha256"] == digest(RUN / "inputs" / dataset / "manifest.json")
        path = Path(run["checkpoint"]).resolve()
        assert digest(path) == run["checkpoint_sha256"] == frozen[str(path)]
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["stage"] == "fine" and state["iteration"] == 14000 and state["total_nominal_steps"] == 17000
        assert set(("python", "numpy", "torch", "cuda")) <= set(state["rng"])
        assert all(k in state for k in ("viewpoint_stack", "temp_list", "deformation_accum"))
        finite = tensors_finite(state)
        assert len(state["model"][1]) == run["final_points"]
        del state; gc.collect()
        inference_dir = rd / "point_cloud/fine_iteration_14000"
        expected = ("point_cloud.ply", "deformation.pth", "deformation_table.pth", "deformation_accum.pth")
        assert all((inference_dir / n).is_file() for n in expected)
        inference = []
        for p in sorted(inference_dir.iterdir()):
            if p.is_file():
                a = identity(p)
                if p.suffix == ".pth": a["finite_state"] = inspect_state(p)
                inference.append(a)
        models.append(dict(id=dataset, role="formal_4DGS", status=run["status"], seed=12345,
                           nominal_iterations=17000, optimizer_updates=16999, final_points=run["final_points"],
                           terminal_resume_state=identity(path), finite_state=finite,
                           inference_files=inference, inference_bytes=sum(a["bytes"] for a in inference),
                           resume_state_bytes=path.stat().st_size, effective_config=identity(rd / "effective_config.json")))
    for dev in ("dev1", "dev2"):
        p = ROOT / "experiments/structured_hoi_20260923" / f"{dev}_S1_v1/checkpoint_008000.pt"
        assert digest(p) == frozen[str(p.resolve())]
        state = torch.load(p, map_location="cpu", weights_only=False)
        assert state["step"] == 8000
        finite = tensors_finite(state); del state; gc.collect()
        models.append(dict(id=f"E0_{dev}", role="historical_complete_S1", status="frozen_and_rendered", original_step=8000,
                           terminal_state=identity(p), finite_state=finite,
                           inference_files=[identity(p)], inference_bytes=p.stat().st_size,
                           resume_state_bytes=p.stat().st_size, size_note="Original full checkpoint is retained; no stripped inference-only S1 export was fabricated."))
    h0 = ROOT / "other data/hosnerf/checkpoints/Backpack.ckpt"
    assert digest(h0) == frozen[str(h0.resolve())]
    state = torch.load(h0, map_location="cpu", weights_only=False)
    assert state["global_step"] == 200000
    finite = tensors_finite(state); del state; gc.collect()
    models.append(dict(id="H0_Backpack", role="released_HOSNeRF_native", status="strict_loaded_and_rendered", released_step=200000,
                       terminal_state=identity(h0), finite_state=finite,
                       inference_files=[identity(h0)], inference_bytes=h0.stat().st_size,
                       resume_state_bytes=h0.stat().st_size,
                       size_note="Released native checkpoint includes its original state; historical training split identity remains unproven."))
    return models


def verify_render_manifests(evaluation_manifests):
    result, h0_paths = {}, {}
    for method in ("E0", "4DGS"):
        for dev in ("dev1", "dev2"):
            p = RUN / "evaluation" / method / dev / "manifest.json"
            m = read(p); assert m["status"] == "completed" and m["optimization_steps"] == 0
            assert m["all_finals_freeze"]["sha256"] == digest(RUN / "protocol/finals.json")
            expected = {"camera1_E": 5 if dev == "dev1" else 4,
                        "camera0_paired_E": 5 if dev == "dev1" else 4,
                        "camera0_full_training_fit": 114 if dev == "dev1" else 98}
            assert Counter(r["group"] for r in m["frames"]) == expected
            assert len(m["frames"]) == m["frame_count"] == sum(expected.values())
            assert len({(r["group"], r["frame_id"]) for r in m["frames"]}) == m["frame_count"]
            for g in GROUPS:
                times = [r["time_seconds"] for r in m["frames"] if r["group"] == g]
                assert len(times) == len(set(times))
            expected_E = {r["frame_id"]: r for r in evaluation_manifests[f"behave_{dev}"]["frames"]}
            for row in m["frames"]:
                check_asset(row["render"])
                if row["group"] == "camera1_E":
                    assert row["source_frame"] == expected_E[row["frame_id"]]
            result[f"{method}_{dev}"] = dict(frames=m["frame_count"], groups=expected, manifest=identity(p))
    h1 = read(RUN / "evaluation/H1/manifest.json")
    assert h1["status"] == "completed" and h1["frame_count"] == len(h1["frames"]) == 284
    assert h1["all_finals_freeze"]["sha256"] == digest(RUN / "protocol/hos_finals.json")
    assert Counter(r["group"] for r in h1["frames"]) == {"test": 16, "input_fit": 268}
    assert len({r["frame_id"] for r in h1["frames"]}) == 284
    for row in h1["frames"]: check_asset(row["render"])
    for tag, count in (("H0_native_first", 1), ("H0_native_remaining", 15)):
        path = RUN / "runs" / tag / "completion.json"
        m = read(path); assert m["status"] == "completed" and m["frames"] == count
        assert m["identity"]["strict_load"] and not m["identity"]["missing_keys"] and not m["identity"]["unexpected_keys"]
        assert len(m["identity"]["render_ids"]) == count
        for fid in m["identity"]["render_ids"]:
            assert fid not in h0_paths
            h0_paths[fid] = path.parent / (fid + ".npz")
    assert set(h0_paths) == {f["frame_id"] for f in evaluation_manifests["hos_backpack"]["frames"]}
    rendered_frozen = read(RUN / "evaluation/hos_comparison/rendered_states_frozen.json")
    assert rendered_frozen["status"] == "all_rendered_states_frozen" and len(rendered_frozen["assets"]) == 300
    for a in rendered_frozen["assets"]: check_asset(a)
    return result, h1, h0_paths


def independently_score(path, gt_path, labels, table_rows):
    with np.load(path, allow_pickle=False) as data:
        raw = data["rgb"].copy()
    gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    assert gt is not None
    gt = gt[..., ::-1].astype(np.float64) / 255
    assert raw.shape == gt.shape and np.isfinite(raw).all() and labels.shape == gt.shape[:2]
    error = ((np.clip(raw, 0, 1).astype(np.float64) - gt) ** 2).mean(-1)
    masks = {"full": np.ones(labels.shape, bool), "human": labels == 1, "object": labels == 2,
             "background": labels == 0, "foreground": labels > 0}
    diffs = []
    for row in table_rows:
        assert digest(path) == row["render_sha256"]
        assert digest(gt_path) == row["GT_sha256"]
        mask = masks[row["region"]]; n = int(mask.sum())
        assert n == int(row["pixels"])
        if n:
            sse = float(error[mask].sum()); mse = sse / n; value = psnr(mse)
            close(sse, row["sse_rgb_mean"], atol=1e-7)
            close(mse, row["mse"], atol=1e-12)
            close(value, row["psnr_db"], atol=1e-8)
            diffs.append(abs(value - float(row["psnr_db"])))
        else:
            assert all(number(row[k]) is None for k in ("sse_rgb_mean", "mse", "psnr_db"))
    return diffs


def verify_metrics(evaluation_manifests, h1, h0_paths):
    base = RUN / "evaluation/comparison"; hos_base = RUN / "evaluation/hos_comparison"
    main, fit = csv_read(base / "metrics_per_frame.csv"), csv_read(base / "input_fit.csv")
    hm, hf = csv_read(hos_base / "metrics_per_frame.csv"), csv_read(hos_base / "input_fit.csv")
    assert (len(main), len(fit), len(hm), len(hf)) == (90, 2210, 96, 804)
    assert len({(r["dev"], r["method"], r["frame_id"], r["region"]) for r in main}) == 90
    assert len({(r["method"], r["frame_id"], r["region"]) for r in hm}) == 96
    differences = []
    for dev in ("dev1", "dev2"):
        for f in evaluation_manifests[f"behave_{dev}"]["frames"]:
            with np.load(check_asset(f["regions"]), allow_pickle=False) as data:
                labels = data["entity_labels"].copy()
            for method in ("E0", "4DGS"):
                rr = [r for r in main if (r["dev"], r["method"], r["frame_id"]) == (dev, method, f["frame_id"])]
                assert len(rr) == 5 and {r["region"] for r in rr} == set(REGIONS)
                differences += independently_score(Path(rr[0]["render_path"]), Path(f["image_path"]), labels, rr)
    h1_test = {r["frame_id"]: r for r in h1["frames"] if r["group"] == "test"}
    for f in evaluation_manifests["hos_backpack"]["frames"]:
        mask = cv2.imread(f["mask_path"], cv2.IMREAD_GRAYSCALE)
        assert mask is not None; labels = (mask >= 128).astype(np.uint8)
        for method in ("H0", "H1"):
            rr = [r for r in hm if (r["method"], r["frame_id"]) == (method, f["frame_id"])]
            assert len(rr) == 3
            p = h0_paths[f["frame_id"]] if method == "H0" else Path(h1_test[f["frame_id"]]["render"]["path"])
            differences += independently_score(p, Path(f["image_path"]), labels, rr)
    # Independent algebra for every stored row, including complete training fits.
    for row in main + fit + hm + hf:
        n, mse, sse = int(row["pixels"]), number(row["mse"]), number(row["sse_rgb_mean"])
        if n:
            close(sse / n, mse, atol=1e-12); close(psnr(mse), row["psnr_db"])
        else: assert mse is None and sse is None and number(row["psnr_db"]) is None
    summary, hs = read(base / "summary.json"), read(hos_base / "summary.json")
    aggregates = 0
    for dev in ("dev1", "dev2"):
        for group in GROUPS:
            for method in ("E0", "4DGS"):
                for region in REGIONS:
                    rows = [r for r in main + fit if (r["dev"], r["group"], r["method"], r["region"]) == (dev, group, method, region)]
                    s = summary["results"][dev][group]["methods"][method][region]
                    check_aggregate(rows, s, nested=True); aggregates += 1
    for method, group in (("H0", "test"), ("H1", "test"), ("H1", "input_fit")):
        for region in ("full", "foreground", "background"):
            rows = [r for r in hm + hf if (r["method"], r["group"], r["region"]) == (method, group, region)]
            check_aggregate(rows, hs["results"][f"{method}_{group}"][region], nested=False); aggregates += 1
    assert hs["direct_fair_H0_H1_delta"] is None
    assert hs["time_boundary_diagnostic"]["H1_normalized_time"] == -1 / 282
    return dict(BEHAVE_test_outputs=18, HOS_test_outputs=32, independent_valid_region_PSNR_checks=len(differences),
                maximum_absolute_PSNR_difference_db=max(differences), row_counts=dict(BEHAVE_main=90, BEHAVE_input_fit=2210, HOS_test=96, HOS_input_fit=804),
                all_row_MSE_SSE_PSNR_algebra_verified=True, mean_median_pooled_aggregate_groups=aggregates)


def check_aggregate(rows, summary, nested):
    assert rows
    valid = [r for r in rows if number(r["mse"]) is not None]
    pixels = sum(int(r["pixels"]) for r in valid)
    mse = sum(float(r["sse_rgb_mean"]) for r in valid) / pixels if pixels else None
    close(summary["pooled_mse"], mse, atol=1e-12)
    close(summary["pooled_psnr_db"], psnr(mse) if mse is not None else None)
    for metric in ("psnr_db", "ssim", "lpips_spatial_mean"):
        vals = [number(r[metric]) for r in rows if number(r[metric]) is not None]
        mean = float(np.mean(vals)) if vals else None
        if nested:
            assert summary[metric]["valid_frames"] == len(vals)
            assert summary[metric]["requested_frames"] == len(rows)
            close(summary[metric]["mean"], mean)
            close(summary[metric]["median"], float(np.median(vals)) if vals else None)
        else: close(summary[metric], mean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    completion_gate()  # No incomplete-run failure artifact; simply refuse early use.
    torch.set_num_threads(args.threads); cv2.setNumThreads(args.threads)
    start = time.perf_counter()
    report = dict(status="checking", source=identity(__file__), optimization_steps=0, renders=0, GPU_used=False)
    try:
        f, frozen, train, evaluation, boundaries = verify_protocol()
        report["frozen_training_sources_inputs_and_final_assets_unchanged"] = True
        report["input_boundaries"] = boundaries
        report["budget"] = verify_budget(f)
        models = verify_models(frozen)
        report["models_checked"] = len(models)
        render_counts, h1, h0_paths = verify_render_manifests(evaluation)
        report["BEHAVE_render_manifests"] = render_counts
        report["HOS_renders"] = dict(H1=284, H0_first=1, H0_remaining=15)
        report["metrics"] = verify_metrics(evaluation, h1, h0_paths)
        report["preserved_exceptions"] = [
            "HOS frame00000 lies outside H1 training time bounds, encoded -1/282; not clamped or dropped.",
            "HOS full-video camera provenance and approximate dynamic foreground triangulation remain disclosed, not certified geometry truth.",
            "H0 native state/preprocessing and historical stage splits are not established as equal-information H1 conditions.",
            "Training-fit camera0 regions are input SAM2; camera1 historical H/O/S are fit-derived, with all pixels retained.",
        ]
        index = dict(status="all_terminal_models_verified", models=models, GPU_used=False,
                     generated_by=identity(__file__), size_semantics="Resume-state bytes and stored inference files are separate; do not add them as a single inference model size.")
        save(RUN / "model_index.json", index)
        report.update(status="passed", seconds=time.perf_counter() - start, model_index=identity(RUN / "model_index.json"))
        save(RUN / "protocol/final_integrity.json", report)
        print(json.dumps(dict(status="passed", models=len(models), metrics=report["metrics"], seconds=report["seconds"])), flush=True)
    except BaseException:
        report.update(status="failed", seconds=time.perf_counter() - start, traceback=traceback.format_exc())
        save(RUN / "protocol/final_integrity.json", report)
        raise


if __name__ == "__main__":
    main()
