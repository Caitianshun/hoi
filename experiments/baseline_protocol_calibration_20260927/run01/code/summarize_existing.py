"""V3 work package A: CPU-only arithmetic on existing, frozen V2 outputs.

No checkpoint loading, rendering, optimization, or new evaluation protocol.
SSE is the sum of per-pixel RGB-mean squared error (not RGB-summed SSE).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REGIONS = {"H": (1, "human"), "O": (2, "object"), "S": (0, "background")}
VARIANTS = ("B0", "B1", "F0", "F1", "F2")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    p = Path(path)
    return {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}


def psnr(mse):
    return -10.0 * math.log10(max(float(mse), 1e-12))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def region_aggregate(rows, region):
    valid = [r for r in rows if r[f"{region}_mse"] is not None]
    n = sum(r[f"{region}_pixels"] for r in valid)
    sse = sum(r[f"{region}_sse"] for r in valid)
    total = sum(r["full_sse"] for r in rows)
    return {
        "pixels_summed": n,
        "sse_rgb_mean_summed": sse if valid else None,
        "pooled_mse": sse / n if n else None,
        "pooled_psnr_db": psnr(sse / n) if n else None,
        "mean_frame_psnr_db": finite_mean([r[f"{region}_psnr_db"] for r in rows]),
        "mean_frame_mse": finite_mean([r[f"{region}_mse"] for r in rows]),
        "pooled_error_share": sse / total if valid and total else None,
        "mean_frame_error_share": finite_mean([r[f"{region}_error_share"] for r in rows]),
        "valid_frames": len(valid),
    }


def collect_camera0(root, out, camera1_rows, sources):
    aux = root / "experiments/aux_ref_object_reconstruction_20260924/run01"
    b1 = root / "experiments/fixed_motion_reconstruction_20260926/run01"
    v2 = root / "experiments/surface_observation_fusion_20260926/run01"
    rows, summary, missing = [], {}, []
    for dev in ("dev1", "dev2"):
        manifest_path = aux / "inputs" / dev / "input_manifest.json"
        manifest = read_json(manifest_path)
        sources.append(identity(manifest_path))
        times = manifest["timestamp_seconds"]
        image_pixels = manifest["width"] * manifest["height"]
        for variant in VARIANTS:
            if variant == "B0":
                fit_path = aux / "runs" / f"{dev}_Ref" / "camera0_fit.json"
            elif variant == "B1":
                fit_path = b1 / "runs" / dev / "camera0_fit.json"
            else:
                fit_path = v2 / "runs" / f"{dev}_{variant}" / "camera0_fit.json"
            if not fit_path.is_file():
                missing.append(str(fit_path))
                continue
            fit = read_json(fit_path)
            fit = fit["rows"] if isinstance(fit, dict) else fit
            assert len(fit) == len(times), (fit_path, len(fit), len(times))
            assert [int(r["frame"]) for r in fit] == list(range(len(times)))
            sources.append(identity(fit_path))
            group = []
            for source_row in fit:
                i = int(source_row["frame"])
                o_psnr = source_row.get("O_PSNR", source_row.get("object_psnr"))
                full_psnr = source_row.get("full_PSNR")
                object_pixels = int(source_row["O_pixels"])
                assert (o_psnr is None) == (object_pixels == 0)
                row = {
                    "dev": dev, "variant": variant, "camera_id": 0,
                    "frame_index": i, "sample_id": manifest["native_rows"][i]["sample_id"],
                    "nominal_time_seconds": times[i], "is_training_fit": True,
                    "full_pixels": image_pixels, "full_psnr_db": full_psnr,
                    "full_mse_from_saved_psnr": 10 ** (-full_psnr / 10) if full_psnr is not None else None,
                    "O_pixels": object_pixels, "O_psnr_db": o_psnr,
                    "O_mse_from_saved_psnr": 10 ** (-o_psnr / 10) if o_psnr is not None else None,
                    "H_pixels": None, "H_psnr_db": None, "S_pixels": None, "S_psnr_db": None,
                    "missing_metrics": "H,S" + (",full" if full_psnr is None else ""),
                    "O_region_source": "camera0 SAM2 input labels; differs from camera1 fit-derived O",
                    "time_basis": manifest["native_rows"][i]["clock"],
                    "metric_source": str(fit_path), "input_manifest": str(manifest_path),
                }
                rows.append(row)
                group.append(row)
            stats = {
                "times_seconds": times, "frame_count": len(group),
                "scope": "historical AUX/V2 Ref-motion object conditions; not complete original S1/E0",
                "H": None, "S": None, "source": str(fit_path),
            }
            for region in ("full", "O"):
                valid = [r for r in group if r[f"{region}_psnr_db"] is not None]
                denominator = sum(r[f"{region}_pixels"] for r in valid)
                pooled = sum(r[f"{region}_mse_from_saved_psnr"] * r[f"{region}_pixels"] for r in valid)
                stats[region] = {
                    "mean_frame_psnr_db": finite_mean([r[f"{region}_psnr_db"] for r in valid]),
                    "pooled_mse_recovered_from_saved_psnr": pooled / denominator if denominator else None,
                    "pooled_psnr_db_recovered_from_saved_psnr": psnr(pooled / denominator) if denominator else None,
                    "valid_frames": len(valid),
                    "valid_times_seconds": [r["nominal_time_seconds"] for r in valid],
                    "warning": "saved historical PSNR; recovered MSE is not new float-render recomputation",
                }
            summary[f"{dev}_{variant}"] = stats
    write_csv(out / "input_fit_existing.csv", rows)
    paired = []
    lookup = {(r["dev"], r["variant"], r["nominal_time_seconds"]): r for r in rows}
    for c1 in camera1_rows:
        c0 = lookup.get((c1["dev"], c1["variant"], c1["nominal_time_seconds"]))
        if c0 is None:
            continue
        row = {"dev": c1["dev"], "variant": c1["variant"], "nominal_time_seconds": c1["nominal_time_seconds"]}
        for region in ("full", "O"):
            a, b = c0[f"{region}_psnr_db"], c1[f"{region}_psnr_db"]
            row[f"camera0_{region}_psnr_db"] = a
            row[f"camera1_{region}_psnr_db"] = b
            row[f"camera0_minus_camera1_{region}_psnr_db"] = a - b if a is not None and b is not None else None
        paired.append(row)
    return summary, paired, missing


def build(root, out):
    aux = root / "experiments/aux_ref_object_reconstruction_20260924/run01"
    v2 = root / "experiments/surface_observation_fusion_20260926/run01"
    region_path = aux / "evaluation/regions/manifest.json"
    metrics_path = v2 / "evaluation/per_frame.json"
    regions, historical = read_json(region_path), read_json(metrics_path)
    assert regions["status"] == "frozen" and regions["role"] == "evaluation_only"
    lookup = {(r["dev"], r["sample_id"]): r for r in regions["rows"]}
    assert len(historical) == 45 and len(lookup) == 9
    sources = [identity(region_path), identity(metrics_path)]
    for name in ("summary.json", "HS_metrics.json", "paired_differences.json"):
        sources.append(identity(v2 / "evaluation" / name))
    cache, rows, checks = {}, [], []
    for old in historical:
        e = lookup[old["dev"], old["sample_id"]]
        assert old["time"] == e["query_time_seconds"]
        key = (e["dev"], e["frame_id"])
        if key not in cache:
            for name in ("rgb", "regions"):
                assert sha256(e[name]["path"]) == e[name]["sha256"], e[name]
                sources.append(identity(e[name]["path"]))
            gt_bgr = cv2.imread(e["rgb"]["path"], cv2.IMREAD_COLOR)
            assert gt_bgr is not None
            gt = gt_bgr[..., ::-1].astype(np.float64) / 255.0
            with np.load(e["regions"]["path"], allow_pickle=False) as data:
                labels = data["entity_labels"].copy()
            assert set(np.unique(labels).tolist()).issubset({0, 1, 2})
            assert gt.shape[:2] == labels.shape
            cache[key] = (gt, labels)
        gt, labels = cache[key]
        pred_path = v2 / "evaluation" / e["dev"] / e["frame_id"] / f"{old['variant']}.npz"
        with np.load(pred_path, allow_pickle=False) as data:
            pred = data["rgb"].astype(np.float64)
        assert pred.shape == gt.shape and np.isfinite(pred).all()
        assert float(pred.min()) >= 0 and float(pred.max()) <= 1
        sources.append(identity(pred_path))
        squared = np.mean((pred - gt) ** 2, axis=-1)
        sse, count = float(squared.sum()), int(squared.size)
        row = {
            "dev": e["dev"], "variant": old["variant"], "camera_id": 1,
            "frame_id": e["frame_id"], "sample_id": e["sample_id"],
            "nominal_time_seconds": e["query_time_seconds"],
            "full_pixels": count, "full_sse": sse, "full_mse": sse / count,
            "full_psnr_db": psnr(sse / count),
        }
        for name, (label, original_name) in REGIONS.items():
            mask = labels == label
            n = int(mask.sum())
            se = float(squared[mask].sum()) if n else None
            mse = se / n if n else None
            row.update({f"{name}_pixels": n, f"{name}_pixel_fraction": n / count,
                        f"{name}_sse": se, f"{name}_mse": mse,
                        f"{name}_psnr_db": psnr(mse) if n else None,
                        f"{name}_error_share": se / sse if n and sse else None})
            assert n == old["metrics"][original_name]["pixels"]
            if n:
                checks.append(abs(row[f"{name}_psnr_db"] - old["metrics"][original_name]["psnr_db"]))
        assert sum(row[f"{r}_pixels"] for r in REGIONS) == count
        assert np.isclose(sum(row[f"{r}_sse"] or 0 for r in REGIONS), sse, rtol=1e-12, atol=1e-8)
        checks.append(abs(row["full_psnr_db"] - old["metrics"]["full"]["psnr_db"]))
        residual = sse - (row["O_sse"] or 0)
        row.update({"algebraic_O_zero_full_mse": residual / count,
                    "algebraic_O_zero_full_psnr_db": psnr(residual / count),
                    "algebraic_O_zero_full_gain_db": psnr(residual / count) - row["full_psnr_db"],
                    "prediction_path": str(pred_path), "GT_path": e["rgb"]["path"],
                    "regions_path": e["regions"]["path"]})
        rows.append(row)
    assert max(checks) < 1e-9, max(checks)
    write_csv(out / "existing_error_budget.csv", rows)
    aggregates = {}
    for dev in ("dev1", "dev2"):
        for variant in VARIANTS:
            group = [r for r in rows if r["dev"] == dev and r["variant"] == variant]
            assert len(group) == (5 if dev == "dev1" else 4)
            total_sse, total_pixels = sum(r["full_sse"] for r in group), sum(r["full_pixels"] for r in group)
            remainder = sum(r["full_sse"] - (r["O_sse"] or 0) for r in group)
            aggregates[f"{dev}_{variant}"] = {
                "dev": dev, "variant": variant, "times_seconds": [r["nominal_time_seconds"] for r in group],
                "frames": len(group), "full": {
                    "mean_frame_psnr_db": finite_mean([r["full_psnr_db"] for r in group]),
                    "pooled_mse": total_sse / total_pixels, "pooled_psnr_db": psnr(total_sse / total_pixels),
                    "sse_rgb_mean_summed": total_sse, "pixels_summed": total_pixels,
                },
                **{name: region_aggregate(group, name) for name in REGIONS},
                "algebraic_O_zero": {
                    "pooled_full_psnr_db": psnr(remainder / total_pixels),
                    "pooled_full_gain_db": psnr(remainder / total_pixels) - psnr(total_sse / total_pixels),
                    "mean_frame_full_psnr_db": finite_mean([r["algebraic_O_zero_full_psnr_db"] for r in group]),
                    "mean_frame_full_gain_db": finite_mean([r["algebraic_O_zero_full_gain_db"] for r in group]),
                },
            }
    input_summary, paired, missing = collect_camera0(root, out, rows, sources)
    summary = {
        "status": "completed_zero_training_existing_outputs_only", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_renders": 0, "optimization_steps": 0, "GPU_used": False,
        "definitions": {
            "error": "mean_RGB((float_prediction-RGB_uint8_GT/255)^2)",
            "SSE": "sum of pixel errors; RGB channel mean, not channel sum",
            "full_MSE": "sum(H_SSE,O_SSE,S_SSE)/full_pixels",
            "region_MSE": "region_SSE/region_pixels; empty region is null/CSV blank",
            "error_share": "region_SSE/full_SSE; aggregate share uses summed SSE, not averaged dB",
            "mean_frame_PSNR": "equal weight over nonempty frames",
            "pooled_PSNR": "-10 log10(sum_region_SSE/sum_region_pixels)",
            "algebraic_O_zero": "Set pixel error inside fixed O to zero, keep H/S errors unchanged. Diagnostic only, not model output or physical upper bound: real object changes can affect pixels outside O.",
            "PSNR_floor": "MSE floor 1e-12, matching history; no current row reaches it",
            "camera0_camera1_pairing": "same official nominal sample times only, not verified exact exposure times; full/O descriptive comparisons remain affected by region/input-image differences",
            "scope": "B0/B1/F0/F1/F2 use Ref object motion, sparse object supervision and frozen H/S. They are auxiliary conditions, not complete original S1/E0 or future 4DGS comparison.",
            "regions": "H=1 O=2 S=0, original fit-derived camera1 labels; disjoint exhaustive full image, never shrink denominator",
        },
        "verification": {"float_render_rows": len(rows), "unique_GT_frames": len(cache),
                         "compared_historical_PSNR_values": len(checks), "maximum_absolute_PSNR_difference_db": max(checks),
                         "all_partitions_exhaustive": True, "fixed_region_and_GT_hashes_match": True},
        "camera1_existing": aggregates, "camera0_existing": input_summary,
        "camera0_camera1_same_nominal_time": paired,
        "missing_assets": missing,
        "missing_statistics": ["camera0 H/S metrics absent for every historical variant", "camera0 full absent for B0", "complete original S1/E0 metrics are not these AUX statistics"],
        "historical_HS_metrics": {"source": str(v2 / "evaluation/HS_metrics.json"), "rows": len(read_json(v2 / "evaluation/HS_metrics.json")),
                                  "meaning": "HS-only actual model render, distinct from algebraic O-zero diagnostic; kept as historical context"},
        "sources": sources,
    }
    (out / "existing_error_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_report(out, summary, rows)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "max_psnr_difference_db": max(checks),
                      "output": str(out)}, ensure_ascii=False))


def write_report(out, summary, rows):
    a = summary["camera1_existing"]
    b = summary["camera0_existing"]
    text = ["# V3 工作包 A：已有误差归因（零训练）", "",
            "本次直接读取历史 V2 的 45 个浮点 RGB 输出及原 AUX 固定区域/GT，用 CPU 重算误差；没有加载检查点、重新渲染或优化。180 个 full/H/O/S PSNR 与历史未四舍五入结果的最大差为 " + f"{summary['verification']['maximum_absolute_PSNR_difference_db']:.3g} dB。", "",
            "## 主要证据", "",
            "固定 camera1 区域中，整图误差主要落在背景 S：下表按整个事件累加 SSE 再求占比。它说明误差发生的位置，尚不能单独区分观察覆盖不足、优化不足或表示缺陷。物体自身低分仍然存在，不能因为区域小而忽略。", "",
            "| 事件 | 版本 | 全图逐帧均值 dB | 全图 pooled dB | H 误差份额 % | O 误差份额 % | S 误差份额 % | O逐帧均值 dB | O pooled dB | O误差归零的代数全图增量 dB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in a.items():
        text.append(f"| {row['dev']} | {row['variant']} | {row['full']['mean_frame_psnr_db']:.6f} | {row['full']['pooled_psnr_db']:.6f} | {100*row['H']['pooled_error_share']:.4f} | {100*row['O']['pooled_error_share']:.4f} | {100*row['S']['pooled_error_share']:.4f} | {row['O']['mean_frame_psnr_db']:.6f} | {row['O']['pooled_psnr_db']:.6f} | {row['algebraic_O_zero']['pooled_full_gain_db']:.6f} |")
    text += ["", "这里的 SSE 定义为逐像素 RGB 均方误差的总和。全图 MSE＝(SSE_H＋SSE_O＋SSE_S)/整图像素数；PSNR＝−10 log10(MSE)。不可直接按像素数加权 dB。逐帧均值先算每帧 dB 再等权平均；pooled 指先把误差与像素累加再转 dB，区域面积变化时差异尤其明显。", "",
             "O 误差归零仅是代数诊断：令固定 O 中误差为零、保留 H/S 原误差。它不是实际模型结果，也不是物体改进的物理上界，因为真实物体更新可能影响 O 外像素。HS-only 是真实去物体渲染，与这一诊断也不同。", "",
             "## 训练相机拟合与时间集合", "",
             "下表只汇总已保存 camera0 结果，属于训练集拟合。B0 仅保存 O；B1/F0/F1/F2 保存 full/O；所有版本缺 H/S，均记 NA，不填零。重建 MSE 由已存 PSNR 反推，并非重新验证浮点前向。", "",
             "| 事件 | 版本 | 帧数 | full逐帧均值 dB | full pooled dB | O有效帧 | O逐帧均值 dB | O pooled dB |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "NA" if v is None else f"{v:.6f}"
    for key, row in b.items():
        dev, variant = key.split("_")
        text.append(f"| {dev} | {variant} | {row['frame_count']} | {fmt(row['full']['mean_frame_psnr_db'])} | {fmt(row['full']['pooled_psnr_db_recovered_from_saved_psnr'])} | {row['O']['valid_frames']} | {fmt(row['O']['mean_frame_psnr_db'])} | {fmt(row['O']['pooled_psnr_db_recovered_from_saved_psnr'])} |")
    for dev in ("dev1", "dev2"):
        text += ["", f"- {dev} camera0 原生名义时刻：{b[dev+'_B1']['times_seconds']}；camera1 固定 E：{a[dev+'_B1']['times_seconds']}。"]
    text += ["", "完整列表保留在 input_fit_existing.csv；JSON 中另给 E 同名义时刻的已有 full/O 成对值。不能将两个不同时间集合的均值差写成纯视角损失；即使同名义时刻，实际曝光同步误差未给出，camera0 O 来自输入 SAM2、camera1 O 来自发布拟合区域，两者语义与处理仍有区别。dev1 t21 的 camera0 O 为空，其 O 配对差为 NA，camera1 的该帧照常保留。", "",
             "## 结论与限制", "",
             "已有辅助模型在输入视角 full 拟合明显好于保留视角，但人体/背景此前用过完整 S1 的 114/98 输入帧；12/9 只是最近物体实验的原生时刻集合，不能描述成整个系统只训练了这么多帧。以上模型包含 Ref 物体运动和固定 H/S，不能替代 E0 完整旧 S1，也不能与 E1/E2 当作同输入排行榜。", "",
             "下一步应恢复完整 S1 的原生时间一致性、输入拟合与固定 E 渲染，并运行共享合法先验初始化的官方 4DGS 对照，才有条件区分系统拟合与输入覆盖问题。本统计未证实新的交互机制，也不改变 V2 的停止结论。", "",
             "camera1 区域是发布拟合派生 H/O/S；其几何支持自洽不能当独立真实可见性真值。所有 9 帧及 t26 的 200 像素 O 均保留。", "",
             "## 可复算文件", "",
             "- existing_error_budget.csv：45 行逐事件/帧/版本的像素、MSE、SSE、误差份额及代数诊断。", "- existing_error_summary.json：10 个事件版本汇总、180 项历史一致性检查、输入拟合与成对名义时刻、全部所用来源哈希。", "- input_fit_existing.csv：105 行历史训练相机拟合与来源/时间，空字段表示缺失或空区域。", "- code/summarize_existing.py：仅 NumPy/OpenCV 的 CPU 统计入口；在项目根运行 envs/gvhmr/bin/python experiments/baseline_protocol_calibration_20260927/run01/code/summarize_existing.py。", ""]
    (out / "ERROR_ATTRIBUTION.md").write_text("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build(args.root, args.output)
