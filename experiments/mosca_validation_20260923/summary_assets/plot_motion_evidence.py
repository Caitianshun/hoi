#!/usr/bin/env python3
"""CPU-only plots of frozen dyn_o counterfactuals against sparse fitted reference.

No model execution, reference interpolation, visibility filtering or alignment.
All input files are read-only. Their hashes are checked before and after plotting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
COLORS = {"gate_on": "#cb543e", "gate_off": "#247fa5", "reference": "#242b38"}
LABELS = {"gate_on": "gate ON：原冻结模型", "gate_off": "gate OFF：冻结反事实"}
SELECTED = ["object_front_brown", "hand_glove_centre"]
TITLES = {"object_front_brown": "箱体棕色表面查询点", "hand_glove_centre": "手套中心查询点"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-dir", type=Path, default=ROOT / "experiments/mosca_validation_20260923/motion_binding")
    p.add_argument("--reference", type=Path, default=ROOT / "experiments/mosca_baseline_20260922/evaluation/fixed_rgb_queries/same_version_joint_reference_observation_times.npz")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {key: args.motion_dir / f"{key}_fixed_source_prediction.npz" for key in LABELS}
    paths["reference"] = args.reference
    paths["freeze_manifest"] = args.motion_dir / "prediction_freeze_manifest.json"
    for key in LABELS:
        paths[key + "_protocol"] = args.motion_dir / f"{key}_prediction_protocol.json"
    input_hashes = {str(v.resolve()): sha256(v) for v in paths.values()}
    freeze = json.loads(paths["freeze_manifest"].read_text())
    for key in LABELS:
        assert sha256(paths[key]) == freeze[paths[key].name], f"Frozen prediction hash mismatch: {key}"
        assert sha256(paths[key + "_protocol"]) == freeze[paths[key + "_protocol"].name]
    data = {k: load_npz(paths[k]) for k in ["gate_on", "gate_off", "reference"]}
    ref = data["reference"]
    times = data["gate_on"]["frame_times"]
    rt = ref["frame_times"]
    ri = ref["prediction_frame_indices"]
    queries = data["gate_on"]["query_id"].tolist()
    assert len(set(queries)) == len(queries)
    assert np.all(np.diff(times) > 0) and np.all(np.diff(rt) > 0)
    assert np.allclose(times[ri], rt, atol=1e-9, rtol=0), "Use independent exact source-RGB association, never nearest time"
    assert ri[0] == 0, "Displacement anchor must be shared first observation"
    assert ref["query_id"].tolist() == queries
    assert ref["coordinate_frame"].item() == "behave_world_k1_color" and ref["units"].item() == "m"
    for k in LABELS:
        z = data[k]
        assert z["query_id"].tolist() == queries
        assert z["coordinate_frame"].item() == ref["coordinate_frame"].item()
        assert z["units"].item() == "m"
        assert np.array_equal(z["frame_times"], times)
        assert z["predicted"].shape == (len(times), len(queries), 3)
        assert np.all(np.isfinite(z["predicted"][z["predicted_valid_mask"]]))

    cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(cjk)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    stats, curves = {}, {}
    for qi, q in enumerate(queries):
        r = ref["reference"][:, qi]
        rv = ref["valid_mask"][:, qi] & np.isfinite(r).all(axis=-1)
        assert rv[0]
        ref_disp = np.linalg.norm(r - r[0], axis=-1)
        stats[q] = {"reference_max_sampled_displacement_m": float(ref_disp[rv].max()),
                    "reference_valid_count": int(rv.sum())}
        curves[q] = {"reference_displacement": ref_disp, "reference_valid": rv}
        for key in LABELS:
            z = data[key]
            pred = z["predicted"][:, qi]
            pv = z["predicted_valid_mask"][:, qi] & np.isfinite(pred).all(axis=-1)
            assert pv[0]
            disp = np.linalg.norm(pred - pred[0], axis=-1)
            valid = rv & pv[ri]
            # Subtracting each series' first position is the displacement metric,
            # not an alignment of absolute predictions used for the error below.
            absolute = np.linalg.norm(pred[ri] - r, axis=-1)
            delta_error = np.linalg.norm((pred[ri] - pred[0]) - (r - r[0]), axis=-1)
            nonanchor = valid.copy()
            nonanchor[0] = False
            stats[q][key] = {
                "predicted_valid_count": int(pv.sum()),
                "max_displacement_all_114_m": float(disp[pv].max()),
                "max_displacement_common_reference_times_m": float(disp[ri][valid].max()),
                "absolute_error_mean_m": float(absolute[valid].mean()),
                "absolute_error_median_m": float(np.median(absolute[valid])),
                "absolute_error_first_m": float(absolute[0]),
                "absolute_error_last_m": float(absolute[-1]),
                "displacement_vector_error_mean_nonanchor_m": float(delta_error[nonanchor].mean()),
                "absolute_common_count": int(valid.sum()),
                "displacement_nonanchor_common_count": int(nonanchor.sum()),
            }
            curves[q][key] = {"displacement": disp, "absolute": absolute,
                              "delta_error": delta_error, "valid": valid, "pred_valid": pv}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.3), sharex=True)
    fig.subplots_adjust(left=.062, right=.983, bottom=.155, top=.79, wspace=.24, hspace=.36)
    fig.suptitle("恢复运动幅度，不等于恢复三维精度", x=.5, y=.97, fontsize=23, fontweight="bold")
    fig.text(.5, .925, "只改冻结模型的 dyn_o 门控；不重训、不改首帧合成权重、不对齐参考", ha="center", fontsize=13)
    for row, q in enumerate(SELECTED):
        c = curves[q]
        ax = axes[row, 0]
        for key in LABELS:
            pred_disp = np.where(c[key]["pred_valid"], c[key]["displacement"], np.nan)
            ax.plot(times, pred_disp, color=COLORS[key], lw=2.2, label=LABELS[key])
        ax.scatter(rt[c["reference_valid"]], c["reference_displacement"][c["reference_valid"]],
                   color=COLORS["reference"], marker="D", s=37, label="同版拟合参考：14 个时刻", zorder=5)
        for col, field in [(1, "absolute"), (2, "delta_error")]:
            for key, marker in [("gate_on", "o"), ("gate_off", "^")]:
                valid = c[key]["valid"].copy()
                if col == 2:
                    valid[0] = False  # The anchor zero is tautological, not evidence.
                axes[row, col].scatter(rt[valid], c[key][field][valid], color=COLORS[key],
                                      marker=marker, s=45, alpha=.92, zorder=4)
        for col, heading in enumerate(["相对各自首帧的位置变化", "绝对位置误差（无对齐）", "位移向量误差（不计锚点）"]):
            a = axes[row, col]
            a.set_title(TITLES[q] + "\n" + heading, fontsize=13, pad=10)
            a.set_ylabel("米")
            # Keep numeric annotations above every plotted sample.
            a.set_ylim(0, a.get_ylim()[1] * 1.2)
            a.grid(True, alpha=.17)
            a.set_xlim(times[0] - .25, times[-1] + .25)
            a.set_xticks(np.arange(18, 32, 2))
            if row == 1:
                a.set_xlabel("原视频实际观测时间（秒）")
        s = stats[q]
        lines = [f"最大位移 ON / OFF：{s['gate_on']['max_displacement_all_114_m']:.3f} / {s['gate_off']['max_displacement_all_114_m']:.3f} m",
                 f"参考 14 时刻最大：{s['reference_max_sampled_displacement_m']:.3f} m"]
        axes[row, 0].text(.03, .96, "\n".join(lines), transform=axes[row, 0].transAxes,
                          va="top", fontsize=10, bbox={"facecolor":"white", "alpha":.82, "edgecolor":"none"})
        axes[row, 1].text(.03, .96,
                         f"14 时刻均值 ON / OFF：{s['gate_on']['absolute_error_mean_m']:.3f} / {s['gate_off']['absolute_error_mean_m']:.3f} m",
                         transform=axes[row, 1].transAxes, va="top", fontsize=10,
                         bbox={"facecolor":"white", "alpha":.82, "edgecolor":"none"})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .894), ncol=3, frameon=False, fontsize=12)
    fig.text(.062, .083,
             "左列连接 114 个预测样本，仅便于看趋势；114 个坐标不代表真值可见或材料表面身份正确。参考只画 14 个点，不插值。",
             fontsize=11)
    fig.text(.062, .053,
             "中列直接比较世界坐标；右列比较从各自首帧出发的三维位移向量，不能消除或替代中列的初始位置误差。",
             fontsize=11)
    fig.text(.062, .023,
             "参考为同版人体／物体拟合表面，非传感器真值；源 RGB 对应时间已核验，拟合名义时间仍有最大 103.745 ms 偏差。手套拟合另有表面对应不确定性。",
             fontsize=10, color="#555555")
    main_png = args.output / "motion_amplitude_and_independent_error.png"
    fig.savefig(main_png, dpi=165)
    plt.close(fig)

    fig = plt.figure(figsize=(14.5, 8.5))
    fig.suptitle("同一世界坐标中的轨迹：预测曲线与稀疏拟合点", y=.965, fontsize=20, fontweight="bold")
    fig.text(.5, .917, "冻结 ON / OFF 反事实；无旋转、平移或尺度对齐；参考点之间不连线", ha="center", fontsize=12)
    for i, q in enumerate(SELECTED):
        a = fig.add_subplot(1, 2, i + 1, projection="3d")
        qi = queries.index(q)
        all_xyz = []
        for key in LABELS:
            xyz = data[key]["predicted"][:, qi]
            valid = curves[q][key]["pred_valid"]
            display = np.where(valid[:, None], xyz, np.nan)
            a.plot(*display.T, lw=1.7, color=COLORS[key], label=LABELS[key])
            a.scatter(*xyz[0], color=COLORS[key], s=95, marker="*", zorder=6)
            all_xyz.append(xyz[valid])
        xyz = ref["reference"][:, qi]
        valid = curves[q]["reference_valid"]
        a.scatter(*xyz[valid].T, color=COLORS["reference"], s=31, marker="D", label="14 时刻拟合参考")
        a.scatter(*xyz[0], color=COLORS["reference"], s=95, marker="*", zorder=6)
        all_xyz.append(xyz[valid])
        stack = np.concatenate(all_xyz)
        centre = (stack.max(axis=0) + stack.min(axis=0)) / 2
        radius = .55 * np.ptp(stack, axis=0).max()
        a.set_xlim(centre[0] - radius, centre[0] + radius)
        a.set_ylim(centre[1] - radius, centre[1] + radius)
        a.set_zlim(centre[2] - radius, centre[2] + radius)
        a.set_box_aspect((1, 1, 1))
        a.set_xlabel("世界 X（m）")
        a.set_ylabel("世界 Y（m）")
        a.set_zlabel("世界 Z（m）")
        a.set_title(TITLES[q] + "\n星号为各自首帧位置", fontsize=13)
        a.view_init(elev=20, azim=-57)
        if i == 0:
            handles3d, labels3d = a.get_legend_handles_labels()
    fig.legend(handles3d, labels3d, loc="upper center", bbox_to_anchor=(.5, .9),
               ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.025, right=.975, top=.79, bottom=.15, wspace=.1)
    fig.text(.05, .09, "坐标：BEHAVE world k1 color，单位米。为保持真实坐标，未改轴符号或进行对齐；显示视角不是原相机视角。", fontsize=11)
    fig.text(.05, .052, "轨迹长度／形状不是独立精度指标。稀疏拟合点未覆盖所有时刻，也未提供准确可见性或接触真值。", fontsize=11)
    trajectory_png = args.output / "motion_world_trajectories.png"
    fig.savefig(trajectory_png, dpi=165)
    plt.close(fig)

    summary = {
        "status": "completed_cpu_only", "selected_plot_queries": SELECTED,
        "coordinate_frame": "behave_world_k1_color", "units": "m",
        "counterfactual_only": True, "retraining": False, "alignment": "none",
        "reference_interpolation": False, "visibility_filtering": False,
        "prediction_count": len(times), "reference_count": len(rt),
        "prediction_actual_times_seconds": times.tolist(),
        "reference_actual_times_seconds": rt.tolist(),
        "reference_prediction_frame_indices": ri.tolist(),
        "reference_max_nominal_time_offset_seconds": float(np.abs(ref["source_image_time_offset_seconds"]).max()),
        "displacement_definition": "Euclidean norm of position minus each series' own first position; absolute comparison is never aligned.",
        "displacement_vector_error_definition": "Norm of (prediction_t-prediction_first) minus (reference_t-reference_first); anchor excluded from means and error scatter.",
        "reference_note": "Same-version fitted surface with fixed triangle/barycentric identities, not sensor ground truth; no reliable ground-truth visibility.",
        "predicted_note": "114 fixed-source Gaussian-weight transport coordinates do not guarantee actual visibility or material surface correspondence.",
        "input_sha256": input_hashes, "per_query": stats,
        "script_sha256": sha256(Path(__file__)),
    }
    summary_path = args.output / "motion_evidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    assert input_hashes == {str(v.resolve()): sha256(v) for v in paths.values()}, "Read-only inputs changed during plotting"
    qa = {
        "status": "numeric_checks_passed_visual_review_pending",
        "frozen_prediction_manifest_hashes_match": True,
        "input_files_unchanged": True, "all_query_orders_match": True,
        "exact_actual_timestamp_mapping_checked": True,
        "reference_interpolated": False, "reference_alignment": "none",
        "all_114_predictions_finite_and_valid": bool(all(np.all(data[k]["predicted_valid_mask"]) for k in LABELS)),
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {str(v.resolve()): sha256(v) for v in [main_png, trajectory_png, summary_path]},
    }
    (args.output / "motion_evidence_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"figures": [str(main_png), str(trajectory_png)], "selected_metrics": {q: stats[q] for q in SELECTED}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
