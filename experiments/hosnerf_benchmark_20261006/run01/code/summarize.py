#!/usr/bin/env python3
"""Summarize local native-HOSNeRF endpoints; absent metrics remain NA.

Published numbers are read from a private provenance file, never embedded here.
No training, rendering, network request or test-based checkpoint selection runs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
SCENES = ["Backpack", "Tennis", "Suitcase", "Playground", "Dance", "Lounge"]
STEPS = {1: 500000, 2: 400000, 3: 200000}


def read(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def stage_state(run, scene, stage):
    directory = run / "runs/formal" / scene / f"stage{stage}"
    receipt = read(directory / "receipt.json")
    current = read(directory / "status.json")
    current_is_authoritative = bool(current and current.get("status") in {"failed", "running"})
    state = current if current_is_authoritative else (receipt or current)
    if state is None:
        return dict(stage=stage, prescribed_steps=STEPS[stage], status="not_started", global_step=0)
    # A failed or active invocation may not report a committed step yet. Keep
    # the previous successful segment's receipt as the last confirmed step;
    # it must not override the current failure/running state.
    confirmed = receipt if current_is_authoritative and receipt else state
    step = int(confirmed.get("global_step", 0))
    status = state.get("status", "unknown")
    if step < STEPS[stage] and status == "completed":
        status = "incomplete_segment"
    result = dict(stage=stage, prescribed_steps=STEPS[stage], global_step=step,
                  last_confirmed_global_step=step, status=status,
                  source_status=str(directory / "status.json") if current_is_authoritative else str(directory / "receipt.json") if receipt else str(directory / "status.json"))
    if current_is_authoritative and receipt:
        result["last_confirmed_receipt"] = str(directory / "receipt.json")
    if status == "failed":
        result["failure"] = state.get("error", "See the stage status.json traceback")
    return result


def summarize(run):
    dataset = read(run / "protocol/dataset_manifest.json") or {}
    reference = read(run / "protocol/published_reference.json") or {}
    pipeline = read(run / "pipeline.json")
    rows = []
    local_rows = []
    for scene in SCENES:
        stages = [stage_state(run, scene, stage) for stage in (1, 2, 3)]
        scene_data = dataset.get("scenes", {}).get(scene, {})
        evaluation = read(run / "evaluation" / scene / "metrics.json")
        valid = bool(evaluation and evaluation.get("status") == "completed"
                     and evaluation.get("source_kind") == "local_retrained"
                     and evaluation.get("frames") == 16
                     and evaluation.get("identity", {}).get("strict_load")
                     and evaluation.get("identity", {}).get("global_step") == STEPS[3])
        if valid:
            identity = evaluation["identity"].get("checkpoint_benchmark_identity") or {}
            valid = identity.get("formal", False) and identity.get("scene") == scene and identity.get("stage") == 3
            valid = valid and evaluation["identity"]["test_ids"] == scene_data.get("test_ids")
        failure = read(run / "evaluation" / scene / "evaluation_failure.json")
        status = "completed" if valid else ("evaluation_failed" if failure else "awaiting_endpoint")
        if not scene_data or scene_data.get("status") != "prepared":
            status = "missing_or_unprepared_dataset"
        elif any(stage["status"] == "failed" for stage in stages):
            status = "training_failed"
        metrics = evaluation["mean_metrics"] if valid else {key: None for key in ["PSNR", "SSIM", "LPIPS"]}
        row = dict(scene=scene, source_kind="local_retrained", status=status,
                   data_status=scene_data.get("status", "missing"), stages=stages,
                   frames=16 if valid else None, metrics=metrics,
                   evaluation=str(run / "evaluation" / scene / "metrics.json") if valid else None,
                   checkpoint_sha256=evaluation["identity"]["checkpoint_sha256"] if valid else None)
        local_rows.append(row)
        rows.append(dict(scene=scene, source_kind="local_retrained", status=status,
                         PSNR=metrics["PSNR"], SSIM=metrics["SSIM"], LPIPS_VGG=metrics["LPIPS"],
                         frames=16 if valid else None))
        paper = reference.get("scenes", {}).get(scene)
        if paper:
            rows.append(dict(scene=scene, source_kind="published_reference", status="reported_by_paper",
                             PSNR=paper.get("PSNR"), SSIM=paper.get("SSIM"), LPIPS_VGG=paper.get("LPIPS_VGG"), frames=None))
    finished = sum(row["status"] == "completed" for row in local_rows)
    aggregate = {key: (sum(row["metrics"][key] for row in local_rows) / len(SCENES)
                       if finished == len(SCENES) else None)
                 for key in ["PSNR", "SSIM", "LPIPS"]}
    rows.append(dict(scene="ALL_SIX_MEAN", source_kind="local_retrained",
                     status="completed" if finished == len(SCENES) else "awaiting_all_six_endpoints",
                     PSNR=aggregate["PSNR"], SSIM=aggregate["SSIM"], LPIPS_VGG=aggregate["LPIPS"],
                     frames=16 * len(SCENES) if finished == len(SCENES) else None))
    result = dict(status="completed" if finished == len(SCENES) else "incomplete",
                  requested_scenes=SCENES, completed_scenes=finished,
                  updated_at_utc=datetime.now(timezone.utc).isoformat(),
                  pipeline=pipeline, local_retrained=local_rows,
                  local_equal_scene_mean=dict(status="completed" if finished == len(SCENES) else "awaiting_all_six_endpoints",
                                             completed_scenes=finished, required_scenes=len(SCENES),
                                             aggregation="equal-weight arithmetic mean of all six scene means; no partial-scene average",
                                             metrics=aggregate),
                  published_reference=reference if reference else None,
                  note="Missing local endpoint metrics are null/NA. Published references are not locally trained results. VGG LPIPS and corrected spatial SSIM are explicit.")
    write_json(run / "summary.json", result)
    with (run / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene", "source_kind", "status", "PSNR", "SSIM", "LPIPS_VGG", "frames"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})
    lines = ["# HOSNeRF 本机基准状态", "", f"本机正式终态完成：{finished}/{len(SCENES)} 场景。缺失指标保持 NA；论文报告值不计入本机完成数量。", "",
             "| 场景 | 状态 | stage1/2/3 当前步 | 本机 PSNR | 本机 SSIM | 本机 VGG-LPIPS |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in local_rows:
        values = ["NA" if row["metrics"][key] is None else f"{row['metrics'][key]:.6f}" for key in ["PSNR", "SSIM", "LPIPS"]]
        steps = "/".join(str(stage["global_step"]) for stage in row["stages"])
        lines.append(f"| {row['scene']} | {row['status']} | {steps} | {' | '.join(values)} |")
    aggregate_values = ["NA" if aggregate[key] is None else f"{aggregate[key]:.6f}"
                        for key in ["PSNR", "SSIM", "LPIPS"]]
    lines.append(f"| 六场景等权均值 | {finished}/{len(SCENES)} 终态完成 | — | {' | '.join(aggregate_values)} |")
    lines.extend(["", "SSIM采用原生尺寸H×W×3图像、channel_axis=2、data_range=1，修正官方展平N×3计算；VGG-LPIPS保留官方网络与标量定义。训练仅使用冻结训练帧，按固定官方三阶段预算评价最终终态。", "",
                  "论文参照的来源与未知复跑细节见 protocol/PAPER_PROTOCOL_AUDIT.md；metrics.csv以source_kind区分local_retrained和published_reference。正式训练与先验条件核实前，不计算公平排名差值。", ""])
    (run / "STATUS.md").write_text("\n".join(lines))
    print(json.dumps(dict(status=result["status"], completed_scenes=finished, summary=str(run / "summary.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    summarize(args.run.expanduser().resolve())
