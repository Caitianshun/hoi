#!/usr/bin/env python3
"""CPU check that actual saved Adam groups follow the official LR schedule.

The checkpoint stores the LR used by the next update. Stage 1 updates LR
before Adam.step; stages 2/3 update it after Adam.step. In each case, after
g completed updates the saved schedule argument must be g-1.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys


STAGES = {
    1: "1st_State-Conditional_Scene",
    2: "2nd_State_Conditional_Human-Object",
    3: "3rd_Complete_HOSNeRF",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def defaults_from_source(path):
    tree = ast.parse(path.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LitMipNeRF360")
    init = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    return {argument.arg: ast.literal_eval(default) for argument, default in zip(init.args.args[-len(init.args.defaults):], init.args.defaults)}


def audit(args):
    import torch
    import yaml
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = payload["benchmark_identity"]
    stage, global_step = identity["stage"], int(payload["global_step"])
    if global_step < 1 or len(payload["optimizer_states"]) != 1:
        raise ValueError("A native checkpoint with completed Adam updates is required")
    if args.expect_step is not None and global_step != args.expect_step:
        raise ValueError("Checkpoint global_step differs from the requested audit point")
    source = args.official / STAGES[stage] / "src/model/mipnerf360/model.py"
    if sha(source) != identity["official_model_sha256"]:
        raise ValueError("Official source changed after the checkpoint was made")
    index = global_step - 1
    optimizer = payload["optimizer_states"][0]
    settings = {}
    if stage == 1:
        settings = defaults_from_source(source)
        # The native harness passes only basedir to this non-gin-configurable
        # constructor, so these explicit source defaults are authoritative.
        delay = 1.0
        if settings["lr_delay_steps"] > 0:
            ratio = min(max(index / settings["lr_delay_steps"], 0), 1)
            delay = settings["lr_delay_mult"] + (1-settings["lr_delay_mult"]) * math.sin(.5*math.pi*ratio)
        fraction = min(max(index / identity["max_steps"], 0), 1)
        reference = delay * math.exp(math.log(settings["lr_init"])*(1-fraction) + math.log(settings["lr_final"])*fraction)
        schedule = dict(kind="delayed logarithmic interpolation", argument=index,
                        denominator=identity["max_steps"], **{name: settings[name] for name in ["lr_init", "lr_final", "lr_delay_steps", "lr_delay_mult"]})
    else:
        directories = [args.config_dir] if args.config_dir else [args.checkpoint.parent, args.checkpoint.parent.parent]
        config_path = next((directory / "resolved.yaml" for directory in directories if (directory / "resolved.yaml").is_file()), None)
        if config_path is None:
            raise FileNotFoundError("The checkpoint's saved resolved.yaml is required")
        config = yaml.safe_load(config_path.read_text())
        settings = config["train"]
        decay_steps = settings["lrate_decay"] * 1000
        decay = .1 ** (index / decay_steps)
        custom = [name[3:] for name in settings if name.startswith("lr_")]
        schedule = dict(kind="per-group exponential decay", argument=index,
                        denominator=decay_steps, decay_factor=decay,
                        config_path=str(config_path), config_sha256=sha(config_path))
    rows = []
    for group_index, group in enumerate(optimizer["param_groups"]):
        name = group.get("name", "stage1_all_parameters")
        if stage == 1:
            expected = reference
            base = settings["lr_init"]
        else:
            matches = [key for key in custom if key in name]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous LR matches need original cfg ordering review: {name}: {matches}")
            base = settings["lr_" + matches[0]] if matches else settings["lr" if stage == 2 else "lr_bkgd"]
            expected = base * decay
        actual = float(group["lr"])
        passed = math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-15)
        rows.append(dict(group_index=group_index, name=name, base_lr=float(base),
                         expected_lr=expected, actual_lr=actual, passed=passed,
                         relative_error=(actual/expected-1) if expected else None))
    result = dict(status="passed" if all(row["passed"] for row in rows) else "failed",
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        stage=stage, scene=identity["scene"], global_step=global_step,
        official_model_sha256=identity["official_model_sha256"],
        groups=len(rows), failed_groups=sum(not row["passed"] for row in rows),
        schedule=schedule, group_audit=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps({name: value for name, value in result.items() if name != "group_audit"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official", type=Path, default=Path(__file__).resolve().parents[4] / "third_party/HOSNeRF")
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--expect-step", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-only", action="store_true", help="Record negative evidence without a failing exit code")
    arguments = parser.parse_args()
    for name in ["checkpoint", "official", "config_dir", "output"]:
        value = getattr(arguments, name)
        if value is not None:
            setattr(arguments, name, value.expanduser().resolve())
    outcome = audit(arguments)
    if outcome["status"] != "passed" and not arguments.report_only:
        sys.exit(2)
