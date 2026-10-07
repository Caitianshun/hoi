#!/usr/bin/env python3
"""CPU audit of an isolated native stage-1 continuation.

Predicts the next single-image batches from the saved epoch RNG, then checks
per-parameter Adam counters against the actual accessed state embeddings.
It reads checkpoints and metadata, never reads RGB pixels or starts CUDA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    before = torch.load(args.before, map_location="cpu", weights_only=False)
    identity = before["benchmark_identity"]
    if identity["stage"] != 1 or len(before["optimizer_states"]) != 1:
        raise ValueError("This audit applies to the native single-Adam stage 1")
    source = args.official / "1st_State-Conditional_Scene"
    if sha(source / "src/model/mipnerf360/model.py") != identity["official_model_sha256"]:
        raise RuntimeError("Official model source differs from the checkpoint identity")
    data = (args.data_root or Path(identity["data_root"])) / identity["scene"]
    for name, digest in identity["metadata_sha256"].items():
        if sha(data / name) != digest:
            raise RuntimeError(f"Dataset metadata changed: {name}")
    os.chdir(source)
    sys.path.insert(0, str(source))
    from src.model.mipnerf360.model import LitMipNeRF360
    model = LitMipNeRF360(basedir=str(data))
    model.load_state_dict(before["state_dict"], strict=True)
    named = list(model.named_parameters())
    optimizer = before["optimizer_states"][0]
    if len(optimizer["param_groups"]) != 1:
        raise RuntimeError("Native stage 1 must have one Adam parameter group")
    parameter_ids = optimizer["param_groups"][0]["params"]
    if len(parameter_ids) != len(named):
        raise RuntimeError("Model and Adam parameter lists differ")
    mapping = {name: int(identifier) for (name, _), identifier in zip(named, parameter_ids)}
    sampling = before["benchmark_sampling"]
    if sampling["committed"] + args.updates > sampling["epoch_length"]:
        raise ValueError("This short audit does not cross a sampler epoch boundary")
    # Reproduce only the small image-choice vector, avoiding the multi-GB
    # per-pixel sampler array. The official sampler chooses images first.
    replay = np.random.RandomState()
    replay.set_state(sampling["epoch_rng"]["numpy"])
    choices = replay.choice(np.arange(len(identity["train_ids"])), sampling["epoch_length"], replace=True)
    chosen = choices[sampling["committed"]:sampling["committed"] + args.updates]
    frames = [identity["train_ids"][int(index)] for index in chosen]
    all_ids = [path.stem for path in sorted((data / "images").glob("*.png"))]
    times = np.linspace(0., 1., len(all_ids)).astype(np.float32)
    time_lookup = dict(zip(all_ids, times))
    expected = {}
    embedding_rows = []
    observed_states = optimizer["state"]
    for level, mlp in enumerate(model.model.mlps):
        transitions = getattr(mlp, "transitions_times", [])
        count = len(mlp.bkgd_stateembeds)
        def active_state(frame):
            time = time_lookup[frame]
            if count == 1:
                return 0
            if time < transitions[0] - 1e-5:
                return 0
            for index in range(1, count - 1):
                if time <= transitions[index] + 1e-5:
                    return index
            return count - 1
        accesses = Counter(active_state(frame) for frame in frames)
        for index in range(count):
            name = f"model.mlps.{level}.bkgd_stateembeds.{index}"
            identifier = mapping[name]
            previous = int(observed_states[identifier]["step"]) if identifier in observed_states else 0
            target = previous + accesses[index]
            expected[identifier] = target
            embedding_rows.append(dict(parameter=name, optimizer_id=identifier,
                before_step=previous if identifier in observed_states else None,
                expected_after_step=target if target else None, new_accesses=accesses[index]))
    shared = []
    for name, identifier in mapping.items():
        if re.search(r"\.bkgd_stateembeds\.\d+$", name):
            continue
        if identifier not in observed_states:
            raise RuntimeError(f"Shared parameter has no saved Adam state: {name}")
        previous = int(observed_states[identifier]["step"])
        expected[identifier] = previous + args.updates
        shared.append(dict(parameter=name, optimizer_id=identifier,
                           before_step=previous, expected_after_step=expected[identifier]))
    result = dict(status="prediction_only", before_checkpoint=str(args.before),
        before_sha256=sha(args.before), before_global_step=int(before["global_step"]),
        expected_after_global_step=int(before["global_step"]) + args.updates,
        expected_frame_ids=frames, shared_parameter_count=len(shared),
        embedding_parameters=embedding_rows, shared_parameters=shared,
        note="State embeddings receive Adam updates only when their time state is accessed. This audit does not claim uninterrupted-training bitwise equivalence.")
    if args.after:
        after = torch.load(args.after, map_location="cpu", weights_only=False)
        for key in ["stage", "scene", "seed", "train_ids", "test_ids", "metadata_sha256",
                    "official_revision", "official_model_sha256", "training_script_sha256",
                    "benchmark_config_sha256", "formal", "max_steps", "workers", "chunk", "netchunk"]:
            if after["benchmark_identity"][key] != identity[key]:
                raise RuntimeError(f"Resumed identity differs: {key}")
        if after["global_step"] != result["expected_after_global_step"]:
            raise RuntimeError("Resume global_step did not increase by the requested update count")
        actual_sampling = after["benchmark_sampling"]
        if actual_sampling["committed"] != sampling["committed"] + args.updates:
            raise RuntimeError("Resumed sampler committed position is wrong")
        for key in ["epoch", "epoch_length"]:
            if actual_sampling[key] != sampling[key]:
                raise RuntimeError(f"Resumed sampler {key} differs")
        first, second = sampling["epoch_rng"], actual_sampling["epoch_rng"]
        if not torch.equal(first["torch"], second["torch"]):
            raise RuntimeError("Sampler epoch-start Torch RNG changed")
        if not all(np.array_equal(a, b) for a, b in zip(first["numpy"], second["numpy"])):
            raise RuntimeError("Sampler epoch-start NumPy RNG changed")
        model.load_state_dict(after["state_dict"], strict=True)
        if len(after["optimizer_states"]) != 1:
            raise RuntimeError("Resumed optimizer count differs")
        actual_optimizer = after["optimizer_states"][0]
        if actual_optimizer["param_groups"][0]["params"] != parameter_ids:
            raise RuntimeError("Resumed Adam parameter mapping changed")
        actual_states = actual_optimizer["state"]
        for identifier, target in expected.items():
            if target == 0:
                if identifier in actual_states:
                    raise RuntimeError(f"Unused embedding unexpectedly has Adam state: {identifier}")
                continue
            if identifier not in actual_states or int(actual_states[identifier]["step"]) != target:
                raise RuntimeError(f"Adam counter reset or sampling position changed: parameter {identifier}, expected {target}")
            for key in ["exp_avg", "exp_avg_sq"]:
                tensor = actual_states[identifier][key]
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"Resumed Adam {key} is nonfinite: {identifier}")
            if not (actual_states[identifier]["exp_avg_sq"] >= 0).all():
                raise RuntimeError(f"Resumed Adam second moment is negative: {identifier}")
        if any(not torch.isfinite(value).all() for value in after["state_dict"].values() if value.is_floating_point()):
            raise RuntimeError("Resumed model contains nonfinite tensors")
        if set(after["benchmark_rng"]) != {"python", "numpy", "torch", "cuda"} or not after["benchmark_rng"]["cuda"]:
            raise RuntimeError("Resumed RNG state is incomplete")
        step = after["global_step"] - 1
        delay = model.lr_delay_mult + (1 - model.lr_delay_mult) * np.sin(.5 * np.pi * np.clip(step / model.lr_delay_steps, 0, 1))
        fraction = np.clip(step / identity["max_steps"], 0, 1)
        expected_lr = delay * np.exp(np.log(model.lr_init)*(1-fraction) + np.log(model.lr_final)*fraction)
        actual_lr = actual_optimizer["param_groups"][0]["lr"]
        if not np.isclose(actual_lr, expected_lr, rtol=1e-10, atol=1e-15):
            raise RuntimeError("Resumed learning rate no longer uses the full prescribed schedule")
        result.update(status="passed", after_checkpoint=str(args.after),
            after_sha256=sha(args.after), actual_after_global_step=int(after["global_step"]),
            actual_sampler_committed=actual_sampling["committed"],
            actual_lr=actual_lr, expected_lr=float(expected_lr),
            shared_adam_continuation_verified=True, state_embedding_accesses_verified=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ["shared_parameters", "embedding_parameters"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--official", type=Path, default=Path(__file__).resolve().parents[4] / "third_party/HOSNeRF")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    for name in ["before", "after", "official", "data_root", "output"]:
        value = getattr(arguments, name)
        if value is not None:
            setattr(arguments, name, value.expanduser().resolve())
    if arguments.updates <= 0:
        parser.error("updates must be positive")
    main(arguments)
