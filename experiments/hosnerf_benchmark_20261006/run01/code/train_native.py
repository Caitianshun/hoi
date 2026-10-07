#!/usr/bin/env python3
"""Train the three native HOSNeRF stages without changing the vendor checkout.

Each invocation runs one stage. Formal budgets are 500000/400000/200000 steps;
shorter runs require --smoke and cannot be reported as the formal benchmark.
The official model, losses, Adam implementation, and learning-rate schedules
are retained. Compatibility patches are applied only in this Python process.
The Lightning wrapper is rebound to Adam's restored parameter groups so that
the official learning-rate updates continue to affect the live optimizer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap
import time
import traceback
import types


STAGES = {
    1: ("1st_State-Conditional_Scene", "configs/state_mipnerf360/Backpack.gin", 500000),
    2: ("2nd_State_Conditional_Human-Object", "configs/human-object/Backpack.gin", 400000),
    3: ("3rd_Complete_HOSNeRF", "configs/HOSNeRF/Backpack.gin", 200000),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=STAGES, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing the scene directory; use a prepared independent copy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official", type=Path,
                        default=Path(__file__).resolve().parents[4] / "third_party/HOSNeRF")
    parser.add_argument("--background-checkpoint", type=Path)
    parser.add_argument("--human-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after-updates", type=int, default=0,
                        help="End this segment after N new updates; full max_steps and LR schedule stay unchanged")
    parser.add_argument("--deadline", type=float,
                        default=datetime(2026, 11, 4, 23, tzinfo=timezone(timedelta(hours=-8))).timestamp(),
                        help="Unix cutoff; stop after the current update and save full state")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", type=int, default=0,
                        help="Index within CUDA_VISIBLE_DEVICES; all official .cuda() calls use this device")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=0,
                        help="Optional ray chunk for stage 2/3 (changes batching only)")
    parser.add_argument("--netchunk", type=int, default=0,
                        help="Optional point chunk for stage 2/3 (changes batching only)")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--imports-only", action="store_true", help="CPU import/config check; no data or model construction")
    parser.add_argument("--check-only", action="store_true", help="CPU model and dataset construction; no optimization")
    parser.add_argument("--no-evaluate", action="store_true", help="Explicitly defer the final stage-3 evaluation")
    parser.add_argument("--evaluate-only", action="store_true", help="Stage 3 only; requires a complete --resume checkpoint")
    args = parser.parse_args()
    for name in ["data_root", "output", "official", "background_checkpoint", "human_checkpoint", "resume"]:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.max_steps = args.max_steps if args.max_steps is not None else STAGES[args.stage][2]
    if args.smoke and not args.stop_after_updates and args.max_steps == STAGES[args.stage][2]:
        args.stop_after_updates = 2
    if args.max_steps <= 0 or not 1 <= args.checkpoint_every <= 2000:
        parser.error("max-steps must be positive and checkpoint-every must be within 1..2000")
    if args.max_steps != STAGES[args.stage][2] and not args.smoke:
        parser.error("A nonofficial step budget requires --smoke")
    if args.workers < 0 or args.device < 0:
        parser.error("workers and device must be nonnegative")
    if args.stop_after_updates < 0:
        parser.error("stop-after-updates must be nonnegative")
    if args.evaluate_only and (args.stage != 3 or not args.resume):
        parser.error("evaluate-only requires --stage 3 and --resume")
    return args


class SamplingCursor:
    """Resume original sampling order without saving multi-GB ray index arrays.

    Stage 1 generates its entire epoch from NumPy's epoch-start state; stage 2/3
    RandomSampler generates an epoch using an internal Torch generator. Replay
    recreates the original iterator and skips only committed training batches.
    Worker-prefetch RNG is still not claimed to be exactly reproducible.
    """
    def __init__(self, np, torch):
        self.np, self.torch = np, torch
        self.sampler = None
        self.pending = None

    def wrap(self, source):
        cursor = self
        class WrappedSampler:
            def __init__(self):
                self.source = source
                self.committed = 0
                self.epoch = 0
                self.epoch_rng = None
            def __len__(self):
                return len(self.source)
            def __iter__(self):
                if self.committed >= len(self.source):
                    self.committed = 0
                    self.epoch_rng = None
                    self.epoch += 1
                if self.epoch_rng is None:
                    self.epoch_rng = dict(numpy=cursor.np.random.get_state(), torch=cursor.torch.get_rng_state())
                    iterator = iter(self.source)
                else:
                    current_np, current_torch = cursor.np.random.get_state(), cursor.torch.get_rng_state()
                    cursor.np.random.set_state(self.epoch_rng["numpy"])
                    cursor.torch.set_rng_state(self.epoch_rng["torch"])
                    iterator = iter(self.source)
                    for _ in range(self.committed):
                        next(iterator)
                    cursor.np.random.set_state(current_np)
                    cursor.torch.set_rng_state(current_torch)
                yield from iterator
            def state_dict(self):
                return dict(committed=self.committed, epoch=self.epoch, epoch_rng=self.epoch_rng,
                            epoch_length=len(self.source))
            def load_state_dict(self, state):
                if state["epoch_length"] != len(self.source):
                    raise RuntimeError("Resumed sampler epoch length differs")
                self.committed, self.epoch, self.epoch_rng = state["committed"], state["epoch"], state["epoch_rng"]
        self.sampler = WrappedSampler()
        if self.pending is not None:
            self.sampler.load_state_dict(self.pending)
            self.pending = None
        return self.sampler

    def state_dict(self):
        return self.sampler.state_dict() if self.sampler is not None else self.pending

    def load_state_dict(self, state):
        if self.sampler is not None:
            self.sampler.load_state_dict(state)
        else:
            self.pending = state

    def commit(self):
        if self.sampler is not None:
            self.sampler.committed += 1


def load_runtime(args):
    import numpy as np
    import torch
    import gin
    import pytorch_lightning as pl

    source = args.official / STAGES[args.stage][0]
    if not source.is_dir():
        raise FileNotFoundError(source)
    os.chdir(source)
    sys.path.insert(0, str(source))
    # NumPy removed np.bool; the vendor uses it only as the boolean dtype.
    if "bool" not in np.__dict__:
        np.bool = np.bool_
    # Official checkpoints and Lightning 1.9 carry full trusted Python state.
    # This also covers the vendor LPIPS loader and Lightning cloud_io imports.
    original_load = torch.load
    @functools.wraps(original_load)
    def legacy_load(*positional, **keywords):
        keywords.setdefault("weights_only", False)
        return original_load(*positional, **keywords)
    torch.load = legacy_load
    torch.set_num_threads(args.threads)
    pl.seed_everything(args.seed, workers=True)
    import pdb
    def fail_debug_trap(*unused, **keywords):
        raise RuntimeError("The official code reached a debugger trap; aborting unattended execution")
    pdb.set_trace = fail_debug_trap

    # Register only the gin entry-point signature; its obsolete DDP run.py is
    # never imported. Stage 1's original optimizer queries run.max_steps.
    @gin.configurable()
    def run(dataset_name=None, datadir=None, model_name=None, max_steps=-1,
            log_every_n_steps=100, grad_max_norm=.001, bkgd_path=None,
            human_path=None):
        raise RuntimeError("The vendor training entry point is intentionally not called")

    import src.model.mipnerf360.model as official_model
    import src.data.litdata as litdata
    sampling = SamplingCursor(np, torch)
    official_model._benchmark_sampling = sampling
    gin.parse_config_file(str(source / STAGES[args.stage][1]))
    gin.bind_parameter("run.max_steps", args.max_steps)

    # Lightning 1.9 no longer passes using_native_amp to optimizer_step.
    # The original body is called with the removed argument set to False;
    # 32-bit precision, Adam closure and stage-specific LR updates are intact.
    original_step = official_model.LitMipNeRF360.optimizer_step
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu=False, using_lbfgs=False, **unused):
        # PL1.9 copies param_groups before Adam.load_state_dict replaces them.
        # Rebind the live groups while retaining Lightning's step/closure path.
        optimizer.param_groups = getattr(optimizer, "optimizer", optimizer).param_groups
        return original_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                             optimizer_closure, on_tpu, False, using_lbfgs)
    official_model.LitMipNeRF360.optimizer_step = optimizer_step

    cfg = None
    if args.stage == 1:
        gin.bind_parameter("LitData.num_workers", args.workers)
        import src.data.interface as data_interface
        import src.data.sampler as samplers
        original_sampler_init = samplers.DDPSampler.__init__
        def single_sampler_init(self, batch_size, num_replicas, rank, tpu):
            return original_sampler_init(self, batch_size, 1, 0, tpu)
        samplers.DDPSampler.__init__ = single_sampler_init
        original_loader = data_interface.DataLoader
        def compatible_loader(*positional, **keywords):
            if "batch_sampler" in keywords:
                keywords["batch_sampler"] = sampling.wrap(keywords["batch_sampler"])
            if keywords.get("num_workers", 0) == 0:
                keywords["persistent_workers"] = False
            return original_loader(*positional, **keywords)
        data_interface.DataLoader = compatible_loader
    else:
        from third_parties.yacs import CfgNode as CN
        cfg = CN()
        cfg.resume = bool(args.resume)
        cfg.eval_iter = 10000000
        cfg.render_folder_name = ""
        cfg.ignore_non_rigid_motions = False
        cfg.render_skip = 1
        cfg.render_frames = 100
        cfg.num_workers = args.workers
        cfg.merge_from_file(str(source / "configs/default.yaml"))
        cfg.merge_from_file(str(source / "configs/human_nerf/wild/monocular/adventure.yaml"))
        cfg.basedir = str(args.data_root / args.scene)
        if args.chunk:
            cfg.chunk = args.chunk
            if args.stage == 3:
                cfg.chunk_bkg = args.chunk
        if args.netchunk:
            cfg.netchunk_per_gpu = args.netchunk
        from core.data.dataset_args import DatasetArgs
        original_get = DatasetArgs.get
        def local_get(configuration, name):
            result = original_get(configuration, name)
            result["dataset_path"] = cfg.basedir
            return result
        DatasetArgs.get = staticmethod(local_get)
        from core.data import create_dataset
        def native_loader(configuration, data_type="train"):
            node = configuration[data_type]
            dataset = create_dataset(configuration, data_type)
            sampler = None
            if data_type == "train":
                sampler = sampling.wrap(torch.utils.data.RandomSampler(dataset))
            return torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=node.batch_size, shuffle=node.shuffle if sampler is None else False,
                drop_last=node.drop_last, pin_memory=True,
                persistent_workers=configuration.num_workers > 0,
                num_workers=configuration.num_workers,
            )
        # None of the five idle rendering loaders is constructed at startup.
        official_model.create_dataloader = lambda *positional, **keywords: None
        official_model.LitMipNeRF360.progress = lambda self: False
        official_model._benchmark_native_loader = native_loader
    return np, torch, gin, pl, source, official_model, litdata, cfg


def input_identity(args, source):
    scene = args.data_root / args.scene
    images = sorted(scene.glob("images/*.png"))
    if len(images) < 16:
        raise ValueError("The official uniform-16 split requires at least 16 PNG frames")
    test_indices = list(range(0, len(images), len(images) // 16))[:16]
    test_ids = [images[index].stem for index in test_indices]
    train_ids = [path.stem for index, path in enumerate(images) if index not in test_indices]
    required = ["transitions_times.json", "poses_bounds.npy", "cameras.pkl",
                "canonical_joints.pkl", "mesh_infos.pkl"]
    if args.stage == 3:
        required.append("cameras_scaleworld.pkl")
    metadata = {name: sha256(scene / name) for name in required}
    if args.stage in [2, 3]:
        absent = [name for name in train_ids if not (scene / "images_flow" / (name + "_bwd.npz")).is_file()]
        if absent:
            raise FileNotFoundError(f"Missing prepared train-only backward fields: {absent[:5]}")
        flow_manifest = Path(__file__).resolve().parents[1] / "protocol" / (args.scene + "_flow.json")
        flow = json.loads(flow_manifest.read_text())
        if flow["status"] != "complete" or flow["identity"]["train_ids"] != train_ids or flow["identity"]["test_ids"] != test_ids:
            raise RuntimeError("A complete matching frozen training-only flow manifest is required")
        if flow["identity"]["heldout_rgb_read"] or len(flow["pairs"]) != len(train_ids):
            raise RuntimeError("Flow cache violates the no-heldout-RGB protocol or is incomplete")
        import numpy as np
        with np.load(scene / "images_flow" / (train_ids[0] + "_bwd.npz")) as first:
            if np.any(first["mask"] != 0):
                raise RuntimeError("The first retained train frame must have an all-zero flow validity mask")
    revision = subprocess.check_output(["git", "-C", str(args.official), "rev-parse", "HEAD"], text=True).strip()
    identity = dict(stage=args.stage, scene=args.scene, seed=args.seed,
                data_root=str(args.data_root), train_ids=train_ids, test_ids=test_ids,
                uniform_test_rule="sorted PNG indices arange(N)[::floor(N/16)][:16]",
                metadata_sha256=metadata, official_revision=revision,
                official_model_sha256=sha256(source / "src/model/mipnerf360/model.py"),
                native_budget=STAGES[args.stage][2], max_steps=args.max_steps,
                formal=not args.smoke, checkpoint_selection="final prescribed training step; no validation selection",
                validation_during_training=False, idle_render_loaders=False,
                flow_protocol="prepared fields must use retained training-list predecessor only; first mask all zero",
                smpl_dependency="published canonical_joints/mesh_infos/cameras; no runtime SMPL network",
                resume_note="Full Lightning Adam/loop state, main-process Python/NumPy/Torch/CUDA RNG and sampler epoch-start RNG/committed position; worker-prefetch is not claimed bitwise identical",
                image_access_note="Stage 1's original loader reads all RGB into CPU arrays before splitting; only i_train enters its loss. Human stages construct only the training loader until final evaluation.",
                adjustments=["dynamic dataset paths", "PL1.9 optimizer hook compatibility",
                             "torch.load full checkpoint compatibility", "NumPy bool dtype alias",
                             "single-device sampler rank", "workers=0 loader compatibility",
                             "disable progress/all-frame/freeview/tpose evaluation",
                             "periodic and final full-state checkpoints; final-only 16-frame evaluation",
                             "fail instead of entering an unattended native debugger trap"])
    identity["training_script_sha256"] = sha256(Path(__file__).resolve())
    identity["benchmark_config_sha256"] = sha256(Path(__file__).resolve().parents[1] / "configs/benchmark.json")
    identity["flow_manifest_sha256"] = sha256(flow_manifest) if args.stage in [2, 3] else None
    return identity


def construct_training(args, runtime, identity):
    np, torch, gin, pl, source, module, litdata, cfg = runtime
    basedir = str(args.data_root / args.scene)
    if args.stage == 1:
        class TrainOnlyScene(litdata.LitDataNeRF360V2):
            def setup(self, stage):
                self.num_devices = self.trainer.num_devices
                if stage == "fit":
                    self.train_dset, _ = self.split_each(
                        self.images, self.masks, self.normals, None, self.i_train, dummy=False)
                    self.train_image_sizes = self.image_sizes[self.i_train]
                    if self.use_near_clip:
                        self.inward_nearfar_heuristic(self.extrinsics[self.i_train][:, :3, 3])
            def val_dataloader(self):
                return []
        data = TrainOnlyScene(datadir=str(args.data_root), scene_name=args.scene)
        model = module.LitMipNeRF360(basedir=basedir)
    else:
        class TrainOnlyHuman(pl.LightningDataModule):
            near_bkg = .1
            far_bkg = 1e6
            def train_dataloader(self):
                if not hasattr(self, "native_train_loader"):
                    self.native_train_loader = module._benchmark_native_loader(cfg, "train")
                return self.native_train_loader
            def val_dataloader(self):
                return []
        data = TrainOnlyHuman()
        model = module.LitMipNeRF360(cfg=cfg, basedir=basedir)
        if args.stage == 3 and not args.resume:
            for name, checkpoint in [("human", args.human_checkpoint), ("background", args.background_checkpoint)]:
                if checkpoint is None:
                    raise ValueError(f"Stage 3 requires its own completed {name} checkpoint")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                parent = payload.get("benchmark_identity")
                expected_stage = 2 if name == "human" else 1
                if parent is None or parent["stage"] != expected_stage or parent["scene"] != args.scene or parent["seed"] != args.seed:
                    raise RuntimeError(f"Stage 3 requires its own native {name} checkpoint from the same scene/seed")
                for key in ["train_ids", "test_ids", "official_revision", "training_script_sha256", "benchmark_config_sha256"]:
                    if parent[key] != identity[key]:
                        raise RuntimeError(f"Stage 3 {name} initialization identity differs: {key}")
                for key, digest in parent["metadata_sha256"].items():
                    if key in identity["metadata_sha256"] and digest != identity["metadata_sha256"][key]:
                        raise RuntimeError(f"Stage 3 {name} initialization dataset metadata differs: {key}")
                if not args.smoke and (not parent["formal"] or payload["global_step"] != STAGES[expected_stage][2]):
                    raise RuntimeError(f"Formal stage 3 requires the completed official-budget {name} stage")
                if name == "human" and parent["flow_manifest_sha256"] != identity["flow_manifest_sha256"]:
                    raise RuntimeError("Stage 3 human initialization flow manifest differs")
                expected_prefix = "human." if name == "human" else "model."
                selected = {key: value for key, value in payload["state_dict"].items() if key.startswith(expected_prefix)}
                target = {key: value for key, value in model.state_dict().items() if key.startswith(expected_prefix)}
                if set(selected) != set(target):
                    raise RuntimeError(f"{name} checkpoint keys differ: missing={set(target)-set(selected)}, extra={set(selected)-set(target)}")
                model.load_state_dict(selected, strict=False)
    model.logdir = str(args.output)
    return model, data


def evaluate_final(args, runtime, model, trainer, identity):
    np, torch, gin, pl, source, module, litdata, cfg = runtime
    from skimage.metrics import structural_similarity
    model.test_dataloader = module._benchmark_native_loader(cfg, "test")
    if list(model.test_dataloader.dataset.framelist) != identity["test_ids"]:
        raise RuntimeError("The final native test loader differs from the frozen uniform-16 split")
    rows = []
    evaluation = args.output / "evaluation"
    evaluation.mkdir(exist_ok=True)
    def capture_float(self, frame_id, prediction, truth, height, width):
        rgb = prediction.reshape(height, width, 3)
        target = truth.reshape(height, width, 3)
        np.savez_compressed(evaluation / (frame_id + ".npz"), rgb=rgb, gt=target)
        return structural_similarity(rgb, target, data_range=1.0, channel_axis=-1)
    def capture_scalar(self, frame_id, psnr, ssim, lpips):
        row = dict(frame_id=frame_id, PSNR=float(psnr), SSIM=float(ssim), LPIPS=float(lpips))
        rows.append(row)
        write_json(evaluation / "per_frame.json", {"frames": rows})
        print("HOS_FRAME", json.dumps(row), flush=True)
    body = textwrap.dedent(inspect.getsource(module.LitMipNeRF360.test_metrics))
    old_ssim = "ssim = skimage.metrics.structural_similarity(rendered, truth, channel_axis=True)"
    old_append = "lpipss.append(lpips)"
    if body.count(old_ssim) != 1 or body.count(old_append) != 1:
        raise RuntimeError("Vendor renderer changed; metric-only patch requires review")
    body = body.replace(old_ssim, "ssim = self.capture_float(frame_name, rendered, truth, int(height), int(width))")
    body = body.replace(old_append, old_append + "\n        self.capture_scalar(frame_name, psnr, ssim, lpips)")
    namespace = {}
    exec(compile(body, str(source / "src/model/mipnerf360/model.py") + ":metric_capture", "exec"), vars(module), namespace)
    model.capture_float = types.MethodType(capture_float, model)
    model.capture_scalar = types.MethodType(capture_scalar, model)
    model.test_metrics = types.MethodType(namespace["test_metrics"], model)
    model.near_bkg = .1
    model.far_bkg = 1e6
    model.to(torch.device("cuda", args.device))
    model.eval()
    with torch.no_grad():
        model.test_metrics()
    if [row["frame_id"] for row in rows] != identity["test_ids"]:
        raise RuntimeError("Final renderer did not complete all 16 frozen test frames")
    results = dict(status="completed", frames=len(rows), global_step=trainer.global_step,
                   mean_metrics={key: float(np.mean([row[key] for row in rows])) for key in ["PSNR", "SSIM", "LPIPS"]},
                   metric_note="Official PSNR and pretrained VGG LPIPS; SSIM on HxWx3 float RGB, data_range=1, instead of vendor flattened Nx3 call. Renderer/sampling/compositing unchanged.")
    write_json(evaluation / "completion.json", results)
    return results


def main():
    args = arguments()
    globals()["_output_directory"] = args.output
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "final.ckpt").exists() and not args.resume:
        raise FileExistsError("An existing result requires explicit --resume or a new --output directory")
    start = time.monotonic()
    runtime = load_runtime(args)
    np, torch, gin, pl, source, module, litdata, cfg = runtime
    if args.imports_only:
        write_json(args.output / "import_check.json", dict(status="passed", stage=args.stage,
                   torch=torch.__version__, lightning=pl.__version__, source=str(source)))
        print("HOS_IMPORTS_OK", args.stage, flush=True)
        return
    identity = input_identity(args, source)
    identity.update(torch_version=torch.__version__, lightning_version=pl.__version__,
                    workers=args.workers, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    device=args.device, chunk=args.chunk, netchunk=args.netchunk,
                    resource=dict(cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                  device=args.device, threads=args.threads,
                                  segment_limit=args.stop_after_updates, deadline_unix=args.deadline))
    for name in ["background_checkpoint", "human_checkpoint", "resume"]:
        checkpoint = getattr(args, name)
        if checkpoint:
            identity[name] = dict(path=str(checkpoint), sha256=sha256(checkpoint))
    write_json(args.output / "identity.json", identity)
    (args.output / "resolved.gin").write_text(gin.config_str())
    if cfg is not None:
        (args.output / "resolved.yaml").write_text(cfg.dump())
    if not args.check_only:
        if not args.evaluate_only and args.deadline and time.time() >= args.deadline:
            raise RuntimeError("Training cutoff passed; no new optimizer update was started")
        if not torch.cuda.is_available():
            raise RuntimeError("Native training requires CUDA; use --imports-only or --check-only on CPU")
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        identity["gpu"] = torch.cuda.get_device_name(args.device)
        write_json(args.output / "identity.json", identity)
    model, data = construct_training(args, runtime, identity)
    if args.stage == 1:
        actual_test = [sorted((args.data_root / args.scene).glob("images/*.png"))[index].stem for index in data.i_test]
        if actual_test != identity["test_ids"]:
            raise RuntimeError("Stage 1 split differs from the frozen uniform-16 split")
    else:
        train_loader = data.train_dataloader()
        if list(train_loader.dataset.framelist) != identity["train_ids"]:
            raise RuntimeError("Native human train loader differs from the frozen training split")
    if args.check_only:
        write_json(args.output / "construction_check.json", dict(status="passed", stage=args.stage,
                   parameters=sum(parameter.numel() for parameter in model.parameters()), identity=identity))
        print("HOS_CONSTRUCTION_OK", args.stage, flush=True)
        return

    class FullState(pl.Callback):
        def on_train_start(self, trainer, lightning_module):
            self.segment_start = trainer.global_step
            self.segment_end = min(args.max_steps, self.segment_start + args.stop_after_updates) if args.stop_after_updates else args.max_steps
            self.stop_reason = "prescribed_budget"
        def on_train_batch_end(self, trainer, lightning_module, outputs, batch, batch_idx):
            module._benchmark_sampling.commit()
            if args.deadline and time.time() >= args.deadline and trainer.global_step < args.max_steps:
                self.stop_reason = "deadline"
                trainer.should_stop = True
            elif trainer.global_step >= self.segment_end and trainer.global_step < args.max_steps:
                self.stop_reason = "segment_updates"
                trainer.should_stop = True
        def on_save_checkpoint(self, trainer, lightning_module, checkpoint):
            checkpoint["benchmark_identity"] = identity
            checkpoint["benchmark_rng"] = dict(python=random.getstate(), numpy=np.random.get_state(),
                   torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
            checkpoint["benchmark_sampling"] = module._benchmark_sampling.state_dict()
        def on_load_checkpoint(self, trainer, lightning_module, checkpoint):
            previous = checkpoint.get("benchmark_identity")
            if previous is None:
                raise RuntimeError("Resume requires this harness's full training checkpoint")
            for key in ["stage", "scene", "seed", "train_ids", "test_ids", "metadata_sha256", "official_revision", "official_model_sha256", "formal", "max_steps", "training_script_sha256", "benchmark_config_sha256", "flow_manifest_sha256", "chunk", "netchunk", "workers"]:
                if previous[key] != identity[key]:
                    raise RuntimeError(f"Resume identity mismatch: {key}")
            rng = checkpoint["benchmark_rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            module._benchmark_sampling.load_state_dict(checkpoint["benchmark_sampling"])
        def on_before_backward(self, trainer, lightning_module, loss):
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite native loss; optimizer was not advanced")

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    checkpoint = ModelCheckpoint(dirpath=str(args.output / "checkpoints"),
                filename="step-{step:09d}", auto_insert_metric_name=False,
                monitor=None, save_top_k=1, save_last=True,
                every_n_train_steps=args.checkpoint_every,
                save_on_train_epoch_end=False, save_weights_only=False)
    full_state = FullState()
    trainer = pl.Trainer(accelerator="gpu", devices=[args.device], strategy="auto",
              max_steps=args.max_steps, max_epochs=-1, precision=32,
              callbacks=[full_state, checkpoint],
              logger=CSVLogger(str(args.output), name="train_metrics"),
              log_every_n_steps=args.log_every, limit_val_batches=0,
              num_sanity_val_steps=0, enable_progress_bar=False,
              enable_model_summary=False, replace_sampler_ddp=False,
              gradient_clip_val=.001, gradient_clip_algorithm="norm")
    write_json(args.output / "status.json", dict(status="running", stage=args.stage,
               formal=not args.smoke, prescribed_steps=args.max_steps))
    if args.evaluate_only:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        FullState().on_load_checkpoint(trainer, model, payload)
        if int(payload["global_step"]) != args.max_steps:
            raise RuntimeError("Evaluation-only requires the prescribed final checkpoint")
        model.load_state_dict(payload["state_dict"], strict=True)
        model._trainer = types.SimpleNamespace(global_step=int(payload["global_step"]))
        model.to(torch.device("cuda", args.device))
        eval_trainer = model._trainer
    else:
        trainer.fit(model, datamodule=data, ckpt_path=str(args.resume) if args.resume else None)
        complete = trainer.global_step == args.max_steps
        if not complete and full_state.stop_reason != "deadline" and (not args.stop_after_updates or trainer.global_step != full_state.segment_end):
            raise RuntimeError(f"Native optimization stopped at {trainer.global_step}/{args.max_steps}")
        if complete:
            trainer.save_checkpoint(str(args.output / "final.ckpt"), weights_only=False)
        trainer.save_checkpoint(str(args.output / "last.ckpt"), weights_only=False)
        trainer.save_checkpoint(str(args.output / "checkpoints/last.ckpt"), weights_only=False)
        eval_trainer = trainer
    complete = eval_trainer.global_step == args.max_steps
    result = dict(status="completed" if complete else "segment_complete", stage=args.stage, formal=not args.smoke,
                  reason=full_state.stop_reason if not args.evaluate_only else "evaluation_only",
                  complete=complete, segment_updates=eval_trainer.global_step-full_state.segment_start if not args.evaluate_only else 0,
                  global_step=eval_trainer.global_step, wall_seconds=time.monotonic()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                  final_checkpoint=str(args.resume if args.evaluate_only else args.output / ("final.ckpt" if complete else "last.ckpt")))
    write_json(args.output / "receipt.json", result)
    if complete and args.stage == 3 and not args.no_evaluate and not args.smoke:
        result["evaluation"] = evaluate_final(args, runtime, model, eval_trainer, identity)
    write_json(args.output / "status.json", result)
    print("HOS_COMPLETED", json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # Preserve a machine-readable failure even when SSH log output is lost.
        try:
            index = sys.argv.index("--output")
            directory = globals().get("_output_directory", Path(sys.argv[index+1]).expanduser().resolve())
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "status.json", dict(status="failed", error=repr(error),
                       traceback=traceback.format_exc()))
        except Exception:
            pass
        raise
