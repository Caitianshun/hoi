"""Read-only runtime lineage instrumentation for MoSca's dynamic leaves.

No extra forward, RNG draws, optimizer calls, tensor writes, or source edits.
CPU IDs are observational metadata, NOT trainable instance/material identities.
Install inside recorded_fit, after initialization, and close in its finally block.
"""
from __future__ import annotations

import csv
import functools
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch


def _array(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


class IdentityTable:
    """Stable monotonically allocated IDs; current rows are only an index map."""
    def __init__(self, count):
        self.live = np.arange(count, dtype=np.int64)
        self.parent = [-1] * count
        self.root = list(range(count))
        self.birth_step = [0] * count
        self.death_step = [-1] * count
        self.birth_kind = ['initial'] * count

    def append(self, parent_rows, count, kind, step):
        ids = np.arange(len(self.parent), len(self.parent) + count, dtype=np.int64)
        if parent_rows is None:
            parents = np.full(count, -1, dtype=np.int64)
            roots = ids.copy()
        else:
            rows = np.asarray(parent_rows, dtype=np.int64)
            if len(rows) != count or (len(rows) and (rows.min() < 0 or rows.max() >= len(self.live))):
                raise AssertionError('Parent row map does not match actual append')
            parents = self.live[rows].copy()
            roots = np.asarray(self.root, dtype=np.int64)[parents]
        self.parent.extend(parents.tolist())
        self.root.extend(roots.tolist())
        self.birth_step.extend([step] * count)
        self.death_step.extend([-1] * count)
        self.birth_kind.extend([kind] * count)
        self.live = np.concatenate([self.live, ids])
        return dict(id=ids, parent_id=parents, root_id=roots)

    def prune(self, mask, step):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.live.shape:
            raise AssertionError('Prune mask does not match current row map')
        rows = np.flatnonzero(mask)
        ids = self.live[mask].copy()
        for value in ids:
            self.death_step[int(value)] = step
        self.live = self.live[~mask]
        return dict(id=ids, old_row=rows,
                    parent_id=np.asarray(self.parent, dtype=np.int64)[ids],
                    root_id=np.asarray(self.root, dtype=np.int64)[ids])

    def arrays(self):
        return dict(live_id=self.live, parent_id=np.asarray(self.parent, dtype=np.int64),
                    root_id=np.asarray(self.root, dtype=np.int64),
                    birth_step=np.asarray(self.birth_step, dtype=np.int64),
                    death_step=np.asarray(self.death_step, dtype=np.int64),
                    birth_kind=np.asarray(self.birth_kind, dtype='U40'))


class LineageHandle:
    def __init__(self, photo_module, d_model, output_dir, *, world_scale=1.0, static_model=None,
                 seed_provenance_path=None, checkpoints=(0, 100, 500, 1000, 2000, 4000, 8000)):
        self.photo = photo_module
        self.model = d_model
        self.static_model = static_model
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / 'events.csv').exists():
            raise FileExistsError('Lineage output is append-protected; use a fresh directory')
        (self.output / 'events').mkdir(exist_ok=True)
        (self.output / 'snapshots').mkdir(exist_ok=True)
        self.scale = float(world_scale)
        if self.scale <= 0:
            raise ValueError('world_scale must be positive')
        self.seed_path = Path(seed_provenance_path) if seed_provenance_path is not None else None
        self.checkpoints = set(map(int, checkpoints))
        self.leaf = IdentityTable(int(d_model.N))
        self.node = IdentityTable(int(d_model.scf.M))
        self.step = 0  # completed updates, event inside step i is tagged i+1
        self.completed_steps = 0
        self.total_steps = None
        self.event_index = 0
        self.snapshots = []
        self.overhead_seconds = 0.0
        self.closed = False
        self.loop_completed = False
        self.originals = []
        self.started = time.perf_counter()
        self.csv_file = (self.output / 'events.csv').open('w', newline='')
        self.csv = csv.DictWriter(self.csv_file, fieldnames=[
            'event', 'completed_step_of_update', 'entity', 'operation', 'caller',
            'count_before', 'count_after', 'affected_count', 'npz', 'observation_seconds'])
        self.csv.writeheader()
        t = time.perf_counter()
        self._install()
        np.savez_compressed(self.output / 'initial_identity.npz',
                            leaf_id=self.leaf.live, leaf_root_id=self.leaf.live,
                            node_id=self.node.live, node_root_id=self.node.live)
        self.overhead_seconds += time.perf_counter() - t

    def _patch(self, obj, name, factory):
        # A plain function on an instance receives exactly the caller's arguments;
        # original is already a bound method. Restore by deleting the override.
        had_own = name in vars(obj)
        old_own = vars(obj).get(name)
        original = getattr(obj, name)
        self.originals.append((obj, name, had_own, old_own))
        setattr(obj, name, factory(original))

    def _event(self, entity, operation, caller, before, after, payload, begun):
        fn = Path('events') / f'{self.event_index:05d}_{entity}_{operation}.npz'
        np.savez_compressed(self.output / fn, **payload,
                            step=np.array(self.step), operation=np.array(operation),
                            entity=np.array(entity), caller=np.array(caller))
        duration = time.perf_counter() - begun
        self.csv.writerow(dict(event=self.event_index, completed_step_of_update=self.step,
                               entity=entity, operation=operation, caller=caller,
                               count_before=before, count_after=after,
                               affected_count=len(payload['id']), npz=str(fn),
                               observation_seconds=duration))
        self.csv_file.flush()
        self.event_index += 1
        return duration

    def _install(self):
        handle = self

        def wrap_append(original):
            @functools.wraps(original)
            def observed(*args, **kwargs):
                t = time.perf_counter()
                caller = inspect.currentframe().f_back
                name, loc = caller.f_code.co_name, caller.f_locals
                before = len(handle.leaf.live)
                if before != int(handle.model.N):
                    raise AssertionError('Unobserved dynamic-row mutation before append')
                if name in ('_densify_and_clone', '_densify_and_split'):
                    parents = np.flatnonzero(_array(loc['selected_pts_mask'], bool))
                    kind = 'clone' if name.endswith('clone') else 'split'
                    if kind == 'split':
                        parents = np.tile(parents, int(loc['N']))
                elif name == 'gradient_based_node_densification':
                    parents = _array(loc['new_gs_ind'], np.int64)
                    kind = 'node_densify_duplicate'
                elif name in ('append_new_gs', 'append_new_node_and_gs'):
                    parents = None
                    kind = 'append_new_root'
                else:
                    raise RuntimeError(f'Unaudited leaf append caller: {name}')
                del caller, loc
                before_cost = time.perf_counter() - t
                result = original(*args, **kwargs)
                t = time.perf_counter()
                after = int(handle.model.N)
                data = handle.leaf.append(parents, after - before, kind, handle.step)
                data['new_row'] = np.arange(before, after, dtype=np.int64)
                if parents is not None:
                    data['parent_row_before_append'] = parents
                duration = handle._event('leaf', kind, name, before, after, data, t)
                handle.overhead_seconds += before_cost + duration
                return result
            return observed

        def wrap_prune(original):
            @functools.wraps(original)
            def observed(optimizer, mask):
                t = time.perf_counter()
                mask_cpu = _array(mask, bool)
                before = len(handle.leaf.live)
                if before != int(handle.model.N):
                    raise AssertionError('Unobserved dynamic-row mutation before prune')
                caller = inspect.currentframe().f_back.f_code.co_name
                pre_cost = time.perf_counter() - t
                result = original(optimizer, mask)
                t = time.perf_counter()
                after = int(handle.model.N)
                if after != before - int(mask_cpu.sum()):
                    raise AssertionError('Actual leaf prune does not match observed mask')
                data = handle.leaf.prune(mask_cpu, handle.step)
                handle.overhead_seconds += pre_cost + handle._event(
                    'leaf', 'prune', caller, before, after, data, t)
                return result
            return observed

        def wrap_node_append(original):
            @functools.wraps(original)
            def observed(*args, **kwargs):
                t = time.perf_counter()
                frame = inspect.currentframe().f_back
                name, loc = frame.f_code.co_name, frame.f_locals
                before = len(handle.node.live)
                if before != int(handle.model.scf.M):
                    raise AssertionError('Unobserved node append')
                source_leaf = None
                if name == 'gradient_based_node_densification':
                    rows = np.flatnonzero(_array(loc['candidate_mask'], bool))[
                        _array(loc['resample_ind'], np.int64)]
                    source_leaf = handle.leaf.live[rows].copy()
                del frame, loc
                pre_cost = time.perf_counter() - t
                result = original(*args, **kwargs)
                t = time.perf_counter()
                after = int(handle.model.scf.M)
                # New nodes are new roots, not material descendants of nearest nodes.
                data = handle.node.append(None, after - before, 'append_node', handle.step)
                data['new_row'] = np.arange(before, after, dtype=np.int64)
                if source_leaf is not None:
                    assert len(source_leaf) == after - before
                    data['source_leaf_id'] = source_leaf
                handle.overhead_seconds += pre_cost + handle._event(
                    'node', 'append', name, before, after, data, t)
                return result
            return observed

        def wrap_node_prune(original):
            @functools.wraps(original)
            def observed(optimizer, node_prune_mask):
                t = time.perf_counter()
                mask = _array(node_prune_mask, bool)
                before = len(handle.node.live)
                pre_cost = time.perf_counter() - t
                result = original(optimizer, node_prune_mask)
                t = time.perf_counter()
                after = int(handle.model.scf.M)
                if after == before and mask.any():
                    data = dict(id=np.empty(0, dtype=np.int64), requested_prune_id=handle.node.live[mask])
                    kind = 'prune_skipped'
                else:
                    assert after == before - int(mask.sum())
                    data = handle.node.prune(mask, handle.step)
                    kind = 'prune'
                handle.overhead_seconds += pre_cost + handle._event(
                    'node', kind, 'remove_nodes', before, after, data, t)
                return result
            return observed

        self._patch(self.model, '_densification_postprocess', wrap_append)
        self._patch(self.model, '_prune_points', wrap_prune)
        self._patch(self.model.scf, 'append_nodes_traj', wrap_node_append)
        self._patch(self.model.scf, 'remove_nodes', wrap_node_prune)

        def wrap_tqdm(original):
            @functools.wraps(original)
            def observed(iterable=None, *args, **kwargs):
                frame = inspect.currentframe().f_back
                loc = frame.f_locals
                matched = (frame.f_code.co_name == 'photometric_fit'
                           and loc.get('d_model') is handle.model
                           and isinstance(iterable, range)
                           and iterable.start == 0 and iterable.step == 1
                           and iterable.stop == loc.get('total_steps'))
                del frame, loc
                progress = original(iterable, *args, **kwargs)
                if not matched:
                    return progress
                if handle.total_steps is not None:
                    raise RuntimeError('One lineage handle must serve exactly one photo loop')
                handle.total_steps = iterable.stop
                def traced():
                    if 0 in handle.checkpoints:
                        handle.snapshot(0)
                    for step in progress:
                        handle.step = int(step) + 1
                        yield step
                        handle.completed_steps = int(step) + 1
                        if handle.completed_steps in handle.checkpoints:
                            handle.snapshot(handle.completed_steps)
                    handle.loop_completed = True
                return traced()
            return observed
        self._patch(self.photo, 'tqdm', wrap_tqdm)

    def snapshot(self, completed_steps):
        t = time.perf_counter()
        if int(self.model.N) != len(self.leaf.live) or int(self.model.scf.M) != len(self.node.live):
            raise AssertionError('Snapshot row mapping does not match model')
        folder = self.output / 'snapshots' / f'{completed_steps:05d}'
        folder.mkdir(exist_ok=False)
        # state_dict is read-only. Copy every tensor to CPU so files never retain
        # CUDA storage or aliases to parameters being updated after the callback.
        state = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                 for k, v in self.model.state_dict().items()}
        torch.save(state, folder / 'dynamic_state.pth')
        if self.static_model is not None:
            static_state = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                            for k, v in self.static_model.state_dict().items()}
            torch.save(static_state, folder / 'static_state.pth')
        ids = self.leaf.live
        attach = _array(self.model.attach_ind, np.int64)
        np.savez_compressed(folder / 'identity.npz', leaf_id=ids,
                            parent_id=np.asarray(self.leaf.parent, dtype=np.int64)[ids],
                            root_id=np.asarray(self.leaf.root, dtype=np.int64)[ids],
                            node_id=self.node.live, attached_node_id=self.node.live[attach],
                            attach_ind=attach, ref_time=_array(self.model.ref_time, np.int64),
                            raw_stored_xyz=_array(self.model._xyz),
                            world_scale=np.array(self.scale), completed_steps=np.array(completed_steps))
        duration = time.perf_counter() - t
        self.overhead_seconds += duration
        self.snapshots.append(dict(completed_steps=int(completed_steps),
                                   dynamic_leaves=int(self.model.N), nodes=int(self.model.scf.M),
                                   directory=str(folder.resolve()), observation_seconds=duration,
                                   state_sha256=_sha(folder / 'dynamic_state.pth'),
                                   identity_sha256=_sha(folder / 'identity.npz')))
        if self.static_model is not None:
            self.snapshots[-1]['static_state_sha256'] = _sha(folder / 'static_state.pth')
        # The hash reads are instrumentation too; include them in total cost.
        self.overhead_seconds += time.perf_counter() - t - duration

    def close(self):
        if self.closed:
            return self.summary
        t = time.perf_counter()
        for obj, name, had_own, old in reversed(self.originals):
            if had_own:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self.csv_file.close()
        np.savez_compressed(self.output / 'leaf_lineage.npz', **self.leaf.arrays())
        np.savez_compressed(self.output / 'node_lineage.npz', **self.node.arrays())
        self.overhead_seconds += time.perf_counter() - t
        self.closed = True
        provenance = None
        if self.seed_path is not None:
            provenance = dict(path=str(self.seed_path.resolve()), exists=self.seed_path.exists())
            if self.seed_path.exists():
                provenance['sha256'] = _sha(self.seed_path)
        self.summary = dict(status='completed' if self.loop_completed else 'closed_incomplete',
                            completed_steps=self.completed_steps, expected_steps=self.total_steps,
                            events=self.event_index, snapshots=self.snapshots,
                            allocated_leaf_ids=len(self.leaf.parent), surviving_leaves=len(self.leaf.live),
                            allocated_node_ids=len(self.node.parent), surviving_nodes=len(self.node.live),
                            instrumentation_wall_seconds=self.overhead_seconds,
                            handle_lifetime_wall_seconds=time.perf_counter() - self.started,
                            world_scale=self.scale, seed_provenance=provenance,
                            static_state_saved=self.static_model is not None,
                            script_sha256=_sha(__file__),
                            no_extra_rng_draws=True, no_extra_forward=True,
                            model_parameters_and_gradients_written=False,
                            step_convention='snapshot k after k complete loop iterations; mutation i belongs to update i (1-based)',
                            coordinates='state uses normalized MoSca world/local storage; divide world positions/scales by world_scale once. raw_stored_xyz is local if leaf_local_flag is true.',
                            identity_scope='Computational lineage only, not proof of material/surface identity; all fresh observations and nodes are independent roots.',
                            node_origin_note='Gradient-created nodes carry source_leaf_id in the append event. No fictitious nearest-node parent is assigned.',
                            reference_used=False)
        (self.output / 'summary.json').write_text(json.dumps(self.summary, indent=2) + '\n')
        return self.summary


def install_lineage_hooks(photo_module, d_model, output_dir, *, world_scale=1.0, static_model=None,
                          seed_provenance_path=None,
                          checkpoints=(0, 100, 500, 1000, 2000, 4000, 8000)):
    """Return handle; caller MUST call handle.close() in photometric_fit's finally."""
    return LineageHandle(photo_module, d_model, output_dir, world_scale=world_scale,
                         static_model=static_model, seed_provenance_path=seed_provenance_path,
                         checkpoints=checkpoints)
