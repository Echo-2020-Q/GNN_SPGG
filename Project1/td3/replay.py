from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .data import (
    TensorReplayActionRecord,
    TensorReplayBatch,
    TensorTransition,
    Transition,
    replay_source_name_to_id,
    topology_id_to_name,
)


def _coerce_transition(transition: TensorTransition | Transition) -> TensorTransition:
    if isinstance(transition, TensorTransition):
        return transition
    if isinstance(transition, Transition):
        return TensorTransition.from_transition(transition)
    raise TypeError("ReplayBuffer.add expects TensorTransition or Transition.")


def _concat_replay_batches(batches: Sequence[TensorReplayBatch]) -> TensorReplayBatch:
    if not batches:
        raise ValueError("batches must contain at least one item.")
    if len(batches) == 1:
        return batches[0].clone()

    first_batch = batches[0]
    replay_source_id = (
        torch.cat([batch.replay_source_id for batch in batches], dim=0)
        if all(batch.replay_source_id is not None for batch in batches)
        else None
    )
    return TensorReplayBatch(
        obs={key: torch.cat([batch.obs[key] for batch in batches], dim=0) for key in first_batch.obs},
        action=TensorReplayActionRecord(
            allocation=torch.cat([batch.action.allocation for batch in batches], dim=0),
        ),
        reward=torch.cat([batch.reward for batch in batches], dim=0),
        next_obs={key: torch.cat([batch.next_obs[key] for batch in batches], dim=0) for key in first_batch.next_obs},
        done=torch.cat([batch.done for batch in batches], dim=0),
        is_demo=torch.cat([batch.is_demo for batch in batches], dim=0),
        collapse_flag=torch.cat([batch.collapse_flag for batch in batches], dim=0),
        topology_id=torch.cat([batch.topology_id for batch in batches], dim=0),
        pool_power_demo_flag=torch.cat([batch.pool_power_demo_flag for batch in batches], dim=0),
        demo_return_target=torch.cat([batch.demo_return_target for batch in batches], dim=0),
        demo_return_valid=torch.cat([batch.demo_return_valid for batch in batches], dim=0),
        replay_source_id=replay_source_id,
    )


def _slice_replay_batch(batch: TensorReplayBatch, indices: Tensor) -> TensorReplayBatch:
    return TensorReplayBatch(
        obs={key: value.index_select(0, indices) for key, value in batch.obs.items()},
        action=TensorReplayActionRecord(
            allocation=batch.action.allocation.index_select(0, indices),
        ),
        reward=batch.reward.index_select(0, indices),
        next_obs={key: value.index_select(0, indices) for key, value in batch.next_obs.items()},
        done=batch.done.index_select(0, indices),
        is_demo=batch.is_demo.index_select(0, indices),
        collapse_flag=batch.collapse_flag.index_select(0, indices),
        topology_id=batch.topology_id.index_select(0, indices),
        pool_power_demo_flag=batch.pool_power_demo_flag.index_select(0, indices),
        demo_return_target=batch.demo_return_target.index_select(0, indices),
        demo_return_valid=batch.demo_return_valid.index_select(0, indices),
        replay_source_id=(
            None if batch.replay_source_id is None else batch.replay_source_id.index_select(0, indices)
        ),
    )


def _slice_replay_batch_range(batch: TensorReplayBatch, start: int, end: int) -> TensorReplayBatch:
    return TensorReplayBatch(
        obs={key: value[start:end] for key, value in batch.obs.items()},
        action=TensorReplayActionRecord(
            allocation=batch.action.allocation[start:end],
        ),
        reward=batch.reward[start:end],
        next_obs={key: value[start:end] for key, value in batch.next_obs.items()},
        done=batch.done[start:end],
        is_demo=batch.is_demo[start:end],
        collapse_flag=batch.collapse_flag[start:end],
        topology_id=batch.topology_id[start:end],
        pool_power_demo_flag=batch.pool_power_demo_flag[start:end],
        demo_return_target=batch.demo_return_target[start:end],
        demo_return_valid=batch.demo_return_valid[start:end],
        replay_source_id=None if batch.replay_source_id is None else batch.replay_source_id[start:end],
    )


def _shuffle_replay_batch(batch: TensorReplayBatch, rng: np.random.Generator) -> TensorReplayBatch:
    if len(batch) <= 1:
        return batch
    indices = torch.as_tensor(rng.permutation(len(batch)), dtype=torch.int64, device="cpu")
    return _slice_replay_batch(batch, indices)


def _with_replay_source_id(batch: TensorReplayBatch, source_name: str) -> TensorReplayBatch:
    source_id = replay_source_name_to_id(source_name)
    return TensorReplayBatch(
        obs=batch.obs,
        action=batch.action,
        reward=batch.reward,
        next_obs=batch.next_obs,
        done=batch.done,
        is_demo=batch.is_demo,
        collapse_flag=batch.collapse_flag,
        topology_id=batch.topology_id,
        pool_power_demo_flag=batch.pool_power_demo_flag,
        demo_return_target=batch.demo_return_target,
        demo_return_valid=batch.demo_return_valid,
        replay_source_id=torch.full((len(batch),), int(source_id), dtype=torch.int64, device=batch.reward.device),
    )


def _split_replay_batch_train_val_impl(
    batch: TensorReplayBatch,
    *,
    validation_fraction: float,
    rng: np.random.Generator,
    only_demo: bool,
) -> tuple[TensorReplayBatch | None, TensorReplayBatch | None]:
    if len(batch) <= 0:
        return None, None
    if validation_fraction <= 0.0:
        return batch.clone(), None

    if only_demo:
        eligible_mask = batch.is_demo.detach().cpu().numpy().astype(np.bool_, copy=False)
    else:
        eligible_mask = np.ones(len(batch), dtype=np.bool_)
    if not np.any(eligible_mask):
        return batch.clone(), None

    topology_ids = batch.topology_id.detach().cpu().numpy().astype(np.int64, copy=False)
    val_mask = np.zeros(len(batch), dtype=np.bool_)
    for topology_id in np.unique(topology_ids[eligible_mask]):
        topology_indices = np.flatnonzero(np.logical_and(eligible_mask, topology_ids == int(topology_id)))
        if topology_indices.size <= 1:
            continue
        requested = int(round(float(topology_indices.size) * float(validation_fraction)))
        requested = max(0, min(requested, int(topology_indices.size) - 1))
        if requested <= 0:
            continue
        selected = rng.choice(topology_indices, size=requested, replace=False)
        val_mask[selected] = True

    if not np.any(val_mask) and int(np.sum(eligible_mask)) > 1:
        selected = rng.choice(np.flatnonzero(eligible_mask), size=1, replace=False)
        val_mask[selected] = True

    train_mask = np.logical_not(val_mask)
    train_batch: TensorReplayBatch | None = None
    val_batch: TensorReplayBatch | None = None
    if np.any(train_mask):
        train_indices = torch.as_tensor(np.flatnonzero(train_mask), dtype=torch.int64, device="cpu")
        train_batch = _slice_replay_batch(batch, train_indices)
    if np.any(val_mask):
        val_indices = torch.as_tensor(np.flatnonzero(val_mask), dtype=torch.int64, device="cpu")
        val_batch = _slice_replay_batch(batch, val_indices)
    return train_batch, val_batch


def split_demo_batch_train_val(
    batch: TensorReplayBatch,
    *,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[TensorReplayBatch | None, TensorReplayBatch | None]:
    return _split_replay_batch_train_val_impl(
        batch,
        validation_fraction=validation_fraction,
        rng=rng,
        only_demo=True,
    )


def split_replay_batch_train_val(
    batch: TensorReplayBatch,
    *,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[TensorReplayBatch | None, TensorReplayBatch | None]:
    return _split_replay_batch_train_val_impl(
        batch,
        validation_fraction=validation_fraction,
        rng=rng,
        only_demo=False,
    )


def _allocate_integer_counts(total: int, weights: Sequence[float]) -> list[int]:
    if total <= 0:
        return [0 for _ in weights]
    if not weights:
        return []

    weight_array = np.asarray(weights, dtype=np.float64)
    if np.any(weight_array < 0.0):
        raise ValueError("weights must be non-negative.")
    if float(weight_array.sum()) <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    normalized = weight_array / weight_array.sum()
    raw = normalized * float(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total - int(counts.sum()))
    if remainder > 0:
        fractional = raw - counts.astype(np.float64)
        order = np.argsort(-fractional, kind="stable")
        for index in order[:remainder]:
            counts[index] += 1
    return [int(item) for item in counts.tolist()]


class _TensorReplayStorage:
    def __init__(self, capacity: int, seed: int | None = None, replacement_policy: str = "ring"):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if replacement_policy not in {"ring", "reservoir"}:
            raise ValueError("replacement_policy must be one of {'ring', 'reservoir'}.")

        self.capacity = int(capacity)
        self.replacement_policy = str(replacement_policy)
        self._next_index = 0
        self._size = 0
        self._seen_count = 0
        self._rng = np.random.default_rng(seed)

        self._obs_buffers: dict[str, Tensor] = {}
        self._next_obs_buffers: dict[str, Tensor] = {}
        self._action_buffers: dict[str, Tensor] = {}
        self._reward_buffer: Tensor | None = None
        self._done_buffer: Tensor | None = None
        self._is_demo_buffer: Tensor | None = None
        self._collapse_flag_buffer: Tensor | None = None
        self._topology_id_buffer: Tensor | None = None
        self._pool_power_demo_flag_buffer: Tensor | None = None
        self._demo_return_target_buffer: Tensor | None = None
        self._demo_return_valid_buffer: Tensor | None = None

    def __len__(self) -> int:
        return int(self._size)

    def _batch_from_indices(self, indices: Tensor) -> TensorReplayBatch:
        obs = {key: buffer.index_select(0, indices) for key, buffer in self._obs_buffers.items()}
        next_obs = {key: buffer.index_select(0, indices) for key, buffer in self._next_obs_buffers.items()}
        assert self._reward_buffer is not None
        assert self._done_buffer is not None
        assert self._is_demo_buffer is not None
        assert self._collapse_flag_buffer is not None
        assert self._topology_id_buffer is not None
        assert self._pool_power_demo_flag_buffer is not None
        assert self._demo_return_target_buffer is not None
        assert self._demo_return_valid_buffer is not None
        return TensorReplayBatch(
            obs=obs,
            action=TensorReplayActionRecord(
                allocation=self._action_buffers["allocation"].index_select(0, indices),
            ),
            reward=self._reward_buffer.index_select(0, indices),
            next_obs=next_obs,
            done=self._done_buffer.index_select(0, indices),
            is_demo=self._is_demo_buffer.index_select(0, indices),
            collapse_flag=self._collapse_flag_buffer.index_select(0, indices),
            topology_id=self._topology_id_buffer.index_select(0, indices),
            pool_power_demo_flag=self._pool_power_demo_flag_buffer.index_select(0, indices),
            demo_return_target=self._demo_return_target_buffer.index_select(0, indices),
            demo_return_valid=self._demo_return_valid_buffer.index_select(0, indices),
        )

    def export_all(self) -> TensorReplayBatch | None:
        if self._size <= 0 or not self._is_initialized():
            return None
        indices = torch.arange(self._size, dtype=torch.int64, device="cpu")
        return self._batch_from_indices(indices)

    def _is_initialized(self) -> bool:
        return self._reward_buffer is not None

    def _allocate_from_transition(self, transition: TensorTransition) -> None:
        self._obs_buffers = {
            key: torch.empty((self.capacity, *tuple(value.shape)), dtype=value.dtype, device="cpu")
            for key, value in transition.obs.items()
        }
        self._next_obs_buffers = {
            key: torch.empty((self.capacity, *tuple(value.shape)), dtype=value.dtype, device="cpu")
            for key, value in transition.next_obs.items()
        }
        self._action_buffers = {
            "allocation": torch.empty(
                (self.capacity, *tuple(transition.action.allocation.shape)),
                dtype=transition.action.allocation.dtype,
                device="cpu",
            ),
        }
        self._reward_buffer = torch.empty(self.capacity, dtype=transition.reward.dtype, device="cpu")
        self._done_buffer = torch.empty(self.capacity, dtype=transition.done.dtype, device="cpu")
        self._is_demo_buffer = torch.empty(self.capacity, dtype=transition.is_demo.dtype, device="cpu")
        self._collapse_flag_buffer = torch.empty(self.capacity, dtype=transition.collapse_flag.dtype, device="cpu")
        self._topology_id_buffer = torch.empty(self.capacity, dtype=transition.topology_id.dtype, device="cpu")
        self._pool_power_demo_flag_buffer = torch.empty(
            self.capacity,
            dtype=transition.pool_power_demo_flag.dtype,
            device="cpu",
        )
        self._demo_return_target_buffer = torch.empty(
            self.capacity,
            dtype=transition.demo_return_target.dtype,
            device="cpu",
        )
        self._demo_return_valid_buffer = torch.empty(
            self.capacity,
            dtype=transition.demo_return_valid.dtype,
            device="cpu",
        )

    def _allocate_from_batch(self, batch: TensorReplayBatch) -> None:
        self._obs_buffers = {
            key: torch.empty((self.capacity, *tuple(value.shape[1:])), dtype=value.dtype, device="cpu")
            for key, value in batch.obs.items()
        }
        self._next_obs_buffers = {
            key: torch.empty((self.capacity, *tuple(value.shape[1:])), dtype=value.dtype, device="cpu")
            for key, value in batch.next_obs.items()
        }
        self._action_buffers = {
            "allocation": torch.empty(
                (self.capacity, *tuple(batch.action.allocation.shape[1:])),
                dtype=batch.action.allocation.dtype,
                device="cpu",
            ),
        }
        self._reward_buffer = torch.empty(self.capacity, dtype=batch.reward.dtype, device="cpu")
        self._done_buffer = torch.empty(self.capacity, dtype=batch.done.dtype, device="cpu")
        self._is_demo_buffer = torch.empty(self.capacity, dtype=batch.is_demo.dtype, device="cpu")
        self._collapse_flag_buffer = torch.empty(self.capacity, dtype=batch.collapse_flag.dtype, device="cpu")
        self._topology_id_buffer = torch.empty(self.capacity, dtype=batch.topology_id.dtype, device="cpu")
        self._pool_power_demo_flag_buffer = torch.empty(
            self.capacity,
            dtype=batch.pool_power_demo_flag.dtype,
            device="cpu",
        )
        self._demo_return_target_buffer = torch.empty(
            self.capacity,
            dtype=batch.demo_return_target.dtype,
            device="cpu",
        )
        self._demo_return_valid_buffer = torch.empty(
            self.capacity,
            dtype=batch.demo_return_valid.dtype,
            device="cpu",
        )

    def _validate_transition_structure(self, transition: TensorTransition) -> None:
        if set(transition.obs.keys()) != set(self._obs_buffers.keys()):
            raise ValueError("Observation keys do not match replay buffer schema.")
        if set(transition.next_obs.keys()) != set(self._next_obs_buffers.keys()):
            raise ValueError("Next-observation keys do not match replay buffer schema.")
        for key, value in transition.obs.items():
            buffer = self._obs_buffers[key]
            if tuple(value.shape) != tuple(buffer.shape[1:]) or value.dtype != buffer.dtype:
                raise ValueError("Observation field '{0}' is incompatible with replay buffer schema.".format(key))
        for key, value in transition.next_obs.items():
            buffer = self._next_obs_buffers[key]
            if tuple(value.shape) != tuple(buffer.shape[1:]) or value.dtype != buffer.dtype:
                raise ValueError("Next-observation field '{0}' is incompatible with replay buffer schema.".format(key))
        action_buffer = self._action_buffers["allocation"]
        if (
            tuple(transition.action.allocation.shape) != tuple(action_buffer.shape[1:])
            or transition.action.allocation.dtype != action_buffer.dtype
        ):
            raise ValueError("Action field 'allocation' is incompatible with replay buffer schema.")
        if self._is_demo_buffer is None or self._collapse_flag_buffer is None:
            raise ValueError("Replay metadata buffers are not initialized.")
        if transition.is_demo.shape != torch.Size([]) or transition.is_demo.dtype != self._is_demo_buffer.dtype:
            raise ValueError("Transition field 'is_demo' is incompatible with replay buffer schema.")
        if (
            transition.collapse_flag.shape != torch.Size([])
            or transition.collapse_flag.dtype != self._collapse_flag_buffer.dtype
        ):
            raise ValueError("Transition field 'collapse_flag' is incompatible with replay buffer schema.")
        if self._topology_id_buffer is None or transition.topology_id.dtype != self._topology_id_buffer.dtype:
            raise ValueError("Transition field 'topology_id' is incompatible with replay buffer schema.")
        if (
            self._pool_power_demo_flag_buffer is None
            or transition.pool_power_demo_flag.dtype != self._pool_power_demo_flag_buffer.dtype
        ):
            raise ValueError("Transition field 'pool_power_demo_flag' is incompatible with replay buffer schema.")
        if (
            self._demo_return_target_buffer is None
            or transition.demo_return_target.shape != torch.Size([])
            or transition.demo_return_target.dtype != self._demo_return_target_buffer.dtype
        ):
            raise ValueError("Transition field 'demo_return_target' is incompatible with replay buffer schema.")
        if (
            self._demo_return_valid_buffer is None
            or transition.demo_return_valid.shape != torch.Size([])
            or transition.demo_return_valid.dtype != self._demo_return_valid_buffer.dtype
        ):
            raise ValueError("Transition field 'demo_return_valid' is incompatible with replay buffer schema.")

    def _validate_batch_structure(self, batch: TensorReplayBatch) -> None:
        if set(batch.obs.keys()) != set(self._obs_buffers.keys()):
            raise ValueError("Observation keys do not match replay buffer schema.")
        if set(batch.next_obs.keys()) != set(self._next_obs_buffers.keys()):
            raise ValueError("Next-observation keys do not match replay buffer schema.")
        for key, value in batch.obs.items():
            buffer = self._obs_buffers[key]
            if value.ndim < 1 or tuple(value.shape[1:]) != tuple(buffer.shape[1:]) or value.dtype != buffer.dtype:
                raise ValueError("Observation batch field '{0}' is incompatible with replay buffer schema.".format(key))
        for key, value in batch.next_obs.items():
            buffer = self._next_obs_buffers[key]
            if value.ndim < 1 or tuple(value.shape[1:]) != tuple(buffer.shape[1:]) or value.dtype != buffer.dtype:
                raise ValueError("Next-observation batch field '{0}' is incompatible with replay buffer schema.".format(key))
        action_buffer = self._action_buffers["allocation"]
        if (
            batch.action.allocation.ndim < 1
            or tuple(batch.action.allocation.shape[1:]) != tuple(action_buffer.shape[1:])
            or batch.action.allocation.dtype != action_buffer.dtype
        ):
            raise ValueError("Action batch field 'allocation' is incompatible with replay buffer schema.")
        if self._is_demo_buffer is None or self._collapse_flag_buffer is None:
            raise ValueError("Replay metadata buffers are not initialized.")
        if batch.is_demo.ndim != 1 or batch.is_demo.dtype != self._is_demo_buffer.dtype:
            raise ValueError("Replay batch field 'is_demo' is incompatible with replay buffer schema.")
        if batch.collapse_flag.ndim != 1 or batch.collapse_flag.dtype != self._collapse_flag_buffer.dtype:
            raise ValueError("Replay batch field 'collapse_flag' is incompatible with replay buffer schema.")
        if self._topology_id_buffer is None or batch.topology_id.dtype != self._topology_id_buffer.dtype:
            raise ValueError("Replay batch field 'topology_id' is incompatible with replay buffer schema.")
        if (
            self._pool_power_demo_flag_buffer is None
            or batch.pool_power_demo_flag.dtype != self._pool_power_demo_flag_buffer.dtype
        ):
            raise ValueError("Replay batch field 'pool_power_demo_flag' is incompatible with replay buffer schema.")
        if (
            self._demo_return_target_buffer is None
            or batch.demo_return_target.ndim != 1
            or batch.demo_return_target.dtype != self._demo_return_target_buffer.dtype
        ):
            raise ValueError("Replay batch field 'demo_return_target' is incompatible with replay buffer schema.")
        if (
            self._demo_return_valid_buffer is None
            or batch.demo_return_valid.ndim != 1
            or batch.demo_return_valid.dtype != self._demo_return_valid_buffer.dtype
        ):
            raise ValueError("Replay batch field 'demo_return_valid' is incompatible with replay buffer schema.")

    @staticmethod
    def _batch_is_cpu(batch: TensorReplayBatch) -> bool:
        tensors = list(batch.obs.values()) + [
            batch.action.allocation,
            batch.reward,
            batch.done,
            batch.is_demo,
            batch.collapse_flag,
            batch.topology_id,
            batch.pool_power_demo_flag,
            batch.demo_return_target,
            batch.demo_return_valid,
        ] + list(batch.next_obs.values())
        return all(tensor.device.type == "cpu" for tensor in tensors)

    def _write_transition_at_index(self, index: int, transition: TensorTransition) -> None:
        for key, value in transition.obs.items():
            self._obs_buffers[key][index].copy_(value)
        for key, value in transition.next_obs.items():
            self._next_obs_buffers[key][index].copy_(value)
        self._action_buffers["allocation"][index].copy_(transition.action.allocation)
        assert self._reward_buffer is not None
        assert self._done_buffer is not None
        assert self._is_demo_buffer is not None
        assert self._collapse_flag_buffer is not None
        assert self._topology_id_buffer is not None
        assert self._pool_power_demo_flag_buffer is not None
        assert self._demo_return_target_buffer is not None
        assert self._demo_return_valid_buffer is not None
        self._reward_buffer[index] = transition.reward
        self._done_buffer[index] = transition.done
        self._is_demo_buffer[index] = transition.is_demo
        self._collapse_flag_buffer[index] = transition.collapse_flag
        self._topology_id_buffer[index] = transition.topology_id
        self._pool_power_demo_flag_buffer[index] = transition.pool_power_demo_flag
        self._demo_return_target_buffer[index] = transition.demo_return_target
        self._demo_return_valid_buffer[index] = transition.demo_return_valid

    def _write_batch_at_indices(self, indices: Tensor, batch: TensorReplayBatch) -> None:
        for key, value in batch.obs.items():
            self._obs_buffers[key].index_copy_(0, indices, value)
        for key, value in batch.next_obs.items():
            self._next_obs_buffers[key].index_copy_(0, indices, value)
        self._action_buffers["allocation"].index_copy_(0, indices, batch.action.allocation)
        assert self._reward_buffer is not None
        assert self._done_buffer is not None
        assert self._is_demo_buffer is not None
        assert self._collapse_flag_buffer is not None
        assert self._topology_id_buffer is not None
        assert self._pool_power_demo_flag_buffer is not None
        assert self._demo_return_target_buffer is not None
        assert self._demo_return_valid_buffer is not None
        self._reward_buffer.index_copy_(0, indices, batch.reward)
        self._done_buffer.index_copy_(0, indices, batch.done)
        self._is_demo_buffer.index_copy_(0, indices, batch.is_demo)
        self._collapse_flag_buffer.index_copy_(0, indices, batch.collapse_flag)
        self._topology_id_buffer.index_copy_(0, indices, batch.topology_id)
        self._pool_power_demo_flag_buffer.index_copy_(0, indices, batch.pool_power_demo_flag)
        self._demo_return_target_buffer.index_copy_(0, indices, batch.demo_return_target)
        self._demo_return_valid_buffer.index_copy_(0, indices, batch.demo_return_valid)

    def _select_write_index(self) -> int | None:
        if self.replacement_policy == "ring":
            index = int(self._next_index)
            self._next_index = int((self._next_index + 1) % self.capacity)
            self._size = min(self._size + 1, self.capacity)
            self._seen_count += 1
            return index

        self._seen_count += 1
        if self._size < self.capacity:
            index = int(self._size)
            self._size += 1
            return index
        replacement_index = int(self._rng.integers(0, self._seen_count))
        if replacement_index >= self.capacity:
            return None
        return replacement_index

    def add(self, transition: TensorTransition | Transition) -> None:
        tensor_transition = _coerce_transition(transition)
        if not self._is_initialized():
            self._allocate_from_transition(tensor_transition)
        else:
            self._validate_transition_structure(tensor_transition)
        index = self._select_write_index()
        if index is None:
            return
        self._write_transition_at_index(index, tensor_transition)

    def extend(self, batch: TensorReplayBatch) -> None:
        if len(batch) == 0:
            return
        cpu_batch = batch if self._batch_is_cpu(batch) else batch.to("cpu")
        if not self._is_initialized():
            self._allocate_from_batch(cpu_batch)
        else:
            self._validate_batch_structure(cpu_batch)

        if self.replacement_policy == "ring":
            batch_size = len(cpu_batch)
            indices = (
                torch.arange(batch_size, dtype=torch.int64, device="cpu") + int(self._next_index)
            ) % int(self.capacity)
            self._write_batch_at_indices(indices, cpu_batch)
            self._next_index = int((self._next_index + batch_size) % self.capacity)
            self._size = min(self._size + batch_size, self.capacity)
            self._seen_count += batch_size
            return

        batch_size = len(cpu_batch)
        batch_offset = 0

        if self._size < self.capacity:
            fill_count = min(int(self.capacity - self._size), int(batch_size))
            if fill_count > 0:
                fill_indices = torch.arange(
                    int(self._size),
                    int(self._size) + fill_count,
                    dtype=torch.int64,
                    device="cpu",
                )
                self._write_batch_at_indices(
                    fill_indices,
                    _slice_replay_batch_range(cpu_batch, 0, fill_count),
                )
                self._size += fill_count
                self._seen_count += fill_count
                batch_offset = fill_count

        remaining = int(batch_size - batch_offset)
        if remaining <= 0:
            return

        assert self._size == self.capacity
        seen_highs = np.arange(
            int(self._seen_count) + 1,
            int(self._seen_count) + remaining + 1,
            dtype=np.int64,
        )
        replacement_indices = np.asarray(
            [int(self._rng.integers(0, int(high))) for high in seen_highs],
            dtype=np.int64,
        )
        valid_mask = replacement_indices < int(self.capacity)
        self._seen_count += remaining
        if not np.any(valid_mask):
            return

        source_indices = torch.as_tensor(
            batch_offset + np.flatnonzero(valid_mask),
            dtype=torch.int64,
            device="cpu",
        )
        target_indices = torch.as_tensor(
            replacement_indices[valid_mask],
            dtype=torch.int64,
            device="cpu",
        )
        deduplicated_replacements: dict[int, int] = {}
        for source_index, target_index in zip(source_indices.tolist(), target_indices.tolist()):
            deduplicated_replacements[int(target_index)] = int(source_index)
        target_indices = torch.as_tensor(
            list(deduplicated_replacements.keys()),
            dtype=torch.int64,
            device="cpu",
        )
        source_indices = torch.as_tensor(
            list(deduplicated_replacements.values()),
            dtype=torch.int64,
            device="cpu",
        )
        self._write_batch_at_indices(
            target_indices,
            _slice_replay_batch(cpu_batch, source_indices),
        )

    def _sample_indices_up_to(
        self,
        batch_size: int,
        max_collapse_ratio: float | None,
        strict_max_collapse_ratio: bool,
    ) -> Tensor:
        if self._size <= 0:
            return torch.empty(0, dtype=torch.int64, device="cpu")
        requested = min(int(batch_size), int(self._size))
        if requested <= 0:
            return torch.empty(0, dtype=torch.int64, device="cpu")
        if max_collapse_ratio is None or self._collapse_flag_buffer is None:
            return torch.as_tensor(
                self._rng.choice(self._size, size=requested, replace=False),
                dtype=torch.int64,
                device="cpu",
            )

        ratio = min(max(float(max_collapse_ratio), 0.0), 1.0)
        if ratio >= 1.0:
            return torch.as_tensor(
                self._rng.choice(self._size, size=requested, replace=False),
                dtype=torch.int64,
                device="cpu",
            )

        collapse_flags = self._collapse_flag_buffer[: self._size].detach().cpu().numpy().astype(np.bool_, copy=False)
        collapse_indices = np.flatnonzero(collapse_flags)
        non_collapse_indices = np.flatnonzero(~collapse_flags)

        if not strict_max_collapse_ratio:
            if collapse_indices.size == 0 or non_collapse_indices.size == 0:
                return torch.as_tensor(
                    self._rng.choice(self._size, size=requested, replace=False),
                    dtype=torch.int64,
                    device="cpu",
                )

        if ratio <= 0.0:
            take_non_collapse = min(int(non_collapse_indices.size), requested)
            if take_non_collapse <= 0:
                return torch.empty(0, dtype=torch.int64, device="cpu")
            sampled_indices = self._rng.choice(non_collapse_indices, size=take_non_collapse, replace=False)
            return torch.as_tensor(sampled_indices, dtype=torch.int64, device="cpu")

        take_non_collapse = min(int(non_collapse_indices.size), requested)
        if strict_max_collapse_ratio:
            allowed_collapse_by_non_collapse = int(
                np.floor((ratio * float(take_non_collapse)) / max(1.0 - ratio, 1e-12))
            )
            take_collapse = min(int(collapse_indices.size), requested - take_non_collapse, allowed_collapse_by_non_collapse)
        else:
            max_collapse = int(np.floor(float(requested) * ratio))
            take_collapse = min(int(collapse_indices.size), max_collapse)
            take_non_collapse = min(int(non_collapse_indices.size), requested - take_collapse)
            remaining = requested - take_non_collapse - take_collapse
            if remaining > 0:
                extra_non_collapse = min(int(non_collapse_indices.size) - take_non_collapse, remaining)
                take_non_collapse += extra_non_collapse
                remaining -= extra_non_collapse
            if remaining > 0:
                extra_collapse = min(int(collapse_indices.size) - take_collapse, remaining)
                take_collapse += extra_collapse

        if take_non_collapse <= 0 and take_collapse <= 0:
            return torch.empty(0, dtype=torch.int64, device="cpu")

        sampled_parts: list[np.ndarray] = []
        if take_non_collapse > 0:
            sampled_parts.append(self._rng.choice(non_collapse_indices, size=take_non_collapse, replace=False))
        if take_collapse > 0:
            sampled_parts.append(self._rng.choice(collapse_indices, size=take_collapse, replace=False))
        sampled_indices = np.concatenate(sampled_parts, axis=0)
        self._rng.shuffle(sampled_indices)
        return torch.as_tensor(sampled_indices, dtype=torch.int64, device="cpu")

    def _sample_candidate_indices_up_to(
        self,
        candidate_indices: np.ndarray,
        batch_size: int,
        max_collapse_ratio: float | None,
        strict_max_collapse_ratio: bool,
    ) -> Tensor:
        if batch_size <= 0 or candidate_indices.size <= 0:
            return torch.empty(0, dtype=torch.int64, device="cpu")

        requested = min(int(batch_size), int(candidate_indices.size))
        if requested <= 0:
            return torch.empty(0, dtype=torch.int64, device="cpu")

        if max_collapse_ratio is None or self._collapse_flag_buffer is None:
            sampled_indices = self._rng.choice(candidate_indices, size=requested, replace=False)
            return torch.as_tensor(sampled_indices, dtype=torch.int64, device="cpu")

        ratio = min(max(float(max_collapse_ratio), 0.0), 1.0)
        if ratio >= 1.0:
            sampled_indices = self._rng.choice(candidate_indices, size=requested, replace=False)
            return torch.as_tensor(sampled_indices, dtype=torch.int64, device="cpu")

        collapse_flags = self._collapse_flag_buffer[candidate_indices].detach().cpu().numpy().astype(np.bool_, copy=False)
        collapse_indices = candidate_indices[np.flatnonzero(collapse_flags)]
        non_collapse_indices = candidate_indices[np.flatnonzero(~collapse_flags)]

        if not strict_max_collapse_ratio:
            if collapse_indices.size == 0 or non_collapse_indices.size == 0:
                sampled_indices = self._rng.choice(candidate_indices, size=requested, replace=False)
                return torch.as_tensor(sampled_indices, dtype=torch.int64, device="cpu")

        if ratio <= 0.0:
            take_non_collapse = min(int(non_collapse_indices.size), requested)
            if take_non_collapse <= 0:
                return torch.empty(0, dtype=torch.int64, device="cpu")
            sampled_indices = self._rng.choice(non_collapse_indices, size=take_non_collapse, replace=False)
            return torch.as_tensor(sampled_indices, dtype=torch.int64, device="cpu")

        take_non_collapse = min(int(non_collapse_indices.size), requested)
        if strict_max_collapse_ratio:
            allowed_collapse_by_non_collapse = int(
                np.floor((ratio * float(take_non_collapse)) / max(1.0 - ratio, 1e-12))
            )
            take_collapse = min(int(collapse_indices.size), requested - take_non_collapse, allowed_collapse_by_non_collapse)
        else:
            max_collapse = int(np.floor(float(requested) * ratio))
            take_collapse = min(int(collapse_indices.size), max_collapse)
            take_non_collapse = min(int(non_collapse_indices.size), requested - take_collapse)
            remaining = requested - take_non_collapse - take_collapse
            if remaining > 0:
                extra_non_collapse = min(int(non_collapse_indices.size) - take_non_collapse, remaining)
                take_non_collapse += extra_non_collapse
                remaining -= extra_non_collapse
            if remaining > 0:
                extra_collapse = min(int(collapse_indices.size) - take_collapse, remaining)
                take_collapse += extra_collapse

        if take_non_collapse <= 0 and take_collapse <= 0:
            return torch.empty(0, dtype=torch.int64, device="cpu")

        sampled_parts: list[np.ndarray] = []
        if take_non_collapse > 0:
            sampled_parts.append(self._rng.choice(non_collapse_indices, size=take_non_collapse, replace=False))
        if take_collapse > 0:
            sampled_parts.append(self._rng.choice(collapse_indices, size=take_collapse, replace=False))
        sampled_indices = np.concatenate(sampled_parts, axis=0)
        self._rng.shuffle(sampled_indices)
        return torch.as_tensor(sampled_indices, dtype=torch.int64, device="cpu")

    def sample_up_to(
        self,
        batch_size: int,
        max_collapse_ratio: float | None = None,
        strict_max_collapse_ratio: bool = False,
    ) -> TensorReplayBatch | None:
        if batch_size <= 0 or self._size <= 0 or not self._is_initialized():
            return None
        indices = self._sample_indices_up_to(
            batch_size=batch_size,
            max_collapse_ratio=max_collapse_ratio,
            strict_max_collapse_ratio=strict_max_collapse_ratio,
        )
        if indices.numel() == 0:
            return None
        return self._batch_from_indices(indices)

    def sample_filtered_up_to(
        self,
        batch_size: int,
        *,
        max_collapse_ratio: float | None = None,
        strict_max_collapse_ratio: bool = False,
        require_demo: bool = False,
        require_pool_power_demo: bool = False,
    ) -> TensorReplayBatch | None:
        if batch_size <= 0 or self._size <= 0 or not self._is_initialized():
            return None
        candidate_mask = np.ones(self._size, dtype=np.bool_)
        if require_demo:
            assert self._is_demo_buffer is not None
            candidate_mask &= self._is_demo_buffer[: self._size].detach().cpu().numpy().astype(np.bool_, copy=False)
        if require_pool_power_demo:
            assert self._pool_power_demo_flag_buffer is not None
            candidate_mask &= self._pool_power_demo_flag_buffer[: self._size].detach().cpu().numpy().astype(np.bool_, copy=False)
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size <= 0:
            return None
        indices = self._sample_candidate_indices_up_to(
            candidate_indices=candidate_indices,
            batch_size=batch_size,
            max_collapse_ratio=max_collapse_ratio,
            strict_max_collapse_ratio=strict_max_collapse_ratio,
        )
        if indices.numel() == 0:
            return None
        return self._batch_from_indices(indices)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        max_collapse_ratio: float | None = None,
    ) -> TensorReplayBatch:
        batch = self.sample_up_to(
            batch_size=batch_size,
            max_collapse_ratio=max_collapse_ratio,
            strict_max_collapse_ratio=False,
        )
        if batch is None:
            raise ValueError("Cannot sample from an empty replay storage.")
        if device is not None:
            return batch.to(device)
        return batch

    def sample_demo(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        max_collapse_ratio: float | None = None,
    ) -> TensorReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        with self._lock:
            if self.replay_strategy == "fifo":
                assert self._fifo_storage is not None
                batch = self._fifo_storage.sample_filtered_up_to(
                    batch_size=batch_size,
                    max_collapse_ratio=max_collapse_ratio,
                    strict_max_collapse_ratio=False,
                    require_demo=True,
                    require_pool_power_demo=(self.demo_behavior_source == "pool_power_mix"),
                )
                if batch is None:
                    raise ValueError("Cannot sample demo batch from an empty replay buffer.")
                batch = _with_replay_source_id(batch, "demo")
                topology_ids = batch.topology_id.detach().cpu().numpy().astype(np.int64, copy=False)
                topology_counts: dict[str, int] = {}
                for topology_id in np.unique(topology_ids):
                    topology_name = self._resolve_topology_name(int(topology_id))
                    topology_counts[topology_name] = int(np.sum(topology_ids == int(topology_id)))
                sample_size = max(len(batch), 1)
                self._last_sample_stats = {
                    "replay_sample_size": float(len(batch)),
                    "replay_source_frac_demo": 1.0,
                    **{
                        "replay_topology_frac_{0}".format(topology_name): float(count) / float(sample_size)
                        for topology_name, count in sorted(topology_counts.items())
                    },
                }
            else:
                active_topologies = [
                    topology_name
                    for topology_name in self.topology_names
                    if (
                        (topology_name in self._demo_storages and len(self._demo_storages[topology_name]) > 0)
                        or (topology_name in self._recent_storages and self._recent_storages[topology_name].demo_size() > 0)
                        or (topology_name in self._long_term_storages and self._long_term_storages[topology_name].demo_size() > 0)
                    )
                ]
                if not active_topologies:
                    raise ValueError("Cannot sample demo batch from an empty replay buffer.")
                topology_counts = _allocate_integer_counts(batch_size, [1.0 for _ in active_topologies])
                sampled_batches: list[TensorReplayBatch] = []
                actual_topology_counts: dict[str, int] = {}

                for topology_name, requested in zip(active_topologies, topology_counts):
                    if requested <= 0:
                        continue
                    remaining_for_topology = int(requested)
                    topology_samples: list[TensorReplayBatch] = []
                    preferred_storage = self._demo_storages.get(topology_name)
                    if preferred_storage is not None and len(preferred_storage) > 0:
                        sample = preferred_storage.sample_up_to(
                            remaining_for_topology,
                            max_collapse_ratio=max_collapse_ratio,
                            strict_max_collapse_ratio=True,
                        )
                        if sample is not None:
                            topology_samples.append(_with_replay_source_id(sample, "demo"))
                            remaining_for_topology -= len(sample)
                    for fallback_storage in (
                        self._recent_storages.get(topology_name),
                        self._long_term_storages.get(topology_name),
                    ):
                        if remaining_for_topology <= 0 or fallback_storage is None or len(fallback_storage) <= 0:
                            continue
                        sample = fallback_storage.sample_filtered_up_to(
                            remaining_for_topology,
                            max_collapse_ratio=max_collapse_ratio,
                            strict_max_collapse_ratio=True,
                            require_demo=True,
                            require_pool_power_demo=(self.demo_behavior_source == "pool_power_mix"),
                        )
                        if sample is None:
                            continue
                        topology_samples.append(_with_replay_source_id(sample, "demo"))
                        remaining_for_topology -= len(sample)
                    if not topology_samples:
                        continue
                    combined_topology_batch = _concat_replay_batches(topology_samples)
                    sampled_batches.append(combined_topology_batch)
                    actual_topology_counts[topology_name] = (
                        actual_topology_counts.get(topology_name, 0) + len(combined_topology_batch)
                    )

                remaining = batch_size - sum(len(batch_item) for batch_item in sampled_batches)
                while remaining > 0:
                    made_progress = False
                    for topology_name in active_topologies:
                        if remaining <= 0:
                            break
                        sample = None
                        preferred_storage = self._demo_storages.get(topology_name)
                        if preferred_storage is not None and len(preferred_storage) > 0:
                            sample = preferred_storage.sample_up_to(
                                1,
                                max_collapse_ratio=max_collapse_ratio,
                                strict_max_collapse_ratio=True,
                            )
                        if sample is None:
                            for fallback_storage in (
                                self._recent_storages.get(topology_name),
                                self._long_term_storages.get(topology_name),
                            ):
                                if fallback_storage is None or len(fallback_storage) <= 0:
                                    continue
                                sample = fallback_storage.sample_filtered_up_to(
                                    1,
                                    max_collapse_ratio=max_collapse_ratio,
                                    strict_max_collapse_ratio=True,
                                    require_demo=True,
                                    require_pool_power_demo=(self.demo_behavior_source == "pool_power_mix"),
                                )
                                if sample is not None:
                                    break
                        if sample is None:
                            continue
                        sampled_batches.append(_with_replay_source_id(sample, "demo"))
                        actual_topology_counts[topology_name] = actual_topology_counts.get(topology_name, 0) + len(sample)
                        remaining -= len(sample)
                        made_progress = True
                    if not made_progress:
                        break

                if not sampled_batches:
                    raise ValueError("Cannot sample demo batch from an empty replay buffer.")
                batch = _shuffle_replay_batch(_concat_replay_batches(sampled_batches), self._rng)
                sample_size = max(len(batch), 1)
                self._last_sample_stats = {
                    "replay_sample_size": float(len(batch)),
                    "replay_source_frac_demo": 1.0,
                    **{
                        "replay_topology_frac_{0}".format(topology_name): float(count) / float(sample_size)
                        for topology_name, count in sorted(actual_topology_counts.items())
                    },
                }

        if device is not None:
            return batch.to(device)
        return batch

    def demo_size(self, *, require_pool_power_demo: bool = False) -> int:
        if self._size <= 0 or self._is_demo_buffer is None:
            return 0
        demo_mask = self._is_demo_buffer[: self._size].detach().cpu().numpy().astype(np.bool_, copy=False)
        if require_pool_power_demo:
            assert self._pool_power_demo_flag_buffer is not None
            pool_power_mask = self._pool_power_demo_flag_buffer[: self._size].detach().cpu().numpy().astype(
                np.bool_,
                copy=False,
            )
            demo_mask = np.logical_and(demo_mask, pool_power_mask)
        return int(np.sum(demo_mask))

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": int(self.capacity),
            "replacement_policy": str(self.replacement_policy),
            "next_index": int(self._next_index),
            "size": int(self._size),
            "seen_count": int(self._seen_count),
            "rng_state": self._rng.bit_generator.state,
            "initialized": self._is_initialized(),
            "obs_buffers": {key: value.detach().cpu().clone() for key, value in self._obs_buffers.items()},
            "next_obs_buffers": {key: value.detach().cpu().clone() for key, value in self._next_obs_buffers.items()},
            "action_buffers": {key: value.detach().cpu().clone() for key, value in self._action_buffers.items()},
            "reward_buffer": None if self._reward_buffer is None else self._reward_buffer.detach().cpu().clone(),
            "done_buffer": None if self._done_buffer is None else self._done_buffer.detach().cpu().clone(),
            "is_demo_buffer": None if self._is_demo_buffer is None else self._is_demo_buffer.detach().cpu().clone(),
            "collapse_flag_buffer": (
                None if self._collapse_flag_buffer is None else self._collapse_flag_buffer.detach().cpu().clone()
            ),
            "topology_id_buffer": (
                None if self._topology_id_buffer is None else self._topology_id_buffer.detach().cpu().clone()
            ),
            "pool_power_demo_flag_buffer": (
                None
                if self._pool_power_demo_flag_buffer is None
                else self._pool_power_demo_flag_buffer.detach().cpu().clone()
            ),
            "demo_return_target_buffer": (
                None
                if self._demo_return_target_buffer is None
                else self._demo_return_target_buffer.detach().cpu().clone()
            ),
            "demo_return_valid_buffer": (
                None
                if self._demo_return_valid_buffer is None
                else self._demo_return_valid_buffer.detach().cpu().clone()
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "buffer" in state_dict:
            self._load_legacy_state_dict(state_dict)
            return

        self.capacity = int(state_dict["capacity"])
        self.replacement_policy = str(state_dict.get("replacement_policy", "ring"))
        self._next_index = int(state_dict["next_index"])
        self._size = int(state_dict["size"])
        self._seen_count = int(state_dict.get("seen_count", self._size))
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state_dict["rng_state"]

        initialized = bool(state_dict.get("initialized", False))
        if not initialized:
            self._obs_buffers = {}
            self._next_obs_buffers = {}
            self._action_buffers = {}
            self._reward_buffer = None
            self._done_buffer = None
            self._is_demo_buffer = None
            self._collapse_flag_buffer = None
            self._topology_id_buffer = None
            self._pool_power_demo_flag_buffer = None
            self._demo_return_target_buffer = None
            self._demo_return_valid_buffer = None
            return

        self._obs_buffers = {
            key: torch.as_tensor(value, device="cpu").detach().clone()
            for key, value in dict(state_dict["obs_buffers"]).items()
        }
        self._next_obs_buffers = {
            key: torch.as_tensor(value, device="cpu").detach().clone()
            for key, value in dict(state_dict["next_obs_buffers"]).items()
        }
        self._action_buffers = {
            key: torch.as_tensor(value, device="cpu").detach().clone()
            for key, value in dict(state_dict["action_buffers"]).items()
        }
        reward_buffer = state_dict.get("reward_buffer")
        done_buffer = state_dict.get("done_buffer")
        is_demo_buffer = state_dict.get("is_demo_buffer")
        collapse_flag_buffer = state_dict.get("collapse_flag_buffer")
        topology_id_buffer = state_dict.get("topology_id_buffer")
        pool_power_demo_flag_buffer = state_dict.get("pool_power_demo_flag_buffer")
        demo_return_target_buffer = state_dict.get("demo_return_target_buffer")
        demo_return_valid_buffer = state_dict.get("demo_return_valid_buffer")
        self._reward_buffer = None if reward_buffer is None else torch.as_tensor(reward_buffer, device="cpu").detach().clone()
        self._done_buffer = None if done_buffer is None else torch.as_tensor(done_buffer, device="cpu").detach().clone()
        self._is_demo_buffer = (
            torch.zeros(self.capacity, dtype=torch.bool, device="cpu")
            if is_demo_buffer is None
            else torch.as_tensor(is_demo_buffer, device="cpu").detach().clone()
        )
        self._collapse_flag_buffer = (
            torch.zeros(self.capacity, dtype=torch.bool, device="cpu")
            if collapse_flag_buffer is None
            else torch.as_tensor(collapse_flag_buffer, device="cpu").detach().clone()
        )
        self._topology_id_buffer = (
            torch.full(self.capacity, 6, dtype=torch.int64, device="cpu")
            if topology_id_buffer is None
            else torch.as_tensor(topology_id_buffer, device="cpu").detach().clone()
        )
        self._pool_power_demo_flag_buffer = (
            torch.zeros(self.capacity, dtype=torch.bool, device="cpu")
            if pool_power_demo_flag_buffer is None
            else torch.as_tensor(pool_power_demo_flag_buffer, device="cpu").detach().clone()
        )
        self._demo_return_target_buffer = (
            torch.zeros(self.capacity, dtype=torch.float32, device="cpu")
            if demo_return_target_buffer is None
            else torch.as_tensor(demo_return_target_buffer, device="cpu").detach().clone()
        )
        self._demo_return_valid_buffer = (
            torch.zeros(self.capacity, dtype=torch.bool, device="cpu")
            if demo_return_valid_buffer is None
            else torch.as_tensor(demo_return_valid_buffer, device="cpu").detach().clone()
        )

    def _load_legacy_state_dict(self, state_dict: dict[str, Any]) -> None:
        capacity = int(state_dict["capacity"])
        buffer_state = list(state_dict["buffer"])
        if len(buffer_state) != capacity:
            raise ValueError("Replay buffer checkpoint is inconsistent with its declared capacity.")

        self.capacity = capacity
        self.replacement_policy = "ring"
        self._next_index = int(state_dict["next_index"])
        self._size = int(state_dict["size"])
        self._seen_count = int(self._size)
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state_dict["rng_state"]
        self._obs_buffers = {}
        self._next_obs_buffers = {}
        self._action_buffers = {}
        self._reward_buffer = None
        self._done_buffer = None
        self._is_demo_buffer = None
        self._collapse_flag_buffer = None
        self._topology_id_buffer = None
        self._pool_power_demo_flag_buffer = None
        self._demo_return_target_buffer = None
        self._demo_return_valid_buffer = None

        first_transition = next((item for item in buffer_state if item is not None), None)
        if first_transition is None:
            return

        reference_transition = _coerce_transition(first_transition)
        self._allocate_from_transition(reference_transition)
        for index, transition in enumerate(buffer_state):
            if transition is None:
                continue
            tensor_transition = _coerce_transition(transition)
            self._validate_transition_structure(tensor_transition)
            self._write_transition_at_index(index, tensor_transition)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        seed: int | None = None,
        replay_strategy: str = "fifo",
        topology_names: Sequence[str] | None = None,
        recent_fraction: float = 0.50,
        long_term_fraction: float = 0.35,
        demo_fraction: float = 0.15,
        demo_behavior_source: str = "pool_power_mix",
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if replay_strategy not in {"fifo", "topology_stratified_mixed"}:
            raise ValueError("replay_strategy must be one of {'fifo', 'topology_stratified_mixed'}.")
        if recent_fraction < 0.0 or long_term_fraction < 0.0 or demo_fraction < 0.0:
            raise ValueError("Replay source fractions must be non-negative.")
        if replay_strategy == "topology_stratified_mixed":
            total_fraction = float(recent_fraction + long_term_fraction + demo_fraction)
            if abs(total_fraction - 1.0) > 1e-6:
                raise ValueError("For topology_stratified_mixed replay, recent/long_term/demo fractions must sum to 1.")

        self.capacity = int(capacity)
        self.replay_strategy = str(replay_strategy)
        self.demo_behavior_source = str(demo_behavior_source)
        self.recent_fraction = float(recent_fraction)
        self.long_term_fraction = float(long_term_fraction)
        self.demo_fraction = float(demo_fraction)
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._last_sample_stats: dict[str, float] = {}

        if topology_names is None:
            normalized_topology_names = ["fixed"]
        else:
            normalized_topology_names = [str(item) for item in topology_names if str(item)]
            if not normalized_topology_names:
                normalized_topology_names = ["fixed"]
        self.topology_names = tuple(dict.fromkeys(normalized_topology_names))

        self._fifo_storage: _TensorReplayStorage | None = None
        self._recent_storages: dict[str, _TensorReplayStorage] = {}
        self._long_term_storages: dict[str, _TensorReplayStorage] = {}
        self._demo_storages: dict[str, _TensorReplayStorage] = {}

        self._initialize_storage()

    def _initialize_storage(self) -> None:
        if self.replay_strategy == "fifo":
            self._fifo_storage = _TensorReplayStorage(self.capacity, seed=self._seed, replacement_policy="ring")
            self._recent_storages = {}
            self._long_term_storages = {}
            self._demo_storages = {}
            return

        source_capacities = _allocate_integer_counts(
            self.capacity,
            [self.recent_fraction, self.long_term_fraction, self.demo_fraction],
        )
        recent_total, long_term_total, demo_total = source_capacities
        recent_caps = _allocate_integer_counts(recent_total, [1.0 for _ in self.topology_names]) if recent_total > 0 else [0 for _ in self.topology_names]
        long_caps = _allocate_integer_counts(long_term_total, [1.0 for _ in self.topology_names]) if long_term_total > 0 else [0 for _ in self.topology_names]
        demo_caps = _allocate_integer_counts(demo_total, [1.0 for _ in self.topology_names]) if demo_total > 0 else [0 for _ in self.topology_names]

        self._fifo_storage = None
        self._recent_storages = {
            name: _TensorReplayStorage(capacity=int(capacity), seed=self._seed, replacement_policy="ring")
            for name, capacity in zip(self.topology_names, recent_caps)
            if int(capacity) > 0
        }
        self._long_term_storages = {
            name: _TensorReplayStorage(capacity=int(capacity), seed=self._seed, replacement_policy="reservoir")
            for name, capacity in zip(self.topology_names, long_caps)
            if int(capacity) > 0
        }
        self._demo_storages = {
            name: _TensorReplayStorage(capacity=int(capacity), seed=self._seed, replacement_policy="ring")
            for name, capacity in zip(self.topology_names, demo_caps)
            if int(capacity) > 0
        }

    def _resolve_topology_name(self, topology_id: int) -> str:
        topology_name = topology_id_to_name(int(topology_id))
        if topology_name in self.topology_names:
            return topology_name
        return str(self.topology_names[0])

    def _canonical_export_storages(self) -> list[_TensorReplayStorage]:
        if self.replay_strategy == "fifo":
            assert self._fifo_storage is not None
            return [self._fifo_storage]
        if self._recent_storages:
            return list(self._recent_storages.values())
        if self._long_term_storages:
            return list(self._long_term_storages.values())
        return list(self._demo_storages.values())

    def _split_batch_by_topology(
        self,
        batch: TensorReplayBatch,
        require_pool_power_demo: bool = False,
    ) -> dict[str, TensorReplayBatch]:
        grouped: dict[str, TensorReplayBatch] = {}
        if len(batch) == 0:
            return grouped
        topology_ids = batch.topology_id.detach().cpu().numpy().astype(np.int64, copy=False)
        demo_flags = batch.pool_power_demo_flag.detach().cpu().numpy().astype(np.bool_, copy=False)
        for topology_id in np.unique(topology_ids):
            topology_mask = topology_ids == int(topology_id)
            if require_pool_power_demo:
                topology_mask = np.logical_and(topology_mask, demo_flags)
            if not np.any(topology_mask):
                continue
            indices = torch.as_tensor(np.flatnonzero(topology_mask), dtype=torch.int64, device="cpu")
            grouped[self._resolve_topology_name(int(topology_id))] = _slice_replay_batch(batch, indices)
        return grouped

    def _source_storages_for_topology(self, topology_name: str) -> list[tuple[str, _TensorReplayStorage, float]]:
        storages: list[tuple[str, _TensorReplayStorage, float]] = []
        demo_storage = self._demo_storages.get(topology_name)
        if demo_storage is not None and len(demo_storage) > 0 and self.demo_fraction > 0.0:
            storages.append(("demo", demo_storage, self.demo_fraction))
        recent_storage = self._recent_storages.get(topology_name)
        if recent_storage is not None and len(recent_storage) > 0 and self.recent_fraction > 0.0:
            storages.append(("recent", recent_storage, self.recent_fraction))
        long_term_storage = self._long_term_storages.get(topology_name)
        if long_term_storage is not None and len(long_term_storage) > 0 and self.long_term_fraction > 0.0:
            storages.append(("long_term", long_term_storage, self.long_term_fraction))
        return storages

    def _sample_from_topology(
        self,
        topology_name: str,
        requested: int,
        max_collapse_ratio: float | None,
    ) -> tuple[TensorReplayBatch | None, dict[str, int]]:
        if requested <= 0:
            return None, {}
        source_storages = self._source_storages_for_topology(topology_name)
        if not source_storages:
            return None, {}

        target_counts = _allocate_integer_counts(requested, [weight for _, _, weight in source_storages])
        sampled_batches: list[TensorReplayBatch] = []
        sampled_count = 0
        source_counts: dict[str, int] = {}

        for (source_name, storage, _), target_count in zip(source_storages, target_counts):
            if target_count <= 0:
                continue
            sample = storage.sample_up_to(
                target_count,
                max_collapse_ratio=max_collapse_ratio,
                strict_max_collapse_ratio=True,
            )
            if sample is None:
                continue
            sampled_batches.append(_with_replay_source_id(sample, source_name))
            sampled_count += len(sample)
            source_counts[source_name] = source_counts.get(source_name, 0) + len(sample)

        remaining = requested - sampled_count
        if remaining > 0:
            for source_name, storage, _ in source_storages:
                if remaining <= 0:
                    break
                sample = storage.sample_up_to(
                    remaining,
                    max_collapse_ratio=max_collapse_ratio,
                    strict_max_collapse_ratio=True,
                )
                if sample is None:
                    continue
                sampled_batches.append(_with_replay_source_id(sample, source_name))
                source_counts[source_name] = source_counts.get(source_name, 0) + len(sample)
                remaining -= len(sample)

        if not sampled_batches:
            return None, {}
        return _concat_replay_batches(sampled_batches), source_counts

    def _fifo_candidate_indices_by_topology(
        self,
        *,
        require_demo: bool = False,
        require_pool_power_demo: bool = False,
    ) -> dict[str, np.ndarray]:
        assert self._fifo_storage is not None
        storage = self._fifo_storage
        if len(storage) <= 0:
            return {}
        assert storage._topology_id_buffer is not None
        size = int(storage._size)
        topology_ids = storage._topology_id_buffer[:size].detach().cpu().numpy().astype(np.int64, copy=False)
        candidate_mask = np.ones(size, dtype=np.bool_)
        if require_demo:
            assert storage._is_demo_buffer is not None
            candidate_mask &= storage._is_demo_buffer[:size].detach().cpu().numpy().astype(np.bool_, copy=False)
        if require_pool_power_demo:
            assert storage._pool_power_demo_flag_buffer is not None
            candidate_mask &= storage._pool_power_demo_flag_buffer[:size].detach().cpu().numpy().astype(
                np.bool_,
                copy=False,
            )
        if not np.any(candidate_mask):
            return {}

        grouped: dict[str, np.ndarray] = {}
        for topology_id in np.unique(topology_ids[candidate_mask]):
            topology_mask = np.logical_and(candidate_mask, topology_ids == int(topology_id))
            topology_indices = np.flatnonzero(topology_mask)
            if topology_indices.size <= 0:
                continue
            grouped[self._resolve_topology_name(int(topology_id))] = topology_indices.astype(np.int64, copy=False)
        return grouped

    def _fifo_max_sample_count_for_topology(
        self,
        candidate_indices: np.ndarray,
        max_collapse_ratio: float | None,
    ) -> int:
        assert self._fifo_storage is not None
        storage = self._fifo_storage
        if candidate_indices.size <= 0:
            return 0
        if max_collapse_ratio is None or storage._collapse_flag_buffer is None:
            return int(candidate_indices.size)

        ratio = min(max(float(max_collapse_ratio), 0.0), 1.0)
        if ratio >= 1.0:
            return int(candidate_indices.size)

        collapse_flags = storage._collapse_flag_buffer[candidate_indices].detach().cpu().numpy().astype(
            np.bool_,
            copy=False,
        )
        collapse_count = int(np.sum(collapse_flags))
        non_collapse_count = int(candidate_indices.size) - collapse_count
        if ratio <= 0.0:
            return non_collapse_count
        if non_collapse_count <= 0:
            return 0
        allowed_collapse = int(
            np.floor((ratio * float(non_collapse_count)) / max(1.0 - ratio, 1e-12))
        )
        return non_collapse_count + min(collapse_count, allowed_collapse)

    def _record_last_sample_topology_stats(
        self,
        topology_counts: Mapping[str, int],
        sample_size: int,
        *,
        demo_only: bool = False,
    ) -> None:
        self._last_sample_stats = {
            "replay_sample_size": float(sample_size),
            **(
                {"replay_source_frac_demo": 1.0}
                if demo_only
                else {}
            ),
            **{
                "replay_topology_frac_{0}".format(topology_name): float(count) / float(max(sample_size, 1))
                for topology_name, count in sorted(topology_counts.items())
            },
        }

    def _sample_fifo_with_topology_collapse_caps(
        self,
        *,
        batch_size: int,
        max_collapse_ratio: float | None,
        require_demo: bool = False,
        require_pool_power_demo: bool = False,
    ) -> TensorReplayBatch | None:
        assert self._fifo_storage is not None
        storage = self._fifo_storage
        candidate_groups = self._fifo_candidate_indices_by_topology(
            require_demo=require_demo,
            require_pool_power_demo=require_pool_power_demo,
        )
        if not candidate_groups:
            return None
        if max_collapse_ratio is None or len(candidate_groups) <= 1:
            return None

        topology_names = list(candidate_groups.keys())
        eligible_counts = [
            self._fifo_max_sample_count_for_topology(candidate_groups[topology_name], max_collapse_ratio)
            for topology_name in topology_names
        ]
        active_pairs = [
            (topology_name, candidate_groups[topology_name], int(eligible_count))
            for topology_name, eligible_count in zip(topology_names, eligible_counts)
            if int(eligible_count) > 0
        ]
        if not active_pairs:
            return None

        requested = min(int(batch_size), sum(eligible_count for _, _, eligible_count in active_pairs))
        if requested <= 0:
            return None

        active_topology_names = [item[0] for item in active_pairs]
        active_candidate_groups = {item[0]: item[1] for item in active_pairs}
        target_counts = _allocate_integer_counts(requested, [float(item[2]) for item in active_pairs])
        selected_by_topology: dict[str, np.ndarray] = {
            topology_name: np.empty(0, dtype=np.int64) for topology_name in active_topology_names
        }
        sampled_parts: list[np.ndarray] = []
        actual_topology_counts: dict[str, int] = {}

        for topology_name, target_count in zip(active_topology_names, target_counts):
            if target_count <= 0:
                continue
            sampled_indices = storage._sample_candidate_indices_up_to(
                candidate_indices=active_candidate_groups[topology_name],
                batch_size=target_count,
                max_collapse_ratio=max_collapse_ratio,
                strict_max_collapse_ratio=True,
            )
            if sampled_indices.numel() <= 0:
                continue
            sampled_np = sampled_indices.detach().cpu().numpy().astype(np.int64, copy=False)
            selected_by_topology[topology_name] = sampled_np
            sampled_parts.append(sampled_np)
            actual_topology_counts[topology_name] = len(sampled_np)

        remaining = requested - sum(actual_topology_counts.values())
        while remaining > 0:
            made_progress = False
            for topology_name in active_topology_names:
                candidate_indices = active_candidate_groups[topology_name]
                selected_indices = selected_by_topology[topology_name]
                if selected_indices.size > 0:
                    residual_candidates = candidate_indices[
                        ~np.isin(candidate_indices, selected_indices, assume_unique=False)
                    ]
                else:
                    residual_candidates = candidate_indices
                if residual_candidates.size <= 0:
                    continue
                sampled_indices = storage._sample_candidate_indices_up_to(
                    candidate_indices=residual_candidates,
                    batch_size=1,
                    max_collapse_ratio=max_collapse_ratio,
                    strict_max_collapse_ratio=True,
                )
                if sampled_indices.numel() <= 0:
                    continue
                sampled_np = sampled_indices.detach().cpu().numpy().astype(np.int64, copy=False)
                selected_by_topology[topology_name] = np.concatenate(
                    [selected_by_topology[topology_name], sampled_np],
                    axis=0,
                )
                sampled_parts.append(sampled_np)
                actual_topology_counts[topology_name] = actual_topology_counts.get(topology_name, 0) + len(sampled_np)
                remaining -= len(sampled_np)
                made_progress = True
                if remaining <= 0:
                    break
            if not made_progress:
                break

        if not sampled_parts:
            return None
        sampled_indices = np.concatenate(sampled_parts, axis=0)
        if sampled_indices.size <= 0:
            return None
        self._rng.shuffle(sampled_indices)
        batch = storage._batch_from_indices(
            torch.as_tensor(sampled_indices, dtype=torch.int64, device="cpu")
        )
        self._record_last_sample_topology_stats(
            actual_topology_counts,
            len(batch),
            demo_only=require_demo,
        )
        return batch

    def get_last_sample_stats(self) -> dict[str, float]:
        with self._lock:
            return dict(self._last_sample_stats)

    def export_demo_batch(self) -> TensorReplayBatch | None:
        with self._lock:
            exported_batches: list[TensorReplayBatch] = []
            for storage in self._canonical_export_storages():
                storage_batch = storage.export_all()
                if storage_batch is None or len(storage_batch) <= 0:
                    continue
                demo_mask = storage_batch.is_demo.detach().cpu().numpy().astype(np.bool_, copy=False)
                if self.demo_behavior_source == "pool_power_mix":
                    pool_mask = storage_batch.pool_power_demo_flag.detach().cpu().numpy().astype(np.bool_, copy=False)
                    demo_mask = np.logical_and(demo_mask, pool_mask)
                if not np.any(demo_mask):
                    continue
                indices = torch.as_tensor(np.flatnonzero(demo_mask), dtype=torch.int64, device="cpu")
                exported_batches.append(_slice_replay_batch(storage_batch, indices))
            if not exported_batches:
                return None
            if len(exported_batches) == 1:
                return exported_batches[0]
            return _concat_replay_batches(exported_batches)

    def add(self, transition: TensorTransition | Transition) -> None:
        tensor_transition = _coerce_transition(transition)
        with self._lock:
            if self.replay_strategy == "fifo":
                assert self._fifo_storage is not None
                self._fifo_storage.add(tensor_transition)
                return

            topology_name = self._resolve_topology_name(int(tensor_transition.topology_id.item()))
            recent_storage = self._recent_storages.get(topology_name)
            if recent_storage is not None:
                recent_storage.add(tensor_transition)
            long_term_storage = self._long_term_storages.get(topology_name)
            if long_term_storage is not None:
                long_term_storage.add(tensor_transition)
            if bool(tensor_transition.pool_power_demo_flag.item()):
                demo_storage = self._demo_storages.get(topology_name)
                if demo_storage is not None:
                    demo_storage.add(tensor_transition)

    def extend(self, batch: TensorReplayBatch) -> None:
        if len(batch) == 0:
            return
        cpu_batch = batch if _TensorReplayStorage._batch_is_cpu(batch) else batch.to("cpu")
        with self._lock:
            if self.replay_strategy == "fifo":
                assert self._fifo_storage is not None
                self._fifo_storage.extend(cpu_batch)
                return

            recent_groups = self._split_batch_by_topology(cpu_batch, require_pool_power_demo=False)
            for topology_name, topology_batch in recent_groups.items():
                recent_storage = self._recent_storages.get(topology_name)
                if recent_storage is not None:
                    recent_storage.extend(topology_batch)
                long_term_storage = self._long_term_storages.get(topology_name)
                if long_term_storage is not None:
                    long_term_storage.extend(topology_batch)

            if bool(cpu_batch.pool_power_demo_flag.any().item()):
                demo_groups = self._split_batch_by_topology(cpu_batch, require_pool_power_demo=True)
                for topology_name, topology_batch in demo_groups.items():
                    demo_storage = self._demo_storages.get(topology_name)
                    if demo_storage is not None:
                        demo_storage.extend(topology_batch)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        max_collapse_ratio: float | None = None,
    ) -> TensorReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        with self._lock:
            if self.replay_strategy == "fifo":
                assert self._fifo_storage is not None
                batch = self._sample_fifo_with_topology_collapse_caps(
                    batch_size=batch_size,
                    max_collapse_ratio=max_collapse_ratio,
                )
                if batch is None:
                    batch = self._fifo_storage.sample(
                        batch_size=batch_size,
                        device=None,
                        max_collapse_ratio=max_collapse_ratio,
                    )
                batch = _with_replay_source_id(batch, "fifo")
                topology_ids = batch.topology_id.detach().cpu().numpy().astype(np.int64, copy=False)
                topology_counts: dict[str, int] = {}
                for topology_id in np.unique(topology_ids):
                    topology_name = self._resolve_topology_name(int(topology_id))
                    topology_counts[topology_name] = int(np.sum(topology_ids == int(topology_id)))
                self._record_last_sample_topology_stats(topology_counts, len(batch))
            else:
                active_topologies = [
                    topology_name
                    for topology_name in self.topology_names
                    if self._source_storages_for_topology(topology_name)
                ]
                if not active_topologies:
                    raise ValueError("Cannot sample from an empty replay buffer.")

                topology_counts = _allocate_integer_counts(batch_size, [1.0 for _ in active_topologies])
                sampled_batches: list[TensorReplayBatch] = []
                sampled_total = 0
                source_counts: dict[str, int] = {}
                actual_topology_counts: dict[str, int] = {}

                for topology_name, requested in zip(active_topologies, topology_counts):
                    topology_batch, topology_source_counts = self._sample_from_topology(
                        topology_name=topology_name,
                        requested=requested,
                        max_collapse_ratio=max_collapse_ratio,
                    )
                    if topology_batch is None:
                        continue
                    sampled_batches.append(topology_batch)
                    sampled_total += len(topology_batch)
                    actual_topology_counts[topology_name] = actual_topology_counts.get(topology_name, 0) + len(topology_batch)
                    for source_name, count in topology_source_counts.items():
                        source_counts[source_name] = source_counts.get(source_name, 0) + int(count)

                remaining = batch_size - sampled_total
                while remaining > 0:
                    made_progress = False
                    for topology_name in active_topologies:
                        topology_batch, topology_source_counts = self._sample_from_topology(
                            topology_name=topology_name,
                            requested=1,
                            max_collapse_ratio=max_collapse_ratio,
                        )
                        if topology_batch is None:
                            continue
                        sampled_batches.append(topology_batch)
                        sampled_total += len(topology_batch)
                        actual_topology_counts[topology_name] = actual_topology_counts.get(topology_name, 0) + len(topology_batch)
                        for source_name, count in topology_source_counts.items():
                            source_counts[source_name] = source_counts.get(source_name, 0) + int(count)
                        remaining -= len(topology_batch)
                        made_progress = True
                        if remaining <= 0:
                            break
                    if not made_progress:
                        break

                if not sampled_batches:
                    raise ValueError("Cannot sample from an empty replay buffer.")
                batch = _shuffle_replay_batch(_concat_replay_batches(sampled_batches), self._rng)
                sample_size = max(len(batch), 1)
                self._last_sample_stats = {
                    "replay_sample_size": float(len(batch)),
                    **{
                        "replay_source_frac_{0}".format(source_name): float(count) / float(sample_size)
                        for source_name, count in sorted(source_counts.items())
                    },
                    **{
                        "replay_topology_frac_{0}".format(topology_name): float(count) / float(sample_size)
                        for topology_name, count in sorted(actual_topology_counts.items())
                    },
                }

        if device is not None:
            return batch.to(device)
        return batch

    def sample_demo(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        max_collapse_ratio: float | None = None,
    ) -> TensorReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        with self._lock:
            if self.replay_strategy == "fifo":
                assert self._fifo_storage is not None
                batch = self._sample_fifo_with_topology_collapse_caps(
                    batch_size=batch_size,
                    max_collapse_ratio=max_collapse_ratio,
                    require_demo=True,
                    require_pool_power_demo=(self.demo_behavior_source == "pool_power_mix"),
                )
                if batch is None:
                    batch = self._fifo_storage.sample_filtered_up_to(
                        batch_size=batch_size,
                        max_collapse_ratio=max_collapse_ratio,
                        strict_max_collapse_ratio=False,
                        require_demo=True,
                        require_pool_power_demo=(self.demo_behavior_source == "pool_power_mix"),
                    )
                if batch is None:
                    raise ValueError("Cannot sample demo batch from an empty replay buffer.")
                batch = _with_replay_source_id(batch, "demo")
                topology_ids = batch.topology_id.detach().cpu().numpy().astype(np.int64, copy=False)
                topology_counts: dict[str, int] = {}
                for topology_id in np.unique(topology_ids):
                    topology_name = self._resolve_topology_name(int(topology_id))
                    topology_counts[topology_name] = int(np.sum(topology_ids == int(topology_id)))
                self._record_last_sample_topology_stats(topology_counts, len(batch), demo_only=True)
            else:
                active_topologies = [
                    topology_name
                    for topology_name in self.topology_names
                    if (
                        (topology_name in self._demo_storages and len(self._demo_storages[topology_name]) > 0)
                        or (topology_name in self._recent_storages and self._recent_storages[topology_name].demo_size() > 0)
                        or (topology_name in self._long_term_storages and self._long_term_storages[topology_name].demo_size() > 0)
                    )
                ]
                if not active_topologies:
                    raise ValueError("Cannot sample demo batch from an empty replay buffer.")
                topology_counts = _allocate_integer_counts(batch_size, [1.0 for _ in active_topologies])
                sampled_batches: list[TensorReplayBatch] = []
                actual_topology_counts: dict[str, int] = {}

                for topology_name, requested in zip(active_topologies, topology_counts):
                    if requested <= 0:
                        continue
                    remaining_for_topology = int(requested)
                    topology_samples: list[TensorReplayBatch] = []
                    preferred_storage = self._demo_storages.get(topology_name)
                    if preferred_storage is not None and len(preferred_storage) > 0:
                        sample = preferred_storage.sample_up_to(
                            remaining_for_topology,
                            max_collapse_ratio=max_collapse_ratio,
                            strict_max_collapse_ratio=True,
                        )
                        if sample is not None:
                            topology_samples.append(sample)
                            remaining_for_topology -= len(sample)
                    for fallback_storage in (
                        self._recent_storages.get(topology_name),
                        self._long_term_storages.get(topology_name),
                    ):
                        if remaining_for_topology <= 0 or fallback_storage is None or len(fallback_storage) <= 0:
                            continue
                        sample = fallback_storage.sample_filtered_up_to(
                            remaining_for_topology,
                            max_collapse_ratio=max_collapse_ratio,
                            strict_max_collapse_ratio=True,
                            require_demo=True,
                            require_pool_power_demo=(self.demo_behavior_source == "pool_power_mix"),
                        )
                        if sample is None:
                            continue
                        topology_samples.append(sample)
                        remaining_for_topology -= len(sample)
                    if not topology_samples:
                        continue
                    combined_topology_batch = _concat_replay_batches(topology_samples)
                    sampled_batches.append(combined_topology_batch)
                    actual_topology_counts[topology_name] = (
                        actual_topology_counts.get(topology_name, 0) + len(combined_topology_batch)
                    )

                remaining = batch_size - sum(len(batch_item) for batch_item in sampled_batches)
                while remaining > 0:
                    made_progress = False
                    for topology_name in active_topologies:
                        if remaining <= 0:
                            break
                        sample = None
                        preferred_storage = self._demo_storages.get(topology_name)
                        if preferred_storage is not None and len(preferred_storage) > 0:
                            sample = preferred_storage.sample_up_to(
                                1,
                                max_collapse_ratio=max_collapse_ratio,
                                strict_max_collapse_ratio=True,
                            )
                        if sample is None:
                            for fallback_storage in (
                                self._recent_storages.get(topology_name),
                                self._long_term_storages.get(topology_name),
                            ):
                                if fallback_storage is None or len(fallback_storage) <= 0:
                                    continue
                                sample = fallback_storage.sample_filtered_up_to(
                                    1,
                                    max_collapse_ratio=max_collapse_ratio,
                                    strict_max_collapse_ratio=True,
                                    require_demo=True,
                                    require_pool_power_demo=(self.demo_behavior_source == "pool_power_mix"),
                                )
                                if sample is not None:
                                    break
                        if sample is None:
                            continue
                        sampled_batches.append(sample)
                        actual_topology_counts[topology_name] = actual_topology_counts.get(topology_name, 0) + len(sample)
                        remaining -= len(sample)
                        made_progress = True
                    if not made_progress:
                        break

                if not sampled_batches:
                    raise ValueError("Cannot sample demo batch from an empty replay buffer.")
                batch = _shuffle_replay_batch(_concat_replay_batches(sampled_batches), self._rng)
                sample_size = max(len(batch), 1)
                self._last_sample_stats = {
                    "replay_sample_size": float(len(batch)),
                    "replay_source_frac_demo": 1.0,
                    **{
                        "replay_topology_frac_{0}".format(topology_name): float(count) / float(sample_size)
                        for topology_name, count in sorted(actual_topology_counts.items())
                    },
                }

        if device is not None:
            return batch.to(device)
        return batch

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            if self.replay_strategy == "fifo":
                assert self._fifo_storage is not None
                payload = self._fifo_storage.state_dict()
                payload["replay_mode"] = "fifo"
                return payload

            return {
                "replay_mode": self.replay_strategy,
                "capacity": int(self.capacity),
                "rng_state": self._rng.bit_generator.state,
                "topology_names": list(self.topology_names),
                "recent_fraction": float(self.recent_fraction),
                "long_term_fraction": float(self.long_term_fraction),
                "demo_fraction": float(self.demo_fraction),
                "demo_behavior_source": str(self.demo_behavior_source),
                "recent_storages": {
                    key: storage.state_dict() for key, storage in self._recent_storages.items()
                },
                "long_term_storages": {
                    key: storage.state_dict() for key, storage in self._long_term_storages.items()
                },
                "demo_storages": {
                    key: storage.state_dict() for key, storage in self._demo_storages.items()
                },
            }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        replay_mode = str(state_dict.get("replay_mode", "fifo"))
        with self._lock:
            if replay_mode == "fifo" and "recent_storages" not in state_dict:
                self.replay_strategy = "fifo"
                self.capacity = int(state_dict["capacity"])
                self._fifo_storage = _TensorReplayStorage(self.capacity, seed=self._seed, replacement_policy="ring")
                self._fifo_storage.load_state_dict(state_dict)
                self._recent_storages = {}
                self._long_term_storages = {}
                self._demo_storages = {}
                return

            self.replay_strategy = replay_mode
            self.capacity = int(state_dict["capacity"])
            self._rng = np.random.default_rng()
            if "rng_state" in state_dict:
                self._rng.bit_generator.state = state_dict["rng_state"]
            self.topology_names = tuple(str(item) for item in state_dict.get("topology_names", ["fixed"]))
            self.recent_fraction = float(state_dict.get("recent_fraction", 0.50))
            self.long_term_fraction = float(state_dict.get("long_term_fraction", 0.35))
            self.demo_fraction = float(state_dict.get("demo_fraction", 0.15))
            self.demo_behavior_source = str(state_dict.get("demo_behavior_source", "pool_power_mix"))
            self._fifo_storage = None
            self._recent_storages = {}
            self._long_term_storages = {}
            self._demo_storages = {}

            for key, storage_state in dict(state_dict.get("recent_storages", {})).items():
                storage = _TensorReplayStorage(int(storage_state["capacity"]), seed=self._seed, replacement_policy="ring")
                storage.load_state_dict(dict(storage_state))
                self._recent_storages[str(key)] = storage
            for key, storage_state in dict(state_dict.get("long_term_storages", {})).items():
                storage = _TensorReplayStorage(
                    int(storage_state["capacity"]),
                    seed=self._seed,
                    replacement_policy="reservoir",
                )
                storage.load_state_dict(dict(storage_state))
                self._long_term_storages[str(key)] = storage
            for key, storage_state in dict(state_dict.get("demo_storages", {})).items():
                storage = _TensorReplayStorage(int(storage_state["capacity"]), seed=self._seed, replacement_policy="ring")
                storage.load_state_dict(dict(storage_state))
                self._demo_storages[str(key)] = storage

    def __len__(self) -> int:
        with self._lock:
            if self.replay_strategy == "fifo":
                assert self._fifo_storage is not None
                return len(self._fifo_storage)
            recent_size = int(sum(len(storage) for storage in self._recent_storages.values()))
            if recent_size > 0:
                return recent_size
            long_term_size = int(sum(len(storage) for storage in self._long_term_storages.values()))
            if long_term_size > 0:
                return long_term_size
            return int(sum(len(storage) for storage in self._demo_storages.values()))

    def demo_size(self) -> int:
        with self._lock:
            if self.replay_strategy == "fifo":
                assert self._fifo_storage is not None
                return int(self._fifo_storage.demo_size(require_pool_power_demo=self.demo_behavior_source == "pool_power_mix"))
            return int(sum(len(storage) for storage in self._demo_storages.values()))
