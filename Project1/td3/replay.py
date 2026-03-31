from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .data import TensorReplayActionRecord, TensorReplayBatch, TensorTransition, Transition


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self._next_index = 0
        self._size = 0
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)

        self._obs_buffers: dict[str, Tensor] = {}
        self._next_obs_buffers: dict[str, Tensor] = {}
        self._action_buffers: dict[str, Tensor] = {}
        self._reward_buffer: Tensor | None = None
        self._done_buffer: Tensor | None = None
        self._is_demo_buffer: Tensor | None = None
        self._collapse_flag_buffer: Tensor | None = None

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
            "allocation": torch.empty((self.capacity, *tuple(transition.action.allocation.shape)), dtype=transition.action.allocation.dtype, device="cpu"),
        }
        self._reward_buffer = torch.empty(self.capacity, dtype=transition.reward.dtype, device="cpu")
        self._done_buffer = torch.empty(self.capacity, dtype=transition.done.dtype, device="cpu")
        self._is_demo_buffer = torch.empty(self.capacity, dtype=transition.is_demo.dtype, device="cpu")
        self._collapse_flag_buffer = torch.empty(self.capacity, dtype=transition.collapse_flag.dtype, device="cpu")

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
            "allocation": torch.empty((self.capacity, *tuple(batch.action.allocation.shape[1:])), dtype=batch.action.allocation.dtype, device="cpu"),
        }
        self._reward_buffer = torch.empty(self.capacity, dtype=batch.reward.dtype, device="cpu")
        self._done_buffer = torch.empty(self.capacity, dtype=batch.done.dtype, device="cpu")
        self._is_demo_buffer = torch.empty(self.capacity, dtype=batch.is_demo.dtype, device="cpu")
        self._collapse_flag_buffer = torch.empty(self.capacity, dtype=batch.collapse_flag.dtype, device="cpu")

    def _coerce_transition(self, transition: TensorTransition | Transition) -> TensorTransition:
        if isinstance(transition, TensorTransition):
            return transition
        if isinstance(transition, Transition):
            return TensorTransition.from_transition(transition)
        raise TypeError("ReplayBuffer.add expects TensorTransition or Transition.")

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

        action_fields = {
            "allocation": transition.action.allocation,
        }
        for key, value in action_fields.items():
            buffer = self._action_buffers[key]
            if tuple(value.shape) != tuple(buffer.shape[1:]) or value.dtype != buffer.dtype:
                raise ValueError("Action field '{0}' is incompatible with replay buffer schema.".format(key))
        if self._is_demo_buffer is None or self._collapse_flag_buffer is None:
            raise ValueError("Replay metadata buffers are not initialized.")
        if transition.is_demo.shape != torch.Size([]) or transition.is_demo.dtype != self._is_demo_buffer.dtype:
            raise ValueError("Transition field 'is_demo' is incompatible with replay buffer schema.")
        if transition.collapse_flag.shape != torch.Size([]) or transition.collapse_flag.dtype != self._collapse_flag_buffer.dtype:
            raise ValueError("Transition field 'collapse_flag' is incompatible with replay buffer schema.")

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
        self._reward_buffer[index] = transition.reward
        self._done_buffer[index] = transition.done
        self._is_demo_buffer[index] = transition.is_demo
        self._collapse_flag_buffer[index] = transition.collapse_flag

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

        action_fields = {
            "allocation": batch.action.allocation,
        }
        for key, value in action_fields.items():
            buffer = self._action_buffers[key]
            if value.ndim < 1 or tuple(value.shape[1:]) != tuple(buffer.shape[1:]) or value.dtype != buffer.dtype:
                raise ValueError("Action batch field '{0}' is incompatible with replay buffer schema.".format(key))
        if self._is_demo_buffer is None or self._collapse_flag_buffer is None:
            raise ValueError("Replay metadata buffers are not initialized.")
        if batch.is_demo.ndim != 1 or batch.is_demo.dtype != self._is_demo_buffer.dtype:
            raise ValueError("Replay batch field 'is_demo' is incompatible with replay buffer schema.")
        if batch.collapse_flag.ndim != 1 or batch.collapse_flag.dtype != self._collapse_flag_buffer.dtype:
            raise ValueError("Replay batch field 'collapse_flag' is incompatible with replay buffer schema.")

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
        self._reward_buffer.index_copy_(0, indices, batch.reward)
        self._done_buffer.index_copy_(0, indices, batch.done)
        self._is_demo_buffer.index_copy_(0, indices, batch.is_demo)
        self._collapse_flag_buffer.index_copy_(0, indices, batch.collapse_flag)

    @staticmethod
    def _batch_is_cpu(batch: TensorReplayBatch) -> bool:
        tensors = list(batch.obs.values()) + [
            batch.action.allocation,
            batch.reward,
            batch.done,
            batch.is_demo,
            batch.collapse_flag,
        ] + list(batch.next_obs.values())
        return all(tensor.device.type == "cpu" for tensor in tensors)

    def _sample_indices(self, batch_size: int, max_collapse_ratio: float | None) -> Tensor:
        if (
            max_collapse_ratio is None
            or self._collapse_flag_buffer is None
            or self._size <= 0
        ):
            return torch.as_tensor(
                self._rng.integers(0, self._size, size=batch_size),
                dtype=torch.int64,
                device="cpu",
            )

        ratio = float(max_collapse_ratio)
        ratio = min(max(ratio, 0.0), 1.0)
        if ratio >= 1.0:
            return torch.as_tensor(
                self._rng.integers(0, self._size, size=batch_size),
                dtype=torch.int64,
                device="cpu",
            )

        collapse_flags = self._collapse_flag_buffer[: self._size].detach().cpu().numpy().astype(np.bool_, copy=False)
        collapse_indices = np.flatnonzero(collapse_flags)
        non_collapse_indices = np.flatnonzero(~collapse_flags)
        if collapse_indices.size == 0 or non_collapse_indices.size == 0:
            return torch.as_tensor(
                self._rng.integers(0, self._size, size=batch_size),
                dtype=torch.int64,
                device="cpu",
            )

        max_collapse = int(np.floor(float(batch_size) * ratio))
        collapse_take = min(int(collapse_indices.size), max_collapse)
        non_collapse_take = min(int(non_collapse_indices.size), batch_size - collapse_take)

        remaining = batch_size - collapse_take - non_collapse_take
        if remaining > 0:
            extra_non_collapse = min(int(non_collapse_indices.size) - non_collapse_take, remaining)
            non_collapse_take += extra_non_collapse
            remaining -= extra_non_collapse
        if remaining > 0:
            extra_collapse = min(int(collapse_indices.size) - collapse_take, remaining)
            collapse_take += extra_collapse
            remaining -= extra_collapse
        if remaining > 0:
            raise RuntimeError("Failed to assemble replay sample indices.")

        sampled_parts: list[np.ndarray] = []
        if non_collapse_take > 0:
            sampled_parts.append(self._rng.choice(non_collapse_indices, size=non_collapse_take, replace=False))
        if collapse_take > 0:
            sampled_parts.append(self._rng.choice(collapse_indices, size=collapse_take, replace=False))
        sampled_indices = np.concatenate(sampled_parts, axis=0)
        self._rng.shuffle(sampled_indices)
        return torch.as_tensor(sampled_indices, dtype=torch.int64, device="cpu")

    def add(self, transition: TensorTransition | Transition) -> None:
        tensor_transition = self._coerce_transition(transition)
        with self._lock:
            if not self._is_initialized():
                self._allocate_from_transition(tensor_transition)
            else:
                self._validate_transition_structure(tensor_transition)

            self._write_transition_at_index(self._next_index, tensor_transition)
            self._next_index = (self._next_index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def extend(self, batch: TensorReplayBatch) -> None:
        if len(batch) == 0:
            return
        cpu_batch = batch if self._batch_is_cpu(batch) else batch.to("cpu")
        with self._lock:
            if not self._is_initialized():
                self._allocate_from_batch(cpu_batch)
            else:
                self._validate_batch_structure(cpu_batch)

            batch_size = len(cpu_batch)
            indices = (torch.arange(batch_size, dtype=torch.int64, device="cpu") + int(self._next_index)) % int(self.capacity)
            self._write_batch_at_indices(indices, cpu_batch)
            self._next_index = int((self._next_index + batch_size) % self.capacity)
            self._size = min(self._size + batch_size, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        max_collapse_ratio: float | None = None,
    ) -> TensorReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        with self._lock:
            if self._size == 0 or not self._is_initialized():
                raise ValueError("Cannot sample from an empty replay buffer.")

            indices = self._sample_indices(batch_size=batch_size, max_collapse_ratio=max_collapse_ratio)
            obs = {key: buffer.index_select(0, indices) for key, buffer in self._obs_buffers.items()}
            next_obs = {key: buffer.index_select(0, indices) for key, buffer in self._next_obs_buffers.items()}
            action = TensorReplayActionRecord(
                allocation=self._action_buffers["allocation"].index_select(0, indices),
            )
            assert self._reward_buffer is not None
            assert self._done_buffer is not None
            assert self._is_demo_buffer is not None
            assert self._collapse_flag_buffer is not None
            batch = TensorReplayBatch(
                obs=obs,
                action=action,
                reward=self._reward_buffer.index_select(0, indices),
                next_obs=next_obs,
                done=self._done_buffer.index_select(0, indices),
                is_demo=self._is_demo_buffer.index_select(0, indices),
                collapse_flag=self._collapse_flag_buffer.index_select(0, indices),
            )

        if device is not None:
            return batch.to(device)
        return batch

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "capacity": int(self.capacity),
                "next_index": int(self._next_index),
                "size": int(self._size),
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
            }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "buffer" in state_dict:
            self._load_legacy_state_dict(state_dict)
            return

        capacity = int(state_dict["capacity"])
        with self._lock:
            self.capacity = capacity
            self._next_index = int(state_dict["next_index"])
            self._size = int(state_dict["size"])
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
            self._reward_buffer = None if reward_buffer is None else torch.as_tensor(reward_buffer, device="cpu").detach().clone()
            self._done_buffer = None if done_buffer is None else torch.as_tensor(done_buffer, device="cpu").detach().clone()
            if is_demo_buffer is None:
                self._is_demo_buffer = torch.zeros(self.capacity, dtype=torch.bool, device="cpu")
            else:
                self._is_demo_buffer = torch.as_tensor(is_demo_buffer, device="cpu").detach().clone()
            if collapse_flag_buffer is None:
                self._collapse_flag_buffer = torch.zeros(self.capacity, dtype=torch.bool, device="cpu")
            else:
                self._collapse_flag_buffer = torch.as_tensor(collapse_flag_buffer, device="cpu").detach().clone()

    def _load_legacy_state_dict(self, state_dict: dict[str, Any]) -> None:
        capacity = int(state_dict["capacity"])
        buffer_state = list(state_dict["buffer"])
        if len(buffer_state) != capacity:
            raise ValueError("Replay buffer checkpoint is inconsistent with its declared capacity.")

        with self._lock:
            self.capacity = capacity
            self._next_index = int(state_dict["next_index"])
            self._size = int(state_dict["size"])
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = state_dict["rng_state"]
            self._obs_buffers = {}
            self._next_obs_buffers = {}
            self._action_buffers = {}
            self._reward_buffer = None
            self._done_buffer = None
            self._is_demo_buffer = None
            self._collapse_flag_buffer = None

            first_transition = next((item for item in buffer_state if item is not None), None)
            if first_transition is None:
                return

            reference_transition = self._coerce_transition(first_transition)
            self._allocate_from_transition(reference_transition)
            for index, transition in enumerate(buffer_state):
                if transition is None:
                    continue
                tensor_transition = self._coerce_transition(transition)
                self._validate_transition_structure(tensor_transition)
                self._write_transition_at_index(index, tensor_transition)

    def __len__(self) -> int:
        with self._lock:
            return self._size
