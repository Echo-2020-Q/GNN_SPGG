from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .data import TensorActionRecord, TensorReplayBatch, TensorTransition, Transition


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
            "logits": torch.empty((self.capacity, *tuple(transition.action.logits.shape)), dtype=transition.action.logits.dtype, device="cpu"),
            "allocation": torch.empty((self.capacity, *tuple(transition.action.allocation.shape)), dtype=transition.action.allocation.dtype, device="cpu"),
            "transfers": torch.empty((self.capacity, *tuple(transition.action.transfers.shape)), dtype=transition.action.transfers.dtype, device="cpu"),
            "incoming": torch.empty((self.capacity, *tuple(transition.action.incoming.shape)), dtype=transition.action.incoming.dtype, device="cpu"),
            "ego_mask": torch.empty((self.capacity, *tuple(transition.action.ego_mask.shape)), dtype=transition.action.ego_mask.dtype, device="cpu"),
            "pool_values": torch.empty((self.capacity, *tuple(transition.action.pool_values.shape)), dtype=transition.action.pool_values.dtype, device="cpu"),
        }
        self._reward_buffer = torch.empty(self.capacity, dtype=transition.reward.dtype, device="cpu")
        self._done_buffer = torch.empty(self.capacity, dtype=transition.done.dtype, device="cpu")

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
            "logits": torch.empty((self.capacity, *tuple(batch.action.logits.shape[1:])), dtype=batch.action.logits.dtype, device="cpu"),
            "allocation": torch.empty((self.capacity, *tuple(batch.action.allocation.shape[1:])), dtype=batch.action.allocation.dtype, device="cpu"),
            "transfers": torch.empty((self.capacity, *tuple(batch.action.transfers.shape[1:])), dtype=batch.action.transfers.dtype, device="cpu"),
            "incoming": torch.empty((self.capacity, *tuple(batch.action.incoming.shape[1:])), dtype=batch.action.incoming.dtype, device="cpu"),
            "ego_mask": torch.empty((self.capacity, *tuple(batch.action.ego_mask.shape[1:])), dtype=batch.action.ego_mask.dtype, device="cpu"),
            "pool_values": torch.empty((self.capacity, *tuple(batch.action.pool_values.shape[1:])), dtype=batch.action.pool_values.dtype, device="cpu"),
        }
        self._reward_buffer = torch.empty(self.capacity, dtype=batch.reward.dtype, device="cpu")
        self._done_buffer = torch.empty(self.capacity, dtype=batch.done.dtype, device="cpu")

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
            "logits": transition.action.logits,
            "allocation": transition.action.allocation,
            "transfers": transition.action.transfers,
            "incoming": transition.action.incoming,
            "ego_mask": transition.action.ego_mask,
            "pool_values": transition.action.pool_values,
        }
        for key, value in action_fields.items():
            buffer = self._action_buffers[key]
            if tuple(value.shape) != tuple(buffer.shape[1:]) or value.dtype != buffer.dtype:
                raise ValueError("Action field '{0}' is incompatible with replay buffer schema.".format(key))

    def _write_transition_at_index(self, index: int, transition: TensorTransition) -> None:
        for key, value in transition.obs.items():
            self._obs_buffers[key][index].copy_(value)
        for key, value in transition.next_obs.items():
            self._next_obs_buffers[key][index].copy_(value)

        self._action_buffers["logits"][index].copy_(transition.action.logits)
        self._action_buffers["allocation"][index].copy_(transition.action.allocation)
        self._action_buffers["transfers"][index].copy_(transition.action.transfers)
        self._action_buffers["incoming"][index].copy_(transition.action.incoming)
        self._action_buffers["ego_mask"][index].copy_(transition.action.ego_mask)
        self._action_buffers["pool_values"][index].copy_(transition.action.pool_values)
        assert self._reward_buffer is not None
        assert self._done_buffer is not None
        self._reward_buffer[index] = transition.reward
        self._done_buffer[index] = transition.done

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
            "logits": batch.action.logits,
            "allocation": batch.action.allocation,
            "transfers": batch.action.transfers,
            "incoming": batch.action.incoming,
            "ego_mask": batch.action.ego_mask,
            "pool_values": batch.action.pool_values,
        }
        for key, value in action_fields.items():
            buffer = self._action_buffers[key]
            if value.ndim < 1 or tuple(value.shape[1:]) != tuple(buffer.shape[1:]) or value.dtype != buffer.dtype:
                raise ValueError("Action batch field '{0}' is incompatible with replay buffer schema.".format(key))

    def _write_batch_at_indices(self, indices: Tensor, batch: TensorReplayBatch) -> None:
        for key, value in batch.obs.items():
            self._obs_buffers[key].index_copy_(0, indices, value)
        for key, value in batch.next_obs.items():
            self._next_obs_buffers[key].index_copy_(0, indices, value)

        self._action_buffers["logits"].index_copy_(0, indices, batch.action.logits)
        self._action_buffers["allocation"].index_copy_(0, indices, batch.action.allocation)
        self._action_buffers["transfers"].index_copy_(0, indices, batch.action.transfers)
        self._action_buffers["incoming"].index_copy_(0, indices, batch.action.incoming)
        self._action_buffers["ego_mask"].index_copy_(0, indices, batch.action.ego_mask)
        self._action_buffers["pool_values"].index_copy_(0, indices, batch.action.pool_values)
        assert self._reward_buffer is not None
        assert self._done_buffer is not None
        self._reward_buffer.index_copy_(0, indices, batch.reward)
        self._done_buffer.index_copy_(0, indices, batch.done)

    @staticmethod
    def _batch_is_cpu(batch: TensorReplayBatch) -> bool:
        tensors = list(batch.obs.values()) + [
            batch.action.logits,
            batch.action.allocation,
            batch.action.transfers,
            batch.action.incoming,
            batch.action.ego_mask,
            batch.action.pool_values,
            batch.reward,
            batch.done,
        ] + list(batch.next_obs.values())
        return all(tensor.device.type == "cpu" for tensor in tensors)

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

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> TensorReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        with self._lock:
            if self._size == 0 or not self._is_initialized():
                raise ValueError("Cannot sample from an empty replay buffer.")

            indices = torch.as_tensor(
                self._rng.integers(0, self._size, size=batch_size),
                dtype=torch.int64,
                device="cpu",
            )
            obs = {key: buffer.index_select(0, indices) for key, buffer in self._obs_buffers.items()}
            next_obs = {key: buffer.index_select(0, indices) for key, buffer in self._next_obs_buffers.items()}
            action = TensorActionRecord(
                logits=self._action_buffers["logits"].index_select(0, indices),
                allocation=self._action_buffers["allocation"].index_select(0, indices),
                transfers=self._action_buffers["transfers"].index_select(0, indices),
                incoming=self._action_buffers["incoming"].index_select(0, indices),
                ego_mask=self._action_buffers["ego_mask"].index_select(0, indices),
                pool_values=self._action_buffers["pool_values"].index_select(0, indices),
            )
            assert self._reward_buffer is not None
            assert self._done_buffer is not None
            batch = TensorReplayBatch(
                obs=obs,
                action=action,
                reward=self._reward_buffer.index_select(0, indices),
                next_obs=next_obs,
                done=self._done_buffer.index_select(0, indices),
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
            self._reward_buffer = None if reward_buffer is None else torch.as_tensor(reward_buffer, device="cpu").detach().clone()
            self._done_buffer = None if done_buffer is None else torch.as_tensor(done_buffer, device="cpu").detach().clone()

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
