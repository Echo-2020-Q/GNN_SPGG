from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .data import Transition


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self._buffer: list[Transition | None] = [None] * self.capacity
        self._next_index = 0
        self._size = 0
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        with self._lock:
            self._buffer[self._next_index] = transition.clone()
            self._next_index = (self._next_index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        with self._lock:
            if self._size == 0:
                raise ValueError("Cannot sample from an empty replay buffer.")
            indices = self._rng.integers(0, self._size, size=batch_size)
            return [self._buffer[int(index)].clone() for index in indices if self._buffer[int(index)] is not None]

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "capacity": int(self.capacity),
                "buffer": [transition.clone() if transition is not None else None for transition in self._buffer],
                "next_index": int(self._next_index),
                "size": int(self._size),
                "rng_state": self._rng.bit_generator.state,
            }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        capacity = int(state_dict["capacity"])
        buffer_state = list(state_dict["buffer"])
        if len(buffer_state) != capacity:
            raise ValueError("Replay buffer checkpoint is inconsistent with its declared capacity.")

        with self._lock:
            self.capacity = capacity
            self._buffer = [transition.clone() if transition is not None else None for transition in buffer_state]
            self._next_index = int(state_dict["next_index"])
            self._size = int(state_dict["size"])
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = state_dict["rng_state"]

    def __len__(self) -> int:
        with self._lock:
            return self._size
