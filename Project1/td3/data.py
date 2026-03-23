from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from Project1.env import Observation


def clone_observation(observation: Observation) -> Observation:
    return {key: np.asarray(value).copy() for key, value in observation.items()}


@dataclass
class TensorActionRecord:
    logits: Tensor
    allocation: Tensor
    transfers: Tensor
    incoming: Tensor
    ego_mask: Tensor
    pool_values: Tensor

    def to_numpy(self) -> "ActionRecord":
        return ActionRecord(
            logits=self.logits.detach().cpu().numpy().copy(),
            allocation=self.allocation.detach().cpu().numpy().copy(),
            transfers=self.transfers.detach().cpu().numpy().copy(),
            incoming=self.incoming.detach().cpu().numpy().copy(),
            ego_mask=self.ego_mask.detach().cpu().numpy().copy(),
            pool_values=self.pool_values.detach().cpu().numpy().copy(),
        )


@dataclass
class ActionRecord:
    logits: np.ndarray
    allocation: np.ndarray
    transfers: np.ndarray
    incoming: np.ndarray
    ego_mask: np.ndarray
    pool_values: np.ndarray

    def clone(self) -> "ActionRecord":
        return ActionRecord(
            logits=np.asarray(self.logits).copy(),
            allocation=np.asarray(self.allocation).copy(),
            transfers=np.asarray(self.transfers).copy(),
            incoming=np.asarray(self.incoming).copy(),
            ego_mask=np.asarray(self.ego_mask).copy(),
            pool_values=np.asarray(self.pool_values).copy(),
        )

    def to_tensors(self, device: torch.device | str) -> TensorActionRecord:
        return TensorActionRecord(
            logits=torch.as_tensor(self.logits, dtype=torch.float32, device=device),
            allocation=torch.as_tensor(self.allocation, dtype=torch.float32, device=device),
            transfers=torch.as_tensor(self.transfers, dtype=torch.float32, device=device),
            incoming=torch.as_tensor(self.incoming, dtype=torch.float32, device=device),
            ego_mask=torch.as_tensor(self.ego_mask, dtype=torch.bool, device=device),
            pool_values=torch.as_tensor(self.pool_values, dtype=torch.float32, device=device),
        )


@dataclass
class Transition:
    obs: Observation
    action: ActionRecord
    reward: float
    next_obs: Observation
    done: bool
    info: dict[str, Any]
    metadata: Mapping[str, Any]

    def clone(self) -> "Transition":
        return Transition(
            obs=clone_observation(self.obs),
            action=self.action.clone(),
            reward=float(self.reward),
            next_obs=clone_observation(self.next_obs),
            done=bool(self.done),
            info=dict(self.info),
            metadata=dict(self.metadata),
        )
