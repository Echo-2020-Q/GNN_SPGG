from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from Project1.env import Observation


TensorObservation = dict[str, Tensor]

# First-stage tensor replay stores only the observation fields consumed by the
# actor/critic training path. This keeps semantics unchanged while removing
# unnecessary CPU memory traffic from unused observation arrays.
REPLAY_OBSERVATION_DTYPES: dict[str, torch.dtype] = {
    "x_actual": torch.float32,
    "resource_norm": torch.float32,
    "pool_raw_norm": torch.float32,
    "degree_norm": torch.float32,
    "strategy_norm": torch.float32,
    "gini": torch.float32,
    "pool_grown": torch.float32,
    "local_mask": torch.bool,
}


def clone_observation(observation: Observation) -> Observation:
    return {key: np.asarray(value).copy() for key, value in observation.items()}


def clone_tensor_observation(observation: TensorObservation) -> TensorObservation:
    return {key: value.detach().cpu().clone() for key, value in observation.items()}


def observation_to_replay_tensors(observation: Mapping[str, Any]) -> TensorObservation:
    tensor_observation: TensorObservation = {}
    for key, dtype in REPLAY_OBSERVATION_DTYPES.items():
        if key not in observation:
            raise KeyError("Observation is missing replay field '{0}'.".format(key))
        tensor_observation[key] = torch.as_tensor(
            observation[key],
            dtype=dtype,
            device="cpu",
        ).detach().clone()
    return tensor_observation


@dataclass
class TensorActionRecord:
    logits: Tensor
    allocation: Tensor
    transfers: Tensor
    incoming: Tensor
    ego_mask: Tensor
    pool_values: Tensor

    def clone(self) -> "TensorActionRecord":
        return TensorActionRecord(
            logits=self.logits.detach().cpu().clone(),
            allocation=self.allocation.detach().cpu().clone(),
            transfers=self.transfers.detach().cpu().clone(),
            incoming=self.incoming.detach().cpu().clone(),
            ego_mask=self.ego_mask.detach().cpu().clone(),
            pool_values=self.pool_values.detach().cpu().clone(),
        )

    def to(self, device: torch.device | str) -> "TensorActionRecord":
        return TensorActionRecord(
            logits=self.logits.to(device=device),
            allocation=self.allocation.to(device=device),
            transfers=self.transfers.to(device=device),
            incoming=self.incoming.to(device=device),
            ego_mask=self.ego_mask.to(device=device),
            pool_values=self.pool_values.to(device=device),
        )

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


@dataclass(frozen=True)
class TensorTransition:
    obs: TensorObservation
    action: TensorActionRecord
    reward: Tensor
    next_obs: TensorObservation
    done: Tensor

    @classmethod
    def from_step(
        cls,
        obs: Mapping[str, Any],
        action: TensorActionRecord,
        reward: float,
        next_obs: Mapping[str, Any],
        done: bool,
    ) -> "TensorTransition":
        return cls(
            obs=observation_to_replay_tensors(obs),
            action=action.clone(),
            reward=torch.tensor(float(reward), dtype=torch.float32, device="cpu"),
            next_obs=observation_to_replay_tensors(next_obs),
            done=torch.tensor(float(done), dtype=torch.float32, device="cpu"),
        )

    @classmethod
    def from_transition(cls, transition: Transition) -> "TensorTransition":
        return cls.from_step(
            obs=transition.obs,
            action=transition.action.to_tensors(device="cpu"),
            reward=float(transition.reward),
            next_obs=transition.next_obs,
            done=bool(transition.done),
        )

    def clone(self) -> "TensorTransition":
        return TensorTransition(
            obs=clone_tensor_observation(self.obs),
            action=self.action.clone(),
            reward=self.reward.detach().cpu().clone(),
            next_obs=clone_tensor_observation(self.next_obs),
            done=self.done.detach().cpu().clone(),
        )


@dataclass(frozen=True)
class TensorReplayBatch:
    obs: TensorObservation
    action: TensorActionRecord
    reward: Tensor
    next_obs: TensorObservation
    done: Tensor

    def to(self, device: torch.device | str) -> "TensorReplayBatch":
        return TensorReplayBatch(
            obs={key: value.to(device=device) for key, value in self.obs.items()},
            action=self.action.to(device=device),
            reward=self.reward.to(device=device),
            next_obs={key: value.to(device=device) for key, value in self.next_obs.items()},
            done=self.done.to(device=device),
        )

    def __len__(self) -> int:
        return int(self.reward.shape[0])

    def clone(self) -> "TensorReplayBatch":
        return TensorReplayBatch(
            obs=clone_tensor_observation(self.obs),
            action=self.action.clone(),
            reward=self.reward.detach().cpu().clone(),
            next_obs=clone_tensor_observation(self.next_obs),
            done=self.done.detach().cpu().clone(),
        )


def stack_tensor_transitions(transitions: Sequence[TensorTransition]) -> TensorReplayBatch:
    if not transitions:
        raise ValueError("transitions must contain at least one item.")

    first_transition = transitions[0]
    obs = {
        key: torch.stack([transition.obs[key] for transition in transitions], dim=0)
        for key in first_transition.obs
    }
    next_obs = {
        key: torch.stack([transition.next_obs[key] for transition in transitions], dim=0)
        for key in first_transition.next_obs
    }
    action = TensorActionRecord(
        logits=torch.stack([transition.action.logits for transition in transitions], dim=0),
        allocation=torch.stack([transition.action.allocation for transition in transitions], dim=0),
        transfers=torch.stack([transition.action.transfers for transition in transitions], dim=0),
        incoming=torch.stack([transition.action.incoming for transition in transitions], dim=0),
        ego_mask=torch.stack([transition.action.ego_mask for transition in transitions], dim=0),
        pool_values=torch.stack([transition.action.pool_values for transition in transitions], dim=0),
    )
    reward = torch.stack([transition.reward for transition in transitions], dim=0)
    done = torch.stack([transition.done for transition in transitions], dim=0)
    return TensorReplayBatch(
        obs=obs,
        action=action,
        reward=reward,
        next_obs=next_obs,
        done=done,
    )
