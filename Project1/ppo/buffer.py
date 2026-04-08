from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


TensorObservationBatch = dict[str, Tensor]


def clone_observation_batch(batch: Mapping[str, Tensor]) -> TensorObservationBatch:
    return {key: value.detach().cpu().clone() for key, value in batch.items()}


def stack_observation_batches(batches: list[Mapping[str, Tensor]]) -> TensorObservationBatch:
    if not batches:
        raise ValueError("batches must contain at least one item.")
    return {
        key: torch.stack([torch.as_tensor(batch[key]).detach().cpu() for batch in batches], dim=0)
        for key in batches[0]
    }


def flatten_observation_time_env(batch: Mapping[str, Tensor]) -> TensorObservationBatch:
    flattened: TensorObservationBatch = {}
    for key, value in batch.items():
        if value.ndim < 2:
            raise ValueError("Observation batch fields must have at least [time, env] dimensions.")
        flattened[key] = value.reshape((-1,) + tuple(value.shape[2:]))
    return flattened


@dataclass(frozen=True)
class RolloutBatch:
    observations: TensorObservationBatch
    actions: Tensor
    log_probs: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor

    def flatten(self) -> "FlattenedRolloutBatch":
        if self.actions.ndim < 3:
            raise ValueError("actions must have shape [time, env, ...].")
        return FlattenedRolloutBatch(
            observations=flatten_observation_time_env(self.observations),
            actions=self.actions.reshape((-1,) + tuple(self.actions.shape[2:])),
            log_probs=self.log_probs.reshape(-1),
            values=self.values.reshape(-1),
            advantages=self.advantages.reshape(-1),
            returns=self.returns.reshape(-1),
        )


@dataclass(frozen=True)
class FlattenedRolloutBatch:
    observations: TensorObservationBatch
    actions: Tensor
    log_probs: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = torch.zeros((), dtype=torch.float32)
        self.var = torch.ones((), dtype=torch.float32)
        self.count = torch.tensor(float(epsilon), dtype=torch.float32)

    def update(self, values: Tensor) -> None:
        flat_values = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
        if flat_values.numel() == 0:
            return
        batch_mean = flat_values.mean()
        batch_var = flat_values.var(unbiased=False)
        batch_count = torch.tensor(float(flat_values.numel()), dtype=torch.float32)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta.square() * (self.count * batch_count / total_count)
        new_var = (m_a + m_b + correction) / total_count
        self.mean = new_mean
        self.var = new_var.clamp_min(1e-8)
        self.count = total_count

    def normalize(self, values: Tensor, eps: float = 1e-8) -> Tensor:
        scale = torch.sqrt(self.var).clamp_min(eps)
        return torch.as_tensor(values, dtype=torch.float32) / scale

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "mean": self.mean.detach().cpu().clone(),
            "var": self.var.detach().cpu().clone(),
            "count": self.count.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        self.mean = torch.as_tensor(state_dict["mean"], dtype=torch.float32).detach().cpu().clone()
        self.var = torch.as_tensor(state_dict["var"], dtype=torch.float32).detach().cpu().clone()
        self.count = torch.as_tensor(state_dict["count"], dtype=torch.float32).detach().cpu().clone()


class RolloutTrajectoryBuffer:
    def __init__(self) -> None:
        self._observations: list[TensorObservationBatch] = []
        self._actions: list[Tensor] = []
        self._log_probs: list[Tensor] = []
        self._rewards: list[Tensor] = []
        self._dones: list[Tensor] = []
        self._values: list[Tensor] = []

    def __len__(self) -> int:
        return len(self._rewards)

    def add(
        self,
        observation_batch: Mapping[str, Tensor],
        action_batch: Tensor,
        log_prob_batch: Tensor,
        reward_batch: Tensor,
        done_batch: Tensor,
        value_batch: Tensor,
    ) -> None:
        self._observations.append(clone_observation_batch(observation_batch))
        self._actions.append(torch.as_tensor(action_batch, dtype=torch.float32).detach().cpu().clone())
        self._log_probs.append(torch.as_tensor(log_prob_batch, dtype=torch.float32).detach().cpu().clone())
        self._rewards.append(torch.as_tensor(reward_batch, dtype=torch.float32).detach().cpu().clone())
        self._dones.append(torch.as_tensor(done_batch, dtype=torch.float32).detach().cpu().clone())
        self._values.append(torch.as_tensor(value_batch, dtype=torch.float32).detach().cpu().clone())

    def build_batch(
        self,
        *,
        last_values: Tensor,
        gamma: float,
        gae_lambda: float,
        reward_normalizer: RunningMeanStd | None = None,
    ) -> RolloutBatch:
        if not self._observations:
            raise ValueError("Cannot build a PPO batch from an empty rollout buffer.")

        observations = stack_observation_batches(self._observations)
        actions = torch.stack(self._actions, dim=0)
        log_probs = torch.stack(self._log_probs, dim=0)
        rewards = torch.stack(self._rewards, dim=0)
        dones = torch.stack(self._dones, dim=0)
        values = torch.stack(self._values, dim=0)
        if reward_normalizer is not None:
            reward_normalizer.update(rewards)
            rewards = reward_normalizer.normalize(rewards)

        last_values = torch.as_tensor(last_values, dtype=torch.float32).detach().cpu().clone()
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_values)
        next_values = last_values

        for step in range(rewards.size(0) - 1, -1, -1):
            next_nonterminal = 1.0 - dones[step]
            delta = rewards[step] + (gamma * next_values * next_nonterminal) - values[step]
            gae = delta + (gamma * gae_lambda * next_nonterminal * gae)
            advantages[step] = gae
            next_values = values[step]

        returns = advantages + values
        return RolloutBatch(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages,
            returns=returns,
        )
