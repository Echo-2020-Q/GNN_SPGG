from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from Project1.policies.gnn_rl import GNNAllocationPolicy

from .buffer import FlattenedRolloutBatch
from .config import GraphPPOConfig


class GraphPPOLearner:
    def __init__(self, actor: GNNAllocationPolicy, config: GraphPPOConfig):
        self.actor = actor.to(config.device)
        self.config = config
        self.device = torch.device(config.device)
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.optimization_updates = 0

    def _current_learning_rate(self) -> float:
        if self.config.lr_schedule_type == "constant":
            return float(self.config.learning_rate)
        decayed = float(self.config.learning_rate) * (
            float(self.config.lr_decay_rate) ** (float(self.optimization_updates) / float(self.config.lr_decay_steps))
        )
        return max(float(self.config.lr_final), decayed)

    def _set_learning_rate(self) -> float:
        learning_rate = self._current_learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def _evaluate_values_full_batch(self, batch: FlattenedRolloutBatch) -> Tensor:
        values: list[Tensor] = []
        minibatch_size = min(int(self.config.ppo_minibatch_size), int(batch.actions.size(0)))
        with torch.no_grad():
            for start in range(0, int(batch.actions.size(0)), minibatch_size):
                end = min(start + minibatch_size, int(batch.actions.size(0)))
                observation_batch = {
                    key: value[start:end].to(device=self.device)
                    for key, value in batch.observations.items()
                }
                action_batch = batch.actions[start:end].to(device=self.device)
                evaluated = self.actor.evaluate_action_tensor_batch(observation_batch, action_batch)
                values.append(evaluated.value.detach().cpu())
        return torch.cat(values, dim=0)

    def update(self, rollout_batch: FlattenedRolloutBatch) -> dict[str, float]:
        if rollout_batch.actions.size(0) <= 0:
            raise ValueError("rollout_batch must contain at least one sample.")

        self.actor.train()
        learning_rate = self._set_learning_rate()

        advantages = rollout_batch.advantages.detach().cpu().clone()
        if self.config.ppo_advantage_normalization and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        num_samples = int(rollout_batch.actions.size(0))
        minibatch_size = min(int(self.config.ppo_minibatch_size), num_samples)
        aggregated_metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "actor_grad_norm_pre_clip": 0.0,
            "actor_grad_norm_post_clip": 0.0,
        }
        num_minibatches = 0
        early_stop = False

        for _ in range(int(self.config.ppo_update_epochs)):
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, minibatch_size):
                indices = permutation[start:start + minibatch_size]
                observation_batch = {
                    key: value.index_select(0, indices).to(device=self.device)
                    for key, value in rollout_batch.observations.items()
                }
                action_batch = rollout_batch.actions.index_select(0, indices).to(device=self.device)
                old_log_probs = rollout_batch.log_probs.index_select(0, indices).to(device=self.device)
                minibatch_advantages = advantages.index_select(0, indices).to(device=self.device)
                returns = rollout_batch.returns.index_select(0, indices).to(device=self.device)

                evaluated = self.actor.evaluate_action_tensor_batch(observation_batch, action_batch)
                new_log_probs = evaluated.log_prob
                if new_log_probs is None:
                    raise RuntimeError("PPO requires policy.evaluate_action_tensor_batch() to return log_prob.")
                entropy = evaluated.entropy
                if entropy is None:
                    entropy = torch.zeros_like(new_log_probs)

                value = evaluated.value
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                unclipped_objective = ratio * minibatch_advantages
                clipped_objective = torch.clamp(
                    ratio,
                    1.0 - float(self.config.ppo_clip_ratio),
                    1.0 + float(self.config.ppo_clip_ratio),
                ) * minibatch_advantages
                policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
                value_loss = F.mse_loss(value, returns)
                entropy_mean = entropy.mean()
                loss = policy_loss + (float(self.config.ppo_value_coef) * value_loss) - (
                    float(self.config.ppo_entropy_coef) * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                grad_norm_pre_clip = 0.0
                grad_norm_post_clip = 0.0
                if self.config.ppo_max_grad_norm is not None:
                    grad_norm_pre_clip = float(
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(),
                            max_norm=float(self.config.ppo_max_grad_norm),
                        ).item()
                    )
                    grad_norm_post_clip = min(grad_norm_pre_clip, float(self.config.ppo_max_grad_norm))
                self.optimizer.step()

                approx_kl = float(torch.mean((ratio - 1.0) - log_ratio).detach().cpu().item())
                clipfrac = float(
                    torch.mean((torch.abs(ratio - 1.0) > float(self.config.ppo_clip_ratio)).to(dtype=torch.float32))
                    .detach()
                    .cpu()
                    .item()
                )
                aggregated_metrics["loss"] += float(loss.detach().cpu().item())
                aggregated_metrics["policy_loss"] += float(policy_loss.detach().cpu().item())
                aggregated_metrics["value_loss"] += float(value_loss.detach().cpu().item())
                aggregated_metrics["entropy"] += float(entropy_mean.detach().cpu().item())
                aggregated_metrics["approx_kl"] += approx_kl
                aggregated_metrics["clipfrac"] += clipfrac
                aggregated_metrics["actor_grad_norm_pre_clip"] += grad_norm_pre_clip
                aggregated_metrics["actor_grad_norm_post_clip"] += grad_norm_post_clip
                num_minibatches += 1

                target_kl = self.config.ppo_target_kl
                if target_kl is not None and approx_kl > (1.5 * float(target_kl)):
                    early_stop = True
                    break
            if early_stop:
                break

        self.optimization_updates += 1
        self.actor.eval()

        if num_minibatches <= 0:
            raise RuntimeError("PPO update produced no minibatches.")
        averaged_metrics = {key: value / float(num_minibatches) for key, value in aggregated_metrics.items()}

        predicted_values = self._evaluate_values_full_batch(rollout_batch)
        returns = rollout_batch.returns.detach().cpu()
        value_error = returns - predicted_values
        returns_variance = float(torch.var(returns, unbiased=False).item())
        explained_variance = 0.0
        if returns_variance > 1e-8:
            explained_variance = 1.0 - float(torch.var(value_error, unbiased=False).item()) / returns_variance

        return {
            "loss": float(averaged_metrics["loss"]),
            "policy_loss": float(averaged_metrics["policy_loss"]),
            "value_loss": float(averaged_metrics["value_loss"]),
            "entropy": float(averaged_metrics["entropy"]),
            "ppo_policy_loss": float(averaged_metrics["policy_loss"]),
            "ppo_value_loss": float(averaged_metrics["value_loss"]),
            "ppo_entropy": float(averaged_metrics["entropy"]),
            "ppo_approx_kl": float(averaged_metrics["approx_kl"]),
            "ppo_clipfrac": float(averaged_metrics["clipfrac"]),
            "explained_variance": float(explained_variance),
            "actor_lr": float(learning_rate),
            "critic_lr": float(learning_rate),
            "actor_grad_norm_pre_clip": float(averaged_metrics["actor_grad_norm_pre_clip"]),
            "actor_grad_norm_post_clip": float(averaged_metrics["actor_grad_norm_post_clip"]),
            "critic_grad_norm_pre_clip": 0.0,
            "critic_grad_norm_post_clip": 0.0,
            "profile_actor_update_seconds": 0.0,
            "profile_critic_update_seconds": 0.0,
            "profile_target_soft_update_seconds": 0.0,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "actor_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.actor.state_dict().items()
            },
            "optimizer_state": self.optimizer.state_dict(),
            "optimization_updates": int(self.optimization_updates),
        }

    def load_checkpoint_state(self, state_dict: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(dict(state_dict["actor_state_dict"]))
        self.optimizer.load_state_dict(dict(state_dict["optimizer_state"]))
        self.optimization_updates = int(state_dict.get("optimization_updates", 0))
