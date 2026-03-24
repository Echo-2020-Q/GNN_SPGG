from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from Project1.policies.gnn_rl import GNNAllocationPolicy

from .config import GraphTD3Config
from .critic import TwinCritic
from .exploration import LogitSpaceExplorer
from .replay import ReplayBuffer


def _copy_state_dict_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _move_nested_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _move_nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_nested_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_to_cpu(item) for item in value)
    return value


def soft_update_module(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau)
            target_param.add_(tau * source_param)


def _mean_allocation_entropy(allocation_matrix: Tensor) -> Tensor:
    safe_allocation = allocation_matrix.clamp_min(1e-12)
    row_entropies = -(allocation_matrix * safe_allocation.log()).sum(dim=1)
    return row_entropies.mean()


def _masked_mean_square(values: Tensor, mask: Tensor) -> Tensor:
    valid_values = values[mask]
    if valid_values.numel() == 0:
        return values.new_zeros(())
    return valid_values.pow(2).mean()


class GraphTD3Learner:
    def __init__(
        self,
        actor: GNNAllocationPolicy,
        critics: TwinCritic,
        target_actor: GNNAllocationPolicy,
        target_critics: TwinCritic,
        replay_buffer: ReplayBuffer,
        target_explorer: LogitSpaceExplorer,
        config: GraphTD3Config,
    ):
        self.config = config
        self.device = torch.device(config.device)

        self.actor = actor.to(self.device)
        self.target_actor = target_actor.to(self.device)
        self.critics = critics.to(self.device)
        self.target_critics = target_critics.to(self.device)
        self.replay_buffer = replay_buffer
        self.target_explorer = target_explorer

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.actor_weight_decay,
        )
        critic_parameters = list(self.critics.critic1.parameters()) + list(self.critics.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=config.critic_lr,
            weight_decay=config.critic_weight_decay,
        )
        self.actor_initial_lr = float(config.actor_lr)
        self.critic_initial_lr = float(config.critic_lr)

        self.update_step_count = 0
        self.actor_version = 0
        self.last_actor_loss = 0.0
        self.last_actor_entropy = 0.0
        self.last_actor_logit_l2 = 0.0
        self.last_actor_reg_loss = 0.0
        self.last_actor_q_loss = 0.0

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())
        self._apply_lr_schedule(step=0)

    def _scheduled_lr(self, initial_lr: float, step: int) -> float:
        if self.config.lr_schedule_type == "constant":
            return float(initial_lr)

        exponent = float(step) / float(self.config.lr_decay_steps)
        decayed_lr = float(initial_lr) * (float(self.config.lr_decay_rate) ** exponent)
        return max(float(self.config.lr_final), decayed_lr)

    def _set_optimizer_lr(self, optimizer: torch.optim.Optimizer, lr: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(lr)

    def _apply_lr_schedule(self, step: int) -> None:
        self._set_optimizer_lr(self.actor_optimizer, self._scheduled_lr(self.actor_initial_lr, step))
        self._set_optimizer_lr(self.critic_optimizer, self._scheduled_lr(self.critic_initial_lr, step))

    def _current_actor_lr(self) -> float:
        return float(self.actor_optimizer.param_groups[0]["lr"])

    def _current_critic_lr(self) -> float:
        return float(self.critic_optimizer.param_groups[0]["lr"])

    def publish_actor_state(self) -> tuple[dict[str, Tensor], int]:
        return _copy_state_dict_to_cpu(self.actor), self.actor_version

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "trainer_config": asdict(self.config),
            "policy_config": asdict(self.actor.config),
            "actor_state_dict": _copy_state_dict_to_cpu(self.actor),
            "target_actor_state_dict": _copy_state_dict_to_cpu(self.target_actor),
            "critics_state_dict": _copy_state_dict_to_cpu(self.critics),
            "target_critics_state_dict": _copy_state_dict_to_cpu(self.target_critics),
            "actor_optimizer_state_dict": _move_nested_to_cpu(self.actor_optimizer.state_dict()),
            "critic_optimizer_state_dict": _move_nested_to_cpu(self.critic_optimizer.state_dict()),
            "update_step_count": int(self.update_step_count),
            "actor_version": int(self.actor_version),
            "last_actor_loss": float(self.last_actor_loss),
            "last_actor_q_loss": float(self.last_actor_q_loss),
            "last_actor_entropy": float(self.last_actor_entropy),
            "last_actor_logit_l2": float(self.last_actor_logit_l2),
            "last_actor_reg_loss": float(self.last_actor_reg_loss),
        }

    def load_checkpoint_state(self, state_dict: dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor_state_dict"])
        self.target_actor.load_state_dict(state_dict["target_actor_state_dict"])
        self.critics.load_state_dict(state_dict["critics_state_dict"])
        self.target_critics.load_state_dict(state_dict["target_critics_state_dict"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer_state_dict"])

        self.update_step_count = int(state_dict["update_step_count"])
        self.actor_version = int(state_dict["actor_version"])
        self.last_actor_loss = float(state_dict["last_actor_loss"])
        self.last_actor_q_loss = float(state_dict["last_actor_q_loss"])
        self.last_actor_entropy = float(state_dict["last_actor_entropy"])
        self.last_actor_logit_l2 = float(state_dict["last_actor_logit_l2"])
        self.last_actor_reg_loss = float(state_dict["last_actor_reg_loss"])

    def train_step(self) -> dict[str, float]:
        if len(self.replay_buffer) < max(1, self.config.batch_size):
            return {
                "critic1_loss": 0.0,
                "critic2_loss": 0.0,
                "critic_loss": 0.0,
                "actor_loss": self.last_actor_loss,
                "actor_q_loss": self.last_actor_q_loss,
                "actor_entropy": self.last_actor_entropy,
                "actor_logit_l2": self.last_actor_logit_l2,
                "actor_reg_loss": self.last_actor_reg_loss,
                "loss": 0.0,
                "replay_size": float(len(self.replay_buffer)),
                "actor_lr": self._current_actor_lr(),
                "critic_lr": self._current_critic_lr(),
            }

        actor_lr = self._current_actor_lr()
        critic_lr = self._current_critic_lr()
        batch = self.replay_buffer.sample(self.config.batch_size)
        critic_metrics = self.update_critics(batch)

        actor_metrics: dict[str, float] = {
            "actor_loss": self.last_actor_loss,
            "actor_q_loss": self.last_actor_q_loss,
            "actor_entropy": self.last_actor_entropy,
            "actor_logit_l2": self.last_actor_logit_l2,
            "actor_reg_loss": self.last_actor_reg_loss,
        }
        if self.update_step_count % self.config.policy_delay == 0:
            actor_metrics = self.update_actor(batch)
            self.soft_update_targets()
            self.actor_version += 1

        self.update_step_count += 1
        self._apply_lr_schedule(step=self.update_step_count)
        return {
            **critic_metrics,
            **actor_metrics,
            "loss": critic_metrics["critic_loss"] + actor_metrics["actor_loss"],
            "replay_size": float(len(self.replay_buffer)),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
        }

    def update_critics(self, batch: list[Any]) -> dict[str, float]:
        observations = [transition.obs for transition in batch]
        next_observations = [transition.next_obs for transition in batch]
        actions = [transition.action.to_tensors(self.device).allocation for transition in batch]
        rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([float(transition.done) for transition in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            target_outputs = [self.target_actor.deterministic_action(obs) for obs in next_observations]
            target_actions = [
                self.target_explorer.apply_to_policy_output(
                    policy_output=output,
                    ego_mask=torch.as_tensor(obs["local_mask"], dtype=torch.bool, device=self.device),
                    pool_values=torch.as_tensor(obs["pool_grown"], dtype=torch.float32, device=self.device),
                    noise_std=self.config.target_logit_noise_std,
                    noise_clip=self.config.target_logit_noise_clip,
                ).allocation
                for output, obs in zip(target_outputs, next_observations)
            ]
            target_q1, target_q2 = self.target_critics.forward_batch(next_observations, target_actions)
            target_q = rewards + (self.config.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2))

        current_q1, current_q2 = self.critics.forward_batch(observations, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "critic_loss": float(critic_loss.item()),
        }

    def update_actor(self, batch: list[Any]) -> dict[str, float]:
        observations = [transition.obs for transition in batch]
        current_outputs = [self.actor.deterministic_action(obs) for obs in observations]
        current_actions = [output.allocation_matrix for output in current_outputs]
        actor_q = self.critics.critic1.forward_batch(observations, current_actions)
        actor_q_loss = -actor_q.mean()

        mean_entropy = actor_q_loss.new_zeros(())
        mean_logit_l2 = actor_q_loss.new_zeros(())
        if current_outputs:
            entropy_terms = [_mean_allocation_entropy(output.allocation_matrix) for output in current_outputs]
            mean_entropy = torch.stack(entropy_terms).mean()

            logit_l2_terms: list[Tensor] = []
            for output, obs in zip(current_outputs, observations):
                if output.logits is None:
                    continue
                mask = torch.as_tensor(obs["local_mask"], dtype=torch.bool, device=self.device)
                logit_l2_terms.append(_masked_mean_square(output.logits, mask))
            if logit_l2_terms:
                mean_logit_l2 = torch.stack(logit_l2_terms).mean()

        actor_reg_loss = (
            (-float(self.config.actor_entropy_coef) * mean_entropy)
            + (float(self.config.actor_logit_l2_coef) * mean_logit_l2)
        )
        actor_loss = actor_q_loss + actor_reg_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.last_actor_loss = float(actor_loss.item())
        self.last_actor_q_loss = float(actor_q_loss.item())
        self.last_actor_entropy = float(mean_entropy.item())
        self.last_actor_logit_l2 = float(mean_logit_l2.item())
        self.last_actor_reg_loss = float(actor_reg_loss.item())
        return {
            "actor_loss": self.last_actor_loss,
            "actor_q_loss": self.last_actor_q_loss,
            "actor_entropy": self.last_actor_entropy,
            "actor_logit_l2": self.last_actor_logit_l2,
            "actor_reg_loss": self.last_actor_reg_loss,
        }

    def soft_update_targets(self) -> None:
        soft_update_module(self.target_actor, self.actor, tau=self.config.tau)
        soft_update_module(self.target_critics.critic1, self.critics.critic1, tau=self.config.tau)
        soft_update_module(self.target_critics.critic2, self.critics.critic2, tau=self.config.tau)
