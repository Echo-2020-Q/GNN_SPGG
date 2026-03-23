from __future__ import annotations

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


def soft_update_module(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau)
            target_param.add_(tau * source_param)


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

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        critic_parameters = list(self.critics.critic1.parameters()) + list(self.critics.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=config.critic_lr)

        self.update_step_count = 0
        self.actor_version = 0
        self.last_actor_loss = 0.0

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())

    def publish_actor_state(self) -> tuple[dict[str, Tensor], int]:
        return _copy_state_dict_to_cpu(self.actor), self.actor_version

    def train_step(self) -> dict[str, float]:
        if len(self.replay_buffer) < max(1, self.config.batch_size):
            return {
                "critic1_loss": 0.0,
                "critic2_loss": 0.0,
                "critic_loss": 0.0,
                "actor_loss": self.last_actor_loss,
                "loss": 0.0,
                "replay_size": float(len(self.replay_buffer)),
            }

        batch = self.replay_buffer.sample(self.config.batch_size)
        critic_metrics = self.update_critics(batch)

        actor_metrics: dict[str, float] = {"actor_loss": self.last_actor_loss}
        if self.update_step_count % self.config.policy_delay == 0:
            actor_metrics = self.update_actor(batch)
            self.soft_update_targets()
            self.actor_version += 1

        self.update_step_count += 1
        return {
            **critic_metrics,
            **actor_metrics,
            "loss": critic_metrics["critic_loss"] + actor_metrics["actor_loss"],
            "replay_size": float(len(self.replay_buffer)),
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
        current_actions = [self.actor.deterministic_action(obs).allocation_matrix for obs in observations]
        actor_q = self.critics.critic1.forward_batch(observations, current_actions)
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.last_actor_loss = float(actor_loss.item())
        return {"actor_loss": self.last_actor_loss}

    def soft_update_targets(self) -> None:
        soft_update_module(self.target_actor, self.actor, tau=self.config.tau)
        soft_update_module(self.target_critics.critic1, self.critics.critic1, tau=self.config.tau)
        soft_update_module(self.target_critics.critic2, self.critics.critic2, tau=self.config.tau)
