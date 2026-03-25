from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from Project1.policies.gnn_rl import BatchedPolicyOutput, GNNAllocationPolicy

from .config import GraphTD3Config
from .critic import GraphActionCritic, TwinCritic
from .data import TensorReplayBatch
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
    row_entropies = -(allocation_matrix * safe_allocation.log()).sum(dim=-1)
    return row_entropies.mean()


def _masked_mean_square(values: Tensor, mask: Tensor) -> Tensor:
    valid_values = values[mask]
    if valid_values.numel() == 0:
        return values.new_zeros(())
    return valid_values.pow(2).mean()


def _batch_size_from_observations(observations: Mapping[str, Tensor]) -> int:
    if not observations:
        return 0
    first_value = next(iter(observations.values()))
    if first_value.ndim == 0:
        raise ValueError("Batched observations must include a leading batch dimension.")
    return int(first_value.shape[0])


def _slice_observation_batch(observations: Mapping[str, Tensor], start: int, end: int) -> dict[str, Tensor]:
    return {key: value[start:end] for key, value in observations.items()}


def _chunk_ranges(batch_size: int, chunk_size: int):
    for start in range(0, batch_size, chunk_size):
        yield start, min(start + chunk_size, batch_size)


def _concat_batched_policy_outputs(outputs: list[BatchedPolicyOutput]) -> BatchedPolicyOutput:
    if not outputs:
        raise ValueError("outputs must contain at least one item.")
    return BatchedPolicyOutput(
        allocation_matrix=torch.cat([output.allocation_matrix for output in outputs], dim=0),
        transferred_resources=torch.cat([output.transferred_resources for output in outputs], dim=0),
        incoming_resources=torch.cat([output.incoming_resources for output in outputs], dim=0),
        value=torch.cat([output.value for output in outputs], dim=0),
        logits=(
            torch.cat([output.logits for output in outputs], dim=0)
            if outputs[0].logits is not None
            else None
        ),
    )


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

    def _batched_actor_outputs(
        self,
        actor: GNNAllocationPolicy,
        observations: Mapping[str, Tensor],
    ) -> BatchedPolicyOutput:
        batch_size = _batch_size_from_observations(observations)
        if batch_size == 0:
            raise ValueError("observations must contain at least one item.")

        chunk_size = max(1, int(self.config.graph_batch_chunk_size))
        chunk_outputs: list[BatchedPolicyOutput] = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk_outputs.append(actor.deterministic_action_tensor_batch(_slice_observation_batch(observations, start, end)))
        return _concat_batched_policy_outputs(chunk_outputs)

    def _critic_forward_batch(
        self,
        critic: GraphActionCritic,
        observations: Mapping[str, Tensor],
        actions: Tensor,
    ) -> Tensor:
        batch_size = _batch_size_from_observations(observations)
        if batch_size == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        chunk_size = max(1, int(self.config.graph_batch_chunk_size))
        chunk_outputs: list[Tensor] = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk_outputs.append(
                critic.forward_tensor_batch(
                    _slice_observation_batch(observations, start, end),
                    actions[start:end],
                )
            )
        return torch.cat(chunk_outputs, dim=0)

    def _twin_critic_forward_batch(
        self,
        critics: TwinCritic,
        observations: Mapping[str, Tensor],
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = _batch_size_from_observations(observations)
        if batch_size == 0:
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return empty, empty

        chunk_size = max(1, int(self.config.graph_batch_chunk_size))
        chunk_outputs_1: list[Tensor] = []
        chunk_outputs_2: list[Tensor] = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            q1, q2 = critics.forward_tensor_batch(
                _slice_observation_batch(observations, start, end),
                actions[start:end],
            )
            chunk_outputs_1.append(q1)
            chunk_outputs_2.append(q2)
        return torch.cat(chunk_outputs_1, dim=0), torch.cat(chunk_outputs_2, dim=0)

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
                "profile_replay_sample_seconds": 0.0,
                "profile_batch_to_device_seconds": 0.0,
                "profile_critic_update_seconds": 0.0,
                "profile_actor_update_seconds": 0.0,
                "profile_target_soft_update_seconds": 0.0,
            }

        actor_lr = self._current_actor_lr()
        critic_lr = self._current_critic_lr()
        replay_sample_start = perf_counter()
        cpu_batch = self.replay_buffer.sample(self.config.batch_size, device=None)
        replay_sample_seconds = float(perf_counter() - replay_sample_start)

        batch_to_device_seconds = 0.0
        if self.device.type == "cpu":
            batch = cpu_batch
        else:
            batch_to_device_start = perf_counter()
            batch = cpu_batch.to(self.device)
            batch_to_device_seconds = float(perf_counter() - batch_to_device_start)

        critic_update_start = perf_counter()
        critic_metrics = self.update_critics(batch)
        critic_update_seconds = float(perf_counter() - critic_update_start)

        actor_metrics: dict[str, float] = {
            "actor_loss": self.last_actor_loss,
            "actor_q_loss": self.last_actor_q_loss,
            "actor_entropy": self.last_actor_entropy,
            "actor_logit_l2": self.last_actor_logit_l2,
            "actor_reg_loss": self.last_actor_reg_loss,
        }
        actor_update_seconds = 0.0
        target_soft_update_seconds = 0.0
        if self.update_step_count % self.config.policy_delay == 0:
            actor_update_start = perf_counter()
            actor_metrics = self.update_actor(batch)
            actor_update_seconds = float(perf_counter() - actor_update_start)

            target_soft_update_start = perf_counter()
            self.soft_update_targets()
            target_soft_update_seconds = float(perf_counter() - target_soft_update_start)
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
            "profile_replay_sample_seconds": replay_sample_seconds,
            "profile_batch_to_device_seconds": batch_to_device_seconds,
            "profile_critic_update_seconds": critic_update_seconds,
            "profile_actor_update_seconds": actor_update_seconds,
            "profile_target_soft_update_seconds": target_soft_update_seconds,
        }

    def update_critics(self, batch: TensorReplayBatch) -> dict[str, float]:
        observations = batch.obs
        next_observations = batch.next_obs
        actions = batch.action.allocation
        rewards = batch.reward
        dones = batch.done
        self.critic_optimizer.zero_grad(set_to_none=True)

        batch_size = _batch_size_from_observations(observations)
        chunk_size = max(1, int(self.config.graph_batch_chunk_size))
        total_critic1_loss = 0.0
        total_critic2_loss = 0.0

        for start, end in _chunk_ranges(batch_size, chunk_size):
            chunk_observations = _slice_observation_batch(observations, start, end)
            chunk_next_observations = _slice_observation_batch(next_observations, start, end)
            chunk_actions = actions[start:end]
            chunk_rewards = rewards[start:end]
            chunk_dones = dones[start:end]

            with torch.no_grad():
                target_outputs = self.target_actor.deterministic_action_tensor_batch(chunk_next_observations)
                if target_outputs.logits is None:
                    raise ValueError("Target actor must provide logits for TD3 target smoothing.")
                target_actions = self.target_explorer.apply_to_logits(
                    logits=target_outputs.logits,
                    ego_mask=chunk_next_observations["local_mask"],
                    pool_values=chunk_next_observations["pool_grown"],
                    noise_std=self.config.target_logit_noise_std,
                    noise_clip=self.config.target_logit_noise_clip,
                ).allocation
                target_q1, target_q2 = self.target_critics.forward_tensor_batch(
                    chunk_next_observations,
                    target_actions,
                )
                target_q = chunk_rewards + (
                    self.config.gamma * (1.0 - chunk_dones) * torch.minimum(target_q1, target_q2)
                )

            current_q1, current_q2 = self.critics.forward_tensor_batch(
                chunk_observations,
                chunk_actions,
            )
            critic1_loss_sum = F.mse_loss(current_q1, target_q, reduction="sum")
            critic2_loss_sum = F.mse_loss(current_q2, target_q, reduction="sum")
            chunk_loss = (critic1_loss_sum + critic2_loss_sum) / float(batch_size)
            chunk_loss.backward()

            total_critic1_loss += float(critic1_loss_sum.item())
            total_critic2_loss += float(critic2_loss_sum.item())

        self.critic_optimizer.step()

        critic1_loss = total_critic1_loss / float(batch_size)
        critic2_loss = total_critic2_loss / float(batch_size)
        critic_loss = critic1_loss + critic2_loss

        return {
            "critic1_loss": float(critic1_loss),
            "critic2_loss": float(critic2_loss),
            "critic_loss": float(critic_loss),
        }

    def update_actor(self, batch: TensorReplayBatch) -> dict[str, float]:
        observations = batch.obs

        self.actor_optimizer.zero_grad(set_to_none=True)

        batch_size = _batch_size_from_observations(observations)
        chunk_size = max(1, int(self.config.graph_batch_chunk_size))
        total_actor_q = 0.0
        total_entropy = 0.0
        total_entropy_rows = 0
        total_logit_square = 0.0
        total_valid_logits = int(observations["local_mask"].sum().item())

        for start, end in _chunk_ranges(batch_size, chunk_size):
            chunk_observations = _slice_observation_batch(observations, start, end)
            current_outputs = self.actor.deterministic_action_tensor_batch(chunk_observations)
            actor_q = self.critics.critic1.forward_tensor_batch(
                chunk_observations,
                current_outputs.allocation_matrix,
            )

            actor_q_loss_chunk = -actor_q.sum() / float(batch_size)

            allocation = current_outputs.allocation_matrix
            safe_allocation = allocation.clamp_min(1e-12)
            entropy_sum = -(allocation * safe_allocation.log()).sum(dim=-1).sum()
            entropy_rows = int(allocation.shape[0] * allocation.shape[1])
            entropy_term = entropy_sum / float(max(batch_size * allocation.shape[1], 1))

            if current_outputs.logits is not None and total_valid_logits > 0:
                valid_logits = current_outputs.logits[chunk_observations["local_mask"]]
                logit_square_sum = valid_logits.pow(2).sum()
                logit_l2_term = logit_square_sum / float(total_valid_logits)
                total_logit_square += float(logit_square_sum.item())
            else:
                logit_l2_term = actor_q_loss_chunk.new_zeros(())

            total_actor_q += float(actor_q.sum().item())
            total_entropy += float(entropy_sum.item())
            total_entropy_rows += entropy_rows

            actor_reg_loss_chunk = (
                (-float(self.config.actor_entropy_coef) * entropy_term)
                + (float(self.config.actor_logit_l2_coef) * logit_l2_term)
            )
            (actor_q_loss_chunk + actor_reg_loss_chunk).backward()

        self.actor_optimizer.step()

        actor_q_loss = -(total_actor_q / float(batch_size))
        mean_entropy = total_entropy / float(max(total_entropy_rows, 1))
        mean_logit_l2 = total_logit_square / float(max(total_valid_logits, 1)) if total_valid_logits > 0 else 0.0
        actor_reg_loss = (
            (-float(self.config.actor_entropy_coef) * mean_entropy)
            + (float(self.config.actor_logit_l2_coef) * mean_logit_l2)
        )
        actor_loss = actor_q_loss + actor_reg_loss

        self.last_actor_loss = float(actor_loss)
        self.last_actor_q_loss = float(actor_q_loss)
        self.last_actor_entropy = float(mean_entropy)
        self.last_actor_logit_l2 = float(mean_logit_l2)
        self.last_actor_reg_loss = float(actor_reg_loss)
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
