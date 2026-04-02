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
from .data import TensorReplayActionRecord, TensorReplayBatch
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


def _slice_replay_batch_range(batch: TensorReplayBatch, start: int, end: int) -> TensorReplayBatch:
    return TensorReplayBatch(
        obs={key: value[start:end] for key, value in batch.obs.items()},
        action=TensorReplayActionRecord(
            allocation=batch.action.allocation[start:end],
        ),
        reward=batch.reward[start:end],
        next_obs={key: value[start:end] for key, value in batch.next_obs.items()},
        done=batch.done[start:end],
        is_demo=batch.is_demo[start:end],
        collapse_flag=batch.collapse_flag[start:end],
        topology_id=batch.topology_id[start:end],
        pool_power_demo_flag=batch.pool_power_demo_flag[start:end],
        demo_return_target=batch.demo_return_target[start:end],
        demo_return_valid=batch.demo_return_valid[start:end],
    )


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([torch.norm(grad, p=2) for grad in grads])
    return float(torch.norm(norms, p=2).item())


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
        self.last_actor_bc_loss = 0.0
        self.last_actor_bc_coef = 0.0
        self.last_actor_q_coef = 1.0
        self.last_replay_demo_frac = 0.0
        self.last_replay_pool_power_demo_frac = 0.0
        self.last_replay_collapse_frac = 0.0
        self.last_actor_grad_norm = 0.0
        self.last_critic_grad_norm = 0.0
        self.last_actor_grad_norm_pre_clip = 0.0
        self.last_actor_grad_norm_post_clip = 0.0
        self.last_critic_grad_norm_pre_clip = 0.0
        self.last_critic_grad_norm_post_clip = 0.0
        self.last_q_filter_enabled = 0.0
        self.last_q_filter_pass_frac = 0.0
        self.last_q_filter_demo_q_mean = 0.0
        self.last_q_filter_actor_q_mean = 0.0
        self.last_q_filter_margin_mean = 0.0

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
            "last_actor_bc_loss": float(self.last_actor_bc_loss),
            "last_actor_bc_coef": float(self.last_actor_bc_coef),
            "last_actor_q_coef": float(self.last_actor_q_coef),
            "last_replay_demo_frac": float(self.last_replay_demo_frac),
            "last_replay_pool_power_demo_frac": float(self.last_replay_pool_power_demo_frac),
            "last_replay_collapse_frac": float(self.last_replay_collapse_frac),
            "last_actor_grad_norm": float(self.last_actor_grad_norm),
            "last_critic_grad_norm": float(self.last_critic_grad_norm),
            "last_actor_grad_norm_pre_clip": float(self.last_actor_grad_norm_pre_clip),
            "last_actor_grad_norm_post_clip": float(self.last_actor_grad_norm_post_clip),
            "last_critic_grad_norm_pre_clip": float(self.last_critic_grad_norm_pre_clip),
            "last_critic_grad_norm_post_clip": float(self.last_critic_grad_norm_post_clip),
            "last_q_filter_enabled": float(self.last_q_filter_enabled),
            "last_q_filter_pass_frac": float(self.last_q_filter_pass_frac),
            "last_q_filter_demo_q_mean": float(self.last_q_filter_demo_q_mean),
            "last_q_filter_actor_q_mean": float(self.last_q_filter_actor_q_mean),
            "last_q_filter_margin_mean": float(self.last_q_filter_margin_mean),
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
        self.last_actor_bc_loss = float(state_dict.get("last_actor_bc_loss", 0.0))
        self.last_actor_bc_coef = float(state_dict.get("last_actor_bc_coef", 0.0))
        self.last_actor_q_coef = float(state_dict.get("last_actor_q_coef", 1.0))
        self.last_replay_demo_frac = float(state_dict.get("last_replay_demo_frac", 0.0))
        self.last_replay_pool_power_demo_frac = float(state_dict.get("last_replay_pool_power_demo_frac", 0.0))
        self.last_replay_collapse_frac = float(state_dict.get("last_replay_collapse_frac", 0.0))
        self.last_actor_grad_norm = float(state_dict.get("last_actor_grad_norm", 0.0))
        self.last_critic_grad_norm = float(state_dict.get("last_critic_grad_norm", 0.0))
        self.last_actor_grad_norm_pre_clip = float(
            state_dict.get("last_actor_grad_norm_pre_clip", self.last_actor_grad_norm)
        )
        self.last_actor_grad_norm_post_clip = float(
            state_dict.get("last_actor_grad_norm_post_clip", self.last_actor_grad_norm)
        )
        self.last_critic_grad_norm_pre_clip = float(
            state_dict.get("last_critic_grad_norm_pre_clip", self.last_critic_grad_norm)
        )
        self.last_critic_grad_norm_post_clip = float(
            state_dict.get("last_critic_grad_norm_post_clip", self.last_critic_grad_norm)
        )
        self.last_q_filter_enabled = float(state_dict.get("last_q_filter_enabled", 0.0))
        self.last_q_filter_pass_frac = float(state_dict.get("last_q_filter_pass_frac", 0.0))
        self.last_q_filter_demo_q_mean = float(state_dict.get("last_q_filter_demo_q_mean", 0.0))
        self.last_q_filter_actor_q_mean = float(state_dict.get("last_q_filter_actor_q_mean", 0.0))
        self.last_q_filter_margin_mean = float(state_dict.get("last_q_filter_margin_mean", 0.0))

    def _current_demo_bc_coef(self, global_env_steps: int | None) -> float:
        if int(self.config.warmup_steps) <= 0:
            return 0.0

        if global_env_steps is None:
            return float(self.config.actor_demo_bc_coef)

        total_rollout_env_steps = int(self.config.total_updates) * int(self.config.steps_per_update) * int(self.config.num_workers)
        warmup_end_step = int(self.config.warmup_steps)
        decay_end_fraction = float(self.config.actor_demo_bc_decay_end_fraction)
        decay_end_step = int(round(float(total_rollout_env_steps) * decay_end_fraction))
        decay_end_step = max(warmup_end_step, decay_end_step)
        current_step = max(0, int(global_env_steps))

        if current_step >= decay_end_step:
            return 0.0
        if decay_end_step <= warmup_end_step:
            return 0.0

        decay_progress = float(current_step - warmup_end_step) / float(decay_end_step - warmup_end_step)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        return float(self.config.actor_demo_bc_coef) * (1.0 - decay_progress)

    def _current_actor_q_coef(self, global_env_steps: int | None) -> float:
        if global_env_steps is None:
            return float(self.config.online_actor_q_coef_final)

        warmup_end_step = int(self.config.warmup_steps)
        total_rollout_env_steps = int(self.config.total_updates) * int(self.config.steps_per_update) * int(self.config.num_workers)
        ramp_end_step = int(round(float(total_rollout_env_steps) * float(self.config.online_actor_q_coef_ramp_end_fraction)))
        ramp_end_step = max(warmup_end_step, ramp_end_step)
        current_step = max(0, int(global_env_steps))

        if current_step <= warmup_end_step:
            return float(self.config.online_actor_q_coef_initial)
        if current_step >= ramp_end_step or ramp_end_step <= warmup_end_step:
            return float(self.config.online_actor_q_coef_final)

        progress = float(current_step - warmup_end_step) / float(ramp_end_step - warmup_end_step)
        progress = min(max(progress, 0.0), 1.0)
        initial_coef = float(self.config.online_actor_q_coef_initial)
        final_coef = float(self.config.online_actor_q_coef_final)
        return initial_coef + (final_coef - initial_coef) * progress

    def _critic_loss_sum(self, prediction: Tensor, target: Tensor, *, valid_mask: Tensor | None = None) -> Tensor:
        if self.config.critic_loss_type == "huber":
            loss = F.huber_loss(
                prediction,
                target,
                reduction="none",
                delta=float(self.config.critic_huber_delta),
            )
        else:
            loss = F.mse_loss(prediction, target, reduction="none")
        if valid_mask is not None:
            loss = loss[valid_mask]
        return loss.sum()

    def _resolve_pretrain_batch_size(self, batch_size: int | None) -> int:
        if batch_size is not None:
            resolved = int(batch_size)
        elif self.config.demo_pretrain_batch_size is not None:
            resolved = int(self.config.demo_pretrain_batch_size)
        else:
            resolved = int(self.config.batch_size)
        if resolved <= 0:
            raise ValueError("Pretrain batch size must be positive.")
        return resolved

    def evaluate_actor_bc_on_demo_batch(
        self,
        cpu_batch: TensorReplayBatch,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        if len(cpu_batch) <= 0:
            return {
                "actor_bc_val_loss": 0.0,
                "actor_bc_val_num_entries": 0.0,
            }

        resolved_batch_size = self._resolve_pretrain_batch_size(batch_size)
        total_squared_error = 0.0
        total_entries = 0
        with torch.no_grad():
            for start, end in _chunk_ranges(len(cpu_batch), resolved_batch_size):
                chunk_cpu_batch = _slice_replay_batch_range(cpu_batch, start, end)
                batch = chunk_cpu_batch if self.device.type == "cpu" else chunk_cpu_batch.to(self.device)
                actor_outputs = self.actor.deterministic_action_tensor_batch(batch.obs)
                valid_demo_mask = batch.obs["local_mask"] & batch.is_demo.view(-1, 1, 1)
                if valid_demo_mask.any():
                    squared_error = (
                        actor_outputs.allocation_matrix - batch.action.allocation
                    ).pow(2)
                    total_squared_error += float(squared_error[valid_demo_mask].sum().item())
                    total_entries += int(valid_demo_mask.sum().item())
        return {
            "actor_bc_val_loss": float(total_squared_error / max(total_entries, 1)),
            "actor_bc_val_num_entries": float(total_entries),
        }

    def evaluate_critic_on_demo_return_batch(
        self,
        cpu_batch: TensorReplayBatch,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        if len(cpu_batch) <= 0:
            return {
                "critic_val_loss": 0.0,
                "critic1_val_loss": 0.0,
                "critic2_val_loss": 0.0,
                "critic_val_num_targets": 0.0,
                "critic_q_pred_mean": 0.0,
                "critic_q_pred_std": 0.0,
                "critic_target_mean": 0.0,
                "critic_target_std": 0.0,
                "critic_error_mean": 0.0,
                "critic_error_std": 0.0,
            }

        resolved_batch_size = self._resolve_pretrain_batch_size(batch_size)
        total_loss_q1 = 0.0
        total_loss_q2 = 0.0
        total_targets = 0
        q_pred_sum = 0.0
        q_pred_sumsq = 0.0
        target_sum = 0.0
        target_sumsq = 0.0
        error_sum = 0.0
        error_sumsq = 0.0
        with torch.no_grad():
            for start, end in _chunk_ranges(len(cpu_batch), resolved_batch_size):
                chunk_cpu_batch = _slice_replay_batch_range(cpu_batch, start, end)
                batch = chunk_cpu_batch if self.device.type == "cpu" else chunk_cpu_batch.to(self.device)
                current_q1, current_q2 = self._twin_critic_forward_batch(
                    self.critics,
                    batch.obs,
                    batch.action.allocation,
                )
                valid_mask = batch.demo_return_valid.bool()
                if not bool(valid_mask.any().item()):
                    continue
                target_q = batch.demo_return_target
                total_loss_q1 += float(self._critic_loss_sum(current_q1, target_q, valid_mask=valid_mask).item())
                total_loss_q2 += float(self._critic_loss_sum(current_q2, target_q, valid_mask=valid_mask).item())
                q_prediction = 0.5 * (current_q1 + current_q2)
                valid_q_prediction = q_prediction[valid_mask]
                valid_target = target_q[valid_mask]
                valid_error = valid_q_prediction - valid_target
                total_targets += int(valid_q_prediction.numel())
                q_pred_sum += float(valid_q_prediction.sum().item())
                q_pred_sumsq += float(valid_q_prediction.pow(2).sum().item())
                target_sum += float(valid_target.sum().item())
                target_sumsq += float(valid_target.pow(2).sum().item())
                error_sum += float(valid_error.sum().item())
                error_sumsq += float(valid_error.pow(2).sum().item())

        mean_q_pred = q_pred_sum / max(total_targets, 1)
        mean_target = target_sum / max(total_targets, 1)
        mean_error = error_sum / max(total_targets, 1)

        def _std(sum_squares: float, mean: float) -> float:
            variance = max(sum_squares / max(total_targets, 1) - mean * mean, 0.0)
            return float(variance ** 0.5)

        critic1_val_loss = total_loss_q1 / max(total_targets, 1)
        critic2_val_loss = total_loss_q2 / max(total_targets, 1)
        return {
            "critic_val_loss": float(0.5 * (critic1_val_loss + critic2_val_loss)),
            "critic1_val_loss": float(critic1_val_loss),
            "critic2_val_loss": float(critic2_val_loss),
            "critic_val_num_targets": float(total_targets),
            "critic_q_pred_mean": float(mean_q_pred),
            "critic_q_pred_std": _std(q_pred_sumsq, mean_q_pred),
            "critic_target_mean": float(mean_target),
            "critic_target_std": _std(target_sumsq, mean_target),
            "critic_error_mean": float(mean_error),
            "critic_error_std": _std(error_sumsq, mean_error),
        }

    def _compute_demo_q_filter_mask(
        self,
        batch: TensorReplayBatch,
        *,
        margin: float,
    ) -> tuple[Tensor, dict[str, float]]:
        demo_mask = batch.is_demo.bool()
        batch_size = _batch_size_from_observations(batch.obs)
        q_filter_mask = torch.zeros_like(demo_mask, dtype=torch.bool)
        total_demo_count = int(demo_mask.sum().item())
        if total_demo_count <= 0:
            return q_filter_mask, {
                "q_filter_enabled": 1.0,
                "q_filter_pass_frac": 0.0,
                "q_filter_demo_q_mean": 0.0,
                "q_filter_actor_q_mean": 0.0,
                "q_filter_margin_mean": 0.0,
            }

        chunk_size = max(1, int(self.config.graph_batch_chunk_size))
        total_pass_count = 0
        total_demo_q = 0.0
        total_actor_q = 0.0
        total_margin = 0.0

        with torch.no_grad():
            for start, end in _chunk_ranges(batch_size, chunk_size):
                chunk_observations = _slice_observation_batch(batch.obs, start, end)
                chunk_demo_mask = demo_mask[start:end]
                if not bool(chunk_demo_mask.any().item()):
                    continue

                actor_outputs = self.actor.deterministic_action_tensor_batch(chunk_observations)
                demo_q1, demo_q2 = self.critics.forward_tensor_batch(
                    chunk_observations,
                    batch.action.allocation[start:end],
                )
                actor_q1, actor_q2 = self.critics.forward_tensor_batch(
                    chunk_observations,
                    actor_outputs.allocation_matrix.detach(),
                )
                demo_q = torch.minimum(demo_q1, demo_q2)
                actor_q = torch.minimum(actor_q1, actor_q2)
                q_margin = demo_q - actor_q

                q_filter_mask[start:end] = chunk_demo_mask & (q_margin > float(margin))
                total_pass_count += int(q_filter_mask[start:end].sum().item())
                total_demo_q += float(demo_q[chunk_demo_mask].sum().item())
                total_actor_q += float(actor_q[chunk_demo_mask].sum().item())
                total_margin += float(q_margin[chunk_demo_mask].sum().item())

        normalization = float(max(total_demo_count, 1))
        return q_filter_mask, {
            "q_filter_enabled": 1.0,
            "q_filter_pass_frac": float(total_pass_count) / normalization,
            "q_filter_demo_q_mean": float(total_demo_q) / normalization,
            "q_filter_actor_q_mean": float(total_actor_q) / normalization,
            "q_filter_margin_mean": float(total_margin) / normalization,
        }

    def actor_bc_pretrain_step(self, batch_size: int | None = None) -> dict[str, float]:
        resolved_batch_size = self._resolve_pretrain_batch_size(batch_size)
        cpu_batch = self.replay_buffer.sample_demo(
            resolved_batch_size,
            device=None,
            max_collapse_ratio=self.config.replay_max_collapse_sample_ratio,
        )
        replay_sample_stats = self.replay_buffer.get_last_sample_stats()
        self.last_replay_demo_frac = float(cpu_batch.is_demo.float().mean().item()) if len(cpu_batch) > 0 else 0.0
        self.last_replay_pool_power_demo_frac = (
            float(cpu_batch.pool_power_demo_flag.float().mean().item()) if len(cpu_batch) > 0 else 0.0
        )
        self.last_replay_collapse_frac = (
            float(cpu_batch.collapse_flag.float().mean().item()) if len(cpu_batch) > 0 else 0.0
        )
        batch = cpu_batch if self.device.type == "cpu" else cpu_batch.to(self.device)
        self.last_actor_bc_coef = float(self.config.warmup_actor_bc_coef)
        self.last_actor_q_coef = 0.0
        actor_metrics = self.update_actor(
            batch,
            actor_q_enabled=False,
            bc_coef=self.last_actor_bc_coef,
            actor_q_coef=self.last_actor_q_coef,
            enable_q_filter=bool(self.config.actor_bc_q_filter_enabled and not self.config.actor_bc_q_filter_online_only),
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        return {
            **actor_metrics,
            "replay_size": float(len(self.replay_buffer)),
            "replay_demo_frac": self.last_replay_demo_frac,
            "replay_pool_power_demo_frac": self.last_replay_pool_power_demo_frac,
            "replay_teacher_frac": self.last_replay_pool_power_demo_frac,
            "replay_collapse_frac": self.last_replay_collapse_frac,
            "actor_q_coef": self.last_actor_q_coef,
            "actor_grad_norm": self.last_actor_grad_norm,
            "critic_grad_norm": self.last_critic_grad_norm,
            "actor_grad_norm_pre_clip": self.last_actor_grad_norm_pre_clip,
            "actor_grad_norm_post_clip": self.last_actor_grad_norm_post_clip,
            "critic_grad_norm_pre_clip": self.last_critic_grad_norm_pre_clip,
            "critic_grad_norm_post_clip": self.last_critic_grad_norm_post_clip,
            "actor_lr": self._current_actor_lr(),
            "critic_lr": self._current_critic_lr(),
            **replay_sample_stats,
        }

    def critic_pretrain_step(self, batch_size: int | None = None) -> dict[str, float]:
        resolved_batch_size = self._resolve_pretrain_batch_size(batch_size)
        cpu_batch = self.replay_buffer.sample_demo(
            resolved_batch_size,
            device=None,
            max_collapse_ratio=self.config.replay_max_collapse_sample_ratio,
        )
        replay_sample_stats = self.replay_buffer.get_last_sample_stats()
        self.last_replay_demo_frac = float(cpu_batch.is_demo.float().mean().item()) if len(cpu_batch) > 0 else 0.0
        self.last_replay_pool_power_demo_frac = (
            float(cpu_batch.pool_power_demo_flag.float().mean().item()) if len(cpu_batch) > 0 else 0.0
        )
        self.last_replay_collapse_frac = (
            float(cpu_batch.collapse_flag.float().mean().item()) if len(cpu_batch) > 0 else 0.0
        )
        batch = cpu_batch if self.device.type == "cpu" else cpu_batch.to(self.device)
        critic_metrics = self.update_critics(batch, use_demo_return_target=True)
        self.soft_update_targets()
        return {
            **critic_metrics,
            "replay_size": float(len(self.replay_buffer)),
            "replay_demo_frac": self.last_replay_demo_frac,
            "replay_pool_power_demo_frac": self.last_replay_pool_power_demo_frac,
            "replay_teacher_frac": self.last_replay_pool_power_demo_frac,
            "replay_collapse_frac": self.last_replay_collapse_frac,
            "actor_q_coef": self.last_actor_q_coef,
            "actor_grad_norm": self.last_actor_grad_norm,
            "critic_grad_norm": self.last_critic_grad_norm,
            "actor_grad_norm_pre_clip": self.last_actor_grad_norm_pre_clip,
            "actor_grad_norm_post_clip": self.last_actor_grad_norm_post_clip,
            "critic_grad_norm_pre_clip": self.last_critic_grad_norm_pre_clip,
            "critic_grad_norm_post_clip": self.last_critic_grad_norm_post_clip,
            "actor_lr": self._current_actor_lr(),
            "critic_lr": self._current_critic_lr(),
            **replay_sample_stats,
        }

    def train_step(
        self,
        global_env_steps: int | None = None,
        *,
        teacher_release_unlocked: bool = False,
    ) -> dict[str, float]:
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
                "actor_bc_loss": self.last_actor_bc_loss,
                "actor_bc_coef": self.last_actor_bc_coef,
                "actor_q_coef": self.last_actor_q_coef,
                "q_filter_enabled": self.last_q_filter_enabled,
                "q_filter_pass_frac": self.last_q_filter_pass_frac,
                "q_filter_demo_q_mean": self.last_q_filter_demo_q_mean,
                "q_filter_actor_q_mean": self.last_q_filter_actor_q_mean,
                "q_filter_margin_mean": self.last_q_filter_margin_mean,
                "loss": 0.0,
                "replay_size": float(len(self.replay_buffer)),
                "replay_demo_frac": self.last_replay_demo_frac,
                "replay_pool_power_demo_frac": self.last_replay_pool_power_demo_frac,
                "replay_teacher_frac": self.last_replay_pool_power_demo_frac,
                "replay_collapse_frac": self.last_replay_collapse_frac,
                "actor_grad_norm": self.last_actor_grad_norm,
                "critic_grad_norm": self.last_critic_grad_norm,
                "actor_grad_norm_pre_clip": self.last_actor_grad_norm_pre_clip,
                "actor_grad_norm_post_clip": self.last_actor_grad_norm_post_clip,
                "critic_grad_norm_pre_clip": self.last_critic_grad_norm_pre_clip,
                "critic_grad_norm_post_clip": self.last_critic_grad_norm_post_clip,
                "actor_lr": self._current_actor_lr(),
                "critic_lr": self._current_critic_lr(),
                "profile_replay_sample_seconds": 0.0,
                "profile_batch_to_device_seconds": 0.0,
                "profile_critic_update_seconds": 0.0,
                "profile_actor_update_seconds": 0.0,
                "profile_target_soft_update_seconds": 0.0,
                **self.replay_buffer.get_last_sample_stats(),
            }

        actor_lr = self._current_actor_lr()
        critic_lr = self._current_critic_lr()
        replay_sample_start = perf_counter()
        cpu_batch = self.replay_buffer.sample(
            self.config.batch_size,
            device=None,
            max_collapse_ratio=self.config.replay_max_collapse_sample_ratio,
        )
        replay_sample_stats = self.replay_buffer.get_last_sample_stats()
        replay_sample_seconds = float(perf_counter() - replay_sample_start)
        self.last_replay_demo_frac = float(cpu_batch.is_demo.float().mean().item()) if len(cpu_batch) > 0 else 0.0
        self.last_replay_pool_power_demo_frac = (
            float(cpu_batch.pool_power_demo_flag.float().mean().item()) if len(cpu_batch) > 0 else 0.0
        )
        self.last_replay_collapse_frac = (
            float(cpu_batch.collapse_flag.float().mean().item()) if len(cpu_batch) > 0 else 0.0
        )

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
            "actor_bc_loss": self.last_actor_bc_loss,
            "actor_bc_coef": self.last_actor_bc_coef,
            "q_filter_enabled": self.last_q_filter_enabled,
            "q_filter_pass_frac": self.last_q_filter_pass_frac,
            "q_filter_demo_q_mean": self.last_q_filter_demo_q_mean,
            "q_filter_actor_q_mean": self.last_q_filter_actor_q_mean,
            "q_filter_margin_mean": self.last_q_filter_margin_mean,
        }
        actor_update_seconds = 0.0
        target_soft_update_seconds = 0.0
        actor_q_coef = 0.0
        actor_q_enabled = not (
            bool(self.config.freeze_actor_q_during_warmup)
            and global_env_steps is not None
            and int(global_env_steps) < int(self.config.warmup_steps)
        )
        bc_coef = (
            self._current_demo_bc_coef(global_env_steps)
            if actor_q_enabled
            else float(self.config.warmup_actor_bc_coef)
        )
        if actor_q_enabled:
            actor_q_coef = self._current_actor_q_coef(global_env_steps)
        q_filter_enabled = bool(self.config.actor_bc_q_filter_enabled) and actor_q_enabled and bc_coef > 0.0
        if q_filter_enabled and bool(self.config.actor_bc_q_filter_require_teacher_release):
            if bool(self.config.adaptive_teacher_release_enabled):
                q_filter_enabled = bool(teacher_release_unlocked)
        self.last_actor_bc_coef = float(bc_coef)
        self.last_actor_q_coef = float(actor_q_coef)
        if self.update_step_count % self.config.policy_delay == 0:
            actor_update_start = perf_counter()
            actor_metrics = self.update_actor(
                batch,
                actor_q_enabled=actor_q_enabled,
                bc_coef=bc_coef,
                actor_q_coef=actor_q_coef,
                enable_q_filter=q_filter_enabled,
            )
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
            "replay_demo_frac": self.last_replay_demo_frac,
            "replay_pool_power_demo_frac": self.last_replay_pool_power_demo_frac,
            "replay_teacher_frac": self.last_replay_pool_power_demo_frac,
            "replay_collapse_frac": self.last_replay_collapse_frac,
            "actor_bc_coef": self.last_actor_bc_coef,
            "actor_q_coef": self.last_actor_q_coef,
            "actor_grad_norm": self.last_actor_grad_norm,
            "critic_grad_norm": self.last_critic_grad_norm,
            "actor_grad_norm_pre_clip": self.last_actor_grad_norm_pre_clip,
            "actor_grad_norm_post_clip": self.last_actor_grad_norm_post_clip,
            "critic_grad_norm_pre_clip": self.last_critic_grad_norm_pre_clip,
            "critic_grad_norm_post_clip": self.last_critic_grad_norm_post_clip,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "profile_replay_sample_seconds": replay_sample_seconds,
            "profile_batch_to_device_seconds": batch_to_device_seconds,
            "profile_critic_update_seconds": critic_update_seconds,
            "profile_actor_update_seconds": actor_update_seconds,
            "profile_target_soft_update_seconds": target_soft_update_seconds,
            **replay_sample_stats,
        }

    def update_critics(self, batch: TensorReplayBatch, *, use_demo_return_target: bool = False) -> dict[str, float]:
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
        valid_target_count = int(batch.demo_return_valid.sum().item()) if use_demo_return_target else int(batch_size)
        if use_demo_return_target and valid_target_count <= 0:
            raise ValueError("critic_pretrain_step requires at least one valid demo_return_target.")
        normalization_count = max(valid_target_count, 1)

        for start, end in _chunk_ranges(batch_size, chunk_size):
            chunk_observations = _slice_observation_batch(observations, start, end)
            chunk_next_observations = _slice_observation_batch(next_observations, start, end)
            chunk_actions = actions[start:end]
            chunk_rewards = rewards[start:end]
            chunk_dones = dones[start:end]
            chunk_valid_mask = batch.demo_return_valid[start:end].bool()

            if use_demo_return_target:
                target_q = batch.demo_return_target[start:end]
            else:
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
            critic1_loss_sum = self._critic_loss_sum(
                current_q1,
                target_q,
                valid_mask=chunk_valid_mask if use_demo_return_target else None,
            )
            critic2_loss_sum = self._critic_loss_sum(
                current_q2,
                target_q,
                valid_mask=chunk_valid_mask if use_demo_return_target else None,
            )
            chunk_loss = (critic1_loss_sum + critic2_loss_sum) / float(normalization_count)
            chunk_loss.backward()

            total_critic1_loss += float(critic1_loss_sum.item())
            total_critic2_loss += float(critic2_loss_sum.item())

        critic_parameters = list(self.critics.critic1.parameters()) + list(self.critics.critic2.parameters())
        if self.config.critic_grad_clip_norm is not None:
            self.last_critic_grad_norm_pre_clip = float(
                torch.nn.utils.clip_grad_norm_(critic_parameters, float(self.config.critic_grad_clip_norm)).item()
            )
            self.last_critic_grad_norm_post_clip = _gradient_norm(critic_parameters)
        else:
            self.last_critic_grad_norm_pre_clip = _gradient_norm(critic_parameters)
            self.last_critic_grad_norm_post_clip = self.last_critic_grad_norm_pre_clip
        self.last_critic_grad_norm = self.last_critic_grad_norm_pre_clip
        self.critic_optimizer.step()

        critic1_loss = total_critic1_loss / float(normalization_count)
        critic2_loss = total_critic2_loss / float(normalization_count)
        critic_loss = critic1_loss + critic2_loss

        return {
            "critic1_loss": float(critic1_loss),
            "critic2_loss": float(critic2_loss),
            "critic_loss": float(critic_loss),
            "critic_grad_norm": float(self.last_critic_grad_norm),
            "critic_grad_norm_pre_clip": float(self.last_critic_grad_norm_pre_clip),
            "critic_grad_norm_post_clip": float(self.last_critic_grad_norm_post_clip),
        }

    def update_actor(
        self,
        batch: TensorReplayBatch,
        *,
        actor_q_enabled: bool,
        bc_coef: float,
        actor_q_coef: float,
        enable_q_filter: bool = False,
    ) -> dict[str, float]:
        observations = batch.obs
        target_actions = batch.action.allocation
        demo_mask = batch.is_demo.bool()
        q_filter_metrics = {
            "q_filter_enabled": 0.0,
            "q_filter_pass_frac": 0.0,
            "q_filter_demo_q_mean": 0.0,
            "q_filter_actor_q_mean": 0.0,
            "q_filter_margin_mean": 0.0,
        }
        effective_demo_mask = demo_mask
        if bool(enable_q_filter):
            effective_demo_mask, q_filter_metrics = self._compute_demo_q_filter_mask(
                batch,
                margin=float(self.config.actor_bc_q_filter_margin),
            )

        self.actor_optimizer.zero_grad(set_to_none=True)

        batch_size = _batch_size_from_observations(observations)
        chunk_size = max(1, int(self.config.graph_batch_chunk_size))
        total_actor_q = 0.0
        total_entropy = 0.0
        total_entropy_rows = 0
        total_logit_square = 0.0
        total_valid_logits = int(observations["local_mask"].sum().item())
        total_demo_valid_entries = int((observations["local_mask"] & effective_demo_mask.view(-1, 1, 1)).sum().item())
        total_bc_square = 0.0

        for start, end in _chunk_ranges(batch_size, chunk_size):
            chunk_observations = _slice_observation_batch(observations, start, end)
            current_outputs = self.actor.deterministic_action_tensor_batch(chunk_observations)
            if actor_q_enabled:
                actor_q = self.critics.critic1.forward_tensor_batch(
                    chunk_observations,
                    current_outputs.allocation_matrix,
                )
                actor_q_loss_chunk = -actor_q.sum() / float(batch_size)
                total_actor_q += float(actor_q.sum().item())
            else:
                actor_q_loss_chunk = current_outputs.allocation_matrix.new_zeros(())

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

            total_entropy += float(entropy_sum.item())
            total_entropy_rows += entropy_rows

            chunk_demo_mask = effective_demo_mask[start:end]
            if bool(chunk_demo_mask.any().item()) and total_demo_valid_entries > 0:
                chunk_valid_demo_mask = chunk_observations["local_mask"] & chunk_demo_mask.view(-1, 1, 1)
                bc_square_sum = (allocation - target_actions[start:end]).pow(2)[chunk_valid_demo_mask].sum()
                bc_loss_chunk = (float(bc_coef) * bc_square_sum) / float(total_demo_valid_entries)
                total_bc_square += float(bc_square_sum.item())
            else:
                bc_loss_chunk = actor_q_loss_chunk.new_zeros(())

            actor_reg_loss_chunk = (
                (-float(self.config.actor_entropy_coef) * entropy_term)
                + (float(self.config.actor_logit_l2_coef) * logit_l2_term)
            )
            ((float(actor_q_coef) * actor_q_loss_chunk) + actor_reg_loss_chunk + bc_loss_chunk).backward()

        actor_parameters = list(self.actor.parameters())
        if self.config.actor_grad_clip_norm is not None:
            self.last_actor_grad_norm_pre_clip = float(
                torch.nn.utils.clip_grad_norm_(actor_parameters, float(self.config.actor_grad_clip_norm)).item()
            )
            self.last_actor_grad_norm_post_clip = _gradient_norm(actor_parameters)
        else:
            self.last_actor_grad_norm_pre_clip = _gradient_norm(actor_parameters)
            self.last_actor_grad_norm_post_clip = self.last_actor_grad_norm_pre_clip
        self.last_actor_grad_norm = self.last_actor_grad_norm_pre_clip
        self.actor_optimizer.step()

        actor_q_loss = -(total_actor_q / float(batch_size))
        mean_entropy = total_entropy / float(max(total_entropy_rows, 1))
        mean_logit_l2 = total_logit_square / float(max(total_valid_logits, 1)) if total_valid_logits > 0 else 0.0
        actor_reg_loss = (
            (-float(self.config.actor_entropy_coef) * mean_entropy)
            + (float(self.config.actor_logit_l2_coef) * mean_logit_l2)
        )
        actor_bc_loss = (
            float(bc_coef) * (total_bc_square / float(total_demo_valid_entries))
            if total_demo_valid_entries > 0
            else 0.0
        )
        actor_loss = (float(actor_q_coef) * actor_q_loss) + actor_reg_loss + actor_bc_loss

        self.last_actor_loss = float(actor_loss)
        self.last_actor_q_loss = float(actor_q_loss)
        self.last_actor_entropy = float(mean_entropy)
        self.last_actor_logit_l2 = float(mean_logit_l2)
        self.last_actor_reg_loss = float(actor_reg_loss)
        self.last_actor_bc_loss = float(actor_bc_loss)
        self.last_q_filter_enabled = float(q_filter_metrics["q_filter_enabled"])
        self.last_q_filter_pass_frac = float(q_filter_metrics["q_filter_pass_frac"])
        self.last_q_filter_demo_q_mean = float(q_filter_metrics["q_filter_demo_q_mean"])
        self.last_q_filter_actor_q_mean = float(q_filter_metrics["q_filter_actor_q_mean"])
        self.last_q_filter_margin_mean = float(q_filter_metrics["q_filter_margin_mean"])
        return {
            "actor_loss": self.last_actor_loss,
            "actor_q_loss": self.last_actor_q_loss,
            "actor_entropy": self.last_actor_entropy,
            "actor_logit_l2": self.last_actor_logit_l2,
            "actor_reg_loss": self.last_actor_reg_loss,
            "actor_bc_loss": self.last_actor_bc_loss,
            "actor_bc_coef": self.last_actor_bc_coef,
            "actor_q_coef": self.last_actor_q_coef,
            "q_filter_enabled": self.last_q_filter_enabled,
            "q_filter_pass_frac": self.last_q_filter_pass_frac,
            "q_filter_demo_q_mean": self.last_q_filter_demo_q_mean,
            "q_filter_actor_q_mean": self.last_q_filter_actor_q_mean,
            "q_filter_margin_mean": self.last_q_filter_margin_mean,
            "actor_grad_norm": self.last_actor_grad_norm,
            "actor_grad_norm_pre_clip": self.last_actor_grad_norm_pre_clip,
            "actor_grad_norm_post_clip": self.last_actor_grad_norm_post_clip,
        }

    def soft_update_targets(self) -> None:
        soft_update_module(self.target_actor, self.actor, tau=self.config.tau)
        soft_update_module(self.target_critics.critic1, self.critics.critic1, tau=self.config.tau)
        soft_update_module(self.target_critics.critic2, self.critics.critic2, tau=self.config.tau)
