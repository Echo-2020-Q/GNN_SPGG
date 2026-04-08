from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphPPOConfig:
    total_updates: int = 100
    steps_per_update: int = 128
    eval_interval: int = 10
    eval_episodes: int = 3
    device: str = "cpu"
    seed: int | None = None

    gamma: float = 0.99
    learning_rate: float = 3e-4
    lr_schedule_type: str = "constant"
    lr_final: float = 1e-5
    lr_decay_rate: float = 0.05
    lr_decay_steps: int = 1_000
    weight_decay: float = 0.0

    ppo_update_epochs: int = 4
    ppo_minibatch_size: int = 256
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 1e-3
    ppo_gae_lambda: float = 0.95
    ppo_max_grad_norm: float | None = 0.5
    ppo_target_kl: float | None = 0.03
    ppo_reward_normalization: bool = False
    ppo_advantage_normalization: bool = True

    num_workers: int = 1
    num_envs_per_worker: int = 1
    rollout_device: str | tuple[str, ...] = "cpu"
    rollout_inference_mode: str = "local"
    rollout_inference_batch_timeout_ms: float = 0.0
    rollout_num_threads: int | None = None
    overlap_rollout_and_update: bool = False
    collapse_resource_threshold: float = 1e-6

    def __post_init__(self) -> None:
        if self.total_updates <= 0:
            raise ValueError("total_updates must be positive.")
        if self.steps_per_update <= 0:
            raise ValueError("steps_per_update must be positive.")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive.")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.lr_schedule_type not in {"constant", "exponential_decay"}:
            raise ValueError("lr_schedule_type must be one of {'constant', 'exponential_decay'}.")
        if self.lr_final <= 0.0:
            raise ValueError("lr_final must be positive.")
        if self.lr_final > self.learning_rate:
            raise ValueError("lr_final must be <= learning_rate.")
        if not 0.0 < self.lr_decay_rate <= 1.0:
            raise ValueError("lr_decay_rate must be in (0, 1].")
        if self.lr_decay_steps <= 0:
            raise ValueError("lr_decay_steps must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.ppo_update_epochs <= 0:
            raise ValueError("ppo_update_epochs must be positive.")
        if self.ppo_minibatch_size <= 0:
            raise ValueError("ppo_minibatch_size must be positive.")
        if self.ppo_clip_ratio <= 0.0:
            raise ValueError("ppo_clip_ratio must be positive.")
        if self.ppo_value_coef < 0.0:
            raise ValueError("ppo_value_coef must be non-negative.")
        if self.ppo_entropy_coef < 0.0:
            raise ValueError("ppo_entropy_coef must be non-negative.")
        if not 0.0 <= self.ppo_gae_lambda <= 1.0:
            raise ValueError("ppo_gae_lambda must be in [0, 1].")
        if self.ppo_max_grad_norm is not None and self.ppo_max_grad_norm <= 0.0:
            raise ValueError("ppo_max_grad_norm must be positive when provided.")
        if self.ppo_target_kl is not None and self.ppo_target_kl <= 0.0:
            raise ValueError("ppo_target_kl must be positive when provided.")
        if not isinstance(self.ppo_reward_normalization, bool):
            raise ValueError("ppo_reward_normalization must be a bool.")
        if not isinstance(self.ppo_advantage_normalization, bool):
            raise ValueError("ppo_advantage_normalization must be a bool.")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive.")
        if self.num_envs_per_worker <= 0:
            raise ValueError("num_envs_per_worker must be positive.")
        if self.rollout_inference_mode not in {"local", "centralized"}:
            raise ValueError("rollout_inference_mode must be one of {'local', 'centralized'}.")
        if self.rollout_inference_batch_timeout_ms < 0.0:
            raise ValueError("rollout_inference_batch_timeout_ms must be non-negative.")
        rollout_device = self.rollout_device
        if isinstance(rollout_device, list):
            rollout_device = tuple(str(item) for item in rollout_device)
        elif isinstance(rollout_device, tuple):
            rollout_device = tuple(str(item) for item in rollout_device)
        elif isinstance(rollout_device, str):
            rollout_device = str(rollout_device)
        else:
            raise ValueError("rollout_device must be a string or a sequence of strings.")
        if isinstance(rollout_device, tuple):
            if not rollout_device:
                raise ValueError("rollout_device sequence must contain at least one device.")
            if any(not item for item in rollout_device):
                raise ValueError("rollout_device entries must be non-empty strings.")
        elif not rollout_device:
            raise ValueError("rollout_device must be a non-empty string.")
        self.rollout_device = rollout_device
        if self.rollout_num_threads is not None and self.rollout_num_threads <= 0:
            raise ValueError("rollout_num_threads must be positive when provided.")
        if not isinstance(self.overlap_rollout_and_update, bool):
            raise ValueError("overlap_rollout_and_update must be a bool.")
        if self.collapse_resource_threshold < 0.0:
            raise ValueError("collapse_resource_threshold must be non-negative.")
