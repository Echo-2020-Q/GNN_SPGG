from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphTD3Config:
    total_updates: int = 100
    steps_per_update: int = 64
    eval_interval: int = 10
    eval_episodes: int = 3
    device: str = "cpu"
    seed: int | None = None

    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    actor_lr: float | None = None
    critic_lr: float | None = None
    lr_schedule_type: str = "constant"
    lr_final: float = 1e-5
    lr_decay_rate: float = 0.05
    lr_decay_steps: int = 1_000
    actor_weight_decay: float = 0.0
    critic_weight_decay: float = 0.0
    actor_entropy_coef: float = 0.0
    actor_logit_l2_coef: float = 0.0
    critic_state_hidden_dim: int | None = None
    critic_action_hidden_dim: int | None = None
    critic_pool_hidden_dim: int | None = None
    critic_q_hidden_dim: int | None = None
    batch_size: int = 32
    graph_batch_chunk_size: int = 16
    replay_capacity: int = 200_000
    warmup_steps: int = 1_000
    warmup_behavior_mode: str = "random_only"
    warmup_selection_granularity: str = "per_episode"
    warmup_uniform_prob: float = 0.0
    warmup_proportional_prob: float = 0.0
    warmup_constant_mix_prob: float = 0.0
    warmup_pool_power_mix_prob: float = 0.0
    warmup_random_logits_prob: float = 1.0
    warmup_constant_mix_omega: float = 0.5
    warmup_pool_power_k: float = 19.0
    warmup_logit_noise_std: float = 0.0
    warmup_logit_noise_clip: float = 0.0
    train_every: int = 1
    gradient_steps_per_update: int = 1
    policy_delay: int = 2

    rollout_logit_noise_std: float = 0.30
    rollout_logit_noise_clip: float = 0.50
    rollout_noise_decay: float = 0.9995
    target_logit_noise_std: float = 0.10
    target_logit_noise_clip: float = 0.25

    num_workers: int = 1
    num_envs_per_worker: int = 1
    worker_sync_interval: int = 1
    overlap_rollout_and_update: bool = True
    worker_rpc_timeout_seconds: float = 300.0
    rollout_device: str | tuple[str, ...] = "cpu"
    rollout_inference_mode: str = "local"
    rollout_inference_batch_timeout_ms: float = 2.0
    rollout_num_threads: int | None = None
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
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.tau <= 0.0 or self.tau > 1.0:
            raise ValueError("tau must be in (0, 1].")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.actor_lr is None:
            self.actor_lr = self.learning_rate
        if self.critic_lr is None:
            self.critic_lr = self.learning_rate
        if self.actor_lr <= 0.0:
            raise ValueError("actor_lr must be positive.")
        if self.critic_lr <= 0.0:
            raise ValueError("critic_lr must be positive.")
        if self.lr_schedule_type not in {"constant", "exponential_decay"}:
            raise ValueError("lr_schedule_type must be one of {'constant', 'exponential_decay'}.")
        if self.lr_final <= 0.0:
            raise ValueError("lr_final must be positive.")
        if self.lr_final > min(self.actor_lr, self.critic_lr):
            raise ValueError("lr_final must be <= min(actor_lr, critic_lr).")
        if self.lr_decay_rate <= 0.0 or self.lr_decay_rate > 1.0:
            raise ValueError("lr_decay_rate must be in (0, 1].")
        if self.lr_decay_steps <= 0:
            raise ValueError("lr_decay_steps must be positive.")
        if self.actor_weight_decay < 0.0:
            raise ValueError("actor_weight_decay must be non-negative.")
        if self.critic_weight_decay < 0.0:
            raise ValueError("critic_weight_decay must be non-negative.")
        if self.actor_entropy_coef < 0.0:
            raise ValueError("actor_entropy_coef must be non-negative.")
        if self.actor_logit_l2_coef < 0.0:
            raise ValueError("actor_logit_l2_coef must be non-negative.")
        if self.critic_state_hidden_dim is not None and self.critic_state_hidden_dim <= 0:
            raise ValueError("critic_state_hidden_dim must be positive when provided.")
        if self.critic_action_hidden_dim is not None and self.critic_action_hidden_dim <= 0:
            raise ValueError("critic_action_hidden_dim must be positive when provided.")
        if self.critic_pool_hidden_dim is not None and self.critic_pool_hidden_dim <= 0:
            raise ValueError("critic_pool_hidden_dim must be positive when provided.")
        if self.critic_q_hidden_dim is not None and self.critic_q_hidden_dim <= 0:
            raise ValueError("critic_q_hidden_dim must be positive when provided.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.graph_batch_chunk_size <= 0:
            raise ValueError("graph_batch_chunk_size must be positive.")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.warmup_behavior_mode not in {"random_only", "heuristic_mix"}:
            raise ValueError("warmup_behavior_mode must be one of {'random_only', 'heuristic_mix'}.")
        if self.warmup_selection_granularity not in {"per_episode", "per_step"}:
            raise ValueError("warmup_selection_granularity must be one of {'per_episode', 'per_step'}.")
        if self.warmup_uniform_prob < 0.0:
            raise ValueError("warmup_uniform_prob must be non-negative.")
        if self.warmup_proportional_prob < 0.0:
            raise ValueError("warmup_proportional_prob must be non-negative.")
        if self.warmup_constant_mix_prob < 0.0:
            raise ValueError("warmup_constant_mix_prob must be non-negative.")
        if self.warmup_pool_power_mix_prob < 0.0:
            raise ValueError("warmup_pool_power_mix_prob must be non-negative.")
        if self.warmup_random_logits_prob < 0.0:
            raise ValueError("warmup_random_logits_prob must be non-negative.")
        if self.warmup_constant_mix_omega < 0.0 or self.warmup_constant_mix_omega > 1.0:
            raise ValueError("warmup_constant_mix_omega must be in [0, 1].")
        if self.warmup_pool_power_k < 0.0:
            raise ValueError("warmup_pool_power_k must be non-negative.")
        if self.warmup_logit_noise_std < 0.0:
            raise ValueError("warmup_logit_noise_std must be non-negative.")
        if self.warmup_logit_noise_clip < 0.0:
            raise ValueError("warmup_logit_noise_clip must be non-negative.")
        if self.warmup_behavior_mode == "heuristic_mix":
            warmup_mix_total = (
                self.warmup_uniform_prob
                + self.warmup_proportional_prob
                + self.warmup_constant_mix_prob
                + self.warmup_pool_power_mix_prob
                + self.warmup_random_logits_prob
            )
            if warmup_mix_total <= 0.0:
                raise ValueError("heuristic warm-up requires at least one positive behavior probability.")
        if self.train_every <= 0:
            raise ValueError("train_every must be positive.")
        if self.gradient_steps_per_update <= 0:
            raise ValueError("gradient_steps_per_update must be positive.")
        if self.policy_delay <= 0:
            raise ValueError("policy_delay must be positive.")
        if self.rollout_logit_noise_std < 0.0:
            raise ValueError("rollout_logit_noise_std must be non-negative.")
        if self.rollout_logit_noise_clip < 0.0:
            raise ValueError("rollout_logit_noise_clip must be non-negative.")
        if self.rollout_noise_decay <= 0.0 or self.rollout_noise_decay > 1.0:
            raise ValueError("rollout_noise_decay must be in (0, 1].")
        if self.target_logit_noise_std < 0.0:
            raise ValueError("target_logit_noise_std must be non-negative.")
        if self.target_logit_noise_clip < 0.0:
            raise ValueError("target_logit_noise_clip must be non-negative.")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive.")
        if self.num_envs_per_worker <= 0:
            raise ValueError("num_envs_per_worker must be positive.")
        if self.worker_sync_interval <= 0:
            raise ValueError("worker_sync_interval must be positive.")
        if not isinstance(self.overlap_rollout_and_update, bool):
            raise ValueError("overlap_rollout_and_update must be a bool.")
        if self.worker_rpc_timeout_seconds <= 0.0:
            raise ValueError("worker_rpc_timeout_seconds must be positive.")
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


@dataclass
class WorkerConfig:
    worker_id: int
    seed: int
    rollout_steps_per_sync: int = 256
    num_envs_per_worker: int = 1
    noise_scale_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.rollout_steps_per_sync <= 0:
            raise ValueError("rollout_steps_per_sync must be positive.")
        if self.num_envs_per_worker <= 0:
            raise ValueError("num_envs_per_worker must be positive.")
        if self.noise_scale_multiplier < 0.0:
            raise ValueError("noise_scale_multiplier must be non-negative.")


@dataclass
class DomainRandomizationConfig:
    enabled: bool = False
    network_types: tuple[str, ...] = ("regular",)
    network_type_weights: tuple[float, ...] | None = None
    num_nodes_choices: tuple[int, ...] = (100,)
    regular_degree_choices: tuple[int, ...] = (4,)
    er_mean_degree_choices: tuple[float, ...] = (4.0,)
    ws_degree_choices: tuple[int, ...] = (4,)
    ws_rewiring_choices: tuple[float, ...] = (0.10,)
    ba_attachment_choices: tuple[int, ...] = (2,)
    initial_resource_range: tuple[float, float] | None = None
    initial_cooperation_prob_range: tuple[float, float] | None = None
    alpha_range: tuple[float, float] | None = None
    r_range: tuple[float, float] | None = None
    p_max_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.network_types:
            raise ValueError("network_types must contain at least one entry.")
        if self.network_type_weights is not None:
            if len(self.network_type_weights) != len(self.network_types):
                raise ValueError("network_type_weights must have the same length as network_types.")
            if any(weight < 0.0 for weight in self.network_type_weights):
                raise ValueError("network_type_weights must be non-negative.")
            if sum(self.network_type_weights) <= 0.0:
                raise ValueError("network_type_weights must sum to a positive value.")


@dataclass
class EvalConfig:
    num_episodes: int = 3
    collapse_resource_threshold: float = 1e-6

    def __post_init__(self) -> None:
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")
