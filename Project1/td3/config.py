from __future__ import annotations

from dataclasses import dataclass, field


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
    batch_size: int = 32
    replay_capacity: int = 200_000
    warmup_steps: int = 1_000
    train_every: int = 1
    gradient_steps_per_update: int = 1
    policy_delay: int = 2

    rollout_logit_noise_std: float = 0.30
    rollout_logit_noise_clip: float = 0.50
    rollout_noise_decay: float = 0.9995
    target_logit_noise_std: float = 0.10
    target_logit_noise_clip: float = 0.25

    num_workers: int = 1
    worker_sync_interval: int = 1
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
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
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
        if self.worker_sync_interval <= 0:
            raise ValueError("worker_sync_interval must be positive.")


@dataclass
class WorkerConfig:
    worker_id: int
    seed: int
    rollout_steps_per_sync: int = 256
    noise_scale_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.rollout_steps_per_sync <= 0:
            raise ValueError("rollout_steps_per_sync must be positive.")
        if self.noise_scale_multiplier < 0.0:
            raise ValueError("noise_scale_multiplier must be non-negative.")


@dataclass
class DomainRandomizationConfig:
    enabled: bool = False
    network_types: tuple[str, ...] = ("regular",)
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


@dataclass
class EvalConfig:
    num_episodes: int = 3
    collapse_resource_threshold: float = 1e-6
    per_network_episode_budget: int = 0
    tracked_network_types: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")
        if self.per_network_episode_budget < 0:
            raise ValueError("per_network_episode_budget must be non-negative.")
