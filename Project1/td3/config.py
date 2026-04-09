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
    replay_strategy: str = "fifo"
    replay_topology_names: tuple[str, ...] = ("fixed",)
    replay_recent_fraction: float = 0.50
    replay_long_term_fraction: float = 0.35
    replay_demo_fraction: float = 0.15
    replay_demo_behavior_source: str = "pool_power_mix"
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
    freeze_actor_during_warmup: bool = False
    freeze_actor_q_during_warmup: bool = True
    warmup_actor_bc_coef: float = 1.0
    actor_demo_bc_coef: float = 0.25
    actor_demo_bc_decay_end_fraction: float = 0.50
    actor_bc_q_filter_enabled: bool = False
    actor_bc_q_filter_margin: float = 0.0
    actor_bc_q_filter_online_only: bool = True
    actor_bc_q_filter_require_teacher_release: bool = True
    demo_pretrain_enabled: bool = False
    demo_collection_env_steps: int = 0
    demo_collection_behavior_source: str = "pool_power_mix"
    demo_collection_use_domain_randomization: bool = True
    demo_collection_network_types: tuple[str, ...] = ()
    demo_collection_runtime: str = "parallel_cpu"
    actor_bc_pretrain_updates: int = 0
    critic_pretrain_updates: int = 0
    demo_pretrain_batch_size: int | None = None
    demo_pretrain_validation_batch_size: int | None = None
    demo_validation_fraction: float = 0.10
    demo_pretrain_eval_interval: int = 200
    demo_pretrain_patience: int = 5
    demo_pretrain_min_relative_improvement: float = 0.01
    demo_dataset_save_path: str | None = None
    demo_critic_pretrain_target_mode: str = "n_step"
    demo_critic_pretrain_n_step: int = 20
    critic_bridge_enabled: bool = False
    critic_bridge_env_steps: int = 0
    critic_bridge_updates: int = 0
    critic_bridge_batch_size: int | None = None
    critic_bridge_validation_fraction: float = 0.10
    critic_bridge_eval_interval: int = 200
    critic_bridge_patience: int = 5
    critic_bridge_min_relative_improvement: float = 0.01
    critic_bridge_behavior_mode: str = "actor_only"
    critic_bridge_teacher_takeover_prob: float = 0.0
    critic_bridge_use_curriculum_stage0_distribution: bool = True
    critic_bridge_teacher_return_aux_schedule: str = "fixed"
    critic_bridge_teacher_return_aux_levels: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)
    critic_bridge_teacher_return_aux_required_evals: int = 2
    critic_bridge_teacher_return_aux_max_val_ratio: float = 1.10
    critic_bridge_teacher_return_aux_max_error_ratio: float = 0.20
    critic_bridge_teacher_return_aux_coef: float = 0.0
    teacher_takeover_enabled: bool = True
    teacher_takeover_behavior_source: str = "pool_power_mix"
    teacher_takeover_granularity: str = "per_step"
    teacher_takeover_start_prob: float = 0.8
    teacher_takeover_end_prob: float = 0.0
    teacher_takeover_decay_end_fraction: float = 0.30
    adaptive_teacher_release_enabled: bool = False
    adaptive_teacher_release_mode: str = "legacy"
    adaptive_teacher_release_min_cooperation: float = 0.80
    adaptive_teacher_release_min_return_ratio: float = 0.90
    adaptive_teacher_release_max_actor_bc_val_ratio: float = 1.20
    adaptive_teacher_release_max_critic_val_ratio: float = 1.20
    adaptive_teacher_release_required_evals: int = 3
    adaptive_teacher_release_min_criteria: int = 2
    freeze_actor_q_until_teacher_release: bool = False
    online_actor_q_coef_initial: float = 0.2
    online_actor_q_coef_final: float = 1.0
    online_actor_q_coef_ramp_end_fraction: float = 0.30
    critic_loss_type: str = "huber"
    critic_huber_delta: float = 1.0
    actor_grad_clip_norm: float | None = 5.0
    critic_grad_clip_norm: float | None = 5.0
    train_every: int = 1
    gradient_steps_per_update: int = 1
    policy_delay: int = 2
    replay_collapse_fc_threshold: float = 0.10
    replay_max_collapse_sample_ratio: float = 0.20

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
        if self.replay_strategy not in {"fifo", "topology_stratified_mixed"}:
            raise ValueError("replay_strategy must be one of {'fifo', 'topology_stratified_mixed'}.")
        if not self.replay_topology_names:
            raise ValueError("replay_topology_names must contain at least one entry.")
        if self.replay_recent_fraction < 0.0:
            raise ValueError("replay_recent_fraction must be non-negative.")
        if self.replay_long_term_fraction < 0.0:
            raise ValueError("replay_long_term_fraction must be non-negative.")
        if self.replay_demo_fraction < 0.0:
            raise ValueError("replay_demo_fraction must be non-negative.")
        if self.replay_strategy == "topology_stratified_mixed":
            total_replay_fraction = (
                float(self.replay_recent_fraction)
                + float(self.replay_long_term_fraction)
                + float(self.replay_demo_fraction)
            )
            if abs(total_replay_fraction - 1.0) > 1e-6:
                raise ValueError(
                    "For topology_stratified_mixed replay, replay_recent_fraction + "
                    "replay_long_term_fraction + replay_demo_fraction must sum to 1."
                )
        if self.replay_demo_behavior_source not in {"pool_power_mix"}:
            raise ValueError("replay_demo_behavior_source currently only supports 'pool_power_mix'.")
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
        if not isinstance(self.freeze_actor_during_warmup, bool):
            raise ValueError("freeze_actor_during_warmup must be a bool.")
        if not isinstance(self.freeze_actor_q_during_warmup, bool):
            raise ValueError("freeze_actor_q_during_warmup must be a bool.")
        if self.warmup_actor_bc_coef < 0.0:
            raise ValueError("warmup_actor_bc_coef must be non-negative.")
        if self.actor_demo_bc_coef < 0.0:
            raise ValueError("actor_demo_bc_coef must be non-negative.")
        if self.actor_demo_bc_decay_end_fraction < 0.0 or self.actor_demo_bc_decay_end_fraction > 1.0:
            raise ValueError("actor_demo_bc_decay_end_fraction must be in [0, 1].")
        if not isinstance(self.actor_bc_q_filter_enabled, bool):
            raise ValueError("actor_bc_q_filter_enabled must be a bool.")
        if self.actor_bc_q_filter_margin < 0.0:
            raise ValueError("actor_bc_q_filter_margin must be non-negative.")
        if not isinstance(self.actor_bc_q_filter_online_only, bool):
            raise ValueError("actor_bc_q_filter_online_only must be a bool.")
        if not isinstance(self.actor_bc_q_filter_require_teacher_release, bool):
            raise ValueError("actor_bc_q_filter_require_teacher_release must be a bool.")
        if not isinstance(self.demo_pretrain_enabled, bool):
            raise ValueError("demo_pretrain_enabled must be a bool.")
        if self.demo_collection_env_steps < 0:
            raise ValueError("demo_collection_env_steps must be non-negative.")
        if self.demo_collection_behavior_source not in {"pool_power_mix"}:
            raise ValueError("demo_collection_behavior_source currently only supports 'pool_power_mix'.")
        if not isinstance(self.demo_collection_use_domain_randomization, bool):
            raise ValueError("demo_collection_use_domain_randomization must be a bool.")
        self.demo_collection_network_types = tuple(str(item) for item in self.demo_collection_network_types if str(item))
        if self.demo_collection_runtime not in {"parallel_cpu", "isolated_cpu", "reuse_workers"}:
            raise ValueError(
                "demo_collection_runtime must be one of {'parallel_cpu', 'isolated_cpu', 'reuse_workers'}."
            )
        if self.actor_bc_pretrain_updates < 0:
            raise ValueError("actor_bc_pretrain_updates must be non-negative.")
        if self.critic_pretrain_updates < 0:
            raise ValueError("critic_pretrain_updates must be non-negative.")
        if self.demo_pretrain_batch_size is not None and self.demo_pretrain_batch_size <= 0:
            raise ValueError("demo_pretrain_batch_size must be positive when provided.")
        if self.demo_pretrain_validation_batch_size is not None and self.demo_pretrain_validation_batch_size <= 0:
            raise ValueError("demo_pretrain_validation_batch_size must be positive when provided.")
        if self.demo_validation_fraction < 0.0 or self.demo_validation_fraction >= 1.0:
            raise ValueError("demo_validation_fraction must be in [0, 1).")
        if self.demo_pretrain_eval_interval <= 0:
            raise ValueError("demo_pretrain_eval_interval must be positive.")
        if self.demo_pretrain_patience <= 0:
            raise ValueError("demo_pretrain_patience must be positive.")
        if self.demo_pretrain_min_relative_improvement < 0.0:
            raise ValueError("demo_pretrain_min_relative_improvement must be non-negative.")
        if self.demo_critic_pretrain_target_mode not in {"n_step", "mc"}:
            raise ValueError("demo_critic_pretrain_target_mode must be one of {'n_step', 'mc'}.")
        if self.demo_critic_pretrain_n_step <= 0:
            raise ValueError("demo_critic_pretrain_n_step must be positive.")
        if not isinstance(self.critic_bridge_enabled, bool):
            raise ValueError("critic_bridge_enabled must be a bool.")
        if self.critic_bridge_env_steps < 0:
            raise ValueError("critic_bridge_env_steps must be non-negative.")
        if self.critic_bridge_updates < 0:
            raise ValueError("critic_bridge_updates must be non-negative.")
        if self.critic_bridge_batch_size is not None and self.critic_bridge_batch_size <= 0:
            raise ValueError("critic_bridge_batch_size must be positive when provided.")
        if self.critic_bridge_validation_fraction < 0.0 or self.critic_bridge_validation_fraction >= 1.0:
            raise ValueError("critic_bridge_validation_fraction must be in [0, 1).")
        if self.critic_bridge_eval_interval <= 0:
            raise ValueError("critic_bridge_eval_interval must be positive.")
        if self.critic_bridge_patience <= 0:
            raise ValueError("critic_bridge_patience must be positive.")
        if self.critic_bridge_min_relative_improvement < 0.0:
            raise ValueError("critic_bridge_min_relative_improvement must be non-negative.")
        if self.critic_bridge_behavior_mode not in {"actor_only", "teacher_actor_mix"}:
            raise ValueError("critic_bridge_behavior_mode must be one of {'actor_only', 'teacher_actor_mix'}.")
        if self.critic_bridge_teacher_takeover_prob < 0.0 or self.critic_bridge_teacher_takeover_prob > 1.0:
            raise ValueError("critic_bridge_teacher_takeover_prob must be in [0, 1].")
        if not isinstance(self.critic_bridge_use_curriculum_stage0_distribution, bool):
            raise ValueError("critic_bridge_use_curriculum_stage0_distribution must be a bool.")
        if self.critic_bridge_teacher_return_aux_schedule not in {"fixed", "adaptive"}:
            raise ValueError("critic_bridge_teacher_return_aux_schedule must be one of {'fixed', 'adaptive'}.")
        if not self.critic_bridge_teacher_return_aux_levels:
            raise ValueError("critic_bridge_teacher_return_aux_levels must contain at least one entry.")
        if any(float(level) < 0.0 for level in self.critic_bridge_teacher_return_aux_levels):
            raise ValueError("critic_bridge_teacher_return_aux_levels must be non-negative.")
        for previous, current in zip(
            self.critic_bridge_teacher_return_aux_levels,
            self.critic_bridge_teacher_return_aux_levels[1:],
        ):
            if float(current) > float(previous):
                raise ValueError(
                    "critic_bridge_teacher_return_aux_levels must be non-increasing so the aux weight only decays."
                )
        if self.critic_bridge_teacher_return_aux_required_evals <= 0:
            raise ValueError("critic_bridge_teacher_return_aux_required_evals must be positive.")
        if self.critic_bridge_teacher_return_aux_max_val_ratio < 1.0:
            raise ValueError("critic_bridge_teacher_return_aux_max_val_ratio must be >= 1.0.")
        if self.critic_bridge_teacher_return_aux_max_error_ratio < 0.0:
            raise ValueError("critic_bridge_teacher_return_aux_max_error_ratio must be non-negative.")
        if self.critic_bridge_teacher_return_aux_coef < 0.0:
            raise ValueError("critic_bridge_teacher_return_aux_coef must be non-negative.")
        if not isinstance(self.teacher_takeover_enabled, bool):
            raise ValueError("teacher_takeover_enabled must be a bool.")
        if self.teacher_takeover_behavior_source not in {"pool_power_mix"}:
            raise ValueError("teacher_takeover_behavior_source currently only supports 'pool_power_mix'.")
        if self.teacher_takeover_granularity not in {"per_step", "per_episode"}:
            raise ValueError("teacher_takeover_granularity must be one of {'per_step', 'per_episode'}.")
        if self.teacher_takeover_start_prob < 0.0 or self.teacher_takeover_start_prob > 1.0:
            raise ValueError("teacher_takeover_start_prob must be in [0, 1].")
        if self.teacher_takeover_end_prob < 0.0 or self.teacher_takeover_end_prob > 1.0:
            raise ValueError("teacher_takeover_end_prob must be in [0, 1].")
        if self.teacher_takeover_decay_end_fraction < 0.0 or self.teacher_takeover_decay_end_fraction > 1.0:
            raise ValueError("teacher_takeover_decay_end_fraction must be in [0, 1].")
        if not isinstance(self.adaptive_teacher_release_enabled, bool):
            raise ValueError("adaptive_teacher_release_enabled must be a bool.")
        if self.adaptive_teacher_release_mode not in {"legacy", "eval_cooperation"}:
            raise ValueError("adaptive_teacher_release_mode must be one of {'legacy', 'eval_cooperation'}.")
        if self.adaptive_teacher_release_min_cooperation < 0.0 or self.adaptive_teacher_release_min_cooperation > 1.0:
            raise ValueError("adaptive_teacher_release_min_cooperation must be in [0, 1].")
        if self.adaptive_teacher_release_min_return_ratio < 0.0:
            raise ValueError("adaptive_teacher_release_min_return_ratio must be non-negative.")
        if self.adaptive_teacher_release_max_actor_bc_val_ratio <= 0.0:
            raise ValueError("adaptive_teacher_release_max_actor_bc_val_ratio must be positive.")
        if self.adaptive_teacher_release_max_critic_val_ratio <= 0.0:
            raise ValueError("adaptive_teacher_release_max_critic_val_ratio must be positive.")
        if self.adaptive_teacher_release_required_evals <= 0:
            raise ValueError("adaptive_teacher_release_required_evals must be positive.")
        if self.adaptive_teacher_release_min_criteria <= 0:
            raise ValueError("adaptive_teacher_release_min_criteria must be positive.")
        if not isinstance(self.freeze_actor_q_until_teacher_release, bool):
            raise ValueError("freeze_actor_q_until_teacher_release must be a bool.")
        if self.online_actor_q_coef_initial < 0.0:
            raise ValueError("online_actor_q_coef_initial must be non-negative.")
        if self.online_actor_q_coef_final < 0.0:
            raise ValueError("online_actor_q_coef_final must be non-negative.")
        if self.online_actor_q_coef_ramp_end_fraction < 0.0 or self.online_actor_q_coef_ramp_end_fraction > 1.0:
            raise ValueError("online_actor_q_coef_ramp_end_fraction must be in [0, 1].")
        if self.critic_loss_type not in {"mse", "huber"}:
            raise ValueError("critic_loss_type must be one of {'mse', 'huber'}.")
        if self.critic_huber_delta <= 0.0:
            raise ValueError("critic_huber_delta must be positive.")
        if self.actor_grad_clip_norm is not None and self.actor_grad_clip_norm <= 0.0:
            raise ValueError("actor_grad_clip_norm must be positive when provided.")
        if self.critic_grad_clip_norm is not None and self.critic_grad_clip_norm <= 0.0:
            raise ValueError("critic_grad_clip_norm must be positive when provided.")
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
        if self.replay_collapse_fc_threshold < 0.0 or self.replay_collapse_fc_threshold > 1.0:
            raise ValueError("replay_collapse_fc_threshold must be in [0, 1].")
        if self.replay_max_collapse_sample_ratio < 0.0 or self.replay_max_collapse_sample_ratio > 1.0:
            raise ValueError("replay_max_collapse_sample_ratio must be in [0, 1].")
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
    fixed_graph_bank_enabled: bool = False
    fixed_graph_bank_size_per_type: int = 0
    fixed_graph_bank_seed: int = 0
    fixed_graph_bank_sampling: str = "uniform"
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
        if not isinstance(self.fixed_graph_bank_enabled, bool):
            raise ValueError("fixed_graph_bank_enabled must be a bool.")
        if self.fixed_graph_bank_size_per_type < 0:
            raise ValueError("fixed_graph_bank_size_per_type must be non-negative.")
        if self.fixed_graph_bank_enabled and self.fixed_graph_bank_size_per_type <= 0:
            raise ValueError("fixed_graph_bank_size_per_type must be positive when fixed_graph_bank_enabled is True.")
        if self.fixed_graph_bank_sampling not in {"uniform", "round_robin"}:
            raise ValueError("fixed_graph_bank_sampling must be one of {'uniform', 'round_robin'}.")


@dataclass
class EvalConfig:
    num_episodes: int = 3
    collapse_resource_threshold: float = 1e-6

    def __post_init__(self) -> None:
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")
