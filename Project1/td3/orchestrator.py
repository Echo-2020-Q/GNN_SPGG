from __future__ import annotations

import copy
from dataclasses import asdict, replace
from collections import Counter
from multiprocessing.connection import wait
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from Project1.env import SPGGEnv
from Project1.policies.gnn_rl import GNNAllocationPolicy

from .config import DomainRandomizationConfig, EvalConfig, GraphTD3Config, WorkerConfig
from .critic import GraphActionCritic, GraphActionCriticConfig, TwinCritic
from .data import TensorReplayActionRecord, TensorReplayBatch
from .evaluator import GraphTD3Evaluator
from .exploration import LogitSpaceExplorer
from .learner import GraphTD3Learner
from .replay import ReplayBuffer, split_demo_batch_train_val, split_replay_batch_train_val
from .worker import (
    ParallelRolloutInferenceServer,
    ParallelRolloutWorker,
    RandomizedEnvFactory,
    RolloutResult,
    RolloutWorker,
)


def _resolve_rollout_device_for_worker(rollout_device: str | tuple[str, ...], worker_id: int) -> str:
    if isinstance(rollout_device, tuple):
        return str(rollout_device[worker_id % len(rollout_device)])
    return str(rollout_device)


def _should_use_centralized_rollout_inference(config: GraphTD3Config) -> bool:
    return config.num_workers > 1 and config.rollout_inference_mode == "centralized"


class GraphTD3Trainer:
    """Central learner with real parallel rollout workers.

    The learner and replay buffer stay in the main process. Rollout workers hold actor
    snapshots in separate processes, collect transitions in parallel, and return tensor
    batches that are appended into the central replay buffer before TD3 updates.
    """

    def __init__(
        self,
        env: SPGGEnv,
        policy: GNNAllocationPolicy,
        config: GraphTD3Config,
        eval_env: SPGGEnv | None = None,
        randomization: DomainRandomizationConfig | None = None,
        eval_env_factories: Sequence[RandomizedEnvFactory] | None = None,
        curriculum_stages: Sequence[Mapping[str, Any]] | None = None,
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.replay_buffer = ReplayBuffer(
            config.replay_capacity,
            seed=config.seed,
            replay_strategy=config.replay_strategy,
            topology_names=config.replay_topology_names,
            recent_fraction=config.replay_recent_fraction,
            long_term_fraction=config.replay_long_term_fraction,
            demo_fraction=config.replay_demo_fraction,
            demo_behavior_source=config.replay_demo_behavior_source,
        )

        default_critic_hidden_dim = int(getattr(policy.config, "hidden_dim", 64))
        critic_config = GraphActionCriticConfig(
            state_hidden_dim=int(config.critic_state_hidden_dim or default_critic_hidden_dim),
            action_hidden_dim=int(config.critic_action_hidden_dim or default_critic_hidden_dim),
            pool_hidden_dim=int(config.critic_pool_hidden_dim or default_critic_hidden_dim),
            q_hidden_dim=int(config.critic_q_hidden_dim or default_critic_hidden_dim),
        )
        critics = TwinCritic(
            GraphActionCritic(critic_config),
            GraphActionCritic(critic_config),
        )
        target_critics = TwinCritic(
            GraphActionCritic(critic_config),
            GraphActionCritic(critic_config),
        )

        self.target_policy = copy.deepcopy(policy)
        self.target_explorer = LogitSpaceExplorer()
        self.learner = GraphTD3Learner(
            actor=self.policy,
            critics=critics,
            target_actor=self.target_policy,
            target_critics=target_critics,
            replay_buffer=self.replay_buffer,
            target_explorer=self.target_explorer,
            config=self.config,
        )

        train_factory = RandomizedEnvFactory.from_env(env, randomization=randomization)
        self.train_factory = train_factory
        self.rollout_explorer = LogitSpaceExplorer()
        self.rollout_inference_servers: list[ParallelRolloutInferenceServer] = []
        self.workers = []
        self._rollout_runtime_initialized = False
        if not self._should_delay_rollout_runtime_initialization():
            self._initialize_rollout_runtime()

        evaluator_factories = list(eval_env_factories) if eval_env_factories is not None else [
            RandomizedEnvFactory.from_env(eval_env or env)
        ]
        self.default_eval_env_factories = list(evaluator_factories)
        self.evaluator = GraphTD3Evaluator(
            env_factories=evaluator_factories,
            config=EvalConfig(
                num_episodes=config.eval_episodes,
                collapse_resource_threshold=config.collapse_resource_threshold,
            ),
            device=config.device,
        )
        self.curriculum_stages = [dict(stage) for stage in curriculum_stages] if curriculum_stages is not None else []
        self.active_curriculum_stage_index: int | None = None
        self.completed_updates = 0
        self.global_env_steps = 0
        self.history: list[dict[str, float]] = []
        self._rollout_collect_inflight = False
        self._last_replay_extend_seconds = 0.0
        self.demo_pretrain_completed = False
        self.demo_pretrain_summary: dict[str, float | str | bool | None] | None = None
        self.demo_validation_batch: TensorReplayBatch | None = None
        self.teacher_takeover_release_env_step: int | None = None
        self.teacher_takeover_stable_eval_count = 0
        self.teacher_takeover_soft_release_env_step: int | None = None
        self.teacher_takeover_full_release_env_step: int | None = None
        self.teacher_handoff_stage = 0
        self.teacher_handoff_stable_eval_count = 0
        self.teacher_handoff_regression_eval_count = 0

    def preload_demo_replay(
        self,
        replay_state: ReplayBuffer | Mapping[str, Any],
        summary: Mapping[str, float | str | bool | None] | None = None,
        validation_batch: TensorReplayBatch | None = None,
    ) -> None:
        if isinstance(replay_state, ReplayBuffer):
            self.replay_buffer = replay_state
        else:
            self.replay_buffer.load_state_dict(dict(replay_state))
        self.learner.replay_buffer = self.replay_buffer
        self.demo_pretrain_completed = False
        if summary is not None:
            self.demo_pretrain_summary = dict(summary)
        self.demo_validation_batch = validation_batch.clone() if validation_batch is not None else None

    def close(self) -> None:
        self._shutdown_rollout_runtime()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _format_progress_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(float(seconds))))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return "{0}h {1:02d}m {2:02d}s".format(hours, minutes, secs)
        if minutes > 0:
            return "{0}m {1:02d}s".format(minutes, secs)
        return "{0}s".format(secs)

    @classmethod
    def _print_pretrain_progress(
        cls,
        stage_label: str,
        completed: int,
        total: int,
        started_at: float,
    ) -> None:
        if total <= 0:
            return
        elapsed = float(perf_counter() - started_at)
        progress = min(max(float(completed) / float(total), 0.0), 1.0)
        eta_seconds = None
        if completed > 0 and progress > 0.0:
            eta_seconds = max(0.0, elapsed * (1.0 - progress) / progress)
        eta_text = (
            cls._format_progress_duration(eta_seconds)
            if eta_seconds is not None
            else "unavailable"
        )
        print(
            "Demo Pretrain | {0} progress | {1}/{2} ({3:.1f}%) | ETA={4} | elapsed={5}".format(
                stage_label,
                int(completed),
                int(total),
                progress * 100.0,
                eta_text,
                cls._format_progress_duration(elapsed),
            )
        )

    @staticmethod
    def _progress_interval(total: int) -> int:
        return max(1, int(np.ceil(float(max(1, total)) / 20.0)))

    def _make_split_rng(self) -> np.random.Generator:
        return np.random.default_rng(int(self.config.seed or 0) + 3_000_000)

    @staticmethod
    def _concat_replay_batches(batches: Sequence[TensorReplayBatch]) -> TensorReplayBatch:
        if not batches:
            raise ValueError("batches must contain at least one item.")
        if len(batches) == 1:
            return batches[0].clone()
        first_batch = batches[0]
        return TensorReplayBatch(
            obs={key: torch.cat([batch.obs[key] for batch in batches], dim=0) for key in first_batch.obs},
            action=TensorReplayActionRecord(
                allocation=torch.cat([batch.action.allocation for batch in batches], dim=0),
            ),
            reward=torch.cat([batch.reward for batch in batches], dim=0),
            next_obs={key: torch.cat([batch.next_obs[key] for batch in batches], dim=0) for key in first_batch.next_obs},
            done=torch.cat([batch.done for batch in batches], dim=0),
            is_demo=torch.cat([batch.is_demo for batch in batches], dim=0),
            collapse_flag=torch.cat([batch.collapse_flag for batch in batches], dim=0),
            topology_id=torch.cat([batch.topology_id for batch in batches], dim=0),
            pool_power_demo_flag=torch.cat([batch.pool_power_demo_flag for batch in batches], dim=0),
            demo_return_target=torch.cat([batch.demo_return_target for batch in batches], dim=0),
            demo_return_valid=torch.cat([batch.demo_return_valid for batch in batches], dim=0),
        )

    def _route_demo_batch_to_train_and_val(
        self,
        batch: TensorReplayBatch,
        *,
        split_rng: np.random.Generator,
        validation_batches: list[TensorReplayBatch],
    ) -> int:
        train_batch, val_batch = split_demo_batch_train_val(
            batch,
            validation_fraction=float(self.config.demo_validation_fraction),
            rng=split_rng,
        )
        collected_now = 0
        if train_batch is not None and len(train_batch) > 0:
            self.replay_buffer.extend(train_batch)
            collected_now += len(train_batch)
        if val_batch is not None and len(val_batch) > 0:
            validation_batches.append(val_batch.clone())
            collected_now += len(val_batch)
        return collected_now

    @staticmethod
    def _route_replay_batch_to_train_and_val(
        replay_buffer: ReplayBuffer,
        batch: TensorReplayBatch,
        *,
        validation_fraction: float,
        split_rng: np.random.Generator,
        validation_batches: list[TensorReplayBatch],
    ) -> int:
        train_batch, val_batch = split_replay_batch_train_val(
            batch,
            validation_fraction=float(validation_fraction),
            rng=split_rng,
        )
        collected_now = 0
        if train_batch is not None and len(train_batch) > 0:
            replay_buffer.extend(train_batch)
            collected_now += len(train_batch)
        if val_batch is not None and len(val_batch) > 0:
            validation_batches.append(val_batch.clone())
            collected_now += len(val_batch)
        return collected_now

    def _resolved_demo_validation_batch(self) -> TensorReplayBatch | None:
        if self.demo_validation_batch is not None and len(self.demo_validation_batch) > 0:
            return self.demo_validation_batch.clone()
        exported = self.replay_buffer.export_demo_batch()
        if exported is None or len(exported) <= 0:
            return None
        return exported

    @staticmethod
    def _metric_improved(
        current: float,
        best: float | None,
        *,
        greater_is_better: bool,
        min_relative_improvement: float,
    ) -> bool:
        if best is None:
            return True
        baseline = max(abs(float(best)), 1e-8)
        if greater_is_better:
            return float(current) > float(best) * (1.0 + float(min_relative_improvement))
        return float(current) < float(best) - baseline * float(min_relative_improvement)

    def _current_learner_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.learner.checkpoint_state())

    def _run_demo_pretrain_validation(
        self,
        *,
        include_quick_eval: bool,
    ) -> dict[str, float]:
        validation_batch = self._resolved_demo_validation_batch()
        metrics: dict[str, float] = {}
        if validation_batch is not None and len(validation_batch) > 0:
            metrics.update(self.learner.evaluate_actor_bc_on_demo_batch(validation_batch))
            metrics.update(self.learner.evaluate_critic_on_demo_return_batch(validation_batch))
        else:
            metrics.update(
                {
                    "actor_bc_val_loss": 0.0,
                    "actor_bc_val_num_entries": 0.0,
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
            )
        if include_quick_eval:
            quick_eval_episodes = max(1, min(int(self.config.eval_episodes), 4))
            quick_eval = self.evaluate(num_episodes=quick_eval_episodes)
            metrics["quick_eval_return_mean"] = float(quick_eval.get("return_mean", 0.0))
            metrics["quick_eval_return_per_step_mean"] = float(quick_eval.get("return_per_step_mean", 0.0))
            metrics["quick_eval_cooperation_mean"] = float(quick_eval.get("cooperation_mean", 0.0))
            metrics["quick_eval_gini_mean"] = float(quick_eval.get("gini_mean", 0.0))
            metrics["quick_eval_collapse_rate"] = float(quick_eval.get("collapse_rate", 0.0))
            metrics["quick_eval_num_episodes"] = float(quick_eval_episodes)
        return metrics

    def _run_critic_bridge_validation(
        self,
        validation_batch: TensorReplayBatch | None,
    ) -> dict[str, float]:
        if validation_batch is None or len(validation_batch) <= 0:
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
        return self.learner.evaluate_critic_on_td_batch(validation_batch)

    @staticmethod
    def _teacher_handoff_stage_label(stage: int) -> str:
        return {
            0: "locked",
            1: "soft_release",
            2: "full_handoff",
        }.get(int(stage), str(stage))

    def _evaluate_teacher_release_gate(
        self,
        *,
        online_eval_cooperation_mean: float,
        online_eval_return_mean: float,
        actor_bc_val_loss: float | None,
        critic_val_loss: float | None,
    ) -> tuple[int, int, bool]:
        if str(self.config.adaptive_teacher_release_mode) == "eval_cooperation":
            available = 1
            passed = 1 if float(online_eval_cooperation_mean) >= float(
                self.config.adaptive_teacher_release_min_cooperation
            ) else 0
            return available, passed, passed >= 1

        demo_summary = self.demo_pretrain_summary or {}
        baseline_return = float(demo_summary.get("quick_eval_return_best", 0.0))
        baseline_actor_val = float(demo_summary.get("actor_bc_val_loss_best", 0.0))
        baseline_critic_val = float(demo_summary.get("critic_val_loss_best", 0.0))
        passed = 0
        available = 0

        if baseline_return > 0.0:
            available += 1
            if float(online_eval_return_mean) >= baseline_return * float(self.config.adaptive_teacher_release_min_return_ratio):
                passed += 1
        if actor_bc_val_loss is not None and baseline_actor_val > 1e-8:
            available += 1
            if float(actor_bc_val_loss) <= baseline_actor_val * float(
                self.config.adaptive_teacher_release_max_actor_bc_val_ratio
            ):
                passed += 1
        if critic_val_loss is not None and baseline_critic_val > 1e-8:
            available += 1
            if float(critic_val_loss) <= baseline_critic_val * float(
                self.config.adaptive_teacher_release_max_critic_val_ratio
            ):
                passed += 1

        required_criteria = min(int(self.config.adaptive_teacher_release_min_criteria), max(1, available))
        gate_passed = available > 0 and passed >= required_criteria
        return available, passed, gate_passed

    def _update_adaptive_teacher_release(
        self,
        *,
        online_eval_cooperation_mean: float,
        online_eval_return_mean: float,
        actor_bc_val_loss: float | None,
        critic_val_loss: float | None,
        behavior_frac_actor_logits: float,
    ) -> dict[str, float]:
        metrics = {
            "teacher_release_enabled": 1.0 if bool(self.config.adaptive_teacher_release_enabled) else 0.0,
            "teacher_release_unlocked": 1.0 if self.teacher_takeover_release_env_step is not None else 0.0,
            "teacher_release_stable_eval_count": float(self.teacher_takeover_stable_eval_count),
            "teacher_release_gate_pass_count": 0.0,
            "teacher_release_gate_available_count": 0.0,
            "teacher_release_passed": 0.0,
            "teacher_release_just_unlocked": 0.0,
            "teacher_handoff_stage": float(self.teacher_handoff_stage),
            "teacher_handoff_soft_released": 1.0 if int(self.teacher_handoff_stage) >= 1 else 0.0,
            "teacher_handoff_full_released": 1.0 if int(self.teacher_handoff_stage) >= 2 else 0.0,
            "teacher_handoff_stage_stable_eval_count": float(self.teacher_handoff_stable_eval_count),
            "teacher_handoff_regression_eval_count": float(self.teacher_handoff_regression_eval_count),
            "teacher_handoff_behavior_frac_actor_logits": float(behavior_frac_actor_logits),
            "teacher_handoff_behavior_gate_passed": 0.0,
            "teacher_handoff_rollback_gate_passed": 1.0,
            "teacher_handoff_stage_just_advanced": 0.0,
            "teacher_handoff_stage_just_regressed": 0.0,
            "teacher_handoff_stage_just_changed": 0.0,
        }
        if not bool(self.config.adaptive_teacher_release_enabled):
            return metrics
        if (
            bool(self.config.adaptive_teacher_release_require_warmup_complete)
            and int(self.global_env_steps) < int(self.config.warmup_steps)
        ):
            self.teacher_takeover_stable_eval_count = 0
            self.teacher_handoff_stable_eval_count = 0
            self.teacher_handoff_regression_eval_count = 0
            metrics["teacher_release_stable_eval_count"] = 0.0
            metrics["teacher_handoff_stage_stable_eval_count"] = 0.0
            metrics["teacher_handoff_regression_eval_count"] = 0.0
            return metrics

        available, passed, gate_passed = self._evaluate_teacher_release_gate(
            online_eval_cooperation_mean=online_eval_cooperation_mean,
            online_eval_return_mean=online_eval_return_mean,
            actor_bc_val_loss=actor_bc_val_loss,
            critic_val_loss=critic_val_loss,
        )
        metrics.update(
            {
                "teacher_release_gate_pass_count": float(passed),
                "teacher_release_gate_available_count": float(available),
                "teacher_release_passed": 1.0 if gate_passed else 0.0,
            }
        )

        if self.teacher_takeover_release_env_step is None:
            if gate_passed:
                self.teacher_takeover_stable_eval_count += 1
            else:
                self.teacher_takeover_stable_eval_count = 0

            if self.teacher_takeover_stable_eval_count >= int(self.config.adaptive_teacher_release_required_evals):
                self.teacher_takeover_release_env_step = int(self.global_env_steps)
                self.teacher_takeover_soft_release_env_step = int(self.global_env_steps)
                self.teacher_takeover_full_release_env_step = None
                self.teacher_handoff_stage = 1
                self.teacher_handoff_stable_eval_count = 0
                self.teacher_handoff_regression_eval_count = 0
                if str(self.config.adaptive_teacher_release_mode) == "eval_cooperation":
                    print(
                        "Teacher Release | stage=soft_release | unlocked at t_env={0} | eval_f_c={1:.6f} | threshold={2:.6f}".format(
                            int(self.global_env_steps),
                            float(online_eval_cooperation_mean),
                            float(self.config.adaptive_teacher_release_min_cooperation),
                        )
                    )
                else:
                    print(
                        "Teacher Release | stage=soft_release | unlocked at t_env={0} | eval_return={1:.6f} | actor_bc_val={2} | critic_val={3}".format(
                            int(self.global_env_steps),
                            float(online_eval_return_mean),
                            "None" if actor_bc_val_loss is None else "{0:.6f}".format(float(actor_bc_val_loss)),
                            "None" if critic_val_loss is None else "{0:.6f}".format(float(critic_val_loss)),
                        )
                    )
                metrics["teacher_release_just_unlocked"] = 1.0
                metrics["teacher_handoff_stage_just_advanced"] = 1.0
                metrics["teacher_handoff_stage_just_changed"] = 1.0

            metrics["teacher_release_unlocked"] = 1.0 if self.teacher_takeover_release_env_step is not None else 0.0
            metrics["teacher_release_stable_eval_count"] = float(self.teacher_takeover_stable_eval_count)
            metrics["teacher_handoff_stage"] = float(self.teacher_handoff_stage)
            metrics["teacher_handoff_soft_released"] = 1.0 if int(self.teacher_handoff_stage) >= 1 else 0.0
            metrics["teacher_handoff_regression_eval_count"] = float(self.teacher_handoff_regression_eval_count)
            return metrics

        metrics["teacher_release_unlocked"] = 1.0
        metrics["teacher_release_stable_eval_count"] = float(self.teacher_takeover_stable_eval_count)
        metrics["teacher_handoff_stage"] = float(self.teacher_handoff_stage)
        metrics["teacher_handoff_soft_released"] = 1.0

        behavior_gate_passed = float(behavior_frac_actor_logits) >= float(
            self.config.adaptive_teacher_handoff_min_actor_behavior
        )
        metrics["teacher_handoff_behavior_gate_passed"] = 1.0 if behavior_gate_passed else 0.0
        if int(self.teacher_handoff_stage) < 2:
            self.teacher_handoff_regression_eval_count = 0
            if gate_passed and behavior_gate_passed:
                self.teacher_handoff_stable_eval_count += 1
            else:
                self.teacher_handoff_stable_eval_count = 0

            if self.teacher_handoff_stable_eval_count >= int(self.config.adaptive_teacher_handoff_required_evals):
                self.teacher_handoff_stage = 2
                self.teacher_takeover_full_release_env_step = int(self.global_env_steps)
                self.teacher_handoff_stable_eval_count = 0
                self.teacher_handoff_regression_eval_count = 0
                metrics["teacher_handoff_stage_just_advanced"] = 1.0
                metrics["teacher_handoff_stage_just_changed"] = 1.0
                print(
                    "Teacher Handoff | stage=full_handoff | unlocked at t_env={0} | actor_logits_frac={1:.6f} | eval_f_c={2:.6f} | eval_return={3:.6f}".format(
                        int(self.global_env_steps),
                        float(behavior_frac_actor_logits),
                        float(online_eval_cooperation_mean),
                        float(online_eval_return_mean),
                    )
                )
        else:
            rollback_gate_passed = gate_passed and (
                float(behavior_frac_actor_logits) >= float(self.config.adaptive_teacher_handoff_rollback_min_actor_behavior)
            )
            metrics["teacher_handoff_rollback_gate_passed"] = 1.0 if rollback_gate_passed else 0.0
            self.teacher_handoff_stable_eval_count = 0
            if bool(self.config.adaptive_teacher_handoff_rollback_enabled) and not rollback_gate_passed:
                self.teacher_handoff_regression_eval_count += 1
            else:
                self.teacher_handoff_regression_eval_count = 0

            if (
                bool(self.config.adaptive_teacher_handoff_rollback_enabled)
                and self.teacher_handoff_regression_eval_count
                >= int(self.config.adaptive_teacher_handoff_rollback_required_evals)
            ):
                self.teacher_handoff_stage = 1
                self.teacher_takeover_soft_release_env_step = int(self.global_env_steps)
                self.teacher_takeover_full_release_env_step = None
                self.teacher_handoff_stable_eval_count = 0
                self.teacher_handoff_regression_eval_count = 0
                metrics["teacher_handoff_stage_just_regressed"] = 1.0
                metrics["teacher_handoff_stage_just_changed"] = 1.0
                print(
                    "Teacher Handoff | rollback to soft_release at t_env={0} | actor_logits_frac={1:.6f} | eval_f_c={2:.6f} | eval_return={3:.6f}".format(
                        int(self.global_env_steps),
                        float(behavior_frac_actor_logits),
                        float(online_eval_cooperation_mean),
                        float(online_eval_return_mean),
                    )
                )

        metrics["teacher_handoff_stage"] = float(self.teacher_handoff_stage)
        metrics["teacher_handoff_soft_released"] = 1.0 if int(self.teacher_handoff_stage) >= 1 else 0.0
        metrics["teacher_handoff_full_released"] = 1.0 if int(self.teacher_handoff_stage) >= 2 else 0.0
        metrics["teacher_handoff_stage_stable_eval_count"] = float(self.teacher_handoff_stable_eval_count)
        metrics["teacher_handoff_regression_eval_count"] = float(self.teacher_handoff_regression_eval_count)
        return metrics

    def _shutdown_rollout_runtime(self) -> None:
        for worker in self.workers:
            close_method = getattr(worker, "close", None)
            if callable(close_method):
                close_method()
        self.workers = []
        for inference_server in self.rollout_inference_servers:
            close_method = getattr(inference_server, "close", None)
            if callable(close_method):
                close_method()
        self.rollout_inference_servers = []
        self._rollout_runtime_initialized = False

    def _should_delay_rollout_runtime_initialization(self) -> bool:
        return (
            bool(self.config.demo_pretrain_enabled)
            and str(self.config.demo_collection_runtime) != "reuse_workers"
        )

    def _initialize_rollout_runtime(self) -> None:
        if self._rollout_runtime_initialized:
            return
        self._shutdown_rollout_runtime()
        actor_source = self.learner.actor
        centralized_rollout_inference = _should_use_centralized_rollout_inference(self.config)
        worker_inference_connections: dict[int, Any] = {}
        if centralized_rollout_inference:
            worker_ids_by_device: dict[str, list[int]] = {}
            for worker_id in range(self.config.num_workers):
                worker_device = _resolve_rollout_device_for_worker(self.config.rollout_device, worker_id)
                worker_ids_by_device.setdefault(worker_device, []).append(worker_id)
            for worker_device, worker_ids in worker_ids_by_device.items():
                inference_server = ParallelRolloutInferenceServer(
                    actor=copy.deepcopy(actor_source),
                    train_config=self.config,
                    device=worker_device,
                    num_clients=len(worker_ids),
                )
                self.rollout_inference_servers.append(inference_server)
                for local_index, worker_id in enumerate(worker_ids):
                    worker_inference_connections[worker_id] = inference_server.take_worker_connection(local_index)

        for worker_id in range(self.config.num_workers):
            worker_config = WorkerConfig(
                worker_id=worker_id,
                seed=(self.config.seed or 0) + worker_id,
                rollout_steps_per_sync=self.config.steps_per_update,
                num_envs_per_worker=self.config.num_envs_per_worker,
            )
            worker_device = _resolve_rollout_device_for_worker(self.config.rollout_device, worker_id)
            if self.config.num_workers > 1:
                worker = ParallelRolloutWorker(
                    actor=copy.deepcopy(actor_source),
                    env_factory=self.train_factory,
                    config=worker_config,
                    train_config=self.config,
                    device="cpu" if centralized_rollout_inference else worker_device,
                    inference_connection=worker_inference_connections.get(worker_id),
                )
            else:
                worker = RolloutWorker(
                    actor=copy.deepcopy(actor_source),
                    explorer=self.rollout_explorer,
                    env_factory=self.train_factory,
                    config=worker_config,
                    train_config=self.config,
                    device=worker_device,
                )
            self.workers.append(worker)
        self._rollout_runtime_initialized = True

    def _resolve_curriculum_stage(self, update: int) -> tuple[int, Mapping[str, Any]] | None:
        if not self.curriculum_stages:
            return None

        active_stage_index = 0
        active_stage = self.curriculum_stages[0]
        for stage_index, stage in enumerate(self.curriculum_stages):
            if int(stage["activate_at_update"]) <= int(update):
                active_stage_index = stage_index
                active_stage = stage
            else:
                break
        return active_stage_index, active_stage

    def _activate_curriculum_stage(
        self,
        update: int,
        stage_index: int,
        stage: Mapping[str, Any],
        reset_workers: bool,
    ) -> None:
        self.train_factory.randomization = stage["train_randomization"]
        for worker in self.workers:
            worker.set_env_factory(self.train_factory, reset_environment=reset_workers)

        stage_eval_factories = stage.get("eval_env_factories")
        self.evaluator.env_factories = (
            list(stage_eval_factories)
            if stage_eval_factories is not None
            else list(self.default_eval_env_factories)
        )
        self.active_curriculum_stage_index = stage_index
        print(
            "Curriculum: update={0}, stage={1}, train={2}, train_weights={3}, eval={4}".format(
                update,
                stage.get("label", "stage_{0}".format(stage_index)),
                list(stage.get("train_network_types", ())),
                [float(item) for item in stage.get("train_network_type_weights", ())],
                list(stage.get("eval_network_types", ())),
            )
        )

    def _apply_curriculum_stage(self, update: int) -> None:
        resolved_stage = self._resolve_curriculum_stage(update)
        if resolved_stage is None:
            return

        stage_index, stage = resolved_stage
        if self.active_curriculum_stage_index == stage_index:
            return
        self._activate_curriculum_stage(update, stage_index, stage, reset_workers=True)

    def _sync_rollout_inference_servers(self, actor_state_dict: Mapping[str, Any]) -> None:
        for inference_server in self.rollout_inference_servers:
            inference_server.sync_actor(dict(actor_state_dict))

    def build_checkpoint(
        self,
        update: int,
        metrics: Mapping[str, float] | None = None,
        checkpoint_mode: str = "full_resume",
    ) -> dict[str, object]:
        if checkpoint_mode not in {"lightweight", "full_resume"}:
            raise ValueError("checkpoint_mode must be one of {'lightweight', 'full_resume'}.")

        payload: dict[str, object] = {
            "update": int(update),
            "metrics": dict(metrics) if metrics is not None else {},
            "checkpoint_mode": checkpoint_mode,
            "trainer_config": asdict(self.config),
            "policy_config": asdict(self.policy.config),
            "learner_state": self.learner.checkpoint_state(),
            "completed_updates": int(self.completed_updates),
            "global_env_steps": int(self.global_env_steps),
            "active_curriculum_stage_index": self.active_curriculum_stage_index,
            "history": [dict(item) for item in self.history],
            "demo_pretrain_completed": bool(self.demo_pretrain_completed),
            "demo_pretrain_summary": None if self.demo_pretrain_summary is None else dict(self.demo_pretrain_summary),
            "teacher_takeover_release_env_step": (
                None if self.teacher_takeover_release_env_step is None else int(self.teacher_takeover_release_env_step)
            ),
            "teacher_takeover_stable_eval_count": int(self.teacher_takeover_stable_eval_count),
            "teacher_takeover_soft_release_env_step": (
                None
                if self.teacher_takeover_soft_release_env_step is None
                else int(self.teacher_takeover_soft_release_env_step)
            ),
            "teacher_takeover_full_release_env_step": (
                None
                if self.teacher_takeover_full_release_env_step is None
                else int(self.teacher_takeover_full_release_env_step)
            ),
            "teacher_handoff_stage": int(self.teacher_handoff_stage),
            "teacher_handoff_stable_eval_count": int(self.teacher_handoff_stable_eval_count),
            "teacher_handoff_regression_eval_count": int(self.teacher_handoff_regression_eval_count),
        }
        if checkpoint_mode == "full_resume":
            payload["replay_buffer_state"] = self.replay_buffer.state_dict()
            payload["worker_states"] = [worker.state_dict() for worker in self.workers]
        return payload

    def load_checkpoint(self, checkpoint: Mapping[str, Any]) -> str:
        self.learner.load_checkpoint_state(dict(checkpoint["learner_state"]))
        self.completed_updates = int(checkpoint.get("completed_updates", checkpoint.get("update", 0)))
        self.global_env_steps = int(checkpoint.get("global_env_steps", 0))
        self.history = [dict(item) for item in checkpoint.get("history", [])]
        self.active_curriculum_stage_index = checkpoint.get("active_curriculum_stage_index")
        self.demo_pretrain_completed = bool(checkpoint.get("demo_pretrain_completed", False))
        demo_pretrain_summary = checkpoint.get("demo_pretrain_summary")
        self.demo_pretrain_summary = None if demo_pretrain_summary is None else dict(demo_pretrain_summary)
        release_env_step = checkpoint.get("teacher_takeover_release_env_step")
        self.teacher_takeover_release_env_step = None if release_env_step is None else int(release_env_step)
        self.teacher_takeover_stable_eval_count = int(checkpoint.get("teacher_takeover_stable_eval_count", 0))
        soft_release_env_step = checkpoint.get("teacher_takeover_soft_release_env_step")
        self.teacher_takeover_soft_release_env_step = (
            None if soft_release_env_step is None else int(soft_release_env_step)
        )
        full_release_env_step = checkpoint.get("teacher_takeover_full_release_env_step")
        self.teacher_takeover_full_release_env_step = (
            None if full_release_env_step is None else int(full_release_env_step)
        )
        self.teacher_handoff_stage = int(
            checkpoint.get(
                "teacher_handoff_stage",
                1 if self.teacher_takeover_release_env_step is not None else 0,
            )
        )
        self.teacher_handoff_stable_eval_count = int(checkpoint.get("teacher_handoff_stable_eval_count", 0))
        self.teacher_handoff_regression_eval_count = int(checkpoint.get("teacher_handoff_regression_eval_count", 0))
        checkpoint_mode = str(
            checkpoint.get(
                "checkpoint_mode",
                "full_resume" if ("replay_buffer_state" in checkpoint and "worker_states" in checkpoint) else "lightweight",
            )
        )

        if checkpoint_mode == "full_resume":
            self.replay_buffer.load_state_dict(dict(checkpoint["replay_buffer_state"]))

            worker_states = list(checkpoint["worker_states"])
            if len(worker_states) != len(self.workers):
                raise ValueError(
                    "Checkpoint worker count {0} does not match current trainer worker count {1}.".format(
                        len(worker_states),
                        len(self.workers),
                    )
                )
            for worker, worker_state in zip(self.workers, worker_states):
                worker.load_state_dict(dict(worker_state))
        else:
            actor_state_dict, actor_version = self.learner.publish_actor_state()
            self._sync_rollout_inference_servers(actor_state_dict)
            for worker in self.workers:
                worker.sync_actor(
                    actor_state_dict,
                    version=actor_version,
                    teacher_takeover_release_env_step=self.teacher_takeover_release_env_step,
                    teacher_takeover_soft_release_env_step=self.teacher_takeover_soft_release_env_step,
                    teacher_takeover_full_release_env_step=self.teacher_takeover_full_release_env_step,
                    teacher_handoff_stage=self.teacher_handoff_stage,
                )

        actor_state_dict, _ = self.learner.publish_actor_state()
        self._sync_rollout_inference_servers(actor_state_dict)

        resolved_stage = self._resolve_curriculum_stage(max(1, self.completed_updates + 1))
        if resolved_stage is not None:
            stage_index, stage = resolved_stage
            self._activate_curriculum_stage(max(1, self.completed_updates + 1), stage_index, stage, reset_workers=False)

        return checkpoint_mode

    def _global_warmup_allocations(self) -> list[int]:
        remaining_global_warmup = max(0, int(self.config.warmup_steps) - int(self.global_env_steps))
        if remaining_global_warmup <= 0:
            return [0 for _ in self.workers]

        allocations: list[int] = []
        remaining_workers = len(self.workers)
        for worker in self.workers:
            per_worker_steps = int(worker.config.rollout_steps_per_sync)
            if remaining_workers <= 1:
                allocation = min(per_worker_steps, remaining_global_warmup)
            else:
                fair_share = int(np.ceil(float(remaining_global_warmup) / float(remaining_workers)))
                allocation = min(per_worker_steps, fair_share, remaining_global_warmup)
            allocations.append(allocation)
            remaining_global_warmup -= allocation
            remaining_workers -= 1
        return allocations

    def _global_step_allocations(self, total_steps: int) -> list[int]:
        return self._global_step_allocations_for_workers(self.workers, total_steps)

    @staticmethod
    def _global_step_allocations_for_workers(workers: Sequence[Any], total_steps: int) -> list[int]:
        remaining_steps = max(0, int(total_steps))
        if remaining_steps <= 0:
            return [0 for _ in workers]

        allocations: list[int] = []
        remaining_workers = len(workers)
        for worker in workers:
            per_worker_steps = int(worker.config.rollout_steps_per_sync)
            if remaining_workers <= 1:
                allocation = min(per_worker_steps, remaining_steps)
            else:
                fair_share = int(np.ceil(float(remaining_steps) / float(remaining_workers)))
                allocation = min(per_worker_steps, fair_share, remaining_steps)
            allocations.append(allocation)
            remaining_steps -= allocation
            remaining_workers -= 1
        return allocations

    def _collect_rollouts_with_allocations(
        self,
        step_allocations: Sequence[int],
        *,
        warmup_allocations: Sequence[int] | None = None,
        forced_behavior_source: str | None = None,
        mark_as_demo: bool | None = None,
        count_env_steps: bool = True,
        demo_return_target_mode: str | None = None,
        demo_return_n_step: int | None = None,
        teacher_takeover_override_prob: float | None = None,
        extend_to_replay: bool = True,
    ) -> list[RolloutResult]:
        return self._collect_rollouts_with_allocations_on_workers(
            self.workers,
            step_allocations,
            warmup_allocations=warmup_allocations,
            forced_behavior_source=forced_behavior_source,
            mark_as_demo=mark_as_demo,
            count_env_steps=count_env_steps,
            demo_return_target_mode=demo_return_target_mode,
            demo_return_n_step=demo_return_n_step,
            teacher_takeover_override_prob=teacher_takeover_override_prob,
            extend_to_replay=extend_to_replay,
        )

    def _collect_rollouts_with_allocations_on_workers(
        self,
        workers: Sequence[Any],
        step_allocations: Sequence[int],
        *,
        warmup_allocations: Sequence[int] | None = None,
        forced_behavior_source: str | None = None,
        mark_as_demo: bool | None = None,
        count_env_steps: bool = True,
        demo_return_target_mode: str | None = None,
        demo_return_n_step: int | None = None,
        teacher_takeover_override_prob: float | None = None,
        extend_to_replay: bool = True,
    ) -> list[RolloutResult]:
        if len(step_allocations) != len(workers):
            raise ValueError("step_allocations must align with workers.")
        if warmup_allocations is None:
            warmup_allocations = [0 for _ in workers]
        if len(warmup_allocations) != len(workers):
            raise ValueError("warmup_allocations must align with workers.")

        positive_requests = [
            (worker, int(num_steps), int(warmup_steps))
            for worker, num_steps, warmup_steps in zip(workers, step_allocations, warmup_allocations)
            if int(num_steps) > 0
        ]
        if not positive_requests:
            return []

        if len(workers) <= 1:
            worker, num_steps, warmup_steps = positive_requests[0]
            rollout_results = [
                worker.collect(
                    num_steps=num_steps,
                    global_warmup_steps=warmup_steps,
                    forced_behavior_source=forced_behavior_source,
                    mark_as_demo=mark_as_demo,
                    count_env_steps=count_env_steps,
                    global_env_start_step=int(self.global_env_steps),
                    demo_return_target_mode=demo_return_target_mode,
                    demo_return_n_step=demo_return_n_step,
                    teacher_takeover_override_prob=teacher_takeover_override_prob,
                )
            ]
        else:
            rollout_results: list[RolloutResult] = []
            started_workers: list[Any] = []
            pending_workers: dict[Any, Any] = {}
            try:
                for worker, num_steps, warmup_steps in positive_requests:
                    worker.start_collect(
                        num_steps=num_steps,
                        global_warmup_steps=warmup_steps,
                        forced_behavior_source=forced_behavior_source,
                        mark_as_demo=mark_as_demo,
                        count_env_steps=count_env_steps,
                        global_env_start_step=int(self.global_env_steps),
                        demo_return_target_mode=demo_return_target_mode,
                        demo_return_n_step=demo_return_n_step,
                        teacher_takeover_override_prob=teacher_takeover_override_prob,
                    )
                    started_workers.append(worker)
                    pending_workers[worker.connection] = worker

                while pending_workers:
                    ready_connections = wait(
                        list(pending_workers.keys()),
                        timeout=float(self.config.worker_rpc_timeout_seconds),
                    )
                    if not ready_connections:
                        raise TimeoutError(
                            "Timed out waiting for rollout workers: {0}".format(
                                [worker.config.worker_id for worker in pending_workers.values()]
                            )
                        )
                    for ready_connection in ready_connections:
                        worker = pending_workers.pop(ready_connection)
                        rollout_results.append(worker.finish_collect_ready())
            except Exception:
                for worker in started_workers:
                    if getattr(worker, "_collect_inflight", False):
                        try:
                            worker.finish_collect()
                        except Exception:
                            pass
                raise

        try:
            if extend_to_replay:
                replay_extend_start = perf_counter()
                for result in rollout_results:
                    self.replay_buffer.extend(result.replay_batch)
                    if count_env_steps:
                        self.global_env_steps += len(result.replay_batch)
                self._last_replay_extend_seconds = float(perf_counter() - replay_extend_start)
            return rollout_results
        finally:
            if extend_to_replay:
                for result in rollout_results:
                    result.release_shared_memory()

    def _build_demo_collection_factory(self) -> RandomizedEnvFactory:
        base_randomization = self.train_factory.randomization
        if not bool(self.config.demo_collection_use_domain_randomization) or not bool(base_randomization.enabled):
            demo_randomization = replace(base_randomization, enabled=False)
            return RandomizedEnvFactory(
                self.train_factory.base_config,
                self.train_factory.base_graph,
                randomization=demo_randomization,
            )

        configured_network_types = tuple(str(item) for item in self.config.demo_collection_network_types if str(item))
        if configured_network_types:
            supported_types = set(str(item) for item in base_randomization.network_types)
            selected_network_types = tuple(item for item in configured_network_types if item in supported_types)
            if not selected_network_types:
                raise ValueError("demo_collection_network_types must overlap domain_randomization.network_types.")
        else:
            selected_network_types = tuple(str(item) for item in base_randomization.network_types)

        selected_weights = None
        if base_randomization.network_type_weights is not None:
            weight_by_type = {
                str(network_type): float(weight)
                for network_type, weight in zip(base_randomization.network_types, base_randomization.network_type_weights)
            }
            selected_weights = tuple(weight_by_type[item] for item in selected_network_types)

        demo_randomization = replace(
            base_randomization,
            enabled=True,
            network_types=selected_network_types,
            network_type_weights=selected_weights,
        )
        return RandomizedEnvFactory(
            self.train_factory.base_config,
            self.train_factory.base_graph,
            randomization=demo_randomization,
        )

    def _build_isolated_demo_collection_worker(self, demo_factory: RandomizedEnvFactory) -> RolloutWorker:
        total_parallel_envs = max(1, int(self.config.num_workers)) * max(1, int(self.config.num_envs_per_worker))
        total_parallel_steps = max(1, int(self.config.num_workers)) * max(1, int(self.config.steps_per_update))
        worker_config = WorkerConfig(
            worker_id=-1,
            seed=int(self.config.seed or 0) + 1_000_000,
            rollout_steps_per_sync=total_parallel_steps,
            num_envs_per_worker=total_parallel_envs,
        )
        return RolloutWorker(
            actor=GNNAllocationPolicy(copy.deepcopy(self.policy.config)),
            explorer=LogitSpaceExplorer(),
            env_factory=demo_factory,
            config=worker_config,
            train_config=self.config,
            device="cpu",
        )

    def _build_parallel_demo_collection_workers(self, demo_factory: RandomizedEnvFactory) -> list[ParallelRolloutWorker]:
        demo_workers: list[ParallelRolloutWorker] = []
        for worker_id in range(int(self.config.num_workers)):
            worker_config = WorkerConfig(
                worker_id=worker_id,
                seed=(int(self.config.seed or 0) + 2_000_000 + worker_id),
                rollout_steps_per_sync=int(self.config.steps_per_update),
                num_envs_per_worker=int(self.config.num_envs_per_worker),
            )
            demo_workers.append(
                ParallelRolloutWorker(
                    actor=GNNAllocationPolicy(copy.deepcopy(self.policy.config)),
                    env_factory=demo_factory,
                    config=worker_config,
                    train_config=self.config,
                    device="cpu",
                )
            )
        return demo_workers

    def _collect_demo_rollouts_with_isolated_worker(
        self,
        *,
        demo_factory: RandomizedEnvFactory,
        total_steps: int,
        demo_batches_to_save: list[Any],
        demo_return_targets: list[np.ndarray],
        validation_batches: list[TensorReplayBatch],
        split_rng: np.random.Generator,
    ) -> None:
        demo_worker = self._build_isolated_demo_collection_worker(demo_factory)
        remaining_steps = max(0, int(total_steps))
        started_at = perf_counter()
        log_interval = self._progress_interval(remaining_steps)
        next_log_at = min(int(total_steps), log_interval)
        last_logged_completed = 0
        while remaining_steps > 0:
            batch_steps = min(int(demo_worker.config.rollout_steps_per_sync), remaining_steps)
            result = demo_worker.collect(
                num_steps=batch_steps,
                global_warmup_steps=0,
                forced_behavior_source=str(self.config.demo_collection_behavior_source),
                mark_as_demo=True,
                count_env_steps=False,
                global_env_start_step=0,
                demo_return_target_mode=str(self.config.demo_critic_pretrain_target_mode),
                demo_return_n_step=int(self.config.demo_critic_pretrain_n_step),
            )
            try:
                collected_now = len(result.replay_batch)
                if collected_now <= 0:
                    raise RuntimeError("Demo collection produced zero transitions.")
                replay_extend_start = perf_counter()
                collected_now = self._route_demo_batch_to_train_and_val(
                    result.replay_batch,
                    split_rng=split_rng,
                    validation_batches=validation_batches,
                )
                self._last_replay_extend_seconds = float(perf_counter() - replay_extend_start)
                if self.config.demo_dataset_save_path:
                    demo_batches_to_save.append(result.replay_batch.clone())
                valid_mask = result.replay_batch.demo_return_valid.detach().cpu().numpy().astype(np.bool_, copy=False)
                if np.any(valid_mask):
                    demo_return_targets.append(
                        result.replay_batch.demo_return_target.detach().cpu().numpy().astype(np.float32, copy=False)[valid_mask]
                    )
                remaining_steps -= collected_now
                completed_steps = int(total_steps) - int(remaining_steps)
                if completed_steps >= next_log_at:
                    self._print_pretrain_progress("collection", completed_steps, int(total_steps), started_at)
                    last_logged_completed = completed_steps
                    next_log_at += log_interval
            finally:
                result.release_shared_memory()
        if last_logged_completed < int(total_steps):
            self._print_pretrain_progress("collection", int(total_steps), int(total_steps), started_at)

    def _collect_demo_rollouts_with_parallel_workers(
        self,
        *,
        demo_factory: RandomizedEnvFactory,
        total_steps: int,
        demo_batches_to_save: list[Any],
        demo_return_targets: list[np.ndarray],
        validation_batches: list[TensorReplayBatch],
        split_rng: np.random.Generator,
    ) -> None:
        demo_workers = self._build_parallel_demo_collection_workers(demo_factory)
        started_at = perf_counter()
        log_interval = self._progress_interval(total_steps)
        next_log_at = min(int(total_steps), log_interval)
        last_logged_completed = 0
        try:
            remaining_steps = max(0, int(total_steps))
            while remaining_steps > 0:
                step_allocations = self._global_step_allocations_for_workers(demo_workers, remaining_steps)
                if sum(step_allocations) <= 0:
                    break
                rollout_results = self._collect_rollouts_with_allocations_on_workers(
                    demo_workers,
                    step_allocations,
                    forced_behavior_source=str(self.config.demo_collection_behavior_source),
                    mark_as_demo=True,
                    count_env_steps=False,
                    demo_return_target_mode=str(self.config.demo_critic_pretrain_target_mode),
                    demo_return_n_step=int(self.config.demo_critic_pretrain_n_step),
                )
                collected_now = sum(len(result.replay_batch) for result in rollout_results)
                if collected_now <= 0:
                    raise RuntimeError("Demo collection produced zero transitions.")
                if self.config.demo_dataset_save_path:
                    demo_batches_to_save.extend(result.replay_batch.clone() for result in rollout_results)
                for result in rollout_results:
                    self._route_demo_batch_to_train_and_val(
                        result.replay_batch,
                        split_rng=split_rng,
                        validation_batches=validation_batches,
                    )
                    valid_mask = result.replay_batch.demo_return_valid.detach().cpu().numpy().astype(np.bool_, copy=False)
                    if np.any(valid_mask):
                        demo_return_targets.append(
                            result.replay_batch.demo_return_target.detach().cpu().numpy().astype(np.float32, copy=False)[valid_mask]
                        )
                remaining_steps -= collected_now
                completed_steps = int(total_steps) - int(remaining_steps)
                if completed_steps >= next_log_at:
                    self._print_pretrain_progress("collection", completed_steps, int(total_steps), started_at)
                    last_logged_completed = completed_steps
                    next_log_at += log_interval
        finally:
            for worker in demo_workers:
                worker.close()
        if last_logged_completed < int(total_steps):
            self._print_pretrain_progress("collection", int(total_steps), int(total_steps), started_at)

    def _build_critic_bridge_collection_factory(self) -> RandomizedEnvFactory:
        if (
            bool(self.config.critic_bridge_use_curriculum_stage0_distribution)
            and self.curriculum_stages
            and self.curriculum_stages[0].get("train_randomization") is not None
        ):
            stage_zero_randomization = self.curriculum_stages[0]["train_randomization"]
            return RandomizedEnvFactory.from_env(self.env, randomization=stage_zero_randomization)
        return self.train_factory

    def _build_critic_bridge_replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(
            self.config.replay_capacity,
            seed=(int(self.config.seed or 0) + 4_000_000),
            replay_strategy=self.config.replay_strategy,
            topology_names=self.config.replay_topology_names,
            recent_fraction=self.config.replay_recent_fraction,
            long_term_fraction=self.config.replay_long_term_fraction,
            demo_fraction=self.config.replay_demo_fraction,
            demo_behavior_source=self.config.replay_demo_behavior_source,
        )

    def _collect_critic_bridge_rollouts(
        self,
        bridge_replay_buffer: ReplayBuffer,
    ) -> tuple[TensorReplayBatch | None, dict[str, float]]:
        bridge_factory = self._build_critic_bridge_collection_factory()
        if not self._rollout_runtime_initialized:
            self._initialize_rollout_runtime()

        self._broadcast_actor_state_to_rollout_runtime()
        for worker in self.workers:
            worker.set_env_factory(bridge_factory, reset_environment=True)

        total_steps = max(0, int(self.config.critic_bridge_env_steps))
        split_rng = np.random.default_rng(int(self.config.seed or 0) + 5_000_000)
        validation_batches: list[TensorReplayBatch] = []
        started_at = perf_counter()
        log_interval = self._progress_interval(total_steps)
        next_log_at = min(total_steps, log_interval)
        last_logged_completed = 0
        remaining_steps = total_steps
        behavior_mode = str(self.config.critic_bridge_behavior_mode)
        teacher_takeover_override_prob = (
            0.0
            if behavior_mode == "actor_only"
            else float(self.config.critic_bridge_teacher_takeover_prob)
        )

        try:
            while remaining_steps > 0:
                step_allocations = self._global_step_allocations(remaining_steps)
                if sum(step_allocations) <= 0:
                    break
                rollout_results = self._collect_rollouts_with_allocations_on_workers(
                    self.workers,
                    step_allocations,
                    warmup_allocations=[0 for _ in self.workers],
                    count_env_steps=False,
                    teacher_takeover_override_prob=teacher_takeover_override_prob,
                    extend_to_replay=False,
                )
                collected_now = 0
                try:
                    for result in rollout_results:
                        collected_now += self._route_replay_batch_to_train_and_val(
                            bridge_replay_buffer,
                            result.replay_batch,
                            validation_fraction=float(self.config.critic_bridge_validation_fraction),
                            split_rng=split_rng,
                            validation_batches=validation_batches,
                        )
                finally:
                    for result in rollout_results:
                        result.release_shared_memory()
                if collected_now <= 0:
                    raise RuntimeError("Critic bridge collection produced zero transitions.")
                remaining_steps -= collected_now
                completed_steps = total_steps - remaining_steps
                if completed_steps >= next_log_at:
                    self._print_pretrain_progress("critic_bridge_collection", completed_steps, total_steps, started_at)
                    last_logged_completed = completed_steps
                    next_log_at += log_interval
        finally:
            for worker in self.workers:
                worker.set_env_factory(self.train_factory, reset_environment=True)

        if total_steps > 0 and last_logged_completed < total_steps:
            self._print_pretrain_progress("critic_bridge_collection", total_steps, total_steps, started_at)

        validation_batch = self._concat_replay_batches(validation_batches) if validation_batches else None
        summary = {
            "critic_bridge_env_steps": float(total_steps),
            "critic_bridge_replay_size_after_collection": float(
                len(bridge_replay_buffer) + (len(validation_batch) if validation_batch is not None else 0)
            ),
            "critic_bridge_train_replay_size_after_split": float(len(bridge_replay_buffer)),
            "critic_bridge_val_replay_size_after_split": float(len(validation_batch) if validation_batch is not None else 0),
            "seconds_critic_bridge_collection": float(perf_counter() - started_at),
        }
        return validation_batch, summary

    def _save_demo_dataset(self, replay_batches: Sequence[Any]) -> str | None:
        save_path = self.config.demo_dataset_save_path
        if not save_path:
            return None

        dataset_path = Path(str(save_path)).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = Path.cwd() / dataset_path
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source": "pool_power_mix_demo_collection",
                "num_batches": int(len(replay_batches)),
                "replay_batches": [batch.clone() for batch in replay_batches],
                "critic_target_mode": str(self.config.demo_critic_pretrain_target_mode),
                "critic_n_step": int(self.config.demo_critic_pretrain_n_step),
            },
            dataset_path,
        )
        return str(dataset_path)

    def _run_demo_pretrain(self) -> dict[str, float | str | bool | None]:
        summary: dict[str, float | str | bool | None] = {
            "enabled": bool(self.config.demo_pretrain_enabled),
            "demo_collection_env_steps": float(self.config.demo_collection_env_steps),
            "demo_replay_size_after_collection": float(self.replay_buffer.demo_size()),
            "demo_train_replay_size_after_split": float(self.replay_buffer.demo_size()),
            "demo_val_replay_size_after_split": float(
                len(self.demo_validation_batch) if self.demo_validation_batch is not None else 0
            ),
            "demo_validation_fraction": float(self.config.demo_validation_fraction),
            "demo_pretrain_eval_interval": float(self.config.demo_pretrain_eval_interval),
            "demo_pretrain_patience": float(self.config.demo_pretrain_patience),
            "demo_pretrain_min_relative_improvement": float(self.config.demo_pretrain_min_relative_improvement),
            "actor_bc_updates": 0.0,
            "critic_pretrain_updates": 0.0,
            "critic_bridge_env_steps": 0.0,
            "critic_bridge_replay_size_after_collection": 0.0,
            "critic_bridge_train_replay_size_after_split": 0.0,
            "critic_bridge_val_replay_size_after_split": 0.0,
            "critic_bridge_updates": 0.0,
            "actor_bc_loss_last": 0.0,
            "critic_loss_last": 0.0,
            "critic_bridge_loss_last": 0.0,
            "critic_bridge_teacher_aux_loss_last": 0.0,
            "critic_bridge_teacher_aux_coef": (
                float(self.config.critic_bridge_teacher_return_aux_levels[0])
                if str(self.config.critic_bridge_teacher_return_aux_schedule) == "adaptive"
                else float(self.config.critic_bridge_teacher_return_aux_coef)
            ),
            "critic_bridge_teacher_aux_level_index": 0.0,
            "critic_bridge_teacher_aux_stable_eval_count": 0.0,
            "critic_bridge_teacher_aux_error_ratio": 0.0,
            "critic_bridge_teacher_aux_reduction_count": 0.0,
            "actor_bc_val_loss_last": 0.0,
            "actor_bc_val_loss_best": 0.0,
            "critic_val_loss_last": 0.0,
            "critic_val_loss_best": 0.0,
            "critic_bridge_val_loss_last": 0.0,
            "critic_bridge_val_loss_best": 0.0,
            "quick_eval_return_last": 0.0,
            "quick_eval_return_best": 0.0,
            "quick_eval_return_per_step_last": 0.0,
            "quick_eval_return_per_step_best": 0.0,
            "actor_bc_eval_count": 0.0,
            "critic_eval_count": 0.0,
            "critic_bridge_eval_count": 0.0,
            "actor_bc_early_stopped": False,
            "critic_pretrain_early_stopped": False,
            "critic_bridge_early_stopped": False,
            "critic_q_pred_mean": 0.0,
            "critic_q_pred_std": 0.0,
            "critic_target_mean": 0.0,
            "critic_target_std": 0.0,
            "critic_error_mean": 0.0,
            "critic_error_std": 0.0,
            "critic_bridge_q_pred_mean": 0.0,
            "critic_bridge_q_pred_std": 0.0,
            "critic_bridge_target_mean": 0.0,
            "critic_bridge_target_std": 0.0,
            "critic_bridge_error_mean": 0.0,
            "critic_bridge_error_std": 0.0,
            "seconds_collection": 0.0,
            "seconds_actor_bc": 0.0,
            "seconds_critic": 0.0,
            "seconds_critic_bridge_collection": 0.0,
            "seconds_critic_bridge": 0.0,
            "dataset_path": None,
            "behavior_source": str(self.config.demo_collection_behavior_source),
            "critic_target_mode": str(self.config.demo_critic_pretrain_target_mode),
            "critic_bridge_teacher_aux_schedule": str(self.config.critic_bridge_teacher_return_aux_schedule),
            "demo_return_target_mean": 0.0,
            "demo_return_target_std": 0.0,
        }
        if self.demo_pretrain_summary is not None:
            summary.update(dict(self.demo_pretrain_summary))
            summary["enabled"] = bool(self.config.demo_pretrain_enabled)
        summary["demo_train_replay_size_after_split"] = float(self.replay_buffer.demo_size())
        summary["demo_val_replay_size_after_split"] = float(
            len(self.demo_validation_batch) if self.demo_validation_batch is not None else 0
        )
        summary["demo_replay_size_after_collection"] = (
            float(summary["demo_train_replay_size_after_split"])
            + float(summary["demo_val_replay_size_after_split"])
        )
        if not bool(self.config.demo_pretrain_enabled):
            self.demo_pretrain_completed = False
            self.demo_pretrain_summary = dict(summary)
            return dict(summary)

        demo_batches_to_save: list[Any] = []
        demo_return_targets: list[np.ndarray] = []
        validation_batches: list[TensorReplayBatch] = []
        split_rng = self._make_split_rng()
        demo_collection_steps = max(0, int(self.config.demo_collection_env_steps))
        if demo_collection_steps > 0:
            print(
                "Demo Pretrain | collection start | env_steps={0} | behavior={1}".format(
                    demo_collection_steps,
                    self.config.demo_collection_behavior_source,
                )
            )
            demo_collection_start = perf_counter()
            demo_factory = self._build_demo_collection_factory()
            demo_collection_runtime = str(self.config.demo_collection_runtime)
            if demo_collection_runtime == "reuse_workers":
                if not self._rollout_runtime_initialized:
                    print("Demo Pretrain | initializing rollout workers for reuse_workers collection")
                    self._initialize_rollout_runtime()
                print("Demo Pretrain | runtime=reuse_workers")
                for worker in self.workers:
                    worker.set_env_factory(demo_factory, reset_environment=True)
                try:
                    remaining_steps = int(demo_collection_steps)
                    while remaining_steps > 0:
                        step_allocations = self._global_step_allocations(remaining_steps)
                        if sum(step_allocations) <= 0:
                            break
                        rollout_results = self._collect_rollouts_with_allocations(
                            step_allocations,
                            forced_behavior_source=str(self.config.demo_collection_behavior_source),
                            mark_as_demo=True,
                            count_env_steps=False,
                            demo_return_target_mode=str(self.config.demo_critic_pretrain_target_mode),
                            demo_return_n_step=int(self.config.demo_critic_pretrain_n_step),
                        )
                        collected_now = sum(len(result.replay_batch) for result in rollout_results)
                        if collected_now <= 0:
                            raise RuntimeError("Demo collection produced zero transitions.")
                        if self.config.demo_dataset_save_path:
                            demo_batches_to_save.extend(result.replay_batch.clone() for result in rollout_results)
                        for result in rollout_results:
                            self._route_demo_batch_to_train_and_val(
                                result.replay_batch,
                                split_rng=split_rng,
                                validation_batches=validation_batches,
                            )
                            valid_mask = result.replay_batch.demo_return_valid.detach().cpu().numpy().astype(
                                np.bool_, copy=False
                            )
                            if np.any(valid_mask):
                                demo_return_targets.append(
                                    result.replay_batch.demo_return_target.detach().cpu().numpy().astype(
                                        np.float32, copy=False
                                    )[valid_mask]
                                )
                        remaining_steps -= collected_now
                finally:
                    for worker in self.workers:
                        worker.set_env_factory(self.train_factory, reset_environment=True)
            elif demo_collection_runtime == "parallel_cpu" and int(self.config.num_workers) > 1:
                print(
                    "Demo Pretrain | runtime=parallel_cpu | envs={0} | steps_per_sync={1}".format(
                        max(1, int(self.config.num_workers)) * max(1, int(self.config.num_envs_per_worker)),
                        max(1, int(self.config.steps_per_update)),
                    )
                )
                had_active_rollout_runtime = bool(self._rollout_runtime_initialized)
                if had_active_rollout_runtime:
                    print("Demo Pretrain | suspending online rollout workers during parallel_cpu collection")
                    self._shutdown_rollout_runtime()
                try:
                    self._collect_demo_rollouts_with_parallel_workers(
                        demo_factory=demo_factory,
                        total_steps=demo_collection_steps,
                        demo_batches_to_save=demo_batches_to_save,
                        demo_return_targets=demo_return_targets,
                        validation_batches=validation_batches,
                        split_rng=split_rng,
                    )
                finally:
                    if had_active_rollout_runtime:
                        print("Demo Pretrain | restoring online rollout workers after parallel_cpu collection")
                        self._initialize_rollout_runtime()
            else:
                print(
                    "Demo Pretrain | runtime=isolated_cpu | envs={0} | steps_per_sync={1}".format(
                        max(1, int(self.config.num_workers)) * max(1, int(self.config.num_envs_per_worker)),
                        max(1, int(self.config.num_workers)) * max(1, int(self.config.steps_per_update)),
                    )
                )
                self._collect_demo_rollouts_with_isolated_worker(
                    demo_factory=demo_factory,
                    total_steps=demo_collection_steps,
                    demo_batches_to_save=demo_batches_to_save,
                    demo_return_targets=demo_return_targets,
                    validation_batches=validation_batches,
                    split_rng=split_rng,
                )

            summary["seconds_collection"] = float(perf_counter() - demo_collection_start)
            summary["dataset_path"] = self._save_demo_dataset(demo_batches_to_save)
            if demo_return_targets:
                concatenated_demo_returns = np.concatenate(demo_return_targets, axis=0)
                summary["demo_return_target_mean"] = float(np.mean(concatenated_demo_returns))
                summary["demo_return_target_std"] = float(np.std(concatenated_demo_returns))

        if validation_batches:
            self.demo_validation_batch = self._concat_replay_batches(validation_batches)

        summary["demo_train_replay_size_after_split"] = float(self.replay_buffer.demo_size())
        summary["demo_val_replay_size_after_split"] = float(
            len(self.demo_validation_batch) if self.demo_validation_batch is not None else 0
        )
        summary["demo_replay_size_after_collection"] = (
            float(summary["demo_train_replay_size_after_split"])
            + float(summary["demo_val_replay_size_after_split"])
        )
        print(
            "Demo Pretrain | collection done | demo_total={0:.0f} | train={1:.0f} | val={2:.0f} | return_mode={3} | target_mean={4:.6f} | target_std={5:.6f} | seconds={6:.3f}".format(
                float(summary["demo_replay_size_after_collection"]),
                float(summary["demo_train_replay_size_after_split"]),
                float(summary["demo_val_replay_size_after_split"]),
                str(summary["critic_target_mode"]),
                float(summary["demo_return_target_mean"]),
                float(summary["demo_return_target_std"]),
                float(summary["seconds_collection"]),
            )
        )

        if self.replay_buffer.demo_size() <= 0:
            raise ValueError("Demo pretrain requires at least one demo transition in replay.")

        actor_updates = max(0, int(self.config.actor_bc_pretrain_updates))
        if actor_updates > 0:
            print("Demo Pretrain | actor BC start | updates={0}".format(actor_updates))
            actor_pretrain_start = perf_counter()
            actor_metrics: dict[str, float] = {}
            actor_log_interval = self._progress_interval(actor_updates)
            next_actor_log_at = actor_log_interval
            last_actor_logged = 0
            actor_eval_interval = max(1, int(self.config.demo_pretrain_eval_interval))
            actor_eval_count = 0
            actor_no_improve = 0
            actor_best_state: dict[str, Any] | None = None
            actor_best_quick_eval: float | None = None
            actor_best_quick_eval_per_step: float | None = None
            actor_best_val_loss: float | None = None
            actor_early_stopped = False
            executed_actor_updates = 0
            for update_index in range(1, actor_updates + 1):
                actor_metrics = self.learner.actor_bc_pretrain_step()
                executed_actor_updates = update_index
                if update_index >= next_actor_log_at:
                    self._print_pretrain_progress("actor_bc", update_index, actor_updates, actor_pretrain_start)
                    last_actor_logged = update_index
                    next_actor_log_at += actor_log_interval
                should_eval = update_index == actor_updates or update_index % actor_eval_interval == 0
                if should_eval:
                    actor_eval_count += 1
                    validation_metrics = self._run_demo_pretrain_validation(include_quick_eval=True)
                    current_quick_eval = float(validation_metrics.get("quick_eval_return_mean", 0.0))
                    current_quick_eval_per_step = float(
                        validation_metrics.get("quick_eval_return_per_step_mean", 0.0)
                    )
                    current_actor_val = float(validation_metrics.get("actor_bc_val_loss", 0.0))
                    summary["actor_bc_val_loss_last"] = current_actor_val
                    summary["quick_eval_return_last"] = current_quick_eval
                    summary["quick_eval_return_per_step_last"] = current_quick_eval_per_step
                    quick_improved = self._metric_improved(
                        current_quick_eval,
                        actor_best_quick_eval,
                        greater_is_better=True,
                        min_relative_improvement=float(self.config.demo_pretrain_min_relative_improvement),
                    )
                    actor_val_improved = self._metric_improved(
                        current_actor_val,
                        actor_best_val_loss,
                        greater_is_better=False,
                        min_relative_improvement=float(self.config.demo_pretrain_min_relative_improvement),
                    )
                    quick_tie_margin = (
                        max(abs(float(actor_best_quick_eval)), 1e-8)
                        * float(self.config.demo_pretrain_min_relative_improvement)
                        if actor_best_quick_eval is not None
                        else 0.0
                    )
                    better_than_best = (
                        actor_best_state is None
                        or quick_improved
                        or (
                            actor_best_quick_eval is not None
                            and abs(current_quick_eval - float(actor_best_quick_eval)) <= quick_tie_margin
                            and actor_val_improved
                        )
                    )
                    if better_than_best:
                        actor_best_state = self._current_learner_state()
                        actor_best_quick_eval = current_quick_eval
                        actor_best_quick_eval_per_step = current_quick_eval_per_step
                        actor_best_val_loss = current_actor_val
                    if quick_improved or actor_val_improved or actor_best_state is None:
                        actor_no_improve = 0
                    else:
                        actor_no_improve += 1
                    print(
                        "Demo Pretrain | actor_bc eval | update={0}/{1} | val_bc={2:.6f} | quick_eval_return={3:.6f} | quick_eval_return_per_step={4:.6f} | patience={5}/{6}".format(
                            update_index,
                            actor_updates,
                            current_actor_val,
                            current_quick_eval,
                            current_quick_eval_per_step,
                            actor_no_improve,
                            int(self.config.demo_pretrain_patience),
                        )
                    )
                    if actor_no_improve >= int(self.config.demo_pretrain_patience) and update_index < actor_updates:
                        actor_early_stopped = True
                        break
            summary["seconds_actor_bc"] = float(perf_counter() - actor_pretrain_start)
            summary["actor_bc_updates"] = float(executed_actor_updates)
            summary["actor_bc_loss_last"] = float(actor_metrics.get("actor_bc_loss", 0.0))
            summary["actor_bc_eval_count"] = float(actor_eval_count)
            summary["actor_bc_early_stopped"] = bool(actor_early_stopped)
            summary["actor_bc_val_loss_best"] = float(actor_best_val_loss or 0.0)
            summary["quick_eval_return_best"] = float(actor_best_quick_eval or 0.0)
            summary["quick_eval_return_per_step_best"] = float(actor_best_quick_eval_per_step or 0.0)
            if actor_best_state is not None:
                self.learner.load_checkpoint_state(actor_best_state)
                self.learner.target_actor.load_state_dict(self.learner.actor.state_dict())
            if last_actor_logged < executed_actor_updates:
                self._print_pretrain_progress("actor_bc", executed_actor_updates, actor_updates, actor_pretrain_start)
            print(
                "Demo Pretrain | actor BC done | executed={0:.0f} | last_bc_loss={1:.6f} | best_val_bc={2:.6f} | best_quick_eval={3:.6f} | best_quick_eval_per_step={4:.6f} | early_stopped={5} | seconds={6:.3f}".format(
                    float(summary["actor_bc_updates"]),
                    float(summary["actor_bc_loss_last"]),
                    float(summary["actor_bc_val_loss_best"]),
                    float(summary["quick_eval_return_best"]),
                    float(summary["quick_eval_return_per_step_best"]),
                    bool(summary["actor_bc_early_stopped"]),
                    float(summary["seconds_actor_bc"]),
                )
            )

        critic_updates = max(0, int(self.config.critic_pretrain_updates))
        if critic_updates > 0:
            print("Demo Pretrain | critic start | updates={0}".format(critic_updates))
            critic_pretrain_start = perf_counter()
            critic_metrics: dict[str, float] = {}
            critic_log_interval = self._progress_interval(critic_updates)
            next_critic_log_at = critic_log_interval
            last_critic_logged = 0
            critic_eval_interval = max(1, int(self.config.demo_pretrain_eval_interval))
            critic_eval_count = 0
            critic_no_improve = 0
            critic_best_state: dict[str, Any] | None = None
            critic_best_val_loss: float | None = None
            critic_early_stopped = False
            executed_critic_updates = 0
            for update_index in range(1, critic_updates + 1):
                critic_metrics = self.learner.critic_pretrain_step()
                executed_critic_updates = update_index
                if update_index >= next_critic_log_at:
                    self._print_pretrain_progress("critic", update_index, critic_updates, critic_pretrain_start)
                    last_critic_logged = update_index
                    next_critic_log_at += critic_log_interval
                should_eval = update_index == critic_updates or update_index % critic_eval_interval == 0
                if should_eval:
                    critic_eval_count += 1
                    validation_metrics = self._run_demo_pretrain_validation(include_quick_eval=False)
                    current_critic_val = float(validation_metrics.get("critic_val_loss", 0.0))
                    summary["critic_val_loss_last"] = current_critic_val
                    summary["critic_q_pred_mean"] = float(validation_metrics.get("critic_q_pred_mean", 0.0))
                    summary["critic_q_pred_std"] = float(validation_metrics.get("critic_q_pred_std", 0.0))
                    summary["critic_target_mean"] = float(validation_metrics.get("critic_target_mean", 0.0))
                    summary["critic_target_std"] = float(validation_metrics.get("critic_target_std", 0.0))
                    summary["critic_error_mean"] = float(validation_metrics.get("critic_error_mean", 0.0))
                    summary["critic_error_std"] = float(validation_metrics.get("critic_error_std", 0.0))
                    critic_improved = self._metric_improved(
                        current_critic_val,
                        critic_best_val_loss,
                        greater_is_better=False,
                        min_relative_improvement=float(self.config.demo_pretrain_min_relative_improvement),
                    )
                    if critic_best_state is None or critic_improved:
                        critic_best_state = self._current_learner_state()
                        critic_best_val_loss = current_critic_val
                    if critic_improved or critic_best_state is None:
                        critic_no_improve = 0
                    else:
                        critic_no_improve += 1
                    print(
                        "Demo Pretrain | critic eval | update={0}/{1} | val_critic={2:.6f} | q_pred_mean={3:.6f} | target_mean={4:.6f} | patience={5}/{6}".format(
                            update_index,
                            critic_updates,
                            current_critic_val,
                            float(summary["critic_q_pred_mean"]),
                            float(summary["critic_target_mean"]),
                            critic_no_improve,
                            int(self.config.demo_pretrain_patience),
                        )
                    )
                    if critic_no_improve >= int(self.config.demo_pretrain_patience) and update_index < critic_updates:
                        critic_early_stopped = True
                        break
            summary["seconds_critic"] = float(perf_counter() - critic_pretrain_start)
            summary["critic_pretrain_updates"] = float(executed_critic_updates)
            summary["critic_loss_last"] = float(critic_metrics.get("critic_loss", 0.0))
            summary["critic_eval_count"] = float(critic_eval_count)
            summary["critic_pretrain_early_stopped"] = bool(critic_early_stopped)
            summary["critic_val_loss_best"] = float(critic_best_val_loss or 0.0)
            if critic_best_state is not None:
                self.learner.load_checkpoint_state(critic_best_state)
            if last_critic_logged < executed_critic_updates:
                self._print_pretrain_progress("critic", executed_critic_updates, critic_updates, critic_pretrain_start)
            print(
                "Demo Pretrain | critic done | executed={0:.0f} | last_critic_loss={1:.6f} | best_val_critic={2:.6f} | early_stopped={3} | seconds={4:.3f}".format(
                    float(summary["critic_pretrain_updates"]),
                    float(summary["critic_loss_last"]),
                    float(summary["critic_val_loss_best"]),
                    bool(summary["critic_pretrain_early_stopped"]),
                    float(summary["seconds_critic"]),
                )
            )

        bridge_env_steps = max(0, int(self.config.critic_bridge_env_steps))
        bridge_updates = max(0, int(self.config.critic_bridge_updates))
        if bool(self.config.critic_bridge_enabled) and bridge_env_steps > 0 and bridge_updates > 0:
            print(
                "Demo Pretrain | critic bridge start | env_steps={0} | updates={1} | mode={2}".format(
                    bridge_env_steps,
                    bridge_updates,
                    str(self.config.critic_bridge_behavior_mode),
                )
            )
            bridge_replay_buffer = self._build_critic_bridge_replay_buffer()
            bridge_validation_batch, bridge_collection_summary = self._collect_critic_bridge_rollouts(bridge_replay_buffer)
            summary.update(bridge_collection_summary)

            bridge_train_size = int(summary.get("critic_bridge_train_replay_size_after_split", 0.0) or 0)
            if bridge_train_size <= 0:
                raise ValueError("Critic bridge requires at least one training transition.")

            bridge_start = perf_counter()
            bridge_metrics: dict[str, float] = {}
            bridge_log_interval = self._progress_interval(bridge_updates)
            next_bridge_log_at = bridge_log_interval
            last_bridge_logged = 0
            bridge_eval_interval = max(1, int(self.config.critic_bridge_eval_interval))
            bridge_eval_count = 0
            bridge_no_improve = 0
            bridge_best_state: dict[str, Any] | None = None
            bridge_best_val_loss: float | None = None
            bridge_early_stopped = False
            executed_bridge_updates = 0
            bridge_aux_schedule = str(self.config.critic_bridge_teacher_return_aux_schedule)
            bridge_aux_levels = (
                tuple(float(level) for level in self.config.critic_bridge_teacher_return_aux_levels)
                if bridge_aux_schedule == "adaptive"
                else (float(self.config.critic_bridge_teacher_return_aux_coef),)
            )
            bridge_aux_level_index = 0
            bridge_aux_reduction_count = 0
            bridge_aux_stable_eval_count = 0
            current_bridge_teacher_aux_coef = float(bridge_aux_levels[bridge_aux_level_index])
            summary["critic_bridge_teacher_aux_coef"] = current_bridge_teacher_aux_coef
            summary["critic_bridge_teacher_aux_level_index"] = float(bridge_aux_level_index)
            for update_index in range(1, bridge_updates + 1):
                bridge_metrics = self.learner.critic_bridge_step(
                    bridge_replay_buffer,
                    teacher_aux_coef_override=current_bridge_teacher_aux_coef,
                )
                executed_bridge_updates = update_index
                if update_index >= next_bridge_log_at:
                    self._print_pretrain_progress("critic_bridge", update_index, bridge_updates, bridge_start)
                    last_bridge_logged = update_index
                    next_bridge_log_at += bridge_log_interval
                should_eval = update_index == bridge_updates or update_index % bridge_eval_interval == 0
                if should_eval:
                    bridge_eval_count += 1
                    validation_metrics = self._run_critic_bridge_validation(bridge_validation_batch)
                    current_bridge_val = float(validation_metrics.get("critic_val_loss", 0.0))
                    summary["critic_bridge_val_loss_last"] = current_bridge_val
                    summary["critic_bridge_q_pred_mean"] = float(validation_metrics.get("critic_q_pred_mean", 0.0))
                    summary["critic_bridge_q_pred_std"] = float(validation_metrics.get("critic_q_pred_std", 0.0))
                    summary["critic_bridge_target_mean"] = float(validation_metrics.get("critic_target_mean", 0.0))
                    summary["critic_bridge_target_std"] = float(validation_metrics.get("critic_target_std", 0.0))
                    summary["critic_bridge_error_mean"] = float(validation_metrics.get("critic_error_mean", 0.0))
                    summary["critic_bridge_error_std"] = float(validation_metrics.get("critic_error_std", 0.0))
                    target_mean_abs = max(abs(float(summary["critic_bridge_target_mean"])), 1.0)
                    bridge_aux_error_ratio = (
                        abs(
                            float(summary["critic_bridge_q_pred_mean"])
                            - float(summary["critic_bridge_target_mean"])
                        )
                        / target_mean_abs
                    )
                    summary["critic_bridge_teacher_aux_error_ratio"] = float(bridge_aux_error_ratio)
                    bridge_improved = self._metric_improved(
                        current_bridge_val,
                        bridge_best_val_loss,
                        greater_is_better=False,
                        min_relative_improvement=float(self.config.critic_bridge_min_relative_improvement),
                    )
                    if bridge_best_state is None or bridge_improved:
                        bridge_best_state = self._current_learner_state()
                        bridge_best_val_loss = current_bridge_val
                    if bridge_improved or bridge_best_state is None:
                        bridge_no_improve = 0
                    else:
                        bridge_no_improve += 1
                    bridge_aux_gate_passed = False
                    if bridge_aux_schedule == "adaptive" and bridge_aux_level_index < len(bridge_aux_levels) - 1:
                        reference_bridge_val = (
                            float(bridge_best_val_loss)
                            if bridge_best_val_loss is not None
                            else float(current_bridge_val)
                        )
                        if reference_bridge_val <= 0.0:
                            bridge_val_ok = float(current_bridge_val) <= 0.0
                        else:
                            bridge_val_ok = float(current_bridge_val) <= (
                                reference_bridge_val
                                * float(self.config.critic_bridge_teacher_return_aux_max_val_ratio)
                            )
                        bridge_error_ok = float(bridge_aux_error_ratio) <= float(
                            self.config.critic_bridge_teacher_return_aux_max_error_ratio
                        )
                        bridge_aux_gate_passed = bool(bridge_val_ok and bridge_error_ok)
                        if bridge_aux_gate_passed:
                            bridge_aux_stable_eval_count += 1
                        else:
                            bridge_aux_stable_eval_count = 0
                        if bridge_aux_stable_eval_count >= int(
                            self.config.critic_bridge_teacher_return_aux_required_evals
                        ):
                            previous_bridge_teacher_aux_coef = float(current_bridge_teacher_aux_coef)
                            bridge_aux_level_index += 1
                            current_bridge_teacher_aux_coef = float(bridge_aux_levels[bridge_aux_level_index])
                            bridge_aux_reduction_count += 1
                            bridge_aux_stable_eval_count = 0
                            bridge_no_improve = 0
                            print(
                                "Demo Pretrain | critic_bridge aux decay | update={0}/{1} | coef={2:.3f}->{3:.3f} | level={4}/{5}".format(
                                    update_index,
                                    bridge_updates,
                                    previous_bridge_teacher_aux_coef,
                                    current_bridge_teacher_aux_coef,
                                    bridge_aux_level_index,
                                    len(bridge_aux_levels) - 1,
                                )
                            )
                    summary["critic_bridge_teacher_aux_coef"] = float(current_bridge_teacher_aux_coef)
                    summary["critic_bridge_teacher_aux_level_index"] = float(bridge_aux_level_index)
                    summary["critic_bridge_teacher_aux_stable_eval_count"] = float(bridge_aux_stable_eval_count)
                    summary["critic_bridge_teacher_aux_reduction_count"] = float(bridge_aux_reduction_count)
                    print(
                        "Demo Pretrain | critic_bridge eval | update={0}/{1} | val_critic={2:.6f} | q_pred_mean={3:.6f} | target_mean={4:.6f} | aux_coef={5:.3f} | aux_stable={6} | aux_gate={7} | patience={8}/{9}".format(
                            update_index,
                            bridge_updates,
                            current_bridge_val,
                            float(summary["critic_bridge_q_pred_mean"]),
                            float(summary["critic_bridge_target_mean"]),
                            float(current_bridge_teacher_aux_coef),
                            bridge_aux_stable_eval_count,
                            1 if bridge_aux_gate_passed else 0,
                            bridge_no_improve,
                            int(self.config.critic_bridge_patience),
                        )
                    )
                    if bridge_no_improve >= int(self.config.critic_bridge_patience) and update_index < bridge_updates:
                        bridge_early_stopped = True
                        break

            summary["seconds_critic_bridge"] = float(perf_counter() - bridge_start)
            summary["critic_bridge_updates"] = float(executed_bridge_updates)
            summary["critic_bridge_loss_last"] = float(bridge_metrics.get("critic_loss", 0.0))
            summary["critic_bridge_teacher_aux_loss_last"] = float(
                bridge_metrics.get("critic_bridge_teacher_aux_loss", 0.0)
            )
            summary["critic_bridge_teacher_aux_coef"] = float(current_bridge_teacher_aux_coef)
            summary["critic_bridge_teacher_aux_level_index"] = float(bridge_aux_level_index)
            summary["critic_bridge_teacher_aux_stable_eval_count"] = float(bridge_aux_stable_eval_count)
            summary["critic_bridge_teacher_aux_reduction_count"] = float(bridge_aux_reduction_count)
            summary["critic_bridge_eval_count"] = float(bridge_eval_count)
            summary["critic_bridge_early_stopped"] = bool(bridge_early_stopped)
            summary["critic_bridge_val_loss_best"] = float(bridge_best_val_loss or 0.0)
            if bridge_best_state is not None:
                self.learner.load_checkpoint_state(bridge_best_state)
            if last_bridge_logged < executed_bridge_updates:
                self._print_pretrain_progress("critic_bridge", executed_bridge_updates, bridge_updates, bridge_start)
            print(
                "Demo Pretrain | critic bridge done | executed={0:.0f} | last_critic_loss={1:.6f} | last_teacher_aux={2:.6f} | aux_coef={3:.3f} | aux_level={4:.0f} | aux_reductions={5:.0f} | best_val_critic={6:.6f} | early_stopped={7} | seconds={8:.3f}".format(
                    float(summary["critic_bridge_updates"]),
                    float(summary["critic_bridge_loss_last"]),
                    float(summary["critic_bridge_teacher_aux_loss_last"]),
                    float(summary["critic_bridge_teacher_aux_coef"]),
                    float(summary["critic_bridge_teacher_aux_level_index"]),
                    float(summary["critic_bridge_teacher_aux_reduction_count"]),
                    float(summary["critic_bridge_val_loss_best"]),
                    bool(summary["critic_bridge_early_stopped"]),
                    float(summary["seconds_critic_bridge"]),
                )
            )

        self.demo_pretrain_completed = True
        self.demo_pretrain_summary = dict(summary)
        return dict(summary)

    def _collect_rollouts(self) -> list[RolloutResult]:
        if self.config.num_workers > 1:
            self._start_rollout_collection()
            return self._finish_rollout_collection()

        warmup_allocations = self._global_warmup_allocations()
        rollout_results = [
            self.workers[0].collect(
                num_steps=int(self.workers[0].config.rollout_steps_per_sync),
                global_warmup_steps=int(warmup_allocations[0]) if warmup_allocations else 0,
                global_env_start_step=int(self.global_env_steps),
            )
        ]
        try:
            replay_extend_start = perf_counter()
            for result in rollout_results:
                self.replay_buffer.extend(result.replay_batch)
                self.global_env_steps += len(result.replay_batch)
            self._last_replay_extend_seconds = float(perf_counter() - replay_extend_start)
            return rollout_results
        finally:
            for result in rollout_results:
                result.release_shared_memory()

    def _start_rollout_collection(self) -> None:
        if self.config.num_workers <= 1:
            raise RuntimeError("Asynchronous rollout collection requires num_workers > 1.")
        if self._rollout_collect_inflight:
            raise RuntimeError("Rollout collection is already in flight.")

        warmup_allocations = self._global_warmup_allocations()
        for worker, warmup_steps in zip(self.workers, warmup_allocations):
            worker.start_collect(
                num_steps=int(worker.config.rollout_steps_per_sync),
                global_warmup_steps=int(warmup_steps),
                global_env_start_step=int(self.global_env_steps),
            )
        self._rollout_collect_inflight = True

    def _finish_rollout_collection(self) -> list[RolloutResult]:
        if self.config.num_workers <= 1:
            raise RuntimeError("_finish_rollout_collection is only valid for num_workers > 1.")
        if not self._rollout_collect_inflight:
            raise RuntimeError("No rollout collection is currently in flight.")

        rollout_results: list[RolloutResult] = []
        try:
            pending_workers = {worker.connection: worker for worker in self.workers}
            while pending_workers:
                ready_connections = wait(
                    list(pending_workers.keys()),
                    timeout=float(self.config.worker_rpc_timeout_seconds),
                )
                if not ready_connections:
                    raise TimeoutError(
                        "Timed out waiting for rollout workers: {0}".format(
                            [worker.config.worker_id for worker in pending_workers.values()]
                        )
                    )
                for ready_connection in ready_connections:
                    worker = pending_workers.pop(ready_connection)
                    rollout_results.append(worker.finish_collect_ready())

            replay_extend_start = perf_counter()
            for result in rollout_results:
                self.replay_buffer.extend(result.replay_batch)
                self.global_env_steps += len(result.replay_batch)
            self._last_replay_extend_seconds = float(perf_counter() - replay_extend_start)
            return rollout_results
        finally:
            self._rollout_collect_inflight = False
            for result in rollout_results:
                result.release_shared_memory()

    @staticmethod
    def _empty_rollout_sync_profile() -> dict[str, float]:
        return {
            "profile_actor_sync_seconds": 0.0,
            "profile_actor_publish_seconds": 0.0,
            "profile_actor_sync_inference_server_seconds": 0.0,
            "profile_actor_sync_worker_rpc_seconds": 0.0,
        }

    @staticmethod
    def _accumulated_learner_profile_keys() -> tuple[str, ...]:
        return (
            "profile_replay_sample_seconds",
            "profile_batch_to_device_seconds",
            "profile_critic_update_seconds",
            "profile_actor_update_seconds",
            "profile_target_soft_update_seconds",
        )

    def _run_learner_updates(self) -> dict[str, float]:
        learner_metrics: dict[str, float] | None = None
        accumulated_profiles = {
            key: 0.0 for key in self._accumulated_learner_profile_keys()
        }
        for _ in range(int(self.config.gradient_steps_per_update)):
            step_metrics = self.learner.train_step(
                global_env_steps=int(self.global_env_steps),
                teacher_release_unlocked=(self.teacher_takeover_release_env_step is not None),
                teacher_release_env_step=self.teacher_takeover_release_env_step,
                teacher_handoff_stage=self.teacher_handoff_stage,
                teacher_full_release_env_step=self.teacher_takeover_full_release_env_step,
            )
            learner_metrics = dict(step_metrics)
            for key in accumulated_profiles:
                accumulated_profiles[key] += float(step_metrics.get(key, 0.0))
        if learner_metrics is None:
            learner_metrics = {}
        learner_metrics.update(accumulated_profiles)
        return learner_metrics

    def _sync_rollout_workers_if_needed(self, update: int) -> dict[str, float]:
        sync_metrics = self._empty_rollout_sync_profile()
        if update == 1 or ((update - 1) % self.config.worker_sync_interval == 0):
            sync_metrics = self._broadcast_actor_state_to_rollout_runtime()
        return sync_metrics

    def _broadcast_actor_state_to_rollout_runtime(self) -> dict[str, float]:
        sync_metrics = self._empty_rollout_sync_profile()
        publish_start = perf_counter()
        actor_state_dict, actor_version = self.learner.publish_actor_state()
        sync_metrics["profile_actor_publish_seconds"] = float(perf_counter() - publish_start)

        inference_server_sync_start = perf_counter()
        self._sync_rollout_inference_servers(actor_state_dict)
        sync_metrics["profile_actor_sync_inference_server_seconds"] = float(
            perf_counter() - inference_server_sync_start
        )

        worker_sync_start = perf_counter()
        for worker in self.workers:
            worker.sync_actor(
                actor_state_dict,
                version=actor_version,
                teacher_takeover_release_env_step=self.teacher_takeover_release_env_step,
                teacher_takeover_soft_release_env_step=self.teacher_takeover_soft_release_env_step,
                teacher_takeover_full_release_env_step=self.teacher_takeover_full_release_env_step,
                teacher_handoff_stage=self.teacher_handoff_stage,
            )
        sync_metrics["profile_actor_sync_worker_rpc_seconds"] = float(perf_counter() - worker_sync_start)
        sync_metrics["profile_actor_sync_seconds"] = (
            sync_metrics["profile_actor_publish_seconds"]
            + sync_metrics["profile_actor_sync_inference_server_seconds"]
            + sync_metrics["profile_actor_sync_worker_rpc_seconds"]
        )
        return sync_metrics

    def _should_overlap_rollout_and_update(self) -> bool:
        return bool(self.config.overlap_rollout_and_update) and self.config.num_workers > 1

    def train(
        self,
        num_updates: int | None = None,
        on_update: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        total_updates = int(num_updates or self.config.total_updates)
        if self.completed_updates >= total_updates:
            return [dict(item) for item in self.history]
        if self.completed_updates == 0 and not self.demo_pretrain_completed and bool(self.config.demo_pretrain_enabled):
            self._run_demo_pretrain()
        if not self._rollout_runtime_initialized:
            print("Rollout Runtime | initializing online rollout workers")
            self._initialize_rollout_runtime()

        overlap_rollout_and_update = self._should_overlap_rollout_and_update()
        pending_collect_started_at: float | None = None
        pending_collect_update: int | None = None
        pending_sync_metrics = self._empty_rollout_sync_profile()

        for update in range(self.completed_updates + 1, total_updates + 1):
            learner_metrics = {
                "critic1_loss": 0.0,
                "critic2_loss": 0.0,
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "actor_q_loss": 0.0,
                "actor_entropy": 0.0,
                "actor_logit_l2": 0.0,
                "actor_reg_loss": 0.0,
                "loss": 0.0,
                "replay_size": float(len(self.replay_buffer)),
                "actor_lr": float(self.learner.actor_optimizer.param_groups[0]["lr"]),
                "critic_lr": float(self.learner.critic_optimizer.param_groups[0]["lr"]),
            }
            learner_update_seconds = 0.0
            rollout_finish_wait_seconds = 0.0
            rollout_overlap_seconds = 0.0
            rollout_sync_metrics = self._empty_rollout_sync_profile()

            if overlap_rollout_and_update and pending_collect_update == update:
                rollout_sync_metrics = dict(pending_sync_metrics)
                learner_update_start = perf_counter()
                if update % self.config.train_every == 0:
                    learner_metrics = self._run_learner_updates()
                learner_update_seconds = float(perf_counter() - learner_update_start)

                rollout_wait_start = perf_counter()
                rollout_results = self._finish_rollout_collection()
                rollout_finish_wait_seconds = float(perf_counter() - rollout_wait_start)
                if pending_collect_started_at is None:
                    raise RuntimeError("Missing rollout collect start time for update {0}.".format(update))
                rollout_collect_seconds = float(perf_counter() - pending_collect_started_at)
                rollout_overlap_seconds = max(0.0, rollout_collect_seconds - rollout_finish_wait_seconds)
                pending_collect_update = None
                pending_collect_started_at = None
                pending_sync_metrics = self._empty_rollout_sync_profile()
            else:
                self._apply_curriculum_stage(update)
                rollout_sync_metrics = self._sync_rollout_workers_if_needed(update)

                rollout_collect_start = perf_counter()
                rollout_results = self._collect_rollouts()
                rollout_collect_seconds = float(perf_counter() - rollout_collect_start)

                learner_update_start = perf_counter()
                if update % self.config.train_every == 0:
                    learner_metrics = self._run_learner_updates()
                learner_update_seconds = float(perf_counter() - learner_update_start)

            rollout_metrics = [result.metrics for result in rollout_results]
            mean_rollout_reward = float(np.mean([item["mean_reward"] for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_cooperation = float(np.mean([item["mean_actual_cooperation_rate"] for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_resource = float(np.mean([item["mean_resource"] for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_gini = float(np.mean([item["mean_gini"] for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_payoff = float(np.mean([item["mean_payoff"] for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_pool_grown = float(np.mean([item["mean_pool_grown"] for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_pool_raw = float(np.mean([item["mean_pool_raw"] for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_collect_worker_seconds = float(np.mean([item.get("collect_wall_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_env_step_seconds = float(np.mean([item.get("env_step_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_inference_wait_seconds = float(np.mean([item.get("inference_wait_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_inference_request_build_seconds = float(
                np.mean([item.get("inference_request_build_seconds", 0.0) for item in rollout_metrics])
            ) if rollout_metrics else 0.0
            mean_rollout_local_policy_forward_seconds = float(np.mean([item.get("local_policy_forward_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_action_to_numpy_seconds = float(np.mean([item.get("action_to_numpy_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_transition_encode_seconds = float(np.mean([item.get("transition_encode_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_stack_transitions_seconds = float(np.mean([item.get("stack_transitions_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_shared_memory_serialize_seconds = float(np.mean([item.get("shared_memory_serialize_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_shared_memory_deserialize_seconds = float(np.mean([item.get("shared_memory_deserialize_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_inference_batch_size = float(np.mean([item.get("inference_batch_size_mean", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            max_rollout_inference_batch_size = float(max((item.get("inference_batch_size_max", 0.0) for item in rollout_metrics), default=0.0))
            mean_teacher_takeover_prob = float(np.mean([item.get("teacher_takeover_prob_mean", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            total_rollout_steps_collected = float(sum(item.get("steps_collected", 0.0) for item in rollout_metrics))
            rollout_steps_per_second = (
                total_rollout_steps_collected / rollout_collect_seconds
                if rollout_collect_seconds > 0.0
                else 0.0
            )
            behavior_counts: Counter[str] = Counter()
            for item in rollout_metrics:
                behavior_counts.update(item.get("behavior_source_counts", {}))

            evaluation_seconds = 0.0
            metrics = {
                "update": float(update),
                "loss": float(learner_metrics["loss"]),
                "policy_loss": float(learner_metrics["actor_loss"]),
                "value_loss": float(learner_metrics["critic_loss"]),
                "entropy": float(learner_metrics["actor_entropy"]),
                "critic1_loss": float(learner_metrics["critic1_loss"]),
                "critic2_loss": float(learner_metrics["critic2_loss"]),
                "critic_loss": float(learner_metrics["critic_loss"]),
                "actor_loss": float(learner_metrics["actor_loss"]),
                "actor_q_loss": float(learner_metrics["actor_q_loss"]),
                "actor_entropy": float(learner_metrics["actor_entropy"]),
                "actor_logit_l2": float(learner_metrics["actor_logit_l2"]),
                "actor_reg_loss": float(learner_metrics["actor_reg_loss"]),
                "actor_bc_loss": float(learner_metrics.get("actor_bc_loss", 0.0)),
                "actor_bc_coef": float(learner_metrics.get("actor_bc_coef", 0.0)),
                "replay_size": float(len(self.replay_buffer)),
                "replay_demo_frac": float(learner_metrics.get("replay_demo_frac", 0.0)),
                "replay_pool_power_demo_frac": float(learner_metrics.get("replay_pool_power_demo_frac", 0.0)),
                "replay_teacher_frac": float(learner_metrics.get("replay_teacher_frac", 0.0)),
                "replay_collapse_frac": float(learner_metrics.get("replay_collapse_frac", 0.0)),
                "teacher_takeover_prob": mean_teacher_takeover_prob,
                "teacher_handoff_stage": float(self.teacher_handoff_stage),
                "teacher_handoff_soft_released": 1.0 if int(self.teacher_handoff_stage) >= 1 else 0.0,
                "teacher_handoff_full_released": 1.0 if int(self.teacher_handoff_stage) >= 2 else 0.0,
                "teacher_handoff_stage_stable_eval_count": float(self.teacher_handoff_stable_eval_count),
                "mean_rollout_reward": mean_rollout_reward,
                "actor_lr": float(learner_metrics["actor_lr"]),
                "critic_lr": float(learner_metrics["critic_lr"]),
                "curriculum_stage": float(self.active_curriculum_stage_index or 0),
                "rollout_f_c": mean_rollout_cooperation,
                "rollout_R_mean": mean_rollout_resource,
                "rollout_gini": mean_rollout_gini,
                "rollout_payoff_mean": mean_rollout_payoff,
                "rollout_pool_grown_mean": mean_rollout_pool_grown,
                "rollout_pool_mean": mean_rollout_pool_raw,
                "profile_rollout_collect_seconds": rollout_collect_seconds,
                "profile_rollout_collect_worker_seconds": mean_rollout_collect_worker_seconds,
                "profile_rollout_env_step_seconds": mean_rollout_env_step_seconds,
                "profile_rollout_inference_wait_seconds": mean_rollout_inference_wait_seconds,
                "profile_rollout_inference_request_build_seconds": mean_rollout_inference_request_build_seconds,
                "profile_rollout_local_policy_forward_seconds": mean_rollout_local_policy_forward_seconds,
                "profile_rollout_action_to_numpy_seconds": mean_rollout_action_to_numpy_seconds,
                "profile_rollout_transition_encode_seconds": mean_rollout_transition_encode_seconds,
                "profile_rollout_stack_transitions_seconds": mean_rollout_stack_transitions_seconds,
                "profile_rollout_shared_memory_serialize_seconds": mean_rollout_shared_memory_serialize_seconds,
                "profile_rollout_shared_memory_deserialize_seconds": mean_rollout_shared_memory_deserialize_seconds,
                "profile_rollout_finish_wait_seconds": rollout_finish_wait_seconds,
                "profile_rollout_overlap_seconds": rollout_overlap_seconds,
                "profile_replay_extend_seconds": float(self._last_replay_extend_seconds),
                "profile_rollout_steps_per_second": rollout_steps_per_second,
                "profile_rollout_inference_batch_size_mean": mean_rollout_inference_batch_size,
                "profile_rollout_inference_batch_size_max": max_rollout_inference_batch_size,
                "profile_learner_update_seconds": learner_update_seconds,
                "profile_replay_sample_seconds": float(learner_metrics.get("profile_replay_sample_seconds", 0.0)),
                "profile_batch_to_device_seconds": float(learner_metrics.get("profile_batch_to_device_seconds", 0.0)),
                "profile_critic_update_seconds": float(learner_metrics.get("profile_critic_update_seconds", 0.0)),
                "profile_actor_update_seconds": float(learner_metrics.get("profile_actor_update_seconds", 0.0)),
                "profile_target_soft_update_seconds": float(
                    learner_metrics.get("profile_target_soft_update_seconds", 0.0)
                ),
                "actor_grad_norm": float(learner_metrics.get("actor_grad_norm", 0.0)),
                "critic_grad_norm": float(learner_metrics.get("critic_grad_norm", 0.0)),
                "actor_q_coef": float(learner_metrics.get("actor_q_coef", 0.0)),
                "profile_eval_seconds": evaluation_seconds,
                "global_env_steps": float(self.global_env_steps),
                **rollout_sync_metrics,
            }
            for key, value in learner_metrics.items():
                if key.startswith("replay_source_frac_") or key.startswith("replay_topology_frac_"):
                    metrics[key] = float(value)
            total_behavior_samples = float(sum(behavior_counts.values()))
            for source in (
                "uniform",
                "proportional",
                "constant_mix",
                "pool_power_mix",
                "random_logits",
                "actor_logits",
            ):
                ratio = 0.0
                if total_behavior_samples > 0.0:
                    ratio = float(behavior_counts.get(source, 0)) / total_behavior_samples
                metrics["behavior_frac_{0}".format(source)] = ratio

            if update % self.config.eval_interval == 0 or update == total_updates:
                evaluation_start = perf_counter()
                evaluation = self.evaluate(self.config.eval_episodes)
                demo_validation_metrics = self._run_demo_pretrain_validation(include_quick_eval=False)
                metrics["online_actor_bc_val_loss"] = float(demo_validation_metrics.get("actor_bc_val_loss", 0.0))
                metrics["online_critic_val_loss"] = float(demo_validation_metrics.get("critic_val_loss", 0.0))
                metrics["online_critic_q_pred_mean"] = float(demo_validation_metrics.get("critic_q_pred_mean", 0.0))
                metrics["online_critic_q_pred_std"] = float(demo_validation_metrics.get("critic_q_pred_std", 0.0))
                metrics["online_critic_target_mean"] = float(demo_validation_metrics.get("critic_target_mean", 0.0))
                metrics["online_critic_target_std"] = float(demo_validation_metrics.get("critic_target_std", 0.0))
                metrics["online_critic_error_mean"] = float(demo_validation_metrics.get("critic_error_mean", 0.0))
                metrics["online_critic_error_std"] = float(demo_validation_metrics.get("critic_error_std", 0.0))
                teacher_release_metrics = self._update_adaptive_teacher_release(
                    online_eval_cooperation_mean=float(evaluation.get("cooperation_mean", 0.0)),
                    online_eval_return_mean=float(evaluation.get("return_mean", 0.0)),
                    actor_bc_val_loss=float(demo_validation_metrics.get("actor_bc_val_loss", 0.0)),
                    critic_val_loss=float(demo_validation_metrics.get("critic_val_loss", 0.0)),
                    behavior_frac_actor_logits=float(metrics.get("behavior_frac_actor_logits", 0.0)),
                )
                for key, value in teacher_release_metrics.items():
                    metrics[key] = float(value)
                if (
                    bool(teacher_release_metrics.get("teacher_release_just_unlocked", 0.0))
                    or bool(teacher_release_metrics.get("teacher_handoff_stage_just_changed", 0.0))
                ) and self.workers:
                    self._broadcast_actor_state_to_rollout_runtime()
                evaluation_seconds = float(perf_counter() - evaluation_start)
                metrics["profile_eval_seconds"] = evaluation_seconds
                metrics["eval_return_mean"] = evaluation["return_mean"]
                metrics["eval_cooperation_mean"] = evaluation["cooperation_mean"]
                metrics["eval_gini_mean"] = evaluation["gini_mean"]
                metrics["eval_mean_total_resource"] = evaluation["mean_total_resource"]
                metrics["eval_collapse_rate"] = evaluation["collapse_rate"]
                for key, value in evaluation.items():
                    prefixed_key = "eval_{0}".format(key)
                    if prefixed_key not in metrics:
                        metrics[prefixed_key] = float(value)

            self.history.append(metrics)
            self.completed_updates = update
            on_update_seconds = 0.0
            if on_update is not None:
                on_update_start = perf_counter()
                on_update(dict(metrics))
                on_update_seconds = float(perf_counter() - on_update_start)
            self.history[-1]["profile_on_update_seconds"] = on_update_seconds

            if overlap_rollout_and_update and update < total_updates:
                next_update = update + 1
                self._apply_curriculum_stage(next_update)
                pending_sync_metrics = self._sync_rollout_workers_if_needed(next_update)
                pending_collect_started_at = perf_counter()
                self._start_rollout_collection()
                pending_collect_update = next_update

        return [dict(item) for item in self.history]

    def evaluate(self, num_episodes: int | None = None) -> dict[str, float]:
        return self.evaluator.evaluate(self.learner.actor, num_episodes=num_episodes)
