from __future__ import annotations

import copy
from dataclasses import asdict
from collections import Counter
from multiprocessing.connection import wait
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from Project1.env import SPGGEnv
from Project1.policies.gnn_rl import GNNAllocationPolicy

from .config import DomainRandomizationConfig, EvalConfig, GraphTD3Config, WorkerConfig
from .critic import GraphActionCritic, GraphActionCriticConfig, TwinCritic
from .evaluator import GraphTD3Evaluator
from .exploration import LogitSpaceExplorer
from .learner import GraphTD3Learner
from .replay import ReplayBuffer
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
        self.replay_buffer = ReplayBuffer(config.replay_capacity, seed=config.seed)

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
        centralized_rollout_inference = _should_use_centralized_rollout_inference(config)
        worker_inference_connections: dict[int, Any] = {}
        if centralized_rollout_inference:
            worker_ids_by_device: dict[str, list[int]] = {}
            for worker_id in range(config.num_workers):
                worker_device = _resolve_rollout_device_for_worker(config.rollout_device, worker_id)
                worker_ids_by_device.setdefault(worker_device, []).append(worker_id)
            for worker_device, worker_ids in worker_ids_by_device.items():
                inference_server = ParallelRolloutInferenceServer(
                    actor=copy.deepcopy(policy),
                    train_config=config,
                    device=worker_device,
                    num_clients=len(worker_ids),
                )
                self.rollout_inference_servers.append(inference_server)
                for local_index, worker_id in enumerate(worker_ids):
                    worker_inference_connections[worker_id] = inference_server.take_worker_connection(local_index)

        for worker_id in range(config.num_workers):
            worker_config = WorkerConfig(
                worker_id=worker_id,
                seed=(config.seed or 0) + worker_id,
                rollout_steps_per_sync=config.steps_per_update,
            )
            worker_device = _resolve_rollout_device_for_worker(config.rollout_device, worker_id)
            if config.num_workers > 1:
                worker = ParallelRolloutWorker(
                    actor=copy.deepcopy(policy),
                    env_factory=train_factory,
                    config=worker_config,
                    train_config=config,
                    device="cpu" if centralized_rollout_inference else worker_device,
                    inference_connection=worker_inference_connections.get(worker_id),
                )
            else:
                worker = RolloutWorker(
                    actor=copy.deepcopy(policy),
                    explorer=self.rollout_explorer,
                    env_factory=train_factory,
                    config=worker_config,
                    train_config=config,
                    device=worker_device,
                )
            self.workers.append(worker)

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

    def close(self) -> None:
        for worker in self.workers:
            close_method = getattr(worker, "close", None)
            if callable(close_method):
                close_method()
        for inference_server in self.rollout_inference_servers:
            close_method = getattr(inference_server, "close", None)
            if callable(close_method):
                close_method()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
                worker.sync_actor(actor_state_dict, version=actor_version)

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

    def _collect_rollouts(self) -> list[RolloutResult]:
        warmup_allocations = self._global_warmup_allocations()
        rollout_results: list[RolloutResult] = []

        try:
            if self.config.num_workers > 1:
                for worker, warmup_steps in zip(self.workers, warmup_allocations):
                    worker.start_collect(
                        num_steps=int(worker.config.rollout_steps_per_sync),
                        global_warmup_steps=int(warmup_steps),
                    )
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
            else:
                rollout_results = [
                    self.workers[0].collect(
                        num_steps=int(self.workers[0].config.rollout_steps_per_sync),
                        global_warmup_steps=int(warmup_allocations[0]) if warmup_allocations else 0,
                    )
                ]

            for result in rollout_results:
                self.replay_buffer.extend(result.replay_batch)
                self.global_env_steps += len(result.replay_batch)
            return rollout_results
        finally:
            for result in rollout_results:
                result.release_shared_memory()

    def train(
        self,
        num_updates: int | None = None,
        on_update: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        total_updates = int(num_updates or self.config.total_updates)
        if self.completed_updates >= total_updates:
            return [dict(item) for item in self.history]

        for update in range(self.completed_updates + 1, total_updates + 1):
            self._apply_curriculum_stage(update)
            if update == 1 or ((update - 1) % self.config.worker_sync_interval == 0):
                actor_state_dict, actor_version = self.learner.publish_actor_state()
                self._sync_rollout_inference_servers(actor_state_dict)
                for worker in self.workers:
                    worker.sync_actor(actor_state_dict, version=actor_version)

            rollout_collect_start = perf_counter()
            rollout_results = self._collect_rollouts()
            rollout_collect_seconds = float(perf_counter() - rollout_collect_start)
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
            mean_rollout_local_policy_forward_seconds = float(np.mean([item.get("local_policy_forward_seconds", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            mean_rollout_inference_batch_size = float(np.mean([item.get("inference_batch_size_mean", 0.0) for item in rollout_metrics])) if rollout_metrics else 0.0
            max_rollout_inference_batch_size = float(max((item.get("inference_batch_size_max", 0.0) for item in rollout_metrics), default=0.0))
            total_rollout_steps_collected = float(sum(item.get("steps_collected", 0.0) for item in rollout_metrics))
            rollout_steps_per_second = (
                total_rollout_steps_collected / rollout_collect_seconds
                if rollout_collect_seconds > 0.0
                else 0.0
            )
            behavior_counts: Counter[str] = Counter()
            for item in rollout_metrics:
                behavior_counts.update(item.get("behavior_source_counts", {}))

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
            learner_update_start = perf_counter()
            if update % self.config.train_every == 0:
                for _ in range(self.config.gradient_steps_per_update):
                    learner_metrics = self.learner.train_step()
            learner_update_seconds = float(perf_counter() - learner_update_start)

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
                "replay_size": float(learner_metrics["replay_size"]),
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
                "profile_rollout_local_policy_forward_seconds": mean_rollout_local_policy_forward_seconds,
                "profile_rollout_steps_per_second": rollout_steps_per_second,
                "profile_rollout_inference_batch_size_mean": mean_rollout_inference_batch_size,
                "profile_rollout_inference_batch_size_max": max_rollout_inference_batch_size,
                "profile_learner_update_seconds": learner_update_seconds,
                "global_env_steps": float(self.global_env_steps),
            }
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
                evaluation = self.evaluate(self.config.eval_episodes)
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
            if on_update is not None:
                on_update(dict(metrics))

        return [dict(item) for item in self.history]

    def evaluate(self, num_episodes: int | None = None) -> dict[str, float]:
        return self.evaluator.evaluate(self.learner.actor, num_episodes=num_episodes)
