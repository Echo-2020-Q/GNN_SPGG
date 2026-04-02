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
                num_envs_per_worker=config.num_envs_per_worker,
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
        self._rollout_collect_inflight = False
        self._last_replay_extend_seconds = 0.0
        self.demo_pretrain_completed = False
        self.demo_pretrain_summary: dict[str, float | str | bool | None] | None = None

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
            "demo_pretrain_completed": bool(self.demo_pretrain_completed),
            "demo_pretrain_summary": None if self.demo_pretrain_summary is None else dict(self.demo_pretrain_summary),
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

    def _global_step_allocations(self, total_steps: int) -> list[int]:
        remaining_steps = max(0, int(total_steps))
        if remaining_steps <= 0:
            return [0 for _ in self.workers]

        allocations: list[int] = []
        remaining_workers = len(self.workers)
        for worker in self.workers:
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
    ) -> list[RolloutResult]:
        if len(step_allocations) != len(self.workers):
            raise ValueError("step_allocations must align with workers.")
        if warmup_allocations is None:
            warmup_allocations = [0 for _ in self.workers]
        if len(warmup_allocations) != len(self.workers):
            raise ValueError("warmup_allocations must align with workers.")

        positive_requests = [
            (worker, int(num_steps), int(warmup_steps))
            for worker, num_steps, warmup_steps in zip(self.workers, step_allocations, warmup_allocations)
            if int(num_steps) > 0
        ]
        if not positive_requests:
            return []

        if self.config.num_workers <= 1:
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
            replay_extend_start = perf_counter()
            for result in rollout_results:
                self.replay_buffer.extend(result.replay_batch)
                if count_env_steps:
                    self.global_env_steps += len(result.replay_batch)
            self._last_replay_extend_seconds = float(perf_counter() - replay_extend_start)
            return rollout_results
        finally:
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
            "actor_bc_updates": float(0),
            "critic_pretrain_updates": float(0),
            "actor_bc_loss_last": 0.0,
            "critic_loss_last": 0.0,
            "seconds_collection": 0.0,
            "seconds_actor_bc": 0.0,
            "seconds_critic": 0.0,
            "dataset_path": None,
            "behavior_source": str(self.config.demo_collection_behavior_source),
            "critic_target_mode": str(self.config.demo_critic_pretrain_target_mode),
            "demo_return_target_mean": 0.0,
            "demo_return_target_std": 0.0,
        }
        if not bool(self.config.demo_pretrain_enabled):
            self.demo_pretrain_completed = False
            self.demo_pretrain_summary = dict(summary)
            return dict(summary)

        demo_batches_to_save: list[Any] = []
        demo_return_targets: list[np.ndarray] = []
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
                        valid_mask = result.replay_batch.demo_return_valid.detach().cpu().numpy().astype(np.bool_, copy=False)
                        if np.any(valid_mask):
                            demo_return_targets.append(
                                result.replay_batch.demo_return_target.detach().cpu().numpy().astype(np.float32, copy=False)[valid_mask]
                            )
                    remaining_steps -= collected_now
            finally:
                for worker in self.workers:
                    worker.set_env_factory(self.train_factory, reset_environment=True)

            summary["seconds_collection"] = float(perf_counter() - demo_collection_start)
            summary["demo_replay_size_after_collection"] = float(self.replay_buffer.demo_size())
            summary["dataset_path"] = self._save_demo_dataset(demo_batches_to_save)
            if demo_return_targets:
                concatenated_demo_returns = np.concatenate(demo_return_targets, axis=0)
                summary["demo_return_target_mean"] = float(np.mean(concatenated_demo_returns))
                summary["demo_return_target_std"] = float(np.std(concatenated_demo_returns))
            print(
                "Demo Pretrain | collection done | demo_replay_size={0:.0f} | return_mode={1} | target_mean={2:.6f} | target_std={3:.6f} | seconds={4:.3f}".format(
                    float(summary["demo_replay_size_after_collection"]),
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
            for _ in range(actor_updates):
                actor_metrics = self.learner.actor_bc_pretrain_step()
            summary["seconds_actor_bc"] = float(perf_counter() - actor_pretrain_start)
            summary["actor_bc_updates"] = float(actor_updates)
            summary["actor_bc_loss_last"] = float(actor_metrics.get("actor_bc_loss", 0.0))
            print(
                "Demo Pretrain | actor BC done | last_bc_loss={0:.6f} | seconds={1:.3f}".format(
                    float(summary["actor_bc_loss_last"]),
                    float(summary["seconds_actor_bc"]),
                )
            )

        critic_updates = max(0, int(self.config.critic_pretrain_updates))
        if critic_updates > 0:
            print("Demo Pretrain | critic start | updates={0}".format(critic_updates))
            critic_pretrain_start = perf_counter()
            critic_metrics: dict[str, float] = {}
            for _ in range(critic_updates):
                critic_metrics = self.learner.critic_pretrain_step()
            summary["seconds_critic"] = float(perf_counter() - critic_pretrain_start)
            summary["critic_pretrain_updates"] = float(critic_updates)
            summary["critic_loss_last"] = float(critic_metrics.get("critic_loss", 0.0))
            print(
                "Demo Pretrain | critic done | last_critic_loss={0:.6f} | seconds={1:.3f}".format(
                    float(summary["critic_loss_last"]),
                    float(summary["seconds_critic"]),
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

    def _sync_rollout_workers_if_needed(self, update: int) -> dict[str, float]:
        sync_metrics = self._empty_rollout_sync_profile()
        if update == 1 or ((update - 1) % self.config.worker_sync_interval == 0):
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
                worker.sync_actor(actor_state_dict, version=actor_version)
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
                    for _ in range(self.config.gradient_steps_per_update):
                        learner_metrics = self.learner.train_step(global_env_steps=int(self.global_env_steps))
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
                    for _ in range(self.config.gradient_steps_per_update):
                        learner_metrics = self.learner.train_step(global_env_steps=int(self.global_env_steps))
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
