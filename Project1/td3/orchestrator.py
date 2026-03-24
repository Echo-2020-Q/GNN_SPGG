from __future__ import annotations

import copy
from dataclasses import asdict
from collections import Counter
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
from .worker import RandomizedEnvFactory, RolloutWorker


class GraphTD3Trainer:
    """Single-process orchestrator for the approved Graph-TD3 training skeleton.

    The design intentionally mirrors the future multi-worker setup, but keeps execution
    local and sequential for now. Workers maintain actor snapshots, write into a shared
    replay buffer, and the centralized learner performs TD3 updates.
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
        self.workers = [
            RolloutWorker(
                actor=copy.deepcopy(policy),
                replay_buffer=self.replay_buffer,
                explorer=self.rollout_explorer,
                env_factory=train_factory,
                config=WorkerConfig(
                    worker_id=worker_id,
                    seed=(config.seed or 0) + worker_id,
                    rollout_steps_per_sync=config.steps_per_update,
                ),
                train_config=config,
                device="cpu",
            )
            for worker_id in range(config.num_workers)
        ]

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
        self.history: list[dict[str, float]] = []

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
            for worker in self.workers:
                worker.sync_actor(actor_state_dict, version=actor_version)

        resolved_stage = self._resolve_curriculum_stage(max(1, self.completed_updates + 1))
        if resolved_stage is not None:
            stage_index, stage = resolved_stage
            self._activate_curriculum_stage(max(1, self.completed_updates + 1), stage_index, stage, reset_workers=False)

        return checkpoint_mode

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
                for worker in self.workers:
                    worker.sync_actor(actor_state_dict, version=actor_version)

            rollout_metrics = [worker.collect(worker.config.rollout_steps_per_sync) for worker in self.workers]
            mean_rollout_reward = float(np.mean([item["mean_reward"] for item in rollout_metrics])) if rollout_metrics else 0.0
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
            if update % self.config.train_every == 0:
                for _ in range(self.config.gradient_steps_per_update):
                    learner_metrics = self.learner.train_step()

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
