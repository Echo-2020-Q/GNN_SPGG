from __future__ import annotations

import copy

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
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.replay_buffer = ReplayBuffer(config.replay_capacity, seed=config.seed)

        critic_hidden_dim = int(getattr(policy.config, "hidden_dim", 64))
        critic_config = GraphActionCriticConfig(
            state_hidden_dim=critic_hidden_dim,
            action_hidden_dim=critic_hidden_dim,
            pool_hidden_dim=critic_hidden_dim,
            q_hidden_dim=critic_hidden_dim,
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

        eval_factory = RandomizedEnvFactory.from_env(eval_env or env)
        self.evaluator = GraphTD3Evaluator(
            env_factories=[eval_factory],
            config=EvalConfig(
                num_episodes=config.eval_episodes,
                collapse_resource_threshold=config.collapse_resource_threshold,
            ),
            device=config.device,
        )

    def train(self, num_updates: int | None = None) -> list[dict[str, float]]:
        total_updates = int(num_updates or self.config.total_updates)
        history: list[dict[str, float]] = []

        for update in range(1, total_updates + 1):
            if update == 1 or ((update - 1) % self.config.worker_sync_interval == 0):
                actor_state_dict, actor_version = self.learner.publish_actor_state()
                for worker in self.workers:
                    worker.sync_actor(actor_state_dict, version=actor_version)

            rollout_metrics = [worker.collect(self.config.steps_per_update) for worker in self.workers]
            mean_rollout_reward = float(np.mean([item["mean_reward"] for item in rollout_metrics])) if rollout_metrics else 0.0

            learner_metrics = {
                "critic1_loss": 0.0,
                "critic2_loss": 0.0,
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "loss": 0.0,
                "replay_size": float(len(self.replay_buffer)),
            }
            if update % self.config.train_every == 0:
                for _ in range(self.config.gradient_steps_per_update):
                    learner_metrics = self.learner.train_step()

            metrics = {
                "update": float(update),
                "loss": float(learner_metrics["loss"]),
                "policy_loss": float(learner_metrics["actor_loss"]),
                "value_loss": float(learner_metrics["critic_loss"]),
                "entropy": 0.0,
                "critic1_loss": float(learner_metrics["critic1_loss"]),
                "critic2_loss": float(learner_metrics["critic2_loss"]),
                "critic_loss": float(learner_metrics["critic_loss"]),
                "actor_loss": float(learner_metrics["actor_loss"]),
                "replay_size": float(learner_metrics["replay_size"]),
                "mean_rollout_reward": mean_rollout_reward,
            }

            if update % self.config.eval_interval == 0 or update == total_updates:
                evaluation = self.evaluate(self.config.eval_episodes)
                metrics["eval_return_mean"] = evaluation["return_mean"]
                metrics["eval_cooperation_mean"] = evaluation["cooperation_mean"]
                metrics["eval_gini_mean"] = evaluation["gini_mean"]
                metrics["eval_mean_total_resource"] = evaluation["mean_total_resource"]
                metrics["eval_collapse_rate"] = evaluation["collapse_rate"]

            history.append(metrics)

        return history

    def evaluate(self, num_episodes: int | None = None) -> dict[str, float]:
        return self.evaluator.evaluate(self.learner.actor, num_episodes=num_episodes)
