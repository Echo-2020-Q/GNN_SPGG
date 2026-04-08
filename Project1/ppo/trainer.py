from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from Project1.env import SPGGEnv
from Project1.policies.gnn_rl import GNNAllocationPolicy
from Project1.td3 import DomainRandomizationConfig, EvalConfig
from Project1.td3.data import REPLAY_OBSERVATION_DTYPES
from Project1.td3.evaluator import GraphTD3Evaluator
from Project1.td3.worker import RandomizedEnvFactory

from .buffer import RunningMeanStd, RolloutTrajectoryBuffer
from .config import GraphPPOConfig
from .learner import GraphPPOLearner


def _stack_observations(observations: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("observations must contain at least one item.")
    return {
        key: torch.stack(
            [torch.as_tensor(observation[key], dtype=dtype) for observation in observations],
            dim=0,
        )
        for key, dtype in REPLAY_OBSERVATION_DTYPES.items()
    }


class GraphPPOTrainer:
    def __init__(
        self,
        env: SPGGEnv,
        policy: GNNAllocationPolicy,
        config: GraphPPOConfig,
        eval_env: SPGGEnv | None = None,
        randomization: DomainRandomizationConfig | None = None,
        eval_env_factories: Sequence[RandomizedEnvFactory] | None = None,
        curriculum_stages: Sequence[Mapping[str, Any]] | None = None,
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.device = torch.device(config.device)
        if config.seed is not None:
            np.random.seed(int(config.seed))
            torch.manual_seed(int(config.seed))

        self.train_factory = RandomizedEnvFactory.from_env(env, randomization=randomization)
        self._validate_num_nodes_support(self.train_factory)
        self.default_eval_env_factories = list(eval_env_factories) if eval_env_factories is not None else [
            RandomizedEnvFactory.from_env(eval_env or env)
        ]
        self.evaluator = GraphTD3Evaluator(
            env_factories=self.default_eval_env_factories,
            config=EvalConfig(
                num_episodes=config.eval_episodes,
                collapse_resource_threshold=config.collapse_resource_threshold,
            ),
            device=config.device,
        )
        self.curriculum_stages = [dict(stage) for stage in curriculum_stages] if curriculum_stages is not None else []
        self.active_curriculum_stage_index: int | None = None
        self.learner = GraphPPOLearner(actor=self.policy, config=config)
        self.history: list[dict[str, float]] = []
        self.completed_updates = 0
        self.global_env_steps = 0
        self.demo_pretrain_completed = False
        self.demo_pretrain_summary: dict[str, float | str | bool | None] | None = None

        self.reward_normalizer = RunningMeanStd() if bool(config.ppo_reward_normalization) else None
        self.rng = np.random.default_rng(config.seed)
        self.total_envs = int(config.num_workers) * int(config.num_envs_per_worker)
        self.envs: list[SPGGEnv] = []
        self.env_metadatas: list[dict[str, Any]] = []
        self.observations: list[dict[str, np.ndarray]] = []
        self._reset_all_envs()

    @staticmethod
    def _validate_num_nodes_support(factory: RandomizedEnvFactory) -> None:
        randomization = factory.randomization
        if randomization.enabled and len(set(int(item) for item in randomization.num_nodes_choices)) > 1:
            raise ValueError(
                "PPO currently requires a fixed num_nodes across rollout environments. "
                "Use one value in domain_randomization.num_nodes_choices."
            )

    def _sample_env(self) -> tuple[SPGGEnv, dict[str, Any]]:
        return self.train_factory.sample_environment(self.rng)

    def _reset_env_slot(self, slot: int) -> None:
        sampled_env, metadata = self._sample_env()
        observation = sampled_env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        if slot < len(self.envs):
            self.envs[slot] = sampled_env
            self.env_metadatas[slot] = dict(metadata)
            self.observations[slot] = observation
        else:
            self.envs.append(sampled_env)
            self.env_metadatas.append(dict(metadata))
            self.observations.append(observation)

    def _reset_all_envs(self) -> None:
        self.envs = []
        self.env_metadatas = []
        self.observations = []
        for slot in range(self.total_envs):
            self._reset_env_slot(slot)

    def _resolve_curriculum_stage(self, update: int) -> tuple[int, Mapping[str, Any]] | None:
        active_stage: tuple[int, Mapping[str, Any]] | None = None
        for stage in self.curriculum_stages:
            if int(stage["activate_at_update"]) <= int(update):
                active_stage = (int(stage["stage_index"]), stage)
        return active_stage

    def _apply_curriculum_stage(self, update: int) -> None:
        resolved = self._resolve_curriculum_stage(update)
        if resolved is None:
            return
        stage_index, stage = resolved
        if self.active_curriculum_stage_index == stage_index:
            return
        self.active_curriculum_stage_index = stage_index
        self.train_factory = RandomizedEnvFactory.from_env(
            self.env,
            randomization=stage.get("train_randomization"),
        )
        self._validate_num_nodes_support(self.train_factory)
        evaluator_factories = stage.get("eval_env_factories") or self.default_eval_env_factories
        self.evaluator = GraphTD3Evaluator(
            env_factories=evaluator_factories,
            config=EvalConfig(
                num_episodes=self.config.eval_episodes,
                collapse_resource_threshold=self.config.collapse_resource_threshold,
            ),
            device=self.config.device,
        )
        self._reset_all_envs()

    def _collect_rollout(self) -> tuple[RolloutTrajectoryBuffer, dict[str, float]]:
        self.policy.eval()
        buffer = RolloutTrajectoryBuffer()
        rewards: list[float] = []
        cooperation_rates: list[float] = []
        mean_resources: list[float] = []
        gini_values: list[float] = []
        mean_payoffs: list[float] = []
        mean_pool_growns: list[float] = []
        mean_pool_raws: list[float] = []

        for _ in range(int(self.config.steps_per_update)):
            observation_batch = _stack_observations(self.observations)
            with torch.no_grad():
                action_output = self.policy.sample_action_tensor_batch(observation_batch)
            if action_output.log_prob is None:
                raise RuntimeError("PPO rollout sampling requires policy.sample_action_tensor_batch() to return log_prob.")
            actions = action_output.allocation_matrix.detach().cpu().numpy()
            reward_batch = np.zeros(self.total_envs, dtype=np.float32)
            done_batch = np.zeros(self.total_envs, dtype=np.float32)

            for env_index in range(self.total_envs):
                current_observation = self.observations[env_index]
                next_observation, reward, done, info = self.envs[env_index].step(actions[env_index])
                reward_batch[env_index] = float(reward)
                done_batch[env_index] = float(done)
                rewards.append(float(reward))
                cooperation_rates.append(float(info.get("actual_cooperation_rate", 0.0)))
                gini_values.append(float(info.get("gini", 0.0)))
                reward_components = dict(info.get("reward_components", {}))
                mean_resources.append(float(reward_components.get("mean_resource_next", 0.0)))
                mean_payoffs.append(float(reward_components.get("mean_payoff", 0.0)))
                mean_pool_growns.append(float(np.asarray(current_observation["pool_grown"]).mean()))
                mean_pool_raws.append(float(np.asarray(current_observation["pool_raw"]).mean()))
                if done:
                    self._reset_env_slot(env_index)
                else:
                    self.observations[env_index] = next_observation

            buffer.add(
                observation_batch=observation_batch,
                action_batch=action_output.allocation_matrix.detach().cpu(),
                log_prob_batch=action_output.log_prob.detach().cpu(),
                reward_batch=torch.as_tensor(reward_batch, dtype=torch.float32),
                done_batch=torch.as_tensor(done_batch, dtype=torch.float32),
                value_batch=action_output.value.detach().cpu(),
            )

        self.global_env_steps += int(self.config.steps_per_update) * self.total_envs
        rollout_metrics = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_actual_cooperation_rate": float(np.mean(cooperation_rates)) if cooperation_rates else 0.0,
            "mean_resource": float(np.mean(mean_resources)) if mean_resources else 0.0,
            "mean_gini": float(np.mean(gini_values)) if gini_values else 0.0,
            "mean_payoff": float(np.mean(mean_payoffs)) if mean_payoffs else 0.0,
            "mean_pool_grown": float(np.mean(mean_pool_growns)) if mean_pool_growns else 0.0,
            "mean_pool_raw": float(np.mean(mean_pool_raws)) if mean_pool_raws else 0.0,
            "steps_collected": float(int(self.config.steps_per_update) * self.total_envs),
        }
        return buffer, rollout_metrics

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
            "rng_state": self.rng.bit_generator.state,
            "reward_normalizer_state": None if self.reward_normalizer is None else self.reward_normalizer.state_dict(),
            "demo_pretrain_completed": False,
            "demo_pretrain_summary": None,
        }
        return payload

    def load_checkpoint(self, checkpoint: Mapping[str, Any]) -> str:
        self.learner.load_checkpoint_state(dict(checkpoint["learner_state"]))
        self.completed_updates = int(checkpoint.get("completed_updates", checkpoint.get("update", 0)))
        self.global_env_steps = int(checkpoint.get("global_env_steps", 0))
        self.history = [dict(item) for item in checkpoint.get("history", [])]
        active_stage_index = checkpoint.get("active_curriculum_stage_index")
        self.active_curriculum_stage_index = None if active_stage_index is None else int(active_stage_index)
        reward_normalizer_state = checkpoint.get("reward_normalizer_state")
        if reward_normalizer_state is not None:
            if self.reward_normalizer is None:
                self.reward_normalizer = RunningMeanStd()
            self.reward_normalizer.load_state_dict(dict(reward_normalizer_state))
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self._reset_all_envs()
        checkpoint_mode = str(checkpoint.get("checkpoint_mode", "lightweight"))
        if self.active_curriculum_stage_index is not None:
            for stage in self.curriculum_stages:
                if int(stage.get("stage_index", -1)) == self.active_curriculum_stage_index:
                    self.train_factory = RandomizedEnvFactory.from_env(
                        self.env,
                        randomization=stage.get("train_randomization"),
                    )
                    evaluator_factories = stage.get("eval_env_factories") or self.default_eval_env_factories
                    self.evaluator = GraphTD3Evaluator(
                        env_factories=evaluator_factories,
                        config=EvalConfig(
                            num_episodes=self.config.eval_episodes,
                            collapse_resource_threshold=self.config.collapse_resource_threshold,
                        ),
                        device=self.config.device,
                    )
                    self._reset_all_envs()
                    break
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
            rollout_collect_start = perf_counter()
            rollout_buffer, rollout_metrics = self._collect_rollout()
            rollout_collect_seconds = float(perf_counter() - rollout_collect_start)

            bootstrap_observations = _stack_observations(self.observations)
            with torch.no_grad():
                bootstrap_output = self.policy.forward_tensor_batch(bootstrap_observations)
            rollout_batch = rollout_buffer.build_batch(
                last_values=bootstrap_output.value.detach().cpu(),
                gamma=float(self.config.gamma),
                gae_lambda=float(self.config.ppo_gae_lambda),
                reward_normalizer=self.reward_normalizer,
            )

            learner_update_start = perf_counter()
            learner_metrics = self.learner.update(rollout_batch.flatten())
            learner_update_seconds = float(perf_counter() - learner_update_start)

            metrics = {
                "update": float(update),
                "loss": float(learner_metrics["loss"]),
                "policy_loss": float(learner_metrics["policy_loss"]),
                "value_loss": float(learner_metrics["value_loss"]),
                "entropy": float(learner_metrics["entropy"]),
                "ppo_policy_loss": float(learner_metrics["ppo_policy_loss"]),
                "ppo_value_loss": float(learner_metrics["ppo_value_loss"]),
                "ppo_entropy": float(learner_metrics["ppo_entropy"]),
                "ppo_approx_kl": float(learner_metrics["ppo_approx_kl"]),
                "ppo_clipfrac": float(learner_metrics["ppo_clipfrac"]),
                "explained_variance": float(learner_metrics["explained_variance"]),
                "actor_lr": float(learner_metrics["actor_lr"]),
                "critic_lr": float(learner_metrics["critic_lr"]),
                "actor_grad_norm_pre_clip": float(learner_metrics["actor_grad_norm_pre_clip"]),
                "actor_grad_norm_post_clip": float(learner_metrics["actor_grad_norm_post_clip"]),
                "critic_grad_norm_pre_clip": float(learner_metrics["critic_grad_norm_pre_clip"]),
                "critic_grad_norm_post_clip": float(learner_metrics["critic_grad_norm_post_clip"]),
                "mean_rollout_reward": float(rollout_metrics["mean_reward"]),
                "rollout_f_c": float(rollout_metrics["mean_actual_cooperation_rate"]),
                "rollout_R_mean": float(rollout_metrics["mean_resource"]),
                "rollout_gini": float(rollout_metrics["mean_gini"]),
                "rollout_payoff_mean": float(rollout_metrics["mean_payoff"]),
                "rollout_pool_grown_mean": float(rollout_metrics["mean_pool_grown"]),
                "rollout_pool_mean": float(rollout_metrics["mean_pool_raw"]),
                "curriculum_stage": float(self.active_curriculum_stage_index or 0),
                "behavior_frac_actor_logits": 1.0,
                "profile_rollout_collect_seconds": rollout_collect_seconds,
                "profile_learner_update_seconds": learner_update_seconds,
                "global_env_steps": float(self.global_env_steps),
            }

            if update % int(self.config.eval_interval) == 0 or update == total_updates:
                evaluation = self.evaluate(self.config.eval_episodes)
                metrics["eval_return_mean"] = float(evaluation["return_mean"])
                metrics["eval_cooperation_mean"] = float(evaluation["cooperation_mean"])
                metrics["eval_gini_mean"] = float(evaluation["gini_mean"])
                metrics["eval_mean_total_resource"] = float(evaluation["mean_total_resource"])
                metrics["eval_collapse_rate"] = float(evaluation["collapse_rate"])
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

    def close(self) -> None:
        return None
