from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from Project1.env import SPGGEnv
from Project1.policies.gnn_rl import GNNAllocationPolicy


@dataclass
class TrainerConfig:
    total_updates: int = 100
    steps_per_update: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    eval_interval: int = 10
    eval_episodes: int = 3
    device: str = "cpu"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.total_updates <= 0:
            raise ValueError("total_updates must be positive.")
        if self.steps_per_update <= 0:
            raise ValueError("steps_per_update must be positive.")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive.")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive.")


class CentralizedActorCriticTrainer:
    def __init__(
        self,
        env: SPGGEnv,
        policy: GNNAllocationPolicy,
        config: TrainerConfig,
        eval_env: SPGGEnv | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.config = config
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)

        self.eval_env = eval_env or SPGGEnv(
            env.config,
            {node: list(neighbors) for node, neighbors in enumerate(env.graph.neighbors)},
        )
        self._observation = self.env.reset(seed=config.seed)

    def train(self, num_updates: int | None = None) -> list[dict[str, float]]:
        total_updates = num_updates or self.config.total_updates
        history: list[dict[str, float]] = []

        for update in range(1, total_updates + 1):
            self.policy.train()
            batch = self.collect_rollout()

            advantages = batch["advantages"]
            advantage_std = advantages.std(unbiased=False).clamp_min(1e-8)
            normalized_advantages = (advantages - advantages.mean()) / advantage_std

            policy_loss = -(normalized_advantages.detach() * batch["log_probs"]).mean()
            value_loss = F.mse_loss(batch["values"], batch["returns"].detach())
            entropy_bonus = batch["entropies"].mean()
            loss = policy_loss + (self.config.value_coef * value_loss) - (self.config.entropy_coef * entropy_bonus)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            metrics = {
                "update": float(update),
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy_bonus.item()),
                "mean_rollout_reward": float(batch["rewards"].mean().item()),
                "mean_episode_return": float(np.mean(batch["episode_returns"])) if batch["episode_returns"] else 0.0,
            }

            if update % self.config.eval_interval == 0 or update == total_updates:
                evaluation = self.evaluate(self.config.eval_episodes)
                metrics["eval_return_mean"] = evaluation["return_mean"]
                metrics["eval_cooperation_mean"] = evaluation["cooperation_mean"]
                metrics["eval_gini_mean"] = evaluation["gini_mean"]

            history.append(metrics)

        return history

    def collect_rollout(self) -> dict[str, torch.Tensor | list[float]]:
        rewards: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        episode_returns: list[float] = []
        running_episode_return = 0.0

        for _ in range(self.config.steps_per_update):
            action_output = self.policy.sample_action(self._observation)
            next_observation, reward, done, _ = self.env.step(
                action_output.allocation_matrix.detach().cpu().numpy()
            )

            rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            values.append(action_output.value.squeeze())
            log_prob = action_output.log_prob if action_output.log_prob is not None else torch.zeros((), device=self.device)
            entropy = action_output.entropy if action_output.entropy is not None else torch.zeros((), device=self.device)
            log_probs.append(log_prob.squeeze())
            entropies.append(entropy.squeeze())
            dones.append(torch.tensor(float(done), dtype=torch.float32, device=self.device))

            running_episode_return += reward
            self._observation = next_observation
            if done:
                episode_returns.append(running_episode_return)
                running_episode_return = 0.0
                self._observation = self.env.reset()

        with torch.no_grad():
            bootstrap_value = self.policy.evaluate_value(self._observation).squeeze()

        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)
        dones_tensor = torch.stack(dones)
        returns_tensor, advantages_tensor = self._compute_returns_and_advantages(
            rewards_tensor,
            values_tensor,
            dones_tensor,
            bootstrap_value,
        )

        return {
            "rewards": rewards_tensor,
            "values": values_tensor,
            "log_probs": log_probs_tensor,
            "entropies": entropies_tensor,
            "returns": returns_tensor,
            "advantages": advantages_tensor,
            "episode_returns": episode_returns,
        }

    def evaluate(self, num_episodes: int = 3) -> dict[str, float]:
        self.policy.eval()

        returns: list[float] = []
        cooperation_rates: list[float] = []
        gini_values: list[float] = []

        for episode in range(num_episodes):
            seed = None if self.config.seed is None else self.config.seed + episode + 1
            observation = self.eval_env.reset(seed=seed)
            done = False
            episode_return = 0.0
            last_info: dict[str, float] | None = None

            while not done:
                with torch.no_grad():
                    action_output = self.policy.deterministic_action(observation)
                observation, reward, done, info = self.eval_env.step(
                    action_output.allocation_matrix.detach().cpu().numpy()
                )
                episode_return += reward
                last_info = info

            returns.append(episode_return)
            cooperation_rates.append(
                float(last_info["actual_cooperation_rate"]) if last_info is not None else float(observation["x_actual"].mean())
            )
            gini_values.append(float(last_info["gini"]) if last_info is not None else 0.0)

        return {
            "return_mean": float(np.mean(returns)),
            "cooperation_mean": float(np.mean(cooperation_rates)),
            "gini_mean": float(np.mean(gini_values)),
        }

    def _compute_returns_and_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        bootstrap_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), device=self.device)
        next_value = bootstrap_value

        for index in range(rewards.size(0) - 1, -1, -1):
            mask = 1.0 - dones[index]
            delta = rewards[index] + (self.config.gamma * next_value * mask) - values[index]
            gae = delta + (self.config.gamma * self.config.gae_lambda * mask * gae)
            advantages[index] = gae
            next_value = values[index]

        returns = advantages + values
        return returns, advantages
