from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from Project1.policies.gnn_rl import GNNAllocationPolicy

from .config import EvalConfig
from .worker import RandomizedEnvFactory


class GraphTD3Evaluator:
    def __init__(
        self,
        env_factories: Sequence[RandomizedEnvFactory],
        config: EvalConfig,
        device: torch.device | str = "cpu",
    ):
        self.env_factories = list(env_factories)
        self.config = config
        self.device = torch.device(device)

    def evaluate(self, actor: GNNAllocationPolicy, num_episodes: int | None = None) -> dict[str, float]:
        actor = actor.to(self.device)
        actor.eval()

        episode_budget = int(num_episodes or self.config.num_episodes)
        returns: list[float] = []
        mean_resources: list[float] = []
        mean_total_resources: list[float] = []
        mean_payoffs: list[float] = []
        mean_pool_raws: list[float] = []
        mean_pool_growns: list[float] = []
        cooperation_rates: list[float] = []
        gini_values: list[float] = []
        collapse_flags: list[float] = []

        per_network_mean_resources: dict[str, list[float]] = {}
        per_network_returns: dict[str, list[float]] = {}
        per_network_mean_total_resources: dict[str, list[float]] = {}
        per_network_mean_payoffs: dict[str, list[float]] = {}
        per_network_mean_pool_raws: dict[str, list[float]] = {}
        per_network_mean_pool_growns: dict[str, list[float]] = {}
        per_network_cooperation_rates: dict[str, list[float]] = {}
        per_network_gini_values: dict[str, list[float]] = {}
        per_network_collapse_flags: dict[str, list[float]] = {}

        for factory in self.env_factories:
            rng = np.random.default_rng(0)
            for _ in range(episode_budget):
                env, metadata = factory.sample_environment(rng)
                observation = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                done = False
                episode_return = 0.0
                resource_mean_trace: list[float] = []
                resource_trace: list[float] = []
                payoff_mean_trace: list[float] = []
                pool_raw_mean_trace: list[float] = []
                pool_grown_mean_trace: list[float] = []
                final_info: dict[str, float] | None = None

                while not done:
                    resource_mean_trace.append(float(np.asarray(observation["resources"]).mean()))
                    resource_trace.append(float(np.asarray(observation["resources"]).sum()))
                    pool_raw_mean_trace.append(float(np.asarray(observation["pool_raw"]).mean()))
                    pool_grown_mean_trace.append(float(np.asarray(observation["pool_grown"]).mean()))
                    with torch.no_grad():
                        action_output = actor.deterministic_action(observation)
                    observation, reward, done, info = env.step(action_output.allocation_matrix.detach().cpu().numpy())
                    episode_return += float(reward)
                    if info is not None and "payoff" in info:
                        payoff_mean_trace.append(float(np.asarray(info["payoff"]).mean()))
                    final_info = info

                final_mean_resource = float(np.asarray(observation["resources"]).mean())
                returns.append(episode_return)
                mean_resource = float(np.mean(resource_mean_trace)) if resource_mean_trace else 0.0
                mean_total_resource = float(np.mean(resource_trace)) if resource_trace else 0.0
                mean_payoff = float(np.mean(payoff_mean_trace)) if payoff_mean_trace else 0.0
                mean_pool_raw = float(np.mean(pool_raw_mean_trace)) if pool_raw_mean_trace else 0.0
                mean_pool_grown = float(np.mean(pool_grown_mean_trace)) if pool_grown_mean_trace else 0.0
                cooperation_rate = (
                    float(final_info["actual_cooperation_rate"])
                    if final_info is not None
                    else float(np.asarray(observation["x_actual"]).mean())
                )
                gini_value = float(final_info["gini"]) if final_info is not None else 0.0
                collapse_flag = float(final_mean_resource <= self.config.collapse_resource_threshold)

                mean_resources.append(mean_resource)
                mean_total_resources.append(mean_total_resource)
                mean_payoffs.append(mean_payoff)
                mean_pool_raws.append(mean_pool_raw)
                mean_pool_growns.append(mean_pool_grown)
                cooperation_rates.append(cooperation_rate)
                gini_values.append(gini_value)
                collapse_flags.append(collapse_flag)

                network_type = str(metadata.get("network_type", "unknown"))
                per_network_returns.setdefault(network_type, []).append(episode_return)
                per_network_mean_resources.setdefault(network_type, []).append(mean_resource)
                per_network_mean_total_resources.setdefault(network_type, []).append(mean_total_resource)
                per_network_mean_payoffs.setdefault(network_type, []).append(mean_payoff)
                per_network_mean_pool_raws.setdefault(network_type, []).append(mean_pool_raw)
                per_network_mean_pool_growns.setdefault(network_type, []).append(mean_pool_grown)
                per_network_cooperation_rates.setdefault(network_type, []).append(cooperation_rate)
                per_network_gini_values.setdefault(network_type, []).append(gini_value)
                per_network_collapse_flags.setdefault(network_type, []).append(collapse_flag)

        metrics = {
            "return_mean": float(np.mean(returns)) if returns else 0.0,
            "mean_resource": float(np.mean(mean_resources)) if mean_resources else 0.0,
            "mean_total_resource": float(np.mean(mean_total_resources)) if mean_total_resources else 0.0,
            "mean_payoff": float(np.mean(mean_payoffs)) if mean_payoffs else 0.0,
            "mean_pool_raw": float(np.mean(mean_pool_raws)) if mean_pool_raws else 0.0,
            "mean_pool_grown": float(np.mean(mean_pool_growns)) if mean_pool_growns else 0.0,
            "cooperation_mean": float(np.mean(cooperation_rates)) if cooperation_rates else 0.0,
            "gini_mean": float(np.mean(gini_values)) if gini_values else 0.0,
            "collapse_rate": float(np.mean(collapse_flags)) if collapse_flags else 0.0,
            "sustainability_rate": float(1.0 - np.mean(collapse_flags)) if collapse_flags else 0.0,
        }
        for network_type, values in per_network_returns.items():
            metrics["return_mean/{0}".format(network_type)] = float(np.mean(values))
        for network_type, values in per_network_mean_resources.items():
            metrics["mean_resource/{0}".format(network_type)] = float(np.mean(values))
        for network_type, values in per_network_mean_total_resources.items():
            metrics["mean_total_resource/{0}".format(network_type)] = float(np.mean(values))
        for network_type, values in per_network_mean_payoffs.items():
            metrics["mean_payoff/{0}".format(network_type)] = float(np.mean(values))
        for network_type, values in per_network_mean_pool_raws.items():
            metrics["mean_pool_raw/{0}".format(network_type)] = float(np.mean(values))
        for network_type, values in per_network_mean_pool_growns.items():
            metrics["mean_pool_grown/{0}".format(network_type)] = float(np.mean(values))
        for network_type, values in per_network_cooperation_rates.items():
            metrics["cooperation_mean/{0}".format(network_type)] = float(np.mean(values))
        for network_type, values in per_network_gini_values.items():
            metrics["gini_mean/{0}".format(network_type)] = float(np.mean(values))
        for network_type, values in per_network_collapse_flags.items():
            metrics["collapse_rate/{0}".format(network_type)] = float(np.mean(values))
        return metrics
