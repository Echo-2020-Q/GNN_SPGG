from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

from Project1.env import (
    RewardConfig,
    SPGGConfig,
    SPGGEnv,
    make_barabasi_albert_graph,
    make_erdos_renyi_graph,
    make_grid_graph,
    make_random_regular_graph,
    make_watts_strogatz_graph,
)
from Project1.policies.rule_based import ProportionalContributionPolicy, UniformAllocationPolicy
from analyze_gnn_policy_decisions import load_actor_from_run_dir, run_deterministic_rollouts


SCRIPT_DEFAULTS = {
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    "checkpoint_name": "best_eval.pt",
    "output_dir": None,
    "output_subdir_name": "0420_prop_equal_mix_ego_resource_scatter",
    "topologies": "regular",
    "episodes": 5,
    "max_steps": 200,
    "episode_length_override": 200,
    "env_r_override": None,
    "seed": None,
    "analysis_mode": "agent_fit",
    "num_weights": 21,
    "weights": None,
    "device": "cpu",
    "max_scatter_points_per_topology": 20000,
}

TOPOLOGY_ALIASES = {
    "original": "original",
    "orig": "original",
    "regular": "regular",
    "reg": "regular",
    "er": "erdos_renyi",
    "erdos_renyi": "erdos_renyi",
    "erdos-renyi": "erdos_renyi",
    "ws": "small_world",
    "small_world": "small_world",
    "small-world": "small_world",
    "ba": "scale_free",
    "scale_free": "scale_free",
    "scale-free": "scale_free",
}
TOPOLOGY_LABELS = {
    "original": "Original",
    "regular": "Regular",
    "erdos_renyi": "ER",
    "small_world": "WS",
    "scale_free": "BA",
}


@dataclass
class RunningMoments:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update(self, value: float) -> None:
        if not np.isfinite(value):
            return
        self.count += 1
        self.total += float(value)
        self.total_sq += float(value) * float(value)

    @property
    def mean(self) -> float:
        if self.count <= 0:
            return float("nan")
        return float(self.total / float(self.count))

    @property
    def std(self) -> float:
        if self.count <= 0:
            return float("nan")
        mean = self.mean
        variance = max((self.total_sq / float(self.count)) - (mean * mean), 0.0)
        return float(math.sqrt(variance))

    @property
    def sem(self) -> float:
        if self.count <= 0:
            return float("nan")
        return float(self.std / math.sqrt(float(self.count)))


class PropEqualMixAllocationPolicy:
    """Allocate by w * proportional + (1 - w) * equal, where w is the proportional weight."""

    def __init__(self, weight_proportional: float):
        if not 0.0 <= weight_proportional <= 1.0:
            raise ValueError("weight_proportional must be in [0, 1].")
        self.weight_proportional = float(weight_proportional)
        self._proportional = ProportionalContributionPolicy()
        self._equal = UniformAllocationPolicy()

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        proportional = np.asarray(self._proportional.allocate(observation), dtype=np.float64)
        equal = np.asarray(self._equal.allocate(observation), dtype=np.float64)
        local_mask = np.asarray(observation["local_mask"], dtype=bool)

        allocation = (self.weight_proportional * proportional) + ((1.0 - self.weight_proportional) * equal)
        allocation = np.where(local_mask, allocation, 0.0)
        row_sums = allocation.sum(axis=1, keepdims=True)
        return np.divide(
            allocation,
            row_sums,
            out=np.zeros_like(allocation, dtype=np.float64),
            where=row_sums > 1e-12,
        )

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def canonicalize_topology_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in TOPOLOGY_ALIASES:
        raise ValueError(f"Unsupported topology name: {name}")
    return TOPOLOGY_ALIASES[normalized]


def parse_topology_list(raw_value: str) -> list[str]:
    if not raw_value.strip():
        raise ValueError("Topology list cannot be empty.")
    topologies: list[str] = []
    for item in raw_value.split(","):
        topology = canonicalize_topology_name(item)
        if topology not in topologies:
            topologies.append(topology)
    return topologies


def _resolve_er_edge_prob(network: Mapping[str, Any]) -> float:
    explicit_prob = network.get("er_edge_prob")
    if explicit_prob is not None:
        return float(explicit_prob)
    num_nodes = int(network["num_nodes"])
    target_mean_degree = float(network.get("er_target_mean_degree", 0.0))
    if num_nodes <= 1:
        return 0.0
    return float(target_mean_degree / float(num_nodes - 1))


def _resolve_even_degree(target_mean_degree: float, num_nodes: int) -> int:
    if num_nodes <= 2:
        return max(0, num_nodes - 1)
    degree = int(round(target_mean_degree))
    degree = max(2, min(num_nodes - 1, degree))
    if degree % 2 != 0:
        if degree < num_nodes - 1:
            degree += 1
        else:
            degree -= 1
    return max(2, min(num_nodes - 1, degree))


def _resolve_target_mean_degree(spec: Mapping[str, Any], graph: Mapping[int, Sequence[int]]) -> float:
    network = spec["network"]
    network_type = str(network["type"])
    if network_type == "regular":
        return float(network["regular_degree"])
    if network_type == "erdos_renyi":
        if network.get("er_target_mean_degree") is not None:
            return float(network["er_target_mean_degree"])
        num_nodes = int(network["num_nodes"])
        return float(_resolve_er_edge_prob(network) * max(num_nodes - 1, 0))
    if network_type == "small_world":
        return float(network["ws_degree"])
    if network_type == "scale_free":
        num_nodes = int(network["num_nodes"])
        attachments = float(network["ba_attachments_per_new_node"])
        return float((2.0 * attachments) - ((attachments * (attachments + 1.0)) / max(num_nodes, 1)))
    if network_type == "grid":
        degrees = [len(neighbors) for neighbors in graph.values()]
        return float(np.mean(degrees)) if degrees else 0.0
    raise ValueError(f"Unsupported network type: {network_type}")


def build_graph_from_spec(spec: Mapping[str, Any]) -> dict[int, list[int]]:
    network = spec["network"]
    network_type = str(network["type"])
    seed = int(spec["seed"])

    if network_type == "regular":
        return make_random_regular_graph(
            num_nodes=int(network["num_nodes"]),
            degree=int(network["regular_degree"]),
            seed=seed,
        )
    if network_type == "erdos_renyi":
        return make_erdos_renyi_graph(
            num_nodes=int(network["num_nodes"]),
            edge_prob=_resolve_er_edge_prob(network),
            seed=seed,
        )
    if network_type == "small_world":
        return make_watts_strogatz_graph(
            num_nodes=int(network["num_nodes"]),
            degree=int(network["ws_degree"]),
            rewiring_prob=float(network["ws_rewiring_prob"]),
            seed=seed,
        )
    if network_type == "scale_free":
        return make_barabasi_albert_graph(
            num_nodes=int(network["num_nodes"]),
            attachments_per_new_node=int(network["ba_attachments_per_new_node"]),
            seed=seed,
        )
    if network_type == "grid":
        return make_grid_graph(
            rows=int(network["grid_rows"]),
            cols=int(network["grid_cols"]),
            periodic=bool(network.get("grid_periodic", False)),
        )
    raise ValueError(f"Unsupported network type: {network_type}")


def resolve_reference_mean_degree(base_spec: Mapping[str, Any]) -> float:
    graph = build_graph_from_spec(base_spec)
    return _resolve_target_mean_degree(base_spec, graph)


def build_spec_for_topology(
    base_spec: Mapping[str, Any],
    topology_name: str,
    *,
    reference_mean_degree: float | None = None,
) -> dict[str, Any]:
    topology = canonicalize_topology_name(topology_name)
    if topology == "original":
        return deepcopy(dict(base_spec))

    spec = deepcopy(dict(base_spec))
    network = spec["network"]
    num_nodes = int(network["num_nodes"])
    target_mean_degree = (
        float(reference_mean_degree)
        if reference_mean_degree is not None
        else resolve_reference_mean_degree(base_spec)
    )

    network["type"] = topology
    if topology == "regular":
        degree = int(network.get("regular_degree", max(1, int(round(target_mean_degree)))))
        degree = max(0, min(num_nodes - 1, degree))
        if (num_nodes * degree) % 2 != 0:
            degree = max(0, degree - 1)
        network["regular_degree"] = degree
    elif topology == "erdos_renyi":
        network["er_target_mean_degree"] = float(network.get("er_target_mean_degree", target_mean_degree))
        network["er_edge_prob"] = float(network["er_target_mean_degree"]) / float(max(num_nodes - 1, 1))
    elif topology == "small_world":
        network["ws_degree"] = int(network.get("ws_degree", _resolve_even_degree(target_mean_degree, num_nodes)))
        if int(network["ws_degree"]) % 2 != 0:
            network["ws_degree"] = _resolve_even_degree(float(network["ws_degree"]), num_nodes)
        network["ws_rewiring_prob"] = float(network.get("ws_rewiring_prob", 0.10))
    elif topology == "scale_free":
        attachments = int(network.get("ba_attachments_per_new_node", max(1, int(round(target_mean_degree / 2.0)))))
        network["ba_attachments_per_new_node"] = max(1, min(num_nodes - 1, attachments))
    else:
        raise ValueError(f"Unsupported topology name: {topology}")

    return spec


def build_env_config_from_spec(spec: Mapping[str, Any], graph: Mapping[int, Sequence[int]]) -> SPGGConfig:
    dynamics = spec["dynamics"]
    reward = spec["reward"]
    return SPGGConfig(
        alpha=float(dynamics["alpha"]),
        r=float(dynamics["r"]),
        p_mode=str(dynamics.get("p_mode", "constant")),
        p_max=float(dynamics["p_max"]),
        p_c=float(dynamics.get("p_c", 1.0)),
        resource_consumption_mode=str(dynamics.get("resource_consumption_mode", "fixed")),
        resource_consumption_fixed_mode=str(dynamics.get("resource_consumption_fixed_mode", "constant")),
        resource_consumption_fixed=float(dynamics.get("resource_consumption_fixed", 0.0)),
        resource_consumption_degree_multiplier=float(dynamics.get("resource_consumption_degree_multiplier", 0.0)),
        resource_consumption_rate=float(dynamics.get("resource_consumption_rate", 0.0)),
        resource_consumption_threshold=float(dynamics.get("resource_consumption_threshold", 0.0)),
        strategy_update_rule=str(dynamics.get("strategy_update_rule", "fermi")),
        beta=float(dynamics["beta"]),
        q_learning_rate=float(dynamics.get("q_learning_rate", 0.1)),
        q_learning_discount=float(dynamics.get("q_learning_discount", 0.95)),
        q_learning_epsilon=float(dynamics.get("q_learning_epsilon", 0.05)),
        q_learning_initial_value=float(dynamics.get("q_learning_initial_value", 0.0)),
        episode_length=int(dynamics["episode_length"]),
        initial_resource=float(dynamics["initial_resource"]),
        initial_cooperation_prob=float(dynamics["initial_cooperation_prob"]),
        target_mean_degree=_resolve_target_mean_degree(spec, graph),
        reward=RewardConfig(
            lambda_payoff=float(reward["lambda_payoff"]),
            lambda_cooperation=float(reward["lambda_cooperation"]),
            lambda_total_resource=float(reward.get("lambda_total_resource", 0.0)),
            lambda_collapse=float(reward.get("lambda_collapse", 0.0)),
            lambda_gini=float(reward["lambda_gini"]),
            epsilon=float(reward["epsilon"]),
        ),
    )


def apply_analysis_env_overrides(spec: Mapping[str, Any], *, env_r_override: float | None) -> dict[str, Any]:
    effective_spec = deepcopy(dict(spec))
    if env_r_override is None:
        return effective_spec
    if "dynamics" not in effective_spec or not isinstance(effective_spec["dynamics"], Mapping):
        raise KeyError("Experiment spec does not contain a dynamics mapping; cannot override r.")
    effective_spec["dynamics"] = dict(effective_spec["dynamics"])
    effective_spec["dynamics"]["r"] = float(env_r_override)
    return effective_spec


def _gini_nonnegative(values: np.ndarray, eps: float = 1e-12) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return float("nan")
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= eps:
        return 0.0
    pairwise_abs = np.abs(array[:, None] - array[None, :]).sum()
    return float(pairwise_abs / (2.0 * float(array.size) * total))


def _safe_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array
    clipped = np.clip(array, 0.0, None)
    total = float(clipped.sum())
    if total <= eps:
        return np.full(clipped.shape, 1.0 / float(max(clipped.size, 1)), dtype=np.float64)
    return clipped / total


def _build_row_baselines(
    observation: Mapping[str, np.ndarray],
    sender: int,
    valid_receivers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    receiver_count = int(valid_receivers.size)
    equal_row = np.full(receiver_count, 1.0 / float(max(receiver_count, 1)), dtype=np.float64)
    receiver_unit_investment = np.asarray(observation["unit_investment"][valid_receivers], dtype=np.float64)
    proportional_row = _safe_normalize(receiver_unit_investment)
    self_matches = np.nonzero(valid_receivers == sender)[0]
    if self_matches.size == 0:
        raise RuntimeError("Sender must appear in its own receiver set.")
    return proportional_row, equal_row


def _fit_prop_equal_lambda(
    allocation_row: np.ndarray,
    proportional_row: np.ndarray,
    equal_row: np.ndarray,
) -> float:
    direction = proportional_row - equal_row
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return float("nan")
    estimate = float(np.dot(allocation_row - equal_row, direction) / denominator)
    return float(np.clip(estimate, 0.0, 1.0))


def _save_figure(figure: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")


def parse_weight_values(weights_text: str | None, *, num_weights: int) -> list[float]:
    if weights_text is not None and str(weights_text).strip():
        raw_values = [item.strip() for item in str(weights_text).split(",") if item.strip()]
        if not raw_values:
            raise ValueError("--weights was provided but no valid numbers were found.")
        weights = [float(item) for item in raw_values]
    else:
        weights = np.linspace(0.0, 1.0, num=max(2, int(num_weights))).tolist()

    cleaned: list[float] = []
    for value in weights:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {value}")
        rounded = round(float(value), 10)
        if rounded not in cleaned:
            cleaned.append(rounded)
    cleaned.sort()
    return cleaned


def _weight_label(weight_proportional: float) -> str:
    return f"{float(weight_proportional):.2f}"


def rollout_weight_summary(
    *,
    env_config: SPGGConfig,
    graph: Mapping[int, Sequence[int]],
    topology_name: str,
    episodes: int,
    seed: int,
    max_steps: int | None,
    weight_proportional: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = SPGGEnv(env_config, graph)
    policy = PropEqualMixAllocationPolicy(weight_proportional=weight_proportional)

    ego_gini_stats = RunningMoments()
    ego_mean_stats = RunningMoments()
    reward_stats = RunningMoments()
    final_gini_stats = RunningMoments()
    final_mean_resource_stats = RunningMoments()
    final_cooperation_stats = RunningMoments()

    episode_rows: list[dict[str, Any]] = []
    total_steps = 0

    for episode_index in range(max(0, int(episodes))):
        observation = env.reset(seed=int(seed) + episode_index)
        done = False
        step_index = 0
        total_reward = 0.0
        last_info: dict[str, Any] | None = None

        while not done:
            if max_steps is not None and step_index >= max_steps:
                break

            local_mask = np.asarray(observation["local_mask"], dtype=bool)
            resources = np.asarray(observation["resources"], dtype=np.float64)
            for sender in range(local_mask.shape[0]):
                valid_receivers = np.flatnonzero(local_mask[sender])
                if valid_receivers.size == 0:
                    continue
                ego_resources = np.asarray(resources[valid_receivers], dtype=np.float64)
                ego_gini_stats.update(_gini_nonnegative(ego_resources))
                ego_mean_stats.update(float(np.mean(ego_resources)))

            allocation_matrix = np.asarray(policy.allocate(observation), dtype=np.float64)
            next_observation, reward, env_done, info = env.step(allocation_matrix)

            total_reward += float(reward)
            reward_stats.update(float(reward))
            last_info = info
            observation = next_observation
            step_index += 1
            done = bool(env_done)

        final_cooperation_rate = float(np.mean(observation["x_actual"]))
        final_gini = float(np.asarray(observation["gini"]).item())
        if last_info is not None:
            final_cooperation_rate = float(last_info.get("actual_cooperation_rate", final_cooperation_rate))
            final_gini = float(last_info.get("gini", final_gini))

        final_mean_resource = float(np.mean(observation["resources"]))
        final_gini_stats.update(final_gini)
        final_mean_resource_stats.update(final_mean_resource)
        final_cooperation_stats.update(final_cooperation_rate)
        total_steps += int(step_index)

        episode_rows.append(
            {
                "topology": str(topology_name),
                "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
                "weight_proportional": float(weight_proportional),
                "weight_equal": float(1.0 - weight_proportional),
                "weight_label": _weight_label(weight_proportional),
                "episode": int(episode_index),
                "steps": int(step_index),
                "total_reward": float(total_reward),
                "mean_reward": float(total_reward / float(max(step_index, 1))),
                "final_cooperation_rate": float(final_cooperation_rate),
                "final_gini": float(final_gini),
                "final_mean_resource": float(final_mean_resource),
            }
        )

    summary_row = {
        "topology": str(topology_name),
        "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
        "weight_proportional": float(weight_proportional),
        "weight_equal": float(1.0 - weight_proportional),
        "weight_label": _weight_label(weight_proportional),
        "episode_count": int(episodes),
        "step_count_total": int(total_steps),
        "ego_metric_row_count": int(ego_mean_stats.count),
        "ego_resource_gini_mean": float(ego_gini_stats.mean),
        "ego_resource_gini_std": float(ego_gini_stats.std),
        "ego_resource_gini_sem": float(ego_gini_stats.sem),
        "ego_resource_mean_mean": float(ego_mean_stats.mean),
        "ego_resource_mean_std": float(ego_mean_stats.std),
        "ego_resource_mean_sem": float(ego_mean_stats.sem),
        "reward_mean": float(reward_stats.mean),
        "final_cooperation_rate_mean": float(final_cooperation_stats.mean),
        "final_gini_mean": float(final_gini_stats.mean),
        "final_mean_resource_mean": float(final_mean_resource_stats.mean),
    }
    return summary_row, episode_rows


def collect_agent_fit_rows(
    *,
    actor: Any,
    env_config: SPGGConfig,
    graph: Mapping[int, Sequence[int]],
    topology_name: str,
    episodes: int,
    seed: int,
    max_steps: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env = SPGGEnv(env_config, graph)
    snapshots, episode_summaries = run_deterministic_rollouts(
        actor,
        env,
        episodes=max(1, int(episodes)),
        seed=int(seed),
        max_steps=max_steps,
    )

    row_records: list[dict[str, Any]] = []
    for snapshot in snapshots:
        observation = snapshot.observation
        policy = snapshot.policy
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.asarray(observation["resources"], dtype=np.float64)
        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            allocation_row = np.asarray(policy.allocation_matrix[sender, valid_receivers], dtype=np.float64)
            proportional_row, equal_row = _build_row_baselines(observation, sender, valid_receivers)
            fitted_w = _fit_prop_equal_lambda(allocation_row, proportional_row, equal_row)
            fitted_equal = float(1.0 - fitted_w) if np.isfinite(fitted_w) else float("nan")
            fitted_row = (
                (fitted_w * proportional_row) + ((1.0 - fitted_w) * equal_row)
                if np.isfinite(fitted_w)
                else np.full_like(allocation_row, np.nan, dtype=np.float64)
            )
            fit_error_l1 = (
                float(np.mean(np.abs(fitted_row - allocation_row))) if np.isfinite(fitted_w) else float("nan")
            )
            ego_resources = np.asarray(resources[valid_receivers], dtype=np.float64)
            row_records.append(
                {
                    "topology": str(topology_name),
                    "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
                    "episode": int(snapshot.episode),
                    "step": int(snapshot.step),
                    "sender": int(sender),
                    "sender_degree": int(max(valid_receivers.size - 1, 0)),
                    "receiver_count": int(valid_receivers.size),
                    "ego_resource_gini": _gini_nonnegative(ego_resources),
                    "ego_resource_mean": float(np.mean(ego_resources)),
                    "fitted_weight_proportional": float(fitted_w),
                    "fitted_weight_equal": float(fitted_equal),
                    "fitted_row_l1_error": float(fit_error_l1),
                    "self_allocation": float(policy.allocation_matrix[sender, sender]),
                    "top1_mass": float(np.max(allocation_row)),
                    "sender_pool_grown": float(observation["pool_grown"][sender]),
                    "sender_resource": float(observation["resources"][sender]),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in row_records:
        grouped.setdefault(str(row["topology"]), []).append(row)
    episode_count = max(1, int(episodes))
    for topology_key, rows in grouped.items():
        finite_w = [float(row["fitted_weight_proportional"]) for row in rows if np.isfinite(float(row["fitted_weight_proportional"]))]
        summary_rows.append(
            {
                "topology": str(topology_key),
                "topology_label": TOPOLOGY_LABELS.get(topology_key, topology_key),
                "row_count": int(len(rows)),
                "episode_count": int(episode_count),
                "ego_resource_gini_mean": float(np.nanmean([float(row["ego_resource_gini"]) for row in rows])),
                "ego_resource_mean_mean": float(np.nanmean([float(row["ego_resource_mean"]) for row in rows])),
                "fitted_weight_proportional_mean": float(np.nanmean(finite_w)) if finite_w else float("nan"),
                "fitted_weight_equal_mean": (
                    float(np.nanmean([1.0 - value for value in finite_w])) if finite_w else float("nan")
                ),
                "fitted_row_l1_error_mean": float(np.nanmean([float(row["fitted_row_l1_error"]) for row in rows])),
                "final_cooperation_rate_mean": float(
                    np.nanmean(
                        [float(summary.final_cooperation_rate) for summary in episode_summaries]
                    )
                )
                if episode_summaries
                else float("nan"),
                "final_gini_mean": float(np.nanmean([float(summary.final_gini) for summary in episode_summaries]))
                if episode_summaries
                else float("nan"),
                "final_mean_resource_mean": float(
                    np.nanmean([float(summary.final_mean_resource) for summary in episode_summaries])
                )
                if episode_summaries
                else float("nan"),
            }
        )
    return row_records, summary_rows


def _annotation_targets(weights: Sequence[float]) -> set[float]:
    rounded = [round(float(weight), 6) for weight in weights]
    if len(rounded) <= 11:
        return set(rounded)
    targets = {0.0, 0.25, 0.5, 0.75, 1.0}
    selected: set[float] = set()
    for weight in rounded:
        if any(abs(weight - target) <= 1e-6 for target in targets):
            selected.add(weight)
    if rounded:
        selected.add(rounded[0])
        selected.add(rounded[-1])
    return selected


def plot_prop_equal_mix_scatter(summary_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not summary_rows:
        return

    topology_names = sorted({str(row["topology"]) for row in summary_rows})
    n_panels = max(1, len(topology_names))
    ncols = min(2, n_panels)
    nrows = int(math.ceil(float(n_panels) / float(ncols)))

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 5.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = list(axes.reshape(-1))
    cmap = plt.cm.Blues
    color_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    scatter_artist = None

    for axis, topology_name in zip(axes_flat, topology_names):
        topology_rows = [row for row in summary_rows if str(row["topology"]) == topology_name]
        topology_rows.sort(key=lambda row: float(row["weight_proportional"]))
        if not topology_rows:
            axis.set_axis_off()
            continue

        x_values = np.asarray([float(row["ego_resource_gini_mean"]) for row in topology_rows], dtype=np.float64)
        y_values = np.asarray([float(row["ego_resource_mean_mean"]) for row in topology_rows], dtype=np.float64)
        weights = np.asarray([float(row["weight_proportional"]) for row in topology_rows], dtype=np.float64)

        axis.plot(x_values, y_values, color="#94a3b8", linewidth=1.4, alpha=0.9, zorder=1)
        scatter_artist = axis.scatter(
            x_values,
            y_values,
            c=weights,
            cmap=cmap,
            norm=color_norm,
            s=78,
            linewidths=0.7,
            edgecolors="#0f172a",
            zorder=2,
        )
        annotate_targets = _annotation_targets(weights.tolist())
        for x_value, y_value, weight in zip(x_values, y_values, weights):
            if round(float(weight), 6) not in annotate_targets:
                continue
            axis.annotate(
                _weight_label(float(weight)),
                xy=(float(x_value), float(y_value)),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
                color="#0f172a",
            )

        axis.set_title(f"{TOPOLOGY_LABELS.get(topology_name, topology_name)}")
        axis.set_xlabel("Ego_Resource_Gini")
        axis.set_ylabel("Ego_Resource_Mean")
        axis.grid(alpha=0.25)

    for axis in axes_flat[len(topology_names):]:
        axis.set_axis_off()

    if scatter_artist is not None:
        colorbar = figure.colorbar(scatter_artist, ax=axes_flat[:len(topology_names)], shrink=0.92)
        colorbar.set_label("w in w*Prop + (1-w)*Equal")

    figure.suptitle("Ego Resource Frontier Under Prop-Equal Mix (darker = more proportional)")
    _save_figure(figure, output_path)


def plot_agent_fit_scatter(
    row_records: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    max_points_per_topology: int,
    rng_seed: int,
) -> None:
    if plt is None or not row_records:
        return

    topology_names = sorted({str(row["topology"]) for row in row_records})
    n_panels = max(1, len(topology_names))
    ncols = min(2, n_panels)
    nrows = int(math.ceil(float(n_panels) / float(ncols)))

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 5.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = list(axes.reshape(-1))
    cmap = plt.cm.Blues
    color_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    scatter_artist = None
    rng = np.random.default_rng(int(rng_seed))

    for axis, topology_name in zip(axes_flat, topology_names):
        topology_rows = [row for row in row_records if str(row["topology"]) == topology_name]
        if not topology_rows:
            axis.set_axis_off()
            continue
        if len(topology_rows) > max(1, int(max_points_per_topology)):
            sampled_indices = rng.choice(
                len(topology_rows),
                size=max(1, int(max_points_per_topology)),
                replace=False,
            )
            sampled_indices.sort()
            topology_rows = [topology_rows[int(index)] for index in sampled_indices.tolist()]

        x_values = np.asarray([float(row["ego_resource_gini"]) for row in topology_rows], dtype=np.float64)
        y_values = np.asarray([float(row["ego_resource_mean"]) for row in topology_rows], dtype=np.float64)
        weights = np.asarray([float(row["fitted_weight_proportional"]) for row in topology_rows], dtype=np.float64)
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(weights)
        if not np.any(valid_mask):
            axis.set_axis_off()
            continue

        scatter_artist = axis.scatter(
            x_values[valid_mask],
            y_values[valid_mask],
            c=weights[valid_mask],
            cmap=cmap,
            norm=color_norm,
            s=14,
            linewidths=0.0,
            alpha=0.18,
            zorder=2,
        )
        axis.set_title(f"{TOPOLOGY_LABELS.get(topology_name, topology_name)}")
        axis.set_xlabel("Ego_Resource_Gini")
        axis.set_ylabel("Ego_Resource_Mean")
        axis.grid(alpha=0.25)

    for axis in axes_flat[len(topology_names):]:
        axis.set_axis_off()

    if scatter_artist is not None:
        colorbar = figure.colorbar(scatter_artist, ax=axes_flat[:len(topology_names)], shrink=0.92)
        colorbar.set_label("Fitted w in RL row ≈ w*Prop + (1-w)*Equal")

    figure.suptitle("RL-Agent Ego Resource Scatter Colored by Fitted Prop-Equal Weight")
    _save_figure(figure, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ego-resource scatter under either RL-Agent fitted prop-equal mixtures or explicit prop-equal mix rollouts."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(str(SCRIPT_DEFAULTS["run_dir"])),
        help="Experiment directory containing results.json.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=str(SCRIPT_DEFAULTS["checkpoint_name"]),
        help="Checkpoint filename inside <run-dir>/checkpoints/ when analysis-mode=agent_fit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None if SCRIPT_DEFAULTS["output_dir"] is None else Path(str(SCRIPT_DEFAULTS["output_dir"])),
        help="Absolute or relative output directory. Overrides --output-subdir-name when set.",
    )
    parser.add_argument(
        "--output-subdir-name",
        type=str,
        default=None
        if SCRIPT_DEFAULTS["output_subdir_name"] is None
        else str(SCRIPT_DEFAULTS["output_subdir_name"]),
        help="When --output-dir is not set, write outputs under <run-dir>/<output-subdir-name>/.",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default=str(SCRIPT_DEFAULTS["topologies"]),
        help="Comma-separated topology list. Examples: original or regular,er,ws,ba",
    )
    parser.add_argument(
        "--analysis-mode",
        type=str,
        choices=("agent_fit", "mix_rollout"),
        default=str(SCRIPT_DEFAULTS["analysis_mode"]),
        help="agent_fit: rollout RL-Agent then fit each row by w*Prop + (1-w)*Equal; mix_rollout: independently rollout each w mixture policy.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(SCRIPT_DEFAULTS["episodes"]),
        help="Number of deterministic episodes to analyze.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=SCRIPT_DEFAULTS["max_steps"],
        help="Optional cap on steps per episode.",
    )
    parser.add_argument(
        "--episode-length-override",
        type=int,
        default=SCRIPT_DEFAULTS["episode_length_override"],
        help="Override the environment episode_length used during analysis.",
    )
    parser.add_argument(
        "--env-r",
        dest="env_r_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_r_override"],
        help="Override dynamics.r during analysis rollouts. Defaults to the value in results.json when unset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SCRIPT_DEFAULTS["seed"],
        help="Shared graph/reset seed. When omitted, reuse the training seed recorded in results.json.",
    )
    parser.add_argument(
        "--num-weights",
        type=int,
        default=int(SCRIPT_DEFAULTS["num_weights"]),
        help="Number of evenly spaced weights on [0, 1] when --weights is not provided.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=SCRIPT_DEFAULTS["weights"],
        help="Explicit comma-separated proportional weights, e.g. 0,0.1,0.2,...,1.0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(SCRIPT_DEFAULTS["device"]),
        help="Torch device used to load the RL-Agent when analysis-mode=agent_fit.",
    )
    parser.add_argument(
        "--max-scatter-points-per-topology",
        type=int,
        default=int(SCRIPT_DEFAULTS["max_scatter_points_per_topology"]),
        help="Maximum number of RL row points drawn per topology in the raw scatter plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found under {run_dir}")

    results_payload = _load_json(results_path)
    experiment_spec = results_payload["experiment"] if "experiment" in results_payload else results_payload
    if not isinstance(experiment_spec, dict):
        raise TypeError("Could not extract experiment spec from results.json.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            (run_dir / args.output_subdir_name).resolve()
            if args.output_subdir_name is not None
            else (run_dir / "prop_equal_mix_ego_resource_scatter").resolve()
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    topologies = parse_topology_list(args.topologies)
    weight_values = parse_weight_values(args.weights, num_weights=max(2, int(args.num_weights)))
    seed_base = int(experiment_spec.get("seed", 42)) if args.seed is None else int(args.seed)
    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)

    summary_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    agent_fit_row_records: list[dict[str, Any]] = []

    actor = None
    checkpoint_payload: Mapping[str, Any] | None = None
    if str(args.analysis_mode) == "agent_fit":
        device = torch.device(str(args.device))
        actor, checkpoint_payload = load_actor_from_run_dir(run_dir, args.checkpoint_name, device)

    for topology_name in topologies:
        topology_spec = build_spec_for_topology(
            experiment_spec,
            topology_name,
            reference_mean_degree=reference_mean_degree,
        )
        topology_spec = apply_analysis_env_overrides(
            topology_spec,
            env_r_override=None if args.env_r_override is None else float(args.env_r_override),
        )
        topology_spec["seed"] = int(seed_base)
        if args.episode_length_override is not None:
            topology_spec["dynamics"] = dict(topology_spec["dynamics"])
            topology_spec["dynamics"]["episode_length"] = int(args.episode_length_override)

        graph = build_graph_from_spec(topology_spec)
        env_config = build_env_config_from_spec(topology_spec, graph)

        if str(args.analysis_mode) == "agent_fit":
            if actor is None:
                raise RuntimeError("RL actor failed to load for agent_fit mode.")
            topology_row_records, topology_summary_rows = collect_agent_fit_rows(
                actor=actor,
                env_config=env_config,
                graph=graph,
                topology_name=topology_name,
                episodes=max(1, int(args.episodes)),
                seed=int(seed_base),
                max_steps=args.max_steps,
            )
            agent_fit_row_records.extend(topology_row_records)
            summary_rows.extend(topology_summary_rows)
        else:
            for weight_proportional in weight_values:
                summary_row, weight_episode_rows = rollout_weight_summary(
                    env_config=env_config,
                    graph=graph,
                    topology_name=topology_name,
                    episodes=max(1, int(args.episodes)),
                    seed=int(seed_base),
                    max_steps=args.max_steps,
                    weight_proportional=float(weight_proportional),
                )
                summary_rows.append(summary_row)
                episode_rows.extend(weight_episode_rows)

    if str(args.analysis_mode) == "agent_fit":
        write_csv(output_dir / "agent_prop_equal_fit_row_records.csv", agent_fit_row_records)
        write_csv(output_dir / "agent_prop_equal_fit_topology_summary.csv", summary_rows)
        plot_agent_fit_scatter(
            agent_fit_row_records,
            output_dir / "agent_prop_equal_fit_ego_resource_scatter.png",
            max_points_per_topology=max(1, int(args.max_scatter_points_per_topology)),
            rng_seed=int(seed_base),
        )
    else:
        write_csv(output_dir / "prop_equal_mix_weight_summary.csv", summary_rows)
        write_csv(output_dir / "prop_equal_mix_episode_summary.csv", episode_rows)
        plot_prop_equal_mix_scatter(summary_rows, output_dir / "prop_equal_mix_ego_resource_scatter.png")

    summary_payload = {
        "run_dir": str(run_dir),
        "topologies": topologies,
        "analysis_mode": str(args.analysis_mode),
        "weights": weight_values if str(args.analysis_mode) == "mix_rollout" else None,
        "graph_seed": int(seed_base),
        "rollout_seed": int(seed_base),
        "episode_count": int(args.episodes),
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "episode_length_override": (
            None if args.episode_length_override is None else int(args.episode_length_override)
        ),
        "env_r_override": None if args.env_r_override is None else float(args.env_r_override),
        "checkpoint_name": str(args.checkpoint_name),
        "device": str(args.device),
        "max_scatter_points_per_topology": int(args.max_scatter_points_per_topology),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    if str(args.analysis_mode) == "agent_fit":
        print(f"RL-Agent prop-equal fit ego-resource scatter complete. Artifacts written to: {output_dir}")
    else:
        print(f"Prop-equal mix ego-resource scatter complete. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
