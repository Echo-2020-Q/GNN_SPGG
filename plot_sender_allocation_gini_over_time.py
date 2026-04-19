from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

from Project1.env import SPGGEnv
from Project1.policies.rule_based import PoolPowerMixAllocationPolicy, ProportionalContributionPolicy, UniformAllocationPolicy

from analyze_gnn_policy_decisions import (
    TOPOLOGY_LABELS,
    EpisodeSummary,
    Snapshot,
    _load_json,
    build_env_config_from_spec,
    build_graph_from_spec,
    build_spec_for_topology,
    canonicalize_topology_name,
    load_actor_from_run_dir,
    resolve_reference_mean_degree,
    run_deterministic_rollouts,
    write_csv,
)
from analyze_gnn_policy_mechanism import _gini_nonnegative, apply_analysis_env_overrides, run_rule_policy_rollouts

TRAIN_DISTRIBUTION_ALIASES = {
    "train_distribution",
    "train-distribution",
    "train_dist",
    "train-dist",
    "domain_randomized",
    "domain-randomized",
    "domain_randomization",
    "domain-randomization",
    "dr",
}


SCRIPT_DEFAULTS = {
    # 训练结果目录。目录下至少需要 results.json 和 checkpoints/。
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",

    # 要加载的 checkpoint 文件名。
    "checkpoint_name": "best_eval.pt",

    # 显式输出目录。设为 None 时，会继续看 output_subdir_name。
    "output_dir": None,

    # 当 output_dir 为 None 时，输出到 <run_dir>/<output_subdir_name>/。
    "output_subdir_name": "0419_sender_allocation_gini_over_time",

    # 拓扑列表，多个值用英文逗号分隔。
    # 支持：
    # - original, regular, er, ws, ba
    # - train_distribution：表示使用训练时 domain_randomization 里的拓扑族
    "topologies": "train_distribution",

    # 当 topologies 里包含 train_distribution 时，是否只使用训练分布里的某个子集。
    # 设为 None：使用 results.json 里 domain_randomization.network_types 的全部类型；
    # 设为 "regular,scale_free"：只用训练分布里的这两个拓扑类型。
    "train_distribution_topologies": None,

    # 当 topologies 里包含 train_distribution 且训练时启用了 fixed_graph_bank 时，
    # 对每个拓扑类型使用哪一个 graph bank index。
    # 例如 0 表示每种拓扑都用 bank 里的第 0 张固定图。
    "train_distribution_graph_bank_index": 0,

    # 每种拓扑跑多少个 deterministic episode。
    "episodes": 10,

    # 每个 episode 最多截到多少步。None 表示跑完整局。
    "max_steps": 200,

    # 是否覆盖分析环境里的 episode_length。None 表示沿用 results.json。
    "episode_length_override": 200,

    # 分析时是否覆盖环境里的 r。None 表示沿用训练记录。
    "env_r_override": None,

    # rollout 的环境随机种子基准值。第 k 个 episode 用 seed + k。
    "seed": 42,

    # Torch 推理设备。
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    # `PoolPowerMixAllocationPolicy(k)(Agent)` 这条基线里的 k。
    "pool_power_mix_agent_k": 19.0,

    # 画 aggregated mixture-vs-resource 图时，把 sender-row 按 ego resource 做多少个分位数箱。
    "aggregated_mixture_resource_bin_count": 20,
}


NON_AGENT_BASELINE_POLICIES = (
    "resource_proportional",
    "unit_investment_proportional",
    "equal",
    "agent",
)

AGENT_REFERENCE_POLICIES = (
    "resource_proportional_agent",
    "unit_investment_proportional_agent",
    "pool_power_mix_agent",
    "agent",
)


class ResourceProportionalAllocationPolicy:
    """Allocate each sender's pool in proportion to receivers' current resource stocks."""

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)

        allocation = np.zeros(local_mask.shape, dtype=np.float64)
        weights = local_mask.astype(np.float64) * resources[None, :]
        row_sums = weights.sum(axis=1, keepdims=True)

        positive_rows = row_sums.squeeze(-1) > 1e-12
        if np.any(positive_rows):
            allocation[positive_rows] = weights[positive_rows] / row_sums[positive_rows]

        if np.any(~positive_rows):
            uniform_weights = local_mask[~positive_rows].astype(np.float64)
            uniform_sums = uniform_weights.sum(axis=1, keepdims=True)
            allocation[~positive_rows] = uniform_weights / uniform_sums

        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)


class UnitInvestmentProportionalAllocationPolicy(ProportionalContributionPolicy):
    """Allocate each sender's pool in proportion to receivers' current unit_investment values."""


def _parse_optional_topology_subset(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    values: list[str] = []
    for item in text.split(","):
        topology = canonicalize_topology_name(item)
        if topology not in values:
            values.append(topology)
    return values or None


def _is_train_distribution_token(name: str) -> bool:
    return str(name).strip().lower() in TRAIN_DISTRIBUTION_ALIASES


def _resolve_train_distribution_topologies(
    base_spec: Mapping[str, Any],
    requested_subset: Sequence[str] | None,
) -> list[str]:
    randomization = dict(base_spec.get("domain_randomization", {}))
    configured = [canonicalize_topology_name(str(item)) for item in randomization.get("network_types", ())]
    configured_unique: list[str] = []
    for topology in configured:
        if topology not in configured_unique:
            configured_unique.append(topology)
    if not configured_unique:
        raise ValueError("results.json does not define any domain_randomization.network_types.")

    if requested_subset is None:
        return configured_unique

    unsupported = [topology for topology in requested_subset if topology not in configured_unique]
    if unsupported:
        raise ValueError(
            "Requested train_distribution_topologies are not present in results.json domain_randomization.network_types: "
            + ",".join(unsupported)
        )
    return list(requested_subset)


def _domain_randomization_type_seed(seed_base: int, topology_name: str) -> int:
    return int(seed_base) + sum((index + 1) * ord(char) for index, char in enumerate(str(topology_name)))


def _build_train_distribution_spec(
    base_spec: Mapping[str, Any],
    topology_name: str,
    *,
    graph_bank_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    topology = canonicalize_topology_name(topology_name)
    randomization = dict(base_spec.get("domain_randomization", {}))
    if not bool(randomization.get("enabled", False)):
        raise ValueError("results.json does not have enabled domain_randomization; cannot use train_distribution.")

    spec = deepcopy(dict(base_spec))
    network = dict(spec["network"])
    spec["network"] = network
    network["type"] = topology

    num_nodes_choices = tuple(
        int(item) for item in randomization.get("num_nodes_choices", (int(network["num_nodes"]),))
    )
    regular_degree_choices = tuple(
        int(item) for item in randomization.get("regular_degree_choices", (int(network.get("regular_degree", 4)),))
    )
    er_mean_degree_choices = tuple(
        float(item) for item in randomization.get("er_mean_degree_choices", (float(network.get("er_target_mean_degree", 4.0)),))
    )
    ws_degree_choices = tuple(
        int(item) for item in randomization.get("ws_degree_choices", (int(network.get("ws_degree", 4)),))
    )
    ws_rewiring_choices = tuple(
        float(item) for item in randomization.get("ws_rewiring_choices", (float(network.get("ws_rewiring_prob", 0.1)),))
    )
    ba_attachment_choices = tuple(
        int(item) for item in randomization.get("ba_attachment_choices", (int(network.get("ba_attachments_per_new_node", 2)),))
    )

    seed_base = int(randomization.get("fixed_graph_bank_seed", spec["seed"]))
    fixed_graph_bank_enabled = bool(randomization.get("fixed_graph_bank_enabled", False))
    fixed_graph_bank_size = int(randomization.get("fixed_graph_bank_size_per_type", 0))
    type_rng = np.random.default_rng(_domain_randomization_type_seed(seed_base, topology))

    if fixed_graph_bank_enabled:
        if fixed_graph_bank_size <= 0:
            raise ValueError("fixed_graph_bank_enabled=True but fixed_graph_bank_size_per_type <= 0.")
        if int(graph_bank_index) < 0 or int(graph_bank_index) >= fixed_graph_bank_size:
            raise ValueError(
                f"train_distribution_graph_bank_index={graph_bank_index} is out of range for bank size {fixed_graph_bank_size}."
            )
        sample_count = int(graph_bank_index) + 1
    else:
        sample_count = 1

    sampled_num_nodes = int(network["num_nodes"])
    sampled_seed = int(spec["seed"])
    sampled_degree = int(network.get("regular_degree", 0))
    sampled_er_mean_degree = float(network.get("er_target_mean_degree", 0.0))
    sampled_ws_degree = int(network.get("ws_degree", 0))
    sampled_ws_rewiring = float(network.get("ws_rewiring_prob", 0.1))
    sampled_ba_attachment = int(network.get("ba_attachments_per_new_node", 1))

    for _ in range(sample_count):
        sampled_num_nodes = int(type_rng.choice(num_nodes_choices))
        sampled_seed = int(type_rng.integers(0, 2**31 - 1))
        if topology == "regular":
            sampled_degree = int(type_rng.choice(regular_degree_choices))
        elif topology == "erdos_renyi":
            sampled_er_mean_degree = float(type_rng.choice(er_mean_degree_choices))
        elif topology == "small_world":
            sampled_ws_degree = int(type_rng.choice(ws_degree_choices))
            sampled_ws_rewiring = float(type_rng.choice(ws_rewiring_choices))
        elif topology == "scale_free":
            sampled_ba_attachment = int(type_rng.choice(ba_attachment_choices))
        else:
            raise ValueError(f"Unsupported train_distribution topology: {topology}")

    spec["seed"] = int(sampled_seed)
    network["num_nodes"] = int(sampled_num_nodes)
    if topology == "regular":
        network["regular_degree"] = int(sampled_degree)
    elif topology == "erdos_renyi":
        network["er_target_mean_degree"] = float(sampled_er_mean_degree)
        network["er_edge_prob"] = float(sampled_er_mean_degree) / float(max(sampled_num_nodes - 1, 1))
    elif topology == "small_world":
        network["ws_degree"] = int(sampled_ws_degree)
        network["ws_rewiring_prob"] = float(sampled_ws_rewiring)
    elif topology == "scale_free":
        network["ba_attachments_per_new_node"] = int(sampled_ba_attachment)

    metadata = {
        "graph_source_mode": (
            "domain_randomization_fixed_graph_bank" if fixed_graph_bank_enabled else "domain_randomization_single_sample"
        ),
        "train_distribution_graph_bank_index": int(graph_bank_index) if fixed_graph_bank_enabled else None,
        "train_distribution_seed_base": int(seed_base),
        "graph_seed": int(sampled_seed),
        "topology": topology,
    }
    return spec, metadata


def resolve_topology_cases(
    raw_topologies: str,
    *,
    base_spec: Mapping[str, Any],
    reference_mean_degree: float,
    train_distribution_topologies: Sequence[str] | None,
    train_distribution_graph_bank_index: int,
) -> list[dict[str, Any]]:
    if not str(raw_topologies).strip():
        raise ValueError("Topology list cannot be empty.")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in str(raw_topologies).split(","):
        token = item.strip()
        if not token:
            continue
        if _is_train_distribution_token(token):
            selected_topologies = _resolve_train_distribution_topologies(base_spec, train_distribution_topologies)
            for topology in selected_topologies:
                if topology in seen:
                    continue
                spec, metadata = _build_train_distribution_spec(
                    base_spec,
                    topology,
                    graph_bank_index=int(train_distribution_graph_bank_index),
                )
                cases.append(
                    {
                        "topology_name": topology,
                        "spec": spec,
                        "metadata": metadata,
                    }
                )
                seen.add(topology)
            continue

        topology = canonicalize_topology_name(token)
        if topology in seen:
            continue
        cases.append(
            {
                "topology_name": topology,
                "spec": build_spec_for_topology(
                    base_spec,
                    topology,
                    reference_mean_degree=reference_mean_degree,
                ),
                "metadata": {
                    "graph_source_mode": "base_spec",
                    "train_distribution_graph_bank_index": None,
                    "train_distribution_seed_base": None,
                    "graph_seed": None,
                    "topology": topology,
                },
            }
        )
        seen.add(topology)

    if not cases:
        raise ValueError("Topology list cannot be empty.")
    return cases


def compute_sender_allocation_gini_rows(
    policy_name: str,
    snapshots: Sequence[Snapshot],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        observation = snapshot.observation
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        pool_grown = np.asarray(observation["pool_grown"], dtype=np.float64)
        allocation_matrix = np.asarray(snapshot.policy.allocation_matrix, dtype=np.float64)

        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            allocation_row = np.asarray(allocation_matrix[sender, valid_receivers], dtype=np.float64)
            rows.append(
                {
                    "policy": str(policy_name),
                    "state_source": str(policy_name),
                    "episode": int(snapshot.episode),
                    "step": int(snapshot.step),
                    "sender": int(sender),
                    "sender_degree": max(int(valid_receivers.size) - 1, 0),
                    "receiver_count": int(valid_receivers.size),
                    "sender_pool_grown": float(pool_grown[sender]),
                    "sender_allocation_gini": _gini_nonnegative(float(pool_grown[sender]) * allocation_row),
                }
            )
    return rows


def compute_sender_allocation_gini_rows_from_policy_on_snapshots(
    policy_name: str,
    snapshots: Sequence[Snapshot],
    allocation_policy: Any,
    *,
    state_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        observation = snapshot.observation
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        pool_grown = np.asarray(observation["pool_grown"], dtype=np.float64)
        allocation_matrix = np.asarray(allocation_policy.allocate(observation), dtype=np.float64)

        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            allocation_row = np.asarray(allocation_matrix[sender, valid_receivers], dtype=np.float64)
            rows.append(
                {
                    "policy": str(policy_name),
                    "state_source": str(state_source),
                    "episode": int(snapshot.episode),
                    "step": int(snapshot.step),
                    "sender": int(sender),
                    "sender_degree": max(int(valid_receivers.size) - 1, 0),
                    "receiver_count": int(valid_receivers.size),
                    "sender_pool_grown": float(pool_grown[sender]),
                    "sender_allocation_gini": _gini_nonnegative(float(pool_grown[sender]) * allocation_row),
                }
            )
    return rows


def summarize_sender_allocation_gini(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []

    grouped_episode: dict[tuple[str, int, int], list[float]] = {}
    for row in rows:
        key = (str(row["policy"]), int(row["episode"]), int(row["step"]))
        grouped_episode.setdefault(key, []).append(float(row["sender_allocation_gini"]))

    episode_step_rows: list[dict[str, Any]] = []
    grouped_step: dict[tuple[str, int], list[float]] = {}
    for (policy_name, episode, step), values in sorted(grouped_episode.items()):
        episode_mean = float(np.mean(np.asarray(values, dtype=np.float64)))
        episode_step_rows.append(
            {
                "policy": policy_name,
                "episode": int(episode),
                "step": int(step),
                "sender_count": int(len(values)),
                "sender_allocation_gini_mean": episode_mean,
                "sender_allocation_gini_std_within_episode": float(np.std(np.asarray(values, dtype=np.float64))),
            }
        )
        grouped_step.setdefault((policy_name, int(step)), []).append(episode_mean)

    step_summary_rows: list[dict[str, Any]] = []
    policy_order = {
        "resource_proportional": 0,
        "resource_proportional_agent": 1,
        "unit_investment_proportional": 2,
        "unit_investment_proportional_agent": 3,
        "pool_power_mix_agent": 4,
        "agent": 5,
        "equal": 6,
    }
    for (policy_name, step), values in sorted(
        grouped_step.items(),
        key=lambda item: (policy_order.get(item[0][0], 99), item[0][1]),
    ):
        array = np.asarray(values, dtype=np.float64)
        std = float(np.std(array)) if array.size > 0 else float("nan")
        sem = float(std / np.sqrt(array.size)) if array.size > 0 else float("nan")
        step_summary_rows.append(
            {
                "policy": policy_name,
                "step": int(step),
                "episode_count": int(array.size),
                "sender_allocation_gini_mean": float(np.mean(array)) if array.size > 0 else float("nan"),
                "sender_allocation_gini_std": std,
                "sender_allocation_gini_sem": sem,
            }
        )

    return episode_step_rows, step_summary_rows


def summarize_sender_allocation_gini_episode_rows_across_topologies(
    episode_step_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not episode_step_rows:
        return []

    grouped_step: dict[tuple[str, int], list[float]] = {}
    for row in episode_step_rows:
        key = (str(row["policy"]), int(row["step"]))
        grouped_step.setdefault(key, []).append(float(row["sender_allocation_gini_mean"]))

    policy_order = {
        "resource_proportional": 0,
        "resource_proportional_agent": 1,
        "unit_investment_proportional": 2,
        "unit_investment_proportional_agent": 3,
        "pool_power_mix_agent": 4,
        "agent": 5,
        "equal": 6,
    }

    aggregated_rows: list[dict[str, Any]] = []
    for (policy_name, step), values in sorted(
        grouped_step.items(),
        key=lambda item: (policy_order.get(item[0][0], 99), item[0][1]),
    ):
        array = np.asarray(values, dtype=np.float64)
        std = float(np.std(array)) if array.size > 0 else float("nan")
        sem = float(std / np.sqrt(array.size)) if array.size > 0 else float("nan")
        aggregated_rows.append(
            {
                "policy": policy_name,
                "step": int(step),
                "sample_count": int(array.size),
                "sender_allocation_gini_mean": float(np.mean(array)) if array.size > 0 else float("nan"),
                "sender_allocation_gini_std": std,
                "sender_allocation_gini_sem": sem,
            }
        )

    return aggregated_rows


def _quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return np.asarray([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, num=max(2, int(num_bins)) + 1)
    edges = np.quantile(array, quantiles)
    edges = np.unique(edges)
    if edges.size == 1:
        edges = np.asarray([edges[0], edges[0] + 1e-9], dtype=np.float64)
    return edges


def _fit_unit_investment_equal_mixture_weight(
    agent_row: np.ndarray,
    unit_investment_row: np.ndarray,
    equal_row: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    delta = np.asarray(unit_investment_row, dtype=np.float64) - np.asarray(equal_row, dtype=np.float64)
    denom = float(np.dot(delta, delta))
    if denom <= float(eps):
        fitted_row = np.asarray(equal_row, dtype=np.float64)
        residual = np.asarray(agent_row, dtype=np.float64) - fitted_row
        return {
            "weight_unit_investment": float("nan"),
            "weight_equal": float("nan"),
            "identifiable": 0,
            "mixture_fit_l1": float(np.mean(np.abs(residual))),
            "mixture_fit_l2": float(np.sqrt(np.mean(np.square(residual)))),
        }

    weight_unit_investment = float(
        np.dot(np.asarray(agent_row, dtype=np.float64) - np.asarray(equal_row, dtype=np.float64), delta) / denom
    )
    weight_unit_investment = float(np.clip(weight_unit_investment, 0.0, 1.0))
    weight_equal = float(1.0 - weight_unit_investment)
    fitted_row = (weight_unit_investment * np.asarray(unit_investment_row, dtype=np.float64)) + (
        weight_equal * np.asarray(equal_row, dtype=np.float64)
    )
    residual = np.asarray(agent_row, dtype=np.float64) - fitted_row
    return {
        "weight_unit_investment": weight_unit_investment,
        "weight_equal": weight_equal,
        "identifiable": 1,
        "mixture_fit_l1": float(np.mean(np.abs(residual))),
        "mixture_fit_l2": float(np.sqrt(np.mean(np.square(residual)))),
    }


def compute_agent_unit_investment_equal_mixture_rows(
    snapshots: Sequence[Snapshot],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unit_investment_policy = UnitInvestmentProportionalAllocationPolicy()
    equal_policy = UniformAllocationPolicy()

    for snapshot in snapshots:
        observation = snapshot.observation
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        pool_grown = np.asarray(observation["pool_grown"], dtype=np.float64)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)
        agent_allocation = np.asarray(snapshot.policy.allocation_matrix, dtype=np.float64)
        unit_investment_allocation = np.asarray(unit_investment_policy.allocate(observation), dtype=np.float64)
        equal_allocation = np.asarray(equal_policy.allocate(observation), dtype=np.float64)

        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue

            agent_row = np.asarray(agent_allocation[sender, valid_receivers], dtype=np.float64)
            unit_investment_row = np.asarray(unit_investment_allocation[sender, valid_receivers], dtype=np.float64)
            equal_row = np.asarray(equal_allocation[sender, valid_receivers], dtype=np.float64)
            fit_result = _fit_unit_investment_equal_mixture_weight(agent_row, unit_investment_row, equal_row)
            ego_resources = np.asarray(resources[valid_receivers], dtype=np.float64)

            rows.append(
                {
                    "episode": int(snapshot.episode),
                    "step": int(snapshot.step),
                    "sender": int(sender),
                    "sender_degree": max(int(valid_receivers.size) - 1, 0),
                    "receiver_count": int(valid_receivers.size),
                    "sender_pool_grown": float(pool_grown[sender]),
                    "ego_total_resource": float(np.sum(ego_resources)),
                    "ego_mean_resource": float(np.mean(ego_resources)) if ego_resources.size > 0 else float("nan"),
                    "weight_unit_investment": fit_result["weight_unit_investment"],
                    "weight_equal": fit_result["weight_equal"],
                    "identifiable": int(fit_result["identifiable"]),
                    "mixture_fit_l1": float(fit_result["mixture_fit_l1"]),
                    "mixture_fit_l2": float(fit_result["mixture_fit_l2"]),
                }
            )

    return rows


def summarize_agent_unit_investment_equal_mixture(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []

    grouped_episode: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["episode"]), int(row["step"]))
        grouped_episode.setdefault(key, []).append(row)

    episode_step_rows: list[dict[str, Any]] = []
    grouped_step: dict[int, list[dict[str, Any]]] = {}

    for (episode, step), group_rows in sorted(grouped_episode.items()):
        sender_count = int(len(group_rows))
        identifiable_rows = [row for row in group_rows if int(row["identifiable"]) == 1]
        identifiable_count = int(len(identifiable_rows))

        unit_investment_weights = np.asarray(
            [float(row["weight_unit_investment"]) for row in identifiable_rows],
            dtype=np.float64,
        )
        equal_weights = np.asarray(
            [float(row["weight_equal"]) for row in identifiable_rows],
            dtype=np.float64,
        )
        fit_l1_values = np.asarray([float(row["mixture_fit_l1"]) for row in group_rows], dtype=np.float64)
        fit_l2_values = np.asarray([float(row["mixture_fit_l2"]) for row in group_rows], dtype=np.float64)

        episode_row = {
            "episode": int(episode),
            "step": int(step),
            "sender_count": sender_count,
            "identifiable_sender_count": identifiable_count,
            "identifiable_sender_fraction": float(identifiable_count / sender_count) if sender_count > 0 else float("nan"),
            "weight_unit_investment_mean": float(np.mean(unit_investment_weights))
            if unit_investment_weights.size > 0
            else float("nan"),
            "weight_equal_mean": float(np.mean(equal_weights)) if equal_weights.size > 0 else float("nan"),
            "mixture_fit_l1_mean": float(np.mean(fit_l1_values)) if fit_l1_values.size > 0 else float("nan"),
            "mixture_fit_l2_mean": float(np.mean(fit_l2_values)) if fit_l2_values.size > 0 else float("nan"),
        }
        episode_step_rows.append(episode_row)
        grouped_step.setdefault(int(step), []).append(episode_row)

    step_summary_rows: list[dict[str, Any]] = []
    for step, group_rows in sorted(grouped_step.items(), key=lambda item: item[0]):
        unit_investment_episode_values = np.asarray(
            [
                float(row["weight_unit_investment_mean"])
                for row in group_rows
                if np.isfinite(float(row["weight_unit_investment_mean"]))
            ],
            dtype=np.float64,
        )
        equal_episode_values = np.asarray(
            [float(row["weight_equal_mean"]) for row in group_rows if np.isfinite(float(row["weight_equal_mean"]))],
            dtype=np.float64,
        )
        identifiable_fraction_values = np.asarray(
            [float(row["identifiable_sender_fraction"]) for row in group_rows],
            dtype=np.float64,
        )
        fit_l1_values = np.asarray([float(row["mixture_fit_l1_mean"]) for row in group_rows], dtype=np.float64)
        fit_l2_values = np.asarray([float(row["mixture_fit_l2_mean"]) for row in group_rows], dtype=np.float64)

        inv_std = float(np.std(unit_investment_episode_values)) if unit_investment_episode_values.size > 0 else float("nan")
        inv_sem = (
            float(inv_std / np.sqrt(unit_investment_episode_values.size))
            if unit_investment_episode_values.size > 0
            else float("nan")
        )
        eq_std = float(np.std(equal_episode_values)) if equal_episode_values.size > 0 else float("nan")
        eq_sem = (
            float(eq_std / np.sqrt(equal_episode_values.size))
            if equal_episode_values.size > 0
            else float("nan")
        )

        step_summary_rows.append(
            {
                "step": int(step),
                "episode_count": int(len(group_rows)),
                "episode_count_identifiable": int(unit_investment_episode_values.size),
                "weight_unit_investment_mean": float(np.mean(unit_investment_episode_values))
                if unit_investment_episode_values.size > 0
                else float("nan"),
                "weight_unit_investment_std": inv_std,
                "weight_unit_investment_sem": inv_sem,
                "weight_equal_mean": float(np.mean(equal_episode_values)) if equal_episode_values.size > 0 else float("nan"),
                "weight_equal_std": eq_std,
                "weight_equal_sem": eq_sem,
                "identifiable_sender_fraction_mean": float(np.mean(identifiable_fraction_values))
                if identifiable_fraction_values.size > 0
                else float("nan"),
                "mixture_fit_l1_mean": float(np.mean(fit_l1_values)) if fit_l1_values.size > 0 else float("nan"),
                "mixture_fit_l2_mean": float(np.mean(fit_l2_values)) if fit_l2_values.size > 0 else float("nan"),
            }
        )

    return episode_step_rows, step_summary_rows


def summarize_agent_unit_investment_equal_mixture_episode_rows_across_topologies(
    episode_step_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not episode_step_rows:
        return []

    grouped_step: dict[int, list[Mapping[str, Any]]] = {}
    for row in episode_step_rows:
        grouped_step.setdefault(int(row["step"]), []).append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for step, group_rows in sorted(grouped_step.items(), key=lambda item: item[0]):
        unit_investment_episode_values = np.asarray(
            [
                float(row["weight_unit_investment_mean"])
                for row in group_rows
                if np.isfinite(float(row["weight_unit_investment_mean"]))
            ],
            dtype=np.float64,
        )
        equal_episode_values = np.asarray(
            [float(row["weight_equal_mean"]) for row in group_rows if np.isfinite(float(row["weight_equal_mean"]))],
            dtype=np.float64,
        )
        identifiable_fraction_values = np.asarray(
            [float(row["identifiable_sender_fraction"]) for row in group_rows],
            dtype=np.float64,
        )
        fit_l1_values = np.asarray([float(row["mixture_fit_l1_mean"]) for row in group_rows], dtype=np.float64)
        fit_l2_values = np.asarray([float(row["mixture_fit_l2_mean"]) for row in group_rows], dtype=np.float64)

        inv_std = float(np.std(unit_investment_episode_values)) if unit_investment_episode_values.size > 0 else float("nan")
        inv_sem = (
            float(inv_std / np.sqrt(unit_investment_episode_values.size))
            if unit_investment_episode_values.size > 0
            else float("nan")
        )
        eq_std = float(np.std(equal_episode_values)) if equal_episode_values.size > 0 else float("nan")
        eq_sem = (
            float(eq_std / np.sqrt(equal_episode_values.size))
            if equal_episode_values.size > 0
            else float("nan")
        )

        aggregated_rows.append(
            {
                "step": int(step),
                "sample_count": int(len(group_rows)),
                "sample_count_identifiable": int(unit_investment_episode_values.size),
                "weight_unit_investment_mean": float(np.mean(unit_investment_episode_values))
                if unit_investment_episode_values.size > 0
                else float("nan"),
                "weight_unit_investment_std": inv_std,
                "weight_unit_investment_sem": inv_sem,
                "weight_equal_mean": float(np.mean(equal_episode_values)) if equal_episode_values.size > 0 else float("nan"),
                "weight_equal_std": eq_std,
                "weight_equal_sem": eq_sem,
                "identifiable_sender_fraction_mean": float(np.mean(identifiable_fraction_values))
                if identifiable_fraction_values.size > 0
                else float("nan"),
                "mixture_fit_l1_mean": float(np.mean(fit_l1_values)) if fit_l1_values.size > 0 else float("nan"),
                "mixture_fit_l2_mean": float(np.mean(fit_l2_values)) if fit_l2_values.size > 0 else float("nan"),
            }
        )

    return aggregated_rows


def summarize_agent_unit_investment_equal_mixture_vs_resource(
    rows: Sequence[Mapping[str, Any]],
    *,
    resource_key: str,
    num_bins: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    valid_rows = [
        row
        for row in rows
        if np.isfinite(float(row.get(resource_key, float("nan"))))
    ]
    if not valid_rows:
        return []

    resource_values = np.asarray([float(row[resource_key]) for row in valid_rows], dtype=np.float64)
    edges = _quantile_edges(resource_values, int(num_bins))
    if edges.size < 2:
        return []

    summary_rows: list[dict[str, Any]] = []
    for bin_index in range(edges.size - 1):
        left_edge = float(edges[bin_index])
        right_edge = float(edges[bin_index + 1])
        if bin_index == edges.size - 2:
            bin_rows = [row for row in valid_rows if left_edge <= float(row[resource_key]) <= right_edge]
        else:
            bin_rows = [row for row in valid_rows if left_edge <= float(row[resource_key]) < right_edge]
        if not bin_rows:
            continue

        identifiable_rows = [row for row in bin_rows if np.isfinite(float(row["weight_unit_investment"]))]
        unit_investment_values = np.asarray(
            [float(row["weight_unit_investment"]) for row in identifiable_rows],
            dtype=np.float64,
        )
        equal_values = np.asarray([float(row["weight_equal"]) for row in identifiable_rows], dtype=np.float64)
        fit_l1_values = np.asarray([float(row["mixture_fit_l1"]) for row in bin_rows], dtype=np.float64)
        fit_l2_values = np.asarray([float(row["mixture_fit_l2"]) for row in bin_rows], dtype=np.float64)
        resource_bin_values = np.asarray([float(row[resource_key]) for row in bin_rows], dtype=np.float64)

        inv_std = float(np.std(unit_investment_values)) if unit_investment_values.size > 0 else float("nan")
        inv_sem = (
            float(inv_std / np.sqrt(unit_investment_values.size))
            if unit_investment_values.size > 0
            else float("nan")
        )
        eq_std = float(np.std(equal_values)) if equal_values.size > 0 else float("nan")
        eq_sem = float(eq_std / np.sqrt(equal_values.size)) if equal_values.size > 0 else float("nan")

        summary_rows.append(
            {
                "resource_key": str(resource_key),
                "bin_index": int(bin_index),
                "bin_left": left_edge,
                "bin_right": right_edge,
                "bin_center": float(np.mean(resource_bin_values)),
                "sample_count": int(len(bin_rows)),
                "sample_count_identifiable": int(len(identifiable_rows)),
                "weight_unit_investment_mean": float(np.mean(unit_investment_values))
                if unit_investment_values.size > 0
                else float("nan"),
                "weight_unit_investment_std": inv_std,
                "weight_unit_investment_sem": inv_sem,
                "weight_equal_mean": float(np.mean(equal_values)) if equal_values.size > 0 else float("nan"),
                "weight_equal_std": eq_std,
                "weight_equal_sem": eq_sem,
                "identifiable_fraction": float(len(identifiable_rows) / len(bin_rows)) if bin_rows else float("nan"),
                "mixture_fit_l1_mean": float(np.mean(fit_l1_values)) if fit_l1_values.size > 0 else float("nan"),
                "mixture_fit_l2_mean": float(np.mean(fit_l2_values)) if fit_l2_values.size > 0 else float("nan"),
            }
        )

    return summary_rows


def summarize_episode_rows(policy_name: str, episode_summaries: Sequence[EpisodeSummary]) -> list[dict[str, Any]]:
    return [
        {
            "policy": str(policy_name),
            "episode": int(summary.episode),
            "steps": int(summary.steps),
            "total_reward": float(summary.total_reward),
            "mean_reward": float(summary.mean_reward),
            "final_cooperation_rate": float(summary.final_cooperation_rate),
            "final_gini": float(summary.final_gini),
            "final_mean_resource": float(summary.final_mean_resource),
        }
        for summary in episode_summaries
    ]


def _sender_allocation_style_by_policy(pool_power_mix_agent_k: float) -> dict[str, dict[str, str]]:
    return {
        "resource_proportional": {
            "label": "Proportional by Current Resources",
            "color": "#d62728",
            "linestyle": "-",
        },
        "resource_proportional_agent": {
            "label": "Proportional by Current Resources (Agent)",
            "color": "#ff7f0e",
            "linestyle": "--",
        },
        "unit_investment_proportional": {
            "label": "Proportional by Current Unit Investment",
            "color": "#9467bd",
            "linestyle": "-",
        },
        "unit_investment_proportional_agent": {
            "label": "Proportional by Current Unit Investment (Agent)",
            "color": "#8c564b",
            "linestyle": "--",
        },
        "pool_power_mix_agent": {
            "label": f"PoolPowerMix k={float(pool_power_mix_agent_k):g} (Agent)",
            "color": "#e377c2",
            "linestyle": "--",
        },
        "agent": {
            "label": "RL Agent",
            "color": "#2ca02c",
            "linestyle": "-",
        },
        "equal": {
            "label": "Equal Allocation",
            "color": "#1f77ff",
            "linestyle": "-",
        },
    }


def plot_sender_allocation_gini_over_time(
    step_summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    style_by_policy: Mapping[str, Mapping[str, str]],
    policies: Sequence[str],
    title: str,
) -> None:
    if plt is None or not step_summary_rows:
        return

    figure, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for policy_name in policies:
        rows = [row for row in step_summary_rows if str(row["policy"]) == policy_name]
        if not rows:
            continue
        rows.sort(key=lambda row: int(row["step"]))
        x_values = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
        y_values = np.asarray([float(row["sender_allocation_gini_mean"]) for row in rows], dtype=np.float64)
        sem_values = np.asarray([float(row["sender_allocation_gini_sem"]) for row in rows], dtype=np.float64)
        style = style_by_policy[policy_name]
        ax.plot(
            x_values,
            y_values,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
        )
        if np.any(np.isfinite(sem_values)):
            lower = np.clip(y_values - sem_values, 0.0, None)
            upper = np.clip(y_values + sem_values, 0.0, None)
            mask = np.isfinite(lower) & np.isfinite(upper)
            if int(np.count_nonzero(mask)) >= 2:
                ax.fill_between(
                    x_values,
                    lower,
                    upper,
                    where=mask,
                    color=style["color"],
                    alpha=0.16,
                )

    ax.set_title(str(title))
    ax.set_xlabel("Step t")
    ax.set_ylabel("Mean Sender Allocation Gini")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_sender_allocation_gini_over_time_by_topology(
    all_step_summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    style_by_policy: Mapping[str, Mapping[str, str]],
    policies: Sequence[str],
    title_prefix: str,
) -> None:
    if plt is None or not all_step_summary_rows:
        return

    grouped_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in all_step_summary_rows:
        topology_name = str(row.get("topology", "unknown"))
        topology_label = str(row.get("topology_label", topology_name))
        grouped_rows.setdefault((topology_name, topology_label), []).append(row)

    if not grouped_rows:
        return

    ordered_groups = sorted(grouped_rows.items(), key=lambda item: item[0][1])
    panel_count = len(ordered_groups)
    cols = min(2, panel_count)
    rows = int(np.ceil(float(panel_count) / float(cols)))
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.2 * float(cols), 4.2 * float(rows)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_flat = list(np.asarray(axes).reshape(-1))

    for axis, ((_, topology_label), topology_rows) in zip(axes_flat, ordered_groups):
        for policy_name in policies:
            rows_for_policy = [row for row in topology_rows if str(row["policy"]) == policy_name]
            if not rows_for_policy:
                continue
            rows_for_policy.sort(key=lambda row: int(row["step"]))
            x_values = np.asarray([int(row["step"]) for row in rows_for_policy], dtype=np.int64)
            y_values = np.asarray([float(row["sender_allocation_gini_mean"]) for row in rows_for_policy], dtype=np.float64)
            style = style_by_policy[policy_name]
            axis.plot(
                x_values,
                y_values,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
            )
        axis.set_title(f"{title_prefix}: {topology_label}")
        axis.set_xlabel("Step t")
        axis.set_ylabel("Mean Sender Allocation Gini")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)

    for axis in axes_flat[len(ordered_groups):]:
        axis.set_axis_off()

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_agent_unit_investment_equal_mixture_weights_over_time(
    step_summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    if plt is None or not step_summary_rows:
        return

    rows = sorted(step_summary_rows, key=lambda row: int(row["step"]))
    x_values = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
    unit_investment_mean = np.asarray([float(row["weight_unit_investment_mean"]) for row in rows], dtype=np.float64)
    unit_investment_sem = np.asarray([float(row["weight_unit_investment_sem"]) for row in rows], dtype=np.float64)
    equal_mean = np.asarray([float(row["weight_equal_mean"]) for row in rows], dtype=np.float64)
    equal_sem = np.asarray([float(row["weight_equal_sem"]) for row in rows], dtype=np.float64)

    figure, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(
        x_values,
        unit_investment_mean,
        color="#9467bd",
        linestyle="-",
        linewidth=2.3,
        label="w: Proportional by Current Unit Investment",
    )
    if np.any(np.isfinite(unit_investment_sem)):
        lower = np.clip(unit_investment_mean - unit_investment_sem, 0.0, 1.0)
        upper = np.clip(unit_investment_mean + unit_investment_sem, 0.0, 1.0)
        mask = np.isfinite(lower) & np.isfinite(upper)
        if int(np.count_nonzero(mask)) >= 2:
            ax.fill_between(x_values, lower, upper, where=mask, color="#9467bd", alpha=0.16)

    ax.plot(
        x_values,
        equal_mean,
        color="#1f77ff",
        linestyle="-",
        linewidth=2.3,
        label="1 - w: Equal Allocation",
    )
    if np.any(np.isfinite(equal_sem)):
        lower = np.clip(equal_mean - equal_sem, 0.0, 1.0)
        upper = np.clip(equal_mean + equal_sem, 0.0, 1.0)
        mask = np.isfinite(lower) & np.isfinite(upper)
        if int(np.count_nonzero(mask)) >= 2:
            ax.fill_between(x_values, lower, upper, where=mask, color="#1f77ff", alpha=0.16)

    ax.set_title("RL-Agent Mixture Weights Over Time")
    ax.set_xlabel("Step t")
    ax.set_ylabel("Mean Mixture Weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_agent_unit_investment_equal_mixture_vs_resource(
    summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    resource_label: str,
) -> None:
    if plt is None or not summary_rows:
        return

    rows = sorted(summary_rows, key=lambda row: float(row["bin_center"]))
    x_values = np.asarray([float(row["bin_center"]) for row in rows], dtype=np.float64)
    unit_investment_mean = np.asarray([float(row["weight_unit_investment_mean"]) for row in rows], dtype=np.float64)
    unit_investment_sem = np.asarray([float(row["weight_unit_investment_sem"]) for row in rows], dtype=np.float64)
    equal_mean = np.asarray([float(row["weight_equal_mean"]) for row in rows], dtype=np.float64)
    equal_sem = np.asarray([float(row["weight_equal_sem"]) for row in rows], dtype=np.float64)

    figure, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(
        x_values,
        unit_investment_mean,
        color="#9467bd",
        linestyle="-",
        linewidth=2.3,
        label="w: Proportional by Current Unit Investment",
    )
    if np.any(np.isfinite(unit_investment_sem)):
        lower = np.clip(unit_investment_mean - unit_investment_sem, 0.0, 1.0)
        upper = np.clip(unit_investment_mean + unit_investment_sem, 0.0, 1.0)
        mask = np.isfinite(lower) & np.isfinite(upper)
        if int(np.count_nonzero(mask)) >= 2:
            ax.fill_between(x_values, lower, upper, where=mask, color="#9467bd", alpha=0.16)

    ax.plot(
        x_values,
        equal_mean,
        color="#1f77ff",
        linestyle="-",
        linewidth=2.3,
        label="1 - w: Equal Allocation",
    )
    if np.any(np.isfinite(equal_sem)):
        lower = np.clip(equal_mean - equal_sem, 0.0, 1.0)
        upper = np.clip(equal_mean + equal_sem, 0.0, 1.0)
        mask = np.isfinite(lower) & np.isfinite(upper)
        if int(np.count_nonzero(mask)) >= 2:
            ax.fill_between(x_values, lower, upper, where=mask, color="#1f77ff", alpha=0.16)

    ax.set_title(f"RL-Agent Mixture Weights vs {resource_label}")
    ax.set_xlabel(resource_label)
    ax.set_ylabel("Mean Mixture Weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sender allocation Gini over time for RL-Agent and two baselines.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(str(SCRIPT_DEFAULTS["run_dir"])),
        help="Experiment directory containing results.json and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=str(SCRIPT_DEFAULTS["checkpoint_name"]),
        help="Checkpoint filename inside <run-dir>/checkpoints/.",
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
        help="Comma-separated topology list. Examples: original, regular,er,ws,ba, or train_distribution",
    )
    parser.add_argument(
        "--train-distribution-topologies",
        type=str,
        default=SCRIPT_DEFAULTS["train_distribution_topologies"],
        help="Optional subset used when --topologies includes train_distribution. Examples: regular,scale_free",
    )
    parser.add_argument(
        "--train-distribution-graph-bank-index",
        type=int,
        default=int(SCRIPT_DEFAULTS["train_distribution_graph_bank_index"]),
        help="When train_distribution uses fixed_graph_bank, choose which graph bank index to reconstruct.",
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
        default=int(SCRIPT_DEFAULTS["seed"]),
        help="Base environment seed for analysis rollouts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(SCRIPT_DEFAULTS["device"]),
        help="Torch device, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--pool-power-mix-agent-k",
        type=float,
        default=float(SCRIPT_DEFAULTS["pool_power_mix_agent_k"]),
        help="k used by PoolPowerMixAllocationPolicy(k)(Agent).",
    )
    parser.add_argument(
        "--aggregated-mixture-resource-bin-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["aggregated_mixture_resource_bin_count"]),
        help="Number of quantile bins for aggregated mixture-vs-resource plots.",
    )
    return parser.parse_args()


def analyze_single_topology(
    *,
    actor: Any,
    run_dir: Path,
    checkpoint_name: str,
    checkpoint_payload: Mapping[str, Any],
    output_dir: Path,
    spec: Mapping[str, Any],
    topology_name: str,
    topology_metadata: Mapping[str, Any],
    episodes: int,
    max_steps: int | None,
    episode_length_override: int | None,
    rollout_seed: int,
    pool_power_mix_agent_k: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_spec = deepcopy(dict(spec))
    if episode_length_override is not None:
        effective_spec["dynamics"] = dict(effective_spec["dynamics"])
        effective_spec["dynamics"]["episode_length"] = int(episode_length_override)

    graph = build_graph_from_spec(effective_spec)
    env_config = build_env_config_from_spec(effective_spec, graph)

    agent_env = SPGGEnv(env_config, graph)
    agent_snapshots, agent_episode_summaries = run_deterministic_rollouts(
        actor,
        agent_env,
        episodes=max(1, int(episodes)),
        seed=int(rollout_seed),
        max_steps=max_steps,
    )
    resource_prop_snapshots, resource_prop_episode_summaries = run_rule_policy_rollouts(
        ResourceProportionalAllocationPolicy(),
        env_config=env_config,
        graph=graph,
        episodes=max(1, int(episodes)),
        seed=int(rollout_seed) + 100_000,
        max_steps=max_steps,
    )
    unit_investment_prop_snapshots, unit_investment_prop_episode_summaries = run_rule_policy_rollouts(
        UnitInvestmentProportionalAllocationPolicy(),
        env_config=env_config,
        graph=graph,
        episodes=max(1, int(episodes)),
        seed=int(rollout_seed) + 150_000,
        max_steps=max_steps,
    )
    equal_snapshots, equal_episode_summaries = run_rule_policy_rollouts(
        UniformAllocationPolicy(),
        env_config=env_config,
        graph=graph,
        episodes=max(1, int(episodes)),
        seed=int(rollout_seed) + 200_000,
        max_steps=max_steps,
    )

    raw_rows = []
    raw_rows.extend(compute_sender_allocation_gini_rows("resource_proportional", resource_prop_snapshots))
    raw_rows.extend(
        compute_sender_allocation_gini_rows_from_policy_on_snapshots(
            "resource_proportional_agent",
            agent_snapshots,
            ResourceProportionalAllocationPolicy(),
            state_source="agent",
        )
    )
    raw_rows.extend(compute_sender_allocation_gini_rows("unit_investment_proportional", unit_investment_prop_snapshots))
    raw_rows.extend(
        compute_sender_allocation_gini_rows_from_policy_on_snapshots(
            "unit_investment_proportional_agent",
            agent_snapshots,
            UnitInvestmentProportionalAllocationPolicy(),
            state_source="agent",
        )
    )
    raw_rows.extend(
        compute_sender_allocation_gini_rows_from_policy_on_snapshots(
            "pool_power_mix_agent",
            agent_snapshots,
            PoolPowerMixAllocationPolicy(float(pool_power_mix_agent_k)),
            state_source="agent",
        )
    )
    raw_rows.extend(compute_sender_allocation_gini_rows("agent", agent_snapshots))
    raw_rows.extend(compute_sender_allocation_gini_rows("equal", equal_snapshots))
    mixture_raw_rows = compute_agent_unit_investment_equal_mixture_rows(agent_snapshots)

    episode_step_rows, step_summary_rows = summarize_sender_allocation_gini(raw_rows)
    mixture_episode_step_rows, mixture_step_summary_rows = summarize_agent_unit_investment_equal_mixture(mixture_raw_rows)

    episode_summary_rows = []
    episode_summary_rows.extend(summarize_episode_rows("resource_proportional", resource_prop_episode_summaries))
    episode_summary_rows.extend(summarize_episode_rows("unit_investment_proportional", unit_investment_prop_episode_summaries))
    episode_summary_rows.extend(summarize_episode_rows("agent", agent_episode_summaries))
    episode_summary_rows.extend(summarize_episode_rows("equal", equal_episode_summaries))

    style_by_policy = _sender_allocation_style_by_policy(float(pool_power_mix_agent_k))
    plot_sender_allocation_gini_over_time(
        step_summary_rows,
        output_dir / "sender_allocation_gini_over_time.png",
        style_by_policy=style_by_policy,
        policies=NON_AGENT_BASELINE_POLICIES,
        title="Sender Allocation Gini Over Time: Non-Agent Baselines",
    )
    plot_sender_allocation_gini_over_time(
        step_summary_rows,
        output_dir / "sender_gini_agent.png",
        style_by_policy=style_by_policy,
        policies=AGENT_REFERENCE_POLICIES,
        title="Sender Allocation Gini Over Time: Agent-State Baselines",
    )
    write_csv(output_dir / "sender_allocation_gini_raw_rows.csv", raw_rows)
    write_csv(output_dir / "sender_allocation_gini_episode_step_summary.csv", episode_step_rows)
    write_csv(output_dir / "sender_allocation_gini_step_summary.csv", step_summary_rows)
    write_csv(output_dir / "policy_episode_summary.csv", episode_summary_rows)
    plot_agent_unit_investment_equal_mixture_weights_over_time(
        mixture_step_summary_rows,
        output_dir / "agent_unitinv_equal_mix_w_over_time.png",
    )
    write_csv(output_dir / "agent_unitinv_equal_mix_rows.csv", mixture_raw_rows)
    write_csv(output_dir / "agent_unitinv_equal_mix_ep_step.csv", mixture_episode_step_rows)
    write_csv(output_dir / "agent_unitinv_equal_mix_step.csv", mixture_step_summary_rows)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_name": checkpoint_name,
        "completed_updates": int(checkpoint_payload.get("completed_updates", checkpoint_payload.get("update", 0))),
        "topology_name": topology_name,
        "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
        "topology_metadata": dict(topology_metadata),
        "episodes": int(episodes),
        "max_steps": None if max_steps is None else int(max_steps),
        "effective_episode_length": int(effective_spec["dynamics"]["episode_length"]),
        "policy_labels": {
            "resource_proportional": "Proportional by Current Resources",
            "resource_proportional_agent": "Proportional by Current Resources (Agent)",
            "unit_investment_proportional": "Proportional by Current Unit Investment",
            "unit_investment_proportional_agent": "Proportional by Current Unit Investment (Agent)",
            "pool_power_mix_agent": f"PoolPowerMix k={float(pool_power_mix_agent_k):g} (Agent)",
            "agent": "RL Agent",
            "equal": "Equal Allocation",
        },
        "matplotlib_available": plt is not None,
        "step_summary_rows": step_summary_rows,
        "agent_unit_investment_equal_mixture_step_summary_rows": mixture_step_summary_rows,
    }
    with (output_dir / "sender_allocation_gini_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    final_step_by_policy: dict[str, float] = {}
    for policy_name in (
        "resource_proportional",
        "resource_proportional_agent",
        "unit_investment_proportional",
        "unit_investment_proportional_agent",
        "pool_power_mix_agent",
        "agent",
        "equal",
    ):
        policy_rows = [row for row in step_summary_rows if str(row["policy"]) == policy_name]
        if not policy_rows:
            continue
        last_row = max(policy_rows, key=lambda row: int(row["step"]))
        final_step_by_policy[policy_name] = float(last_row["sender_allocation_gini_mean"])

    return {
        "topology_summary_row": {
            "topology": topology_name,
            "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
            "graph_source_mode": str(topology_metadata.get("graph_source_mode", "unknown")),
            "train_distribution_graph_bank_index": topology_metadata.get("train_distribution_graph_bank_index"),
            "graph_seed": topology_metadata.get("graph_seed"),
            "output_dir": str(output_dir),
            "final_step_sender_allocation_gini": final_step_by_policy,
        },
        "step_summary_rows": [
            {
                **dict(row),
                "topology": topology_name,
                "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
            }
            for row in step_summary_rows
        ],
        "episode_step_rows": [
            {
                **dict(row),
                "topology": topology_name,
                "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
            }
            for row in episode_step_rows
        ],
        "mixture_episode_step_rows": [
            {
                **dict(row),
                "topology": topology_name,
                "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
            }
            for row in mixture_episode_step_rows
        ],
        "mixture_raw_rows": [
            {
                **dict(row),
                "topology": topology_name,
                "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
            }
            for row in mixture_raw_rows
        ],
    }


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
            else (run_dir / "sender_allocation_gini_over_time").resolve()
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_distribution_topologies = _parse_optional_topology_subset(args.train_distribution_topologies)

    import torch

    device = torch.device(args.device)
    actor, checkpoint_payload = load_actor_from_run_dir(run_dir, args.checkpoint_name, device)
    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)
    topology_cases = resolve_topology_cases(
        args.topologies,
        base_spec=experiment_spec,
        reference_mean_degree=reference_mean_degree,
        train_distribution_topologies=train_distribution_topologies,
        train_distribution_graph_bank_index=int(args.train_distribution_graph_bank_index),
    )

    topology_summary_rows: list[dict[str, Any]] = []
    all_topology_step_summary_rows: list[dict[str, Any]] = []
    all_topology_episode_step_rows: list[dict[str, Any]] = []
    all_topology_mixture_episode_step_rows: list[dict[str, Any]] = []
    all_topology_mixture_raw_rows: list[dict[str, Any]] = []
    for topology_index, case in enumerate(topology_cases):
        topology_name = str(case["topology_name"])
        topology_spec = deepcopy(dict(case["spec"]))
        topology_spec = apply_analysis_env_overrides(
            topology_spec,
            env_r_override=None if args.env_r_override is None else float(args.env_r_override),
        )
        topology_output_dir = output_dir if len(topology_cases) == 1 else (output_dir / topology_name)
        result = analyze_single_topology(
            actor=actor,
            run_dir=run_dir,
            checkpoint_name=args.checkpoint_name,
            checkpoint_payload=checkpoint_payload,
            output_dir=topology_output_dir,
            spec=topology_spec,
            topology_name=topology_name,
            topology_metadata=dict(case.get("metadata", {})),
            episodes=max(1, int(args.episodes)),
            max_steps=args.max_steps,
            episode_length_override=args.episode_length_override,
            rollout_seed=int(args.seed) + topology_index,
            pool_power_mix_agent_k=float(args.pool_power_mix_agent_k),
        )
        topology_summary_rows.append(dict(result["topology_summary_row"]))
        all_topology_step_summary_rows.extend([dict(row) for row in result["step_summary_rows"]])
        all_topology_episode_step_rows.extend([dict(row) for row in result["episode_step_rows"]])
        all_topology_mixture_episode_step_rows.extend([dict(row) for row in result["mixture_episode_step_rows"]])
        all_topology_mixture_raw_rows.extend([dict(row) for row in result["mixture_raw_rows"]])

    write_csv(output_dir / "topology_summary.csv", topology_summary_rows)
    write_csv(output_dir / "all_topologies_sender_allocation_gini_step_summary.csv", all_topology_step_summary_rows)
    with (output_dir / "topology_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(topology_summary_rows, handle, ensure_ascii=False, indent=2)
    if len(topology_cases) > 1:
        style_by_policy = _sender_allocation_style_by_policy(float(args.pool_power_mix_agent_k))
        aggregated_step_summary_rows = summarize_sender_allocation_gini_episode_rows_across_topologies(
            all_topology_episode_step_rows
        )
        aggregated_mixture_step_summary_rows = summarize_agent_unit_investment_equal_mixture_episode_rows_across_topologies(
            all_topology_mixture_episode_step_rows
        )
        aggregated_mixture_vs_ego_mean_rows = summarize_agent_unit_investment_equal_mixture_vs_resource(
            all_topology_mixture_raw_rows,
            resource_key="ego_mean_resource",
            num_bins=int(args.aggregated_mixture_resource_bin_count),
        )
        aggregated_mixture_vs_ego_total_rows = summarize_agent_unit_investment_equal_mixture_vs_resource(
            all_topology_mixture_raw_rows,
            resource_key="ego_total_resource",
            num_bins=int(args.aggregated_mixture_resource_bin_count),
        )
        write_csv(
            output_dir / "aggregated_topologies_sender_allocation_gini_step_summary.csv",
            aggregated_step_summary_rows,
        )
        write_csv(
            output_dir / "agg_topos_agent_unitinv_equal_mix_step.csv",
            aggregated_mixture_step_summary_rows,
        )
        write_csv(
            output_dir / "agg_topos_agent_unitinv_equal_mix_vs_ego_mean_res.csv",
            aggregated_mixture_vs_ego_mean_rows,
        )
        write_csv(
            output_dir / "agg_topos_agent_unitinv_equal_mix_vs_ego_total_res.csv",
            aggregated_mixture_vs_ego_total_rows,
        )
        plot_sender_allocation_gini_over_time(
            aggregated_step_summary_rows,
            output_dir / "aggregated_topologies_sender_allocation_gini_over_time.png",
            style_by_policy=style_by_policy,
            policies=NON_AGENT_BASELINE_POLICIES,
            title="Aggregated Sender Allocation Gini Over Time: Non-Agent Baselines",
        )
        plot_sender_allocation_gini_over_time(
            aggregated_step_summary_rows,
            output_dir / "agg_topos_sender_gini_agent.png",
            style_by_policy=style_by_policy,
            policies=AGENT_REFERENCE_POLICIES,
            title="Aggregated Sender Allocation Gini Over Time: Agent-State Baselines",
        )
        plot_agent_unit_investment_equal_mixture_weights_over_time(
            aggregated_mixture_step_summary_rows,
            output_dir / "agg_topos_agent_unitinv_equal_mix_w_over_time.png",
        )
        plot_agent_unit_investment_equal_mixture_vs_resource(
            aggregated_mixture_vs_ego_mean_rows,
            output_dir / "agg_topos_agent_unitinv_equal_mix_vs_ego_mean_res.png",
            resource_label="Sender Ego Mean Resource",
        )
        plot_agent_unit_investment_equal_mixture_vs_resource(
            aggregated_mixture_vs_ego_total_rows,
            output_dir / "agg_topos_agent_unitinv_equal_mix_vs_ego_total_res.png",
            resource_label="Sender Ego Total Resource",
        )
        plot_sender_allocation_gini_over_time_by_topology(
            all_topology_step_summary_rows,
            output_dir / "all_topologies_sender_allocation_gini_over_time.png",
            style_by_policy=style_by_policy,
            policies=NON_AGENT_BASELINE_POLICIES,
            title_prefix="Non-Agent Baselines",
        )
        plot_sender_allocation_gini_over_time_by_topology(
            all_topology_step_summary_rows,
            output_dir / "all_topos_sender_gini_agent.png",
            style_by_policy=style_by_policy,
            policies=AGENT_REFERENCE_POLICIES,
            title_prefix="Agent-State Baselines",
        )

    print(f"Sender allocation gini analysis complete. Artifacts written to: {output_dir}")
    if plt is None:
        print("Plot export skipped because matplotlib is not installed in the current interpreter.")


if __name__ == "__main__":
    main()
