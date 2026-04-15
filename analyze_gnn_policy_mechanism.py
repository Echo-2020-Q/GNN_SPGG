from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

from Project1.env import SPGGEnv

from analyze_gnn_policy_decisions import (
    TOPOLOGY_LABELS,
    EpisodeSummary,
    Snapshot,
    _chunked_policy_forward,
    _load_json,
    _observation_copy,
    build_env_config_from_spec,
    build_graph_from_spec,
    build_spec_for_topology,
    collect_graph_stats,
    compute_policy_behavior_stats,
    load_actor_from_run_dir,
    parse_topology_list,
    resolve_reference_mean_degree,
    run_deterministic_rollouts,
    write_csv,
    write_episode_summary_csv,
)


SCRIPT_DEFAULTS = {
    # 训练结果目录。
    # 目录下需要至少包含：
    # 1. results.json：用于恢复实验配置；
    # 2. checkpoints/：用于加载训练好的策略权重。
    "run_dir": "outputs/Pool_dynamic/0414_20Mspgg_GNN_50Nodes_200length_Fermi_TD3_Regular&BA_Codex'sParam",

    # 要加载的 checkpoint 文件名。
    # 常用值通常是 best_eval.pt / final.pt / latest.pt。
    "checkpoint_name": "best_eval.pt",

    # 机制分析结果输出目录。
    # 设为 None 时，会继续看 output_subdir_name。
    # 两者都为 None 时，默认输出到：
    # <run_dir>/policy_mechanism/<checkpoint_name去掉后缀>/
    "output_dir": None,

    # 当 output_dir 为 None 时，把结果输出到：
    # <run_dir>/<output_subdir_name>/
    # 设为 None 时，回退到默认目录：
    # <run_dir>/policy_mechanism/<checkpoint_name去掉后缀>/
    "output_subdir_name": "0414_200length_policy_mechanism",

    # 要分析的图拓扑，多个值用英文逗号分隔。
    # 支持：original, regular, er, ws, ba。
    # 这个脚本只做机制分析，所以默认给 original，避免一上来就做跨拓扑批量跑。
    "topologies": "original",

    # 每种拓扑下跑多少个 deterministic episode。
    # 越大越稳，但 CSV 也会更大、运行也更慢。
    "episodes": 10,

    # 每个 episode 最多分析多少步。
    # 设为 None 表示跑完整个 episode_length。
    "max_steps": 500,

    # 是否覆盖分析环境里的 episode_length。
    # 如果你想看“200 步训练策略在 500 步长期演化下的机制”，这里就可以设成 500。
    "episode_length_override": 200,

    # 把 P_grown / P_upperbound 按多少个分位数区间做分箱统计。
    # 更小更平滑，更大更细。
    "mechanism_bin_count": 10,

    # labor 基线和 equal 基线的 L1 距离阈值。
    # 小于这个阈值时，说明这两种基线在当前行上几乎重合，
    # 不应该再强行解读成“更偏 labor 还是更偏 equal”。
    "labor_equal_gap_threshold": 0.05,

    # 是否执行“提高某个接收节点贡献信号后，发送节点是否给它更多分配”的反事实测试。
    # True 会额外输出 counterfactual_contribution_response.csv / summary.csv / png。
    "enable_counterfactual_analysis": True,

    # 反事实测试最多抽多少个 sender-row 作为样本。
    # 每个样本行会对该 sender 的所有可接收节点分别做一次“贡献抬高”测试。
    "counterfactual_row_sample_size": 512,

    # 反事实里把目标节点的 strategy_norm 最多上调多少。
    # 这里是“相对当前值加多少”，然后再截断到 [0, 1]。
    "counterfactual_contribution_delta": 0.20,

    # 反事实离线前向推理的 batch size。
    # 更大更快，但更吃显存/内存。
    "counterfactual_batch_size": 128,

    # rollout 的环境随机种子基准值。
    # 第 0 个 episode 用 seed，第 1 个用 seed+1，以此类推。
    "seed": 42,

    # Torch 推理设备。
    # 这份脚本默认设成 cpu，避免在没有 CUDA 的环境里直接报错。
    "device": "cuda:0",
}


def _safe_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum())
    if total <= eps:
        if array.size == 0:
            return array.copy()
        return np.full(array.shape, 1.0 / float(array.size), dtype=np.float64)
    return array / total


def _nanmean_or_nan(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return float("nan")
    return float(np.mean(array[finite_mask]))


def _project_to_simplex(weights: np.ndarray) -> np.ndarray:
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        return vector.copy()
    sorted_vector = np.sort(vector)[::-1]
    cssv = np.cumsum(sorted_vector) - 1.0
    rho_candidates = sorted_vector - (cssv / (np.arange(vector.size) + 1))
    positive = np.nonzero(rho_candidates > 0.0)[0]
    if positive.size == 0:
        return np.full(vector.shape, 1.0 / float(vector.size), dtype=np.float64)
    rho = int(positive[-1])
    theta = cssv[rho] / float(rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    projected_sum = float(projected.sum())
    if projected_sum <= 1e-12:
        return np.full(vector.shape, 1.0 / float(vector.size), dtype=np.float64)
    return projected / projected_sum


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array.copy()
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _pearson_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x_array = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x_array.size != y_array.size or x_array.size < 2:
        return float("nan")
    x_centered = x_array - x_array.mean()
    y_centered = y_array - y_array.mean()
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def _spearman_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x_array = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y_values, dtype=np.float64).reshape(-1)
    valid_mask = np.isfinite(x_array) & np.isfinite(y_array)
    if valid_mask.sum() < 2:
        return float("nan")
    x_rank = _rankdata_average(x_array[valid_mask])
    y_rank = _rankdata_average(y_array[valid_mask])
    return _pearson_corr(x_rank, y_rank)


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


def _entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(-(safe * np.log(safe)).sum())


def _js_divergence(prob_p: np.ndarray, prob_q: np.ndarray) -> float:
    p = np.clip(np.asarray(prob_p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(prob_q, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    mean = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * (np.log(p) - np.log(mean))) + 0.5 * np.sum(q * (np.log(q) - np.log(mean)))
    )


def _fit_labor_equal_lambda(
    allocation_row: np.ndarray,
    labor_row: np.ndarray,
    equal_row: np.ndarray,
) -> float:
    direction = labor_row - equal_row
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return float("nan")
    estimate = float(np.dot(allocation_row - equal_row, direction) / denominator)
    return float(np.clip(estimate, 0.0, 1.0))


def _fit_three_way_mechanism_weights(
    allocation_row: np.ndarray,
    labor_row: np.ndarray,
    equal_row: np.ndarray,
    self_row: np.ndarray,
) -> tuple[np.ndarray, float]:
    if allocation_row.size == 0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float64), float("nan")

    reduced_basis = np.column_stack([labor_row - equal_row, self_row - equal_row])
    target = allocation_row - equal_row
    if np.allclose(reduced_basis, 0.0):
        weights = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
        fit_error = float(np.mean(np.abs(allocation_row - equal_row)))
        return weights, fit_error

    raw_solution, _, _, _ = np.linalg.lstsq(reduced_basis, target, rcond=None)
    raw_labor = float(raw_solution[0]) if raw_solution.size >= 1 else 0.0
    raw_self = float(raw_solution[1]) if raw_solution.size >= 2 else 0.0
    raw_equal = 1.0 - raw_labor - raw_self
    weights = _project_to_simplex(np.asarray([raw_labor, raw_equal, raw_self], dtype=np.float64))
    fitted = (weights[0] * labor_row) + (weights[1] * equal_row) + (weights[2] * self_row)
    fit_error = float(np.mean(np.abs(fitted - allocation_row)))
    return weights, fit_error


def _build_row_baselines(
    observation: Mapping[str, np.ndarray],
    sender: int,
    valid_receivers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    receiver_count = int(valid_receivers.size)
    equal_row = np.full(receiver_count, 1.0 / float(max(receiver_count, 1)), dtype=np.float64)
    receiver_investment = np.asarray(observation["investment"][valid_receivers], dtype=np.float64)
    labor_row = _safe_normalize(receiver_investment)

    self_row = np.zeros(receiver_count, dtype=np.float64)
    self_matches = np.nonzero(valid_receivers == sender)[0]
    if self_matches.size == 0:
        raise RuntimeError("Sender must appear in its own receiver set.")
    self_row[int(self_matches[0])] = 1.0
    return labor_row, equal_row, self_row


def _labor_equal_gap_l1(labor_row: np.ndarray, equal_row: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(labor_row, dtype=np.float64) - np.asarray(equal_row, dtype=np.float64))))


def _compute_pool_theoretical_max_for_observation(env_config: Any, thresholds: np.ndarray) -> np.ndarray:
    if str(env_config.p_mode) == "constant":
        return np.full(thresholds.shape, float(env_config.p_max), dtype=np.float64)
    if str(env_config.p_mode) == "dynamic":
        return float(env_config.p_c) * thresholds
    raise RuntimeError(f"Unsupported p_mode: {env_config.p_mode}")


def _compute_pool_capacity_for_observation(
    env_config: Any,
    local_actual_cooperators: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    if str(env_config.p_mode) == "constant":
        return np.full(thresholds.shape, float(env_config.p_max), dtype=np.float64)
    if str(env_config.p_mode) == "dynamic":
        return float(env_config.p_c) * np.square(local_actual_cooperators) / thresholds
    raise RuntimeError(f"Unsupported p_mode: {env_config.p_mode}")


def _build_counterfactual_contribution_observation(
    observation: Mapping[str, np.ndarray],
    *,
    env_config: Any,
    target_node: int,
    contribution_delta: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    perturbed = _observation_copy(observation)
    resources = np.asarray(perturbed["resources"], dtype=np.float64)
    degrees = np.asarray(perturbed["degrees"], dtype=np.float64)
    thresholds = degrees + 1.0
    investment = np.asarray(perturbed["investment"], dtype=np.float64)
    strategy_norm = np.asarray(perturbed["strategy_norm"], dtype=np.float64)
    x_actual = np.asarray(perturbed["x_actual"], dtype=np.float64)
    x_nominal = np.asarray(perturbed["x_nominal"])
    local_mask_float = np.asarray(perturbed["local_mask"], dtype=np.float64)

    resource_value = float(resources[target_node])
    threshold_value = float(thresholds[target_node])
    before_investment = float(investment[target_node])
    before_strategy_norm = float(strategy_norm[target_node])
    before_x_actual = float(x_actual[target_node])
    desired_strategy_norm = float(np.clip(before_strategy_norm + contribution_delta, 0.0, 1.0))

    if resource_value <= 1e-12 or resource_value < threshold_value - 1e-12:
        after_investment = before_investment
        after_strategy_norm = before_strategy_norm
        after_x_actual = before_x_actual
    else:
        after_investment = min(resource_value, max(before_investment, desired_strategy_norm * resource_value))
        after_strategy_norm = after_investment / resource_value if resource_value > 1e-12 else 0.0
        after_x_actual = 1.0 if after_investment > 1e-12 else 0.0

    investment[target_node] = after_investment
    strategy_norm[target_node] = after_strategy_norm
    x_actual[target_node] = after_x_actual
    x_nominal[target_node] = np.asarray(int(after_x_actual > 0.5), dtype=x_nominal.dtype)

    unit_investment = np.divide(
        investment,
        thresholds,
        out=np.zeros_like(investment, dtype=np.float64),
        where=thresholds > 1e-12,
    )
    pool_raw = local_mask_float @ unit_investment
    local_actual_cooperators = local_mask_float @ x_actual
    pool_theoretical_max = _compute_pool_theoretical_max_for_observation(env_config, thresholds)
    pool_capacity = _compute_pool_capacity_for_observation(env_config, local_actual_cooperators, thresholds)
    pool_grown = np.minimum((1.0 + float(env_config.r)) * pool_raw, pool_capacity)
    pool_raw_norm = np.divide(
        pool_raw,
        pool_theoretical_max,
        out=np.zeros_like(pool_raw, dtype=np.float64),
        where=pool_theoretical_max > 1e-8,
    )

    perturbed["investment"] = investment.astype(observation["investment"].dtype, copy=False)
    perturbed["strategy_norm"] = strategy_norm.astype(observation["strategy_norm"].dtype, copy=False)
    perturbed["x_actual"] = x_actual.astype(observation["x_actual"].dtype, copy=False)
    perturbed["x_nominal"] = x_nominal.astype(observation["x_nominal"].dtype, copy=False)
    perturbed["unit_investment"] = unit_investment.astype(np.float64, copy=False)
    perturbed["pool_raw"] = pool_raw.astype(np.float64, copy=False)
    perturbed["local_actual_cooperators"] = local_actual_cooperators.astype(np.float64, copy=False)
    perturbed["pool_theoretical_max"] = pool_theoretical_max.astype(np.float64, copy=False)
    perturbed["pool_capacity"] = pool_capacity.astype(np.float64, copy=False)
    perturbed["pool_grown"] = pool_grown.astype(np.float64, copy=False)
    perturbed["pool_raw_norm"] = pool_raw_norm.astype(np.float64, copy=False)
    perturbed["p_max"] = pool_theoretical_max.astype(np.float64, copy=False)

    metadata = {
        "target_resource": resource_value,
        "target_threshold": threshold_value,
        "baseline_target_investment": before_investment,
        "counterfactual_target_investment": after_investment,
        "baseline_target_strategy_norm": before_strategy_norm,
        "counterfactual_target_strategy_norm": after_strategy_norm,
        "baseline_target_x_actual": before_x_actual,
        "counterfactual_target_x_actual": after_x_actual,
        "target_strategy_norm_delta": after_strategy_norm - before_strategy_norm,
        "target_investment_delta": after_investment - before_investment,
        "target_x_actual_delta": after_x_actual - before_x_actual,
        "target_can_meet_threshold": int(resource_value >= threshold_value - 1e-12),
        "effective_intervention": int(
            (abs(after_investment - before_investment) > 1e-12)
            or (abs(after_x_actual - before_x_actual) > 1e-12)
        ),
    }
    return perturbed, metadata


def compute_row_mechanism_records(
    snapshots: Sequence[Snapshot],
    *,
    p_c: float,
    labor_equal_gap_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        observation = snapshot.observation
        policy = snapshot.policy
        local_mask = observation["local_mask"].astype(bool, copy=False)
        num_nodes = local_mask.shape[0]

        for sender in range(num_nodes):
            valid_receivers = np.flatnonzero(local_mask[sender])
            allocation_row = np.asarray(policy.allocation_matrix[sender, valid_receivers], dtype=np.float64)
            labor_row, equal_row, self_row = _build_row_baselines(observation, sender, valid_receivers)
            labor_equal_gap_l1 = _labor_equal_gap_l1(labor_row, equal_row)
            js_labor = _js_divergence(allocation_row, labor_row)
            js_equal = _js_divergence(allocation_row, equal_row)
            js_self = _js_divergence(allocation_row, self_row)
            lambda_labor_equal = _fit_labor_equal_lambda(allocation_row, labor_row, equal_row)
            mechanism_weights, fit_error = _fit_three_way_mechanism_weights(
                allocation_row,
                labor_row,
                equal_row,
                self_row,
            )

            pool_value = float(observation["pool_grown"][sender])
            sender_degree = max(int(valid_receivers.size) - 1, 0)
            pool_upperbound = float(sender_degree) * float(p_c)
            pool_position_ratio = pool_value / pool_upperbound if pool_upperbound > 1e-12 else float("nan")
            ego_resources = np.asarray(observation["resources"][valid_receivers], dtype=np.float64)
            ego_investment = np.asarray(observation["investment"][valid_receivers], dtype=np.float64)
            self_index = int(np.nonzero(valid_receivers == sender)[0][0])
            top_receiver = int(valid_receivers[np.argmax(allocation_row)])

            rows.append(
                {
                    "episode": snapshot.episode,
                    "step": snapshot.step,
                    "sender": sender,
                    "sender_degree": sender_degree,
                    "receiver_count": int(valid_receivers.size),
                    "sender_p_c": float(p_c),
                    "sender_pool_grown": pool_value,
                    "sender_pool_upperbound": pool_upperbound,
                    "pool_grown_over_upperbound": pool_position_ratio,
                    "ego_total_resources": float(ego_resources.sum()),
                    "ego_total_investment": float(ego_investment.sum()),
                    "sender_resource": float(observation["resources"][sender]),
                    "sender_investment": float(observation["investment"][sender]),
                    "sender_strategy_norm": float(observation["strategy_norm"][sender]),
                    "sender_x_actual": float(observation["x_actual"][sender]),
                    "self_allocation": float(allocation_row[self_index]),
                    "top1_receiver": top_receiver,
                    "top1_is_self": int(top_receiver == sender),
                    "row_entropy": _entropy(allocation_row),
                    "top1_mass": float(np.max(allocation_row)),
                    "js_to_labor": js_labor,
                    "js_to_equal": js_equal,
                    "js_to_self": js_self,
                    "labor_equal_gap_l1": labor_equal_gap_l1,
                    "labor_equal_gap_tv": 0.5 * labor_equal_gap_l1,
                    "labor_equal_identifiable": int(labor_equal_gap_l1 >= float(labor_equal_gap_threshold)),
                    "delta_equal_minus_labor": js_equal - js_labor,
                    "delta_self_minus_labor": js_self - js_labor,
                    "lambda_labor_equal": lambda_labor_equal,
                    "weight_labor": float(mechanism_weights[0]) if np.isfinite(mechanism_weights[0]) else float("nan"),
                    "weight_equal": float(mechanism_weights[1]) if np.isfinite(mechanism_weights[1]) else float("nan"),
                    "weight_self": float(mechanism_weights[2]) if np.isfinite(mechanism_weights[2]) else float("nan"),
                    "three_way_fit_l1": fit_error,
                }
            )
    return rows


def compute_node_income_decomposition(snapshots: Sequence[Snapshot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        observation = snapshot.observation
        policy = snapshot.policy
        local_mask = observation["local_mask"].astype(bool, copy=False)
        num_nodes = local_mask.shape[0]

        equal_incoming = np.zeros(num_nodes, dtype=np.float64)
        labor_incoming = np.zeros(num_nodes, dtype=np.float64)
        self_incoming = np.zeros(num_nodes, dtype=np.float64)
        sender_counts = np.zeros(num_nodes, dtype=np.int64)

        for sender in range(num_nodes):
            valid_receivers = np.flatnonzero(local_mask[sender])
            labor_row, equal_row, self_row = _build_row_baselines(observation, sender, valid_receivers)
            pool_value = float(observation["pool_grown"][sender])
            equal_incoming[valid_receivers] += pool_value * equal_row
            labor_incoming[valid_receivers] += pool_value * labor_row
            self_incoming[valid_receivers] += pool_value * self_row
            sender_counts[valid_receivers] += 1

        actual_incoming = np.asarray(policy.incoming_resources, dtype=np.float64)
        for receiver in range(num_nodes):
            equal_denom = float(equal_incoming[receiver])
            labor_denom = float(labor_incoming[receiver])
            rows.append(
                {
                    "episode": snapshot.episode,
                    "step": snapshot.step,
                    "receiver": receiver,
                    "actual_incoming": float(actual_incoming[receiver]),
                    "expected_incoming_equal": float(equal_incoming[receiver]),
                    "expected_incoming_labor": float(labor_incoming[receiver]),
                    "expected_incoming_self": float(self_incoming[receiver]),
                    "actual_minus_equal": float(actual_incoming[receiver] - equal_incoming[receiver]),
                    "actual_minus_labor": float(actual_incoming[receiver] - labor_incoming[receiver]),
                    "actual_over_equal": (
                        float(actual_incoming[receiver] / equal_denom) if equal_denom > 1e-12 else float("nan")
                    ),
                    "actual_over_labor": (
                        float(actual_incoming[receiver] / labor_denom) if labor_denom > 1e-12 else float("nan")
                    ),
                    "receiver_resource": float(observation["resources"][receiver]),
                    "receiver_investment": float(observation["investment"][receiver]),
                    "receiver_strategy_norm": float(observation["strategy_norm"][receiver]),
                    "receiver_x_actual": float(observation["x_actual"][receiver]),
                    "receiver_degree": int(sender_counts[receiver] - 1),
                    "receiver_exposure_count": int(sender_counts[receiver]),
                }
            )
    return rows


def summarize_row_mechanism_records(
    row_records: Sequence[Mapping[str, Any]],
    *,
    bin_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not row_records:
        return {}, []

    position_ratio = np.asarray([float(row["pool_grown_over_upperbound"]) for row in row_records], dtype=np.float64)
    lambda_values = np.asarray([float(row["lambda_labor_equal"]) for row in row_records], dtype=np.float64)
    delta_values = np.asarray([float(row["delta_equal_minus_labor"]) for row in row_records], dtype=np.float64)
    gap_values = np.asarray([float(row["labor_equal_gap_l1"]) for row in row_records], dtype=np.float64)
    identifiable_flags = np.asarray([float(row["labor_equal_identifiable"]) for row in row_records], dtype=np.float64)
    labor_weights = np.asarray([float(row["weight_labor"]) for row in row_records], dtype=np.float64)
    equal_weights = np.asarray([float(row["weight_equal"]) for row in row_records], dtype=np.float64)
    self_weights = np.asarray([float(row["weight_self"]) for row in row_records], dtype=np.float64)
    js_labor = np.asarray([float(row["js_to_labor"]) for row in row_records], dtype=np.float64)
    js_equal = np.asarray([float(row["js_to_equal"]) for row in row_records], dtype=np.float64)
    js_self = np.asarray([float(row["js_to_self"]) for row in row_records], dtype=np.float64)
    fit_errors = np.asarray([float(row["three_way_fit_l1"]) for row in row_records], dtype=np.float64)

    identifiable_mask = identifiable_flags > 0.5
    summary = {
        "row_count": int(len(row_records)),
        "pool_grown_over_upperbound_mean": _nanmean_or_nan(position_ratio),
        "labor_equal_gap_l1_mean": _nanmean_or_nan(gap_values),
        "labor_equal_gap_l1_median": float(np.nanmedian(gap_values)),
        "labor_equal_identifiable_frac": _nanmean_or_nan(identifiable_flags),
        "lambda_labor_equal_mean": _nanmean_or_nan(lambda_values),
        "lambda_labor_equal_mean_identifiable": _nanmean_or_nan(lambda_values[identifiable_mask]),
        "delta_equal_minus_labor_mean": _nanmean_or_nan(delta_values),
        "delta_equal_minus_labor_mean_identifiable": _nanmean_or_nan(delta_values[identifiable_mask]),
        "weight_labor_mean": _nanmean_or_nan(labor_weights),
        "weight_labor_mean_identifiable": _nanmean_or_nan(labor_weights[identifiable_mask]),
        "weight_equal_mean": _nanmean_or_nan(equal_weights),
        "weight_equal_mean_identifiable": _nanmean_or_nan(equal_weights[identifiable_mask]),
        "weight_self_mean": _nanmean_or_nan(self_weights),
        "js_to_labor_mean": _nanmean_or_nan(js_labor),
        "js_to_equal_mean": _nanmean_or_nan(js_equal),
        "js_to_self_mean": _nanmean_or_nan(js_self),
        "three_way_fit_l1_mean": _nanmean_or_nan(fit_errors),
        "pool_grown_over_upperbound_vs_lambda_spearman": _spearman_corr(position_ratio, lambda_values),
        "pool_grown_over_upperbound_vs_delta_spearman": _spearman_corr(position_ratio, delta_values),
        "pool_grown_over_upperbound_vs_gap_spearman": _spearman_corr(position_ratio, gap_values),
        "pool_grown_over_upperbound_vs_weight_labor_spearman": _spearman_corr(position_ratio, labor_weights),
        "pool_grown_over_upperbound_vs_weight_equal_spearman": _spearman_corr(position_ratio, equal_weights),
        "pool_grown_over_upperbound_vs_weight_self_spearman": _spearman_corr(position_ratio, self_weights),
    }

    edges = _quantile_edges(position_ratio[np.isfinite(position_ratio)], max(2, int(bin_count)))
    bin_rows: list[dict[str, Any]] = []
    if edges.size >= 2:
        for bin_index in range(edges.size - 1):
            left = float(edges[bin_index])
            right = float(edges[bin_index + 1])
            if bin_index == edges.size - 2:
                mask = (position_ratio >= left) & (position_ratio <= right)
            else:
                mask = (position_ratio >= left) & (position_ratio < right)
            if not np.any(mask):
                continue
            bin_rows.append(
                {
                    "bin_index": int(bin_index),
                    "pool_grown_over_upperbound_left": left,
                    "pool_grown_over_upperbound_right": right,
                    "count_rows": int(mask.sum()),
                    "pool_grown_over_upperbound_mean": _nanmean_or_nan(position_ratio[mask]),
                    "labor_equal_gap_l1_mean": _nanmean_or_nan(gap_values[mask]),
                    "labor_equal_identifiable_frac": _nanmean_or_nan(identifiable_flags[mask]),
                    "lambda_labor_equal_mean": _nanmean_or_nan(lambda_values[mask]),
                    "delta_equal_minus_labor_mean": _nanmean_or_nan(delta_values[mask]),
                    "weight_labor_mean": _nanmean_or_nan(labor_weights[mask]),
                    "weight_equal_mean": _nanmean_or_nan(equal_weights[mask]),
                    "weight_self_mean": _nanmean_or_nan(self_weights[mask]),
                    "js_to_labor_mean": _nanmean_or_nan(js_labor[mask]),
                    "js_to_equal_mean": _nanmean_or_nan(js_equal[mask]),
                    "js_to_self_mean": _nanmean_or_nan(js_self[mask]),
                    "three_way_fit_l1_mean": _nanmean_or_nan(fit_errors[mask]),
                }
            )
    return summary, bin_rows


def summarize_node_income_records(node_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not node_records:
        return {}
    actual_over_equal = np.asarray([float(row["actual_over_equal"]) for row in node_records], dtype=np.float64)
    actual_over_labor = np.asarray([float(row["actual_over_labor"]) for row in node_records], dtype=np.float64)
    actual_minus_equal = np.asarray([float(row["actual_minus_equal"]) for row in node_records], dtype=np.float64)
    actual_minus_labor = np.asarray([float(row["actual_minus_labor"]) for row in node_records], dtype=np.float64)
    return {
        "node_record_count": int(len(node_records)),
        "actual_over_equal_mean": float(np.nanmean(actual_over_equal)),
        "actual_over_labor_mean": float(np.nanmean(actual_over_labor)),
        "actual_minus_equal_mean": float(np.nanmean(actual_minus_equal)),
        "actual_minus_labor_mean": float(np.nanmean(actual_minus_labor)),
    }


def summarize_position_fixed_bins(
    row_records: Sequence[Mapping[str, Any]],
    *,
    bin_count: int,
) -> list[dict[str, Any]]:
    if not row_records:
        return []

    ratio = np.asarray([float(row["pool_grown_over_upperbound"]) for row in row_records], dtype=np.float64)
    lambda_values = np.asarray([float(row["lambda_labor_equal"]) for row in row_records], dtype=np.float64)
    delta_values = np.asarray([float(row["delta_equal_minus_labor"]) for row in row_records], dtype=np.float64)
    gap_values = np.asarray([float(row["labor_equal_gap_l1"]) for row in row_records], dtype=np.float64)
    identifiable_flags = np.asarray([float(row["labor_equal_identifiable"]) for row in row_records], dtype=np.float64)
    labor_weights = np.asarray([float(row["weight_labor"]) for row in row_records], dtype=np.float64)
    equal_weights = np.asarray([float(row["weight_equal"]) for row in row_records], dtype=np.float64)
    self_weights = np.asarray([float(row["weight_self"]) for row in row_records], dtype=np.float64)
    self_alloc = np.asarray([float(row["self_allocation"]) for row in row_records], dtype=np.float64)
    finite_ratio = ratio[np.isfinite(ratio)]
    if finite_ratio.size == 0:
        return []

    left_edge = float(np.min(finite_ratio))
    right_edge = float(np.max(finite_ratio))
    if right_edge <= left_edge:
        right_edge = left_edge + 1e-9
    edges = np.linspace(left_edge, right_edge, num=max(2, int(bin_count)) + 1)

    bin_rows: list[dict[str, Any]] = []
    for bin_index in range(edges.size - 1):
        left = float(edges[bin_index])
        right = float(edges[bin_index + 1])
        if bin_index == edges.size - 2:
            mask = (ratio >= left) & (ratio <= right)
        else:
            mask = (ratio >= left) & (ratio < right)
        if not np.any(mask):
            continue
        bin_rows.append(
            {
                "bin_index": int(bin_index),
                "pool_grown_over_upperbound_left": left,
                "pool_grown_over_upperbound_right": right,
                "count_rows": int(mask.sum()),
                "pool_grown_over_upperbound_mean": _nanmean_or_nan(ratio[mask]),
                "labor_equal_gap_l1_mean": _nanmean_or_nan(gap_values[mask]),
                "labor_equal_identifiable_frac": _nanmean_or_nan(identifiable_flags[mask]),
                "lambda_labor_equal_mean": _nanmean_or_nan(lambda_values[mask]),
                "delta_equal_minus_labor_mean": _nanmean_or_nan(delta_values[mask]),
                "weight_labor_mean": _nanmean_or_nan(labor_weights[mask]),
                "weight_equal_mean": _nanmean_or_nan(equal_weights[mask]),
                "weight_self_mean": _nanmean_or_nan(self_weights[mask]),
                "self_allocation_mean": _nanmean_or_nan(self_alloc[mask]),
            }
        )
    return bin_rows


def summarize_step_mechanism_records(row_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not row_records:
        return []

    per_step: dict[int, dict[str, list[float]]] = {}
    for row in row_records:
        step = int(row["step"])
        bucket = per_step.setdefault(
            step,
            {
                "ratio": [],
                "lambda": [],
                "gap": [],
                "identifiable": [],
                "labor": [],
                "equal": [],
                "self": [],
                "self_alloc": [],
            },
        )
        bucket["ratio"].append(float(row["pool_grown_over_upperbound"]))
        bucket["lambda"].append(float(row["lambda_labor_equal"]))
        bucket["gap"].append(float(row["labor_equal_gap_l1"]))
        bucket["identifiable"].append(float(row["labor_equal_identifiable"]))
        bucket["labor"].append(float(row["weight_labor"]))
        bucket["equal"].append(float(row["weight_equal"]))
        bucket["self"].append(float(row["weight_self"]))
        bucket["self_alloc"].append(float(row["self_allocation"]))

    rows: list[dict[str, Any]] = []
    for step in sorted(per_step):
        bucket = per_step[step]
        rows.append(
            {
                "step": int(step),
                "count_rows": int(len(bucket["ratio"])),
                "pool_grown_over_upperbound_mean": _nanmean_or_nan(bucket["ratio"]),
                "labor_equal_gap_l1_mean": _nanmean_or_nan(bucket["gap"]),
                "labor_equal_identifiable_frac": _nanmean_or_nan(bucket["identifiable"]),
                "lambda_labor_equal_mean": _nanmean_or_nan(bucket["lambda"]),
                "weight_labor_mean": _nanmean_or_nan(bucket["labor"]),
                "weight_equal_mean": _nanmean_or_nan(bucket["equal"]),
                "weight_self_mean": _nanmean_or_nan(bucket["self"]),
                "self_allocation_mean": _nanmean_or_nan(bucket["self_alloc"]),
            }
        )
    return rows


def summarize_sender_state_mechanism(row_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not row_records:
        return []

    groups = {
        "cooperator": [],
        "defector": [],
    }
    for row in row_records:
        key = "cooperator" if float(row["sender_x_actual"]) > 0.5 else "defector"
        groups[key].append(row)

    rows: list[dict[str, Any]] = []
    for state_name, group in groups.items():
        if not group:
            continue
        rows.append(
            {
                "sender_state": state_name,
                "count_rows": int(len(group)),
                "pool_grown_over_upperbound_mean": float(
                    np.mean([float(row["pool_grown_over_upperbound"]) for row in group])
                ),
                "labor_equal_gap_l1_mean": _nanmean_or_nan([float(row["labor_equal_gap_l1"]) for row in group]),
                "labor_equal_identifiable_frac": _nanmean_or_nan(
                    [float(row["labor_equal_identifiable"]) for row in group]
                ),
                "weight_labor_mean": _nanmean_or_nan([float(row["weight_labor"]) for row in group]),
                "weight_equal_mean": _nanmean_or_nan([float(row["weight_equal"]) for row in group]),
                "weight_self_mean": _nanmean_or_nan([float(row["weight_self"]) for row in group]),
                "self_allocation_mean": _nanmean_or_nan([float(row["self_allocation"]) for row in group]),
                "lambda_labor_equal_mean": _nanmean_or_nan([float(row["lambda_labor_equal"]) for row in group]),
            }
        )
    return rows


def summarize_receiver_state_income(node_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not node_records:
        return []

    groups = {
        "cooperator": [],
        "defector": [],
    }
    for row in node_records:
        key = "cooperator" if float(row["receiver_x_actual"]) > 0.5 else "defector"
        groups[key].append(row)

    rows: list[dict[str, Any]] = []
    for state_name, group in groups.items():
        if not group:
            continue
        rows.append(
            {
                "receiver_state": state_name,
                "count_rows": int(len(group)),
                "actual_incoming_mean": float(np.mean([float(row["actual_incoming"]) for row in group])),
                "expected_incoming_equal_mean": float(
                    np.mean([float(row["expected_incoming_equal"]) for row in group])
                ),
                "expected_incoming_labor_mean": float(
                    np.mean([float(row["expected_incoming_labor"]) for row in group])
                ),
                "expected_incoming_self_mean": float(
                    np.mean([float(row["expected_incoming_self"]) for row in group])
                ),
            }
        )
    return rows


def compute_counterfactual_contribution_response(
    actor: Any,
    snapshots: Sequence[Snapshot],
    row_records: Sequence[Mapping[str, Any]],
    *,
    env_config: Any,
    row_sample_size: int,
    contribution_delta: float,
    batch_size: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    if not snapshots or not row_records:
        return []

    snapshot_lookup = {(snapshot.episode, snapshot.step): snapshot for snapshot in snapshots}
    eligible_indices = [
        index
        for index, row in enumerate(row_records)
        if (int(row["episode"]), int(row["step"])) in snapshot_lookup
    ]
    if not eligible_indices:
        return []

    rng = np.random.default_rng(int(rng_seed))
    sample_count = min(len(eligible_indices), max(1, int(row_sample_size)))
    sampled_indices = rng.choice(np.asarray(eligible_indices, dtype=np.int64), size=sample_count, replace=False)

    pending_metadata: list[dict[str, Any]] = []
    perturbed_observations: list[dict[str, np.ndarray]] = []
    for row_index in sorted(int(index) for index in sampled_indices.tolist()):
        row = row_records[row_index]
        episode = int(row["episode"])
        step = int(row["step"])
        sender = int(row["sender"])
        snapshot = snapshot_lookup[(episode, step)]
        local_mask = snapshot.observation["local_mask"].astype(bool, copy=False)
        valid_receivers = np.flatnonzero(local_mask[sender])
        baseline_row = np.asarray(snapshot.policy.allocation_matrix[sender, valid_receivers], dtype=np.float64)
        baseline_top1_receiver = int(valid_receivers[np.argmax(baseline_row)])

        for target_receiver in valid_receivers:
            perturbed_observation, perturbation_metadata = _build_counterfactual_contribution_observation(
                snapshot.observation,
                env_config=env_config,
                target_node=int(target_receiver),
                contribution_delta=float(contribution_delta),
            )
            pending_metadata.append(
                {
                    "episode": episode,
                    "step": step,
                    "sender": sender,
                    "target_receiver": int(target_receiver),
                    "target_is_self": int(int(target_receiver) == sender),
                    "receiver_count": int(valid_receivers.size),
                    "pool_grown_over_upperbound": float(row["pool_grown_over_upperbound"]),
                    "labor_equal_gap_l1": float(row["labor_equal_gap_l1"]),
                    "labor_equal_identifiable": int(row["labor_equal_identifiable"]),
                    "sender_x_actual": float(row["sender_x_actual"]),
                    "baseline_target_allocation": float(snapshot.policy.allocation_matrix[sender, int(target_receiver)]),
                    "baseline_sender_self_allocation": float(snapshot.policy.allocation_matrix[sender, sender]),
                    "baseline_row_entropy": _entropy(baseline_row),
                    "baseline_top1_receiver": baseline_top1_receiver,
                    "_valid_receivers": valid_receivers,
                    "_baseline_row": baseline_row,
                    **perturbation_metadata,
                }
            )
            perturbed_observations.append(perturbed_observation)

    if not perturbed_observations:
        return []

    perturbed_policies = _chunked_policy_forward(actor, perturbed_observations, batch_size=max(1, int(batch_size)))
    response_rows: list[dict[str, Any]] = []
    for metadata, perturbed_policy in zip(pending_metadata, perturbed_policies):
        valid_receivers = np.asarray(metadata["_valid_receivers"], dtype=np.int64)
        baseline_row = np.asarray(metadata["_baseline_row"], dtype=np.float64)
        sender = int(metadata["sender"])
        target_receiver = int(metadata["target_receiver"])

        counterfactual_row = np.asarray(perturbed_policy.allocation_matrix[sender, valid_receivers], dtype=np.float64)
        counterfactual_target_allocation = float(perturbed_policy.allocation_matrix[sender, target_receiver])
        counterfactual_sender_self_allocation = float(perturbed_policy.allocation_matrix[sender, sender])
        counterfactual_top1_receiver = int(valid_receivers[np.argmax(counterfactual_row)])

        response_rows.append(
            {
                "episode": int(metadata["episode"]),
                "step": int(metadata["step"]),
                "sender": sender,
                "target_receiver": target_receiver,
                "target_is_self": int(metadata["target_is_self"]),
                "receiver_count": int(metadata["receiver_count"]),
                "pool_grown_over_upperbound": float(metadata["pool_grown_over_upperbound"]),
                "labor_equal_gap_l1": float(metadata["labor_equal_gap_l1"]),
                "labor_equal_identifiable": int(metadata["labor_equal_identifiable"]),
                "sender_x_actual": float(metadata["sender_x_actual"]),
                "baseline_target_allocation": float(metadata["baseline_target_allocation"]),
                "counterfactual_target_allocation": counterfactual_target_allocation,
                "target_allocation_delta": counterfactual_target_allocation - float(metadata["baseline_target_allocation"]),
                "baseline_sender_self_allocation": float(metadata["baseline_sender_self_allocation"]),
                "counterfactual_sender_self_allocation": counterfactual_sender_self_allocation,
                "sender_self_allocation_delta": (
                    counterfactual_sender_self_allocation - float(metadata["baseline_sender_self_allocation"])
                ),
                "baseline_row_entropy": float(metadata["baseline_row_entropy"]),
                "counterfactual_row_entropy": _entropy(counterfactual_row),
                "row_entropy_delta": _entropy(counterfactual_row) - float(metadata["baseline_row_entropy"]),
                "row_js_to_baseline": _js_divergence(baseline_row, counterfactual_row),
                "baseline_top1_receiver": int(metadata["baseline_top1_receiver"]),
                "counterfactual_top1_receiver": counterfactual_top1_receiver,
                "top1_switched": int(counterfactual_top1_receiver != int(metadata["baseline_top1_receiver"])),
                "target_becomes_top1": int(counterfactual_top1_receiver == target_receiver),
                "target_resource": float(metadata["target_resource"]),
                "target_threshold": float(metadata["target_threshold"]),
                "baseline_target_investment": float(metadata["baseline_target_investment"]),
                "counterfactual_target_investment": float(metadata["counterfactual_target_investment"]),
                "target_investment_delta": float(metadata["target_investment_delta"]),
                "baseline_target_strategy_norm": float(metadata["baseline_target_strategy_norm"]),
                "counterfactual_target_strategy_norm": float(metadata["counterfactual_target_strategy_norm"]),
                "target_strategy_norm_delta": float(metadata["target_strategy_norm_delta"]),
                "baseline_target_x_actual": float(metadata["baseline_target_x_actual"]),
                "counterfactual_target_x_actual": float(metadata["counterfactual_target_x_actual"]),
                "target_x_actual_delta": float(metadata["target_x_actual_delta"]),
                "target_can_meet_threshold": int(metadata["target_can_meet_threshold"]),
                "effective_intervention": int(metadata["effective_intervention"]),
            }
        )
    return response_rows


def _summarize_counterfactual_group(
    group_type: str,
    group_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    deltas = [float(row["target_allocation_delta"]) for row in rows]
    before_values = [float(row["baseline_target_allocation"]) for row in rows]
    after_values = [float(row["counterfactual_target_allocation"]) for row in rows]
    self_deltas = [float(row["sender_self_allocation_delta"]) for row in rows]
    row_js = [float(row["row_js_to_baseline"]) for row in rows]
    entropy_delta = [float(row["row_entropy_delta"]) for row in rows]
    identifiable = [float(row["labor_equal_identifiable"]) for row in rows]
    effective = [float(row["effective_intervention"]) for row in rows]
    positive = [1.0 if float(row["target_allocation_delta"]) > 1e-8 else 0.0 for row in rows]

    return {
        "group_type": group_type,
        "group_name": group_name,
        "count_tests": int(len(rows)),
        "target_allocation_before_mean": _nanmean_or_nan(before_values),
        "target_allocation_after_mean": _nanmean_or_nan(after_values),
        "target_allocation_delta_mean": _nanmean_or_nan(deltas),
        "positive_response_frac": _nanmean_or_nan(positive),
        "effective_intervention_frac": _nanmean_or_nan(effective),
        "sender_self_allocation_delta_mean": _nanmean_or_nan(self_deltas),
        "row_js_to_baseline_mean": _nanmean_or_nan(row_js),
        "row_entropy_delta_mean": _nanmean_or_nan(entropy_delta),
        "labor_equal_identifiable_frac": _nanmean_or_nan(identifiable),
    }


def summarize_counterfactual_response_rows(
    response_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not response_rows:
        return [], {}

    summary_rows = [
        _summarize_counterfactual_group("overall", "all", response_rows),
    ]

    group_specs = [
        (
            "identifiability",
            [
                ("identifiable", [row for row in response_rows if int(row["labor_equal_identifiable"]) == 1]),
                ("indistinguishable", [row for row in response_rows if int(row["labor_equal_identifiable"]) == 0]),
            ],
        ),
        (
            "target_role",
            [
                ("self_target", [row for row in response_rows if int(row["target_is_self"]) == 1]),
                ("other_target", [row for row in response_rows if int(row["target_is_self"]) == 0]),
            ],
        ),
        (
            "target_state",
            [
                ("cooperator", [row for row in response_rows if float(row["baseline_target_x_actual"]) > 0.5]),
                ("defector", [row for row in response_rows if float(row["baseline_target_x_actual"]) <= 0.5]),
            ],
        ),
        (
            "intervention_effect",
            [
                ("effective", [row for row in response_rows if int(row["effective_intervention"]) == 1]),
                ("no_effect", [row for row in response_rows if int(row["effective_intervention"]) == 0]),
            ],
        ),
    ]
    for group_type, groups in group_specs:
        for group_name, group_rows in groups:
            if group_rows:
                summary_rows.append(_summarize_counterfactual_group(group_type, group_name, group_rows))

    summary_lookup = {(row["group_type"], row["group_name"]): row for row in summary_rows}
    overall_row = summary_lookup.get(("overall", "all"), {})
    identifiable_row = summary_lookup.get(("identifiability", "identifiable"), {})
    indistinguishable_row = summary_lookup.get(("identifiability", "indistinguishable"), {})

    flat_summary = {
        "counterfactual_test_count": int(len(response_rows)),
        "counterfactual_target_allocation_delta_mean": overall_row.get("target_allocation_delta_mean", float("nan")),
        "counterfactual_positive_response_frac": overall_row.get("positive_response_frac", float("nan")),
        "counterfactual_effective_intervention_frac": overall_row.get("effective_intervention_frac", float("nan")),
        "counterfactual_target_allocation_delta_identifiable_mean": identifiable_row.get(
            "target_allocation_delta_mean",
            float("nan"),
        ),
        "counterfactual_target_allocation_delta_indistinguishable_mean": indistinguishable_row.get(
            "target_allocation_delta_mean",
            float("nan"),
        ),
        "counterfactual_positive_response_identifiable_frac": identifiable_row.get(
            "positive_response_frac",
            float("nan"),
        ),
        "counterfactual_positive_response_indistinguishable_frac": indistinguishable_row.get(
            "positive_response_frac",
            float("nan"),
        ),
    }
    return summary_rows, flat_summary


def plot_position_mechanism_bins(bin_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not bin_rows:
        return

    x_values = np.asarray([float(row["pool_grown_over_upperbound_mean"]) for row in bin_rows], dtype=np.float64)
    labor_weight = np.asarray([float(row["weight_labor_mean"]) for row in bin_rows], dtype=np.float64)
    equal_weight = np.asarray([float(row["weight_equal_mean"]) for row in bin_rows], dtype=np.float64)
    self_weight = np.asarray([float(row["weight_self_mean"]) for row in bin_rows], dtype=np.float64)
    delta_values = np.asarray([float(row["delta_equal_minus_labor_mean"]) for row in bin_rows], dtype=np.float64)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].plot(x_values, labor_weight, marker="o", label="labor")
    axes[0].plot(x_values, equal_weight, marker="o", label="equal")
    axes[0].plot(x_values, self_weight, marker="o", label="self")
    axes[0].set_title("Mechanism Weights vs P_grown / P_upperbound")
    axes[0].set_xlabel("Mean P_grown / P_upperbound")
    axes[0].set_ylabel("Mean Weight")
    axes[0].legend()

    axes[1].plot(x_values, delta_values, marker="o", color="#d62728")
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_title("Delta(JS_equal - JS_labor) vs P_grown / P_upperbound")
    axes[1].set_xlabel("Mean P_grown / P_upperbound")
    axes[1].set_ylabel("Mean Delta")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_position_identifiability(bin_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not bin_rows:
        return

    x_values = np.asarray([float(row["pool_grown_over_upperbound_mean"]) for row in bin_rows], dtype=np.float64)
    gap_values = np.asarray([float(row["labor_equal_gap_l1_mean"]) for row in bin_rows], dtype=np.float64)
    identifiable_frac = np.asarray([float(row["labor_equal_identifiable_frac"]) for row in bin_rows], dtype=np.float64)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(x_values, gap_values, marker="o", color="#1f77b4")
    axes[0].set_title("Labor/Equal Gap vs P_grown / P_upperbound")
    axes[0].set_xlabel("Mean P_grown / P_upperbound")
    axes[0].set_ylabel("Mean L1 Gap")

    axes[1].plot(x_values, identifiable_frac, marker="o", color="#d62728")
    axes[1].set_title("Identifiable Fraction vs P_grown / P_upperbound")
    axes[1].set_xlabel("Mean P_grown / P_upperbound")
    axes[1].set_ylabel("Fraction with Gap >= Threshold")
    axes[1].set_ylim(0.0, 1.05)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_position_distribution(row_records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not row_records:
        return

    ratio = np.asarray([float(row["pool_grown_over_upperbound"]) for row in row_records], dtype=np.float64)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].hist(ratio, bins=40, color="#4c78a8", edgecolor="white")
    axes[0].set_title("Distribution of P_grown / P_upperbound")
    axes[0].set_xlabel("P_grown / P_upperbound")
    axes[0].set_ylabel("Row Count")

    axes[1].boxplot(ratio, vert=False)
    axes[1].set_title("Position Ratio Boxplot")
    axes[1].set_xlabel("P_grown / P_upperbound")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_counterfactual_summary(summary_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not summary_rows:
        return

    def _subset(group_type: str) -> list[Mapping[str, Any]]:
        return [row for row in summary_rows if str(row["group_type"]) == group_type]

    group_specs = [
        ("identifiability", "Response by Identifiability"),
        ("target_role", "Response by Target Role"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, (group_type, title) in zip(axes, group_specs):
        rows = _subset(group_type)
        if not rows:
            ax.set_axis_off()
            continue

        labels = [str(row["group_name"]) for row in rows]
        delta_values = [float(row["target_allocation_delta_mean"]) for row in rows]
        positive_frac = [float(row["positive_response_frac"]) for row in rows]
        x = np.arange(len(labels))

        bars = ax.bar(x, delta_values, color="#4c78a8", width=0.6, label="mean alloc delta")
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(title)
        ax.set_ylabel("Mean Target Allocation Delta")

        twin = ax.twinx()
        twin.plot(x, positive_frac, color="#d62728", marker="o", label="positive frac")
        twin.set_ylim(0.0, 1.05)
        twin.set_ylabel("Positive Response Fraction")

        handles = [bars, twin.lines[0]]
        labels = ["mean alloc delta", "positive frac"]
        ax.legend(handles, labels, loc="best")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_mechanism_over_time(step_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not step_rows:
        return

    steps = np.asarray([int(row["step"]) for row in step_rows], dtype=np.int64)
    ratio = np.asarray([float(row["pool_grown_over_upperbound_mean"]) for row in step_rows], dtype=np.float64)
    labor = np.asarray([float(row["weight_labor_mean"]) for row in step_rows], dtype=np.float64)
    equal = np.asarray([float(row["weight_equal_mean"]) for row in step_rows], dtype=np.float64)
    self_weight = np.asarray([float(row["weight_self_mean"]) for row in step_rows], dtype=np.float64)
    self_alloc = np.asarray([float(row["self_allocation_mean"]) for row in step_rows], dtype=np.float64)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)

    axes[0].plot(steps, labor, label="labor", color="#1f77b4")
    axes[0].plot(steps, equal, label="equal", color="#ff7f0e")
    axes[0].plot(steps, self_weight, label="self", color="#2ca02c")
    axes[0].set_title("Mechanism Weights Over Time")
    axes[0].set_ylabel("Mean Weight")
    axes[0].legend()

    axes[1].plot(steps, ratio, label="P/Pmax", color="#9467bd")
    axes[1].plot(steps, self_alloc, label="self allocation", color="#d62728")
    axes[1].set_title("Position Ratio and Self Allocation Over Time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean Value")
    axes[1].legend()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_sender_state_mechanism(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return

    labels = [str(row["sender_state"]) for row in rows]
    x = np.arange(len(labels))
    labor = [float(row["weight_labor_mean"]) for row in rows]
    equal = [float(row["weight_equal_mean"]) for row in rows]
    self_weight = [float(row["weight_self_mean"]) for row in rows]
    self_alloc = [float(row["self_allocation_mean"]) for row in rows]
    width = 0.2

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].bar(x - width, labor, width=width, label="labor")
    axes[0].bar(x, equal, width=width, label="equal")
    axes[0].bar(x + width, self_weight, width=width, label="self")
    axes[0].set_title("Mechanism Weights by Sender State")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean Weight")
    axes[0].legend()

    axes[1].bar(x, self_alloc, color="#d62728")
    axes[1].set_title("Self Allocation by Sender State")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean Self Allocation")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_receiver_state_income(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return

    labels = [str(row["receiver_state"]) for row in rows]
    x = np.arange(len(labels))
    actual = [float(row["actual_incoming_mean"]) for row in rows]
    equal = [float(row["expected_incoming_equal_mean"]) for row in rows]
    labor = [float(row["expected_incoming_labor_mean"]) for row in rows]
    self_income = [float(row["expected_incoming_self_mean"]) for row in rows]
    width = 0.2

    figure, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(x - 1.5 * width, actual, width=width, label="actual")
    ax.bar(x - 0.5 * width, equal, width=width, label="equal")
    ax.bar(x + 0.5 * width, labor, width=width, label="labor")
    ax.bar(x + 1.5 * width, self_income, width=width, label="self")
    ax.set_title("Incoming Resources by Receiver State")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Incoming Resource")
    ax.legend()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_mechanism_summary(
    *,
    run_dir: Path,
    checkpoint_name: str,
    checkpoint_payload: Mapping[str, Any],
    topology_name: str,
    network_config: Mapping[str, Any],
    effective_episode_length: int,
    requested_episode_length_override: int | None,
    graph_stats: Mapping[str, Any],
    policy_behavior: Mapping[str, Any],
    mechanism_summary: Mapping[str, Any],
    node_income_summary: Mapping[str, Any],
    labor_equal_gap_threshold: float,
    counterfactual_summary_rows: Sequence[Mapping[str, Any]],
    counterfactual_flat_summary: Mapping[str, Any],
    counterfactual_config: Mapping[str, Any],
    episode_summaries: Sequence[EpisodeSummary],
) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "checkpoint_name": checkpoint_name,
        "completed_updates": int(checkpoint_payload.get("completed_updates", checkpoint_payload.get("update", 0))),
        "global_env_steps": int(checkpoint_payload.get("global_env_steps", 0)),
        "matplotlib_available": plt is not None,
        "topology_name": topology_name,
        "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
        "network_config": dict(network_config),
        "effective_episode_length": int(effective_episode_length),
        "requested_episode_length_override": (
            None if requested_episode_length_override is None else int(requested_episode_length_override)
        ),
        "position_metric": {
            "name": "pool_grown_over_upperbound",
            "formula": "pool_grown / (sender_degree * p_c)",
            "sender_degree_definition": "number of neighbors excluding self",
        },
        "labor_equal_identifiability": {
            "metric": "L1 distance between labor baseline and equal baseline within each sender row",
            "threshold": float(labor_equal_gap_threshold),
        },
        "graph_stats": dict(graph_stats),
        "policy_behavior": dict(policy_behavior),
        "mechanism_summary": dict(mechanism_summary),
        "node_income_summary": dict(node_income_summary),
        "counterfactual_analysis": {
            "config": dict(counterfactual_config),
            "summary": dict(counterfactual_flat_summary),
            "summary_rows": [dict(row) for row in counterfactual_summary_rows],
        },
        "episode_summaries": [
            {
                "episode": summary.episode,
                "steps": summary.steps,
                "total_reward": summary.total_reward,
                "mean_reward": summary.mean_reward,
                "final_cooperation_rate": summary.final_cooperation_rate,
                "final_gini": summary.final_gini,
                "final_mean_resource": summary.final_mean_resource,
            }
            for summary in episode_summaries
        ],
    }


def summarize_topology_case(
    *,
    topology_name: str,
    graph_stats: Mapping[str, Any],
    policy_behavior: Mapping[str, Any],
    mechanism_summary: Mapping[str, Any],
    node_income_summary: Mapping[str, Any],
    counterfactual_flat_summary: Mapping[str, Any],
    episode_summaries: Sequence[EpisodeSummary],
) -> dict[str, Any]:
    total_rewards = [summary.total_reward for summary in episode_summaries]
    mean_rewards = [summary.mean_reward for summary in episode_summaries]
    final_coops = [summary.final_cooperation_rate for summary in episode_summaries]
    final_ginis = [summary.final_gini for summary in episode_summaries]
    final_resources = [summary.final_mean_resource for summary in episode_summaries]

    row = {
        "topology": topology_name,
        "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
        "episodes": int(len(episode_summaries)),
        "return_mean": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "reward_per_step_mean": float(np.mean(mean_rewards)) if mean_rewards else 0.0,
        "final_cooperation_mean": float(np.mean(final_coops)) if final_coops else 0.0,
        "final_gini_mean": float(np.mean(final_ginis)) if final_ginis else 0.0,
        "final_mean_resource_mean": float(np.mean(final_resources)) if final_resources else 0.0,
    }
    row.update({key: value for key, value in graph_stats.items()})
    row.update({key: value for key, value in policy_behavior.items()})
    row.update({key: value for key, value in mechanism_summary.items()})
    row.update({key: value for key, value in node_income_summary.items()})
    row.update({key: value for key, value in counterfactual_flat_summary.items()})
    return row


def analyze_single_topology(
    *,
    actor: Any,
    checkpoint_payload: Mapping[str, Any],
    run_dir: Path,
    output_dir: Path,
    spec: Mapping[str, Any],
    topology_name: str,
    checkpoint_name: str,
    episodes: int,
    max_steps: int | None,
    episode_length_override: int | None,
    mechanism_bin_count: int,
    labor_equal_gap_threshold: float,
    enable_counterfactual_analysis: bool,
    counterfactual_row_sample_size: int,
    counterfactual_contribution_delta: float,
    counterfactual_batch_size: int,
    rollout_seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_spec = deepcopy(dict(spec))
    if episode_length_override is not None:
        effective_spec["dynamics"] = dict(effective_spec["dynamics"])
        effective_spec["dynamics"]["episode_length"] = int(episode_length_override)

    graph = build_graph_from_spec(effective_spec)
    env_config = build_env_config_from_spec(effective_spec, graph)
    env = SPGGEnv(env_config, graph)

    snapshots, episode_summaries = run_deterministic_rollouts(
        actor,
        env,
        episodes=episodes,
        seed=rollout_seed,
        max_steps=max_steps,
    )
    if not snapshots:
        raise RuntimeError(f"No snapshots collected for topology {topology_name}.")

    graph_stats = collect_graph_stats(graph)
    policy_behavior = compute_policy_behavior_stats(snapshots)
    row_mechanism_records = compute_row_mechanism_records(
        snapshots,
        p_c=float(env_config.p_c),
        labor_equal_gap_threshold=float(labor_equal_gap_threshold),
    )
    node_income_records = compute_node_income_decomposition(snapshots)
    mechanism_summary, position_bin_rows = summarize_row_mechanism_records(
        row_mechanism_records,
        bin_count=mechanism_bin_count,
    )
    position_fixed_bin_rows = summarize_position_fixed_bins(
        row_mechanism_records,
        bin_count=mechanism_bin_count,
    )
    step_mechanism_rows = summarize_step_mechanism_records(row_mechanism_records)
    sender_state_rows = summarize_sender_state_mechanism(row_mechanism_records)
    node_income_summary = summarize_node_income_records(node_income_records)
    receiver_state_rows = summarize_receiver_state_income(node_income_records)
    counterfactual_response_rows: list[dict[str, Any]] = []
    counterfactual_summary_rows: list[dict[str, Any]] = []
    counterfactual_flat_summary: dict[str, Any] = {}
    if enable_counterfactual_analysis:
        counterfactual_response_rows = compute_counterfactual_contribution_response(
            actor,
            snapshots,
            row_mechanism_records,
            env_config=env_config,
            row_sample_size=max(1, int(counterfactual_row_sample_size)),
            contribution_delta=float(counterfactual_contribution_delta),
            batch_size=max(1, int(counterfactual_batch_size)),
            rng_seed=int(rollout_seed) + 17,
        )
        counterfactual_summary_rows, counterfactual_flat_summary = summarize_counterfactual_response_rows(
            counterfactual_response_rows
        )

    plot_position_mechanism_bins(position_bin_rows, output_dir / "position_vs_mechanism.png")
    plot_position_mechanism_bins(position_fixed_bin_rows, output_dir / "position_fixed_bins.png")
    plot_position_identifiability(position_fixed_bin_rows, output_dir / "position_identifiability.png")
    plot_position_distribution(row_mechanism_records, output_dir / "position_distribution.png")
    plot_mechanism_over_time(step_mechanism_rows, output_dir / "mechanism_over_time.png")
    plot_sender_state_mechanism(sender_state_rows, output_dir / "sender_state_mechanism.png")
    plot_receiver_state_income(receiver_state_rows, output_dir / "receiver_state_income.png")
    plot_counterfactual_summary(counterfactual_summary_rows, output_dir / "counterfactual_contribution_summary.png")
    write_episode_summary_csv(output_dir / "episode_summary.csv", episode_summaries)
    write_csv(output_dir / "row_mechanism.csv", row_mechanism_records)
    write_csv(output_dir / "position_bins.csv", position_bin_rows)
    write_csv(output_dir / "position_fixed_bins.csv", position_fixed_bin_rows)
    write_csv(output_dir / "step_mechanism.csv", step_mechanism_rows)
    write_csv(output_dir / "sender_state_mechanism.csv", sender_state_rows)
    write_csv(output_dir / "node_income_decomposition.csv", node_income_records)
    write_csv(output_dir / "receiver_state_income.csv", receiver_state_rows)
    write_csv(output_dir / "counterfactual_contribution_response.csv", counterfactual_response_rows)
    write_csv(output_dir / "counterfactual_contribution_summary.csv", counterfactual_summary_rows)

    summary = build_mechanism_summary(
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_payload=checkpoint_payload,
        topology_name=topology_name,
        network_config=effective_spec["network"],
        effective_episode_length=int(effective_spec["dynamics"]["episode_length"]),
        requested_episode_length_override=episode_length_override,
        graph_stats=graph_stats,
        policy_behavior=policy_behavior,
        mechanism_summary=mechanism_summary,
        node_income_summary=node_income_summary,
        labor_equal_gap_threshold=float(labor_equal_gap_threshold),
        counterfactual_summary_rows=counterfactual_summary_rows,
        counterfactual_flat_summary=counterfactual_flat_summary,
        counterfactual_config={
            "enabled": bool(enable_counterfactual_analysis),
            "row_sample_size": int(counterfactual_row_sample_size),
            "contribution_delta_on_strategy_norm": float(counterfactual_contribution_delta),
            "batch_size": int(counterfactual_batch_size),
        },
        episode_summaries=episode_summaries,
    )
    with (output_dir / "mechanism_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return {
        "topology": topology_name,
        "topology_row": summarize_topology_case(
            topology_name=topology_name,
            graph_stats=graph_stats,
            policy_behavior=policy_behavior,
            mechanism_summary=mechanism_summary,
            node_income_summary=node_income_summary,
            counterfactual_flat_summary=counterfactual_flat_summary,
            episode_summaries=episode_summaries,
        ),
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused mechanism analysis for a trained GNN allocator.")
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
        help="Comma-separated topology list. Examples: original or regular,er,ws,ba",
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
        "--mechanism-bin-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["mechanism_bin_count"]),
        help="Quantile bin count for P_grown / P_upperbound summaries.",
    )
    parser.add_argument(
        "--labor-equal-gap-threshold",
        type=float,
        default=float(SCRIPT_DEFAULTS["labor_equal_gap_threshold"]),
        help="Rows with L1(labor, equal) below this threshold are marked as indistinguishable.",
    )
    counterfactual_group = parser.add_mutually_exclusive_group()
    counterfactual_group.add_argument(
        "--enable-counterfactual-analysis",
        dest="enable_counterfactual_analysis",
        action="store_true",
        help="Run counterfactual contribution-response analysis.",
    )
    counterfactual_group.add_argument(
        "--disable-counterfactual-analysis",
        dest="enable_counterfactual_analysis",
        action="store_false",
        help="Skip counterfactual contribution-response analysis.",
    )
    parser.set_defaults(enable_counterfactual_analysis=bool(SCRIPT_DEFAULTS["enable_counterfactual_analysis"]))
    parser.add_argument(
        "--counterfactual-row-sample-size",
        type=int,
        default=int(SCRIPT_DEFAULTS["counterfactual_row_sample_size"]),
        help="Maximum number of sender rows sampled for counterfactual tests.",
    )
    parser.add_argument(
        "--counterfactual-contribution-delta",
        type=float,
        default=float(SCRIPT_DEFAULTS["counterfactual_contribution_delta"]),
        help="How much to increase target strategy_norm in the counterfactual test.",
    )
    parser.add_argument(
        "--counterfactual-batch-size",
        type=int,
        default=int(SCRIPT_DEFAULTS["counterfactual_batch_size"]),
        help="Batch size for counterfactual policy inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(SCRIPT_DEFAULTS["seed"]),
        help="Base environment reset seed for analysis rollouts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(SCRIPT_DEFAULTS["device"]),
        help="Torch device, e.g. cpu or cuda:0.",
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
            else (run_dir / "policy_mechanism" / Path(args.checkpoint_name).stem).resolve()
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    topologies = parse_topology_list(args.topologies)

    import torch

    device = torch.device(args.device)
    actor, checkpoint_payload = load_actor_from_run_dir(run_dir, args.checkpoint_name, device)
    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)

    topology_rows: list[dict[str, Any]] = []
    for topology_index, topology_name in enumerate(topologies):
        topology_spec = build_spec_for_topology(
            experiment_spec,
            topology_name,
            reference_mean_degree=reference_mean_degree,
        )
        topology_output_dir = output_dir if len(topologies) == 1 else (output_dir / topology_name)
        result = analyze_single_topology(
            actor=actor,
            checkpoint_payload=checkpoint_payload,
            run_dir=run_dir,
            output_dir=topology_output_dir,
            spec=topology_spec,
            topology_name=topology_name,
            checkpoint_name=args.checkpoint_name,
            episodes=max(1, int(args.episodes)),
            max_steps=args.max_steps,
            episode_length_override=args.episode_length_override,
            mechanism_bin_count=max(2, int(args.mechanism_bin_count)),
            labor_equal_gap_threshold=max(0.0, float(args.labor_equal_gap_threshold)),
            enable_counterfactual_analysis=bool(args.enable_counterfactual_analysis),
            counterfactual_row_sample_size=max(1, int(args.counterfactual_row_sample_size)),
            counterfactual_contribution_delta=max(0.0, float(args.counterfactual_contribution_delta)),
            counterfactual_batch_size=max(1, int(args.counterfactual_batch_size)),
            rollout_seed=int(args.seed) + topology_index,
        )
        topology_rows.append(result["topology_row"])

    write_csv(output_dir / "topology_summary.csv", topology_rows)
    with (output_dir / "topology_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(topology_rows, handle, ensure_ascii=False, indent=2)

    print(f"Mechanism analysis complete. Artifacts written to: {output_dir}")
    if plt is None:
        print("Plot export skipped because matplotlib is not installed in the current interpreter.")
    print("Per-topology summary:")
    for row in topology_rows:
        parts = [
            f"{row['topology_label']:<8}",
            f"return={float(row['return_mean']):.6f}",
            f"coop={float(row['final_cooperation_mean']):.4f}",
            f"gini={float(row['final_gini_mean']):.4f}",
            f"self={float(row['mean_self_allocation']):.4f}",
            f"lambda={float(row['lambda_labor_equal_mean']):.4f}",
            f"id_frac={float(row['labor_equal_identifiable_frac']):.4f}",
            f"eq_w={float(row['weight_equal_mean']):.4f}",
            f"labor_w={float(row['weight_labor_mean']):.4f}",
            f"cf_delta={float(row.get('counterfactual_target_allocation_delta_mean', float('nan'))):.4f}",
            f"P/Pmax->lambda={float(row['pool_grown_over_upperbound_vs_lambda_spearman']):.4f}",
        ]
        print("  " + " ".join(parts))


if __name__ == "__main__":
    main()
