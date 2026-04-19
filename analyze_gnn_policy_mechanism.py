from __future__ import annotations

import argparse
import csv
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

from Project1.env import SPGGEnv, gini_coefficient
from Project1.policies.rule_based import ProportionalContributionPolicy, UniformAllocationPolicy

from analyze_gnn_policy_decisions import (
    TOPOLOGY_LABELS,
    EpisodeSummary,
    PolicyArrays,
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
    # 如果你更习惯“直接改脚本里的默认参数”，优先改这一整段即可。
    # parse_args() 里的命令行参数默认值，都会从这里读取。

    # 训练结果目录。
    # 目录下至少要有：
    # 1. results.json：用于恢复训练时的 experiment 配置；
    # 2. checkpoints/：用于加载训练好的策略网络权重。
    # 如果你在 Windows/PyCharm 里直接跑，可以保留 Windows 路径写法；
    # 如果你在 bash / WSL 里跑，命令行里最好显式传 --run-dir，避免路径解析问题。
    "run_dir": r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    #"run_dir": r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic\0416_demo_regularized_graph_td3_regular_ba_guard\BC_floor\BC_floor\Q_cap",

    # 要加载的 checkpoint 文件名。
    # 常用值：
    # - best_eval.pt：一般优先看它，表示评估集上表现最好的模型；
    # - final.pt：训练结束时最后一次保存的模型；
    # - latest.pt：最近一次保存的模型；
    # - update_00xxxx.pt：中间阶段模型，适合看策略随训练进度怎么变。
    "checkpoint_name": "best_eval.pt",

    # 机制分析结果输出目录。
    # 设为 None 时，会继续看 output_subdir_name。
    # 两者都为 None 时，默认输出到：
    # <run_dir>/policy_mechanism/<checkpoint_name去掉后缀>/
    # 如果你想把多次分析结果严格分开，也可以直接在这里写一个完整路径。
    "output_dir": None,

    # 当 output_dir 为 None 时，把结果输出到：
    # <run_dir>/<output_subdir_name>/
    # 设为 None 时，回退到默认目录：
    # <run_dir>/policy_mechanism/<checkpoint_name去掉后缀>/
    # 建议你把这个名字改成能反映配置的形式，比如：
    # 0416_pool_intervention_original_10ep
   # "output_subdir_name": "0418_regu_ba_resourcepool_intervention_policy_mechanism",
    "output_subdir_name": "0418_total_regu_ba_policy_mechanism",

    # 要分析的图拓扑，多个值用英文逗号分隔。
    # 支持：original, regular, er, ws, ba。
    # - original：完全按这次训练原本 results.json 里的拓扑来建图；
    # - regular / er / ws / ba：把同一个训练好的策略迁移到其他图上测。
    # 这个脚本默认给 original，避免一上来就做跨拓扑批量跑。
    "topologies": "regular,ba",

    # 每种拓扑下跑多少个 deterministic episode。
    # 越大越稳，但 CSV/JSON 会更大、运行也更慢。
    # 经验上：
    # - 1：快速 smoke test；
    # - 3~10：初步分析；
    # - 10+：更稳定的统计。
    "episodes": 10,

    # 每个 episode 最多分析多少步。
    # 设为 None 表示跑完整个 episode_length。
    # 注意它只是“分析截断上限”，不是环境本身的长度定义。
    # 真正一局最多能跑多长，还要看下面的 episode_length_override。
    "max_steps": 200,

    # 是否覆盖分析环境里的 episode_length。
    # 如果设为 None，就沿用 results.json 里的原始 episode_length。
    # 如果你想看“200 步训练策略在 500 步长期演化下的机制”，这里就设成 500。
    # 只有当这个值 >= max_steps 时，max_steps 才真的可能跑到那么长。
    "episode_length_override": 200,

    # 把 P_grown / P_upperbound 按多少个分位数区间做分箱统计。
    # 这影响的是“自然状态机制分析”的各种分箱表和图，
    # 不影响 pool intervention 图的横轴倍率。
    # 更小更平滑，更大更细。
    "mechanism_bin_count": 10,

    # labor 基线和 equal 基线的 L1 距离阈值。
    # 小于这个阈值时，说明这两种基线在当前行上几乎重合，
    # 不应该再强行解读成“更偏 labor 还是更偏 equal”。
    # 因而这些行会被标为 labor_equal_identifiable = 0。
    "labor_equal_gap_threshold": 0.05,

    # 是否执行“提高某个接收节点贡献信号后，发送节点是否给它更多分配”的反事实测试。
    # True 会额外输出 counterfactual_contribution_response.csv / summary.csv / png。
    # 如果你当前只关心 pool intervention 图，把它关掉可以省不少时间。
    "enable_counterfactual_analysis": False,

    # 反事实测试最多抽多少个 sender-row 作为样本。
    # 每个样本行会对该 sender 的所有可接收节点分别做一次“贡献抬高”测试。
    # 这个值越大，统计越稳，但离线前向推理量也会明显上升。
    "counterfactual_row_sample_size": 512,

    # 反事实里把目标节点的 strategy_norm 最多上调多少。
    # 这里是“相对当前值加多少”，然后再截断到 [0, 1]。
    # 例如 0.20 表示 target 的 strategy_norm 最多上调 0.2。
    "counterfactual_contribution_delta": 0.20,

    # 反事实离线前向推理的 batch size。
    # 更大更快，但更吃显存/内存。
    "counterfactual_batch_size": 128,

    # 用 rule-based baseline 在同一环境里独立 rollout 多少个 episode。
    # 这会影响三类输出：
    # 1. observed-position 参考图里的 proportional/equal 曲线；
    # 2. policy_rollout_outcomes_over_time.png / policy_rollout_behavior_over_time.png
    #    里 proportional/equal 的时间序列；
    # 3. proportional 这条 manipulated-pool intervention 曲线所基于的 rollout 状态。
    # 如果你想让 RL Agent / Proportional / Equal 的时间序列统计口径尽量一致，
    # 一般建议把它设成和 episodes 相同。
    "baseline_gini_episodes": 10,

    # 是否执行“手动操纵 sender 看到的当前 pool 大小，然后重新前向推理”的干预实验。
    # True 时会把 allocation_gini_vs_position.png 改成 intervention 图，
    # 横轴是被操纵后的 pool 相对 reference pool 的倍数。
    # 这是现在最接近你贴的 Fig S3 的那部分分析。
    "enable_pool_intervention_analysis": True,

    # intervention 图里使用哪个 reference pool：
    # - original：以该 sender 在当前状态下原本观察到的 pool_grown 为基准；
    # - upperbound：以 (degree+1) * p_c 为基准。
    # 如果你想尽量复刻 Fig S3，通常优先用 original。
    # 如果你想研究“相对理论上限 P_upperbound”的响应，再改成 upperbound。
    "pool_intervention_reference_mode": "upperbound",

    # intervention 图的横轴倍率列表，多个值用英文逗号分隔。
    # 例如 0.1 表示把当前 pool 信号压到 reference pool 的 10%，
    # 6.0 表示放大到 6 倍。
    # 这组默认值就是按你给的 Fig S3 风格设置的。
    # 注意这些值必须都是正数，因为图会用对数坐标。
    "pool_intervention_multipliers": "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0",

    # intervention 最多抽多少个 sender-row 做实验。
    # 设为 0 或负数时，表示对所有 sender-row 都做 intervention。
    # 这个参数主要是为了控制离线前向推理量。
    # 如果你要做最终版图，建议设成 0；
    # 如果你只想先快速预览趋势，可以设成 256 / 512 / 1024。
    "pool_intervention_row_sample_size": 2048,

    # intervention 离线前向推理的 batch size。
    # 更大更快，但也更吃显存/内存。
    # CPU 下一般 64~256 都比较稳；如果以后切 GPU 可以再适当调大。
    "pool_intervention_batch_size": 128,

    # RL-Agent 和 Equal Baseline 这两条 intervention 曲线，抽样 row 来自哪些 step。
    # 这里的 step 是“episode 内的时间步 t”，不是训练 update。
    # - None 表示不限制下界；
    # - 设成整数，例如 50，表示只从 t >= 50 的 row 里抽样。
    # 说明：当前脚本里的 Equal Baseline 仍然是基于 agent rollout 状态做参考，
    # 所以它和 RL-Agent 共用这一组 step 范围参数。
    "agent_equal_pool_intervention_step_start": 1,

    # RL-Agent 和 Equal Baseline intervention 抽样的 step 上界。
    # - None 表示不限制上界；
    # - 设成整数，例如 199，表示只从 t <= 199 的 row 里抽样。
    # 上下界都是闭区间。
    "agent_equal_pool_intervention_step_end": 50,

    # Proportional Baseline 这条 intervention 曲线，抽样 row 来自哪些 step。
    # 它使用的是“从 step 0 开始一直按 proportional rollout”产生的状态，
    # 因此单独给一组 step 范围参数。
    "proportional_pool_intervention_step_start": None,

    # Proportional Baseline intervention 抽样的 step 上界。
    # 同样是闭区间；None 表示不限制。
    "proportional_pool_intervention_step_end": None,

    # 是否执行“手动操纵 sender 看到的当前 resource 大小，然后重新前向推理”的干预实验。
    # 这和 pool intervention 不同：这里看的不是 pool 通道，而是资源通道。
    # 当前实现的语义是“篡改 agent 对 sender 资源量的感知”，不会把改过的 observation 继续 rollout 到后续步。
    "enable_resource_intervention_analysis": True,

    # resource intervention 使用哪个参考量：
    # - original：以该 sender 当前 observation 里的原始 resources[sender] 为基准；
    # - norm_reference：以环境的 resource_norm_reference 为基准。
    # 如果你想看“相对当前资源量”的敏感性，优先用 original。
    "resource_intervention_reference_mode": "original",

    # resource intervention 横轴倍率列表，多个值用英文逗号分隔。
    # 含义和 pool intervention 类似，只是这里操纵的是 sender 的 resource 信号。
    # 如果 reference_mode=original，1.0 表示保持该 sender 原本的资源不变。
    "resource_intervention_multipliers": "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0",

    # resource intervention 最多抽多少个 sender-row 做实验。
    # 设为 0 或负数时，表示对所有 sender-row 都做 intervention。
    "resource_intervention_row_sample_size": 2048,

    # resource intervention 里 RL-Agent 批量前向推理的 batch size。
    "resource_intervention_batch_size": 128,

    # RL-Agent 和 Equal Baseline 这两条 resource intervention 曲线，抽样 row 的 step 下界。
    # 当前 Equal 仍然使用 agent rollout 状态做参考，所以和 RL-Agent 共用这一组。
    "agent_equal_resource_intervention_step_start":1,

    # RL-Agent 和 Equal Baseline 这两条 resource intervention 曲线，抽样 row 的 step 上界。
    "agent_equal_resource_intervention_step_end": 50,

    # Proportional Baseline 这条 resource intervention 曲线，抽样 row 的 step 下界。
    "proportional_resource_intervention_step_start": 1,

    # Proportional Baseline 这条 resource intervention 曲线，抽样 row 的 step 上界。
    "proportional_resource_intervention_step_end": 50,

    # 分析时是否覆盖环境里的 r。
    # - 设为 None：沿用 results.json 里训练时记录的 r；
    # - 设为具体数值，例如 0.25 / 0.50：表示在分析阶段强制用这个 r 重新 rollout。
    # 这会同时影响 agent、rule-based baseline，以及后续所有基于 observation 的分析。
    "env_r_override": 0.5,

    # rollout 的环境随机种子基准值。
    # 第 0 个 episode 用 seed，第 1 个用 seed+1，以此类推。
    # 如果你想复现实验结果，固定这个值即可。
    "seed": 42,

    # Torch 推理设备。
    # 常见写法：
    # - "cpu"：最稳，所有环境基本都能跑；
    # - "cuda:0"：如果当前解释器能用 GPU，就会更快。
    # 如果你在没有 CUDA 的解释器里写成 cuda:0，会直接报错。
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


def _step_in_selected_range(step: int, step_start: int | None, step_end: int | None) -> bool:
    if step_start is not None and int(step) < int(step_start):
        return False
    if step_end is not None and int(step) > int(step_end):
        return False
    return True


def _parse_float_list(raw_value: str) -> list[float]:
    values: list[float] = []
    for part in str(raw_value).split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


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


def apply_analysis_env_overrides(
    spec: Mapping[str, Any],
    *,
    env_r_override: float | None,
) -> dict[str, Any]:
    effective_spec = deepcopy(dict(spec))
    if env_r_override is None:
        return effective_spec
    if "dynamics" not in effective_spec or not isinstance(effective_spec["dynamics"], Mapping):
        raise KeyError("Experiment spec does not contain a dynamics mapping; cannot override r.")
    effective_spec["dynamics"] = dict(effective_spec["dynamics"])
    effective_spec["dynamics"]["r"] = float(env_r_override)
    return effective_spec


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


def _build_pool_intervention_observation(
    observation: Mapping[str, np.ndarray],
    *,
    sender: int,
    manipulated_pool_value: float,
    sender_pool_upperbound: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    perturbed = _observation_copy(observation)
    original_pool = float(np.asarray(observation["pool_grown"], dtype=np.float64)[sender])
    original_pool_raw_norm = float(np.asarray(observation["pool_raw_norm"], dtype=np.float64)[sender])
    pool_scale = (
        float(manipulated_pool_value / original_pool) if abs(original_pool) > 1e-12 else float("nan")
    )

    pool_grown = np.asarray(perturbed["pool_grown"], dtype=np.float64)
    pool_grown[sender] = float(manipulated_pool_value)
    perturbed["pool_grown"] = pool_grown.astype(np.asarray(observation["pool_grown"]).dtype, copy=False)

    pool_raw_norm = np.asarray(perturbed["pool_raw_norm"], dtype=np.float64)
    if np.isfinite(pool_scale):
        pool_raw_norm[sender] = original_pool_raw_norm * pool_scale
    elif sender_pool_upperbound > 1e-12:
        pool_raw_norm[sender] = float(manipulated_pool_value / sender_pool_upperbound)
    else:
        pool_raw_norm[sender] = 0.0
    perturbed["pool_raw_norm"] = pool_raw_norm.astype(np.asarray(observation["pool_raw_norm"]).dtype, copy=False)

    if "pool_raw" in perturbed:
        pool_raw = np.asarray(perturbed["pool_raw"], dtype=np.float64)
        if np.isfinite(pool_scale):
            pool_raw[sender] = float(pool_raw[sender]) * pool_scale
        else:
            pool_raw[sender] = float(pool_raw_norm[sender] * sender_pool_upperbound)
        perturbed["pool_raw"] = pool_raw.astype(np.asarray(observation["pool_raw"]).dtype, copy=False)

    return (
        perturbed,
        {
            "original_pool_grown": original_pool,
            "original_pool_raw_norm": original_pool_raw_norm,
            "manipulated_pool_grown": float(manipulated_pool_value),
            "manipulated_pool_raw_norm": float(pool_raw_norm[sender]),
            "sender_pool_upperbound": float(sender_pool_upperbound),
            "pool_scale_over_original": pool_scale,
            "manipulated_pool_over_upperbound": (
                float(manipulated_pool_value / sender_pool_upperbound)
                if sender_pool_upperbound > 1e-12
                else float("nan")
            ),
        },
    )


def _build_resource_intervention_observation(
    observation: Mapping[str, np.ndarray],
    *,
    sender: int,
    manipulated_resource_value: float,
    resource_norm_reference: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    perturbed = _observation_copy(observation)
    original_resource = float(np.asarray(observation["resources"], dtype=np.float64)[sender])
    original_resource_norm = float(np.asarray(observation["resource_norm"], dtype=np.float64)[sender])
    original_strategy_norm = float(np.asarray(observation["strategy_norm"], dtype=np.float64)[sender])
    sender_investment = float(np.asarray(observation["investment"], dtype=np.float64)[sender])
    safe_resource_value = max(float(manipulated_resource_value), 0.0)
    resource_scale = (
        float(safe_resource_value / original_resource) if abs(original_resource) > 1e-12 else float("nan")
    )

    resources = np.asarray(perturbed["resources"], dtype=np.float64)
    resources[sender] = safe_resource_value
    perturbed["resources"] = resources.astype(np.asarray(observation["resources"]).dtype, copy=False)

    resource_norm = np.asarray(perturbed["resource_norm"], dtype=np.float64)
    if resource_norm_reference > 1e-12:
        resource_norm[sender] = float(safe_resource_value / resource_norm_reference)
    elif np.isfinite(resource_scale):
        resource_norm[sender] = original_resource_norm * resource_scale
    else:
        resource_norm[sender] = 0.0
    perturbed["resource_norm"] = resource_norm.astype(np.asarray(observation["resource_norm"]).dtype, copy=False)

    strategy_norm = np.asarray(perturbed["strategy_norm"], dtype=np.float64)
    if safe_resource_value > 1e-12:
        strategy_norm[sender] = float(np.clip(sender_investment / safe_resource_value, 0.0, 1.0))
    else:
        strategy_norm[sender] = 0.0
    perturbed["strategy_norm"] = strategy_norm.astype(np.asarray(observation["strategy_norm"]).dtype, copy=False)

    if "gini" in perturbed:
        perturbed["gini"] = np.asarray(
            gini_coefficient(resources, epsilon=1e-8),
            dtype=np.asarray(observation["gini"]).dtype,
        )

    return (
        perturbed,
        {
            "original_resource": original_resource,
            "original_resource_norm": original_resource_norm,
            "original_sender_strategy_norm": original_strategy_norm,
            "original_sender_investment": sender_investment,
            "manipulated_resource": safe_resource_value,
            "manipulated_resource_norm": float(resource_norm[sender]),
            "manipulated_sender_strategy_norm": float(strategy_norm[sender]),
            "resource_norm_reference": float(resource_norm_reference),
            "resource_scale_over_original": resource_scale,
            "manipulated_resource_over_reference": (
                float(safe_resource_value / resource_norm_reference) if resource_norm_reference > 1e-12 else float("nan")
            ),
        },
    )


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
            pool_upperbound = float(sender_degree + 1) * float(p_c)
            pool_position_ratio = pool_value / pool_upperbound if pool_upperbound > 1e-12 else float("nan")
            ego_resources = np.asarray(observation["resources"][valid_receivers], dtype=np.float64)
            ego_investment = np.asarray(observation["investment"][valid_receivers], dtype=np.float64)
            self_index = int(np.nonzero(valid_receivers == sender)[0][0])
            top_receiver = int(valid_receivers[np.argmax(allocation_row)])
            agent_allocated_resources = pool_value * allocation_row

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
                    "agent_allocated_resource_gini": _gini_nonnegative(agent_allocated_resources),
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
    agent_gini = np.asarray([float(row["agent_allocated_resource_gini"]) for row in row_records], dtype=np.float64)
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
                "agent_allocated_resource_gini_mean": _nanmean_or_nan(agent_gini[mask]),
            }
        )
    return bin_rows


def compute_rule_policy_allocation_gini_records(
    policy: Any,
    *,
    policy_name: str,
    env_config: Any,
    graph: Mapping[int, Sequence[int]],
    episodes: int,
    seed: int,
    max_steps: int | None,
) -> tuple[list[dict[str, Any]], list[EpisodeSummary]]:
    snapshots, episode_summaries = run_rule_policy_rollouts(
        policy,
        env_config=env_config,
        graph=graph,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )
    return extract_policy_allocation_gini_records_from_snapshots(policy_name, snapshots), episode_summaries


def run_rule_policy_rollouts(
    policy: Any,
    *,
    env_config: Any,
    graph: Mapping[int, Sequence[int]],
    episodes: int,
    seed: int,
    max_steps: int | None,
) -> tuple[list[Snapshot], list[EpisodeSummary]]:
    env = SPGGEnv(env_config, graph)
    snapshots: list[Snapshot] = []
    episode_summaries: list[EpisodeSummary] = []

    for episode_index in range(max(0, int(episodes))):
        observation = env.reset(seed=int(seed) + episode_index)
        done = False
        step_index = 0
        total_reward = 0.0
        last_info: dict[str, Any] | None = None

        while not done:
            if max_steps is not None and step_index >= max_steps:
                break

            allocation_matrix = np.asarray(policy.allocate(observation), dtype=np.float64)
            pool_values = np.asarray(observation["pool_grown"], dtype=np.float64)
            transferred_resources = allocation_matrix * pool_values[:, None]
            incoming_resources = np.sum(transferred_resources, axis=0)

            next_observation, reward, env_done, info = env.step(allocation_matrix)
            snapshots.append(
                Snapshot(
                    episode=episode_index,
                    step=step_index,
                    observation=_observation_copy(observation),
                    policy=PolicyArrays(
                        allocation_matrix=np.asarray(allocation_matrix, dtype=np.float64),
                        transferred_resources=np.asarray(transferred_resources, dtype=np.float64),
                        incoming_resources=np.asarray(incoming_resources, dtype=np.float64),
                        logits=np.zeros_like(allocation_matrix, dtype=np.float64),
                        value=0.0,
                    ),
                    reward=float(reward),
                    actual_cooperation_rate=float(info.get("actual_cooperation_rate", np.mean(next_observation["x_actual"]))),
                    gini=float(info.get("gini", next_observation["gini"])),
                )
            )
            total_reward += float(reward)
            last_info = info
            observation = next_observation
            step_index += 1
            done = bool(env_done)

        final_cooperation_rate = float(np.mean(observation["x_actual"]))
        final_gini = float(np.asarray(observation["gini"]).item())
        if last_info is not None:
            final_cooperation_rate = float(last_info.get("actual_cooperation_rate", final_cooperation_rate))
            final_gini = float(last_info.get("gini", final_gini))
        episode_summaries.append(
            EpisodeSummary(
                episode=episode_index,
                steps=step_index,
                total_reward=total_reward,
                mean_reward=total_reward / float(max(step_index, 1)),
                final_cooperation_rate=final_cooperation_rate,
                final_gini=final_gini,
                final_mean_resource=float(np.mean(observation["resources"])),
            )
        )

    return snapshots, episode_summaries


def extract_policy_allocation_gini_records_from_snapshots(
    policy_name: str,
    snapshots: Sequence[Snapshot],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for snapshot in snapshots:
        observation = snapshot.observation
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            allocation_row = np.asarray(snapshot.policy.allocation_matrix[sender, valid_receivers], dtype=np.float64)
            pool_value = float(observation["pool_grown"][sender])
            sender_degree = max(int(valid_receivers.size) - 1, 0)
            if "p_max" in snapshot.observation:
                sender_pool_upperbound = float(np.asarray(snapshot.observation["p_max"], dtype=np.float64)[sender])
            else:
                sender_pool_upperbound = float("nan")
            records.append(
                {
                    "policy": str(policy_name),
                    "episode": int(snapshot.episode),
                    "step": int(snapshot.step),
                    "sender": int(sender),
                    "sender_degree": int(sender_degree),
                    "receiver_count": int(valid_receivers.size),
                    "sender_pool_grown": pool_value,
                    "sender_pool_upperbound": sender_pool_upperbound,
                    "pool_grown_over_upperbound": (
                        pool_value / sender_pool_upperbound if sender_pool_upperbound > 1e-12 else float("nan")
                    ),
                    "allocated_resource_gini": _gini_nonnegative(pool_value * allocation_row),
                }
            )
    return records


def summarize_allocation_gini_rows(
    rows_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bin_count: int,
) -> list[dict[str, Any]]:
    all_ratios: list[float] = []
    for rows in rows_by_policy.values():
        for row in rows:
            ratio = float(row["pool_grown_over_upperbound"])
            if np.isfinite(ratio):
                all_ratios.append(ratio)
    if not all_ratios:
        return []

    ratio_array = np.asarray(all_ratios, dtype=np.float64)
    left_edge = float(np.min(ratio_array))
    right_edge = float(np.max(ratio_array))
    if right_edge <= left_edge:
        right_edge = left_edge + 1e-9
    edges = np.linspace(left_edge, right_edge, num=max(2, int(bin_count)) + 1)

    summary_rows: list[dict[str, Any]] = []
    for policy_name, policy_rows in rows_by_policy.items():
        ratios = np.asarray([float(row["pool_grown_over_upperbound"]) for row in policy_rows], dtype=np.float64)
        ginis = np.asarray([float(row["allocated_resource_gini"]) for row in policy_rows], dtype=np.float64)
        for bin_index in range(edges.size - 1):
            left = float(edges[bin_index])
            right = float(edges[bin_index + 1])
            if bin_index == edges.size - 2:
                mask = (ratios >= left) & (ratios <= right)
            else:
                mask = (ratios >= left) & (ratios < right)
            if not np.any(mask):
                continue
            summary_rows.append(
                {
                    "policy": str(policy_name),
                    "bin_index": int(bin_index),
                    "pool_grown_over_upperbound_left": left,
                    "pool_grown_over_upperbound_right": right,
                    "count_rows": int(mask.sum()),
                    "pool_grown_over_upperbound_mean": _nanmean_or_nan(ratios[mask]),
                    "allocated_resource_gini_mean": _nanmean_or_nan(ginis[mask]),
                    }
                )
    return summary_rows


def compute_pool_intervention_records(
    actor: Any,
    snapshots: Sequence[Snapshot],
    row_records: Sequence[Mapping[str, Any]],
    *,
    reference_mode: str,
    multiplier_values: Sequence[float],
    row_sample_size: int,
    batch_size: int,
    rng_seed: int,
    step_start: int | None,
    step_end: int | None,
) -> list[dict[str, Any]]:
    if not snapshots or not row_records or not multiplier_values:
        return []

    mode = str(reference_mode).strip().lower()
    if mode not in {"original", "upperbound"}:
        raise ValueError(f"Unsupported pool intervention reference mode: {reference_mode}")

    snapshot_lookup = {(snapshot.episode, snapshot.step): snapshot for snapshot in snapshots}
    eligible_indices = [
        index
        for index, row in enumerate(row_records)
        if (int(row["episode"]), int(row["step"])) in snapshot_lookup
        and _step_in_selected_range(
            int(row["step"]),
            None if step_start is None else int(step_start),
            None if step_end is None else int(step_end),
        )
    ]
    if not eligible_indices:
        return []

    rng = np.random.default_rng(int(rng_seed))
    if int(row_sample_size) <= 0:
        sample_count = len(eligible_indices)
    else:
        sample_count = min(len(eligible_indices), int(row_sample_size))
    sampled_indices = rng.choice(np.asarray(eligible_indices, dtype=np.int64), size=sample_count, replace=False)

    equal_policy = UniformAllocationPolicy()

    pending_metadata: list[dict[str, Any]] = []
    perturbed_observations: list[dict[str, np.ndarray]] = []
    for row_index in sorted(int(index) for index in sampled_indices.tolist()):
        row = row_records[row_index]
        episode = int(row["episode"])
        step = int(row["step"])
        sender = int(row["sender"])
        snapshot = snapshot_lookup[(episode, step)]
        observation = snapshot.observation
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        valid_receivers = np.flatnonzero(local_mask[sender])
        if valid_receivers.size == 0:
            continue

        equal_allocation = np.asarray(equal_policy.allocate(observation), dtype=np.float64)
        equal_row = np.asarray(equal_allocation[sender, valid_receivers], dtype=np.float64)

        original_pool = float(np.asarray(observation["pool_grown"], dtype=np.float64)[sender])
        sender_pool_upperbound = float(row["sender_pool_upperbound"])
        reference_pool = original_pool if mode == "original" else sender_pool_upperbound

        for multiplier in multiplier_values:
            multiplier_value = float(multiplier)
            manipulated_pool = multiplier_value * reference_pool
            perturbed_observation, intervention_metadata = _build_pool_intervention_observation(
                observation,
                sender=sender,
                manipulated_pool_value=manipulated_pool,
                sender_pool_upperbound=sender_pool_upperbound,
            )
            pending_metadata.append(
                {
                    "episode": episode,
                    "step": step,
                    "sender": sender,
                    "sender_degree": int(row["sender_degree"]),
                    "receiver_count": int(valid_receivers.size),
                    "reference_mode": mode,
                    "multiplier": multiplier_value,
                    "reference_pool_value": float(reference_pool),
                    "original_pool_grown": original_pool,
                    "manipulated_pool_over_original": (
                        float(manipulated_pool / original_pool) if abs(original_pool) > 1e-12 else float("nan")
                    ),
                    "equal_allocated_resource_gini": _gini_nonnegative(manipulated_pool * equal_row),
                    "_valid_receivers": valid_receivers,
                    **intervention_metadata,
                }
            )
            perturbed_observations.append(perturbed_observation)

    if not perturbed_observations:
        return []

    perturbed_policies = _chunked_policy_forward(actor, perturbed_observations, batch_size=max(1, int(batch_size)))
    response_rows: list[dict[str, Any]] = []
    for metadata, perturbed_policy in zip(pending_metadata, perturbed_policies):
        sender = int(metadata["sender"])
        valid_receivers = np.asarray(metadata["_valid_receivers"], dtype=np.int64)
        allocation_row = np.asarray(perturbed_policy.allocation_matrix[sender, valid_receivers], dtype=np.float64)
        common = {
            "episode": int(metadata["episode"]),
            "step": int(metadata["step"]),
            "sender": sender,
            "sender_degree": int(metadata["sender_degree"]),
            "receiver_count": int(metadata["receiver_count"]),
            "reference_mode": str(metadata["reference_mode"]),
            "multiplier": float(metadata["multiplier"]),
            "reference_pool_value": float(metadata["reference_pool_value"]),
            "original_pool_grown": float(metadata["original_pool_grown"]),
            "manipulated_pool_grown": float(metadata["manipulated_pool_grown"]),
            "sender_pool_upperbound": float(metadata["sender_pool_upperbound"]),
            "manipulated_pool_over_original": float(metadata["manipulated_pool_over_original"]),
            "manipulated_pool_over_upperbound": float(metadata["manipulated_pool_over_upperbound"]),
        }
        response_rows.append(
            {
                **common,
                "policy": "agent",
                "allocated_resource_gini": _gini_nonnegative(float(metadata["manipulated_pool_grown"]) * allocation_row),
            }
        )
        response_rows.append(
            {
                **common,
                "policy": "equal",
                "allocated_resource_gini": float(metadata["equal_allocated_resource_gini"]),
            }
        )
    return response_rows


def compute_rule_policy_pool_intervention_records(
    policy: Any,
    *,
    policy_name: str,
    snapshots: Sequence[Snapshot],
    row_records: Sequence[Mapping[str, Any]],
    reference_mode: str,
    multiplier_values: Sequence[float],
    row_sample_size: int,
    rng_seed: int,
    step_start: int | None,
    step_end: int | None,
) -> list[dict[str, Any]]:
    if not snapshots or not row_records or not multiplier_values:
        return []

    mode = str(reference_mode).strip().lower()
    if mode not in {"original", "upperbound"}:
        raise ValueError(f"Unsupported pool intervention reference mode: {reference_mode}")

    snapshot_lookup = {(snapshot.episode, snapshot.step): snapshot for snapshot in snapshots}
    eligible_indices = [
        index
        for index, row in enumerate(row_records)
        if (int(row["episode"]), int(row["step"])) in snapshot_lookup
        and _step_in_selected_range(
            int(row["step"]),
            None if step_start is None else int(step_start),
            None if step_end is None else int(step_end),
        )
    ]
    if not eligible_indices:
        return []

    rng = np.random.default_rng(int(rng_seed))
    if int(row_sample_size) <= 0:
        sample_count = len(eligible_indices)
    else:
        sample_count = min(len(eligible_indices), int(row_sample_size))
    sampled_indices = rng.choice(np.asarray(eligible_indices, dtype=np.int64), size=sample_count, replace=False)

    response_rows: list[dict[str, Any]] = []
    for row_index in sorted(int(index) for index in sampled_indices.tolist()):
        row = row_records[row_index]
        episode = int(row["episode"])
        step = int(row["step"])
        sender = int(row["sender"])
        snapshot = snapshot_lookup[(episode, step)]
        observation = snapshot.observation
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        valid_receivers = np.flatnonzero(local_mask[sender])
        if valid_receivers.size == 0:
            continue

        original_pool = float(np.asarray(observation["pool_grown"], dtype=np.float64)[sender])
        sender_pool_upperbound = float(row["sender_pool_upperbound"])
        reference_pool = original_pool if mode == "original" else sender_pool_upperbound

        for multiplier in multiplier_values:
            multiplier_value = float(multiplier)
            manipulated_pool = multiplier_value * reference_pool
            perturbed_observation, intervention_metadata = _build_pool_intervention_observation(
                observation,
                sender=sender,
                manipulated_pool_value=manipulated_pool,
                sender_pool_upperbound=sender_pool_upperbound,
            )
            allocation_matrix = np.asarray(policy.allocate(perturbed_observation), dtype=np.float64)
            allocation_row = np.asarray(allocation_matrix[sender, valid_receivers], dtype=np.float64)
            response_rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "sender": sender,
                    "sender_degree": int(row["sender_degree"]),
                    "receiver_count": int(valid_receivers.size),
                    "reference_mode": mode,
                    "multiplier": multiplier_value,
                    "reference_pool_value": float(reference_pool),
                    "original_pool_grown": original_pool,
                    "manipulated_pool_grown": float(intervention_metadata["manipulated_pool_grown"]),
                    "sender_pool_upperbound": float(sender_pool_upperbound),
                    "manipulated_pool_over_original": (
                        float(manipulated_pool / original_pool) if abs(original_pool) > 1e-12 else float("nan")
                    ),
                    "manipulated_pool_over_upperbound": float(intervention_metadata["manipulated_pool_over_upperbound"]),
                    "policy": str(policy_name),
                    "allocated_resource_gini": _gini_nonnegative(
                        float(intervention_metadata["manipulated_pool_grown"]) * allocation_row
                    ),
                }
            )
    return response_rows


def compute_resource_intervention_records(
    actor: Any,
    snapshots: Sequence[Snapshot],
    row_records: Sequence[Mapping[str, Any]],
    *,
    reference_mode: str,
    multiplier_values: Sequence[float],
    row_sample_size: int,
    batch_size: int,
    rng_seed: int,
    step_start: int | None,
    step_end: int | None,
    resource_norm_reference: float,
) -> list[dict[str, Any]]:
    if not snapshots or not row_records or not multiplier_values:
        return []

    mode = str(reference_mode).strip().lower()
    if mode not in {"original", "norm_reference"}:
        raise ValueError(f"Unsupported resource intervention reference mode: {reference_mode}")

    snapshot_lookup = {(snapshot.episode, snapshot.step): snapshot for snapshot in snapshots}
    eligible_indices = [
        index
        for index, row in enumerate(row_records)
        if (int(row["episode"]), int(row["step"])) in snapshot_lookup
        and _step_in_selected_range(
            int(row["step"]),
            None if step_start is None else int(step_start),
            None if step_end is None else int(step_end),
        )
    ]
    if not eligible_indices:
        return []

    rng = np.random.default_rng(int(rng_seed))
    if int(row_sample_size) <= 0:
        sample_count = len(eligible_indices)
    else:
        sample_count = min(len(eligible_indices), int(row_sample_size))
    sampled_indices = rng.choice(np.asarray(eligible_indices, dtype=np.int64), size=sample_count, replace=False)

    equal_policy = UniformAllocationPolicy()

    pending_metadata: list[dict[str, Any]] = []
    perturbed_observations: list[dict[str, np.ndarray]] = []
    for row_index in sorted(int(index) for index in sampled_indices.tolist()):
        row = row_records[row_index]
        episode = int(row["episode"])
        step = int(row["step"])
        sender = int(row["sender"])
        snapshot = snapshot_lookup[(episode, step)]
        observation = snapshot.observation
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        valid_receivers = np.flatnonzero(local_mask[sender])
        if valid_receivers.size == 0:
            continue

        equal_allocation = np.asarray(equal_policy.allocate(observation), dtype=np.float64)
        equal_row = np.asarray(equal_allocation[sender, valid_receivers], dtype=np.float64)

        original_resource = float(np.asarray(observation["resources"], dtype=np.float64)[sender])
        reference_resource = original_resource if mode == "original" else float(resource_norm_reference)
        pool_grown = float(np.asarray(observation["pool_grown"], dtype=np.float64)[sender])

        for multiplier in multiplier_values:
            multiplier_value = float(multiplier)
            manipulated_resource = multiplier_value * reference_resource
            perturbed_observation, intervention_metadata = _build_resource_intervention_observation(
                observation,
                sender=sender,
                manipulated_resource_value=manipulated_resource,
                resource_norm_reference=float(resource_norm_reference),
            )
            pending_metadata.append(
                {
                    "episode": episode,
                    "step": step,
                    "sender": sender,
                    "sender_degree": int(row["sender_degree"]),
                    "receiver_count": int(valid_receivers.size),
                    "reference_mode": mode,
                    "multiplier": multiplier_value,
                    "reference_resource_value": float(reference_resource),
                    "original_resource": original_resource,
                    "original_pool_grown": pool_grown,
                    "manipulated_resource_over_original": (
                        float(manipulated_resource / original_resource) if abs(original_resource) > 1e-12 else float("nan")
                    ),
                    "equal_allocated_resource_gini": _gini_nonnegative(pool_grown * equal_row),
                    "_valid_receivers": valid_receivers,
                    **intervention_metadata,
                }
            )
            perturbed_observations.append(perturbed_observation)

    if not perturbed_observations:
        return []

    perturbed_policies = _chunked_policy_forward(actor, perturbed_observations, batch_size=max(1, int(batch_size)))
    response_rows: list[dict[str, Any]] = []
    for metadata, perturbed_policy in zip(pending_metadata, perturbed_policies):
        sender = int(metadata["sender"])
        valid_receivers = np.asarray(metadata["_valid_receivers"], dtype=np.int64)
        allocation_row = np.asarray(perturbed_policy.allocation_matrix[sender, valid_receivers], dtype=np.float64)
        common = {
            "episode": int(metadata["episode"]),
            "step": int(metadata["step"]),
            "sender": sender,
            "sender_degree": int(metadata["sender_degree"]),
            "receiver_count": int(metadata["receiver_count"]),
            "reference_mode": str(metadata["reference_mode"]),
            "multiplier": float(metadata["multiplier"]),
            "reference_resource_value": float(metadata["reference_resource_value"]),
            "original_resource": float(metadata["original_resource"]),
            "manipulated_resource": float(metadata["manipulated_resource"]),
            "resource_norm_reference": float(metadata["resource_norm_reference"]),
            "manipulated_resource_over_original": float(metadata["manipulated_resource_over_original"]),
            "manipulated_resource_over_reference": float(metadata["manipulated_resource_over_reference"]),
            "original_pool_grown": float(metadata["original_pool_grown"]),
        }
        response_rows.append(
            {
                **common,
                "policy": "agent",
                "allocated_resource_gini": _gini_nonnegative(float(metadata["original_pool_grown"]) * allocation_row),
            }
        )
        response_rows.append(
            {
                **common,
                "policy": "equal",
                "allocated_resource_gini": float(metadata["equal_allocated_resource_gini"]),
            }
        )
    return response_rows


def compute_rule_policy_resource_intervention_records(
    policy: Any,
    *,
    policy_name: str,
    snapshots: Sequence[Snapshot],
    row_records: Sequence[Mapping[str, Any]],
    reference_mode: str,
    multiplier_values: Sequence[float],
    row_sample_size: int,
    rng_seed: int,
    step_start: int | None,
    step_end: int | None,
    resource_norm_reference: float,
) -> list[dict[str, Any]]:
    if not snapshots or not row_records or not multiplier_values:
        return []

    mode = str(reference_mode).strip().lower()
    if mode not in {"original", "norm_reference"}:
        raise ValueError(f"Unsupported resource intervention reference mode: {reference_mode}")

    snapshot_lookup = {(snapshot.episode, snapshot.step): snapshot for snapshot in snapshots}
    eligible_indices = [
        index
        for index, row in enumerate(row_records)
        if (int(row["episode"]), int(row["step"])) in snapshot_lookup
        and _step_in_selected_range(
            int(row["step"]),
            None if step_start is None else int(step_start),
            None if step_end is None else int(step_end),
        )
    ]
    if not eligible_indices:
        return []

    rng = np.random.default_rng(int(rng_seed))
    if int(row_sample_size) <= 0:
        sample_count = len(eligible_indices)
    else:
        sample_count = min(len(eligible_indices), int(row_sample_size))
    sampled_indices = rng.choice(np.asarray(eligible_indices, dtype=np.int64), size=sample_count, replace=False)

    response_rows: list[dict[str, Any]] = []
    for row_index in sorted(int(index) for index in sampled_indices.tolist()):
        row = row_records[row_index]
        episode = int(row["episode"])
        step = int(row["step"])
        sender = int(row["sender"])
        snapshot = snapshot_lookup[(episode, step)]
        observation = snapshot.observation
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        valid_receivers = np.flatnonzero(local_mask[sender])
        if valid_receivers.size == 0:
            continue

        original_resource = float(np.asarray(observation["resources"], dtype=np.float64)[sender])
        reference_resource = original_resource if mode == "original" else float(resource_norm_reference)
        pool_grown = float(np.asarray(observation["pool_grown"], dtype=np.float64)[sender])

        for multiplier in multiplier_values:
            multiplier_value = float(multiplier)
            manipulated_resource = multiplier_value * reference_resource
            perturbed_observation, intervention_metadata = _build_resource_intervention_observation(
                observation,
                sender=sender,
                manipulated_resource_value=manipulated_resource,
                resource_norm_reference=float(resource_norm_reference),
            )
            allocation_matrix = np.asarray(policy.allocate(perturbed_observation), dtype=np.float64)
            allocation_row = np.asarray(allocation_matrix[sender, valid_receivers], dtype=np.float64)
            response_rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "sender": sender,
                    "sender_degree": int(row["sender_degree"]),
                    "receiver_count": int(valid_receivers.size),
                    "reference_mode": mode,
                    "multiplier": multiplier_value,
                    "reference_resource_value": float(reference_resource),
                    "original_resource": original_resource,
                    "manipulated_resource": float(intervention_metadata["manipulated_resource"]),
                    "resource_norm_reference": float(resource_norm_reference),
                    "manipulated_resource_over_original": (
                        float(manipulated_resource / original_resource) if abs(original_resource) > 1e-12 else float("nan")
                    ),
                    "manipulated_resource_over_reference": float(intervention_metadata["manipulated_resource_over_reference"]),
                    "original_pool_grown": pool_grown,
                    "policy": str(policy_name),
                    "allocated_resource_gini": _gini_nonnegative(pool_grown * allocation_row),
                }
            )
    return response_rows


def summarize_pool_intervention_records(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {}

    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy"]), float(row["multiplier"]))
        grouped.setdefault(key, []).append(row)

    policy_order = {"equal": 0, "proportional": 1, "agent": 2}
    summary_rows: list[dict[str, Any]] = []
    for (policy_name, multiplier), group in sorted(
        grouped.items(),
        key=lambda item: (policy_order.get(item[0][0], 99), item[0][1]),
    ):
        gini_values = np.asarray([float(row["allocated_resource_gini"]) for row in group], dtype=np.float64)
        count_rows = int(gini_values.size)
        gini_std = float(np.std(gini_values)) if count_rows > 0 else float("nan")
        gini_sem = float(gini_std / np.sqrt(count_rows)) if count_rows > 0 else float("nan")
        summary_rows.append(
            {
                "policy": str(policy_name),
                "multiplier": float(multiplier),
                "count_rows": count_rows,
                "allocated_resource_gini_mean": _nanmean_or_nan(gini_values),
                "allocated_resource_gini_std": gini_std,
                "allocated_resource_gini_sem": gini_sem,
                "manipulated_pool_grown_mean": _nanmean_or_nan(
                    [float(row["manipulated_pool_grown"]) for row in group]
                ),
                "manipulated_pool_over_original_mean": _nanmean_or_nan(
                    [float(row["manipulated_pool_over_original"]) for row in group]
                ),
                "manipulated_pool_over_upperbound_mean": _nanmean_or_nan(
                    [float(row["manipulated_pool_over_upperbound"]) for row in group]
                ),
            }
        )

    agent_rows = [row for row in summary_rows if str(row["policy"]) == "agent"]
    unique_sender_rows = {
        (int(row["episode"]), int(row["step"]), int(row["sender"]))
        for row in rows
        if str(row["policy"]) == "agent"
    }
    flat_summary = {
        "pool_intervention_record_count": int(len(rows)),
        "pool_intervention_sender_row_count": int(len(unique_sender_rows)),
        "pool_intervention_multiplier_count": int(len({float(row["multiplier"]) for row in rows})),
    }
    if agent_rows:
        sorted_agent_rows = sorted(agent_rows, key=lambda row: float(row["multiplier"]))
        reference_row = min(sorted_agent_rows, key=lambda row: abs(float(row["multiplier"]) - 1.0))
        lowest_row = sorted_agent_rows[0]
        highest_row = sorted_agent_rows[-1]
        flat_summary.update(
            {
                "pool_intervention_agent_gini_reference_mean": float(
                    reference_row["allocated_resource_gini_mean"]
                ),
                "pool_intervention_agent_gini_lowest_multiplier_mean": float(
                    lowest_row["allocated_resource_gini_mean"]
                ),
                "pool_intervention_agent_gini_highest_multiplier_mean": float(
                    highest_row["allocated_resource_gini_mean"]
                ),
                "pool_intervention_agent_gini_high_minus_low": float(
                    highest_row["allocated_resource_gini_mean"] - lowest_row["allocated_resource_gini_mean"]
                ),
            }
        )
    return summary_rows, flat_summary


def summarize_resource_intervention_records(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {}

    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy"]), float(row["multiplier"]))
        grouped.setdefault(key, []).append(row)

    policy_order = {"equal": 0, "proportional": 1, "agent": 2}
    summary_rows: list[dict[str, Any]] = []
    for (policy_name, multiplier), group in sorted(
        grouped.items(),
        key=lambda item: (policy_order.get(item[0][0], 99), item[0][1]),
    ):
        gini_values = np.asarray([float(row["allocated_resource_gini"]) for row in group], dtype=np.float64)
        count_rows = int(gini_values.size)
        gini_std = float(np.std(gini_values)) if count_rows > 0 else float("nan")
        gini_sem = float(gini_std / np.sqrt(count_rows)) if count_rows > 0 else float("nan")
        summary_rows.append(
            {
                "policy": str(policy_name),
                "multiplier": float(multiplier),
                "count_rows": count_rows,
                "allocated_resource_gini_mean": _nanmean_or_nan(gini_values),
                "allocated_resource_gini_std": gini_std,
                "allocated_resource_gini_sem": gini_sem,
                "manipulated_resource_mean": _nanmean_or_nan(
                    [float(row["manipulated_resource"]) for row in group]
                ),
                "manipulated_resource_over_original_mean": _nanmean_or_nan(
                    [float(row["manipulated_resource_over_original"]) for row in group]
                ),
                "manipulated_resource_over_reference_mean": _nanmean_or_nan(
                    [float(row["manipulated_resource_over_reference"]) for row in group]
                ),
            }
        )

    agent_rows = [row for row in summary_rows if str(row["policy"]) == "agent"]
    unique_sender_rows = {
        (int(row["episode"]), int(row["step"]), int(row["sender"]))
        for row in rows
        if str(row["policy"]) == "agent"
    }
    flat_summary = {
        "resource_intervention_record_count": int(len(rows)),
        "resource_intervention_sender_row_count": int(len(unique_sender_rows)),
        "resource_intervention_multiplier_count": int(len({float(row["multiplier"]) for row in rows})),
    }
    if agent_rows:
        sorted_agent_rows = sorted(agent_rows, key=lambda row: float(row["multiplier"]))
        reference_row = min(sorted_agent_rows, key=lambda row: abs(float(row["multiplier"]) - 1.0))
        lowest_row = sorted_agent_rows[0]
        highest_row = sorted_agent_rows[-1]
        flat_summary.update(
            {
                "resource_intervention_agent_gini_reference_mean": float(
                    reference_row["allocated_resource_gini_mean"]
                ),
                "resource_intervention_agent_gini_lowest_multiplier_mean": float(
                    lowest_row["allocated_resource_gini_mean"]
                ),
                "resource_intervention_agent_gini_highest_multiplier_mean": float(
                    highest_row["allocated_resource_gini_mean"]
                ),
                "resource_intervention_agent_gini_high_minus_low": float(
                    highest_row["allocated_resource_gini_mean"] - lowest_row["allocated_resource_gini_mean"]
                ),
            }
        )
    return summary_rows, flat_summary


def summarize_policy_rollout_step_metrics(
    snapshots_by_policy: Mapping[str, Sequence[Snapshot]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for policy_name, snapshots in snapshots_by_policy.items():
        if not snapshots:
            continue
        per_step: dict[int, dict[str, list[float]]] = {}
        for snapshot in snapshots:
            step = int(snapshot.step)
            bucket = per_step.setdefault(
                step,
                {
                    "reward": [],
                    "cooperation": [],
                    "resource_gini": [],
                    "mean_resource": [],
                    "mean_pool_grown": [],
                    "mean_self_allocation": [],
                    "offer_gini": [],
                },
            )
            observation = snapshot.observation
            allocation_matrix = np.asarray(snapshot.policy.allocation_matrix, dtype=np.float64)
            local_mask = np.asarray(observation["local_mask"], dtype=bool)
            pool_grown = np.asarray(observation["pool_grown"], dtype=np.float64)

            row_ginis: list[float] = []
            for sender in range(local_mask.shape[0]):
                valid_receivers = np.flatnonzero(local_mask[sender])
                allocation_row = np.asarray(allocation_matrix[sender, valid_receivers], dtype=np.float64)
                row_ginis.append(_gini_nonnegative(float(pool_grown[sender]) * allocation_row))

            bucket["reward"].append(float(snapshot.reward))
            bucket["cooperation"].append(float(snapshot.actual_cooperation_rate))
            bucket["resource_gini"].append(float(snapshot.gini))
            bucket["mean_resource"].append(float(np.mean(observation["resources"])))
            bucket["mean_pool_grown"].append(float(np.mean(pool_grown)))
            bucket["mean_self_allocation"].append(float(np.mean(np.diag(allocation_matrix))))
            bucket["offer_gini"].append(_nanmean_or_nan(row_ginis))

        for step, bucket in sorted(per_step.items()):
            summary_rows.append(
                {
                    "policy": str(policy_name),
                    "step": int(step),
                    "count_episodes": int(len(bucket["reward"])),
                    "reward_mean": _nanmean_or_nan(bucket["reward"]),
                    "cooperation_rate_mean": _nanmean_or_nan(bucket["cooperation"]),
                    "resource_gini_mean": _nanmean_or_nan(bucket["resource_gini"]),
                    "mean_resource_mean": _nanmean_or_nan(bucket["mean_resource"]),
                    "mean_pool_grown_mean": _nanmean_or_nan(bucket["mean_pool_grown"]),
                    "mean_self_allocation_mean": _nanmean_or_nan(bucket["mean_self_allocation"]),
                    "offer_gini_mean": _nanmean_or_nan(bucket["offer_gini"]),
                }
            )
    return summary_rows


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


def _save_figure(figure: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


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

    _save_figure(figure, output_path)


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

    _save_figure(figure, output_path)


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

    _save_figure(figure, output_path)


def plot_allocation_gini_vs_observed_position(summary_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not summary_rows:
        return

    figure, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    style_by_policy = {
        "agent": {"linestyle": "-", "marker": "o", "linewidth": 2.0},
        "proportional": {"linestyle": "--", "marker": None, "linewidth": 2.0},
        "equal": {"linestyle": "--", "marker": None, "linewidth": 2.0},
    }
    policy_names = sorted({str(row["policy"]) for row in summary_rows})
    for policy_name in policy_names:
        rows = [row for row in summary_rows if str(row["policy"]) == policy_name]
        rows.sort(key=lambda row: int(row["bin_index"]))
        x_values = np.asarray([float(row["pool_grown_over_upperbound_mean"]) for row in rows], dtype=np.float64)
        gini_values = np.asarray([float(row["allocated_resource_gini_mean"]) for row in rows], dtype=np.float64)
        style = style_by_policy.get(policy_name, {"linestyle": "-", "marker": "o", "linewidth": 1.5})
        ax.plot(x_values, gini_values, label=policy_name, **style)
    ax.set_title("Allocated Resource Gini vs rho_p")
    ax.set_xlabel("rho_p = P_grown / P_upperbound")
    ax.set_ylabel("Gini over ego allocated resources")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    ax.grid(alpha=0.25)

    _save_figure(figure, output_path)


def plot_allocation_gini_vs_position(
    summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    reference_mode: str,
) -> None:
    if plt is None or not summary_rows:
        return

    figure, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    style_by_policy = {
        "equal": {
            "color": "#1f77ff",
            "label": "Equal Baseline",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.0,
        },
        "proportional": {
            "color": "#d62728",
            "label": "Proportional Baseline",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.0,
        },
        "agent": {
            "color": "#2ca02c",
            "label": "RL Agent",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.2,
        },
    }
    for policy_name in ("equal", "proportional", "agent"):
        rows = [row for row in summary_rows if str(row["policy"]) == policy_name]
        if not rows:
            continue
        rows.sort(key=lambda row: float(row["multiplier"]))
        x_values = np.asarray([float(row["multiplier"]) for row in rows], dtype=np.float64)
        gini_values = np.asarray([float(row["allocated_resource_gini_mean"]) for row in rows], dtype=np.float64)
        sem_values = np.asarray([float(row["allocated_resource_gini_sem"]) for row in rows], dtype=np.float64)
        style = style_by_policy[policy_name]
        ax.plot(
            x_values,
            gini_values,
            color=style["color"],
            label=style["label"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=4.0,
        )
        if np.any(np.isfinite(sem_values)):
            lower = np.clip(gini_values - sem_values, 0.0, None)
            upper = np.clip(gini_values + sem_values, 0.0, None)
            band_mask = np.isfinite(x_values) & np.isfinite(lower) & np.isfinite(upper)
            if int(np.count_nonzero(band_mask)) >= 2:
                ax.fill_between(
                    x_values,
                    lower,
                    upper,
                    where=band_mask,
                    color=style["color"],
                    alpha=0.16,
                )

    reference_text = "original pool" if str(reference_mode) == "original" else "P_upperbound"
    ax.set_title("Allocation Gini under Pool Intervention")
    ax.set_xscale("log")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_xlabel(f"Manipulated pool as a fraction of {reference_text} (log scale)")
    ax.set_ylabel("Gini of manipulated offers")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    ax.grid(alpha=0.25, which="both")

    _save_figure(figure, output_path)


def plot_allocation_gini_vs_resource_intervention(
    summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    reference_mode: str,
) -> None:
    if plt is None or not summary_rows:
        return

    figure, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    style_by_policy = {
        "equal": {
            "color": "#1f77ff",
            "label": "Equal Baseline",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.0,
        },
        "proportional": {
            "color": "#d62728",
            "label": "Proportional Baseline",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.0,
        },
        "agent": {
            "color": "#2ca02c",
            "label": "RL Agent",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.2,
        },
    }
    for policy_name in ("equal", "proportional", "agent"):
        rows = [row for row in summary_rows if str(row["policy"]) == policy_name]
        if not rows:
            continue
        rows.sort(key=lambda row: float(row["multiplier"]))
        x_values = np.asarray([float(row["multiplier"]) for row in rows], dtype=np.float64)
        gini_values = np.asarray([float(row["allocated_resource_gini_mean"]) for row in rows], dtype=np.float64)
        sem_values = np.asarray([float(row["allocated_resource_gini_sem"]) for row in rows], dtype=np.float64)
        style = style_by_policy[policy_name]
        ax.plot(
            x_values,
            gini_values,
            color=style["color"],
            label=style["label"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=4.0,
        )
        if np.any(np.isfinite(sem_values)):
            lower = np.clip(gini_values - sem_values, 0.0, None)
            upper = np.clip(gini_values + sem_values, 0.0, None)
            band_mask = np.isfinite(x_values) & np.isfinite(lower) & np.isfinite(upper)
            if int(np.count_nonzero(band_mask)) >= 2:
                ax.fill_between(
                    x_values,
                    lower,
                    upper,
                    where=band_mask,
                    color=style["color"],
                    alpha=0.16,
                )

    reference_text = "original sender resource" if str(reference_mode) == "original" else "resource_norm_reference"
    ax.set_title("Allocation Gini under Resource Intervention")
    ax.set_xscale("log")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_xlabel(f"Manipulated sender resource as a fraction of {reference_text} (log scale)")
    ax.set_ylabel("Gini of manipulated offers")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    ax.grid(alpha=0.25, which="both")

    _save_figure(figure, output_path)


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

    _save_figure(figure, output_path)


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

    axes[1].plot(steps, ratio, label="P/Pupper", color="#9467bd")
    axes[1].plot(steps, self_alloc, label="self allocation", color="#d62728")
    axes[1].set_title("Position Ratio and Self Allocation Over Time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean Value")
    axes[1].legend()

    _save_figure(figure, output_path)


def _plot_policy_rollout_metric_grid(
    step_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    metric_specs: Sequence[tuple[str, str, str]],
    layout: tuple[int, int],
) -> None:
    if plt is None or not step_rows:
        return

    style_by_policy = {
        "agent": {"color": "#2ca02c", "label": "RL Agent"},
        "proportional": {"color": "#d62728", "label": "Proportional"},
        "equal": {"color": "#1f77ff", "label": "Equal"},
    }
    rows, cols = int(layout[0]), int(layout[1])
    figure_width = 6.2 * float(cols)
    figure_height = 3.8 * float(rows)
    figure, axes = plt.subplots(rows, cols, figsize=(figure_width, figure_height), constrained_layout=True, sharex=True)
    axes_flat = list(np.asarray(axes).reshape(-1))

    for axis, (metric_key, title, ylabel) in zip(axes_flat, metric_specs):
        for policy_name in ("agent", "proportional", "equal"):
            rows = [row for row in step_rows if str(row["policy"]) == policy_name]
            if not rows:
                continue
            rows.sort(key=lambda row: int(row["step"]))
            x_values = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
            y_values = np.asarray([float(row[metric_key]) for row in rows], dtype=np.float64)
            style = style_by_policy[policy_name]
            axis.plot(
                x_values,
                y_values,
                color=style["color"],
                label=style["label"],
                linewidth=2.0,
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    for axis in axes_flat[len(metric_specs):]:
        axis.set_axis_off()
    for axis in axes_flat[max(0, len(axes_flat) - cols):]:
        axis.set_xlabel("Step t")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    _save_figure(figure, output_path)


def plot_policy_rollout_outcomes_over_time(
    step_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    metric_specs = [
        ("reward_mean", "Reward", "Mean Reward"),
        ("cooperation_rate_mean", "Cooperation Rate", "Actual Cooperation Rate"),
        ("resource_gini_mean", "Resource Gini", "Resource Gini"),
        ("mean_resource_mean", "Mean Resource", "Mean Resource"),
    ]
    _plot_policy_rollout_metric_grid(
        step_rows,
        output_path,
        metric_specs=metric_specs,
        layout=(2, 2),
    )


def plot_policy_rollout_behavior_over_time(
    step_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    metric_specs = [
        ("mean_self_allocation_mean", "Mean Self Allocation", "Mean Self Allocation"),
        ("offer_gini_mean", "Offer Gini", "Mean Offer Gini"),
    ]
    _plot_policy_rollout_metric_grid(
        step_rows,
        output_path,
        metric_specs=metric_specs,
        layout=(1, 2),
    )


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

    _save_figure(figure, output_path)


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

    _save_figure(figure, output_path)


def build_mechanism_summary(
    *,
    run_dir: Path,
    checkpoint_name: str,
    checkpoint_payload: Mapping[str, Any],
    topology_name: str,
    network_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    effective_episode_length: int,
    requested_episode_length_override: int | None,
    graph_stats: Mapping[str, Any],
    policy_behavior: Mapping[str, Any],
    mechanism_summary: Mapping[str, Any],
    node_income_summary: Mapping[str, Any],
    labor_equal_gap_threshold: float,
    pool_intervention_summary_rows: Sequence[Mapping[str, Any]],
    pool_intervention_flat_summary: Mapping[str, Any],
    pool_intervention_config: Mapping[str, Any],
    resource_intervention_summary_rows: Sequence[Mapping[str, Any]],
    resource_intervention_flat_summary: Mapping[str, Any],
    resource_intervention_config: Mapping[str, Any],
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
        "dynamics_config": dict(dynamics_config),
        "effective_episode_length": int(effective_episode_length),
        "requested_episode_length_override": (
            None if requested_episode_length_override is None else int(requested_episode_length_override)
        ),
        "position_metric": {
            "name": "pool_grown_over_upperbound",
            "formula": "pool_grown / ((sender_degree + 1) * p_c)",
            "sender_degree_definition": "number of neighbors excluding self; +1 includes sender itself",
        },
        "labor_equal_identifiability": {
            "metric": "L1 distance between labor baseline and equal baseline within each sender row",
            "threshold": float(labor_equal_gap_threshold),
        },
        "graph_stats": dict(graph_stats),
        "policy_behavior": dict(policy_behavior),
        "mechanism_summary": dict(mechanism_summary),
        "node_income_summary": dict(node_income_summary),
        "pool_intervention_analysis": {
            "config": dict(pool_intervention_config),
            "summary": dict(pool_intervention_flat_summary),
            "summary_rows": [dict(row) for row in pool_intervention_summary_rows],
        },
        "resource_intervention_analysis": {
            "config": dict(resource_intervention_config),
            "summary": dict(resource_intervention_flat_summary),
            "summary_rows": [dict(row) for row in resource_intervention_summary_rows],
        },
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
    pool_intervention_flat_summary: Mapping[str, Any],
    resource_intervention_flat_summary: Mapping[str, Any],
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
    row.update({key: value for key, value in pool_intervention_flat_summary.items()})
    row.update({key: value for key, value in resource_intervention_flat_summary.items()})
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
    enable_pool_intervention_analysis: bool,
    pool_intervention_reference_mode: str,
    pool_intervention_multipliers: Sequence[float],
    pool_intervention_row_sample_size: int,
    pool_intervention_batch_size: int,
    agent_equal_pool_intervention_step_start: int | None,
    agent_equal_pool_intervention_step_end: int | None,
    proportional_pool_intervention_step_start: int | None,
    proportional_pool_intervention_step_end: int | None,
    enable_resource_intervention_analysis: bool,
    resource_intervention_reference_mode: str,
    resource_intervention_multipliers: Sequence[float],
    resource_intervention_row_sample_size: int,
    resource_intervention_batch_size: int,
    agent_equal_resource_intervention_step_start: int | None,
    agent_equal_resource_intervention_step_end: int | None,
    proportional_resource_intervention_step_start: int | None,
    proportional_resource_intervention_step_end: int | None,
    enable_counterfactual_analysis: bool,
    counterfactual_row_sample_size: int,
    counterfactual_contribution_delta: float,
    counterfactual_batch_size: int,
    baseline_gini_episodes: int,
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
    agent_allocation_gini_records = [
        {
            "policy": "agent",
            "episode": int(row["episode"]),
            "step": int(row["step"]),
            "sender": int(row["sender"]),
            "sender_degree": int(row["sender_degree"]),
            "receiver_count": int(row["receiver_count"]),
            "sender_pool_grown": float(row["sender_pool_grown"]),
            "sender_pool_upperbound": float(row["sender_pool_upperbound"]),
            "pool_grown_over_upperbound": float(row["pool_grown_over_upperbound"]),
            "allocated_resource_gini": float(row["agent_allocated_resource_gini"]),
        }
        for row in row_mechanism_records
    ]
    baseline_allocation_gini_records: list[dict[str, Any]] = []
    baseline_episode_summary_rows: list[dict[str, Any]] = []
    baseline_snapshots_by_policy: dict[str, list[Snapshot]] = {}
    baseline_episode_count = max(0, int(baseline_gini_episodes))
    if baseline_episode_count > 0:
        baseline_specs = [
            ("proportional", ProportionalContributionPolicy(), int(rollout_seed) + 100_000),
            ("equal", UniformAllocationPolicy(), int(rollout_seed) + 200_000),
        ]
        for policy_name, policy, policy_seed in baseline_specs:
            policy_snapshots, policy_episode_summaries = run_rule_policy_rollouts(
                policy,
                env_config=env_config,
                graph=graph,
                episodes=baseline_episode_count,
                seed=policy_seed,
                max_steps=max_steps,
            )
            baseline_snapshots_by_policy[str(policy_name)] = list(policy_snapshots)
            policy_records = extract_policy_allocation_gini_records_from_snapshots(policy_name, policy_snapshots)
            baseline_allocation_gini_records.extend(policy_records)
            baseline_episode_summary_rows.extend(
                {
                    "policy": policy_name,
                    "episode": summary.episode,
                    "steps": summary.steps,
                    "total_reward": summary.total_reward,
                    "mean_reward": summary.mean_reward,
                    "final_cooperation_rate": summary.final_cooperation_rate,
                    "final_gini": summary.final_gini,
                    "final_mean_resource": summary.final_mean_resource,
                }
                for summary in policy_episode_summaries
            )
    allocation_gini_position_rows = summarize_allocation_gini_rows(
        {
            "agent": agent_allocation_gini_records,
            "proportional": [
                row for row in baseline_allocation_gini_records if str(row["policy"]) == "proportional"
            ],
            "equal": [row for row in baseline_allocation_gini_records if str(row["policy"]) == "equal"],
        },
        bin_count=mechanism_bin_count,
    )
    policy_rollout_step_rows = summarize_policy_rollout_step_metrics(
        {
            "agent": snapshots,
            "proportional": baseline_snapshots_by_policy.get("proportional", []),
            "equal": baseline_snapshots_by_policy.get("equal", []),
        }
    )
    step_mechanism_rows = summarize_step_mechanism_records(row_mechanism_records)
    sender_state_rows = summarize_sender_state_mechanism(row_mechanism_records)
    node_income_summary = summarize_node_income_records(node_income_records)
    receiver_state_rows = summarize_receiver_state_income(node_income_records)
    proportional_intervention_policy = ProportionalContributionPolicy()
    proportional_intervention_snapshots = baseline_snapshots_by_policy.get("proportional", [])
    proportional_intervention_row_records: list[dict[str, Any]] | None = None
    pool_intervention_rows: list[dict[str, Any]] = []
    pool_intervention_summary_rows: list[dict[str, Any]] = []
    pool_intervention_flat_summary: dict[str, Any] = {}
    if enable_pool_intervention_analysis:
        if not proportional_intervention_snapshots:
            proportional_intervention_snapshots, _ = run_rule_policy_rollouts(
                proportional_intervention_policy,
                env_config=env_config,
                graph=graph,
                episodes=episodes,
                seed=int(rollout_seed) + 100_000,
                max_steps=max_steps,
            )
        if proportional_intervention_row_records is None:
            proportional_intervention_row_records = compute_row_mechanism_records(
                proportional_intervention_snapshots,
                p_c=float(env_config.p_c),
                labor_equal_gap_threshold=float(labor_equal_gap_threshold),
            )
        pool_intervention_rows = compute_pool_intervention_records(
            actor,
            snapshots,
            row_mechanism_records,
            reference_mode=str(pool_intervention_reference_mode),
            multiplier_values=pool_intervention_multipliers,
            row_sample_size=int(pool_intervention_row_sample_size),
            batch_size=max(1, int(pool_intervention_batch_size)),
            rng_seed=int(rollout_seed) + 11,
            step_start=(
                None
                if agent_equal_pool_intervention_step_start is None
                else int(agent_equal_pool_intervention_step_start)
            ),
            step_end=(
                None
                if agent_equal_pool_intervention_step_end is None
                else int(agent_equal_pool_intervention_step_end)
            ),
        )
        pool_intervention_rows.extend(
            compute_rule_policy_pool_intervention_records(
                proportional_intervention_policy,
                policy_name="proportional",
                snapshots=proportional_intervention_snapshots,
                row_records=proportional_intervention_row_records,
                reference_mode=str(pool_intervention_reference_mode),
                multiplier_values=pool_intervention_multipliers,
                row_sample_size=int(pool_intervention_row_sample_size),
                rng_seed=int(rollout_seed) + 100_011,
                step_start=(
                    None
                    if proportional_pool_intervention_step_start is None
                    else int(proportional_pool_intervention_step_start)
                ),
                step_end=(
                    None
                    if proportional_pool_intervention_step_end is None
                    else int(proportional_pool_intervention_step_end)
                ),
            )
        )
        pool_intervention_summary_rows, pool_intervention_flat_summary = summarize_pool_intervention_records(
            pool_intervention_rows
        )
    resource_intervention_rows: list[dict[str, Any]] = []
    resource_intervention_summary_rows: list[dict[str, Any]] = []
    resource_intervention_flat_summary: dict[str, Any] = {}
    if enable_resource_intervention_analysis:
        if not proportional_intervention_snapshots:
            proportional_intervention_snapshots, _ = run_rule_policy_rollouts(
                proportional_intervention_policy,
                env_config=env_config,
                graph=graph,
                episodes=episodes,
                seed=int(rollout_seed) + 100_000,
                max_steps=max_steps,
            )
        if proportional_intervention_row_records is None:
            proportional_intervention_row_records = compute_row_mechanism_records(
                proportional_intervention_snapshots,
                p_c=float(env_config.p_c),
                labor_equal_gap_threshold=float(labor_equal_gap_threshold),
            )
        resource_intervention_rows = compute_resource_intervention_records(
            actor,
            snapshots,
            row_mechanism_records,
            reference_mode=str(resource_intervention_reference_mode),
            multiplier_values=resource_intervention_multipliers,
            row_sample_size=int(resource_intervention_row_sample_size),
            batch_size=max(1, int(resource_intervention_batch_size)),
            rng_seed=int(rollout_seed) + 21,
            step_start=(
                None
                if agent_equal_resource_intervention_step_start is None
                else int(agent_equal_resource_intervention_step_start)
            ),
            step_end=(
                None
                if agent_equal_resource_intervention_step_end is None
                else int(agent_equal_resource_intervention_step_end)
            ),
            resource_norm_reference=float(env.resource_norm_reference),
        )
        resource_intervention_rows.extend(
            compute_rule_policy_resource_intervention_records(
                proportional_intervention_policy,
                policy_name="proportional",
                snapshots=proportional_intervention_snapshots,
                row_records=proportional_intervention_row_records,
                reference_mode=str(resource_intervention_reference_mode),
                multiplier_values=resource_intervention_multipliers,
                row_sample_size=int(resource_intervention_row_sample_size),
                rng_seed=int(rollout_seed) + 100_021,
                step_start=(
                    None
                    if proportional_resource_intervention_step_start is None
                    else int(proportional_resource_intervention_step_start)
                ),
                step_end=(
                    None
                    if proportional_resource_intervention_step_end is None
                    else int(proportional_resource_intervention_step_end)
                ),
                resource_norm_reference=float(env.resource_norm_reference),
            )
        )
        resource_intervention_summary_rows, resource_intervention_flat_summary = summarize_resource_intervention_records(
            resource_intervention_rows
        )
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
    plot_allocation_gini_vs_position(
        pool_intervention_summary_rows,
        output_dir / "allocation_gini_vs_position.png",
        reference_mode=str(pool_intervention_reference_mode),
    )
    plot_allocation_gini_vs_resource_intervention(
        resource_intervention_summary_rows,
        output_dir / "allocation_gini_vs_resource_intervention.png",
        reference_mode=str(resource_intervention_reference_mode),
    )
    plot_allocation_gini_vs_observed_position(
        allocation_gini_position_rows,
        output_dir / "allocation_gini_vs_observed_position.png",
    )
    plot_policy_rollout_outcomes_over_time(
        policy_rollout_step_rows,
        output_dir / "policy_rollout_outcomes_over_time.png",
    )
    plot_policy_rollout_behavior_over_time(
        policy_rollout_step_rows,
        output_dir / "policy_rollout_behavior_over_time.png",
    )
    plot_mechanism_over_time(step_mechanism_rows, output_dir / "mechanism_over_time.png")
    plot_sender_state_mechanism(sender_state_rows, output_dir / "sender_state_mechanism.png")
    plot_receiver_state_income(receiver_state_rows, output_dir / "receiver_state_income.png")
    plot_counterfactual_summary(counterfactual_summary_rows, output_dir / "counterfactual_contribution_summary.png")
    write_episode_summary_csv(output_dir / "episode_summary.csv", episode_summaries)
    write_csv(output_dir / "row_mechanism.csv", row_mechanism_records)
    write_csv(output_dir / "position_bins.csv", position_bin_rows)
    write_csv(output_dir / "position_fixed_bins.csv", position_fixed_bin_rows)
    write_csv(output_dir / "allocation_gini_observed_position_bins.csv", allocation_gini_position_rows)
    write_csv(output_dir / "baseline_allocation_gini_records.csv", baseline_allocation_gini_records)
    write_csv(output_dir / "baseline_episode_summary.csv", baseline_episode_summary_rows)
    write_csv(output_dir / "policy_rollout_step_metrics.csv", policy_rollout_step_rows)
    write_csv(output_dir / "step_mechanism.csv", step_mechanism_rows)
    write_csv(output_dir / "sender_state_mechanism.csv", sender_state_rows)
    write_csv(output_dir / "node_income_decomposition.csv", node_income_records)
    write_csv(output_dir / "receiver_state_income.csv", receiver_state_rows)
    write_csv(output_dir / "pool_intervention_raw_rows.csv", pool_intervention_rows)
    write_csv(output_dir / "pool_intervention_summary.csv", pool_intervention_summary_rows)
    write_csv(output_dir / "resource_intervention_raw_rows.csv", resource_intervention_rows)
    write_csv(output_dir / "resource_intervention_summary.csv", resource_intervention_summary_rows)
    write_csv(output_dir / "counterfactual_contribution_response.csv", counterfactual_response_rows)
    write_csv(output_dir / "counterfactual_contribution_summary.csv", counterfactual_summary_rows)

    summary = build_mechanism_summary(
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_payload=checkpoint_payload,
        topology_name=topology_name,
        network_config=effective_spec["network"],
        dynamics_config=effective_spec["dynamics"],
        effective_episode_length=int(effective_spec["dynamics"]["episode_length"]),
        requested_episode_length_override=episode_length_override,
        graph_stats=graph_stats,
        policy_behavior=policy_behavior,
        mechanism_summary=mechanism_summary,
        node_income_summary=node_income_summary,
        labor_equal_gap_threshold=float(labor_equal_gap_threshold),
        pool_intervention_summary_rows=pool_intervention_summary_rows,
        pool_intervention_flat_summary=pool_intervention_flat_summary,
        pool_intervention_config={
            "enabled": bool(enable_pool_intervention_analysis),
            "reference_mode": str(pool_intervention_reference_mode),
            "multipliers": [float(value) for value in pool_intervention_multipliers],
            "row_sample_size": int(pool_intervention_row_sample_size),
            "batch_size": int(pool_intervention_batch_size),
            "agent_equal_step_start": (
                None
                if agent_equal_pool_intervention_step_start is None
                else int(agent_equal_pool_intervention_step_start)
            ),
            "agent_equal_step_end": (
                None if agent_equal_pool_intervention_step_end is None else int(agent_equal_pool_intervention_step_end)
            ),
            "proportional_step_start": (
                None
                if proportional_pool_intervention_step_start is None
                else int(proportional_pool_intervention_step_start)
            ),
            "proportional_step_end": (
                None
                if proportional_pool_intervention_step_end is None
                else int(proportional_pool_intervention_step_end)
            ),
        },
        resource_intervention_summary_rows=resource_intervention_summary_rows,
        resource_intervention_flat_summary=resource_intervention_flat_summary,
        resource_intervention_config={
            "enabled": bool(enable_resource_intervention_analysis),
            "reference_mode": str(resource_intervention_reference_mode),
            "multipliers": [float(value) for value in resource_intervention_multipliers],
            "row_sample_size": int(resource_intervention_row_sample_size),
            "batch_size": int(resource_intervention_batch_size),
            "agent_equal_step_start": (
                None
                if agent_equal_resource_intervention_step_start is None
                else int(agent_equal_resource_intervention_step_start)
            ),
            "agent_equal_step_end": (
                None
                if agent_equal_resource_intervention_step_end is None
                else int(agent_equal_resource_intervention_step_end)
            ),
            "proportional_step_start": (
                None
                if proportional_resource_intervention_step_start is None
                else int(proportional_resource_intervention_step_start)
            ),
            "proportional_step_end": (
                None
                if proportional_resource_intervention_step_end is None
                else int(proportional_resource_intervention_step_end)
            ),
            "resource_norm_reference": float(env.resource_norm_reference),
        },
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
            pool_intervention_flat_summary=pool_intervention_flat_summary,
            resource_intervention_flat_summary=resource_intervention_flat_summary,
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
    pool_intervention_group = parser.add_mutually_exclusive_group()
    pool_intervention_group.add_argument(
        "--enable-pool-intervention-analysis",
        dest="enable_pool_intervention_analysis",
        action="store_true",
        help="Run the manipulated-pool intervention analysis used for allocation_gini_vs_position.png.",
    )
    pool_intervention_group.add_argument(
        "--disable-pool-intervention-analysis",
        dest="enable_pool_intervention_analysis",
        action="store_false",
        help="Skip the manipulated-pool intervention analysis.",
    )
    parser.set_defaults(enable_pool_intervention_analysis=bool(SCRIPT_DEFAULTS["enable_pool_intervention_analysis"]))
    parser.add_argument(
        "--pool-intervention-reference-mode",
        type=str,
        default=str(SCRIPT_DEFAULTS["pool_intervention_reference_mode"]),
        choices=("original", "upperbound"),
        help="Reference pool used on the intervention x-axis.",
    )
    parser.add_argument(
        "--pool-intervention-multipliers",
        type=str,
        default=str(SCRIPT_DEFAULTS["pool_intervention_multipliers"]),
        help="Comma-separated manipulated-pool multipliers such as 0.1,0.2,...,6.0.",
    )
    parser.add_argument(
        "--pool-intervention-row-sample-size",
        type=int,
        default=int(SCRIPT_DEFAULTS["pool_intervention_row_sample_size"]),
        help="Maximum number of sender rows used for the pool intervention analysis. <=0 means use all rows.",
    )
    parser.add_argument(
        "--pool-intervention-batch-size",
        type=int,
        default=int(SCRIPT_DEFAULTS["pool_intervention_batch_size"]),
        help="Batch size for batched actor forward passes in the pool intervention analysis.",
    )
    parser.add_argument(
        "--agent-equal-pool-intervention-step-start",
        type=int,
        default=SCRIPT_DEFAULTS["agent_equal_pool_intervention_step_start"],
        help="Inclusive lower step bound for sampled RL-Agent / Equal intervention rows within each episode.",
    )
    parser.add_argument(
        "--agent-equal-pool-intervention-step-end",
        type=int,
        default=SCRIPT_DEFAULTS["agent_equal_pool_intervention_step_end"],
        help="Inclusive upper step bound for sampled RL-Agent / Equal intervention rows within each episode.",
    )
    parser.add_argument(
        "--proportional-pool-intervention-step-start",
        type=int,
        default=SCRIPT_DEFAULTS["proportional_pool_intervention_step_start"],
        help="Inclusive lower step bound for sampled Proportional intervention rows within each episode.",
    )
    parser.add_argument(
        "--proportional-pool-intervention-step-end",
        type=int,
        default=SCRIPT_DEFAULTS["proportional_pool_intervention_step_end"],
        help="Inclusive upper step bound for sampled Proportional intervention rows within each episode.",
    )
    resource_intervention_group = parser.add_mutually_exclusive_group()
    resource_intervention_group.add_argument(
        "--enable-resource-intervention-analysis",
        dest="enable_resource_intervention_analysis",
        action="store_true",
        help="Run the manipulated-resource intervention analysis.",
    )
    resource_intervention_group.add_argument(
        "--disable-resource-intervention-analysis",
        dest="enable_resource_intervention_analysis",
        action="store_false",
        help="Skip the manipulated-resource intervention analysis.",
    )
    parser.set_defaults(enable_resource_intervention_analysis=bool(SCRIPT_DEFAULTS["enable_resource_intervention_analysis"]))
    parser.add_argument(
        "--resource-intervention-reference-mode",
        type=str,
        default=str(SCRIPT_DEFAULTS["resource_intervention_reference_mode"]),
        choices=("original", "norm_reference"),
        help="Reference resource used on the resource-intervention x-axis.",
    )
    parser.add_argument(
        "--resource-intervention-multipliers",
        type=str,
        default=str(SCRIPT_DEFAULTS["resource_intervention_multipliers"]),
        help="Comma-separated manipulated-resource multipliers such as 0.1,0.2,...,6.0.",
    )
    parser.add_argument(
        "--resource-intervention-row-sample-size",
        type=int,
        default=int(SCRIPT_DEFAULTS["resource_intervention_row_sample_size"]),
        help="Maximum number of sender rows used for the resource intervention analysis. <=0 means use all rows.",
    )
    parser.add_argument(
        "--resource-intervention-batch-size",
        type=int,
        default=int(SCRIPT_DEFAULTS["resource_intervention_batch_size"]),
        help="Batch size for batched actor forward passes in the resource intervention analysis.",
    )
    parser.add_argument(
        "--agent-equal-resource-intervention-step-start",
        type=int,
        default=SCRIPT_DEFAULTS["agent_equal_resource_intervention_step_start"],
        help="Inclusive lower step bound for sampled RL-Agent / Equal resource-intervention rows within each episode.",
    )
    parser.add_argument(
        "--agent-equal-resource-intervention-step-end",
        type=int,
        default=SCRIPT_DEFAULTS["agent_equal_resource_intervention_step_end"],
        help="Inclusive upper step bound for sampled RL-Agent / Equal resource-intervention rows within each episode.",
    )
    parser.add_argument(
        "--proportional-resource-intervention-step-start",
        type=int,
        default=SCRIPT_DEFAULTS["proportional_resource_intervention_step_start"],
        help="Inclusive lower step bound for sampled Proportional resource-intervention rows within each episode.",
    )
    parser.add_argument(
        "--proportional-resource-intervention-step-end",
        type=int,
        default=SCRIPT_DEFAULTS["proportional_resource_intervention_step_end"],
        help="Inclusive upper step bound for sampled Proportional resource-intervention rows within each episode.",
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
        "--baseline-gini-episodes",
        type=int,
        default=int(SCRIPT_DEFAULTS["baseline_gini_episodes"]),
        help="Number of independent equal/proportional baseline rollout episodes used for observed-position plots and policy-over-time comparisons.",
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
        help="Base environment reset seed for analysis rollouts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(SCRIPT_DEFAULTS["device"]),
        help="Torch device, e.g. cpu or cuda:0.",
    )
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _to_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)


def _to_float(value: Any, *, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _apply_episode_offset(rows: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    if int(offset) == 0:
        return rows
    for row in rows:
        if "episode" not in row:
            continue
        raw_value = row.get("episode")
        if raw_value is None:
            continue
        text = str(raw_value).strip()
        if not text:
            continue
        row["episode"] = _to_int(text) + int(offset)
    return rows


def _combine_policy_rollout_step_metrics(step_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not step_rows:
        return []

    metric_keys = [
        "reward_mean",
        "cooperation_rate_mean",
        "resource_gini_mean",
        "mean_resource_mean",
        "mean_pool_grown_mean",
        "mean_self_allocation_mean",
        "offer_gini_mean",
    ]

    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in step_rows:
        policy = str(row.get("policy", ""))
        step = _to_int(row.get("step", 0))
        count = max(0, _to_int(row.get("count_episodes", 0)))

        bucket = grouped.setdefault(
            (policy, step),
            {
                "policy": policy,
                "step": step,
                "count_episodes": 0,
                "_sum": {key: 0.0 for key in metric_keys},
                "_weight": {key: 0 for key in metric_keys},
            },
        )
        bucket["count_episodes"] += count
        if count <= 0:
            continue
        for key in metric_keys:
            value = _to_float(row.get(key, float("nan")))
            if not np.isfinite(value):
                continue
            bucket["_sum"][key] += value * float(count)
            bucket["_weight"][key] += int(count)

    policy_order = {"equal": 0, "proportional": 1, "agent": 2}
    combined_rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        output: dict[str, Any] = {
            "policy": str(bucket["policy"]),
            "step": int(bucket["step"]),
            "count_episodes": int(bucket["count_episodes"]),
        }
        for key in metric_keys:
            weight = int(bucket["_weight"][key])
            output[key] = float(bucket["_sum"][key]) / float(weight) if weight > 0 else float("nan")
        combined_rows.append(output)

    combined_rows.sort(key=lambda item: (policy_order.get(str(item["policy"]), 99), int(item["step"])))
    return combined_rows


def _write_multi_topology_aggregate(
    *,
    root_output_dir: Path,
    topology_names: Sequence[str],
    combined_output_dir: Path,
    run_dir: Path,
    checkpoint_name: str,
    checkpoint_payload: Mapping[str, Any],
    mechanism_bin_count: int,
    labor_equal_gap_threshold: float,
    pool_intervention_reference_mode: str,
    resource_intervention_reference_mode: str,
) -> None:
    if len(topology_names) < 2:
        return

    combined_output_dir.mkdir(parents=True, exist_ok=True)

    per_topology_dirs = {str(name): (root_output_dir / str(name)) for name in topology_names}

    episode_offsets: dict[str, int] = {}
    episode_counts: dict[str, int] = {}
    cursor = 0
    for name in topology_names:
        topology_name = str(name)
        topo_dir = per_topology_dirs[topology_name]
        episode_rows = _read_csv_rows(topo_dir / "episode_summary.csv")
        count = int(len(episode_rows))
        episode_offsets[topology_name] = int(cursor)
        episode_counts[topology_name] = int(count)
        cursor += count

    episode_summary_rows: list[dict[str, Any]] = []
    row_mechanism_records: list[dict[str, Any]] = []
    node_income_records: list[dict[str, Any]] = []
    baseline_allocation_gini_records: list[dict[str, Any]] = []
    baseline_episode_summary_rows: list[dict[str, Any]] = []
    policy_rollout_step_rows_raw: list[dict[str, Any]] = []
    pool_intervention_rows: list[dict[str, Any]] = []
    resource_intervention_rows: list[dict[str, Any]] = []
    counterfactual_response_rows: list[dict[str, Any]] = []

    per_topology_graph_stats: dict[str, Any] = {}
    per_topology_policy_behavior: dict[str, Any] = {}
    first_summary: dict[str, Any] | None = None

    for name in topology_names:
        topology_name = str(name)
        topo_dir = per_topology_dirs[topology_name]
        offset = int(episode_offsets.get(topology_name, 0))

        episode_summary_rows.extend(
            _apply_episode_offset(_read_csv_rows(topo_dir / "episode_summary.csv"), offset)
        )
        row_mechanism_records.extend(
            _apply_episode_offset(_read_csv_rows(topo_dir / "row_mechanism.csv"), offset)
        )
        node_income_records.extend(
            _apply_episode_offset(_read_csv_rows(topo_dir / "node_income_decomposition.csv"), offset)
        )
        baseline_allocation_gini_records.extend(
            _apply_episode_offset(_read_csv_rows(topo_dir / "baseline_allocation_gini_records.csv"), offset)
        )
        baseline_episode_summary_rows.extend(
            _apply_episode_offset(_read_csv_rows(topo_dir / "baseline_episode_summary.csv"), offset)
        )
        policy_rollout_step_rows_raw.extend(_read_csv_rows(topo_dir / "policy_rollout_step_metrics.csv"))

        pool_intervention_rows.extend(
            _apply_episode_offset(_read_csv_rows(topo_dir / "pool_intervention_raw_rows.csv"), offset)
        )
        resource_intervention_rows.extend(
            _apply_episode_offset(_read_csv_rows(topo_dir / "resource_intervention_raw_rows.csv"), offset)
        )
        counterfactual_response_rows.extend(
            _apply_episode_offset(_read_csv_rows(topo_dir / "counterfactual_contribution_response.csv"), offset)
        )

        summary_path = topo_dir / "mechanism_summary.json"
        if summary_path.exists():
            payload = _load_json(summary_path)
            if isinstance(payload, dict):
                if first_summary is None:
                    first_summary = payload
                per_topology_graph_stats[topology_name] = payload.get("graph_stats", {})
                per_topology_policy_behavior[topology_name] = payload.get("policy_behavior", {})

    if not row_mechanism_records:
        return

    mechanism_summary, position_bin_rows = summarize_row_mechanism_records(
        row_mechanism_records,
        bin_count=max(2, int(mechanism_bin_count)),
    )
    position_fixed_bin_rows = summarize_position_fixed_bins(
        row_mechanism_records,
        bin_count=max(2, int(mechanism_bin_count)),
    )

    agent_allocation_gini_records = [
        {
            "policy": "agent",
            "episode": _to_int(row.get("episode", 0)),
            "step": _to_int(row.get("step", 0)),
            "sender": _to_int(row.get("sender", 0)),
            "sender_degree": _to_int(row.get("sender_degree", 0)),
            "receiver_count": _to_int(row.get("receiver_count", 0)),
            "sender_pool_grown": _to_float(row.get("sender_pool_grown", float("nan"))),
            "sender_pool_upperbound": _to_float(row.get("sender_pool_upperbound", float("nan"))),
            "pool_grown_over_upperbound": _to_float(row.get("pool_grown_over_upperbound", float("nan"))),
            "allocated_resource_gini": _to_float(row.get("agent_allocated_resource_gini", float("nan"))),
        }
        for row in row_mechanism_records
    ]

    proportional_baseline_records = [
        row
        for row in baseline_allocation_gini_records
        if str(row.get("policy", "")) == "proportional"
    ]
    equal_baseline_records = [
        row for row in baseline_allocation_gini_records if str(row.get("policy", "")) == "equal"
    ]
    allocation_gini_position_rows = summarize_allocation_gini_rows(
        {
            "agent": agent_allocation_gini_records,
            "proportional": proportional_baseline_records,
            "equal": equal_baseline_records,
        },
        bin_count=max(2, int(mechanism_bin_count)),
    )

    policy_rollout_step_rows = _combine_policy_rollout_step_metrics(policy_rollout_step_rows_raw)
    step_mechanism_rows = summarize_step_mechanism_records(row_mechanism_records)
    sender_state_rows = summarize_sender_state_mechanism(row_mechanism_records)
    node_income_summary = summarize_node_income_records(node_income_records)
    receiver_state_rows = summarize_receiver_state_income(node_income_records)

    pool_intervention_summary_rows, pool_intervention_flat_summary = summarize_pool_intervention_records(
        pool_intervention_rows
    )
    resource_intervention_summary_rows, resource_intervention_flat_summary = summarize_resource_intervention_records(
        resource_intervention_rows
    )
    counterfactual_summary_rows, counterfactual_flat_summary = summarize_counterfactual_response_rows(
        counterfactual_response_rows
    )

    plot_position_mechanism_bins(position_bin_rows, combined_output_dir / "position_vs_mechanism.png")
    plot_position_mechanism_bins(position_fixed_bin_rows, combined_output_dir / "position_fixed_bins.png")
    plot_position_identifiability(position_fixed_bin_rows, combined_output_dir / "position_identifiability.png")
    plot_position_distribution(row_mechanism_records, combined_output_dir / "position_distribution.png")
    plot_allocation_gini_vs_position(
        pool_intervention_summary_rows,
        combined_output_dir / "allocation_gini_vs_position.png",
        reference_mode=str(pool_intervention_reference_mode),
    )
    plot_allocation_gini_vs_resource_intervention(
        resource_intervention_summary_rows,
        combined_output_dir / "allocation_gini_vs_resource_intervention.png",
        reference_mode=str(resource_intervention_reference_mode),
    )
    plot_allocation_gini_vs_observed_position(
        allocation_gini_position_rows,
        combined_output_dir / "allocation_gini_vs_observed_position.png",
    )
    plot_policy_rollout_outcomes_over_time(
        policy_rollout_step_rows,
        combined_output_dir / "policy_rollout_outcomes_over_time.png",
    )
    plot_policy_rollout_behavior_over_time(
        policy_rollout_step_rows,
        combined_output_dir / "policy_rollout_behavior_over_time.png",
    )
    plot_mechanism_over_time(step_mechanism_rows, combined_output_dir / "mechanism_over_time.png")
    plot_sender_state_mechanism(sender_state_rows, combined_output_dir / "sender_state_mechanism.png")
    plot_receiver_state_income(receiver_state_rows, combined_output_dir / "receiver_state_income.png")
    plot_counterfactual_summary(
        counterfactual_summary_rows,
        combined_output_dir / "counterfactual_contribution_summary.png",
    )

    write_csv(combined_output_dir / "episode_summary.csv", episode_summary_rows)
    write_csv(combined_output_dir / "row_mechanism.csv", row_mechanism_records)
    write_csv(combined_output_dir / "position_bins.csv", position_bin_rows)
    write_csv(combined_output_dir / "position_fixed_bins.csv", position_fixed_bin_rows)
    write_csv(combined_output_dir / "allocation_gini_observed_position_bins.csv", allocation_gini_position_rows)
    write_csv(combined_output_dir / "baseline_allocation_gini_records.csv", baseline_allocation_gini_records)
    write_csv(combined_output_dir / "baseline_episode_summary.csv", baseline_episode_summary_rows)
    write_csv(combined_output_dir / "policy_rollout_step_metrics.csv", policy_rollout_step_rows)
    write_csv(combined_output_dir / "step_mechanism.csv", step_mechanism_rows)
    write_csv(combined_output_dir / "sender_state_mechanism.csv", sender_state_rows)
    write_csv(combined_output_dir / "node_income_decomposition.csv", node_income_records)
    write_csv(combined_output_dir / "receiver_state_income.csv", receiver_state_rows)
    write_csv(combined_output_dir / "pool_intervention_raw_rows.csv", pool_intervention_rows)
    write_csv(combined_output_dir / "pool_intervention_summary.csv", pool_intervention_summary_rows)
    write_csv(combined_output_dir / "resource_intervention_raw_rows.csv", resource_intervention_rows)
    write_csv(combined_output_dir / "resource_intervention_summary.csv", resource_intervention_summary_rows)
    write_csv(combined_output_dir / "counterfactual_contribution_response.csv", counterfactual_response_rows)
    write_csv(combined_output_dir / "counterfactual_contribution_summary.csv", counterfactual_summary_rows)

    episode_summaries: list[EpisodeSummary] = []
    for row in episode_summary_rows:
        try:
            episode_summaries.append(
                EpisodeSummary(
                    episode=_to_int(row.get("episode", 0)),
                    steps=_to_int(row.get("steps", 0)),
                    total_reward=_to_float(row.get("total_reward", float("nan"))),
                    mean_reward=_to_float(row.get("mean_reward", float("nan"))),
                    final_cooperation_rate=_to_float(row.get("final_cooperation_rate", float("nan"))),
                    final_gini=_to_float(row.get("final_gini", float("nan"))),
                    final_mean_resource=_to_float(row.get("final_mean_resource", float("nan"))),
                )
            )
        except TypeError:
            continue

    network_config: Mapping[str, Any] = {"combined_topologies": [str(name) for name in topology_names]}
    dynamics_config: Mapping[str, Any] = {}
    effective_episode_length = 0
    requested_episode_length_override: int | None = None
    pool_intervention_config: Mapping[str, Any] = {"enabled": False}
    resource_intervention_config: Mapping[str, Any] = {"enabled": False}
    counterfactual_config: Mapping[str, Any] = {"enabled": False}

    if first_summary is not None:
        if isinstance(first_summary.get("network_config"), Mapping):
            network_config = first_summary["network_config"]
        if isinstance(first_summary.get("dynamics_config"), Mapping):
            dynamics_config = first_summary["dynamics_config"]
        effective_episode_length = _to_int(first_summary.get("effective_episode_length", 0))
        requested_override_raw = first_summary.get("requested_episode_length_override")
        requested_episode_length_override = (
            None if requested_override_raw is None else _to_int(requested_override_raw)
        )
        pool_intervention_config = dict(first_summary.get("pool_intervention_analysis", {}).get("config", {}))
        resource_intervention_config = dict(first_summary.get("resource_intervention_analysis", {}).get("config", {}))
        counterfactual_config = dict(first_summary.get("counterfactual_analysis", {}).get("config", {}))

    if effective_episode_length <= 0 and isinstance(dynamics_config, Mapping):
        effective_episode_length = _to_int(dynamics_config.get("episode_length", 0))

    summary_payload = build_mechanism_summary(
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_payload=checkpoint_payload,
        topology_name="all_topologies",
        network_config=dict(network_config),
        dynamics_config=dict(dynamics_config),
        effective_episode_length=int(effective_episode_length),
        requested_episode_length_override=requested_episode_length_override,
        graph_stats={"per_topology": dict(per_topology_graph_stats)},
        policy_behavior={"per_topology": dict(per_topology_policy_behavior)},
        mechanism_summary=mechanism_summary,
        node_income_summary=node_income_summary,
        labor_equal_gap_threshold=float(labor_equal_gap_threshold),
        pool_intervention_summary_rows=pool_intervention_summary_rows,
        pool_intervention_flat_summary=pool_intervention_flat_summary,
        pool_intervention_config=dict(pool_intervention_config),
        resource_intervention_summary_rows=resource_intervention_summary_rows,
        resource_intervention_flat_summary=resource_intervention_flat_summary,
        resource_intervention_config=dict(resource_intervention_config),
        counterfactual_summary_rows=counterfactual_summary_rows,
        counterfactual_flat_summary=counterfactual_flat_summary,
        counterfactual_config=dict(counterfactual_config),
        episode_summaries=episode_summaries,
    )
    summary_payload["multi_topology"] = {
        "included_topologies": [str(name) for name in topology_names],
        "episode_offsets": dict(episode_offsets),
        "episode_counts": dict(episode_counts),
        "source_output_dir": str(root_output_dir),
    }
    with (combined_output_dir / "mechanism_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)


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
    pool_intervention_multipliers = _parse_float_list(args.pool_intervention_multipliers)
    resource_intervention_multipliers = _parse_float_list(args.resource_intervention_multipliers)
    if any(value <= 0.0 for value in pool_intervention_multipliers):
        raise ValueError("pool intervention multipliers must all be positive because the x-axis is logarithmic.")
    if any(value <= 0.0 for value in resource_intervention_multipliers):
        raise ValueError("resource intervention multipliers must all be positive because the x-axis is logarithmic.")
    if (
        args.agent_equal_pool_intervention_step_start is not None
        and args.agent_equal_pool_intervention_step_end is not None
        and int(args.agent_equal_pool_intervention_step_start) > int(args.agent_equal_pool_intervention_step_end)
    ):
        raise ValueError("agent-equal pool intervention step range is invalid: start must be <= end.")
    if (
        args.proportional_pool_intervention_step_start is not None
        and args.proportional_pool_intervention_step_end is not None
        and int(args.proportional_pool_intervention_step_start) > int(args.proportional_pool_intervention_step_end)
    ):
        raise ValueError("proportional pool intervention step range is invalid: start must be <= end.")
    if (
        args.agent_equal_resource_intervention_step_start is not None
        and args.agent_equal_resource_intervention_step_end is not None
        and int(args.agent_equal_resource_intervention_step_start) > int(args.agent_equal_resource_intervention_step_end)
    ):
        raise ValueError("agent-equal resource intervention step range is invalid: start must be <= end.")
    if (
        args.proportional_resource_intervention_step_start is not None
        and args.proportional_resource_intervention_step_end is not None
        and int(args.proportional_resource_intervention_step_start) > int(args.proportional_resource_intervention_step_end)
    ):
        raise ValueError("proportional resource intervention step range is invalid: start must be <= end.")

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
        topology_spec = apply_analysis_env_overrides(
            topology_spec,
            env_r_override=None if args.env_r_override is None else float(args.env_r_override),
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
            enable_pool_intervention_analysis=bool(args.enable_pool_intervention_analysis),
            pool_intervention_reference_mode=str(args.pool_intervention_reference_mode),
            pool_intervention_multipliers=pool_intervention_multipliers,
            pool_intervention_row_sample_size=int(args.pool_intervention_row_sample_size),
            pool_intervention_batch_size=max(1, int(args.pool_intervention_batch_size)),
            agent_equal_pool_intervention_step_start=args.agent_equal_pool_intervention_step_start,
            agent_equal_pool_intervention_step_end=args.agent_equal_pool_intervention_step_end,
            proportional_pool_intervention_step_start=args.proportional_pool_intervention_step_start,
            proportional_pool_intervention_step_end=args.proportional_pool_intervention_step_end,
            enable_resource_intervention_analysis=bool(args.enable_resource_intervention_analysis),
            resource_intervention_reference_mode=str(args.resource_intervention_reference_mode),
            resource_intervention_multipliers=resource_intervention_multipliers,
            resource_intervention_row_sample_size=int(args.resource_intervention_row_sample_size),
            resource_intervention_batch_size=max(1, int(args.resource_intervention_batch_size)),
            agent_equal_resource_intervention_step_start=args.agent_equal_resource_intervention_step_start,
            agent_equal_resource_intervention_step_end=args.agent_equal_resource_intervention_step_end,
            proportional_resource_intervention_step_start=args.proportional_resource_intervention_step_start,
            proportional_resource_intervention_step_end=args.proportional_resource_intervention_step_end,
            enable_counterfactual_analysis=bool(args.enable_counterfactual_analysis),
            counterfactual_row_sample_size=max(1, int(args.counterfactual_row_sample_size)),
            counterfactual_contribution_delta=max(0.0, float(args.counterfactual_contribution_delta)),
            counterfactual_batch_size=max(1, int(args.counterfactual_batch_size)),
            baseline_gini_episodes=max(0, int(args.baseline_gini_episodes)),
            rollout_seed=int(args.seed) + topology_index,
        )
        topology_rows.append(result["topology_row"])

    if len(topologies) > 1:
        _write_multi_topology_aggregate(
            root_output_dir=output_dir,
            topology_names=topologies,
            combined_output_dir=output_dir / "_all_topologies",
            run_dir=run_dir,
            checkpoint_name=args.checkpoint_name,
            checkpoint_payload=checkpoint_payload,
            mechanism_bin_count=max(2, int(args.mechanism_bin_count)),
            labor_equal_gap_threshold=max(0.0, float(args.labor_equal_gap_threshold)),
            pool_intervention_reference_mode=str(args.pool_intervention_reference_mode),
            resource_intervention_reference_mode=str(args.resource_intervention_reference_mode),
        )

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
            f"poolIntΔ={float(row.get('pool_intervention_agent_gini_high_minus_low', float('nan'))):.4f}",
            f"cf_delta={float(row.get('counterfactual_target_allocation_delta_mean', float('nan'))):.4f}",
            f"P/Pupper->lambda={float(row['pool_grown_over_upperbound_vs_lambda_spearman']):.4f}",
        ]
        print("  " + " ".join(parts))


if __name__ == "__main__":
    main()
