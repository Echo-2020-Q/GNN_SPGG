from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig, PolicyOutput


NODE_FEATURE_KEYS = (
    "pool_raw_norm",
    "resource_norm",
    "degree_norm",
    "strategy_norm",
)
ANALYSIS_FEATURE_KEYS = (
    "pool_raw_norm",
    "resource_norm",
    "degree_norm",
    "strategy_norm",
    "x_actual",
    "gini",
    "pool_grown",
)
FEATURE_GROUPS = {
    "pool_raw_norm": "node+global_mean",
    "resource_norm": "node+global_mean",
    "degree_norm": "node",
    "strategy_norm": "node",
    "x_actual": "global_mean",
    "gini": "global_scalar",
    "pool_grown": "local_pool_scalar",
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

# Edit these defaults directly if you prefer not to pass many CLI arguments.
SCRIPT_DEFAULTS = {
    # 训练结果目录。
    # 这个目录下面至少要有：
    # 1. results.json，用来恢复实验配置；
    # 2. checkpoints/，里面放训练好的模型权重。
    # 一般填某次具体实验的输出目录，而不是 outputs/Pool_dynamic 这种总目录。
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",

    # 要加载的 checkpoint 文件名，脚本会去 <run_dir>/checkpoints/ 下面找它。
    # 常见可选值：
    # - best_eval.pt：通常优先推荐，表示验证表现最好的模型；
    # - final.pt：训练结束时最后一次保存的模型；
    # - latest.pt：最近一次保存的模型；
    # - update_00xxxx.pt：某个中间训练阶段的模型，适合看策略演化。
    "checkpoint_name": "best_eval.pt",

    # 分析结果输出目录。
    # 设为 None 时，会自动输出到：
    # <run_dir>/policy_analysis/<checkpoint_name去掉后缀>/
    # 如果你想把不同分析结果分开放，改成一个显式路径字符串即可。
    "output_dir": "D:\\PyCharm_Community_Edition_2024_01_04\\Py_Projects\\GNN_SPGG\\outputs\\Pool_dynamic\\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher\\0416_policy_analysis_200length",

    # 要评估的图拓扑类型，多个值用英文逗号分隔。
    # 支持：
    # - original：严格使用该实验训练时 results.json 里记录的原始网络类型；
    # - regular：随机正则图；
    # - er：Erdos-Renyi 随机图；
    # - ws：Watts-Strogatz small-world 图；
    # - ba：Barabasi-Albert scale-free 图。
    # 当前这个默认值表示：固定模型不变，分别放到 Regular / ER / WS / BA 四种图上做迁移测试。
    "topologies": "regular,ba",

    # 每种拓扑下跑多少个 deterministic episode。
    # 越大，统计越稳定，但运行越慢、decision_records.csv 也会更大。
    # 经验上：
    # - 1：快速 smoke test；
    # - 3~10：初步分析；
    # - 10+：更稳的跨拓扑对比。
    "episodes": 5,

    # 每个 episode 最多分析多少步。
    # 设为 None 表示跑完整个 episode_length。
    # 如果你只是想快速看脚本能不能跑通，或者先做小样本预览，可以改成 5、10、20 之类的小值。
    "max_steps": 200,

    # 是否覆盖分析环境本身的 episode_length。
    # 设为 None：沿用训练时 results.json 里的 episode_length。
    # 设为整数，例如 500：分析环境最多就会跑 500 步。
    # 注意区分：
    # - episode_length_override 控制环境“最多能跑多长”；
    # - max_steps 控制这次分析“最多截到多少步”。
    # 如果你想比较 200 步训练模型在 200 步长期演化下的行为，这里就应该设为 200。
    "episode_length_override": 200,

    # 每种拓扑导出多少个“时刻快照”用于画图。
    # 脚本会从整个 rollout 过程里均匀抽取这些时间点。
    # 注意：只有当前解释器装了 matplotlib 才会真的输出 PNG 图。
    "snapshot_count": 4,

    # 是否导出 rollout 快照图和 snapshot_index.csv。
    # True：按 snapshot_count 均匀抽样，输出若干时刻的图结构快照；
    # False：完全跳过这部分，适合你只关心表格统计、不关心示意图的时候。
    "enable_snapshot_plots": True,

    # 是否执行“特征扰动敏感度分析”。
    # True：会逐个特征通道做扰动，输出 feature_importance.csv / feature_importance.png；
    # False：跳过这部分，可显著减少一次分析的离线推理量。
    # 如果你当前更关心“分配机制”而不是“输入特征重要性”，可以先关掉。
    "enable_feature_perturbation": True,

    # 做“特征通道影响分析”时，对单个特征采用什么扰动方式。
    # 支持：
    # - zero：把该特征整通道清零，最直接；
    # - mean：把该特征替换成该状态下的平均值，适合看“去掉差异性”后的影响；
    # - shuffle：只打乱该特征在节点之间的对应关系，保留总体分布但破坏节点语义。
    # 一般首选 zero，因为最容易解释。
    "perturbation_mode": "zero",

    # 扰动分析用到的随机种子。
    # 主要影响 shuffle 模式；在 zero / mean 模式下基本不会改变结果。
    # 固定它有利于复现实验。
    "perturbation_seed": 666,

    # 离线推理时的 batch 大小，主要影响“特征扰动分析”的速度和内存占用。
    # 更大：通常更快，但更吃内存；
    # 更小：更稳、更省内存，但会慢一些。
    # 如果你以后改用 GPU，可以适当调大；CPU 上 32~128 通常都比较稳。
    "batch_size":128,

    # 是否执行“资源宽裕度 vs 分配机制”分析。
    # True：会把每一行分配拟合成 labor / equal / self 三种机制的组合，
    #       输出 row_mechanism.csv、scarcity_bins.csv、node_income_decomposition.csv 等。
    # False：跳过这部分，适合你只看拓扑表现或特征扰动时使用。
    "enable_mechanism_analysis": False,

    # 做“P_grown / P_upperbound vs 分配机制”分析时，按多少个分位数区间做分箱统计。
    # 更小：更平滑、更稳；
    # 更大：更细，但每个 bin 的样本数会变少。
    # 经验上 8~12 比较合适。
    "mechanism_bin_count": 10,

    # 如果 topologies 里写了多个拓扑，是否额外导出“跨拓扑总表/总图”。
    # True：会写 topology_comparison.csv/json/png 以及 topology_feature_importance.csv；
    # False：仍然会逐个拓扑分别跑，但不会再额外生成跨拓扑汇总结果。
    # 适合你只是想批量跑多个拓扑、事后自己再单独处理数据的情况。
    "enable_topology_comparison": True,

    # rollout 的环境随机种子基准值。
    # 第 0 个 episode 用这个种子，第 1 个 episode 用 seed+1，以此类推。
    # 改它可以换一批初始状态，从而检查结论是否只依赖某一组随机初始化。
    "seed": 42,

    # Torch 推理设备。
    # 常见写法：
    # - "cpu"：最稳，兼容性最好；
    # - "cuda:0"：如果你的环境里 GPU 可用，可以显著加速推理。
    # 如果设成 GPU，但当前环境没有对应 CUDA，会直接报错。
    "device": "cuda:0",
}


@dataclass(frozen=True)
class PolicyArrays:
    allocation_matrix: np.ndarray
    transferred_resources: np.ndarray
    incoming_resources: np.ndarray
    logits: np.ndarray
    value: float


@dataclass(frozen=True)
class Snapshot:
    episode: int
    step: int
    observation: dict[str, np.ndarray]
    policy: PolicyArrays
    reward: float
    actual_cooperation_rate: float
    gini: float


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    steps: int
    total_reward: float
    mean_reward: float
    final_cooperation_rate: float
    final_gini: float
    final_mean_resource: float


def _safe_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum())
    if total <= eps:
        if array.size == 0:
            return array.copy()
        return np.full(array.shape, 1.0 / float(array.size), dtype=np.float64)
    return array / total


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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint at {path} is not a dictionary payload.")
    return payload


def _observation_copy(observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in observation.items()}


def _policy_output_to_arrays(output: PolicyOutput) -> PolicyArrays:
    logits = output.logits
    if logits is None:
        raise RuntimeError("Policy output does not contain logits.")
    return PolicyArrays(
        allocation_matrix=output.allocation_matrix.detach().cpu().numpy(),
        transferred_resources=output.transferred_resources.detach().cpu().numpy(),
        incoming_resources=output.incoming_resources.detach().cpu().numpy(),
        logits=logits.detach().cpu().numpy(),
        value=float(output.value.detach().cpu().item()),
    )


def _resolve_er_edge_prob(network: Mapping[str, Any]) -> float:
    explicit_prob = network.get("er_edge_prob")
    if explicit_prob is not None:
        return float(explicit_prob)
    num_nodes = int(network["num_nodes"])
    target_mean_degree = float(network.get("er_target_mean_degree", 0.0))
    if num_nodes <= 1:
        return 0.0
    return float(target_mean_degree / float(num_nodes - 1))


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


def _parse_toggle_arg(raw_value: str) -> bool:
    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected on/off or true/false, got: {raw_value}")


def _toggle_label(value: bool) -> str:
    return "on" if bool(value) else "off"


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
        network["er_target_mean_degree"] = float(
            network.get("er_target_mean_degree", target_mean_degree)
        )
        network["er_edge_prob"] = float(network["er_target_mean_degree"]) / float(max(num_nodes - 1, 1))
    elif topology == "small_world":
        network["ws_degree"] = int(
            network.get("ws_degree", _resolve_even_degree(target_mean_degree, num_nodes))
        )
        if int(network["ws_degree"]) % 2 != 0:
            network["ws_degree"] = _resolve_even_degree(float(network["ws_degree"]), num_nodes)
        network["ws_rewiring_prob"] = float(network.get("ws_rewiring_prob", 0.10))
    elif topology == "scale_free":
        attachments = int(network.get("ba_attachments_per_new_node", max(1, int(round(target_mean_degree / 2.0)))))
        network["ba_attachments_per_new_node"] = max(1, min(num_nodes - 1, attachments))
    else:
        raise ValueError(f"Unsupported topology name: {topology}")

    return spec


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


def load_actor_from_run_dir(run_dir: Path, checkpoint_name: str, device: torch.device) -> tuple[GNNAllocationPolicy, dict[str, Any]]:
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = _load_checkpoint(checkpoint_path)
    policy_payload = payload.get("policy_config")
    if not isinstance(policy_payload, dict):
        learner_state = payload.get("learner_state", {})
        policy_payload = learner_state.get("policy_config")
    if not isinstance(policy_payload, dict):
        raise KeyError("Checkpoint does not contain a policy_config dictionary.")

    learner_state = payload.get("learner_state")
    if not isinstance(learner_state, dict):
        raise KeyError("Checkpoint does not contain learner_state.")
    actor_state = learner_state.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise KeyError("Checkpoint learner_state does not contain actor_state_dict.")

    actor = GNNAllocationPolicy(GNNPolicyConfig(**policy_payload))
    actor.load_state_dict(actor_state)
    actor.to(device)
    actor.eval()
    return actor, payload


def run_deterministic_rollouts(
    actor: GNNAllocationPolicy,
    env: SPGGEnv,
    *,
    episodes: int,
    seed: int,
    max_steps: int | None,
) -> tuple[list[Snapshot], list[EpisodeSummary]]:
    snapshots: list[Snapshot] = []
    episode_summaries: list[EpisodeSummary] = []

    for episode_index in range(episodes):
        observation = env.reset(seed=seed + episode_index)
        done = False
        step_index = 0
        total_reward = 0.0
        last_info: dict[str, Any] | None = None

        while not done:
            if max_steps is not None and step_index >= max_steps:
                break

            with torch.no_grad():
                policy_output = actor.deterministic_action(observation)
            policy_arrays = _policy_output_to_arrays(policy_output)
            next_observation, reward, env_done, info = env.step(policy_arrays.allocation_matrix)

            snapshots.append(
                Snapshot(
                    episode=episode_index,
                    step=step_index,
                    observation=_observation_copy(observation),
                    policy=policy_arrays,
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
                mean_reward=(total_reward / float(max(step_index, 1))),
                final_cooperation_rate=final_cooperation_rate,
                final_gini=final_gini,
                final_mean_resource=float(np.mean(observation["resources"])),
            )
        )

    return snapshots, episode_summaries


def _chunked_policy_forward(
    actor: GNNAllocationPolicy,
    observations: Sequence[Mapping[str, np.ndarray]],
    batch_size: int,
) -> list[PolicyArrays]:
    outputs: list[PolicyArrays] = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            stop = min(start + batch_size, len(observations))
            batch_outputs = actor.deterministic_action_batch(observations[start:stop])
            outputs.extend(_policy_output_to_arrays(output) for output in batch_outputs)
    return outputs


def _perturb_observation(
    observation: Mapping[str, np.ndarray],
    feature_key: str,
    mode: str,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    perturbed = _observation_copy(observation)
    value = np.asarray(perturbed[feature_key])

    if mode == "zero":
        perturbed_value = np.zeros_like(value, dtype=np.float64)
    elif mode == "mean":
        mean_value = float(np.mean(value))
        perturbed_value = np.full_like(value, mean_value, dtype=np.float64)
    elif mode == "shuffle":
        flat = value.astype(np.float64, copy=True).reshape(-1)
        if flat.size > 1:
            rng.shuffle(flat)
        perturbed_value = flat.reshape(value.shape)
    else:
        raise ValueError(f"Unsupported perturbation mode: {mode}")

    perturbed[feature_key] = perturbed_value.astype(value.dtype if hasattr(value, "dtype") else np.float64, copy=False)
    return perturbed


def _entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities.astype(np.float64, copy=False), 1e-12, 1.0)
    return float(-(safe * np.log(safe)).sum())


def _js_divergence(prob_p: np.ndarray, prob_q: np.ndarray) -> float:
    p = np.clip(prob_p.astype(np.float64, copy=False), 1e-12, 1.0)
    q = np.clip(prob_q.astype(np.float64, copy=False), 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    mean = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * (np.log(p) - np.log(mean))) + 0.5 * np.sum(q * (np.log(q) - np.log(mean)))
    )


def _gini_coefficient(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size <= 1:
        return 0.0

    data = np.clip(data, 0.0, None)
    total = float(np.sum(data))
    if total <= 1e-12:
        return 0.0

    sorted_data = np.sort(data)
    n = int(sorted_data.size)
    indices = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * float(np.sum(indices * sorted_data)) / (float(n) * total)) - (float(n) + 1.0) / float(n)
    return float(np.clip(gini, 0.0, 1.0))


def _sender_row_ginis(allocation_matrix: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(local_mask, dtype=bool)
    allocation = np.asarray(allocation_matrix, dtype=np.float64)
    num_nodes = int(mask.shape[0])
    ginis = np.zeros(num_nodes, dtype=np.float64)
    for sender in range(num_nodes):
        valid = mask[sender]
        row = allocation[sender, valid]
        ginis[sender] = _gini_coefficient(row)
    return ginis


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
        raise RuntimeError("Sender must be present in its own valid receiver set.")
    self_row[int(self_matches[0])] = 1.0
    return labor_row, equal_row, self_row


def compute_feature_importance(
    actor: GNNAllocationPolicy,
    snapshots: Sequence[Snapshot],
    *,
    feature_keys: Sequence[str],
    perturbation_mode: str,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    observations = [snapshot.observation for snapshot in snapshots]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    base_sender_ginis = [
        _sender_row_ginis(
            snapshot.policy.allocation_matrix,
            snapshot.observation["local_mask"].astype(bool, copy=False),
        )
        for snapshot in snapshots
    ]
    base_sender_gini_means = [float(np.mean(ginis)) for ginis in base_sender_ginis]
    base_sender_gini_mean_overall = float(np.mean(base_sender_gini_means)) if base_sender_gini_means else 0.0

    for feature_key in feature_keys:
        perturbed_observations = [
            _perturb_observation(observation, feature_key, perturbation_mode, rng) for observation in observations
        ]
        perturbed_outputs = _chunked_policy_forward(actor, perturbed_observations, batch_size)

        allocation_l1_values: list[float] = []
        transfer_l1_values: list[float] = []
        logit_l1_values: list[float] = []
        self_delta_values: list[float] = []
        row_entropy_delta_values: list[float] = []
        row_js_values: list[float] = []
        top1_change_values: list[float] = []
        sender_row_gini_delta_values: list[float] = []
        sender_row_gini_cf_means: list[float] = []

        for snapshot_index, (snapshot, perturbed_output) in enumerate(zip(snapshots, perturbed_outputs)):
            mask = snapshot.observation["local_mask"].astype(bool, copy=False)
            base_policy = snapshot.policy

            allocation_l1_values.append(
                float(np.mean(np.abs(perturbed_output.allocation_matrix[mask] - base_policy.allocation_matrix[mask])))
            )
            transfer_l1_values.append(
                float(
                    np.mean(
                        np.abs(perturbed_output.transferred_resources[mask] - base_policy.transferred_resources[mask])
                    )
                )
            )
            logit_l1_values.append(float(np.mean(np.abs(perturbed_output.logits[mask] - base_policy.logits[mask]))))

            num_nodes = mask.shape[0]
            self_indices = np.arange(num_nodes)
            self_delta_values.append(
                float(
                    np.mean(
                        np.abs(
                            perturbed_output.allocation_matrix[self_indices, self_indices]
                            - base_policy.allocation_matrix[self_indices, self_indices]
                        )
                    )
                )
            )

            row_entropy_delta = 0.0
            row_js = 0.0
            row_top1_switch = 0.0
            for sender in range(num_nodes):
                valid_receivers = mask[sender]
                base_row = base_policy.allocation_matrix[sender, valid_receivers]
                perturbed_row = perturbed_output.allocation_matrix[sender, valid_receivers]
                row_entropy_delta += abs(_entropy(base_row) - _entropy(perturbed_row))
                row_js += _js_divergence(base_row, perturbed_row)
                row_top1_switch += float(int(np.argmax(base_row) != np.argmax(perturbed_row)))
            normalizer = float(max(num_nodes, 1))
            row_entropy_delta_values.append(row_entropy_delta / normalizer)
            row_js_values.append(row_js / normalizer)
            top1_change_values.append(row_top1_switch / normalizer)

            base_ginis = base_sender_ginis[snapshot_index]
            cf_ginis = _sender_row_ginis(perturbed_output.allocation_matrix, mask)
            sender_row_gini_delta_values.append(float(np.mean(np.abs(cf_ginis - base_ginis))))
            sender_row_gini_cf_means.append(float(np.mean(cf_ginis)))

        rows.append(
            {
                "feature": feature_key,
                "feature_group": FEATURE_GROUPS.get(feature_key, "unknown"),
                "perturbation_mode": perturbation_mode,
                "allocation_l1_mean": float(np.mean(allocation_l1_values)) if allocation_l1_values else 0.0,
                "transfer_l1_mean": float(np.mean(transfer_l1_values)) if transfer_l1_values else 0.0,
                "logit_l1_mean": float(np.mean(logit_l1_values)) if logit_l1_values else 0.0,
                "self_allocation_delta_mean": float(np.mean(self_delta_values)) if self_delta_values else 0.0,
                "row_entropy_delta_mean": float(np.mean(row_entropy_delta_values)) if row_entropy_delta_values else 0.0,
                "row_js_divergence_mean": float(np.mean(row_js_values)) if row_js_values else 0.0,
                "row_top1_change_rate": float(np.mean(top1_change_values)) if top1_change_values else 0.0,
                "sender_row_gini_base_mean": base_sender_gini_mean_overall,
                "sender_row_gini_counterfactual_mean": (
                    float(np.mean(sender_row_gini_cf_means)) if sender_row_gini_cf_means else 0.0
                ),
                "sender_row_gini_delta_mean": float(np.mean(sender_row_gini_delta_values)) if sender_row_gini_delta_values else 0.0,
            }
        )

    rows.sort(key=lambda item: item["allocation_l1_mean"], reverse=True)
    return rows


def compute_feature_channel_stats(snapshots: Sequence[Snapshot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_key in ANALYSIS_FEATURE_KEYS:
        per_state_std: list[float] = []
        values: list[np.ndarray] = []
        for snapshot in snapshots:
            raw_value = np.asarray(snapshot.observation[feature_key], dtype=np.float64)
            flat_value = raw_value.reshape(-1)
            values.append(flat_value)
            per_state_std.append(float(np.std(flat_value)) if flat_value.size > 1 else 0.0)

        concatenated = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
        rows.append(
            {
                "feature": feature_key,
                "feature_group": FEATURE_GROUPS.get(feature_key, "unknown"),
                "global_mean": float(np.mean(concatenated)) if concatenated.size else 0.0,
                "global_std": float(np.std(concatenated)) if concatenated.size else 0.0,
                "global_min": float(np.min(concatenated)) if concatenated.size else 0.0,
                "global_max": float(np.max(concatenated)) if concatenated.size else 0.0,
                "within_state_std_mean": float(np.mean(per_state_std)) if per_state_std else 0.0,
            }
        )
    return rows


def collect_graph_stats(graph: Mapping[int, Sequence[int]]) -> dict[str, Any]:
    degrees = np.asarray([len(neighbors) for neighbors in graph.values()], dtype=np.float64)
    edge_count = int(sum(int(len(neighbors)) for neighbors in graph.values()) // 2)
    return {
        "num_nodes": int(len(graph)),
        "edge_count": edge_count,
        "mean_degree": float(np.mean(degrees)) if degrees.size else 0.0,
        "min_degree": int(np.min(degrees)) if degrees.size else 0,
        "max_degree": int(np.max(degrees)) if degrees.size else 0,
        "degree_std": float(np.std(degrees)) if degrees.size else 0.0,
    }


def compute_policy_behavior_stats(snapshots: Sequence[Snapshot]) -> dict[str, Any]:
    self_allocations: list[float] = []
    row_entropies: list[float] = []
    top1_self_flags: list[float] = []
    top1_masses: list[float] = []
    incoming_stds: list[float] = []
    neighborhood_sizes: list[float] = []
    cooperation_values: list[float] = []

    for snapshot in snapshots:
        observation = snapshot.observation
        policy = snapshot.policy
        local_mask = observation["local_mask"].astype(bool, copy=False)
        num_nodes = local_mask.shape[0]

        neighborhood_sizes.extend(local_mask.sum(axis=1).astype(np.float64).tolist())
        incoming_stds.append(float(np.std(policy.incoming_resources)))
        cooperation_values.append(float(np.mean(observation["x_actual"])))

        for sender in range(num_nodes):
            valid_receivers = np.flatnonzero(local_mask[sender])
            row = policy.allocation_matrix[sender, valid_receivers]
            top1_receiver = int(valid_receivers[np.argmax(row)])
            row_entropies.append(_entropy(row))
            top1_masses.append(float(np.max(row)))
            top1_self_flags.append(float(int(top1_receiver == sender)))
            self_allocations.append(float(policy.allocation_matrix[sender, sender]))

    return {
        "mean_self_allocation": float(np.mean(self_allocations)) if self_allocations else 0.0,
        "mean_row_entropy": float(np.mean(row_entropies)) if row_entropies else 0.0,
        "mean_top1_mass": float(np.mean(top1_masses)) if top1_masses else 0.0,
        "mean_top1_self_rate": float(np.mean(top1_self_flags)) if top1_self_flags else 0.0,
        "mean_incoming_std": float(np.mean(incoming_stds)) if incoming_stds else 0.0,
        "mean_local_neighborhood_size": float(np.mean(neighborhood_sizes)) if neighborhood_sizes else 0.0,
        "mean_actual_cooperation_state": float(np.mean(cooperation_values)) if cooperation_values else 0.0,
    }


def compute_row_mechanism_records(
    snapshots: Sequence[Snapshot],
    *,
    p_c: float,
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

            ego_resources = np.asarray(observation["resources"][valid_receivers], dtype=np.float64)
            ego_investment = np.asarray(observation["investment"][valid_receivers], dtype=np.float64)
            pool_value = float(observation["pool_grown"][sender])
            ego_total_resources = float(ego_resources.sum())
            ego_total_investment = float(ego_investment.sum())
            sender_degree = max(int(valid_receivers.size) - 1, 0)
            pool_upperbound = float(sender_degree) * float(p_c)
            pool_position_ratio = pool_value / pool_upperbound if pool_upperbound > 1e-12 else float("nan")
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
                    "ego_total_resources": ego_total_resources,
                    "ego_total_investment": ego_total_investment,
                    "pool_over_ego_resources": pool_value / max(ego_total_resources, 1e-12),
                    "pool_over_ego_investment": pool_value / max(ego_total_investment, 1e-12),
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
    labor_weights = np.asarray([float(row["weight_labor"]) for row in row_records], dtype=np.float64)
    equal_weights = np.asarray([float(row["weight_equal"]) for row in row_records], dtype=np.float64)
    self_weights = np.asarray([float(row["weight_self"]) for row in row_records], dtype=np.float64)
    js_labor = np.asarray([float(row["js_to_labor"]) for row in row_records], dtype=np.float64)
    js_equal = np.asarray([float(row["js_to_equal"]) for row in row_records], dtype=np.float64)
    js_self = np.asarray([float(row["js_to_self"]) for row in row_records], dtype=np.float64)
    fit_errors = np.asarray([float(row["three_way_fit_l1"]) for row in row_records], dtype=np.float64)

    summary = {
        "row_count": int(len(row_records)),
        "pool_grown_over_upperbound_mean": float(np.nanmean(position_ratio)),
        "lambda_labor_equal_mean": float(np.nanmean(lambda_values)),
        "delta_equal_minus_labor_mean": float(np.nanmean(delta_values)),
        "weight_labor_mean": float(np.nanmean(labor_weights)),
        "weight_equal_mean": float(np.nanmean(equal_weights)),
        "weight_self_mean": float(np.nanmean(self_weights)),
        "js_to_labor_mean": float(np.nanmean(js_labor)),
        "js_to_equal_mean": float(np.nanmean(js_equal)),
        "js_to_self_mean": float(np.nanmean(js_self)),
        "three_way_fit_l1_mean": float(np.nanmean(fit_errors)),
        "pool_grown_over_upperbound_vs_lambda_spearman": _spearman_corr(position_ratio, lambda_values),
        "pool_grown_over_upperbound_vs_delta_spearman": _spearman_corr(position_ratio, delta_values),
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
                    "pool_grown_over_upperbound_mean": float(np.nanmean(position_ratio[mask])),
                    "lambda_labor_equal_mean": float(np.nanmean(lambda_values[mask])),
                    "delta_equal_minus_labor_mean": float(np.nanmean(delta_values[mask])),
                    "weight_labor_mean": float(np.nanmean(labor_weights[mask])),
                    "weight_equal_mean": float(np.nanmean(equal_weights[mask])),
                    "weight_self_mean": float(np.nanmean(self_weights[mask])),
                    "js_to_labor_mean": float(np.nanmean(js_labor[mask])),
                    "js_to_equal_mean": float(np.nanmean(js_equal[mask])),
                    "js_to_self_mean": float(np.nanmean(js_self[mask])),
                    "three_way_fit_l1_mean": float(np.nanmean(fit_errors[mask])),
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


def default_mechanism_summary() -> dict[str, Any]:
    return {
        "row_count": 0,
        "pool_grown_over_upperbound_mean": float("nan"),
        "lambda_labor_equal_mean": float("nan"),
        "delta_equal_minus_labor_mean": float("nan"),
        "weight_labor_mean": float("nan"),
        "weight_equal_mean": float("nan"),
        "weight_self_mean": float("nan"),
        "js_to_labor_mean": float("nan"),
        "js_to_equal_mean": float("nan"),
        "js_to_self_mean": float("nan"),
        "three_way_fit_l1_mean": float("nan"),
        "pool_grown_over_upperbound_vs_lambda_spearman": float("nan"),
        "pool_grown_over_upperbound_vs_delta_spearman": float("nan"),
        "pool_grown_over_upperbound_vs_weight_labor_spearman": float("nan"),
        "pool_grown_over_upperbound_vs_weight_equal_spearman": float("nan"),
        "pool_grown_over_upperbound_vs_weight_self_spearman": float("nan"),
    }


def default_node_income_summary() -> dict[str, Any]:
    return {
        "node_record_count": 0,
        "actual_over_equal_mean": float("nan"),
        "actual_over_labor_mean": float("nan"),
        "actual_minus_equal_mean": float("nan"),
        "actual_minus_labor_mean": float("nan"),
    }


def plot_scarcity_mechanism_bins(bin_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_episode_summary_csv(path: Path, summaries: Sequence[EpisodeSummary]) -> None:
    rows = [
        {
            "episode": summary.episode,
            "steps": summary.steps,
            "total_reward": summary.total_reward,
            "mean_reward": summary.mean_reward,
            "final_cooperation_rate": summary.final_cooperation_rate,
            "final_gini": summary.final_gini,
            "final_mean_resource": summary.final_mean_resource,
        }
        for summary in summaries
    ]
    write_csv(path, rows)


def write_decision_records_csv(path: Path, snapshots: Sequence[Snapshot]) -> None:
    fieldnames = [
        "episode",
        "step",
        "sender",
        "receiver",
        "is_self",
        "is_top_receiver",
        "reward",
        "cooperation_rate",
        "gini",
        "sender_pool_grown",
        "sender_resource",
        "sender_pool_raw_norm",
        "sender_resource_norm",
        "sender_degree_norm",
        "sender_strategy_norm",
        "sender_x_actual",
        "receiver_resource",
        "receiver_pool_raw_norm",
        "receiver_resource_norm",
        "receiver_degree_norm",
        "receiver_strategy_norm",
        "receiver_x_actual",
        "incoming_receiver",
        "allocation",
        "transferred",
        "logit",
        "self_allocation",
        "row_entropy",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for snapshot in snapshots:
            observation = snapshot.observation
            policy = snapshot.policy
            local_mask = observation["local_mask"].astype(bool, copy=False)
            num_nodes = local_mask.shape[0]

            for sender in range(num_nodes):
                valid_receivers = np.flatnonzero(local_mask[sender])
                row_allocation = policy.allocation_matrix[sender, valid_receivers]
                row_entropy = _entropy(row_allocation)
                top_receiver = int(valid_receivers[np.argmax(row_allocation)])
                self_allocation = float(policy.allocation_matrix[sender, sender])

                for receiver in valid_receivers:
                    writer.writerow(
                        {
                            "episode": snapshot.episode,
                            "step": snapshot.step,
                            "sender": sender,
                            "receiver": int(receiver),
                            "is_self": int(sender == receiver),
                            "is_top_receiver": int(top_receiver == int(receiver)),
                            "reward": snapshot.reward,
                            "cooperation_rate": snapshot.actual_cooperation_rate,
                            "gini": snapshot.gini,
                            "sender_pool_grown": float(observation["pool_grown"][sender]),
                            "sender_resource": float(observation["resources"][sender]),
                            "sender_pool_raw_norm": float(observation["pool_raw_norm"][sender]),
                            "sender_resource_norm": float(observation["resource_norm"][sender]),
                            "sender_degree_norm": float(observation["degree_norm"][sender]),
                            "sender_strategy_norm": float(observation["strategy_norm"][sender]),
                            "sender_x_actual": float(observation["x_actual"][sender]),
                            "receiver_resource": float(observation["resources"][receiver]),
                            "receiver_pool_raw_norm": float(observation["pool_raw_norm"][receiver]),
                            "receiver_resource_norm": float(observation["resource_norm"][receiver]),
                            "receiver_degree_norm": float(observation["degree_norm"][receiver]),
                            "receiver_strategy_norm": float(observation["strategy_norm"][receiver]),
                            "receiver_x_actual": float(observation["x_actual"][receiver]),
                            "incoming_receiver": float(policy.incoming_resources[receiver]),
                            "allocation": float(policy.allocation_matrix[sender, receiver]),
                            "transferred": float(policy.transferred_resources[sender, receiver]),
                            "logit": float(policy.logits[sender, receiver]),
                            "self_allocation": self_allocation,
                            "row_entropy": row_entropy,
                        }
                    )


def _select_snapshot_indices(total_count: int, snapshot_count: int) -> list[int]:
    if total_count <= 0:
        return []
    if snapshot_count <= 1:
        return [0]
    raw = np.linspace(0, total_count - 1, num=min(snapshot_count, total_count))
    return sorted({int(round(value)) for value in raw})


def plot_selected_snapshots(
    snapshots: Sequence[Snapshot],
    *,
    output_dir: Path,
    snapshot_count: int,
) -> list[dict[str, Any]]:
    if plt is None:
        return []

    selected_indices = _select_snapshot_indices(len(snapshots), snapshot_count)
    selected_metadata: list[dict[str, Any]] = []

    for snapshot_index in selected_indices:
        snapshot = snapshots[snapshot_index]
        observation = snapshot.observation
        policy = snapshot.policy
        feature_matrix = np.stack([observation[key] for key in NODE_FEATURE_KEYS], axis=1)
        top_incoming = np.argsort(policy.incoming_resources)[-10:][::-1]

        figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        alloc_image = axes[0].imshow(policy.allocation_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        axes[0].set_title("Allocation Matrix")
        axes[0].set_xlabel("Receiver")
        axes[0].set_ylabel("Sender")
        figure.colorbar(alloc_image, ax=axes[0], fraction=0.046, pad=0.04)

        feat_image = axes[1].imshow(feature_matrix, aspect="auto", cmap="coolwarm")
        axes[1].set_title("Node Features")
        axes[1].set_xlabel("Feature Channel")
        axes[1].set_ylabel("Node")
        axes[1].set_xticks(range(len(NODE_FEATURE_KEYS)))
        axes[1].set_xticklabels(NODE_FEATURE_KEYS, rotation=30, ha="right")
        figure.colorbar(feat_image, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].bar(np.arange(len(top_incoming)), policy.incoming_resources[top_incoming], color="#1f77b4")
        axes[2].set_title("Top Incoming Resources")
        axes[2].set_xlabel("Receiver Rank")
        axes[2].set_ylabel("Incoming Resource")
        axes[2].set_xticks(np.arange(len(top_incoming)))
        axes[2].set_xticklabels([str(int(node)) for node in top_incoming], rotation=45, ha="right")

        figure.suptitle(
            (
                f"Episode {snapshot.episode} Step {snapshot.step} | "
                f"reward={snapshot.reward:.4f}, coop={snapshot.actual_cooperation_rate:.3f}, "
                f"gini={snapshot.gini:.3f}, value={snapshot.policy.value:.3f}"
            ),
            fontsize=12,
        )

        filename = f"snapshot_ep{snapshot.episode:02d}_step{snapshot.step:03d}.png"
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)

        selected_metadata.append(
            {
                "snapshot_index": snapshot_index,
                "episode": snapshot.episode,
                "step": snapshot.step,
                "filename": filename,
            }
        )

    return selected_metadata


def plot_feature_importance(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return

    labels = [str(row["feature"]) for row in rows]
    allocation_scores = [float(row["allocation_l1_mean"]) for row in rows]
    transfer_scores = [float(row["transfer_l1_mean"]) for row in rows]
    top1_scores = [float(row["row_top1_change_rate"]) for row in rows]

    has_sender_gini = any("sender_row_gini_delta_mean" in row for row in rows)
    sender_gini_scores = [float(row.get("sender_row_gini_delta_mean", 0.0)) for row in rows]

    panel_count = 4 if has_sender_gini else 3
    figure, axes = plt.subplots(1, panel_count, figsize=(16 + (4 if has_sender_gini else 0), 6), constrained_layout=True)
    y_positions = np.arange(len(labels))

    axes[0].barh(y_positions, allocation_scores, color="#1f77b4")
    axes[0].set_title("Allocation Sensitivity")
    axes[0].set_xlabel("Mean L1 Delta")
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()

    axes[1].barh(y_positions, transfer_scores, color="#ff7f0e")
    axes[1].set_title("Transferred Resource Sensitivity")
    axes[1].set_xlabel("Mean L1 Delta")
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()

    axes[2].barh(y_positions, top1_scores, color="#2ca02c")
    axes[2].set_title("Top-1 Receiver Switch Rate")
    axes[2].set_xlabel("Rate")
    axes[2].set_yticks(y_positions)
    axes[2].set_yticklabels(labels)
    axes[2].invert_yaxis()

    if has_sender_gini:
        axes[3].barh(y_positions, sender_gini_scores, color="#9467bd")
        axes[3].set_title("Sender Row Gini Delta")
        axes[3].set_xlabel("Mean |Δ Gini|")
        axes[3].set_yticks(y_positions)
        axes[3].set_yticklabels(labels)
        axes[3].invert_yaxis()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_sender_row_gini_feature_summary(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return

    sorted_rows = sorted(rows, key=lambda row: float(row.get("sender_row_gini_delta_mean", 0.0)), reverse=True)
    labels = [str(row.get("feature", "")) for row in sorted_rows]
    base_means = [float(row.get("sender_row_gini_base_mean", float("nan"))) for row in sorted_rows]
    cf_means = [float(row.get("sender_row_gini_counterfactual_mean", float("nan"))) for row in sorted_rows]
    delta_means = [float(row.get("sender_row_gini_delta_mean", float("nan"))) for row in sorted_rows]

    y_positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    axes[0].plot(base_means, y_positions, "o", label="base", color="#1f77b4")
    axes[0].plot(cf_means, y_positions, "o", label="counterfactual", color="#ff7f0e")
    for idx, (base_value, cf_value) in enumerate(zip(base_means, cf_means)):
        if np.isfinite(base_value) and np.isfinite(cf_value):
            axes[0].plot([base_value, cf_value], [idx, idx], color="gray", alpha=0.4, linewidth=1.0)

    axes[0].set_title("Sender Row Gini: Base vs Counterfactual Mean")
    axes[0].set_xlabel("Mean Gini")
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 1.0)
    axes[0].legend()

    axes[1].barh(y_positions, delta_means, color="#9467bd")
    axes[1].set_title("Sender Row Gini Sensitivity")
    axes[1].set_xlabel("Mean |Δ Gini| (across senders)")
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()

    finite_deltas = [float(value) for value in delta_means if np.isfinite(value)]
    if finite_deltas:
        axes[1].set_xlim(0.0, float(max(finite_deltas)) * 1.05)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_sender_row_gini_distribution_for_feature(
    actor: GNNAllocationPolicy,
    snapshots: Sequence[Snapshot],
    *,
    feature_key: str,
    perturbation_mode: str,
    perturbation_seed: int,
    batch_size: int,
    output_path: Path,
) -> None:
    if plt is None or not snapshots:
        return

    observations = [snapshot.observation for snapshot in snapshots]
    rng = np.random.default_rng(int(perturbation_seed))
    perturbed_observations = [
        _perturb_observation(observation, feature_key, perturbation_mode, rng) for observation in observations
    ]
    perturbed_outputs = _chunked_policy_forward(actor, perturbed_observations, batch_size)

    base_values: list[np.ndarray] = []
    counterfactual_values: list[np.ndarray] = []
    for snapshot, perturbed_output in zip(snapshots, perturbed_outputs):
        mask = snapshot.observation["local_mask"].astype(bool, copy=False)
        base_values.append(_sender_row_ginis(snapshot.policy.allocation_matrix, mask))
        counterfactual_values.append(_sender_row_ginis(perturbed_output.allocation_matrix, mask))

    base_flat = np.concatenate(base_values) if base_values else np.empty(0, dtype=np.float64)
    cf_flat = (
        np.concatenate(counterfactual_values) if counterfactual_values else np.empty(0, dtype=np.float64)
    )
    base_flat = base_flat[np.isfinite(base_flat)]
    cf_flat = cf_flat[np.isfinite(cf_flat)]
    if base_flat.size == 0 or cf_flat.size == 0:
        return

    aligned_delta_mean = float(np.mean(np.abs(cf_flat - base_flat))) if base_flat.size == cf_flat.size else float("nan")

    bins = np.linspace(0.0, 1.0, 41)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].hist(base_flat, bins=bins, density=True, alpha=0.6, label="base", color="#1f77b4")
    axes[0].hist(
        cf_flat,
        bins=bins,
        density=True,
        alpha=0.6,
        label=f"cf({feature_key})",
        color="#ff7f0e",
    )
    axes[0].set_title("Sender Row Gini Distribution")
    axes[0].set_xlabel("Gini")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].legend()

    base_sorted = np.sort(base_flat)
    cf_sorted = np.sort(cf_flat)
    axes[1].plot(
        base_sorted,
        np.arange(1, base_sorted.size + 1, dtype=np.float64) / float(base_sorted.size),
        label="base",
        color="#1f77b4",
    )
    axes[1].plot(
        cf_sorted,
        np.arange(1, cf_sorted.size + 1, dtype=np.float64) / float(cf_sorted.size),
        label=f"cf({feature_key})",
        color="#ff7f0e",
    )
    axes[1].set_title("Empirical CDF")
    axes[1].set_xlabel("Gini")
    axes[1].set_ylabel("CDF")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    figure.suptitle(
        (
            f"Feature={feature_key} | mode={perturbation_mode} | "
            f"base_mean={float(np.mean(base_flat)):.3f} cf_mean={float(np.mean(cf_flat)):.3f} "
            f"mean|Δ|={aligned_delta_mean:.3f}"
        ),
        fontsize=12,
    )

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_analysis_summary(
    *,
    run_dir: Path,
    checkpoint_name: str,
    checkpoint_payload: Mapping[str, Any],
    topology_name: str,
    topology_label: str,
    network_config: Mapping[str, Any],
    effective_episode_length: int,
    requested_episode_length_override: int | None,
    enabled_modules: Mapping[str, Any],
    graph_stats: Mapping[str, Any],
    policy_behavior: Mapping[str, Any],
    mechanism_summary: Mapping[str, Any],
    node_income_summary: Mapping[str, Any],
    episode_summaries: Sequence[EpisodeSummary],
    feature_stats: Sequence[Mapping[str, Any]],
    feature_importance: Sequence[Mapping[str, Any]],
    snapshot_index_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "checkpoint_name": checkpoint_name,
        "completed_updates": int(checkpoint_payload.get("completed_updates", checkpoint_payload.get("update", 0))),
        "global_env_steps": int(checkpoint_payload.get("global_env_steps", 0)),
        "active_curriculum_stage_index": checkpoint_payload.get("active_curriculum_stage_index"),
        "teacher_handoff_stage": checkpoint_payload.get("teacher_handoff_stage"),
        "matplotlib_available": plt is not None,
        "topology_name": topology_name,
        "topology_label": topology_label,
        "network_config": dict(network_config),
        "effective_episode_length": int(effective_episode_length),
        "requested_episode_length_override": (
            None if requested_episode_length_override is None else int(requested_episode_length_override)
        ),
        "enabled_modules": dict(enabled_modules),
        "graph_stats": dict(graph_stats),
        "policy_behavior": dict(policy_behavior),
        "mechanism_summary": dict(mechanism_summary),
        "node_income_summary": dict(node_income_summary),
        "policy_mechanics": [
            "Actor first encodes the whole graph with a two-layer GraphNet backbone.",
            "Node channels are pool_raw_norm, resource_norm, degree_norm, strategy_norm; global channels are mean(x_actual), mean(resource_norm), mean(pool_raw_norm), gini.",
            "For each sender node i, the policy extracts the ego-subgraph S_i = {i} union N(i), adds a center-indicator, and also injects sender pool_grown and ego size as local-global context.",
            "ScoreReadout gives one score to each candidate receiver j in S_i, and softmax over that local candidate set becomes row i of allocation_matrix.",
            "transferred_resources[i, j] = allocation_matrix[i, j] * pool_grown[i], so pool_grown affects absolute transfer volume even when proportions stay similar.",
        ],
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
        "feature_channel_stats": list(feature_stats),
        "feature_importance": list(feature_importance),
        "selected_snapshots": list(snapshot_index_rows),
    }


def summarize_topology_case(
    *,
    topology_name: str,
    graph_stats: Mapping[str, Any],
    policy_behavior: Mapping[str, Any],
    mechanism_summary: Mapping[str, Any],
    node_income_summary: Mapping[str, Any],
    episode_summaries: Sequence[EpisodeSummary],
    feature_importance: Sequence[Mapping[str, Any]],
    snapshot_count: int,
) -> dict[str, Any]:
    total_rewards = [summary.total_reward for summary in episode_summaries]
    mean_rewards = [summary.mean_reward for summary in episode_summaries]
    final_coops = [summary.final_cooperation_rate for summary in episode_summaries]
    final_ginis = [summary.final_gini for summary in episode_summaries]
    final_resources = [summary.final_mean_resource for summary in episode_summaries]
    top_feature = feature_importance[0] if feature_importance else None

    row = {
        "topology": topology_name,
        "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
        "episodes": int(len(episode_summaries)),
        "snapshot_count": int(snapshot_count),
        "return_mean": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "reward_per_step_mean": float(np.mean(mean_rewards)) if mean_rewards else 0.0,
        "final_cooperation_mean": float(np.mean(final_coops)) if final_coops else 0.0,
        "final_gini_mean": float(np.mean(final_ginis)) if final_ginis else 0.0,
        "final_mean_resource_mean": float(np.mean(final_resources)) if final_resources else 0.0,
        "top_feature": "" if top_feature is None else str(top_feature["feature"]),
        "top_feature_allocation_l1": 0.0 if top_feature is None else float(top_feature["allocation_l1_mean"]),
    }
    row.update({key: value for key, value in graph_stats.items()})
    row.update({key: value for key, value in policy_behavior.items()})
    row.update({key: value for key, value in mechanism_summary.items()})
    row.update({key: value for key, value in node_income_summary.items()})
    return row


def build_topology_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    baseline_topology = "regular" if any(str(row["topology"]) == "regular" for row in rows) else str(rows[0]["topology"])
    baseline_row = next(row for row in rows if str(row["topology"]) == baseline_topology)
    comparison_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["baseline_topology"] = baseline_topology
        enriched["delta_return_mean_vs_baseline"] = float(row["return_mean"]) - float(baseline_row["return_mean"])
        enriched["delta_reward_per_step_mean_vs_baseline"] = float(row["reward_per_step_mean"]) - float(
            baseline_row["reward_per_step_mean"]
        )
        enriched["delta_final_cooperation_mean_vs_baseline"] = float(row["final_cooperation_mean"]) - float(
            baseline_row["final_cooperation_mean"]
        )
        enriched["delta_final_gini_mean_vs_baseline"] = float(row["final_gini_mean"]) - float(
            baseline_row["final_gini_mean"]
        )
        enriched["delta_final_mean_resource_mean_vs_baseline"] = float(row["final_mean_resource_mean"]) - float(
            baseline_row["final_mean_resource_mean"]
        )
        enriched["delta_mean_self_allocation_vs_baseline"] = float(row["mean_self_allocation"]) - float(
            baseline_row["mean_self_allocation"]
        )
        enriched["delta_mean_row_entropy_vs_baseline"] = float(row["mean_row_entropy"]) - float(
            baseline_row["mean_row_entropy"]
        )
        enriched["delta_lambda_labor_equal_mean_vs_baseline"] = float(row["lambda_labor_equal_mean"]) - float(
            baseline_row["lambda_labor_equal_mean"]
        )
        enriched["delta_weight_labor_mean_vs_baseline"] = float(row["weight_labor_mean"]) - float(
            baseline_row["weight_labor_mean"]
        )
        enriched["delta_weight_equal_mean_vs_baseline"] = float(row["weight_equal_mean"]) - float(
            baseline_row["weight_equal_mean"]
        )
        enriched["delta_weight_self_mean_vs_baseline"] = float(row["weight_self_mean"]) - float(
            baseline_row["weight_self_mean"]
        )
        comparison_rows.append(enriched)
    return comparison_rows


def plot_topology_comparison(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return

    labels = [str(row["topology_label"]) for row in rows]
    return_mean = [float(row["return_mean"]) for row in rows]
    final_coop = [float(row["final_cooperation_mean"]) for row in rows]
    self_allocation = [float(row["mean_self_allocation"]) for row in rows]

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    x = np.arange(len(labels))

    axes[0].bar(x, return_mean, color="#1f77b4")
    axes[0].set_title("Return Mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)

    axes[1].bar(x, final_coop, color="#2ca02c")
    axes[1].set_title("Final Cooperation Mean")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    axes[2].bar(x, self_allocation, color="#ff7f0e")
    axes[2].set_title("Mean Self Allocation")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyze_single_topology(
    *,
    actor: GNNAllocationPolicy,
    checkpoint_payload: Mapping[str, Any],
    run_dir: Path,
    output_dir: Path,
    spec: Mapping[str, Any],
    topology_name: str,
    checkpoint_name: str,
    episodes: int,
    max_steps: int | None,
    episode_length_override: int | None,
    snapshot_count: int,
    enable_snapshot_plots: bool,
    enable_feature_perturbation: bool,
    perturbation_mode: str,
    perturbation_seed: int,
    batch_size: int,
    enable_mechanism_analysis: bool,
    mechanism_bin_count: int,
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

    feature_stats = compute_feature_channel_stats(snapshots)
    feature_importance: list[dict[str, Any]] = []
    if enable_feature_perturbation:
        feature_importance = compute_feature_importance(
            actor,
            snapshots,
            feature_keys=ANALYSIS_FEATURE_KEYS,
            perturbation_mode=perturbation_mode,
            batch_size=batch_size,
            seed=perturbation_seed,
        )
    graph_stats = collect_graph_stats(graph)
    policy_behavior = compute_policy_behavior_stats(snapshots)
    row_mechanism_records: list[dict[str, Any]] = []
    scarcity_bin_rows: list[dict[str, Any]] = []
    node_income_records: list[dict[str, Any]] = []
    mechanism_summary = default_mechanism_summary()
    node_income_summary = default_node_income_summary()
    if enable_mechanism_analysis:
        row_mechanism_records = compute_row_mechanism_records(
            snapshots,
            p_c=float(env_config.p_c),
        )
        node_income_records = compute_node_income_decomposition(snapshots)
        mechanism_summary_raw, scarcity_bin_rows = summarize_row_mechanism_records(
            row_mechanism_records,
            bin_count=mechanism_bin_count,
        )
        node_income_summary_raw = summarize_node_income_records(node_income_records)
        mechanism_summary.update(mechanism_summary_raw)
        node_income_summary.update(node_income_summary_raw)

    snapshot_index_rows: list[dict[str, Any]] = []
    if enable_snapshot_plots:
        snapshot_index_rows = plot_selected_snapshots(
            snapshots,
            output_dir=output_dir,
            snapshot_count=snapshot_count,
        )
    if enable_feature_perturbation:
        plot_feature_importance(feature_importance, output_dir / "feature_importance.png")
        plot_sender_row_gini_feature_summary(feature_importance, output_dir / "sender_row_gini_feature_summary.png")

        if feature_importance:
            top_row = max(feature_importance, key=lambda row: float(row.get("sender_row_gini_delta_mean", 0.0)))
            top_feature = str(top_row.get("feature", "unknown"))
            plot_sender_row_gini_distribution_for_feature(
                actor,
                snapshots,
                feature_key=top_feature,
                perturbation_mode=perturbation_mode,
                perturbation_seed=perturbation_seed,
                batch_size=batch_size,
                output_path=output_dir / f"sender_row_gini_distribution_{top_feature}.png",
            )
    if enable_mechanism_analysis:
        plot_scarcity_mechanism_bins(scarcity_bin_rows, output_dir / "scarcity_vs_mechanism.png")

    write_episode_summary_csv(output_dir / "episode_summary.csv", episode_summaries)
    write_csv(output_dir / "feature_channel_stats.csv", feature_stats)
    write_decision_records_csv(output_dir / "decision_records.csv", snapshots)
    if enable_feature_perturbation:
        write_csv(output_dir / "feature_importance.csv", feature_importance)
    if enable_snapshot_plots:
        write_csv(output_dir / "snapshot_index.csv", snapshot_index_rows)
    if enable_mechanism_analysis:
        write_csv(output_dir / "row_mechanism.csv", row_mechanism_records)
        write_csv(output_dir / "scarcity_bins.csv", scarcity_bin_rows)
        write_csv(output_dir / "node_income_decomposition.csv", node_income_records)

    summary = build_analysis_summary(
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_payload=checkpoint_payload,
        topology_name=topology_name,
        topology_label=TOPOLOGY_LABELS.get(topology_name, topology_name),
        network_config=effective_spec["network"],
        effective_episode_length=int(effective_spec["dynamics"]["episode_length"]),
        requested_episode_length_override=episode_length_override,
        enabled_modules={
            "snapshot_plots": bool(enable_snapshot_plots),
            "feature_perturbation": bool(enable_feature_perturbation),
            "mechanism_analysis": bool(enable_mechanism_analysis),
        },
        graph_stats=graph_stats,
        policy_behavior=policy_behavior,
        mechanism_summary=mechanism_summary,
        node_income_summary=node_income_summary,
        episode_summaries=episode_summaries,
        feature_stats=feature_stats,
        feature_importance=feature_importance,
        snapshot_index_rows=snapshot_index_rows,
    )
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    topology_row = summarize_topology_case(
        topology_name=topology_name,
        graph_stats=graph_stats,
        policy_behavior=policy_behavior,
        mechanism_summary=mechanism_summary,
        node_income_summary=node_income_summary,
        episode_summaries=episode_summaries,
        feature_importance=feature_importance,
        snapshot_count=len(snapshots),
    )
    feature_rows = [dict(row, topology=topology_name, topology_label=TOPOLOGY_LABELS.get(topology_name, topology_name)) for row in feature_importance]

    return {
        "topology": topology_name,
        "graph": graph,
        "snapshots": snapshots,
        "episode_summaries": episode_summaries,
        "feature_stats": feature_stats,
        "feature_importance": feature_importance,
        "graph_stats": graph_stats,
        "policy_behavior": policy_behavior,
        "mechanism_summary": mechanism_summary,
        "node_income_summary": node_income_summary,
        "row_mechanism_records": row_mechanism_records,
        "scarcity_bin_rows": scarcity_bin_rows,
        "node_income_records": node_income_records,
        "topology_row": topology_row,
        "feature_rows": feature_rows,
        "summary": summary,
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze how a trained GNN-RL allocator makes decisions.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(str(SCRIPT_DEFAULTS["run_dir"])),
        help="Experiment output directory containing results.json and checkpoints/.",
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
        help="Directory for analysis artifacts. Defaults to <run-dir>/policy_analysis/<checkpoint-stem>/.",
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
        help="Optional cap on steps per episode for a quick pass.",
    )
    parser.add_argument(
        "--episode-length-override",
        type=int,
        default=SCRIPT_DEFAULTS["episode_length_override"],
        help="Override the environment episode_length used during analysis.",
    )
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["snapshot_count"]),
        help="How many evenly spaced snapshots to export as figures.",
    )
    parser.add_argument(
        "--enable-snapshot-plots",
        type=_parse_toggle_arg,
        default=bool(SCRIPT_DEFAULTS["enable_snapshot_plots"]),
        metavar="{on,off}",
        help="Whether to export rollout snapshot figures and snapshot_index.csv.",
    )
    parser.add_argument(
        "--enable-feature-perturbation",
        type=_parse_toggle_arg,
        default=bool(SCRIPT_DEFAULTS["enable_feature_perturbation"]),
        metavar="{on,off}",
        help="Whether to run feature perturbation sensitivity analysis.",
    )
    parser.add_argument(
        "--perturbation-mode",
        type=str,
        default=str(SCRIPT_DEFAULTS["perturbation_mode"]),
        choices=("zero", "mean", "shuffle"),
        help="How to perturb a feature when estimating its influence.",
    )
    parser.add_argument(
        "--perturbation-seed",
        type=int,
        default=int(SCRIPT_DEFAULTS["perturbation_seed"]),
        help="Random seed used by shuffle perturbations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(SCRIPT_DEFAULTS["batch_size"]),
        help="Batch size for offline actor inference during perturbation analysis.",
    )
    parser.add_argument(
        "--enable-mechanism-analysis",
        type=_parse_toggle_arg,
        default=bool(SCRIPT_DEFAULTS["enable_mechanism_analysis"]),
        metavar="{on,off}",
        help="Whether to run labor/equal/self mechanism decomposition analysis.",
    )
    parser.add_argument(
        "--mechanism-bin-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["mechanism_bin_count"]),
        help="Quantile bin count for P_grown / P_upperbound vs mechanism summaries.",
    )
    parser.add_argument(
        "--enable-topology-comparison",
        type=_parse_toggle_arg,
        default=bool(SCRIPT_DEFAULTS["enable_topology_comparison"]),
        metavar="{on,off}",
        help="Whether to export aggregated cross-topology comparison artifacts when multiple topologies are analyzed.",
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
        else (run_dir / "policy_analysis" / Path(args.checkpoint_name).stem).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    topologies = parse_topology_list(args.topologies)
    enable_snapshot_plots = bool(args.enable_snapshot_plots)
    enable_feature_perturbation = bool(args.enable_feature_perturbation)
    enable_mechanism_analysis = bool(args.enable_mechanism_analysis)
    enable_topology_comparison = bool(args.enable_topology_comparison)

    device = torch.device(args.device)
    actor, checkpoint_payload = load_actor_from_run_dir(run_dir, args.checkpoint_name, device)
    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)

    topology_results: list[dict[str, Any]] = []
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
            episodes=int(args.episodes),
            max_steps=args.max_steps,
            episode_length_override=args.episode_length_override,
            snapshot_count=max(1, int(args.snapshot_count)),
            enable_snapshot_plots=enable_snapshot_plots,
            enable_feature_perturbation=enable_feature_perturbation,
            perturbation_mode=args.perturbation_mode,
            perturbation_seed=int(args.perturbation_seed) + topology_index,
            batch_size=max(1, int(args.batch_size)),
            enable_mechanism_analysis=enable_mechanism_analysis,
            mechanism_bin_count=max(2, int(args.mechanism_bin_count)),
            rollout_seed=int(args.seed),
            checkpoint_name=args.checkpoint_name,
        )
        topology_results.append(result)

    comparison_rows = build_topology_comparison_rows([result["topology_row"] for result in topology_results])
    comparison_feature_rows = [row for result in topology_results for row in result["feature_rows"]]

    if len(topologies) > 1 and enable_topology_comparison:
        write_csv(output_dir / "topology_comparison.csv", comparison_rows)
        if enable_feature_perturbation:
            write_csv(output_dir / "topology_feature_importance.csv", comparison_feature_rows)
        plot_topology_comparison(comparison_rows, output_dir / "topology_comparison.png")
        multi_summary = {
            "run_dir": str(run_dir),
            "checkpoint_name": args.checkpoint_name,
            "trained_topology": str(experiment_spec["network"]["type"]),
            "requested_topologies": topologies,
            "reference_mean_degree": reference_mean_degree,
            "enabled_modules": {
                "snapshot_plots": enable_snapshot_plots,
                "feature_perturbation": enable_feature_perturbation,
                "mechanism_analysis": enable_mechanism_analysis,
                "topology_comparison": enable_topology_comparison,
            },
            "requested_episode_length_override": (
                None if args.episode_length_override is None else int(args.episode_length_override)
            ),
            "matplotlib_available": plt is not None,
            "topology_network_configs": {
                result["topology"]: dict(result["summary"]["network_config"]) for result in topology_results
            },
            "topology_comparison": comparison_rows,
        }
        with (output_dir / "topology_comparison.json").open("w", encoding="utf-8") as handle:
            json.dump(multi_summary, handle, ensure_ascii=False, indent=2)

    print(f"Analysis complete. Artifacts written to: {output_dir}")
    if plt is None:
        print("Plot export skipped because matplotlib is not installed in the current interpreter.")
    print(
        "Enabled modules: "
        f"snapshot_plots={_toggle_label(enable_snapshot_plots)} "
        f"feature_perturbation={_toggle_label(enable_feature_perturbation)} "
        f"mechanism_analysis={_toggle_label(enable_mechanism_analysis)} "
        f"topology_comparison={_toggle_label(enable_topology_comparison)}"
    )
    print("Per-topology summary:")
    for row in comparison_rows:
        parts = [
            f"{row['topology_label']:<8}",
            f"return={float(row['return_mean']):.6f}",
            f"coop={float(row['final_cooperation_mean']):.4f}",
            f"gini={float(row['final_gini_mean']):.4f}",
            f"self={float(row['mean_self_allocation']):.4f}",
        ]
        if enable_mechanism_analysis:
            parts.extend(
                [
                    f"lambda={float(row['lambda_labor_equal_mean']):.4f}",
                    f"eq_w={float(row['weight_equal_mean']):.4f}",
                    f"labor_w={float(row['weight_labor_mean']):.4f}",
                    f"P/Pmax->lambda={float(row['pool_grown_over_upperbound_vs_lambda_spearman']):.4f}",
                ]
            )
        if enable_feature_perturbation:
            parts.append(f"top_feature={row['top_feature']}")
        print("  " + " ".join(parts))


if __name__ == "__main__":
    main()
