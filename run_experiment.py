from __future__ import annotations

"""
单文件实验入口。

使用方式：
1. 直接在这个文件顶部改参数。
2. 运行：
   python run_experiment.py

设计目标：
- 把论文实验常改的参数集中在一个文件中。
- 支持单次实验，也支持一次跑多组不同网络/模型参数。
- 支持两类可视化：
  1. 微观图：每个时间步一张网络快照图，观察节点、边、策略和资源状态。
  2. 宏观图：同一实验过程中统计量随时间的变化曲线。
"""

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from Project1 import (
    RewardConfig,
    SPGGConfig,
    SPGGEnv,
    gini_coefficient,
    make_barabasi_albert_graph,
    make_erdos_renyi_graph,
    make_grid_graph,
    make_random_regular_graph,
    make_watts_strogatz_graph,
)
from Project1.policies.rule_based import ProportionalContributionPolicy, UniformAllocationPolicy


# =============================================================================
# 基础实验配置：如果你只想跑一组实验，通常只改这里
# =============================================================================

BASE_EXPERIMENT = {
    # 这次实验的名字。
    # 它会决定输出目录名、结果 JSON 中的实验名，也方便你区分不同实验。
    "experiment_name": "spgg_regular_uniform_demo",

    # 全局随机种子。
    # 用来控制网络生成、环境初始化、批量实验中的随机性。
    # 如果你想复现结果，就固定成一个整数。
    "seed": 42,

    # 运行模式：
    # - "uniform"      ：人工规则，均匀分配
    # - "proportional" ：人工规则，按贡献比例分配
    # - "gnn_train"    ：训练 GNN-RL 分配器，再做训练后评估
    "run_mode": "uniform",

    # ---------------------------
    # 网络配置
    # ---------------------------
    "network": {
        # 网络类型：
        # - "regular"     ：随机正则网络
        # - "erdos_renyi" ：ER 随机网络
        # - "small_world" ：WS 小世界网络
        # - "scale_free"  ：BA 无标度网络
        # - "grid"        ：二维格子，仅保留做对照
        "type": "regular",

        # 节点总数 n。
        # 对 regular / erdos_renyi / small_world / scale_free 都使用这个值。
        # 对 grid 网络，这个值会被 grid_rows * grid_cols 取代。
        "num_nodes": 100,

        # regular 网络参数：每个节点的度。
        # 例如 degree=4 表示每个节点都恰好连 4 条边。
        "regular_degree": 4,

        # ER 网络参数：目标平均度。
        # 如果这里不是 None，程序会自动按
        #   p = er_target_mean_degree / (num_nodes - 1)
        # 计算 ER 连边概率。
        # 例如目标平均度为 4：
        # - n=100  时，p≈4/99   ≈ 0.0404
        # - n=2500 时，p≈4/2499 ≈ 0.0016
        "er_target_mean_degree": 4.0,

        # ER 网络参数：手动指定任意两点之间连边的概率。
        # 仅当 er_target_mean_degree 为 None 时使用。
        "er_edge_prob": None,

        # WS 小世界网络参数：初始规则环网络的度。
        # 当前实现要求是偶数。
        "ws_degree": 4,

        # WS 小世界网络参数：重连概率。
        # 越大，随机捷径越多。
        "ws_rewiring_prob": 0.10,

        # BA 无标度网络参数：每个新节点接到几个已有节点上。
        # 一般用 2、3、4 这类小整数。
        "ba_attachments_per_new_node": 2,

        # grid 网络参数：只有 type == "grid" 时使用。
        "grid_rows": 10,
        "grid_cols": 10,
        "grid_periodic": False,
    },

    # ---------------------------
    # 环境动力学参数
    # ---------------------------
    "dynamics": {
        # alpha：控制合作投入随资源增长的线性变化强度。
        # 对应：
        #   e_i,t = min(R_i,t, max(0, (d_i+1) + alpha * (R_i,t - (d_i+1))))
        "alpha": 0.5,

        # r：资源池放大参数。
        # 对应：
        #   G_i,t = min((1 + r) * P_i,t, P_max)
        "r": 2,

        # P_max：资源池增长后的饱和上限。
        "p_max": 100.0,#40

        # 资源自然消耗模式：
        # - "fixed"            ：每轮固定消耗
        # - "proportional"     ：按当前资源按比例消耗
        # - "piecewise_linear" ：先固定消耗，超过阈值后再按比例增加
        "resource_consumption_mode": "piecewise_linear",

        # 固定消耗项的计算模式：
        # - "constant"      ：所有节点用同一个固定常数
        # - "degree_scaled" ：固定项 = 倍数 × 节点度 d_i
        "resource_consumption_fixed_mode": "degree_scaled",

        # 固定消耗constant项。
        # - fixed            ：每轮直接消耗这个值
        # - piecewise_linear ：作为基础固定消耗
        # 仅当 resource_consumption_fixed_mode == "constant" 时生效。
        "resource_consumption_fixed": 4, #暂时设置等于平均度

        # 度比例固定消耗倍数。
        # 当 resource_consumption_fixed_mode == "degree_scaled" 时：
        # fixed_term_i = resource_consumption_degree_multiplier * degree_i
        "resource_consumption_degree_multiplier": 1.0,

        # 比例消耗系数。
        # - proportional     ：consumption = rate * resources
        # - piecewise_linear ：consumption = fixed + rate * max(resources - threshold, 0)
        "resource_consumption_rate": 0.05, #0.1

        # 分段线性消耗阈值。
        # 仅当 resource_consumption_mode == "piecewise_linear" 时使用。
        "resource_consumption_threshold": 4.0, #50

        # 个体策略更新规则：
        # - "fermi"        ：同步 Fermi 更新
        # - "q_learning"   ：每个节点使用二动作无状态 Q-learning 更新
        # - "q_learning_2x2"：每个节点使用 2状态×2动作 Q-learning，
        #                     状态=自己上一轮动作，动作=本轮选 C/D
        # - "imitate_best" ：最优邻居模仿 / Best-takes-over
        "strategy_update_rule": "q_learning",

        # beta：同步 Fermi 更新的选择强度。
        # 仅当 strategy_update_rule == "fermi" 时使用。
        # 越大，节点越偏向模仿高收益邻居。
        "beta": 1.0,

        # 以下参数仅当 strategy_update_rule == "q_learning"
        # 或 "q_learning_2x2" 时使用。
        "q_learning_rate": 0.1,
        "q_learning_discount": 0.1,
        "q_learning_epsilon": 0.05,
        "q_learning_initial_value": 0.0,

        # 每个 episode 的时间步上限。
        # 到达这个步数后，本 episode 结束。
        "episode_length": 10000, #150 10000

        # 所有节点统一的初始资源。
        "initial_resource": 20.0,#10

        # 初始名义合作概率。
        # 如果不单独指定节点策略向量，reset 时每个节点按这个概率初始化为合作。
        "initial_cooperation_prob": 0.5,
    },

    # ---------------------------
    # planner 奖励参数
    # ---------------------------
    "reward": {
        # 平均净收益项的权重。
        "lambda_payoff": 1.0,

        # 下一时刻实际合作比例项的权重。
        "lambda_cooperation": 0.0,

        # Gini 不平等惩罚项的权重。
        "lambda_gini": 0.0,

        # Gini 分母的极小修正项，通常不需要改。
        "epsilon": 1e-8,
    },

    # ---------------------------
    # GNN 策略参数
    # ---------------------------
    "gnn": {
        # 节点嵌入隐藏维度。
        "hidden_dim": 64,

        # 消息传递层数。
        "num_message_passing_layers": 2,

        # 局部 softmax 温度参数 tau。
        "temperature": 1.0,

        # Dirichlet 训练时的浓度缩放系数。
        "dirichlet_concentration_scale": 1.0,

        # Dirichlet 浓度下界，避免数值不稳定。
        "dirichlet_concentration_floor": 0.1,
    },

    # ---------------------------
    # 训练参数
    # ---------------------------
    "training": {
        # 总 update 次数。
        "total_updates": 50,

        # 每次 update 收集多少个环境步。
        "steps_per_update": 64,

        # 折扣因子 gamma。
        "gamma": 0.99,

        # GAE 的 lambda。
        "gae_lambda": 0.95,

        # Adam 学习率。
        "learning_rate": 3e-4,

        # 熵正则系数。
        "entropy_coef": 1e-3,

        # 价值函数损失权重。
        "value_coef": 0.5,

        # 梯度裁剪阈值。
        "max_grad_norm": 1.0,

        # 每隔多少个 update 做一次评估。
        "eval_interval": 10,

        # 每次评估多少个 episode。
        "eval_episodes": 3,

        # 训练设备：cpu 或 cuda。
        "device": "cuda",
    },

    # ---------------------------
    # 规则模式 / 评估模式运行长度
    # ---------------------------
    "rollout": {
        # 在人工规则模式下，一共跑多少个 episode。
        # 例如 5 表示从头 reset 5 次，每次都跑到 episode_length。
        "episodes": 5,

        # 在 gnn_train 模式下，训练结束后用训练好的策略再评估多少个 episode。
        "post_training_eval_episodes": 2,
    },

    # ---------------------------
    # 可视化参数
    # ---------------------------
    "visualization": {
        # 是否生成微观网络快照图。
        # 开启后，每个选中的 episode 都会在每个时间步保存一张图。
        "enable_micro_snapshots": False,

        # 是否生成宏观时间序列图。
        # 开启后，每个选中的 episode 都会保存一张统计量随时间变化的折线图。
        "enable_macro_timeseries": True,

        # 想要可视化哪些 episode。
        # 用 1-based 编号，例如 [1, 3] 表示只可视化第 1 和第 3 个 episode。
        # 设为 [] 表示所有 episode 都可视化。
        "episodes_to_visualize": [1],

        # 微观图每隔多少步保存一张。
        # 1 表示每步都存。
        # 5 表示 t=0,5,10,15,... 才存，能显著减少图片数量。
        "frame_stride": 1,

        # 网络布局方式：
        # - "spring"   ：力导向布局，适合一般网络
        # - "circular" ：圆形布局，简单稳定
        # - "grid"     ：只有 grid 网络建议用，其他网络不要用
        "layout": "spring",

        # spring 布局迭代次数。
        # 越大布局通常越舒展，但也越慢。
        "spring_iterations": 80,

        # 微观图里节点颜色映射到哪个节点级参数。
        # 可选典型值：
        # - "x_actual"   ：实际执行策略（合作/背叛）
        # - "x_nominal"  ：名义策略
        # - "resources"  ：累计资源
        # - "pool_raw"   ：原始池子资源
        # - "pool_grown" ：增长后池子资源
        # - "investment" ：个体投入
        # - "income"     ：个体收入
        # - "consumption": 个体消耗
        # - "payoff"     ：个体净收益
        "node_color_metric": "x_actual",

        # 微观图里节点大小映射到哪个节点级参数。
        # 常见搭配：
        # - 颜色看策略，大小看资源
        # - 颜色看资源，大小看资源
        "node_size_metric": "resources",

        # 节点中心显示哪个节点级参数。
        # 设为 None、"" 时表示不在节点中心显示数值。
        # 常见值：
        # - "resources"  ：显示当前个体资源
        # - "investment" ：显示当前投入
        # - "income"     ：显示当前收入
        # - "consumption": 显示当前消耗
        "node_value_metric": "resources",

        # 节点中心数值保留的小数位数。
        "node_value_decimals": 1,

        # 是否在微观图中标出节点编号。
        # 如果同时显示中心数值，编号会绘制在节点上方。
        # 节点很多时建议关掉，否则会很乱。
        "label_nodes": True,

        # 宏观图要画哪些统计量。
        # 横轴统一是时间 t，纵轴是这些统计量的值。
        "macro_metrics": [
            "actual_cooperation_rate",
            "mean_resource",
            "mean_pool_grown",
            "mean_consumption",
            "mean_payoff",
            "gini",
        ],

        # 画图保存分辨率。
        "save_dpi": 160,
    },

    # ---------------------------
    # 输出与保存参数
    # ---------------------------
    "output": {
        # 所有输出结果的根目录。
        # 每个实验会在下面单独建一个子目录。
        "root_dir": "outputs/200frame_piecewise_degree_Pmax100_R020_r2_len200",#"outputs",

        # 是否保存结果 JSON。
        "save_results_json": True,

        # 是否把微观快照图真正落盘。
        # 如果 enable_micro_snapshots=True 但这里 False，则不保存文件。
        "save_micro_snapshots": True,

        # 是否把宏观时间序列图真正落盘。
        "save_macro_timeseries": True,
    },
}


# =============================================================================
# 批量实验配置：如果你想“一次跑多组参数”，就在这里加条目
# =============================================================================
#
# 使用方式：
# 1. 保持 BASE_EXPERIMENT 作为共同基础配置
# 2. 在下面每个 dict 里只写你想覆盖的字段
# 3. 程序会自动把它们逐个和 BASE_EXPERIMENT 合并，然后依次运行
#
# 覆盖规则示例：
# {
#     "experiment_name": "regular_d4_alpha05",
#     "network": {"type": "regular", "regular_degree": 4},
#     "dynamics": {"alpha": 0.5},
#     "visualization": {"enable_macro_timeseries": True},
# }
#
BATCH_EXPERIMENTS = [
    {
        "experiment_name": "regular_d4_prop_r15_q_learning",
        "network": {"type": "regular", "regular_degree": 4},
        "run_mode": "proportional",
        "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    },
        {
        "experiment_name": "regular_d4_uniform_r15_q_learning",
        "network": {"type": "regular", "regular_degree": 4},
        "run_mode": "uniform",
        "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    },
        {
        "experiment_name": "ba_m2_proportional_r15_q_learning",
        "network": {"type": "scale_free", "ba_attachments_per_new_node": 2},
        "run_mode": "proportional",
        "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    },
        {
        "experiment_name": "ba_m2_uniform_r15_q_learning",
        "network": {"type": "scale_free", "ba_attachments_per_new_node": 2},
        "run_mode": "uniform",
        "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    },
        {
        "experiment_name": "er_k4_proportional_r15_q_learning",
        "network": {"type": "erdos_renyi", "er_target_mean_degree": 4.0},
        "run_mode": "proportional",
        "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    },
        {
        "experiment_name": "er_k4_uniform_r15_q_learning",
        "network": {"type": "erdos_renyi", "er_target_mean_degree": 4.0},
        "run_mode": "uniform",
        "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    },
        {
        "experiment_name": "ws_k4_p01_proportional_r15_q_learning",
        "network": {"type": "small_world", "ws_degree": 4, "ws_rewiring_prob": 0.1},
        "run_mode": "proportional",
        "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    },
        {
        "experiment_name": "ws_k4_p01_uniform_r15_q_learning",
        "network": {"type": "small_world", "ws_degree": 4, "ws_rewiring_prob": 0.1},
        "run_mode": "uniform",
        "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    },
]


# =============================================================================
# 参数扫描配置：启用后会忽略上面的 BATCH_EXPERIMENTS，自动生成扫描实验
# =============================================================================
SCAN_EXPERIMENT = {
    "enabled": True,
    "name": "3_18_num_nodes_r_network_consumption_strategy_scan_proportional",
    "output_root_dir": "outputs/10000frame_r_network_consumption_strategy_scan_0.01_0.1tau",
    "parallel": True,
    "max_workers": 32,#自己的电脑为16核
    "r_values": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    "resource_consumption_rate": [0.01, 0.05,0.1],
    "num_nodes": [100,1000,2500],
    "network_types": ["regular", "erdos_renyi", "small_world", "scale_free"],
    "resource_consumption_modes": ["piecewise_linear"],#["fixed", "proportional", "piecewise_linear"],
    "resource_consumption_fixed_modes": ["constant", "degree_scaled"],
    "strategy_update_rules": ["q_learning", "fermi"],
    "run_mode": ["proportional"],
}

def deep_update(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _format_float_token(value: float) -> str:
    token = format(float(value), "g")
    return token.replace("-", "m").replace(".", "p")


def _resolve_er_edge_prob(network: Mapping[str, Any]) -> float:
    num_nodes = int(network["num_nodes"])
    target_mean_degree = network.get("er_target_mean_degree")
    if target_mean_degree is not None:
        target_mean_degree = float(target_mean_degree)
        if target_mean_degree < 0.0:
            raise ValueError("er_target_mean_degree must be non-negative.")
        max_mean_degree = max(num_nodes - 1, 0)
        if target_mean_degree > max_mean_degree:
            raise ValueError(
                "er_target_mean_degree must be <= num_nodes - 1."
            )
        if num_nodes <= 1:
            return 0.0
        return target_mean_degree / float(num_nodes - 1)

    edge_prob = network.get("er_edge_prob")
    if edge_prob is None:
        raise ValueError("ER network requires er_target_mean_degree or er_edge_prob.")
    edge_prob = float(edge_prob)
    if not 0.0 <= edge_prob <= 1.0:
        raise ValueError("er_edge_prob must be in [0, 1].")
    return edge_prob


def _er_parameter_summary(network: Mapping[str, Any]) -> str:
    edge_prob = _resolve_er_edge_prob(network)
    target_mean_degree = network.get("er_target_mean_degree")
    if target_mean_degree is not None:
        return "target_mean_degree={0}, edge_prob={1:.6g}".format(
            float(target_mean_degree),
            edge_prob,
        )
    expected_mean_degree = edge_prob * max(int(network["num_nodes"]) - 1, 0)
    return "edge_prob={0:.6g}, expected_mean_degree={1:.6g}".format(
        edge_prob,
        expected_mean_degree,
    )


def _network_variant_label(network: Mapping[str, Any]) -> str:
    network_type = network["type"]
    if network_type == "regular":
        return "regular_d{0}".format(network["regular_degree"])
    if network_type == "erdos_renyi":
        if network.get("er_target_mean_degree") is not None:
            return "er_k{0}".format(_format_float_token(network["er_target_mean_degree"]))
        return "er_p{0}".format(_format_float_token(_resolve_er_edge_prob(network)))
    if network_type == "small_world":
        return "ws_k{0}_p{1}".format(
            network["ws_degree"],
            _format_float_token(network["ws_rewiring_prob"]),
        )
    if network_type == "scale_free":
        return "ba_m{0}".format(network["ba_attachments_per_new_node"])
    if network_type == "grid":
        return "grid_{0}x{1}".format(network["grid_rows"], network["grid_cols"])
    raise ValueError("Unsupported network type: {0}".format(network_type))


def _network_override_for_type(base_network: Mapping[str, Any], network_type: str) -> Dict[str, Any]:
    override: Dict[str, Any] = {"type": network_type, "num_nodes": base_network["num_nodes"]}
    if network_type == "regular":
        override["regular_degree"] = base_network["regular_degree"]
    elif network_type == "erdos_renyi":
        if "er_target_mean_degree" in base_network:
            override["er_target_mean_degree"] = base_network["er_target_mean_degree"]
        if "er_edge_prob" in base_network:
            override["er_edge_prob"] = base_network["er_edge_prob"]
    elif network_type == "small_world":
        override["ws_degree"] = base_network["ws_degree"]
        override["ws_rewiring_prob"] = base_network["ws_rewiring_prob"]
    elif network_type == "scale_free":
        override["ba_attachments_per_new_node"] = base_network["ba_attachments_per_new_node"]
    elif network_type == "grid":
        override["grid_rows"] = base_network["grid_rows"]
        override["grid_cols"] = base_network["grid_cols"]
        override["grid_periodic"] = base_network["grid_periodic"]
    else:
        raise ValueError("Unsupported network type in scan: {0}".format(network_type))
    return override


def _consumption_variants(scan_config: Mapping[str, Any]) -> List[tuple[str, Optional[str]]]:
    variants: List[tuple[str, Optional[str]]] = []
    for mode in scan_config["resource_consumption_modes"]:
        if mode == "proportional":
            variants.append((mode, None))
            continue
        for fixed_mode in scan_config["resource_consumption_fixed_modes"]:
            variants.append((mode, fixed_mode))
    return variants


def _consumption_variant_label(mode: str, fixed_mode: Optional[str]) -> str:
    if mode == "proportional":
        return "proportional"
    if fixed_mode is None:
        raise ValueError("fixed_mode is required for mode {0}".format(mode))
    return "{0}_{1}".format(mode, fixed_mode)


def _scan_run_modes(scan_config: Mapping[str, Any], base_run_mode: str) -> List[str]:
    configured_run_mode = scan_config.get("run_mode", base_run_mode)
    if isinstance(configured_run_mode, str):
        run_modes = [configured_run_mode]
    else:
        run_modes = [str(item) for item in configured_run_mode]

    if not run_modes:
        raise ValueError("SCAN_EXPERIMENT['run_mode'] must contain at least one run mode.")
    invalid_run_modes = [item for item in run_modes if item not in {"uniform", "proportional"}]
    if invalid_run_modes:
        raise ValueError(
            "Scan mode currently supports only rule-based run_mode values {'uniform', 'proportional'}."
            " Invalid values: {0}".format(invalid_run_modes)
        )
    return run_modes


def _scan_num_nodes(scan_config: Mapping[str, Any], base_num_nodes: int) -> List[int]:
    configured_num_nodes = scan_config.get("num_nodes", base_num_nodes)
    if isinstance(configured_num_nodes, int):
        num_nodes_values = [configured_num_nodes]
    else:
        num_nodes_values = [int(item) for item in configured_num_nodes]

    if not num_nodes_values:
        raise ValueError("SCAN_EXPERIMENT['num_nodes'] must contain at least one value.")
    if any(item <= 0 for item in num_nodes_values):
        raise ValueError("SCAN_EXPERIMENT['num_nodes'] values must be positive.")
    return num_nodes_values


def build_scan_experiment_specs() -> List[Dict[str, Any]]:
    base = deepcopy(BASE_EXPERIMENT)
    scan = SCAN_EXPERIMENT
    run_modes = _scan_run_modes(scan, str(base["run_mode"]))
    num_nodes_values = _scan_num_nodes(scan, int(base["network"]["num_nodes"]))
    specs: List[Dict[str, Any]] = []

    for run_mode in run_modes:
        for strategy_update_rule in scan["strategy_update_rules"]:
            for resource_consumption_mode, resource_consumption_fixed_mode in _consumption_variants(scan):
                consumption_label = _consumption_variant_label(
                    resource_consumption_mode,
                    resource_consumption_fixed_mode,
                )
                for num_nodes in num_nodes_values:
                    base_network = deepcopy(base["network"])
                    base_network["num_nodes"] = int(num_nodes)
                    for network_type in scan["network_types"]:
                        network_override = _network_override_for_type(base_network, network_type)
                        network_label = "n{0}_{1}".format(num_nodes, _network_variant_label(network_override))
                        for r_value in scan["r_values"]:
                            experiment_name = "{0}__{1}__{2}__{3}__r{4}".format(
                                network_label,
                                run_mode,
                                consumption_label,
                                strategy_update_rule,
                                _format_float_token(r_value),
                            )
                            spec = deep_update(
                                deepcopy(base),
                                {
                                    "experiment_name": experiment_name,
                                    "run_mode": run_mode,
                                    "network": network_override,
                                    "dynamics": {
                                        "r": float(r_value),
                                        "resource_consumption_mode": resource_consumption_mode,
                                        "strategy_update_rule": strategy_update_rule,
                                    },
                                    "output": {
                                        "root_dir": scan["output_root_dir"],
                                    },
                                },
                            )
                            if resource_consumption_fixed_mode is not None:
                                spec["dynamics"]["resource_consumption_fixed_mode"] = resource_consumption_fixed_mode
                            spec["scan_tags"] = {
                                "scan_name": scan["name"],
                                "run_mode": run_mode,
                                "num_nodes": int(num_nodes),
                                "network_label": network_label,
                                "consumption_label": consumption_label,
                                "resource_consumption_mode": resource_consumption_mode,
                                "resource_consumption_fixed_mode": resource_consumption_fixed_mode,
                                "strategy_update_rule": strategy_update_rule,
                                "r": float(r_value),
                            }
                            specs.append(spec)

    return specs


def build_experiment_specs() -> List[Dict[str, Any]]:
    if SCAN_EXPERIMENT["enabled"]:
        return build_scan_experiment_specs()

    base = deepcopy(BASE_EXPERIMENT)
    if not BATCH_EXPERIMENTS:
        return [base]

    specs: List[Dict[str, Any]] = []
    for override in BATCH_EXPERIMENTS:
        specs.append(deep_update(deepcopy(base), override))
    return specs


def build_graph(spec: Mapping[str, Any]) -> Dict[int, List[int]]:
    network = spec["network"]
    network_type = network["type"]
    seed = spec["seed"]

    if network_type == "regular":
        return make_random_regular_graph(
            num_nodes=network["num_nodes"],
            degree=network["regular_degree"],
            seed=seed,
        )
    if network_type == "erdos_renyi":
        return make_erdos_renyi_graph(
            num_nodes=network["num_nodes"],
            edge_prob=_resolve_er_edge_prob(network),
            seed=seed,
        )
    if network_type == "small_world":
        return make_watts_strogatz_graph(
            num_nodes=network["num_nodes"],
            degree=network["ws_degree"],
            rewiring_prob=network["ws_rewiring_prob"],
            seed=seed,
        )
    if network_type == "scale_free":
        return make_barabasi_albert_graph(
            num_nodes=network["num_nodes"],
            attachments_per_new_node=network["ba_attachments_per_new_node"],
            seed=seed,
        )
    if network_type == "grid":
        return make_grid_graph(
            rows=network["grid_rows"],
            cols=network["grid_cols"],
            periodic=network["grid_periodic"],
        )
    raise ValueError("Unsupported network type: {0}".format(network_type))


def build_env_config(spec: Mapping[str, Any]) -> SPGGConfig:
    dynamics = spec["dynamics"]
    reward = spec["reward"]
    return SPGGConfig(
        alpha=dynamics["alpha"],
        r=dynamics["r"],
        p_max=dynamics["p_max"],
        resource_consumption_mode=dynamics.get("resource_consumption_mode", "fixed"),
        resource_consumption_fixed_mode=dynamics.get("resource_consumption_fixed_mode", "constant"),
        resource_consumption_fixed=dynamics.get("resource_consumption_fixed", 0.0),
        resource_consumption_degree_multiplier=dynamics.get("resource_consumption_degree_multiplier", 0.0),
        resource_consumption_rate=dynamics.get("resource_consumption_rate", 0.0),
        resource_consumption_threshold=dynamics.get("resource_consumption_threshold", 0.0),
        strategy_update_rule=dynamics.get("strategy_update_rule", "fermi"),
        beta=dynamics["beta"],
        q_learning_rate=dynamics.get("q_learning_rate", 0.1),
        q_learning_discount=dynamics.get("q_learning_discount", 0.95),
        q_learning_epsilon=dynamics.get("q_learning_epsilon", 0.05),
        q_learning_initial_value=dynamics.get("q_learning_initial_value", 0.0),
        episode_length=dynamics["episode_length"],
        initial_resource=dynamics["initial_resource"],
        initial_cooperation_prob=dynamics["initial_cooperation_prob"],
        reward=RewardConfig(
            lambda_payoff=reward["lambda_payoff"],
            lambda_cooperation=reward["lambda_cooperation"],
            lambda_gini=reward["lambda_gini"],
            epsilon=reward["epsilon"],
        ),
    )


def build_gnn_policy(spec: Mapping[str, Any]) -> Any:
    from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig

    gnn = spec["gnn"]
    return GNNAllocationPolicy(
        GNNPolicyConfig(
            hidden_dim=gnn["hidden_dim"],
            num_message_passing_layers=gnn["num_message_passing_layers"],
            temperature=gnn["temperature"],
            dirichlet_concentration_scale=gnn["dirichlet_concentration_scale"],
            dirichlet_concentration_floor=gnn["dirichlet_concentration_floor"],
        )
    )


def build_trainer_config(spec: Mapping[str, Any]) -> Any:
    from Project1.trainer import TrainerConfig

    training = spec["training"]
    return TrainerConfig(
        total_updates=training["total_updates"],
        steps_per_update=training["steps_per_update"],
        gamma=training["gamma"],
        gae_lambda=training["gae_lambda"],
        learning_rate=training["learning_rate"],
        entropy_coef=training["entropy_coef"],
        value_coef=training["value_coef"],
        max_grad_norm=training["max_grad_norm"],
        eval_interval=training["eval_interval"],
        eval_episodes=training["eval_episodes"],
        device=training["device"],
        seed=spec["seed"],
    )


def graph_summary(graph: Mapping[int, Sequence[int]]) -> Dict[str, float]:
    degrees = [len(neighbors) for neighbors in graph.values()]
    num_edges = sum(degrees) // 2
    return {
        "num_nodes": float(len(graph)),
        "num_edges": float(num_edges),
        "degree_min": float(min(degrees) if degrees else 0),
        "degree_max": float(max(degrees) if degrees else 0),
        "degree_mean": float(mean(degrees) if degrees else 0.0),
    }


def print_header(spec: Mapping[str, Any], graph: Mapping[int, Sequence[int]], env_config: SPGGConfig) -> None:
    summary = graph_summary(graph)
    print("=" * 80)
    print("Experiment: {0}".format(spec["experiment_name"]))
    print("Run mode  : {0}".format(spec["run_mode"]))
    print("Network   : {0}".format(spec["network"]["type"]))
    if spec["network"]["type"] == "erdos_renyi":
        print("ER Params : {0}".format(_er_parameter_summary(spec["network"])))
    print(
        "Graph     : nodes={0}, edges={1}, degree_min={2:.0f}, degree_max={3:.0f}, degree_mean={4:.3f}".format(
            int(summary["num_nodes"]),
            int(summary["num_edges"]),
            summary["degree_min"],
            summary["degree_max"],
            summary["degree_mean"],
        )
    )
    print(
        "Dynamics  : alpha={0}, r={1}, p_max={2}, episode_length={3}".format(
            env_config.alpha,
            env_config.r,
            env_config.p_max,
            env_config.episode_length,
        )
    )
    if env_config.resource_consumption_mode == "fixed":
        if env_config.resource_consumption_fixed_mode == "degree_scaled":
            print(
                "Consume   : mode=fixed, fixed_mode=degree_scaled, degree_multiplier={0}".format(
                    env_config.resource_consumption_degree_multiplier
                )
            )
        else:
            print(
                "Consume   : mode=fixed, fixed_mode=constant, fixed={0}".format(
                    env_config.resource_consumption_fixed
                )
            )
    elif env_config.resource_consumption_mode == "proportional":
        print("Consume   : mode=proportional, rate={0}".format(env_config.resource_consumption_rate))
    else:
        if env_config.resource_consumption_fixed_mode == "degree_scaled":
            print(
                "Consume   : mode=piecewise_linear, fixed_mode=degree_scaled, degree_multiplier={0}, rate={1}, threshold={2}".format(
                    env_config.resource_consumption_degree_multiplier,
                    env_config.resource_consumption_rate,
                    env_config.resource_consumption_threshold,
                )
            )
        else:
            print(
                "Consume   : mode=piecewise_linear, fixed_mode=constant, fixed={0}, rate={1}, threshold={2}".format(
                    env_config.resource_consumption_fixed,
                    env_config.resource_consumption_rate,
                    env_config.resource_consumption_threshold,
                )
            )
    if env_config.strategy_update_rule == "fermi":
        print("Strategy  : rule=fermi, beta={0}".format(env_config.beta))
    elif env_config.strategy_update_rule == "q_learning":
        print(
            "Strategy  : rule=q_learning, state=none, lr={0}, gamma={1}, epsilon={2}, q0={3}".format(
                env_config.q_learning_rate,
                env_config.q_learning_discount,
                env_config.q_learning_epsilon,
                env_config.q_learning_initial_value,
            )
        )
    elif env_config.strategy_update_rule == "q_learning_2x2":
        print(
            "Strategy  : rule=q_learning_2x2, state=last_action, lr={0}, gamma={1}, epsilon={2}, q0={3}".format(
                env_config.q_learning_rate,
                env_config.q_learning_discount,
                env_config.q_learning_epsilon,
                env_config.q_learning_initial_value,
            )
        )
    else:
        print("Strategy  : rule=imitate_best")
    print(
        "Reward    : lambda_payoff={0}, lambda_cooperation={1}, lambda_gini={2}".format(
            env_config.reward.lambda_payoff,
            env_config.reward.lambda_cooperation,
            env_config.reward.lambda_gini,
        )
    )
    print("=" * 80)


def build_output_dir(spec: Mapping[str, Any]) -> Path:
    root_dir = Path(spec["output"]["root_dir"])
    experiment_dir = root_dir / spec["experiment_name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def should_visualize_episode(spec: Mapping[str, Any], episode_index: int) -> bool:
    selected = spec["visualization"]["episodes_to_visualize"]
    if not selected:
        return True
    return episode_index in selected


def capture_record(
    time_index: int,
    observation: Mapping[str, np.ndarray],
    reward: float = 0.0,
    info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resources = observation["resources"].copy()
    zeros = np.zeros_like(resources, dtype=np.float64)
    income = np.asarray(info["income"], dtype=np.float64).copy() if info is not None else zeros.copy()
    consumption = np.asarray(info["consumption"], dtype=np.float64).copy() if info is not None else zeros.copy()
    payoff = np.asarray(info["payoff"], dtype=np.float64).copy() if info is not None else zeros.copy()

    record = {
        "time": int(time_index),
        "x_nominal": observation["x_nominal"].copy(),
        "x_actual": observation["x_actual"].copy(),
        "resources": resources,
        "investment": observation["investment"].copy(),
        "pool_raw": observation["pool_raw"].copy(),
        "pool_grown": observation["pool_grown"].copy(),
        "income": income,
        "consumption": consumption,
        "payoff": payoff,
        "reward": float(reward),
        "gini": float(info["gini"]) if info is not None else gini_coefficient(resources),
        "actual_cooperation_rate": float(np.mean(observation["x_actual"])),
        "nominal_cooperation_rate": float(np.mean(observation["x_nominal"])),
        "mean_resource": float(np.mean(resources)),
        "mean_pool_raw": float(np.mean(observation["pool_raw"])),
        "mean_pool_grown": float(np.mean(observation["pool_grown"])),
        "mean_investment": float(np.mean(observation["investment"])),
        "mean_income": float(np.mean(income)),
        "mean_consumption": float(np.mean(consumption)),
        "mean_payoff": float(np.mean(payoff)),
    }
    return record


def save_visualizations_for_history(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    history: Sequence[Dict[str, Any]],
    output_dir: Path,
    episode_index: int,
    phase_name: str,
) -> None:
    visualization = spec["visualization"]
    output = spec["output"]
    if not visualization["enable_micro_snapshots"] and not visualization["enable_macro_timeseries"]:
        return
    if not should_visualize_episode(spec, episode_index):
        return

    from Project1.visualization import build_layout, save_macro_timeseries, save_network_snapshots

    if visualization["enable_micro_snapshots"] and output["save_micro_snapshots"]:
        network = spec["network"]
        grid_shape = None
        if network["type"] == "grid":
            grid_shape = (network["grid_rows"], network["grid_cols"])

        layout = build_layout(
            graph,
            layout_name=visualization["layout"],
            seed=spec["seed"],
            spring_iterations=visualization["spring_iterations"],
            grid_shape=grid_shape,
        )
        micro_dir = output_dir / "{0}_episode_{1:03d}_micro".format(phase_name, episode_index)
        save_network_snapshots(
            graph=graph,
            history=history,
            output_dir=micro_dir,
            positions=layout,
            node_color_metric=visualization["node_color_metric"],
            node_size_metric=visualization["node_size_metric"],
            node_value_metric=visualization.get("node_value_metric"),
            node_value_decimals=int(visualization.get("node_value_decimals", 1)),
            frame_stride=visualization["frame_stride"],
            dpi=visualization["save_dpi"],
            label_nodes=visualization["label_nodes"],
            title_prefix="{0} | {1} | ep={2:03d} | ".format(spec["experiment_name"], phase_name, episode_index),
        )

    if visualization["enable_macro_timeseries"] and output["save_macro_timeseries"]:
        macro_path = output_dir / "{0}_episode_{1:03d}_macro.png".format(phase_name, episode_index)
        save_macro_timeseries(
            history=history,
            output_path=macro_path,
            metrics=visualization["macro_metrics"],
            title="{0} | {1} | episode {2:03d}".format(spec["experiment_name"], phase_name, episode_index),
            dpi=visualization["save_dpi"],
        )


def summarize_episode(history: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    final_record = history[-1]
    return {
        "final_actual_cooperation": float(final_record["actual_cooperation_rate"]),
        "final_mean_resource": float(final_record["mean_resource"]),
        "final_mean_pool_grown": float(final_record["mean_pool_grown"]),
        "final_mean_consumption": float(final_record["mean_consumption"]),
        "final_mean_payoff": float(final_record["mean_payoff"]),
        "final_gini": float(final_record["gini"]),
    }


def summarize_rule_based_episodes(
    episode_summaries: Sequence[Mapping[str, float]],
    episode_returns: Sequence[float],
) -> Dict[str, float]:
    final_actual_cooperation_mean = float(np.mean([item["final_actual_cooperation"] for item in episode_summaries]))
    final_mean_resource_mean = float(np.mean([item["final_mean_resource"] for item in episode_summaries]))
    final_mean_pool_grown_mean = float(np.mean([item["final_mean_pool_grown"] for item in episode_summaries]))
    final_mean_consumption_mean = float(np.mean([item["final_mean_consumption"] for item in episode_summaries]))
    final_mean_payoff_mean = float(np.mean([item["final_mean_payoff"] for item in episode_summaries]))
    final_gini_mean = float(np.mean([item["final_gini"] for item in episode_summaries]))

    return {
        "return_mean": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "final_cooperation_mean": final_actual_cooperation_mean,
        "final_actual_cooperation_mean": final_actual_cooperation_mean,
        "final_mean_resource_mean": final_mean_resource_mean,
        "final_mean_pool_grown_mean": final_mean_pool_grown_mean,
        "final_consumption_mean": final_mean_consumption_mean,
        "final_mean_consumption_mean": final_mean_consumption_mean,
        "final_mean_payoff_mean": final_mean_payoff_mean,
        "final_gini_mean": final_gini_mean,
    }


def run_rule_based_mode(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    env_config: SPGGConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    env = SPGGEnv(env_config, graph)
    run_mode = spec["run_mode"]
    rollout = spec["rollout"]

    if run_mode == "uniform":
        policy = UniformAllocationPolicy()
    elif run_mode == "proportional":
        policy = ProportionalContributionPolicy()
    else:
        raise ValueError("Unsupported rule-based run_mode: {0}".format(run_mode))

    episode_summaries: List[Dict[str, float]] = []
    episode_returns: List[float] = []

    for episode_index in range(1, rollout["episodes"] + 1):
        observation = env.reset(seed=spec["seed"] + episode_index - 1)
        done = False
        episode_return = 0.0
        time_index = 0
        history = [capture_record(time_index, observation)]

        while not done:
            allocation = policy.allocate(observation)
            observation, reward, done, info = env.step(allocation)
            time_index += 1
            episode_return += reward
            history.append(capture_record(time_index, observation, reward=reward, info=info))

        summary = summarize_episode(history)
        summary["episode_return"] = float(episode_return)
        summary["episode_index"] = float(episode_index)

        episode_summaries.append(summary)
        episode_returns.append(episode_return)

        print(
            "[Episode {0:03d}] return={1:.6f}, final_actual_cooperation={2:.6f}, final_gini={3:.6f}".format(
                episode_index,
                episode_return,
                summary["final_actual_cooperation"],
                summary["final_gini"],
            )
        )

        save_visualizations_for_history(
            spec=spec,
            graph=graph,
            history=history,
            output_dir=output_dir,
            episode_index=episode_index,
            phase_name="rollout",
        )

    return {
        "experiment_name": spec["experiment_name"],
        "run_mode": run_mode,
        "network_type": spec["network"]["type"],
        "episode_summaries": episode_summaries,
        "summary": summarize_rule_based_episodes(episode_summaries, episode_returns),
    }


def run_trained_policy_evaluation(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    env_config: SPGGConfig,
    policy: Any,
    output_dir: Path,
) -> List[Dict[str, float]]:
    import torch

    eval_env = SPGGEnv(env_config, graph)
    episode_summaries: List[Dict[str, float]] = []
    rollout = spec["rollout"]

    for episode_index in range(1, rollout["post_training_eval_episodes"] + 1):
        observation = eval_env.reset(seed=spec["seed"] + 10_000 + episode_index)
        done = False
        episode_return = 0.0
        time_index = 0
        history = [capture_record(time_index, observation)]

        while not done:
            with torch.no_grad():
                action_output = policy.deterministic_action(observation)
            allocation = action_output.allocation_matrix.detach().cpu().numpy()
            observation, reward, done, info = eval_env.step(allocation)
            time_index += 1
            episode_return += reward
            history.append(capture_record(time_index, observation, reward=reward, info=info))

        summary = summarize_episode(history)
        summary["episode_return"] = float(episode_return)
        summary["episode_index"] = float(episode_index)
        episode_summaries.append(summary)

        print(
            "[Post-Train Eval {0:03d}] return={1:.6f}, final_actual_cooperation={2:.6f}, final_gini={3:.6f}".format(
                episode_index,
                episode_return,
                summary["final_actual_cooperation"],
                summary["final_gini"],
            )
        )

        save_visualizations_for_history(
            spec=spec,
            graph=graph,
            history=history,
            output_dir=output_dir,
            episode_index=episode_index,
            phase_name="post_train_eval",
        )

    return episode_summaries


def run_gnn_training_mode(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    env_config: SPGGConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    from Project1.trainer import CentralizedActorCriticTrainer

    env = SPGGEnv(env_config, graph)
    eval_env = SPGGEnv(env_config, graph)
    policy = build_gnn_policy(spec)
    trainer_config = build_trainer_config(spec)

    trainer = CentralizedActorCriticTrainer(
        env=env,
        policy=policy,
        eval_env=eval_env,
        config=trainer_config,
    )

    history = trainer.train(num_updates=spec["training"]["total_updates"])
    for item in history:
        summary_text = (
            "[Update {0:03d}] loss={1:.6f}, policy_loss={2:.6f}, value_loss={3:.6f}, entropy={4:.6f}, mean_rollout_reward={5:.6f}".format(
                int(item["update"]),
                item["loss"],
                item["policy_loss"],
                item["value_loss"],
                item["entropy"],
                item["mean_rollout_reward"],
            )
        )
        if "eval_return_mean" in item:
            summary_text += ", eval_return_mean={0:.6f}, eval_cooperation_mean={1:.6f}, eval_gini_mean={2:.6f}".format(
                item["eval_return_mean"],
                item["eval_cooperation_mean"],
                item["eval_gini_mean"],
            )
        print(summary_text)

    evaluation_summaries = run_trained_policy_evaluation(
        spec=spec,
        graph=graph,
        env_config=env_config,
        policy=policy,
        output_dir=output_dir,
    )

    return {
        "experiment_name": spec["experiment_name"],
        "run_mode": spec["run_mode"],
        "network_type": spec["network"]["type"],
        "trainer_config": asdict(trainer_config),
        "history": history,
        "post_training_evaluation": evaluation_summaries,
        "final_metrics": history[-1] if history else {},
    }


def save_results_json(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    env_config: SPGGConfig,
    results: Mapping[str, Any],
    output_dir: Path,
) -> None:
    if not spec["output"]["save_results_json"]:
        return

    payload = {
        "experiment": spec,
        "graph_summary": graph_summary(graph),
        "env_config": asdict(env_config),
        "results": results,
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Results saved to: {0}".format(output_path))


def print_final_summary(results: Mapping[str, Any]) -> None:
    summary = results.get("summary")
    if summary is None:
        summary = results.get("final_metrics", {})
    print("Final summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_one_experiment(spec: Mapping[str, Any]) -> Dict[str, Any]:
    np.random.seed(spec["seed"])

    graph = build_graph(spec)
    env_config = build_env_config(spec)
    output_dir = build_output_dir(spec)
    print_header(spec, graph, env_config)

    if spec["run_mode"] in {"uniform", "proportional"}:
        results = run_rule_based_mode(spec, graph, env_config, output_dir)
    elif spec["run_mode"] == "gnn_train":
        results = run_gnn_training_mode(spec, graph, env_config, output_dir)
    else:
        raise ValueError("Unsupported run_mode: {0}".format(spec["run_mode"]))

    print_final_summary(results)
    save_results_json(spec, graph, env_config, results, output_dir)
    return results


def _run_scan_experiment_worker(spec: Mapping[str, Any]) -> Dict[str, Any]:
    results = run_one_experiment(spec)
    return _scan_record_from_results(spec, results)


def _scan_record_from_results(spec: Mapping[str, Any], results: Mapping[str, Any]) -> Dict[str, Any]:
    summary = results["summary"]
    scan_tags = spec["scan_tags"]
    return {
        "experiment_name": spec["experiment_name"],
        "run_mode": scan_tags["run_mode"],
        "num_nodes": int(scan_tags["num_nodes"]),
        "network_type": spec["network"]["type"],
        "network_label": scan_tags["network_label"],
        "resource_consumption_mode": scan_tags["resource_consumption_mode"],
        "resource_consumption_fixed_mode": scan_tags["resource_consumption_fixed_mode"],
        "consumption_label": scan_tags["consumption_label"],
        "strategy_update_rule": scan_tags["strategy_update_rule"],
        "r": float(scan_tags["r"]),
        "final_actual_cooperation_mean": float(summary["final_actual_cooperation_mean"]),
        "final_mean_resource_mean": float(summary["final_mean_resource_mean"]),
        "final_mean_pool_grown_mean": float(summary["final_mean_pool_grown_mean"]),
        "final_mean_consumption_mean": float(summary["final_mean_consumption_mean"]),
        "final_mean_payoff_mean": float(summary["final_mean_payoff_mean"]),
        "final_gini_mean": float(summary["final_gini_mean"]),
        "return_mean": float(summary["return_mean"]),
        "return_std": float(summary["return_std"]),
    }


def _save_scan_summary_tables(
    output_root: Path,
    scan_records: Sequence[Mapping[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "scan_summary.json"
    json_path.write_text(json.dumps(list(scan_records), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Scan summary saved to: {0}".format(json_path))

    csv_path = output_root / "scan_summary.csv"
    fieldnames = [
        "experiment_name",
        "run_mode",
        "num_nodes",
        "network_type",
        "network_label",
        "resource_consumption_mode",
        "resource_consumption_fixed_mode",
        "consumption_label",
        "strategy_update_rule",
        "r",
        "final_actual_cooperation_mean",
        "final_mean_resource_mean",
        "final_mean_pool_grown_mean",
        "final_mean_consumption_mean",
        "final_mean_payoff_mean",
        "final_gini_mean",
        "return_mean",
        "return_std",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in scan_records:
            writer.writerow({field: record.get(field) for field in fieldnames})
    print("Scan CSV saved to: {0}".format(csv_path))


def _save_scan_steady_state_plots(
    output_root: Path,
    scan_records: Sequence[Mapping[str, Any]],
) -> None:
    from Project1.visualization import save_scan_metric_grid

    metrics = [
        "final_actual_cooperation_mean",
        "final_mean_resource_mean",
        "final_mean_pool_grown_mean",
        "final_mean_consumption_mean",
        "final_mean_payoff_mean",
        "final_gini_mean",
    ]

    grouped_records: Dict[tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for record in scan_records:
        group_key = (
            str(record["run_mode"]),
            str(record["strategy_update_rule"]),
            str(record["consumption_label"]),
        )
        grouped_records.setdefault(group_key, []).append(record)

    steady_state_dir = output_root / "steady_state_vs_r"
    steady_state_dir.mkdir(parents=True, exist_ok=True)

    for (run_mode, strategy_update_rule, consumption_label), records in grouped_records.items():
        output_path = steady_state_dir / "{0}__{1}__{2}__steady_state_vs_r.png".format(
            run_mode,
            strategy_update_rule,
            consumption_label,
        )
        title = "Steady-state vs r | run_mode={0} | strategy={1} | consumption={2}".format(
            run_mode,
            strategy_update_rule,
            consumption_label,
        )
        save_scan_metric_grid(
            records=records,
            output_path=output_path,
            metrics=metrics,
            title=title,
            dpi=int(BASE_EXPERIMENT["visualization"]["save_dpi"]),
        )


def run_scan_experiments() -> None:
    specs = build_scan_experiment_specs()
    output_root = Path(SCAN_EXPERIMENT["output_root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scan_config": deepcopy(SCAN_EXPERIMENT),
        "base_experiment_name": BASE_EXPERIMENT["experiment_name"],
        "base_run_mode": BASE_EXPERIMENT["run_mode"],
        "num_experiments": len(specs),
    }
    manifest_path = output_root / "scan_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Scan manifest saved to: {0}".format(manifest_path))

    scan_records: List[Dict[str, Any]] = []
    if SCAN_EXPERIMENT["parallel"]:
        requested_workers = SCAN_EXPERIMENT["max_workers"]
        cpu_count = os.cpu_count() or 1
        max_workers = cpu_count if requested_workers in (None, 0) else int(requested_workers)
        max_workers = max(1, min(max_workers, len(specs)))
        print("Running scan in parallel with {0} worker processes.".format(max_workers))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {
                executor.submit(_run_scan_experiment_worker, spec): (index, spec["experiment_name"])
                for index, spec in enumerate(specs, start=1)
            }
            for completed_count, future in enumerate(as_completed(future_to_meta), start=1):
                index, experiment_name = future_to_meta[future]
                print(
                    "[Scan completed {0}/{1}] #{2} {3}".format(
                        completed_count,
                        len(specs),
                        index,
                        experiment_name,
                    )
                )
                scan_records.append(future.result())
    else:
        for index, spec in enumerate(specs, start=1):
            print("\n" + "#" * 80)
            print("Scan experiment {0}/{1}".format(index, len(specs)))
            print("#" * 80)
            results = run_one_experiment(spec)
            scan_records.append(_scan_record_from_results(spec, results))

    scan_records.sort(
        key=lambda item: (
            str(item["run_mode"]),
            str(item["strategy_update_rule"]),
            str(item["consumption_label"]),
            str(item["network_label"]),
            float(item["r"]),
        )
    )
    _save_scan_summary_tables(output_root, scan_records)
    _save_scan_steady_state_plots(output_root, scan_records)


def main() -> None:
    if SCAN_EXPERIMENT["enabled"]:
        run_scan_experiments()
        return

    specs = build_experiment_specs()
    for index, spec in enumerate(specs, start=1):
        if len(specs) > 1:
            print("\n" + "#" * 80)
            print("Batch experiment {0}/{1}".format(index, len(specs)))
            print("#" * 80)
        run_one_experiment(spec)


if __name__ == "__main__":
    main()
