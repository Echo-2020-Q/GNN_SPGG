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

from analyze_gnn_policy_decisions import (
    _load_json,
    build_env_config_from_spec,
    build_graph_from_spec,
    canonicalize_topology_name,
    parse_topology_list,
    resolve_reference_mean_degree,
)
from Project1.env import SPGGEnv
from Project1.policies.rule_based import UniformAllocationPolicy
from training_graph_bank_utils import build_topology_spec_for_analysis, normalize_graph_source_mode
import plot_policy_dynamics_over_topologies as pp


SCRIPT_DEFAULTS = {
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    "output_dir": None,
    "output_subdir_name": "0820_200_all_cooperation_Gini",
    "topologies": "regular,scale_free,erdos_renyi,small_world",
    "episodes": 10,
    "max_steps": 200,
    "episode_length_override": 200,
    # None 表示沿用 run_dir/results.json 中的环境参数。
    "env_r_override": None,
    "env_p_c_override": None,
    "env_p_max_override": None,
    "env_initial_resource_override": None,
    "graph_source_mode": "random_per_episode",
    # True：使用下方 EMBEDDED_REFERENCE_GRAPH_SEEDS 中内置的 40 张地图。
    "use_embedded_reference_graphs": True,
    # 指定后按 CSV 中的 topology + episode 精确重放地图种子和建图参数。
    # 非 None 时优先级高于 use_embedded_reference_graphs。
    "graph_metadata_csv": None,
    # 仅在 graph_source_mode=training_fixed_graph_bank 时使用。
    "graph_bank_index": 0,
    "time_axis_scale": "symlog",
    "metric_axis_scale": "linear",
    "gini_average_tail_fraction": 0.05,
    "seed": None,
    "figsize": (8.0, 5.33),
    # 与 plot_policy_dynamics_over_topologies.py 的 triptych 主图字号保持一致。
    "title_fontsize": 14.0,
    "axis_label_fontsize": 16.0,
    "tick_labelsize": 14.0,
    "legend_fontsize": 14.0,
    "legend_ncol": 1,
    "legend_loc": "upper center",
    "legend_bbox_x": 0.51,
    "legend_bbox_y": 0.97,
    "marker_count": 10,
    "marker_size": 5.5,
}


# 来自用户提供的 episode_graph_metadata.csv。每种拓扑对应 episode 0..9。
EMBEDDED_REFERENCE_GRAPH_SEEDS = {
    "regular": (3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064),
    "scale_free": (5698, 5699, 5700, 5701, 5702, 5703, 5704, 5705, 5706, 5707),
    "erdos_renyi": (7221, 7222, 7223, 7224, 7225, 7226, 7227, 7228, 7229, 7230),
    "small_world": (7135, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144),
}
EMBEDDED_REFERENCE_NETWORK_PARAMETERS = {
    "graph_source_mode": "random_per_episode",
    "graph_num_nodes": 50,
    "regular_degree": 4,
    "er_target_mean_degree": 4.0,
    "ws_degree": 4,
    "ws_rewiring_prob": 0.1,
    "ba_attachments_per_new_node": 2,
}


TOPOLOGY_ORDER = ("regular", "scale_free", "erdos_renyi", "small_world")
TOPOLOGY_LABELS = {
    "regular": "Regular",
    "scale_free": "BA",
    "erdos_renyi": "ER",
    "small_world": "WS",
}
TOPOLOGY_STYLES = {
    "regular": {"color": "#88CCEE", "linestyle": "-", "marker": "o"},
    "erdos_renyi": {"color": "#CC6677", "linestyle": "--", "marker": "s"},
    "small_world": {"color": "#44AA99", "linestyle": "-.", "marker": "^"},
    "scale_free": {"color": "#DDCC77", "linestyle": ":", "marker": "D"},
}


class AllCooperationSPGGEnv(SPGGEnv):
    """SPGG environment whose nominal strategy is fixed to cooperation."""

    def _update_nominal_strategies(
        self,
        nominal_strategies: np.ndarray,
        actual_strategies: np.ndarray,
        payoff: np.ndarray,
    ) -> np.ndarray:
        del actual_strategies, payoff
        return np.ones_like(nominal_strategies, dtype=np.int8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure resource Gini dynamics over network topologies when every individual "
            "always chooses cooperation and every pool allocates uniformly."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=Path(str(SCRIPT_DEFAULTS["run_dir"])))
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DEFAULTS["output_dir"])
    parser.add_argument("--output-subdir-name", type=str, default=str(SCRIPT_DEFAULTS["output_subdir_name"]))
    parser.add_argument("--topologies", type=str, default=str(SCRIPT_DEFAULTS["topologies"]))
    parser.add_argument("--episodes", type=int, default=int(SCRIPT_DEFAULTS["episodes"]))
    parser.add_argument("--max-steps", type=int, default=int(SCRIPT_DEFAULTS["max_steps"]))
    parser.add_argument(
        "--episode-length-override",
        type=int,
        default=int(SCRIPT_DEFAULTS["episode_length_override"]),
    )
    parser.add_argument("--env-r", dest="env_r_override", type=float, default=SCRIPT_DEFAULTS["env_r_override"])
    parser.add_argument(
        "--env-p-c",
        dest="env_p_c_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_p_c_override"],
    )
    parser.add_argument(
        "--env-p-max",
        dest="env_p_max_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_p_max_override"],
    )
    parser.add_argument(
        "--env-initial-resource",
        dest="env_initial_resource_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_initial_resource_override"],
    )
    parser.add_argument(
        "--graph-source-mode",
        type=str,
        default=str(SCRIPT_DEFAULTS["graph_source_mode"]),
        help="training_fixed_graph_bank, spec_seed, or random_per_episode.",
    )
    parser.add_argument(
        "--graph-metadata-csv",
        type=Path,
        default=(
            None
            if SCRIPT_DEFAULTS["graph_metadata_csv"] is None
            else Path(str(SCRIPT_DEFAULTS["graph_metadata_csv"]))
        ),
        help=(
            "Replay graph_seed and network parameters by topology + episode from an "
            "episode_graph_metadata.csv file. This takes precedence over --graph-source-mode."
        ),
    )
    parser.add_argument(
        "--use-embedded-reference-graphs",
        dest="use_embedded_reference_graphs",
        action="store_true",
        help="Replay the graph seeds and network parameters embedded in this source file.",
    )
    parser.add_argument(
        "--no-embedded-reference-graphs",
        dest="use_embedded_reference_graphs",
        action="store_false",
        help="Disable embedded graph replay and use --graph-source-mode instead.",
    )
    parser.set_defaults(
        use_embedded_reference_graphs=bool(SCRIPT_DEFAULTS["use_embedded_reference_graphs"])
    )
    parser.add_argument(
        "--graph-bank-index",
        type=int,
        default=int(SCRIPT_DEFAULTS["graph_bank_index"]),
        help="Used only when --graph-source-mode=training_fixed_graph_bank.",
    )
    parser.add_argument(
        "--time-axis-scale",
        type=str,
        choices=("linear", "symlog", "log"),
        default=str(SCRIPT_DEFAULTS["time_axis_scale"]),
    )
    parser.add_argument(
        "--metric-axis-scale",
        type=str,
        choices=("linear", "symlog", "log"),
        default=str(SCRIPT_DEFAULTS["metric_axis_scale"]),
    )
    parser.add_argument(
        "--gini-average-tail-fraction",
        type=float,
        default=float(SCRIPT_DEFAULTS["gini_average_tail_fraction"]),
        help="Fraction of each episode's final time points used for Gini mean and std across episodes.",
    )
    parser.add_argument("--seed", type=int, default=SCRIPT_DEFAULTS["seed"])
    parser.add_argument("--figsize-width", type=float, default=float(SCRIPT_DEFAULTS["figsize"][0]))
    parser.add_argument("--figsize-height", type=float, default=float(SCRIPT_DEFAULTS["figsize"][1]))
    parser.add_argument("--title-fontsize", type=float, default=float(SCRIPT_DEFAULTS["title_fontsize"]))
    parser.add_argument(
        "--axis-label-fontsize",
        type=float,
        default=float(SCRIPT_DEFAULTS["axis_label_fontsize"]),
    )
    parser.add_argument("--tick-labelsize", type=float, default=float(SCRIPT_DEFAULTS["tick_labelsize"]))
    parser.add_argument("--legend-fontsize", type=float, default=float(SCRIPT_DEFAULTS["legend_fontsize"]))
    parser.add_argument("--legend-ncol", type=int, default=int(SCRIPT_DEFAULTS["legend_ncol"]))
    parser.add_argument("--legend-loc", type=str, default=str(SCRIPT_DEFAULTS["legend_loc"]))
    parser.add_argument("--legend-bbox-x", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_x"]))
    parser.add_argument("--legend-bbox-y", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_y"]))
    parser.add_argument("--marker-count", type=int, default=int(SCRIPT_DEFAULTS["marker_count"]))
    parser.add_argument("--marker-size", type=float, default=float(SCRIPT_DEFAULTS["marker_size"]))
    return parser.parse_args()


def _ordered_topologies(raw_topologies: str) -> list[str]:
    selected = list(parse_topology_list(str(raw_topologies)))
    ordered = [name for name in TOPOLOGY_ORDER if name in selected]
    return ordered + [name for name in selected if name not in ordered]


def _topology_seed_offset(topology_name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(str(topology_name)))


def _optional_csv_number(
    row: Mapping[str, Any],
    key: str,
    *,
    number_type: type[int] | type[float],
) -> int | float | None:
    raw_value = row.get(key)
    if raw_value is None or str(raw_value).strip() == "":
        return None
    if number_type is int:
        return int(float(str(raw_value).strip()))
    return float(str(raw_value).strip())


def _load_graph_metadata_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Graph metadata CSV not found: {resolved_path}")

    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"topology", "episode", "graph_seed"}
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            raise KeyError(
                f"Graph metadata CSV is missing required columns: {sorted(missing_fields)}"
            )
        for csv_row in reader:
            topology_name = canonicalize_topology_name(str(csv_row["topology"]))
            episode_index = int(csv_row["episode"])
            key = (str(topology_name), int(episode_index))
            if key in rows_by_key:
                raise ValueError(
                    "Graph metadata CSV contains a duplicate topology/episode row: "
                    f"{topology_name}, episode={episode_index}"
                )
            normalized_row = dict(csv_row)
            normalized_row["topology"] = str(topology_name)
            normalized_row["episode"] = int(episode_index)
            normalized_row["graph_seed"] = int(csv_row["graph_seed"])
            rows_by_key[key] = normalized_row
    if not rows_by_key:
        raise ValueError(f"Graph metadata CSV contains no data rows: {resolved_path}")
    return rows_by_key


def _build_embedded_reference_graph_rows() -> dict[tuple[str, int], dict[str, Any]]:
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for topology_name, graph_seeds in EMBEDDED_REFERENCE_GRAPH_SEEDS.items():
        for episode_index, graph_seed in enumerate(graph_seeds):
            row = {
                "topology": str(topology_name),
                "topology_label": str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
                "episode": int(episode_index),
                "graph_seed": int(graph_seed),
                "graph_bank_index": None,
                "fixed_graph_bank_seed": None,
                "fixed_graph_bank_size_per_type": None,
                "fixed_graph_bank_sampling": None,
                **EMBEDDED_REFERENCE_NETWORK_PARAMETERS,
            }
            rows_by_key[(str(topology_name), int(episode_index))] = row
    return rows_by_key


def _apply_replayed_graph_metadata(
    spec: Mapping[str, Any],
    metadata_row: Mapping[str, Any],
    *,
    replay_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay_spec = deepcopy(dict(spec))
    replay_spec["seed"] = int(metadata_row["graph_seed"])
    network = dict(replay_spec["network"])
    replay_spec["network"] = network

    integer_fields = (
        "graph_num_nodes",
        "regular_degree",
        "ws_degree",
        "ba_attachments_per_new_node",
    )
    float_fields = ("er_target_mean_degree", "ws_rewiring_prob")
    network_key_by_csv_key = {
        "graph_num_nodes": "num_nodes",
        "regular_degree": "regular_degree",
        "er_target_mean_degree": "er_target_mean_degree",
        "ws_degree": "ws_degree",
        "ws_rewiring_prob": "ws_rewiring_prob",
        "ba_attachments_per_new_node": "ba_attachments_per_new_node",
    }
    for csv_key in integer_fields:
        value = _optional_csv_number(metadata_row, csv_key, number_type=int)
        if value is not None:
            network[network_key_by_csv_key[csv_key]] = int(value)
    for csv_key in float_fields:
        value = _optional_csv_number(metadata_row, csv_key, number_type=float)
        if value is not None:
            network[network_key_by_csv_key[csv_key]] = float(value)
    if network.get("er_target_mean_degree") is not None:
        network["er_edge_prob"] = float(network["er_target_mean_degree"]) / float(
            max(int(network["num_nodes"]) - 1, 1)
        )

    graph_metadata = {
        "topology": str(metadata_row["topology"]),
        "graph_replay_mode": str(replay_mode),
        "graph_source_mode": str(metadata_row.get("graph_source_mode", "random_per_episode")),
        "graph_seed": int(metadata_row["graph_seed"]),
        "graph_bank_index": _optional_csv_number(metadata_row, "graph_bank_index", number_type=int),
        "fixed_graph_bank_seed": _optional_csv_number(
            metadata_row, "fixed_graph_bank_seed", number_type=int
        ),
        "fixed_graph_bank_size_per_type": _optional_csv_number(
            metadata_row, "fixed_graph_bank_size_per_type", number_type=int
        ),
        "fixed_graph_bank_sampling": (
            None
            if str(metadata_row.get("fixed_graph_bank_sampling", "")).strip() == ""
            else str(metadata_row["fixed_graph_bank_sampling"])
        ),
    }
    return replay_spec, graph_metadata


def _bounded_gini(value: Any) -> float:
    """Remove floating-point noise outside the theoretical Gini range."""

    return float(np.clip(float(np.asarray(value).item()), 0.0, 1.0))


def _apply_environment_overrides(
    spec: Mapping[str, Any],
    *,
    episode_length_override: int,
    env_r_override: float | None,
    env_p_c_override: float | None,
    env_p_max_override: float | None,
    env_initial_resource_override: float | None,
) -> dict[str, Any]:
    effective_spec = deepcopy(dict(spec))
    dynamics = dict(effective_spec.get("dynamics", {}))
    if not dynamics:
        raise KeyError("Experiment spec does not contain a dynamics mapping.")

    overrides = {
        "r": env_r_override,
        "p_c": env_p_c_override,
        "p_max": env_p_max_override,
        "initial_resource": env_initial_resource_override,
    }
    for key, value in overrides.items():
        if value is not None:
            dynamics[key] = float(value)
    dynamics["episode_length"] = int(episode_length_override)
    # 仅用于记录语义；实际 reset 和每一步更新都由 AllCooperationSPGGEnv 强制合作。
    dynamics["initial_cooperation_prob"] = 1.0
    effective_spec["dynamics"] = dynamics
    return effective_spec


def _run_episode(
    *,
    env_config: Any,
    graph: Mapping[int, Sequence[int]],
    topology_name: str,
    episode_index: int,
    episode_seed: int,
    max_steps: int,
    graph_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = AllCooperationSPGGEnv(env_config, graph)
    policy = UniformAllocationPolicy()
    observation = env.reset(initial_strategies=1, seed=int(episode_seed))

    step_rows: list[dict[str, Any]] = []
    step_index = 0
    done = False
    minimum_actual_cooperation = float(np.mean(observation["x_actual"]))

    while not done and step_index < int(max_steps):
        if not np.all(np.asarray(observation["x_nominal"], dtype=np.int8) == 1):
            raise RuntimeError("All-cooperation invariant was violated before env.step().")

        allocation_matrix = np.asarray(policy.allocate(observation), dtype=np.float64)
        next_observation, _, env_done, info = env.step(allocation_matrix)
        nominal_cooperation_rate = float(np.mean(next_observation["x_nominal"]))
        if not np.isclose(nominal_cooperation_rate, 1.0):
            raise RuntimeError("All-cooperation invariant was violated after env.step().")

        actual_cooperation_rate = float(
            info.get("actual_cooperation_rate", np.mean(next_observation["x_actual"]))
        )
        minimum_actual_cooperation = min(minimum_actual_cooperation, actual_cooperation_rate)
        resources = np.asarray(next_observation["resources"], dtype=np.float64)
        step_rows.append(
            {
                "topology": str(topology_name),
                "topology_label": str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
                "episode": int(episode_index),
                # 与 analyze_five_policy_dynamics_grid.py 保持一致：step=0 表示第一次环境更新后的状态。
                "step": int(step_index),
                "resource_gini": _bounded_gini(info.get("gini", next_observation["gini"])),
                "mean_resource": float(np.mean(resources)),
                "total_resource": float(np.sum(resources)),
                "nominal_cooperation_rate": float(nominal_cooperation_rate),
                "actual_cooperation_rate": float(actual_cooperation_rate),
                "graph_source_mode": str(graph_metadata.get("graph_source_mode", "")),
                "graph_seed": int(graph_metadata.get("graph_seed", episode_seed)),
                "graph_bank_index": graph_metadata.get("graph_bank_index"),
            }
        )
        observation = next_observation
        step_index += 1
        done = bool(env_done)

    final_resources = np.asarray(observation["resources"], dtype=np.float64)
    episode_row = {
        "topology": str(topology_name),
        "topology_label": str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
        "episode": int(episode_index),
        "steps": int(step_index),
        "final_resource_gini": _bounded_gini(observation["gini"]),
        "final_mean_resource": float(np.mean(final_resources)),
        "final_total_resource": float(np.sum(final_resources)),
        "final_nominal_cooperation_rate": float(np.mean(observation["x_nominal"])),
        "final_actual_cooperation_rate": float(np.mean(observation["x_actual"])),
        "minimum_actual_cooperation_rate": float(minimum_actual_cooperation),
        "graph_source_mode": str(graph_metadata.get("graph_source_mode", "")),
        "graph_seed": int(graph_metadata.get("graph_seed", episode_seed)),
        "graph_bank_index": graph_metadata.get("graph_bank_index"),
    }
    return step_rows, episode_row


def _summarize_step_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["topology"]), int(row["step"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for topology_name in sorted({key[0] for key in grouped}, key=_topology_sort_key):
        steps = sorted(step for topology, step in grouped if topology == topology_name)
        for step in steps:
            bucket = grouped[(topology_name, step)]
            gini_values = np.asarray([float(row["resource_gini"]) for row in bucket], dtype=np.float64)
            resource_values = np.asarray([float(row["mean_resource"]) for row in bucket], dtype=np.float64)
            actual_cooperation = np.asarray(
                [float(row["actual_cooperation_rate"]) for row in bucket], dtype=np.float64
            )
            summary_rows.append(
                {
                    "topology": str(topology_name),
                    "topology_label": str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
                    "step": int(step),
                    "count_episodes": int(len(bucket)),
                    "resource_gini_mean": float(np.mean(gini_values)),
                    "resource_gini_std": float(np.std(gini_values, ddof=0)),
                    "resource_gini_min": float(np.min(gini_values)),
                    "resource_gini_max": float(np.max(gini_values)),
                    "mean_resource_mean": float(np.mean(resource_values)),
                    "actual_cooperation_rate_mean": float(np.mean(actual_cooperation)),
                    "nominal_cooperation_rate_mean": float(
                        np.mean([float(row["nominal_cooperation_rate"]) for row in bucket])
                    ),
                }
            )
    return summary_rows


def _summarize_topology_average_gini(
    step_summary_rows: Sequence[Mapping[str, Any]],
    *,
    step_rows: Sequence[Mapping[str, Any]],
    tail_fraction: float,
) -> list[dict[str, Any]]:
    """Summarize episode tail means, with population std across episodes (ddof=0)."""

    episode_rows_by_topology: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    for row in step_rows:
        episodes = episode_rows_by_topology.setdefault(str(row["topology"]), {})
        episodes.setdefault(int(row["episode"]), []).append(row)

    output_rows: list[dict[str, Any]] = []
    topology_names = sorted(
        {str(row["topology"]) for row in step_summary_rows},
        key=_topology_sort_key,
    )
    for topology_name in topology_names:
        topology_rows = [
            row for row in step_summary_rows if str(row["topology"]) == str(topology_name)
        ]
        topology_rows.sort(key=lambda row: int(row["step"]))
        tail_count = max(1, int(np.ceil(len(topology_rows) * float(tail_fraction))))
        tail_rows = topology_rows[-tail_count:]
        gini_values = np.asarray(
            [float(row["resource_gini_mean"]) for row in tail_rows],
            dtype=np.float64,
        )
        episode_counts = np.asarray(
            [int(row["count_episodes"]) for row in tail_rows],
            dtype=np.int64,
        )
        # 先对每轮末尾时间段取均值，再计算轮次之间的标准差；
        # 不同于下方 gini_over_time_std（平均曲线随时间的波动）。
        episode_tail_means: list[float] = []
        for episode_rows in episode_rows_by_topology[topology_name].values():
            episode_rows.sort(key=lambda row: int(row["step"]))
            episode_tail_count = max(1, int(np.ceil(len(episode_rows) * float(tail_fraction))))
            episode_tail_means.append(
                float(np.mean([
                    float(row["resource_gini"]) for row in episode_rows[-episode_tail_count:]
                ]))
            )
        output_rows.append(
            {
                "topology": str(topology_name),
                "topology_label": str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
                "total_time_points": int(len(topology_rows)),
                "tail_fraction": float(tail_fraction),
                "tail_time_points": int(gini_values.size),
                "tail_start_step": int(tail_rows[0]["step"]),
                "tail_end_step": int(tail_rows[-1]["step"]),
                "episodes_per_step_min": int(np.min(episode_counts)),
                "episodes_per_step_max": int(np.max(episode_counts)),
                "count_episodes": int(len(episode_tail_means)),
                "average_gini_level": float(np.mean(episode_tail_means)),
                "average_gini_std": float(np.std(episode_tail_means, ddof=0)),
                "gini_over_time_std": float(np.std(gini_values, ddof=0)),
                "gini_over_time_min": float(np.min(gini_values)),
                "gini_over_time_max": float(np.max(gini_values)),
            }
        )
    return output_rows


def _topology_sort_key(topology_name: str) -> tuple[int, str]:
    text = str(topology_name)
    try:
        return (TOPOLOGY_ORDER.index(text), text)
    except ValueError:
        return (len(TOPOLOGY_ORDER), text)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_even_markevery(x_values: np.ndarray, *, axis_scale: str, target_count: int) -> list[int]:
    x_array = np.asarray(x_values, dtype=np.float64).reshape(-1)
    if x_array.size <= 1:
        return [0]
    count = max(2, min(int(target_count), int(x_array.size)))
    if str(axis_scale) == "linear":
        transformed = x_array
    elif str(axis_scale) == "log":
        transformed = np.log10(np.clip(x_array, 1e-12, None))
    else:
        transformed = np.where(
            np.abs(x_array) <= 1.0,
            x_array,
            np.sign(x_array) * (1.0 + np.log10(np.clip(np.abs(x_array), 1e-12, None))),
        )
    targets = np.linspace(float(transformed[0]), float(transformed[-1]), num=count)
    indices = sorted({int(np.argmin(np.abs(transformed - target))) for target in targets})
    if 0 not in indices:
        indices.insert(0, 0)
    if int(x_array.size - 1) not in indices:
        indices.append(int(x_array.size - 1))
    return indices


def _plot_gini_dynamics(
    rows: Sequence[Mapping[str, Any]],
    *,
    topology_average_rows: Sequence[Mapping[str, Any]],
    topologies: Sequence[str],
    output_stem: Path,
    figsize: tuple[float, float],
    title_fontsize: float,
    axis_label_fontsize: float,
    tick_labelsize: float,
    legend_fontsize: float,
    time_axis_scale: str,
    metric_axis_scale: str,
    marker_count: int,
    marker_size: float,
    legend_ncol: int,
    legend_loc: str,
    legend_bbox_x: float,
    legend_bbox_y: float,
) -> tuple[Path | None, Path | None]:
    if plt is None:
        return None, None

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    average_gini_by_topology = {
        str(row["topology"]): float(row["average_gini_level"])
        for row in topology_average_rows
    }
    average_gini_std_by_topology = {
        str(row["topology"]): float(row["average_gini_std"])
        for row in topology_average_rows
    }
    tail_percent_by_topology = {
        str(row["topology"]): 100.0 * float(row["tail_fraction"])
        for row in topology_average_rows
    }
    for topology_name in topologies:
        topology_rows = [row for row in rows if str(row["topology"]) == str(topology_name)]
        topology_rows.sort(key=lambda row: int(row["step"]))
        if not topology_rows:
            continue
        x_values = np.asarray([int(row["step"]) for row in topology_rows], dtype=np.int64)
        mean_values = np.asarray([float(row["resource_gini_mean"]) for row in topology_rows], dtype=np.float64)
        std_values = np.asarray([float(row["resource_gini_std"]) for row in topology_rows], dtype=np.float64)
        style = TOPOLOGY_STYLES.get(str(topology_name), {})
        color = str(style.get("color", "#333333"))
        average_gini = average_gini_by_topology.get(str(topology_name), float("nan"))
        average_gini_std = average_gini_std_by_topology.get(str(topology_name), float("nan"))
        tail_percent = tail_percent_by_topology.get(str(topology_name), 5.0)
        topology_label = str(TOPOLOGY_LABELS.get(str(topology_name), topology_name))
        legend_label = (
            f"{topology_label} (last {tail_percent:g}%: {average_gini:.4f} ± {average_gini_std:.4f})"
            if np.isfinite(average_gini)
            else topology_label
        )
        axis.plot(
            x_values,
            mean_values,
            label=legend_label,
            color=color,
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", None),
            markersize=float(marker_size),
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            markevery=_build_even_markevery(
                x_values,
                axis_scale=str(time_axis_scale),
                target_count=max(2, int(marker_count)),
            ),
            linewidth=2.0,
        )
        if np.isfinite(average_gini):
            axis.axhline(
                average_gini,
                color=color,
                linestyle=(0, (3, 3)),
                linewidth=1.2,
                alpha=0.65,
                zorder=1,
            )
        if np.any(std_values > 0.0):
            lower_values = np.clip(mean_values - std_values, 0.0, 1.0)
            upper_values = np.clip(mean_values + std_values, 0.0, 1.0)
            finite_mask = (
                np.isfinite(x_values)
                & np.isfinite(lower_values)
                & np.isfinite(upper_values)
            )
            axis.fill_between(
                x_values,
                lower_values,
                upper_values,
                # Matplotlib 3.5.x may turn where=None into a scalar mask and
                # fail in contiguous_regions. Always provide a 1-D mask.
                where=finite_mask,
                color=color,
                alpha=0.14,
                linewidth=0.0,
            )

    resolved_time_scale = str(time_axis_scale)
    if resolved_time_scale == "log" and any(int(row["step"]) <= 0 for row in rows):
        resolved_time_scale = "symlog"
    if resolved_time_scale == "symlog":
        axis.set_xscale("symlog", linthresh=1.0)
    else:
        axis.set_xscale(resolved_time_scale)

    resolved_metric_scale = str(metric_axis_scale)
    if resolved_metric_scale == "log" and any(float(row["resource_gini_mean"]) <= 0.0 for row in rows):
        resolved_metric_scale = "symlog"
    if resolved_metric_scale == "symlog":
        axis.set_yscale("symlog", linthresh=1e-3)
    else:
        axis.set_yscale(resolved_metric_scale)
    if resolved_metric_scale == "linear":
        axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel(
        "t",
        fontsize=float(axis_label_fontsize),
        fontweight=pp.FONT_WEIGHT_BOLD,
        fontfamily=pp.FONT_FAMILY,
    )
    axis.set_ylabel(
        r"$\mathit{Gini}$",
        fontsize=float(axis_label_fontsize),
        fontweight=pp.FONT_WEIGHT_BOLD,
        fontfamily=pp.FONT_FAMILY,
    )
    axis.set_title(
        "All Cooperation + Equal Allocation",
        fontsize=float(title_fontsize),
        fontweight=pp.FONT_WEIGHT_TITLE,
        fontfamily=pp.FONT_FAMILY,
        pad=14.0,
    )
    axis.margins(x=0.03)
    axis.grid(False)
    axis.tick_params(axis="both", labelsize=float(tick_labelsize))
    pp._style_axis_text(
        axis,
        title_fontsize=float(title_fontsize),
        axis_label_fontsize=float(axis_label_fontsize),
        tick_labelsize=float(tick_labelsize),
    )
    legend = axis.legend(
        loc=str(legend_loc),
        bbox_to_anchor=(float(legend_bbox_x), float(legend_bbox_y)),
        ncol=max(1, int(legend_ncol)),
        frameon=False,
        prop={
            "family": pp.FONT_FAMILY,
            "weight": pp.FONT_WEIGHT_BOLD,
            "size": float(legend_fontsize),
        },
    )
    if legend is not None:
        pp._style_legend_text(legend, fontsize=float(legend_fontsize))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    return png_path.resolve(), pdf_path.resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive.")
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive.")
    if int(args.episode_length_override) <= 0:
        raise ValueError("--episode-length-override must be positive.")
    if int(args.graph_bank_index) < 0:
        raise ValueError("--graph-bank-index must be non-negative.")
    if not 0.0 < float(args.gini_average_tail_fraction) <= 1.0:
        raise ValueError("--gini-average-tail-fraction must be in (0, 1].")


def main() -> None:
    pp._configure_matplotlib_fonts()
    args = parse_args()
    _validate_args(args)

    run_dir = args.run_dir.expanduser().resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found under {run_dir}")

    results_payload = _load_json(results_path)
    experiment_spec = results_payload.get("experiment", results_payload)
    if not isinstance(experiment_spec, Mapping):
        raise TypeError("Could not extract experiment spec from results.json.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / str(args.output_subdir_name)).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    topologies = _ordered_topologies(str(args.topologies))
    if not topologies:
        raise ValueError("No topology was selected.")
    graph_metadata_csv = (
        None
        if args.graph_metadata_csv is None
        else args.graph_metadata_csv.expanduser().resolve()
    )
    replay_graph_rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    graph_metadata_source: str | None = None
    if graph_metadata_csv is not None:
        replay_graph_rows_by_key = _load_graph_metadata_rows(graph_metadata_csv)
        graph_metadata_source = "metadata_csv_replay"
    elif bool(args.use_embedded_reference_graphs):
        replay_graph_rows_by_key = _build_embedded_reference_graph_rows()
        graph_metadata_source = "embedded_reference_replay"

    if replay_graph_rows_by_key:
        missing_replay_keys = [
            (str(topology_name), int(episode_index))
            for topology_name in topologies
            for episode_index in range(int(args.episodes))
            if (str(topology_name), int(episode_index)) not in replay_graph_rows_by_key
        ]
        if missing_replay_keys:
            preview = ", ".join(
                f"{topology}/episode={episode}" for topology, episode in missing_replay_keys[:8]
            )
            raise KeyError(
                f"Graph replay data does not contain all requested topology/episode rows: {preview}"
            )
        graph_source_mode = str(graph_metadata_source)
    else:
        graph_source_mode = normalize_graph_source_mode(str(args.graph_source_mode))
    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)
    seed_base = int(experiment_spec.get("seed", 42)) if args.seed is None else int(args.seed)

    all_step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    effective_dynamics_by_topology: dict[str, Any] = {}

    total_runs = len(topologies) * int(args.episodes)
    completed_runs = 0
    for topology_name in topologies:
        for episode_index in range(int(args.episodes)):
            replay_metadata_row = replay_graph_rows_by_key.get(
                (str(topology_name), int(episode_index))
            )
            if replay_metadata_row is not None:
                topology_spec, _ = build_topology_spec_for_analysis(
                    experiment_spec,
                    str(topology_name),
                    reference_mean_degree=float(reference_mean_degree),
                    graph_source_mode="spec_seed",
                    graph_bank_index=0,
                    spec_seed=int(replay_metadata_row["graph_seed"]),
                )
                topology_spec, graph_metadata = _apply_replayed_graph_metadata(
                    topology_spec,
                    replay_metadata_row,
                    replay_mode=str(graph_metadata_source),
                )
            else:
                if graph_source_mode == "random_per_episode":
                    graph_spec_seed: int | None = (
                        int(seed_base) + _topology_seed_offset(str(topology_name)) + int(episode_index)
                    )
                elif graph_source_mode == "spec_seed":
                    graph_spec_seed = int(seed_base)
                else:
                    graph_spec_seed = None

                topology_spec, graph_metadata = build_topology_spec_for_analysis(
                    experiment_spec,
                    str(topology_name),
                    reference_mean_degree=float(reference_mean_degree),
                    graph_source_mode=str(graph_source_mode),
                    graph_bank_index=int(args.graph_bank_index),
                    spec_seed=graph_spec_seed,
                )
            topology_spec = _apply_environment_overrides(
                topology_spec,
                episode_length_override=int(args.episode_length_override),
                env_r_override=args.env_r_override,
                env_p_c_override=args.env_p_c_override,
                env_p_max_override=args.env_p_max_override,
                env_initial_resource_override=args.env_initial_resource_override,
            )
            graph = build_graph_from_spec(topology_spec)
            env_config = build_env_config_from_spec(topology_spec, graph)
            episode_seed = int(seed_base) + int(episode_index)
            current_step_rows, episode_row = _run_episode(
                env_config=env_config,
                graph=graph,
                topology_name=str(topology_name),
                episode_index=int(episode_index),
                episode_seed=int(episode_seed),
                max_steps=int(args.max_steps),
                graph_metadata=graph_metadata,
            )
            all_step_rows.extend(current_step_rows)
            episode_rows.append(episode_row)
            graph_rows.append(
                {
                    "topology": str(topology_name),
                    "topology_label": str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
                    "episode": int(episode_index),
                    "graph_replay_mode": graph_metadata.get("graph_replay_mode"),
                    "graph_source_mode": str(graph_metadata.get("graph_source_mode", graph_source_mode)),
                    "graph_seed": int(graph_metadata.get("graph_seed", topology_spec["seed"])),
                    "graph_bank_index": graph_metadata.get("graph_bank_index"),
                    "fixed_graph_bank_seed": graph_metadata.get("fixed_graph_bank_seed"),
                    "fixed_graph_bank_size_per_type": graph_metadata.get("fixed_graph_bank_size_per_type"),
                    "fixed_graph_bank_sampling": graph_metadata.get("fixed_graph_bank_sampling"),
                    "graph_num_nodes": int(topology_spec["network"]["num_nodes"]),
                    "graph_mean_degree": float(np.mean([len(neighbors) for neighbors in graph.values()])),
                    "regular_degree": topology_spec["network"].get("regular_degree"),
                    "er_target_mean_degree": topology_spec["network"].get("er_target_mean_degree"),
                    "ws_degree": topology_spec["network"].get("ws_degree"),
                    "ws_rewiring_prob": topology_spec["network"].get("ws_rewiring_prob"),
                    "ba_attachments_per_new_node": topology_spec["network"].get(
                        "ba_attachments_per_new_node"
                    ),
                }
            )
            effective_dynamics_by_topology[str(topology_name)] = dict(topology_spec["dynamics"])
            completed_runs += 1
            print(
                f"[{completed_runs}/{total_runs}] {TOPOLOGY_LABELS.get(str(topology_name), topology_name)} "
                f"episode {episode_index + 1}/{int(args.episodes)} finished"
            )

    summary_rows = _summarize_step_rows(all_step_rows)
    topology_average_rows = _summarize_topology_average_gini(
        summary_rows,
        step_rows=all_step_rows,
        tail_fraction=float(args.gini_average_tail_fraction),
    )
    _write_csv(output_dir / "all_cooperation_equal_step_metrics.csv", all_step_rows)
    _write_csv(output_dir / "all_cooperation_equal_gini_dynamics.csv", summary_rows)
    _write_csv(output_dir / "topology_average_gini.csv", topology_average_rows)
    _write_csv(output_dir / "all_cooperation_equal_episode_summary.csv", episode_rows)
    _write_csv(output_dir / "episode_graph_metadata.csv", graph_rows)

    print(
        f"Gini over the last {100.0 * float(args.gini_average_tail_fraction):g}% of each episode "
        "(mean ± std across episodes, ddof=0):"
    )
    for row in topology_average_rows:
        print(
            f"  {row['topology_label']}: {float(row['average_gini_level']):.6f} "
            f"± {float(row['average_gini_std']):.6f}"
        )

    figure_png, figure_pdf = _plot_gini_dynamics(
        summary_rows,
        topology_average_rows=topology_average_rows,
        topologies=topologies,
        output_stem=output_dir / "all_cooperation_equal_gini",
        figsize=(float(args.figsize_width), float(args.figsize_height)),
        title_fontsize=float(args.title_fontsize),
        axis_label_fontsize=float(args.axis_label_fontsize),
        tick_labelsize=float(args.tick_labelsize),
        legend_fontsize=float(args.legend_fontsize),
        time_axis_scale=str(args.time_axis_scale),
        metric_axis_scale=str(args.metric_axis_scale),
        marker_count=int(args.marker_count),
        marker_size=float(args.marker_size),
        legend_ncol=int(args.legend_ncol),
        legend_loc=str(args.legend_loc),
        legend_bbox_x=float(args.legend_bbox_x),
        legend_bbox_y=float(args.legend_bbox_y),
    )

    all_nominal_cooperation_verified = all(
        np.isclose(float(row["nominal_cooperation_rate"]), 1.0) for row in all_step_rows
    )
    summary_payload = {
        "scenario": {
            "nominal_strategy": "all_cooperation_fixed",
            "allocation_policy": "UniformAllocationPolicy",
            "note": (
                "Nominal cooperation is forced to 1. Actual cooperation can be below 1 when a node "
                "does not have enough resources to meet the environment's cooperation threshold."
            ),
        },
        "run_dir": str(run_dir),
        "topologies": [str(item) for item in topologies],
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "episode_length_override": int(args.episode_length_override),
        "graph_source_mode": str(graph_source_mode),
        "use_embedded_reference_graphs": bool(args.use_embedded_reference_graphs),
        "graph_metadata_source": graph_metadata_source,
        "graph_metadata_csv": None if graph_metadata_csv is None else str(graph_metadata_csv),
        "graph_bank_index": int(args.graph_bank_index) if graph_source_mode == "training_fixed_graph_bank" else None,
        "seed": int(seed_base),
        "environment_overrides": {
            "r": args.env_r_override,
            "p_c": args.env_p_c_override,
            "p_max": args.env_p_max_override,
            "initial_resource": args.env_initial_resource_override,
        },
        "effective_dynamics_by_topology": effective_dynamics_by_topology,
        "all_nominal_cooperation_verified": bool(all_nominal_cooperation_verified),
        "gini_average_tail_fraction": float(args.gini_average_tail_fraction),
        "topology_average_gini": {
            str(row["topology"]): float(row["average_gini_level"])
            for row in topology_average_rows
        },
        "topology_average_gini_std": {
            str(row["topology"]): float(row["average_gini_std"])
            for row in topology_average_rows
        },
        "gini_std_ddof": 0,
        "gini_average_std_description": (
            "Population standard deviation across episodes of each episode's tail-mean Gini."
        ),
        "output_files": {
            "step_metrics_csv": str((output_dir / "all_cooperation_equal_step_metrics.csv").resolve()),
            "gini_dynamics_csv": str((output_dir / "all_cooperation_equal_gini_dynamics.csv").resolve()),
            "topology_average_gini_csv": str((output_dir / "topology_average_gini.csv").resolve()),
            "episode_summary_csv": str((output_dir / "all_cooperation_equal_episode_summary.csv").resolve()),
            "graph_metadata_csv": str((output_dir / "episode_graph_metadata.csv").resolve()),
            "figure_png": None if figure_png is None else str(figure_png),
            "figure_pdf": None if figure_pdf is None else str(figure_pdf),
        },
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
