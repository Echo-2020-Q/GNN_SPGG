from __future__ import annotations

import argparse
import gc
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

import analyze_rule_allocation_over_time as ar
import plot_policy_dynamics_over_topologies as pp
from Project1.env import SPGGEnv
from Project1.policies.rule_based import (
    PoolPowerMixAllocationPolicy,
    ProportionalContributionPolicy,
    UniformAllocationPolicy,
)
from analyze_gnn_policy_decisions import (
    _load_json,
    build_env_config_from_spec,
    build_graph_from_spec,
    load_actor_from_run_dir,
    parse_topology_list,
    resolve_reference_mean_degree,
)
from run_saved_binned_unitinv_equal_policy import _load_saved_binned_policy
from run_saved_resource_conditioned_spline_policy import _load_saved_conditioned_policy
from training_graph_bank_utils import build_topology_spec_for_analysis, normalize_graph_source_mode


SCRIPT_DEFAULTS = {
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    "checkpoint_name": "best_eval.pt",
     "m1_summary_json": (
        "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
        "0429_binned_three_way_profile/regular/binned_unitinv_equal_profile_summary.json"
    ),
    "m2_summary_json": (
        "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
        "0421_rule_allocation_over_time_ba_degree_conditioned_discrete/analysis_summary.json"
    ),
    "output_dir": None,
    # "output_subdir_name": "0622_1000_two_policy_param_sweep_grid_r",
    "output_subdir_name": "0823_1000_powpoolk_param_sweep_grid_k10_17",
    "topologies": "regular,scale_free,erdos_renyi,small_world",
    "policies": "equal,prop,adaptive",  #"m1,m2,agent,equal,prop,adaptive",
    "episodes": 2,
    "max_steps": 1000,
    "episode_length_override": 1000,
    "x_var": "pool_power_k",
    "x_values": None,
    "env_r_override": None,
    "env_p_c_override": None,
    "env_beta_override": None,
    "env_num_nodes_override": None,
    "env_strategy_update_rule_override": None,
    "env_p_max_override": None,
    "env_initial_resource_override": None,
    "env_initial_cooperation_prob_override": None,
    "graph_source_mode": "random_per_episode",
    "agent_graph_episode_sampling": "uniform_per_episode",
    "agent_graph_bank_index": 0,
    "m2_ego_resource_x_mode": "auto",
    "m2_ego_resource_normalizer": None,
    "m2_saved_ego_resource_normalizer": 443.18,
    "m2_pc_normalizer_knots": "25:215.91,50:443.18,62.5:556.82,75:670.45,87.5:784.09,100:897.73",
    "pool_power_k": 19.0,
    "seed": None,
    "figsize": (12.0, 12.0),
    "legend_ncol": 5,
    "legend_loc": "upper center",
    "legend_bbox_x": 0.50,
    "legend_bbox_y": 1.03,
    "show_legend": False,
    "device": "cpu" if torch.cuda.is_available() else "cpu",
}


DEFAULT_X_VALUES = {
    "pc": [25.0, 50.0, 62.5, 75.0, 87.5, 100.0],
    "r": [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0,7.0,8.0,9.0],#[0.50, 0.625, 0.75, 0.875, 1.00],
    "beta": [0.25, 0.5, 1.0, 3.0, 5.0],
    "n": [10, 25, 50, 75, 100],
    "pool_power_k": [10,11,12,13,14,15.0, 17.0],
}

TOPOLOGY_ORDER = ("regular", "scale_free", "erdos_renyi", "small_world")
TOPOLOGY_LABELS = {
    "regular": "Regular",
    "scale_free": "BA",
    "erdos_renyi": "ER",
    "small_world": "WS",
}

POLICY_ORDER = ("m1", "m2", "agent", "equal", "prop", "adaptive")
POLICY_LABELS = {
    "m1": "M1",
    "m2": "M2",
    "agent": "RL-Agent",
    "equal": "Equal",
    "prop": "Proportional",
    "adaptive": "Interpolation",
}
POLICY_STYLES = {
    "m1": {"color": "#2E8B57", "label": "M1", "linestyle": "-.", "marker": "s", "zorder": 4},
    "m2": {"color": "#BD93DA", "label": "M2", "linestyle": "--", "marker": "^", "zorder": 5},
    "agent": {"color": "#E3A008", "label": "RL-Agent", "linestyle": "-", "marker": "o", "zorder": 4},
    "equal": {"color": "#E56051", "label": "Equal", "linestyle": (0, (5, 2)), "marker": "X", "zorder": 3},
    "prop": {"color": "#1E4688", "label": "Proportional", "linestyle": ":", "marker": "D", "zorder": 3},
    "adaptive": {"color": "#008B8B", "label": "Interpolation", "linestyle": "-", "marker": "P", "zorder": 1},
}
POLICY_ALIASES = {
    "m1": "m1",
    "m2": "m2",
    "agent": "agent",
    "rl": "agent",
    "rl_agent": "agent",
    "rl-agent": "agent",
    "equal": "equal",
    "uniform": "equal",
    "prop": "prop",
    "proportional": "prop",
    "proportionnal": "prop",
    "adaptive": "adaptive",
    "interpolation": "adaptive",
}

BASE_METRIC_SPECS = (
    ("final_cooperation_mean", r"$f_c$"),
    ("final_mean_resource_mean", r"$\bar{R}(t)$"),
    ("final_gini_mean", "Gini"),
)

X_LABELS = {
    "pc": r"$p_c$",
    "r": r"$r$",
    "beta": r"$\kappa$",
    "n": r"$N$",
    "pool_power_k": r"$k$",
}


class _TqdmProgressBar:
    def __init__(self, total: int, *, desc: str) -> None:
        if _tqdm is None:
            raise RuntimeError("tqdm is not available.")
        self._bar = _tqdm(total=max(0, int(total)), desc=str(desc), dynamic_ncols=True, leave=True)

    def set_detail(self, detail: str) -> None:
        self._bar.set_postfix_str(str(detail), refresh=False)

    def update(self, step: int = 1, *, detail: str | None = None) -> None:
        if detail is not None:
            self._bar.set_postfix_str(str(detail), refresh=False)
        self._bar.update(max(0, int(step)))

    def close(self) -> None:
        self._bar.close()


def _build_progress_bar(total: int, *, desc: str) -> Any:
    if _tqdm is not None:
        return _TqdmProgressBar(total=int(total), desc=str(desc))
    return pp._ConsoleProgressBar(total=int(total), desc=str(desc))


def _coerce_path(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
    windows_match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if windows_match and os.name != "nt":
        drive = windows_match.group(1).lower()
        suffix = windows_match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{suffix}").expanduser().resolve()
    return Path(text).expanduser().resolve()


def _parse_float_list(raw_value: str | None, *, fallback: Sequence[float]) -> list[float]:
    if raw_value is None:
        return [float(item) for item in fallback]
    text = str(raw_value).strip()
    if not text:
        return [float(item) for item in fallback]
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    values: list[float] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(float(item))
    return values or [float(item) for item in fallback]


def _parse_positive_int_list(raw_value: str | None, *, fallback: Sequence[int]) -> list[int]:
    float_values = _parse_float_list(raw_value, fallback=fallback)
    values: list[int] = []
    for value in float_values:
        if not np.isfinite(value):
            raise ValueError(f"Node count N must be finite, got {value}.")
        int_value = int(value)
        if float(int_value) != float(value):
            raise ValueError(f"Node count N must be an integer, got {value}.")
        if int_value <= 0:
            raise ValueError(f"Node count N must be positive, got {int_value}.")
        if int_value not in values:
            values.append(int_value)
    return values


def _apply_num_nodes_override(spec: Mapping[str, Any], num_nodes: int | None) -> dict[str, Any]:
    effective_spec = deepcopy(dict(spec))
    if num_nodes is None:
        return effective_spec
    if int(num_nodes) <= 0:
        raise ValueError(f"Node count N must be positive, got {num_nodes}.")
    if "network" not in effective_spec or not isinstance(effective_spec["network"], Mapping):
        raise KeyError("Experiment spec does not contain a network mapping; cannot override node count N.")

    network = dict(effective_spec["network"])
    effective_spec["network"] = network
    node_count = int(num_nodes)
    network["num_nodes"] = node_count
    network_type = str(network.get("type", ""))

    if network_type == "regular":
        degree = max(0, min(node_count - 1, int(network.get("regular_degree", 0))))
        if (node_count * degree) % 2 != 0:
            degree = max(0, degree - 1)
        network["regular_degree"] = degree
    elif network_type == "erdos_renyi":
        target_mean_degree = min(
            float(max(node_count - 1, 0)),
            max(0.0, float(network.get("er_target_mean_degree", 0.0))),
        )
        network["er_target_mean_degree"] = target_mean_degree
        network["er_edge_prob"] = target_mean_degree / float(max(node_count - 1, 1))
    elif network_type == "small_world":
        degree = max(0, min(node_count - 1, int(network.get("ws_degree", 0))))
        if degree % 2 != 0:
            degree -= 1
        network["ws_degree"] = max(0, degree)
    elif network_type == "scale_free":
        if node_count < 2:
            raise ValueError("Scale-free topology requires node count N >= 2.")
        attachments = int(network.get("ba_attachments_per_new_node", 1))
        network["ba_attachments_per_new_node"] = max(1, min(node_count - 1, attachments))

    return effective_spec


def _ordered_topologies(raw_text: str) -> list[str]:
    selected = list(parse_topology_list(str(raw_text)))
    ordered = [name for name in TOPOLOGY_ORDER if name in selected]
    extras = [name for name in selected if name not in ordered]
    return ordered + extras


def _parse_policy_list(raw_text: str | Sequence[str]) -> list[str]:
    if isinstance(raw_text, str):
        chunks = [chunk.strip() for chunk in raw_text.split(",")]
    else:
        chunks = [str(chunk).strip() for chunk in raw_text]

    selected: list[str] = []
    unknown: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        key = chunk.lower().replace(" ", "_")
        policy_name = POLICY_ALIASES.get(key)
        if policy_name is None:
            unknown.append(chunk)
            continue
        if policy_name not in selected:
            selected.append(policy_name)

    if unknown:
        valid_names = ", ".join(POLICY_ORDER)
        raise ValueError(f"Unknown policy name(s): {', '.join(unknown)}. Valid policies: {valid_names}.")
    if not selected:
        raise ValueError("At least one policy must be selected.")

    return [policy_name for policy_name in POLICY_ORDER if policy_name in selected]


def _resolve_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) > 1:
            raise ValueError(f"Expected one numeric value, got {value}.")
        value = value[0]
    return float(value)


def _policy_sort_key(policy_name: str) -> tuple[int, str]:
    text = str(policy_name)
    try:
        return (POLICY_ORDER.index(text), text)
    except ValueError:
        return (len(POLICY_ORDER), text)


def _topology_sort_key(topology_name: str) -> tuple[int, str]:
    text = str(topology_name)
    try:
        return (TOPOLOGY_ORDER.index(text), text)
    except ValueError:
        return (len(TOPOLOGY_ORDER), text)


def _format_sweep_value(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _default_saved_reference_pc(experiment_spec: Mapping[str, Any]) -> float:
    dynamics = dict(experiment_spec.get("dynamics", {}))
    return float(dynamics.get("p_c", 50.0))


def _parse_xy_knots(raw_value: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    pairs = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if len(pairs) < 2:
        raise ValueError("pc normalizer knots need at least two pairs like 50:443.18,75:670.45.")

    x_values: list[float] = []
    y_values: list[float] = []
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid knot pair: {pair}. Expected <x>:<y>.")
        left, right = pair.split(":", maxsplit=1)
        x_values.append(float(left))
        y_values.append(float(right))

    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    order = np.argsort(x_array)
    x_array = x_array[order]
    y_array = y_array[order]
    return x_array, y_array


def _interp_with_linear_extrapolation(x_value: float, x_knots: np.ndarray, y_knots: np.ndarray) -> float:
    x_array = np.asarray(x_knots, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y_knots, dtype=np.float64).reshape(-1)
    if x_array.size < 2 or y_array.size != x_array.size:
        raise ValueError("Interpolation knots must have matching size >= 2.")

    x = float(x_value)
    if x <= float(x_array[0]):
        slope = (float(y_array[1]) - float(y_array[0])) / (float(x_array[1]) - float(x_array[0]))
        return float(y_array[0]) + slope * (x - float(x_array[0]))
    if x >= float(x_array[-1]):
        slope = (float(y_array[-1]) - float(y_array[-2])) / (float(x_array[-1]) - float(x_array[-2]))
        return float(y_array[-1]) + slope * (x - float(x_array[-1]))
    return float(np.interp(x, x_array, y_array))


def _resolve_m2_transfer_config(
    *,
    x_var: str,
    x_value: float,
    requested_mode: str,
    requested_current_normalizer: float | None,
    requested_saved_normalizer: float | None,
    saved_reference_pc: float,
    requested_pc_normalizer_knots: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[str, float | None, float | None]:
    mode_raw = str(requested_mode).strip().lower()
    if mode_raw not in {"auto", "absolute", "normalized"}:
        raise ValueError(f"Unsupported M2 ego_resource_x_mode: {requested_mode}")

    saved_normalizer = None if requested_saved_normalizer is None else float(requested_saved_normalizer)
    if mode_raw == "auto":
        resolved_mode = "normalized" if str(x_var) == "pc" else "absolute"
    else:
        resolved_mode = mode_raw

    current_normalizer = None if requested_current_normalizer is None else float(requested_current_normalizer)
    if resolved_mode == "normalized" and str(x_var) == "pc" and current_normalizer is None:
        if requested_pc_normalizer_knots is not None:
            knot_x, knot_y = requested_pc_normalizer_knots
            current_normalizer = _interp_with_linear_extrapolation(float(x_value), knot_x, knot_y)
        else:
            if saved_normalizer is None or saved_normalizer <= 0.0:
                raise ValueError(
                    "pc sweep with normalized M2 transfer requires --m2-saved-ego-resource-normalizer."
                )
            if float(saved_reference_pc) <= 0.0:
                raise ValueError("Saved reference p_c must be positive when auto-scaling M2 normalized transfer.")
            current_normalizer = float(saved_normalizer) * float(x_value) / float(saved_reference_pc)

    return resolved_mode, current_normalizer, saved_normalizer


def _resolve_pc_resource_axis_normalizer(
    *,
    pc_value: float,
    requested_current_normalizer: float | None,
    requested_saved_normalizer: float | None,
    saved_reference_pc: float,
    requested_pc_normalizer_knots: tuple[np.ndarray, np.ndarray] | None,
) -> float | None:
    current_normalizer = None if requested_current_normalizer is None else float(requested_current_normalizer)
    if current_normalizer is not None:
        return current_normalizer
    if requested_pc_normalizer_knots is not None:
        knot_x, knot_y = requested_pc_normalizer_knots
        return _interp_with_linear_extrapolation(float(pc_value), knot_x, knot_y)
    if requested_saved_normalizer is None or float(requested_saved_normalizer) <= 0.0:
        return None
    if float(saved_reference_pc) <= 0.0:
        return None
    return float(requested_saved_normalizer) * float(pc_value) / float(saved_reference_pc)


def _graph_episode_sampling_mode_label(graph_source_mode_resolved: str, agent_graph_episode_sampling: str) -> str:
    if str(graph_source_mode_resolved) == "training_fixed_graph_bank":
        return str(agent_graph_episode_sampling)
    return "per_episode_spec_seed" if str(graph_source_mode_resolved) == "random_per_episode" else "fixed_spec"


def _build_episode_contexts_for_topology(
    *,
    experiment_spec: Mapping[str, Any],
    reference_mean_degree: float,
    topology_name: str,
    episodes: int,
    rollout_seed: int,
    graph_source_mode: str,
    agent_graph_episode_sampling: str,
    agent_graph_bank_index: int,
    episode_length_override: int | None,
    env_r_override: float | None,
    env_p_c_override: float | None,
    env_beta_override: float | None,
    env_num_nodes_override: int | None,
    env_strategy_update_rule_override: str | None,
    env_p_max_override: float | None,
    env_initial_resource_override: float | None,
    env_initial_cooperation_prob_override: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_episode_count = max(0, int(episodes))
    topology_label = TOPOLOGY_LABELS.get(str(topology_name), str(topology_name))
    graph_source_mode_resolved = normalize_graph_source_mode(str(graph_source_mode))

    if graph_source_mode_resolved == "training_fixed_graph_bank":
        randomization = dict(experiment_spec.get("domain_randomization", {}))
        fixed_graph_bank_size_per_type = int(randomization.get("fixed_graph_bank_size_per_type", 0))
        fixed_graph_bank_seed = int(randomization.get("fixed_graph_bank_seed", experiment_spec["seed"]))
        fixed_graph_bank_sampling = str(randomization.get("fixed_graph_bank_sampling", "uniform"))
        sampled_graph_bank_indices = pp._sample_graph_bank_indices_for_topology(
            topology_name=str(topology_name),
            bank_size=int(fixed_graph_bank_size_per_type),
            episodes=int(requested_episode_count),
            rollout_seed=int(rollout_seed),
            sampling_mode=str(agent_graph_episode_sampling),
            fixed_index=int(agent_graph_bank_index),
        )
    else:
        fixed_graph_bank_size_per_type = None
        fixed_graph_bank_seed = None
        fixed_graph_bank_sampling = None
        sampled_graph_bank_indices = [None for _ in range(requested_episode_count)]

    episode_contexts: list[dict[str, Any]] = []
    episode_graph_rows: list[dict[str, Any]] = []
    attempted_graph_bank_indices: list[int | None] = []

    for episode_index in range(requested_episode_count):
        current_graph_bank_index = sampled_graph_bank_indices[episode_index] if episode_index < len(sampled_graph_bank_indices) else None
        attempted_graph_bank_indices.append(current_graph_bank_index)
        current_spec, current_graph_metadata = build_topology_spec_for_analysis(
            experiment_spec,
            str(topology_name),
            reference_mean_degree=reference_mean_degree,
            graph_source_mode=str(graph_source_mode_resolved),
            graph_bank_index=(0 if current_graph_bank_index is None else int(current_graph_bank_index)),
            spec_seed=(
                None
                if graph_source_mode_resolved == "training_fixed_graph_bank"
                else (
                    int(rollout_seed) + pp._topology_seed_offset(topology_name) + int(episode_index)
                    if graph_source_mode_resolved == "random_per_episode"
                    else int(rollout_seed)
                )
            ),
        )
        current_spec = ar.apply_analysis_env_overrides(
            current_spec,
            env_r_override=env_r_override,
            env_p_c_override=env_p_c_override,
            env_beta_override=env_beta_override,
            env_strategy_update_rule_override=env_strategy_update_rule_override,
            env_p_max_override=env_p_max_override,
            env_initial_resource_override=env_initial_resource_override,
            env_initial_cooperation_prob_override=env_initial_cooperation_prob_override,
        )
        current_spec = _apply_num_nodes_override(current_spec, env_num_nodes_override)
        if episode_length_override is not None:
            current_spec["dynamics"] = dict(current_spec["dynamics"])
            current_spec["dynamics"]["episode_length"] = int(episode_length_override)

        graph_row = {
            "topology": str(topology_name),
            "topology_label": str(topology_label),
            "episode": int(episode_index),
            "graph_source_mode": str(current_graph_metadata.get("graph_source_mode", "spec_seed")),
            "graph_seed": int(current_graph_metadata.get("graph_seed", current_spec["seed"])),
            "graph_bank_index": current_graph_metadata.get("graph_bank_index"),
            "fixed_graph_bank_seed": current_graph_metadata.get("fixed_graph_bank_seed"),
            "fixed_graph_bank_size_per_type": current_graph_metadata.get("fixed_graph_bank_size_per_type"),
            "fixed_graph_bank_sampling": current_graph_metadata.get("fixed_graph_bank_sampling"),
            "graph_num_nodes": int(current_spec["network"]["num_nodes"]),
            "regular_degree": current_spec["network"].get("regular_degree"),
            "er_target_mean_degree": current_spec["network"].get("er_target_mean_degree"),
            "ws_degree": current_spec["network"].get("ws_degree"),
            "ws_rewiring_prob": current_spec["network"].get("ws_rewiring_prob"),
            "ba_attachments_per_new_node": current_spec["network"].get("ba_attachments_per_new_node"),
        }
        episode_contexts.append(
            {
                "episode": int(episode_index),
                "spec": current_spec,
                "episode_seed": int(rollout_seed) + int(episode_index),
                "graph_row": graph_row,
            }
        )
        episode_graph_rows.append(dict(graph_row))

    topology_graph_metadata = {
        "graph_source_mode": str(graph_source_mode_resolved),
        "graph_episode_sampling_mode": _graph_episode_sampling_mode_label(
            str(graph_source_mode_resolved),
            str(agent_graph_episode_sampling),
        ),
        "attempted_episode_count": int(requested_episode_count),
        "kept_episode_count": int(requested_episode_count),
        "graph_seed": (
            int(episode_graph_rows[0]["graph_seed"])
            if episode_graph_rows and len({int(row["graph_seed"]) for row in episode_graph_rows}) == 1
            else None
        ),
        "graph_bank_index": (
            episode_graph_rows[0]["graph_bank_index"]
            if episode_graph_rows and len({str(row["graph_bank_index"]) for row in episode_graph_rows}) == 1
            else None
        ),
        "sampled_graph_bank_indices": [
            None if row["graph_bank_index"] is None else int(row["graph_bank_index"])
            for row in episode_graph_rows
        ],
        "attempted_graph_bank_indices": [
            None if index is None else int(index) for index in attempted_graph_bank_indices
        ],
        "unique_graph_bank_indices": sorted(
            {
                int(row["graph_bank_index"])
                for row in episode_graph_rows
                if row.get("graph_bank_index") is not None
            }
        ),
        "fixed_graph_bank_seed": fixed_graph_bank_seed,
        "fixed_graph_bank_size_per_type": fixed_graph_bank_size_per_type,
        "fixed_graph_bank_sampling": fixed_graph_bank_sampling,
        "graph_num_nodes": (
            int(episode_graph_rows[0]["graph_num_nodes"])
            if episode_graph_rows
            and len({int(row["graph_num_nodes"]) for row in episode_graph_rows}) == 1
            else None
        ),
    }
    return episode_contexts, topology_graph_metadata


def _rollout_episode_summary(
    *,
    actor: Any | None,
    policy: Any | None,
    policy_name: str,
    spec: Mapping[str, Any],
    episode_seed: int,
    episode_index: int,
    max_steps: int | None,
) -> dict[str, Any]:
    graph = build_graph_from_spec(spec)
    env_config = build_env_config_from_spec(spec, graph)
    env = SPGGEnv(env_config, graph)
    observation = env.reset(seed=int(episode_seed))
    done = False
    step_index = 0
    total_reward = 0.0
    last_info: dict[str, Any] | None = None

    while not done:
        if max_steps is not None and step_index >= max_steps:
            break
        if str(policy_name) == "agent":
            if actor is None:
                raise ValueError("RL-Agent policy was selected but actor checkpoint was not loaded.")
            with torch.no_grad():
                policy_output = actor.deterministic_action(observation)
            allocation_matrix = policy_output.allocation_matrix.detach().cpu().numpy()
        else:
            if policy is None:
                raise ValueError(f"Policy object missing for policy {policy_name}.")
            allocation_matrix = np.asarray(policy.allocate(observation), dtype=np.float64)

        next_observation, reward, env_done, info = env.step(np.asarray(allocation_matrix, dtype=np.float64))
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

    return {
        "policy": str(policy_name),
        "episode": int(episode_index),
        "steps": int(step_index),
        "total_reward": float(total_reward),
        "mean_reward": float(total_reward / float(max(step_index, 1))),
        "final_cooperation_rate": float(final_cooperation_rate),
        "final_gini": float(final_gini),
        "final_mean_resource": float(np.mean(observation["resources"])),
    }


def _run_single_topology_episode_only(
    *,
    actor: Any | None,
    experiment_spec: Mapping[str, Any],
    reference_mean_degree: float,
    topology_name: str,
    selected_policies: Sequence[str],
    policy_objects: Mapping[str, Any],
    episodes: int,
    max_steps: int | None,
    episode_length_override: int | None,
    rollout_seed: int,
    graph_source_mode: str,
    agent_graph_episode_sampling: str,
    agent_graph_bank_index: int,
    env_r_override: float | None,
    env_p_c_override: float | None,
    env_beta_override: float | None,
    env_num_nodes_override: int | None,
    env_strategy_update_rule_override: str | None,
    env_p_max_override: float | None,
    env_initial_resource_override: float | None,
    env_initial_cooperation_prob_override: float | None,
    progress_bar: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    topology_label = TOPOLOGY_LABELS.get(str(topology_name), str(topology_name))
    episode_contexts, topology_graph_metadata = _build_episode_contexts_for_topology(
        experiment_spec=experiment_spec,
        reference_mean_degree=reference_mean_degree,
        topology_name=str(topology_name),
        episodes=int(episodes),
        rollout_seed=int(rollout_seed),
        graph_source_mode=str(graph_source_mode),
        agent_graph_episode_sampling=str(agent_graph_episode_sampling),
        agent_graph_bank_index=int(agent_graph_bank_index),
        episode_length_override=episode_length_override,
        env_r_override=env_r_override,
        env_p_c_override=env_p_c_override,
        env_beta_override=env_beta_override,
        env_num_nodes_override=env_num_nodes_override,
        env_strategy_update_rule_override=env_strategy_update_rule_override,
        env_p_max_override=env_p_max_override,
        env_initial_resource_override=env_initial_resource_override,
        env_initial_cooperation_prob_override=env_initial_cooperation_prob_override,
    )

    episode_rows: list[dict[str, Any]] = []
    for policy_name in selected_policies:
        if str(policy_name) != "agent" and policy_objects.get(str(policy_name)) is None:
            continue
        for context in episode_contexts:
            if progress_bar is not None:
                progress_bar.set_detail(
                    f"Running {topology_label} | {POLICY_LABELS.get(str(policy_name), str(policy_name))} | episode {int(context['episode']) + 1}"
                )
            summary_row = _rollout_episode_summary(
                actor=actor if str(policy_name) == "agent" else None,
                policy=None if str(policy_name) == "agent" else policy_objects.get(str(policy_name)),
                policy_name=str(policy_name),
                spec=context["spec"],
                episode_seed=int(context["episode_seed"]),
                episode_index=int(context["episode"]),
                max_steps=max_steps,
            )
            merged_row = dict(summary_row)
            graph_row = dict(context["graph_row"])
            merged_row.update(
                {
                    "graph_source_mode": graph_row.get("graph_source_mode"),
                    "graph_seed": graph_row.get("graph_seed"),
                    "graph_bank_index": graph_row.get("graph_bank_index"),
                    "fixed_graph_bank_seed": graph_row.get("fixed_graph_bank_seed"),
                    "fixed_graph_bank_size_per_type": graph_row.get("fixed_graph_bank_size_per_type"),
                    "fixed_graph_bank_sampling": graph_row.get("fixed_graph_bank_sampling"),
                    "graph_num_nodes": graph_row.get("graph_num_nodes"),
                    "topology": str(topology_name),
                    "topology_label": str(topology_label),
                }
            )
            episode_rows.append(merged_row)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if progress_bar is not None:
            progress_bar.update(detail=f"Finished {topology_label} | {POLICY_LABELS.get(str(policy_name), str(policy_name))}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return episode_rows, topology_graph_metadata


def _build_policy_objects_for_sweep_point(
    *,
    m1_summary_json: Path,
    m2_summary_json: Path,
    selected_policies: Sequence[str],
    x_var: str,
    x_value: float,
    requested_m2_mode: str,
    requested_m2_current_normalizer: float | None,
    requested_m2_saved_normalizer: float | None,
    saved_reference_pc: float,
    requested_pc_normalizer_knots: tuple[np.ndarray, np.ndarray] | None,
    pool_power_k: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = {str(policy_name) for policy_name in selected_policies}
    policy_objects: dict[str, Any] = {}
    policy_metadata: dict[str, Any] = {}

    if "m1" in selected:
        m1_policy, m1_payload, m1_policy_name, m1_policy_label = _load_saved_binned_policy(m1_summary_json)
        policy_objects["m1"] = m1_policy
        policy_metadata.update(
            {
                "m1_summary_json": str(m1_summary_json),
                "m1_policy_name": str(m1_policy_name),
                "m1_policy_label": str(m1_policy_label),
                "m1_source_topology": str(m1_payload.get("topology", "")),
            }
        )

    if "m2" in selected:
        m2_mode, m2_current_normalizer, m2_saved_normalizer = _resolve_m2_transfer_config(
            x_var=str(x_var),
            x_value=float(x_value),
            requested_mode=str(requested_m2_mode),
            requested_current_normalizer=requested_m2_current_normalizer,
            requested_saved_normalizer=requested_m2_saved_normalizer,
            saved_reference_pc=float(saved_reference_pc),
            requested_pc_normalizer_knots=requested_pc_normalizer_knots,
        )
        m2_policy, m2_payload, m2_profile_summary, _, _ = _load_saved_conditioned_policy(
            m2_summary_json,
            replay_ego_resource_x_mode=str(m2_mode),
            replay_ego_resource_normalizer=m2_current_normalizer,
            saved_ego_resource_normalizer=m2_saved_normalizer,
        )
        policy_objects["m2"] = m2_policy
        policy_metadata.update(
            {
                "m2_source_topology": str(m2_payload.get("topology", "")),
                "m2_profile_summary": dict(m2_profile_summary),
                "m2_replay_ego_resource_x_mode": str(m2_mode),
                "m2_replay_ego_resource_normalizer": (
                    None if m2_current_normalizer is None else float(m2_current_normalizer)
                ),
                "m2_saved_ego_resource_normalizer": (
                    None if m2_saved_normalizer is None else float(m2_saved_normalizer)
                ),
                "m2_saved_reference_pc": float(saved_reference_pc),
                "m2_pc_normalizer_knots": (
                    None
                    if requested_pc_normalizer_knots is None
                    else [
                        {"pc": float(pc_value), "normalizer": float(norm_value)}
                        for pc_value, norm_value in zip(
                            requested_pc_normalizer_knots[0].tolist(),
                            requested_pc_normalizer_knots[1].tolist(),
                        )
                    ]
                ),
            }
        )

    if "equal" in selected:
        policy_objects["equal"] = UniformAllocationPolicy()
    if "prop" in selected:
        policy_objects["prop"] = ProportionalContributionPolicy()
    if "adaptive" in selected:
        policy_objects["adaptive"] = PoolPowerMixAllocationPolicy(power_k=float(pool_power_k))
        policy_metadata.update(
            {
                "adaptive_policy": "PoolPowerMixAllocationPolicy",
                "adaptive_policy_label": str(POLICY_LABELS["adaptive"]),
                "pool_power_k": float(pool_power_k),
            }
        )

    return policy_objects, policy_metadata


def _resolve_pool_power_k_for_sweep_point(
    *,
    x_var: str,
    x_value: float,
    fixed_pool_power_k: float,
) -> float:
    power_k = float(x_value) if str(x_var) == "pool_power_k" else float(fixed_pool_power_k)
    if not np.isfinite(power_k):
        raise ValueError(f"pool_power_k must be finite, got {power_k}.")
    if power_k < 0.0:
        raise ValueError(f"pool_power_k must be non-negative, got {power_k}.")
    return power_k


def _summarize_sweep_point_rows(
    *,
    topology_name: str,
    x_var: str,
    x_value: float,
    episode_rows: Sequence[Mapping[str, Any]],
    graph_metadata: Mapping[str, Any],
    policy_metadata: Mapping[str, Any],
    env_r_value: float | None,
    env_p_c_value: float | None,
    env_beta_value: float | None,
    env_num_nodes_value: int | None,
    resource_axis_normalizer: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name in POLICY_ORDER:
        policy_episode_rows = [row for row in episode_rows if str(row.get("policy")) == str(policy_name)]
        if not policy_episode_rows:
            continue
        raw_mean_resource = float(np.mean([float(row["final_mean_resource"]) for row in policy_episode_rows]))
        normalized_mean_resource = (
            raw_mean_resource / float(resource_axis_normalizer)
            if str(x_var) == "pc" and resource_axis_normalizer is not None and float(resource_axis_normalizer) > 0.0
            else raw_mean_resource
        )
        rows.append(
            {
                "x_var": str(x_var),
                "x_label": str(X_LABELS.get(str(x_var), x_var)),
                "x_value": float(x_value),
                "topology": str(topology_name),
                "topology_label": str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
                "policy": str(policy_name),
                "policy_label": str(POLICY_LABELS.get(str(policy_name), policy_name)),
                "episodes": int(len(policy_episode_rows)),
                "return_mean": float(np.mean([float(row["total_reward"]) for row in policy_episode_rows])),
                "reward_per_step_mean": float(np.mean([float(row["mean_reward"]) for row in policy_episode_rows])),
                "final_cooperation_mean": float(np.mean([float(row["final_cooperation_rate"]) for row in policy_episode_rows])),
                "final_gini_mean": float(np.mean([float(row["final_gini"]) for row in policy_episode_rows])),
                "final_mean_resource_mean": raw_mean_resource,
                "final_mean_resource_normalized_mean": normalized_mean_resource,
                "resource_axis_normalizer": (
                    None if resource_axis_normalizer is None else float(resource_axis_normalizer)
                ),
                "graph_source_mode": str(graph_metadata.get("graph_source_mode", "")),
                "graph_episode_sampling_mode": str(graph_metadata.get("graph_episode_sampling_mode", "")),
                "graph_num_nodes": graph_metadata.get("graph_num_nodes"),
                "env_r": None if env_r_value is None else float(env_r_value),
                "env_p_c": None if env_p_c_value is None else float(env_p_c_value),
                "env_beta": None if env_beta_value is None else float(env_beta_value),
                "env_num_nodes": None if env_num_nodes_value is None else int(env_num_nodes_value),
                "m2_replay_ego_resource_x_mode": policy_metadata.get("m2_replay_ego_resource_x_mode"),
                "m2_replay_ego_resource_normalizer": policy_metadata.get("m2_replay_ego_resource_normalizer"),
                "m2_saved_ego_resource_normalizer": policy_metadata.get("m2_saved_ego_resource_normalizer"),
                "m2_saved_reference_pc": policy_metadata.get("m2_saved_reference_pc"),
                "adaptive_policy": policy_metadata.get("adaptive_policy"),
                "pool_power_k": policy_metadata.get("pool_power_k"),
            }
        )
    return rows


def _metric_specs_for_x_var(x_var: str) -> tuple[tuple[str, str], ...]:
    if str(x_var) == "pc":
        return (
            ("final_cooperation_mean", r"$f_c$"),
            ("final_mean_resource_normalized_mean", r"$\bar{R}_{\mathrm{normal}}(t)$"),
            ("final_gini_mean", "Gini"),
        )
    return BASE_METRIC_SPECS


def _apply_sweep_metric_ylim(axis: Any, *, metric_key: str, values: np.ndarray) -> None:
    if str(metric_key) in {"final_cooperation_mean", "final_gini_mean", "final_mean_resource_normalized_mean"}:
        axis.set_ylim(-0.03, 1.03)
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        return
    if str(metric_key) == "final_mean_resource_mean":
        axis.set_ylim(-12.0, 462.0)
        axis.set_yticks([0.0, 100.0, 200.0, 300.0, 400.0, 450.0])
        return
    safe_values = np.asarray(values, dtype=np.float64)
    if safe_values.size == 0 or not np.any(np.isfinite(safe_values)):
        return
    value_min = float(np.nanmin(safe_values))
    value_max = float(np.nanmax(safe_values))
    span = max(1e-9, value_max - value_min)
    axis.set_ylim(value_min - 0.05 * span, value_max + 0.05 * span)


def _plot_sweep_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_var: str,
    topologies: Sequence[str],
    output_stem: Path,
    figsize: tuple[float, float],
    legend_ncol: int,
    legend_loc: str,
    legend_bbox_x: float,
    legend_bbox_y: float,
    show_legend: bool,
) -> None:
    if ar.plt is None or not rows:
        return

    metric_specs = _metric_specs_for_x_var(str(x_var))
    figure, axes = ar.plt.subplots(
        len(topologies),
        len(metric_specs),
        figsize=figsize,
        sharex="col",
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(len(topologies), len(metric_specs))

    for row_index, topology_name in enumerate(topologies):
        topology_rows = [row for row in rows if str(row.get("topology")) == str(topology_name)]
        topology_label = TOPOLOGY_LABELS.get(str(topology_name), str(topology_name))
        for col_index, (metric_key, ylabel) in enumerate(metric_specs):
            axis = axes_array[row_index, col_index]
            for policy_name in POLICY_ORDER:
                policy_rows = [row for row in topology_rows if str(row.get("policy")) == str(policy_name)]
                if not policy_rows:
                    continue
                policy_rows.sort(key=lambda row: float(row["x_value"]))
                x_values = np.asarray([float(row["x_value"]) for row in policy_rows], dtype=np.float64)
                y_values = np.asarray([float(row[metric_key]) for row in policy_rows], dtype=np.float64)
                style = POLICY_STYLES[str(policy_name)]
                axis.plot(
                    x_values,
                    y_values,
                    label=str(style["label"]),
                    color=str(style["color"]),
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker", None),
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgecolor=str(style["color"]),
                    markeredgewidth=1.1,
                    linewidth=2.3,
                    zorder=float(style.get("zorder", 3)),
                )

            axis.set_yscale("linear")
            _apply_sweep_metric_ylim(
                axis,
                metric_key=str(metric_key),
                values=np.asarray([float(row[metric_key]) for row in topology_rows], dtype=np.float64),
            )
            if row_index == 0:
                axis.set_title(str(ylabel), fontweight=pp.FONT_WEIGHT_TITLE, pad=10.0)
            if row_index == len(topologies) - 1:
                axis.set_xlabel(str(X_LABELS.get(str(x_var), x_var)), fontweight=pp.FONT_WEIGHT_BOLD)
            if col_index == 0:
                axis.set_ylabel(f"{topology_label}\n{ylabel}", fontweight=pp.FONT_WEIGHT_BOLD)
            else:
                axis.set_ylabel(str(ylabel), fontweight=pp.FONT_WEIGHT_BOLD)
            axis.grid(False)
            pp._style_axis_text(axis, title_fontsize=16, axis_label_fontsize=14, tick_labelsize=12)
            if str(metric_key) == "final_gini_mean":
                axis.title.set_fontstyle("italic")
                axis.title.set_fontweight(pp.FONT_WEIGHT_BOLD)
                axis.yaxis.get_label().set_fontstyle("italic")
                axis.yaxis.get_label().set_fontweight(pp.FONT_WEIGHT_BOLD)

    handles, labels = axes_array[0, 0].get_legend_handles_labels()
    if show_legend and handles:
        legend = figure.legend(
            handles,
            labels,
            loc=str(legend_loc),
            bbox_to_anchor=(float(legend_bbox_x), float(legend_bbox_y)),
            ncol=max(1, int(legend_ncol)),
            frameon=False,
            borderaxespad=0.0,
            prop={"family": pp.FONT_FAMILY, "weight": pp.FONT_WEIGHT_BOLD, "size": 13},
        )
        pp._style_legend_text(legend, fontsize=13)

    pp._save_figure_pdf_png(figure, output_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep pc/r/beta/N/pool_power_k and plot a 4x3 grid for selected policies."
    )
    parser.add_argument("--run-dir", type=Path, default=Path(str(SCRIPT_DEFAULTS["run_dir"])))
    parser.add_argument("--checkpoint-name", type=str, default=str(SCRIPT_DEFAULTS["checkpoint_name"]))
    parser.add_argument("--m1-summary-json", type=Path, default=Path(str(SCRIPT_DEFAULTS["m1_summary_json"])))
    parser.add_argument("--m2-summary-json", type=Path, default=Path(str(SCRIPT_DEFAULTS["m2_summary_json"])))
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DEFAULTS["output_dir"])
    parser.add_argument("--output-subdir-name", type=str, default=str(SCRIPT_DEFAULTS["output_subdir_name"]))
    parser.add_argument("--topologies", type=str, default=str(SCRIPT_DEFAULTS["topologies"]))
    parser.add_argument(
        "--policies",
        type=str,
        default=str(SCRIPT_DEFAULTS["policies"]),
        help=(
            "Comma-separated policies to scan. Valid: m1,m2,agent,equal,prop,adaptive. "
            "Aliases include proportional/proportionnal/interpolation."
        ),
    )
    parser.add_argument("--episodes", type=int, default=int(SCRIPT_DEFAULTS["episodes"]))
    parser.add_argument("--max-steps", type=int, default=SCRIPT_DEFAULTS["max_steps"])
    parser.add_argument("--episode-length-override", type=int, default=SCRIPT_DEFAULTS["episode_length_override"])
    parser.add_argument(
        "--x-var",
        type=lambda value: str(value).strip().lower(),
        choices=("pc", "r", "beta", "n", "pool_power_k"),
        default=str(SCRIPT_DEFAULTS["x_var"]).lower(),
    )
    parser.add_argument("--x-values", type=str, default=SCRIPT_DEFAULTS["x_values"])
    parser.add_argument("--env-r", dest="env_r_override", type=float, default=SCRIPT_DEFAULTS["env_r_override"])
    parser.add_argument("--env-p-c", dest="env_p_c_override", type=float, default=SCRIPT_DEFAULTS["env_p_c_override"])
    parser.add_argument("--env-beta", dest="env_beta_override", type=float, default=SCRIPT_DEFAULTS["env_beta_override"])
    parser.add_argument(
        "--env-num-nodes",
        dest="env_num_nodes_override",
        type=int,
        default=SCRIPT_DEFAULTS["env_num_nodes_override"],
        help="Override node count N when sweeping pc/r/beta/pool_power_k.",
    )
    parser.add_argument(
        "--env-strategy-update-rule",
        dest="env_strategy_update_rule_override",
        type=str,
        choices=("fermi", "q_learning", "q_learning_2x2", "imitate_best"),
        default=SCRIPT_DEFAULTS["env_strategy_update_rule_override"],
    )
    parser.add_argument("--env-p-max", dest="env_p_max_override", type=float, default=SCRIPT_DEFAULTS["env_p_max_override"])
    parser.add_argument("--env-initial-resource", dest="env_initial_resource_override", type=float, default=SCRIPT_DEFAULTS["env_initial_resource_override"])
    parser.add_argument(
        "--env-initial-cooperation-prob",
        dest="env_initial_cooperation_prob_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_initial_cooperation_prob_override"],
    )
    parser.add_argument(
        "--graph-source-mode",
        type=str,
        default=str(SCRIPT_DEFAULTS["graph_source_mode"]),
        help="training_fixed_graph_bank, spec_seed, or random_per_episode.",
    )
    parser.add_argument(
        "--agent-graph-episode-sampling",
        type=str,
        choices=("uniform_per_episode", "fixed_index"),
        default=str(SCRIPT_DEFAULTS["agent_graph_episode_sampling"]),
    )
    parser.add_argument("--agent-graph-bank-index", type=int, default=int(SCRIPT_DEFAULTS["agent_graph_bank_index"]))
    parser.add_argument(
        "--m2-ego-resource-x-mode",
        type=str,
        choices=("auto", "absolute", "normalized"),
        default=str(SCRIPT_DEFAULTS["m2_ego_resource_x_mode"]),
        help=(
            "When x-var=pc, auto resolves to normalized; for r/beta/N/pool_power_k, "
            "auto resolves to absolute."
        ),
    )
    parser.add_argument(
        "--m2-ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["m2_ego_resource_normalizer"],
        help="Current runtime normalizer. When omitted in pc sweep, it is auto-scaled from saved p_c.",
    )
    parser.add_argument(
        "--m2-saved-ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["m2_saved_ego_resource_normalizer"],
        help="Saved/the original theoretical maximum for M2 normalized transfer.",
    )
    parser.add_argument(
        "--m2-pc-normalizer-knots",
        type=str,
        default=str(SCRIPT_DEFAULTS["m2_pc_normalizer_knots"]),
        help="Optional pc:normalizer knot pairs for M2 normalized transfer during pc sweep.",
    )
    parser.add_argument(
        "--pool-power-k",
        type=float,
        default=float(SCRIPT_DEFAULTS["pool_power_k"]),
        help=(
            "Fixed exponent k used by the Interpolation PoolPowerMixAllocationPolicy. "
            "When --x-var=pool_power_k, the values from --x-values take precedence."
        ),
    )
    parser.add_argument("--seed", type=int, default=SCRIPT_DEFAULTS["seed"])
    parser.add_argument("--figsize-width", type=float, default=float(SCRIPT_DEFAULTS["figsize"][0]))
    parser.add_argument("--figsize-height", type=float, default=float(SCRIPT_DEFAULTS["figsize"][1]))
    parser.add_argument("--legend-ncol", type=int, default=int(SCRIPT_DEFAULTS["legend_ncol"]))
    parser.add_argument("--legend-loc", type=str, default=str(SCRIPT_DEFAULTS["legend_loc"]))
    parser.add_argument("--legend-bbox-x", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_x"]))
    parser.add_argument("--legend-bbox-y", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_y"]))
    parser.add_argument(
        "--hide-legend",
        dest="show_legend",
        action="store_false",
        help="Disable the figure-level legend.",
    )
    parser.set_defaults(show_legend=bool(SCRIPT_DEFAULTS["show_legend"]))
    parser.add_argument("--device", type=str, default=str(SCRIPT_DEFAULTS["device"]))
    args = parser.parse_args()
    try:
        args.policies = _parse_policy_list(str(args.policies))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    pp._configure_matplotlib_fonts()
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found under {run_dir}")

    selected_policies = list(args.policies)
    m1_summary_json = _coerce_path(args.m1_summary_json)
    m2_summary_json = _coerce_path(args.m2_summary_json)
    if "m1" in selected_policies and not m1_summary_json.exists():
        raise FileNotFoundError(f"M1 summary JSON not found: {m1_summary_json}")
    if "m2" in selected_policies and not m2_summary_json.exists():
        raise FileNotFoundError(f"M2 summary JSON not found: {m2_summary_json}")

    results_payload = _load_json(results_path)
    experiment_spec = results_payload["experiment"] if "experiment" in results_payload else results_payload
    if not isinstance(experiment_spec, dict):
        raise TypeError("Could not extract experiment spec from results.json.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / args.output_subdir_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    x_var = str(args.x_var)
    try:
        x_values = sorted(
            _parse_positive_int_list(args.x_values, fallback=DEFAULT_X_VALUES["n"])
            if x_var == "n"
            else _parse_float_list(args.x_values, fallback=DEFAULT_X_VALUES[str(x_var)])
        )
    except ValueError as exc:
        raise ValueError(f"Invalid --x-values for x-var={x_var}: {exc}") from exc
    if x_var == "pool_power_k":
        if "adaptive" not in selected_policies:
            raise ValueError(
                "--x-var pool_power_k requires the adaptive policy; include adaptive in --policies."
            )
        invalid_power_values = [
            float(value)
            for value in x_values
            if not np.isfinite(value) or float(value) < 0.0
        ]
        if invalid_power_values:
            raise ValueError(
                "Every pool_power_k value must be finite and non-negative; "
                f"got {invalid_power_values}."
            )
    elif "adaptive" in selected_policies and (
        not np.isfinite(float(args.pool_power_k)) or float(args.pool_power_k) < 0.0
    ):
        raise ValueError(
            f"--pool-power-k must be finite and non-negative, got {args.pool_power_k}."
        )
    topologies = _ordered_topologies(str(args.topologies))
    graph_source_mode = normalize_graph_source_mode(str(args.graph_source_mode))
    fixed_num_nodes_override = (
        None if args.env_num_nodes_override is None else int(args.env_num_nodes_override)
    )
    if fixed_num_nodes_override is not None and fixed_num_nodes_override <= 0:
        raise ValueError(f"--env-num-nodes must be positive, got {fixed_num_nodes_override}.")
    if x_var == "n" and fixed_num_nodes_override is not None:
        raise ValueError("Do not combine --x-var N with --env-num-nodes; provide node counts through --x-values.")
    requested_num_nodes = x_values if x_var == "n" else [fixed_num_nodes_override]
    if "scale_free" in topologies and any(
        value is not None and int(value) < 2 for value in requested_num_nodes
    ):
        raise ValueError("Scale-free topology requires every requested node count N to be at least 2.")
    if graph_source_mode == "training_fixed_graph_bank" and (
        x_var == "n" or fixed_num_nodes_override is not None
    ):
        raise ValueError(
            "Node-count overrides cannot replay the training fixed graph bank because its graphs have fixed N. "
            "Use --graph-source-mode spec_seed or random_per_episode."
        )
    seed_base = int(experiment_spec.get("seed", 42)) if args.seed is None else int(args.seed)
    saved_reference_pc = _default_saved_reference_pc(experiment_spec)
    m2_pc_normalizer_knots = _parse_xy_knots(args.m2_pc_normalizer_knots)

    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)
    device = torch.device(str(args.device))
    actor = None
    if "agent" in selected_policies:
        actor, _ = load_actor_from_run_dir(run_dir, str(args.checkpoint_name), device)

    all_episode_rows: list[dict[str, Any]] = []
    all_sweep_rows: list[dict[str, Any]] = []
    sweep_point_metadata: list[dict[str, Any]] = []

    progress_bar = _build_progress_bar(
        total=max(1, len(x_values) * len(topologies) * len(selected_policies)),
        desc=f"Sweeping {x_var}",
    )
    try:
        for x_value in x_values:
            env_r_value = float(x_value) if x_var == "r" else _resolve_optional_float(args.env_r_override)
            env_p_c_value = float(x_value) if x_var == "pc" else _resolve_optional_float(args.env_p_c_override)
            env_beta_value = float(x_value) if x_var == "beta" else _resolve_optional_float(args.env_beta_override)
            env_num_nodes_value = int(x_value) if x_var == "n" else fixed_num_nodes_override
            resource_axis_normalizer = (
                _resolve_pc_resource_axis_normalizer(
                    pc_value=float(x_value),
                    requested_current_normalizer=_resolve_optional_float(args.m2_ego_resource_normalizer),
                    requested_saved_normalizer=_resolve_optional_float(args.m2_saved_ego_resource_normalizer),
                    saved_reference_pc=float(saved_reference_pc),
                    requested_pc_normalizer_knots=m2_pc_normalizer_knots,
                )
                if str(x_var) == "pc"
                else None
            )
            effective_pool_power_k = _resolve_pool_power_k_for_sweep_point(
                x_var=str(x_var),
                x_value=float(x_value),
                fixed_pool_power_k=float(args.pool_power_k),
            )

            policy_objects, policy_metadata = _build_policy_objects_for_sweep_point(
                m1_summary_json=m1_summary_json,
                m2_summary_json=m2_summary_json,
                selected_policies=selected_policies,
                x_var=str(x_var),
                x_value=float(x_value),
                requested_m2_mode=str(args.m2_ego_resource_x_mode),
                requested_m2_current_normalizer=_resolve_optional_float(args.m2_ego_resource_normalizer),
                requested_m2_saved_normalizer=_resolve_optional_float(args.m2_saved_ego_resource_normalizer),
                saved_reference_pc=float(saved_reference_pc),
                requested_pc_normalizer_knots=m2_pc_normalizer_knots,
                pool_power_k=float(effective_pool_power_k),
            )

            for topology_name in topologies:
                episode_rows, graph_metadata = _run_single_topology_episode_only(
                    actor=actor,
                    experiment_spec=experiment_spec,
                    reference_mean_degree=reference_mean_degree,
                    topology_name=str(topology_name),
                    selected_policies=selected_policies,
                    policy_objects=policy_objects,
                    episodes=max(1, int(args.episodes)),
                    max_steps=None if args.max_steps is None else int(args.max_steps),
                    episode_length_override=(
                        None if args.episode_length_override is None else int(args.episode_length_override)
                    ),
                    rollout_seed=int(seed_base),
                    graph_source_mode=str(graph_source_mode),
                    agent_graph_episode_sampling=str(args.agent_graph_episode_sampling),
                    agent_graph_bank_index=int(args.agent_graph_bank_index),
                    env_r_override=env_r_value,
                    env_p_c_override=env_p_c_value,
                    env_beta_override=env_beta_value,
                    env_num_nodes_override=env_num_nodes_value,
                    env_strategy_update_rule_override=args.env_strategy_update_rule_override,
                    env_p_max_override=_resolve_optional_float(args.env_p_max_override),
                    env_initial_resource_override=_resolve_optional_float(args.env_initial_resource_override),
                    env_initial_cooperation_prob_override=_resolve_optional_float(args.env_initial_cooperation_prob_override),
                    progress_bar=progress_bar,
                )

                enriched_episode_rows: list[dict[str, Any]] = []
                for row in episode_rows:
                    merged = dict(row)
                    merged["x_var"] = str(x_var)
                    merged["x_value"] = float(x_value)
                    merged["x_label"] = str(X_LABELS.get(str(x_var), x_var))
                    merged["pool_power_k"] = policy_metadata.get("pool_power_k")
                    all_episode_rows.append(merged)
                    enriched_episode_rows.append(merged)

                all_sweep_rows.extend(
                    _summarize_sweep_point_rows(
                        topology_name=str(topology_name),
                        x_var=str(x_var),
                        x_value=float(x_value),
                        episode_rows=enriched_episode_rows,
                        graph_metadata=graph_metadata,
                        policy_metadata=policy_metadata,
                        env_r_value=env_r_value,
                        env_p_c_value=env_p_c_value,
                        env_beta_value=env_beta_value,
                        env_num_nodes_value=env_num_nodes_value,
                        resource_axis_normalizer=resource_axis_normalizer,
                    )
                )
                sweep_point_metadata.append(
                    {
                        "x_var": str(x_var),
                        "x_value": float(x_value),
                        "topology": str(topology_name),
                        "topology_label": str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
                        "graph_source_mode": str(graph_metadata.get("graph_source_mode", "")),
                        "graph_episode_sampling_mode": str(graph_metadata.get("graph_episode_sampling_mode", "")),
                        "graph_num_nodes": graph_metadata.get("graph_num_nodes"),
                        "attempted_episode_count": graph_metadata.get("attempted_episode_count"),
                        "kept_episode_count": graph_metadata.get("kept_episode_count"),
                        "m1_summary_json": policy_metadata.get("m1_summary_json"),
                        "m1_policy_name": policy_metadata.get("m1_policy_name"),
                        "m1_policy_label": policy_metadata.get("m1_policy_label"),
                        "m1_source_topology": policy_metadata.get("m1_source_topology"),
                        "m2_replay_ego_resource_x_mode": policy_metadata.get("m2_replay_ego_resource_x_mode"),
                        "m2_replay_ego_resource_normalizer": policy_metadata.get("m2_replay_ego_resource_normalizer"),
                        "m2_saved_ego_resource_normalizer": policy_metadata.get("m2_saved_ego_resource_normalizer"),
                        "m2_saved_reference_pc": policy_metadata.get("m2_saved_reference_pc"),
                        "adaptive_policy": policy_metadata.get("adaptive_policy"),
                        "adaptive_policy_label": policy_metadata.get("adaptive_policy_label"),
                        "pool_power_k": policy_metadata.get("pool_power_k"),
                        "resource_axis_normalizer": (
                            None if resource_axis_normalizer is None else float(resource_axis_normalizer)
                        ),
                        "episode_row_count": int(len(enriched_episode_rows)),
                        "step_row_count": 0,
                    }
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        progress_bar.close()

    all_episode_rows = sorted(
        all_episode_rows,
        key=lambda row: (
            float(row.get("x_value", 0.0)),
            _topology_sort_key(str(row.get("topology", ""))),
            _policy_sort_key(str(row.get("policy", ""))),
            int(row.get("episode", 0)),
        ),
    )
    all_sweep_rows = sorted(
        all_sweep_rows,
        key=lambda row: (
            float(row.get("x_value", 0.0)),
            _topology_sort_key(str(row.get("topology", ""))),
            _policy_sort_key(str(row.get("policy", ""))),
        ),
    )
    sweep_point_metadata = sorted(
        sweep_point_metadata,
        key=lambda row: (
            float(row.get("x_value", 0.0)),
            _topology_sort_key(str(row.get("topology", ""))),
        ),
    )

    ar.write_csv(output_dir / "all_episode_summary.csv", all_episode_rows)
    ar.write_csv(output_dir / "parameter_sweep_summary.csv", all_sweep_rows)
    ar.write_csv(output_dir / "sweep_point_metadata.csv", sweep_point_metadata)

    output_stem = output_dir / f"{x_var}_sweep_four_policy_grid"
    _plot_sweep_grid(
        all_sweep_rows,
        x_var=str(x_var),
        topologies=topologies,
        output_stem=output_stem,
        figsize=(float(args.figsize_width), float(args.figsize_height)),
        legend_ncol=int(args.legend_ncol),
        legend_loc=str(args.legend_loc),
        legend_bbox_x=float(args.legend_bbox_x),
        legend_bbox_y=float(args.legend_bbox_y),
        show_legend=bool(args.show_legend),
    )

    summary_payload = {
        "run_dir": str(run_dir),
        "checkpoint_name": str(args.checkpoint_name),
        "m1_summary_json": str(m1_summary_json),
        "m2_summary_json": str(m2_summary_json),
        "x_var": str(x_var),
        "x_label": str(X_LABELS.get(str(x_var), x_var)),
        "x_values": [int(item) for item in x_values] if x_var == "n" else [float(item) for item in x_values],
        "topologies": [str(item) for item in topologies],
        "policies": [str(item) for item in selected_policies],
        "episodes": int(args.episodes),
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "episode_length_override": None if args.episode_length_override is None else int(args.episode_length_override),
        "graph_source_mode": str(graph_source_mode),
        "agent_graph_episode_sampling": str(args.agent_graph_episode_sampling),
        "agent_graph_bank_index": int(args.agent_graph_bank_index),
        "seed": int(seed_base),
        "saved_reference_pc": float(saved_reference_pc),
        "m2_transfer_request": {
            "mode": str(args.m2_ego_resource_x_mode),
            "current_normalizer": _resolve_optional_float(args.m2_ego_resource_normalizer),
            "saved_normalizer": _resolve_optional_float(args.m2_saved_ego_resource_normalizer),
            "pc_normalizer_knots": str(args.m2_pc_normalizer_knots),
        },
        "adaptive_policy": (
            {
                "class": "PoolPowerMixAllocationPolicy",
                "label": str(POLICY_LABELS["adaptive"]),
                "pool_power_k": (
                    None if x_var == "pool_power_k" else float(args.pool_power_k)
                ),
                "pool_power_k_values": (
                    [float(item) for item in x_values]
                    if x_var == "pool_power_k"
                    else [float(args.pool_power_k)]
                ),
            }
            if "adaptive" in selected_policies
            else None
        ),
        "env_base_overrides": {
            "r": _resolve_optional_float(args.env_r_override),
            "p_c": _resolve_optional_float(args.env_p_c_override),
            "beta": _resolve_optional_float(args.env_beta_override),
            "num_nodes": fixed_num_nodes_override,
            "strategy_update_rule": args.env_strategy_update_rule_override,
            "p_max": _resolve_optional_float(args.env_p_max_override),
            "initial_resource": _resolve_optional_float(args.env_initial_resource_override),
            "initial_cooperation_prob": _resolve_optional_float(args.env_initial_cooperation_prob_override),
        },
        "output_files": {
            "all_episode_summary_csv": str(output_dir / "all_episode_summary.csv"),
            "parameter_sweep_summary_csv": str(output_dir / "parameter_sweep_summary.csv"),
            "sweep_point_metadata_csv": str(output_dir / "sweep_point_metadata.csv"),
            "figure_png": str(output_stem.with_suffix(".png")),
            "figure_pdf": str(output_stem.with_suffix(".pdf")),
        },
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
