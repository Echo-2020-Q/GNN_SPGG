from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
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
from Project1.policies.rule_based import (
    PoolPowerMixAllocationPolicy,
    ProportionalContributionPolicy,
    UniformAllocationPolicy,
)
from analyze_gnn_policy_decisions import _load_json, load_actor_from_run_dir, parse_topology_list, resolve_reference_mean_degree
from run_saved_binned_unitinv_equal_policy import _load_saved_binned_policy
from run_saved_resource_conditioned_spline_policy import _load_saved_conditioned_policy
from training_graph_bank_utils import normalize_graph_source_mode


SCRIPT_DEFAULTS = {
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    "checkpoint_name": "best_eval.pt",
    "m1_summary_json": (
        "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
        "0429_binned_three_way_profile/regular/binned_unitinv_equal_profile_summary.json"
    ),
    # 0429 那张 publication 图对应的是这份已保存的 degree-conditioned policy。
    "m2_summary_json": (
        "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
        "0421_rule_allocation_over_time_ba_degree_conditioned_discrete/analysis_summary.json"
    ),
    "output_dir": None,
    "output_subdir_name": "0820_200_six_policy_dynamics_grid",
    "topologies": "regular,scale_free,erdos_renyi,small_world",
    "episodes": 2,
    "max_steps": 200,
    "episode_length_override": 200,
    "env_r_override": None,
    "env_p_c_override": None,
    "env_beta_override": None,
    "env_strategy_update_rule_override": None,
    "env_p_max_override": None,
    "env_initial_resource_override": None,
    "env_initial_cooperation_prob_override": None,
    "graph_source_mode": "random_per_episode",
    "agent_graph_episode_sampling": "uniform_per_episode",
    "agent_graph_bank_index": 0,
    "time_axis_scale": "symlog",
    "metric_axis_scale": "linear",
    "m2_ego_resource_x_mode": "absolute",
    "m2_ego_resource_normalizer": None,
    "m2_saved_ego_resource_normalizer": None,
    "pool_power_k": 19.0,
    "seed": None,
    "figsize": (12.0, 12.0),
    "legend_ncol": 1,
    "legend_loc": "upper center",
    "legend_bbox_x": 0.51,
    "legend_bbox_y": 0.97,
    "marker_count": 10,
    "marker_size": 5.5,
    "num_workers": 1,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


TOPOLOGY_ORDER = ("regular", "scale_free", "erdos_renyi", "small_world")
TOPOLOGY_LABELS = {
    "regular": "Regular",
    "scale_free": "BA",
    "erdos_renyi": "ER",
    "small_world": "WS",
}

POLICY_ORDER = ("m1", "m2", "prop", "equal", "adaptive", "agent")
POLICY_LABELS = {
    "m1": "M1",
    "m2": "M2",
    "prop": "Prop",
    "equal": "Equal",
    "adaptive": "Interpolation",
    "agent": "RL-Agent",
}
POLICY_STYLES = {
    "agent": {"color": "#E3A008", "label": "RL-Agent", "linestyle": "-", "marker": "o", "zorder": 3},
    "m1": {"color": "#2E8B57", "label": "M1", "linestyle": "-.", "marker": "s", "zorder": 4},
    "m2": {"color": "#BD93DA", "label": "M2", "linestyle": "--", "marker": "^", "zorder": 5},
    "prop": {"color": "#1E4688", "label": "Prop", "linestyle": ":", "marker": "D", "zorder": 3},
    "equal": {"color": "#E56051", "label": "Equal", "linestyle": (0, (5, 2)), "marker": "X", "zorder": 3},
    "adaptive": {"color": "#008B8B", "label": "Adaptive", "linestyle": "-", "marker": "P", "zorder": 1},
}

METRIC_SPECS = (
    ("cooperation_rate_mean", r"$f_c$"),
    ("mean_resource_mean", r"$\bar{R}(t)$"),
    ("resource_gini_mean", "Gini"),
)


def _panel_label(index: int) -> str:
    return f"({chr(ord('a') + int(index))})"


def _build_even_markevery(
    x_values: np.ndarray,
    *,
    axis_scale: str,
    target_count: int,
    symlog_linthresh: float = 1.0,
) -> list[int] | int:
    x_array = np.asarray(x_values, dtype=np.float64).reshape(-1)
    if x_array.size <= 1:
        return 1

    safe_target_count = max(2, min(int(target_count), int(x_array.size)))
    if safe_target_count >= x_array.size:
        return [int(index) for index in range(int(x_array.size))]

    scale = str(axis_scale)
    if scale == "linear":
        transformed = x_array
    elif scale == "log":
        transformed = np.log10(np.clip(x_array, 1e-12, None))
    else:
        abs_x = np.abs(x_array)
        transformed = np.where(
            abs_x <= float(symlog_linthresh),
            abs_x / float(symlog_linthresh),
            1.0 + np.log10(np.clip(abs_x / float(symlog_linthresh), 1e-12, None)),
        )
        transformed = transformed * np.sign(x_array)

    target_positions = np.linspace(float(transformed[0]), float(transformed[-1]), num=safe_target_count)
    marker_indices = sorted(
        {
            int(np.argmin(np.abs(transformed - target_position)))
            for target_position in target_positions.tolist()
        }
    )
    if 0 not in marker_indices:
        marker_indices.insert(0, 0)
    last_index = int(x_array.size - 1)
    if marker_indices[-1] != last_index:
        marker_indices.append(last_index)
    return marker_indices


class _TqdmProgressBar:
    def __init__(self, total: int, *, desc: str) -> None:
        if _tqdm is None:
            raise RuntimeError("tqdm is not available.")
        self._bar = _tqdm(
            total=max(0, int(total)),
            desc=str(desc),
            dynamic_ncols=True,
            leave=True,
        )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and plot a 4x3 grid of policy dynamics over topologies for "
            "M1/M2/Prop/Equal/Adaptive/RL-Agent."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=Path(str(SCRIPT_DEFAULTS["run_dir"])))
    parser.add_argument("--checkpoint-name", type=str, default=str(SCRIPT_DEFAULTS["checkpoint_name"]))
    parser.add_argument("--m1-summary-json", type=Path, default=Path(str(SCRIPT_DEFAULTS["m1_summary_json"])))
    parser.add_argument("--m2-summary-json", type=Path, default=Path(str(SCRIPT_DEFAULTS["m2_summary_json"])))
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DEFAULTS["output_dir"])
    parser.add_argument("--output-subdir-name", type=str, default=str(SCRIPT_DEFAULTS["output_subdir_name"]))
    parser.add_argument("--topologies", type=str, default=str(SCRIPT_DEFAULTS["topologies"]))
    parser.add_argument("--episodes", type=int, default=int(SCRIPT_DEFAULTS["episodes"]))
    parser.add_argument("--max-steps", type=int, default=SCRIPT_DEFAULTS["max_steps"])
    parser.add_argument("--episode-length-override", type=int, default=SCRIPT_DEFAULTS["episode_length_override"])
    parser.add_argument("--env-r", dest="env_r_override", type=float, default=SCRIPT_DEFAULTS["env_r_override"])
    parser.add_argument("--env-p-c", dest="env_p_c_override", type=float, default=SCRIPT_DEFAULTS["env_p_c_override"])
    parser.add_argument("--env-beta", dest="env_beta_override", type=float, default=SCRIPT_DEFAULTS["env_beta_override"])
    parser.add_argument(
        "--env-strategy-update-rule",
        dest="env_strategy_update_rule_override",
        type=str,
        choices=("fermi", "q_learning", "q_learning_2x2", "imitate_best"),
        default=SCRIPT_DEFAULTS["env_strategy_update_rule_override"],
    )
    parser.add_argument("--env-p-max", dest="env_p_max_override", type=float, default=SCRIPT_DEFAULTS["env_p_max_override"])
    parser.add_argument(
        "--env-initial-resource",
        dest="env_initial_resource_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_initial_resource_override"],
    )
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
    parser.add_argument(
        "--agent-graph-bank-index",
        type=int,
        default=int(SCRIPT_DEFAULTS["agent_graph_bank_index"]),
        help="Used only when --agent-graph-episode-sampling=fixed_index.",
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
        "--m2-ego-resource-x-mode",
        type=str,
        choices=("absolute", "normalized"),
        default=str(SCRIPT_DEFAULTS["m2_ego_resource_x_mode"]),
    )
    parser.add_argument(
        "--m2-ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["m2_ego_resource_normalizer"],
    )
    parser.add_argument(
        "--m2-saved-ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["m2_saved_ego_resource_normalizer"],
    )
    parser.add_argument(
        "--pool-power-k",
        type=float,
        default=float(SCRIPT_DEFAULTS["pool_power_k"]),
        help="Exponent k used by the Adaptive PoolPowerMixAllocationPolicy.",
    )
    parser.add_argument("--seed", type=int, default=SCRIPT_DEFAULTS["seed"])
    parser.add_argument("--figsize-width", type=float, default=float(SCRIPT_DEFAULTS["figsize"][0]))
    parser.add_argument("--figsize-height", type=float, default=float(SCRIPT_DEFAULTS["figsize"][1]))
    parser.add_argument("--legend-ncol", type=int, default=int(SCRIPT_DEFAULTS["legend_ncol"]))
    parser.add_argument("--legend-loc", type=str, default=str(SCRIPT_DEFAULTS["legend_loc"]))
    parser.add_argument("--legend-bbox-x", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_x"]))
    parser.add_argument("--legend-bbox-y", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_y"]))
    parser.add_argument("--marker-count", type=int, default=int(SCRIPT_DEFAULTS["marker_count"]))
    parser.add_argument("--marker-size", type=float, default=float(SCRIPT_DEFAULTS["marker_size"]))
    parser.add_argument("--num-workers", type=int, default=int(SCRIPT_DEFAULTS["num_workers"]))
    parser.add_argument("--device", type=str, default=str(SCRIPT_DEFAULTS["device"]))
    return parser.parse_args()


def _coerce_path(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
    windows_match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if windows_match and os.name != "nt":
        drive = windows_match.group(1).lower()
        suffix = windows_match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{suffix}").expanduser().resolve()
    return Path(text).expanduser().resolve()


def _ordered_topologies(topologies: Sequence[str]) -> list[str]:
    selected = list(parse_topology_list(",".join(str(item) for item in topologies)))
    ordered = [name for name in TOPOLOGY_ORDER if name in selected]
    extras = [name for name in selected if name not in ordered]
    return ordered + extras


def _build_policy_objects(
    *,
    m1_summary_json: Path,
    m2_summary_json: Path,
    m2_ego_resource_x_mode: str,
    m2_ego_resource_normalizer: float | None,
    m2_saved_ego_resource_normalizer: float | None,
    pool_power_k: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    m1_policy, m1_payload, _, _ = _load_saved_binned_policy(m1_summary_json)
    m2_policy, m2_payload, m2_profile_summary, _, _ = _load_saved_conditioned_policy(
        m2_summary_json,
        replay_ego_resource_x_mode=str(m2_ego_resource_x_mode),
        replay_ego_resource_normalizer=m2_ego_resource_normalizer,
        saved_ego_resource_normalizer=m2_saved_ego_resource_normalizer,
    )
    return (
        {
            "m1": m1_policy,
            "m2": m2_policy,
            "prop": ProportionalContributionPolicy(),
            "equal": UniformAllocationPolicy(),
            "adaptive": PoolPowerMixAllocationPolicy(power_k=float(pool_power_k)),
        },
        {
            "m1_summary_json": str(m1_summary_json),
            "m1_payload": dict(m1_payload),
            "m2_summary_json": str(m2_summary_json),
            "m2_payload_topology": str(m2_payload.get("topology", "")),
            "m2_profile_summary": dict(m2_profile_summary),
            "adaptive_policy": "PoolPowerMixAllocationPolicy",
            "pool_power_k": float(pool_power_k),
        },
    )


def _summarize_topology_rows(
    *,
    topology_name: str,
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name in POLICY_ORDER:
        policy_episode_rows = [row for row in episode_rows if str(row.get("policy")) == str(policy_name)]
        if not policy_episode_rows:
            continue
        policy_step_rows = [row for row in step_rows if str(row.get("policy")) == str(policy_name)]
        rows.append(
            {
                "topology": str(topology_name),
                "topology_label": str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
                "policy": str(policy_name),
                "policy_label": str(POLICY_LABELS.get(str(policy_name), policy_name)),
                "episodes": int(len(policy_episode_rows)),
                "return_mean": float(np.mean([float(row["total_reward"]) for row in policy_episode_rows])),
                "reward_per_step_mean": float(np.mean([float(row["mean_reward"]) for row in policy_episode_rows])),
                "final_cooperation_mean": float(np.mean([float(row["final_cooperation_rate"]) for row in policy_episode_rows])),
                "final_gini_mean": float(np.mean([float(row["final_gini"]) for row in policy_episode_rows])),
                "final_mean_resource_mean": float(np.mean([float(row["final_mean_resource"]) for row in policy_episode_rows])),
                "mean_self_allocation_mean": (
                    float(np.mean([float(row["mean_self_allocation_mean"]) for row in policy_step_rows]))
                    if policy_step_rows
                    else float("nan")
                ),
            }
        )
    return rows


def _plot_policy_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    topologies: Sequence[str],
    output_stem: Path,
    figsize: tuple[float, float],
    legend_ncol: int,
    legend_loc: str,
    legend_bbox_x: float,
    legend_bbox_y: float,
    marker_count: int,
    marker_size: float,
    time_axis_scale: str,
    metric_axis_scale: str,
) -> None:
    if ar.plt is None:
        return

    figure, axes = ar.plt.subplots(
        len(topologies),
        len(METRIC_SPECS),
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(len(topologies), len(METRIC_SPECS))

    for row_index, topology_name in enumerate(topologies):
        topology_rows = [row for row in rows if str(row.get("topology")) == str(topology_name)]
        topology_label = TOPOLOGY_LABELS.get(str(topology_name), str(topology_name))
        for col_index, (metric_key, ylabel) in enumerate(METRIC_SPECS):
            axis = axes_array[row_index, col_index]
            panel_index = row_index * len(METRIC_SPECS) + col_index
            for policy_name in POLICY_ORDER:
                policy_rows = [row for row in topology_rows if str(row.get("policy")) == str(policy_name)]
                if not policy_rows:
                    continue
                policy_rows.sort(key=lambda row: int(row["step"]))
                x_values = np.asarray([int(row["step"]) for row in policy_rows], dtype=np.int64)
                y_values = np.asarray([float(row[metric_key]) for row in policy_rows], dtype=np.float64)
                style = POLICY_STYLES[str(policy_name)]
                axis.plot(
                    x_values,
                    y_values,
                    label=str(style["label"]),
                    color=str(style["color"]),
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker", None),
                    markersize=float(marker_size),
                    markerfacecolor="white",
                    markeredgecolor=str(style["color"]),
                    markeredgewidth=1.1,
                    markevery=_build_even_markevery(
                        x_values,
                        axis_scale=str(time_axis_scale),
                        target_count=max(2, int(marker_count)),
                    ),
                    linewidth=2.5 if policy_name == "agent" else 2.0,
                    zorder=float(style.get("zorder", 3)),
                )
                pp._apply_axis_scale(axis, axis_name="x", requested_scale=str(time_axis_scale), values=x_values)
                if str(metric_axis_scale) == "linear":
                    axis.set_yscale("linear")
                    pp._apply_metric_ylim(axis, metric_key=str(metric_key), requested_scale="linear", values=y_values)
                else:
                    pp._apply_axis_scale(axis, axis_name="y", requested_scale=str(metric_axis_scale), values=y_values)

            if row_index == 0:
                axis.set_title(str(ylabel), fontweight=pp.FONT_WEIGHT_TITLE, pad=10.0)
            if row_index == len(topologies) - 1:
                axis.set_xlabel("t", fontweight=pp.FONT_WEIGHT_BOLD)
            if col_index == 0:
                axis.set_ylabel(f"{topology_label}\n{ylabel}", fontweight=pp.FONT_WEIGHT_BOLD)
            else:
                axis.set_ylabel(str(ylabel), fontweight=pp.FONT_WEIGHT_BOLD)
            axis.grid(False)
            axis.margins(x=0.02)
            axis.text(
                0.02,
                0.98,
                _panel_label(panel_index),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontfamily=pp.FONT_FAMILY,
                fontweight=pp.FONT_WEIGHT_BOLD,
                fontsize=15,
            )
            pp._style_axis_text(axis, title_fontsize=16, axis_label_fontsize=14, tick_labelsize=12)

    handles, labels = axes_array[0, 0].get_legend_handles_labels()
    if handles:
        legend = figure.legend(
            handles,
            labels,
            loc=str(legend_loc),
            bbox_to_anchor=(float(legend_bbox_x), float(legend_bbox_y)),
            ncol=max(1, int(legend_ncol)),
            frameon=False,
            prop={"family": pp.FONT_FAMILY, "weight": pp.FONT_WEIGHT_BOLD, "size": 13},
        )
        pp._style_legend_text(legend, fontsize=13)

    pp._save_figure_pdf_png(figure, output_stem)


def _topology_sort_key(topology_name: str) -> tuple[int, str]:
    text = str(topology_name)
    try:
        return (TOPOLOGY_ORDER.index(text), text)
    except ValueError:
        return (len(TOPOLOGY_ORDER), text)


def _policy_sort_key(policy_name: str) -> tuple[int, str]:
    text = str(policy_name)
    try:
        return (POLICY_ORDER.index(text), text)
    except ValueError:
        return (len(POLICY_ORDER), text)


def _sort_step_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _topology_sort_key(str(row.get("topology", ""))),
            _policy_sort_key(str(row.get("policy", ""))),
            int(row.get("step", 0)),
        ),
    )


def _sort_episode_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _topology_sort_key(str(row.get("topology", ""))),
            _policy_sort_key(str(row.get("policy", ""))),
            int(row.get("episode", 0)),
        ),
    )


def _sort_graph_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _topology_sort_key(str(row.get("topology", ""))),
            int(row.get("episode", 0)),
        ),
    )


def _sort_topology_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _topology_sort_key(str(row.get("topology", ""))),
            _policy_sort_key(str(row.get("policy", ""))),
        ),
    )


def _run_single_topology_worker(
    *,
    run_dir: str,
    checkpoint_name: str,
    m1_summary_json: str,
    m2_summary_json: str,
    m2_ego_resource_x_mode: str,
    m2_ego_resource_normalizer: float | None,
    m2_saved_ego_resource_normalizer: float | None,
    pool_power_k: float,
    experiment_spec: Mapping[str, Any],
    reference_mean_degree: float,
    topology_name: str,
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
    env_strategy_update_rule_override: str | None,
    env_p_max_override: float | None,
    env_initial_resource_override: float | None,
    env_initial_cooperation_prob_override: float | None,
    device_name: str,
) -> dict[str, Any]:
    policy_objects, _ = _build_policy_objects(
        m1_summary_json=Path(str(m1_summary_json)),
        m2_summary_json=Path(str(m2_summary_json)),
        m2_ego_resource_x_mode=str(m2_ego_resource_x_mode),
        m2_ego_resource_normalizer=m2_ego_resource_normalizer,
        m2_saved_ego_resource_normalizer=m2_saved_ego_resource_normalizer,
        pool_power_k=float(pool_power_k),
    )
    device = torch.device(str(device_name))
    actor, _ = load_actor_from_run_dir(Path(str(run_dir)), str(checkpoint_name), device)

    step_rows, _, episode_rows, graph_rows, graph_metadata = pp._run_single_topology(
        actor=actor,
        experiment_spec=experiment_spec,
        reference_mean_degree=float(reference_mean_degree),
        topology_name=str(topology_name),
        selected_policies=list(POLICY_ORDER),
        policy_objects=policy_objects,
        episodes=max(1, int(episodes)),
        max_steps=None if max_steps is None else int(max_steps),
        episode_length_override=None if episode_length_override is None else int(episode_length_override),
        rollout_seed=int(rollout_seed),
        graph_source_mode=str(graph_source_mode),
        agent_graph_episode_sampling=str(agent_graph_episode_sampling),
        agent_graph_bank_index=int(agent_graph_bank_index),
        env_r_override=env_r_override,
        env_p_c_override=env_p_c_override,
        env_beta_override=env_beta_override,
        env_strategy_update_rule_override=env_strategy_update_rule_override,
        env_p_max_override=env_p_max_override,
        env_initial_resource_override=env_initial_resource_override,
        env_initial_cooperation_prob_override=env_initial_cooperation_prob_override,
        agent_csv_final_fc_min_for_ws_er=0.85,
        agent_csv_min_kept_episodes_for_ws_er=20,
        agent_csv_max_attempt_episodes_for_ws_er=800,
        agent_plot_final_fc_min=0.85,
        agent_plot_final_gini_max_for_ws_er=0.8,
        progress_bar=None,
    )
    topology_summary_rows = _summarize_topology_rows(
        topology_name=str(topology_name),
        episode_rows=episode_rows,
        step_rows=step_rows,
    )
    return {
        "topology_name": str(topology_name),
        "step_rows": list(step_rows),
        "episode_rows": list(episode_rows),
        "graph_rows": list(graph_rows),
        "graph_metadata": dict(graph_metadata),
        "topology_summary_rows": list(topology_summary_rows),
    }


def main() -> None:
    pp._configure_matplotlib_fonts()
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found under {run_dir}")

    m1_summary_json = _coerce_path(args.m1_summary_json)
    m2_summary_json = _coerce_path(args.m2_summary_json)
    if not m1_summary_json.exists():
        raise FileNotFoundError(f"M1 summary JSON not found: {m1_summary_json}")
    if not m2_summary_json.exists():
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

    policy_objects, policy_metadata = _build_policy_objects(
        m1_summary_json=m1_summary_json,
        m2_summary_json=m2_summary_json,
        m2_ego_resource_x_mode=str(args.m2_ego_resource_x_mode),
        m2_ego_resource_normalizer=args.m2_ego_resource_normalizer,
        m2_saved_ego_resource_normalizer=args.m2_saved_ego_resource_normalizer,
        pool_power_k=float(args.pool_power_k),
    )

    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)
    topologies = _ordered_topologies(parse_topology_list(str(args.topologies)))
    graph_source_mode = normalize_graph_source_mode(str(args.graph_source_mode))
    seed_base = int(experiment_spec.get("seed", 42)) if args.seed is None else int(args.seed)

    device = torch.device(str(args.device))
    actor, _ = load_actor_from_run_dir(run_dir, str(args.checkpoint_name), device)

    all_step_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    all_graph_rows: list[dict[str, Any]] = []
    all_topology_summary_rows: list[dict[str, Any]] = []
    topology_metadata: dict[str, Any] = {}

    progress_bar = _build_progress_bar(
        total=max(1, len(topologies) * len(POLICY_ORDER)),
        desc="Running six-policy dynamics",
    )
    try:
        for topology_name in topologies:
            step_rows, _, episode_rows, graph_rows, graph_metadata = pp._run_single_topology(
                actor=actor,
                experiment_spec=experiment_spec,
                reference_mean_degree=reference_mean_degree,
                topology_name=str(topology_name),
                selected_policies=list(POLICY_ORDER),
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
                env_r_override=args.env_r_override,
                env_p_c_override=args.env_p_c_override,
                env_beta_override=args.env_beta_override,
                env_strategy_update_rule_override=args.env_strategy_update_rule_override,
                env_p_max_override=args.env_p_max_override,
                env_initial_resource_override=args.env_initial_resource_override,
                env_initial_cooperation_prob_override=args.env_initial_cooperation_prob_override,
                agent_csv_final_fc_min_for_ws_er=0.85,
                agent_csv_min_kept_episodes_for_ws_er=20,
                agent_csv_max_attempt_episodes_for_ws_er=800,
                agent_plot_final_fc_min=0.85,
                agent_plot_final_gini_max_for_ws_er=0.8,
                progress_bar=progress_bar,
            )
            all_step_rows.extend(step_rows)
            all_episode_rows.extend(episode_rows)
            all_graph_rows.extend(graph_rows)
            all_topology_summary_rows.extend(
                _summarize_topology_rows(
                    topology_name=str(topology_name),
                    episode_rows=episode_rows,
                    step_rows=step_rows,
                )
            )
            topology_metadata[str(topology_name)] = dict(graph_metadata)
    finally:
        progress_bar.close()

    ar.write_csv(output_dir / "policy_rollout_step_metrics.csv", all_step_rows)
    ar.write_csv(output_dir / "episode_summary.csv", all_episode_rows)
    ar.write_csv(output_dir / "episode_graph_metadata.csv", all_graph_rows)
    ar.write_csv(output_dir / "topology_summary.csv", all_topology_summary_rows)

    _plot_policy_grid(
        all_step_rows,
        topologies=topologies,
        output_stem=output_dir / "six_policy_dynamics_grid",
        figsize=(float(args.figsize_width), float(args.figsize_height)),
        legend_ncol=int(args.legend_ncol),
        legend_loc=str(args.legend_loc),
        legend_bbox_x=float(args.legend_bbox_x),
        legend_bbox_y=float(args.legend_bbox_y),
        marker_count=int(args.marker_count),
        marker_size=float(args.marker_size),
        time_axis_scale=str(args.time_axis_scale),
        metric_axis_scale=str(args.metric_axis_scale),
    )

    summary_payload = {
        "run_dir": str(run_dir),
        "checkpoint_name": str(args.checkpoint_name),
        "topologies": [str(item) for item in topologies],
        "policies": [str(item) for item in POLICY_ORDER],
        "episodes": int(args.episodes),
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "episode_length_override": None if args.episode_length_override is None else int(args.episode_length_override),
        "graph_source_mode": str(graph_source_mode),
        "agent_graph_episode_sampling": str(args.agent_graph_episode_sampling),
        "agent_graph_bank_index": int(args.agent_graph_bank_index),
        "seed": int(seed_base),
        "env_overrides": {
            "r": args.env_r_override,
            "p_c": args.env_p_c_override,
            "beta": args.env_beta_override,
            "strategy_update_rule": args.env_strategy_update_rule_override,
            "p_max": args.env_p_max_override,
            "initial_resource": args.env_initial_resource_override,
            "initial_cooperation_prob": args.env_initial_cooperation_prob_override,
        },
        "policy_sources": dict(policy_metadata),
        "topology_graph_metadata": dict(topology_metadata),
        "output_files": {
            "step_metrics_csv": str(output_dir / "policy_rollout_step_metrics.csv"),
            "episode_summary_csv": str(output_dir / "episode_summary.csv"),
            "topology_summary_csv": str(output_dir / "topology_summary.csv"),
            "figure_png": str((output_dir / "six_policy_dynamics_grid.png").resolve()),
            "figure_pdf": str((output_dir / "six_policy_dynamics_grid.pdf").resolve()),
        },
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
