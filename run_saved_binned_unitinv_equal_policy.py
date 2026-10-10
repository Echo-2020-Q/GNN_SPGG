from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import analyze_rule_allocation_over_time as ar
from analyze_gnn_policy_decisions import (
    _load_json,
    build_env_config_from_spec,
    build_graph_from_spec,
    build_spec_for_topology,
    load_actor_from_run_dir,
    parse_topology_list,
    resolve_reference_mean_degree,
    run_deterministic_rollouts,
)
from Project1.env import SPGGEnv


SCRIPT_DEFAULTS = {
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    "checkpoint_name": "best_eval.pt",
    "saved_profile_summary_json": (
        "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
        "0429_binned_three_way_profile/regular/binned_unitinv_equal_profile_summary.json"
    ),
    "output_dir": None,
    "output_subdir_name": "0429_saved_binned_unitinv_equal_rollout",
    "topologies": "regular",
    "episodes": 10,
    "max_steps": 200,
    "episode_length_override": 200,
    "env_r_override": None,
    "env_r_list": None,
    "env_p_c_override": None,
    "env_beta_override": None,
    "env_strategy_update_rule_override": None,
    "env_p_max_override": None,
    "env_initial_resource_override": None,
    "env_initial_cooperation_prob_override": None,
    "seed": None,
    "compare_with_agent": True,
    "early_alignment_max_step": 15,
    "plot_final_fc_min": 0.9,
    "triptych_output_stem": "regular_policy_dynamics_triptych",
    "triptych_figsize": (12.0, 4.0),
    "triptych_axis_label_fontsize": 16,
    "triptych_tick_labelsize": 14,
    "triptych_legend_fontsize": 14,
    "triptych_legend_ncol": 1,
    "triptych_legend_bbox_x": 0.5,
    "triptych_legend_bbox_y": 1.02,
    "triptych_m1_color": "#3F9A3F",
    "triptych_agent_color": "#E2A802",
    "time_axis_scale": "symlog",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


FONT_FAMILY = "Times New Roman"
FONT_WEIGHT_BOLD = "bold"
FONT_WEIGHT_TITLE = "normal"

TRIPTYCH_METRIC_SPECS = [
    ("cooperation_rate_mean", r"$f_c$"),
    ("mean_resource_mean", r"$\bar{R}(t)$"),
    ("resource_gini_mean", "Gini"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a saved discrete Unit-Investment + Equal profile on selected topologies."
    )
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
        help="Checkpoint filename inside <run-dir>/checkpoints/ when comparing against RL-Agent.",
    )
    parser.add_argument(
        "--saved-profile-summary-json",
        type=Path,
        default=Path(str(SCRIPT_DEFAULTS["saved_profile_summary_json"])),
        help="Path to the JSON exported by build_binned_unitinv_equal_profile.py.",
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
        default=None if SCRIPT_DEFAULTS["output_subdir_name"] is None else str(SCRIPT_DEFAULTS["output_subdir_name"]),
        help="When --output-dir is not set, write outputs under <run-dir>/<output-subdir-name>/.",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default=str(SCRIPT_DEFAULTS["topologies"]),
        help="Comma-separated topology list, e.g. regular,ba or regular,er,ws,ba.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(SCRIPT_DEFAULTS["episodes"]),
        help="Number of deterministic episodes to roll out.",
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
        help="Override dynamics.r during analysis rollouts.",
    )
    parser.add_argument(
        "--env-r-list",
        type=str,
        default=SCRIPT_DEFAULTS["env_r_list"],
        help="Optional comma-separated sweep of dynamics.r values, e.g. 0.25,0.50,0.75,1.00.",
    )
    parser.add_argument(
        "--env-p-c",
        dest="env_p_c_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_p_c_override"],
        help="Override dynamics.p_c during analysis rollouts.",
    )
    parser.add_argument(
        "--env-beta",
        dest="env_beta_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_beta_override"],
        help="Override dynamics.beta during analysis rollouts.",
    )
    parser.add_argument(
        "--env-strategy-update-rule",
        dest="env_strategy_update_rule_override",
        type=str,
        choices=("fermi", "q_learning", "q_learning_2x2", "imitate_best"),
        default=SCRIPT_DEFAULTS["env_strategy_update_rule_override"],
        help="Override dynamics.strategy_update_rule during analysis rollouts.",
    )
    parser.add_argument(
        "--env-p-max",
        dest="env_p_max_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_p_max_override"],
        help="Override dynamics.p_max during analysis rollouts.",
    )
    parser.add_argument(
        "--env-initial-resource",
        dest="env_initial_resource_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_initial_resource_override"],
        help="Override dynamics.initial_resource during analysis rollouts.",
    )
    parser.add_argument(
        "--env-initial-cooperation-prob",
        dest="env_initial_cooperation_prob_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_initial_cooperation_prob_override"],
        help="Override dynamics.initial_cooperation_prob during analysis rollouts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SCRIPT_DEFAULTS["seed"],
        help="Shared graph/reset seed. Defaults to the run seed from results.json when omitted.",
    )
    parser.add_argument(
        "--compare-with-agent",
        dest="compare_with_agent",
        action="store_true",
        help="Also run RL-Agent on the same topologies for direct comparison.",
    )
    parser.add_argument(
        "--no-compare-with-agent",
        dest="compare_with_agent",
        action="store_false",
        help="Only run the saved discrete policy.",
    )
    parser.set_defaults(compare_with_agent=bool(SCRIPT_DEFAULTS["compare_with_agent"]))
    parser.add_argument(
        "--early-alignment-max-step",
        type=int,
        default=int(SCRIPT_DEFAULTS["early_alignment_max_step"]),
        help="Maximum step used when computing alignment against RL-Agent.",
    )
    parser.add_argument(
        "--plot-final-fc-min",
        type=float,
        default=float(SCRIPT_DEFAULTS["plot_final_fc_min"]),
        help="Only keep episodes with final_cooperation_rate >= this threshold when building the triptych plot.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(SCRIPT_DEFAULTS["device"]),
        help="Torch device, e.g. cpu or cuda:0.",
    )
    return parser.parse_args()


def _parse_env_r_list(raw_value: str | None) -> list[float] | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    values: list[float] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(float(item))
    return values or None


def _coerce_optional_scalar(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) > 1:
            raise ValueError(
                f"{name} expects a single numeric value here, but got multiple values: {value}. "
                f"Use a single value like 75 or [75]."
            )
        value = value[0]
    return float(value)


def _format_r_tag(r_value: float) -> str:
    text = f"{float(r_value):.2f}"
    return f"r{text.replace('-', 'm').replace('.', 'p')}"


def _configure_matplotlib_fonts() -> None:
    if ar.matplotlib is None:
        return
    ar.matplotlib.rcParams["font.family"] = FONT_FAMILY
    ar.matplotlib.rcParams["font.weight"] = FONT_WEIGHT_BOLD
    ar.matplotlib.rcParams["mathtext.fontset"] = "custom"
    ar.matplotlib.rcParams["mathtext.rm"] = FONT_FAMILY
    ar.matplotlib.rcParams["mathtext.it"] = f"{FONT_FAMILY}:italic"
    ar.matplotlib.rcParams["mathtext.bf"] = f"{FONT_FAMILY}:bold"
    ar.matplotlib.rcParams["axes.unicode_minus"] = False


def _style_axis_text(
    axis: Any,
    *,
    axis_label_fontsize: float | None = None,
    tick_labelsize: float | None = None,
) -> None:
    xlabel = axis.xaxis.get_label()
    ylabel = axis.yaxis.get_label()
    xlabel.set_fontfamily(FONT_FAMILY)
    ylabel.set_fontfamily(FONT_FAMILY)
    xlabel.set_fontweight(FONT_WEIGHT_BOLD)
    ylabel.set_fontweight(FONT_WEIGHT_BOLD)
    if axis_label_fontsize is not None:
        xlabel.set_fontsize(axis_label_fontsize)
        ylabel.set_fontsize(axis_label_fontsize)
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontfamily(FONT_FAMILY)
        tick_label.set_fontweight(FONT_WEIGHT_BOLD)
        if tick_labelsize is not None:
            tick_label.set_fontsize(tick_labelsize)


def _style_legend_text(legend: Any, *, fontsize: float | None = None) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontweight(FONT_WEIGHT_BOLD)
        if fontsize is not None:
            text.set_fontsize(fontsize)


def _save_figure_pdf_png(figure: Any, output_stem: Path) -> None:
    png_path = ar._prepare_output_path(Path(output_stem).with_suffix(".png"))
    pdf_path = ar._prepare_output_path(Path(output_stem).with_suffix(".pdf"))
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(pdf_path, dpi=600, bbox_inches="tight")
    if ar.plt is not None:
        ar.plt.close(figure)


def _apply_time_axis_scale(axis: Any, *, x_values: np.ndarray, requested_scale: str) -> None:
    scale = str(requested_scale).strip().lower()
    if scale == "symlog":
        axis.set_xscale("symlog", linthresh=1.0)
        return
    if scale == "log":
        positive_values = x_values[np.isfinite(x_values) & (x_values > 0)]
        if positive_values.size > 0:
            axis.set_xscale("log")
        else:
            axis.set_xscale("linear")
        return
    axis.set_xscale("linear")


def _force_plot_fc_start(
    *,
    metric_key: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    adjusted = np.asarray(y_values, dtype=np.float64).copy()
    if str(metric_key) != "cooperation_rate_mean":
        return adjusted
    if adjusted.size == 0:
        return adjusted
    zero_mask = np.asarray(x_values, dtype=np.int64) == 0
    if np.any(zero_mask):
        adjusted[zero_mask] = 0.5
    return adjusted


def _build_plot_episode_allowlist(
    episode_rows: list[dict[str, Any]],
    *,
    final_fc_min: float,
) -> dict[str, set[int]]:
    allowlist: dict[str, set[int]] = {}
    threshold = float(final_fc_min)
    for row in episode_rows:
        try:
            final_fc = float(row.get("final_cooperation_rate", float("nan")))
            policy_name = str(row.get("policy", ""))
            episode_index = int(row.get("episode", -1))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(final_fc) or not policy_name or episode_index < 0:
            continue
        if final_fc >= threshold:
            allowlist.setdefault(policy_name, set()).add(int(episode_index))
    return allowlist


def _filter_episode_rows_for_plot(
    episode_rows: list[dict[str, Any]],
    *,
    allowlist: Mapping[str, set[int]],
) -> list[dict[str, Any]]:
    filtered_rows: list[dict[str, Any]] = []
    for row in episode_rows:
        try:
            policy_name = str(row.get("policy", ""))
            episode_index = int(row.get("episode", -1))
        except (TypeError, ValueError):
            continue
        if episode_index < 0:
            continue
        allowed_episodes = allowlist.get(policy_name, set())
        if int(episode_index) in allowed_episodes:
            filtered_rows.append(dict(row))
    return filtered_rows


def _filter_snapshots_by_plot_episode_allowlist(
    snapshots_by_policy: Mapping[str, list[Any]],
    *,
    allowlist: Mapping[str, set[int]],
) -> dict[str, list[Any]]:
    filtered: dict[str, list[Any]] = {}
    for policy_name, snapshots in snapshots_by_policy.items():
        allowed_episodes = allowlist.get(str(policy_name), set())
        filtered[str(policy_name)] = [
            snapshot for snapshot in snapshots if int(getattr(snapshot, "episode", -1)) in allowed_episodes
        ]
    return filtered


def _plot_policy_dynamics_triptych(
    step_rows: list[dict[str, Any]],
    output_stem: Path,
    *,
    saved_policy_name: str,
    compare_with_agent: bool,
) -> None:
    if ar.plt is None or not step_rows:
        return

    figure, axes = ar.plt.subplots(
        1,
        3,
        figsize=tuple(SCRIPT_DEFAULTS["triptych_figsize"]),
        constrained_layout=True,
    )
    axes_flat = list(np.asarray(axes).reshape(-1))
    triptych_axis_label_fontsize = float(SCRIPT_DEFAULTS["triptych_axis_label_fontsize"])
    triptych_tick_labelsize = float(SCRIPT_DEFAULTS["triptych_tick_labelsize"])
    triptych_legend_fontsize = float(SCRIPT_DEFAULTS["triptych_legend_fontsize"])
    triptych_legend_ncol = int(SCRIPT_DEFAULTS["triptych_legend_ncol"])
    triptych_legend_bbox_x = float(SCRIPT_DEFAULTS["triptych_legend_bbox_x"])
    triptych_legend_bbox_y = float(SCRIPT_DEFAULTS["triptych_legend_bbox_y"])
    time_axis_scale = str(SCRIPT_DEFAULTS["time_axis_scale"])

    policy_styles: list[tuple[str, str, str]] = [(str(saved_policy_name), "M1", str(SCRIPT_DEFAULTS["triptych_m1_color"]))]
    if bool(compare_with_agent):
        policy_styles.append(("agent", "RL-Agent", str(SCRIPT_DEFAULTS["triptych_agent_color"])))

    for axis, (metric_key, ylabel) in zip(axes_flat, TRIPTYCH_METRIC_SPECS):
        for policy_name, legend_label, color in policy_styles:
            policy_rows = [row for row in step_rows if str(row.get("policy")) == str(policy_name)]
            if not policy_rows:
                continue
            policy_rows.sort(key=lambda row: int(row["step"]))
            x_values = np.asarray([int(row["step"]) for row in policy_rows], dtype=np.int64)
            y_values = np.asarray([float(row[metric_key]) for row in policy_rows], dtype=np.float64)
            y_values = _force_plot_fc_start(metric_key=str(metric_key), x_values=x_values, y_values=y_values)
            axis.plot(
                x_values,
                y_values,
                label=str(legend_label),
                color=str(color),
                linewidth=2.6,
            )
            _apply_time_axis_scale(axis, x_values=x_values, requested_scale=time_axis_scale)
        axis.set_xlabel("t", fontsize=triptych_axis_label_fontsize, fontweight=FONT_WEIGHT_BOLD, fontfamily=FONT_FAMILY)
        axis.set_ylabel(
            str(ylabel),
            fontsize=triptych_axis_label_fontsize,
            fontweight=FONT_WEIGHT_BOLD,
            fontfamily=FONT_FAMILY,
        )
        axis.grid(False)
        axis.margins(x=0.03)
        if str(metric_key) == "cooperation_rate_mean":
            axis.set_ylim(-0.03, 1.03)
        elif str(metric_key) == "resource_gini_mean":
            axis.set_ylim(-0.03, 1.03)
        _style_axis_text(
            axis,
            axis_label_fontsize=triptych_axis_label_fontsize,
            tick_labelsize=triptych_tick_labelsize,
        )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        legend = figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(triptych_legend_bbox_x, triptych_legend_bbox_y),
            ncol=min(len(handles), max(1, triptych_legend_ncol)),
            frameon=False,
            prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT_BOLD, "size": triptych_legend_fontsize},
        )
        _style_legend_text(legend, fontsize=triptych_legend_fontsize)
    _save_figure_pdf_png(figure, output_stem)


def _coerce_path(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
    windows_match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if windows_match and os.name != "nt":
        drive = windows_match.group(1).lower()
        suffix = windows_match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{suffix}").expanduser().resolve()
    return Path(text).expanduser().resolve()


def _load_saved_binned_policy(
    summary_path: Path,
) -> tuple[ar.BinnedEgoResourceMixtureAllocationPolicy, dict[str, Any], str, str]:
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Saved profile summary at {summary_path} is not a JSON object.")

    policy_summary = payload.get("policy_summary")
    if not isinstance(policy_summary, Mapping):
        raise KeyError(f"{summary_path} does not contain a policy_summary object.")

    bin_edges_resource = policy_summary.get("bin_edges_resource")
    bin_weight_rows = policy_summary.get("bin_weight_rows")
    if bin_edges_resource is None or bin_weight_rows is None:
        raise KeyError(f"{summary_path} policy_summary is missing bin_edges_resource or bin_weight_rows.")

    saved_policy = ar.BinnedEgoResourceMixtureAllocationPolicy(
        mode=str(policy_summary.get("mode", "binned_3way")),
        bin_edges_resource=bin_edges_resource,
        bin_weight_rows=bin_weight_rows,
        profile_source=f"saved_profile:{summary_path}",
        fit_summary=dict(policy_summary),
        tail_cap_mode=str(policy_summary.get("tail_cap_mode", "none")),
        tail_cap_resource=policy_summary.get("tail_cap_resource"),
        tail_cap_soft_scale=float(policy_summary.get("tail_cap_soft_scale", 0.25)),
        ego_resource_x_mode=str(policy_summary.get("ego_resource_x_mode", "absolute")),
        ego_resource_normalizer=policy_summary.get("ego_resource_normalizer"),
    )
    policy_name = str(payload.get("policy_name", "saved_binned_unitinv_equal_mix"))
    policy_label = str(payload.get("policy_label", "Saved Binned UnitInv + Equal"))
    return saved_policy, dict(payload), policy_name, policy_label


def main() -> None:
    _configure_matplotlib_fonts()
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found under {run_dir}")

    results_payload = _load_json(results_path)
    experiment_spec = results_payload["experiment"] if "experiment" in results_payload else results_payload
    if not isinstance(experiment_spec, dict):
        raise TypeError("Could not extract experiment spec from results.json.")

    saved_profile_summary_path = _coerce_path(args.saved_profile_summary_json)
    if not saved_profile_summary_path.exists():
        raise FileNotFoundError(f"Saved profile summary JSON not found: {saved_profile_summary_path}")
    saved_policy, saved_profile_payload, saved_policy_name, saved_policy_label = _load_saved_binned_policy(
        saved_profile_summary_path
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            (run_dir / args.output_subdir_name).resolve()
            if args.output_subdir_name is not None
            else (run_dir / "saved_binned_unitinv_equal_rollout").resolve()
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    topologies = parse_topology_list(str(args.topologies))
    env_r_values = _parse_env_r_list(args.env_r_list)
    if env_r_values is None:
        env_r_values = [None if args.env_r_override is None else float(args.env_r_override)]

    episodes = max(1, int(args.episodes))
    max_steps = None if args.max_steps is None else int(args.max_steps)
    seed_base = int(experiment_spec.get("seed", 42)) if args.seed is None else int(args.seed)

    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)
    device = torch.device(str(args.device))
    actor = None
    if bool(args.compare_with_agent):
        actor, _ = load_actor_from_run_dir(run_dir, args.checkpoint_name, device)

    ar.POLICY_STYLES["agent"] = {
        "color": str(SCRIPT_DEFAULTS["triptych_agent_color"]),
        "label": "RL-Agent",
    }
    ar.POLICY_STYLES[str(saved_policy_name)] = {
        "color": str(SCRIPT_DEFAULTS["triptych_m1_color"]),
        "label": "M1",
    }

    topology_summary_rows: list[dict[str, Any]] = []
    for env_r_value in env_r_values:
        r_output_dir = (
            output_dir
            if len(env_r_values) == 1
            else (output_dir / _format_r_tag(float(env_r_value if env_r_value is not None else 0.0)))
        )
        r_output_dir.mkdir(parents=True, exist_ok=True)

        for topology_name in topologies:
            topology_spec = build_spec_for_topology(
                experiment_spec,
                topology_name,
                reference_mean_degree=reference_mean_degree,
            )
            topology_spec = ar.apply_analysis_env_overrides(
                topology_spec,
                env_r_override=None if env_r_value is None else float(env_r_value),
                env_p_c_override=_coerce_optional_scalar(args.env_p_c_override, name="env_p_c_override"),
                env_beta_override=_coerce_optional_scalar(args.env_beta_override, name="env_beta_override"),
                env_strategy_update_rule_override=(
                    None if args.env_strategy_update_rule_override is None else str(args.env_strategy_update_rule_override)
                ),
                env_p_max_override=_coerce_optional_scalar(args.env_p_max_override, name="env_p_max_override"),
                env_initial_resource_override=_coerce_optional_scalar(
                    args.env_initial_resource_override,
                    name="env_initial_resource_override",
                ),
                env_initial_cooperation_prob_override=_coerce_optional_scalar(
                    args.env_initial_cooperation_prob_override,
                    name="env_initial_cooperation_prob_override",
                ),
            )
            topology_spec["seed"] = int(seed_base)
            if args.episode_length_override is not None:
                topology_spec["dynamics"] = dict(topology_spec["dynamics"])
                topology_spec["dynamics"]["episode_length"] = int(args.episode_length_override)

            topology_output_dir = r_output_dir if len(topologies) == 1 else (r_output_dir / topology_name)
            topology_output_dir.mkdir(parents=True, exist_ok=True)

            graph = build_graph_from_spec(topology_spec)
            env_config = build_env_config_from_spec(topology_spec, graph)
            env = SPGGEnv(env_config, graph)

            policy_specs: list[tuple[str, Any]] = []
            snapshots_by_policy: dict[str, list[Any]] = {}
            episode_rows: list[dict[str, Any]] = []

            if bool(args.compare_with_agent):
                agent_snapshots, agent_summaries = run_deterministic_rollouts(
                    actor,
                    env,
                    episodes=episodes,
                    seed=int(seed_base),
                    max_steps=max_steps,
                )
                snapshots_by_policy["agent"] = list(agent_snapshots)
                episode_rows.extend(ar.episode_summaries_to_rows("agent", agent_summaries))
                policy_specs.append(("agent", None))

            saved_snapshots, saved_summaries = ar.run_rule_policy_rollouts(
                saved_policy,
                env_config=env_config,
                graph=graph,
                episodes=episodes,
                seed=int(seed_base),
                max_steps=max_steps,
            )
            snapshots_by_policy[str(saved_policy_name)] = list(saved_snapshots)
            episode_rows.extend(ar.episode_summaries_to_rows(str(saved_policy_name), saved_summaries))
            policy_specs.append((str(saved_policy_name), saved_policy))
            ar.POLICY_SPECS = tuple(policy_specs)

            step_rows = ar.summarize_policy_rollout_step_metrics(snapshots_by_policy)
            ar.write_csv(topology_output_dir / "policy_rollout_step_metrics.csv", step_rows)
            ar.write_csv(topology_output_dir / "episode_summary.csv", episode_rows)

            plot_episode_allowlist = _build_plot_episode_allowlist(
                episode_rows,
                final_fc_min=float(args.plot_final_fc_min),
            )
            plot_episode_rows = _filter_episode_rows_for_plot(
                episode_rows,
                allowlist=plot_episode_allowlist,
            )
            plot_snapshots_by_policy = _filter_snapshots_by_plot_episode_allowlist(
                snapshots_by_policy,
                allowlist=plot_episode_allowlist,
            )
            plot_step_rows = ar.summarize_policy_rollout_step_metrics(plot_snapshots_by_policy)
            ar.write_csv(topology_output_dir / "episode_summary_plot_filtered.csv", plot_episode_rows)
            ar.write_csv(topology_output_dir / "policy_rollout_step_metrics_plot_filtered.csv", plot_step_rows)
            _plot_policy_dynamics_triptych(
                plot_step_rows,
                topology_output_dir / str(SCRIPT_DEFAULTS["triptych_output_stem"]),
                saved_policy_name=str(saved_policy_name),
                compare_with_agent=bool(args.compare_with_agent),
            )

            if bool(args.compare_with_agent):
                early_rows, early_summary = ar.summarize_early_alignment_error_rows(
                    step_rows,
                    topology_name=topology_name,
                    adaptive_policy_name=str(saved_policy_name),
                    max_step=max(0, int(args.early_alignment_max_step)),
                )
                ar.write_csv(topology_output_dir / "saved_policy_early_alignment.csv", early_rows)
                ar.plot_early_alignment_error(
                    early_rows,
                    topology_output_dir / "saved_policy_early_alignment.png",
                    adaptive_policy_name=str(saved_policy_name),
                )
                early_alignment_by_policy: dict[str, Any] = {
                    str(saved_policy_name): dict(early_summary)
                }
            else:
                early_alignment_by_policy = {}

            topology_rows = ar.summarize_topology_case(
                topology_name,
                episode_rows,
                step_rows=step_rows,
                adaptive_early_alignment_by_policy=early_alignment_by_policy,
            )
            for row in topology_rows:
                row["env_r"] = float(topology_spec["dynamics"]["r"])
            topology_summary_rows.extend(topology_rows)
            ar.write_csv(topology_output_dir / "topology_summary.csv", topology_rows)

            summary_payload = {
                "run_dir": str(run_dir),
                "saved_profile_summary_json": str(saved_profile_summary_path),
                "topology": str(topology_name),
                "graph_seed": int(topology_spec["seed"]),
                "rollout_seed": int(seed_base),
                "episode_count": int(episodes),
                "max_steps": None if max_steps is None else int(max_steps),
                "episode_length": int(topology_spec["dynamics"]["episode_length"]),
                "env_r": float(topology_spec["dynamics"]["r"]),
                "env_p_c": float(topology_spec["dynamics"]["p_c"]),
                "env_beta": float(topology_spec["dynamics"]["beta"]),
                "env_strategy_update_rule": str(topology_spec["dynamics"]["strategy_update_rule"]),
                "env_p_max": float(topology_spec["dynamics"]["p_max"]),
                "env_initial_resource": float(topology_spec["dynamics"]["initial_resource"]),
                "env_initial_cooperation_prob": float(topology_spec["dynamics"]["initial_cooperation_prob"]),
                "compare_with_agent": bool(args.compare_with_agent),
                "saved_policy_name": str(saved_policy_name),
                "saved_policy_label": str(saved_policy_label),
                "saved_policy_profile_summary": dict(saved_policy.profile_summary()),
                "saved_profile_payload": dict(saved_profile_payload),
                "plot_final_fc_min": float(args.plot_final_fc_min),
                "plot_filtered_episode_counts": {
                    str(policy_name): int(len(plot_episode_allowlist.get(str(policy_name), set())))
                    for policy_name, _ in policy_specs
                },
                "early_alignment_by_policy": dict(early_alignment_by_policy),
            }
            ar._write_json(topology_output_dir / "saved_policy_summary.json", summary_payload)

    ar.write_csv(output_dir / "topology_summary.csv", topology_summary_rows)
    ar._write_json(output_dir / "topology_summary.json", topology_summary_rows)
    if len(env_r_values) > 1:
        ar.write_csv(output_dir / "env_r_sweep_summary.csv", topology_summary_rows)
        ar._write_json(output_dir / "env_r_sweep_summary.json", topology_summary_rows)

    print(f"Saved binned Unit-Investment + Equal rollout complete. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
