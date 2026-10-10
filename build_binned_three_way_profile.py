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
    load_actor_from_run_dir,
    resolve_reference_mean_degree,
    run_deterministic_rollouts,
)
from Project1.env import SPGGEnv
from training_graph_bank_utils import (
    GRAPH_SOURCE_MODE_CHOICES,
    build_topology_spec_for_analysis,
    normalize_graph_source_mode,
)


SCRIPT_DEFAULTS = {
    "run_dir": (
        r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic"
        r"\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
    ),
    "checkpoint_name": "best_eval.pt",
    "output_dir": (
        r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic"
        r"\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
        r"\0429_binned_three_way_profile\regular"
    ),
    "output_stem": "binned_unitinv_equal_profile",
    "topology": "regular",
    "episodes": 10,
    "max_steps": 200,
    "episode_length_override": 200,
    "env_r_override": None,
    "env_p_c_override": None,
    "env_beta_override": None,
    "env_strategy_update_rule_override": None,
    "env_p_max_override": None,
    "env_initial_resource_override": None,
    "env_initial_cooperation_prob_override": None,
    "graph_source_mode": "training_fixed_graph_bank",
    "graph_bank_index": 0,
    "seed": None,
    "bin_count": 20,
    "ego_resource_x_mode": "absolute",
    "ego_resource_normalizer": None,
    "tail_cap_mode": "hard",
    "tail_cap_resource": 443.18,
    "tail_cap_soft_scale": 0.25,
    "title": "",
    "x_label": r"$\bar{R}_{ego}(t)$",
    "y_label": "Mixture weight",
    "fig_width": 4.0,
    "fig_height": 4.0,
    "serif_font": "Times New Roman",
    "dpi": 600,
    "title_fontsize": 14,
    "axis_label_fontsize": 13,
    "tick_label_fontsize": 12,
    "legend_fontsize": 11,
    "proportional_color": "#1E468A",
    "average_color": "#E66051",
    "self_color": "#9467BD",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a discrete three-way ego-resource profile directly from RL-agent rollouts. "
            "The saved profile contains Unit-Investment, Equal, and Self weights."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(SCRIPT_DEFAULTS["run_dir"]),
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
        type=str,
        default=str(SCRIPT_DEFAULTS["output_dir"]),
        help="Directory for generated CSV/PNG/JSON artifacts.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=str(SCRIPT_DEFAULTS["output_stem"]),
        help="Filename stem for generated artifacts.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=str(SCRIPT_DEFAULTS["topology"]),
        help="Topology to analyze, e.g. regular, ba, er, ws.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(SCRIPT_DEFAULTS["episodes"]),
        help="Number of deterministic RL-agent episodes used to build reference rows.",
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
        "--graph-source-mode",
        type=str,
        choices=GRAPH_SOURCE_MODE_CHOICES,
        default=str(SCRIPT_DEFAULTS["graph_source_mode"]),
        help="How to construct the analysis graph: replay the training fixed graph bank, or use spec_seed directly.",
    )
    parser.add_argument(
        "--graph-bank-index",
        type=int,
        default=int(SCRIPT_DEFAULTS["graph_bank_index"]),
        help="Which graph inside the training fixed graph bank to replay when --graph-source-mode=training_fixed_graph_bank.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SCRIPT_DEFAULTS["seed"],
        help="Rollout reset seed. When --graph-source-mode=spec_seed, it also controls the graph seed.",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["bin_count"]),
        help="Number of discrete ego-resource bins.",
    )
    parser.add_argument(
        "--ego-resource-x-mode",
        type=str,
        choices=("absolute", "normalized"),
        default=str(SCRIPT_DEFAULTS["ego_resource_x_mode"]),
        help="Whether the saved bins are stored in absolute or normalized ego-resource coordinates.",
    )
    parser.add_argument(
        "--ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["ego_resource_normalizer"],
        help="Required when --ego-resource-x-mode=normalized.",
    )
    parser.add_argument(
        "--tail-cap-mode",
        type=str,
        choices=("none", "hard", "soft"),
        default=str(SCRIPT_DEFAULTS["tail_cap_mode"]),
        help="Tail-cap mode applied before converting resource to profile-x space.",
    )
    parser.add_argument(
        "--tail-cap-resource",
        type=float,
        default=SCRIPT_DEFAULTS["tail_cap_resource"],
        help="Cap threshold used by hard/soft tail-cap modes.",
    )
    parser.add_argument(
        "--tail-cap-soft-scale",
        type=float,
        default=float(SCRIPT_DEFAULTS["tail_cap_soft_scale"]),
        help="Soft tail-cap slope for values above the cap.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=str(SCRIPT_DEFAULTS["title"]),
        help="Plot title. Pass an empty string to omit it.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=int(SCRIPT_DEFAULTS["dpi"]),
        help="PNG export DPI.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(SCRIPT_DEFAULTS["device"]),
        help="Torch device, e.g. cpu or cuda:0.",
    )
    return parser.parse_args()


def _coerce_path(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
    windows_match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if windows_match and os.name != "nt":
        drive = windows_match.group(1).lower()
        suffix = windows_match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{suffix}").expanduser().resolve()
    return Path(text).expanduser().resolve()


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


def _configure_publication_style(serif_font: str) -> None:
    if ar.matplotlib is None:
        return
    ar.matplotlib.rcParams["font.family"] = str(serif_font)
    ar.matplotlib.rcParams["font.serif"] = [
        str(serif_font),
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "DejaVu Serif",
    ]
    ar.matplotlib.rcParams["font.weight"] = "bold"
    ar.matplotlib.rcParams["mathtext.fontset"] = "custom"
    ar.matplotlib.rcParams["mathtext.rm"] = str(serif_font)
    ar.matplotlib.rcParams["mathtext.it"] = f"{serif_font}:italic"
    ar.matplotlib.rcParams["mathtext.bf"] = f"{serif_font}:bold"
    ar.matplotlib.rcParams["mathtext.cal"] = f"{serif_font}:italic"


def _build_reference_rows_from_rollout(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], dict[str, Any], int, list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = _coerce_path(args.run_dir)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found under {run_dir}")

    results_payload = _load_json(results_path)
    experiment_spec = results_payload["experiment"] if "experiment" in results_payload else results_payload
    if not isinstance(experiment_spec, dict):
        raise TypeError("Could not extract experiment spec from results.json.")

    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)
    rollout_seed_base = int(experiment_spec.get("seed", 42)) if args.seed is None else int(args.seed)
    graph_source_mode = normalize_graph_source_mode(str(args.graph_source_mode))
    topology_spec, graph_metadata = build_topology_spec_for_analysis(
        experiment_spec,
        str(args.topology),
        reference_mean_degree=reference_mean_degree,
        graph_source_mode=graph_source_mode,
        graph_bank_index=int(args.graph_bank_index),
        spec_seed=(None if graph_source_mode == "training_fixed_graph_bank" else int(rollout_seed_base)),
    )
    topology_spec = ar.apply_analysis_env_overrides(
        topology_spec,
        env_r_override=_coerce_optional_scalar(args.env_r_override, name="env_r_override"),
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

    if args.episode_length_override is not None:
        topology_spec["dynamics"] = dict(topology_spec["dynamics"])
        topology_spec["dynamics"]["episode_length"] = int(args.episode_length_override)

    graph = build_graph_from_spec(topology_spec)
    env_config = build_env_config_from_spec(topology_spec, graph)
    env = SPGGEnv(env_config, graph)

    actor, _ = load_actor_from_run_dir(run_dir, str(args.checkpoint_name), torch.device(str(args.device)))
    snapshots, episode_summaries = run_deterministic_rollouts(
        actor,
        env,
        episodes=max(1, int(args.episodes)),
        seed=int(rollout_seed_base),
        max_steps=None if args.max_steps is None else int(args.max_steps),
    )
    reference_rows = ar.collect_agent_reference_rows(snapshots)
    return (
        run_dir,
        topology_spec,
        graph_metadata,
        int(rollout_seed_base),
        reference_rows,
        ar.episode_summaries_to_rows("agent", episode_summaries),
    )


def _build_binned_rows_from_fit(
    policy: ar.BinnedEgoResourceMixtureAllocationPolicy,
    fit_summary: Mapping[str, Any],
    prepared_rows: list[dict[str, Any]],
    *,
    profile_source: str,
) -> list[dict[str, Any]]:
    grouped_rows: dict[int, list[dict[str, Any]]] = {}
    for row in prepared_rows:
        raw_resource = float(row["ego_mean_resource"])
        profile_x = float(policy.profile_x_value_for_ego_mean_resource(raw_resource))
        bin_index = int(policy._resolve_bin_index(profile_x))
        grouped_rows.setdefault(bin_index, []).append(dict(row))

    summary_rows = list(fit_summary.get("bin_rows_summary", []))
    if not summary_rows:
        raise ValueError("fit_summary does not contain bin_rows_summary for the discrete three-way profile.")

    output_rows: list[dict[str, Any]] = []
    for summary_row in summary_rows:
        bin_index = int(summary_row["bin_index"])
        members = grouped_rows.get(bin_index, [])
        if not members:
            continue

        raw_values = np.asarray([float(row["ego_mean_resource"]) for row in members], dtype=np.float64)
        effective_values = np.asarray(
            [float(policy.effective_ego_mean_resource(value)) for value in raw_values.tolist()],
            dtype=np.float64,
        )
        x_values = np.asarray(
            [float(policy.profile_x_value_for_ego_mean_resource(value)) for value in raw_values.tolist()],
            dtype=np.float64,
        )
        fit_l1_values = np.asarray([float(row["fit_l1"]) for row in members], dtype=np.float64)

        output_rows.append(
            {
                "row_kind": "reference_bin",
                "order_index": int(bin_index),
                "ego_mean_resource": float(np.mean(raw_values)),
                "ego_mean_resource_effective": float(np.mean(effective_values)),
                "ego_resource_x": float(np.mean(x_values)),
                "ego_resource_x_mode": str(policy.ego_resource_x_mode),
                "ego_resource_normalizer": (
                    None if policy.ego_resource_normalizer is None else float(policy.ego_resource_normalizer)
                ),
                "log_ego_mean_resource": float(np.log1p(max(0.0, float(np.mean(x_values))))),
                "weight_unit_investment": float(summary_row["weight_unit_investment"]),
                "weight_equal": float(summary_row["weight_equal"]),
                "weight_self": float(summary_row["weight_self"]),
                "bin_left": float(summary_row["bin_left"]),
                "bin_right": float(summary_row["bin_right"]),
                "sample_count": int(summary_row["sample_count"]),
                "reference_fit_l1_mean": float(np.nanmean(fit_l1_values)),
                "profile_source": str(profile_source),
            }
        )

    output_rows.sort(key=lambda row: float(row["bin_left"]))
    return output_rows


def _plot_discrete_profile(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    dpi: int,
    fig_width: float,
    fig_height: float,
    title_fontsize: float,
    axis_label_fontsize: float,
    tick_label_fontsize: float,
    legend_fontsize: float,
    proportional_color: str,
    average_color: str,
    self_color: str,
) -> bool:
    if ar.plt is None or not rows:
        return False

    resolved_x_label = str(x_label).strip() or ar._ego_resource_x_label(
        str(rows[0].get("ego_resource_x_mode", "absolute")),
        rows[0].get("ego_resource_normalizer"),
    )
    resolved_y_label = str(y_label).strip() or "Mixture weight"
    figure, axis = ar.plt.subplots(figsize=(float(fig_width), float(fig_height)), constrained_layout=True)

    bin_edges = np.asarray(
        [float(rows[0]["bin_left"])] + [float(row["bin_right"]) for row in rows],
        dtype=np.float64,
    )
    centers = np.asarray([float(row["ego_resource_x"]) for row in rows], dtype=np.float64)
    sample_counts = np.asarray([int(row["sample_count"]) for row in rows], dtype=np.int64)
    alpha_scale = np.clip(sample_counts / float(max(int(np.max(sample_counts)), 1)), 0.35, 1.0)

    component_specs = [
        ("weight_unit_investment", str(proportional_color), "Proportional"),
        ("weight_equal", str(average_color), "Equal"),
        ("weight_self", str(self_color), "Self"),
    ]

    for key, color, label in component_specs:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if np.all(~np.isfinite(values)):
            continue
        step_y = np.concatenate([values, values[-1:]], axis=0)
        axis.step(
            bin_edges,
            step_y,
            where="post",
            color=color,
            linewidth=2.2,
            label=label,
            zorder=2,
        )
        axis.scatter(
            centers,
            values,
            s=28,
            color=color,
            alpha=float(np.mean(alpha_scale)),
            zorder=3,
        )

    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel(resolved_x_label, fontsize=float(axis_label_fontsize), fontweight="bold")
    axis.set_ylabel(resolved_y_label, fontsize=float(axis_label_fontsize), fontweight="bold")
    if title.strip():
        axis.set_title(title, fontsize=float(title_fontsize))
    axis.tick_params(axis="both", labelsize=float(tick_label_fontsize))
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    for tick_label in list(axis.get_xticklabels()) + list(axis.get_yticklabels()):
        tick_label.set_fontweight("bold")
    legend = axis.legend(frameon=False, fontsize=float(legend_fontsize))
    if legend is not None:
        for legend_text in legend.get_texts():
            legend_text.set_fontweight("bold")
    figure.savefig(output_path, dpi=max(72, int(dpi)), bbox_inches="tight", facecolor="white")
    ar.plt.close(figure)
    return True


def _write_summary_json(
    output_path: Path,
    *,
    run_dir: Path,
    topology_spec: Mapping[str, Any],
    graph_metadata: Mapping[str, Any],
    rollout_seed: int,
    checkpoint_name: str,
    output_dir: Path,
    output_stem: str,
    episode_rows: list[dict[str, Any]],
    reference_row_count: int,
    policy_summary: Mapping[str, Any],
) -> None:
    payload = {
        "policy_name": "saved_binned_three_way_mix",
        "policy_label": "Saved Binned UnitInv + Equal + Self",
        "profile_type": "binned_three_way",
        "reference_source": "agent_rollout_3way",
        "run_dir": str(run_dir),
        "checkpoint_name": str(checkpoint_name),
        "topology": str(topology_spec["network"]["type"]),
        "graph_seed": int(topology_spec["seed"]),
        "rollout_seed": int(rollout_seed),
        "graph_source_mode": str(graph_metadata.get("graph_source_mode", "spec_seed")),
        "graph_bank_index": graph_metadata.get("graph_bank_index"),
        "fixed_graph_bank_seed": graph_metadata.get("fixed_graph_bank_seed"),
        "fixed_graph_bank_size_per_type": graph_metadata.get("fixed_graph_bank_size_per_type"),
        "fixed_graph_bank_sampling": graph_metadata.get("fixed_graph_bank_sampling"),
        "output_dir": str(output_dir),
        "output_stem": str(output_stem),
        "episode_count": int(len(episode_rows)),
        "reference_row_count": int(reference_row_count),
        "env_r": float(topology_spec["dynamics"]["r"]),
        "env_p_c": float(topology_spec["dynamics"]["p_c"]),
        "env_beta": float(topology_spec["dynamics"]["beta"]),
        "env_strategy_update_rule": str(topology_spec["dynamics"]["strategy_update_rule"]),
        "env_p_max": float(topology_spec["dynamics"]["p_max"]),
        "env_initial_resource": float(topology_spec["dynamics"]["initial_resource"]),
        "env_initial_cooperation_prob": float(topology_spec["dynamics"]["initial_cooperation_prob"]),
        "policy_summary": dict(policy_summary),
    }
    ar._write_json(output_path, payload)


def _build_fit_metrics_payload(
    *,
    fit_summary: Mapping[str, Any],
    reference_row_count: int,
    prepared_row_count: int,
) -> dict[str, Any]:
    return {
        "fit_status": str(fit_summary.get("fit_status", "")),
        "reference_row_count": int(reference_row_count),
        "fit_row_count": int(fit_summary.get("fit_row_count", prepared_row_count)),
        "prepared_row_count": int(prepared_row_count),
        "adaptive_ego_resource_profile_mode": str(
            fit_summary.get("adaptive_ego_resource_profile_mode", "discrete")
        ),
        "adaptive_ego_resource_bin_count": int(fit_summary.get("adaptive_ego_resource_bin_count", 0)),
        "ego_resource_x_mode": str(fit_summary.get("ego_resource_x_mode", "absolute")),
        "ego_resource_normalizer": fit_summary.get("ego_resource_normalizer"),
        "tail_cap_mode": str(fit_summary.get("tail_cap_mode", "none")),
        "tail_cap_resource": fit_summary.get("tail_cap_resource"),
        "tail_cap_soft_scale": float(fit_summary.get("tail_cap_soft_scale", 0.25)),
        "weight_simplex_l1_mean": float(fit_summary.get("weight_simplex_l1_mean", float("nan"))),
        "reference_row_fit_l1_mean": float(fit_summary.get("reference_row_fit_l1_mean", float("nan"))),
    }


def main() -> None:
    args = parse_args()
    _configure_publication_style(str(SCRIPT_DEFAULTS["serif_font"]))
    if str(args.ego_resource_x_mode) == "normalized":
        if args.ego_resource_normalizer is None or float(args.ego_resource_normalizer) <= 0.0:
            raise ValueError("--ego-resource-normalizer must be positive when --ego-resource-x-mode=normalized.")

    output_dir = _coerce_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = str(args.output_stem).strip()
    if not output_stem:
        raise ValueError("--output-stem cannot be empty.")

    run_dir, topology_spec, graph_metadata, rollout_seed, reference_rows, episode_rows = _build_reference_rows_from_rollout(
        args
    )
    if not reference_rows:
        raise ValueError("No agent reference rows were collected from the rollout.")

    policy, prepared_rows, fit_summary = ar.fit_binned_three_way_policy(
        reference_rows,
        bin_count=max(2, int(args.bin_count)),
        tail_cap_mode=str(args.tail_cap_mode),
        tail_cap_resource=None if args.tail_cap_resource is None else float(args.tail_cap_resource),
        tail_cap_soft_scale=float(args.tail_cap_soft_scale),
        ego_resource_x_mode=str(args.ego_resource_x_mode),
        ego_resource_normalizer=(
            None if args.ego_resource_normalizer is None else float(args.ego_resource_normalizer)
        ),
    )
    if not isinstance(policy, ar.BinnedEgoResourceMixtureAllocationPolicy):
        raise ValueError(
            "Discrete three-way fitting fell back to a continuous policy. "
            "Increase --episodes / --max-steps or reduce --bin-count."
        )

    binned_rows = _build_binned_rows_from_fit(
        policy,
        fit_summary,
        prepared_rows,
        profile_source=f"agent_rollout_reference_rows_binned::{run_dir}::{args.topology}",
    )
    if not binned_rows:
        raise ValueError("No non-empty discrete bins were produced for the three-way profile.")

    profile_csv_path = output_dir / f"{output_stem}.csv"
    profile_png_path = output_dir / f"{output_stem}.png"
    profile_json_path = output_dir / f"{output_stem}_summary.json"
    fit_metrics_csv_path = output_dir / f"{output_stem}_fit_metrics.csv"
    fit_metrics_json_path = output_dir / f"{output_stem}_fit_metrics.json"
    reference_csv_path = output_dir / f"{output_stem}_reference_rows.csv"
    episode_summary_csv_path = output_dir / "agent_episode_summary.csv"
    fit_metrics_payload = _build_fit_metrics_payload(
        fit_summary=fit_summary,
        reference_row_count=len(reference_rows),
        prepared_row_count=len(prepared_rows),
    )

    ar.write_csv(profile_csv_path, binned_rows)
    ar.write_csv(reference_csv_path, reference_rows)
    ar.write_csv(episode_summary_csv_path, episode_rows)
    ar.write_csv(fit_metrics_csv_path, [fit_metrics_payload])
    ar._write_json(fit_metrics_json_path, fit_metrics_payload)
    png_written = _plot_discrete_profile(
        binned_rows,
        profile_png_path,
        title=str(args.title),
        x_label=str(SCRIPT_DEFAULTS["x_label"]),
        y_label=str(SCRIPT_DEFAULTS["y_label"]),
        dpi=int(args.dpi),
        fig_width=float(SCRIPT_DEFAULTS["fig_width"]),
        fig_height=float(SCRIPT_DEFAULTS["fig_height"]),
        title_fontsize=float(SCRIPT_DEFAULTS["title_fontsize"]),
        axis_label_fontsize=float(SCRIPT_DEFAULTS["axis_label_fontsize"]),
        tick_label_fontsize=float(SCRIPT_DEFAULTS["tick_label_fontsize"]),
        legend_fontsize=float(SCRIPT_DEFAULTS["legend_fontsize"]),
        proportional_color=str(SCRIPT_DEFAULTS["proportional_color"]),
        average_color=str(SCRIPT_DEFAULTS["average_color"]),
        self_color=str(SCRIPT_DEFAULTS["self_color"]),
    )
    _write_summary_json(
        profile_json_path,
        run_dir=run_dir,
        topology_spec=topology_spec,
        graph_metadata=graph_metadata,
        rollout_seed=int(rollout_seed),
        checkpoint_name=str(args.checkpoint_name),
        output_dir=output_dir,
        output_stem=output_stem,
        episode_rows=episode_rows,
        reference_row_count=len(reference_rows),
        policy_summary=policy.profile_summary(),
    )

    print(f"Resolved run dir: {run_dir}")
    print(f"Resolved topology: {topology_spec['network']['type']}")
    print(
        "Graph source: "
        f"mode={graph_metadata.get('graph_source_mode')}, "
        f"graph_seed={graph_metadata.get('graph_seed')}, "
        f"graph_bank_index={graph_metadata.get('graph_bank_index')}, "
        f"fixed_graph_bank_seed={graph_metadata.get('fixed_graph_bank_seed')}"
    )
    print(f"Collected agent reference rows: {len(reference_rows)}")
    print(f"Saved raw reference rows CSV to: {reference_csv_path}")
    print(f"Saved discrete profile CSV to: {profile_csv_path}")
    if png_written and profile_png_path.exists():
        print(f"Saved discrete profile PNG to: {profile_png_path}")
    else:
        print("Skipped discrete profile PNG because matplotlib is not available in the current interpreter.")
    print(f"Saved discrete profile summary JSON to: {profile_json_path}")
    print(f"Saved fit metrics CSV to: {fit_metrics_csv_path}")
    print(f"Saved fit metrics JSON to: {fit_metrics_json_path}")
    print(f"Saved episode summary CSV to: {episode_summary_csv_path}")
    print(f"Tail cap: mode={args.tail_cap_mode}, resource={args.tail_cap_resource}")
    print(f"Discrete bin count: {policy.bin_weight_rows.shape[0]}")
    print(
        "Fit metrics: "
        f"fit_status={fit_metrics_payload['fit_status']}, "
        f"weight_simplex_l1_mean={fit_metrics_payload['weight_simplex_l1_mean']:.6f}, "
        f"reference_row_fit_l1_mean={fit_metrics_payload['reference_row_fit_l1_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
