from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    "saved_summary_json": (
        "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
        "0421_rule_allocation_over_time_ba_degree_conditioned_discrete/analysis_summary.json"
    ),
    "output_dir": None,
    "output_subdir_name": "0422_saved_conditionedtransfer_degree_discrete_beta5",
    "topologies": "regular,er,ws,ba",
    "episodes": 10,
    "max_steps": None,
    "episode_length_override": None,
    "env_r_override": None,
    "env_r_list": 0.5,#[0.25, 0.50, 0.75, 1.00],
    "env_p_c_override": 50,
    "env_beta_override": [5],
    "env_strategy_update_rule_override": None,
    "env_p_max_override": None,
    "env_initial_resource_override": None,
    "env_initial_cooperation_prob_override": None,
    "ego_resource_x_mode": "absolute",#"normalized",
    "ego_resource_normalizer": 897.73,    #当前环境的理论最大值，或者说归一化分母
    "saved_ego_resource_normalizer": 443.18,    #原策略对应的理论最大值
    "seed": None,
    "compare_with_agent": True,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


def _parse_env_r_list(raw_value: str | None) -> list[float] | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            text = text[1:-1].strip()
        else:
            if isinstance(parsed, list):
                values = [float(item) for item in parsed]
                return values or None
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


def _build_saved_policy_key_and_label(
    conditioning_mode: str,
    profile_family: str,
) -> tuple[str, str]:
    mode = str(conditioning_mode)
    family = str(profile_family)
    if mode == "sender_degree_bin":
        prefix = "saved_degree_conditioned"
        label_prefix = "Saved Degree-Conditioned"
    elif mode == "sender_resource_norm_bin":
        prefix = "saved_resource_conditioned"
        label_prefix = "Saved Resource-Conditioned"
    else:
        prefix = "saved_conditioned"
        label_prefix = "Saved Conditioned"

    if family == "unitself":
        return f"{prefix}_unitself_profile", f"{label_prefix} UnitInv + Self"
    return f"{prefix}_spline_profile", f"{label_prefix} Spline"


class _TransferredEgoResourceSubpolicy:
    """Replay a saved ego-resource rule under a new theoretical resource scale."""

    def __init__(
        self,
        *,
        base_policy: Any,
        replay_x_mode: str,
        replay_normalizer: float | None,
        saved_reference_normalizer: float | None,
    ) -> None:
        self.base_policy = base_policy
        self.replay_x_mode = str(replay_x_mode)
        self.replay_normalizer = None if replay_normalizer is None else float(replay_normalizer)
        self.saved_reference_normalizer = (
            None if saved_reference_normalizer is None else float(saved_reference_normalizer)
        )
        if self.replay_x_mode not in {"absolute", "normalized"}:
            raise ValueError(f"Unsupported replay ego resource x mode: {replay_x_mode}")

    def _map_ego_mean_resource(self, ego_mean_resource: float) -> float:
        raw_value = float(max(0.0, float(ego_mean_resource)))
        if self.replay_x_mode == "absolute":
            return raw_value

        current_normalizer = self.replay_normalizer
        if current_normalizer is None or current_normalizer <= 0.0:
            raise ValueError(
                "When --ego-resource-x-mode=normalized in run_saved_resource_conditioned_spline_policy.py, "
                "--ego-resource-normalizer must be positive."
            )
        normalized_fraction = raw_value / float(current_normalizer)

        saved_x_mode = str(getattr(self.base_policy, "ego_resource_x_mode", "absolute"))
        if saved_x_mode == "normalized":
            saved_normalizer = getattr(self.base_policy, "ego_resource_normalizer", None)
            if saved_normalizer is None or float(saved_normalizer) <= 0.0:
                raise ValueError(
                    "Saved profile uses normalized ego-resource x mode but has no positive saved normalizer."
                )
            return float(normalized_fraction * float(saved_normalizer))

        reference_normalizer = self.saved_reference_normalizer
        if reference_normalizer is None or float(reference_normalizer) <= 0.0:
            raise ValueError(
                "Normalized transfer for a saved absolute-x profile requires the original theoretical maximum. "
                "Pass --saved-ego-resource-normalizer."
            )
        return float(normalized_fraction * float(reference_normalizer))

    def weights_for_ego_mean_resource(self, ego_mean_resource: float) -> tuple[float, float, float]:
        mapped_resource = self._map_ego_mean_resource(ego_mean_resource)
        return self.base_policy.weights_for_ego_mean_resource(float(mapped_resource))

    def profile_summary(self) -> dict[str, Any]:
        summary = dict(self.base_policy.profile_summary())
        summary["replay_ego_resource_x_mode"] = str(self.replay_x_mode)
        summary["replay_ego_resource_normalizer"] = (
            None if self.replay_normalizer is None else float(self.replay_normalizer)
        )
        summary["saved_reference_normalizer_for_transfer"] = (
            None if self.saved_reference_normalizer is None else float(self.saved_reference_normalizer)
        )
        return summary


def _load_saved_conditioned_policy(
    summary_path: Path,
    *,
    replay_ego_resource_x_mode: str,
    replay_ego_resource_normalizer: float | None,
    saved_ego_resource_normalizer: float | None,
) -> tuple[ar.ConditionedSplineMixtureAllocationPolicy, dict[str, Any], dict[str, Any], str, str]:
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Saved summary at {summary_path} is not a JSON object.")

    profile_summary = payload.get("adaptive_unitinv_equal_profile")
    if not isinstance(profile_summary, Mapping):
        raise KeyError(f"{summary_path} does not contain adaptive_unitinv_equal_profile.")
    conditioning_mode = str(profile_summary.get("conditioning_mode", ""))
    if conditioning_mode not in {"sender_resource_norm_bin", "sender_degree_bin"}:
        raise ValueError(
            f"Saved summary conditioning_mode is {profile_summary.get('conditioning_mode')}, "
            "expected one of sender_resource_norm_bin or sender_degree_bin."
        )

    condition_spec = dict(profile_summary.get("condition_spec", {}))
    if conditioning_mode == "sender_degree_bin":
        # Saved summaries omit the in-memory lookup, but retain the degree
        # intervals. Restore the lookup using the same interval selection as
        # 0429_rule_allocation_over_time_ba_degree_conditioned_discrete/compute.py.
        degree_lookup: dict[int, dict[str, Any]] = {}
        for raw_bin in condition_spec.get("bins", []):
            degree_bin = dict(raw_bin)
            left, right = int(degree_bin["degree_left"]), int(degree_bin["degree_right"])
            if left > right:
                raise ValueError(f"Invalid saved degree interval: {left}..{right}.")
            for degree in range(left, right + 1):
                if degree in degree_lookup:
                    raise ValueError(f"Overlapping saved degree intervals at degree {degree}.")
                degree_lookup[degree] = degree_bin
        if not degree_lookup:
            raise ValueError("Saved degree-conditioned policy contains no degree intervals.")
        condition_spec["value_to_bin"] = degree_lookup
    bin_profiles_raw = profile_summary.get("bin_profiles", {})
    if not isinstance(bin_profiles_raw, Mapping) or not bin_profiles_raw:
        raise ValueError(f"No bin_profiles found in {summary_path}.")

    policies_by_bin: dict[str, Any] = {}
    for bin_label, bin_profile in dict(bin_profiles_raw).items():
        if not isinstance(bin_profile, Mapping):
            continue
        mode = str(bin_profile.get("mode", ""))
        if mode in {"smooth_3way", "spline_3way"}:
            coeffs_unit = bin_profile.get("coeffs_unit_investment")
            coeffs_self = bin_profile.get("coeffs_self")
            if coeffs_unit is None or coeffs_self is None:
                raise ValueError(f"Saved bin profile {bin_label} is missing coefficients.")
            subpolicy = ar.SmoothLogEgoResourceMixtureAllocationPolicy(
                mode=mode,
                coeffs_unit_investment=coeffs_unit,
                coeffs_self=coeffs_self,
                profile_source=f"saved_summary:{summary_path}::{bin_label}",
                fit_summary={key: value for key, value in dict(bin_profile).items() if key not in {"coeffs_unit_investment", "coeffs_self", "coeffs_equal"}},
                spline_knots_resource=bin_profile.get("spline_knots_resource"),
                tail_cap_mode=str(bin_profile.get("tail_cap_mode", "none")),
                tail_cap_resource=bin_profile.get("tail_cap_resource"),
                tail_cap_soft_scale=float(bin_profile.get("tail_cap_soft_scale", 0.25)),
                ego_resource_x_mode=str(bin_profile.get("ego_resource_x_mode", "absolute")),
                ego_resource_normalizer=bin_profile.get("ego_resource_normalizer"),
            )
        elif mode in {"binned_3way", "binned_unitself"}:
            bin_edges_resource = bin_profile.get("bin_edges_resource")
            bin_weight_rows = bin_profile.get("bin_weight_rows")
            if bin_edges_resource is None or bin_weight_rows is None:
                raise ValueError(f"Saved binned bin profile {bin_label} is missing edges or bin weights.")
            subpolicy = ar.BinnedEgoResourceMixtureAllocationPolicy(
                mode=mode,
                bin_edges_resource=bin_edges_resource,
                bin_weight_rows=bin_weight_rows,
                profile_source=f"saved_summary:{summary_path}::{bin_label}",
                fit_summary={
                    key: value
                    for key, value in dict(bin_profile).items()
                    if key not in {"bin_edges_resource", "bin_weight_rows"}
                },
                tail_cap_mode=str(bin_profile.get("tail_cap_mode", "none")),
                tail_cap_resource=bin_profile.get("tail_cap_resource"),
                tail_cap_soft_scale=float(bin_profile.get("tail_cap_soft_scale", 0.25)),
                ego_resource_x_mode=str(bin_profile.get("ego_resource_x_mode", "absolute")),
                ego_resource_normalizer=bin_profile.get("ego_resource_normalizer"),
            )
        else:
            raise ValueError(f"Unsupported saved subpolicy mode: {mode}")
        policies_by_bin[str(bin_label)] = _TransferredEgoResourceSubpolicy(
            base_policy=subpolicy,
            replay_x_mode=str(replay_ego_resource_x_mode),
            replay_normalizer=replay_ego_resource_normalizer,
            saved_reference_normalizer=saved_ego_resource_normalizer,
        )

    conditioned_policy = ar.ConditionedSplineMixtureAllocationPolicy(
        conditioning_mode=conditioning_mode,
        profile_family=str(profile_summary.get("profile_family", "three_way")),
        condition_spec=condition_spec,
        policies_by_bin=policies_by_bin,
        profile_source=f"saved_summary:{summary_path}",
        fit_summary=dict(profile_summary),
    )
    policy_key, policy_label = _build_saved_policy_key_and_label(
        conditioning_mode=conditioning_mode,
        profile_family=str(profile_summary.get("profile_family", "three_way")),
    )
    return conditioned_policy, dict(payload), dict(profile_summary), str(policy_key), str(policy_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a saved conditioned spline/binned surrogate rule from analysis_summary.json on selected topologies."
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
        "--saved-summary-json",
        type=Path,
        default=Path(str(SCRIPT_DEFAULTS["saved_summary_json"])),
        help="Path to the original conditioned analysis_summary.json that stores the exact spline/binned coefficients.",
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
        default=SCRIPT_DEFAULTS["topologies"],
        help="Comma-separated topology list. Defaults to the saved summary topology when omitted.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=SCRIPT_DEFAULTS["episodes"],
        help="Number of deterministic episodes to roll out. Defaults to the saved summary episode_count when omitted.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=SCRIPT_DEFAULTS["max_steps"],
        help="Optional cap on steps per episode. Defaults to the saved summary max_steps when omitted.",
    )
    parser.add_argument(
        "--episode-length-override",
        type=int,
        default=SCRIPT_DEFAULTS["episode_length_override"],
        help="Override the environment episode_length used during analysis. Defaults to the saved summary episode_length when omitted.",
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
        help="Comma-separated sweep of dynamics.r values, e.g. 0.25,0.50,0.75,1.00. When set, runs one subdirectory per r.",
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
        help="Override dynamics.beta during analysis rollouts. For Fermi this is the update intensity k/beta.",
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
        "--ego-resource-x-mode",
        type=str,
        choices=("absolute", "normalized"),
        default=str(SCRIPT_DEFAULTS["ego_resource_x_mode"]),
        help=(
            "How to interpret ego_mean_resource during saved-policy replay. "
            "absolute keeps the old behavior. normalized first divides the current ego_mean_resource "
            "by --ego-resource-normalizer, then maps that fraction back onto the saved policy scale."
        ),
    )
    parser.add_argument(
        "--ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["ego_resource_normalizer"],
        help=(
            "Current/the runtime theoretical maximum used when --ego-resource-x-mode=normalized. "
            "This is the denominator for the current environment."
        ),
    )
    parser.add_argument(
        "--saved-ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["saved_ego_resource_normalizer"],
        help=(
            "Original/the saved theoretical maximum. Needed when the saved profile was fit on absolute "
            "ego_mean_resource but you want normalized transfer replay."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SCRIPT_DEFAULTS["seed"],
        help="Shared graph/reset seed. Defaults to the saved summary rollout_seed when omitted.",
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
        help="Only run the saved spline rule.",
    )
    parser.set_defaults(compare_with_agent=bool(SCRIPT_DEFAULTS["compare_with_agent"]))
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

    saved_summary_path = args.saved_summary_json.expanduser().resolve()
    saved_policy, saved_summary_payload, saved_profile_summary, saved_policy_name, saved_policy_label = (
        _load_saved_conditioned_policy(
            saved_summary_path,
            replay_ego_resource_x_mode=str(args.ego_resource_x_mode),
            replay_ego_resource_normalizer=(
                None if args.ego_resource_normalizer is None else float(args.ego_resource_normalizer)
            ),
            saved_ego_resource_normalizer=(
                None if args.saved_ego_resource_normalizer is None else float(args.saved_ego_resource_normalizer)
            ),
        )
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            (run_dir / args.output_subdir_name).resolve()
            if args.output_subdir_name is not None
            else (run_dir / "saved_conditioned_spline_transfer").resolve()
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_topology = str(saved_summary_payload.get("topology", "ba"))
    topologies_raw = saved_topology if args.topologies in {None, ""} else str(args.topologies)
    topologies = parse_topology_list(topologies_raw)
    env_r_values = _parse_env_r_list(args.env_r_list)
    if env_r_values is None:
        env_r_values = [None if args.env_r_override is None else float(args.env_r_override)]

    episodes = int(saved_summary_payload.get("episode_count", 5)) if args.episodes is None else int(args.episodes)
    max_steps = saved_summary_payload.get("max_steps", None) if args.max_steps is None else int(args.max_steps)
    episode_length = (
        int(saved_summary_payload.get("episode_length", 200))
        if args.episode_length_override is None
        else int(args.episode_length_override)
    )
    seed_base = (
        int(saved_summary_payload.get("rollout_seed", experiment_spec.get("seed", 42)))
        if args.seed is None
        else int(args.seed)
    )

    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)
    device = torch.device(str(args.device))
    actor = None
    if bool(args.compare_with_agent):
        actor, _ = load_actor_from_run_dir(run_dir, args.checkpoint_name, device)

    ar.POLICY_STYLES[str(saved_policy_name)] = {
        "color": "#0f766e",
        "label": str(saved_policy_label),
    }

    topology_summary_rows: list[dict[str, Any]] = []
    for env_r_value in env_r_values:
        r_output_dir = output_dir if len(env_r_values) == 1 else (output_dir / _format_r_tag(float(env_r_value if env_r_value is not None else saved_summary_payload.get("env_r", 0.0))))
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
            if episode_length is not None:
                topology_spec["dynamics"] = dict(topology_spec["dynamics"])
                topology_spec["dynamics"]["episode_length"] = int(episode_length)

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
                    episodes=max(1, int(episodes)),
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
                episodes=max(1, int(episodes)),
                seed=int(seed_base),
                max_steps=max_steps,
            )
            snapshots_by_policy[str(saved_policy_name)] = list(saved_snapshots)
            episode_rows.extend(
                ar.episode_summaries_to_rows(str(saved_policy_name), saved_summaries)
            )
            policy_specs.append((str(saved_policy_name), saved_policy))
            ar.POLICY_SPECS = tuple(policy_specs)

            step_rows = ar.summarize_policy_rollout_step_metrics(snapshots_by_policy)
            ar.write_csv(topology_output_dir / "policy_rollout_step_metrics.csv", step_rows)
            ar.write_csv(topology_output_dir / "episode_summary.csv", episode_rows)
            ar.plot_policy_rollout_outcomes_over_time(step_rows, topology_output_dir / "policy_rollout_outcomes_over_time.png")
            ar.plot_policy_rollout_behavior_over_time(step_rows, topology_output_dir / "policy_rollout_behavior_over_time.png")

            if bool(args.compare_with_agent):
                early_rows, early_summary = ar.summarize_early_alignment_error_rows(
                    step_rows,
                    topology_name=topology_name,
                    adaptive_policy_name=str(saved_policy_name),
                    max_step=15,
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
                "saved_summary_json": str(saved_summary_path),
                "topology": str(topology_name),
                "saved_topology": str(saved_topology),
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
                "replay_ego_resource_x_mode": str(args.ego_resource_x_mode),
                "replay_ego_resource_normalizer": (
                    None if args.ego_resource_normalizer is None else float(args.ego_resource_normalizer)
                ),
                "saved_ego_resource_normalizer": (
                    None
                    if args.saved_ego_resource_normalizer is None
                    else float(args.saved_ego_resource_normalizer)
                ),
                "compare_with_agent": bool(args.compare_with_agent),
                "saved_policy_name": str(saved_policy_name),
                "saved_policy_label": str(saved_policy_label),
                "saved_policy_profile_summary": dict(saved_policy.profile_summary()),
                "saved_profile_source_summary": dict(saved_profile_summary),
                "early_alignment_by_policy": dict(early_alignment_by_policy),
            }
            ar._write_json(topology_output_dir / "saved_policy_summary.json", summary_payload)

    ar.write_csv(output_dir / "topology_summary.csv", topology_summary_rows)
    ar._write_json(output_dir / "topology_summary.json", topology_summary_rows)
    if len(env_r_values) > 1:
        ar.write_csv(output_dir / "env_r_sweep_summary.csv", topology_summary_rows)
        ar._write_json(output_dir / "env_r_sweep_summary.json", topology_summary_rows)

    print(f"Saved conditioned spline transfer rollout complete. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
