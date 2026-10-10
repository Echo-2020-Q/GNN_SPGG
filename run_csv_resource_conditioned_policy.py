from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    write_csv,
)
from Project1.env import SPGGEnv
from Project1.policies.rule_based import ProportionalContributionPolicy, UniformAllocationPolicy


SCRIPT_DEFAULTS = {
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    "checkpoint_name": "best_eval.pt",
    "profile_csv": (
        "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
        "0421_rule_allocation_over_time_ba_resource_conditioned/resource_conditioned_profile.csv"
    ),
    "output_dir": None,
    "output_subdir_name": "0421_resource_conditioned_transfer",
    "topologies": "regular,er,ws,ba",
    "episodes": 10,
    "max_steps": 200,
    "episode_length_override": 200,
    "env_r_override": None,
    "seed": None,
    "compare_with_agent": True,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


class UnitInvestmentProportionalAllocationPolicy(ProportionalContributionPolicy):
    """Allocate each sender's pool in proportion to receivers' current unit_investment values."""


class CSVResourceConditionedAllocationPolicy:
    """Replay a resource-conditioned ego-resource profile CSV as an executable allocation rule."""

    def __init__(self, profile_csv_path: Path):
        self.profile_csv_path = Path(profile_csv_path).expanduser().resolve()
        if not self.profile_csv_path.exists():
            raise FileNotFoundError(f"Profile CSV not found: {self.profile_csv_path}")

        self._unit_investment = UnitInvestmentProportionalAllocationPolicy()
        self._equal = UniformAllocationPolicy()
        self.bin_profiles = self._load_profile(self.profile_csv_path)
        if not self.bin_profiles:
            raise ValueError(f"No usable fitted_curve rows found in {self.profile_csv_path}")

    @staticmethod
    def _normalize_weight_triplet(unit_weight: float, equal_weight: float, self_weight: float) -> tuple[float, float, float]:
        weights = np.asarray([unit_weight, equal_weight, self_weight], dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        total = float(np.sum(weights))
        if total <= 1e-12:
            return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        normalized = weights / total
        return float(normalized[0]), float(normalized[1]), float(normalized[2])

    @staticmethod
    def _coalesce_curve_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        sorted_rows = sorted(rows, key=lambda row: float(row["ego_mean_resource"]))
        resource_values: list[float] = []
        unit_values: list[float] = []
        equal_values: list[float] = []
        self_values: list[float] = []
        for row in sorted_rows:
            ego_mean_resource = float(row["ego_mean_resource"])
            unit_weight = float(row["weight_unit_investment"])
            equal_weight = float(row["weight_equal"])
            self_weight = float(row["weight_self"])
            if resource_values and abs(ego_mean_resource - resource_values[-1]) <= 1e-12:
                unit_values[-1] = 0.5 * (unit_values[-1] + unit_weight)
                equal_values[-1] = 0.5 * (equal_values[-1] + equal_weight)
                self_values[-1] = 0.5 * (self_values[-1] + self_weight)
            else:
                resource_values.append(float(ego_mean_resource))
                unit_values.append(float(unit_weight))
                equal_values.append(float(equal_weight))
                self_values.append(float(self_weight))
        return {
            "ego_mean_resource": np.asarray(resource_values, dtype=np.float64),
            "weight_unit_investment": np.asarray(unit_values, dtype=np.float64),
            "weight_equal": np.asarray(equal_values, dtype=np.float64),
            "weight_self": np.asarray(self_values, dtype=np.float64),
        }

    @classmethod
    def _load_profile(cls, profile_csv_path: Path) -> list[dict[str, Any]]:
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        bin_metadata: dict[str, dict[str, Any]] = {}
        with profile_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row or str(row.get("row_kind", "")) != "fitted_curve":
                    continue
                bin_label = str(row.get("sender_resource_norm_bin", "NA"))
                grouped_rows.setdefault(bin_label, []).append(dict(row))
                if bin_label not in bin_metadata:
                    bin_metadata[bin_label] = {
                        "bin_index": int(float(row.get("sender_resource_norm_bin_index", -1))),
                        "bin_left": float(row.get("sender_resource_norm_bin_left", float("nan"))),
                        "bin_right": float(row.get("sender_resource_norm_bin_right", float("nan"))),
                    }

        profiles: list[dict[str, Any]] = []
        for bin_label, rows in grouped_rows.items():
            curve = cls._coalesce_curve_rows(rows)
            metadata = dict(bin_metadata.get(bin_label, {}))
            profiles.append(
                {
                    "bin_label": str(bin_label),
                    "bin_index": int(metadata.get("bin_index", -1)),
                    "bin_left": float(metadata.get("bin_left", float("nan"))),
                    "bin_right": float(metadata.get("bin_right", float("nan"))),
                    **curve,
                }
            )
        profiles.sort(key=lambda row: int(row["bin_index"]))
        return profiles

    def _resolve_bin(self, sender_resource_norm: float) -> dict[str, Any]:
        safe_value = float(sender_resource_norm)
        profiles = list(self.bin_profiles)
        if not profiles:
            raise RuntimeError("CSV resource-conditioned policy has no bin profiles.")
        if not np.isfinite(safe_value):
            return dict(profiles[-1])
        for index, bin_profile in enumerate(profiles):
            left = float(bin_profile["bin_left"])
            right = float(bin_profile["bin_right"])
            is_last = index == len(profiles) - 1
            if (left <= safe_value <= right) if is_last else (left <= safe_value < right):
                return dict(bin_profile)
        return dict(profiles[-1])

    def weights_for_sender(self, observation: Mapping[str, np.ndarray], sender: int) -> tuple[float, float, float, dict[str, Any]]:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)
        resource_norm = np.asarray(observation["resource_norm"], dtype=np.float64)
        valid_receivers = np.flatnonzero(local_mask[sender])
        if valid_receivers.size == 0:
            return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, {
                "sender_resource_norm": float(resource_norm[sender]),
                "ego_mean_resource": float("nan"),
                "sender_resource_norm_bin": "NA",
            }

        ego_mean_resource = float(np.mean(resources[valid_receivers]))
        sender_resource_norm = float(resource_norm[sender])
        bin_profile = self._resolve_bin(sender_resource_norm)
        x_values = np.asarray(bin_profile["ego_mean_resource"], dtype=np.float64)
        unit_values = np.asarray(bin_profile["weight_unit_investment"], dtype=np.float64)
        equal_values = np.asarray(bin_profile["weight_equal"], dtype=np.float64)
        self_values = np.asarray(bin_profile["weight_self"], dtype=np.float64)

        unit_weight = float(
            np.interp(
                float(ego_mean_resource),
                x_values,
                unit_values,
                left=float(unit_values[0]),
                right=float(unit_values[-1]),
            )
        )
        equal_weight = float(
            np.interp(
                float(ego_mean_resource),
                x_values,
                equal_values,
                left=float(equal_values[0]),
                right=float(equal_values[-1]),
            )
        )
        self_weight = float(
            np.interp(
                float(ego_mean_resource),
                x_values,
                self_values,
                left=float(self_values[0]),
                right=float(self_values[-1]),
            )
        )
        normalized = self._normalize_weight_triplet(unit_weight, equal_weight, self_weight)
        return (
            normalized[0],
            normalized[1],
            normalized[2],
            {
                "sender_resource_norm": float(sender_resource_norm),
                "ego_mean_resource": float(ego_mean_resource),
                "sender_resource_norm_bin": str(bin_profile["bin_label"]),
                "sender_resource_norm_bin_left": float(bin_profile["bin_left"]),
                "sender_resource_norm_bin_right": float(bin_profile["bin_right"]),
            },
        )

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        unit_investment_allocation = np.asarray(self._unit_investment.allocate(observation), dtype=np.float64)
        equal_allocation = np.asarray(self._equal.allocate(observation), dtype=np.float64)
        self_allocation = np.zeros(local_mask.shape, dtype=np.float64)

        for sender in range(local_mask.shape[0]):
            if local_mask[sender, sender]:
                self_allocation[sender, sender] = 1.0
            else:
                valid_receivers = np.flatnonzero(local_mask[sender])
                if valid_receivers.size > 0:
                    self_allocation[sender, valid_receivers] = 1.0 / float(valid_receivers.size)

        allocation = np.zeros(local_mask.shape, dtype=np.float64)
        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            weight_unit, weight_equal, weight_self, _ = self.weights_for_sender(observation, sender)
            allocation[sender] = (
                weight_unit * unit_investment_allocation[sender]
                + weight_equal * equal_allocation[sender]
                + weight_self * self_allocation[sender]
            )

        allocation = np.where(local_mask, allocation, 0.0)
        row_sums = allocation.sum(axis=1, keepdims=True)
        positive_rows = row_sums.squeeze(-1) > 1e-12
        if np.any(positive_rows):
            allocation[positive_rows] = allocation[positive_rows] / row_sums[positive_rows]
        if np.any(~positive_rows):
            allocation[~positive_rows] = equal_allocation[~positive_rows]
        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)

    def profile_summary(self) -> dict[str, Any]:
        return {
            "mode": "csv_resource_conditioned_lookup",
            "profile_csv": str(self.profile_csv_path),
            "bin_count": int(len(self.bin_profiles)),
            "bins": [
                {
                    "bin_label": str(bin_profile["bin_label"]),
                    "bin_index": int(bin_profile["bin_index"]),
                    "bin_left": float(bin_profile["bin_left"]),
                    "bin_right": float(bin_profile["bin_right"]),
                    "curve_point_count": int(len(np.asarray(bin_profile["ego_mean_resource"], dtype=np.float64))),
                }
                for bin_profile in self.bin_profiles
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CSV-defined resource-conditioned surrogate allocation policy on selected topologies."
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
        "--profile-csv",
        type=Path,
        default=Path(str(SCRIPT_DEFAULTS["profile_csv"])),
        help="Path to resource_conditioned_profile.csv to replay as a rollout policy.",
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
        help="Comma-separated topology list. Examples: regular,er,ws,ba",
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
        "--seed",
        type=int,
        default=SCRIPT_DEFAULTS["seed"],
        help="Shared graph/reset seed. When omitted, reuse the training seed recorded in results.json.",
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
        help="Only run the CSV-defined rule policy.",
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

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            (run_dir / args.output_subdir_name).resolve()
            if args.output_subdir_name is not None
            else (run_dir / "csv_resource_conditioned_transfer").resolve()
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_base = int(experiment_spec.get("seed", 42)) if args.seed is None else int(args.seed)
    topologies = parse_topology_list(args.topologies)
    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)
    device = torch.device(str(args.device))

    csv_policy = CSVResourceConditionedAllocationPolicy(args.profile_csv)
    actor = None
    if bool(args.compare_with_agent):
        actor, _ = load_actor_from_run_dir(run_dir, args.checkpoint_name, device)

    ar.POLICY_STYLES["csv_resource_conditioned_profile"] = {
        "color": "#0f766e",
        "label": "CSV Resource-Conditioned Rule",
    }

    topology_summary_rows: list[dict[str, Any]] = []
    for topology_name in topologies:
        topology_spec = build_spec_for_topology(
            experiment_spec,
            topology_name,
            reference_mean_degree=reference_mean_degree,
        )
        topology_spec = ar.apply_analysis_env_overrides(
            topology_spec,
            env_r_override=None if args.env_r_override is None else float(args.env_r_override),
        )
        topology_spec["seed"] = int(seed_base)
        topology_output_dir = output_dir if len(topologies) == 1 else (output_dir / topology_name)
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
                episodes=max(1, int(args.episodes)),
                seed=int(seed_base),
                max_steps=args.max_steps,
            )
            snapshots_by_policy["agent"] = list(agent_snapshots)
            episode_rows.extend(ar.episode_summaries_to_rows("agent", agent_summaries))
            policy_specs.append(("agent", None))

        csv_snapshots, csv_summaries = ar.run_rule_policy_rollouts(
            csv_policy,
            env_config=env_config,
            graph=graph,
            episodes=max(1, int(args.episodes)),
            seed=int(seed_base),
            max_steps=args.max_steps,
        )
        snapshots_by_policy["csv_resource_conditioned_profile"] = list(csv_snapshots)
        episode_rows.extend(ar.episode_summaries_to_rows("csv_resource_conditioned_profile", csv_summaries))
        policy_specs.append(("csv_resource_conditioned_profile", csv_policy))
        ar.POLICY_SPECS = tuple(policy_specs)

        step_rows = ar.summarize_policy_rollout_step_metrics(snapshots_by_policy)
        write_csv(topology_output_dir / "policy_rollout_step_metrics.csv", step_rows)
        write_csv(topology_output_dir / "episode_summary.csv", episode_rows)

        if bool(args.compare_with_agent):
            early_rows, early_summary = ar.summarize_early_alignment_error_rows(
                step_rows,
                topology_name=topology_name,
                adaptive_policy_name="csv_resource_conditioned_profile",
                max_step=15,
            )
            write_csv(topology_output_dir / "csv_policy_early_alignment.csv", early_rows)
            ar.plot_early_alignment_error(
                early_rows,
                topology_output_dir / "csv_policy_early_alignment.png",
                adaptive_policy_name="csv_resource_conditioned_profile",
            )
            early_alignment_by_policy: dict[str, Any] = {"csv_resource_conditioned_profile": dict(early_summary)}
        else:
            early_alignment_by_policy = {}

        ar.plot_policy_rollout_outcomes_over_time(step_rows, topology_output_dir / "policy_rollout_outcomes_over_time.png")
        ar.plot_policy_rollout_behavior_over_time(step_rows, topology_output_dir / "policy_rollout_behavior_over_time.png")

        topology_rows = ar.summarize_topology_case(
            topology_name,
            episode_rows,
            step_rows=step_rows,
            adaptive_early_alignment_by_policy=early_alignment_by_policy,
        )
        topology_summary_rows.extend(topology_rows)
        write_csv(topology_output_dir / "topology_summary.csv", topology_rows)

        summary_payload = {
            "run_dir": str(run_dir),
            "profile_csv": str(csv_policy.profile_csv_path),
            "topology": str(topology_name),
            "graph_seed": int(topology_spec["seed"]),
            "rollout_seed": int(seed_base),
            "episode_count": int(args.episodes),
            "max_steps": None if args.max_steps is None else int(args.max_steps),
            "episode_length": int(topology_spec["dynamics"]["episode_length"]),
            "compare_with_agent": bool(args.compare_with_agent),
            "csv_policy_profile_summary": dict(csv_policy.profile_summary()),
            "early_alignment_by_policy": dict(early_alignment_by_policy),
        }
        with (topology_output_dir / "csv_policy_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    write_csv(output_dir / "topology_summary.csv", topology_summary_rows)
    with (output_dir / "topology_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(topology_summary_rows, handle, ensure_ascii=False, indent=2)

    print(f"CSV resource-conditioned transfer rollout complete. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
