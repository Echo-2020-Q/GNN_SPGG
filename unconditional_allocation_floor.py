from __future__ import annotations

"""Robustness test for a small unconditional (equal-allocation) floor.

For every pool/sender ``j`` and eligible receiver ``i`` this script uses

    A_epsilon[j, i] = (1 - epsilon) * A_prop[j, i]
                        + epsilon / (degree[j] + 1),

where ``A_prop`` is proportional to the receiver's unit investment
``e_i / (degree[i] + 1)`` inside ``N_j union {j}``.  If every eligible unit
investment is zero, the repository's proportional baseline falls back to the
equal allocation; this script preserves that convention.

The experiment is simulation-only: it does not load or retrain an RL policy.
It writes paired episode results and steady-state (tail-window) summaries for
f_C, mean resource, and resource Gini.
"""

import argparse
import csv
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from Project1.env import (
    SPGGConfig,
    SPGGEnv,
    make_barabasi_albert_graph,
    make_erdos_renyi_graph,
    make_random_regular_graph,
    make_watts_strogatz_graph,
)
from Project1.policies.rule_based import ProportionalContributionPolicy


DEFAULT_RUN_DIR = Path(
    "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
)
DEFAULT_EPSILONS = (0.0, 0.01, 0.05, 0.10)
DEFAULT_TOPOLOGIES = ("regular", "scale_free", "erdos_renyi", "small_world")

TOPOLOGY_ALIASES = {
    "regular": "regular",
    "rr": "regular",
    "scale_free": "scale_free",
    "scale-free": "scale_free",
    "ba": "scale_free",
    "erdos_renyi": "erdos_renyi",
    "erdos-renyi": "erdos_renyi",
    "er": "erdos_renyi",
    "small_world": "small_world",
    "small-world": "small_world",
    "ws": "small_world",
}

TOPOLOGY_LABELS = {
    "regular": "Regular",
    "scale_free": "BA",
    "erdos_renyi": "ER",
    "small_world": "WS",
}

METRICS = (
    ("tail_cooperation", "tail_cooperation_mean", "tail_cooperation_sem", r"$f_C$"),
    ("tail_mean_resource", "tail_mean_resource_mean", "tail_mean_resource_sem", r"$\bar{R}$"),
    ("tail_gini", "tail_gini_mean", "tail_gini_sem", "Gini"),
)


class UnconditionalAllocationFloorPolicy:
    """Blend proportional allocation with an unconditional equal floor."""

    def __init__(self, epsilon: float) -> None:
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be a finite number in [0, 1].")
        self.epsilon = epsilon

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        unit_investment = np.asarray(observation["unit_investment"], dtype=np.float64)
        if local_mask.ndim != 2 or local_mask.shape[0] != local_mask.shape[1]:
            raise ValueError("observation['local_mask'] must be a square matrix.")
        if unit_investment.shape != (local_mask.shape[1],):
            raise ValueError("observation['unit_investment'] has an incompatible shape.")
        if np.any(~np.isfinite(unit_investment)) or np.any(unit_investment < 0.0):
            raise ValueError("unit investment must be finite and non-negative.")

        group_sizes = local_mask.sum(axis=1, keepdims=True).astype(np.float64)
        if np.any(group_sizes <= 0.0):
            raise ValueError("Every sender must be eligible to allocate to at least itself.")
        equal = local_mask.astype(np.float64) / group_sizes

        weighted = local_mask.astype(np.float64) * unit_investment[None, :]
        denominators = weighted.sum(axis=1, keepdims=True)
        proportional = equal.copy()
        np.divide(weighted, denominators, out=proportional, where=denominators > 0.0)

        allocation = ((1.0 - self.epsilon) * proportional) + (self.epsilon * equal)
        allocation[~local_mask] = 0.0
        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one epsilon is required.")
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid epsilon {value}; all values must lie in [0, 1].")
    return list(dict.fromkeys(values))


def _parse_topologies(raw: str) -> list[str]:
    selected: list[str] = []
    for part in str(raw).split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in TOPOLOGY_ALIASES:
            raise ValueError(f"Unsupported topology: {part}")
        canonical = TOPOLOGY_ALIASES[name]
        if canonical not in selected:
            selected.append(canonical)
    if not selected:
        raise ValueError("At least one topology is required.")
    return selected


def _extract_experiment_spec(results_path: Path) -> dict[str, Any]:
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    spec = payload.get("experiment", payload)
    if not isinstance(spec, dict):
        raise TypeError(f"Could not extract an experiment spec from {results_path}.")
    for key in ("seed", "network", "dynamics", "reward"):
        if key not in spec:
            raise KeyError(f"Experiment spec is missing {key!r}.")
    return deepcopy(spec)


def _reference_mean_degree(spec: Mapping[str, Any]) -> float:
    network = spec["network"]
    network_type = str(network["type"])
    if network_type == "regular":
        return float(network["regular_degree"])
    if network_type == "erdos_renyi":
        return float(network.get("er_target_mean_degree", 0.0))
    if network_type == "small_world":
        return float(network["ws_degree"])
    if network_type == "scale_free":
        return 2.0 * float(network["ba_attachments_per_new_node"])
    raise ValueError(f"Unsupported base topology: {network_type}")


def _even_degree(target: float, num_nodes: int) -> int:
    degree = max(2, min(num_nodes - 1, int(round(target))))
    if degree % 2:
        degree = degree + 1 if degree < num_nodes - 1 else degree - 1
    return max(2, degree)


def _spec_for_topology(
    base_spec: Mapping[str, Any],
    topology: str,
    *,
    reference_mean_degree: float,
    episode_length: int | None,
) -> dict[str, Any]:
    spec = deepcopy(dict(base_spec))
    network = spec["network"]
    num_nodes = int(network["num_nodes"])
    network["type"] = topology
    if topology == "regular":
        degree = int(network.get("regular_degree", round(reference_mean_degree)))
        degree = max(0, min(num_nodes - 1, degree))
        if (num_nodes * degree) % 2:
            degree -= 1
        network["regular_degree"] = degree
    elif topology == "erdos_renyi":
        target = float(network.get("er_target_mean_degree", reference_mean_degree))
        network["er_target_mean_degree"] = target
        network["er_edge_prob"] = target / float(max(num_nodes - 1, 1))
    elif topology == "small_world":
        network["ws_degree"] = _even_degree(
            float(network.get("ws_degree", reference_mean_degree)), num_nodes
        )
        network["ws_rewiring_prob"] = float(network.get("ws_rewiring_prob", 0.10))
    elif topology == "scale_free":
        attachments = int(
            network.get("ba_attachments_per_new_node", max(1, round(reference_mean_degree / 2.0)))
        )
        network["ba_attachments_per_new_node"] = max(1, min(num_nodes - 1, attachments))
    else:  # guarded by _parse_topologies
        raise ValueError(f"Unsupported topology: {topology}")
    if episode_length is not None:
        spec["dynamics"]["episode_length"] = int(episode_length)
    return spec


def _build_graph(spec: Mapping[str, Any]) -> dict[int, list[int]]:
    network = spec["network"]
    topology = str(network["type"])
    num_nodes = int(network["num_nodes"])
    seed = int(spec["seed"])
    if topology == "regular":
        return make_random_regular_graph(num_nodes, int(network["regular_degree"]), seed=seed)
    if topology == "erdos_renyi":
        edge_prob = network.get("er_edge_prob")
        if edge_prob is None:
            edge_prob = float(network["er_target_mean_degree"]) / float(max(num_nodes - 1, 1))
        return make_erdos_renyi_graph(num_nodes, float(edge_prob), seed=seed)
    if topology == "small_world":
        return make_watts_strogatz_graph(
            num_nodes,
            int(network["ws_degree"]),
            float(network["ws_rewiring_prob"]),
            seed=seed,
        )
    if topology == "scale_free":
        return make_barabasi_albert_graph(
            num_nodes, int(network["ba_attachments_per_new_node"]), seed=seed
        )
    raise ValueError(f"Unsupported topology: {topology}")


def _build_env_config(
    spec: Mapping[str, Any], graph: Mapping[int, Sequence[int]]
) -> SPGGConfig:
    dynamics = dict(spec["dynamics"])
    reward = dict(spec["reward"])
    config = SPGGConfig(**dynamics)
    config.reward = replace(
        config.reward,
        lambda_payoff=float(reward.get("lambda_payoff", 1.0)),
        lambda_cooperation=float(reward.get("lambda_cooperation", 0.0)),
        lambda_total_resource=float(reward.get("lambda_total_resource", 0.0)),
        lambda_collapse=float(reward.get("lambda_collapse", 0.0)),
        lambda_gini=float(reward.get("lambda_gini", 0.0)),
        epsilon=float(reward.get("epsilon", 1e-8)),
    )
    config.target_mean_degree = float(np.mean([len(neighbors) for neighbors in graph.values()]))
    return config


def _mean_sem(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array)) if array.size else 0.0
    sem = float(np.std(array, ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, sem


def _run_episode(
    *,
    config: SPGGConfig,
    graph: Mapping[int, Sequence[int]],
    policy: UnconditionalAllocationFloorPolicy,
    episode: int,
    episode_seed: int,
    max_steps: int | None,
    tail_fraction: float,
    topology: str,
) -> dict[str, Any]:
    env = SPGGEnv(config, graph)
    observation = env.reset(seed=episode_seed)
    cooperation: list[float] = []
    resources: list[float] = []
    ginis: list[float] = []
    max_allocation_row_error = 0.0
    max_transfer_conservation_error = 0.0
    step = 0
    done = False
    while not done and (max_steps is None or step < max_steps):
        pool_grown_total = float(np.sum(observation["pool_grown"]))
        allocation = policy.allocate(observation)
        max_allocation_row_error = max(
            max_allocation_row_error,
            float(np.max(np.abs(allocation.sum(axis=1) - 1.0))),
        )
        observation, _, done, info = env.step(allocation)
        max_transfer_conservation_error = max(
            max_transfer_conservation_error,
            abs(float(np.sum(info["transfer_matrix"])) - pool_grown_total),
        )
        cooperation.append(float(np.mean(observation["x_actual"])))
        resources.append(float(np.mean(observation["resources"])))
        ginis.append(float(np.asarray(observation["gini"]).item()))
        step += 1

    if not cooperation:
        raise RuntimeError("The rollout produced no steps.")
    tail_steps = max(1, int(math.ceil(len(cooperation) * tail_fraction)))
    tail = slice(len(cooperation) - tail_steps, len(cooperation))
    return {
        "topology": topology,
        "topology_label": TOPOLOGY_LABELS[topology],
        "epsilon": policy.epsilon,
        "episode": int(episode),
        "episode_seed": int(episode_seed),
        "steps": int(len(cooperation)),
        "tail_steps": int(tail_steps),
        "final_cooperation": cooperation[-1],
        "final_mean_resource": resources[-1],
        "final_gini": ginis[-1],
        "tail_cooperation": float(np.mean(cooperation[tail])),
        "tail_mean_resource": float(np.mean(resources[tail])),
        "tail_gini": float(np.mean(ginis[tail])),
        "max_allocation_row_error": max_allocation_row_error,
        "max_transfer_conservation_error": max_transfer_conservation_error,
    }


def _summarize(episode_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = (
        "final_cooperation",
        "final_mean_resource",
        "final_gini",
        "tail_cooperation",
        "tail_mean_resource",
        "tail_gini",
    )
    cases = sorted({(str(row["topology"]), float(row["epsilon"])) for row in episode_rows})
    for topology, epsilon in cases:
        selected = [
            row
            for row in episode_rows
            if str(row["topology"]) == topology and float(row["epsilon"]) == epsilon
        ]
        summary: dict[str, Any] = {
            "topology": topology,
            "topology_label": TOPOLOGY_LABELS[topology],
            "epsilon": epsilon,
            "episodes": len(selected),
        }
        for key in keys:
            mean, sem = _mean_sem([float(row[key]) for row in selected])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sem"] = sem
        summary["max_allocation_row_error"] = max(
            float(row["max_allocation_row_error"]) for row in selected
        )
        summary["max_transfer_conservation_error"] = max(
            float(row["max_transfer_conservation_error"]) for row in selected
        )
        summaries.append(summary)
    return summaries


def _paired_deltas(episode_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        (str(row["topology"]), int(row["episode"])): row
        for row in episode_rows
        if np.isclose(float(row["epsilon"]), 0.0)
    }
    delta_rows: list[dict[str, Any]] = []
    metric_keys = ("tail_cooperation", "tail_mean_resource", "tail_gini")
    for row in episode_rows:
        epsilon = float(row["epsilon"])
        if np.isclose(epsilon, 0.0):
            continue
        key = (str(row["topology"]), int(row["episode"]))
        if key not in baselines:
            continue
        baseline = baselines[key]
        delta = {
            "topology": str(row["topology"]),
            "topology_label": str(row["topology_label"]),
            "epsilon": epsilon,
            "episode": int(row["episode"]),
        }
        for metric in metric_keys:
            delta[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        delta_rows.append(delta)

    summaries: list[dict[str, Any]] = []
    cases = sorted({(str(row["topology"]), float(row["epsilon"])) for row in delta_rows})
    for topology, epsilon in cases:
        selected = [
            row
            for row in delta_rows
            if str(row["topology"]) == topology and float(row["epsilon"]) == epsilon
        ]
        summary: dict[str, Any] = {
            "topology": topology,
            "topology_label": TOPOLOGY_LABELS[topology],
            "epsilon": epsilon,
            "paired_episodes": len(selected),
        }
        for metric in metric_keys:
            key = f"delta_{metric}"
            summary[f"{key}_mean"], summary[f"{key}_sem"] = _mean_sem(
                [float(row[key]) for row in selected]
            )
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    summary_rows: Sequence[Mapping[str, Any]],
    topologies: Sequence[str],
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping the Appendix figure.")
        return []

    figure, axes = plt.subplots(
        len(topologies), 3, figsize=(10.5, 2.7 * len(topologies)), squeeze=False
    )
    for row_index, topology in enumerate(topologies):
        selected = sorted(
            [row for row in summary_rows if str(row["topology"]) == topology],
            key=lambda row: float(row["epsilon"]),
        )
        epsilon_values = [float(row["epsilon"]) for row in selected]
        x = np.arange(len(epsilon_values), dtype=np.float64)
        for column_index, (_, mean_key, sem_key, ylabel) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            y = np.asarray([float(row[mean_key]) for row in selected], dtype=np.float64)
            yerr = np.asarray([float(row[sem_key]) for row in selected], dtype=np.float64)
            axis.errorbar(x, y, yerr=yerr, color="#1E4688", marker="o", capsize=3)
            axis.grid(alpha=0.25, linewidth=0.6)
            axis.set_xticks(x)
            axis.set_xticklabels([f"{value:.2f}" for value in epsilon_values])
            axis.set_xlabel(r"Floor $\epsilon$")
            axis.set_ylabel(ylabel)
            if column_index == 0:
                axis.set_title(TOPOLOGY_LABELS[topology])
            if mean_key in {"tail_cooperation_mean", "tail_gini_mean"}:
                axis.set_ylim(-0.03, 1.03)
    figure.suptitle("Unconditional allocation floor: steady-state robustness")
    figure.tight_layout()
    output_paths = [
        output_dir / "unconditional_allocation_floor_appendix.png",
        output_dir / "unconditional_allocation_floor_appendix.pdf",
    ]
    for path in output_paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [str(path) for path in output_paths]


def run_self_checks() -> dict[str, float]:
    """Check formula endpoints, support, normalization, and transfer conservation."""
    mask = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    unit = np.asarray([1.0, 0.0, 2.0, 4.0], dtype=np.float64)
    observation = {"local_mask": mask, "unit_investment": unit}
    equal = mask.astype(np.float64) / mask.sum(axis=1, keepdims=True)
    weighted = mask.astype(np.float64) * unit[None, :]
    proportional = weighted / weighted.sum(axis=1, keepdims=True)
    allocation_zero = UnconditionalAllocationFloorPolicy(0.0).allocate(observation)
    repository_baseline = ProportionalContributionPolicy().allocate(observation)
    allocation_one = UnconditionalAllocationFloorPolicy(1.0).allocate(observation)
    allocation_small = UnconditionalAllocationFloorPolicy(0.05).allocate(observation)
    np.testing.assert_allclose(allocation_zero, proportional, atol=1e-14)
    np.testing.assert_allclose(allocation_zero, repository_baseline, atol=1e-14)
    np.testing.assert_allclose(allocation_one, equal, atol=1e-14)
    np.testing.assert_allclose(allocation_small.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(allocation_small[~mask], 0.0, atol=1e-14)

    zero_observation = {"local_mask": mask, "unit_investment": np.zeros(4)}
    np.testing.assert_allclose(
        UnconditionalAllocationFloorPolicy(0.05).allocate(zero_observation), equal, atol=1e-14
    )

    graph = make_random_regular_graph(8, 2, seed=7)
    config = SPGGConfig(
        alpha=0.5,
        r=0.5,
        p_mode="dynamic",
        p_c=5.0,
        episode_length=2,
        initial_resource=10.0,
    )
    env = SPGGEnv(config, graph)
    env_observation = env.reset(seed=11)
    allocation = UnconditionalAllocationFloorPolicy(0.05).allocate(env_observation)
    _, _, _, info = env.step(allocation)
    transfer_error = abs(
        float(np.sum(info["transfer_matrix"])) - float(np.sum(env_observation["pool_grown"]))
    )
    if transfer_error > 1e-10:
        raise AssertionError(f"Transfer conservation error is too large: {transfer_error}")
    return {
        "epsilon_zero_max_error": float(np.max(np.abs(allocation_zero - proportional))),
        "epsilon_zero_vs_repository_baseline_max_error": float(
            np.max(np.abs(allocation_zero - repository_baseline))
        ),
        "epsilon_one_max_error": float(np.max(np.abs(allocation_one - equal))),
        "row_sum_max_error": float(np.max(np.abs(allocation_small.sum(axis=1) - 1.0))),
        "transfer_conservation_error": transfer_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--epsilons",
        type=str,
        default=",".join(str(value) for value in DEFAULT_EPSILONS),
    )
    parser.add_argument("--topologies", type=str, default=",".join(DEFAULT_TOPOLOGIES))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Override dynamics.episode_length; default uses results.json.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional rollout cap; default runs the full episode.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.20,
        help="Fraction of the end of each episode used as the steady-state window.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = run_self_checks()
    print("Self-checks passed:", json.dumps(checks, sort_keys=True))
    if args.check_only:
        return
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.episode_length is not None and args.episode_length <= 0:
        raise ValueError("--episode-length must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must lie in (0, 1].")

    epsilons = _parse_float_list(args.epsilons)
    if not any(np.isclose(value, 0.0) for value in epsilons):
        raise ValueError("Include epsilon=0 so the proportional baseline and paired deltas are defined.")
    topologies = _parse_topologies(args.topologies)
    run_dir = args.run_dir.expanduser().resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / "unconditional_allocation_floor").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_spec = _extract_experiment_spec(results_path)
    seed_base = int(base_spec.get("seed", 42)) if args.seed is None else int(args.seed)
    reference_degree = _reference_mean_degree(base_spec)
    episode_rows: list[dict[str, Any]] = []
    total_cases = len(topologies) * len(epsilons)
    case_index = 0
    for topology in topologies:
        topology_spec = _spec_for_topology(
            base_spec,
            topology,
            reference_mean_degree=reference_degree,
            episode_length=args.episode_length,
        )
        topology_spec["seed"] = seed_base
        graph = _build_graph(topology_spec)
        config = _build_env_config(topology_spec, graph)
        for epsilon in epsilons:
            case_index += 1
            print(
                f"[{case_index}/{total_cases}] topology={topology}, epsilon={epsilon:g}, "
                f"episodes={args.episodes}, length={config.episode_length}"
            )
            policy = UnconditionalAllocationFloorPolicy(epsilon)
            for episode in range(args.episodes):
                episode_rows.append(
                    _run_episode(
                        config=config,
                        graph=graph,
                        policy=policy,
                        episode=episode,
                        episode_seed=seed_base + episode,
                        max_steps=args.max_steps,
                        tail_fraction=float(args.tail_fraction),
                        topology=topology,
                    )
                )

    summary_rows = _summarize(episode_rows)
    paired_delta_rows = _paired_deltas(episode_rows)
    _write_csv(output_dir / "episode_metrics.csv", episode_rows)
    _write_csv(output_dir / "robustness_summary.csv", summary_rows)
    _write_csv(output_dir / "paired_deltas_vs_epsilon0.csv", paired_delta_rows)
    figure_paths = _plot_summary(summary_rows, topologies, output_dir)

    payload = {
        "method": "unconditional_allocation_floor",
        "formula": "A_epsilon = (1-epsilon)*A_proportional + epsilon*A_equal",
        "proportional_weight": "unit_investment = e_i/(d_i+1)",
        "zero_contribution_fallback": "equal allocation",
        "run_dir": str(run_dir),
        "results_json": str(results_path),
        "output_dir": str(output_dir),
        "epsilons": epsilons,
        "topologies": topologies,
        "episodes": int(args.episodes),
        "episode_length_override": args.episode_length,
        "max_steps": args.max_steps,
        "tail_fraction": float(args.tail_fraction),
        "seed": seed_base,
        "paired_design": True,
        "self_checks": checks,
        "summary": summary_rows,
        "paired_deltas_vs_epsilon0": paired_delta_rows,
        "figures": figure_paths,
    }
    with (output_dir / "robustness_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"Finished. Results written to {output_dir}")
    for row in summary_rows:
        print(
            f"{row['topology_label']:>7s} eps={float(row['epsilon']):.2f} | "
            f"f_C={float(row['tail_cooperation_mean']):.4f} | "
            f"Rbar={float(row['tail_mean_resource_mean']):.4f} | "
            f"Gini={float(row['tail_gini_mean']):.4f}"
        )


if __name__ == "__main__":
    main()
