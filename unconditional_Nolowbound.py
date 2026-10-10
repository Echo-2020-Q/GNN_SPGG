from __future__ import annotations

"""Run the proportional baseline without an individual resource lower bound.

This intervention changes only the individual resource constraint:

* nominal cooperation is always realized, even when resources are below d_i+1;
* a cooperator pays its prescribed investment without an affordability cap;
* consequently, cooperation costs may make cumulative resources negative;
* ordinary consumption remains capped by the non-negative available balance.

The strategy-update rule, pool dynamics, and proportional allocation rule are
otherwise identical to the source experiment in ``results.json``.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from Project1.env import SPGGEnv, Observation, gini_coefficient
from Project1.policies.rule_based import ProportionalContributionPolicy
from unconditional_allocation_floor import (
    DEFAULT_RUN_DIR,
    DEFAULT_TOPOLOGIES,
    TOPOLOGY_LABELS,
    _build_env_config,
    _build_graph,
    _extract_experiment_spec,
    _parse_topologies,
    _reference_mean_degree,
    _spec_for_topology,
)


class NoLowerBoundSPGGEnv(SPGGEnv):
    """SPGG environment in which cooperation costs can create debt."""

    def _precompute_observation(
        self,
        nominal_strategies: np.ndarray,
        resources: np.ndarray,
    ) -> Observation:
        # The intervention: available resources do not alter the chosen strategy.
        actual_strategies = nominal_strategies.astype(np.float64, copy=False)

        investment_base = self._thresholds_float64 + self.config.alpha * (
            resources - self._thresholds_float64
        )
        # Keep the model's non-negative prescribed investment, but remove the
        # original min(resources, prescribed_investment) affordability cap.
        investment = actual_strategies * np.maximum(0.0, investment_base)
        unit_investment = investment / self._thresholds_float64

        pool_raw = self._local_mask_float64 @ unit_investment
        local_actual_cooperators = self._local_mask_float64 @ actual_strategies
        pool_theoretical_max = self._compute_pool_theoretical_max()
        pool_capacity = self._compute_pool_capacity(local_actual_cooperators)
        pool_grown = np.minimum((1.0 + self.config.r) * pool_raw, pool_capacity)
        pool_raw_norm = np.divide(
            pool_raw,
            pool_theoretical_max,
            out=np.zeros_like(pool_raw, dtype=np.float64),
            where=pool_theoretical_max > 1e-8,
        )
        resource_norm = resources / self.resource_norm_reference
        strategy_norm = np.divide(
            investment,
            resources,
            out=np.zeros_like(resources, dtype=np.float64),
            where=np.abs(resources) > 1e-12,
        )

        # This is the generalized Gini used by the repository. With negative
        # observations it can exceed [0, 1], so debt statistics are also
        # reported explicitly by this script.
        resource_gini = gini_coefficient(resources, self.config.reward.epsilon)
        return {
            "x_nominal": nominal_strategies.astype(np.int8, copy=True),
            "x_actual": actual_strategies.astype(np.float64, copy=True),
            "resources": resources.astype(np.float64, copy=True),
            "investment": investment.astype(np.float64, copy=True),
            "unit_investment": unit_investment.astype(np.float64, copy=True),
            "pool_raw": pool_raw.astype(np.float64, copy=True),
            "pool_grown": pool_grown.astype(np.float64, copy=True),
            "pool_capacity": pool_capacity.astype(np.float64, copy=True),
            "pool_theoretical_max": pool_theoretical_max.astype(np.float64, copy=True),
            "local_actual_cooperators": local_actual_cooperators.astype(np.float64, copy=True),
            "degrees": self._degrees_int64.copy(),
            "pool_raw_norm": pool_raw_norm.astype(np.float64, copy=True),
            "resource_norm": resource_norm.astype(np.float64, copy=True),
            "degree_norm": self._degree_norm_float64.copy(),
            "strategy_norm": strategy_norm.astype(np.float64, copy=True),
            "gini": np.asarray(resource_gini, dtype=np.float64),
            "p_max": pool_theoretical_max.astype(np.float64, copy=True),
            "local_mask": self._local_mask_bool.copy(),
        }

    def _compute_effective_consumption(
        self,
        resources: np.ndarray,
        resources_after_exchange: np.ndarray,
    ) -> np.ndarray:
        desired = np.maximum(self._compute_nominal_consumption(resources), 0.0)
        # Preserve the consumption affordability constraint. The additional
        # zero floor is necessary because the original environment assumed the
        # post-exchange balance could never be negative; without it, min() would
        # create negative consumption and incorrectly erase cooperation debt.
        available = np.maximum(resources_after_exchange, 0.0)
        return np.minimum(desired, available)


def _mean_sem(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array)) if array.size else 0.0
    sem = float(np.std(array, ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, sem


def _run_episode(
    *,
    config: Any,
    graph: Mapping[int, Sequence[int]],
    policy: ProportionalContributionPolicy,
    topology: str,
    episode: int,
    episode_seed: int,
    max_steps: int | None,
    tail_fraction: float,
) -> dict[str, Any]:
    env = NoLowerBoundSPGGEnv(config, graph)
    observation = env.reset(seed=episode_seed)
    cooperation: list[float] = []
    resources_mean: list[float] = []
    resource_gini: list[float] = []
    negative_fraction: list[float] = []
    minimum_resource: list[float] = []
    max_strategy_mismatch = 0.0
    max_allocation_row_error = 0.0
    max_transfer_conservation_error = 0.0
    max_unfunded_investment = 0.0
    done = False
    step = 0

    while not done and (max_steps is None or step < max_steps):
        max_strategy_mismatch = max(
            max_strategy_mismatch,
            float(
                np.max(
                    np.abs(
                        observation["x_actual"]
                        - observation["x_nominal"].astype(np.float64)
                    )
                )
            ),
        )
        max_unfunded_investment = max(
            max_unfunded_investment,
            float(
                np.max(
                    observation["investment"]
                    - np.maximum(observation["resources"], 0.0)
                )
            ),
        )
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
        current_resources = np.asarray(observation["resources"], dtype=np.float64)
        cooperation.append(float(np.mean(observation["x_actual"])))
        resources_mean.append(float(np.mean(current_resources)))
        resource_gini.append(float(np.asarray(observation["gini"]).item()))
        negative_fraction.append(float(np.mean(current_resources < 0.0)))
        minimum_resource.append(float(np.min(current_resources)))
        step += 1

    if not cooperation:
        raise RuntimeError("The rollout produced no steps.")
    tail_steps = max(1, int(math.ceil(len(cooperation) * tail_fraction)))
    tail = slice(len(cooperation) - tail_steps, len(cooperation))
    return {
        "topology": topology,
        "topology_label": TOPOLOGY_LABELS[topology],
        "policy": "proportional_no_lower_bound",
        "episode": int(episode),
        "episode_seed": int(episode_seed),
        "steps": int(len(cooperation)),
        "tail_steps": int(tail_steps),
        "final_cooperation": cooperation[-1],
        "final_mean_resource": resources_mean[-1],
        "final_gini_generalized": resource_gini[-1],
        "final_negative_resource_fraction": negative_fraction[-1],
        "final_minimum_resource": minimum_resource[-1],
        "tail_cooperation": float(np.mean(cooperation[tail])),
        "tail_mean_resource": float(np.mean(resources_mean[tail])),
        "tail_gini_generalized": float(np.mean(resource_gini[tail])),
        "tail_negative_resource_fraction": float(np.mean(negative_fraction[tail])),
        "tail_minimum_resource": float(np.mean(minimum_resource[tail])),
        "minimum_resource_over_episode": float(np.min(minimum_resource)),
        "ever_negative": bool(np.min(minimum_resource) < 0.0),
        "max_strategy_mismatch": max_strategy_mismatch,
        "max_unfunded_investment": max_unfunded_investment,
        "max_allocation_row_error": max_allocation_row_error,
        "max_transfer_conservation_error": max_transfer_conservation_error,
    }


def _summarize(episode_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = (
        "final_cooperation",
        "final_mean_resource",
        "final_gini_generalized",
        "final_negative_resource_fraction",
        "final_minimum_resource",
        "tail_cooperation",
        "tail_mean_resource",
        "tail_gini_generalized",
        "tail_negative_resource_fraction",
        "tail_minimum_resource",
        "minimum_resource_over_episode",
    )
    summaries: list[dict[str, Any]] = []
    for topology in DEFAULT_TOPOLOGIES:
        selected = [row for row in episode_rows if str(row["topology"]) == topology]
        if not selected:
            continue
        summary: dict[str, Any] = {
            "topology": topology,
            "topology_label": TOPOLOGY_LABELS[topology],
            "policy": "proportional_no_lower_bound",
            "episodes": len(selected),
            "episodes_ever_negative": sum(bool(row["ever_negative"]) for row in selected),
        }
        for key in metric_keys:
            summary[f"{key}_mean"], summary[f"{key}_sem"] = _mean_sem(
                [float(row[key]) for row in selected]
            )
        for key in (
            "max_strategy_mismatch",
            "max_unfunded_investment",
            "max_allocation_row_error",
            "max_transfer_conservation_error",
        ):
            summary[key] = max(float(row[key]) for row in selected)
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(summary_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping summary figures.")
        return []

    metric_specs = (
        ("tail_cooperation_mean", r"$f_C$"),
        ("tail_mean_resource_mean", r"$\bar{R}$"),
        ("tail_gini_generalized_mean", "Generalized Gini"),
    )
    labels = [str(row["topology_label"]) for row in summary_rows]
    positions = np.arange(len(summary_rows), dtype=np.float64)
    figure, axes = plt.subplots(1, len(metric_specs), figsize=(10.5, 5.2), squeeze=False)
    for index, (mean_key, ylabel) in enumerate(metric_specs):
        axis = axes[0, index]
        sem_key = mean_key.removesuffix("_mean") + "_sem"
        values = [float(row[mean_key]) for row in summary_rows]
        errors = [float(row[sem_key]) for row in summary_rows]
        axis.bar(positions, values, yerr=errors, capsize=4, color="#1E4688", alpha=0.88)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=14)
        axis.set_ylabel(ylabel, fontsize=16, fontweight="bold", labelpad=10)
        axis.tick_params(axis="y", labelsize=13)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    figure.suptitle(
        "Proportional allocation without an individual resource lower bound",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), w_pad=2.2)
    paths = [
        output_dir / "proportional_no_lower_bound_summary.png",
        output_dir / "proportional_no_lower_bound_summary.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [str(path) for path in paths]


def run_self_checks() -> dict[str, float]:
    from Project1.env import SPGGConfig

    graph = {0: []}
    config = SPGGConfig(
        alpha=0.0,
        r=0.0,
        p_max=0.1,
        resource_consumption_mode="fixed",
        resource_consumption_fixed=1.0,
        beta=0.0,
        episode_length=2,
    )
    env = NoLowerBoundSPGGEnv(config, graph)
    observation = env.reset(
        initial_resources=[0.5],
        initial_strategies=[1],
        seed=0,
    )
    np.testing.assert_array_equal(observation["x_actual"], np.ones(1))
    np.testing.assert_allclose(observation["investment"], np.ones(1))
    if not np.all(observation["investment"] > observation["resources"]):
        raise AssertionError("The cooperation affordability constraint was not removed.")

    policy = ProportionalContributionPolicy()
    allocation = policy.allocate(observation)
    np.testing.assert_allclose(allocation.sum(axis=1), 1.0, atol=1e-14)
    next_observation, _, _, info = env.step(allocation)
    np.testing.assert_allclose(next_observation["resources"], np.full(1, -0.4), atol=1e-14)
    np.testing.assert_array_equal(next_observation["x_actual"], np.ones(1))
    np.testing.assert_allclose(next_observation["investment"], np.ones(1))
    transfer_error = abs(
        float(np.sum(info["transfer_matrix"])) - float(np.sum(observation["pool_grown"]))
    )
    if transfer_error > 1e-10:
        raise AssertionError(f"Transfer conservation error is too large: {transfer_error}")
    return {
        "initial_resource": float(observation["resources"][0]),
        "unconstrained_cooperation_cost": float(observation["investment"][0]),
        "next_resource": float(next_observation["resources"][0]),
        "actual_strategy_at_negative_resource": float(next_observation["x_actual"][0]),
        "transfer_conservation_error": transfer_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--topologies", type=str, default=",".join(DEFAULT_TOPOLOGIES))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--tail-fraction", type=float, default=0.20)
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

    run_dir = args.run_dir.expanduser().resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / "proportional_no_lower_bound").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    topologies = _parse_topologies(args.topologies)
    base_spec = _extract_experiment_spec(results_path)
    reference_degree = _reference_mean_degree(base_spec)
    seed_base = int(base_spec.get("seed", 42)) if args.seed is None else int(args.seed)
    policy = ProportionalContributionPolicy()
    episode_rows: list[dict[str, Any]] = []

    for topology_index, topology in enumerate(topologies, start=1):
        topology_spec = _spec_for_topology(
            base_spec,
            topology,
            reference_mean_degree=reference_degree,
            episode_length=args.episode_length,
        )
        topology_spec["seed"] = seed_base
        graph = _build_graph(topology_spec)
        config = _build_env_config(topology_spec, graph)
        print(
            f"[{topology_index}/{len(topologies)}] topology={topology}, "
            f"policy=proportional, episodes={args.episodes}, length={config.episode_length}"
        )
        for episode in range(args.episodes):
            episode_rows.append(
                _run_episode(
                    config=config,
                    graph=graph,
                    policy=policy,
                    topology=topology,
                    episode=episode,
                    episode_seed=seed_base + episode,
                    max_steps=args.max_steps,
                    tail_fraction=float(args.tail_fraction),
                )
            )

    summary_rows = _summarize(episode_rows)
    _write_csv(output_dir / "episode_metrics.csv", episode_rows)
    _write_csv(output_dir / "topology_summary.csv", summary_rows)
    figure_paths = _plot_summary(summary_rows, output_dir)
    payload = {
        "method": "proportional_allocation_without_individual_resource_lower_bound",
        "allocation_policy": "ProportionalContributionPolicy",
        "intervention": {
            "actual_strategy_equals_nominal_strategy": True,
            "cooperation_investment_affordability_cap": False,
            "consumption_affordability_cap": True,
            "negative_individual_resources_allowed": True,
        },
        "gini_note": (
            "The repository generalized Gini is retained for continuity; with negative resources "
            "it is not restricted to [0, 1]. Use the explicit debt metrics alongside it."
        ),
        "run_dir": str(run_dir),
        "results_json": str(results_path),
        "output_dir": str(output_dir),
        "topologies": topologies,
        "episodes": int(args.episodes),
        "episode_length_override": args.episode_length,
        "max_steps": args.max_steps,
        "tail_fraction": float(args.tail_fraction),
        "seed": seed_base,
        "self_checks": checks,
        "summary": summary_rows,
        "figures": figure_paths,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"Finished. Results written to {output_dir}")
    for row in summary_rows:
        print(
            f"{row['topology_label']:>7s} | "
            f"f_C={float(row['tail_cooperation_mean']):.4f} | "
            f"Rbar={float(row['tail_mean_resource_mean']):.4f} | "
            f"Gini*={float(row['tail_gini_generalized_mean']):.4f} | "
            f"R<0={float(row['tail_negative_resource_fraction_mean']):.4f} | "
            f"minR={float(row['minimum_resource_over_episode_mean']):.4f}"
        )


if __name__ == "__main__":
    main()
