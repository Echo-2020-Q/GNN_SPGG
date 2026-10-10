"""Run complete SPGG episodes with the pretrained GNN or saved M1/M2 rules."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from env import (
    RewardConfig,
    SPGGConfig,
    SPGGEnv,
    make_barabasi_albert_graph,
    make_erdos_renyi_graph,
    make_random_regular_graph,
    make_watts_strogatz_graph,
)
from mechanisms import load_m1, load_m2


TOPOLOGIES = {
    "regular": "regular",
    "ba": "scale_free",
    "scale_free": "scale_free",
    "er": "erdos_renyi",
    "erdos_renyi": "erdos_renyi",
    "ws": "small_world",
    "small_world": "small_world",
}


def load_config(path: str | Path | None = None) -> dict:
    resolved = Path(path) if path is not None else Path(__file__).parent / "environment.json"
    with resolved.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_environment(
    config: dict,
    *,
    topology: str = "regular",
    graph_index: int = 0,
    graph_seed: int | None = None,
    steps: int | None = None,
) -> tuple[SPGGEnv, int]:
    """Build a fixed graph and the matching SPGG observation normalization."""
    topology = TOPOLOGIES[topology]
    if graph_index < 0:
        raise ValueError("graph_index must be non-negative.")
    if graph_seed is None:
        seeds = config["reference_graph_seeds"][topology]
        if graph_index >= len(seeds):
            raise ValueError(f"graph_index must be between 0 and {len(seeds) - 1} for {topology}.")
        graph_seed = int(seeds[graph_index])
    if graph_seed < 0:
        raise ValueError("graph_seed must be non-negative.")
    network = config["network"]
    count = int(network["num_nodes"])
    if topology == "regular":
        degree = int(network["regular_degree"])
        graph = make_random_regular_graph(count, degree, seed=graph_seed)
        target_degree = float(degree)
    elif topology == "scale_free":
        attachments = int(network["ba_attachments_per_new_node"])
        graph = make_barabasi_albert_graph(count, attachments, seed=graph_seed)
        target_degree = float(2 * attachments - attachments * (attachments + 1) / max(count, 1))
    elif topology == "erdos_renyi":
        target_degree = float(network["er_target_mean_degree"])
        graph = make_erdos_renyi_graph(count, target_degree / max(count - 1, 1), seed=graph_seed)
    else:
        degree = int(network["ws_degree"])
        graph = make_watts_strogatz_graph(count, degree, float(network["ws_rewiring_prob"]), seed=graph_seed)
        target_degree = float(degree)
    dynamics = dict(config["dynamics"])
    if steps is not None:
        dynamics["episode_length"] = int(steps)
    env_config = SPGGConfig(
        **dynamics,
        target_mean_degree=target_degree,
        reward=RewardConfig(**config["reward"]),
    )
    return SPGGEnv(env_config, graph), int(graph_seed)


def episode_seed(seed_base: int, topology: str, graph_index: int, repeat: int) -> int:
    # Same seed construction as the fixed-graph M2 repeat experiment.
    topology = TOPOLOGIES[topology]
    offset = sum((index + 1) * ord(char) for index, char in enumerate(topology))
    return int(np.random.SeedSequence([seed_base, offset, graph_index, repeat]).generate_state(1, dtype=np.uint64)[0])


def _metrics(observation: dict) -> dict:
    return {
        "gini": float(np.asarray(observation["gini"]).item()),
        "mean_resource": float(np.mean(observation["resources"])),
        "cooperation_rate": float(np.mean(observation["x_actual"])),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("all", "gnn", "m1", "m2"), default="all")
    parser.add_argument("--topology", choices=tuple(TOPOLOGIES), default="regular")
    parser.add_argument("--graph-index", type=int, default=0, help="Reference graph index (0..9).")
    parser.add_argument("--graph-seed", type=int, help="Override the reference graph seed.")
    parser.add_argument("--steps", type=int, help="Steps per run; defaults to 200 in environment.json.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats on the same fixed graph.")
    parser.add_argument("--seed", type=int, help="Base simulation seed; defaults to environment.json.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "environment.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or (args.steps is not None and args.steps <= 0):
        parser.error("--repeats and --steps must be positive.")
    if args.graph_index < 0 or (args.graph_seed is not None and args.graph_seed < 0):
        parser.error("Graph index and seed must be non-negative.")
    config = load_config(args.config)
    topology = TOPOLOGIES[args.topology]
    seed_base = int(config["seed"]) if args.seed is None else args.seed
    if seed_base < 0:
        parser.error("Simulation seed must be non-negative.")
    env, graph_seed = build_environment(
        config, topology=topology, graph_index=args.graph_index,
        graph_seed=args.graph_seed, steps=args.steps,
    )
    steps = env.config.episode_length
    names = ("gnn", "m1", "m2") if args.policy == "all" else (args.policy,)
    policies = {}
    for name in names:
        if name == "gnn":
            import torch
            from gnn import load_gnn

            torch.set_num_threads(1)
            policies[name] = load_gnn(device=args.device)
        else:
            policies[name] = load_m1() if name == "m1" else load_m2()
    output_dir = args.output_dir or (
        Path(__file__).parent / "results" /
        f"{topology}_graph{args.graph_index}_seed{graph_seed}_{args.policy}_{steps}steps_{args.repeats}repeats"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_rows = []
    summary_rows = []
    trajectory_path = output_dir / "trajectories.csv"
    with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for name, policy in policies.items():
            for repeat in range(args.repeats):
                seed = episode_seed(seed_base, topology, args.graph_index, repeat)
                # Every policy starts from the same initial state and random seed.
                observation = env.reset(seed=seed)
                identity = {
                    "policy": name, "topology": topology, "graph_index": args.graph_index,
                    "graph_seed": graph_seed, "repeat": repeat, "episode_seed": seed,
                }
                for step in range(steps + 1):
                    if step > 0:
                        observation, _, done, _ = env.step(policy.allocate(observation))
                        if done != (step == steps):
                            raise RuntimeError("Environment termination does not match the requested step count.")
                    metrics = _metrics(observation)
                    row = {**identity, "step": step, **metrics}
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise RuntimeError("Non-finite simulation metric encountered.")
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                final_rows.append({
                    **identity, "steps": steps,
                    **{f"final_{key}": value for key, value in _metrics(observation).items()},
                })
                print(f"{name.upper()} {repeat + 1}/{args.repeats}: step={steps}, Gini={row['gini']:.6f}", flush=True)
            runs = [row for row in final_rows if row["policy"] == name]
            values = np.asarray([row["final_gini"] for row in runs], dtype=np.float64)
            summary_rows.append({
                "policy": name, "topology": topology, "graph_index": args.graph_index,
                "graph_seed": graph_seed, "steps": steps, "count_runs": len(runs),
                "final_gini_mean": float(np.mean(values)),
                "final_gini_std": float(np.std(values, ddof=0)), "std_ddof": 0,
            })
    _write_csv(output_dir / "episodes.csv", final_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    record = {
        "policies": list(names), "topology": topology, "graph_index": args.graph_index,
        "graph_seed": graph_seed, "seed_base": seed_base, "steps": steps, "repeats": args.repeats,
        "environment": asdict(env.config),
        "network": {**config["network"], "type": topology},
        "episode_seed_method": "SeedSequence([seed_base, topology_offset, graph_index, repeat]) -> uint64",
        "step_convention": "step=0 is the initial state; step=200 is after 200 environment updates.",
        "statistic": "Final-step Gini across repeats on the same graph; population std (ddof=0).",
        "m2_replay": "Select Q1/Q2/Q3/Q4 by sender degree intervals; retain Q4 fallback only for degrees outside the fitted 2..19 range.",
    }
    (output_dir / "run_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Final Gini (mean ± std):")
    for row in summary_rows:
        print(f"  {row['policy'].upper()}: {row['final_gini_mean']:.6f} ± {row['final_gini_std']:.6g}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
