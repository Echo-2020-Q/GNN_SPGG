"""固定参考图，在 M2 机制下重复仿真，统计最后一步资源 Gini 的均值和标准差。

默认沿用 all_cooperation_Gini.py 的 40 张图（每种拓扑编号 0..9）。
每张图固定建图种子，改变仿真随机种子重复 50 次，每次运行 200 步。
统计第 200 次更新后的 Gini，不对末尾时间段取平均；标准差使用 ddof=0。
M2 与 analyze_five_policy_dynamics_grid.py 使用同一份已保存的条件分配策略。
按 0429 分析脚本的度数区间选择分组；输出另存为 degree_conditioned 目录。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from all_cooperation_Gini import (
    TOPOLOGY_LABELS,
    _apply_replayed_graph_metadata,
    _build_embedded_reference_graph_rows,
    _load_graph_metadata_rows,
    _ordered_topologies,
    _topology_seed_offset,
    _write_csv,
)
from analyze_gnn_policy_decisions import (
    _load_json,
    build_env_config_from_spec,
    build_graph_from_spec,
    build_spec_for_topology,
    resolve_reference_mean_degree,
)
from Project1.env import SPGGEnv
from run_saved_resource_conditioned_spline_policy import _load_saved_conditioned_policy


SCRIPT_DEFAULTS = {
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    "m2_summary_json": (
        "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
        "0421_rule_allocation_over_time_ba_degree_conditioned_discrete/analysis_summary.json"
    ),
    "output_dir": None,
    "topologies": "regular,scale_free,erdos_renyi,small_world",
    # all 表示使用每种拓扑的全部参考图，也可写成 "0" 或 "0,1,2"。
    "graph_indices": "all",
    # None 表示使用 all_cooperation_Gini.py 内置的图编号、种子和建图参数。
    "graph_metadata_csv": None,
    "repeats": 50,
    "steps": 200,
    # None 表示使用 results.json 中的 seed 作为仿真随机种子的基础。
    "seed": None,
    "num_workers": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(SCRIPT_DEFAULTS["run_dir"]))
    parser.add_argument("--m2-summary-json", type=Path, default=Path(SCRIPT_DEFAULTS["m2_summary_json"]))
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DEFAULTS["output_dir"])
    parser.add_argument("--topologies", default=SCRIPT_DEFAULTS["topologies"])
    parser.add_argument("--graph-indices", default=SCRIPT_DEFAULTS["graph_indices"])
    parser.add_argument("--graph-metadata-csv", type=Path, default=SCRIPT_DEFAULTS["graph_metadata_csv"])
    parser.add_argument("--repeats", type=int, default=SCRIPT_DEFAULTS["repeats"], help="Runs per fixed graph.")
    parser.add_argument("--steps", "--max-steps", dest="steps", type=int, default=SCRIPT_DEFAULTS["steps"])
    parser.add_argument("--seed", type=int, default=SCRIPT_DEFAULTS["seed"])
    parser.add_argument("--num-workers", type=int, default=SCRIPT_DEFAULTS["num_workers"])
    args = parser.parse_args()
    for name in ("repeats", "steps", "num_workers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative.")
    return args


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def _select_graph_rows(
    rows_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    topologies: Sequence[str],
    graph_indices: str,
) -> list[dict[str, Any]]:
    requested = None
    if str(graph_indices).strip().lower() != "all":
        requested = sorted({int(item.strip()) for item in str(graph_indices).split(",") if item.strip()})
        if not requested or min(requested) < 0:
            raise ValueError("--graph-indices must be 'all' or comma-separated non-negative indices.")
    selected = []
    for topology in topologies:
        available = sorted(index for name, index in rows_by_key if name == topology)
        if not available:
            raise ValueError(f"No reference graphs found for {topology}.")
        indices = available if requested is None else requested
        for index in indices:
            if (topology, index) not in rows_by_key:
                raise ValueError(f"Reference graph missing: {topology}, graph_index={index}.")
            if index < 0:
                raise ValueError("Reference graph indices must be non-negative.")
            selected.append(dict(rows_by_key[(topology, index)]))
    return selected


def _episode_seed(seed_base: int, topology: str, graph_index: int, repeat_index: int) -> int:
    # 种子仅由这四个值决定，与 worker 数量、图的选择及执行顺序无关。
    sequence = np.random.SeedSequence([
        int(seed_base), _topology_seed_offset(topology), int(graph_index), int(repeat_index)
    ])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _gini_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty set of runs.")
    values = np.asarray([float(row["final_gini"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite final Gini encountered.")
    return {
        "count_runs": int(values.size),
        "final_gini_mean": float(np.mean(values)),
        "final_gini_std": float(np.std(values, ddof=0)),
        "final_gini_min": float(np.min(values)),
        "final_gini_max": float(np.max(values)),
        "std_ddof": 0,
    }


def _run_fixed_graph(task: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = task["spec"]
    topology = str(task["topology"])
    graph_index = int(task["graph_index"])
    graph = build_graph_from_spec(spec)
    adjacency = [[int(node), sorted(int(item) for item in graph[node])] for node in sorted(graph)]
    graph_sha256 = hashlib.sha256(json.dumps(adjacency, separators=(",", ":")).encode()).hexdigest()
    env = SPGGEnv(build_env_config_from_spec(spec, graph), graph)
    policy, _, _, _, _ = _load_saved_conditioned_policy(
        Path(task["m2_summary_json"]),
        replay_ego_resource_x_mode="absolute",
        replay_ego_resource_normalizer=None,
        saved_ego_resource_normalizer=None,
    )
    common = {
        "topology": topology,
        "topology_label": TOPOLOGY_LABELS.get(topology, topology),
        "graph_index": graph_index,
        "graph_seed": int(spec["seed"]),
        "graph_sha256": graph_sha256,
    }
    runs = []
    for repeat_index in range(int(task["repeats"])):
        seed = _episode_seed(int(task["seed_base"]), topology, graph_index, repeat_index)
        observation = env.reset(seed=seed)
        for step in range(int(task["steps"])):
            allocation = np.asarray(policy.allocate(observation), dtype=np.float64)
            observation, _, done, _ = env.step(allocation)
            if done and step + 1 != int(task["steps"]):
                raise RuntimeError("Environment terminated before the requested final step.")
        runs.append({
            **common,
            "repeat": repeat_index,
            "episode_seed": seed,
            "steps": int(task["steps"]),
            "final_gini": float(np.asarray(observation["gini"]).item()),
            "final_mean_resource": float(np.mean(observation["resources"])),
            "final_actual_cooperation_rate": float(np.mean(observation["x_actual"])),
        })
    metadata = {
        **common,
        # 保留原 CSV 的 episode 编号，方便核对参考图。
        "episode": graph_index,
        "graph_num_nodes": len(graph),
        "graph_mean_degree": float(np.mean([len(neighbors) for neighbors in graph.values()])),
        "regular_degree": spec["network"].get("regular_degree"),
        "er_target_mean_degree": spec["network"].get("er_target_mean_degree"),
        "ws_degree": spec["network"].get("ws_degree"),
        "ws_rewiring_prob": spec["network"].get("ws_rewiring_prob"),
        "ba_attachments_per_new_node": spec["network"].get("ba_attachments_per_new_node"),
    }
    return runs, metadata


def main() -> None:
    args = parse_args()
    run_dir = _resolve_path(args.run_dir)
    m2_summary_json = _resolve_path(args.m2_summary_json)
    payload = _load_json(run_dir / "results.json")
    experiment_spec = payload.get("experiment", payload)
    if not isinstance(experiment_spec, Mapping):
        raise TypeError("results.json must contain an experiment mapping.")
    # 在开始仿真前验证 M2 参数文件。
    _, _, profile_summary, policy_key, _ = _load_saved_conditioned_policy(
        m2_summary_json,
        replay_ego_resource_x_mode="absolute",
        replay_ego_resource_normalizer=None,
        saved_ego_resource_normalizer=None,
    )
    seed_base = int(experiment_spec.get("seed", 42)) if args.seed is None else int(args.seed)
    if seed_base < 0:
        raise ValueError("Simulation seed must be non-negative.")
    topologies = _ordered_topologies(args.topologies)
    graph_metadata_csv = None if args.graph_metadata_csv is None else _resolve_path(args.graph_metadata_csv)
    graph_rows = _select_graph_rows(
        _build_embedded_reference_graph_rows() if graph_metadata_csv is None
        else _load_graph_metadata_rows(graph_metadata_csv),
        topologies,
        args.graph_indices,
    )
    reference_degree = resolve_reference_mean_degree(experiment_spec)
    tasks = []
    for row in graph_rows:
        spec = build_spec_for_topology(
            experiment_spec, str(row["topology"]), reference_mean_degree=reference_degree
        )
        spec, _ = _apply_replayed_graph_metadata(spec, row, replay_mode="m2_fixed_graph_replay")
        spec["dynamics"] = deepcopy(dict(spec["dynamics"]))
        # 只设置总步长；其余环境参数（含初始合作率和策略演化规则）沿用训练配置。
        spec["dynamics"]["episode_length"] = int(args.steps)
        tasks.append({
            "spec": spec,
            "topology": str(row["topology"]),
            "graph_index": int(row["episode"]),
            "m2_summary_json": str(m2_summary_json),
            "repeats": int(args.repeats),
            "steps": int(args.steps),
            "seed_base": seed_base,
        })
    output_dir = (
        _resolve_path(args.output_dir) if args.output_dir is not None
        else run_dir / f"M2_Gini_{args.steps}steps_{args.repeats}repeats_degree_conditioned"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"M2: {len(tasks)} fixed graphs x {args.repeats} repeats x {args.steps} steps "
        f"= {len(tasks) * args.repeats} runs; std ddof=0",
        flush=True,
    )
    all_runs: list[dict[str, Any]] = []
    graph_summaries: list[dict[str, Any]] = []
    graph_metadata: list[dict[str, Any]] = []
    executor = None
    try:
        if args.num_workers == 1:
            results = map(_run_fixed_graph, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=min(int(args.num_workers), len(tasks)))
            results = executor.map(_run_fixed_graph, tasks)
        for index, (runs, metadata) in enumerate(results, start=1):
            stats = _gini_statistics(runs)
            all_runs.extend(runs)
            graph_metadata.append(metadata)
            graph_summaries.append({
                "topology": metadata["topology"],
                "topology_label": metadata["topology_label"],
                "graph_index": metadata["graph_index"],
                "graph_seed": metadata["graph_seed"],
                "steps": int(args.steps),
                **stats,
            })
            # 每完成一张图就保存，长时间运行时可直接查看已完成的结果。
            _write_csv(output_dir / "m2_episode_results.csv", all_runs)
            _write_csv(output_dir / "m2_graph_gini_summary.csv", graph_summaries)
            _write_csv(output_dir / "episode_graph_metadata.csv", graph_metadata)
            print(
                f"[{index}/{len(tasks)} graphs, {len(all_runs)}/{len(tasks) * args.repeats} runs] "
                f"{metadata['topology_label']} graph={metadata['graph_index']} seed={metadata['graph_seed']}: "
                f"{stats['final_gini_mean']:.6f} ± {stats['final_gini_std']:.6f}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    topology_summaries = []
    print(f"Final Gini after {args.steps} steps (mean ± std across all runs of each topology):")
    for topology in topologies:
        runs = [row for row in all_runs if row["topology"] == topology]
        stats = _gini_statistics(runs)
        topology_summaries.append({
            "topology": topology,
            "topology_label": TOPOLOGY_LABELS.get(topology, topology),
            "count_graphs": len({row["graph_index"] for row in runs}),
            "repeats_per_graph": int(args.repeats),
            "steps": int(args.steps),
            **stats,
        })
        print(
            f"  {TOPOLOGY_LABELS.get(topology, topology)}: "
            f"{stats['final_gini_mean']:.6f} ± {stats['final_gini_std']:.6f} "
            f"(n={stats['count_runs']})"
        )
    _write_csv(output_dir / "m2_topology_gini_summary.csv", topology_summaries)
    summary = {
        "policy": "M2",
        "policy_key": policy_key,
        "run_dir": str(run_dir),
        "m2_summary_json": str(m2_summary_json),
        "m2_profile_summary": profile_summary,
        "m2_ego_resource_x_mode": "absolute",
        "m2_degree_selection": "Saved sender-degree intervals; Q4 fallback only outside the fitted degree range.",
        "m2_selection_reference": "0429_rule_allocation_over_time_ba_degree_conditioned_discrete/compute.py:_resolve_degree_profile",
        "graph_metadata_source": str(graph_metadata_csv) if graph_metadata_csv else "all_cooperation_Gini.py embedded references",
        "topologies": topologies,
        "count_graphs": len(tasks),
        "repeats_per_graph": int(args.repeats),
        "steps": int(args.steps),
        "total_runs": len(all_runs),
        "seed_base": seed_base,
        "episode_seed_method": "SeedSequence([seed_base, topology_seed_offset, graph_index, repeat_index]) -> uint64",
        "num_workers": int(args.num_workers),
        "effective_dynamics": tasks[0]["spec"]["dynamics"],
        "statistic": "Gini immediately after the final environment update; no time averaging.",
        "std_ddof": 0,
        "topology_summary_scope": "All final Gini observations pooled across graphs and repeats within each topology.",
        "topology_summaries": topology_summaries,
        "graph_summaries": graph_summaries,
        "output_files": {
            "episode_results_csv": str(output_dir / "m2_episode_results.csv"),
            "graph_summary_csv": str(output_dir / "m2_graph_gini_summary.csv"),
            "topology_summary_csv": str(output_dir / "m2_topology_gini_summary.csv"),
            "graph_metadata_csv": str(output_dir / "episode_graph_metadata.csv"),
        },
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
