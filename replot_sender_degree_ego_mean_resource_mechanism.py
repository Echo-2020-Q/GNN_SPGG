from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_gnn_policy_decisions import write_csv
from analyze_gnn_policy_mechanism import (
    _read_csv_rows,
    _to_float,
    _to_int,
    plot_sender_degree_ego_resource_mechanism,
    summarize_sender_degree_bin_mechanism,
    summarize_sender_degree_ego_resource_mechanism,
)


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _ensure_ego_mean_resource(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        current = row.get("ego_mean_resource")
        if current not in (None, ""):
            enriched.append(updated)
            continue
        ego_total = _to_float(row.get("ego_total_resources", float("nan")))
        receiver_count = max(_to_int(row.get("receiver_count", 0)), 1)
        updated["ego_mean_resource"] = (
            float(ego_total / float(receiver_count)) if ego_total == ego_total else float("nan")
        )
        enriched.append(updated)
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild sender_degree_ego_mean_resource_mechanism outputs from existing CSVs without rerunning rollouts."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Existing per-topology mechanism analysis directory that already contains row_mechanism.csv.",
    )
    parser.add_argument(
        "--ego-resource-bin-count",
        type=int,
        default=12,
        help="How many quantile bins to use for ego_mean_resource. Typical values are 10 to 20.",
    )
    parser.add_argument(
        "--tail-trim-frac",
        type=float,
        default=0.05,
        help="Trim this fraction from both the low and high tails of ego_mean_resource within each sender-degree bin before rebining. Use 0 to disable.",
    )
    parser.add_argument(
        "--degree-bin-mode",
        type=str,
        choices=("quantile", "log"),
        default=None,
        help="Override sender degree bin mode. Defaults to the value stored in mechanism_summary.json.",
    )
    parser.add_argument(
        "--degree-bin-count",
        type=int,
        default=None,
        help="Override sender degree bin count. Defaults to the value stored in mechanism_summary.json.",
    )
    parser.add_argument(
        "--degree-bin-min-count",
        type=int,
        default=None,
        help="Override sender degree min count. Defaults to the value stored in mechanism_summary.json.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite sender_degree_ego_mean_resource_mechanism.(csv|png) instead of writing *_binsN variants.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    row_path = analysis_dir / "row_mechanism.csv"
    if not row_path.exists():
        raise FileNotFoundError(f"row_mechanism.csv not found under {analysis_dir}")

    summary = _load_summary(analysis_dir / "mechanism_summary.json")
    degree_summary = summary.get("degree_analysis", {}) if isinstance(summary, dict) else {}
    topology_name = str(summary.get("topology_name", analysis_dir.name))
    if topology_name == "all_topologies":
        topology_name = "_all_topologies"

    degree_bin_mode = (
        str(args.degree_bin_mode)
        if args.degree_bin_mode is not None
        else str(degree_summary.get("degree_bin_mode", "quantile"))
    )
    degree_bin_count = (
        int(args.degree_bin_count)
        if args.degree_bin_count is not None
        else max(1, _to_int(degree_summary.get("degree_bin_count", 4)))
    )
    degree_bin_min_count = (
        int(args.degree_bin_min_count)
        if args.degree_bin_min_count is not None
        else max(1, _to_int(degree_summary.get("degree_bin_min_count", 100)))
    )
    ego_resource_bin_count = max(1, int(args.ego_resource_bin_count))
    tail_trim_frac = min(max(float(args.tail_trim_frac), 0.0), 0.49)

    row_records = _ensure_ego_mean_resource(_read_csv_rows(row_path))
    if not row_records:
        raise RuntimeError(f"No rows found in {row_path}")

    _, _, sender_spec, _ = summarize_sender_degree_bin_mechanism(
        row_records,
        topology_name=topology_name,
        degree_bin_mode=degree_bin_mode,
        degree_bin_count=degree_bin_count,
        degree_bin_min_count=degree_bin_min_count,
    )
    output_rows = summarize_sender_degree_ego_resource_mechanism(
        row_records,
        topology_name=topology_name,
        sender_spec=sender_spec,
        ego_resource_bin_count=ego_resource_bin_count,
        tail_trim_frac=tail_trim_frac,
    )

    if args.overwrite_output:
        csv_path = analysis_dir / "sender_degree_ego_mean_resource_mechanism.csv"
        png_path = analysis_dir / "sender_degree_ego_mean_resource_mechanism.png"
    else:
        suffix = (
            f"bins{ego_resource_bin_count}_trim{int(round(tail_trim_frac * 100.0))}"
            if tail_trim_frac > 0.0
            else f"bins{ego_resource_bin_count}"
        )
        csv_path = analysis_dir / f"sender_degree_ego_mean_resource_mechanism_{suffix}.csv"
        png_path = analysis_dir / f"sender_degree_ego_mean_resource_mechanism_{suffix}.png"

    write_csv(csv_path, output_rows)
    plot_sender_degree_ego_resource_mechanism(output_rows, png_path)

    print(f"Rebuilt sender-degree × ego-mean-resource mechanism CSV: {csv_path}")
    print(f"Rebuilt sender-degree × ego-mean-resource mechanism plot: {png_path}")


if __name__ == "__main__":
    main()
