from __future__ import annotations

"""
Rebuild steady-state summary plots from an existing scan_summary.json.

Usage:
1. Edit REDRAW_CONFIG below if you prefer configuring this file directly.
2. Run:
       python redraw_scan_steady_state.py

Optional command-line overrides:
    python redraw_scan_steady_state.py --root-dir outputs/r_network_consumption_strategy_scan
    python redraw_scan_steady_state.py --summary-json outputs/my_scan/scan_summary.json --dpi 200
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from Project1.visualization import save_scan_metric_grid


# =============================================================================
# Direct file configuration
# =============================================================================
#
# Typical usage:
# - Only edit REDRAW_CONFIG
# - Then run: python redraw_scan_steady_state.py
#
# Notes:
# - summary_json = None means automatically use <root_dir>/scan_summary.json
# - Empty filter lists mean "do not filter"
#
REDRAW_CONFIG = {
    # Scan output root directory.
    "root_dir": Path("outputs/r_network_consumption_strategy_scan"),

    # If not None, read this file directly instead of <root_dir>/scan_summary.json.
    "summary_json": None,

    # Output figure DPI.
    "dpi": 160,

    # Which steady-state metrics to draw in each aggregated figure.
    "metrics": [
        "final_actual_cooperation_mean",
        "final_mean_resource_mean",
        "final_mean_pool_grown_mean",
        "final_mean_consumption_mean",
        "final_mean_payoff_mean",
        "final_gini_mean",
    ],

    # Optional filters. Use [] to keep all records.
    "run_modes": [],
    "strategy_update_rules": [],
    "consumption_labels": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw steady-state summary plots from an existing scan summary file."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("outputs/r_network_consumption_strategy_scan"),
        help="Root directory of the scan outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to scan_summary.json. Defaults to <root-dir>/scan_summary.json.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output DPI for regenerated figures.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(REDRAW_CONFIG)
    config["root_dir"] = Path(args.root_dir) if args.root_dir is not None else Path(config["root_dir"])
    config["summary_json"] = (
        Path(args.summary_json)
        if args.summary_json is not None
        else config["summary_json"]
    )
    config["dpi"] = int(args.dpi) if args.dpi is not None else int(config["dpi"])
    config["metrics"] = list(config["metrics"])
    config["run_modes"] = list(config["run_modes"])
    config["strategy_update_rules"] = list(config["strategy_update_rules"])
    config["consumption_labels"] = list(config["consumption_labels"])
    return config


def load_scan_records(summary_json_path: Path) -> List[Dict[str, Any]]:
    if not summary_json_path.exists():
        raise FileNotFoundError("scan summary file not found: {0}".format(summary_json_path))

    records = json.loads(summary_json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("scan summary json must contain a list of records.")
    return [dict(record) for record in records]


def filter_scan_records(
    records: Sequence[Mapping[str, Any]],
    run_modes: Sequence[str],
    strategy_update_rules: Sequence[str],
    consumption_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    filtered_records: List[Dict[str, Any]] = []
    run_mode_set = set(run_modes)
    strategy_set = set(strategy_update_rules)
    consumption_set = set(consumption_labels)

    for record in records:
        if run_mode_set and str(record["run_mode"]) not in run_mode_set:
            continue
        if strategy_set and str(record["strategy_update_rule"]) not in strategy_set:
            continue
        if consumption_set and str(record["consumption_label"]) not in consumption_set:
            continue
        filtered_records.append(dict(record))

    return filtered_records


def sort_scan_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda item: (
            str(item["run_mode"]),
            str(item["strategy_update_rule"]),
            str(item["consumption_label"]),
            str(item["network_label"]),
            float(item["r"]),
        ),
    )


def redraw_steady_state_plots(
    output_root: Path,
    scan_records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    dpi: int,
) -> None:
    grouped_records: Dict[tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for record in scan_records:
        group_key = (
            str(record["run_mode"]),
            str(record["strategy_update_rule"]),
            str(record["consumption_label"]),
        )
        grouped_records.setdefault(group_key, []).append(record)

    steady_state_dir = output_root / "steady_state_vs_r"
    steady_state_dir.mkdir(parents=True, exist_ok=True)

    for (run_mode, strategy_update_rule, consumption_label), records in grouped_records.items():
        output_path = steady_state_dir / "{0}__{1}__{2}__steady_state_vs_r.png".format(
            run_mode,
            strategy_update_rule,
            consumption_label,
        )
        title = "Steady-state vs r | run_mode={0} | strategy={1} | consumption={2}".format(
            run_mode,
            strategy_update_rule,
            consumption_label,
        )
        save_scan_metric_grid(
            records=records,
            output_path=output_path,
            metrics=metrics,
            title=title,
            dpi=dpi,
        )
        print("Saved: {0}".format(output_path))


def main() -> None:
    config = resolve_config(parse_args())
    output_root = Path(config["root_dir"])
    summary_json_path = (
        Path(config["summary_json"])
        if config["summary_json"] is not None
        else (output_root / "scan_summary.json")
    )

    scan_records = load_scan_records(summary_json_path)
    filtered_records = filter_scan_records(
        scan_records,
        run_modes=config["run_modes"],
        strategy_update_rules=config["strategy_update_rules"],
        consumption_labels=config["consumption_labels"],
    )
    sorted_records = sort_scan_records(filtered_records)
    if not sorted_records:
        raise ValueError("No scan records matched the current REDRAW_CONFIG filters.")

    print("Root dir      : {0}".format(output_root))
    print("Summary json  : {0}".format(summary_json_path))
    print("Matched plots : {0}".format(len(sorted_records)))
    redraw_steady_state_plots(
        output_root=output_root,
        scan_records=sorted_records,
        metrics=config["metrics"],
        dpi=int(config["dpi"]),
    )


if __name__ == "__main__":
    main()
