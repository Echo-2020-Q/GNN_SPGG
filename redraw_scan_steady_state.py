from __future__ import annotations

"""
Rebuild steady-state summary plots from an existing scan_summary.json.

Usage:
    python redraw_scan_steady_state.py
    python redraw_scan_steady_state.py --root-dir outputs/r_network_consumption_strategy_scan
    python redraw_scan_steady_state.py --summary-json outputs/my_scan/scan_summary.json --dpi 200
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from Project1.visualization import save_scan_metric_grid


STEADY_STATE_METRICS = [
    "final_actual_cooperation_mean",
    "final_mean_resource_mean",
    "final_mean_pool_grown_mean",
    "final_mean_consumption_mean",
    "final_mean_payoff_mean",
    "final_gini_mean",
]


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


def load_scan_records(summary_json_path: Path) -> List[Dict[str, Any]]:
    if not summary_json_path.exists():
        raise FileNotFoundError("scan summary file not found: {0}".format(summary_json_path))

    records = json.loads(summary_json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("scan summary json must contain a list of records.")
    return [dict(record) for record in records]


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
            metrics=STEADY_STATE_METRICS,
            title=title,
            dpi=dpi,
        )
        print("Saved: {0}".format(output_path))


def main() -> None:
    args = parse_args()
    output_root = args.root_dir
    summary_json_path = args.summary_json or (output_root / "scan_summary.json")

    scan_records = load_scan_records(summary_json_path)
    sorted_records = sort_scan_records(scan_records)
    redraw_steady_state_plots(output_root=output_root, scan_records=sorted_records, dpi=args.dpi)


if __name__ == "__main__":
    main()
