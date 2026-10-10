from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

import analyze_rule_allocation_over_time as ar


SCRIPT_DEFAULTS = {
    # 输入既可以是原始的 agent_mix_vs_ego_mean.csv，
    # 也可以是 adaptive_unitinv_equal_w_vs_ego_mean_resource.csv。
    "input_csv": (
        r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic"
        r"\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
        r"\0419_regu_ba_sender_allocation_gini_over_time\regular\agent_mix_vs_ego_mean.csv"
    ),
    # 输出目录。
    "output_dir": (
        r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic"
        r"\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
        r"\0429_binned_unitinv_equal_profile\regular"
    ),
    # 输出文件名前缀。
    "output_stem": "binned_unitinv_equal_profile",
    # 资源横轴解释方式。
    "ego_resource_x_mode": "absolute",
    "ego_resource_normalizer": None,
    # Ego_Resource_Mean 在 443.18 处做硬截断；超过这个值的一律按 443.18 处理。
    "tail_cap_mode": "hard",
    "tail_cap_resource": 443.18,
    "tail_cap_soft_scale": 0.25,
    # 图标题。
    "title": "Discrete Binned Unit-Investment + Equal Profile",
    # PNG 分辨率。
    "dpi": 220,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert agent_mix/adaptive_unitinv_equal profile CSV into a saved discrete-bin "
            "Unit-Investment + Equal profile with a step-plot artifact."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(SCRIPT_DEFAULTS["input_csv"]),
        help="Path to agent_mix_vs_ego_mean.csv or adaptive_unitinv_equal_w_vs_ego_mean_resource.csv.",
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
    return parser.parse_args()


def _coerce_path(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
    windows_match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if windows_match and os.name != "nt":
        drive = windows_match.group(1).lower()
        suffix = windows_match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{suffix}").expanduser().resolve()
    return Path(text).expanduser().resolve()


def _load_rows(csv_path: Path) -> tuple[list[dict[str, Any]], str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    headers = set(rows[0].keys())
    if {"bin_left", "bin_right", "weight_unit_investment_mean", "weight_equal_mean"}.issubset(headers):
        return rows, "agent_mix_summary"
    if {"row_kind", "bin_left", "bin_right", "weight_unit_investment", "weight_equal"}.issubset(headers):
        return rows, "adaptive_profile"
    raise ValueError(f"Unsupported CSV schema for {csv_path}")


def _profile_x_value(
    resource_value: float,
    *,
    x_mode: str,
    normalizer: float | None,
    tail_cap_mode: str,
    tail_cap_resource: float | None,
    tail_cap_soft_scale: float,
) -> float:
    resource_array = np.asarray([float(resource_value)], dtype=np.float64)
    effective_value = float(
        ar._apply_tail_cap_to_resources(
            resource_array,
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
        )[0]
    )
    return float(
        ar._normalize_ego_resource_value(
            effective_value,
            x_mode=str(x_mode),
            normalizer=normalizer,
        )
    )


def _normalize_source_rows(
    rows: list[dict[str, Any]],
    *,
    source_format: str,
    ego_resource_x_mode: str,
    ego_resource_normalizer: float | None,
    tail_cap_mode: str,
    tail_cap_resource: float | None,
    tail_cap_soft_scale: float,
    profile_source: str,
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []

    if source_format == "agent_mix_summary":
        source_rows = rows
    else:
        source_rows = [row for row in rows if str(row.get("row_kind", "")) == "reference_bin"]
        if not source_rows:
            raise ValueError("No row_kind=reference_bin rows found in adaptive profile CSV.")

    for order_index, row in enumerate(source_rows):
        raw_left = float(row["bin_left"])
        raw_right = float(row["bin_right"])
        raw_center = (
            float(row["bin_center"])
            if row.get("bin_center") not in {None, ""}
            else 0.5 * (raw_left + raw_right)
        )
        raw_count = int(float(row.get("sample_count", 0) or 0))
        if source_format == "agent_mix_summary":
            weight_unit = float(row["weight_unit_investment_mean"])
            weight_equal = float(row["weight_equal_mean"])
            fit_l1 = (
                float(row["mixture_fit_l1_mean"])
                if row.get("mixture_fit_l1_mean") not in {None, ""}
                else float("nan")
            )
        else:
            weight_unit = float(row["weight_unit_investment"])
            weight_equal = float(row["weight_equal"])
            fit_l1 = (
                float(row["reference_fit_l1_mean"])
                if row.get("reference_fit_l1_mean") not in {None, ""}
                else float("nan")
            )

        profile_left = _profile_x_value(
            raw_left,
            x_mode=ego_resource_x_mode,
            normalizer=ego_resource_normalizer,
            tail_cap_mode=tail_cap_mode,
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=tail_cap_soft_scale,
        )
        profile_right = _profile_x_value(
            raw_right,
            x_mode=ego_resource_x_mode,
            normalizer=ego_resource_normalizer,
            tail_cap_mode=tail_cap_mode,
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=tail_cap_soft_scale,
        )
        profile_center = _profile_x_value(
            raw_center,
            x_mode=ego_resource_x_mode,
            normalizer=ego_resource_normalizer,
            tail_cap_mode=tail_cap_mode,
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=tail_cap_soft_scale,
        )

        normalized_rows.append(
            {
                "row_kind": "reference_bin",
                "order_index": int(order_index),
                "ego_mean_resource": float(raw_center),
                "ego_mean_resource_effective": float(
                    ar._apply_tail_cap_to_resources(
                        np.asarray([raw_center], dtype=np.float64),
                        tail_cap_mode=str(tail_cap_mode),
                        tail_cap_resource=tail_cap_resource,
                        tail_cap_soft_scale=float(tail_cap_soft_scale),
                    )[0]
                ),
                "ego_resource_x": float(profile_center),
                "ego_resource_x_mode": str(ego_resource_x_mode),
                "ego_resource_normalizer": (
                    None if ego_resource_normalizer is None else float(ego_resource_normalizer)
                ),
                "log_ego_mean_resource": float(np.log1p(max(0.0, profile_center))),
                "weight_unit_investment": float(weight_unit),
                "weight_equal": float(weight_equal),
                "weight_self": 0.0,
                "bin_left": float(profile_left),
                "bin_right": float(profile_right),
                "raw_bin_left": float(raw_left),
                "raw_bin_right": float(raw_right),
                "raw_bin_center": float(raw_center),
                "sample_count": int(raw_count),
                "reference_fit_l1_mean": float(fit_l1),
                "profile_source": str(profile_source),
            }
        )

    normalized_rows.sort(key=lambda row: float(row["bin_left"]))
    return normalized_rows


def _build_policy_from_rows(
    rows: list[dict[str, Any]],
    *,
    tail_cap_mode: str,
    tail_cap_resource: float | None,
    tail_cap_soft_scale: float,
    ego_resource_x_mode: str,
    ego_resource_normalizer: float | None,
    profile_source: str,
) -> tuple[ar.BinnedEgoResourceMixtureAllocationPolicy, dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("No normalized rows were produced for the discrete profile.")

    cleaned_edges = [float(rows[0]["bin_left"])]
    cleaned_weight_rows: list[list[float]] = []
    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        right_value = float(row["bin_right"])
        if right_value < cleaned_edges[-1] - 1e-12:
            raise ValueError("Discrete profile bin_right values must be non-decreasing.")
        if right_value <= cleaned_edges[-1] + 1e-12:
            continue
        left_value = float(cleaned_edges[-1])
        row_copy = dict(row)
        row_copy["order_index"] = int(len(cleaned_rows))
        row_copy["bin_left"] = float(left_value)
        row_copy["bin_right"] = float(right_value)
        row_copy["ego_resource_x"] = float(
            np.clip(float(row_copy["ego_resource_x"]), left_value, right_value)
        )
        cleaned_edges.append(float(right_value))
        cleaned_weight_rows.append(
            [
                float(row_copy["weight_unit_investment"]),
                float(row_copy["weight_equal"]),
                0.0,
            ]
        )
        cleaned_rows.append(row_copy)
    if len(cleaned_edges) < 2:
        raise ValueError("Discrete profile requires at least one non-empty bin interval.")

    fit_summary = {
        "fit_status": "imported_reference_bins_2way",
        "adaptive_ego_resource_profile_mode": "discrete",
        "adaptive_ego_resource_bin_count": int(len(cleaned_weight_rows)),
        "ego_resource_x_mode": str(ego_resource_x_mode),
        "ego_resource_normalizer": None if ego_resource_normalizer is None else float(ego_resource_normalizer),
        "tail_cap_mode": str(tail_cap_mode),
        "tail_cap_resource": None if tail_cap_resource is None else float(tail_cap_resource),
        "tail_cap_soft_scale": float(tail_cap_soft_scale),
        "source_reference_row_count": int(len(rows)),
        "collapsed_zero_width_row_count": int(len(rows) - len(cleaned_rows)),
        "weight_simplex_l1_mean": (
            float(np.nanmean(np.asarray([float(row["reference_fit_l1_mean"]) for row in cleaned_rows], dtype=np.float64)))
            if cleaned_rows
            else float("nan")
        ),
        "reference_row_fit_l1_mean": (
            float(np.nanmean(np.asarray([float(row["reference_fit_l1_mean"]) for row in cleaned_rows], dtype=np.float64)))
            if cleaned_rows
            else float("nan")
        ),
        "bin_rows_summary": [
            {
                "bin_index": int(index),
                "bin_left": float(row["bin_left"]),
                "bin_right": float(row["bin_right"]),
                "raw_bin_left": float(row["raw_bin_left"]),
                "raw_bin_right": float(row["raw_bin_right"]),
                "sample_count": int(row["sample_count"]),
                "weight_unit_investment": float(row["weight_unit_investment"]),
                "weight_equal": float(row["weight_equal"]),
                "weight_self": 0.0,
            }
            for index, row in enumerate(cleaned_rows)
        ],
        "bin_edges_resource": [float(value) for value in cleaned_edges],
    }

    policy = ar.BinnedEgoResourceMixtureAllocationPolicy(
        mode="binned_3way",
        bin_edges_resource=cleaned_edges,
        bin_weight_rows=cleaned_weight_rows,
        profile_source=str(profile_source),
        fit_summary=fit_summary,
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
        ego_resource_x_mode=str(ego_resource_x_mode),
        ego_resource_normalizer=ego_resource_normalizer,
    )
    return policy, fit_summary, cleaned_rows


def _plot_discrete_profile(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> bool:
    if ar.plt is None or not rows:
        return False

    x_label = ar._ego_resource_x_label(
        str(rows[0].get("ego_resource_x_mode", "absolute")),
        rows[0].get("ego_resource_normalizer"),
    )
    figure, axis = ar.plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    bin_edges = np.asarray(
        [float(rows[0]["bin_left"])] + [float(row["bin_right"]) for row in rows],
        dtype=np.float64,
    )
    centers = np.asarray([float(row["ego_resource_x"]) for row in rows], dtype=np.float64)
    sample_counts = np.asarray([int(row["sample_count"]) for row in rows], dtype=np.int64)
    alpha_scale = np.clip(sample_counts / float(max(int(np.max(sample_counts)), 1)), 0.35, 1.0)

    component_specs = [
        ("weight_unit_investment", "#8c564b", "Unit-Investment"),
        ("weight_equal", "#1f77ff", "Equal"),
    ]

    for edge_value in bin_edges[1:-1]:
        axis.axvline(float(edge_value), color="#d1d5db", linewidth=0.8, alpha=0.8, zorder=0)

    for key, color, label in component_specs:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
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

    axis.set_xlabel(x_label)
    axis.set_ylabel("Mixture Weight")
    axis.set_ylim(-0.02, 1.02)
    if title.strip():
        axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=max(72, int(dpi)), bbox_inches="tight", facecolor="white")
    ar.plt.close(figure)
    return True


def _write_summary_json(
    output_path: Path,
    *,
    input_csv: Path,
    output_dir: Path,
    output_stem: str,
    source_format: str,
    policy_summary: Mapping[str, Any],
) -> None:
    payload = {
        "policy_name": "saved_binned_unitinv_equal_mix",
        "policy_label": "Saved Binned UnitInv + Equal",
        "profile_type": "binned_unitinv_equal",
        "source_csv": str(input_csv),
        "source_format": str(source_format),
        "output_dir": str(output_dir),
        "output_stem": str(output_stem),
        "policy_summary": dict(policy_summary),
    }
    ar._write_json(output_path, payload)


def main() -> None:
    args = parse_args()
    input_csv = _coerce_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if str(args.ego_resource_x_mode) == "normalized":
        if args.ego_resource_normalizer is None or float(args.ego_resource_normalizer) <= 0.0:
            raise ValueError("--ego-resource-normalizer must be positive when --ego-resource-x-mode=normalized.")

    output_dir = _coerce_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = str(args.output_stem).strip()
    if not output_stem:
        raise ValueError("--output-stem cannot be empty.")

    raw_rows, source_format = _load_rows(input_csv)
    normalized_rows = _normalize_source_rows(
        raw_rows,
        source_format=source_format,
        ego_resource_x_mode=str(args.ego_resource_x_mode),
        ego_resource_normalizer=(
            None if args.ego_resource_normalizer is None else float(args.ego_resource_normalizer)
        ),
        tail_cap_mode=str(args.tail_cap_mode),
        tail_cap_resource=None if args.tail_cap_resource is None else float(args.tail_cap_resource),
        tail_cap_soft_scale=float(args.tail_cap_soft_scale),
        profile_source=f"{input_csv}",
    )
    policy, fit_summary, effective_rows = _build_policy_from_rows(
        normalized_rows,
        tail_cap_mode=str(args.tail_cap_mode),
        tail_cap_resource=None if args.tail_cap_resource is None else float(args.tail_cap_resource),
        tail_cap_soft_scale=float(args.tail_cap_soft_scale),
        ego_resource_x_mode=str(args.ego_resource_x_mode),
        ego_resource_normalizer=(
            None if args.ego_resource_normalizer is None else float(args.ego_resource_normalizer)
        ),
        profile_source=f"{input_csv}",
    )

    profile_csv_path = output_dir / f"{output_stem}.csv"
    profile_png_path = output_dir / f"{output_stem}.png"
    profile_json_path = output_dir / f"{output_stem}_summary.json"

    ar.write_csv(profile_csv_path, effective_rows)
    png_written = _plot_discrete_profile(effective_rows, profile_png_path, title=str(args.title), dpi=int(args.dpi))
    _write_summary_json(
        profile_json_path,
        input_csv=input_csv,
        output_dir=output_dir,
        output_stem=output_stem,
        source_format=source_format,
        policy_summary=policy.profile_summary(),
    )

    print(f"Resolved input CSV: {input_csv}")
    print(f"Detected input format: {source_format}")
    print(f"Saved discrete profile CSV to: {profile_csv_path}")
    if png_written and profile_png_path.exists():
        print(f"Saved discrete profile PNG to: {profile_png_path}")
    else:
        print("Skipped discrete profile PNG because matplotlib is not available in the current interpreter.")
    print(f"Saved discrete profile summary JSON to: {profile_json_path}")
    print(f"Tail cap: mode={args.tail_cap_mode}, resource={args.tail_cap_resource}")
    print(f"Source reference row count: {fit_summary['source_reference_row_count']}")
    print(f"Collapsed zero-width rows: {fit_summary['collapsed_zero_width_row_count']}")
    print(f"Discrete bin count: {fit_summary['adaptive_ego_resource_bin_count']}")


if __name__ == "__main__":
    main()
