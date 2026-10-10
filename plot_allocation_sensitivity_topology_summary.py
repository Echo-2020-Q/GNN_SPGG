from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("matplotlib is required to export the topology summary figure.") from exc

#分析总得特征重要性/敏感度，用于放在附录里面
FEATURE_ORDER = (
    "resource_norm",
    "strategy_norm",
    "pool_grown",
    "degree_norm",
)

FEATURE_LABELS = {
    "resource_norm": r"$R_i(t)$",
    "strategy_norm": r"$e_i(t)$",
    "pool_grown": r"$G_i(t)$",
    "degree_norm": r"$d_i$",
}

TOPOLOGY_ORDER = (
    "regular",
    "erdos_renyi",
    "small_world",
    "scale_free",
)

TOPOLOGY_LABELS = {
    "regular": "Regular",
    "erdos_renyi": "ER",
    "small_world": "WS",
    "scale_free": "BA",
}

# Edit these defaults directly if you prefer not to pass CLI arguments.
SCRIPT_DEFAULTS = {
    # 输入可以写成：
    # 1) policy_analysis 目录；
    # 2) best_eval 目录；
    # 3) topology_feature_importance.csv 的完整路径。
    # 如果给的是目录，脚本会自动向下找 topology_feature_importance.csv。
    "input_path": r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher\policy_analysis",

    # 输出文件路径“前缀”。
    # 脚本会自动导出：
    # - <output_stem>.png
    # - <output_stem>.pdf
    "output_stem": r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher\0428_policy_analysis_summary\best_eval\allocation_sensitivity_topology_aggregate",

    # 图总标题；如果不想要标题，改成空字符串 ""。
    "title": "Feature Sensitivity of the Allocation Policy (Multi-Topology)",

    # 横轴标题。
    "x_label": "Mean absolute change in allocation",

    # 误差线模式：
    # - "std"：显示跨 topology 的标准差；
    # - "sem"：显示标准误；
    # - "none"：不显示误差线。
    "error_bar_mode": "std",

    # PNG 分辨率。
    "dpi": 600,

    # 论文图配色。保持单一低饱和主色，避免过花。
    "bar_fill_color": "#57BE7A",
    "bar_edge_color": "#3BA75F",
    "grid_color": "#E0E0E0",
    "spine_color": "#000000",
    "text_color": "#000000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a paper-ready multi-topology Allocation Sensitivity summary figure."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SCRIPT_DEFAULTS["input_path"]),
        help="policy_analysis directory, best_eval directory, or topology_feature_importance.csv path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DEFAULTS["output_stem"]),
        help="Output file or output stem.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=str(SCRIPT_DEFAULTS["title"]),
        help="Figure title. Pass an empty string to omit it.",
    )
    parser.add_argument(
        "--x-label",
        type=str,
        default=str(SCRIPT_DEFAULTS["x_label"]),
        help="Shared x-axis label.",
    )
    parser.add_argument(
        "--error-bar-mode",
        type=str,
        default=str(SCRIPT_DEFAULTS["error_bar_mode"]),
        choices=("std", "sem", "none"),
        help="How to summarize cross-topology variability for the aggregated figure.",
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


def _resolve_serif_font() -> str:
    available_fonts = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in ("Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"):
        if candidate in available_fonts:
            return candidate
    return "DejaVu Serif"


def _configure_publication_style() -> None:
    serif_font = _resolve_serif_font()
    rc_params: dict[str, Any] = {
        "font.family": serif_font,
        "font.serif": [serif_font, "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.weight": "bold",
        "axes.unicode_minus": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": serif_font,
        "mathtext.it": f"{serif_font}:italic",
        "mathtext.bf": f"{serif_font}:bold",
        "mathtext.cal": f"{serif_font}:italic",
    }
    plt.rcParams.update(rc_params)


def _style_axis_text(
    axis: Any,
    *,
    axis_label_fontsize: float,
    x_tick_labelsize: float,
    y_tick_labelsize: float,
) -> None:
    serif_font = _resolve_serif_font()
    xlabel = axis.xaxis.get_label()
    ylabel = axis.yaxis.get_label()
    xlabel.set_fontfamily(serif_font)
    ylabel.set_fontfamily(serif_font)
    xlabel.set_fontweight("bold")
    ylabel.set_fontweight("bold")
    xlabel.set_fontsize(axis_label_fontsize)
    ylabel.set_fontsize(axis_label_fontsize)
    for tick_label in axis.get_xticklabels():
        tick_label.set_fontfamily(serif_font)
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(x_tick_labelsize)
    for tick_label in axis.get_yticklabels():
        tick_label.set_fontfamily(serif_font)
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(y_tick_labelsize)


def _resolve_input_csv(input_arg: str) -> Path:
    candidate = _coerce_path(input_arg)
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Input path not found: {candidate}")

    direct_csv = candidate / "topology_feature_importance.csv"
    if direct_csv.exists():
        return direct_csv

    recursive_matches = sorted(candidate.glob("**/topology_feature_importance.csv"))
    if len(recursive_matches) == 1:
        return recursive_matches[0]
    if len(recursive_matches) > 1:
        best_eval_matches = [path for path in recursive_matches if path.parent.name == "best_eval"]
        if len(best_eval_matches) == 1:
            return best_eval_matches[0]
        raise RuntimeError(
            "Multiple topology_feature_importance.csv files found. Please pass the exact CSV path explicitly."
        )

    raise FileNotFoundError(
        f"Could not find topology_feature_importance.csv under directory: {candidate}"
    )


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    if "feature" not in rows[0] or "topology" not in rows[0] or "allocation_l1_mean" not in rows[0]:
        raise ValueError(f"{csv_path} does not look like a topology_feature_importance.csv file.")
    return rows


def _aggregate_feature_rows(rows: list[dict[str, Any]], error_bar_mode: str) -> tuple[list[dict[str, Any]], list[str]]:
    aggregated_rows: list[dict[str, Any]] = []
    included_topologies: list[str] = []

    for topology_name in TOPOLOGY_ORDER:
        if any(str(row.get("topology")) == topology_name for row in rows):
            included_topologies.append(topology_name)

    for feature_key in FEATURE_ORDER:
        feature_values: list[float] = []
        feature_topologies: list[str] = []
        for topology_name in TOPOLOGY_ORDER:
            match = next(
                (
                    row
                    for row in rows
                    if str(row.get("topology")) == topology_name and str(row.get("feature")) == feature_key
                ),
                None,
            )
            if match is None:
                continue
            feature_values.append(float(match["allocation_l1_mean"]))
            feature_topologies.append(topology_name)

        if not feature_values:
            continue

        values_array = np.asarray(feature_values, dtype=np.float64)
        std_value = float(np.std(values_array, ddof=0)) if values_array.size > 1 else 0.0
        sem_value = float(std_value / np.sqrt(values_array.size)) if values_array.size > 0 else 0.0
        error_value = 0.0
        if error_bar_mode == "std":
            error_value = std_value
        elif error_bar_mode == "sem":
            error_value = sem_value

        aggregated_rows.append(
            {
                "feature": feature_key,
                "paper_label": FEATURE_LABELS[feature_key],
                "allocation_l1_mean": float(np.mean(values_array)),
                "allocation_l1_std": std_value,
                "allocation_l1_sem": sem_value,
                "allocation_l1_min": float(np.min(values_array)),
                "allocation_l1_max": float(np.max(values_array)),
                "allocation_l1_error": error_value,
                "topology_count": int(values_array.size),
                "topologies": ",".join(feature_topologies),
            }
        )

    if not aggregated_rows:
        raise ValueError("No matching topology/feature rows were found in the input CSV.")
    return aggregated_rows, included_topologies


def _resolve_output_stem(output_arg: str) -> Path:
    output_path = _coerce_path(output_arg)
    if output_path.suffix.lower() in {".png", ".pdf", ".svg"}:
        return output_path.with_suffix("")
    return output_path


def _resolve_output_targets(output_stem: Path) -> list[Path]:
    return [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]


def _write_summary_csv(output_stem: Path, rows: list[dict[str, Any]]) -> Path:
    output_csv = output_stem.with_suffix(".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature",
        "paper_label",
        "allocation_l1_mean",
        "allocation_l1_std",
        "allocation_l1_sem",
        "allocation_l1_min",
        "allocation_l1_max",
        "topology_count",
        "topologies",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    return output_csv


def render_figure(
    rows: list[dict[str, Any]],
    title: str,
    x_label: str,
    output_targets: list[Path],
    dpi: int,
) -> None:
    _configure_publication_style()

    bar_fill_color = str(SCRIPT_DEFAULTS["bar_fill_color"])
    bar_edge_color = str(SCRIPT_DEFAULTS["bar_edge_color"])
    grid_color = str(SCRIPT_DEFAULTS["grid_color"])
    spine_color = str(SCRIPT_DEFAULTS["spine_color"])
    text_color = str(SCRIPT_DEFAULTS["text_color"])

    global_xmax = max(float(row["allocation_l1_mean"]) + float(row["allocation_l1_error"]) for row in rows)
    x_padding = global_xmax * 0.12 if global_xmax > 0.0 else 1.0
    labels = [str(row["paper_label"]) for row in rows]
    scores = [float(row["allocation_l1_mean"]) for row in rows]
    errors = [float(row["allocation_l1_error"]) for row in rows]
    positions = list(range(len(labels)))

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.barh(
        positions,
        scores,
        xerr=errors,
        color=bar_fill_color,
        edgecolor=bar_edge_color,
        linewidth=0.9,
        height=0.62,
        ecolor="#6F6F6F",
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=13, color=text_color, fontweight="bold")
    axis.invert_yaxis()
    axis.set_xlabel(x_label, fontsize=13, color=text_color, fontweight="bold")
    if title.strip():
        axis.set_title(title, fontsize=14, color=text_color, pad=10)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=grid_color, linewidth=0.8)
    axis.tick_params(axis="x", labelsize=12, colors=text_color)
    axis.tick_params(axis="y", length=0)
    _style_axis_text(axis, axis_label_fontsize=13, x_tick_labelsize=12, y_tick_labelsize=13)
    axis.set_xlim(0.0, global_xmax + x_padding)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(spine_color)
    axis.spines["bottom"].set_color(spine_color)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)

    for output_path in output_targets:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if output_path.suffix.lower() == ".png":
            save_kwargs["dpi"] = max(72, int(dpi))
        figure.savefig(output_path, **save_kwargs)

    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_csv = _resolve_input_csv(args.input)
    raw_rows = _load_rows(input_csv)
    aggregated_rows, included_topologies = _aggregate_feature_rows(raw_rows, error_bar_mode=args.error_bar_mode)
    output_stem = _resolve_output_stem(args.output)
    output_targets = _resolve_output_targets(output_stem)
    summary_csv = _write_summary_csv(output_stem, aggregated_rows)
    render_figure(
        aggregated_rows,
        title=args.title,
        x_label=args.x_label,
        output_targets=output_targets,
        dpi=int(args.dpi),
    )

    print(f"Resolved input CSV: {input_csv}")
    print(f"Saved aggregated summary CSV to: {summary_csv}")
    print(f"Saved topology summary figure to: {', '.join(str(path) for path in output_targets)}")
    print("Included topologies:", ", ".join(included_topologies))


if __name__ == "__main__":
    main()
