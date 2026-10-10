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
    raise SystemExit("matplotlib is required to export the publication figure.") from exc


POLICY_ORDER = ("equal", "proportional", "agent")

# Edit these defaults directly if you prefer not to pass CLI arguments.
SCRIPT_DEFAULTS = {
    # 输入的资源干预汇总表。
    "input_csv": r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher\0418_regu_ba_2Wlength_policy_mechanism\regular\resource_intervention_summary.csv",

    # 输出文件路径“前缀”。
    # 脚本会自动导出：
    # - <output_stem>.png
    # - <output_stem>.pdf
    "output_stem": r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher\0428_policy_analysis_200length\regular\allocation_gini_vs_resource_intervention_publication",

    # 图标题；如果正文里不想单独放标题，改成空字符串 ""。
    "title": "",

    # 坐标轴名称。
    "x_label": r"Manipulated resource as a fraction of original value",
    "y_label": "Allocation Gini Coefficient",

    # 字号。和 plot_allocation_sensitivity_publication.py 保持一致。
    "title_fontsize": 14,
    "axis_label_fontsize": 12,
    "tick_label_fontsize": 12,
    "legend_fontsize": 11,

    # 图片比例。
    "fig_width": 4.0,
    "fig_height": 4.0,

    # y 轴下边额外留白，避免 0 紧贴横轴。
    "y_margin_lower": 0.02,
    "y_margin_upper_ratio": 0.06,

    # 误差带展示方式。
    # - "sem": 使用 summary.csv 里的标准误；
    # - "std": 使用 summary.csv 里的标准差；
    # - "none": 不画误差带。
    "band_mode": "sem",

    # PNG 分辨率。
    "dpi": 600,

    # 配色。正文图用低饱和、非刺眼的三色。
    "equal_color":  "#E66051",
    "proportional_color": "#1E468A",
    "agent_color": "#E2A802",
    "grid_color": "#E0E0E0",
    "spine_color": "#000000",
    "text_color": "#000000",
    "reference_line_color": "#8A8A8A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-ready Allocation Gini vs resource intervention figure."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SCRIPT_DEFAULTS["input_csv"]),
        help="Path to resource_intervention_summary.csv.",
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
        help="X-axis label.",
    )
    parser.add_argument(
        "--y-label",
        type=str,
        default=str(SCRIPT_DEFAULTS["y_label"]),
        help="Y-axis label.",
    )
    parser.add_argument(
        "--band-mode",
        type=str,
        default=str(SCRIPT_DEFAULTS["band_mode"]),
        choices=("sem", "std", "none"),
        help="Which uncertainty band to draw.",
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
        "font.family": "serif",
        "font.serif": [serif_font, "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
    }
    if serif_font == "Times New Roman":
        rc_params.update(
            {
                "mathtext.fontset": "custom",
                "mathtext.rm": serif_font,
                "mathtext.it": f"{serif_font}:italic",
                "mathtext.bf": f"{serif_font}:bold",
                "mathtext.cal": f"{serif_font}:italic",
            }
        )
    else:
        rc_params["mathtext.fontset"] = "stix"
    plt.rcParams.update(rc_params)


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    required = {
        "policy",
        "multiplier",
        "allocated_resource_gini_mean",
        "allocated_resource_gini_std",
        "allocated_resource_gini_sem",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for policy_name in POLICY_ORDER:
        policy_rows = [row for row in rows if str(row.get("policy")) == policy_name]
        for row in policy_rows:
            row["multiplier"] = float(row["multiplier"])
            row["allocated_resource_gini_mean"] = float(row["allocated_resource_gini_mean"])
            row["allocated_resource_gini_std"] = float(row["allocated_resource_gini_std"])
            row["allocated_resource_gini_sem"] = float(row["allocated_resource_gini_sem"])
        policy_rows.sort(key=lambda row: float(row["multiplier"]))
        if policy_rows:
            grouped[policy_name] = policy_rows
    if not grouped:
        raise ValueError("No valid policy rows were found in the input CSV.")
    return grouped


def _resolve_output_targets(output_arg: str) -> list[Path]:
    output_path = _coerce_path(output_arg)
    if output_path.suffix.lower() in {".png", ".pdf", ".svg"}:
        return [output_path]
    return [output_path.with_suffix(".png"), output_path.with_suffix(".pdf")]


def render_figure(
    grouped_rows: dict[str, list[dict[str, Any]]],
    title: str,
    x_label: str,
    y_label: str,
    band_mode: str,
    output_targets: list[Path],
    dpi: int,
) -> None:
    _configure_publication_style()

    style_by_policy = {
        "equal": {
            "color": str(SCRIPT_DEFAULTS["equal_color"]),
            "label": "Equal Baseline",
            "linewidth": 1.9,
        },
        "proportional": {
            "color": str(SCRIPT_DEFAULTS["proportional_color"]),
            "label": "Proportional Baseline",
            "linewidth": 1.9,
        },
        "agent": {
            "color": str(SCRIPT_DEFAULTS["agent_color"]),
            "label": "RL Agent",
            "linewidth": 2.1,
        },
    }
    grid_color = str(SCRIPT_DEFAULTS["grid_color"])
    spine_color = str(SCRIPT_DEFAULTS["spine_color"])
    text_color = str(SCRIPT_DEFAULTS["text_color"])
    reference_line_color = str(SCRIPT_DEFAULTS["reference_line_color"])
    title_fontsize = float(SCRIPT_DEFAULTS["title_fontsize"])
    axis_label_fontsize = float(SCRIPT_DEFAULTS["axis_label_fontsize"])
    tick_label_fontsize = float(SCRIPT_DEFAULTS["tick_label_fontsize"])
    legend_fontsize = float(SCRIPT_DEFAULTS["legend_fontsize"])

    all_y_values = np.asarray(
        [float(row["allocated_resource_gini_mean"]) for rows in grouped_rows.values() for row in rows],
        dtype=np.float64,
    )
    y_max = float(np.nanmax(all_y_values)) if all_y_values.size > 0 else 1.0
    y_upper = y_max + max(float(SCRIPT_DEFAULTS["y_margin_lower"]), y_max * float(SCRIPT_DEFAULTS["y_margin_upper_ratio"]))
    y_lower = -float(SCRIPT_DEFAULTS["y_margin_lower"])

    figure, axis = plt.subplots(
        figsize=(float(SCRIPT_DEFAULTS["fig_width"]), float(SCRIPT_DEFAULTS["fig_height"])),
        constrained_layout=True,
    )

    for policy_name in POLICY_ORDER:
        rows = grouped_rows.get(policy_name)
        if not rows:
            continue
        x_values = np.asarray([float(row["multiplier"]) for row in rows], dtype=np.float64)
        y_values = np.asarray([float(row["allocated_resource_gini_mean"]) for row in rows], dtype=np.float64)
        style = style_by_policy[policy_name]
        axis.plot(
            x_values,
            y_values,
            color=style["color"],
            label=style["label"],
            linewidth=float(style["linewidth"]),
            marker="o",
            markersize=3.8,
        )
        if band_mode != "none":
            error_key = "allocated_resource_gini_sem" if band_mode == "sem" else "allocated_resource_gini_std"
            band_values = np.asarray([float(row[error_key]) for row in rows], dtype=np.float64)
            lower = np.clip(y_values - band_values, 0.0, None)
            upper = np.clip(y_values + band_values, 0.0, None)
            mask = np.isfinite(x_values) & np.isfinite(lower) & np.isfinite(upper)
            if int(np.count_nonzero(mask)) >= 2:
                axis.fill_between(
                    x_values,
                    lower,
                    upper,
                    where=mask,
                    color=style["color"],
                    alpha=0.12,
                )

    axis.set_xscale("log")
    axis.axvline(1.0, color=reference_line_color, linestyle="--", linewidth=1.0, alpha=0.95)
    axis.set_xlabel(x_label, fontsize=axis_label_fontsize, color=text_color, fontweight="bold")
    axis.set_ylabel(y_label, fontsize=axis_label_fontsize, color=text_color, fontweight="bold")
    if title.strip():
        axis.set_title(title, fontsize=title_fontsize, color=text_color, pad=8)

    axis.set_ylim(y_lower, y_upper)
    axis.grid(alpha=0.28, which="both", color=grid_color, linewidth=0.8)
    axis.tick_params(axis="both", labelsize=tick_label_fontsize, colors=text_color)
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    for tick_label in list(axis.get_xticklabels()) + list(axis.get_yticklabels()):
        tick_label.set_fontweight("bold")
    legend = axis.legend(frameon=False, fontsize=legend_fontsize, loc="best")
    if legend is not None:
        for legend_text in legend.get_texts():
            legend_text.set_fontweight("bold")

    for spine_name in ("top", "right", "left", "bottom"):
        axis.spines[spine_name].set_visible(True)
        axis.spines[spine_name].set_color(spine_color)
        axis.spines[spine_name].set_linewidth(0.8)

    for output_path in output_targets:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if output_path.suffix.lower() == ".png":
            save_kwargs["dpi"] = max(72, int(dpi))
        figure.savefig(output_path, **save_kwargs)

    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_csv = _coerce_path(args.input)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    raw_rows = _load_rows(input_csv)
    grouped_rows = _group_rows(raw_rows)
    output_targets = _resolve_output_targets(args.output)
    render_figure(
        grouped_rows,
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
        band_mode=args.band_mode,
        output_targets=output_targets,
        dpi=int(args.dpi),
    )

    print(f"Resolved input CSV: {input_csv}")
    print(f"Saved publication figure to: {', '.join(str(path) for path in output_targets)}")
    print("Included policies:", ", ".join(name for name in POLICY_ORDER if name in grouped_rows))


if __name__ == "__main__":
    main()
