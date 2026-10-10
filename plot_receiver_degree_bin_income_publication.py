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


PHASE_ORDER = ("early", "mid", "late")

# Edit these defaults directly if you prefer not to pass CLI arguments.
SCRIPT_DEFAULTS = {
    # 输入的 receiver_degree_bin_income.csv。
    "input_csv": (
        r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic"
        r"\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
        r"\0420_total_regu_ba_BA_policy_mechanism\scale_free\receiver_degree_bin_income.csv"
    ),

    # 输出文件路径“前缀”。
    # 脚本会自动导出：
    # - <output_stem>.png
    # - <output_stem>.pdf
        "output_stem": (
        r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic"
        r"\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
        r"\0821_total_regu_ba_BA_policy_mechanism\scale_free\receiver_degree_bin_income_publication"
    ),

    # "output_stem": (
    #     r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic"
    #     r"\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
    #     r"\0429_total_regu_ba_BA_policy_mechanism\scale_free\receiver_degree_bin_income_publication"
    # ),

    # 图标题；如果正文里不想放标题，保持空字符串。
    "title": "",

    # 三个 panel 的标题。
    "early_panel_title": "First 20%",
    "mid_panel_title": "Middle 60%",
    "late_panel_title": "Last 20%",

    # 坐标轴名称。
    "x_label": "Receiver degree bin",
    "y_label": "Incoming resource ratio",

    # legend 文案。
    "equal_ratio_label": "Observed / Equal baseline",
    "labor_ratio_label": "Observed / Proportional baseline",

    # 只画哪个 flow_scope。
    # - "all"：包含 self-flow
    # - "excluding_self"：排除 self-flow
    "flow_scope": "all",

    # 是否从同目录下的 node_income_decomposition.csv 计算并绘制
    # episode-level SEM 阴影带。先在每个 episode 内汇总一个阶段，再跨
    # episode 计算 sample SD / sqrt(n)，避免把相邻 step 当作独立重复。
    "show_sem": False,
    "sem_alpha": 0.18,
    "sem_errorbar_capsize": 4.0,
    "sem_errorbar_linewidth": 1.2,

    # 图片比例。
    "fig_width": 12.0,
    "fig_height": 4.0,

    # 字号。
    "title_fontsize": 14,
    "axis_label_fontsize": 13,
    "tick_label_fontsize": 12,
    "legend_fontsize": 11,
    "panel_title_fontsize": 13,

    # legend 位置与列数。
    "legend_loc": "upper center",
    "legend_ncol": 1,
    "legend_bbox_x": 0.52,
    "legend_bbox_y": 0.95,

    # PNG 分辨率。
    "dpi": 600,

    # 配色。正文图用低饱和、非刺眼的双色。
    "equal_ratio_color": "#E46051",
    "labor_ratio_color": "#1E468A",
    "reference_line_color": "#8A8A8A",
    "grid_color": "#E6E6E6",
    "spine_color": "#000000",
    "text_color": "#000000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-ready receiver-degree incoming-resource ratio figure."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SCRIPT_DEFAULTS["input_csv"]),
        help="Path to receiver_degree_bin_income.csv.",
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
        "--y-label",
        type=str,
        default=str(SCRIPT_DEFAULTS["y_label"]),
        help="Shared y-axis label.",
    )
    parser.add_argument(
        "--flow-scope",
        type=str,
        default=str(SCRIPT_DEFAULTS["flow_scope"]),
        choices=("all", "excluding_self"),
        help="Which flow_scope rows to plot.",
    )
    parser.add_argument(
        "--sem-input",
        type=str,
        default=None,
        help=(
            "Path to node_income_decomposition.csv used for episode-level SEM. "
            "By default it is inferred from --input."
        ),
    )
    sem_group = parser.add_mutually_exclusive_group()
    sem_group.add_argument(
        "--show-sem",
        dest="show_sem",
        action="store_true",
        help="Draw mean +/- one episode-level SEM bands (default).",
    )
    sem_group.add_argument(
        "--no-sem",
        dest="show_sem",
        action="store_false",
        help="Do not draw SEM bands.",
    )
    parser.set_defaults(show_sem=bool(SCRIPT_DEFAULTS["show_sem"]))
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


def _style_axis_text(
    axis: Any,
    *,
    axis_label_fontsize: float,
    tick_labelsize: float,
    title_fontsize: float,
    title_fontweight: str,
    text_color: str,
) -> None:
    serif_font = _resolve_serif_font()
    xlabel = axis.xaxis.get_label()
    ylabel = axis.yaxis.get_label()
    title = axis.title
    xlabel.set_fontfamily(serif_font)
    ylabel.set_fontfamily(serif_font)
    title.set_fontfamily(serif_font)
    xlabel.set_fontweight("bold")
    ylabel.set_fontweight("bold")
    title.set_fontweight(title_fontweight)
    xlabel.set_fontsize(axis_label_fontsize)
    ylabel.set_fontsize(axis_label_fontsize)
    title.set_fontsize(title_fontsize)
    xlabel.set_color(text_color)
    ylabel.set_color(text_color)
    title.set_color(text_color)
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontfamily(serif_font)
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(tick_labelsize)
        tick_label.set_color(text_color)


def _style_legend_text(legend: Any, *, fontsize: float, text_color: str) -> None:
    if legend is None:
        return
    serif_font = _resolve_serif_font()
    for text in legend.get_texts():
        text.set_fontfamily(serif_font)
        text.set_fontweight("bold")
        text.set_fontsize(fontsize)
        text.set_color(text_color)
    legend_title = legend.get_title()
    if legend_title is not None:
        legend_title.set_fontfamily(serif_font)
        legend_title.set_fontweight("bold")
        legend_title.set_fontsize(fontsize)
        legend_title.set_color(text_color)


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    required = {
        "time_phase",
        "flow_scope",
        "receiver_degree_bin",
        "receiver_degree_bin_index",
        "actual_over_equal_mean",
        "actual_over_labor_mean",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return rows


def _select_plot_rows(rows: list[dict[str, Any]], *, flow_scope: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for phase_name in PHASE_ORDER:
        phase_rows = [
            dict(row)
            for row in rows
            if str(row.get("time_phase", "")) == phase_name and str(row.get("flow_scope", "")) == flow_scope
        ]
        for row in phase_rows:
            row["receiver_degree_bin_index"] = int(float(row["receiver_degree_bin_index"]))
            row["actual_over_equal_mean"] = float(row["actual_over_equal_mean"])
            row["actual_over_labor_mean"] = float(row["actual_over_labor_mean"])
        phase_rows.sort(key=lambda row: int(row["receiver_degree_bin_index"]))
        grouped[phase_name] = phase_rows
    if not any(grouped.values()):
        raise ValueError(f"No rows found for flow_scope={flow_scope!r}.")
    return grouped


def _infer_sem_csv_path(input_csv: Path, *, flow_scope: str) -> Path:
    del flow_scope  # Both flow scopes are present as separate columns in the raw file.
    return input_csv.with_name("node_income_decomposition.csv")


def _load_sem_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _attach_episode_sem(
    grouped_rows: dict[str, list[dict[str, Any]]],
    node_rows: list[dict[str, Any]],
    *,
    flow_scope: str,
) -> int:
    """Attach SEM across episode-level phase means; return the minimum episode n."""
    metric_sources = {
        "actual_over_equal_mean": (
            "actual_over_equal_excluding_self" if flow_scope == "excluding_self" else "actual_over_equal"
        ),
        "actual_over_labor_mean": (
            "actual_over_labor_excluding_self" if flow_scope == "excluding_self" else "actual_over_labor"
        ),
    }
    required = {
        "episode",
        "time_phase",
        "receiver_degree",
        *metric_sources.values(),
    }
    missing = required.difference(node_rows[0].keys())
    if missing:
        raise ValueError(f"SEM input is missing required columns: {sorted(missing)}")

    bin_ranges = {
        phase_name: [
            (
                int(row["receiver_degree_bin_index"]),
                float(row["receiver_degree_min"]),
                float(row["receiver_degree_max"]),
            )
            for row in phase_rows
        ]
        for phase_name, phase_rows in grouped_rows.items()
    }
    episode_values: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    for row in node_rows:
        phase_name = str(row.get("time_phase", ""))
        if phase_name not in PHASE_ORDER:
            continue
        degree = float(row["receiver_degree"])
        bin_index = next(
            (
                candidate_index
                for candidate_index, left, right in bin_ranges.get(phase_name, [])
                if left <= degree <= right
            ),
            None,
        )
        if bin_index is None:
            continue
        key = (
            phase_name,
            int(bin_index),
            int(float(row["episode"])),
        )
        destination = episode_values.setdefault(
            key,
            {metric: [] for metric in metric_sources},
        )
        for metric, source_column in metric_sources.items():
            value = float(row[source_column])
            if np.isfinite(value):
                destination[metric].append(value)

    episode_means: dict[tuple[str, int], dict[str, list[float]]] = {}
    for (phase_name, bin_index, _episode), values_by_metric in episode_values.items():
        destination = episode_means.setdefault(
            (phase_name, bin_index),
            {metric: [] for metric in metric_sources},
        )
        for metric, raw_values in values_by_metric.items():
            if raw_values:
                destination[metric].append(float(np.mean(raw_values)))

    sample_sizes: list[int] = []
    for phase_name, phase_rows in grouped_rows.items():
        for row in phase_rows:
            bin_index = int(row["receiver_degree_bin_index"])
            metric_values = episode_means.get((phase_name, bin_index), {})
            for metric in metric_sources:
                values = np.asarray(metric_values.get(metric, []), dtype=np.float64)
                values = values[np.isfinite(values)]
                sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size >= 2 else float("nan")
                row[f"{metric}_sem"] = sem
                row[f"{metric}_sem_n"] = int(values.size)
                if values.size > 0:
                    sample_sizes.append(int(values.size))
    return min(sample_sizes) if sample_sizes else 0


def _resolve_output_targets(output_arg: str) -> list[Path]:
    output_path = _coerce_path(output_arg)
    if output_path.suffix.lower() in {".png", ".pdf", ".svg"}:
        return [output_path]
    return [output_path.with_suffix(".png"), output_path.with_suffix(".pdf")]


def render_figure(
    grouped_rows: dict[str, list[dict[str, Any]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    output_targets: list[Path],
    dpi: int,
    sem_episode_n: int = 0,
) -> None:
    _configure_publication_style()

    fig_width = float(SCRIPT_DEFAULTS["fig_width"])
    fig_height = float(SCRIPT_DEFAULTS["fig_height"])
    title_fontsize = float(SCRIPT_DEFAULTS["title_fontsize"])
    panel_title_fontsize = float(SCRIPT_DEFAULTS["panel_title_fontsize"])
    axis_label_fontsize = float(SCRIPT_DEFAULTS["axis_label_fontsize"])
    tick_label_fontsize = float(SCRIPT_DEFAULTS["tick_label_fontsize"])
    legend_fontsize = float(SCRIPT_DEFAULTS["legend_fontsize"])
    legend_loc = str(SCRIPT_DEFAULTS["legend_loc"])
    legend_ncol = int(SCRIPT_DEFAULTS["legend_ncol"])
    legend_bbox_x = float(SCRIPT_DEFAULTS["legend_bbox_x"])
    legend_bbox_y = float(SCRIPT_DEFAULTS["legend_bbox_y"])
    panel_title_fontweight = "normal"
    panel_titles = {
        "early": str(SCRIPT_DEFAULTS["early_panel_title"]),
        "mid": str(SCRIPT_DEFAULTS["mid_panel_title"]),
        "late": str(SCRIPT_DEFAULTS["late_panel_title"]),
    }
    equal_ratio_label = str(SCRIPT_DEFAULTS["equal_ratio_label"])
    labor_ratio_label = str(SCRIPT_DEFAULTS["labor_ratio_label"])
    equal_ratio_color = str(SCRIPT_DEFAULTS["equal_ratio_color"])
    labor_ratio_color = str(SCRIPT_DEFAULTS["labor_ratio_color"])
    reference_line_color = str(SCRIPT_DEFAULTS["reference_line_color"])
    grid_color = str(SCRIPT_DEFAULTS["grid_color"])
    spine_color = str(SCRIPT_DEFAULTS["spine_color"])
    text_color = str(SCRIPT_DEFAULTS["text_color"])
    sem_alpha = float(SCRIPT_DEFAULTS["sem_alpha"])
    sem_errorbar_capsize = float(SCRIPT_DEFAULTS["sem_errorbar_capsize"])
    sem_errorbar_linewidth = float(SCRIPT_DEFAULTS["sem_errorbar_linewidth"])

    figure, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), constrained_layout=True, sharey=True)
    axes_array = np.atleast_1d(axes).reshape(-1)

    global_ymax = 1.0
    for phase_name in PHASE_ORDER:
        for row in grouped_rows.get(phase_name, []):
            global_ymax = max(
                global_ymax,
                float(row["actual_over_equal_mean"])
                + (float(row.get("actual_over_equal_mean_sem", 0.0)) if np.isfinite(float(row.get("actual_over_equal_mean_sem", 0.0))) else 0.0),
                float(row["actual_over_labor_mean"])
                + (float(row.get("actual_over_labor_mean_sem", 0.0)) if np.isfinite(float(row.get("actual_over_labor_mean_sem", 0.0))) else 0.0),
            )
    y_upper = global_ymax * 1.08 if global_ymax > 0.0 else 1.1

    for axis, phase_name in zip(axes_array, PHASE_ORDER):
        phase_rows = grouped_rows.get(phase_name, [])
        if not phase_rows:
            axis.set_axis_off()
            continue

        x_labels = [str(row["receiver_degree_bin"]) for row in phase_rows]
        x_values = np.arange(len(x_labels), dtype=np.int64)
        over_equal = np.asarray([float(row["actual_over_equal_mean"]) for row in phase_rows], dtype=np.float64)
        over_labor = np.asarray([float(row["actual_over_labor_mean"]) for row in phase_rows], dtype=np.float64)
        equal_sem = np.asarray(
            [float(row.get("actual_over_equal_mean_sem", float("nan"))) for row in phase_rows],
            dtype=np.float64,
        )
        labor_sem = np.asarray(
            [float(row.get("actual_over_labor_mean_sem", float("nan"))) for row in phase_rows],
            dtype=np.float64,
        )

        if np.any(np.isfinite(equal_sem)):
            equal_lower = over_equal - equal_sem
            equal_upper = over_equal + equal_sem
            equal_where = np.isfinite(x_values) & np.isfinite(equal_lower) & np.isfinite(equal_upper)
            axis.fill_between(
                x_values,
                equal_lower,
                equal_upper,
                where=equal_where,
                color=equal_ratio_color,
                alpha=sem_alpha,
                linewidth=0.0,
            )
        if np.any(np.isfinite(labor_sem)):
            labor_lower = over_labor - labor_sem
            labor_upper = over_labor + labor_sem
            labor_where = np.isfinite(x_values) & np.isfinite(labor_lower) & np.isfinite(labor_upper)
            axis.fill_between(
                x_values,
                labor_lower,
                labor_upper,
                where=labor_where,
                color=labor_ratio_color,
                alpha=sem_alpha,
                linewidth=0.0,
            )

        equal_yerr = np.where(np.isfinite(equal_sem), equal_sem, 0.0) if np.any(np.isfinite(equal_sem)) else None
        labor_yerr = np.where(np.isfinite(labor_sem), labor_sem, 0.0) if np.any(np.isfinite(labor_sem)) else None
        axis.errorbar(
            x_values,
            over_equal,
            yerr=equal_yerr,
            marker="o",
            markersize=4.2,
            linewidth=2.0,
            color=equal_ratio_color,
            ecolor=equal_ratio_color,
            elinewidth=sem_errorbar_linewidth,
            capsize=sem_errorbar_capsize,
            capthick=sem_errorbar_linewidth,
            label=equal_ratio_label,
            zorder=3,
        )
        axis.errorbar(
            x_values,
            over_labor,
            yerr=labor_yerr,
            marker="o",
            markersize=4.2,
            linewidth=2.0,
            color=labor_ratio_color,
            ecolor=labor_ratio_color,
            elinewidth=sem_errorbar_linewidth,
            capsize=sem_errorbar_capsize,
            capthick=sem_errorbar_linewidth,
            label=labor_ratio_label,
            zorder=3,
        )
        axis.axhline(1.0, color=reference_line_color, linewidth=1.0, linestyle="--", alpha=0.95)
        axis.set_title(panel_titles.get(phase_name, phase_name), pad=8)
        axis.set_xticks(x_values)
        axis.set_xticklabels(x_labels, rotation=0)
        axis.set_xlabel(x_label if phase_name == "mid" else "")
        axis.set_ylim(0.0, y_upper)
        axis.grid(axis="y", color=grid_color, linewidth=0.8)
        axis.tick_params(axis="both", labelsize=tick_label_fontsize, colors=text_color)
        _style_axis_text(
            axis,
            axis_label_fontsize=axis_label_fontsize,
            tick_labelsize=tick_label_fontsize,
            title_fontsize=panel_title_fontsize,
            title_fontweight=panel_title_fontweight,
            text_color=text_color,
        )
        for spine_name in ("top", "right", "left", "bottom"):
            axis.spines[spine_name].set_visible(True)
            axis.spines[spine_name].set_color(spine_color)
            axis.spines[spine_name].set_linewidth(0.8)

    axes_array[0].set_ylabel(y_label, fontsize=axis_label_fontsize, color=text_color, fontweight="bold")

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        legend = figure.legend(
            handles,
            labels,
            title=(
                f"Mean +/- 1 SEM (n={sem_episode_n} episodes)"
                if sem_episode_n > 0
                else None
            ),
            loc=legend_loc,
            bbox_to_anchor=(legend_bbox_x, legend_bbox_y),
            ncol=max(1, legend_ncol),
            frameon=False,
            fontsize=legend_fontsize,
        )
        _style_legend_text(legend, fontsize=legend_fontsize, text_color=text_color)

    if title.strip():
        suptitle = figure.suptitle(title, fontsize=title_fontsize, color=text_color, fontweight="bold", y=1.06)
        suptitle.set_fontfamily(_resolve_serif_font())

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

    rows = _load_rows(input_csv)
    grouped_rows = _select_plot_rows(rows, flow_scope=str(args.flow_scope))
    sem_csv: Path | None = None
    sem_episode_n = 0
    if bool(args.show_sem):
        sem_csv = (
            _coerce_path(args.sem_input)
            if args.sem_input is not None
            else _infer_sem_csv_path(input_csv, flow_scope=str(args.flow_scope))
        )
        if not sem_csv.exists():
            raise FileNotFoundError(
                f"SEM input CSV not found: {sem_csv}. Pass --no-sem to render without uncertainty bands."
            )
        sem_episode_n = _attach_episode_sem(
            grouped_rows,
            _load_sem_rows(sem_csv),
            flow_scope=str(args.flow_scope),
        )
    output_targets = _resolve_output_targets(args.output)
    render_figure(
        grouped_rows,
        title=str(args.title),
        x_label=str(args.x_label),
        y_label=str(args.y_label),
        output_targets=output_targets,
        dpi=int(args.dpi),
        sem_episode_n=sem_episode_n,
    )

    print(f"Resolved input CSV: {input_csv}")
    print(f"Flow scope: {args.flow_scope}")
    if sem_csv is not None:
        print(f"SEM input CSV: {sem_csv}")
        print(f"SEM unit: episode (minimum n={sem_episode_n})")
    print(f"Saved publication figure to: {', '.join(str(path) for path in output_targets)}")


if __name__ == "__main__":
    main()
