from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("numpy is required to export the publication figure.") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("matplotlib is required to export the publication figure.") from exc


SCRIPT_DEFAULTS = {
    "input_csv": (
        r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic"
        r"\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
        r"\0421_rule_allocation_over_time_ba_degree_conditioned_discrete\degree_conditioned_profile.csv"
    ),
    "output_stem": (
        r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\GNN_SPGG\outputs\Pool_dynamic"
        r"\0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher"
        r"\0429_rule_allocation_over_time_ba_degree_conditioned_discrete\degree_conditioned_profile_publication"
    ),
    "title": "",
    "x_label": r"$\bar{R}_{ego}(t)$",
    "y_label": "Mixture weight",
    "fig_width": 12.0,
    "fig_height": 3.5,
    "title_fontsize": 13,
    "panel_title_fontsize": 13,
    "axis_label_fontsize": 11.0,
    "tick_label_fontsize": 9.5,
    "legend_fontsize": 9.5,
    "legend_unit_investment_label": "Proportional",
    "legend_equal_label": "Equal",
    "legend_self_label": "Self",
    "legend_loc": "upper center",
    "legend_bbox_to_anchor": (0.59, 0.86),
    "legend_ncol": 1,
    "legend_handlelength": 2.4,
    "panel_label_fontsize": 11.5,
    "show_panel_labels": False,
    "dpi": 600,
    "unit_investment_color": "#1E468B",
    "equal_color": "#E56051",
    "self_color": "#9467BC",
    "grid_color": "#D9D9D9",
    "spine_color": "#000000",
    "text_color": "#000000",
}

_RESOLVED_TIMES_FONT_NAME: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-ready degree-conditioned mixture-weight figure."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SCRIPT_DEFAULTS["input_csv"]),
        help="Path to degree_conditioned_profile.csv.",
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


def _resolve_times_new_roman_font() -> str:
    global _RESOLVED_TIMES_FONT_NAME

    if _RESOLVED_TIMES_FONT_NAME is not None:
        return _RESOLVED_TIMES_FONT_NAME

    candidate_names = (
        "Times New Roman",
        "TimesNewRoman",
        "Times New Roman PS MT",
    )
    for candidate_name in candidate_names:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=[candidate_name]),
                fallback_to_default=False,
            )
            _RESOLVED_TIMES_FONT_NAME = candidate_name
            return _RESOLVED_TIMES_FONT_NAME
        except Exception:
            continue

    available_fonts = sorted({entry.name for entry in font_manager.fontManager.ttflist})
    raise RuntimeError(
        "Times New Roman was not found by matplotlib. "
        f"Available font count: {len(available_fonts)}."
    )


def _configure_publication_style() -> None:
    serif_font = _resolve_times_new_roman_font()
    rc_params: dict[str, Any] = {
        "font.family": serif_font,
        "font.serif": [serif_font],
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
    tick_labelsize: float,
    title_fontsize: float,
    text_color: str,
) -> None:
    serif_font = _resolve_times_new_roman_font()
    xlabel = axis.xaxis.get_label()
    ylabel = axis.yaxis.get_label()
    title = axis.title
    xlabel.set_fontfamily(serif_font)
    ylabel.set_fontfamily(serif_font)
    title.set_fontfamily(serif_font)
    xlabel.set_fontweight("bold")
    ylabel.set_fontweight("bold")
    title.set_fontweight("bold")
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
    serif_font = _resolve_times_new_roman_font()
    for text in legend.get_texts():
        text.set_fontfamily(serif_font)
        text.set_fontweight("bold")
        text.set_fontsize(fontsize)
        text.set_color(text_color)


def _resolve_output_targets(output_arg: str) -> list[Path]:
    output_path = _coerce_path(output_arg)
    if output_path.suffix.lower() in {".png", ".pdf", ".svg"}:
        return [output_path]
    return [output_path.with_suffix(".png"), output_path.with_suffix(".pdf")]


def _parse_float(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    required = {
        "row_kind",
        "ego_mean_resource",
        "weight_unit_investment",
        "weight_equal",
        "weight_self",
        "sender_degree_bin",
        "sender_degree_bin_index",
        "sender_degree_bin_left",
        "sender_degree_bin_right",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed = dict(row)
        parsed["ego_mean_resource"] = _parse_float(row.get("ego_mean_resource"))
        parsed["weight_unit_investment"] = _parse_float(row.get("weight_unit_investment"))
        parsed["weight_equal"] = _parse_float(row.get("weight_equal"))
        parsed["weight_self"] = _parse_float(row.get("weight_self"))
        parsed["sender_degree_bin_index"] = int(float(row.get("sender_degree_bin_index", 0)))
        parsed["sender_degree_bin_left"] = _parse_float(row.get("sender_degree_bin_left"))
        parsed["sender_degree_bin_right"] = _parse_float(row.get("sender_degree_bin_right"))
        parsed_rows.append(parsed)
    return parsed_rows


def _format_degree_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    rounded = int(round(float(value)))
    if abs(float(value) - float(rounded)) <= 1e-9:
        return str(rounded)
    return f"{float(value):g}"


def _panel_title(rows: list[dict[str, Any]]) -> str:
    example = rows[0]
    return str(example.get("sender_degree_bin", "Degree Bin"))


def _group_rows_by_degree_bin(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["sender_degree_bin_index"]), []).append(dict(row))
    return [sorted(grouped[key], key=lambda item: float(item["ego_mean_resource"])) for key in sorted(grouped)]


def render_figure(
    grouped_rows: list[list[dict[str, Any]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    output_targets: list[Path],
    dpi: int,
) -> None:
    _configure_publication_style()

    fig_width = float(SCRIPT_DEFAULTS["fig_width"])
    fig_height = float(SCRIPT_DEFAULTS["fig_height"])
    title_fontsize = float(SCRIPT_DEFAULTS["title_fontsize"])
    panel_title_fontsize = float(SCRIPT_DEFAULTS["panel_title_fontsize"])
    axis_label_fontsize = float(SCRIPT_DEFAULTS["axis_label_fontsize"])
    tick_label_fontsize = float(SCRIPT_DEFAULTS["tick_label_fontsize"])
    legend_fontsize = float(SCRIPT_DEFAULTS["legend_fontsize"])
    legend_unit_investment_label = str(SCRIPT_DEFAULTS["legend_unit_investment_label"])
    legend_equal_label = str(SCRIPT_DEFAULTS["legend_equal_label"])
    legend_self_label = str(SCRIPT_DEFAULTS["legend_self_label"])
    legend_loc = str(SCRIPT_DEFAULTS["legend_loc"])
    legend_bbox_to_anchor_raw = SCRIPT_DEFAULTS["legend_bbox_to_anchor"]
    legend_ncol = max(1, int(SCRIPT_DEFAULTS["legend_ncol"]))
    legend_handlelength = float(SCRIPT_DEFAULTS["legend_handlelength"])
    panel_label_fontsize = float(SCRIPT_DEFAULTS["panel_label_fontsize"])
    show_panel_labels = bool(SCRIPT_DEFAULTS["show_panel_labels"])
    unit_investment_color = str(SCRIPT_DEFAULTS["unit_investment_color"])
    equal_color = str(SCRIPT_DEFAULTS["equal_color"])
    self_color = str(SCRIPT_DEFAULTS["self_color"])
    grid_color = str(SCRIPT_DEFAULTS["grid_color"])
    spine_color = str(SCRIPT_DEFAULTS["spine_color"])
    text_color = str(SCRIPT_DEFAULTS["text_color"])

    component_specs = [
        ("weight_unit_investment", unit_investment_color, legend_unit_investment_label),
        ("weight_equal", equal_color, legend_equal_label),
        ("weight_self", self_color, legend_self_label),
    ]
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    if not isinstance(legend_bbox_to_anchor_raw, (list, tuple)) or len(legend_bbox_to_anchor_raw) != 2:
        raise ValueError("SCRIPT_DEFAULTS['legend_bbox_to_anchor'] must be a 2-item tuple/list like (0.5, 1.02).")
    legend_bbox_to_anchor = (
        float(legend_bbox_to_anchor_raw[0]),
        float(legend_bbox_to_anchor_raw[1]),
    )

    figure, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height), constrained_layout=True, sharey=True)
    axes_array = np.asarray(axes).reshape(-1)
    serif_font = _resolve_times_new_roman_font()

    for axis_index, axis in enumerate(axes_array):
        if axis_index >= len(grouped_rows):
            axis.set_axis_off()
            continue

        panel_rows = grouped_rows[axis_index]
        reference_rows = [row for row in panel_rows if str(row.get("row_kind", "")) == "reference_bin"]
        curve_rows = [
            row for row in panel_rows if str(row.get("row_kind", "")) in {"interpolated_curve", "fitted_curve"}
        ]
        panel_x_values = [
            float(row["ego_mean_resource"])
            for row in panel_rows
            if np.isfinite(float(row["ego_mean_resource"]))
        ]
        if not panel_x_values:
            raise ValueError(f"No finite ego_mean_resource values found for panel index {axis_index}.")
        panel_x_min = min(panel_x_values)
        panel_x_max = max(panel_x_values)
        panel_x_pad = max(1.0, 0.02 * max(1.0, panel_x_max - panel_x_min))

        for component_index, (key, color, label) in enumerate(component_specs):
            curve_x = np.asarray([float(row["ego_mean_resource"]) for row in curve_rows], dtype=np.float64)
            curve_y = np.asarray([float(row[key]) for row in curve_rows], dtype=np.float64)
            reference_x = np.asarray([float(row["ego_mean_resource"]) for row in reference_rows], dtype=np.float64)
            reference_y = np.asarray([float(row[key]) for row in reference_rows], dtype=np.float64)
            show_label = label if axis_index == 0 else None

            if reference_rows and np.any(np.isfinite(reference_y)):
                axis.scatter(
                    reference_x,
                    reference_y,
                    s=42,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.1,
                    alpha=0.35,
                    zorder=2,
                )

            if curve_rows and np.any(np.isfinite(curve_y)):
                axis.plot(
                    curve_x,
                    curve_y,
                    color=color,
                    linewidth=2.4,
                    drawstyle="steps-mid",
                    label=show_label,
                    zorder=3 + component_index,
                )

        axis.set_title(_panel_title(panel_rows), pad=9)
        axis.set_xlim(panel_x_min - panel_x_pad, panel_x_max + panel_x_pad)
        axis.set_ylim(-0.02, 1.02)
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.xaxis.set_major_locator(MultipleLocator(50.0))
        axis.grid(axis="y", color=grid_color, linewidth=0.8, alpha=0.9)
        axis.tick_params(axis="both", labelsize=tick_label_fontsize, colors=text_color)
        axis.tick_params(axis="x", labelbottom=True)
        _style_axis_text(
            axis,
            axis_label_fontsize=axis_label_fontsize,
            tick_labelsize=tick_label_fontsize,
            title_fontsize=panel_title_fontsize,
            text_color=text_color,
        )
        for spine_name in ("top", "right", "left", "bottom"):
            axis.spines[spine_name].set_visible(True)
            axis.spines[spine_name].set_color(spine_color)
            axis.spines[spine_name].set_linewidth(0.8)
        if show_panel_labels and axis_index < len(panel_labels):
            axis.text(
                0.0,
                1.015,
                panel_labels[axis_index],
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=panel_label_fontsize,
                fontweight="bold",
                fontfamily=serif_font,
                color=text_color,
            )

    supxlabel = figure.supxlabel(x_label, fontsize=axis_label_fontsize, fontweight="bold", color=text_color)
    supylabel = figure.supylabel(y_label, fontsize=axis_label_fontsize, fontweight="bold", color=text_color)
    supxlabel.set_fontfamily(serif_font)
    supylabel.set_fontfamily(serif_font)

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        legend = figure.legend(
            handles,
            labels,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=legend_ncol,
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=legend_handlelength,
        )
        _style_legend_text(legend, fontsize=legend_fontsize, text_color=text_color)

    if title.strip():
        suptitle = figure.suptitle(title, fontsize=title_fontsize, color=text_color, fontweight="bold", y=1.07)
        suptitle.set_fontfamily(serif_font)

    for output_path in output_targets:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.08}
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
    grouped_rows = _group_rows_by_degree_bin(rows)
    output_targets = _resolve_output_targets(args.output)
    render_figure(
        grouped_rows,
        title=str(args.title),
        x_label=str(args.x_label),
        y_label=str(args.y_label),
        output_targets=output_targets,
        dpi=int(args.dpi),
    )

    print(f"Resolved input CSV: {input_csv}")
    print(f"Saved publication figure to: {', '.join(str(path) for path in output_targets)}")


if __name__ == "__main__":
    main()
