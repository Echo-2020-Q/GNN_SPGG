from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_CSV = Path(
    "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
    "0823_20000_sixpolicy_dynamics_grid/policy_rollout_step_metrics.csv"
)

SCRIPT_DEFAULTS = {
    "input_csv": DEFAULT_INPUT_CSV,
    "output_dir": None,
    "output_name": "five_policy_dynamics_grid_PLOT",
    "figsize": (12.0, 12.0),
    "legend_ncol": 1,
    "legend_loc": "upper center",
    "legend_bbox_x": 0.465,
    "legend_bbox_y": 0.957,
    "marker_count": 10,
    "marker_size": 5.5,
    "time_axis_scale": "symlog",
    "metric_axis_scale": "linear",
}

TOPOLOGY_ORDER = ("regular", "scale_free", "erdos_renyi", "small_world")
TOPOLOGY_LABELS = {
    "regular": "Regular",
    "scale_free": "BA",
    "erdos_renyi": "ER",
    "small_world": "WS",
}

POLICY_ORDER = ("adaptive","agent", "equal","prop","m1", "m2")
POLICY_STYLES = {
    "agent": {"color": "#E3A008", "label": "RL-Agent", "linestyle": "-", "marker": "o", "zorder": 3},
    "m1": {"color": "#2E8B57", "label": "M1", "linestyle": "-.", "marker": "s", "zorder": 4},
    "m2": {"color": "#BD93DA", "label": "M2", "linestyle": "--", "marker": "^", "zorder": 5},
    "prop": {"color": "#1E4688", "label": "Prop", "linestyle": ":", "marker": "D", "zorder": 3},
    "equal": {"color": "#E56051", "label": "Equal", "linestyle": (0, (5, 2)), "marker": "X", "zorder": 3},
    "adaptive": {"color": "#008B8B", "label": "Interpolation", "linestyle": "-", "marker": "P", "zorder": 1},
}

STEP_METRIC_SPECS = (
    ("cooperation_rate_mean", r"$f_c$"),
    ("mean_resource_mean", r"$\bar{R}(t)$"),
    ("resource_gini_mean", "Gini"),
)
EPISODE_METRIC_SPECS = (
    ("final_cooperation_rate", r"$f_c$"),
    ("final_mean_resource", r"$\bar{R}(t)$"),
    ("final_gini", "Gini"),
)

METRIC_LIMIT_GROUPS = {
    "cooperation_rate_mean": "cooperation",
    "final_cooperation_rate": "cooperation",
    "mean_resource_mean": "resource",
    "final_mean_resource": "resource",
    "resource_gini_mean": "gini",
    "final_gini": "gini",
}
METRIC_Y_LIMITS = {
    "cooperation": (-0.03, 1.03),
    "resource": (-12.0, 462.0),
    "gini": (-0.03, 1.03),
}
METRIC_Y_TICKS = {
    "cooperation": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "resource": (0.0, 100.0, 200.0, 300.0, 400.0, 450.0),
    "gini": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
}

FONT_FAMILY = "Times New Roman"
FONT_WEIGHT_BOLD = "bold"
FONT_WEIGHT_TITLE = "normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the policy grid from an existing episode_summary.csv or "
            "policy_rollout_step_metrics.csv. This script does not run simulations."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=SCRIPT_DEFAULTS["input_csv"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DEFAULTS["output_dir"],
        help="Defaults to the input CSV directory.",
    )
    parser.add_argument("--output-name", type=str, default=str(SCRIPT_DEFAULTS["output_name"]))
    parser.add_argument(
        "--topologies",
        type=str,
        default=None,
        help="Optional comma-separated topology subset/order; defaults to the order used by the original script.",
    )
    parser.add_argument("--figsize-width", type=float, default=float(SCRIPT_DEFAULTS["figsize"][0]))
    parser.add_argument("--figsize-height", type=float, default=float(SCRIPT_DEFAULTS["figsize"][1]))
    parser.add_argument("--legend-ncol", type=int, default=int(SCRIPT_DEFAULTS["legend_ncol"]))
    parser.add_argument("--legend-loc", type=str, default=str(SCRIPT_DEFAULTS["legend_loc"]))
    parser.add_argument("--legend-bbox-x", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_x"]))
    parser.add_argument("--legend-bbox-y", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_y"]))
    parser.add_argument("--marker-count", type=int, default=int(SCRIPT_DEFAULTS["marker_count"]))
    parser.add_argument("--marker-size", type=float, default=float(SCRIPT_DEFAULTS["marker_size"]))
    parser.add_argument(
        "--time-axis-scale",
        choices=("linear", "symlog", "log"),
        default=str(SCRIPT_DEFAULTS["time_axis_scale"]),
    )
    parser.add_argument(
        "--metric-axis-scale",
        choices=("linear", "symlog", "log"),
        default=str(SCRIPT_DEFAULTS["metric_axis_scale"]),
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


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
        fieldnames = tuple(str(name) for name in reader.fieldnames)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows, fieldnames


def _detect_schema(fieldnames: Sequence[str]) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    names = set(fieldnames)
    shared = {"topology", "policy"}
    step_fields = shared | {"step"} | {item[0] for item in STEP_METRIC_SPECS}
    episode_fields = shared | {"episode"} | {item[0] for item in EPISODE_METRIC_SPECS}
    if step_fields.issubset(names):
        return "step", "step", STEP_METRIC_SPECS
    if episode_fields.issubset(names):
        return "episode", "episode", EPISODE_METRIC_SPECS
    missing_step = sorted(step_fields - names)
    missing_episode = sorted(episode_fields - names)
    raise ValueError(
        "Unsupported CSV schema. Missing step-summary fields: "
        f"{missing_step}; missing episode-summary fields: {missing_episode}."
    )


def _ordered_topologies(rows: Sequence[Mapping[str, str]], requested: str | None) -> list[str]:
    available = {str(row["topology"]).strip() for row in rows if str(row.get("topology", "")).strip()}
    if requested:
        aliases = {"ba": "scale_free", "er": "erdos_renyi", "ws": "small_world"}
        selected = [aliases.get(item.strip().lower(), item.strip().lower()) for item in requested.split(",") if item.strip()]
        missing = [item for item in selected if item not in available]
        if missing:
            raise ValueError(f"Requested topologies are absent from the CSV: {missing}")
        return list(dict.fromkeys(selected))
    ordered = [item for item in TOPOLOGY_ORDER if item in available]
    return ordered + sorted(available - set(ordered))


def _panel_label(index: int) -> str:
    return f"({chr(ord('a') + int(index))})"


def _build_even_markevery(
    x_values: np.ndarray,
    *,
    axis_scale: str,
    target_count: int,
    symlog_linthresh: float = 1.0,
) -> list[int] | int:
    x_array = np.asarray(x_values, dtype=np.float64).reshape(-1)
    if x_array.size <= 1:
        return 1
    safe_target_count = max(2, min(int(target_count), int(x_array.size)))
    if safe_target_count >= x_array.size:
        return list(range(int(x_array.size)))

    if axis_scale == "linear":
        transformed = x_array
    elif axis_scale == "log":
        transformed = np.log10(np.clip(x_array, 1e-12, None))
    else:
        abs_x = np.abs(x_array)
        transformed = np.where(
            abs_x <= float(symlog_linthresh),
            abs_x / float(symlog_linthresh),
            1.0 + np.log10(np.clip(abs_x / float(symlog_linthresh), 1e-12, None)),
        ) * np.sign(x_array)

    target_positions = np.linspace(float(transformed[0]), float(transformed[-1]), num=safe_target_count)
    marker_indices = sorted({int(np.argmin(np.abs(transformed - target))) for target in target_positions})
    if 0 not in marker_indices:
        marker_indices.insert(0, 0)
    last_index = int(x_array.size - 1)
    if marker_indices[-1] != last_index:
        marker_indices.append(last_index)
    return marker_indices


def _configure_matplotlib_fonts() -> None:
    matplotlib.rcParams["font.family"] = FONT_FAMILY
    matplotlib.rcParams["font.weight"] = FONT_WEIGHT_BOLD
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = FONT_FAMILY
    matplotlib.rcParams["mathtext.it"] = f"{FONT_FAMILY}:italic"
    matplotlib.rcParams["mathtext.bf"] = f"{FONT_FAMILY}:bold"
    matplotlib.rcParams["axes.unicode_minus"] = False


def _resolve_axis_scale(requested_scale: str, values: np.ndarray) -> str:
    if requested_scale == "log" and (
        values.size == 0 or np.any(~np.isfinite(values)) or np.nanmin(values) <= 0.0
    ):
        return "symlog"
    return requested_scale


def _apply_axis_scale(axis: Any, *, axis_name: str, requested_scale: str, values: np.ndarray) -> None:
    actual_scale = _resolve_axis_scale(requested_scale, values)
    setter = axis.set_xscale if axis_name == "x" else axis.set_yscale
    if actual_scale == "symlog":
        setter("symlog", linthresh=1.0)
    else:
        setter(actual_scale)


def _apply_metric_ylim(axis: Any, metric_key: str) -> None:
    group = METRIC_LIMIT_GROUPS[metric_key]
    axis.set_ylim(*METRIC_Y_LIMITS[group])
    axis.set_yticks(METRIC_Y_TICKS[group])


def _style_axis_text(axis: Any) -> None:
    axis.xaxis.get_label().set(fontfamily=FONT_FAMILY, fontweight=FONT_WEIGHT_BOLD, fontsize=14)
    axis.yaxis.get_label().set(fontfamily=FONT_FAMILY, fontweight=FONT_WEIGHT_BOLD, fontsize=14)
    axis.title.set(fontfamily=FONT_FAMILY, fontweight=FONT_WEIGHT_TITLE, fontsize=16)
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontfamily(FONT_FAMILY)
        tick_label.set_fontweight(FONT_WEIGHT_BOLD)
        tick_label.set_fontsize(12)


def _plot_policy_grid(
    rows: Sequence[Mapping[str, str]],
    *,
    topologies: Sequence[str],
    x_key: str,
    metric_specs: Sequence[tuple[str, str]],
    x_label: str,
    output_stem: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        len(topologies),
        len(metric_specs),
        figsize=(float(args.figsize_width), float(args.figsize_height)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, topology_name in enumerate(topologies):
        topology_rows = [row for row in rows if str(row.get("topology")) == topology_name]
        topology_label = TOPOLOGY_LABELS.get(topology_name, topology_name)
        for col_index, (metric_key, ylabel) in enumerate(metric_specs):
            axis = axes[row_index, col_index]
            panel_index = row_index * len(metric_specs) + col_index
            for policy_name in POLICY_ORDER:
                policy_rows = [row for row in topology_rows if str(row.get("policy")) == policy_name]
                if not policy_rows:
                    continue
                policy_rows.sort(key=lambda row: int(float(row[x_key])))
                x_values = np.asarray([int(float(row[x_key])) for row in policy_rows], dtype=np.int64)
                y_values = np.asarray([float(row[metric_key]) for row in policy_rows], dtype=np.float64)
                finite = np.isfinite(x_values) & np.isfinite(y_values)
                x_values = x_values[finite]
                y_values = y_values[finite]
                if not x_values.size:
                    continue
                style = POLICY_STYLES[policy_name]
                axis.plot(
                    x_values,
                    y_values,
                    label=str(style["label"]),
                    color=str(style["color"]),
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=float(args.marker_size),
                    markerfacecolor="white",
                    markeredgecolor=str(style["color"]),
                    markeredgewidth=1.1,
                    markevery=_build_even_markevery(
                        x_values,
                        axis_scale=str(args.time_axis_scale),
                        target_count=max(2, int(args.marker_count)),
                    ),
                    linewidth=2.5 if policy_name == "agent" else 2.0,
                    zorder=float(style["zorder"]),
                )
                _apply_axis_scale(
                    axis,
                    axis_name="x",
                    requested_scale=str(args.time_axis_scale),
                    values=x_values,
                )
                if args.metric_axis_scale == "linear":
                    axis.set_yscale("linear")
                    _apply_metric_ylim(axis, metric_key)
                else:
                    _apply_axis_scale(
                        axis,
                        axis_name="y",
                        requested_scale=str(args.metric_axis_scale),
                        values=y_values,
                    )

            if row_index == 0:
                axis.set_title(ylabel, fontweight=FONT_WEIGHT_TITLE, pad=10.0)
            if row_index == len(topologies) - 1:
                axis.set_xlabel(x_label, fontweight=FONT_WEIGHT_BOLD)
            if col_index == 0:
                axis.set_ylabel(f"{topology_label}\n{ylabel}", fontweight=FONT_WEIGHT_BOLD)
            else:
                axis.set_ylabel(ylabel, fontweight=FONT_WEIGHT_BOLD)
            axis.grid(False)
            axis.margins(x=0.02)
            axis.text(
                0.02,
                0.98,
                _panel_label(panel_index),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontfamily=FONT_FAMILY,
                fontweight=FONT_WEIGHT_BOLD,
                fontsize=15,
            )
            _style_axis_text(axis)

    handles: list[Any] = []
    labels: list[str] = []
    for axis in axes.flat:
        candidate_handles, candidate_labels = axis.get_legend_handles_labels()
        if candidate_handles:
            handles, labels = candidate_handles, candidate_labels
            break
    if handles:
        legend = figure.legend(
            handles,
            labels,
            loc=str(args.legend_loc),
            bbox_to_anchor=(float(args.legend_bbox_x), float(args.legend_bbox_y)),
            ncol=max(1, int(args.legend_ncol)),
            frameon=False,
            prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT_BOLD, "size": 13},
        )
        for text in legend.get_texts():
            text.set_fontfamily(FONT_FAMILY)
            text.set_fontweight(FONT_WEIGHT_BOLD)
            text.set_fontsize(13)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    _configure_matplotlib_fonts()

    input_csv = _coerce_path(args.input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    rows, fieldnames = _read_csv(input_csv)
    schema, x_key, metric_specs = _detect_schema(fieldnames)
    topologies = _ordered_topologies(rows, args.topologies)
    if not topologies:
        raise ValueError("No topology values were found in the input CSV.")

    output_dir = _coerce_path(args.output_dir) if args.output_dir is not None else input_csv.parent
    output_stem = output_dir / str(args.output_name)
    png_path, pdf_path = _plot_policy_grid(
        rows,
        topologies=topologies,
        x_key=x_key,
        metric_specs=metric_specs,
        x_label="t" if schema == "step" else "Episode",
        output_stem=output_stem,
        args=args,
    )

    print(f"Input schema: {schema}")
    print(f"Loaded rows: {len(rows)}")
    print(f"Topologies: {', '.join(topologies)}")
    print(f"PNG: {png_path.resolve()}")
    print(f"PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
