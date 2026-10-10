from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


DEFAULT_INPUT_CSV = Path(
    "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
    "0823_20000_policy_param_sweep_r/all_episode_summary.csv"
)

SCRIPT_DEFAULTS = {
    "input_csv": DEFAULT_INPUT_CSV,
    "output_dir": None,
    "output_name": None,
    "figsize": (12.0, 12.0),
    "legend_ncol": 6,
    "legend_loc": "upper center",
    "legend_bbox_x": 0.50,
    "legend_bbox_y": 1.03,
    "show_legend": True,
    "marker_size": 5.5,
    "line_width": 2.3,
    "pc_normalizer_knots": "25:215.91,50:443.18,62.5:556.82,75:670.45,87.5:784.09,100:897.73",
}

TOPOLOGY_ORDER = ("regular", "scale_free", "erdos_renyi", "small_world")
TOPOLOGY_LABELS = {
    "regular": "Regular",
    "scale_free": "BA",
    "erdos_renyi": "ER",
    "small_world": "WS",
}

# 修改这里即可改变曲线绘制顺序和图例顺序。
POLICY_ORDER = ("m1", "m2", "agent", "equal", "prop", "adaptive")
POLICY_STYLES = {
    "m1": {"color": "#2E8B57", "label": "M1", "linestyle": "-.", "marker": "s", "zorder": 4},
    "m2": {"color": "#BD93DA", "label": "M2", "linestyle": "--", "marker": "^", "zorder": 5},
    "agent": {"color": "#E3A008", "label": "RL-Agent", "linestyle": "-", "marker": "o", "zorder": 4},
    "equal": {"color": "#E56051", "label": "Equal", "linestyle": (0, (5, 2)), "marker": "X", "zorder": 3},
    "prop": {"color": "#1E4688", "label": "Proportional", "linestyle": ":", "marker": "D", "zorder": 3},
    "adaptive": {"color": "#008B8B", "label": "Interpolation", "linestyle": "-", "marker": "P", "zorder": 1},
}

BASE_METRIC_SPECS = (
    ("final_cooperation_mean", r"$f_c$"),
    ("final_mean_resource_mean", r"$\bar{R}(t)$"),
    ("final_gini_mean", "Gini"),
)
X_LABELS = {
    "pc": r"$p_c$",
    "r": r"$r$",
    "beta": r"$\kappa$",
    "n": r"$N$",
    "pool_power_k": r"$k$",
}

FONT_FAMILY = "Times New Roman"
FONT_WEIGHT_BOLD = "bold"
FONT_WEIGHT_TITLE = "normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a parameter-sweep grid from an existing all_episode_summary.csv "
            "or parameter_sweep_summary.csv. No simulations are run."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=SCRIPT_DEFAULTS["input_csv"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DEFAULTS["output_dir"],
        help="Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=SCRIPT_DEFAULTS["output_name"],
        help="Defaults to <x_var>_sweep_four_policy_grid_PLOT.",
    )
    parser.add_argument(
        "--x-var",
        choices=("pc", "r", "beta", "n", "pool_power_k"),
        default=None,
        help="Select one variable when the CSV contains more than one; otherwise detected automatically.",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default=None,
        help="Optional comma-separated topology subset/order.",
    )
    parser.add_argument("--figsize-width", type=float, default=float(SCRIPT_DEFAULTS["figsize"][0]))
    parser.add_argument("--figsize-height", type=float, default=float(SCRIPT_DEFAULTS["figsize"][1]))
    parser.add_argument("--legend-ncol", type=int, default=int(SCRIPT_DEFAULTS["legend_ncol"]))
    parser.add_argument("--legend-loc", type=str, default=str(SCRIPT_DEFAULTS["legend_loc"]))
    parser.add_argument("--legend-bbox-x", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_x"]))
    parser.add_argument("--legend-bbox-y", type=float, default=float(SCRIPT_DEFAULTS["legend_bbox_y"]))
    legend_group = parser.add_mutually_exclusive_group()
    legend_group.add_argument("--show-legend", dest="show_legend", action="store_true")
    legend_group.add_argument("--hide-legend", dest="show_legend", action="store_false")
    parser.set_defaults(show_legend=bool(SCRIPT_DEFAULTS["show_legend"]))
    parser.add_argument("--marker-size", type=float, default=float(SCRIPT_DEFAULTS["marker_size"]))
    parser.add_argument("--line-width", type=float, default=float(SCRIPT_DEFAULTS["line_width"]))
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=None,
        help="Optional directory containing times.ttf, timesbd.ttf, timesi.ttf, and timesbi.ttf.",
    )
    parser.add_argument(
        "--pc-normalizer-knots",
        type=str,
        default=str(SCRIPT_DEFAULTS["pc_normalizer_knots"]),
        help="Used only to reconstruct the normalized resource metric from raw pc-sweep episode rows.",
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


def _parse_pc_normalizer_knots(raw_value: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [chunk.strip() for chunk in str(raw_value).split(",") if chunk.strip()]
    if len(pairs) < 2:
        raise ValueError("pc normalizer knots need at least two <pc>:<normalizer> pairs.")
    x_values: list[float] = []
    y_values: list[float] = []
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid pc normalizer knot: {pair}")
        left, right = pair.split(":", maxsplit=1)
        x_values.append(float(left))
        y_values.append(float(right))
    order = np.argsort(np.asarray(x_values, dtype=np.float64))
    return np.asarray(x_values, dtype=np.float64)[order], np.asarray(y_values, dtype=np.float64)[order]


def _interp_with_linear_extrapolation(x_value: float, x_knots: np.ndarray, y_knots: np.ndarray) -> float:
    x = float(x_value)
    if x <= float(x_knots[0]):
        slope = (float(y_knots[1]) - float(y_knots[0])) / (float(x_knots[1]) - float(x_knots[0]))
        return float(y_knots[0]) + slope * (x - float(x_knots[0]))
    if x >= float(x_knots[-1]):
        slope = (float(y_knots[-1]) - float(y_knots[-2])) / (float(x_knots[-1]) - float(x_knots[-2]))
        return float(y_knots[-1]) + slope * (x - float(x_knots[-1]))
    return float(np.interp(x, x_knots, y_knots))


def _detect_schema(fieldnames: Sequence[str]) -> str:
    names = set(fieldnames)
    common = {"x_var", "x_value", "topology", "policy"}
    episode_fields = common | {
        "episode",
        "final_cooperation_rate",
        "final_mean_resource",
        "final_gini",
    }
    summary_fields = common | {
        "final_cooperation_mean",
        "final_mean_resource_mean",
        "final_gini_mean",
    }
    if episode_fields.issubset(names):
        return "episode"
    if summary_fields.issubset(names):
        return "summary"
    raise ValueError(
        "Unsupported CSV schema. Expected all_episode_summary.csv or parameter_sweep_summary.csv fields."
    )


def _resolve_x_var(rows: Sequence[Mapping[str, str]], requested: str | None) -> str:
    available = sorted({str(row.get("x_var", "")).strip().lower() for row in rows if row.get("x_var")})
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Requested x_var={requested!r} is absent; CSV contains {available}.")
        return requested
    if len(available) != 1:
        raise ValueError(f"Could not select one x_var automatically; CSV contains {available}. Use --x-var.")
    return available[0]


def _aggregate_episode_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    x_var: str,
    pc_normalizer_knots: tuple[np.ndarray, np.ndarray],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("x_var", "")).strip().lower() != x_var:
            continue
        key = (float(row["x_value"]), str(row["topology"]), str(row["policy"]))
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (x_value, topology, policy), group_rows in grouped.items():
        cooperation = np.asarray([float(row["final_cooperation_rate"]) for row in group_rows], dtype=np.float64)
        resource = np.asarray([float(row["final_mean_resource"]) for row in group_rows], dtype=np.float64)
        gini = np.asarray([float(row["final_gini"]) for row in group_rows], dtype=np.float64)
        resource_mean = float(np.mean(resource))
        summary: dict[str, Any] = {
            "x_var": x_var,
            "x_value": float(x_value),
            "topology": topology,
            "policy": policy,
            "episodes": len(group_rows),
            "final_cooperation_mean": float(np.mean(cooperation)),
            "final_mean_resource_mean": resource_mean,
            "final_gini_mean": float(np.mean(gini)),
        }
        if x_var == "pc":
            normalizer = _interp_with_linear_extrapolation(
                x_value,
                pc_normalizer_knots[0],
                pc_normalizer_knots[1],
            )
            summary["final_mean_resource_normalized_mean"] = resource_mean / normalizer
        summary_rows.append(summary)
    return summary_rows


def _prepare_summary_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    schema: str,
    x_var: str,
    pc_normalizer_knots: tuple[np.ndarray, np.ndarray],
) -> list[dict[str, Any]]:
    if schema == "episode":
        return _aggregate_episode_rows(rows, x_var=x_var, pc_normalizer_knots=pc_normalizer_knots)

    prepared: list[dict[str, Any]] = []
    for source_row in rows:
        if str(source_row.get("x_var", "")).strip().lower() != x_var:
            continue
        row: dict[str, Any] = dict(source_row)
        row["x_value"] = float(source_row["x_value"])
        for metric_key, _ in BASE_METRIC_SPECS:
            row[metric_key] = float(source_row[metric_key])
        if x_var == "pc":
            normalized_value = str(source_row.get("final_mean_resource_normalized_mean", "")).strip()
            if normalized_value:
                row["final_mean_resource_normalized_mean"] = float(normalized_value)
            else:
                normalizer = _interp_with_linear_extrapolation(
                    float(row["x_value"]),
                    pc_normalizer_knots[0],
                    pc_normalizer_knots[1],
                )
                row["final_mean_resource_normalized_mean"] = float(row["final_mean_resource_mean"]) / normalizer
        prepared.append(row)
    return prepared


def _ordered_topologies(rows: Sequence[Mapping[str, Any]], requested: str | None) -> list[str]:
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


def _metric_specs_for_x_var(x_var: str) -> tuple[tuple[str, str], ...]:
    if x_var == "pc":
        return (
            ("final_cooperation_mean", r"$f_c$"),
            ("final_mean_resource_normalized_mean", r"$\bar{R}_{\mathrm{normal}}(t)$"),
            ("final_gini_mean", "Gini"),
        )
    return BASE_METRIC_SPECS


def _register_times_new_roman(font_dir: Path | None) -> Path:
    candidate_dirs: list[Path] = []
    if font_dir is not None:
        candidate_dirs.append(_coerce_path(font_dir))
    windows_dir = os.environ.get("WINDIR")
    if windows_dir:
        candidate_dirs.append(Path(windows_dir) / "Fonts")
    candidate_dirs.extend(
        [
            Path("C:/Windows/Fonts"),
            Path("/mnt/c/Windows/Fonts"),
            Path("/usr/share/fonts/truetype/msttcorefonts"),
        ]
    )

    registered_paths: list[Path] = []
    for directory in dict.fromkeys(candidate_dirs):
        for filename in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
            font_path = directory / filename
            if font_path.is_file():
                font_manager.fontManager.addfont(str(font_path))
                registered_paths.append(font_path)

    try:
        resolved_path = Path(font_manager.findfont(FONT_FAMILY, fallback_to_default=False))
    except ValueError as exc:
        searched = ", ".join(str(path) for path in dict.fromkeys(candidate_dirs))
        raise RuntimeError(
            "Times New Roman was not found, so plotting was stopped instead of using a fallback font. "
            f"Searched: {searched}. Use --font-dir to specify the Windows Fonts directory."
        ) from exc
    return resolved_path


def _configure_matplotlib_fonts(font_dir: Path | None) -> Path:
    resolved_font_path = _register_times_new_roman(font_dir)
    matplotlib.rcParams["font.family"] = FONT_FAMILY
    matplotlib.rcParams["font.weight"] = FONT_WEIGHT_BOLD
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = FONT_FAMILY
    matplotlib.rcParams["mathtext.it"] = f"{FONT_FAMILY}:italic"
    matplotlib.rcParams["mathtext.bf"] = f"{FONT_FAMILY}:bold"
    matplotlib.rcParams["mathtext.cal"] = FONT_FAMILY
    matplotlib.rcParams["mathtext.sf"] = FONT_FAMILY
    matplotlib.rcParams["mathtext.tt"] = FONT_FAMILY
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    return resolved_font_path


def _style_axis_text(axis: Any) -> None:
    axis.xaxis.get_label().set(fontfamily=FONT_FAMILY, fontweight=FONT_WEIGHT_BOLD, fontsize=14)
    axis.yaxis.get_label().set(fontfamily=FONT_FAMILY, fontweight=FONT_WEIGHT_BOLD, fontsize=14)
    axis.title.set(fontfamily=FONT_FAMILY, fontweight=FONT_WEIGHT_TITLE, fontsize=16)
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontfamily(FONT_FAMILY)
        tick_label.set_fontweight(FONT_WEIGHT_BOLD)
        tick_label.set_fontsize(12)


def _apply_sweep_metric_ylim(axis: Any, metric_key: str, values: np.ndarray) -> None:
    if metric_key in {"final_cooperation_mean", "final_gini_mean", "final_mean_resource_normalized_mean"}:
        axis.set_ylim(-0.03, 1.03)
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    elif metric_key == "final_mean_resource_mean":
        axis.set_ylim(-12.0, 462.0)
        axis.set_yticks([0.0, 100.0, 200.0, 300.0, 400.0, 450.0])
    elif values.size and np.any(np.isfinite(values)):
        value_min = float(np.nanmin(values))
        value_max = float(np.nanmax(values))
        span = max(1e-9, value_max - value_min)
        axis.set_ylim(value_min - 0.05 * span, value_max + 0.05 * span)


def _plot_sweep_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_var: str,
    topologies: Sequence[str],
    output_stem: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    metric_specs = _metric_specs_for_x_var(x_var)
    figure, axes = plt.subplots(
        len(topologies),
        len(metric_specs),
        figsize=(float(args.figsize_width), float(args.figsize_height)),
        sharex="col",
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, topology_name in enumerate(topologies):
        topology_rows = [row for row in rows if str(row.get("topology")) == topology_name]
        topology_label = TOPOLOGY_LABELS.get(topology_name, topology_name)
        for col_index, (metric_key, ylabel) in enumerate(metric_specs):
            axis = axes[row_index, col_index]
            for policy_name in POLICY_ORDER:
                policy_rows = [row for row in topology_rows if str(row.get("policy")) == policy_name]
                if not policy_rows:
                    continue
                policy_rows.sort(key=lambda row: float(row["x_value"]))
                x_values = np.asarray([float(row["x_value"]) for row in policy_rows], dtype=np.float64)
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
                    linewidth=float(args.line_width),
                    zorder=float(style["zorder"]),
                )

            axis.set_yscale("linear")
            metric_values = np.asarray([float(row[metric_key]) for row in topology_rows], dtype=np.float64)
            _apply_sweep_metric_ylim(axis, metric_key, metric_values)
            if row_index == 0:
                axis.set_title(ylabel, fontweight=FONT_WEIGHT_TITLE, pad=10.0)
            if row_index == len(topologies) - 1:
                axis.set_xlabel(X_LABELS.get(x_var, x_var), fontweight=FONT_WEIGHT_BOLD)
            if col_index == 0:
                axis.set_ylabel(f"{topology_label}\n{ylabel}", fontweight=FONT_WEIGHT_BOLD)
            else:
                axis.set_ylabel(ylabel, fontweight=FONT_WEIGHT_BOLD)
            axis.grid(False)
            _style_axis_text(axis)
            if metric_key == "final_gini_mean":
                axis.title.set_fontstyle("italic")
                axis.title.set_fontweight(FONT_WEIGHT_BOLD)
                axis.yaxis.get_label().set_fontstyle("italic")
                axis.yaxis.get_label().set_fontweight(FONT_WEIGHT_BOLD)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if bool(args.show_legend) and handles:
        legend = figure.legend(
            handles,
            labels,
            loc=str(args.legend_loc),
            bbox_to_anchor=(float(args.legend_bbox_x), float(args.legend_bbox_y)),
            ncol=max(1, int(args.legend_ncol)),
            frameon=False,
            borderaxespad=0.0,
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
    resolved_font_path = _configure_matplotlib_fonts(args.font_dir)

    input_csv = _coerce_path(args.input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    source_rows, fieldnames = _read_csv(input_csv)
    schema = _detect_schema(fieldnames)
    x_var = _resolve_x_var(source_rows, args.x_var)
    pc_normalizer_knots = _parse_pc_normalizer_knots(args.pc_normalizer_knots)
    summary_rows = _prepare_summary_rows(
        source_rows,
        schema=schema,
        x_var=x_var,
        pc_normalizer_knots=pc_normalizer_knots,
    )
    if not summary_rows:
        raise ValueError(f"No rows remain for x_var={x_var}.")
    topologies = _ordered_topologies(summary_rows, args.topologies)
    if not topologies:
        raise ValueError("No topology values were found in the input CSV.")

    output_dir = _coerce_path(args.output_dir) if args.output_dir is not None else input_csv.parent
    output_name = args.output_name or f"{x_var}_sweep_four_policy_grid_PLOT"
    png_path, pdf_path = _plot_sweep_grid(
        summary_rows,
        x_var=x_var,
        topologies=topologies,
        output_stem=output_dir / output_name,
        args=args,
    )

    x_values = sorted({float(row["x_value"]) for row in summary_rows})
    policies = [name for name in POLICY_ORDER if any(str(row.get("policy")) == name for row in summary_rows)]
    print(f"Input schema: {schema}")
    print(f"Loaded episode rows: {len(source_rows)}" if schema == "episode" else f"Loaded summary rows: {len(source_rows)}")
    print(f"Aggregated plot rows: {len(summary_rows)}")
    print(f"Sweep: {x_var} = {x_values}")
    print(f"Topologies: {', '.join(topologies)}")
    print(f"Policies: {', '.join(policies)}")
    print(f"Font: {FONT_FAMILY} ({resolved_font_path})")
    print(f"PNG: {png_path.resolve()}")
    print(f"PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
