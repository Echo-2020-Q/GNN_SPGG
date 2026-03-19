from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
# Use a non-interactive backend because this module only saves figures to disk.
# This avoids Tk backend crashes on Windows during long batch rendering.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


Graph = Dict[int, List[int]]
HistoryRecord = Dict[str, Any]
Layout = Dict[int, Tuple[float, float]]


SCALAR_LABELS = {
    "actual_cooperation_rate": "Actual Cooperation f_c",
    "nominal_cooperation_rate": "Nominal Cooperation",
    "mean_resource": "Mean Resource",
    "mean_pool_raw": "Mean Raw Pool",
    "mean_pool_grown": "Mean Grown Pool",
    "mean_investment": "Mean Investment",
    "mean_income": "Mean Income",
    "mean_consumption": "Mean Consumption",
    "mean_payoff": "Mean Payoff",
    "gini": "Gini",
    "reward": "Reward",
}

STEADY_STATE_LABELS = {
    "final_actual_cooperation_mean": "Steady-State f_c",
    "final_mean_resource_mean": "Steady-State Mean Resource",
    "final_mean_pool_grown_mean": "Steady-State Mean Grown Pool",
    "final_mean_consumption_mean": "Steady-State Mean Consumption",
    "final_mean_payoff_mean": "Steady-State Mean Payoff",
    "final_gini_mean": "Steady-State Gini",
}


def build_layout(
    graph: Graph,
    layout_name: str = "spring",
    seed: Optional[int] = None,
    spring_iterations: int = 80,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Layout:
    if layout_name == "spring":
        return _spring_layout(graph, seed=seed, iterations=spring_iterations)
    if layout_name == "circular":
        return _circular_layout(graph)
    if layout_name == "grid":
        if grid_shape is None:
            raise ValueError("grid_shape is required when layout_name is 'grid'.")
        return _grid_layout(graph, rows=grid_shape[0], cols=grid_shape[1])
    raise ValueError("Unsupported layout_name: {0}".format(layout_name))


def save_network_snapshots(
    graph: Graph,
    history: Sequence[HistoryRecord],
    output_dir: Path,
    positions: Layout,
    node_color_metric: str,
    node_size_metric: str,
    node_value_metric: Optional[str] = None,
    node_value_decimals: int = 1,
    frame_stride: int = 1,
    dpi: int = 160,
    label_nodes: bool = False,
    title_prefix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = _edge_list(graph)

    for record in history:
        time_index = int(record["time"])
        if frame_stride > 1 and (time_index % frame_stride) != 0:
            continue

        figure, axis = plt.subplots(figsize=(10, 8))
        _draw_edges(axis, edges, positions)
        node_ids, node_sizes = _draw_nodes(
            axis,
            positions,
            record,
            node_color_metric=node_color_metric,
            node_size_metric=node_size_metric,
        )

        if label_nodes or node_value_metric:
            _draw_node_annotations(
                axis,
                positions,
                record,
                node_ids=node_ids,
                node_sizes=node_sizes,
                label_nodes=label_nodes,
                node_value_metric=node_value_metric,
                node_value_decimals=node_value_decimals,
            )

        axis.set_axis_off()
        axis.set_title(
            "{0}t={1} | f_c={2:.3f} | mean_R={3:.3f} | mean_G={4:.3f}".format(
                title_prefix,
                time_index,
                float(record["actual_cooperation_rate"]),
                float(record["mean_resource"]),
                float(record["mean_pool_grown"]),
            )
        )

        frame_path = output_dir / "frame_{0:04d}.png".format(time_index)
        figure.tight_layout()
        figure.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def save_macro_timeseries(
    history: Sequence[HistoryRecord],
    output_path: Path,
    metrics: Sequence[str],
    title: str = "",
    dpi: int = 160,
) -> None:
    if not metrics:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = [int(record["time"]) for record in history]

    figure, axes = plt.subplots(len(metrics), 1, figsize=(12, 2.8 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        values = [float(record[metric]) for record in history]
        axis.plot(times, values, linewidth=2.0)
        axis.set_ylabel(SCALAR_LABELS.get(metric, metric))
        axis.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Time step")
    figure.suptitle(title or "Macro Statistics Over Time", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_scan_metric_grid(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    metrics: Sequence[str],
    title: str = "",
    dpi: int = 160,
) -> None:
    if not records or not metrics:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    network_labels = sorted({str(record["network_label"]) for record in records})
    num_metrics = len(metrics)
    num_cols = 2
    num_rows = int(np.ceil(num_metrics / num_cols))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    line_style_cycle = ["-", "--", "-.", ":"]

    figure, axes = plt.subplots(num_rows, num_cols, figsize=(14, 3.5 * num_rows), sharex=True)
    axes_array = np.atleast_1d(axes).reshape(-1)

    for axis, metric in zip(axes_array, metrics):
        for network_index, network_label in enumerate(network_labels):
            network_records = [
                record for record in records
                if str(record["network_label"]) == network_label
            ]
            network_records.sort(key=lambda item: float(item["r"]))
            r_values = [float(item["r"]) for item in network_records]
            y_values = [float(item[metric]) for item in network_records]
            axis.plot(
                r_values,
                y_values,
                marker=marker_cycle[network_index % len(marker_cycle)],
                linestyle=line_style_cycle[network_index % len(line_style_cycle)],
                linewidth=2.0,
                markersize=6.5,
                alpha=0.85,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=network_label,
            )

        axis.set_ylabel(STEADY_STATE_LABELS.get(metric, metric))
        axis.grid(True, linestyle="--", alpha=0.35)

    for axis in axes_array[num_metrics:]:
        axis.axis("off")

    for axis in axes_array[-num_cols:]:
        axis.set_xlabel("r")

    if network_labels:
        handles, labels = axes_array[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            title="Network",
        )

    figure.suptitle(title or "Steady-State Metrics vs r", fontsize=14, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 0.84, 0.96))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _draw_edges(axis: plt.Axes, edges: Sequence[Tuple[int, int]], positions: Layout) -> None:
    segments = []
    for source, target in edges:
        segments.append([positions[source], positions[target]])
    if segments:
        line_collection = LineCollection(segments, colors="lightgray", linewidths=0.8, alpha=0.45)
        axis.add_collection(line_collection)


def _draw_nodes(
    axis: plt.Axes,
    positions: Layout,
    record: HistoryRecord,
    node_color_metric: str,
    node_size_metric: str,
) -> Tuple[List[int], np.ndarray]:
    node_ids = sorted(positions.keys())
    x_values = [positions[node][0] for node in node_ids]
    y_values = [positions[node][1] for node in node_ids]

    color_values = _extract_node_metric(record, node_color_metric)
    size_values = _extract_node_metric(record, node_size_metric)
    sizes = _scale_sizes(size_values)

    if node_color_metric in {"x_actual", "x_nominal"}:
        colors = ["#2a9d8f" if value >= 0.5 else "#d62828" for value in color_values]
        axis.scatter(x_values, y_values, c=colors, s=sizes, edgecolors="black", linewidths=0.5)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Cooperate", markerfacecolor="#2a9d8f", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Defect", markerfacecolor="#d62828", markersize=8),
        ]
        axis.legend(handles=legend_handles, loc="upper right", frameon=True)
    else:
        scatter = axis.scatter(
            x_values,
            y_values,
            c=color_values,
            s=sizes,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.5,
        )
        colorbar = plt.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(node_color_metric)

    return node_ids, sizes


def _draw_node_annotations(
    axis: plt.Axes,
    positions: Layout,
    record: HistoryRecord,
    node_ids: Sequence[int],
    node_sizes: np.ndarray,
    label_nodes: bool,
    node_value_metric: Optional[str],
    node_value_decimals: int,
) -> None:
    node_value_texts: Optional[List[str]] = None
    if node_value_metric:
        node_values = _extract_node_metric(record, node_value_metric)
        node_value_texts = [
            _format_node_value(value, decimals=node_value_decimals)
            for value in node_values
        ]

    for index, node in enumerate(node_ids):
        x_coord, y_coord = positions[node]
        size = float(node_sizes[index])

        if node_value_texts is not None:
            value_text = axis.text(
                x_coord,
                y_coord,
                node_value_texts[index],
                fontsize=float(np.clip(4.8 + (size / 140.0), 5.0, 8.0)),
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                zorder=6,
            )
            value_text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1.6, foreground="black", alpha=0.75),
                    path_effects.Normal(),
                ]
            )

        if label_nodes:
            axis.text(
                x_coord,
                y_coord + (0.015 + 0.00004 * size),
                str(node),
                fontsize=6,
                color="#1f1f1f",
                ha="center",
                va="bottom",
                zorder=7,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                    "pad": 0.2,
                },
            )


def _extract_node_metric(record: HistoryRecord, metric: str) -> np.ndarray:
    if metric in record:
        values = np.asarray(record[metric], dtype=np.float64)
        if values.ndim == 1:
            return values
    raise ValueError("Unsupported node metric: {0}".format(metric))


def _scale_sizes(values: np.ndarray, minimum: float = 80.0, maximum: float = 380.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, values[0]):
        return np.full(values.shape, (minimum + maximum) / 2.0)
    normalized = (values - values.min()) / (values.max() - values.min())
    return minimum + normalized * (maximum - minimum)


def _format_node_value(value: float, decimals: int = 1) -> str:
    rounded = round(float(value), max(decimals, 0))
    if np.isclose(rounded, round(rounded)):
        return str(int(round(rounded)))
    return f"{rounded:.{max(decimals, 0)}f}".rstrip("0").rstrip(".")


def _edge_list(graph: Mapping[int, Iterable[int]]) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for source, neighbors in graph.items():
        for target in neighbors:
            if source < target:
                edges.append((int(source), int(target)))
    return edges


def _circular_layout(graph: Graph) -> Layout:
    node_ids = sorted(graph.keys())
    num_nodes = len(node_ids)
    if num_nodes == 1:
        return {node_ids[0]: (0.5, 0.5)}

    positions: Layout = {}
    for index, node in enumerate(node_ids):
        angle = (2.0 * np.pi * index) / num_nodes
        positions[node] = (0.5 + 0.42 * np.cos(angle), 0.5 + 0.42 * np.sin(angle))
    return positions


def _grid_layout(graph: Graph, rows: int, cols: int) -> Layout:
    if rows * cols != len(graph):
        raise ValueError("grid layout requires rows * cols == number of nodes.")

    positions: Layout = {}
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            x_coord = col / max(cols - 1, 1)
            y_coord = 1.0 - (row / max(rows - 1, 1))
            positions[node] = (x_coord, y_coord)
    return positions


def _spring_layout(graph: Graph, seed: Optional[int] = None, iterations: int = 80) -> Layout:
    node_ids = sorted(graph.keys())
    num_nodes = len(node_ids)
    if num_nodes == 1:
        return {node_ids[0]: (0.5, 0.5)}

    node_to_index = {node: index for index, node in enumerate(node_ids)}
    edge_pairs = [(node_to_index[source], node_to_index[target]) for source, target in _edge_list(graph)]

    rng = np.random.default_rng(seed)
    positions = rng.uniform(-0.5, 0.5, size=(num_nodes, 2))
    area = 1.0
    optimal_distance = np.sqrt(area / max(num_nodes, 1))
    temperature = 0.15
    epsilon = 1e-9

    for _ in range(max(iterations, 1)):
        displacements = np.zeros_like(positions)

        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1) + epsilon
        repulsive_forces = (optimal_distance ** 2 / distances ** 2)[:, :, None] * deltas
        displacements += repulsive_forces.sum(axis=1)

        for source_index, target_index in edge_pairs:
            delta = positions[source_index] - positions[target_index]
            distance = np.linalg.norm(delta) + epsilon
            attractive_force = (distance ** 2 / optimal_distance) * (delta / distance)
            displacements[source_index] -= attractive_force
            displacements[target_index] += attractive_force

        displacement_norms = np.linalg.norm(displacements, axis=1, keepdims=True)
        displacement_norms = np.where(displacement_norms < epsilon, 1.0, displacement_norms)
        positions += (displacements / displacement_norms) * np.minimum(displacement_norms, temperature)
        positions = np.clip(positions, -1.0, 1.0)
        temperature *= 0.95

    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    x_coords = (x_coords - x_coords.min()) / max(x_coords.max() - x_coords.min(), epsilon)
    y_coords = (y_coords - y_coords.min()) / max(y_coords.max() - y_coords.min(), epsilon)

    return {
        node: (float(x_coords[index]), float(y_coords[index]))
        for index, node in enumerate(node_ids)
    }
