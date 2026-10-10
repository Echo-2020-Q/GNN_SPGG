from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


"""State-conditioned split (row-level) for GNN_SPGG policy analysis.

This script is intentionally *row-level* (sample = (episode, step, sender)), not pair-level.

Inputs
------
- A folder that contains `decision_records.csv` produced by:
  - analyze_gnn_policy_decisions.py
  - analyze_gnn_policy_mechanism.py

Core idea
---------
Instead of asking "why did behavior change at t=...", we ask
"when state variable z falls into different regimes, how does the row behavior change?".

Outputs (minimal MVP)
--------------------
1) Per split-variable bin summary CSVs (quantile bins)
2) A variance-explained importance table (sorted)
3) A depth<=2 shallow tree rule table (leaf stats)
"""


SCRIPT_DEFAULTS: dict[str, Any] = {
    "analysis_dir": "outputs/_tmp_policy_analysis_sender_gini_smoke",
    "decision_records": None,
    "output_dir": None,
    "bin_count": 10,
    "eps": 1e-12,
    "tree_target": "weight_equal",
    "tree_max_depth": 2,
    "tree_min_leaf": 200,
    "tree_threshold_candidates": 64,
}


SplitVar = str
TargetVar = str


DEFAULT_SPLIT_VARS: tuple[SplitVar, ...] = (
    "pool_grown_over_upperbound",
    "sender_resource_norm",
    "sender_strategy_norm",
    "sender_degree",
    "local_investment_cv",
    "local_cooperation_count",
)


DEFAULT_TARGET_VARS: tuple[TargetVar, ...] = (
    "weight_unit_investment",
    "weight_equal",
    "self_allocation",
)


def _parse_toggle_arg(raw_value: str) -> bool:
    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected on/off or true/false, got: {raw_value}")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.sum(array))
    if array.size == 0:
        return array.copy()
    if not np.isfinite(total) or total <= eps:
        return np.full(array.shape, 1.0 / float(array.size), dtype=np.float64)
    return array / total


def _entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if not np.isfinite(s) or s <= eps:
        return float("nan")
    p = p / s
    mask = p > eps
    return float(-np.sum(p[mask] * np.log(p[mask])))


def _safe_cv(values: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("nan")
    mean = float(np.mean(x))
    std = float(np.std(x))
    return float(std / (abs(mean) + eps))


def _project_to_simplex(weights: np.ndarray) -> np.ndarray:
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        return vector.copy()
    sorted_vector = np.sort(vector)[::-1]
    cssv = np.cumsum(sorted_vector) - 1.0
    rho_candidates = sorted_vector - (cssv / (np.arange(vector.size) + 1))
    positive = np.nonzero(rho_candidates > 0.0)[0]
    if positive.size == 0:
        return np.full(vector.shape, 1.0 / float(vector.size), dtype=np.float64)
    rho = int(positive[-1])
    theta = cssv[rho] / float(rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    projected_sum = float(projected.sum())
    if projected_sum <= 1e-12:
        return np.full(vector.shape, 1.0 / float(vector.size), dtype=np.float64)
    return projected / projected_sum


def _fit_three_way_weights(
    allocation_row: np.ndarray,
    proportional_row: np.ndarray,
    equal_row: np.ndarray,
    self_row: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit weights w = (w_prop, w_equal, w_self) on the simplex."""
    alloc = np.asarray(allocation_row, dtype=np.float64).reshape(-1)
    if alloc.size == 0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float64), float("nan")

    basis = np.column_stack([
        np.asarray(proportional_row, dtype=np.float64) - np.asarray(equal_row, dtype=np.float64),
        np.asarray(self_row, dtype=np.float64) - np.asarray(equal_row, dtype=np.float64),
    ])
    target = alloc - np.asarray(equal_row, dtype=np.float64)
    if np.allclose(basis, 0.0):
        weights = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
        fit_error = float(np.mean(np.abs(alloc - np.asarray(equal_row, dtype=np.float64))))
        return weights, fit_error

    raw_solution, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
    raw_prop = float(raw_solution[0]) if raw_solution.size >= 1 else 0.0
    raw_self = float(raw_solution[1]) if raw_solution.size >= 2 else 0.0
    raw_equal = 1.0 - raw_prop - raw_self
    weights = _project_to_simplex(np.asarray([raw_prop, raw_equal, raw_self], dtype=np.float64))
    fitted = (weights[0] * proportional_row) + (weights[1] * equal_row) + (weights[2] * self_row)
    fit_error = float(np.mean(np.abs(np.asarray(fitted, dtype=np.float64) - alloc)))
    return weights, fit_error


def _quantile_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    bin_count = max(2, int(bin_count))
    qs = np.linspace(0.0, 1.0, num=bin_count + 1)
    edges = np.unique(np.quantile(x, qs))
    return edges.astype(np.float64)


def _variance_explained_r2(y: np.ndarray, bin_ids: np.ndarray, bin_count: int) -> float:
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    bins = np.asarray(bin_ids, dtype=np.int64).reshape(-1)
    if yy.size == 0 or yy.size != bins.size:
        return float("nan")
    finite = np.isfinite(yy) & np.isfinite(bins)
    yy = yy[finite]
    bins = bins[finite]
    if yy.size < 2:
        return float("nan")
    y_mean = float(np.mean(yy))
    sst = float(np.sum((yy - y_mean) ** 2))
    if not np.isfinite(sst) or sst <= 1e-12:
        return float("nan")

    sse = 0.0
    for b in range(int(bin_count)):
        mask = bins == b
        if not np.any(mask):
            continue
        yb = yy[mask]
        mean_b = float(np.mean(yb))
        sse += float(np.sum((yb - mean_b) ** 2))
    return float(1.0 - (sse / sst))


def _assign_quantile_bins(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if edges.size < 2:
        return np.full(x.shape, -1, dtype=np.int64), 0
    # bins: 0..(B-1)
    bin_ids = np.searchsorted(edges[1:-1], x, side="right").astype(np.int64)
    return bin_ids, int(edges.size - 1)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_decision_records_path(analysis_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if path.is_dir():
            candidate = path / "decision_records.csv"
            if candidate.exists():
                return candidate
        if path.exists():
            return path
        raise FileNotFoundError(f"decision_records not found: {path}")

    candidate = analysis_dir / "decision_records.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"decision_records.csv not found in: {analysis_dir}")


def _infer_receiver_degrees_from_first_snapshot(decision_records_path: Path) -> np.ndarray:
    """Infer receiver degrees from exposure counts in the first (episode, step) snapshot.

    For an undirected graph with self included in local_mask, each receiver appears in exactly
    (degree(receiver) + 1) sender rows at a fixed snapshot.
    """

    first_episode: int | None = None
    first_step: int | None = None
    exposure_counts: dict[int, int] = {}
    max_receiver = -1

    with decision_records_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("decision_records.csv has no header")

        required = {"episode", "step", "receiver"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise KeyError(f"decision_records.csv missing columns: {sorted(missing)}")

        for row in reader:
            episode = _safe_int(row.get("episode"))
            step = _safe_int(row.get("step"))
            if first_episode is None:
                first_episode = episode
                first_step = step
            if episode != first_episode or step != first_step:
                break
            receiver = _safe_int(row.get("receiver"))
            max_receiver = max(max_receiver, receiver)
            exposure_counts[receiver] = exposure_counts.get(receiver, 0) + 1

    if max_receiver < 0:
        return np.empty(0, dtype=np.int64)
    counts = np.zeros(max_receiver + 1, dtype=np.int64)
    for receiver, count in exposure_counts.items():
        if 0 <= receiver < counts.size:
            counts[receiver] = int(count)
    degrees = np.maximum(counts - 1, 0)
    return degrees


@dataclass
class RowDataset:
    # identity
    episode: np.ndarray
    step: np.ndarray
    sender: np.ndarray

    # split variables
    pool_grown_over_upperbound: np.ndarray
    sender_resource_norm: np.ndarray
    sender_strategy_norm: np.ndarray
    sender_degree: np.ndarray
    local_investment_cv: np.ndarray
    local_cooperation_count: np.ndarray

    # targets
    weight_unit_investment: np.ndarray
    weight_equal: np.ndarray
    weight_self: np.ndarray
    self_allocation: np.ndarray

    # extra for tree leaf summary
    row_entropy: np.ndarray


def _build_row_dataset_from_decision_records(
    decision_records_path: Path,
    *,
    receiver_degrees: np.ndarray,
    eps: float,
) -> RowDataset:
    required_columns = {
        "episode",
        "step",
        "sender",
        "receiver",
        "allocation",
        "sender_pool_grown",
        "sender_pool_raw_norm",
        "sender_resource_norm",
        "sender_strategy_norm",
        "receiver_resource",
        "receiver_strategy_norm",
        "receiver_x_actual",
    }

    episodes: list[int] = []
    steps: list[int] = []
    senders: list[int] = []

    pool_grown_over_upperbound: list[float] = []
    sender_resource_norm: list[float] = []
    sender_strategy_norm: list[float] = []
    sender_degree: list[int] = []
    local_investment_cv: list[float] = []
    local_cooperation_count: list[float] = []

    weight_unit_investment: list[float] = []
    weight_equal: list[float] = []
    weight_self: list[float] = []
    self_allocation: list[float] = []
    row_entropy: list[float] = []

    def _finalize_group(
        key: tuple[int, int, int],
        group_sender_pool_grown: float,
        group_sender_pool_raw_norm: float,
        group_sender_resource_norm: float,
        group_sender_strategy_norm: float,
        receiver_ids: list[int],
        allocations: list[float],
        receiver_resources: list[float],
        receiver_strategy_norms: list[float],
        receiver_x_actuals: list[float],
    ) -> None:
        episode, step, sender = key
        receiver_count = int(len(receiver_ids))
        if receiver_count <= 0:
            return

        alloc = _safe_normalize(np.asarray(allocations, dtype=np.float64), eps=eps)
        receivers = np.asarray(receiver_ids, dtype=np.int64)

        # derive investment and unit investment
        recv_resource = np.asarray(receiver_resources, dtype=np.float64)
        recv_strat = np.asarray(receiver_strategy_norms, dtype=np.float64)
        recv_investment = np.clip(recv_resource, 0.0, None) * np.clip(recv_strat, 0.0, None)
        if receiver_degrees.size > 0:
            degree_lookup = receiver_degrees
            thresholds = np.asarray([float(degree_lookup[r]) + 1.0 if 0 <= r < degree_lookup.size else float("nan") for r in receivers], dtype=np.float64)
            unit_investment = np.divide(
                recv_investment,
                thresholds,
                out=np.zeros_like(recv_investment, dtype=np.float64),
                where=np.isfinite(thresholds) & (thresholds > eps),
            )
        else:
            unit_investment = recv_investment

        proportional_row = _safe_normalize(unit_investment, eps=eps)
        equal_row = np.full(receiver_count, 1.0 / float(receiver_count), dtype=np.float64)
        self_row = np.zeros(receiver_count, dtype=np.float64)
        self_matches = np.nonzero(receivers == int(sender))[0]
        if self_matches.size > 0:
            self_row[int(self_matches[0])] = 1.0
        else:
            # fallback: still produce a valid self row
            self_row[int(np.argmax(alloc))] = 1.0

        weights, _fit_error = _fit_three_way_weights(alloc, proportional_row, equal_row, self_row)

        # pool upperbound (robust, no need p_c): pool_theoretical_max = pool_raw / pool_raw_norm
        pool_raw = float(np.sum(unit_investment))
        if np.isfinite(group_sender_pool_raw_norm) and group_sender_pool_raw_norm > eps and pool_raw > eps:
            pool_upperbound = float(pool_raw / group_sender_pool_raw_norm)
            ratio = float(group_sender_pool_grown / pool_upperbound) if pool_upperbound > eps else float("nan")
        else:
            ratio = float("nan")

        # local summaries
        coop_count = float(np.sum(np.asarray(receiver_x_actuals, dtype=np.float64)))
        invest_cv = _safe_cv(recv_investment, eps=eps)

        deg = receiver_count - 1
        # self allocation from alloc
        if self_matches.size > 0:
            self_alloc = float(alloc[int(self_matches[0])])
        else:
            self_alloc = float("nan")

        episodes.append(int(episode))
        steps.append(int(step))
        senders.append(int(sender))

        pool_grown_over_upperbound.append(float(ratio))
        sender_resource_norm.append(float(group_sender_resource_norm))
        sender_strategy_norm.append(float(group_sender_strategy_norm))
        sender_degree.append(int(deg))
        local_investment_cv.append(float(invest_cv))
        local_cooperation_count.append(float(coop_count))

        weight_unit_investment.append(float(weights[0]) if np.isfinite(weights[0]) else float("nan"))
        weight_equal.append(float(weights[1]) if np.isfinite(weights[1]) else float("nan"))
        weight_self.append(float(weights[2]) if np.isfinite(weights[2]) else float("nan"))
        self_allocation.append(float(self_alloc))
        row_entropy.append(_entropy(alloc, eps=eps))

    with decision_records_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("decision_records.csv has no header")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise KeyError(f"decision_records.csv missing columns: {sorted(missing)}")

        current_key: tuple[int, int, int] | None = None

        group_sender_pool_grown = float("nan")
        group_sender_pool_raw_norm = float("nan")
        group_sender_resource_norm = float("nan")
        group_sender_strategy_norm = float("nan")
        receiver_ids: list[int] = []
        allocations: list[float] = []
        receiver_resources: list[float] = []
        receiver_strategy_norms: list[float] = []
        receiver_x_actuals: list[float] = []

        for row in reader:
            key = (_safe_int(row.get("episode")), _safe_int(row.get("step")), _safe_int(row.get("sender")))
            if current_key is None:
                current_key = key

            if key != current_key:
                _finalize_group(
                    current_key,
                    group_sender_pool_grown,
                    group_sender_pool_raw_norm,
                    group_sender_resource_norm,
                    group_sender_strategy_norm,
                    receiver_ids,
                    allocations,
                    receiver_resources,
                    receiver_strategy_norms,
                    receiver_x_actuals,
                )
                current_key = key
                receiver_ids = []
                allocations = []
                receiver_resources = []
                receiver_strategy_norms = []
                receiver_x_actuals = []

            # sender fields (repeated per receiver)
            group_sender_pool_grown = _safe_float(row.get("sender_pool_grown"))
            group_sender_pool_raw_norm = _safe_float(row.get("sender_pool_raw_norm"))
            group_sender_resource_norm = _safe_float(row.get("sender_resource_norm"))
            group_sender_strategy_norm = _safe_float(row.get("sender_strategy_norm"))

            receiver_ids.append(_safe_int(row.get("receiver")))
            allocations.append(_safe_float(row.get("allocation")))
            receiver_resources.append(_safe_float(row.get("receiver_resource")))
            receiver_strategy_norms.append(_safe_float(row.get("receiver_strategy_norm")))
            receiver_x_actuals.append(_safe_float(row.get("receiver_x_actual")))

        if current_key is not None:
            _finalize_group(
                current_key,
                group_sender_pool_grown,
                group_sender_pool_raw_norm,
                group_sender_resource_norm,
                group_sender_strategy_norm,
                receiver_ids,
                allocations,
                receiver_resources,
                receiver_strategy_norms,
                receiver_x_actuals,
            )

    return RowDataset(
        episode=np.asarray(episodes, dtype=np.int64),
        step=np.asarray(steps, dtype=np.int64),
        sender=np.asarray(senders, dtype=np.int64),
        pool_grown_over_upperbound=np.asarray(pool_grown_over_upperbound, dtype=np.float64),
        sender_resource_norm=np.asarray(sender_resource_norm, dtype=np.float64),
        sender_strategy_norm=np.asarray(sender_strategy_norm, dtype=np.float64),
        sender_degree=np.asarray(sender_degree, dtype=np.int64),
        local_investment_cv=np.asarray(local_investment_cv, dtype=np.float64),
        local_cooperation_count=np.asarray(local_cooperation_count, dtype=np.float64),
        weight_unit_investment=np.asarray(weight_unit_investment, dtype=np.float64),
        weight_equal=np.asarray(weight_equal, dtype=np.float64),
        weight_self=np.asarray(weight_self, dtype=np.float64),
        self_allocation=np.asarray(self_allocation, dtype=np.float64),
        row_entropy=np.asarray(row_entropy, dtype=np.float64),
    )


def _bin_summary_rows(
    dataset: RowDataset,
    *,
    split_var: str,
    target_vars: tuple[str, ...],
    bin_count: int,
) -> list[dict[str, Any]]:
    x = np.asarray(getattr(dataset, split_var), dtype=np.float64)
    finite_x = np.isfinite(x)
    if not np.any(finite_x):
        return []
    edges = _quantile_edges(x[finite_x], bin_count)
    if edges.size < 2:
        return []
    bin_ids, effective_bins = _assign_quantile_bins(x, edges)

    rows: list[dict[str, Any]] = []
    for b in range(effective_bins):
        mask = (bin_ids == b) & finite_x
        if not np.any(mask):
            continue
        left = float(edges[b])
        right = float(edges[b + 1])
        out: dict[str, Any] = {
            "split_variable": str(split_var),
            "bin_index": int(b),
            "bin_left": left,
            "bin_right": right,
            "count_rows": int(np.sum(mask)),
            "split_mean": float(np.mean(x[mask])),
        }
        for target in target_vars:
            y = np.asarray(getattr(dataset, target), dtype=np.float64)
            out[f"{target}_mean"] = float(np.nanmean(y[mask]))
        rows.append(out)
    return rows


def _variable_importance_rows(
    dataset: RowDataset,
    *,
    split_vars: tuple[str, ...],
    target_vars: tuple[str, ...],
    bin_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_var in split_vars:
        x = np.asarray(getattr(dataset, split_var), dtype=np.float64)
        finite_x = np.isfinite(x)
        if not np.any(finite_x):
            continue
        edges = _quantile_edges(x[finite_x], bin_count)
        if edges.size < 2:
            continue
        bin_ids, effective_bins = _assign_quantile_bins(x, edges)

        out: dict[str, Any] = {
            "split_variable": str(split_var),
            "sample_count": int(np.sum(finite_x)),
            "bin_count": int(effective_bins),
        }
        r2_values: list[float] = []
        for target in target_vars:
            y = np.asarray(getattr(dataset, target), dtype=np.float64)
            r2 = _variance_explained_r2(y, bin_ids, effective_bins)
            out[f"r2_{target}"] = float(r2) if np.isfinite(r2) else float("nan")
            if np.isfinite(r2):
                r2_values.append(float(r2))
        out["r2_mean"] = float(np.mean(r2_values)) if r2_values else float("nan")
        rows.append(out)

    rows.sort(key=lambda r: float(r.get("r2_mean", float("nan"))) if np.isfinite(float(r.get("r2_mean", float("nan")))) else -1.0, reverse=True)
    return rows


@dataclass
class TreeNode:
    depth: int
    mask: np.ndarray
    split_var: str | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def _best_split_for_node(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_leaf: int,
    threshold_candidates: int,
) -> tuple[float, float] | None:
    """Return (threshold, sse) for best split on one feature."""
    finite = np.isfinite(x) & np.isfinite(y)
    xf = x[finite]
    yf = y[finite]
    if yf.size < max(2 * int(min_leaf), 4):
        return None
    order = np.argsort(xf)
    xs = xf[order]
    ys = yf[order]

    # thresholds from quantiles of xs
    q = np.linspace(0.0, 1.0, num=max(4, int(threshold_candidates) + 2))[1:-1]
    thresholds = np.unique(np.quantile(xs, q))
    if thresholds.size == 0:
        return None

    cum_sum = np.cumsum(ys)
    cum_sumsq = np.cumsum(ys ** 2)
    total_sum = float(cum_sum[-1])
    total_sumsq = float(cum_sumsq[-1])
    n = int(ys.size)

    best_thr: float | None = None
    best_sse = float("inf")

    for thr in thresholds.tolist():
        idx = int(np.searchsorted(xs, thr, side="right"))
        n_left = idx
        n_right = n - idx
        if n_left < int(min_leaf) or n_right < int(min_leaf):
            continue
        sum_left = float(cum_sum[idx - 1]) if idx > 0 else 0.0
        sumsq_left = float(cum_sumsq[idx - 1]) if idx > 0 else 0.0
        sum_right = total_sum - sum_left
        sumsq_right = total_sumsq - sumsq_left

        sse_left = sumsq_left - (sum_left * sum_left) / float(max(n_left, 1))
        sse_right = sumsq_right - (sum_right * sum_right) / float(max(n_right, 1))
        sse = float(sse_left + sse_right)
        if sse < best_sse:
            best_sse = sse
            best_thr = float(thr)

    if best_thr is None or not np.isfinite(best_sse):
        return None
    return best_thr, best_sse


def _fit_shallow_tree(
    dataset: RowDataset,
    *,
    split_vars: tuple[str, ...],
    target: str,
    max_depth: int,
    min_leaf: int,
    threshold_candidates: int,
) -> TreeNode:
    y_all = np.asarray(getattr(dataset, target), dtype=np.float64)
    base_mask = np.isfinite(y_all)
    root = TreeNode(depth=0, mask=base_mask)

    def _node_sse(mask: np.ndarray) -> float:
        y = y_all[mask]
        if y.size < 2:
            return float("inf")
        mean = float(np.mean(y))
        return float(np.sum((y - mean) ** 2))

    queue: list[TreeNode] = [root]
    while queue:
        node = queue.pop(0)
        if node.depth >= int(max_depth):
            continue
        if int(np.sum(node.mask)) < max(2 * int(min_leaf), 4):
            continue

        parent_sse = _node_sse(node.mask)
        best_gain = 0.0
        best_var: str | None = None
        best_thr: float | None = None
        best_left_mask: np.ndarray | None = None
        best_right_mask: np.ndarray | None = None

        for var in split_vars:
            x_full = np.asarray(getattr(dataset, var), dtype=np.float64)
            x = x_full[node.mask]
            y = y_all[node.mask]
            result = _best_split_for_node(x, y, min_leaf=min_leaf, threshold_candidates=threshold_candidates)
            if result is None:
                continue
            thr, child_sse = result
            gain = float(parent_sse - child_sse)
            if gain > best_gain:
                best_gain = gain
                best_var = var
                best_thr = float(thr)
                left = node.mask & (np.asarray(getattr(dataset, var), dtype=np.float64) <= float(thr))
                right = node.mask & (np.asarray(getattr(dataset, var), dtype=np.float64) > float(thr))
                best_left_mask = left
                best_right_mask = right

        if best_var is None or best_thr is None or best_left_mask is None or best_right_mask is None:
            continue

        node.split_var = best_var
        node.threshold = float(best_thr)
        node.left = TreeNode(depth=node.depth + 1, mask=best_left_mask)
        node.right = TreeNode(depth=node.depth + 1, mask=best_right_mask)
        queue.append(node.left)
        queue.append(node.right)

    return root


def _tree_leaf_rows(
    dataset: RowDataset,
    *,
    root: TreeNode,
    primary_targets: tuple[str, ...],
    leaf_extra_metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    leaves: list[tuple[str, TreeNode]] = []

    def _walk(node: TreeNode, path: str) -> None:
        if node.left is None or node.right is None or node.split_var is None or node.threshold is None:
            leaves.append((path, node))
            return
        _walk(node.left, f"{path} & {node.split_var}<= {node.threshold:.6g}")
        _walk(node.right, f"{path} & {node.split_var}> {node.threshold:.6g}")

    _walk(root, "ALL")

    rows: list[dict[str, Any]] = []
    for leaf_index, (path, node) in enumerate(leaves):
        mask = node.mask
        row: dict[str, Any] = {
            "leaf_index": int(leaf_index),
            "rule": str(path),
            "count_rows": int(np.sum(mask)),
        }
        for target in primary_targets:
            y = np.asarray(getattr(dataset, target), dtype=np.float64)
            row[f"{target}_mean"] = float(np.nanmean(y[mask])) if np.any(mask) else float("nan")
        for metric in leaf_extra_metrics:
            y = np.asarray(getattr(dataset, metric), dtype=np.float64)
            row[f"{metric}_mean"] = float(np.nanmean(y[mask])) if np.any(mask) else float("nan")
        rows.append(row)

    # largest leaves first
    rows.sort(key=lambda r: int(r.get("count_rows", 0)), reverse=True)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="State-conditioned split (sender-row) analysis.")
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(SCRIPT_DEFAULTS["analysis_dir"]),
        help="Folder that contains decision_records.csv (from the policy analysis scripts).",
    )
    parser.add_argument(
        "--decision-records",
        type=str,
        default=SCRIPT_DEFAULTS["decision_records"],
        help="Optional explicit path to decision_records.csv (or its containing folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=SCRIPT_DEFAULTS["output_dir"],
        help="Output directory (default: <analysis-dir>/state_conditioned_split).",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["bin_count"]),
        help="Quantile bin count K for univariate splits.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=float(SCRIPT_DEFAULTS["eps"]),
        help="Small epsilon for normalization and CV.",
    )
    parser.add_argument(
        "--tree-target",
        type=str,
        default=str(SCRIPT_DEFAULTS["tree_target"]),
        choices=("weight_unit_investment", "weight_equal", "self_allocation"),
        help="Which target to optimize when fitting the shallow tree.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=int(SCRIPT_DEFAULTS["tree_max_depth"]),
        help="Max depth of the shallow tree (<=2 recommended).",
    )
    parser.add_argument(
        "--tree-min-leaf",
        type=int,
        default=int(SCRIPT_DEFAULTS["tree_min_leaf"]),
        help="Minimum rows per leaf.",
    )
    parser.add_argument(
        "--tree-threshold-candidates",
        type=int,
        default=int(SCRIPT_DEFAULTS["tree_threshold_candidates"]),
        help="How many quantile-based threshold candidates per feature.",
    )
    parser.add_argument(
        "--save-row-dataset",
        type=_parse_toggle_arg,
        default=False,
        help="Whether to also write the aggregated sender-row dataset as CSV (can be large).",
    )
    return parser.parse_args(argv)


def _row_dataset_to_csv_rows(dataset: RowDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(dataset.episode.size)
    for i in range(n):
        rows.append(
            {
                "episode": int(dataset.episode[i]),
                "step": int(dataset.step[i]),
                "sender": int(dataset.sender[i]),
                "pool_grown_over_upperbound": float(dataset.pool_grown_over_upperbound[i]),
                "sender_resource_norm": float(dataset.sender_resource_norm[i]),
                "sender_strategy_norm": float(dataset.sender_strategy_norm[i]),
                "sender_degree": int(dataset.sender_degree[i]),
                "local_investment_cv": float(dataset.local_investment_cv[i]),
                "local_cooperation_count": float(dataset.local_cooperation_count[i]),
                "weight_unit_investment": float(dataset.weight_unit_investment[i]),
                "weight_equal": float(dataset.weight_equal[i]),
                "weight_self": float(dataset.weight_self[i]),
                "self_allocation": float(dataset.self_allocation[i]),
                "row_entropy": float(dataset.row_entropy[i]),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis_dir = Path(args.analysis_dir)
    decision_records_path = _find_decision_records_path(analysis_dir, args.decision_records)

    output_dir = Path(args.output_dir) if args.output_dir else (decision_records_path.parent / "state_conditioned_split")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[state-split] decision_records: {decision_records_path}")
    print(f"[state-split] output_dir: {output_dir}")

    receiver_degrees = _infer_receiver_degrees_from_first_snapshot(decision_records_path)
    if receiver_degrees.size:
        print(
            f"[state-split] inferred receiver degrees from first snapshot: N={receiver_degrees.size}, "
            f"mean={float(np.mean(receiver_degrees)):.3g}, min={int(np.min(receiver_degrees))}, max={int(np.max(receiver_degrees))}"
        )
    else:
        print("[state-split] WARNING: could not infer receiver degrees; falling back to investment baseline")

    dataset = _build_row_dataset_from_decision_records(
        decision_records_path,
        receiver_degrees=receiver_degrees,
        eps=float(args.eps),
    )
    print(f"[state-split] aggregated sender-rows: {int(dataset.episode.size)}")

    if bool(args.save_row_dataset):
        _write_csv(output_dir / "sender_row_dataset.csv", _row_dataset_to_csv_rows(dataset))

    # 1) bin summaries
    for split_var in DEFAULT_SPLIT_VARS:
        summary_rows = _bin_summary_rows(
            dataset,
            split_var=split_var,
            target_vars=DEFAULT_TARGET_VARS,
            bin_count=int(args.bin_count),
        )
        if summary_rows:
            _write_csv(output_dir / f"bin_summary_{split_var}.csv", summary_rows)

    # 2) importance
    importance_rows = _variable_importance_rows(
        dataset,
        split_vars=DEFAULT_SPLIT_VARS,
        target_vars=DEFAULT_TARGET_VARS,
        bin_count=int(args.bin_count),
    )
    _write_csv(output_dir / "variable_importance.csv", importance_rows)

    # 3) shallow tree
    tree = _fit_shallow_tree(
        dataset,
        split_vars=DEFAULT_SPLIT_VARS,
        target=str(args.tree_target),
        max_depth=min(2, int(args.tree_max_depth)),
        min_leaf=max(2, int(args.tree_min_leaf)),
        threshold_candidates=max(8, int(args.tree_threshold_candidates)),
    )
    leaf_rows = _tree_leaf_rows(
        dataset,
        root=tree,
        primary_targets=DEFAULT_TARGET_VARS,
        leaf_extra_metrics=("row_entropy",),
    )
    _write_csv(output_dir / "shallow_tree_rules.csv", leaf_rows)

    print("[state-split] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
