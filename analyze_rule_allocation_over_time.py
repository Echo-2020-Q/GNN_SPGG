from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

from Project1.env import SPGGEnv
from Project1.policies.rule_based import ProportionalContributionPolicy, UniformAllocationPolicy

from analyze_gnn_policy_decisions import (
    TOPOLOGY_LABELS,
    EpisodeSummary,
    PolicyArrays,
    Snapshot,
    _load_json,
    _observation_copy,
    build_env_config_from_spec,
    build_graph_from_spec,
    build_spec_for_topology,
    load_actor_from_run_dir,
    parse_topology_list,
    resolve_reference_mean_degree,
    run_deterministic_rollouts,
    write_csv as _write_csv_raw,
)
from analyze_gnn_policy_mechanism import (
    _assign_degree_bin_fields,
    _build_degree_bin_spec,
    _build_row_baselines,
    _fit_labor_equal_lambda,
    _fit_three_way_mechanism_weights,
)


SCRIPT_DEFAULTS = {
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    "checkpoint_name": "best_eval.pt",
    "output_dir": None,
    "output_subdir_name": "0421_rule_allocation_over_time_ba_degree_conditioned_sparse",
    "topologies": "ba",
    "episodes": 5,
    "max_steps": 200,
    "episode_length_override": 200,
    "env_r_override": None,
    "env_p_c_override": None,
    "env_beta_override": None,
    "env_strategy_update_rule_override": None,
    "env_p_max_override": None,
    "env_initial_resource_override": None,
    "env_initial_cooperation_prob_override": None,
    "seed": None,
    "ego_resource_x_mode": "absolute",
    "ego_resource_normalizer": None,
    "adaptive_policy_mode": "continuous2d_suite",
    "adaptive_unitinv_equal_profile_csv": "0419_regu_ba_sender_allocation_gini_over_time/regular/agent_mix_vs_ego_mean.csv",
    "adaptive_unitinv_equal_knots": None,
    "adaptive_unitinv_equal_smooth_window": 1,
    "adaptive_reference_bin_count": 16,
    "early_alignment_max_step": 15,
    "adaptive_ego_resource_profile_mode": "continuous",
    "adaptive_ego_resource_bin_count": 12,
    "spline_high_knot_style": "sparse",
    "tail_cap_mode": "none",
    "tail_cap_resource": 430.0,
    "tail_cap_soft_scale": 0.25,
    "spline_l2_penalty": 1e-3,
    "adaptive_conditioning_mode": "none",
    "adaptive_conditioned_profile_family": "three_way",
    "conditioned_degree_bin_mode": "quantile",
    "conditioned_degree_bin_count": 4,
    "conditioned_degree_bin_min_count": 100,
    "conditioned_resource_bin_count": 4,
    "continuous_second_variable": "pool_grown_over_upperbound",
    "continuous_basis_mode": "linear_interaction",
    "continuous_x1_hinge_count": 3,
    "continuous_x2_hinge_count": 3,
    "continuous_surface_grid_size": 48,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


class ResourceProportionalAllocationPolicy:
    """Allocate each sender's pool in proportion to receivers' current resource stocks."""

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)

        allocation = np.zeros(local_mask.shape, dtype=np.float64)
        weights = local_mask.astype(np.float64) * resources[None, :]
        row_sums = weights.sum(axis=1, keepdims=True)

        positive_rows = row_sums.squeeze(-1) > 1e-12
        if np.any(positive_rows):
            allocation[positive_rows] = weights[positive_rows] / row_sums[positive_rows]

        if np.any(~positive_rows):
            uniform_weights = local_mask[~positive_rows].astype(np.float64)
            uniform_sums = uniform_weights.sum(axis=1, keepdims=True)
            allocation[~positive_rows] = uniform_weights / uniform_sums

        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)


class UnitInvestmentProportionalAllocationPolicy(ProportionalContributionPolicy):
    """Allocate each sender's pool in proportion to receivers' current unit_investment values."""


class AdaptiveUnitInvestmentEqualAllocationPolicy:
    """Blend unit-investment proportional and equal allocation with a sender-specific w(ego_mean_resource)."""

    def __init__(
        self,
        ego_mean_resource_knots: Sequence[float],
        weight_knots: Sequence[float],
        *,
        profile_source: str,
    ):
        resource_knots = np.asarray(ego_mean_resource_knots, dtype=np.float64).reshape(-1)
        weights = np.asarray(weight_knots, dtype=np.float64).reshape(-1)
        if resource_knots.size < 2:
            raise ValueError("adaptive unit-investment/equal mix requires at least two resource knots.")
        if resource_knots.size != weights.size:
            raise ValueError("adaptive unit-investment/equal mix requires matching knot counts.")
        if not np.all(np.isfinite(resource_knots)):
            raise ValueError("adaptive unit-investment/equal mix resource knots must be finite.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("adaptive unit-investment/equal mix weight knots must be finite.")

        order = np.argsort(resource_knots)
        sorted_resources = resource_knots[order]
        sorted_weights = np.clip(weights[order], 0.0, 1.0)

        unique_resources: list[float] = []
        unique_weights: list[float] = []
        for resource_value, weight_value in zip(sorted_resources.tolist(), sorted_weights.tolist()):
            if unique_resources and abs(resource_value - unique_resources[-1]) <= 1e-12:
                unique_weights[-1] = 0.5 * (unique_weights[-1] + float(weight_value))
            else:
                unique_resources.append(float(resource_value))
                unique_weights.append(float(weight_value))
        if len(unique_resources) < 2:
            raise ValueError("adaptive unit-investment/equal mix knots collapse to fewer than two unique resources.")

        self.ego_mean_resource_knots = np.asarray(unique_resources, dtype=np.float64)
        self.weight_knots = np.asarray(unique_weights, dtype=np.float64)
        self.profile_source = str(profile_source)
        self._unit_investment = UnitInvestmentProportionalAllocationPolicy()
        self._equal = UniformAllocationPolicy()

    def _weight_for_ego_mean_resource(self, ego_mean_resource: float) -> float:
        return float(
            np.interp(
                float(ego_mean_resource),
                self.ego_mean_resource_knots,
                self.weight_knots,
                left=float(self.weight_knots[0]),
                right=float(self.weight_knots[-1]),
            )
        )

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)
        unit_investment_allocation = np.asarray(self._unit_investment.allocate(observation), dtype=np.float64)
        equal_allocation = np.asarray(self._equal.allocate(observation), dtype=np.float64)

        sender_weights = np.zeros(local_mask.shape[0], dtype=np.float64)
        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            sender_weights[sender] = self._weight_for_ego_mean_resource(float(np.mean(resources[valid_receivers])))

        allocation = (
            sender_weights[:, None] * unit_investment_allocation
            + (1.0 - sender_weights[:, None]) * equal_allocation
        )
        allocation = np.where(local_mask, allocation, 0.0)
        row_sums = allocation.sum(axis=1, keepdims=True)

        positive_rows = row_sums.squeeze(-1) > 1e-12
        if np.any(positive_rows):
            allocation[positive_rows] = allocation[positive_rows] / row_sums[positive_rows]
        if np.any(~positive_rows):
            allocation[~positive_rows] = equal_allocation[~positive_rows]
        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)

    def weights_for_ego_mean_resource(self, ego_mean_resource: float) -> tuple[float, float, float]:
        weight_unit_investment = float(self._weight_for_ego_mean_resource(float(ego_mean_resource)))
        return weight_unit_investment, float(1.0 - weight_unit_investment), 0.0

    def profile_summary(self) -> dict[str, Any]:
        return {
            "mode": "lookup_2way",
            "profile_source": self.profile_source,
            "ego_mean_resource_knots": [float(value) for value in self.ego_mean_resource_knots.tolist()],
            "weight_unit_investment_knots": [float(value) for value in self.weight_knots.tolist()],
            "weight_equal_knots": [float(1.0 - value) for value in self.weight_knots.tolist()],
        }


class SmoothLogEgoResourceMixtureAllocationPolicy:
    """Map ego_mean_resource to a simplex over unit-investment / equal / self using smooth or spline logits."""

    def __init__(
        self,
        *,
        mode: str,
        coeffs_unit_investment: Sequence[float],
        coeffs_self: Sequence[float] | None = None,
        profile_source: str,
        fit_summary: Mapping[str, Any] | None = None,
        spline_knots_resource: Sequence[float] | None = None,
        tail_cap_mode: str = "none",
        tail_cap_resource: float | None = None,
        tail_cap_soft_scale: float = 0.25,
        ego_resource_x_mode: str = "absolute",
        ego_resource_normalizer: float | None = None,
    ):
        self.mode = str(mode)
        if self.mode not in {"smooth_2way", "smooth_3way", "spline_3way"}:
            raise ValueError(f"Unsupported smooth mixture mode: {self.mode}")
        self.coeffs_unit_investment = np.asarray(coeffs_unit_investment, dtype=np.float64).reshape(-1)
        self.spline_knots_resource = (
            np.asarray([], dtype=np.float64)
            if spline_knots_resource is None
            else np.asarray(spline_knots_resource, dtype=np.float64).reshape(-1)
        )
        self.spline_knots_log = np.log1p(np.clip(self.spline_knots_resource, 0.0, None))
        expected_basis_size = 3 if self.mode in {"smooth_2way", "smooth_3way"} else (2 + self.spline_knots_log.size)
        if self.coeffs_unit_investment.size != expected_basis_size:
            raise ValueError(
                f"{self.mode} expects {expected_basis_size} coefficients for unit-investment, got {self.coeffs_unit_investment.size}."
            )
        if self.mode in {"smooth_3way", "spline_3way"}:
            if coeffs_self is None:
                raise ValueError(f"{self.mode} requires self coefficients.")
            self.coeffs_self = np.asarray(coeffs_self, dtype=np.float64).reshape(-1)
            if self.coeffs_self.size != expected_basis_size:
                raise ValueError(
                    f"{self.mode} expects {expected_basis_size} coefficients for self, got {self.coeffs_self.size}."
                )
        else:
            self.coeffs_self = np.zeros(expected_basis_size, dtype=np.float64)

        self.profile_source = str(profile_source)
        self.fit_summary_dict = {} if fit_summary is None else dict(fit_summary)
        self.tail_cap_mode = str(tail_cap_mode)
        if self.tail_cap_mode not in {"none", "hard", "soft"}:
            raise ValueError(f"Unsupported tail cap mode: {tail_cap_mode}")
        self.tail_cap_resource = (
            None if tail_cap_resource is None else float(max(0.0, float(tail_cap_resource)))
        )
        self.tail_cap_soft_scale = float(np.clip(float(tail_cap_soft_scale), 0.0, 1.0))
        self.ego_resource_x_mode = str(ego_resource_x_mode)
        if self.ego_resource_x_mode not in {"absolute", "normalized"}:
            raise ValueError(f"Unsupported ego resource x mode: {ego_resource_x_mode}")
        self.ego_resource_normalizer = (
            None if ego_resource_normalizer is None else float(ego_resource_normalizer)
        )
        self._unit_investment = UnitInvestmentProportionalAllocationPolicy()
        self._equal = UniformAllocationPolicy()

    @staticmethod
    def _smooth_basis(log_ego_mean_resource: float) -> np.ndarray:
        x = float(log_ego_mean_resource)
        return np.asarray([1.0, x, x * x], dtype=np.float64)

    def _basis(self, log_ego_mean_resource: float) -> np.ndarray:
        if self.mode in {"smooth_2way", "smooth_3way"}:
            return self._smooth_basis(log_ego_mean_resource)
        x = float(log_ego_mean_resource)
        basis_values = [1.0, x]
        basis_values.extend(float(max(0.0, x - knot)) for knot in self.spline_knots_log.tolist())
        return np.asarray(basis_values, dtype=np.float64)

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = float(np.clip(value, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-clipped)))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        shifted = logits - float(np.max(logits))
        exp_values = np.exp(np.clip(shifted, -60.0, 60.0))
        total = float(np.sum(exp_values))
        if total <= 1e-12:
            return np.full(logits.shape, 1.0 / float(max(logits.size, 1)), dtype=np.float64)
        return exp_values / total

    def effective_ego_mean_resource(self, ego_mean_resource: float) -> float:
        safe_resource = max(0.0, float(ego_mean_resource))
        cap_value = None if self.tail_cap_resource is None else float(self.tail_cap_resource)
        if self.tail_cap_mode == "none" or cap_value is None:
            return float(safe_resource)
        if self.tail_cap_mode == "hard":
            return float(min(safe_resource, cap_value))
        if safe_resource <= cap_value:
            return float(safe_resource)
        return float(cap_value + self.tail_cap_soft_scale * (safe_resource - cap_value))

    def profile_x_value_for_ego_mean_resource(self, ego_mean_resource: float) -> float:
        effective_resource = self.effective_ego_mean_resource(float(ego_mean_resource))
        return _normalize_ego_resource_value(
            effective_resource,
            x_mode=str(self.ego_resource_x_mode),
            normalizer=self.ego_resource_normalizer,
        )

    def weights_for_ego_mean_resource(self, ego_mean_resource: float) -> tuple[float, float, float]:
        profile_x_value = self.profile_x_value_for_ego_mean_resource(float(ego_mean_resource))
        log_resource = float(np.log1p(profile_x_value))
        basis = self._basis(log_resource)

        if self.mode == "smooth_2way":
            unit_logit = float(np.dot(self.coeffs_unit_investment, basis))
            weight_unit_investment = self._sigmoid(unit_logit)
            return weight_unit_investment, float(1.0 - weight_unit_investment), 0.0

        unit_logit = float(np.dot(self.coeffs_unit_investment, basis))
        self_logit = float(np.dot(self.coeffs_self, basis))
        weights = self._softmax(np.asarray([unit_logit, 0.0, self_logit], dtype=np.float64))
        return float(weights[0]), float(weights[1]), float(weights[2])

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)
        unit_investment_allocation = np.asarray(self._unit_investment.allocate(observation), dtype=np.float64)
        equal_allocation = np.asarray(self._equal.allocate(observation), dtype=np.float64)
        self_allocation = np.zeros(local_mask.shape, dtype=np.float64)

        for sender in range(local_mask.shape[0]):
            if local_mask[sender, sender]:
                self_allocation[sender, sender] = 1.0
            else:
                valid_receivers = np.flatnonzero(local_mask[sender])
                if valid_receivers.size > 0:
                    self_allocation[sender, valid_receivers] = 1.0 / float(valid_receivers.size)

        allocation = np.zeros(local_mask.shape, dtype=np.float64)
        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            ego_mean_resource = float(np.mean(resources[valid_receivers]))
            weight_unit_investment, weight_equal, weight_self = self.weights_for_ego_mean_resource(ego_mean_resource)
            allocation[sender] = (
                weight_unit_investment * unit_investment_allocation[sender]
                + weight_equal * equal_allocation[sender]
                + weight_self * self_allocation[sender]
            )

        allocation = np.where(local_mask, allocation, 0.0)
        row_sums = allocation.sum(axis=1, keepdims=True)
        positive_rows = row_sums.squeeze(-1) > 1e-12
        if np.any(positive_rows):
            allocation[positive_rows] = allocation[positive_rows] / row_sums[positive_rows]
        if np.any(~positive_rows):
            allocation[~positive_rows] = equal_allocation[~positive_rows]
        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)

    def profile_summary(self) -> dict[str, Any]:
        if self.mode in {"smooth_2way", "smooth_3way"}:
            basis_terms = ["1", "x", "x^2"]
            input_transform = (
                "log1p(ego_mean_resource)"
                if self.ego_resource_x_mode == "absolute"
                else "log1p(ego_mean_resource / theoretical_max)"
            )
        else:
            basis_terms = ["1", "x"] + [f"relu(x-log1p({float(knot):.3f}))" for knot in self.spline_knots_resource.tolist()]
            input_transform = (
                "log1p(ego_mean_resource) with linear spline hinges"
                if self.ego_resource_x_mode == "absolute"
                else "log1p(ego_mean_resource / theoretical_max) with linear spline hinges"
            )
        summary = {
            "mode": self.mode,
            "profile_source": self.profile_source,
            "input_transform": input_transform,
            "ego_resource_x_mode": str(self.ego_resource_x_mode),
            "ego_resource_normalizer": (
                None if self.ego_resource_normalizer is None else float(self.ego_resource_normalizer)
            ),
            "basis_terms": basis_terms,
            "coeffs_unit_investment": [float(value) for value in self.coeffs_unit_investment.tolist()],
            "coeffs_equal": [0.0 for _ in range(int(self.coeffs_unit_investment.size))] if self.mode in {"smooth_3way", "spline_3way"} else None,
            "coeffs_self": [float(value) for value in self.coeffs_self.tolist()] if self.mode in {"smooth_3way", "spline_3way"} else None,
            "spline_knots_resource": [float(value) for value in self.spline_knots_resource.tolist()] if self.mode == "spline_3way" else None,
            "tail_cap_mode": str(self.tail_cap_mode),
            "tail_cap_resource": (
                None if self.tail_cap_resource is None else float(self.tail_cap_resource)
            ),
            "tail_cap_soft_scale": float(self.tail_cap_soft_scale),
        }
        summary.update(self.fit_summary_dict)
        return summary


class BinnedEgoResourceMixtureAllocationPolicy:
    """Map ego_mean_resource to a simplex over unit-investment / equal / self using discrete resource bins."""

    def __init__(
        self,
        *,
        mode: str,
        bin_edges_resource: Sequence[float],
        bin_weight_rows: Sequence[Sequence[float]],
        profile_source: str,
        fit_summary: Mapping[str, Any] | None = None,
        tail_cap_mode: str = "none",
        tail_cap_resource: float | None = None,
        tail_cap_soft_scale: float = 0.25,
        ego_resource_x_mode: str = "absolute",
        ego_resource_normalizer: float | None = None,
    ):
        self.mode = str(mode)
        if self.mode not in {"binned_3way", "binned_unitself"}:
            raise ValueError(f"Unsupported binned mixture mode: {mode}")
        self.bin_edges_resource = np.asarray(bin_edges_resource, dtype=np.float64).reshape(-1)
        if self.bin_edges_resource.size < 2:
            raise ValueError("Binned ego-resource policy requires at least two bin edges.")
        self.bin_weight_rows = _safe_normalize_rows(np.asarray(bin_weight_rows, dtype=np.float64))
        if self.bin_weight_rows.shape[0] != self.bin_edges_resource.size - 1 or self.bin_weight_rows.shape[1] != 3:
            raise ValueError("Binned ego-resource policy requires one 3-weight row per bin interval.")
        self.profile_source = str(profile_source)
        self.fit_summary_dict = {} if fit_summary is None else dict(fit_summary)
        self.tail_cap_mode = str(tail_cap_mode)
        if self.tail_cap_mode not in {"none", "hard", "soft"}:
            raise ValueError(f"Unsupported tail cap mode: {tail_cap_mode}")
        self.tail_cap_resource = None if tail_cap_resource is None else float(max(0.0, float(tail_cap_resource)))
        self.tail_cap_soft_scale = float(np.clip(float(tail_cap_soft_scale), 0.0, 1.0))
        self.ego_resource_x_mode = str(ego_resource_x_mode)
        if self.ego_resource_x_mode not in {"absolute", "normalized"}:
            raise ValueError(f"Unsupported ego resource x mode: {ego_resource_x_mode}")
        self.ego_resource_normalizer = (
            None if ego_resource_normalizer is None else float(ego_resource_normalizer)
        )
        self._unit_investment = UnitInvestmentProportionalAllocationPolicy()
        self._equal = UniformAllocationPolicy()

    def effective_ego_mean_resource(self, ego_mean_resource: float) -> float:
        safe_resource = max(0.0, float(ego_mean_resource))
        cap_value = None if self.tail_cap_resource is None else float(self.tail_cap_resource)
        if self.tail_cap_mode == "none" or cap_value is None:
            return float(safe_resource)
        if self.tail_cap_mode == "hard":
            return float(min(safe_resource, cap_value))
        if safe_resource <= cap_value:
            return float(safe_resource)
        return float(cap_value + self.tail_cap_soft_scale * (safe_resource - cap_value))

    def profile_x_value_for_ego_mean_resource(self, ego_mean_resource: float) -> float:
        effective_resource = self.effective_ego_mean_resource(float(ego_mean_resource))
        return _normalize_ego_resource_value(
            effective_resource,
            x_mode=str(self.ego_resource_x_mode),
            normalizer=self.ego_resource_normalizer,
        )

    def _resolve_bin_index(self, effective_ego_mean_resource: float) -> int:
        safe_value = float(max(0.0, effective_ego_mean_resource))
        if safe_value <= float(self.bin_edges_resource[0]):
            return 0
        if safe_value >= float(self.bin_edges_resource[-1]):
            return int(self.bin_weight_rows.shape[0] - 1)
        index = int(np.searchsorted(self.bin_edges_resource, safe_value, side="right") - 1)
        return int(np.clip(index, 0, self.bin_weight_rows.shape[0] - 1))

    def weights_for_ego_mean_resource(self, ego_mean_resource: float) -> tuple[float, float, float]:
        profile_x_value = self.profile_x_value_for_ego_mean_resource(float(ego_mean_resource))
        row = self.bin_weight_rows[self._resolve_bin_index(profile_x_value)]
        return float(row[0]), float(row[1]), float(row[2])

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)
        unit_investment_allocation = np.asarray(self._unit_investment.allocate(observation), dtype=np.float64)
        equal_allocation = np.asarray(self._equal.allocate(observation), dtype=np.float64)
        self_allocation = np.zeros(local_mask.shape, dtype=np.float64)

        for sender in range(local_mask.shape[0]):
            if local_mask[sender, sender]:
                self_allocation[sender, sender] = 1.0
            else:
                valid_receivers = np.flatnonzero(local_mask[sender])
                if valid_receivers.size > 0:
                    self_allocation[sender, valid_receivers] = 1.0 / float(valid_receivers.size)

        allocation = np.zeros(local_mask.shape, dtype=np.float64)
        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            ego_mean_resource = float(np.mean(resources[valid_receivers]))
            weight_unit_investment, weight_equal, weight_self = self.weights_for_ego_mean_resource(ego_mean_resource)
            allocation[sender] = (
                weight_unit_investment * unit_investment_allocation[sender]
                + weight_equal * equal_allocation[sender]
                + weight_self * self_allocation[sender]
            )

        allocation = np.where(local_mask, allocation, 0.0)
        row_sums = allocation.sum(axis=1, keepdims=True)
        positive_rows = row_sums.squeeze(-1) > 1e-12
        if np.any(positive_rows):
            allocation[positive_rows] = allocation[positive_rows] / row_sums[positive_rows]
        if np.any(~positive_rows):
            allocation[~positive_rows] = equal_allocation[~positive_rows]
        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)

    def profile_summary(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "profile_source": self.profile_source,
            "input_transform": (
                "ego_mean_resource discrete bins"
                if self.ego_resource_x_mode == "absolute"
                else "ego_mean_resource / theoretical_max discrete bins"
            ),
            "ego_resource_x_mode": str(self.ego_resource_x_mode),
            "ego_resource_normalizer": (
                None if self.ego_resource_normalizer is None else float(self.ego_resource_normalizer)
            ),
            "bin_edges_resource": [float(value) for value in self.bin_edges_resource.tolist()],
            "bin_weight_rows": [[float(item) for item in row] for row in self.bin_weight_rows.tolist()],
            "tail_cap_mode": str(self.tail_cap_mode),
            "tail_cap_resource": None if self.tail_cap_resource is None else float(self.tail_cap_resource),
            "tail_cap_soft_scale": float(self.tail_cap_soft_scale),
        }
        payload.update(self.fit_summary_dict)
        return payload


class ConditionedSplineMixtureAllocationPolicy:
    """Select a bin-specific 1D ego-resource profile before mixing unit-investment / equal / self."""

    def __init__(
        self,
        *,
        conditioning_mode: str,
        profile_family: str,
        condition_spec: Mapping[str, Any],
        policies_by_bin: Mapping[str, Any],
        profile_source: str,
        fit_summary: Mapping[str, Any] | None = None,
    ):
        self.conditioning_mode = str(conditioning_mode)
        self.profile_family = str(profile_family)
        self.condition_spec = dict(condition_spec)
        self.policies_by_bin = {str(key): value for key, value in dict(policies_by_bin).items()}
        self.profile_source = str(profile_source)
        self.fit_summary_dict = {} if fit_summary is None else dict(fit_summary)
        self._unit_investment = UnitInvestmentProportionalAllocationPolicy()
        self._equal = UniformAllocationPolicy()
        self._fallback_bin = None
        bins = list(self.condition_spec.get("bins", []))
        if bins:
            self._fallback_bin = str(bins[-1]["bin_label"])

    def _resolve_sender_bin_label(self, observation: Mapping[str, np.ndarray], sender: int) -> str:
        if self.conditioning_mode == "sender_degree_bin":
            degrees = np.asarray(observation["degrees"], dtype=np.float64)
            sender_degree = int(round(float(degrees[sender])))
            value_to_bin = dict(self.condition_spec.get("value_to_bin", {}))
            bin_info = value_to_bin.get(sender_degree)
            if bin_info is None:
                return str(self._fallback_bin)
            return str(bin_info["bin_label"])
        if self.conditioning_mode == "sender_resource_norm_bin":
            resource_norm = np.asarray(observation["resource_norm"], dtype=np.float64)
            bin_info = _lookup_continuous_bin_info(self.condition_spec, float(resource_norm[sender]))
            if bin_info is None:
                return str(self._fallback_bin)
            return str(bin_info["bin_label"])
        raise ValueError(f"Unsupported conditioning mode: {self.conditioning_mode}")

    def weights_for_sender(
        self,
        observation: Mapping[str, np.ndarray],
        sender: int,
        ego_mean_resource: float,
    ) -> tuple[float, float, float, str]:
        bin_label = self._resolve_sender_bin_label(observation, sender)
        policy = self.policies_by_bin.get(str(bin_label))
        if policy is None:
            raise KeyError(f"No conditioned spline policy found for bin {bin_label}.")
        weight_unit_investment, weight_equal, weight_self = policy.weights_for_ego_mean_resource(
            float(ego_mean_resource)
        )
        return weight_unit_investment, weight_equal, weight_self, str(bin_label)

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)
        unit_investment_allocation = np.asarray(self._unit_investment.allocate(observation), dtype=np.float64)
        equal_allocation = np.asarray(self._equal.allocate(observation), dtype=np.float64)
        self_allocation = np.zeros(local_mask.shape, dtype=np.float64)

        for sender in range(local_mask.shape[0]):
            if local_mask[sender, sender]:
                self_allocation[sender, sender] = 1.0
            else:
                valid_receivers = np.flatnonzero(local_mask[sender])
                if valid_receivers.size > 0:
                    self_allocation[sender, valid_receivers] = 1.0 / float(valid_receivers.size)

        allocation = np.zeros(local_mask.shape, dtype=np.float64)
        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            ego_mean_resource = float(np.mean(resources[valid_receivers]))
            weight_unit_investment, weight_equal, weight_self, _ = self.weights_for_sender(
                observation,
                sender,
                ego_mean_resource,
            )
            allocation[sender] = (
                weight_unit_investment * unit_investment_allocation[sender]
                + weight_equal * equal_allocation[sender]
                + weight_self * self_allocation[sender]
            )

        allocation = np.where(local_mask, allocation, 0.0)
        row_sums = allocation.sum(axis=1, keepdims=True)
        positive_rows = row_sums.squeeze(-1) > 1e-12
        if np.any(positive_rows):
            allocation[positive_rows] = allocation[positive_rows] / row_sums[positive_rows]
        if np.any(~positive_rows):
            allocation[~positive_rows] = equal_allocation[~positive_rows]
        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)

    def profile_summary(self) -> dict[str, Any]:
        summary = {
            "mode": "conditioned_spline_profile",
            "conditioning_mode": self.conditioning_mode,
            "profile_family": self.profile_family,
            "profile_source": self.profile_source,
            "condition_spec": {
                key: value
                for key, value in self.condition_spec.items()
                if key != "value_to_bin"
            },
            "bin_profiles": {
                str(bin_label): dict(policy.profile_summary()) for bin_label, policy in self.policies_by_bin.items()
            },
        }
        summary.update(self.fit_summary_dict)
        return summary


def _extract_continuous_second_value_from_row(
    row: Mapping[str, Any],
    second_variable_name: str,
) -> float:
    variable_name = str(second_variable_name)
    if variable_name == "pool_grown_over_upperbound":
        return float(row.get("pool_grown_over_upperbound", float("nan")))
    if variable_name == "sender_pool_raw_norm":
        return float(row.get("sender_pool_raw_norm", float("nan")))
    raise ValueError(f"Unsupported continuous second variable: {second_variable_name}")


def _extract_continuous_state_from_observation(
    observation: Mapping[str, np.ndarray],
    sender: int,
    second_variable_name: str,
) -> tuple[float, float, float]:
    local_mask = np.asarray(observation["local_mask"], dtype=bool)
    resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)
    valid_receivers = np.flatnonzero(local_mask[sender])
    if valid_receivers.size == 0:
        return 0.0, 0.0, 0.0

    ego_mean_resource = float(np.mean(resources[valid_receivers]))
    x1_value = float(np.log1p(max(0.0, ego_mean_resource)))

    variable_name = str(second_variable_name)
    if variable_name == "pool_grown_over_upperbound":
        pool_grown = float(np.asarray(observation["pool_grown"], dtype=np.float64)[sender])
        pool_upperbound = float(np.asarray(observation["p_max"], dtype=np.float64)[sender])
        x2_value = pool_grown / pool_upperbound if pool_upperbound > 1e-12 else 0.0
    elif variable_name == "sender_pool_raw_norm":
        x2_value = float(np.asarray(observation["pool_raw_norm"], dtype=np.float64)[sender])
    else:
        raise ValueError(f"Unsupported continuous second variable: {second_variable_name}")
    return x1_value, float(x2_value), ego_mean_resource


def _build_hinge_knots(values: Sequence[float], knot_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size < max(8, int(knot_count) * 4):
        return np.asarray([], dtype=np.float64)
    if int(knot_count) <= 0:
        return np.asarray([], dtype=np.float64)
    quantiles = np.linspace(0.15, 0.85, num=max(1, int(knot_count)) + 2)[1:-1]
    return np.unique(np.asarray(np.quantile(array, quantiles), dtype=np.float64))


def _design_matrix_from_two_continuous_inputs(
    x1_values: Sequence[float],
    x2_values: Sequence[float],
    *,
    basis_mode: str,
    x1_hinge_knots: Sequence[float] | None = None,
    x2_hinge_knots: Sequence[float] | None = None,
) -> np.ndarray:
    x1 = np.asarray(x1_values, dtype=np.float64).reshape(-1)
    x2 = np.asarray(x2_values, dtype=np.float64).reshape(-1)
    if x1.size != x2.size:
        raise ValueError("x1/x2 lengths do not match for 2D continuous design matrix.")

    columns: list[np.ndarray] = [
        np.ones_like(x1, dtype=np.float64),
        x1,
        x2,
        x1 * x2,
    ]
    mode = str(basis_mode)
    if mode == "hinge":
        for knot in np.asarray(x1_hinge_knots if x1_hinge_knots is not None else [], dtype=np.float64).tolist():
            columns.append(np.maximum(0.0, x1 - float(knot)))
        for knot in np.asarray(x2_hinge_knots if x2_hinge_knots is not None else [], dtype=np.float64).tolist():
            columns.append(np.maximum(0.0, x2 - float(knot)))
    elif mode != "linear_interaction":
        raise ValueError(f"Unsupported 2D continuous basis mode: {basis_mode}")
    return np.column_stack(columns)


def _continuous_basis_term_names(
    *,
    basis_mode: str,
    x1_hinge_knots: Sequence[float],
    x2_hinge_knots: Sequence[float],
) -> list[str]:
    terms = ["1", "x1", "x2", "x1*x2"]
    if str(basis_mode) == "hinge":
        terms.extend(f"relu(x1-{float(knot):.6f})" for knot in np.asarray(x1_hinge_knots, dtype=np.float64).tolist())
        terms.extend(f"relu(x2-{float(knot):.6f})" for knot in np.asarray(x2_hinge_knots, dtype=np.float64).tolist())
    return terms


class ContinuousTwoStateMixtureAllocationPolicy:
    """Use two continuous state variables to build an interpretable simplex-constrained surrogate rule."""

    def __init__(
        self,
        *,
        policy_name: str,
        profile_family: str,
        second_variable_name: str,
        basis_mode: str,
        coeffs_unit_investment: Sequence[float] | None = None,
        coeffs_self: Sequence[float] | None = None,
        coeffs_alpha: Sequence[float] | None = None,
        x1_hinge_knots: Sequence[float] | None = None,
        x2_hinge_knots: Sequence[float] | None = None,
        profile_source: str,
        fit_summary: Mapping[str, Any] | None = None,
    ):
        self.policy_name = str(policy_name)
        self.profile_family = str(profile_family)
        self.second_variable_name = str(second_variable_name)
        self.basis_mode = str(basis_mode)
        if self.profile_family not in {"three_way", "unitself"}:
            raise ValueError(f"Unsupported continuous profile family: {self.profile_family}")
        if self.basis_mode not in {"linear_interaction", "hinge"}:
            raise ValueError(f"Unsupported continuous basis mode: {self.basis_mode}")

        self.x1_hinge_knots = np.asarray(
            [] if x1_hinge_knots is None else x1_hinge_knots,
            dtype=np.float64,
        ).reshape(-1)
        self.x2_hinge_knots = np.asarray(
            [] if x2_hinge_knots is None else x2_hinge_knots,
            dtype=np.float64,
        ).reshape(-1)
        self.basis_terms = _continuous_basis_term_names(
            basis_mode=self.basis_mode,
            x1_hinge_knots=self.x1_hinge_knots,
            x2_hinge_knots=self.x2_hinge_knots,
        )
        basis_size = int(len(self.basis_terms))

        self.coeffs_unit_investment = (
            np.asarray(coeffs_unit_investment, dtype=np.float64).reshape(-1)
            if coeffs_unit_investment is not None
            else np.zeros(basis_size, dtype=np.float64)
        )
        if self.coeffs_unit_investment.size != basis_size:
            raise ValueError("2D continuous unit-investment coefficient count does not match basis size.")
        if self.profile_family == "three_way":
            if coeffs_self is None:
                raise ValueError("3-way 2D continuous policy requires self coefficients.")
            self.coeffs_self = np.asarray(coeffs_self, dtype=np.float64).reshape(-1)
            if self.coeffs_self.size != basis_size:
                raise ValueError("2D continuous self coefficient count does not match basis size.")
            self.coeffs_alpha = np.zeros(basis_size, dtype=np.float64)
        else:
            if coeffs_alpha is None:
                raise ValueError("UnitInv+Self 2D continuous policy requires alpha coefficients.")
            self.coeffs_alpha = np.asarray(coeffs_alpha, dtype=np.float64).reshape(-1)
            if self.coeffs_alpha.size != basis_size:
                raise ValueError("2D continuous alpha coefficient count does not match basis size.")
            self.coeffs_self = np.zeros(basis_size, dtype=np.float64)

        self.profile_source = str(profile_source)
        self.fit_summary_dict = {} if fit_summary is None else dict(fit_summary)
        self._unit_investment = UnitInvestmentProportionalAllocationPolicy()
        self._equal = UniformAllocationPolicy()

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = float(np.clip(value, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-clipped)))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        shifted = logits - float(np.max(logits))
        exp_values = np.exp(np.clip(shifted, -60.0, 60.0))
        total = float(np.sum(exp_values))
        if total <= 1e-12:
            return np.full(logits.shape, 1.0 / float(max(logits.size, 1)), dtype=np.float64)
        return exp_values / total

    def _basis(self, x1_value: float, x2_value: float) -> np.ndarray:
        return _design_matrix_from_two_continuous_inputs(
            [float(x1_value)],
            [float(x2_value)],
            basis_mode=self.basis_mode,
            x1_hinge_knots=self.x1_hinge_knots,
            x2_hinge_knots=self.x2_hinge_knots,
        )[0]

    def weights_for_inputs(self, x1_value: float, x2_value: float) -> tuple[float, float, float]:
        basis = self._basis(float(x1_value), float(x2_value))
        if self.profile_family == "three_way":
            unit_logit = float(np.dot(self.coeffs_unit_investment, basis))
            self_logit = float(np.dot(self.coeffs_self, basis))
            weights = self._softmax(np.asarray([unit_logit, 0.0, self_logit], dtype=np.float64))
            return float(weights[0]), float(weights[1]), float(weights[2])

        alpha = self._sigmoid(float(np.dot(self.coeffs_alpha, basis)))
        return float(1.0 - alpha), 0.0, float(alpha)

    def weights_for_sender(
        self,
        observation: Mapping[str, np.ndarray],
        sender: int,
    ) -> tuple[float, float, float, dict[str, float]]:
        x1_value, x2_value, ego_mean_resource = _extract_continuous_state_from_observation(
            observation,
            int(sender),
            self.second_variable_name,
        )
        weight_unit_investment, weight_equal, weight_self = self.weights_for_inputs(x1_value, x2_value)
        return (
            weight_unit_investment,
            weight_equal,
            weight_self,
            {
                "x1_log_ego_mean_resource": float(x1_value),
                "ego_mean_resource": float(ego_mean_resource),
                "x2_second_variable": float(x2_value),
            },
        )

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        unit_investment_allocation = np.asarray(self._unit_investment.allocate(observation), dtype=np.float64)
        equal_allocation = np.asarray(self._equal.allocate(observation), dtype=np.float64)
        self_allocation = np.zeros(local_mask.shape, dtype=np.float64)
        for sender in range(local_mask.shape[0]):
            if local_mask[sender, sender]:
                self_allocation[sender, sender] = 1.0
            else:
                valid_receivers = np.flatnonzero(local_mask[sender])
                if valid_receivers.size > 0:
                    self_allocation[sender, valid_receivers] = 1.0 / float(valid_receivers.size)

        allocation = np.zeros(local_mask.shape, dtype=np.float64)
        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            weight_unit_investment, weight_equal, weight_self, _ = self.weights_for_sender(observation, sender)
            allocation[sender] = (
                weight_unit_investment * unit_investment_allocation[sender]
                + weight_equal * equal_allocation[sender]
                + weight_self * self_allocation[sender]
            )

        allocation = np.where(local_mask, allocation, 0.0)
        row_sums = allocation.sum(axis=1, keepdims=True)
        positive_rows = row_sums.squeeze(-1) > 1e-12
        if np.any(positive_rows):
            allocation[positive_rows] = allocation[positive_rows] / row_sums[positive_rows]
        if np.any(~positive_rows):
            allocation[~positive_rows] = equal_allocation[~positive_rows]
        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)

    def profile_summary(self) -> dict[str, Any]:
        payload = {
            "mode": "continuous_2d_surrogate",
            "policy_name": self.policy_name,
            "profile_family": self.profile_family,
            "profile_source": self.profile_source,
            "x1_name": "log1p(ego_mean_resource)",
            "x2_name": self.second_variable_name,
            "basis_mode": self.basis_mode,
            "basis_terms": list(self.basis_terms),
            "x1_hinge_knots": [float(value) for value in self.x1_hinge_knots.tolist()],
            "x2_hinge_knots": [float(value) for value in self.x2_hinge_knots.tolist()],
            "coeffs_unit_investment": [float(value) for value in self.coeffs_unit_investment.tolist()],
            "coeffs_equal": [0.0 for _ in range(int(len(self.basis_terms)))] if self.profile_family == "three_way" else None,
            "coeffs_self": [float(value) for value in self.coeffs_self.tolist()] if self.profile_family == "three_way" else None,
            "coeffs_alpha": [float(value) for value in self.coeffs_alpha.tolist()] if self.profile_family == "unitself" else None,
        }
        payload.update(self.fit_summary_dict)
        return payload


def build_policy_specs(
    adaptive_policy_name: str | None = None,
    adaptive_policy: Any | None = None,
    *,
    extra_policy_specs: Sequence[tuple[str, Any]] | None = None,
) -> tuple[tuple[str, Any], ...]:
    policy_specs: list[tuple[str, Any]] = [
        ("agent", None),
        ("resource_proportional", ResourceProportionalAllocationPolicy()),
        ("unit_investment_proportional", UnitInvestmentProportionalAllocationPolicy()),
    ]
    if adaptive_policy_name is not None and adaptive_policy is not None:
        policy_specs.append((str(adaptive_policy_name), adaptive_policy))
    if extra_policy_specs:
        for policy_name, policy in extra_policy_specs:
            policy_specs.append((str(policy_name), policy))
    policy_specs.append(("equal", UniformAllocationPolicy()))
    return tuple(policy_specs)


POLICY_SPECS: tuple[tuple[str, Any], ...] = ()

POLICY_STYLES = {
    "agent": {"color": "#2ca02c", "label": "RL Agent"},
    "resource_proportional": {"color": "#d62728", "label": "Proportional by Resources"},
    "unit_investment_proportional": {"color": "#9467bd", "label": "Proportional by Unit Investment"},
    "adaptive_unitinv_equal_mix": {"color": "#8c564b", "label": "Adaptive Unit-Investment + Equal"},
    "adaptive_smooth_2way_mix": {"color": "#8c564b", "label": "Smooth log(1+m): UnitInv + Equal"},
    "adaptive_smooth_3way_mix": {"color": "#7f3c8d", "label": "Smooth log(1+m): UnitInv + Equal + Self"},
    "adaptive_spline_3way_mix": {"color": "#6a3d9a", "label": "Spline log(1+m): UnitInv + Equal + Self"},
    "adaptive_binned_3way_mix": {"color": "#4c1d95", "label": "Binned ego-resource: UnitInv + Equal + Self"},
    "adaptive_degree_conditioned_spline_3way_mix": {"color": "#5b2a86", "label": "Degree-Conditioned Spline 3-Way"},
    "adaptive_degree_conditioned_unitself_mix": {"color": "#7a5195", "label": "Degree-Conditioned UnitInv + Self"},
    "adaptive_resource_conditioned_spline_3way_mix": {"color": "#2f5597", "label": "Resource-Conditioned Spline 3-Way"},
    "adaptive_resource_conditioned_unitself_mix": {"color": "#4575b4", "label": "Resource-Conditioned UnitInv + Self"},
    "adaptive_continuous2d_softmax_mix": {"color": "#0f766e", "label": "2D Continuous Softmax"},
    "adaptive_continuous2d_unitself_mix": {"color": "#b45309", "label": "2D Continuous UnitInv + Self"},
    "equal": {"color": "#1f77ff", "label": "Equal Baseline"},
}


def _default_adaptive_unitinv_equal_knots() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([10.0, 40.0, 500.0], dtype=np.float64),
        np.asarray([0.95, 0.99, 0.90], dtype=np.float64),
    )


def _parse_adaptive_knots(raw_value: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if len(pairs) < 2:
        raise ValueError("adaptive-unitinv-equal-knots requires at least two knot pairs like 10:0.95,40:0.99.")

    resource_values: list[float] = []
    weight_values: list[float] = []
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid knot pair: {pair}. Expected format <ego_mean_resource>:<w>.")
        left, right = pair.split(":", maxsplit=1)
        resource_value = float(left)
        weight_value = float(right)
        if not 0.0 <= weight_value <= 1.0:
            raise ValueError(f"Adaptive knot weight must be in [0, 1], got {weight_value}.")
        resource_values.append(resource_value)
        weight_values.append(weight_value)
    return np.asarray(resource_values, dtype=np.float64), np.asarray(weight_values, dtype=np.float64)


def _centered_moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size <= 1:
        return array.copy()
    window = max(1, int(window_size))
    if window <= 1:
        return array.copy()
    if window % 2 == 0:
        window += 1

    half = window // 2
    smoothed = np.empty_like(array, dtype=np.float64)
    for index in range(array.size):
        left = max(0, index - half)
        right = min(array.size, index + half + 1)
        smoothed[index] = float(np.mean(array[left:right]))
    return smoothed


def _prepare_adaptive_profile_knots(
    resource_values: Sequence[float],
    weight_values: Sequence[float],
    *,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    resources = np.asarray(resource_values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weight_values, dtype=np.float64).reshape(-1)
    valid_mask = np.isfinite(resources) & np.isfinite(weights)
    resources = resources[valid_mask]
    weights = np.clip(weights[valid_mask], 0.0, 1.0)
    if resources.size < 2:
        raise ValueError("Cannot derive adaptive unit-investment/equal profile from fewer than two valid points.")

    order = np.argsort(resources)
    resources = resources[order]
    weights = weights[order]

    unique_resources: list[float] = []
    unique_weights: list[float] = []
    for resource_value, weight_value in zip(resources.tolist(), weights.tolist()):
        if unique_resources and abs(resource_value - unique_resources[-1]) <= 1e-12:
            unique_weights[-1] = 0.5 * (unique_weights[-1] + float(weight_value))
        else:
            unique_resources.append(float(resource_value))
            unique_weights.append(float(weight_value))
    resources = np.asarray(unique_resources, dtype=np.float64)
    weights = np.asarray(unique_weights, dtype=np.float64)
    if resources.size < 2:
        raise ValueError("Adaptive unit-investment/equal profile contains fewer than two unique resources.")

    smoothed_weights = _centered_moving_average(weights, max(1, int(smooth_window)))
    smoothed_weights = np.clip(smoothed_weights, 0.0, 1.0)
    return resources, smoothed_weights


def _load_adaptive_unitinv_equal_profile_from_csv(
    profile_csv_path: Path,
    *,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    import csv

    with profile_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        resource_values: list[float] = []
        weight_values: list[float] = []
        for row in reader:
            if not row:
                continue
            resource_raw = row.get("bin_center")
            weight_raw = row.get("weight_unit_investment_mean")
            if resource_raw is None or weight_raw is None:
                continue
            resource_value = float(resource_raw)
            weight_value = float(weight_raw)
            if not np.isfinite(resource_value) or not np.isfinite(weight_value):
                continue
            resource_values.append(resource_value)
            weight_values.append(weight_value)
    if not resource_values:
        raise ValueError(f"No valid bin_center / weight_unit_investment_mean rows found in {profile_csv_path}")
    return _prepare_adaptive_profile_knots(
        resource_values,
        weight_values,
        smooth_window=max(1, int(smooth_window)),
    )


def resolve_adaptive_unitinv_equal_policy(
    *,
    run_dir: Path,
    profile_csv_arg: str | None,
    knots_arg: str | None,
    smooth_window: int,
) -> AdaptiveUnitInvestmentEqualAllocationPolicy:
    if knots_arg is not None and str(knots_arg).strip():
        resource_knots, weight_knots = _parse_adaptive_knots(str(knots_arg))
        return AdaptiveUnitInvestmentEqualAllocationPolicy(
            resource_knots,
            weight_knots,
            profile_source=f"manual_knots(smooth_window={max(1, int(smooth_window))})",
        )

    fallback_resources, fallback_weights = _default_adaptive_unitinv_equal_knots()
    if profile_csv_arg is None or not str(profile_csv_arg).strip():
        return AdaptiveUnitInvestmentEqualAllocationPolicy(
            fallback_resources,
            fallback_weights,
            profile_source="fallback_default_knots",
        )

    profile_path = Path(str(profile_csv_arg))
    if not profile_path.is_absolute():
        profile_path = (run_dir / profile_path).resolve()
    if not profile_path.exists():
        return AdaptiveUnitInvestmentEqualAllocationPolicy(
            fallback_resources,
            fallback_weights,
            profile_source=f"fallback_default_knots_missing:{profile_path}",
        )

    resource_knots, weight_knots = _load_adaptive_unitinv_equal_profile_from_csv(
        profile_path,
        smooth_window=max(1, int(smooth_window)),
    )
    return AdaptiveUnitInvestmentEqualAllocationPolicy(
        resource_knots,
        weight_knots,
        profile_source=f"{profile_path}::smooth_window={max(1, int(smooth_window))}",
    )


_module_default_resources, _module_default_weights = _default_adaptive_unitinv_equal_knots()
POLICY_SPECS = build_policy_specs(
    "adaptive_unitinv_equal_mix",
    AdaptiveUnitInvestmentEqualAllocationPolicy(
        _module_default_resources,
        _module_default_weights,
        profile_source="module_default_knots",
    )
)


def _weighted_lstsq(design: np.ndarray, target: np.ndarray, sample_weights: np.ndarray | None = None) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if design.ndim != 2 or design.shape[0] != target.size:
        raise ValueError("weighted least-squares design/target shapes do not match.")
    if sample_weights is None:
        solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        return np.asarray(solution, dtype=np.float64)

    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.size != target.size:
        raise ValueError("Sample weight count does not match target count.")
    weights = np.clip(weights, 0.0, None)
    sqrt_weights = np.sqrt(weights + 1e-12)
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = target * sqrt_weights
    solution, _, _, _ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    return np.asarray(solution, dtype=np.float64)


def _clip_probability(value: float, eps: float = 1e-6) -> float:
    return float(np.clip(float(value), eps, 1.0 - eps))


def _clip_probability_array(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)


def _normalize_ego_resource_value(
    ego_mean_resource: float,
    *,
    x_mode: str,
    normalizer: float | None,
) -> float:
    safe_resource = max(0.0, float(ego_mean_resource))
    mode = str(x_mode)
    if mode == "absolute":
        return float(safe_resource)
    if mode != "normalized":
        raise ValueError(f"Unsupported ego resource x mode: {x_mode}")
    if normalizer is None or float(normalizer) <= 1e-12:
        raise ValueError("ego_resource_normalizer must be positive when ego_resource_x_mode='normalized'.")
    return float(safe_resource / float(normalizer))


def _ego_resource_x_label(x_mode: str, normalizer: float | None) -> str:
    if str(x_mode) == "normalized":
        return "Ego_Resource_Mean / Theoretical_Max"
    return "Ego_Resource_Mean"


def _extract_ego_mean_resource_from_row(row: Mapping[str, Any]) -> float:
    ego_mean_raw = row.get("ego_mean_resource")
    if ego_mean_raw not in {None, ""}:
        ego_mean_resource = float(ego_mean_raw)
        if np.isfinite(ego_mean_resource):
            return float(max(0.0, ego_mean_resource))
    total_raw = row.get("ego_total_resources")
    receiver_count_raw = row.get("receiver_count")
    if total_raw in {None, ""} or receiver_count_raw in {None, ""}:
        return float("nan")
    total_resource = float(total_raw)
    receiver_count = max(1.0, float(receiver_count_raw))
    ego_mean_resource = total_resource / receiver_count
    return float(max(0.0, ego_mean_resource)) if np.isfinite(ego_mean_resource) else float("nan")


def collect_agent_reference_rows(agent_snapshots: Sequence[Snapshot]) -> list[dict[str, Any]]:
    reference_rows: list[dict[str, Any]] = []
    for snapshot in agent_snapshots:
        observation = snapshot.observation
        allocation_matrix = np.asarray(snapshot.policy.allocation_matrix, dtype=np.float64)
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)
        degrees = np.asarray(observation["degrees"], dtype=np.float64)
        resource_norm = np.asarray(observation["resource_norm"], dtype=np.float64)
        strategy_norm = np.asarray(observation["strategy_norm"], dtype=np.float64)
        x_actual = np.asarray(observation["x_actual"], dtype=np.float64)
        pool_grown = np.asarray(observation["pool_grown"], dtype=np.float64)
        pool_raw_norm = np.asarray(observation["pool_raw_norm"], dtype=np.float64)
        pool_upperbound = np.asarray(observation["p_max"], dtype=np.float64)

        for sender in range(local_mask.shape[0]):
            valid_receivers = np.flatnonzero(local_mask[sender])
            if valid_receivers.size == 0:
                continue
            allocation_row = np.asarray(allocation_matrix[sender, valid_receivers], dtype=np.float64)
            unit_row, equal_row, self_row = _build_row_baselines(observation, sender, valid_receivers)
            two_way_weight_unit_investment = _fit_labor_equal_lambda(allocation_row, unit_row, equal_row)
            if np.isfinite(two_way_weight_unit_investment):
                two_way_weight_equal = float(1.0 - two_way_weight_unit_investment)
                two_way_fit_l1 = float(
                    np.mean(
                        np.abs(
                            allocation_row
                            - (
                                (two_way_weight_unit_investment * unit_row)
                                + (two_way_weight_equal * equal_row)
                            )
                        )
                    )
                )
            else:
                two_way_weight_equal = float("nan")
                two_way_fit_l1 = float("nan")

            three_way_weights, three_way_fit_l1 = _fit_three_way_mechanism_weights(
                allocation_row,
                unit_row,
                equal_row,
                self_row,
            )
            self_index = np.nonzero(valid_receivers == sender)[0]
            self_allocation = float(allocation_row[int(self_index[0])]) if self_index.size > 0 else float("nan")
            sender_pool_grown = float(pool_grown[sender])
            sender_pool_upperbound = float(pool_upperbound[sender])
            pool_grown_over_upperbound = (
                sender_pool_grown / sender_pool_upperbound if sender_pool_upperbound > 1e-12 else float("nan")
            )

            reference_rows.append(
                {
                    "episode": int(snapshot.episode),
                    "step": int(snapshot.step),
                    "sender": int(sender),
                    "sender_degree": int(round(float(degrees[sender]))),
                    "sender_resource": float(resources[sender]),
                    "sender_resource_norm": float(resource_norm[sender]),
                    "sender_strategy_norm": float(strategy_norm[sender]),
                    "sender_x_actual": float(x_actual[sender]),
                    "ego_mean_resource": float(np.mean(resources[valid_receivers])),
                    "sender_pool_grown": float(sender_pool_grown),
                    "sender_pool_upperbound": float(sender_pool_upperbound),
                    "pool_grown_over_upperbound": float(pool_grown_over_upperbound),
                    "sender_pool_raw_norm": float(pool_raw_norm[sender]),
                    "two_way_weight_unit_investment": float(two_way_weight_unit_investment),
                    "two_way_weight_equal": float(two_way_weight_equal),
                    "two_way_fit_l1": float(two_way_fit_l1),
                    "three_way_weight_unit_investment": (
                        float(three_way_weights[0]) if np.isfinite(three_way_weights[0]) else float("nan")
                    ),
                    "three_way_weight_equal": (
                        float(three_way_weights[1]) if np.isfinite(three_way_weights[1]) else float("nan")
                    ),
                    "three_way_weight_self": (
                        float(three_way_weights[2]) if np.isfinite(three_way_weights[2]) else float("nan")
                    ),
                    "three_way_fit_l1": float(three_way_fit_l1),
                    "self_allocation": float(self_allocation),
                }
            )
    return reference_rows


def _design_matrix_from_ego_mean_resource(resource_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe_resources = np.clip(np.asarray(resource_values, dtype=np.float64).reshape(-1), 0.0, None)
    log_resources = np.log1p(safe_resources)
    design = np.column_stack([np.ones_like(log_resources), log_resources, np.square(log_resources)])
    return log_resources, design


def _apply_tail_cap_to_resources(
    resource_values: Sequence[float] | np.ndarray,
    *,
    tail_cap_mode: str,
    tail_cap_resource: float | None,
    tail_cap_soft_scale: float,
) -> np.ndarray:
    resources = np.clip(np.asarray(resource_values, dtype=np.float64).reshape(-1), 0.0, None)
    mode = str(tail_cap_mode)
    cap_value = None if tail_cap_resource is None else float(max(0.0, float(tail_cap_resource)))
    if mode == "none" or cap_value is None:
        return resources
    if mode == "hard":
        return np.minimum(resources, cap_value)
    if mode == "soft":
        scale = float(np.clip(float(tail_cap_soft_scale), 0.0, 1.0))
        return np.where(resources <= cap_value, resources, cap_value + scale * (resources - cap_value))
    raise ValueError(f"Unsupported tail cap mode: {tail_cap_mode}")


def _build_spline_resource_knots(
    resource_values: Sequence[float],
    *,
    high_knot_style: str = "sparse",
) -> np.ndarray:
    resources = np.asarray(resource_values, dtype=np.float64).reshape(-1)
    resources = resources[np.isfinite(resources)]
    if resources.size < 8:
        return np.asarray([], dtype=np.float64)

    resource_min = float(np.min(resources))
    resource_max = float(np.max(resources))
    quantile_knots = np.quantile(resources, [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92])
    high_knot_mode = str(high_knot_style)
    if high_knot_mode == "dense":
        high_resource_knots = np.asarray([400.0, 420.0, 430.0, 435.0, 440.0, 445.0, 450.0, 460.0, 480.0], dtype=np.float64)
    elif high_knot_mode == "sparse":
        high_resource_knots = np.asarray([400.0, 430.0, 460.0], dtype=np.float64)
    elif high_knot_mode == "none":
        high_resource_knots = np.asarray([], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported spline high-knot style: {high_knot_style}")
    candidates = np.concatenate([np.asarray(quantile_knots, dtype=np.float64), high_resource_knots], axis=0)
    candidates = candidates[(candidates > resource_min + 1e-6) & (candidates < resource_max - 1e-6)]
    if candidates.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.unique(np.round(candidates, decimals=6))


def _design_matrix_from_ego_mean_resource_spline(
    resource_values: np.ndarray,
    spline_knots_resource: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    safe_resources = np.clip(np.asarray(resource_values, dtype=np.float64).reshape(-1), 0.0, None)
    log_resources = np.log1p(safe_resources)
    knot_resources = np.asarray(spline_knots_resource, dtype=np.float64).reshape(-1)
    if knot_resources.size == 0:
        design = np.column_stack([np.ones_like(log_resources), log_resources])
        return log_resources, design
    knot_logs = np.log1p(np.clip(knot_resources, 0.0, None))
    hinge_columns = [np.maximum(0.0, log_resources - float(knot_value)) for knot_value in knot_logs.tolist()]
    design = np.column_stack([np.ones_like(log_resources), log_resources, *hinge_columns])
    return log_resources, design


def _safe_normalize_rows(weight_matrix: np.ndarray) -> np.ndarray:
    weights = np.asarray(weight_matrix, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    row_sums = weights.sum(axis=1, keepdims=True)
    valid = row_sums.squeeze(-1) > 1e-12
    normalized = np.full(weights.shape, 1.0 / float(max(weights.shape[1], 1)), dtype=np.float64)
    if np.any(valid):
        normalized[valid] = weights[valid] / row_sums[valid]
    return normalized


def _fit_three_way_softmax_direct(
    design: np.ndarray,
    target_weights: np.ndarray,
    *,
    init_unit: np.ndarray | None = None,
    init_self: np.ndarray | None = None,
    max_iterations: int = 900,
    learning_rate: float = 0.05,
    l2_penalty: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    design_array = np.asarray(design, dtype=np.float32)
    target_array = _safe_normalize_rows(np.asarray(target_weights, dtype=np.float64)).astype(np.float32, copy=False)
    basis_dim = int(design_array.shape[1])

    design_tensor = torch.from_numpy(design_array)
    target_tensor = torch.from_numpy(target_array)
    coeff_unit_tensor = torch.nn.Parameter(
        torch.from_numpy(
            np.asarray(init_unit if init_unit is not None else np.zeros(basis_dim, dtype=np.float64), dtype=np.float32)
        )
    )
    coeff_self_tensor = torch.nn.Parameter(
        torch.from_numpy(
            np.asarray(init_self if init_self is not None else np.zeros(basis_dim, dtype=np.float64), dtype=np.float32)
        )
    )

    optimizer = torch.optim.Adam([coeff_unit_tensor, coeff_self_tensor], lr=float(learning_rate))
    best_loss = float("inf")
    best_coeff_unit = coeff_unit_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
    best_coeff_self = coeff_self_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
    best_iteration = -1
    stale_steps = 0

    for iteration in range(max(1, int(max_iterations))):
        optimizer.zero_grad()
        unit_logits = design_tensor @ coeff_unit_tensor
        self_logits = design_tensor @ coeff_self_tensor
        logits = torch.stack(
            [
                unit_logits,
                torch.zeros_like(unit_logits),
                self_logits,
            ],
            dim=1,
        )
        predicted = torch.softmax(logits, dim=1)
        simplex_l1 = torch.mean(torch.sum(torch.abs(predicted - target_tensor), dim=1))
        regularizer = float(l2_penalty) * (
            torch.mean(coeff_unit_tensor.square()) + torch.mean(coeff_self_tensor.square())
        )
        loss = simplex_l1 + regularizer
        loss.backward()
        optimizer.step()

        loss_value = float(simplex_l1.detach().cpu().item())
        if loss_value + 1e-8 < best_loss:
            best_loss = loss_value
            best_iteration = int(iteration)
            best_coeff_unit = coeff_unit_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
            best_coeff_self = coeff_self_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= 120:
            break

    summary = {
        "optimization_iterations": int(best_iteration + 1 if best_iteration >= 0 else 0),
        "optimization_best_simplex_l1": float(best_loss),
        "optimization_learning_rate": float(learning_rate),
        "optimization_l2_penalty": float(l2_penalty),
    }
    return best_coeff_unit, best_coeff_self, summary


def _fit_binary_sigmoid_direct(
    design: np.ndarray,
    target_alpha: np.ndarray,
    *,
    init_coeffs: np.ndarray | None = None,
    max_iterations: int = 900,
    learning_rate: float = 0.05,
    l2_penalty: float = 1e-4,
) -> tuple[np.ndarray, dict[str, Any]]:
    design_array = np.asarray(design, dtype=np.float32)
    target_array = np.clip(np.asarray(target_alpha, dtype=np.float64).reshape(-1), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )
    basis_dim = int(design_array.shape[1])

    design_tensor = torch.from_numpy(design_array)
    target_tensor = torch.from_numpy(target_array)
    coeff_tensor = torch.nn.Parameter(
        torch.from_numpy(
            np.asarray(
                init_coeffs if init_coeffs is not None else np.zeros(basis_dim, dtype=np.float64),
                dtype=np.float32,
            )
        )
    )

    optimizer = torch.optim.Adam([coeff_tensor], lr=float(learning_rate))
    best_loss = float("inf")
    best_coeffs = coeff_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
    best_iteration = -1
    stale_steps = 0

    for iteration in range(max(1, int(max_iterations))):
        optimizer.zero_grad()
        prediction = torch.sigmoid(design_tensor @ coeff_tensor)
        alpha_l1 = torch.mean(torch.abs(prediction - target_tensor))
        regularizer = float(l2_penalty) * torch.mean(coeff_tensor.square())
        loss = alpha_l1 + regularizer
        loss.backward()
        optimizer.step()

        loss_value = float(alpha_l1.detach().cpu().item())
        if loss_value + 1e-8 < best_loss:
            best_loss = loss_value
            best_iteration = int(iteration)
            best_coeffs = coeff_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= 120:
            break

    return best_coeffs, {
        "optimization_iterations": int(best_iteration + 1 if best_iteration >= 0 else 0),
        "optimization_best_alpha_l1": float(best_loss),
        "optimization_learning_rate": float(learning_rate),
        "optimization_l2_penalty": float(l2_penalty),
    }


def fit_continuous_2d_softmax_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    second_variable_name: str,
    basis_mode: str,
    x1_hinge_count: int,
    x2_hinge_count: int,
) -> tuple[ContinuousTwoStateMixtureAllocationPolicy, dict[str, Any]]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(_extract_continuous_second_value_from_row(row, second_variable_name))
        and np.isfinite(float(row.get("three_way_weight_unit_investment", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_equal", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_self", float("nan"))))
    ]
    if len(valid_rows) < 12:
        fallback_policy, fallback_rows, fallback_summary = fit_spline_three_way_policy(reference_rows)
        fallback_summary = {
            "fit_status": "fallback_spline_3way_insufficient_rows_for_continuous2d_softmax",
            "policy_name": "adaptive_continuous2d_softmax_mix",
            "profile_family": "three_way",
            "fit_row_count": int(len(valid_rows)),
            "second_variable_name": str(second_variable_name),
            "basis_mode": str(basis_mode),
            **dict(fallback_summary),
        }
        if isinstance(fallback_policy, SmoothLogEgoResourceMixtureAllocationPolicy):
            fallback_policy.fit_summary_dict = dict(fallback_summary)
        return fallback_policy, fallback_summary

    ego_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    x1_values = np.log1p(np.clip(ego_values, 0.0, None))
    x2_values = np.asarray(
        [_extract_continuous_second_value_from_row(row, second_variable_name) for row in valid_rows],
        dtype=np.float64,
    )
    x1_hinge_knots = _build_hinge_knots(x1_values, int(x1_hinge_count)) if str(basis_mode) == "hinge" else np.asarray([], dtype=np.float64)
    x2_hinge_knots = _build_hinge_knots(x2_values, int(x2_hinge_count)) if str(basis_mode) == "hinge" else np.asarray([], dtype=np.float64)
    design = _design_matrix_from_two_continuous_inputs(
        x1_values,
        x2_values,
        basis_mode=str(basis_mode),
        x1_hinge_knots=x1_hinge_knots,
        x2_hinge_knots=x2_hinge_knots,
    )

    unit_weights = np.asarray([float(row["three_way_weight_unit_investment"]) for row in valid_rows], dtype=np.float64)
    equal_weights = np.asarray([float(row["three_way_weight_equal"]) for row in valid_rows], dtype=np.float64)
    self_weights = np.asarray([float(row["three_way_weight_self"]) for row in valid_rows], dtype=np.float64)
    target_weights = _safe_normalize_rows(np.column_stack([unit_weights, equal_weights, self_weights]))

    init_unit = _weighted_lstsq(
        design,
        np.log(_clip_probability_array(unit_weights) / _clip_probability_array(equal_weights)),
    )
    init_self = _weighted_lstsq(
        design,
        np.log(_clip_probability_array(self_weights) / _clip_probability_array(equal_weights)),
    )
    coeffs_unit, coeffs_self, optimization_summary = _fit_three_way_softmax_direct(
        design,
        target_weights,
        init_unit=init_unit,
        init_self=init_self,
    )

    policy = ContinuousTwoStateMixtureAllocationPolicy(
        policy_name="adaptive_continuous2d_softmax_mix",
        profile_family="three_way",
        second_variable_name=str(second_variable_name),
        basis_mode=str(basis_mode),
        coeffs_unit_investment=coeffs_unit,
        coeffs_self=coeffs_self,
        x1_hinge_knots=x1_hinge_knots,
        x2_hinge_knots=x2_hinge_knots,
        profile_source="agent_rollout_reference_rows_2d",
    )

    simplex_l1_errors: list[float] = []
    for row, target_weight_vector in zip(valid_rows, target_weights.tolist()):
        predicted = np.asarray(
            policy.weights_for_inputs(
                float(np.log1p(max(0.0, float(row["ego_mean_resource"])))),
                float(_extract_continuous_second_value_from_row(row, second_variable_name)),
            ),
            dtype=np.float64,
        )
        simplex_l1_errors.append(float(np.sum(np.abs(predicted - np.asarray(target_weight_vector, dtype=np.float64)))))

    fit_summary = {
        "fit_status": "agent_rollout_continuous2d_softmax",
        "policy_name": "adaptive_continuous2d_softmax_mix",
        "profile_family": "three_way",
        "second_variable_name": str(second_variable_name),
        "basis_mode": str(basis_mode),
        "fit_row_count": int(len(valid_rows)),
        "basis_dim": int(design.shape[1]),
        "weight_simplex_l1_mean": float(np.mean(simplex_l1_errors)) if simplex_l1_errors else float("nan"),
        "reference_row_fit_l1_mean": (
            float(np.mean([float(row["three_way_fit_l1"]) for row in valid_rows])) if valid_rows else float("nan")
        ),
        "x1_hinge_knots": [float(value) for value in x1_hinge_knots.tolist()],
        "x2_hinge_knots": [float(value) for value in x2_hinge_knots.tolist()],
        **optimization_summary,
    }
    policy.fit_summary_dict = dict(fit_summary)
    return policy, fit_summary


def fit_continuous_2d_unitself_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    second_variable_name: str,
    basis_mode: str,
    x1_hinge_count: int,
    x2_hinge_count: int,
) -> tuple[ContinuousTwoStateMixtureAllocationPolicy, dict[str, Any]]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(_extract_continuous_second_value_from_row(row, second_variable_name))
        and np.isfinite(float(row.get("three_way_weight_unit_investment", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_equal", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_self", float("nan"))))
    ]
    if len(valid_rows) < 12:
        fallback_policy, fallback_rows, fallback_summary = fit_spline_unitself_policy(reference_rows)
        fallback_summary = {
            "fit_status": "fallback_spline_unitself_insufficient_rows_for_continuous2d_unitself",
            "policy_name": "adaptive_continuous2d_unitself_mix",
            "profile_family": "unitself",
            "fit_row_count": int(len(valid_rows)),
            "second_variable_name": str(second_variable_name),
            "basis_mode": str(basis_mode),
            **dict(fallback_summary),
        }
        if isinstance(fallback_policy, SmoothLogEgoResourceMixtureAllocationPolicy):
            fallback_policy.fit_summary_dict = dict(fallback_summary)
        return fallback_policy, fallback_summary

    ego_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    x1_values = np.log1p(np.clip(ego_values, 0.0, None))
    x2_values = np.asarray(
        [_extract_continuous_second_value_from_row(row, second_variable_name) for row in valid_rows],
        dtype=np.float64,
    )
    x1_hinge_knots = _build_hinge_knots(x1_values, int(x1_hinge_count)) if str(basis_mode) == "hinge" else np.asarray([], dtype=np.float64)
    x2_hinge_knots = _build_hinge_knots(x2_values, int(x2_hinge_count)) if str(basis_mode) == "hinge" else np.asarray([], dtype=np.float64)
    design = _design_matrix_from_two_continuous_inputs(
        x1_values,
        x2_values,
        basis_mode=str(basis_mode),
        x1_hinge_knots=x1_hinge_knots,
        x2_hinge_knots=x2_hinge_knots,
    )

    unit_raw = np.clip(
        np.asarray([float(row["three_way_weight_unit_investment"]) for row in valid_rows], dtype=np.float64),
        0.0,
        None,
    )
    equal_raw = np.clip(
        np.asarray([float(row["three_way_weight_equal"]) for row in valid_rows], dtype=np.float64),
        0.0,
        None,
    )
    self_raw = np.clip(
        np.asarray([float(row["three_way_weight_self"]) for row in valid_rows], dtype=np.float64),
        0.0,
        None,
    )
    unitself_sum = unit_raw + self_raw
    safe_sum = np.where(unitself_sum > 1e-8, unitself_sum, 1.0)
    target_alpha = np.where(unitself_sum > 1e-8, self_raw / safe_sum, 0.5)
    init_alpha = _weighted_lstsq(design, np.log(_clip_probability_array(target_alpha) / _clip_probability_array(1.0 - target_alpha)))
    coeffs_alpha, optimization_summary = _fit_binary_sigmoid_direct(
        design,
        target_alpha,
        init_coeffs=init_alpha,
    )

    policy = ContinuousTwoStateMixtureAllocationPolicy(
        policy_name="adaptive_continuous2d_unitself_mix",
        profile_family="unitself",
        second_variable_name=str(second_variable_name),
        basis_mode=str(basis_mode),
        coeffs_alpha=coeffs_alpha,
        x1_hinge_knots=x1_hinge_knots,
        x2_hinge_knots=x2_hinge_knots,
        profile_source="agent_rollout_reference_rows_2d",
    )

    alpha_l1_errors: list[float] = []
    simplex_l1_errors: list[float] = []
    equal_dropped_mass: list[float] = []
    for row in valid_rows:
        predicted = np.asarray(
            policy.weights_for_inputs(
                float(np.log1p(max(0.0, float(row["ego_mean_resource"])))),
                float(_extract_continuous_second_value_from_row(row, second_variable_name)),
            ),
            dtype=np.float64,
        )
        target_full = np.asarray(
            [
                float(row["three_way_weight_unit_investment"]),
                float(row["three_way_weight_equal"]),
                float(row["three_way_weight_self"]),
            ],
            dtype=np.float64,
        )
        target_full = _safe_normalize_rows(target_full.reshape(1, -1))[0]
        alpha_target = float(
            target_full[2] / max(target_full[0] + target_full[2], 1e-8)
            if (target_full[0] + target_full[2]) > 1e-8
            else 0.5
        )
        alpha_l1_errors.append(float(abs(predicted[2] - alpha_target)))
        simplex_l1_errors.append(float(np.sum(np.abs(predicted - target_full))))
        equal_dropped_mass.append(float(target_full[1]))

    fit_summary = {
        "fit_status": "agent_rollout_continuous2d_unitself",
        "policy_name": "adaptive_continuous2d_unitself_mix",
        "profile_family": "unitself",
        "second_variable_name": str(second_variable_name),
        "basis_mode": str(basis_mode),
        "fit_row_count": int(len(valid_rows)),
        "basis_dim": int(design.shape[1]),
        "alpha_l1_mean": float(np.mean(alpha_l1_errors)) if alpha_l1_errors else float("nan"),
        "weight_simplex_l1_mean": float(np.mean(simplex_l1_errors)) if simplex_l1_errors else float("nan"),
        "reference_row_fit_l1_mean": (
            float(np.mean([float(row["three_way_fit_l1"]) for row in valid_rows])) if valid_rows else float("nan")
        ),
        "equal_dropped_mass_mean": float(np.mean(equal_dropped_mass)) if equal_dropped_mass else float("nan"),
        "x1_hinge_knots": [float(value) for value in x1_hinge_knots.tolist()],
        "x2_hinge_knots": [float(value) for value in x2_hinge_knots.tolist()],
        **optimization_summary,
    }
    policy.fit_summary_dict = dict(fit_summary)
    return policy, fit_summary


def fit_smooth_two_way_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    tail_cap_mode: str = "none",
    tail_cap_resource: float | None = None,
    tail_cap_soft_scale: float = 0.25,
    ego_resource_x_mode: str = "absolute",
    ego_resource_normalizer: float | None = None,
) -> tuple[SmoothLogEgoResourceMixtureAllocationPolicy, list[dict[str, Any]], dict[str, Any]]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(float(row.get("two_way_weight_unit_investment", float("nan"))))
    ]
    if len(valid_rows) < 3:
        fallback_resources, fallback_weights = _default_adaptive_unitinv_equal_knots()
        fallback_policy = AdaptiveUnitInvestmentEqualAllocationPolicy(
            fallback_resources,
            fallback_weights,
            profile_source="fallback_default_knots_insufficient_agent_rows",
        )
        return fallback_policy, [], {
            "fit_status": "fallback_default_knots",
            "fit_row_count": int(len(valid_rows)),
        }

    resource_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    effective_resource_values = _apply_tail_cap_to_resources(
        resource_values,
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
    )
    profile_x_values = np.asarray(
        [
            _normalize_ego_resource_value(
                value,
                x_mode=str(ego_resource_x_mode),
                normalizer=ego_resource_normalizer,
            )
            for value in effective_resource_values.tolist()
        ],
        dtype=np.float64,
    )
    target_weights = np.asarray(
        [_clip_probability(float(row["two_way_weight_unit_investment"])) for row in valid_rows],
        dtype=np.float64,
    )
    target_logits = np.log(target_weights / (1.0 - target_weights))
    _, design = _design_matrix_from_ego_mean_resource(profile_x_values)
    coeffs = _weighted_lstsq(design, target_logits)

    policy = SmoothLogEgoResourceMixtureAllocationPolicy(
        mode="smooth_2way",
        coeffs_unit_investment=coeffs,
        coeffs_self=None,
        profile_source="agent_rollout_reference_rows",
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
        ego_resource_x_mode=str(ego_resource_x_mode),
        ego_resource_normalizer=ego_resource_normalizer,
    )

    prepared_rows: list[dict[str, Any]] = []
    l1_errors: list[float] = []
    for row in valid_rows:
        weight_unit_investment, weight_equal, weight_self = policy.weights_for_ego_mean_resource(
            float(row["ego_mean_resource"])
        )
        target_unit = float(row["two_way_weight_unit_investment"])
        target_equal = float(row["two_way_weight_equal"])
        l1_error = abs(weight_unit_investment - target_unit) + abs(weight_equal - target_equal)
        l1_errors.append(float(l1_error))
        prepared_rows.append(
            {
                "ego_mean_resource": float(row["ego_mean_resource"]),
                "weight_unit_investment": float(target_unit),
                "weight_equal": float(target_equal),
                "weight_self": 0.0,
                "fit_l1": float(row["two_way_fit_l1"]),
            }
        )

    fit_summary = {
        "fit_status": "agent_rollout_smooth_2way",
        "fit_row_count": int(len(valid_rows)),
        "ego_resource_x_mode": str(ego_resource_x_mode),
        "ego_resource_normalizer": None if ego_resource_normalizer is None else float(ego_resource_normalizer),
        "tail_cap_mode": str(tail_cap_mode),
        "tail_cap_resource": None if tail_cap_resource is None else float(tail_cap_resource),
        "tail_cap_soft_scale": float(tail_cap_soft_scale),
        "weight_simplex_l1_mean": float(np.mean(l1_errors)) if l1_errors else float("nan"),
        "reference_row_fit_l1_mean": (
            float(np.mean([float(row["two_way_fit_l1"]) for row in valid_rows])) if valid_rows else float("nan")
        ),
    }
    policy.fit_summary_dict = dict(fit_summary)
    return policy, prepared_rows, fit_summary


def fit_smooth_three_way_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    tail_cap_mode: str = "none",
    tail_cap_resource: float | None = None,
    tail_cap_soft_scale: float = 0.25,
    ego_resource_x_mode: str = "absolute",
    ego_resource_normalizer: float | None = None,
) -> tuple[SmoothLogEgoResourceMixtureAllocationPolicy, list[dict[str, Any]], dict[str, Any]]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_unit_investment", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_equal", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_self", float("nan"))))
    ]
    if len(valid_rows) < 3:
        fallback_policy, fallback_rows, fallback_summary = fit_smooth_two_way_policy(reference_rows)
        return fallback_policy, fallback_rows, {
            "fit_status": "fallback_smooth_2way_insufficient_rows",
            "fit_row_count": int(len(valid_rows)),
            **dict(fallback_summary),
        }

    resource_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    effective_resource_values = _apply_tail_cap_to_resources(
        resource_values,
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
    )
    profile_x_values = np.asarray(
        [
            _normalize_ego_resource_value(
                value,
                x_mode=str(ego_resource_x_mode),
                normalizer=ego_resource_normalizer,
            )
            for value in effective_resource_values.tolist()
        ],
        dtype=np.float64,
    )
    unit_weights = np.asarray(
        [_clip_probability(float(row["three_way_weight_unit_investment"])) for row in valid_rows],
        dtype=np.float64,
    )
    equal_weights = np.asarray(
        [_clip_probability(float(row["three_way_weight_equal"])) for row in valid_rows],
        dtype=np.float64,
    )
    self_weights = np.asarray(
        [_clip_probability(float(row["three_way_weight_self"])) for row in valid_rows],
        dtype=np.float64,
    )

    _, design = _design_matrix_from_ego_mean_resource(profile_x_values)
    unit_log_ratio = np.log(unit_weights / equal_weights)
    self_log_ratio = np.log(self_weights / equal_weights)
    coeffs_unit = _weighted_lstsq(design, unit_log_ratio)
    coeffs_self = _weighted_lstsq(design, self_log_ratio)

    policy = SmoothLogEgoResourceMixtureAllocationPolicy(
        mode="smooth_3way",
        coeffs_unit_investment=coeffs_unit,
        coeffs_self=coeffs_self,
        profile_source="agent_rollout_reference_rows",
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
        ego_resource_x_mode=str(ego_resource_x_mode),
        ego_resource_normalizer=ego_resource_normalizer,
    )

    prepared_rows: list[dict[str, Any]] = []
    l1_errors: list[float] = []
    for row in valid_rows:
        weight_unit_investment, weight_equal, weight_self = policy.weights_for_ego_mean_resource(
            float(row["ego_mean_resource"])
        )
        target_unit = float(row["three_way_weight_unit_investment"])
        target_equal = float(row["three_way_weight_equal"])
        target_self = float(row["three_way_weight_self"])
        l1_error = (
            abs(weight_unit_investment - target_unit)
            + abs(weight_equal - target_equal)
            + abs(weight_self - target_self)
        )
        l1_errors.append(float(l1_error))
        prepared_rows.append(
            {
                "ego_mean_resource": float(row["ego_mean_resource"]),
                "weight_unit_investment": float(target_unit),
                "weight_equal": float(target_equal),
                "weight_self": float(target_self),
                "fit_l1": float(row["three_way_fit_l1"]),
            }
        )

    fit_summary = {
        "fit_status": "agent_rollout_smooth_3way",
        "fit_row_count": int(len(valid_rows)),
        "ego_resource_x_mode": str(ego_resource_x_mode),
        "ego_resource_normalizer": None if ego_resource_normalizer is None else float(ego_resource_normalizer),
        "tail_cap_mode": str(tail_cap_mode),
        "tail_cap_resource": None if tail_cap_resource is None else float(tail_cap_resource),
        "tail_cap_soft_scale": float(tail_cap_soft_scale),
        "weight_simplex_l1_mean": float(np.mean(l1_errors)) if l1_errors else float("nan"),
        "reference_row_fit_l1_mean": (
            float(np.mean([float(row["three_way_fit_l1"]) for row in valid_rows])) if valid_rows else float("nan")
        ),
    }
    policy.fit_summary_dict = dict(fit_summary)
    return policy, prepared_rows, fit_summary


def fit_spline_three_way_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    spline_high_knot_style: str = "sparse",
    tail_cap_mode: str = "none",
    tail_cap_resource: float | None = None,
    tail_cap_soft_scale: float = 0.25,
    l2_penalty: float = 1e-3,
    ego_resource_x_mode: str = "absolute",
    ego_resource_normalizer: float | None = None,
) -> tuple[SmoothLogEgoResourceMixtureAllocationPolicy, list[dict[str, Any]], dict[str, Any]]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_unit_investment", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_equal", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_self", float("nan"))))
    ]
    if len(valid_rows) < 8:
        fallback_policy, fallback_rows, fallback_summary = fit_smooth_three_way_policy(
            reference_rows,
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
            ego_resource_x_mode=str(ego_resource_x_mode),
            ego_resource_normalizer=ego_resource_normalizer,
        )
        return fallback_policy, fallback_rows, {
            "fit_status": "fallback_smooth_3way_insufficient_rows_for_spline",
            "fit_row_count": int(len(valid_rows)),
            **dict(fallback_summary),
        }

    resource_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    effective_resource_values = _apply_tail_cap_to_resources(
        resource_values,
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
    )
    profile_x_values = np.asarray(
        [
            _normalize_ego_resource_value(
                value,
                x_mode=str(ego_resource_x_mode),
                normalizer=ego_resource_normalizer,
            )
            for value in effective_resource_values.tolist()
        ],
        dtype=np.float64,
    )
    spline_knots_resource = _build_spline_resource_knots(
        profile_x_values,
        high_knot_style=str(spline_high_knot_style),
    )
    if spline_knots_resource.size == 0:
        fallback_policy, fallback_rows, fallback_summary = fit_smooth_three_way_policy(
            reference_rows,
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
            ego_resource_x_mode=str(ego_resource_x_mode),
            ego_resource_normalizer=ego_resource_normalizer,
        )
        return fallback_policy, fallback_rows, {
            "fit_status": "fallback_smooth_3way_no_valid_spline_knots",
            "fit_row_count": int(len(valid_rows)),
            **dict(fallback_summary),
        }

    unit_weights = np.asarray(
        [float(row["three_way_weight_unit_investment"]) for row in valid_rows],
        dtype=np.float64,
    )
    equal_weights = np.asarray(
        [float(row["three_way_weight_equal"]) for row in valid_rows],
        dtype=np.float64,
    )
    self_weights = np.asarray(
        [float(row["three_way_weight_self"]) for row in valid_rows],
        dtype=np.float64,
    )
    target_weights = _safe_normalize_rows(np.column_stack([unit_weights, equal_weights, self_weights]))

    _, design = _design_matrix_from_ego_mean_resource_spline(profile_x_values, spline_knots_resource)
    init_unit = _weighted_lstsq(
        design,
        np.log(_clip_probability_array(unit_weights) / _clip_probability_array(equal_weights)),
    )
    init_self = _weighted_lstsq(
        design,
        np.log(_clip_probability_array(self_weights) / _clip_probability_array(equal_weights)),
    )
    coeffs_unit, coeffs_self, optimization_summary = _fit_three_way_softmax_direct(
        design,
        target_weights,
        init_unit=init_unit,
        init_self=init_self,
        l2_penalty=float(l2_penalty),
    )

    policy = SmoothLogEgoResourceMixtureAllocationPolicy(
        mode="spline_3way",
        coeffs_unit_investment=coeffs_unit,
        coeffs_self=coeffs_self,
        profile_source="agent_rollout_reference_rows",
        spline_knots_resource=spline_knots_resource,
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
        ego_resource_x_mode=str(ego_resource_x_mode),
        ego_resource_normalizer=ego_resource_normalizer,
    )

    prepared_rows: list[dict[str, Any]] = []
    l1_errors: list[float] = []
    high_resource_mask = resource_values >= 430.0
    high_resource_errors: list[float] = []
    for row, target_weight_vector in zip(valid_rows, target_weights.tolist()):
        weight_unit_investment, weight_equal, weight_self = policy.weights_for_ego_mean_resource(
            float(row["ego_mean_resource"])
        )
        predicted = np.asarray([weight_unit_investment, weight_equal, weight_self], dtype=np.float64)
        target = np.asarray(target_weight_vector, dtype=np.float64)
        simplex_l1 = float(np.sum(np.abs(predicted - target)))
        l1_errors.append(simplex_l1)
        if float(row["ego_mean_resource"]) >= 430.0:
            high_resource_errors.append(simplex_l1)
        prepared_rows.append(
            {
                "ego_mean_resource": float(row["ego_mean_resource"]),
                "weight_unit_investment": float(target[0]),
                "weight_equal": float(target[1]),
                "weight_self": float(target[2]),
                "fit_l1": float(row["three_way_fit_l1"]),
            }
        )

    fit_summary = {
        "fit_status": "agent_rollout_spline_3way",
        "fit_row_count": int(len(valid_rows)),
        "ego_resource_x_mode": str(ego_resource_x_mode),
        "ego_resource_normalizer": None if ego_resource_normalizer is None else float(ego_resource_normalizer),
        "spline_high_knot_style": str(spline_high_knot_style),
        "tail_cap_mode": str(tail_cap_mode),
        "tail_cap_resource": None if tail_cap_resource is None else float(tail_cap_resource),
        "tail_cap_soft_scale": float(tail_cap_soft_scale),
        "effective_resource_max": float(np.max(effective_resource_values)) if effective_resource_values.size else float("nan"),
        "raw_resource_max": float(np.max(resource_values)) if resource_values.size else float("nan"),
        "weight_simplex_l1_mean": float(np.mean(l1_errors)) if l1_errors else float("nan"),
        "weight_simplex_l1_mean_high_resource_430plus": (
            float(np.mean(high_resource_errors)) if high_resource_errors else float("nan")
        ),
        "reference_row_fit_l1_mean": (
            float(np.mean([float(row["three_way_fit_l1"]) for row in valid_rows])) if valid_rows else float("nan")
        ),
        "spline_knot_count": int(spline_knots_resource.size),
        "spline_high_resource_knot_count_430_450": int(np.sum((spline_knots_resource >= 430.0) & (spline_knots_resource <= 450.0))),
        "spline_knots_resource": [float(value) for value in spline_knots_resource.tolist()],
        **optimization_summary,
    }
    policy.fit_summary_dict = dict(fit_summary)
    return policy, prepared_rows, fit_summary


def fit_spline_unitself_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    spline_high_knot_style: str = "sparse",
    tail_cap_mode: str = "none",
    tail_cap_resource: float | None = None,
    tail_cap_soft_scale: float = 0.25,
    l2_penalty: float = 1e-3,
    ego_resource_x_mode: str = "absolute",
    ego_resource_normalizer: float | None = None,
) -> tuple[SmoothLogEgoResourceMixtureAllocationPolicy, list[dict[str, Any]], dict[str, Any]]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_unit_investment", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_self", float("nan"))))
    ]
    if len(valid_rows) < 8:
        fallback_policy, fallback_rows, fallback_summary = fit_spline_three_way_policy(
            reference_rows,
            spline_high_knot_style=str(spline_high_knot_style),
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
            l2_penalty=float(l2_penalty),
            ego_resource_x_mode=str(ego_resource_x_mode),
            ego_resource_normalizer=ego_resource_normalizer,
        )
        return fallback_policy, fallback_rows, {
            "fit_status": "fallback_spline_3way_insufficient_rows_for_unitself",
            "fit_row_count": int(len(valid_rows)),
            **dict(fallback_summary),
        }

    resource_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    effective_resource_values = _apply_tail_cap_to_resources(
        resource_values,
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
    )
    profile_x_values = np.asarray(
        [
            _normalize_ego_resource_value(
                value,
                x_mode=str(ego_resource_x_mode),
                normalizer=ego_resource_normalizer,
            )
            for value in effective_resource_values.tolist()
        ],
        dtype=np.float64,
    )
    spline_knots_resource = _build_spline_resource_knots(
        profile_x_values,
        high_knot_style=str(spline_high_knot_style),
    )
    if spline_knots_resource.size == 0:
        fallback_policy, fallback_rows, fallback_summary = fit_spline_three_way_policy(
            reference_rows,
            spline_high_knot_style=str(spline_high_knot_style),
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
            l2_penalty=float(l2_penalty),
            ego_resource_x_mode=str(ego_resource_x_mode),
            ego_resource_normalizer=ego_resource_normalizer,
        )
        return fallback_policy, fallback_rows, {
            "fit_status": "fallback_spline_3way_no_valid_unitself_knots",
            "fit_row_count": int(len(valid_rows)),
            **dict(fallback_summary),
        }

    unit_raw = np.clip(
        np.asarray([float(row["three_way_weight_unit_investment"]) for row in valid_rows], dtype=np.float64),
        0.0,
        None,
    )
    self_raw = np.clip(
        np.asarray([float(row["three_way_weight_self"]) for row in valid_rows], dtype=np.float64),
        0.0,
        None,
    )
    unitself_sum = unit_raw + self_raw
    safe_sum = np.where(unitself_sum > 1e-8, unitself_sum, 1.0)
    target_unit = np.where(unitself_sum > 1e-8, unit_raw / safe_sum, 0.5)
    target_self = np.where(unitself_sum > 1e-8, self_raw / safe_sum, 0.5)
    target_weights = _safe_normalize_rows(
        np.column_stack([target_unit, np.zeros_like(target_unit), target_self])
    )

    _, design = _design_matrix_from_ego_mean_resource_spline(profile_x_values, spline_knots_resource)
    coeffs_unit, coeffs_self, optimization_summary = _fit_three_way_softmax_direct(
        design,
        target_weights,
        init_unit=np.zeros(design.shape[1], dtype=np.float64),
        init_self=np.zeros(design.shape[1], dtype=np.float64),
        l2_penalty=float(l2_penalty),
    )

    policy = SmoothLogEgoResourceMixtureAllocationPolicy(
        mode="spline_3way",
        coeffs_unit_investment=coeffs_unit,
        coeffs_self=coeffs_self,
        profile_source="agent_rollout_reference_rows_unitself",
        spline_knots_resource=spline_knots_resource,
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
        ego_resource_x_mode=str(ego_resource_x_mode),
        ego_resource_normalizer=ego_resource_normalizer,
    )

    prepared_rows: list[dict[str, Any]] = []
    l1_errors: list[float] = []
    for row, target_weight_vector in zip(valid_rows, target_weights.tolist()):
        weight_unit_investment, weight_equal, weight_self = policy.weights_for_ego_mean_resource(
            float(row["ego_mean_resource"])
        )
        predicted = np.asarray([weight_unit_investment, weight_equal, weight_self], dtype=np.float64)
        target = np.asarray(target_weight_vector, dtype=np.float64)
        simplex_l1 = float(np.sum(np.abs(predicted - target)))
        l1_errors.append(simplex_l1)
        prepared_rows.append(
            {
                "ego_mean_resource": float(row["ego_mean_resource"]),
                "weight_unit_investment": float(target[0]),
                "weight_equal": 0.0,
                "weight_self": float(target[2]),
                "fit_l1": float(row["three_way_fit_l1"]),
            }
        )

    fit_summary = {
        "fit_status": "agent_rollout_spline_unitself",
        "fit_row_count": int(len(valid_rows)),
        "ego_resource_x_mode": str(ego_resource_x_mode),
        "ego_resource_normalizer": None if ego_resource_normalizer is None else float(ego_resource_normalizer),
        "spline_high_knot_style": str(spline_high_knot_style),
        "tail_cap_mode": str(tail_cap_mode),
        "tail_cap_resource": None if tail_cap_resource is None else float(tail_cap_resource),
        "tail_cap_soft_scale": float(tail_cap_soft_scale),
        "effective_resource_max": float(np.max(effective_resource_values)) if effective_resource_values.size else float("nan"),
        "raw_resource_max": float(np.max(resource_values)) if resource_values.size else float("nan"),
        "weight_simplex_l1_mean": float(np.mean(l1_errors)) if l1_errors else float("nan"),
        "reference_row_fit_l1_mean": (
            float(np.mean([float(row["three_way_fit_l1"]) for row in valid_rows])) if valid_rows else float("nan")
        ),
        "spline_knot_count": int(spline_knots_resource.size),
        "spline_high_resource_knot_count_430_450": int(np.sum((spline_knots_resource >= 430.0) & (spline_knots_resource <= 450.0))),
        "spline_knots_resource": [float(value) for value in spline_knots_resource.tolist()],
        **optimization_summary,
    }
    policy.fit_summary_dict = dict(fit_summary)
    return policy, prepared_rows, fit_summary


def _prepare_binned_three_way_targets(
    reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_unit_investment", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_equal", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_self", float("nan"))))
    ]
    resource_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    target_weights = _safe_normalize_rows(
        np.column_stack(
            [
                np.asarray([float(row["three_way_weight_unit_investment"]) for row in valid_rows], dtype=np.float64),
                np.asarray([float(row["three_way_weight_equal"]) for row in valid_rows], dtype=np.float64),
                np.asarray([float(row["three_way_weight_self"]) for row in valid_rows], dtype=np.float64),
            ]
        )
    )
    return valid_rows, resource_values, resource_values.copy(), target_weights


def _build_binned_weight_rows(
    effective_resource_values: np.ndarray,
    target_weights: np.ndarray,
    *,
    bin_count: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    edges = _quantile_edges(np.asarray(effective_resource_values, dtype=np.float64), max(2, int(bin_count)))
    if edges.size < 2:
        minimum = float(np.min(effective_resource_values))
        maximum = float(np.max(effective_resource_values))
        if abs(maximum - minimum) <= 1e-12:
            maximum = minimum + 1e-9
        edges = np.asarray([minimum, maximum], dtype=np.float64)

    bin_weight_rows: list[np.ndarray] = []
    bin_rows_summary: list[dict[str, Any]] = []
    for bin_index in range(max(0, edges.size - 1)):
        left = float(edges[bin_index])
        right = float(edges[bin_index + 1])
        if bin_index == edges.size - 2:
            mask = (effective_resource_values >= left) & (effective_resource_values <= right)
        else:
            mask = (effective_resource_values >= left) & (effective_resource_values < right)
        if not np.any(mask):
            continue
        mean_weights = _safe_normalize_rows(np.mean(target_weights[mask], axis=0, keepdims=True))[0]
        bin_weight_rows.append(np.asarray(mean_weights, dtype=np.float64))
        bin_rows_summary.append(
            {
                "bin_index": int(len(bin_weight_rows) - 1),
                "bin_left": float(left),
                "bin_right": float(right),
                "sample_count": int(np.sum(mask)),
                "weight_unit_investment": float(mean_weights[0]),
                "weight_equal": float(mean_weights[1]),
                "weight_self": float(mean_weights[2]),
            }
        )

    if not bin_weight_rows:
        raise ValueError("Binned profile construction failed because no effective ego-resource bins had samples.")

    effective_edges = np.asarray(
        [bin_rows_summary[0]["bin_left"], *[row["bin_right"] for row in bin_rows_summary]],
        dtype=np.float64,
    )
    return effective_edges, np.vstack(bin_weight_rows), bin_rows_summary


def fit_binned_three_way_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    bin_count: int,
    tail_cap_mode: str = "none",
    tail_cap_resource: float | None = None,
    tail_cap_soft_scale: float = 0.25,
    ego_resource_x_mode: str = "absolute",
    ego_resource_normalizer: float | None = None,
) -> tuple[BinnedEgoResourceMixtureAllocationPolicy | SmoothLogEgoResourceMixtureAllocationPolicy, list[dict[str, Any]], dict[str, Any]]:
    valid_rows, resource_values, _, target_weights = _prepare_binned_three_way_targets(reference_rows)
    if len(valid_rows) < max(6, int(bin_count)):
        fallback_policy, fallback_rows, fallback_summary = fit_spline_three_way_policy(
            reference_rows,
            spline_high_knot_style="sparse",
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
            ego_resource_x_mode=str(ego_resource_x_mode),
            ego_resource_normalizer=ego_resource_normalizer,
            l2_penalty=1e-3,
        )
        return fallback_policy, fallback_rows, {
            "fit_status": "fallback_spline_3way_insufficient_rows_for_binned",
            "fit_row_count": int(len(valid_rows)),
            **dict(fallback_summary),
        }

    effective_resource_values = _apply_tail_cap_to_resources(
        resource_values,
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
    )
    profile_x_values = np.asarray(
        [
            _normalize_ego_resource_value(
                value,
                x_mode=str(ego_resource_x_mode),
                normalizer=ego_resource_normalizer,
            )
            for value in effective_resource_values.tolist()
        ],
        dtype=np.float64,
    )
    bin_edges, bin_weight_rows, bin_rows_summary = _build_binned_weight_rows(
        profile_x_values,
        target_weights,
        bin_count=max(2, int(bin_count)),
    )
    policy = BinnedEgoResourceMixtureAllocationPolicy(
        mode="binned_3way",
        bin_edges_resource=bin_edges,
        bin_weight_rows=bin_weight_rows,
        profile_source="agent_rollout_reference_rows_binned",
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
        ego_resource_x_mode=str(ego_resource_x_mode),
        ego_resource_normalizer=ego_resource_normalizer,
    )

    prepared_rows: list[dict[str, Any]] = []
    l1_errors: list[float] = []
    for row, target_weight_vector in zip(valid_rows, target_weights.tolist()):
        predicted = np.asarray(policy.weights_for_ego_mean_resource(float(row["ego_mean_resource"])), dtype=np.float64)
        target = np.asarray(target_weight_vector, dtype=np.float64)
        simplex_l1 = float(np.sum(np.abs(predicted - target)))
        l1_errors.append(simplex_l1)
        prepared_rows.append(
            {
                "ego_mean_resource": float(row["ego_mean_resource"]),
                "weight_unit_investment": float(target[0]),
                "weight_equal": float(target[1]),
                "weight_self": float(target[2]),
                "fit_l1": float(row["three_way_fit_l1"]),
            }
        )

    fit_summary = {
        "fit_status": "agent_rollout_binned_3way",
        "fit_row_count": int(len(valid_rows)),
        "adaptive_ego_resource_profile_mode": "discrete",
        "adaptive_ego_resource_bin_count": int(bin_weight_rows.shape[0]),
        "ego_resource_x_mode": str(ego_resource_x_mode),
        "ego_resource_normalizer": None if ego_resource_normalizer is None else float(ego_resource_normalizer),
        "tail_cap_mode": str(tail_cap_mode),
        "tail_cap_resource": None if tail_cap_resource is None else float(tail_cap_resource),
        "tail_cap_soft_scale": float(tail_cap_soft_scale),
        "weight_simplex_l1_mean": float(np.mean(l1_errors)) if l1_errors else float("nan"),
        "reference_row_fit_l1_mean": (
            float(np.mean([float(row["three_way_fit_l1"]) for row in valid_rows])) if valid_rows else float("nan")
        ),
        "bin_rows_summary": bin_rows_summary,
        "bin_edges_resource": [float(value) for value in bin_edges.tolist()],
    }
    policy.fit_summary_dict = dict(fit_summary)
    return policy, prepared_rows, fit_summary


def fit_binned_unitself_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    bin_count: int,
    tail_cap_mode: str = "none",
    tail_cap_resource: float | None = None,
    tail_cap_soft_scale: float = 0.25,
    ego_resource_x_mode: str = "absolute",
    ego_resource_normalizer: float | None = None,
) -> tuple[BinnedEgoResourceMixtureAllocationPolicy | SmoothLogEgoResourceMixtureAllocationPolicy, list[dict[str, Any]], dict[str, Any]]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_unit_investment", float("nan"))))
        and np.isfinite(float(row.get("three_way_weight_self", float("nan"))))
    ]
    if len(valid_rows) < max(6, int(bin_count)):
        fallback_policy, fallback_rows, fallback_summary = fit_spline_unitself_policy(
            reference_rows,
            spline_high_knot_style="sparse",
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
            ego_resource_x_mode=str(ego_resource_x_mode),
            ego_resource_normalizer=ego_resource_normalizer,
            l2_penalty=1e-3,
        )
        return fallback_policy, fallback_rows, {
            "fit_status": "fallback_spline_unitself_insufficient_rows_for_binned",
            "fit_row_count": int(len(valid_rows)),
            **dict(fallback_summary),
        }

    resource_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    effective_resource_values = _apply_tail_cap_to_resources(
        resource_values,
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
    )
    profile_x_values = np.asarray(
        [
            _normalize_ego_resource_value(
                value,
                x_mode=str(ego_resource_x_mode),
                normalizer=ego_resource_normalizer,
            )
            for value in effective_resource_values.tolist()
        ],
        dtype=np.float64,
    )
    unit_raw = np.clip(
        np.asarray([float(row["three_way_weight_unit_investment"]) for row in valid_rows], dtype=np.float64),
        0.0,
        None,
    )
    self_raw = np.clip(
        np.asarray([float(row["three_way_weight_self"]) for row in valid_rows], dtype=np.float64),
        0.0,
        None,
    )
    unitself_sum = unit_raw + self_raw
    target_unit = np.where(unitself_sum > 1e-8, unit_raw / np.where(unitself_sum > 1e-8, unitself_sum, 1.0), 0.5)
    target_self = np.where(unitself_sum > 1e-8, self_raw / np.where(unitself_sum > 1e-8, unitself_sum, 1.0), 0.5)
    target_weights = _safe_normalize_rows(np.column_stack([target_unit, np.zeros_like(target_unit), target_self]))
    bin_edges, bin_weight_rows, bin_rows_summary = _build_binned_weight_rows(
        profile_x_values,
        target_weights,
        bin_count=max(2, int(bin_count)),
    )
    policy = BinnedEgoResourceMixtureAllocationPolicy(
        mode="binned_unitself",
        bin_edges_resource=bin_edges,
        bin_weight_rows=bin_weight_rows,
        profile_source="agent_rollout_reference_rows_binned_unitself",
        tail_cap_mode=str(tail_cap_mode),
        tail_cap_resource=tail_cap_resource,
        tail_cap_soft_scale=float(tail_cap_soft_scale),
        ego_resource_x_mode=str(ego_resource_x_mode),
        ego_resource_normalizer=ego_resource_normalizer,
    )

    prepared_rows: list[dict[str, Any]] = []
    l1_errors: list[float] = []
    for row, target_weight_vector in zip(valid_rows, target_weights.tolist()):
        predicted = np.asarray(policy.weights_for_ego_mean_resource(float(row["ego_mean_resource"])), dtype=np.float64)
        target = np.asarray(target_weight_vector, dtype=np.float64)
        simplex_l1 = float(np.sum(np.abs(predicted - target)))
        l1_errors.append(simplex_l1)
        prepared_rows.append(
            {
                "ego_mean_resource": float(row["ego_mean_resource"]),
                "weight_unit_investment": float(target[0]),
                "weight_equal": 0.0,
                "weight_self": float(target[2]),
                "fit_l1": float(row["three_way_fit_l1"]),
            }
        )

    fit_summary = {
        "fit_status": "agent_rollout_binned_unitself",
        "fit_row_count": int(len(valid_rows)),
        "adaptive_ego_resource_profile_mode": "discrete",
        "adaptive_ego_resource_bin_count": int(bin_weight_rows.shape[0]),
        "ego_resource_x_mode": str(ego_resource_x_mode),
        "ego_resource_normalizer": None if ego_resource_normalizer is None else float(ego_resource_normalizer),
        "tail_cap_mode": str(tail_cap_mode),
        "tail_cap_resource": None if tail_cap_resource is None else float(tail_cap_resource),
        "tail_cap_soft_scale": float(tail_cap_soft_scale),
        "weight_simplex_l1_mean": float(np.mean(l1_errors)) if l1_errors else float("nan"),
        "reference_row_fit_l1_mean": (
            float(np.mean([float(row["three_way_fit_l1"]) for row in valid_rows])) if valid_rows else float("nan")
        ),
        "bin_rows_summary": bin_rows_summary,
        "bin_edges_resource": [float(value) for value in bin_edges.tolist()],
    }
    policy.fit_summary_dict = dict(fit_summary)
    return policy, prepared_rows, fit_summary


def fit_degree_conditioned_spline_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    degree_bin_mode: str,
    degree_bin_count: int,
    degree_bin_min_count: int,
    profile_family: str,
    reference_bin_count: int,
    ego_resource_profile_mode: str,
    ego_resource_bin_count: int,
    ego_resource_x_mode: str,
    ego_resource_normalizer: float | None,
    spline_high_knot_style: str,
    tail_cap_mode: str,
    tail_cap_resource: float | None,
    tail_cap_soft_scale: float,
    spline_l2_penalty: float,
) -> tuple[ConditionedSplineMixtureAllocationPolicy, list[dict[str, Any]], dict[str, Any], str]:
    annotated_rows = _assign_degree_bin_fields(
        reference_rows,
        degree_key="sender_degree",
        spec=_build_degree_bin_spec(
            [row.get("sender_degree", 0) for row in reference_rows],
            mode=str(degree_bin_mode),
            bin_count=max(1, int(degree_bin_count)),
            min_count=max(1, int(degree_bin_min_count)),
        ),
        prefix="sender",
    )
    degree_spec = _build_degree_bin_spec(
        [row.get("sender_degree", 0) for row in reference_rows],
        mode=str(degree_bin_mode),
        bin_count=max(1, int(degree_bin_count)),
        min_count=max(1, int(degree_bin_min_count)),
    )
    annotated_rows = _assign_degree_bin_fields(
        reference_rows,
        degree_key="sender_degree",
        spec=degree_spec,
        prefix="sender",
    )

    policies_by_bin: dict[str, Any] = {}
    conditioned_profile_rows: list[dict[str, Any]] = []
    per_bin_summary: dict[str, Any] = {}
    weighted_error_terms: list[tuple[float, int]] = []

    for bin_info in list(degree_spec.get("bins", [])):
        bin_label = str(bin_info["bin_label"])
        bin_index = int(bin_info["bin_index"])
        bin_rows = [row for row in annotated_rows if str(row["sender_degree_bin"]) == bin_label]
        if str(profile_family) == "unitself":
            if str(ego_resource_profile_mode) == "discrete":
                subpolicy, prepared_rows, fit_summary = fit_binned_unitself_policy(
                    bin_rows,
                    bin_count=max(2, int(ego_resource_bin_count)),
                    tail_cap_mode=str(tail_cap_mode),
                    tail_cap_resource=tail_cap_resource,
                    tail_cap_soft_scale=float(tail_cap_soft_scale),
                    ego_resource_x_mode=str(ego_resource_x_mode),
                    ego_resource_normalizer=ego_resource_normalizer,
                )
            else:
                subpolicy, prepared_rows, fit_summary = fit_spline_unitself_policy(
                    bin_rows,
                    spline_high_knot_style=str(spline_high_knot_style),
                    tail_cap_mode=str(tail_cap_mode),
                    tail_cap_resource=tail_cap_resource,
                    tail_cap_soft_scale=float(tail_cap_soft_scale),
                    l2_penalty=float(spline_l2_penalty),
                    ego_resource_x_mode=str(ego_resource_x_mode),
                    ego_resource_normalizer=ego_resource_normalizer,
                )
            policy_name = "adaptive_degree_conditioned_unitself_mix"
        else:
            if str(ego_resource_profile_mode) == "discrete":
                subpolicy, prepared_rows, fit_summary = fit_binned_three_way_policy(
                    bin_rows,
                    bin_count=max(2, int(ego_resource_bin_count)),
                    tail_cap_mode=str(tail_cap_mode),
                    tail_cap_resource=tail_cap_resource,
                    tail_cap_soft_scale=float(tail_cap_soft_scale),
                    ego_resource_x_mode=str(ego_resource_x_mode),
                    ego_resource_normalizer=ego_resource_normalizer,
                )
            else:
                subpolicy, prepared_rows, fit_summary = fit_spline_three_way_policy(
                    bin_rows,
                    spline_high_knot_style=str(spline_high_knot_style),
                    tail_cap_mode=str(tail_cap_mode),
                    tail_cap_resource=tail_cap_resource,
                    tail_cap_soft_scale=float(tail_cap_soft_scale),
                    l2_penalty=float(spline_l2_penalty),
                    ego_resource_x_mode=str(ego_resource_x_mode),
                    ego_resource_normalizer=ego_resource_normalizer,
                )
            policy_name = "adaptive_degree_conditioned_spline_3way_mix"
        policies_by_bin[bin_label] = subpolicy
        per_bin_summary[bin_label] = dict(fit_summary)
        weighted_error_terms.append((float(fit_summary.get("weight_simplex_l1_mean", float("nan"))), len(bin_rows)))

        profile_rows = build_adaptive_profile_rows(
            subpolicy,
            reference_rows=prepared_rows,
            reference_bin_count=max(2, int(reference_bin_count)),
        )
        for row in profile_rows:
            enriched = dict(row)
            enriched["sender_degree_bin"] = bin_label
            enriched["sender_degree_bin_index"] = int(bin_index)
            enriched["sender_degree_bin_left"] = int(bin_info["degree_left"])
            enriched["sender_degree_bin_right"] = int(bin_info["degree_right"])
            conditioned_profile_rows.append(enriched)

    weighted_error_values = [
        error * float(count)
        for error, count in weighted_error_terms
        if np.isfinite(float(error)) and int(count) > 0
    ]
    total_weight_count = int(sum(int(count) for error, count in weighted_error_terms if np.isfinite(float(error)) and int(count) > 0))
    fit_summary = {
        "fit_status": f"agent_rollout_degree_conditioned_{profile_family}",
        "conditioning_mode": "sender_degree_bin",
        "profile_family": str(profile_family),
        "adaptive_ego_resource_profile_mode": str(ego_resource_profile_mode),
        "adaptive_ego_resource_bin_count": int(ego_resource_bin_count),
        "ego_resource_x_mode": str(ego_resource_x_mode),
        "ego_resource_normalizer": None if ego_resource_normalizer is None else float(ego_resource_normalizer),
        "spline_high_knot_style": str(spline_high_knot_style),
        "tail_cap_mode": str(tail_cap_mode),
        "tail_cap_resource": None if tail_cap_resource is None else float(tail_cap_resource),
        "tail_cap_soft_scale": float(tail_cap_soft_scale),
        "spline_l2_penalty": float(spline_l2_penalty),
        "condition_bin_count": int(len(policies_by_bin)),
        "weight_simplex_l1_mean": (
            float(np.sum(weighted_error_values) / float(max(total_weight_count, 1)))
            if weighted_error_values
            else float("nan")
        ),
        "condition_spec": {
            key: value
            for key, value in degree_spec.items()
            if key != "value_to_bin"
        },
        "per_bin_fit_summary": per_bin_summary,
    }
    conditioned_policy = ConditionedSplineMixtureAllocationPolicy(
        conditioning_mode="sender_degree_bin",
        profile_family=str(profile_family),
        condition_spec=degree_spec,
        policies_by_bin=policies_by_bin,
        profile_source="agent_rollout_reference_rows",
        fit_summary=fit_summary,
    )
    return conditioned_policy, conditioned_profile_rows, fit_summary, policy_name


def fit_resource_conditioned_spline_policy(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    resource_bin_count: int,
    profile_family: str,
    reference_bin_count: int,
    ego_resource_profile_mode: str,
    ego_resource_bin_count: int,
    ego_resource_x_mode: str,
    ego_resource_normalizer: float | None,
    spline_high_knot_style: str,
    tail_cap_mode: str,
    tail_cap_resource: float | None,
    tail_cap_soft_scale: float,
    spline_l2_penalty: float,
) -> tuple[ConditionedSplineMixtureAllocationPolicy, list[dict[str, Any]], dict[str, Any], str]:
    resource_spec = _build_continuous_bin_spec(
        [row.get("sender_resource_norm", float("nan")) for row in reference_rows],
        bin_count=max(1, int(resource_bin_count)),
        label_prefix="Q",
    )
    annotated_rows = _assign_continuous_bin_fields(
        reference_rows,
        value_key="sender_resource_norm",
        spec=resource_spec,
        prefix="sender_resource_norm",
    )

    policies_by_bin: dict[str, Any] = {}
    conditioned_profile_rows: list[dict[str, Any]] = []
    per_bin_summary: dict[str, Any] = {}
    weighted_error_terms: list[tuple[float, int]] = []

    for bin_info in list(resource_spec.get("bins", [])):
        bin_label = str(bin_info["bin_label"])
        bin_index = int(bin_info["bin_index"])
        bin_rows = [row for row in annotated_rows if str(row["sender_resource_norm_bin"]) == bin_label]
        if str(profile_family) == "unitself":
            if str(ego_resource_profile_mode) == "discrete":
                subpolicy, prepared_rows, fit_summary = fit_binned_unitself_policy(
                    bin_rows,
                    bin_count=max(2, int(ego_resource_bin_count)),
                    tail_cap_mode=str(tail_cap_mode),
                    tail_cap_resource=tail_cap_resource,
                    tail_cap_soft_scale=float(tail_cap_soft_scale),
                    ego_resource_x_mode=str(ego_resource_x_mode),
                    ego_resource_normalizer=ego_resource_normalizer,
                )
            else:
                subpolicy, prepared_rows, fit_summary = fit_spline_unitself_policy(
                    bin_rows,
                    spline_high_knot_style=str(spline_high_knot_style),
                    tail_cap_mode=str(tail_cap_mode),
                    tail_cap_resource=tail_cap_resource,
                    tail_cap_soft_scale=float(tail_cap_soft_scale),
                    l2_penalty=float(spline_l2_penalty),
                    ego_resource_x_mode=str(ego_resource_x_mode),
                    ego_resource_normalizer=ego_resource_normalizer,
                )
            policy_name = "adaptive_resource_conditioned_unitself_mix"
        else:
            if str(ego_resource_profile_mode) == "discrete":
                subpolicy, prepared_rows, fit_summary = fit_binned_three_way_policy(
                    bin_rows,
                    bin_count=max(2, int(ego_resource_bin_count)),
                    tail_cap_mode=str(tail_cap_mode),
                    tail_cap_resource=tail_cap_resource,
                    tail_cap_soft_scale=float(tail_cap_soft_scale),
                    ego_resource_x_mode=str(ego_resource_x_mode),
                    ego_resource_normalizer=ego_resource_normalizer,
                )
            else:
                subpolicy, prepared_rows, fit_summary = fit_spline_three_way_policy(
                    bin_rows,
                    spline_high_knot_style=str(spline_high_knot_style),
                    tail_cap_mode=str(tail_cap_mode),
                    tail_cap_resource=tail_cap_resource,
                    tail_cap_soft_scale=float(tail_cap_soft_scale),
                    l2_penalty=float(spline_l2_penalty),
                    ego_resource_x_mode=str(ego_resource_x_mode),
                    ego_resource_normalizer=ego_resource_normalizer,
                )
            policy_name = "adaptive_resource_conditioned_spline_3way_mix"
        policies_by_bin[bin_label] = subpolicy
        per_bin_summary[bin_label] = dict(fit_summary)
        weighted_error_terms.append((float(fit_summary.get("weight_simplex_l1_mean", float("nan"))), len(bin_rows)))

        profile_rows = build_adaptive_profile_rows(
            subpolicy,
            reference_rows=prepared_rows,
            reference_bin_count=max(2, int(reference_bin_count)),
        )
        for row in profile_rows:
            enriched = dict(row)
            enriched["sender_resource_norm_bin"] = bin_label
            enriched["sender_resource_norm_bin_index"] = int(bin_index)
            enriched["sender_resource_norm_bin_left"] = float(bin_info["value_left"])
            enriched["sender_resource_norm_bin_right"] = float(bin_info["value_right"])
            conditioned_profile_rows.append(enriched)

    weighted_error_values = [
        error * float(count)
        for error, count in weighted_error_terms
        if np.isfinite(float(error)) and int(count) > 0
    ]
    total_weight_count = int(sum(int(count) for error, count in weighted_error_terms if np.isfinite(float(error)) and int(count) > 0))
    fit_summary = {
        "fit_status": f"agent_rollout_resource_conditioned_{profile_family}",
        "conditioning_mode": "sender_resource_norm_bin",
        "profile_family": str(profile_family),
        "adaptive_ego_resource_profile_mode": str(ego_resource_profile_mode),
        "adaptive_ego_resource_bin_count": int(ego_resource_bin_count),
        "ego_resource_x_mode": str(ego_resource_x_mode),
        "ego_resource_normalizer": None if ego_resource_normalizer is None else float(ego_resource_normalizer),
        "spline_high_knot_style": str(spline_high_knot_style),
        "tail_cap_mode": str(tail_cap_mode),
        "tail_cap_resource": None if tail_cap_resource is None else float(tail_cap_resource),
        "tail_cap_soft_scale": float(tail_cap_soft_scale),
        "spline_l2_penalty": float(spline_l2_penalty),
        "condition_bin_count": int(len(policies_by_bin)),
        "weight_simplex_l1_mean": (
            float(np.sum(weighted_error_values) / float(max(total_weight_count, 1)))
            if weighted_error_values
            else float("nan")
        ),
        "condition_spec": dict(resource_spec),
        "per_bin_fit_summary": per_bin_summary,
    }
    conditioned_policy = ConditionedSplineMixtureAllocationPolicy(
        conditioning_mode="sender_resource_norm_bin",
        profile_family=str(profile_family),
        condition_spec=resource_spec,
        policies_by_bin=policies_by_bin,
        profile_source="agent_rollout_reference_rows",
        fit_summary=fit_summary,
    )
    return conditioned_policy, conditioned_profile_rows, fit_summary, policy_name


def _quantile_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.asarray([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, num=max(2, int(bin_count) + 1))
    edges = np.quantile(array, quantiles)
    return np.unique(np.asarray(edges, dtype=np.float64))


def _build_continuous_bin_spec(
    values: Sequence[Any],
    *,
    bin_count: int,
    label_prefix: str = "Q",
) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"bins": [], "edges": [], "effective_bin_count": 0, "requested_bin_count": int(bin_count)}

    edges = _quantile_edges(array, max(1, int(bin_count)))
    if edges.size < 2:
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if abs(maximum - minimum) <= 1e-12:
            maximum = minimum + 1e-9
        edges = np.asarray([minimum, maximum], dtype=np.float64)

    bins: list[dict[str, Any]] = []
    for index in range(max(0, edges.size - 1)):
        left = float(edges[index])
        right = float(edges[index + 1])
        if index == edges.size - 2:
            mask = (array >= left) & (array <= right)
        else:
            mask = (array >= left) & (array < right)
        bins.append(
            {
                "bin_index": int(index),
                "bin_label": f"{label_prefix}{index + 1}",
                "value_left": float(left),
                "value_right": float(right),
                "count_rows": int(np.sum(mask)),
            }
        )
    return {
        "bins": bins,
        "edges": [float(value) for value in edges.tolist()],
        "effective_bin_count": int(len(bins)),
        "requested_bin_count": int(bin_count),
    }


def _lookup_continuous_bin_info(spec: Mapping[str, Any], value: float) -> dict[str, Any] | None:
    bins = list(spec.get("bins", []))
    if not bins:
        return None
    safe_value = float(value)
    if not np.isfinite(safe_value):
        return dict(bins[-1])
    for index, bin_info in enumerate(bins):
        left = float(bin_info["value_left"])
        right = float(bin_info["value_right"])
        is_last = index == len(bins) - 1
        if (left <= safe_value <= right) if is_last else (left <= safe_value < right):
            return dict(bin_info)
    return dict(bins[-1])


def _assign_continuous_bin_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    spec: Mapping[str, Any],
    prefix: str,
) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        value = float(row.get(value_key, float("nan")))
        bin_info = _lookup_continuous_bin_info(spec, value)
        enriched[f"{prefix}_value"] = float(value)
        if bin_info is None:
            enriched[f"{prefix}_bin"] = "NA"
            enriched[f"{prefix}_bin_index"] = -1
            enriched[f"{prefix}_bin_left"] = float("nan")
            enriched[f"{prefix}_bin_right"] = float("nan")
        else:
            enriched[f"{prefix}_bin"] = str(bin_info["bin_label"])
            enriched[f"{prefix}_bin_index"] = int(bin_info["bin_index"])
            enriched[f"{prefix}_bin_left"] = float(bin_info["value_left"])
            enriched[f"{prefix}_bin_right"] = float(bin_info["value_right"])
        enriched_rows.append(enriched)
    return enriched_rows


def build_adaptive_profile_rows(
    policy: Any,
    *,
    reference_rows: Sequence[Mapping[str, Any]] | None = None,
    reference_bin_count: int = 16,
    curve_sample_count: int = 256,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    processed_reference_rows: list[dict[str, Any]] = []

    if reference_rows:
        for row in reference_rows:
            ego_mean_resource = _extract_ego_mean_resource_from_row(row)
            weight_unit_investment = row.get("weight_unit_investment")
            weight_equal = row.get("weight_equal")
            weight_self = row.get("weight_self", 0.0)
            if not (
                np.isfinite(ego_mean_resource)
                and weight_unit_investment not in {None, ""}
                and weight_equal not in {None, ""}
                and weight_self not in {None, ""}
            ):
                continue
            processed_reference_rows.append(
                {
                    "ego_mean_resource": float(ego_mean_resource),
                    "ego_resource_x": (
                        float(policy.profile_x_value_for_ego_mean_resource(float(ego_mean_resource)))
                        if hasattr(policy, "profile_x_value_for_ego_mean_resource")
                        else float(ego_mean_resource)
                    ),
                    "weight_unit_investment": float(weight_unit_investment),
                    "weight_equal": float(weight_equal),
                    "weight_self": float(weight_self),
                    "fit_l1": (
                        float(row["fit_l1"])
                        if row.get("fit_l1") not in {None, ""}
                        else float("nan")
                    ),
                }
            )

    if processed_reference_rows:
        resource_values = np.asarray([float(row["ego_resource_x"]) for row in processed_reference_rows], dtype=np.float64)
        edges = _quantile_edges(resource_values, max(2, int(reference_bin_count)))
        if edges.size < 2:
            edges = np.asarray([float(np.min(resource_values)), float(np.max(resource_values))], dtype=np.float64)
        for bin_index in range(max(0, edges.size - 1)):
            left = float(edges[bin_index])
            right = float(edges[bin_index + 1])
            if bin_index == edges.size - 2:
                group = [row for row in processed_reference_rows if left <= float(row["ego_resource_x"]) <= right]
            else:
                group = [row for row in processed_reference_rows if left <= float(row["ego_resource_x"]) < right]
            if not group:
                continue
            bin_center_raw = float(np.mean([float(row["ego_mean_resource"]) for row in group]))
            bin_center = float(np.mean([float(row["ego_resource_x"]) for row in group]))
            rows.append(
                {
                    "row_kind": "reference_bin",
                    "order_index": int(bin_index),
                    "ego_mean_resource": float(bin_center_raw),
                    "ego_resource_x": float(bin_center),
                    "ego_resource_x_mode": str(getattr(policy, "ego_resource_x_mode", "absolute")),
                    "ego_resource_normalizer": getattr(policy, "ego_resource_normalizer", None),
                    "ego_mean_resource_effective": (
                        float(policy.effective_ego_mean_resource(float(bin_center_raw)))
                        if hasattr(policy, "effective_ego_mean_resource")
                        else float(bin_center_raw)
                    ),
                    "log_ego_mean_resource": float(np.log1p(bin_center)),
                    "weight_unit_investment": float(np.mean([float(row["weight_unit_investment"]) for row in group])),
                    "weight_equal": float(np.mean([float(row["weight_equal"]) for row in group])),
                    "weight_self": float(np.mean([float(row["weight_self"]) for row in group])),
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "sample_count": int(len(group)),
                    "reference_fit_l1_mean": (
                        float(np.mean([float(row["fit_l1"]) for row in group if np.isfinite(float(row["fit_l1"]))]))
                        if any(np.isfinite(float(row["fit_l1"])) for row in group)
                        else float("nan")
                    ),
                    "profile_source": str(getattr(policy, "profile_source", "unknown")),
                }
            )

    if processed_reference_rows:
        curve_min = float(np.min([float(row["ego_mean_resource"]) for row in processed_reference_rows]))
        curve_max = float(np.max([float(row["ego_mean_resource"]) for row in processed_reference_rows]))
    elif isinstance(policy, AdaptiveUnitInvestmentEqualAllocationPolicy):
        curve_min = float(policy.ego_mean_resource_knots[0])
        curve_max = float(policy.ego_mean_resource_knots[-1])
    else:
        curve_min = 0.0
        curve_max = 1.0

    if abs(curve_max - curve_min) <= 1e-12:
        curve_values = np.asarray([curve_min], dtype=np.float64)
    else:
        curve_values = np.linspace(curve_min, curve_max, num=max(2, int(curve_sample_count)))

    for order_index, ego_mean_resource in enumerate(curve_values.tolist()):
        effective_ego_mean_resource = (
            float(policy.effective_ego_mean_resource(float(ego_mean_resource)))
            if hasattr(policy, "effective_ego_mean_resource")
            else float(ego_mean_resource)
        )
        weight_unit_investment, weight_equal, weight_self = policy.weights_for_ego_mean_resource(float(ego_mean_resource))
        rows.append(
            {
                "row_kind": "fitted_curve",
                "order_index": int(order_index),
                "ego_mean_resource": float(ego_mean_resource),
                "ego_resource_x": (
                    float(policy.profile_x_value_for_ego_mean_resource(float(ego_mean_resource)))
                    if hasattr(policy, "profile_x_value_for_ego_mean_resource")
                    else float(ego_mean_resource)
                ),
                "ego_resource_x_mode": str(getattr(policy, "ego_resource_x_mode", "absolute")),
                "ego_resource_normalizer": getattr(policy, "ego_resource_normalizer", None),
                "ego_mean_resource_effective": float(effective_ego_mean_resource),
                "log_ego_mean_resource": float(np.log1p(max(0.0, float(ego_mean_resource)))),
                "weight_unit_investment": float(weight_unit_investment),
                "weight_equal": float(weight_equal),
                "weight_self": float(weight_self),
                "bin_left": float("nan"),
                "bin_right": float("nan"),
                "sample_count": 0,
                "reference_fit_l1_mean": float("nan"),
                "profile_source": str(getattr(policy, "profile_source", "unknown")),
            }
        )
    return rows


def build_continuous_2d_surface_rows(
    policy: ContinuousTwoStateMixtureAllocationPolicy,
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    second_variable_name: str,
    grid_size: int,
) -> list[dict[str, Any]]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(_extract_continuous_second_value_from_row(row, second_variable_name))
    ]
    if not valid_rows:
        return []

    ego_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    second_values = np.asarray(
        [_extract_continuous_second_value_from_row(row, second_variable_name) for row in valid_rows],
        dtype=np.float64,
    )
    ego_grid = np.linspace(
        float(np.min(ego_values)),
        float(np.max(ego_values)),
        num=max(8, int(grid_size)),
    )
    second_grid = np.linspace(
        float(np.min(second_values)),
        float(np.max(second_values)),
        num=max(8, int(grid_size)),
    )

    rows: list[dict[str, Any]] = []
    order_index = 0
    for x2_value in second_grid.tolist():
        for ego_mean_resource in ego_grid.tolist():
            x1_value = float(np.log1p(max(0.0, float(ego_mean_resource))))
            weight_unit_investment, weight_equal, weight_self = policy.weights_for_inputs(x1_value, float(x2_value))
            rows.append(
                {
                    "order_index": int(order_index),
                    "ego_mean_resource": float(ego_mean_resource),
                    "x1_log_ego_mean_resource": float(x1_value),
                    "second_variable_name": str(second_variable_name),
                    "x2_second_variable": float(x2_value),
                    "weight_unit_investment": float(weight_unit_investment),
                    "weight_equal": float(weight_equal),
                    "weight_self": float(weight_self),
                    "policy_name": str(policy.policy_name),
                    "profile_family": str(policy.profile_family),
                }
            )
            order_index += 1
    return rows


def _pivot_continuous_surface(
    surface_rows: Sequence[Mapping[str, Any]],
    weight_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_x = np.asarray(
        sorted({float(row["ego_mean_resource"]) for row in surface_rows}),
        dtype=np.float64,
    )
    unique_y = np.asarray(
        sorted({float(row["x2_second_variable"]) for row in surface_rows}),
        dtype=np.float64,
    )
    z_values = np.full((unique_y.size, unique_x.size), np.nan, dtype=np.float64)
    x_lookup = {float(value): int(index) for index, value in enumerate(unique_x.tolist())}
    y_lookup = {float(value): int(index) for index, value in enumerate(unique_y.tolist())}
    for row in surface_rows:
        z_values[y_lookup[float(row["x2_second_variable"])]][x_lookup[float(row["ego_mean_resource"])]] = float(
            row[weight_key]
        )
    return unique_x, unique_y, z_values


def plot_continuous_2d_weight_surfaces(
    surface_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    title_prefix: str,
    second_variable_label: str,
) -> None:
    if plt is None or not surface_rows:
        return

    component_specs = [
        ("weight_unit_investment", "#9467bd", "Unit-Inv Weight"),
        ("weight_equal", "#1f77ff", "Equal Weight"),
        ("weight_self", "#ef4444", "Self Weight"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True, sharex=True, sharey=True)
    for axis, (weight_key, _, title) in zip(np.asarray(axes).reshape(-1), component_specs):
        x_values, y_values, z_values = _pivot_continuous_surface(surface_rows, weight_key)
        image = axis.contourf(x_values, y_values, z_values, levels=18, cmap="viridis")
        axis.set_title(f"{title_prefix}: {title}")
        axis.set_xlabel("ego_mean_resource")
        axis.grid(alpha=0.20)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(second_variable_label)
    _save_figure(figure, output_path)


def build_continuous_reference_fit_summary_row(
    policy: ContinuousTwoStateMixtureAllocationPolicy | SmoothLogEgoResourceMixtureAllocationPolicy,
    fit_summary: Mapping[str, Any],
) -> dict[str, Any]:
    profile_summary = dict(policy.profile_summary())
    row = {
        "policy": str(profile_summary.get("policy_name", fit_summary.get("policy_name", "unknown"))),
        "mode": str(profile_summary.get("mode", "")),
        "profile_family": str(profile_summary.get("profile_family", fit_summary.get("profile_family", ""))),
        "x1_name": str(profile_summary.get("x1_name", "log1p(ego_mean_resource)")),
        "x2_name": str(profile_summary.get("x2_name", fit_summary.get("second_variable_name", ""))),
        "basis_mode": str(profile_summary.get("basis_mode", fit_summary.get("basis_mode", ""))),
        "basis_dim": int(fit_summary.get("basis_dim", len(profile_summary.get("basis_terms", [])))),
        "fit_row_count": int(fit_summary.get("fit_row_count", 0)),
        "weight_simplex_l1_mean": float(fit_summary.get("weight_simplex_l1_mean", float("nan"))),
        "reference_row_fit_l1_mean": float(fit_summary.get("reference_row_fit_l1_mean", float("nan"))),
        "optimization_best_simplex_l1": float(fit_summary.get("optimization_best_simplex_l1", float("nan"))),
        "optimization_best_alpha_l1": float(fit_summary.get("optimization_best_alpha_l1", float("nan"))),
        "equal_dropped_mass_mean": float(fit_summary.get("equal_dropped_mass_mean", float("nan"))),
        "interaction_coeff_unit_investment": float("nan"),
        "interaction_coeff_self": float("nan"),
        "interaction_coeff_alpha": float("nan"),
    }
    if isinstance(policy, ContinuousTwoStateMixtureAllocationPolicy):
        term_lookup = {term: index for index, term in enumerate(policy.basis_terms)}
        interaction_index = term_lookup.get("x1*x2", -1)
        if policy.profile_family == "three_way":
            row["interaction_coeff_unit_investment"] = (
                float(policy.coeffs_unit_investment[interaction_index]) if interaction_index >= 0 else float("nan")
            )
            row["interaction_coeff_self"] = (
                float(policy.coeffs_self[interaction_index]) if interaction_index >= 0 else float("nan")
            )
        else:
            row["interaction_coeff_alpha"] = (
                float(policy.coeffs_alpha[interaction_index]) if interaction_index >= 0 else float("nan")
            )
    return row


def summarize_continuous_2d_rule_interpretation(
    policy: ContinuousTwoStateMixtureAllocationPolicy,
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    second_variable_name: str,
) -> dict[str, Any]:
    valid_rows = [
        row
        for row in reference_rows
        if np.isfinite(float(row.get("ego_mean_resource", float("nan"))))
        and np.isfinite(_extract_continuous_second_value_from_row(row, second_variable_name))
    ]
    if not valid_rows:
        return {}

    ego_values = np.asarray([float(row["ego_mean_resource"]) for row in valid_rows], dtype=np.float64)
    x2_values = np.asarray(
        [_extract_continuous_second_value_from_row(row, second_variable_name) for row in valid_rows],
        dtype=np.float64,
    )
    ego_low = float(np.quantile(ego_values, 0.2))
    ego_high = float(np.quantile(ego_values, 0.8))
    x2_low = float(np.quantile(x2_values, 0.2))
    x2_high = float(np.quantile(x2_values, 0.8))
    corners = {
        "low_x1_low_x2": (ego_low, x2_low),
        "low_x1_high_x2": (ego_low, x2_high),
        "high_x1_low_x2": (ego_high, x2_low),
        "high_x1_high_x2": (ego_high, x2_high),
    }

    corner_predictions: dict[str, Any] = {}
    for key, (ego_mean_resource, x2_value) in corners.items():
        weights = policy.weights_for_inputs(float(np.log1p(max(0.0, ego_mean_resource))), float(x2_value))
        dominant_index = int(np.argmax(np.asarray(weights, dtype=np.float64)))
        dominant_name = ("unit_investment", "equal", "self")[dominant_index]
        corner_predictions[key] = {
            "ego_mean_resource": float(ego_mean_resource),
            "x2_value": float(x2_value),
            "weight_unit_investment": float(weights[0]),
            "weight_equal": float(weights[1]),
            "weight_self": float(weights[2]),
            "dominant_mechanism": str(dominant_name),
        }

    interaction_payload: dict[str, Any]
    term_lookup = {term: index for index, term in enumerate(policy.basis_terms)}
    interaction_index = term_lookup.get("x1*x2", -1)
    if policy.profile_family == "three_way":
        interaction_payload = {
            "interaction_coeff_unit_investment": (
                float(policy.coeffs_unit_investment[interaction_index]) if interaction_index >= 0 else float("nan")
            ),
            "interaction_coeff_self": (
                float(policy.coeffs_self[interaction_index]) if interaction_index >= 0 else float("nan")
            ),
        }
    else:
        interaction_payload = {
            "interaction_coeff_alpha": (
                float(policy.coeffs_alpha[interaction_index]) if interaction_index >= 0 else float("nan")
            ),
        }

    return {
        "x1_name": "log1p(ego_mean_resource)",
        "x2_name": str(second_variable_name),
        "reference_quantiles": {
            "ego_mean_resource_q20": float(ego_low),
            "ego_mean_resource_q80": float(ego_high),
            "x2_q20": float(x2_low),
            "x2_q80": float(x2_high),
        },
        "corner_predictions": corner_predictions,
        **interaction_payload,
    }


def _gini_nonnegative(values: np.ndarray, eps: float = 1e-12) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return float("nan")
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= eps:
        return 0.0
    pairwise_abs = np.abs(array[:, None] - array[None, :]).sum()
    return float(pairwise_abs / (2.0 * float(array.size) * total))


def _windows_style_path(path: Path) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive_letter = text[5].upper()
        text = f"{drive_letter}:{text[6:]}"
    return text.replace("/", "\\")


def _short_alias(text: str, *, prefix: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(text)).strip("_")
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8]
    head = sanitized[:12] or prefix
    return f"{head}_{digest}"


def _shortened_filename(path: Path, *, max_total_length: int = 240) -> str:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:8]
    parent_length = len(_windows_style_path(path.parent))
    reserve = len(suffix) + len(digest) + 2
    max_stem_length = max(1, max_total_length - parent_length - reserve)
    sanitized_stem = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in stem
    ).strip("_")
    short_stem = (sanitized_stem[:max_stem_length] or "out").rstrip("_-") or "out"
    candidate = f"{short_stem}_{digest}{suffix}"
    if len(_windows_style_path(path.with_name(candidate))) <= max_total_length:
        return candidate
    return f"f_{digest}{suffix}"


def _prepare_output_path(path: Path, *, max_total_length: int = 240) -> Path:
    candidate = Path(path)
    if len(_windows_style_path(candidate)) <= max_total_length:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    candidate = candidate.with_name(_shortened_filename(candidate, max_total_length=max_total_length))
    if len(_windows_style_path(candidate)) <= max_total_length:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    short_parent = candidate.parent.parent / _short_alias(candidate.parent.name, prefix="out")
    candidate = short_parent / _shortened_filename(short_parent / candidate.name, max_total_length=max_total_length)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv_raw(_prepare_output_path(Path(path)), rows)


def _write_json(path: Path, payload: Any) -> None:
    safe_path = _prepare_output_path(Path(path))
    with safe_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _save_figure(figure: Any, output_path: Path) -> None:
    safe_path = _prepare_output_path(Path(output_path))
    figure.savefig(safe_path, dpi=220, bbox_inches="tight")


def _policy_artifact_stem(policy_name: str) -> str:
    mapping = {
        "adaptive_continuous2d_softmax_mix": "c2d_softmax",
        "adaptive_continuous2d_unitself_mix": "c2d_unitself",
    }
    return str(mapping.get(str(policy_name), str(policy_name)))


def _read_adaptive_unitinv_equal_reference_rows(profile_source: str) -> list[dict[str, Any]]:
    profile_source_text = str(profile_source)
    profile_path_text = profile_source_text.split("::", maxsplit=1)[0]
    profile_path = Path(profile_path_text)
    if profile_path.suffix.lower() != ".csv" or not profile_path.exists():
        return []

    import csv

    rows: list[dict[str, Any]] = []
    with profile_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for order_index, row in enumerate(reader):
            if not row:
                continue
            resource_raw = row.get("bin_center")
            weight_raw = row.get("weight_unit_investment_mean")
            if resource_raw is None or weight_raw is None:
                continue
            resource_value = float(resource_raw)
            weight_value = float(weight_raw)
            if not np.isfinite(resource_value) or not np.isfinite(weight_value):
                continue
            rows.append(
                {
                    "row_kind": "reference_bin",
                    "order_index": int(order_index),
                    "ego_mean_resource": float(resource_value),
                    "weight_unit_investment": float(np.clip(weight_value, 0.0, 1.0)),
                    "weight_equal": float(np.clip(1.0 - weight_value, 0.0, 1.0)),
                    "bin_left": float(row["bin_left"]) if row.get("bin_left") not in {None, ""} else float("nan"),
                    "bin_right": float(row["bin_right"]) if row.get("bin_right") not in {None, ""} else float("nan"),
                    "sample_count": int(float(row["sample_count"])) if row.get("sample_count") not in {None, ""} else 0,
                    "sample_count_identifiable": (
                        int(float(row["sample_count_identifiable"]))
                        if row.get("sample_count_identifiable") not in {None, ""}
                        else 0
                    ),
                    "identifiable_fraction": (
                        float(row["identifiable_fraction"])
                        if row.get("identifiable_fraction") not in {None, ""}
                        else float("nan")
                    ),
                    "profile_source": profile_source_text,
                }
            )
    return rows


def build_adaptive_unitinv_equal_profile_rows(
    policy: AdaptiveUnitInvestmentEqualAllocationPolicy,
    *,
    curve_sample_count: int = 256,
) -> list[dict[str, Any]]:
    legacy_rows = _read_adaptive_unitinv_equal_reference_rows(policy.profile_source)
    normalized_legacy_rows: list[dict[str, Any]] = []
    for row in legacy_rows:
        normalized = dict(row)
        normalized["weight_self"] = 0.0
        normalized["log_ego_mean_resource"] = float(np.log1p(max(0.0, float(row["ego_mean_resource"]))))
        normalized["reference_fit_l1_mean"] = float("nan")
        normalized_legacy_rows.append(normalized)

    curve_rows = build_adaptive_profile_rows(
        policy,
        reference_rows=None,
        curve_sample_count=curve_sample_count,
    )
    for row in curve_rows:
        if row.get("row_kind") == "fitted_curve":
            row["row_kind"] = "interpolated_curve"
    knot_x = np.asarray(policy.ego_mean_resource_knots, dtype=np.float64)
    knot_w = np.asarray(policy.weight_knots, dtype=np.float64)
    knot_rows: list[dict[str, Any]] = []
    for order_index, (ego_mean_resource, weight_value) in enumerate(zip(knot_x.tolist(), knot_w.tolist())):
        knot_rows.append(
            {
                "row_kind": "knot",
                "order_index": int(order_index),
                "ego_mean_resource": float(ego_mean_resource),
                "log_ego_mean_resource": float(np.log1p(max(0.0, float(ego_mean_resource)))),
                "weight_unit_investment": float(weight_value),
                "weight_equal": float(1.0 - weight_value),
                "weight_self": 0.0,
                "bin_left": float("nan"),
                "bin_right": float("nan"),
                "sample_count": 0,
                "reference_fit_l1_mean": float("nan"),
                "profile_source": str(policy.profile_source),
            }
        )
    return normalized_legacy_rows + knot_rows + curve_rows


def plot_adaptive_unitinv_equal_profile(
    profile_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    if plt is None or not profile_rows:
        return

    curve_rows = [
        row for row in profile_rows if str(row.get("row_kind", "")) in {"interpolated_curve", "fitted_curve"}
    ]
    knot_rows = [row for row in profile_rows if str(row.get("row_kind", "")) == "knot"]
    reference_rows = [row for row in profile_rows if str(row.get("row_kind", "")) == "reference_bin"]
    if not curve_rows and not knot_rows and not reference_rows:
        return

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    x_label = _ego_resource_x_label(
        str(profile_rows[0].get("ego_resource_x_mode", "absolute")) if profile_rows else "absolute",
        profile_rows[0].get("ego_resource_normalizer") if profile_rows else None,
    )
    component_specs = [
        ("weight_unit_investment", "#9467bd", "Unit-Investment"),
        ("weight_equal", "#1f77ff", "Equal"),
        ("weight_self", "#ff7f0e", "Self"),
    ]

    if reference_rows:
        reference_rows = sorted(reference_rows, key=lambda row: float(row.get("ego_resource_x", row["ego_mean_resource"])))
        ref_x = np.asarray([float(row.get("ego_resource_x", row["ego_mean_resource"])) for row in reference_rows], dtype=np.float64)
        for key, color, label in component_specs:
            if not any(row.get(key) not in {None, ""} for row in reference_rows):
                continue
            ref_y = np.asarray([float(row.get(key, float("nan"))) for row in reference_rows], dtype=np.float64)
            if np.all(~np.isfinite(ref_y)):
                continue
            axis.scatter(
                ref_x,
                ref_y,
                s=22,
                color=color,
                alpha=0.22,
                label=f"Reference {label}",
                zorder=1,
            )

    if curve_rows:
        curve_rows = sorted(curve_rows, key=lambda row: float(row.get("ego_resource_x", row["ego_mean_resource"])))
        curve_x = np.asarray([float(row.get("ego_resource_x", row["ego_mean_resource"])) for row in curve_rows], dtype=np.float64)
        for key, color, label in component_specs:
            curve_y = np.asarray([float(row.get(key, float("nan"))) for row in curve_rows], dtype=np.float64)
            if np.all(~np.isfinite(curve_y)):
                continue
            if key == "weight_self" and np.nanmax(np.abs(curve_y)) <= 1e-8:
                continue
            axis.plot(
                curve_x,
                curve_y,
                color=color,
                linewidth=2.2,
                label=f"Fitted {label}",
                zorder=2,
            )

    if knot_rows:
        knot_rows = sorted(knot_rows, key=lambda row: float(row.get("ego_resource_x", row["ego_mean_resource"])))
        knot_x = np.asarray([float(row.get("ego_resource_x", row["ego_mean_resource"])) for row in knot_rows], dtype=np.float64)
        knot_y = np.asarray([float(row["weight_unit_investment"]) for row in knot_rows], dtype=np.float64)
        axis.scatter(
            knot_x,
            knot_y,
            s=56,
            color="#8c564b",
            edgecolors="#ffffff",
            linewidths=0.8,
            label="Interpolation Knots",
            zorder=3,
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel("Mixture Weight")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title(f"Adaptive Mixture Weights vs {x_label}")
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_degree_conditioned_profile(
    profile_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    if plt is None or not profile_rows:
        return

    degree_bins = sorted(
        {
            (int(row.get("sender_degree_bin_index", 0)), str(row.get("sender_degree_bin", "NA")))
            for row in profile_rows
        },
        key=lambda item: item[0],
    )
    if not degree_bins:
        return

    panel_count = len(degree_bins)
    ncols = min(2, panel_count)
    nrows = int(np.ceil(float(panel_count) / float(max(ncols, 1))))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.8 * float(ncols), 4.6 * float(nrows)),
        constrained_layout=True,
        sharey=True,
    )
    axes_array = list(np.asarray(axes).reshape(-1))
    x_label = _ego_resource_x_label(
        str(profile_rows[0].get("ego_resource_x_mode", "absolute")) if profile_rows else "absolute",
        profile_rows[0].get("ego_resource_normalizer") if profile_rows else None,
    )
    component_specs = [
        ("weight_unit_investment", "#9467bd", "Unit-Investment"),
        ("weight_equal", "#1f77ff", "Equal"),
        ("weight_self", "#ff7f0e", "Self"),
    ]

    for axis, (bin_index, bin_label) in zip(axes_array, degree_bins):
        panel_rows = [
            row for row in profile_rows if int(row.get("sender_degree_bin_index", -1)) == int(bin_index)
        ]
        reference_rows = sorted(
            [row for row in panel_rows if str(row.get("row_kind", "")) == "reference_bin"],
            key=lambda row: float(row.get("ego_resource_x", row["ego_mean_resource"])),
        )
        curve_rows = sorted(
            [row for row in panel_rows if str(row.get("row_kind", "")) in {"interpolated_curve", "fitted_curve"}],
            key=lambda row: float(row.get("ego_resource_x", row["ego_mean_resource"])),
        )

        if reference_rows:
            ref_x = np.asarray([float(row.get("ego_resource_x", row["ego_mean_resource"])) for row in reference_rows], dtype=np.float64)
            for key, color, label in component_specs:
                ref_y = np.asarray([float(row.get(key, float("nan"))) for row in reference_rows], dtype=np.float64)
                if np.all(~np.isfinite(ref_y)):
                    continue
                axis.scatter(ref_x, ref_y, s=18, color=color, alpha=0.20, zorder=1)

        if curve_rows:
            curve_x = np.asarray([float(row.get("ego_resource_x", row["ego_mean_resource"])) for row in curve_rows], dtype=np.float64)
            for key, color, label in component_specs:
                curve_y = np.asarray([float(row.get(key, float("nan"))) for row in curve_rows], dtype=np.float64)
                if np.all(~np.isfinite(curve_y)):
                    continue
                if key == "weight_self" and np.nanmax(np.abs(curve_y)) <= 1e-8:
                    continue
                axis.plot(curve_x, curve_y, color=color, linewidth=2.2, label=label, zorder=2)

        axis.set_title(f"Sender Degree {bin_label}")
        axis.set_xlabel(x_label)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)

    for axis in axes_array[0::ncols]:
        axis.set_ylabel("Mixture Weight")
    for axis in axes_array[len(degree_bins):]:
        axis.set_axis_off()
    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)), frameon=False)
    _save_figure(figure, output_path)


def plot_resource_conditioned_profile(
    profile_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    if plt is None or not profile_rows:
        return

    resource_bins = sorted(
        {
            (int(row.get("sender_resource_norm_bin_index", 0)), str(row.get("sender_resource_norm_bin", "NA")))
            for row in profile_rows
        },
        key=lambda item: item[0],
    )
    if not resource_bins:
        return

    panel_count = len(resource_bins)
    ncols = min(2, panel_count)
    nrows = int(np.ceil(float(panel_count) / float(max(ncols, 1))))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.8 * float(ncols), 4.6 * float(nrows)),
        constrained_layout=True,
        sharey=True,
    )
    axes_array = list(np.asarray(axes).reshape(-1))
    x_label = _ego_resource_x_label(
        str(profile_rows[0].get("ego_resource_x_mode", "absolute")) if profile_rows else "absolute",
        profile_rows[0].get("ego_resource_normalizer") if profile_rows else None,
    )
    component_specs = [
        ("weight_unit_investment", "#9467bd", "Unit-Investment"),
        ("weight_equal", "#1f77ff", "Equal"),
        ("weight_self", "#ff7f0e", "Self"),
    ]

    for axis, (bin_index, bin_label) in zip(axes_array, resource_bins):
        panel_rows = [
            row for row in profile_rows if int(row.get("sender_resource_norm_bin_index", -1)) == int(bin_index)
        ]
        reference_rows = sorted(
            [row for row in panel_rows if str(row.get("row_kind", "")) == "reference_bin"],
            key=lambda row: float(row.get("ego_resource_x", row["ego_mean_resource"])),
        )
        curve_rows = sorted(
            [row for row in panel_rows if str(row.get("row_kind", "")) in {"interpolated_curve", "fitted_curve"}],
            key=lambda row: float(row.get("ego_resource_x", row["ego_mean_resource"])),
        )

        if reference_rows:
            ref_x = np.asarray([float(row.get("ego_resource_x", row["ego_mean_resource"])) for row in reference_rows], dtype=np.float64)
            for key, color, label in component_specs:
                ref_y = np.asarray([float(row.get(key, float("nan"))) for row in reference_rows], dtype=np.float64)
                if np.all(~np.isfinite(ref_y)):
                    continue
                axis.scatter(ref_x, ref_y, s=18, color=color, alpha=0.20, zorder=1)

        if curve_rows:
            curve_x = np.asarray([float(row.get("ego_resource_x", row["ego_mean_resource"])) for row in curve_rows], dtype=np.float64)
            for key, color, label in component_specs:
                curve_y = np.asarray([float(row.get(key, float("nan"))) for row in curve_rows], dtype=np.float64)
                if np.all(~np.isfinite(curve_y)):
                    continue
                if key == "weight_self" and np.nanmax(np.abs(curve_y)) <= 1e-8:
                    continue
                axis.plot(curve_x, curve_y, color=color, linewidth=2.2, label=label, zorder=2)

        left = panel_rows[0].get("sender_resource_norm_bin_left", float("nan")) if panel_rows else float("nan")
        right = panel_rows[0].get("sender_resource_norm_bin_right", float("nan")) if panel_rows else float("nan")
        axis.set_title(f"Sender Resource Norm {bin_label} [{float(left):.3f}, {float(right):.3f}]")
        axis.set_xlabel(x_label)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)

    for axis in axes_array[0::ncols]:
        axis.set_ylabel("Mixture Weight")
    for axis in axes_array[len(resource_bins):]:
        axis.set_axis_off()
    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)), frameon=False)
    _save_figure(figure, output_path)


def apply_analysis_env_overrides(
    spec: Mapping[str, Any],
    *,
    env_r_override: float | None,
    env_p_c_override: float | None = None,
    env_beta_override: float | None = None,
    env_strategy_update_rule_override: str | None = None,
    env_p_max_override: float | None = None,
    env_initial_resource_override: float | None = None,
    env_initial_cooperation_prob_override: float | None = None,
) -> dict[str, Any]:
    effective_spec = deepcopy(dict(spec))
    override_items = {
        "r": env_r_override,
        "p_c": env_p_c_override,
        "beta": env_beta_override,
        "strategy_update_rule": env_strategy_update_rule_override,
        "p_max": env_p_max_override,
        "initial_resource": env_initial_resource_override,
        "initial_cooperation_prob": env_initial_cooperation_prob_override,
    }
    if all(value is None for value in override_items.values()):
        return effective_spec
    if "dynamics" not in effective_spec or not isinstance(effective_spec["dynamics"], Mapping):
        raise KeyError("Experiment spec does not contain a dynamics mapping; cannot override environment dynamics.")
    effective_spec["dynamics"] = dict(effective_spec["dynamics"])
    if env_r_override is not None:
        effective_spec["dynamics"]["r"] = float(env_r_override)
    if env_p_c_override is not None:
        effective_spec["dynamics"]["p_c"] = float(env_p_c_override)
    if env_beta_override is not None:
        effective_spec["dynamics"]["beta"] = float(env_beta_override)
    if env_strategy_update_rule_override is not None:
        effective_spec["dynamics"]["strategy_update_rule"] = str(env_strategy_update_rule_override)
    if env_p_max_override is not None:
        effective_spec["dynamics"]["p_max"] = float(env_p_max_override)
    if env_initial_resource_override is not None:
        effective_spec["dynamics"]["initial_resource"] = float(env_initial_resource_override)
    if env_initial_cooperation_prob_override is not None:
        effective_spec["dynamics"]["initial_cooperation_prob"] = float(env_initial_cooperation_prob_override)
    return effective_spec


def run_rule_policy_rollouts(
    policy: Any,
    *,
    env_config: Any,
    graph: Mapping[int, Sequence[int]],
    episodes: int,
    seed: int,
    max_steps: int | None,
) -> tuple[list[Snapshot], list[EpisodeSummary]]:
    env = SPGGEnv(env_config, graph)
    snapshots: list[Snapshot] = []
    episode_summaries: list[EpisodeSummary] = []

    for episode_index in range(max(0, int(episodes))):
        observation = env.reset(seed=int(seed) + episode_index)
        done = False
        step_index = 0
        total_reward = 0.0
        last_info: dict[str, Any] | None = None

        while not done:
            if max_steps is not None and step_index >= max_steps:
                break

            allocation_matrix = np.asarray(policy.allocate(observation), dtype=np.float64)
            pool_values = np.asarray(observation["pool_grown"], dtype=np.float64)
            transferred_resources = allocation_matrix * pool_values[:, None]
            incoming_resources = np.sum(transferred_resources, axis=0)

            next_observation, reward, env_done, info = env.step(allocation_matrix)
            snapshots.append(
                Snapshot(
                    episode=episode_index,
                    step=step_index,
                    observation=_observation_copy(observation),
                    policy=PolicyArrays(
                        allocation_matrix=np.asarray(allocation_matrix, dtype=np.float64),
                        transferred_resources=np.asarray(transferred_resources, dtype=np.float64),
                        incoming_resources=np.asarray(incoming_resources, dtype=np.float64),
                        logits=np.zeros_like(allocation_matrix, dtype=np.float64),
                        value=0.0,
                    ),
                    reward=float(reward),
                    actual_cooperation_rate=float(info.get("actual_cooperation_rate", np.mean(next_observation["x_actual"]))),
                    gini=float(info.get("gini", next_observation["gini"])),
                )
            )
            total_reward += float(reward)
            last_info = info
            observation = next_observation
            step_index += 1
            done = bool(env_done)

        final_cooperation_rate = float(np.mean(observation["x_actual"]))
        final_gini = float(np.asarray(observation["gini"]).item())
        if last_info is not None:
            final_cooperation_rate = float(last_info.get("actual_cooperation_rate", final_cooperation_rate))
            final_gini = float(last_info.get("gini", final_gini))
        episode_summaries.append(
            EpisodeSummary(
                episode=episode_index,
                steps=step_index,
                total_reward=total_reward,
                mean_reward=total_reward / float(max(step_index, 1)),
                final_cooperation_rate=final_cooperation_rate,
                final_gini=final_gini,
                final_mean_resource=float(np.mean(observation["resources"])),
            )
        )

    return snapshots, episode_summaries


def summarize_policy_rollout_step_metrics(
    snapshots_by_policy: Mapping[str, Sequence[Snapshot]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for policy_name, snapshots in snapshots_by_policy.items():
        if not snapshots:
            continue
        per_step: dict[int, dict[str, list[float]]] = {}
        for snapshot in snapshots:
            step = int(snapshot.step)
            bucket = per_step.setdefault(
                step,
                {
                    "reward": [],
                    "cooperation": [],
                    "resource_gini": [],
                    "mean_resource": [],
                    "mean_pool_grown": [],
                    "mean_self_allocation": [],
                    "offer_gini": [],
                },
            )
            observation = snapshot.observation
            allocation_matrix = np.asarray(snapshot.policy.allocation_matrix, dtype=np.float64)
            local_mask = np.asarray(observation["local_mask"], dtype=bool)
            pool_grown = np.asarray(observation["pool_grown"], dtype=np.float64)

            row_ginis: list[float] = []
            for sender in range(local_mask.shape[0]):
                valid_receivers = np.flatnonzero(local_mask[sender])
                allocation_row = np.asarray(allocation_matrix[sender, valid_receivers], dtype=np.float64)
                row_ginis.append(_gini_nonnegative(float(pool_grown[sender]) * allocation_row))

            bucket["reward"].append(float(snapshot.reward))
            bucket["cooperation"].append(float(snapshot.actual_cooperation_rate))
            bucket["resource_gini"].append(float(snapshot.gini))
            bucket["mean_resource"].append(float(np.mean(observation["resources"])))
            bucket["mean_pool_grown"].append(float(np.mean(pool_grown)))
            bucket["mean_self_allocation"].append(float(np.mean(np.diag(allocation_matrix))))
            bucket["offer_gini"].append(float(np.nanmean(np.asarray(row_ginis, dtype=np.float64))))

        for step, bucket in sorted(per_step.items()):
            summary_rows.append(
                {
                    "policy": str(policy_name),
                    "step": int(step),
                    "count_episodes": int(len(bucket["reward"])),
                    "reward_mean": float(np.mean(bucket["reward"])) if bucket["reward"] else float("nan"),
                    "cooperation_rate_mean": (
                        float(np.mean(bucket["cooperation"])) if bucket["cooperation"] else float("nan")
                    ),
                    "resource_gini_mean": (
                        float(np.mean(bucket["resource_gini"])) if bucket["resource_gini"] else float("nan")
                    ),
                    "mean_resource_mean": (
                        float(np.mean(bucket["mean_resource"])) if bucket["mean_resource"] else float("nan")
                    ),
                    "mean_pool_grown_mean": (
                        float(np.mean(bucket["mean_pool_grown"])) if bucket["mean_pool_grown"] else float("nan")
                    ),
                    "mean_self_allocation_mean": (
                        float(np.mean(bucket["mean_self_allocation"]))
                        if bucket["mean_self_allocation"]
                        else float("nan")
                    ),
                    "offer_gini_mean": float(np.mean(bucket["offer_gini"])) if bucket["offer_gini"] else float("nan"),
                }
            )
    return summary_rows


def _plot_policy_rollout_metric_grid(
    step_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    metric_specs: Sequence[tuple[str, str, str]],
    layout: tuple[int, int],
) -> None:
    if plt is None or not step_rows:
        return

    rows_count, cols_count = int(layout[0]), int(layout[1])
    figure, axes = plt.subplots(
        rows_count,
        cols_count,
        figsize=(6.2 * float(cols_count), 3.8 * float(rows_count)),
        constrained_layout=True,
        sharex=True,
    )
    axes_flat = list(np.asarray(axes).reshape(-1))

    for axis, (metric_key, title, ylabel) in zip(axes_flat, metric_specs):
        for policy_name, _ in POLICY_SPECS:
            policy_rows = [row for row in step_rows if str(row["policy"]) == policy_name]
            if not policy_rows:
                continue
            policy_rows.sort(key=lambda row: int(row["step"]))
            x_values = np.asarray([int(row["step"]) for row in policy_rows], dtype=np.int64)
            y_values = np.asarray([float(row[metric_key]) for row in policy_rows], dtype=np.float64)
            style = POLICY_STYLES[policy_name]
            axis.plot(
                x_values,
                y_values,
                color=str(style["color"]),
                label=str(style["label"]),
                linewidth=2.2 if policy_name == "agent" else 2.0,
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    for axis in axes_flat[len(metric_specs) :]:
        axis.set_axis_off()
    for axis in axes_flat[max(0, len(axes_flat) - cols_count) :]:
        axis.set_xlabel("Step t")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)), frameon=False)
    _save_figure(figure, output_path)


def plot_policy_rollout_outcomes_over_time(
    step_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    _plot_policy_rollout_metric_grid(
        step_rows,
        output_path,
        metric_specs=[
            ("reward_mean", "Reward", "Mean Reward"),
            ("cooperation_rate_mean", "Cooperation Rate", "Actual Cooperation Rate"),
            ("resource_gini_mean", "Resource Gini", "Resource Gini"),
            ("mean_resource_mean", "Mean Resource", "Mean Resource"),
        ],
        layout=(2, 2),
    )


def plot_policy_rollout_behavior_over_time(
    step_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    _plot_policy_rollout_metric_grid(
        step_rows,
        output_path,
        metric_specs=[
            ("mean_pool_grown_mean", "Mean Pool Grown", "Mean Pool Grown"),
            ("mean_self_allocation_mean", "Mean Self Allocation", "Mean Self Allocation"),
            ("offer_gini_mean", "Offer Gini", "Offer Gini"),
        ],
        layout=(2, 2),
    )


def summarize_early_alignment_error_rows(
    step_rows: Sequence[Mapping[str, Any]],
    *,
    topology_name: str,
    adaptive_policy_name: str,
    max_step: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    agent_rows_by_step = {
        int(row["step"]): row
        for row in step_rows
        if str(row.get("policy")) == "agent" and int(row.get("step", -1)) <= int(max_step)
    }
    adaptive_rows_by_step = {
        int(row["step"]): row
        for row in step_rows
        if str(row.get("policy")) == str(adaptive_policy_name) and int(row.get("step", -1)) <= int(max_step)
    }

    steps = sorted(set(agent_rows_by_step.keys()) & set(adaptive_rows_by_step.keys()))
    rows: list[dict[str, Any]] = []
    combined_errors: list[float] = []
    for step in steps:
        agent_row = agent_rows_by_step[step]
        adaptive_row = adaptive_rows_by_step[step]
        abs_error_cooperation = abs(
            float(adaptive_row["cooperation_rate_mean"]) - float(agent_row["cooperation_rate_mean"])
        )
        abs_error_resource_gini = abs(
            float(adaptive_row["resource_gini_mean"]) - float(agent_row["resource_gini_mean"])
        )
        abs_error_self_allocation = abs(
            float(adaptive_row["mean_self_allocation_mean"]) - float(agent_row["mean_self_allocation_mean"])
        )
        abs_error_offer_gini = abs(
            float(adaptive_row["offer_gini_mean"]) - float(agent_row["offer_gini_mean"])
        )
        combined_alignment_error = float(
            np.mean(
                [
                    abs_error_cooperation,
                    abs_error_resource_gini,
                    abs_error_self_allocation,
                    abs_error_offer_gini,
                ]
            )
        )
        combined_errors.append(combined_alignment_error)
        rows.append(
            {
                "topology": str(topology_name),
                "adaptive_policy": str(adaptive_policy_name),
                "step": int(step),
                "agent_count_episodes": int(agent_row["count_episodes"]),
                "adaptive_count_episodes": int(adaptive_row["count_episodes"]),
                "agent_cooperation_rate_mean": float(agent_row["cooperation_rate_mean"]),
                "adaptive_cooperation_rate_mean": float(adaptive_row["cooperation_rate_mean"]),
                "abs_error_cooperation_rate": float(abs_error_cooperation),
                "agent_resource_gini_mean": float(agent_row["resource_gini_mean"]),
                "adaptive_resource_gini_mean": float(adaptive_row["resource_gini_mean"]),
                "abs_error_resource_gini": float(abs_error_resource_gini),
                "agent_mean_self_allocation_mean": float(agent_row["mean_self_allocation_mean"]),
                "adaptive_mean_self_allocation_mean": float(adaptive_row["mean_self_allocation_mean"]),
                "abs_error_mean_self_allocation": float(abs_error_self_allocation),
                "agent_offer_gini_mean": float(agent_row["offer_gini_mean"]),
                "adaptive_offer_gini_mean": float(adaptive_row["offer_gini_mean"]),
                "abs_error_offer_gini": float(abs_error_offer_gini),
                "combined_alignment_error": float(combined_alignment_error),
            }
        )

    summary = {
        "topology": str(topology_name),
        "adaptive_policy": str(adaptive_policy_name),
        "max_step": int(max_step),
        "step_count": int(len(rows)),
        "combined_alignment_error_mean": float(np.mean(combined_errors)) if combined_errors else float("nan"),
        "combined_alignment_error_max": float(np.max(combined_errors)) if combined_errors else float("nan"),
        "first_divergence_step_over_0p05": (
            int(next((row["step"] for row in rows if float(row["combined_alignment_error"]) > 0.05), -1))
            if rows
            else -1
        ),
        "first_divergence_step_over_0p10": (
            int(next((row["step"] for row in rows if float(row["combined_alignment_error"]) > 0.10), -1))
            if rows
            else -1
        ),
    }
    return rows, summary


def plot_early_alignment_error(
    alignment_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    adaptive_policy_name: str,
) -> None:
    if plt is None or not alignment_rows:
        return

    sorted_rows = sorted(alignment_rows, key=lambda row: int(row["step"]))
    x_values = np.asarray([int(row["step"]) for row in sorted_rows], dtype=np.int64)
    combined = np.asarray([float(row["combined_alignment_error"]) for row in sorted_rows], dtype=np.float64)
    cooperation = np.asarray([float(row["abs_error_cooperation_rate"]) for row in sorted_rows], dtype=np.float64)
    resource_gini = np.asarray([float(row["abs_error_resource_gini"]) for row in sorted_rows], dtype=np.float64)
    self_allocation = np.asarray(
        [float(row["abs_error_mean_self_allocation"]) for row in sorted_rows],
        dtype=np.float64,
    )
    offer_gini = np.asarray([float(row["abs_error_offer_gini"]) for row in sorted_rows], dtype=np.float64)

    figure, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), constrained_layout=True, sharex=True)

    axes[0].plot(x_values, combined, color="#111827", linewidth=2.4, marker="o", markersize=4.2)
    axes[0].axhline(0.05, color="#9ca3af", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].axhline(0.10, color="#d97706", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("Combined Error")
    axes[0].set_title(f"Early Alignment Error vs RL-Agent ({POLICY_STYLES.get(adaptive_policy_name, {}).get('label', adaptive_policy_name)})")
    axes[0].grid(alpha=0.25)

    component_specs = [
        (cooperation, "#1f77ff", "Cooperation"),
        (resource_gini, "#ef4444", "Resource Gini"),
        (self_allocation, "#ff7f0e", "Self Allocation"),
        (offer_gini, "#9467bd", "Offer Gini"),
    ]
    for values, color, label in component_specs:
        axes[1].plot(x_values, values, color=color, linewidth=2.0, marker="o", markersize=3.6, label=label)
    axes[1].set_xlabel("Step t")
    axes[1].set_ylabel("Absolute Error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)

    _save_figure(figure, output_path)


def episode_summaries_to_rows(policy_name: str, summaries: Sequence[EpisodeSummary]) -> list[dict[str, Any]]:
    return [
        {
            "policy": str(policy_name),
            "episode": int(summary.episode),
            "steps": int(summary.steps),
            "total_reward": float(summary.total_reward),
            "mean_reward": float(summary.mean_reward),
            "final_cooperation_rate": float(summary.final_cooperation_rate),
            "final_gini": float(summary.final_gini),
            "final_mean_resource": float(summary.final_mean_resource),
        }
        for summary in summaries
    ]


def summarize_topology_case(
    topology_name: str,
    episode_summary_rows: Sequence[Mapping[str, Any]],
    *,
    step_rows: Sequence[Mapping[str, Any]] | None = None,
    adaptive_early_alignment: Mapping[str, Any] | None = None,
    adaptive_early_alignment_by_policy: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, _ in POLICY_SPECS:
        policy_rows = [row for row in episode_summary_rows if str(row["policy"]) == policy_name]
        if not policy_rows:
            continue
        policy_step_rows = [] if step_rows is None else [row for row in step_rows if str(row.get("policy")) == policy_name]
        mean_self_allocation = (
            float(np.mean([float(row["mean_self_allocation_mean"]) for row in policy_step_rows]))
            if policy_step_rows
            else float("nan")
        )
        if (
            str(policy_name).startswith("adaptive_")
            and adaptive_early_alignment_by_policy is not None
            and str(policy_name) in adaptive_early_alignment_by_policy
        ):
            early_alignment_payload = dict(adaptive_early_alignment_by_policy[str(policy_name)])
            early_alignment_mean = float(early_alignment_payload.get("combined_alignment_error_mean", float("nan")))
            early_alignment_max = float(early_alignment_payload.get("combined_alignment_error_max", float("nan")))
            first_divergence = int(early_alignment_payload.get("first_divergence_step_over_0p05", -1))
        elif str(policy_name).startswith("adaptive_") and adaptive_early_alignment is not None:
            early_alignment_mean = float(adaptive_early_alignment.get("combined_alignment_error_mean", float("nan")))
            early_alignment_max = float(adaptive_early_alignment.get("combined_alignment_error_max", float("nan")))
            first_divergence = int(adaptive_early_alignment.get("first_divergence_step_over_0p05", -1))
        else:
            early_alignment_mean = float("nan")
            early_alignment_max = float("nan")
            first_divergence = -1
        rows.append(
            {
                "topology": topology_name,
                "topology_label": TOPOLOGY_LABELS.get(topology_name, topology_name),
                "policy": policy_name,
                "episodes": int(len(policy_rows)),
                "return_mean": float(np.mean([float(row["total_reward"]) for row in policy_rows])),
                "reward_per_step_mean": float(np.mean([float(row["mean_reward"]) for row in policy_rows])),
                "final_cooperation_mean": float(
                    np.mean([float(row["final_cooperation_rate"]) for row in policy_rows])
                ),
                "final_gini_mean": float(np.mean([float(row["final_gini"]) for row in policy_rows])),
                "final_mean_resource_mean": float(
                    np.mean([float(row["final_mean_resource"]) for row in policy_rows])
                ),
                "mean_self_allocation_mean": float(mean_self_allocation),
                "adaptive_early_alignment_mean": float(early_alignment_mean),
                "adaptive_early_alignment_max": float(early_alignment_max),
                "first_divergence_step_over_0p05": int(first_divergence),
            }
        )
    return rows


def _build_method_summary_row(
    *,
    method_name: str,
    analysis_dir: Path,
    policy_name: str,
    topology_summary_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    analysis_summary: Mapping[str, Any],
) -> dict[str, Any]:
    topology_row = next((row for row in topology_summary_rows if str(row.get("policy")) == str(policy_name)), None)
    policy_step_rows = [row for row in step_rows if str(row.get("policy")) == str(policy_name)]
    mean_self_allocation = (
        float(np.mean([float(row["mean_self_allocation_mean"]) for row in policy_step_rows]))
        if policy_step_rows
        else float("nan")
    )
    early_alignment_by_policy = analysis_summary.get("adaptive_early_alignment_by_policy")
    if isinstance(early_alignment_by_policy, Mapping) and str(policy_name) in early_alignment_by_policy:
        early_alignment = dict(early_alignment_by_policy[str(policy_name)])
    else:
        early_alignment = dict(analysis_summary.get("adaptive_early_alignment", {}))
    return {
        "method": str(method_name),
        "analysis_dir": str(analysis_dir),
        "policy": str(policy_name),
        "topology": None if topology_row is None else str(topology_row.get("topology")),
        "reward_per_step_mean": None if topology_row is None else float(topology_row["reward_per_step_mean"]),
        "final_cooperation_mean": None if topology_row is None else float(topology_row["final_cooperation_mean"]),
        "final_gini_mean": None if topology_row is None else float(topology_row["final_gini_mean"]),
        "final_mean_resource_mean": None if topology_row is None else float(topology_row["final_mean_resource_mean"]),
        "mean_self_allocation_mean": float(mean_self_allocation),
        "adaptive_early_alignment_mean": float(early_alignment.get("combined_alignment_error_mean", float("nan"))),
        "adaptive_early_alignment_max": float(early_alignment.get("combined_alignment_error_max", float("nan"))),
        "first_divergence_step_over_0p05": int(early_alignment.get("first_divergence_step_over_0p05", -1)),
    }


def _load_existing_run_summary_rows(analysis_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import csv

    topology_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    analysis_summary: dict[str, Any] = {}

    topology_path = analysis_dir / "topology_summary.csv"
    if topology_path.exists():
        with topology_path.open("r", encoding="utf-8", newline="") as handle:
            topology_rows = [dict(row) for row in csv.DictReader(handle)]

    step_path = analysis_dir / "policy_rollout_step_metrics.csv"
    if step_path.exists():
        with step_path.open("r", encoding="utf-8", newline="") as handle:
            step_rows = [dict(row) for row in csv.DictReader(handle)]

    summary_path = analysis_dir / "analysis_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                analysis_summary = dict(loaded)
    return topology_rows, step_rows, analysis_summary


def resolve_continuous_2d_policy_suite(
    *,
    agent_reference_rows: Sequence[Mapping[str, Any]],
    adaptive_policy_mode: str,
    second_variable_name: str,
    basis_mode: str,
    x1_hinge_count: int,
    x2_hinge_count: int,
    surface_grid_size: int,
) -> dict[str, Any]:
    mode = str(adaptive_policy_mode)
    include_softmax = mode in {"continuous2d_suite", "continuous2d_softmax"}
    include_unitself = mode in {"continuous2d_suite", "continuous2d_unitself"}
    if not include_softmax and not include_unitself:
        raise ValueError(f"Unsupported 2D continuous adaptive mode: {adaptive_policy_mode}")

    policy_specs: list[tuple[str, Any]] = []
    fit_summaries_by_policy: dict[str, Any] = {}
    surface_rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    interpretations_by_policy: dict[str, Any] = {}
    reference_fit_summary_rows: list[dict[str, Any]] = []

    if include_softmax:
        softmax_policy, softmax_fit_summary = fit_continuous_2d_softmax_policy(
            agent_reference_rows,
            second_variable_name=str(second_variable_name),
            basis_mode=str(basis_mode),
            x1_hinge_count=max(0, int(x1_hinge_count)),
            x2_hinge_count=max(0, int(x2_hinge_count)),
        )
        softmax_policy_name = str(softmax_fit_summary.get("policy_name", "adaptive_continuous2d_softmax_mix"))
        policy_specs.append((softmax_policy_name, softmax_policy))
        fit_summaries_by_policy[softmax_policy_name] = dict(softmax_fit_summary)
        reference_fit_summary_rows.append(
            build_continuous_reference_fit_summary_row(softmax_policy, softmax_fit_summary)
        )
        if isinstance(softmax_policy, ContinuousTwoStateMixtureAllocationPolicy):
            surface_rows_by_policy[softmax_policy_name] = build_continuous_2d_surface_rows(
                softmax_policy,
                reference_rows=agent_reference_rows,
                second_variable_name=str(second_variable_name),
                grid_size=max(8, int(surface_grid_size)),
            )
            interpretations_by_policy[softmax_policy_name] = summarize_continuous_2d_rule_interpretation(
                softmax_policy,
                reference_rows=agent_reference_rows,
                second_variable_name=str(second_variable_name),
            )

    if include_unitself:
        unitself_policy, unitself_fit_summary = fit_continuous_2d_unitself_policy(
            agent_reference_rows,
            second_variable_name=str(second_variable_name),
            basis_mode=str(basis_mode),
            x1_hinge_count=max(0, int(x1_hinge_count)),
            x2_hinge_count=max(0, int(x2_hinge_count)),
        )
        unitself_policy_name = str(unitself_fit_summary.get("policy_name", "adaptive_continuous2d_unitself_mix"))
        policy_specs.append((unitself_policy_name, unitself_policy))
        fit_summaries_by_policy[unitself_policy_name] = dict(unitself_fit_summary)
        reference_fit_summary_rows.append(
            build_continuous_reference_fit_summary_row(unitself_policy, unitself_fit_summary)
        )
        if isinstance(unitself_policy, ContinuousTwoStateMixtureAllocationPolicy):
            surface_rows_by_policy[unitself_policy_name] = build_continuous_2d_surface_rows(
                unitself_policy,
                reference_rows=agent_reference_rows,
                second_variable_name=str(second_variable_name),
                grid_size=max(8, int(surface_grid_size)),
            )
            interpretations_by_policy[unitself_policy_name] = summarize_continuous_2d_rule_interpretation(
                unitself_policy,
                reference_rows=agent_reference_rows,
                second_variable_name=str(second_variable_name),
            )

    return {
        "policy_specs": policy_specs,
        "fit_summaries_by_policy": fit_summaries_by_policy,
        "surface_rows_by_policy": surface_rows_by_policy,
        "interpretations_by_policy": interpretations_by_policy,
        "reference_fit_summary_rows": reference_fit_summary_rows,
    }


def resolve_adaptive_policy_for_topology(
    *,
    run_dir: Path,
    conditioning_mode: str,
    conditioned_profile_family: str,
    conditioned_degree_bin_mode: str,
    conditioned_degree_bin_count: int,
    conditioned_degree_bin_min_count: int,
    conditioned_resource_bin_count: int,
    adaptive_policy_mode: str,
    continuous_second_variable: str,
    continuous_basis_mode: str,
    continuous_x1_hinge_count: int,
    continuous_x2_hinge_count: int,
    continuous_surface_grid_size: int,
    ego_resource_x_mode: str,
    ego_resource_normalizer: float | None,
    adaptive_ego_resource_profile_mode: str,
    adaptive_ego_resource_bin_count: int,
    spline_high_knot_style: str,
    tail_cap_mode: str,
    tail_cap_resource: float | None,
    tail_cap_soft_scale: float,
    spline_l2_penalty: float,
    legacy_profile_csv: str | None,
    legacy_knots: str | None,
    legacy_smooth_window: int,
    reference_bin_count: int,
    agent_reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, Any, list[dict[str, Any]], dict[str, Any]]:
    if str(adaptive_policy_mode) in {"continuous2d_suite", "continuous2d_softmax", "continuous2d_unitself"}:
        suite = resolve_continuous_2d_policy_suite(
            agent_reference_rows=agent_reference_rows,
            adaptive_policy_mode=str(adaptive_policy_mode),
            second_variable_name=str(continuous_second_variable),
            basis_mode=str(continuous_basis_mode),
            x1_hinge_count=max(0, int(continuous_x1_hinge_count)),
            x2_hinge_count=max(0, int(continuous_x2_hinge_count)),
            surface_grid_size=max(8, int(continuous_surface_grid_size)),
        )
        primary_policy_name, primary_policy = suite["policy_specs"][0]
        primary_summary = dict(suite["fit_summaries_by_policy"][primary_policy_name])
        primary_summary.update(
            {
                "suite_policy_names": [str(name) for name, _ in suite["policy_specs"]],
                "continuous_second_variable": str(continuous_second_variable),
                "continuous_basis_mode": str(continuous_basis_mode),
            }
        )
        return str(primary_policy_name), primary_policy, [], {**dict(primary_policy.profile_summary()), **primary_summary}
    if str(conditioning_mode) == "sender_degree_bin":
        policy, profile_rows, fit_summary, policy_name = fit_degree_conditioned_spline_policy(
            agent_reference_rows,
            degree_bin_mode=str(conditioned_degree_bin_mode),
            degree_bin_count=max(1, int(conditioned_degree_bin_count)),
            degree_bin_min_count=max(1, int(conditioned_degree_bin_min_count)),
            profile_family=str(conditioned_profile_family),
            reference_bin_count=max(2, int(reference_bin_count)),
            ego_resource_x_mode=str(ego_resource_x_mode),
            ego_resource_normalizer=ego_resource_normalizer,
            ego_resource_profile_mode=str(adaptive_ego_resource_profile_mode),
            ego_resource_bin_count=max(2, int(adaptive_ego_resource_bin_count)),
            spline_high_knot_style=str(spline_high_knot_style),
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
            spline_l2_penalty=float(spline_l2_penalty),
        )
        return policy_name, policy, profile_rows, {**dict(policy.profile_summary()), **dict(fit_summary)}
    if str(conditioning_mode) == "sender_resource_norm_bin":
        policy, profile_rows, fit_summary, policy_name = fit_resource_conditioned_spline_policy(
            agent_reference_rows,
            resource_bin_count=max(1, int(conditioned_resource_bin_count)),
            profile_family=str(conditioned_profile_family),
            reference_bin_count=max(2, int(reference_bin_count)),
            ego_resource_x_mode=str(ego_resource_x_mode),
            ego_resource_normalizer=ego_resource_normalizer,
            ego_resource_profile_mode=str(adaptive_ego_resource_profile_mode),
            ego_resource_bin_count=max(2, int(adaptive_ego_resource_bin_count)),
            spline_high_knot_style=str(spline_high_knot_style),
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
            spline_l2_penalty=float(spline_l2_penalty),
        )
        return policy_name, policy, profile_rows, {**dict(policy.profile_summary()), **dict(fit_summary)}

    mode = str(adaptive_policy_mode)
    if mode == "lookup_2way":
        policy = resolve_adaptive_unitinv_equal_policy(
            run_dir=run_dir,
            profile_csv_arg=legacy_profile_csv,
            knots_arg=legacy_knots,
            smooth_window=max(1, int(legacy_smooth_window)),
        )
        return (
            "adaptive_unitinv_equal_mix",
            policy,
            build_adaptive_unitinv_equal_profile_rows(policy),
            dict(policy.profile_summary()),
        )
    if mode == "smooth_2way":
        policy, prepared_rows, fit_summary = fit_smooth_two_way_policy(
            agent_reference_rows,
            tail_cap_mode=str(tail_cap_mode),
            tail_cap_resource=tail_cap_resource,
            tail_cap_soft_scale=float(tail_cap_soft_scale),
        )
        policy_name = "adaptive_smooth_2way_mix" if isinstance(policy, SmoothLogEgoResourceMixtureAllocationPolicy) else "adaptive_unitinv_equal_mix"
        profile_rows = (
            build_adaptive_profile_rows(
                policy,
                reference_rows=prepared_rows,
                reference_bin_count=max(2, int(reference_bin_count)),
            )
            if isinstance(policy, SmoothLogEgoResourceMixtureAllocationPolicy)
            else build_adaptive_unitinv_equal_profile_rows(policy)
        )
        return policy_name, policy, profile_rows, {**dict(policy.profile_summary()), **dict(fit_summary)}
    if mode == "smooth_3way":
        if str(adaptive_ego_resource_profile_mode) == "discrete":
            policy, prepared_rows, fit_summary = fit_binned_three_way_policy(
                agent_reference_rows,
                bin_count=max(2, int(adaptive_ego_resource_bin_count)),
                tail_cap_mode=str(tail_cap_mode),
                tail_cap_resource=tail_cap_resource,
                tail_cap_soft_scale=float(tail_cap_soft_scale),
                ego_resource_x_mode=str(ego_resource_x_mode),
                ego_resource_normalizer=ego_resource_normalizer,
            )
        else:
            policy, prepared_rows, fit_summary = fit_smooth_three_way_policy(
                agent_reference_rows,
                tail_cap_mode=str(tail_cap_mode),
                tail_cap_resource=tail_cap_resource,
                tail_cap_soft_scale=float(tail_cap_soft_scale),
                ego_resource_x_mode=str(ego_resource_x_mode),
                ego_resource_normalizer=ego_resource_normalizer,
            )
        policy_name = (
            "adaptive_smooth_3way_mix" if isinstance(policy, SmoothLogEgoResourceMixtureAllocationPolicy) else "adaptive_unitinv_equal_mix"
        )
        profile_rows = (
            build_adaptive_profile_rows(
                policy,
                reference_rows=prepared_rows,
                reference_bin_count=max(2, int(reference_bin_count)),
            )
            if isinstance(policy, SmoothLogEgoResourceMixtureAllocationPolicy)
            else build_adaptive_unitinv_equal_profile_rows(policy)
        )
        return policy_name, policy, profile_rows, {**dict(policy.profile_summary()), **dict(fit_summary)}
    if mode == "spline_3way":
        if str(adaptive_ego_resource_profile_mode) == "discrete":
            policy, prepared_rows, fit_summary = fit_binned_three_way_policy(
                agent_reference_rows,
                bin_count=max(2, int(adaptive_ego_resource_bin_count)),
                tail_cap_mode=str(tail_cap_mode),
                tail_cap_resource=tail_cap_resource,
                tail_cap_soft_scale=float(tail_cap_soft_scale),
                ego_resource_x_mode=str(ego_resource_x_mode),
                ego_resource_normalizer=ego_resource_normalizer,
            )
        else:
            policy, prepared_rows, fit_summary = fit_spline_three_way_policy(
                agent_reference_rows,
                spline_high_knot_style=str(spline_high_knot_style),
                tail_cap_mode=str(tail_cap_mode),
                tail_cap_resource=tail_cap_resource,
                tail_cap_soft_scale=float(tail_cap_soft_scale),
                l2_penalty=float(spline_l2_penalty),
                ego_resource_x_mode=str(ego_resource_x_mode),
                ego_resource_normalizer=ego_resource_normalizer,
            )
        policy_name = (
            (
                "adaptive_spline_3way_mix"
                if isinstance(policy, SmoothLogEgoResourceMixtureAllocationPolicy)
                else "adaptive_binned_3way_mix"
            )
        )
        profile_rows = (
            build_adaptive_profile_rows(
                policy,
                reference_rows=prepared_rows,
                reference_bin_count=max(2, int(reference_bin_count)),
            )
            if isinstance(policy, SmoothLogEgoResourceMixtureAllocationPolicy)
            else build_adaptive_unitinv_equal_profile_rows(policy)
        )
        return policy_name, policy, profile_rows, {**dict(policy.profile_summary()), **dict(fit_summary)}
    raise ValueError(f"Unsupported adaptive-policy-mode: {adaptive_policy_mode}")


def analyze_single_topology(
    *,
    actor: Any,
    run_dir: Path,
    checkpoint_name: str,
    output_dir: Path,
    spec: Mapping[str, Any],
    topology_name: str,
    episodes: int,
    max_steps: int | None,
    episode_length_override: int | None,
    rollout_seed: int,
    adaptive_policy_mode: str,
    adaptive_conditioning_mode: str,
    adaptive_conditioned_profile_family: str,
    conditioned_degree_bin_mode: str,
    conditioned_degree_bin_count: int,
    conditioned_degree_bin_min_count: int,
    conditioned_resource_bin_count: int,
    adaptive_unitinv_equal_profile_csv: str | None,
    adaptive_unitinv_equal_knots: str | None,
    adaptive_unitinv_equal_smooth_window: int,
    adaptive_reference_bin_count: int,
    continuous_second_variable: str,
    continuous_basis_mode: str,
    continuous_x1_hinge_count: int,
    continuous_x2_hinge_count: int,
    continuous_surface_grid_size: int,
    ego_resource_x_mode: str,
    ego_resource_normalizer: float | None,
    adaptive_ego_resource_profile_mode: str,
    adaptive_ego_resource_bin_count: int,
    spline_high_knot_style: str,
    tail_cap_mode: str,
    tail_cap_resource: float | None,
    tail_cap_soft_scale: float,
    spline_l2_penalty: float,
    early_alignment_max_step: int,
) -> list[dict[str, Any]]:
    global POLICY_SPECS

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_spec = deepcopy(dict(spec))
    if episode_length_override is not None:
        effective_spec["dynamics"] = dict(effective_spec["dynamics"])
        effective_spec["dynamics"]["episode_length"] = int(episode_length_override)

    graph = build_graph_from_spec(effective_spec)
    env_config = build_env_config_from_spec(effective_spec, graph)
    env = SPGGEnv(env_config, graph)

    snapshots_by_policy: dict[str, list[Snapshot]] = {}
    episode_rows: list[dict[str, Any]] = []

    agent_snapshots, agent_summaries = run_deterministic_rollouts(
        actor,
        env,
        episodes=max(1, int(episodes)),
        seed=int(rollout_seed),
        max_steps=max_steps,
    )
    snapshots_by_policy["agent"] = list(agent_snapshots)
    episode_rows.extend(episode_summaries_to_rows("agent", agent_summaries))

    agent_reference_rows = collect_agent_reference_rows(agent_snapshots)
    is_continuous_2d_mode = str(adaptive_policy_mode) in {
        "continuous2d_suite",
        "continuous2d_softmax",
        "continuous2d_unitself",
    }
    current_adaptive_policy_specs: list[tuple[str, Any]] = []
    adaptive_policy_name = ""
    adaptive_policy_summary: dict[str, Any] = {}
    adaptive_policy_summaries_by_policy: dict[str, Any] = {}
    adaptive_early_alignment_by_policy: dict[str, Any] = {}

    if is_continuous_2d_mode:
        continuous_suite = resolve_continuous_2d_policy_suite(
            agent_reference_rows=agent_reference_rows,
            adaptive_policy_mode=str(adaptive_policy_mode),
            second_variable_name=str(continuous_second_variable),
            basis_mode=str(continuous_basis_mode),
            x1_hinge_count=max(0, int(continuous_x1_hinge_count)),
            x2_hinge_count=max(0, int(continuous_x2_hinge_count)),
            surface_grid_size=max(8, int(continuous_surface_grid_size)),
        )
        current_adaptive_policy_specs = list(continuous_suite["policy_specs"])
        POLICY_SPECS = build_policy_specs(extra_policy_specs=current_adaptive_policy_specs)
        adaptive_policy_name = str(current_adaptive_policy_specs[0][0])
        adaptive_policy_summary = dict(continuous_suite["fit_summaries_by_policy"][adaptive_policy_name])

        reference_fit_summary_rows = list(continuous_suite["reference_fit_summary_rows"])
        if reference_fit_summary_rows:
            write_csv(output_dir / "reference_fit_summary.csv", reference_fit_summary_rows)

        fit_coefficients_payload: dict[str, Any] = {}
        for policy_name, policy in current_adaptive_policy_specs:
            artifact_stem = _policy_artifact_stem(str(policy_name))
            fit_coefficients_payload[str(policy_name)] = {
                "profile_summary": dict(policy.profile_summary()),
                "rule_interpretation": dict(continuous_suite["interpretations_by_policy"].get(str(policy_name), {})),
            }
            adaptive_policy_summaries_by_policy[str(policy_name)] = fit_coefficients_payload[str(policy_name)][
                "profile_summary"
            ]

            surface_rows = list(continuous_suite["surface_rows_by_policy"].get(str(policy_name), []))
            if surface_rows:
                write_csv(output_dir / f"{artifact_stem}_surface.csv", surface_rows)
                plot_continuous_2d_weight_surfaces(
                    surface_rows,
                    output_dir / f"{artifact_stem}_surface.png",
                    title_prefix=str(POLICY_STYLES.get(str(policy_name), {}).get("label", policy_name)),
                    second_variable_label=str(continuous_second_variable),
                )

        with (output_dir / "fit_coefficients.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "adaptive_policy_mode": str(adaptive_policy_mode),
                    "continuous_second_variable": str(continuous_second_variable),
                    "continuous_basis_mode": str(continuous_basis_mode),
                    "policies": fit_coefficients_payload,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    else:
        adaptive_policy_name, adaptive_policy, adaptive_profile_rows, adaptive_policy_summary = (
            resolve_adaptive_policy_for_topology(
                run_dir=run_dir,
                conditioning_mode=str(adaptive_conditioning_mode),
                conditioned_profile_family=str(adaptive_conditioned_profile_family),
                conditioned_degree_bin_mode=str(conditioned_degree_bin_mode),
                conditioned_degree_bin_count=max(1, int(conditioned_degree_bin_count)),
                conditioned_degree_bin_min_count=max(1, int(conditioned_degree_bin_min_count)),
                conditioned_resource_bin_count=max(1, int(conditioned_resource_bin_count)),
                adaptive_policy_mode=str(adaptive_policy_mode),
                continuous_second_variable=str(continuous_second_variable),
                continuous_basis_mode=str(continuous_basis_mode),
                continuous_x1_hinge_count=max(0, int(continuous_x1_hinge_count)),
                continuous_x2_hinge_count=max(0, int(continuous_x2_hinge_count)),
                continuous_surface_grid_size=max(8, int(continuous_surface_grid_size)),
                ego_resource_x_mode=str(ego_resource_x_mode),
                ego_resource_normalizer=ego_resource_normalizer,
                adaptive_ego_resource_profile_mode=str(adaptive_ego_resource_profile_mode),
                adaptive_ego_resource_bin_count=max(2, int(adaptive_ego_resource_bin_count)),
                spline_high_knot_style=str(spline_high_knot_style),
                tail_cap_mode=str(tail_cap_mode),
                tail_cap_resource=tail_cap_resource,
                tail_cap_soft_scale=float(tail_cap_soft_scale),
                spline_l2_penalty=float(spline_l2_penalty),
                legacy_profile_csv=adaptive_unitinv_equal_profile_csv,
                legacy_knots=adaptive_unitinv_equal_knots,
                legacy_smooth_window=max(1, int(adaptive_unitinv_equal_smooth_window)),
                reference_bin_count=max(2, int(adaptive_reference_bin_count)),
                agent_reference_rows=agent_reference_rows,
            )
        )
        current_adaptive_policy_specs = [(adaptive_policy_name, adaptive_policy)]
        POLICY_SPECS = build_policy_specs(adaptive_policy_name, adaptive_policy)
        adaptive_policy_summaries_by_policy[str(adaptive_policy_name)] = dict(adaptive_policy_summary)

        if str(adaptive_conditioning_mode) == "sender_degree_bin":
            write_csv(output_dir / "degree_conditioned_profile.csv", adaptive_profile_rows)
            plot_degree_conditioned_profile(adaptive_profile_rows, output_dir / "degree_conditioned_profile.png")
        elif str(adaptive_conditioning_mode) == "sender_resource_norm_bin":
            write_csv(output_dir / "resource_conditioned_profile.csv", adaptive_profile_rows)
            plot_resource_conditioned_profile(adaptive_profile_rows, output_dir / "resource_conditioned_profile.png")
        else:
            adaptive_profile_csv_path = output_dir / "adaptive_ego_mean_resource_profile.csv"
            adaptive_profile_png_path = output_dir / "adaptive_ego_mean_resource_profile.png"
            write_csv(adaptive_profile_csv_path, adaptive_profile_rows)
            plot_adaptive_unitinv_equal_profile(adaptive_profile_rows, adaptive_profile_png_path)
            write_csv(output_dir / "adaptive_unitinv_equal_w_vs_ego_mean_resource.csv", adaptive_profile_rows)
            plot_adaptive_unitinv_equal_profile(
                adaptive_profile_rows,
                output_dir / "adaptive_unitinv_equal_w_vs_ego_mean_resource.png",
            )

    for policy_name, policy in POLICY_SPECS:
        if policy_name == "agent":
            continue
        rule_snapshots, rule_summaries = run_rule_policy_rollouts(
            policy,
            env_config=env_config,
            graph=graph,
            episodes=max(1, int(episodes)),
            seed=int(rollout_seed),
            max_steps=max_steps,
        )
        snapshots_by_policy[policy_name] = list(rule_snapshots)
        episode_rows.extend(episode_summaries_to_rows(policy_name, rule_summaries))

    step_rows = summarize_policy_rollout_step_metrics(snapshots_by_policy)
    plot_policy_rollout_outcomes_over_time(step_rows, output_dir / "policy_rollout_outcomes_over_time.png")
    plot_policy_rollout_behavior_over_time(step_rows, output_dir / "policy_rollout_behavior_over_time.png")
    write_csv(output_dir / "policy_rollout_step_metrics.csv", step_rows)
    write_csv(output_dir / "episode_summary.csv", episode_rows)

    combined_early_alignment_rows: list[dict[str, Any]] = []
    for current_policy_name, _ in current_adaptive_policy_specs:
        early_alignment_rows, early_alignment_summary = summarize_early_alignment_error_rows(
            step_rows,
            topology_name=topology_name,
            adaptive_policy_name=str(current_policy_name),
            max_step=max(0, int(early_alignment_max_step)),
        )
        adaptive_early_alignment_by_policy[str(current_policy_name)] = dict(early_alignment_summary)
        combined_early_alignment_rows.extend(early_alignment_rows)
        png_name = (
            "adaptive_early_alignment_error.png"
            if len(current_adaptive_policy_specs) == 1
            else f"{_policy_artifact_stem(str(current_policy_name))}_early_alignment.png"
        )
        plot_early_alignment_error(
            early_alignment_rows,
            output_dir / png_name,
            adaptive_policy_name=str(current_policy_name),
        )
    write_csv(output_dir / "adaptive_early_alignment_error.csv", combined_early_alignment_rows)

    summary_payload = {
        "run_dir": str(run_dir),
        "checkpoint_name": str(checkpoint_name),
        "topology": str(topology_name),
        "graph_seed": int(effective_spec["seed"]),
        "rollout_seed": int(rollout_seed),
        "policies": [policy_name for policy_name, _ in POLICY_SPECS],
        "episode_count": int(episodes),
        "max_steps": None if max_steps is None else int(max_steps),
        "episode_length": int(effective_spec["dynamics"]["episode_length"]),
        "adaptive_unitinv_equal_profile": dict(adaptive_policy_summary),
        "adaptive_policy_mode": str(adaptive_policy_mode),
        "adaptive_conditioning_mode": str(adaptive_conditioning_mode),
        "adaptive_conditioned_profile_family": str(adaptive_conditioned_profile_family),
        "continuous_second_variable": str(continuous_second_variable),
        "continuous_basis_mode": str(continuous_basis_mode),
        "adaptive_reference_row_count": int(len(agent_reference_rows)),
        "adaptive_early_alignment": dict(adaptive_early_alignment_by_policy.get(str(adaptive_policy_name), {})),
        "adaptive_early_alignment_by_policy": dict(adaptive_early_alignment_by_policy),
        "adaptive_policy_summaries": dict(adaptive_policy_summaries_by_policy),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    topology_summary_rows = summarize_topology_case(
        topology_name,
        episode_rows,
        step_rows=step_rows,
        adaptive_early_alignment=dict(adaptive_early_alignment_by_policy.get(str(adaptive_policy_name), {})),
        adaptive_early_alignment_by_policy=adaptive_early_alignment_by_policy,
    )

    if str(adaptive_conditioning_mode) == "sender_degree_bin":
        baseline_dir = run_dir / "0421_rule_allocation_over_time_ba"
        if baseline_dir.exists():
            baseline_topology_rows, baseline_step_rows, baseline_summary = _load_existing_run_summary_rows(baseline_dir)
            baseline_policy_candidates = [row for row in baseline_topology_rows if str(row.get("policy", "")).startswith("adaptive_")]
            current_policy_candidates = [row for row in topology_summary_rows if str(row.get("policy", "")).startswith("adaptive_")]
            comparison_rows: list[dict[str, Any]] = []
            if baseline_policy_candidates:
                comparison_rows.append(
                    _build_method_summary_row(
                        method_name="global ego_mean_resource only",
                        analysis_dir=baseline_dir,
                        policy_name=str(baseline_policy_candidates[0]["policy"]),
                        topology_summary_rows=baseline_topology_rows,
                        step_rows=baseline_step_rows,
                        analysis_summary=baseline_summary,
                    )
                )
            if current_policy_candidates:
                comparison_rows.append(
                    _build_method_summary_row(
                        method_name="degree-conditioned ego_mean_resource",
                        analysis_dir=output_dir,
                        policy_name=str(current_policy_candidates[0]["policy"]),
                        topology_summary_rows=topology_summary_rows,
                        step_rows=step_rows,
                        analysis_summary=summary_payload,
                    )
                )
            if comparison_rows:
                write_csv(output_dir / "stage_a_comparison_summary.csv", comparison_rows)
    if str(adaptive_conditioning_mode) == "sender_resource_norm_bin":
        comparison_rows: list[dict[str, Any]] = []
        baseline_dir = run_dir / "0421_rule_allocation_over_time_ba"
        degree_dir = run_dir / "0421_rule_allocation_over_time_ba_degree_conditioned"
        current_policy_candidates = [row for row in topology_summary_rows if str(row.get("policy", "")).startswith("adaptive_")]
        if baseline_dir.exists():
            baseline_topology_rows, baseline_step_rows, baseline_summary = _load_existing_run_summary_rows(baseline_dir)
            baseline_policy_candidates = [row for row in baseline_topology_rows if str(row.get("policy", "")).startswith("adaptive_")]
            if baseline_policy_candidates:
                comparison_rows.append(
                    _build_method_summary_row(
                        method_name="global ego_mean_resource only",
                        analysis_dir=baseline_dir,
                        policy_name=str(baseline_policy_candidates[0]["policy"]),
                        topology_summary_rows=baseline_topology_rows,
                        step_rows=baseline_step_rows,
                        analysis_summary=baseline_summary,
                    )
                )
        if degree_dir.exists():
            degree_topology_rows, degree_step_rows, degree_summary = _load_existing_run_summary_rows(degree_dir)
            degree_policy_candidates = [row for row in degree_topology_rows if str(row.get("policy", "")).startswith("adaptive_")]
            if degree_policy_candidates:
                comparison_rows.append(
                    _build_method_summary_row(
                        method_name="degree-conditioned ego_mean_resource",
                        analysis_dir=degree_dir,
                        policy_name=str(degree_policy_candidates[0]["policy"]),
                        topology_summary_rows=degree_topology_rows,
                        step_rows=degree_step_rows,
                        analysis_summary=degree_summary,
                    )
                )
        if current_policy_candidates:
            comparison_rows.append(
                _build_method_summary_row(
                    method_name="resource_norm-conditioned ego_mean_resource",
                    analysis_dir=output_dir,
                    policy_name=str(current_policy_candidates[0]["policy"]),
                    topology_summary_rows=topology_summary_rows,
                    step_rows=step_rows,
                    analysis_summary=summary_payload,
                )
            )
        if comparison_rows:
            write_csv(output_dir / "comparison_summary.csv", comparison_rows)
    if is_continuous_2d_mode:
        comparison_rows: list[dict[str, Any]] = []
        baseline_dir = run_dir / "0421_rule_allocation_over_time_ba"
        degree_dir = run_dir / "0421_rule_allocation_over_time_ba_degree_conditioned"
        resource_dir = run_dir / "0421_rule_allocation_over_time_ba_resource_conditioned"
        if baseline_dir.exists():
            baseline_topology_rows, baseline_step_rows, baseline_summary = _load_existing_run_summary_rows(baseline_dir)
            baseline_policy_candidates = [row for row in baseline_topology_rows if str(row.get("policy", "")).startswith("adaptive_")]
            if baseline_policy_candidates:
                comparison_rows.append(
                    _build_method_summary_row(
                        method_name="global ego_mean_resource only",
                        analysis_dir=baseline_dir,
                        policy_name=str(baseline_policy_candidates[0]["policy"]),
                        topology_summary_rows=baseline_topology_rows,
                        step_rows=baseline_step_rows,
                        analysis_summary=baseline_summary,
                    )
                )
        if degree_dir.exists():
            degree_topology_rows, degree_step_rows, degree_summary = _load_existing_run_summary_rows(degree_dir)
            degree_policy_candidates = [row for row in degree_topology_rows if str(row.get("policy", "")).startswith("adaptive_")]
            if degree_policy_candidates:
                comparison_rows.append(
                    _build_method_summary_row(
                        method_name="degree-conditioned ego_mean_resource",
                        analysis_dir=degree_dir,
                        policy_name=str(degree_policy_candidates[0]["policy"]),
                        topology_summary_rows=degree_topology_rows,
                        step_rows=degree_step_rows,
                        analysis_summary=degree_summary,
                    )
                )
        if resource_dir.exists():
            resource_topology_rows, resource_step_rows, resource_summary = _load_existing_run_summary_rows(resource_dir)
            resource_policy_candidates = [row for row in resource_topology_rows if str(row.get("policy", "")).startswith("adaptive_")]
            if resource_policy_candidates:
                comparison_rows.append(
                    _build_method_summary_row(
                        method_name="resource_norm-conditioned ego_mean_resource",
                        analysis_dir=resource_dir,
                        policy_name=str(resource_policy_candidates[0]["policy"]),
                        topology_summary_rows=resource_topology_rows,
                        step_rows=resource_step_rows,
                        analysis_summary=resource_summary,
                    )
                )
        for current_policy_name, _ in current_adaptive_policy_specs:
            method_name = (
                "new 2D continuous simplex model"
                if str(current_policy_name) == "adaptive_continuous2d_softmax_mix"
                else "new 2D UnitInv+Self model"
            )
            comparison_rows.append(
                _build_method_summary_row(
                    method_name=method_name,
                    analysis_dir=output_dir,
                    policy_name=str(current_policy_name),
                    topology_summary_rows=topology_summary_rows,
                    step_rows=step_rows,
                    analysis_summary=summary_payload,
                )
            )
        if comparison_rows:
            write_csv(output_dir / "comparison_summary.csv", comparison_rows)

    return topology_summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RL Agent with rule-based allocation baselines over time."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(str(SCRIPT_DEFAULTS["run_dir"])),
        help="Experiment directory containing results.json and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=str(SCRIPT_DEFAULTS["checkpoint_name"]),
        help="Checkpoint filename inside <run-dir>/checkpoints/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None if SCRIPT_DEFAULTS["output_dir"] is None else Path(str(SCRIPT_DEFAULTS["output_dir"])),
        help="Absolute or relative output directory. Overrides --output-subdir-name when set.",
    )
    parser.add_argument(
        "--output-subdir-name",
        type=str,
        default=None
        if SCRIPT_DEFAULTS["output_subdir_name"] is None
        else str(SCRIPT_DEFAULTS["output_subdir_name"]),
        help="When --output-dir is not set, write outputs under <run-dir>/<output-subdir-name>/.",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default=str(SCRIPT_DEFAULTS["topologies"]),
        help="Comma-separated topology list. Examples: original or regular,er,ws,ba",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(SCRIPT_DEFAULTS["episodes"]),
        help="Number of deterministic episodes to roll out.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=SCRIPT_DEFAULTS["max_steps"],
        help="Optional cap on steps per episode.",
    )
    parser.add_argument(
        "--episode-length-override",
        type=int,
        default=SCRIPT_DEFAULTS["episode_length_override"],
        help="Override the environment episode_length used during analysis.",
    )
    parser.add_argument(
        "--env-r",
        dest="env_r_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_r_override"],
        help="Override dynamics.r during analysis rollouts. Defaults to the value in results.json when unset.",
    )
    parser.add_argument(
        "--env-p-c",
        dest="env_p_c_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_p_c_override"],
        help="Override dynamics.p_c during analysis rollouts.",
    )
    parser.add_argument(
        "--env-beta",
        dest="env_beta_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_beta_override"],
        help="Override dynamics.beta during analysis rollouts. For Fermi this is the update intensity k/beta.",
    )
    parser.add_argument(
        "--env-strategy-update-rule",
        dest="env_strategy_update_rule_override",
        type=str,
        choices=("fermi", "q_learning", "q_learning_2x2", "imitate_best"),
        default=SCRIPT_DEFAULTS["env_strategy_update_rule_override"],
        help="Override dynamics.strategy_update_rule during analysis rollouts.",
    )
    parser.add_argument(
        "--env-p-max",
        dest="env_p_max_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_p_max_override"],
        help="Override dynamics.p_max during analysis rollouts.",
    )
    parser.add_argument(
        "--env-initial-resource",
        dest="env_initial_resource_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_initial_resource_override"],
        help="Override dynamics.initial_resource during analysis rollouts.",
    )
    parser.add_argument(
        "--env-initial-cooperation-prob",
        dest="env_initial_cooperation_prob_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_initial_cooperation_prob_override"],
        help="Override dynamics.initial_cooperation_prob during analysis rollouts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SCRIPT_DEFAULTS["seed"],
        help="Shared graph/reset seed. When omitted, reuse the training seed recorded in results.json.",
    )
    parser.add_argument(
        "--ego-resource-x-mode",
        type=str,
        choices=("absolute", "normalized"),
        default=str(SCRIPT_DEFAULTS["ego_resource_x_mode"]),
        help="Use raw ego_mean_resource or normalize it by a user-provided theoretical maximum before fitting/lookup.",
    )
    parser.add_argument(
        "--ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["ego_resource_normalizer"],
        help="Theoretical maximum used when --ego-resource-x-mode=normalized. Fill a different value for each p_c if needed.",
    )
    parser.add_argument(
        "--adaptive-policy-mode",
        type=str,
        choices=(
            "lookup_2way",
            "smooth_2way",
            "smooth_3way",
            "spline_3way",
            "continuous2d_suite",
            "continuous2d_softmax",
            "continuous2d_unitself",
        ),
        default=str(SCRIPT_DEFAULTS["adaptive_policy_mode"]),
        help=(
            "How to build the ego_mean_resource-only adaptive surrogate. "
            "lookup_2way reproduces the old noisy CSV interpolation; "
            "smooth_2way fits sigmoid(a+bx+cx^2) on log(1+m); "
            "smooth_3way fits a low-degree softmax over {unit_investment, equal, self}; "
            "spline_3way keeps the same 3-way softmax but replaces the global quadratic basis with a spline basis "
            "that is denser around ego_mean_resource≈430-450 and fits simplex weights directly; "
            "continuous2d_suite runs both new 2D continuous surrogates; "
            "continuous2d_softmax runs only the simplex-constrained 2D softmax model; "
            "continuous2d_unitself runs only the 2D UnitInv+Self control."
        ),
    )
    parser.add_argument(
        "--adaptive-conditioning-mode",
        type=str,
        choices=("none", "sender_degree_bin", "sender_resource_norm_bin"),
        default=str(SCRIPT_DEFAULTS["adaptive_conditioning_mode"]),
        help="Optional first-stage conditioning before the 1D ego_mean_resource profile. Stage A uses sender_degree_bin; Stage B uses sender_resource_norm_bin.",
    )
    parser.add_argument(
        "--adaptive-conditioned-profile-family",
        type=str,
        choices=("three_way", "unitself"),
        default=str(SCRIPT_DEFAULTS["adaptive_conditioned_profile_family"]),
        help="Inside each conditioning bin, fit either a full 3-way profile or a simplified UnitInv+Self profile.",
    )
    parser.add_argument(
        "--conditioned-degree-bin-mode",
        type=str,
        choices=("quantile", "log"),
        default=str(SCRIPT_DEFAULTS["conditioned_degree_bin_mode"]),
        help="Degree binning mode for sender_degree_bin conditioning. Reuse quantile bins to match existing BA analysis.",
    )
    parser.add_argument(
        "--conditioned-degree-bin-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["conditioned_degree_bin_count"]),
        help="Requested number of sender degree bins for degree-conditioned profiles.",
    )
    parser.add_argument(
        "--conditioned-degree-bin-min-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["conditioned_degree_bin_min_count"]),
        help="Minimum preferred sample count per conditioned sender degree bin; sparse adjacent bins are merged.",
    )
    parser.add_argument(
        "--conditioned-resource-bin-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["conditioned_resource_bin_count"]),
        help="Requested number of sender_resource_norm quantile bins for resource-conditioned profiles.",
    )
    parser.add_argument(
        "--continuous-second-variable",
        type=str,
        choices=("pool_grown_over_upperbound", "sender_pool_raw_norm"),
        default=str(SCRIPT_DEFAULTS["continuous_second_variable"]),
        help="Second continuous sender-state variable for the new 2D surrogate rules.",
    )
    parser.add_argument(
        "--continuous-basis-mode",
        type=str,
        choices=("linear_interaction", "hinge"),
        default=str(SCRIPT_DEFAULTS["continuous_basis_mode"]),
        help="Basis family for the new 2D continuous surrogate rules.",
    )
    parser.add_argument(
        "--continuous-x1-hinge-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["continuous_x1_hinge_count"]),
        help="Number of hinge knots on x1=log1p(ego_mean_resource) when --continuous-basis-mode=hinge.",
    )
    parser.add_argument(
        "--continuous-x2-hinge-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["continuous_x2_hinge_count"]),
        help="Number of hinge knots on x2 when --continuous-basis-mode=hinge.",
    )
    parser.add_argument(
        "--continuous-surface-grid-size",
        type=int,
        default=int(SCRIPT_DEFAULTS["continuous_surface_grid_size"]),
        help="Grid resolution per axis for exported 2D surrogate weight surfaces.",
    )
    parser.add_argument(
        "--adaptive-unitinv-equal-profile-csv",
        type=str,
        default=SCRIPT_DEFAULTS["adaptive_unitinv_equal_profile_csv"],
        help=(
            "Legacy CSV used only when --adaptive-policy-mode=lookup_2way. "
            "Expected columns: bin_center / weight_unit_investment_mean. Relative paths are resolved under --run-dir."
        ),
    )
    parser.add_argument(
        "--adaptive-unitinv-equal-knots",
        type=str,
        default=SCRIPT_DEFAULTS["adaptive_unitinv_equal_knots"],
        help=(
            "Legacy manual override for --adaptive-policy-mode=lookup_2way. "
            "Format: ego_mean_resource:w,ego_mean_resource:w . Example: 10:0.95,40:0.99,500:0.90"
        ),
    )
    parser.add_argument(
        "--adaptive-unitinv-equal-smooth-window",
        type=int,
        default=int(SCRIPT_DEFAULTS["adaptive_unitinv_equal_smooth_window"]),
        help=(
            "Legacy centered moving-average window for lookup_2way preprocessing. "
            "1 means no smoothing and is the most faithful to the reference CSV."
        ),
    )
    parser.add_argument(
        "--adaptive-reference-bin-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["adaptive_reference_bin_count"]),
        help="How many quantile bins to use when summarizing RL reference rows into the adaptive profile CSV/PNG.",
    )
    parser.add_argument(
        "--adaptive-ego-resource-profile-mode",
        type=str,
        choices=("continuous", "discrete"),
        default=str(SCRIPT_DEFAULTS["adaptive_ego_resource_profile_mode"]),
        help="Use a continuous spline/smooth profile or a discrete quantile-binned ego_mean_resource lookup profile.",
    )
    parser.add_argument(
        "--adaptive-ego-resource-bin-count",
        type=int,
        default=int(SCRIPT_DEFAULTS["adaptive_ego_resource_bin_count"]),
        help="Number of ego_mean_resource bins when --adaptive-ego-resource-profile-mode=discrete.",
    )
    parser.add_argument(
        "--spline-high-knot-style",
        type=str,
        choices=("dense", "sparse", "none"),
        default=str(SCRIPT_DEFAULTS["spline_high_knot_style"]),
        help="High-resource hand-crafted knot density for spline_3way fitting. sparse is the new smoother default.",
    )
    parser.add_argument(
        "--tail-cap-mode",
        type=str,
        choices=("none", "hard", "soft"),
        default=str(SCRIPT_DEFAULTS["tail_cap_mode"]),
        help="Optional tail cap applied to ego_mean_resource before smooth/spline profile evaluation.",
    )
    parser.add_argument(
        "--tail-cap-resource",
        type=float,
        default=float(SCRIPT_DEFAULTS["tail_cap_resource"]),
        help="Tail cap threshold/resource. hard mode clamps to this value; soft mode starts compressing above this value.",
    )
    parser.add_argument(
        "--tail-cap-soft-scale",
        type=float,
        default=float(SCRIPT_DEFAULTS["tail_cap_soft_scale"]),
        help="Compression factor for --tail-cap-mode=soft. Example: 0.25 means only keep 25%% of tail growth.",
    )
    parser.add_argument(
        "--spline-l2-penalty",
        type=float,
        default=float(SCRIPT_DEFAULTS["spline_l2_penalty"]),
        help="L2 penalty used by spline softmax fitting. Raised default is 1e-3 to stabilize BA tail behavior.",
    )
    parser.add_argument(
        "--early-alignment-max-step",
        type=int,
        default=int(SCRIPT_DEFAULTS["early_alignment_max_step"]),
        help="Compute the dedicated early trajectory alignment error CSV/PNG over steps t=0..this value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(SCRIPT_DEFAULTS["device"]),
        help="Torch device, e.g. cpu or cuda:0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found under {run_dir}")

    results_payload = _load_json(results_path)
    experiment_spec = results_payload["experiment"] if "experiment" in results_payload else results_payload
    if not isinstance(experiment_spec, dict):
        raise TypeError("Could not extract experiment spec from results.json.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            (run_dir / args.output_subdir_name).resolve()
            if args.output_subdir_name is not None
            else (run_dir / "rule_allocation_over_time" / Path(args.checkpoint_name).stem).resolve()
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    topologies = parse_topology_list(args.topologies)
    seed_base = int(experiment_spec.get("seed", 42)) if args.seed is None else int(args.seed)

    device = torch.device(args.device)
    actor, _ = load_actor_from_run_dir(run_dir, args.checkpoint_name, device)
    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)

    topology_summary_rows: list[dict[str, Any]] = []
    for topology_name in topologies:
        topology_spec = build_spec_for_topology(
            experiment_spec,
            topology_name,
            reference_mean_degree=reference_mean_degree,
        )
        topology_spec = apply_analysis_env_overrides(
            topology_spec,
            env_r_override=None if args.env_r_override is None else float(args.env_r_override),
            env_p_c_override=None if args.env_p_c_override is None else float(args.env_p_c_override),
            env_beta_override=None if args.env_beta_override is None else float(args.env_beta_override),
            env_strategy_update_rule_override=(
                None if args.env_strategy_update_rule_override is None else str(args.env_strategy_update_rule_override)
            ),
            env_p_max_override=None if args.env_p_max_override is None else float(args.env_p_max_override),
            env_initial_resource_override=(
                None if args.env_initial_resource_override is None else float(args.env_initial_resource_override)
            ),
            env_initial_cooperation_prob_override=(
                None
                if args.env_initial_cooperation_prob_override is None
                else float(args.env_initial_cooperation_prob_override)
            ),
        )
        topology_spec["seed"] = int(seed_base)
        topology_output_dir = output_dir if len(topologies) == 1 else (output_dir / topology_name)
        topology_summary_rows.extend(
            analyze_single_topology(
                actor=actor,
                run_dir=run_dir,
                checkpoint_name=args.checkpoint_name,
                output_dir=topology_output_dir,
                spec=topology_spec,
                topology_name=topology_name,
                episodes=max(1, int(args.episodes)),
                max_steps=args.max_steps,
                episode_length_override=args.episode_length_override,
                rollout_seed=int(seed_base),
                adaptive_policy_mode=str(args.adaptive_policy_mode),
                adaptive_conditioning_mode=str(args.adaptive_conditioning_mode),
                adaptive_conditioned_profile_family=str(args.adaptive_conditioned_profile_family),
                conditioned_degree_bin_mode=str(args.conditioned_degree_bin_mode),
                conditioned_degree_bin_count=max(1, int(args.conditioned_degree_bin_count)),
                conditioned_degree_bin_min_count=max(1, int(args.conditioned_degree_bin_min_count)),
                conditioned_resource_bin_count=max(1, int(args.conditioned_resource_bin_count)),
                adaptive_unitinv_equal_profile_csv=args.adaptive_unitinv_equal_profile_csv,
                adaptive_unitinv_equal_knots=args.adaptive_unitinv_equal_knots,
                adaptive_unitinv_equal_smooth_window=max(1, int(args.adaptive_unitinv_equal_smooth_window)),
                adaptive_reference_bin_count=max(2, int(args.adaptive_reference_bin_count)),
                continuous_second_variable=str(args.continuous_second_variable),
                continuous_basis_mode=str(args.continuous_basis_mode),
                continuous_x1_hinge_count=max(0, int(args.continuous_x1_hinge_count)),
                continuous_x2_hinge_count=max(0, int(args.continuous_x2_hinge_count)),
                continuous_surface_grid_size=max(8, int(args.continuous_surface_grid_size)),
                ego_resource_x_mode=str(args.ego_resource_x_mode),
                ego_resource_normalizer=(
                    None if args.ego_resource_normalizer is None else float(args.ego_resource_normalizer)
                ),
                adaptive_ego_resource_profile_mode=str(args.adaptive_ego_resource_profile_mode),
                adaptive_ego_resource_bin_count=max(2, int(args.adaptive_ego_resource_bin_count)),
                spline_high_knot_style=str(args.spline_high_knot_style),
                tail_cap_mode=str(args.tail_cap_mode),
                tail_cap_resource=float(args.tail_cap_resource),
                tail_cap_soft_scale=float(args.tail_cap_soft_scale),
                spline_l2_penalty=float(args.spline_l2_penalty),
                early_alignment_max_step=max(0, int(args.early_alignment_max_step)),
            )
        )

    write_csv(output_dir / "topology_summary.csv", topology_summary_rows)
    with (output_dir / "topology_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(topology_summary_rows, handle, ensure_ascii=False, indent=2)

    print(f"Rule allocation over-time analysis complete. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
