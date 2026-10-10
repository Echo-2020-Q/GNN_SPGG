"""Saved M1/M2 allocation mechanisms; NumPy is the only dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


class _ResourceProfile:
    """Resource-bin weights for proportional, equal and self allocation."""

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.edges = np.asarray(parameters["bin_edges_resource"], dtype=np.float64)
        weights = np.clip(np.asarray(parameters["bin_weight_rows"], dtype=np.float64), 0.0, None)
        if self.edges.ndim != 1 or self.edges.size < 2 or np.any(np.diff(self.edges) <= 0):
            raise ValueError("Resource-bin edges must be strictly increasing.")
        if weights.shape != (self.edges.size - 1, 3):
            raise ValueError("Each resource bin needs three mixture weights.")
        sums = weights.sum(axis=1, keepdims=True)
        valid = sums.squeeze(-1) > 1e-12
        self.weights = np.full(weights.shape, 1.0 / 3.0, dtype=np.float64)
        self.weights[valid] = weights[valid] / sums[valid]

    def weights_for_resource(self, resource: float) -> np.ndarray:
        value = max(0.0, float(resource))
        mode = self.parameters.get("tail_cap_mode", "none")
        cap = self.parameters.get("tail_cap_resource")
        if cap is not None and mode == "hard":
            value = min(value, float(cap))
        elif cap is not None and mode == "soft" and value > float(cap):
            value = float(cap) + float(self.parameters.get("tail_cap_soft_scale", 0.25)) * (value - float(cap))
        if self.parameters.get("ego_resource_x_mode", "absolute") == "normalized":
            normalizer = self.parameters.get("ego_resource_normalizer")
            if normalizer is None or float(normalizer) <= 1e-12:
                raise ValueError("A normalized profile needs a positive resource normalizer.")
            value /= float(normalizer)
        index = int(np.clip(np.searchsorted(self.edges, value, side="right") - 1, 0, len(self.weights) - 1))
        return self.weights[index]


class _MixturePolicy:
    def _profile_for_sender(self, observation: Mapping[str, np.ndarray], sender: int) -> _ResourceProfile:
        raise NotImplementedError

    def allocate(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        mask = np.asarray(observation["local_mask"], dtype=bool)
        resources = np.clip(np.asarray(observation["resources"], dtype=np.float64), 0.0, None)
        unit_investment = np.asarray(observation["unit_investment"], dtype=np.float64)
        if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
            raise ValueError("local_mask must be an N x N matrix.")
        if resources.shape != (len(mask),) or unit_investment.shape != (len(mask),):
            raise ValueError("resources and unit_investment must have length N.")
        sizes = mask.sum(axis=1, keepdims=True)
        if np.any(sizes == 0):
            raise ValueError("Each sender must have at least one eligible receiver.")
        equal = mask.astype(np.float64) / sizes
        weighted = mask.astype(np.float64) * unit_investment[None, :]
        sums = weighted.sum(axis=1, keepdims=True)
        positive = sums.squeeze(-1) > 0.0
        proportional = equal.copy()
        proportional[positive] = weighted[positive] / sums[positive]
        self_allocation = np.zeros(mask.shape, dtype=np.float64)
        allocation = np.zeros(mask.shape, dtype=np.float64)
        for sender in range(len(mask)):
            receivers = np.flatnonzero(mask[sender])
            if mask[sender, sender]:
                self_allocation[sender, sender] = 1.0
            else:
                self_allocation[sender, receivers] = 1.0 / float(receivers.size)
            resource = float(np.mean(resources[receivers]))
            w_prop, w_equal, w_self = self._profile_for_sender(observation, sender).weights_for_resource(resource)
            allocation[sender] = (
                w_prop * proportional[sender] + w_equal * equal[sender] + w_self * self_allocation[sender]
            )
        allocation = np.where(mask, allocation, 0.0)
        sums = allocation.sum(axis=1, keepdims=True)
        positive = sums.squeeze(-1) > 1e-12
        allocation[positive] /= sums[positive]
        allocation[~positive] = equal[~positive]
        return allocation

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.allocate(observation)


class M1Policy(_MixturePolicy):
    def __init__(self, parameters: dict):
        self.profile = _ResourceProfile(parameters)

    def _profile_for_sender(self, observation: Mapping[str, np.ndarray], sender: int) -> _ResourceProfile:
        return self.profile


class M2Policy(_MixturePolicy):
    def __init__(self, parameters: dict):
        self.profiles = {label: _ResourceProfile(profile) for label, profile in parameters["bin_profiles"].items()}
        self.degree_bins = parameters["degree_bins"]
        self.fallback_bin = parameters["fallback_bin"]
        if parameters["degree_selection"] != "degree_intervals":
            raise ValueError("M2 requires degree-interval selection.")

    def degree_bin(self, degree: int) -> str:
        for group in self.degree_bins:
            if int(group["degree_left"]) <= degree <= int(group["degree_right"]):
                return str(group["bin_label"])
        # Keep the original policy's fallback for degrees outside the fitted range.
        return self.fallback_bin

    def _profile_for_sender(self, observation: Mapping[str, np.ndarray], sender: int) -> _ResourceProfile:
        degree = int(round(float(np.asarray(observation["degrees"], dtype=np.float64)[sender])))
        return self.profiles[self.degree_bin(degree)]


def _load_parameters(path: str | Path | None, filename: str) -> dict:
    resolved = Path(path) if path is not None else Path(__file__).parent / "models" / filename
    with resolved.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_m1(path: str | Path | None = None) -> M1Policy:
    return M1Policy(_load_parameters(path, "m1.json"))


def load_m2(path: str | Path | None = None) -> M2Policy:
    return M2Policy(_load_parameters(path, "m2.json"))
